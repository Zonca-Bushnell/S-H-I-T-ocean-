"""Render raw OFES density, annual-MSS density anomaly, and 500-km W panels.

This diagnostic is deliberately independent from detection, QC, vertical
continuation, and rebuild-W.  It consumes the already selected strict-core
Jan-1 trajectories, reads the native daily ``prho`` and ``w`` fields, and
builds only the local 1993-2012 annual density climatology required by each
selected object.
"""
from __future__ import annotations

import argparse
import calendar
import csv
import json
import os
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage
from PIL import Image, ImageDraw, ImageFont

from ..ofes_io import ctl_path, expected_dta_bytes, open_dta_memmap, parse_ctl, require_daily_file
from ..run_ofes_rebuild_w import align_native_w_vertical, local_lon_lat_grid, sample_stack
from ..w_rebuild_config import DEFAULT_DATA_ROOT
from .build_ofes_meso_filter import nan_gaussian_lowpass

try:
    from netCDF4 import num2date
    from pydap.client import open_url
except ImportError:  # pragma: no cover - reported at runtime with a useful error
    num2date = None
    open_url = None


EARTH_RADIUS_M = 6_371_000.0
BASE_URL = "https://www.jamstec.go.jp/esc/fes/dods/OFES2/Monthly"
DEFAULT_VERTICAL_ROOT = Path(
    r"E:\DATA\01_Eddy_correspond\02_OFES\Fromthebeginning\04_Vertical\hua_center_section_selection_19910101\local_step_2cells_section_bipolar_tiebreak"
)
DEFAULT_OUTPUT_ROOT = Path(
    r"E:\DATA\01_Eddy_correspond\02_OFES\Fromthebeginning\05_TEMP\jan01_strict_core_density_w_19910101"
)
DEFAULT_OBJECTS = (
    "19910101_01073", "19910101_00131", "19910101_03856",
    "19910101_04637", "19910101_05716", "19910101_04887",
)


@dataclass(frozen=True)
class ObjectSpec:
    object_id: str
    region: str
    rank: int
    layer_rows: pd.DataFrame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vertical-root", type=Path, default=DEFAULT_VERTICAL_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--date", default="1991-01-01")
    parser.add_argument("--base-url", default=BASE_URL)
    parser.add_argument("--start-year", type=int, default=1993)
    parser.add_argument("--end-year", type=int, default=2012)
    parser.add_argument("--grid-n", type=int, default=81)
    parser.add_argument(
        "--extent-r",
        type=float,
        default=1.0,
        help="Map half-width in surface equivalent radii; 1.0 gives a 2R_eq square.",
    )
    parser.add_argument("--w-highpass-fwhm-km", type=float, default=500.0)
    parser.add_argument("--w-highpass-padding-sigma", type=float, default=3.0)
    parser.add_argument("--object-id", action="append", dest="object_ids")
    parser.add_argument("--block-retries", type=int, default=5)
    parser.add_argument("--retry-seconds", type=float, default=10.0)
    parser.add_argument("--render-only", action="store_true", help="Use complete local data packages without remote reads.")
    parser.add_argument(
        "--w-only",
        action="store_true",
        help="Render only Jan-1 Gaussian 500-km high-pass OFES w; do not read density or contact the climatology source.",
    )
    return parser.parse_args()


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def finite_percentile(values: np.ndarray, q: float, fallback: float = 1.0) -> float:
    finite = np.asarray(values, dtype="f8")
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        return fallback
    return max(float(np.nanpercentile(np.abs(finite), q)), np.finfo("f4").eps)


def meters_per_degree(latitude: float) -> tuple[float, float]:
    lat_rad = np.deg2rad(latitude)
    return (
        EARTH_RADIUS_M * np.cos(lat_rad) * np.pi / 180.0,
        EARTH_RADIUS_M * np.pi / 180.0,
    )


def find_specs(vertical_root: Path, requested: tuple[str, ...]) -> list[ObjectSpec]:
    selected_path = vertical_root / "curve_section_bipolar70_strict_core_local_lat_scaled_pil" / "selected_family_panels.csv"
    layers_path = vertical_root / "section_bipolar_vertical_catalog_local_lat_scaled" / "section_bipolar_strict_core_layer_catalog.csv"
    selected = pd.read_csv(selected_path)
    layers = pd.read_csv(layers_path)
    selected_id = "hua_object_id" if "hua_object_id" in selected.columns else "object_id"
    wanted = selected.loc[selected[selected_id].isin(requested)].copy()
    missing = sorted(set(requested) - set(wanted[selected_id]))
    if missing:
        raise ValueError(f"Requested strict-core objects not found: {missing}")
    specs: list[ObjectSpec] = []
    for _, row in wanted.sort_values(["region", "rank"]).iterrows():
        object_id = str(row[selected_id])
        obj_layers = layers.loc[layers["hua_object_id"].eq(object_id)].sort_values("depth_index").copy()
        if obj_layers.empty or not obj_layers["depth_index"].eq(0).any():
            raise ValueError(f"No surface-to-depth strict-core trajectory for {object_id}")
        specs.append(ObjectSpec(object_id, str(row["region"]), int(row["rank"]), obj_layers))
    return specs


def read_raw_fields(args: argparse.Namespace, spec: ObjectSpec, *, include_density: bool = True) -> dict[str, np.ndarray]:
    day = date.fromisoformat(args.date)
    metas = {name: parse_ctl(ctl_path(args.data_root, name)) for name in ("prho", "w")}
    paths = {
        name: require_daily_file(args.data_root, name, day, expected_dta_bytes(metas[name]))
        for name in metas
    }
    raw = {name: open_dta_memmap(paths[name], metas[name]) for name in metas}
    trajectory = spec.layer_rows.sort_values("depth_index").copy()
    surface = trajectory.loc[trajectory["depth_index"].eq(0)].iloc[0]
    nlev = int(trajectory["depth_index"].max()) + 1
    radius_km = float(surface["radius_km"])
    extent_km = float(args.extent_r) * radius_km
    axis = np.linspace(-extent_km, extent_km, int(args.grid_n), dtype="f4")
    x_km, y_km = np.meshgrid(axis, axis)
    lon_grid, lat_grid = local_lon_lat_grid(x_km / radius_km, y_km / radius_km, float(surface["center_lon_refined"]), float(surface["center_lat_refined"]), radius_km)

    # The vertical trajectory uses a daily Gaussian 500-km high-pass velocity
    # scale.  Give the local calculation 3 sigma of physical padding before
    # cropping, so the plotted window is not itself a filter boundary.
    step_km = float(axis[1] - axis[0])
    sigma_km = float(args.w_highpass_fwhm_km) / 2.354820045
    padding_cells = max(1, int(np.ceil(float(args.w_highpass_padding_sigma) * sigma_km / step_km)))
    extended_axis = np.arange(-padding_cells, int(args.grid_n) + padding_cells, dtype="f4") * step_km - extent_km
    ext_x_km, ext_y_km = np.meshgrid(extended_axis, extended_axis)
    ext_lon, ext_lat = local_lon_lat_grid(ext_x_km / radius_km, ext_y_km / radius_km, float(surface["center_lon_refined"]), float(surface["center_lat_refined"]), radius_km)
    w_source_extended = sample_stack(raw["w"], metas["w"], ext_lon, ext_lat, min(metas["w"].z.count, nlev)) / 100.0
    w_hp_extended = np.full_like(w_source_extended, np.nan, dtype="f4")
    for level in range(w_source_extended.shape[0]):
        lowpass = nan_gaussian_lowpass(
            w_source_extended[level], ext_lon[0], ext_lat[:, 0], float(args.w_highpass_fwhm_km),
            zonal_scale_mode="km", meridional_scale_mode="median", min_valid_weight_fraction=0.0,
            spatial_kernel="gaussian",
        )
        w_hp_extended[level] = w_source_extended[level] - lowpass
    crop = slice(padding_cells, padding_cells + int(args.grid_n))
    w_source = w_hp_extended[:, crop, crop]
    depth = np.asarray(metas["prho"].z.values[:nlev], dtype="f4")
    w, w_info = align_native_w_vertical(w_source, np.asarray(metas["w"].z.values[:w_source.shape[0]], dtype="f8"), depth.astype("f8"), "layer_center")
    payload = {
        "w_hp500_m_s": w.astype("f4"), "depth_m": depth,
        "x_km": x_km.astype("f4"), "y_km": y_km.astype("f4"), "longitude": lon_grid.astype("f4"), "latitude": lat_grid.astype("f4"),
        "trajectory_depth_index": trajectory["depth_index"].to_numpy(dtype="i2"),
        "trajectory_depth_m": trajectory["depth_m"].to_numpy(dtype="f4"),
        "trajectory_lon": trajectory["center_lon_refined"].to_numpy(dtype="f4"),
        "trajectory_lat": trajectory["center_lat_refined"].to_numpy(dtype="f4"),
        "surface_radius_km": np.asarray([radius_km], dtype="f4"),
        "w_alignment": np.asarray([json.dumps(w_info, default=lambda value: np.asarray(value).tolist())]),
        "w_highpass_definition": np.asarray(["daily OFES w - NaN-aware Gaussian LP_500km(w); no temporal filter"]),
        "w_highpass_padding_sigma": np.asarray([float(args.w_highpass_padding_sigma)], dtype="f4"),
    }
    if include_density:
        density = sample_stack(raw["prho"], metas["prho"], lon_grid, lat_grid, nlev)
        finite = density[np.isfinite(density)]
        density_reference = "absolute_density_kg_m3" if finite.size and float(np.nanmedian(finite)) > 1000.0 else "potential_density_anomaly_like_source_units"
        payload["density_raw"] = density.astype("f4")
        payload["density_reference"] = np.asarray([density_reference])
    return payload


def remote_array(dataset: object, key: str, selection: object) -> np.ndarray:
    return np.ma.asarray(dataset[key][selection].data).filled(np.nan)  # type: ignore[index,union-attr]


def remote_metadata(url: str, start_year: int, end_year: int) -> tuple[object, dict[tuple[int, int], int], np.ndarray, np.ndarray]:
    if open_url is None or num2date is None:
        raise RuntimeError("pydap and netCDF4 are required in OFES_detection for local prho climatology download")
    dataset = open_url(url, protocol="dap2")
    time_var = dataset["time"]
    dates = num2date(remote_array(dataset, "time", slice(None)), str(time_var.attributes["units"]), only_use_cftime_datetimes=True)
    time_index = {(item.year, item.month): pos for pos, item in enumerate(dates)}
    expected = {(year, month) for year in range(start_year, end_year + 1) for month in range(1, 13)}
    missing = expected - set(time_index)
    if missing:
        raise RuntimeError(f"Monthly prho source missing months: {sorted(missing)}")
    return dataset, time_index, np.asarray(remote_array(dataset, "lon", slice(None)), dtype="f8"), np.asarray(remote_array(dataset, "lat", slice(None)), dtype="f8")


def local_indices(lon: np.ndarray, lat: np.ndarray, lon_grid: np.ndarray, lat_grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]]:
    lon_target = np.asarray(lon_grid, dtype="f8") % 360.0
    lat_target = np.asarray(lat_grid, dtype="f8")
    # All six selected windows are away from the longitude wrap; fail loudly rather than splice a corrupt block.
    if np.ptp(lon_target) > 180.0:
        raise ValueError("Local climatology window crosses the longitude seam; split-window support is required")
    ii = np.interp(lon_target, lon, np.arange(lon.size, dtype="f8"))
    jj = np.interp(lat_target, lat, np.arange(lat.size, dtype="f8"))
    i0, i1 = max(0, int(np.floor(ii.min())) - 2), min(lon.size, int(np.ceil(ii.max())) + 3)
    j0, j1 = max(0, int(np.floor(jj.min())) - 2), min(lat.size, int(np.ceil(jj.max())) + 3)
    return ii, jj, (i0, i1, j0, j1)


def interpolate_remote_block(block: np.ndarray, ii: np.ndarray, jj: np.ndarray, bounds: tuple[int, int, int, int]) -> np.ndarray:
    i0, _, j0, _ = bounds
    coordinates = np.vstack([(jj - j0).ravel(), (ii - i0).ravel()])
    output = np.empty((block.shape[0], *ii.shape), dtype="f4")
    for level in range(block.shape[0]):
        field = np.asarray(block[level], dtype="f4")
        field[np.abs(field) > 1.0e30] = np.nan
        values = ndimage.map_coordinates(field, coordinates, order=1, mode="nearest", cval=np.nan).reshape(ii.shape)
        output[level] = values
    return output


def local_climatology(args: argparse.Namespace, spec: ObjectSpec, raw: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    # A cache is tied to its diagnostic grid.  This permits a compact 2R_eq
    # redraw to coexist with an interrupted wider-window request.
    cache = (
        args.output_root / "climatology_local" / spec.object_id
        / f"prho_annual_mss_extent{float(args.extent_r):g}r_grid{int(args.grid_n)}.npz"
    )
    nlev = int(raw["density_raw"].shape[0])
    shape = raw["density_raw"].shape
    months = [(year, month) for year in range(args.start_year, args.end_year + 1) for month in range(1, 13)]
    if cache.exists():
        with np.load(cache, allow_pickle=False) as saved:
            total = np.asarray(saved["weighted_sum"], dtype="f8")
            weight = np.asarray(saved["valid_day_weight"], dtype="f8")
            done = np.asarray(saved["completed_month"], dtype=bool)
        if total.shape != shape or weight.shape != shape or done.shape != (len(months),):
            raise ValueError(f"Invalid climatology cache shape: {cache}")
    else:
        total, weight, done = np.zeros(shape, dtype="f8"), np.zeros(shape, dtype="f8"), np.zeros(len(months), dtype=bool)
    if not done.all():
        url = f"{args.base_url}/prho"
        dataset, indices, lon, lat = remote_metadata(url, args.start_year, args.end_year)
        ii, jj, bounds = local_indices(lon, lat, raw["longitude"], raw["latitude"])
        i0, i1, j0, j1 = bounds
        for position, (year, month) in enumerate(months):
            if done[position]:
                continue
            days = calendar.monthrange(year, month)[1]
            remote_index = indices[(year, month)]
            for attempt in range(args.block_retries + 1):
                try:
                    source = dataset if attempt == 0 else open_url(url, protocol="dap2")
                    block = remote_array(source, "prho", (slice(remote_index, remote_index + 1), slice(0, nlev), slice(j0, j1), slice(i0, i1)))[0]
                    values = interpolate_remote_block(block, ii, jj, bounds)
                    valid = np.isfinite(values) & (np.abs(values) < 1.0e30)
                    total[valid] += values[valid] * days
                    weight[valid] += days
                    done[position] = True
                    atomic_npz(cache, weighted_sum=total, valid_day_weight=weight, completed_month=done.astype("u1"))
                    print(f"[density-climatology] {spec.object_id} {year:04d}-{month:02d} complete", flush=True)
                    break
                except Exception as exc:
                    if attempt >= args.block_retries:
                        raise RuntimeError(f"{spec.object_id} {year:04d}-{month:02d} failed after {attempt + 1} requests") from exc
                    delay = args.retry_seconds * (attempt + 1)
                    print(f"[density-climatology] retry {spec.object_id} {year:04d}-{month:02d}: {exc}", flush=True)
                    time.sleep(delay)
    climatology = np.divide(total, weight, out=np.full(shape, np.nan, dtype="f8"), where=weight > 0).astype("f4")
    metadata = {
        "source_url": f"{args.base_url}/prho", "years": [args.start_year, args.end_year],
        "months_expected": len(months), "months_completed": int(done.sum()),
        "day_weight_total_max": float(np.nanmax(weight)), "finite_fraction": float(np.isfinite(climatology).mean()),
        "cache": str(cache), "method": "native monthly prho sampled to diagnostic grid then finite-value day-weighted annual mean",
    }
    return climatology, weight.astype("f4"), metadata


def trajectory_xy(raw: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    lon0, lat0 = float(raw["trajectory_lon"][0]), float(raw["trajectory_lat"][0])
    mx, my = meters_per_degree(lat0)
    dx = ((np.asarray(raw["trajectory_lon"], dtype="f8") - lon0 + 180.0) % 360.0 - 180.0) * mx / 1000.0
    dy = (np.asarray(raw["trajectory_lat"], dtype="f8") - lat0) * my / 1000.0
    return dx.astype("f4"), dy.astype("f4")


def crop_payload_to_extent(raw: dict[str, np.ndarray], extent_r: float) -> dict[str, np.ndarray]:
    """Crop an existing local package to a centered square of +/- extent_r R_eq."""
    radius_km = float(np.asarray(raw["surface_radius_km"], dtype="f4")[0])
    half_width_km = float(extent_r) * radius_km
    x = np.asarray(raw["x_km"], dtype="f4")[0]
    y = np.asarray(raw["y_km"], dtype="f4")[:, 0]
    ix = np.flatnonzero(np.abs(x) <= half_width_km + 1.0e-5)
    iy = np.flatnonzero(np.abs(y) <= half_width_km + 1.0e-5)
    if ix.size < 9 or iy.size < 9:
        raise ValueError("Existing field package is too small for requested map extent")
    cropped: dict[str, np.ndarray] = {}
    for key, value in raw.items():
        array = np.asarray(value)
        if array.ndim == 3 and array.shape[-2:] == (y.size, x.size):
            cropped[key] = array[:, iy][:, :, ix]
        elif array.ndim == 2 and array.shape == (y.size, x.size):
            cropped[key] = array[iy][:, ix]
        else:
            cropped[key] = array
    return cropped


def selected_levels(trajectory_indices: np.ndarray) -> list[int]:
    top, bottom = int(trajectory_indices.min()), int(trajectory_indices.max())
    wanted = np.rint(np.linspace(top, bottom, 4)).astype(int)
    available = np.asarray(trajectory_indices, dtype=int)
    return [int(available[np.argmin(np.abs(available - level))]) for level in wanted]


def render_panel(output: Path, spec: ObjectSpec, raw: dict[str, np.ndarray], climatology: np.ndarray, weight: np.ndarray) -> dict[str, float]:
    density = raw["density_raw"]
    anomaly = density - climatology
    w = raw["w_hp500_m_s"]
    levels = selected_levels(raw["trajectory_depth_index"])
    x = raw["x_km"][0]
    y = raw["y_km"][:, 0]
    tx, ty = trajectory_xy(raw)
    trajectory_index = raw["trajectory_depth_index"].astype(int)
    depth = raw["depth_m"]
    raw_limit = finite_percentile(density, 99.0)
    anomaly_limit = finite_percentile(anomaly, 99.0)
    w_limit = finite_percentile(w * 1.0e6, 99.0)
    fields = [(density, "Raw OFES prho", "viridis", 0.0, raw_limit, "kg m^-3"), (anomaly, "prho - annual MSS", "RdBu_r", -anomaly_limit, anomaly_limit, "kg m^-3"), (w * 1.0e6, "OFES w_HP500", "RdBu_r", -w_limit, w_limit, "10^-6 m s^-1")]

    def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
        filename = "segoeuib.ttf" if bold else "segoeui.ttf"
        path = Path(r"C:\Windows\Fonts") / filename
        return ImageFont.truetype(path, size) if path.exists() else ImageFont.load_default()

    def colorize(values: np.ndarray, cmap_name: str, vmin: float, vmax: float) -> Image.Image:
        scaled = np.clip((np.asarray(values, dtype="f8") - vmin) / max(vmax - vmin, np.finfo("f8").eps), 0.0, 1.0)
        rgba = matplotlib.colormaps[cmap_name](scaled, bytes=True)[..., :3]
        rgba[~np.isfinite(values)] = (92, 92, 92)
        return Image.fromarray(rgba.astype("u1"), mode="RGB")

    canvas = Image.new("RGB", (3220, 2140), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((36, 25), f"{spec.region} rank {spec.rank} | {spec.object_id} | raw prho, annual-MSS density anomaly, and Gaussian 500-km high-pass W | 1991-01-01", fill=(20, 30, 45), font=font(27, True))
    draw.text((38, 67), "Maps use an equal-distance local grid; black line/points are the accepted vertical center trajectory.", fill=(65, 75, 90), font=font(17))
    panel_y = [125, 795, 1465]
    section_box = (30, 0, 730, 590)
    map_left = [790, 1380, 1970, 2560]
    map_size = 550
    x_span = max(float(np.ptp(x)), 1.0)
    y_span = max(float(np.ptp(y)), 1.0)
    depth_span = max(float(depth[-1]), 1.0)
    for row, (cube, title, cmap, vmin, vmax, unit) in enumerate(fields):
        top = panel_y[row]
        draw.text((38, top), f"{title} | {unit} | shared range [{vmin:.3g}, {vmax:.3g}]", fill=(20, 30, 45), font=font(20, True))
        section = colorize(cube[:, int(len(y) // 2), :], cmap, vmin, vmax).resize((section_box[2] - section_box[0] - 20, section_box[3] - 52), Image.Resampling.BILINEAR)
        sx0, sy0 = section_box[0] + 10, top + 42
        canvas.paste(section, (sx0, sy0))
        draw.rectangle((section_box[0], top + 28, section_box[2], top + section_box[3]), outline=(35, 45, 60), width=2)
        draw.text((section_box[0] + 8, top + 31), "trajectory-axis section", fill=(20, 30, 45), font=font(17, True))
        section_points = [(int(sx0 + (px - x.min()) / x_span * (section.size[0] - 1)), int(sy0 + pz / depth_span * (section.size[1] - 1))) for px, pz in zip(tx, raw["trajectory_depth_m"]) if np.isfinite(px) and np.isfinite(pz)]
        if len(section_points) > 1:
            draw.line(section_points, fill=(15, 15, 15), width=2)
        for px, py in section_points:
            draw.ellipse((px - 3, py - 3, px + 3, py + 3), fill=(15, 15, 15))
        for left, level in zip(map_left, levels):
            panel = colorize(cube[level], cmap, vmin, vmax).resize((map_size, map_size), Image.Resampling.BILINEAR)
            canvas.paste(panel, (left, top + 42))
            draw.rectangle((left, top + 28, left + map_size, top + 42 + map_size), outline=(35, 45, 60), width=2)
            draw.text((left + 8, top + 31), f"z={depth[level]:.0f} m", fill=(20, 30, 45), font=font(17, True))
            draw.text((left + 8, top + 568), "equal-distance local map", fill=(65, 75, 90), font=font(13))
            map_points = [(int(left + (px - x.min()) / x_span * map_size), int(top + 42 + (y.max() - py) / y_span * map_size)) for px, py in zip(tx, ty) if np.isfinite(px) and np.isfinite(py)]
            if len(map_points) > 1:
                draw.line(map_points, fill=(15, 15, 15), width=1)
            match = np.where(trajectory_index == level)[0]
            if match.size:
                px, py = map_points[int(match[0])]
                draw.line((px - 9, py, px + 9, py), fill=(10, 10, 10), width=2)
                draw.line((px, py - 9, px, py + 9), fill=(10, 10, 10), width=2)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
    canvas.save(output.with_suffix(".pdf"), "PDF", resolution=180)
    return {
        "raw_density_p01": float(np.nanpercentile(density, 1)), "raw_density_p99": float(np.nanpercentile(density, 99)),
        "density_anomaly_abs_p99": anomaly_limit, "w_hp500_abs_p99_1e6_m_s": w_limit,
        "climatology_finite_fraction": float(np.isfinite(climatology).mean()), "climatology_day_weight_min": float(np.nanmin(weight)),
    }


def render_w_only_panel(output: Path, spec: ObjectSpec, raw: dict[str, np.ndarray]) -> dict[str, float]:
    """Render high-pass W views without density/climatology reads."""
    w = np.asarray(raw["w_hp500_m_s"], dtype="f4") * 1.0e6
    levels = selected_levels(raw["trajectory_depth_index"])
    x = raw["x_km"][0]
    y = raw["y_km"][:, 0]
    tx, ty = trajectory_xy(raw)
    trajectory_index = raw["trajectory_depth_index"].astype(int)
    depth = raw["depth_m"]
    limit = finite_percentile(w, 99.0)
    def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
        name = "segoeuib.ttf" if bold else "segoeui.ttf"
        path = Path(r"C:\Windows\Fonts") / name
        return ImageFont.truetype(path, size) if path.exists() else ImageFont.load_default()

    def signed_rgb(values: np.ndarray) -> Image.Image:
        norm = np.clip(np.asarray(values, dtype="f8") / limit, -1.0, 1.0)
        rgb = np.full((*norm.shape, 3), 246, dtype=np.uint8)
        negative, positive = norm < 0, norm > 0
        rgb[..., 0][negative] = (245 * (1.0 + norm[negative])).astype(np.uint8)
        rgb[..., 1][negative] = (245 * (1.0 + norm[negative])).astype(np.uint8)
        rgb[..., 2][negative] = 175
        rgb[..., 0][positive] = 185
        rgb[..., 1][positive] = (245 * (1.0 - norm[positive])).astype(np.uint8)
        rgb[..., 2][positive] = (245 * (1.0 - norm[positive])).astype(np.uint8)
        rgb[~np.isfinite(norm)] = (95, 95, 95)
        return Image.fromarray(rgb, mode="RGB")

    image = Image.new("RGB", (2400, 650), "white")
    draw = ImageDraw.Draw(image)
    draw.text((36, 24), f"OFES w_HP500 | {spec.region} rank {spec.rank} | {spec.object_id} | 1991-01-01", fill=(20, 30, 45), font=font(27, True))
    draw.text((38, 64), f"daily w - Gaussian LP500 km(w), layer-center aligned; shared range +/- {limit:.1f} x10^-6 m/s", fill=(65, 75, 90), font=font(18))
    boxes = [(30, 110, 570, 600), (610, 110, 1050, 550), (1100, 110, 1540, 550), (1590, 110, 2030, 550), (2080, 110, 2390, 550)]
    section = signed_rgb(w[:, int(len(y) // 2), :]).resize((boxes[0][2] - boxes[0][0] - 20, boxes[0][3] - boxes[0][1] - 62), Image.Resampling.BILINEAR)
    image.paste(section, (boxes[0][0] + 10, boxes[0][1] + 40))
    draw.rectangle(boxes[0], outline=(35, 45, 60), width=2)
    draw.text((boxes[0][0] + 8, boxes[0][1] + 8), "trajectory-axis section", fill=(20, 30, 45), font=font(18, True))
    sx = boxes[0][0] + 10 + (tx - x.min()) / max(np.ptp(x), 1.0) * (boxes[0][2] - boxes[0][0] - 20)
    sy = boxes[0][1] + 40 + raw["trajectory_depth_m"] / max(float(depth[-1]), 1.0) * (boxes[0][3] - boxes[0][1] - 62)
    points = [(int(px), int(py)) for px, py in zip(sx, sy) if np.isfinite(px) and np.isfinite(py)]
    if len(points) > 1:
        draw.line(points, fill=(20, 20, 20), width=2)
    for point in points:
        draw.ellipse((point[0] - 3, point[1] - 3, point[0] + 3, point[1] + 3), fill=(20, 20, 20))
    for box, level in zip(boxes[1:], levels):
        panel = signed_rgb(w[level]).resize((box[2] - box[0] - 16, box[3] - box[1] - 54), Image.Resampling.BILINEAR)
        image.paste(panel, (box[0] + 8, box[1] + 38))
        draw.rectangle(box, outline=(35, 45, 60), width=2)
        draw.text((box[0] + 8, box[1] + 8), f"z={depth[level]:.0f} m", fill=(20, 30, 45), font=font(18, True))
        draw.text((box[0] + 8, box[3] - 24), "equal-distance local map", fill=(65, 75, 90), font=font(14))
        point = np.where(trajectory_index == level)[0]
        if point.size:
            px = box[0] + 8 + (tx[int(point[0])] - x.min()) / max(np.ptp(x), 1.0) * (box[2] - box[0] - 16)
            py = box[1] + 38 + (y.max() - ty[int(point[0])]) / max(np.ptp(y), 1.0) * (box[3] - box[1] - 54)
            draw.line((int(px) - 8, int(py), int(px) + 8, int(py)), fill=(0, 0, 0), width=2)
            draw.line((int(px), int(py) - 8, int(px), int(py) + 8), fill=(0, 0, 0), width=2)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)
    image.save(output.with_suffix(".pdf"), "PDF", resolution=180)
    return {"w_hp500_abs_p99_1e6_m_s": limit, "w_hp500_valid_fraction": float(np.isfinite(w).mean())}


def main() -> None:
    args = parse_args()
    if args.start_year > args.end_year or args.grid_n < 9:
        raise ValueError("Invalid climatology years or grid size")
    objects = tuple(args.object_ids or DEFAULT_OBJECTS)
    specs = find_specs(args.vertical_root, objects)
    args.output_root.mkdir(parents=True, exist_ok=True)
    summary: list[dict[str, object]] = []
    for spec in specs:
        if args.w_only:
            raw = read_raw_fields(args, spec, include_density=False)
            package = args.output_root / "object_data" / spec.object_id / "w_hp500_fields.npz"
            atomic_npz(package, **raw)
            metrics = render_w_only_panel(args.output_root / "figures_w_hp500" / spec.region / f"{spec.rank:02d}_{spec.object_id}_w_hp500.png", spec, raw)
            record = {
                "hua_object_id": spec.object_id, "region": spec.region, "rank": spec.rank,
                "surface_radius_km": float(raw["surface_radius_km"][0]), "continuation_layers": int(len(raw["trajectory_depth_index"])),
                "max_depth_m": float(np.nanmax(raw["trajectory_depth_m"])), **metrics,
            }
            summary.append(record)
            atomic_json(args.output_root / "manifests_w_hp500" / f"{spec.object_id}.json", record)
            continue
        package = args.output_root / "object_data" / spec.object_id / "density_w_fields.npz"
        if args.render_only:
            with np.load(package, allow_pickle=False) as saved:
                payload = {key: np.asarray(saved[key]) for key in saved.files}
            climatology = payload.pop("density_mss_1993_2012")
            weight = payload.pop("density_mss_valid_day_weight")
            raw = crop_payload_to_extent(payload, float(args.extent_r))
            climatology = crop_payload_to_extent(
                {
                    "surface_radius_km": raw["surface_radius_km"],
                    "x_km": payload["x_km"], "y_km": payload["y_km"],
                    "density_mss": climatology,
                },
                float(args.extent_r),
            )["density_mss"]
            weight = crop_payload_to_extent(
                {
                    "surface_radius_km": raw["surface_radius_km"],
                    "x_km": payload["x_km"], "y_km": payload["y_km"],
                    "density_mss": weight,
                },
                float(args.extent_r),
            )["density_mss"]
            climate_info = {"mode": "render_only"}
        else:
            raw = read_raw_fields(args, spec)
            climatology, weight, climate_info = local_climatology(args, spec, raw)
            raw["density_mss_1993_2012"] = climatology
            raw["density_mss_valid_day_weight"] = weight
            atomic_npz(package, **raw)
            raw.pop("density_mss_1993_2012")
            raw.pop("density_mss_valid_day_weight")
        metrics = render_panel(args.output_root / "figures" / spec.region / f"{spec.rank:02d}_{spec.object_id}_density_w_family.png", spec, raw, climatology, weight)
        record = {
            "hua_object_id": spec.object_id, "region": spec.region, "rank": spec.rank,
            "surface_radius_km": float(raw["surface_radius_km"][0]), "continuation_layers": int(len(raw["trajectory_depth_index"])),
            "max_depth_m": float(np.nanmax(raw["trajectory_depth_m"])), **metrics, **climate_info,
        }
        summary.append(record)
        atomic_json(args.output_root / "manifests" / f"{spec.object_id}.json", record)
    summary_name = "w_hp500_summary" if args.w_only else "summary"
    with (args.output_root / f"{summary_name}.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in summary for key in row}))
        writer.writeheader()
        writer.writerows(summary)
    atomic_json(args.output_root / f"{summary_name}.json", summary)
    atomic_json(args.output_root / ("w_hp500_manifest.json" if args.w_only else "manifest.json"), {
        "status": "complete", "date": args.date, "objects": list(objects),
        "vertical_velocity": "daily OFES w - NaN-aware Gaussian LP_500km(w), converted from cm/s to m/s and aligned to prho layer centers",
        "vertical_velocity_filter": {"kernel": "Gaussian", "fwhm_km": float(args.w_highpass_fwhm_km), "temporal_filter": "none", "padding_sigma": float(args.w_highpass_padding_sigma)},
        "mode": "w_hp500_only" if args.w_only else "density_and_w_hp500",
        "no_processing": ["no temporal filter", "no rebuild-W", "no modification of detection or vertical catalog"],
    })
    print(json.dumps({"objects": len(summary), "output_root": str(args.output_root)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
