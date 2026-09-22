"""Build and plot five Jan1 OFES raw-SSH preprocessing treatments.

The program is intentionally isolated from production catalog roots.  It uses
Jan1-Jan5 only to support the final Jan1 persistence decision.
"""

from __future__ import annotations

import argparse
import calendar
import ctypes
import json
import math
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from netCDF4 import Dataset
from PIL import Image, ImageDraw

from Detection_for_OFES.ofes_io import ctl_path, parse_ctl, read_variable_latlon_daily_only
from Detection_for_OFES.tools.export_origin_netcdf import regrid_scalar_to_velocity
from Detection_for_OFES.tools.plot_latest_ssh_vector_overview import diverging_ramp, font, lat_label, lon_label, px, py
from Detection_for_OFES.tools.plot_ofes_raw_ssh_regions_pillow import draw_vectors


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(r"F:\OFES\external_OFES2")
VELOCITY_SOURCE_ROOT = Path(r"E:\DATA\01_Eddy_correspond\02_OFES\origin_compatible_filter")
MONTHLY_ROOT = Path(r"E:\DATA\01_Eddy_correspond\02_OFES\ofes2_monthly_climatology_1993_2012")
OUTPUT_ROOT = Path(
    r"E:\DATA\01_Eddy_correspond\02_OFES\Fromthebeginning"
    r"\02_ssh_preprocessing_seed_radius_comparison_19910101"
)
ROSSBY_RADIUS = Path(
    r"E:\DATA\01_Eddy_correspond\02_OFES\rossby_radius_chelton1998"
    r"\unzip\fecampos-campos2025-a5d24c0\rossrad.nc"
)

REGIONS = {
    "global": {"label": "Global", "bbox": (0.0, 360.0, -76.0, 76.0)},
    "south_pacific_stcc": {"label": "South Pacific STCC", "bbox": (165.0, 230.0, -29.0, -21.0)},
    "kuroshio_extension": {"label": "Kuroshio Extension", "bbox": (140.0, 180.0, 28.0, 40.0)},
    "taiwan_hawaii_stcc_hlcc": {"label": "Taiwan-Hawaii STCC-HLCC corridor", "bbox": (122.0, 203.0, 18.0, 27.0)},
}


@dataclass(frozen=True)
class Treatment:
    key: str
    label: str
    baseline: str
    filter_mode: str | None
    kernel: str | None


TREATMENTS = (
    Treatment("simple_highpass_500km", "Raw H | Gaussian high-pass 500 km", "raw_h", "highpass", "gaussian"),
    Treatment("annual_mss_unfiltered", "Raw H minus 1993-2012 annual H-MSS", "annual_h_mss", None, None),
    Treatment("annual_mss_gaussian_dog", "Annual H-MSS | Gaussian Rossby DoG", "annual_h_mss", "rossby_dog", "gaussian"),
    Treatment("annual_mss_lanczos_dog", "Annual H-MSS | Lanczos Rossby DoG", "annual_h_mss", "rossby_dog", "lanczos"),
    Treatment("annual_mss_bessel_dog", "Annual H-MSS | Bessel-3 Rossby DoG", "annual_h_mss", "rossby_dog", "bessel"),
)

RAW_H_DOG_CANDIDATE_TREATMENTS = (
    Treatment("raw_h_gaussian_dog", "Raw H | Gaussian Rossby DoG (no annual MSS)", "raw_h", "rossby_dog", "gaussian"),
    Treatment("raw_h_lanczos_dog", "Raw H | Lanczos Rossby DoG (no annual MSS)", "raw_h", "rossby_dog", "lanczos"),
    Treatment("raw_h_bessel_dog", "Raw H | Bessel-3 Rossby DoG (no annual MSS)", "raw_h", "rossby_dog", "bessel"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--velocity-source-root", type=Path, default=VELOCITY_SOURCE_ROOT)
    parser.add_argument("--monthly-root", type=Path, default=MONTHLY_ROOT)
    parser.add_argument("--rossby-radius-path", type=Path, default=ROSSBY_RADIUS)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--start", default="1991-01-01")
    parser.add_argument("--end", default="1991-01-05")
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--rerun-existing", action="store_true")
    parser.add_argument("--skip-catalog", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument(
        "--replot-gaussian-candidate-zooms",
        action="store_true",
        help="Only redraw the three regional Gaussian-DoG candidate maps with saved SSH contours and equivalent-radius circles.",
    )
    parser.add_argument(
        "--replot-all-candidate-contour-zooms",
        action="store_true",
        help="Only redraw the three regional candidate maps for every treatment in local equal-distance coordinates.",
    )
    parser.add_argument(
        "--replot-all-global-colorbars",
        action="store_true",
        help="Only redraw all treatment global candidate/final maps with a labelled SSH colorbar.",
    )
    parser.add_argument(
        "--build-raw-h-dog-candidates",
        action="store_true",
        help="Build Jan1 raw-H Gaussian/Lanczos/Bessel DoG inputs and raw candidate contour figures only.",
    )
    parser.add_argument(
        "--build-simple-highpass-no-tile-cap-candidates",
        action="store_true",
        help="Re-run the Jan1 simple 500-km high-pass candidate stage without tile top-N truncation.",
    )
    parser.add_argument(
        "--plot-simple-highpass-no-tile-failure-cases",
        action="store_true",
        help="Draw four representative Kuroshio candidate failures from the completed no-tile-cap run.",
    )
    return parser.parse_args()


def dates(start: str, end: str) -> list[date]:
    first, last = date.fromisoformat(start), date.fromisoformat(end)
    if last < first:
        raise ValueError("--end must not precede --start")
    return [first + timedelta(days=i) for i in range((last - first).days + 1)]


def read_h_mss(monthly_root: Path, cache_path: Path, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as saved:
            cached_lon = saved["longitude"]
            cached_lat = saved["latitude"]
            field = saved["h_annual_mss_cm"].astype("f4")
        if field.shape == (lat.size, lon.size) and np.allclose(cached_lon, lon) and np.allclose(cached_lat, lat):
            return field
    total = np.zeros((lat.size, lon.size), dtype="f8")
    weights = np.zeros((lat.size, lon.size), dtype="f8")
    raw = monthly_root / "raw_monthly"
    for year in range(1993, 2013):
        for month in range(1, 13):
            stamp = f"{year}{month:02d}"
            eta_path = raw / stamp / "eta.npy"
            pair_path = raw / stamp / "pair.npy"
            if not eta_path.exists() or not pair_path.exists():
                raise FileNotFoundError(f"Missing cached monthly eta/pair for {stamp}")
            eta = np.asarray(np.load(eta_path, mmap_mode="r"), dtype="f4")
            pair = np.asarray(np.load(pair_path, mmap_mode="r"), dtype="f4")
            h = eta - (pair - 1000.0)
            valid = np.isfinite(h)
            day_weight = calendar.monthrange(year, month)[1]
            total[valid] += h[valid] * day_weight
            weights[valid] += day_weight
    field = np.divide(total, weights, out=np.full_like(total, np.nan), where=weights > 0).astype("f4")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        longitude=lon,
        latitude=lat,
        h_annual_mss_cm=field,
        valid_day_weight=weights.astype("u2"),
        months_used=np.asarray([f"{year}{month:02d}" for year in range(1993, 2013) for month in range(1, 13)]),
        definition=np.asarray("H=eta-(pair-1000); day-weighted annual MSS from monthly 1993-2012 fields"),
    )
    return field


def copy_attrs(source, target) -> None:
    for name in source.ncattrs():
        if name != "_FillValue":
            target.setncattr(name, source.getncattr(name))


def write_input(source_path: Path, output_path: Path, ssh: np.ndarray, *, treatment: Treatment) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".partial")
    temporary.unlink(missing_ok=True)
    with Dataset(source_path) as source, Dataset(temporary, "w", format="NETCDF4") as out:
        lon = np.asarray(source.variables["longitude"][:])
        lat = np.asarray(source.variables["latitude"][:])
        depth = np.asarray(source.variables["depth"][:1])
        out.createDimension("time", 1)
        out.createDimension("depth", 1)
        out.createDimension("latitude", lat.size)
        out.createDimension("longitude", lon.size)
        variables = {
            "time": out.createVariable("time", "f8", ("time",)),
            "depth": out.createVariable("depth", "f4", ("depth",)),
            "latitude": out.createVariable("latitude", "f4", ("latitude",)),
            "longitude": out.createVariable("longitude", "f4", ("longitude",)),
            "zos_glor": out.createVariable("zos_glor", "f4", ("time", "latitude", "longitude"), zlib=True, complevel=3),
            "uo_glor": out.createVariable("uo_glor", "f4", ("time", "depth", "latitude", "longitude"), zlib=True, complevel=3),
            "vo_glor": out.createVariable("vo_glor", "f4", ("time", "depth", "latitude", "longitude"), zlib=True, complevel=3),
        }
        for name, variable in variables.items():
            copy_attrs(source.variables[name], variable)
        variables["time"][:] = source.variables["time"][:1]
        variables["depth"][:] = depth
        variables["latitude"][:] = lat
        variables["longitude"][:] = lon
        variables["zos_glor"][0] = ssh.astype("f4")
        variables["uo_glor"][0, 0] = source.variables["uo_glor"][0, 0]
        variables["vo_glor"][0, 0] = source.variables["vo_glor"][0, 0]
        out.science_tag = treatment.key
        out.baseline_definition = "H_OFES=eta-(pressur-1000)"
        out.baseline_formula = treatment.label
        out.atmospheric_pressure_policy = "explicit pair correction retained by requested raw-H definition"
        out.processing_scope = "raw SSH baseline preparation only; spatial filter applied by the treatment driver where requested"
    temporary.replace(output_path)


def build_base_inputs(args: argparse.Namespace, all_days: list[date], treatment: Treatment, h_mss: np.ndarray | None, eta_lon: np.ndarray, eta_lat: np.ndarray) -> Path:
    root = args.output_root / treatment.key / "base_inputs"
    for current in all_days:
        destination = root / f"global_phy_{current:%Y%m%d}.nc"
        if destination.exists() and not args.rerun_existing:
            continue
        eta = np.asarray(read_variable_latlon_daily_only(args.data_root, "eta", current), dtype="f4")
        pair = np.asarray(read_variable_latlon_daily_only(args.data_root, "pressur", current), dtype="f4")
        raw_h = eta - (pair - 1000.0)
        raw_h[~np.isfinite(eta) | ~np.isfinite(pair)] = np.nan
        if treatment.baseline == "raw_h":
            native = raw_h
        else:
            if h_mss is None:
                raise ValueError("Annual-MSS treatments require a supplied H-MSS field")
            native = raw_h - h_mss
        source = args.velocity_source_root / f"global_phy_{current:%Y%m%d}.nc"
        with Dataset(source) as ds:
            velocity_lon = np.asarray(ds.variables["longitude"][:], dtype="f8")
            velocity_lat = np.asarray(ds.variables["latitude"][:], dtype="f8")
        remapped = regrid_scalar_to_velocity(native, eta_lon, eta_lat, velocity_lon, velocity_lat)
        write_input(source, destination, remapped, treatment=treatment)
    return root


def filtered_root(args: argparse.Namespace, treatment: Treatment, base_root: Path, all_days: list[date]) -> Path:
    if treatment.filter_mode is None:
        return base_root
    root = args.output_root / treatment.key / "filtered_inputs"
    day_args = ["--input-root", str(base_root), "--output-root", str(root), "--start", all_days[0].isoformat(), "--end", all_days[-1].isoformat(), "--available-start", all_days[0].isoformat(), "--available-end", all_days[-1].isoformat(), "--temporal-window-days", "1", "--max-depth-layers", "1", "--spatial-kernel", str(treatment.kernel), "--zonal-scale-mode", "km", "--meridional-scale-mode", "median", "--overwrite"]
    if treatment.filter_mode == "highpass":
        mode_args = ["--filter-mode", "highpass", "--large-cutoff-mode", "fixed", "--large-cutoff-km", "500", "--small-cutoff-km", "25", "--science-tag", "raw_h_gaussian_highpass_500km"]
    else:
        baseline_tag = "raw_h" if treatment.baseline == "raw_h" else "annual_h_mss"
        mode_args = ["--filter-mode", "bandpass", "--large-cutoff-mode", "rossby_lower_latadaptive_upper", "--rossby-radius-path", str(args.rossby_radius_path), "--rossby-small-factor", "0.5", "--rossby-min-km", "10", "--rossby-max-km", "180", "--adaptive-large-cutoff-min-km", "180", "--adaptive-large-cutoff-max-km", "180", "--science-tag", f"{baseline_tag}_rossby_dog_{treatment.kernel}"]
    command = [sys.executable, "-m", "Detection_for_OFES.tools.build_ofes_meso_filter", *day_args, *mode_args]
    subprocess.run(command, cwd=REPO_ROOT, check=True)
    return root


def run_catalog(args: argparse.Namespace, treatment: Treatment, input_root: Path, all_days: list[date]) -> Path:
    catalog = args.output_root / treatment.key / "catalog"
    if args.skip_catalog:
        return catalog
    command = [
        sys.executable, "-m", "Detection_for_OFES.tools.build_unified_eddy_catalog",
        "--filter-input-root", str(input_root), "--filter-output-root", str(input_root),
        "--output-root", str(catalog), "--start", all_days[0].isoformat(), "--end", all_days[-1].isoformat(),
        "--max-depth-m", "3", "--filter-max-depth-layers", "1", "--workers", str(args.workers),
        "--skip-filter", "--open-ocean-no-streamline-gate",
    ]
    if args.rerun_existing:
        command.append("--rerun-existing")
    subprocess.run(command, cwd=REPO_ROOT, check=True)
    return catalog


def read_surface(input_root: Path, current: date):
    path = input_root / f"global_phy_{current:%Y%m%d}.nc"
    with Dataset(path) as data:
        return (
            np.asarray(data.variables["longitude"][:], dtype="f8"),
            np.asarray(data.variables["latitude"][:], dtype="f8"),
            np.asarray(data.variables["zos_glor"][0], dtype="f4"),
            np.asarray(data.variables["uo_glor"][0, 0], dtype="f4"),
            np.asarray(data.variables["vo_glor"][0, 0], dtype="f4"),
        )


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def coords(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    lon = pd.to_numeric(frame.get("center_lon", frame.get("seed_lon", pd.Series(dtype=float))), errors="coerce")
    lat = pd.to_numeric(frame.get("center_lat", frame.get("seed_lat", pd.Series(dtype=float))), errors="coerce")
    return lon, lat


def subset(frame: pd.DataFrame, bbox: tuple[float, float, float, float]) -> pd.DataFrame:
    if frame.empty:
        return frame
    lon, lat = coords(frame)
    return frame[lon.between(bbox[0], bbox[1]) & lat.between(bbox[2], bbox[3])].copy()


def scalar_image(field: np.ndarray, limit: float, width: int, height: int) -> Image.Image:
    values = np.asarray(field, dtype="f4")
    rgb = np.asarray(diverging_ramp(np.nan_to_num(values, nan=0.0), -limit, limit), dtype="u1")
    rgb[~np.isfinite(values)] = (35, 40, 48)
    return Image.fromarray(rgb[::-1, :, :], "RGB").resize((width, height), Image.Resampling.BILINEAR)


def grid(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], bbox: tuple[float, float, float, float]) -> None:
    x0, y0, x1, y1 = box
    lo0, lo1, la0, la1 = bbox
    lon_step = 60 if lo1 - lo0 > 100 else 5
    lat_step = 30 if la1 - la0 > 70 else 5
    for value in np.arange(math.ceil(lo0 / lon_step) * lon_step, lo1 + 0.1, lon_step):
        x = px(float(value), lo0, lo1, x0, x1)
        draw.line((x, y0, x, y1), fill=(20, 25, 30), width=1)
        draw.text((x - 16, y1 + 5), lon_label(float(value)), fill=(30, 34, 44), font=font(14))
    for value in np.arange(math.ceil(la0 / lat_step) * lat_step, la1 + 0.1, lat_step):
        y = py(float(value), la0, la1, y0, y1)
        draw.line((x0, y, x1, y), fill=(20, 25, 30), width=1)
        draw.text((x0 - 48, y - 8), lat_label(float(value)), fill=(30, 34, 44), font=font(14))
    draw.rectangle(box, outline=(16, 20, 28), width=2)


def polarity_color(row: pd.Series) -> tuple[int, int, int]:
    return (40, 105, 230) if str(row.get("polarity", "")).lower() == "cyclonic" else (225, 45, 45)


def contour_indices(row: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Return the saved, angularly ordered SSH contour boundary indices."""
    def parse(value: object) -> np.ndarray:
        if value is None or pd.isna(value):
            return np.empty(0, dtype=int)
        try:
            return np.asarray([int(item) for item in str(value).split(";") if item != ""], dtype=int)
        except ValueError:
            return np.empty(0, dtype=int)

    ii = parse(row.get("ssh_contour_boundary_i"))
    jj = parse(row.get("ssh_contour_boundary_j"))
    n = min(ii.size, jj.size)
    return ii[:n], jj[:n]


def draw_dashed_ellipse(
    draw: ImageDraw.ImageDraw,
    ellipse: tuple[float, float, float, float],
    *,
    color: tuple[int, int, int],
    width: int = 2,
) -> None:
    for start in range(0, 360, 24):
        draw.arc(ellipse, start=start, end=start + 13, fill=color, width=width)


def draw_seed_layer(
    draw: ImageDraw.ImageDraw,
    rows: pd.DataFrame,
    lon_axis: np.ndarray,
    lat_axis: np.ndarray,
    box: tuple[int, int, int, int],
    bbox: tuple[float, float, float, float],
) -> None:
    lo0, lo1, la0, la1 = bbox
    x0, y0, x1, y1 = box
    for _, row in rows.iterrows():
        ii, jj = contour_indices(row)
        valid = (ii >= 0) & (ii < lon_axis.size) & (jj >= 0) & (jj < lat_axis.size)
        ii, jj = ii[valid], jj[valid]
        if ii.size < 3:
            continue
        color = polarity_color(row)
        points = [(px(float(lon_axis[i]), lo0, lo1, x0, x1), py(float(lat_axis[j]), la0, la1, y0, y1)) for i, j in zip(ii, jj)]
        draw.line(points + [points[0]], fill=color, width=2, joint="curve")

        center_lon = float(row.get("ssh_contour_center_lon", row.get("seed_lon", np.nan)))
        center_lat = float(row.get("ssh_contour_center_lat", row.get("seed_lat", np.nan)))
        radius = equivalent_radius_km(row)
        if np.isfinite(center_lon) and np.isfinite(center_lat) and np.isfinite(radius):
            dlat = radius / 111.32
            dlon = radius / max(111.32 * abs(math.cos(math.radians(center_lat))), 10.0)
            cx, cy = px(center_lon, lo0, lo1, x0, x1), py(center_lat, la0, la1, y0, y1)
            rx = abs(px(center_lon + dlon, lo0, lo1, x0, x1) - cx)
            ry = abs(py(center_lat + dlat, la0, la1, y0, y1) - cy)
            draw_dashed_ellipse(draw, (cx - rx, cy - ry, cx + rx, cy + ry), color=color)

        seed_lon = float(row.get("seed_lon", np.nan))
        seed_lat = float(row.get("seed_lat", np.nan))
        if np.isfinite(seed_lon) and np.isfinite(seed_lat):
            sx, sy = px(seed_lon, lo0, lo1, x0, x1), py(seed_lat, la0, la1, y0, y1)
            draw.line((sx - 4, sy, sx + 4, sy), fill=color, width=1)
            draw.line((sx, sy - 4, sx, sy + 4), fill=color, width=1)


def equivalent_radius_km(row: pd.Series) -> float:
    for name in ("radius_km", "ssh_contour_radius_cells", "accepted_radius_cells"):
        value = pd.to_numeric(pd.Series([row.get(name, np.nan)]), errors="coerce").iloc[0]
        if np.isfinite(value) and value > 0:
            return float(value if name == "radius_km" else value * 11.12)
    return float("nan")


def draw_final_layer(draw: ImageDraw.ImageDraw, rows: pd.DataFrame, box: tuple[int, int, int, int], bbox: tuple[float, float, float, float]) -> None:
    lo0, lo1, la0, la1 = bbox
    x0, y0, x1, y1 = box
    for _, row in rows.iterrows():
        lon = float(row.get("center_lon", row.get("seed_lon", np.nan)))
        lat = float(row.get("center_lat", row.get("seed_lat", np.nan)))
        radius = equivalent_radius_km(row)
        if not (np.isfinite(lon) and np.isfinite(lat) and np.isfinite(radius)):
            continue
        dlat = radius / 111.32
        dlon = radius / max(111.32 * abs(math.cos(math.radians(lat))), 10.0)
        cx, cy = px(lon, lo0, lo1, x0, x1), py(lat, la0, la1, y0, y1)
        rx = abs(px(lon + dlon, lo0, lo1, x0, x1) - cx)
        ry = abs(py(lat + dlat, la0, la1, y0, y1) - cy)
        color = polarity_color(row)
        draw.ellipse((cx - rx, cy - ry, cx + rx, cy + ry), outline=color, width=2)
        draw.ellipse((cx - 3, cy - 3, cx + 3, cy + 3), fill=color, outline=(255, 255, 255), width=1)


def draw_colorbar(image: Image.Image, draw: ImageDraw.ImageDraw, *, limit: float, box: tuple[int, int, int, int]) -> None:
    """Draw the actual diverging SSH scale used by the map panel."""
    x0, y0, x1, y1 = box
    values = np.linspace(float(limit), -float(limit), 256, dtype="f4")[:, None]
    rgb = np.asarray(diverging_ramp(values, -float(limit), float(limit)), dtype="u1")
    bar = Image.fromarray(rgb, "RGB").resize((x1 - x0, y1 - y0), Image.Resampling.NEAREST)
    image.paste(bar, (x0, y0))
    draw.rectangle(box, outline=(20, 24, 32), width=1)
    for fraction, value in ((0.0, limit), (0.25, 0.5 * limit), (0.5, 0.0), (0.75, -0.5 * limit), (1.0, -limit)):
        y = int(round(y0 + fraction * (y1 - y0)))
        draw.line((x1, y, x1 + 7, y), fill=(30, 34, 42), width=1)
        draw.text((x1 + 12, y - 8), f"{value:.1f}", fill=(35, 40, 50), font=font(15))
    draw.text((x0 - 2, y0 - 28), "SSH (cm)", fill=(35, 40, 50), font=font(16))


def draw_map(field: np.ndarray, lon: np.ndarray, lat: np.ndarray, u: np.ndarray, v: np.ndarray, rows: pd.DataFrame, *, bbox: tuple[float, float, float, float], title: str, layer: str, output: Path, vector_spacing: float) -> dict[str, int]:
    lo0, lo1, la0, la1 = bbox
    ix = np.where((lon >= lo0) & (lon <= lo1))[0]
    iy = np.where((lat >= la0) & (lat <= la1))[0]
    cropped = field[np.ix_(iy, ix)]
    limit = float(np.nanpercentile(np.abs(cropped), 99.0)) if np.isfinite(cropped).any() else 1.0
    image = Image.new("RGB", (2200, 1300), "white")
    box = (110, 120, 1950, 1110)
    image.paste(scalar_image(cropped, max(limit, 1e-6), box[2] - box[0], box[3] - box[1]), (box[0], box[1]))
    draw = ImageDraw.Draw(image)
    grid(draw, box, bbox)
    vector_info = draw_vectors(draw, lon=lon, lat=lat, u=u, v=v, box=box, bbox=bbox, spacing_deg=vector_spacing, arrow_pixels=22.0 if vector_spacing <= 0.5 else 18.0)
    selected = subset(rows, bbox)
    if layer == "candidate_seed":
        closed = selected.get("ssh_contour_closed", pd.Series(False, index=selected.index)).fillna(False).astype(bool)
        selected = selected.loc[closed].copy()
        draw_seed_layer(draw, selected, lon, lat, box, bbox)
        layer_title = f"Closed SSH-contour seeds: {len(selected)}; solid=contour, dashed=R_eq"
    else:
        draw_final_layer(draw, selected, box, bbox)
        layer_title = f"Final QC + persistence objects: {len(selected)}"
    cyclonic = int(selected.get("polarity", pd.Series(dtype=str)).astype(str).eq("cyclonic").sum())
    anticyclonic = int(selected.get("polarity", pd.Series(dtype=str)).astype(str).eq("anticyclonic").sum())
    draw.text((110, 32), title, fill=(18, 24, 38), font=font(34))
    draw.text((110, 76), f"{layer_title}; cyclonic={cyclonic}, anticyclonic={anticyclonic}; raw surface u/v, {vector_spacing:g} degree sampling", fill=(65, 75, 90), font=font(18))
    is_global = (lo1 - lo0) > 100.0 and (la1 - la0) > 70.0
    if is_global:
        draw_colorbar(image, draw, limit=limit, box=(1995, 150, 2028, 1010))
    else:
        draw.text((1980, 130), f"SSH\n+/-{limit:.1f} cm", fill=(35, 45, 60), font=font(18))
    save_to_long_output(image, output, image_format="PNG")
    save_to_long_output(image, output.with_suffix(".pdf"), image_format="PDF", resolution=180.0)
    return {"objects": int(len(selected)), "cyclonic": cyclonic, "anticyclonic": anticyclonic, "vector_stride_cells": int(vector_info["stride_cells"])}


def plot_treatment(args: argparse.Namespace, treatment: Treatment, input_root: Path, catalog: Path, current: date) -> dict[str, object]:
    lon, lat, ssh, u, v = read_surface(input_root, current)
    stamp = f"{current:%Y%m%d}"
    raw = read_csv(catalog / "raw_detection" / "daily_runs" / stamp / "centers_hua_style.csv")
    final = read_csv(catalog / "final_catalog" / "daily_runs" / stamp / "centers_hua_style.csv")
    products: dict[str, object] = {}
    for layer, frame in (("candidate_seed", raw), ("final_equivalent_radius", final)):
        layer_products = {}
        for region_key, region in REGIONS.items():
            spacing = 2.0 if region_key == "global" else 0.5
            path = args.output_root / treatment.key / "figures" / layer / f"{region_key}_{stamp}.png"
            layer_products[region_key] = {
                "path": str(path),
                **draw_map(ssh, lon, lat, u, v, frame, bbox=region["bbox"], title=f"{treatment.label} | {region['label']} | {current.isoformat()}", layer=layer, output=path, vector_spacing=spacing),
            }
        products[layer] = layer_products
    raw_surface = raw[raw.get("depth_index", 0).eq(0)] if not raw.empty and "depth_index" in raw else raw
    final_surface = final[final.get("depth_index", 0).eq(0)] if not final.empty and "depth_index" in final else final
    return {
        "treatment": treatment.key,
        "label": treatment.label,
        "raw_seed_count": int(len(raw_surface)),
        "closed_ssh_contour_count": int(raw_surface.get("ssh_contour_closed", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()),
        "qc_rejected_count": int(raw_surface.get("hua_pass", pd.Series(dtype=bool)).fillna(False).astype(bool).sum() - len(final_surface)),
        "transient_count": int(raw_surface.get("persistence_class", pd.Series(dtype=str)).astype(str).eq("transient").sum()),
        "final_count": int(len(final_surface)),
        "radius_median_km": float(pd.to_numeric(final_surface.get("radius_km", pd.Series(dtype=float)), errors="coerce").median()) if not final_surface.empty else np.nan,
        "amplitude_median_cm": float(pd.to_numeric(final_surface.get("ssh_contour_amplitude_cm", pd.Series(dtype=float)), errors="coerce").median()) if not final_surface.empty else np.nan,
        "products": products,
    }


def replot_gaussian_candidate_zooms(args: argparse.Namespace) -> None:
    """Refresh only the requested three candidate maps; no science files are rewritten."""
    treatment = next(item for item in TREATMENTS if item.key == "annual_mss_gaussian_dog")
    input_root = args.output_root / treatment.key / "filtered_inputs"
    catalog = args.output_root / treatment.key / "catalog"
    current = date.fromisoformat(args.start)
    lon, lat, ssh, u, v = read_surface(input_root, current)
    stamp = f"{current:%Y%m%d}"
    raw = read_csv(catalog / "raw_detection" / "daily_runs" / stamp / "centers_hua_style.csv")
    if raw.empty:
        raise FileNotFoundError(f"Missing raw candidate table for {treatment.key} on {current:%Y-%m-%d}")

    products: dict[str, object] = {}
    for region_key, region in REGIONS.items():
        if region_key == "global":
            continue
        path = args.output_root / treatment.key / "figures" / "candidate_seed" / f"{region_key}_{stamp}.png"
        products[region_key] = {
            "path": str(path),
            **draw_map(
                ssh,
                lon,
                lat,
                u,
                v,
                raw,
                bbox=region["bbox"],
                title=f"{treatment.label} | {region['label']} | {current.isoformat()}",
                layer="candidate_seed",
                output=path,
                vector_spacing=0.8,
            ),
        }
    manifest = {
        "treatment": treatment.key,
        "date": current.isoformat(),
        "scope": "candidate_seed regional redraw only",
        "candidate_policy": "only saved closed SSH contours; solid contour boundary, dashed equivalent-radius circle, seed cross",
        "vector_spacing_degrees": 0.8,
        "products": products,
    }
    target = args.output_root / treatment.key / "figures" / "candidate_seed" / f"candidate_contour_overlay_manifest_{stamp}.json"
    target.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


def local_equal_distance_extent_km(bbox: tuple[float, float, float, float]) -> tuple[float, float, float]:
    """Return local equirectangular width, height, and reference latitude."""
    lon0, lon1, lat0, lat1 = bbox
    lat_ref = 0.5 * (lat0 + lat1)
    width_km = (lon1 - lon0) * 111.32 * max(abs(math.cos(math.radians(lat_ref))), 0.1)
    height_km = (lat1 - lat0) * 111.32
    return float(width_km), float(height_km), float(lat_ref)


def save_to_long_output(image: Image.Image, path: Path, *, image_format: str, **kwargs: object) -> None:
    """Save via a short temporary path, then atomically replace a long Windows path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=path.suffix, delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        image.save(temporary_path, format=image_format, **kwargs)
        if os.name != "nt":
            os.replace(temporary_path, path)
            return
        destination = "\\\\?\\" + str(path.resolve())
        if not ctypes.windll.kernel32.MoveFileExW(str(temporary_path.resolve()), destination, 0x3):
            error = ctypes.get_last_error()
            raise OSError(error, f"MoveFileExW failed for {path}")
    finally:
        temporary_path.unlink(missing_ok=True)


def write_to_long_output(path: Path, text: str) -> None:
    with tempfile.NamedTemporaryFile(mode="w", suffix=path.suffix, delete=False, encoding="utf-8") as temporary:
        temporary.write(text)
        temporary_path = Path(temporary.name)
    try:
        if os.name != "nt":
            path.parent.mkdir(parents=True, exist_ok=True)
            os.replace(temporary_path, path)
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        destination = "\\\\?\\" + str(path.resolve())
        if not ctypes.windll.kernel32.MoveFileExW(str(temporary_path.resolve()), destination, 0x3):
            error = ctypes.get_last_error()
            raise OSError(error, f"MoveFileExW failed for {path}")
    finally:
        temporary_path.unlink(missing_ok=True)


def draw_equal_distance_candidate_map(
    field: np.ndarray,
    lon: np.ndarray,
    lat: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    rows: pd.DataFrame,
    *,
    bbox: tuple[float, float, float, float],
    title: str,
    output: Path,
    layer_label: str = "Closed SSH-contour seeds",
) -> dict[str, object]:
    """Render a regional candidate panel with equal x/y kilometre scale."""
    lo0, lo1, la0, la1 = bbox
    ix = np.where((lon >= lo0) & (lon <= lo1))[0]
    iy = np.where((lat >= la0) & (lat <= la1))[0]
    cropped = field[np.ix_(iy, ix)]
    limit = float(np.nanpercentile(np.abs(cropped), 99.0)) if np.isfinite(cropped).any() else 1.0
    width_km, height_km, lat_ref = local_equal_distance_extent_km(bbox)
    map_width = 1840
    map_height = max(1, int(round(map_width * height_km / max(width_km, 1.0))))
    left, top = 110, 120
    right, bottom = left + map_width, top + map_height
    canvas_height = bottom + 145
    image = Image.new("RGB", (2200, canvas_height), "white")
    image.paste(scalar_image(cropped, max(limit, 1e-6), map_width, map_height), (left, top))
    draw = ImageDraw.Draw(image)
    grid(draw, (left, top, right, bottom), bbox)
    vector_info = draw_vectors(
        draw,
        lon=lon,
        lat=lat,
        u=u,
        v=v,
        box=(left, top, right, bottom),
        bbox=bbox,
        spacing_deg=0.8,
        arrow_pixels=18.0,
    )
    selected = subset(rows, bbox)
    closed = selected.get("ssh_contour_closed", pd.Series(False, index=selected.index)).fillna(False).astype(bool)
    selected = selected.loc[closed].copy()
    draw_seed_layer(draw, selected, lon, lat, (left, top, right, bottom), bbox)
    cyclonic = int(selected.get("polarity", pd.Series(dtype=str)).astype(str).eq("cyclonic").sum())
    anticyclonic = int(selected.get("polarity", pd.Series(dtype=str)).astype(str).eq("anticyclonic").sum())
    draw.text((left, 32), title, fill=(18, 24, 38), font=font(34))
    draw.text(
        (left, 76),
        f"{layer_label}: {len(selected)}; solid=contour, dashed=R_eq; cyclonic={cyclonic}, anticyclonic={anticyclonic}; u/v 0.8 degree sampling",
        fill=(65, 75, 90),
        font=font(18),
    )
    colorbar_x0, colorbar_x1 = right + 42, right + 72
    color_values = np.linspace(limit, -limit, map_height, dtype="f4")[:, None]
    color_rgb = np.asarray(diverging_ramp(color_values, -limit, limit), dtype="u1")
    colorbar = Image.fromarray(color_rgb, "RGB").resize((colorbar_x1 - colorbar_x0, map_height), Image.Resampling.NEAREST)
    image.paste(colorbar, (colorbar_x0, top))
    draw.rectangle((colorbar_x0, top, colorbar_x1, bottom), outline=(35, 45, 60), width=1)
    draw.text((colorbar_x0 - 2, top - 28), "SSH (cm)", fill=(35, 45, 60), font=font(16))
    draw.text((colorbar_x1 + 8, top - 9), f"{limit:.1f}", fill=(35, 45, 60), font=font(15))
    draw.text((colorbar_x1 + 8, 0.5 * (top + bottom) - 9), "0", fill=(35, 45, 60), font=font(15))
    draw.text((colorbar_x1 + 8, bottom - 9), f"{-limit:.1f}", fill=(35, 45, 60), font=font(15))
    save_to_long_output(image, output, image_format="PNG")
    save_to_long_output(image, output.with_suffix(".pdf"), image_format="PDF", resolution=180.0)
    return {
        "objects": int(len(selected)),
        "cyclonic": cyclonic,
        "anticyclonic": anticyclonic,
        "vector_stride_cells": int(vector_info["stride_cells"]),
        "projection": "local_equirectangular_equal_distance",
        "reference_latitude_degrees": lat_ref,
        "map_width_km": width_km,
        "map_height_km": height_km,
        "map_pixel_width": map_width,
        "map_pixel_height": map_height,
    }


def replot_all_candidate_contour_zooms(args: argparse.Namespace) -> None:
    """Refresh all treatment candidate zooms without reopening any detection path."""
    current = date.fromisoformat(args.start)
    stamp = f"{current:%Y%m%d}"
    regional_keys = [key for key in REGIONS if key != "global"]
    for treatment in TREATMENTS:
        input_root = (
            args.output_root / treatment.key / "filtered_inputs"
            if treatment.filter_mode is not None
            else args.output_root / treatment.key / "base_inputs"
        )
        catalog = args.output_root / treatment.key / "catalog"
        lon, lat, ssh, u, v = read_surface(input_root, current)
        raw = read_csv(catalog / "raw_detection" / "daily_runs" / stamp / "centers_hua_style.csv")
        if raw.empty:
            raise FileNotFoundError(f"Missing raw candidate table for {treatment.key} on {current:%Y-%m-%d}")
        products: dict[str, object] = {}
        for region_key in regional_keys:
            region = REGIONS[region_key]
            path = args.output_root / treatment.key / "figures" / "candidate_seed" / f"{region_key}_{stamp}.png"
            products[region_key] = {
                "path": str(path),
                **draw_equal_distance_candidate_map(
                    ssh,
                    lon,
                    lat,
                    u,
                    v,
                    raw,
                    bbox=region["bbox"],
                    title=f"{treatment.label} | {region['label']} | {current.isoformat()}",
                    output=path,
                ),
            }
        manifest = {
            "treatment": treatment.key,
            "date": current.isoformat(),
            "scope": "candidate_seed regional redraw only; global figure preserved",
            "candidate_policy": "only saved closed SSH contours; solid contour boundary, dashed equivalent-radius circle, seed cross",
            "projection": "local equirectangular equal-distance at each region center latitude",
            "vector_spacing_degrees": 0.8,
            "products": products,
        }
        target = args.output_root / treatment.key / "figures" / "candidate_seed" / f"candidate_contour_overlay_manifest_{stamp}.json"
        write_to_long_output(target, json.dumps(manifest, ensure_ascii=False, indent=2))
        print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


def run_candidate_only_detection(
    args: argparse.Namespace,
    treatment: Treatment,
    input_root: Path,
    current: date,
    *,
    output: Path | None = None,
    candidate_selection: str = "tile_topn",
) -> Path:
    """Run one day through raw seed/contour detection only, without post-processing."""
    output = output or args.output_root / treatment.key / "candidate_detection"
    command = [
        sys.executable,
        "-m",
        "Detection_for_OFES.tools.build_unified_eddy_catalog",
        "--filter-input-root",
        str(input_root),
        "--filter-output-root",
        str(input_root),
        "--output-root",
        str(output),
        "--start",
        current.isoformat(),
        "--end",
        current.isoformat(),
        "--max-depth-m",
        "3",
        "--filter-max-depth-layers",
        "1",
        "--workers",
        "1",
        "--skip-filter",
        "--candidate-only",
        "--open-ocean-no-streamline-gate",
        "--candidate-selection",
        candidate_selection,
    ]
    if args.rerun_existing:
        command.append("--rerun-existing")
    subprocess.run(command, cwd=REPO_ROOT, check=True)
    return output


def build_simple_highpass_no_tile_cap_candidates(args: argparse.Namespace) -> None:
    """Compare all merged extrema against the existing tile-limited high-pass run."""
    current = date.fromisoformat(args.start)
    stamp = f"{current:%Y%m%d}"
    treatment = TREATMENTS[0]
    input_root = args.output_root / treatment.key / "filtered_inputs"
    if not (input_root / f"global_phy_{stamp}.nc").exists():
        raise FileNotFoundError(f"Missing existing simple-highpass input: {input_root}")

    comparison_root = args.output_root / treatment.key / "no_tile_cap_candidate_comparison"
    detection_root = run_candidate_only_detection(
        args,
        treatment,
        input_root,
        current,
        output=comparison_root / "raw_detection",
        candidate_selection="global_topn",
    )
    raw = read_csv(detection_root / "raw_detection" / "daily_runs" / stamp / "centers_hua_style.csv")
    if raw.empty:
        raise FileNotFoundError("No raw centers were written for the no-tile-cap comparison")
    lon, lat, ssh, u, v = read_surface(input_root, current)
    products: dict[str, object] = {}
    for region_key, region in REGIONS.items():
        if region_key == "global":
            continue
        path = comparison_root / "figures" / "candidate_seed" / f"{region_key}_{stamp}.png"
        products[region_key] = {
            "path": str(path),
            **draw_equal_distance_candidate_map(
                ssh,
                lon,
                lat,
                u,
                v,
                raw,
                bbox=region["bbox"],
                title=f"Raw H | Gaussian high-pass 500 km | no tile cap | {region['label']} | {current.isoformat()}",
                output=path,
            ),
        }
    manifest = {
        "treatment": treatment.key,
        "comparison": "tile_topn default versus global_topn with max_candidates_per_day=0",
        "date": current.isoformat(),
        "input": "existing simple_highpass_500km filtered input; no filter recomputation",
        "candidate_selection": "global_topn",
        "tile_limit": "disabled",
        "processing_stopped_after": "raw seed and SSH contour detection",
        "not_run": ["shape_overlap_qc", "persistence", "final_catalog"],
        "candidate_policy": "only saved closed SSH contours; solid contour boundary, dashed equivalent-radius circle, seed cross",
        "projection": "local equirectangular equal-distance at each region center latitude",
        "vector_spacing_degrees": 0.8,
        "raw_seed_count": int(len(raw)),
        "closed_ssh_contour_count": int(raw.get("ssh_contour_closed", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()),
        "products": products,
    }
    write_to_long_output(comparison_root / f"candidate_no_tile_cap_manifest_{stamp}.json", json.dumps(manifest, ensure_ascii=False, indent=2))
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


def plot_simple_highpass_no_tile_failure_cases(args: argparse.Namespace) -> None:
    """Visualize one reproducible seed example for each pre-QC contour failure."""
    current = date.fromisoformat(args.start)
    stamp = f"{current:%Y%m%d}"
    treatment = TREATMENTS[0]
    comparison_root = args.output_root / treatment.key / "no_tile_cap_candidate_comparison"
    raw_path = comparison_root / "raw_detection" / "raw_detection" / "daily_runs" / stamp / "centers_hua_style.csv"
    raw = read_csv(raw_path)
    if raw.empty:
        raise FileNotFoundError(f"Missing completed no-tile-cap raw detection: {raw_path}")

    input_root = args.output_root / treatment.key / "filtered_inputs"
    lon, lat, ssh, u, v = read_surface(input_root, current)
    bbox = REGIONS["kuroshio_extension"]["bbox"]
    cases = (
        ("touches_boundary", "T", "connected to search-window boundary", (245, 132, 31), 145.2, 36.2),
        ("amplitude_below_min", "A", "usable contour amplitude < 1.0 cm", (204, 60, 155), 150.9, 38.0),
        ("no_closed_contour", "N", "no closed component at tested levels", (225, 175, 24), 145.8, 33.5),
        ("multiple_extrema", "M", "one contour contains multiple extrema", (30, 165, 190), 145.6, 29.9),
    )

    width_km, height_km, lat_ref = local_equal_distance_extent_km(bbox)
    map_width = 1840
    map_height = max(1, int(round(map_width * height_km / max(width_km, 1.0))))
    left, top = 110, 140
    right, bottom = left + map_width, top + map_height
    image = Image.new("RGB", (2200, bottom + 220), "white")
    ix = np.where((lon >= bbox[0]) & (lon <= bbox[1]))[0]
    iy = np.where((lat >= bbox[2]) & (lat <= bbox[3]))[0]
    cropped = ssh[np.ix_(iy, ix)]
    limit = float(np.nanpercentile(np.abs(cropped), 99.0)) if np.isfinite(cropped).any() else 1.0
    image.paste(scalar_image(cropped, max(limit, 1e-6), map_width, map_height), (left, top))
    draw = ImageDraw.Draw(image)
    grid(draw, (left, top, right, bottom), bbox)
    vector_info = draw_vectors(
        draw,
        lon=lon,
        lat=lat,
        u=u,
        v=v,
        box=(left, top, right, bottom),
        bbox=bbox,
        spacing_deg=0.8,
        arrow_pixels=18.0,
    )
    closed = raw.get("ssh_contour_closed", pd.Series(False, index=raw.index)).fillna(False).astype(bool)
    draw_seed_layer(draw, subset(raw.loc[closed].copy(), bbox), lon, lat, (left, top, right, bottom), bbox)

    case_rows = []
    for reason, code, label, color, target_lon, target_lat in cases:
        dist2 = (pd.to_numeric(raw["seed_lon"], errors="coerce") - target_lon) ** 2 + (pd.to_numeric(raw["seed_lat"], errors="coerce") - target_lat) ** 2
        row = raw.loc[dist2.idxmin()].copy()
        actual = str(row.get("ssh_primary_discovery_reason", ""))
        if reason not in actual:
            raise RuntimeError(f"Expected {reason} near {target_lon},{target_lat}; found {actual}")
        seed_lon, seed_lat = float(row["seed_lon"]), float(row["seed_lat"])
        cx = px(seed_lon, bbox[0], bbox[1], left, right)
        cy = py(seed_lat, bbox[2], bbox[3], top, bottom)
        radius_km = 90.0
        dlon = radius_km / (111.32 * max(abs(math.cos(math.radians(lat_ref))), 0.1))
        dlat = radius_km / 111.32
        rx = abs(px(seed_lon + dlon, bbox[0], bbox[1], left, right) - cx)
        ry = abs(py(seed_lat + dlat, bbox[2], bbox[3], top, bottom) - cy)
        draw.ellipse((cx - rx, cy - ry, cx + rx, cy + ry), outline=color, width=4)
        draw.line((cx - 7, cy, cx + 7, cy), fill=color, width=3)
        draw.line((cx, cy - 7, cx, cy + 7), fill=color, width=3)
        draw.rectangle((cx + 8, cy - 25, cx + 31, cy - 2), fill=(255, 255, 255), outline=color, width=2)
        draw.text((cx + 13, cy - 24), code, fill=color, font=font(18))
        case_rows.append(
            {
                "code": code,
                "reason": actual,
                "label": label,
                "seed_lon": seed_lon,
                "seed_lat": seed_lat,
                "seed_scale_cells": int(row["seed_scale_cells"]),
                "seed_ssh_cm": float(row["ssh_value_m"]),
                "min_amplitude_cm": float(row["ssh_primary_min_amplitude_cm"]),
                "requested_window_half_cells": int(row["ssh_primary_requested_window_half_cells"]),
                "effective_window_half_cells": int(row["ssh_primary_effective_window_half_cells"]),
            }
        )

    draw.text((left, 30), "500-km high-pass | no tile cap | representative SSH-contour failures | Kuroshio Extension | 1991-01-01", fill=(18, 24, 38), font=font(29))
    draw.text((left, 76), "Colored rings isolate failed seeds; normal blue/red lines remain saved closed contours. These failures occur before shape, overlap, persistence, or final catalog.", fill=(65, 75, 90), font=font(16))
    legend_y = bottom + 35
    for index, item in enumerate(case_rows):
        color = cases[index][3]
        x = left + index * 445
        draw.ellipse((x, legend_y, x + 18, legend_y + 18), outline=color, width=3)
        draw.text((x + 27, legend_y - 2), f"{item['code']}: {item['label']}", fill=(35, 42, 55), font=font(16))
        draw.text((x + 27, legend_y + 22), f"{item['seed_lon']:.1f}E, {item['seed_lat']:.1f}; scale={item['seed_scale_cells']} cells", fill=(65, 75, 90), font=font(14))
    draw.text((left, bottom + 102), "A is a contour-search gate: each tested SSH level must differ from the seed by at least 1.0 cm here. It is not shape/overlap/persistence QC.", fill=(45, 55, 68), font=font(16))
    draw.text((left, bottom + 128), f"u/v sampling: 0.8 degrees; equal-distance local map at {lat_ref:.1f}N; callout radius: 90 km", fill=(65, 75, 90), font=font(15))
    colorbar_x0, colorbar_x1 = right + 42, right + 72
    color_values = np.linspace(limit, -limit, map_height, dtype="f4")[:, None]
    color_rgb = np.asarray(diverging_ramp(color_values, -limit, limit), dtype="u1")
    image.paste(Image.fromarray(color_rgb, "RGB").resize((colorbar_x1 - colorbar_x0, map_height), Image.Resampling.NEAREST), (colorbar_x0, top))
    draw.rectangle((colorbar_x0, top, colorbar_x1, bottom), outline=(35, 45, 60), width=1)
    draw.text((colorbar_x0 - 2, top - 28), "SSH (cm)", fill=(35, 45, 60), font=font(16))

    target = comparison_root / "figures" / "failure_cases" / f"kuroshio_extension_failure_cases_{stamp}.png"
    save_to_long_output(image, target, image_format="PNG")
    save_to_long_output(image, target.with_suffix(".pdf"), image_format="PDF", resolution=180.0)
    manifest = {
        "date": current.isoformat(),
        "input": str(raw_path),
        "scope": "four representative raw SSH-contour failure cases from completed no-tile-cap detection",
        "not_run": ["shape_overlap_qc", "persistence", "final_catalog"],
        "minimum_contour_amplitude_cm": 1.0,
        "cases": case_rows,
        "vector_stride_cells": int(vector_info["stride_cells"]),
        "projection": "local_equirectangular_equal_distance",
        "reference_latitude_degrees": lat_ref,
        "output": str(target),
    }
    write_to_long_output(target.with_name(f"kuroshio_extension_failure_cases_manifest_{stamp}.json"), json.dumps(manifest, ensure_ascii=False, indent=2))
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


def build_raw_h_dog_candidates(args: argparse.Namespace) -> None:
    """Create Jan1 no-MSS raw-H DoG candidate-only comparisons for all three kernels."""
    current = date.fromisoformat(args.start)
    eta_meta = parse_ctl(ctl_path(args.data_root, "eta"))
    eta_lon = np.asarray(eta_meta.x.values, dtype="f8")
    eta_lat = np.asarray(eta_meta.y.values, dtype="f8")
    stamp = f"{current:%Y%m%d}"
    regional_keys = [key for key in REGIONS if key != "global"]
    for treatment in RAW_H_DOG_CANDIDATE_TREATMENTS:
        base = build_base_inputs(args, [current], treatment, None, eta_lon, eta_lat)
        input_root = filtered_root(args, treatment, base, [current])
        detection_root = run_candidate_only_detection(args, treatment, input_root, current)
        raw = read_csv(detection_root / "raw_detection" / "daily_runs" / stamp / "centers_hua_style.csv")
        if raw.empty:
            raise FileNotFoundError(f"Raw candidate detection did not produce centers for {treatment.key}")
        lon, lat, ssh, u, v = read_surface(input_root, current)
        products: dict[str, object] = {}
        for region_key in regional_keys:
            region = REGIONS[region_key]
            path = args.output_root / treatment.key / "figures" / "candidate_seed" / f"{region_key}_{stamp}.png"
            products[region_key] = {
                "path": str(path),
                **draw_equal_distance_candidate_map(
                    ssh,
                    lon,
                    lat,
                    u,
                    v,
                    raw,
                    bbox=region["bbox"],
                    title=f"{treatment.label} | {region['label']} | {current.isoformat()}",
                    output=path,
                ),
            }
        manifest = {
            "treatment": treatment.key,
            "date": current.isoformat(),
            "baseline_definition": "H_OFES=eta-(pressur-1000), with no annual MSS subtraction",
            "spatial_filter": f"Rossby DoG with {treatment.kernel} kernel: LP(0.5 R1)-LP(180 km)",
            "processing_stopped_after": "raw seed and SSH contour detection",
            "not_run": ["shape_overlap_qc", "persistence", "final_catalog"],
            "candidate_policy": "only saved closed SSH contours; solid contour boundary, dashed equivalent-radius circle, seed cross",
            "projection": "local equirectangular equal-distance at each region center latitude",
            "vector_spacing_degrees": 0.8,
            "raw_seed_count": int(len(raw)),
            "closed_ssh_contour_count": int(raw.get("ssh_contour_closed", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()),
            "products": products,
        }
        target = args.output_root / treatment.key / "figures" / "candidate_seed" / f"candidate_only_manifest_{stamp}.json"
        write_to_long_output(target, json.dumps(manifest, ensure_ascii=False, indent=2))
        print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


def replot_all_global_colorbars(args: argparse.Namespace) -> None:
    """Refresh only global figures to add the SSH colorbar, preserving regional zooms."""
    current = date.fromisoformat(args.start)
    stamp = f"{current:%Y%m%d}"
    region = REGIONS["global"]
    for treatment in TREATMENTS:
        input_root = (
            args.output_root / treatment.key / "filtered_inputs"
            if treatment.filter_mode is not None
            else args.output_root / treatment.key / "base_inputs"
        )
        catalog = args.output_root / treatment.key / "catalog"
        lon, lat, ssh, u, v = read_surface(input_root, current)
        for layer, relative in (
            ("candidate_seed", "raw_detection/daily_runs"),
            ("final_equivalent_radius", "final_catalog/daily_runs"),
        ):
            rows = read_csv(catalog / relative / stamp / "centers_hua_style.csv")
            if rows.empty:
                raise FileNotFoundError(f"Missing {layer} table for {treatment.key} on {current:%Y-%m-%d}")
            path = args.output_root / treatment.key / "figures" / layer / f"global_{stamp}.png"
            draw_map(
                ssh,
                lon,
                lat,
                u,
                v,
                rows,
                bbox=region["bbox"],
                title=f"{treatment.label} | Global | {current.isoformat()}",
                layer=layer,
                output=path,
                vector_spacing=2.0,
            )
    print(json.dumps({"scope": "all treatment global candidate/final maps", "colorbar": "diverging SSH cm"}, ensure_ascii=False), flush=True)


def main() -> None:
    args = parse_args()
    if args.build_raw_h_dog_candidates:
        build_raw_h_dog_candidates(args)
        return
    if args.build_simple_highpass_no_tile_cap_candidates:
        build_simple_highpass_no_tile_cap_candidates(args)
        return
    if args.plot_simple_highpass_no_tile_failure_cases:
        plot_simple_highpass_no_tile_failure_cases(args)
        return
    if args.replot_all_global_colorbars:
        replot_all_global_colorbars(args)
        return
    if args.replot_all_candidate_contour_zooms:
        replot_all_candidate_contour_zooms(args)
        return
    if args.replot_gaussian_candidate_zooms:
        replot_gaussian_candidate_zooms(args)
        return
    all_days = dates(args.start, args.end)
    eta_meta = parse_ctl(ctl_path(args.data_root, "eta"))
    eta_lon = np.asarray(eta_meta.x.values, dtype="f8")
    eta_lat = np.asarray(eta_meta.y.values, dtype="f8")
    args.output_root.mkdir(parents=True, exist_ok=True)
    h_mss = read_h_mss(args.monthly_root, args.output_root / "shared_inputs" / "ofes_h_annual_mss_1993_2012.npz", eta_lon, eta_lat)
    results = []
    for treatment in TREATMENTS:
        base = build_base_inputs(args, all_days, treatment, h_mss, eta_lon, eta_lat)
        inputs = filtered_root(args, treatment, base, all_days)
        catalog = run_catalog(args, treatment, inputs, all_days)
        if not args.skip_plots:
            results.append(plot_treatment(args, treatment, inputs, catalog, all_days[0]))
    summary = pd.DataFrame([{key: value for key, value in item.items() if key != "products"} for item in results])
    summary.to_csv(args.output_root / "five_treatment_summary_19910101.csv", index=False)
    manifest = {
        "status": "complete",
        "date_window": [all_days[0].isoformat(), all_days[-1].isoformat()],
        "raw_ssh_definition": "H_OFES=eta-(pressur-1000)",
        "annual_mss_definition": "day-weighted H annual MSS using eta-(pair-1000), monthly 1993-2012 fields",
        "treatments": [treatment.__dict__ for treatment in TREATMENTS],
        "visual_policy": "candidate SSH seeds and final QC+persistence equivalent-radius objects are separate layers",
        "vector_policy": "raw surface u/v; global 2 degrees, regional 0.5 degrees",
        "results": results,
    }
    (args.output_root / "five_treatment_manifest_19910101.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"output_root": str(args.output_root), "summary": summary.to_dict(orient="records")}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
