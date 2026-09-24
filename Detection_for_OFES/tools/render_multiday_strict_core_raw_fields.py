"""Render raw OFES ``prho``/native-``w`` family panels for strict-core object-days.

This tool is deliberately diagnostic-only.  It reads the already selected
object-days and their saved vertical tracks; it neither changes a track nor
uses filtered/rebuilt vertical velocity.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

from ..ofes_io import ctl_path, expected_dta_bytes, open_dta_memmap, parse_ctl, require_daily_file
from ..run_ofes_rebuild_w import align_native_w_vertical, local_lon_lat_grid, parse_iso_date, sample_stack
from ..w_rebuild_config import DEFAULT_DATA_ROOT


VERTICAL_ROOT = Path(
    r"E:\DATA\01_Eddy_correspond\02_OFES\Fromthebeginning\04_Vertical\eta_highpass_500km_no_tilecap_ssh_geometry_section_bipolar_jan01_jan19"
)
DEFAULT_SELECTION = Path(
    r"E:\DATA\01_Eddy_correspond\02_OFES\Fromthebeginning\05_TEMP\nh_cyclonic_strict_core_native_w_pointwise_unrotated_eta_19910101_19910119\selected_object_days.csv"
)
DEFAULT_OUTPUT = Path(
    r"E:\DATA\01_Eddy_correspond\02_OFES\Fromthebeginning\05_TEMP\nh_cyclonic_strict_core_raw_fields_19910101_19910119"
)
DEFAULT_CLIMATOLOGY = Path(
    r"E:\DATA\01_Eddy_correspond\02_OFES\ofes2_monthly_climatology_1993_2012\climatology\ofes2_prho_annual_mss_1993_2012.npy"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--vertical-root", type=Path, default=VERTICAL_ROOT)
    parser.add_argument(
        "--vertical-run-root", type=Path,
        help="Explicit vertical continuation directory containing raw_detection/daily_runs. Preferred for new profile diagnostics.",
    )
    parser.add_argument("--selection-file", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--polarity", choices=("cyclonic", "anticyclonic"), default="cyclonic")
    parser.add_argument("--mode", choices=("raw", "outer-ring-reference", "with-anomaly"), default="raw")
    parser.add_argument("--climatology-file", type=Path, default=DEFAULT_CLIMATOLOGY)
    parser.add_argument("--grid-n", type=int, default=81)
    parser.add_argument("--extent-r", type=float, default=3.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-object-days", type=int, default=0)
    return parser.parse_args()


def finite_percentile(values: np.ndarray, percentile: float, fallback: float = 1.0) -> float:
    finite = np.asarray(values, dtype="f8")
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        return fallback
    value = float(np.percentile(finite, percentile))
    return value if np.isfinite(value) and value > 0 else fallback


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    names = ("segoeuib.ttf", "arialbd.ttf") if bold else ("segoeui.ttf", "arial.ttf")
    for name in names:
        candidate = Path(r"C:\Windows\Fonts") / name
        if candidate.exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def coolwarm_rgb(values: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Match the established strict-core composite figure colour convention."""
    normalized = np.nan_to_num(np.clip((values - lo) / max(hi - lo, 1.0e-12), 0.0, 1.0), nan=0.5)
    cool = np.asarray((59, 76, 192), dtype="f4")
    neutral = np.asarray((221, 220, 220), dtype="f4")
    warm = np.asarray((180, 4, 38), dtype="f4")
    lower = (2.0 * normalized[..., None]) * neutral + (1.0 - 2.0 * normalized[..., None]) * cool
    upper = (2.0 * (normalized[..., None] - 0.5)) * warm + (2.0 * (1.0 - normalized[..., None])) * neutral
    rgb = np.where(normalized[..., None] <= 0.5, lower, upper).astype(np.uint8)
    rgb[~np.isfinite(values)] = (55, 60, 68)
    return rgb


def paste_raster(image: Image.Image, values: np.ndarray, box: tuple[int, int, int, int], rgb: np.ndarray) -> None:
    raster = Image.fromarray(rgb, mode="RGB").resize((box[2] - box[0], box[3] - box[1]), Image.Resampling.BILINEAR)
    image.paste(raster, (box[0], box[1]))


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".part")
    partial.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    os.replace(partial, path)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def day_token(value: object) -> str:
    return str(value).replace("-", "")


def local_coordinates(radius_km: float, extent_r: float, grid_n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    extent_km = max(120.0, float(radius_km) * extent_r)
    axis = np.linspace(-extent_km, extent_km, grid_n, dtype="f8")
    x_km, y_km = np.meshgrid(axis, axis)
    return axis, x_km, y_km, np.hypot(x_km, y_km)


def sample_climatology_stack(raw: np.ndarray, meta, lon_grid: np.ndarray, lat_grid: np.ndarray, nlev: int) -> np.ndarray:
    """Bilinearly sample z,y,x annual-MSS storage using OFES native coordinates."""
    lon = np.asarray(meta.x.values, dtype="f8")
    lat = np.asarray(meta.y.values, dtype="f8")
    lon_step = float(np.nanmedian(np.diff(lon)))
    lat_step = float(np.nanmedian(np.diff(lat)))
    ii = ((lon_grid - float(lon[0])) / lon_step) % lon.size
    jj = np.clip((lat_grid - float(lat[0])) / lat_step, 0, lat.size - 1)
    coords = np.vstack([jj.ravel(), ii.ravel()])
    result = np.empty((nlev, *lon_grid.shape), dtype="f4")
    for depth_index in range(nlev):
        layer = np.asarray(raw[depth_index], dtype="f4")
        layer[np.abs(layer) > 1.0e30] = np.nan
        result[depth_index] = ndimage.map_coordinates(layer, coords, order=1, mode="nearest", cval=np.nan).reshape(lon_grid.shape)
    return result


def select_depth_indices(rows: pd.DataFrame) -> list[int]:
    available = sorted(set(rows.depth_index.astype(int)))
    maximum = max(available)
    selected: list[int] = []
    for target in (0.0, maximum / 3.0, 2.0 * maximum / 3.0, float(maximum)):
        choice = min(available, key=lambda value: abs(value - target))
        if choice not in selected:
            selected.append(choice)
    while len(selected) < 4:
        selected.append(selected[-1])
    return selected[:4]


def track_xy(rows: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    ordered = rows.sort_values("depth_index")
    surface = ordered.iloc[0]
    meters_lon = 111_320.0 * np.cos(np.deg2rad(float(surface.center_lat_refined)))
    x = ((ordered.center_lon_refined.to_numpy(dtype="f8") - float(surface.center_lon_refined) + 180.0) % 360.0 - 180.0) * meters_lon / 1000.0
    y = (ordered.center_lat_refined.to_numpy(dtype="f8") - float(surface.center_lat_refined)) * 111_132.0 / 1000.0
    return x, y


def contour_levels(values: np.ndarray, signed: bool) -> np.ndarray:
    if signed:
        vmax = finite_percentile(np.abs(values), 98, fallback=1.0e-10)
        return np.linspace(-vmax, vmax, 11)[1:-1]
    lo = float(np.nanpercentile(values, 5)) if np.isfinite(values).any() else 0.0
    hi = float(np.nanpercentile(values, 95)) if np.isfinite(values).any() else 1.0
    return np.linspace(lo, hi, 9)


def value_scale(values: np.ndarray, signed: bool) -> tuple[float, float, str]:
    if signed:
        bound = finite_percentile(np.abs(values), 98, fallback=1.0e-10)
        return -bound, bound, f"+/- {bound:.3e}"
    lo = float(np.nanpercentile(values, 2)) if np.isfinite(values).any() else 0.0
    hi = float(np.nanpercentile(values, 98)) if np.isfinite(values).any() else 1.0
    return lo, hi, f"{lo:.3f} to {hi:.3f}"


def draw_colorbar(draw: ImageDraw.ImageDraw, image: Image.Image, box: tuple[int, int, int, int], lo: float, hi: float, title: str) -> None:
    values = np.linspace(hi, lo, 256, dtype="f4")[:, None]
    strip = Image.fromarray(coolwarm_rgb(values, lo, hi), mode="RGB").resize((box[2] - box[0], box[3] - box[1]))
    image.paste(strip, (box[0], box[1]))
    draw.rectangle(box, outline=(30, 35, 45), width=1)
    draw.text((box[0] - 8, box[1] - 24), title, fill=(20, 30, 45), font=font(12, True))
    draw.text((box[2] + 5, box[1] - 5), f"{hi:.3g}", fill=(45, 50, 60), font=font(11))
    draw.text((box[2] + 5, (box[1] + box[3]) // 2 - 5), f"{(lo + hi) / 2.0:.3g}", fill=(45, 50, 60), font=font(11))
    draw.text((box[2] + 5, box[3] - 12), f"{lo:.3g}", fill=(45, 50, 60), font=font(11))


def draw_reference_rings(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], axis: np.ndarray, radius_km: float) -> None:
    x0, y0, x1, y1 = box
    center_x, center_y = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    for multiple, width, dashed in ((1.0, 2, False), (1.5, 1, True), (2.0, 1, True)):
        fraction = multiple * radius_km / (axis[-1] - axis[0])
        rx, ry = fraction * (x1 - x0), fraction * (y1 - y0)
        bounds = (center_x - rx, center_y - ry, center_x + rx, center_y + ry)
        if not dashed:
            draw.ellipse(bounds, outline=(10, 10, 10), width=width)
        else:
            for start in range(0, 360, 18):
                draw.arc(bounds, start=start, end=start + 9, fill=(25, 25, 25), width=width)


def draw_map_axes(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], axis: np.ndarray) -> None:
    x0, y0, x1, y1 = box
    label = font(10)
    for value in (float(axis[0]), 0.0, float(axis[-1])):
        fraction = (value - axis[0]) / (axis[-1] - axis[0])
        px = x0 + fraction * (x1 - x0)
        draw.line((px, y1, px, y1 + 4), fill=(25, 30, 40), width=1)
        draw.text((px - 15, y1 + 5), f"{value:.0f}", fill=(60, 65, 75), font=label)
    for value in (float(axis[-1]), 0.0, float(axis[0])):
        fraction = (axis[-1] - value) / (axis[-1] - axis[0])
        py = y0 + fraction * (y1 - y0)
        draw.line((x0 - 4, py, x0, py), fill=(25, 30, 40), width=1)
        draw.text((x0 - 37, py - 6), f"{value:.0f}", fill=(60, 65, 75), font=label)
    draw.text(((x0 + x1) // 2 - 36, y1 + 20), "east (km)", fill=(60, 65, 75), font=label)
    draw.text((x0 - 57, (y0 + y1) // 2 - 5), "north", fill=(60, 65, 75), font=label)


def draw_section_axes(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], axis: np.ndarray, depth: np.ndarray) -> None:
    x0, y0, x1, y1 = box
    label = font(10)
    for value in (float(axis[0]), 0.0, float(axis[-1])):
        fraction = (value - axis[0]) / (axis[-1] - axis[0])
        px = x0 + fraction * (x1 - x0)
        draw.line((px, y1, px, y1 + 4), fill=(25, 30, 40), width=1)
        draw.text((px - 15, y1 + 5), f"{value:.0f}", fill=(60, 65, 75), font=label)
    for value in (0.0, float(depth[-1]) / 2.0, float(depth[-1])):
        fraction = value / float(depth[-1])
        py = y0 + fraction * (y1 - y0)
        draw.line((x0 - 4, py, x0, py), fill=(25, 30, 40), width=1)
        draw.text((x0 - 38, py - 6), f"{value:.0f}", fill=(60, 65, 75), font=label)
    draw.text(((x0 + x1) // 2 - 45, y1 + 20), "east (km)", fill=(60, 65, 75), font=label)
    draw.text((x0 - 55, (y0 + y1) // 2 - 5), "depth", fill=(60, 65, 75), font=label)


def draw_map(draw: ImageDraw.ImageDraw, image: Image.Image, field: np.ndarray, axis: np.ndarray, radius_km: float, x_track: np.ndarray, y_track: np.ndarray, rows: pd.DataFrame, depth_index: int, box: tuple[int, int, int, int], label: str, signed: bool, low: float, high: float, scale: str) -> str:
    paste_raster(image, field, box, coolwarm_rgb(field, low, high))
    draw_reference_rings(draw, box, axis, radius_km)
    draw.rectangle(box, outline=(25, 30, 40), width=2)
    visible = rows.loc[rows.depth_index.le(depth_index)]
    count = len(visible)
    x0, y0, x1, y1 = box
    points = [(x0 + (x + axis[-1]) / (2.0 * axis[-1]) * (x1 - x0), y1 - (y + axis[-1]) / (2.0 * axis[-1]) * (y1 - y0)) for x, y in zip(x_track[:count], y_track[:count])]
    if len(points) > 1:
        draw.line(points, fill=(10, 10, 10), width=2)
    for point in points:
        draw.ellipse((point[0] - 2, point[1] - 2, point[0] + 2, point[1] + 2), fill=(5, 5, 5))
    point = points[-1]
    draw.line((point[0] - 5, point[1] - 5, point[0] + 5, point[1] + 5), fill=(255, 215, 0), width=2)
    draw.line((point[0] - 5, point[1] + 5, point[0] + 5, point[1] - 5), fill=(255, 215, 0), width=2)
    draw.text((x0, y0 - 25), label, fill=(20, 30, 45), font=font(14, True))
    draw.text((x0 + 3, y0 + 3), scale, fill=(15, 18, 24), font=font(11, True))
    draw_map_axes(draw, box, axis)
    return scale


def draw_section(draw: ImageDraw.ImageDraw, image: Image.Image, field: np.ndarray, depth: np.ndarray, axis: np.ndarray, rows: pd.DataFrame, x_track: np.ndarray, box: tuple[int, int, int, int], label: str, signed: bool, low: float, high: float, scale: str) -> str:
    section = field[:, field.shape[1] // 2, :]
    paste_raster(image, section, box, coolwarm_rgb(section, low, high))
    draw.rectangle(box, outline=(25, 30, 40), width=2)
    x0, y0, x1, y1 = box
    track = rows.sort_values("depth_index")
    points = [(x0 + (x + axis[-1]) / (2.0 * axis[-1]) * (x1 - x0), y0 + float(z) / float(depth[-1]) * (y1 - y0)) for x, z in zip(x_track, track.depth_m.to_numpy(dtype="f8"))]
    if len(points) > 1:
        draw.line(points, fill=(10, 10, 10), width=2)
    for point in points:
        draw.ellipse((point[0] - 2, point[1] - 2, point[0] + 2, point[1] + 2), fill=(5, 5, 5))
    draw.text((x0, y0 - 25), label, fill=(20, 30, 45), font=font(14, True))
    draw.text((x0 + 3, y0 + 3), scale, fill=(15, 18, 24), font=font(11, True))
    draw_section_axes(draw, box, axis, depth)
    return scale


def outer_ring_anomaly(prho: np.ndarray, axis: np.ndarray, radius_km: float) -> np.ndarray:
    x_km, y_km = np.meshgrid(axis, axis)
    ring = (np.hypot(x_km, y_km) >= 1.5 * radius_km) & (np.hypot(x_km, y_km) <= 2.0 * radius_km)
    reference = np.nanmedian(np.where(ring[None, :, :], prho, np.nan), axis=(1, 2))
    return prho - reference[:, None, None]


def render_panel(path: Path, obj: pd.Series, rows: pd.DataFrame, axis: np.ndarray, radius_km: float, prho: np.ndarray, native_w: np.ndarray, depth: np.ndarray, climatology_anomaly: np.ndarray | None, include_outer_ring: bool) -> dict[str, object]:
    fields: list[tuple[str, str, np.ndarray, bool, str]] = [("rho", "Raw OFES prho", prho, False, "prho (kg m-3)")]
    if include_outer_ring:
        fields.append(("rho_rel", "Outer-ring-relative density", outer_ring_anomaly(prho, axis, radius_km), True, "delta prho (kg m-3)"))
    if climatology_anomaly is not None:
        fields.append(("rho_prime", "Density anomaly: raw prho - annual MSS", climatology_anomaly, True, "rho prime (kg m-3)"))
    fields.append(("w", "Raw native OFES w", native_w, True, "w (m s-1)"))
    selected = select_depth_indices(rows)
    x_track, y_track = track_xy(rows)
    width, row_height = 2320, 410
    height = 135 + len(fields) * row_height + 55
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    region = str(obj.get("region", "latest strict core")).replace("_", " ")
    polarity = str(obj.polarity).replace("_", " ")
    draw.text((42, 24), f"{obj.composite_date} | {obj.hua_object_id} | {region} | {polarity} strict core", fill=(15, 25, 42), font=font(27, True))
    draw.text((44, 64), "Native OFES prho and W. Field contours disabled. Dashed rings: 1.5R/2R local density reference; black path: accepted vertical centers; gold x: current center.", fill=(65, 70, 82), font=font(15))
    left, section_width, map_size, gap = 65, 400, 330, 45
    for row_number, (short_label, full_label, field, signed, unit_label) in enumerate(fields):
        top = 125 + row_number * row_height
        low, high, scale = value_scale(field, signed)
        section_box = (left, top, left + section_width, top + map_size)
        draw_section(draw, image, field, depth, axis, rows, x_track, section_box, f"{short_label} | east-depth", signed, low, high, scale)
        draw.text((section_box[0] + 155, section_box[1] - 25), full_label, fill=(75, 78, 85), font=font(10))
        for col, depth_index in enumerate(selected):
            x0 = left + section_width + gap + col * (map_size + gap)
            map_box = (x0, top, x0 + map_size, top + map_size)
            draw_map(draw, image, field[depth_index], axis, radius_km, x_track, y_track, rows, depth_index, map_box, f"{short_label} | {depth[depth_index]:.1f} m", signed, low, high, scale)
        draw_colorbar(draw, image, (2215, top, 2245, top + map_size), low, high, unit_label)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    image.save(path.with_suffix(".pdf"), "PDF", resolution=160)
    return {"figure_png": str(path), "figure_pdf": str(path.with_suffix('.pdf')), "display_depth_indices": selected}


def load_track(vertical_root: Path, vertical_run_root: Path | None, token: str, object_id: str) -> pd.DataFrame:
    run = vertical_run_root or (vertical_root / "vertical_continuation_local_step_2cells_section_bipolar")
    path = run / "raw_detection" / "daily_runs" / token / "structures_hua_style.csv"
    table = pd.read_csv(path)
    rows = table.loc[table.hua_object_id.astype(str).eq(object_id)].copy()
    if rows.empty:
        raise RuntimeError(f"No accepted vertical track for {object_id} in {path}")
    return rows.drop_duplicates("depth_index").sort_values("depth_index").reset_index(drop=True)


def prepare_selection(args: argparse.Namespace) -> pd.DataFrame:
    selected = pd.read_csv(args.selection_file)
    required = {"composite_date", "hua_object_id", "polarity", "radius_km", "bipolar_fraction_0p6r", "bipolar_fraction_1p0r"}
    missing = required - set(selected.columns)
    if missing:
        raise RuntimeError(f"Selection file missing columns: {sorted(missing)}")
    selected = selected.loc[selected.polarity.astype(str).str.lower().eq(args.polarity)].copy()
    selected["strict_core_min_fraction"] = selected[["bipolar_fraction_0p6r", "bipolar_fraction_1p0r"]].min(axis=1)
    selected["strict_core_pass"] = selected.strict_core_min_fraction.ge(0.70)
    if not bool(selected.strict_core_pass.all()):
        bad = selected.loc[~selected.strict_core_pass, ["composite_date", "hua_object_id", "strict_core_min_fraction"]]
        raise RuntimeError(f"Selection contains non-strict-core rows: {bad.to_dict('records')[:5]}")
    return selected.sort_values(["composite_date", "hua_object_id"]).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    if args.grid_n < 21 or args.extent_r <= 0:
        raise ValueError("grid and extent values are invalid")
    selected = prepare_selection(args)
    if args.max_object_days > 0:
        selected = selected.head(args.max_object_days).copy()
    output_phase = {"raw": "raw_native", "outer-ring-reference": "with_outer_ring_reference", "with-anomaly": "with_annual_mss"}[args.mode]
    phase_root = args.output_root / output_phase
    phase_root.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_root / "selected_object_days_validation.csv", selected.to_dict("records"))
    write_json(args.output_root / "selection_manifest.json", {
        "selection_source": str(args.selection_file), "selected_object_days": int(len(selected)),
        "polarity": args.polarity,
        "strict_core_definition": "min(bipolar_fraction_0p6r, bipolar_fraction_1p0r) >= 0.70",
        "selection_regions": sorted(selected["region"].dropna().astype(str).unique().tolist()) if "region" in selected else [],
    })
    climatology = None
    if args.mode == "with-anomaly":
        if not args.climatology_file.exists():
            raise RuntimeError(f"Annual prho climatology is not available: {args.climatology_file}")
        climatology = np.load(args.climatology_file, mmap_mode="r", allow_pickle=False)
        if climatology.shape != (105, 1520, 3600):
            raise RuntimeError(f"Unexpected climatology shape: {climatology.shape}")

    metas = {name: parse_ctl(ctl_path(args.data_root, name)) for name in ("prho", "w")}
    nlev = min(105, metas["prho"].z.count)
    depth = np.asarray(metas["prho"].z.values[:nlev], dtype="f8")
    results: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    for date_value, group in selected.groupby("composite_date", sort=True):
        token = day_token(date_value)
        day = parse_iso_date(str(date_value))
        paths = {name: require_daily_file(args.data_root, name, day, expected_dta_bytes(meta)) for name, meta in metas.items()}
        raw = {name: open_dta_memmap(paths[name], metas[name]) for name in metas}
        for ordinal, (_, obj) in enumerate(group.iterrows(), start=1):
            object_id = str(obj.hua_object_id)
            object_dir = phase_root / token / object_id
            figure_name = {
                "raw": "raw_prho_native_w_family.png",
                "outer-ring-reference": "raw_prho_outer_ring_anomaly_native_w_family.png",
                "with-anomaly": "raw_prho_density_anomaly_native_w_family.png",
            }[args.mode]
            figure = object_dir / figure_name
            if figure.exists() and figure.with_suffix(".pdf").exists() and not args.overwrite:
                results.append({"composite_date": str(date_value), "hua_object_id": object_id, "status": "skipped_existing", "figure_png": str(figure)})
                continue
            try:
                rows = load_track(args.vertical_root, args.vertical_run_root, token, object_id)
                surface = rows.iloc[0]
                radius_km = float(obj.radius_km if np.isfinite(obj.radius_km) else surface.radius_km)
                axis, xx_km, yy_km, _ = local_coordinates(radius_km, args.extent_r, args.grid_n)
                lon, lat = local_lon_lat_grid(xx_km / radius_km, yy_km / radius_km, float(surface.center_lon_refined), float(surface.center_lat_refined), radius_km)
                prho = sample_stack(raw["prho"], metas["prho"], lon, lat, nlev)
                source_levels = min(metas["w"].z.count, nlev + 1)
                w_raw = sample_stack(raw["w"], metas["w"], lon, lat, source_levels) / 100.0
                native_w, alignment = align_native_w_vertical(w_raw, np.asarray(metas["w"].z.values[:source_levels], dtype="f8"), depth, "layer_center")
                anomaly = None if climatology is None else prho - sample_climatology_stack(climatology, metas["prho"], lon, lat, nlev)
                rendered = render_panel(figure, obj, rows, axis, radius_km, prho, native_w, depth, anomaly, args.mode == "outer-ring-reference")
                record = {
                    "composite_date": str(date_value), "hua_object_id": object_id, "status": "rendered", "surface_lon": float(surface.center_lon_refined),
                    "surface_lat": float(surface.center_lat_refined), "surface_radius_km": radius_km, "accepted_layers": int(len(rows)),
                    "max_depth_m": float(rows.depth_m.max()), "raw_prho_p02": float(np.nanpercentile(prho, 2)),
                    "raw_prho_p98": float(np.nanpercentile(prho, 98)), "native_w_abs_p98_m_s": finite_percentile(np.abs(native_w), 98, 1.0e-10),
                    "anomaly_abs_p98": finite_percentile(np.abs(anomaly), 98, 1.0e-10) if anomaly is not None else np.nan,
                    **alignment, **rendered,
                }
                results.append(record)
                print(f"[strict-core-fields] {token} {ordinal}/{len(group)} {object_id} rendered", flush=True)
            except Exception as exc:
                failures.append({"composite_date": str(date_value), "hua_object_id": object_id, "error": repr(exc)})
                print(f"[strict-core-fields] {token} {object_id} failed: {exc}", flush=True)
            write_csv(phase_root / "object_summary.csv", results)
            write_csv(phase_root / "failures.csv", failures)
    manifest = {
        "phase": output_phase, "input_object_days": int(len(selected)), "rendered_or_existing": len(results), "failed": len(failures),
        "source_vertical_root": str(args.vertical_root), "source_selection": str(args.selection_file), "data_root": str(args.data_root),
        "density_definition": "native OFES prho; no spatial or temporal filter", "native_w_definition": "native OFES w / 100, interpolated to prho layer centers; no high-pass or rebuild-W",
        "density_anomaly_definition": (
            "raw prho - day-weighted 1993-2012 annual prho MSS" if climatology is not None
            else "outer-ring-relative density: raw prho - same-depth median in 1.5R<=r<=2R; not a climatological density anomaly" if args.mode == "outer-ring-reference"
            else "not generated in raw phase"
        ),
        "climatology_file": str(args.climatology_file) if climatology is not None else None, "grid_n": args.grid_n, "extent_r": args.extent_r,
        "map_projection": "local equidistant approximation with equal x/y kilometre axes", "axes": "kilometres horizontally and vertically; depth in metres", "contours": "disabled (linestyle=NaN equivalent)",
    }
    write_json(phase_root / "manifest.json", manifest)
    print(json.dumps({"phase": output_phase, "rendered_or_existing": len(results), "failed": len(failures), "output": str(phase_root)}))


if __name__ == "__main__":
    main()
