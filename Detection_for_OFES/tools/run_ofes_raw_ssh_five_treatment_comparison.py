"""Build and plot five Jan1 OFES raw-SSH preprocessing treatments.

The program is intentionally isolated from production catalog roots.  It uses
Jan1-Jan5 only to support the final Jan1 persistence decision.
"""

from __future__ import annotations

import argparse
import calendar
import json
import math
import subprocess
import sys
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


def build_base_inputs(args: argparse.Namespace, all_days: list[date], treatment: Treatment, h_mss: np.ndarray, eta_lon: np.ndarray, eta_lat: np.ndarray) -> Path:
    root = args.output_root / treatment.key / "base_inputs"
    for current in all_days:
        destination = root / f"global_phy_{current:%Y%m%d}.nc"
        if destination.exists() and not args.rerun_existing:
            continue
        eta = np.asarray(read_variable_latlon_daily_only(args.data_root, "eta", current), dtype="f4")
        pair = np.asarray(read_variable_latlon_daily_only(args.data_root, "pressur", current), dtype="f4")
        raw_h = eta - (pair - 1000.0)
        raw_h[~np.isfinite(eta) | ~np.isfinite(pair)] = np.nan
        native = raw_h if treatment.baseline == "raw_h" else raw_h - h_mss
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
        mode_args = ["--filter-mode", "bandpass", "--large-cutoff-mode", "rossby_lower_latadaptive_upper", "--rossby-radius-path", str(args.rossby_radius_path), "--rossby-small-factor", "0.5", "--rossby-min-km", "10", "--rossby-max-km", "180", "--adaptive-large-cutoff-min-km", "180", "--adaptive-large-cutoff-max-km", "180", "--science-tag", f"annual_h_mss_rossby_dog_{treatment.kernel}"]
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


def draw_seed_layer(draw: ImageDraw.ImageDraw, rows: pd.DataFrame, box: tuple[int, int, int, int], bbox: tuple[float, float, float, float]) -> None:
    lo0, lo1, la0, la1 = bbox
    x0, y0, x1, y1 = box
    for _, row in rows.iterrows():
        lon = float(row.get("seed_lon", row.get("center_lon", np.nan)))
        lat = float(row.get("seed_lat", row.get("center_lat", np.nan)))
        if not (np.isfinite(lon) and np.isfinite(lat)):
            continue
        x, y = px(lon, lo0, lo1, x0, x1), py(lat, la0, la1, y0, y1)
        color = polarity_color(row)
        draw.line((x - 3, y, x + 3, y), fill=color, width=1)
        draw.line((x, y - 3, x, y + 3), fill=color, width=1)


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
        draw_seed_layer(draw, selected, box, bbox)
        layer_title = f"SSH candidate seeds: {len(selected)}"
    else:
        draw_final_layer(draw, selected, box, bbox)
        layer_title = f"Final QC + persistence objects: {len(selected)}"
    cyclonic = int(selected.get("polarity", pd.Series(dtype=str)).astype(str).eq("cyclonic").sum())
    anticyclonic = int(selected.get("polarity", pd.Series(dtype=str)).astype(str).eq("anticyclonic").sum())
    draw.text((110, 32), title, fill=(18, 24, 38), font=font(34))
    draw.text((110, 76), f"{layer_title}; cyclonic={cyclonic}, anticyclonic={anticyclonic}; raw surface u/v, {vector_spacing:g} degree sampling", fill=(65, 75, 90), font=font(18))
    draw.text((1980, 130), f"SSH\n+/-{limit:.1f} cm", fill=(35, 45, 60), font=font(18))
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)
    image.save(output.with_suffix(".pdf"), "PDF", resolution=180.0)
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


def main() -> None:
    args = parse_args()
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
