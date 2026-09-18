from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

from .ofes_io import (
    ctl_path,
    expected_dta_bytes,
    open_dta_memmap,
    parse_ctl,
    require_daily_file,
)
from .tools.extract_ofes_daily import extract_w_prho
from .w_rebuild_config import DEFAULT_DATA_ROOT, DEFAULT_RESULT_ROOT, config_from_args


EARTH_RADIUS_M = 6_371_000.0
SCIENCE_TAG = "raw_minus_jan_mean_diagnostic"


@dataclass(frozen=True)
class SelectedObject:
    hua_object_id: str
    date: str
    polarity: str
    pass_layers: int
    max_jump_km: float
    max_jump_over_r: float
    radius_km: float
    center_lon: float
    center_lat: float


def main() -> None:
    cfg = config_from_args(build_parser().parse_args())
    args = cfg.to_runtime_args()
    data_root = cfg.io.data_root
    result_root = cfg.io.result_root
    output_root = cfg.output_root
    metadata_dir = output_root / "metadata"
    grids_dir = output_root / "grids"
    figures_dir = output_root / "figures"
    for path in [metadata_dir, grids_dir, figures_dir]:
        path.mkdir(parents=True, exist_ok=True)

    days = date_range(parse_iso_date(args.start), parse_iso_date(args.end))
    stages = parse_stage_set(str(args.stages))
    if "extract" in stages:
        extract_w_prho(data_root, metadata_dir, days, int(args.extract_workers))
    if stages & {"diagnose", "crossing", "band", "crossing_composite"}:
        run_w_diagnostics(data_root, result_root, output_root, grids_dir, figures_dir, args, stages)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare OFES native W against Dipole-style rebuilt W.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--stages", default="all", help="Comma-separated: extract,diagnose,crossing,band,crossing_composite,all. all runs extract plus the default band diagnostic.")
    parser.add_argument("--start", default="1991-01-01")
    parser.add_argument("--end", default="1991-01-19")
    parser.add_argument("--date", default="1991-01-10", help="Diagnostic object date.")
    parser.add_argument("--hua-object-id", default=None)
    parser.add_argument("--max-objects", type=int, default=8)
    parser.add_argument("--max-depth-layers", type=int, default=105)
    parser.add_argument("--grid-n", type=int, default=81)
    parser.add_argument("--extent-r", type=float, default=4.0)
    parser.add_argument("--farfield-inner-r", type=float, default=2.0)
    parser.add_argument("--farfield-outer-r", type=float, default=4.0)
    parser.add_argument("--smooth-sigma-cells", type=float, default=1.0)
    parser.add_argument("--rho-bg-source", choices=["farfield_ring", "regional_box"], default="regional_box")
    parser.add_argument("--regional-bg-lon-half-width-deg", type=float, default=5.0)
    parser.add_argument("--regional-bg-lat-half-width-deg", type=float, default=2.0)
    parser.add_argument("--regional-bg-exclude-r", type=float, default=1.5)
    parser.add_argument("--regional-bg-min-valid-fraction", type=float, default=0.25)
    parser.add_argument("--disable-rho-stabilization", action="store_true", help="Use the legacy raw rho_bg/rho_z/eta_rho calculation.")
    parser.add_argument("--rho-bg-smooth-sigma-layers", type=float, default=2.0)
    parser.add_argument("--rho-z-min", type=float, default=2.0e-5)
    parser.add_argument("--eta-rho-cap-m", type=float, default=500.0)
    parser.add_argument("--eta-rho-cap-mode", choices=["fixed"], default="fixed")
    parser.add_argument("--bad-rho-z-mode", choices=["mask"], default="mask")
    parser.add_argument("--farfield-valid-min-fraction", type=float, default=0.35)
    parser.add_argument("--rebuild-formula", choices=["relative_advection", "legacy_double_c"], default="relative_advection", help="Default uses W = c_rel dot grad(eta_rho) - U_rel dot grad(eta_rho).")
    parser.add_argument("--velocity-reference", choices=["farfield_relative", "absolute"], default="farfield_relative", help="Use farfield-relative horizontal velocity and translation speed by default.")
    parser.add_argument("--translation-profile", choices=["barotropic", "layerwise", "layer_tracking"], default="layer_tracking", help="Use same-depth center tracking for c_abs(z) by default.")
    parser.add_argument("--layer-tracking-max-distance-r", type=float, default=2.0, help="Maximum same-depth center matching distance in eddy radii.")
    parser.add_argument("--layer-tracking-max-distance-km", type=float, default=250.0, help="Maximum same-depth center matching distance in km.")
    parser.add_argument("--native-w-temporal-filter", choices=["none", "lowpass_running_mean"], default="none", help="Optional temporal filter for OFES native W reference only.")
    parser.add_argument("--native-w-filter-window-days", type=int, default=10, help="Running-mean window used when --native-w-temporal-filter=lowpass_running_mean.")
    parser.add_argument("--rebuild-density-filter", choices=["none", "joint_lowpass", "bg_perturb_decomp"], default="joint_lowpass", help="Scale separation applied to the density path used by rebuild W.")
    parser.add_argument("--rebuild-density-filter-window-days", type=int, default=10)
    parser.add_argument("--eta-horizontal-lowpass-sigma-r", type=float, default=0.5)
    parser.add_argument("--density-bg-detrend-order", type=int, choices=[0, 1], default=1)
    parser.add_argument("--density-bg-detrend-fit-ring", default="1.5,4.0", help="Inner,outer radii in R for per-layer density trend fitting.")
    parser.add_argument("--rebuild-velocity-filter", choices=["none", "joint_lowpass"], default="joint_lowpass", help="Scale separation applied to u/v and c(z) used by rebuild W.")
    parser.add_argument("--rebuild-velocity-filter-window-days", type=int, default=10)
    parser.add_argument("--uv-horizontal-lowpass-sigma-r", type=float, default=0.5)
    parser.add_argument("--translation-profile-smooth-sigma-layers", type=float, default=2.0)
    parser.add_argument("--translation-profile-max-speed-m-s", type=float, default=0.5)
    parser.add_argument("--translation-background-layers", type=int, default=10, help="Upper farfield layers used to remove background advection from eddy translation speed.")
    parser.add_argument("--native-w-meso-filter", choices=["none", "temporal10d_spatial50_500km"], default="temporal10d_spatial50_500km", help="Native-W field used by multipole classification and native-W composites.")
    parser.add_argument("--native-w-meso-time-window-days", type=int, default=10, help="Available-day running mean window for the native-W mesoscale diagnostic.")
    parser.add_argument("--native-w-meso-small-cutoff-km", type=float, default=50.0, help="Approximate small-scale cutoff; implemented as Gaussian FWHM before subtracting the large-scale field.")
    parser.add_argument("--native-w-meso-large-cutoff-km", type=float, default=500.0, help="Approximate large-scale cutoff; implemented as Gaussian FWHM and subtracted from the small-scale lowpass.")
    parser.add_argument("--native-w-phase-align", action=argparse.BooleanOptionalAction, default=True, help="Rotate dipole native-W mesoscale fields by their mode-1 phase before compositing.")
    parser.add_argument("--extract-workers", type=int, default=1)
    parser.add_argument("--backend", choices=["matplotlib", "pillow"], default="pillow")
    parser.add_argument("--crossing-lats", default="20,40", help="Comma-separated target latitudes for x-depth W section diagnostics.")
    parser.add_argument("--intersect-radius-r", type=float, default=1.0, help="Strict crossing radius threshold in R.")
    parser.add_argument("--strict-crossing-only", action="store_true", help="Skip a target latitude when no object crosses within intersect-radius-r.")
    parser.add_argument("--band-lat-min", type=float, default=30.0, help="Minimum latitude for match_all band diagnostics.")
    parser.add_argument("--band-lat-max", type=float, default=35.0, help="Maximum latitude for match_all band diagnostics.")
    parser.add_argument("--band-max-objects", type=int, default=8, help="Maximum match_all band objects to plot; use <=0 for all.")
    parser.add_argument("--composite-lat", type=float, default=20.0, help="Target latitude for strict crossing Cressman composites.")
    parser.add_argument("--composite-polarities", default="cyclonic,anticyclonic", help="Comma-separated polarities for crossing composites.")
    parser.add_argument("--composite-max-objects", type=int, default=0, help="Maximum strict-crossing objects per polarity; use <=0 for all.")
    parser.add_argument("--cressman-radius-r", type=float, default=1.0, help="Cressman influence radius in eddy-radius units for composites.")
    parser.add_argument("--cressman-min-objects", type=int, default=8, help="Minimum contributing objects required for a composite cell.")
    parser.add_argument("--composite-workers", type=int, default=1, help="Parallel object rebuild workers for crossing composites.")
    parser.add_argument("--composite-selection-mode", choices=["crossing", "domain_bbox"], default="crossing", help="Select composite source objects by strict crossing or by a lon/lat domain bbox.")
    parser.add_argument("--composite-domain-bbox", default="120,145,20,35", help="Domain bbox for --composite-selection-mode=domain_bbox: lon_min,lon_max,lat_min,lat_max.")
    parser.add_argument("--composite-domain-name", default="kuroshio_domain", help="Output label for --composite-selection-mode=domain_bbox.")
    parser.add_argument("--multipole-depth-min-m", type=float, default=300.0, help="Upper bound of native-W depth range used for multipole classification.")
    parser.add_argument("--multipole-depth-max-m", type=float, default=500.0, help="Lower bound of native-W depth range used for multipole classification.")
    parser.add_argument("--multipole-radius-inner-r", type=float, default=0.0, help="Inner radius for native-W radial integration used by multipole classification.")
    parser.add_argument("--multipole-radius-outer-r", type=float, default=1.0, help="Outer radius for native-W radial integration used by multipole classification.")
    parser.add_argument("--multipole-azimuth-count", type=int, default=60, help="Number of azimuth sectors used by native-W multipole classification.")
    parser.add_argument("--multipole-min-valid-azimuth-fraction", type=float, default=0.80, help="Minimum fraction of valid azimuth sectors required by multipole QC.")
    parser.add_argument("--multipole-min-sector-valid-fraction", type=float, default=0.35, help="Minimum valid grid fraction inside an azimuth sector.")
    parser.add_argument("--multipole-boundary-max-nan-fraction", type=float, default=0.35, help="Maximum NaN fraction allowed near the outer classification radius.")
    parser.add_argument("--multipole-min-amp-1e6-m-s", type=float, default=0.5, help="Minimum azimuthal native-W amplitude in 10^-6 m/s.")
    parser.add_argument("--multipole-min-snr", type=float, default=2.0, help="Minimum azimuthal signal-to-noise ratio for native-W multipole classification.")
    parser.add_argument("--multipole-min-harmonic-dominance", type=float, default=1.1, help="Minimum dominance of the selected azimuthal harmonic over other low modes.")
    parser.add_argument("--composite-selection-class", default="", help="Optional native-W multipole class to keep for composite output, e.g. dipole.")
    parser.add_argument("--composite-region-mode", choices=["none", "longitude_bins"], default="none", help="Optional regional grouping for crossing composites.")
    parser.add_argument("--composite-region-boxes", default="western_boundary:60,140;interior:140,240;eastern_basin:240,360", help="Semicolon-separated region:lon_min,lon_max boxes in 0-360 degrees. Multiple boxes per region use |.")
    parser.add_argument("--composite-combine-polarities", action="store_true", help="Combine requested polarities before regional composite output.")
    return parser


def parse_stage_set(value: str) -> set[str]:
    requested = {item.strip().lower() for item in value.split(",") if item.strip()}
    if not requested:
        raise ValueError("--stages must include at least one stage")
    valid = {"extract", "diagnose", "crossing", "band", "crossing_composite", "all"}
    invalid = sorted(requested - valid)
    if invalid:
        raise ValueError(f"Unsupported --stages value(s): {', '.join(invalid)}. Valid stages: {', '.join(sorted(valid))}")
    if "all" in requested:
        return {"extract", "band"}
    return requested


def run_w_diagnostics(data_root: Path, result_root: Path, output_root: Path, grids_dir: Path, figures_dir: Path, args: argparse.Namespace, stages: set[str]) -> None:
    if "diagnose" in stages:
        _run_object_diagnostics(data_root, result_root, output_root, grids_dir, figures_dir, args)
    if "crossing" in stages:
        _run_crossing_diagnostics(data_root, result_root, output_root, grids_dir, figures_dir, args)
    if "band" in stages:
        _run_band_diagnostics(data_root, result_root, output_root, grids_dir, figures_dir, args)
    if "crossing_composite" in stages:
        _run_crossing_composite_diagnostics(data_root, result_root, output_root, args)


def _run_object_diagnostics(data_root: Path, result_root: Path, output_root: Path, grids_dir: Path, figures_dir: Path, args: argparse.Namespace) -> None:
    target_day = parse_iso_date(args.date)
    detection_dir = resolve_detection_table_dir(result_root)
    centers, structures = load_detection_tables(detection_dir)
    selected = select_objects(centers, structures, target_day, args.hua_object_id, int(args.max_objects))

    metas = {name: parse_ctl(ctl_path(data_root, name)) for name in ["u", "v", "w", "prho"]}
    expected = {name: expected_dta_bytes(meta) for name, meta in metas.items()}
    paths = {name: require_daily_file(data_root, name, target_day, expected[name]) for name in ["u", "v", "w", "prho"]}
    raw = {name: open_dta_memmap(paths[name], metas[name]) for name in ["u", "v", "w", "prho"]}
    prev_next = open_neighbor_center_tables(centers, target_day)

    rows = []
    for idx, obj in enumerate(selected, start=1):
        print(f"[w-rebuild] {idx}/{len(selected)} {obj.hua_object_id}", flush=True)
        grid = rebuild_object_w(raw, metas, centers, obj, prev_next, args)
        suffix = rho_stabilization_suffix(args)
        stem = f"ofes_w_rebuild_4panel_{target_day:%Y%m%d}_{obj.hua_object_id}{suffix}"
        npz_path = grids_dir / f"w_rebuild_grid_{target_day:%Y%m%d}_{obj.hua_object_id}{suffix}.npz"
        json_path = grids_dir / f"w_rebuild_grid_{target_day:%Y%m%d}_{obj.hua_object_id}{suffix}.json"
        np.savez_compressed(npz_path, **{key: value for key, value in grid.items() if isinstance(value, np.ndarray)})
        scalar_payload = {key: json_safe(value) for key, value in grid.items() if not isinstance(value, np.ndarray)}
        write_json(json_path, scalar_payload)
        figure_path = figures_dir / f"{stem}.png"
        pdf_path = figures_dir / f"{stem}.pdf"
        plot_four_panel(grid, figure_path, pdf_path, backend=str(args.backend))
        rows.append({"image_path": str(figure_path), "grid_npz": str(npz_path), "grid_json": str(json_path), **scalar_payload})
    pd.DataFrame(rows).to_csv(output_root / "run_summary.csv", index=False)
    write_json(output_root / "run_summary.json", rows)
    write_method_doc(output_root / "METHOD_OFES_REBUILD_W_ZH.md", args, rows)


def _run_crossing_diagnostics(data_root: Path, result_root: Path, output_root: Path, grids_dir: Path, figures_dir: Path, args: argparse.Namespace) -> None:
    detection_dir = resolve_detection_table_dir(result_root)
    centers, structures = load_detection_tables(detection_dir)
    target_lats = parse_float_list(str(args.crossing_lats))
    selections = select_crossing_objects(structures, target_lats, float(args.intersect_radius_r), bool(args.strict_crossing_only))
    section_dir = figures_dir / "crossing_sections"
    section_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    metas = {name: parse_ctl(ctl_path(data_root, name)) for name in ["u", "v", "w", "prho"]}
    expected = {name: expected_dta_bytes(meta) for name, meta in metas.items()}
    raw_cache: dict[str, dict[str, np.memmap]] = {}
    for selection in selections:
        target_day = parse_iso_date(str(selection["date"]))
        day_key = target_day.isoformat()
        if day_key not in raw_cache:
            paths = {name: require_daily_file(data_root, name, target_day, expected[name]) for name in ["u", "v", "w", "prho"]}
            raw_cache[day_key] = {name: open_dta_memmap(paths[name], metas[name]) for name in ["u", "v", "w", "prho"]}
        obj = selection["object"]
        print(f"[crossing] {lat_label(float(selection['target_lat']))} {selection['selection_mode']} {obj.hua_object_id}", flush=True)
        neighbors = open_neighbor_center_tables(centers, target_day)
        grid = rebuild_object_w(raw_cache[day_key], metas, centers, obj, neighbors, args)
        grid.update(
            {
                "target_lat": float(selection["target_lat"]),
                "crossing_distance_km": float(selection["distance_km"]),
                "crossing_distance_over_r": float(selection["distance_over_r"]),
                "crossing_selection_mode": str(selection["selection_mode"]),
                "crossing_status": str(selection["crossing_status"]),
                "strict_crossing": bool(selection["strict_crossing"]),
                "nearest_fallback": bool(selection["nearest_fallback"]),
                "skipped_no_crossing": bool(selection["skipped_no_crossing"]),
                "intersect_radius_r": float(args.intersect_radius_r),
            }
        )
        suffix = rho_stabilization_suffix(args)
        stem = f"ofes_w_crossing_4panel_{lat_token(float(selection['target_lat']))}_{target_day:%Y%m%d}_{obj.hua_object_id}{suffix}"
        npz_path = grids_dir / f"w_crossing_grid_{lat_token(float(selection['target_lat']))}_{target_day:%Y%m%d}_{obj.hua_object_id}{suffix}.npz"
        json_path = grids_dir / f"w_crossing_grid_{lat_token(float(selection['target_lat']))}_{target_day:%Y%m%d}_{obj.hua_object_id}{suffix}.json"
        np.savez_compressed(npz_path, **{key: value for key, value in grid.items() if isinstance(value, np.ndarray)})
        scalar_payload = {key: json_safe(value) for key, value in grid.items() if not isinstance(value, np.ndarray)}
        write_json(json_path, scalar_payload)
        png_path = section_dir / f"{stem}.png"
        plot_crossing_four_panel_pillow(grid, png_path)
        rows.append({"image_path": str(png_path), "grid_npz": str(npz_path), "grid_json": str(json_path), **scalar_payload})

    write_csv(output_root / "crossing_summary.csv", rows)
    write_json(output_root / "crossing_summary.json", rows)


def _run_band_diagnostics(data_root: Path, result_root: Path, output_root: Path, grids_dir: Path, figures_dir: Path, args: argparse.Namespace) -> None:
    detection_dir = resolve_detection_table_dir(result_root)
    centers, structures = load_detection_tables(detection_dir)
    lat_min = float(args.band_lat_min)
    lat_max = float(args.band_lat_max)
    objects = select_band_objects(structures, lat_min, lat_max, int(args.band_max_objects))
    if not objects:
        raise ValueError(f"No structured OFES eddies found in latitude band {lat_min:g} to {lat_max:g}.")

    token = f"{lat_token(lat_min)}_{lat_token(lat_max)}"
    section_dir = figures_dir / f"band_{token}_sections"
    horizontal_dir = figures_dir / f"band_{token}_horizontal"
    section_dir.mkdir(parents=True, exist_ok=True)
    horizontal_dir.mkdir(parents=True, exist_ok=True)

    metas = {name: parse_ctl(ctl_path(data_root, name)) for name in ["u", "v", "w", "prho"]}
    expected = {name: expected_dta_bytes(meta) for name, meta in metas.items()}
    raw_cache: dict[str, dict[str, np.memmap]] = {}
    rows = []
    for idx, obj in enumerate(objects, start=1):
        target_day = parse_iso_date(obj.date)
        day_key = target_day.isoformat()
        if day_key not in raw_cache:
            paths = {name: require_daily_file(data_root, name, target_day, expected[name]) for name in ["u", "v", "w", "prho"]}
            raw_cache[day_key] = {name: open_dta_memmap(paths[name], metas[name]) for name in ["u", "v", "w", "prho"]}
        print(f"[band] {idx}/{len(objects)} {lat_min:g}-{lat_max:g}N {obj.hua_object_id}", flush=True)
        neighbors = open_neighbor_center_tables(centers, target_day)
        grid = rebuild_object_w(raw_cache[day_key], metas, centers, obj, neighbors, args)
        grid.update(
            {
                "target_lat": float(obj.center_lat),
                "band_lat_min": lat_min,
                "band_lat_max": lat_max,
                "band_selection_mode": "match_all_center_in_band",
                "crossing_selection_mode": "match_all_center_slice",
                "crossing_distance_km": 0.0,
                "crossing_distance_over_r": 0.0,
                "intersect_radius_r": float("nan"),
            }
        )
        suffix = rho_stabilization_suffix(args)
        stem = f"ofes_w_band_match_all_4panel_{token}_{target_day:%Y%m%d}_{obj.hua_object_id}{suffix}"
        npz_path = grids_dir / f"w_band_match_all_grid_{token}_{target_day:%Y%m%d}_{obj.hua_object_id}{suffix}.npz"
        json_path = grids_dir / f"w_band_match_all_grid_{token}_{target_day:%Y%m%d}_{obj.hua_object_id}{suffix}.json"
        np.savez_compressed(npz_path, **{key: value for key, value in grid.items() if isinstance(value, np.ndarray)})
        scalar_payload = {key: json_safe(value) for key, value in grid.items() if not isinstance(value, np.ndarray)}
        write_json(json_path, scalar_payload)

        section_png = section_dir / f"{stem}_slice.png"
        horizontal_png = horizontal_dir / f"{stem}_horizontal.png"
        plot_crossing_four_panel_pillow(grid, section_png)
        plot_four_panel_pillow(grid, horizontal_png)
        rows.append(
            {
                "slice_image_path": str(section_png),
                "horizontal_image_path": str(horizontal_png),
                "grid_npz": str(npz_path),
                "grid_json": str(json_path),
                **scalar_payload,
            }
        )

    suffix = rho_stabilization_suffix(args)
    write_csv(output_root / f"band_match_all_summary_{token}{suffix}.csv", rows)
    write_json(output_root / f"band_match_all_summary_{token}{suffix}.json", rows)


def resolve_detection_table_dir(result_root: Path) -> Path:
    candidates = [
        result_root,
        result_root / "hua_b3_start2_detection",
        result_root / "detection_hua_global_jan1991",
    ]
    for candidate in candidates:
        if detection_table_path(candidate, "centers_hua_style") and detection_table_path(candidate, "structures_hua_style"):
            return candidate
    checked = "; ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "Could not find centers_hua_style and structures_hua_style tables. "
        f"Checked: {checked}"
    )


def detection_table_path(directory: Path, stem: str) -> Path | None:
    for suffix in [".parquet", ".csv"]:
        path = directory / f"{stem}{suffix}"
        if path.exists():
            return path
    return None


def read_detection_table(directory: Path, stem: str) -> pd.DataFrame:
    path = detection_table_path(directory, stem)
    if path is None:
        raise FileNotFoundError(f"Missing {stem}.parquet or {stem}.csv under {directory}")
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def load_detection_tables(detection_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    centers = read_detection_table(detection_dir, "centers_hua_style")
    structures = read_detection_table(detection_dir, "structures_hua_style")
    for table in [centers, structures]:
        if "date" in table.columns:
            table["date"] = pd.to_datetime(table["date"]).dt.strftime("%Y-%m-%d")
    return centers, structures


def _run_crossing_composite_diagnostics(data_root: Path, result_root: Path, output_root: Path, args: argparse.Namespace) -> None:
    detection_dir = resolve_detection_table_dir(result_root)
    centers, structures = load_detection_tables(detection_dir)
    selection_mode = str(getattr(args, "composite_selection_mode", "crossing")).lower()
    domain_bbox: tuple[float, float, float, float] | None = None
    if selection_mode == "domain_bbox":
        domain_bbox = parse_bbox(str(getattr(args, "composite_domain_bbox", "120,145,20,35")))
        target_lat = 0.5 * (domain_bbox[2] + domain_bbox[3])
        token = safe_token(str(getattr(args, "composite_domain_name", "kuroshio_domain")))
    else:
        target_lat = float(args.composite_lat)
        token = lat_token(target_lat)
    polarities = parse_string_list(str(args.composite_polarities))
    if not polarities:
        raise ValueError("--composite-polarities must include at least one polarity")

    composite_root = output_root / "w_native_meso50_500km_mode1_aligned_by_polarity" / f"composite_{token}_crossing"
    figures_dir = composite_root / "figures"
    grids_dir = composite_root / "grids"
    figures_dir.mkdir(parents=True, exist_ok=True)
    grids_dir.mkdir(parents=True, exist_ok=True)

    metas = {name: parse_ctl(ctl_path(data_root, name)) for name in ["u", "v", "w", "prho"]}
    expected = {name: expected_dta_bytes(meta) for name, meta in metas.items()}
    raw_cache: dict[str, dict[str, np.memmap]] = {}
    summary_rows: list[dict[str, object]] = []
    detail_rows: list[dict[str, object]] = []
    workers = max(1, int(getattr(args, "composite_workers", 1)))

    def build_object_grid(item: tuple[int, SelectedObject]) -> tuple[int, SelectedObject, dict[str, object]]:
        idx, obj = item
        target_day = parse_iso_date(obj.date)
        day_key = target_day.isoformat()
        if workers <= 1:
            if day_key not in raw_cache:
                paths = {name: require_daily_file(data_root, name, target_day, expected[name]) for name in ["u", "v", "w", "prho"]}
                raw_cache[day_key] = {name: open_dta_memmap(paths[name], metas[name]) for name in ["u", "v", "w", "prho"]}
            raw = raw_cache[day_key]
        else:
            paths = {name: require_daily_file(data_root, name, target_day, expected[name]) for name in ["u", "v", "w", "prho"]}
            raw = {name: open_dta_memmap(paths[name], metas[name]) for name in ["u", "v", "w", "prho"]}
        neighbors = open_neighbor_center_tables(centers, target_day)
        grid = rebuild_object_w(raw, metas, centers, obj, neighbors, args)
        distance_km, distance_over_r = crossing_distance_for_object(obj, target_lat)
        grid.update(
            {
                "target_lat": target_lat,
                "crossing_distance_km": distance_km,
                "crossing_distance_over_r": distance_over_r,
                "crossing_selection_mode": "domain_bbox_composite" if selection_mode == "domain_bbox" else "strict_1r_crossing_composite",
                "crossing_status": "domain_bbox" if selection_mode == "domain_bbox" else "strict_crossing",
                "strict_crossing": selection_mode != "domain_bbox",
                "nearest_fallback": False,
                "skipped_no_crossing": False,
                "intersect_radius_r": float(args.intersect_radius_r),
            }
        )
        grid.update(classify_native_w_multipole(grid, args))
        grid.update(align_native_w_to_dipole_phase(grid, args))
        return idx, obj, grid

    if selection_mode == "domain_bbox":
        assert domain_bbox is not None
        selection_class = str(getattr(args, "composite_selection_class", "")).strip().lower()
        if not selection_class:
            selection_class = "dipole"
        domain_name = str(getattr(args, "composite_domain_name", "kuroshio_domain")).strip() or "domain_bbox"
        composite_root = output_root / "w_native_meso50_500km_mode1_aligned_by_polarity" / f"composite_{token}_domain"
        figures_dir = composite_root / "figures"
        grids_dir = composite_root / "grids"
        figures_dir.mkdir(parents=True, exist_ok=True)
        grids_dir.mkdir(parents=True, exist_ok=True)

        object_pool: list[SelectedObject] = []
        for polarity in polarities:
            object_pool.extend(
                select_domain_objects(
                    structures,
                    bbox=domain_bbox,
                    polarity=polarity,
                    max_objects=int(args.composite_max_objects),
                )
            )
        object_pool.sort(key=lambda item: (item.date, item.hua_object_id))
        print(
            f"[domain-composite] {domain_name}: {len(object_pool)} objects in bbox {format_bbox(domain_bbox)} before class filter; "
            f"class={selection_class}; workers={workers}",
            flush=True,
        )

        all_selected_accumulator: dict[str, object] | None = None
        polarity_accumulators: dict[str, dict[str, object]] = {}
        domain_detail_rows: list[dict[str, object]] = []
        domain_summary_rows: list[dict[str, object]] = []

        def consume_domain_grid(idx: int, obj: SelectedObject, grid: dict[str, object]) -> None:
            nonlocal all_selected_accumulator
            multipole_class = str(grid.get("multipole_class", "")).lower()
            selected = multipole_class == selection_class
            distance_km = float(grid["crossing_distance_km"])
            distance_over_r = float(grid["crossing_distance_over_r"])
            row = object_detail_row(target_lat, obj.polarity, obj, grid, distance_km, distance_over_r)
            row.update(
                {
                    "region": domain_name,
                    "region_mode": "domain_bbox",
                    "domain_bbox": format_bbox(domain_bbox),
                    "selected_for_region_composite": bool(selected),
                    "selection_class": selection_class,
                    "combine_polarities": False,
                }
            )
            domain_detail_rows.append(row)
            print(
                f"[domain-composite] {idx}/{len(object_pool)} {obj.hua_object_id} "
                f"{obj.polarity} {multipole_class} selected={selected}",
                flush=True,
            )
            if not selected:
                return
            if all_selected_accumulator is None:
                all_selected_accumulator = init_composite_accumulator(grid, target_lat, "all_polarities", args, multipole_class=selection_class)
                all_selected_accumulator["region"] = domain_name
                all_selected_accumulator["region_boxes"] = format_bbox(domain_bbox)
            update_composite_accumulator(all_selected_accumulator, grid, args)
            if obj.polarity not in polarity_accumulators:
                polarity_accumulators[obj.polarity] = init_composite_accumulator(grid, target_lat, obj.polarity, args, multipole_class=selection_class)
                polarity_accumulators[obj.polarity]["region"] = domain_name
                polarity_accumulators[obj.polarity]["region_boxes"] = format_bbox(domain_bbox)
            update_composite_accumulator(polarity_accumulators[obj.polarity], grid, args)

        if workers > 1:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [executor.submit(build_object_grid, item) for item in enumerate(object_pool, start=1)]
                for future in as_completed(futures):
                    idx, obj, grid = future.result()
                    consume_domain_grid(idx, obj, grid)
        else:
            for item in enumerate(object_pool, start=1):
                idx, obj, grid = build_object_grid(item)
                consume_domain_grid(idx, obj, grid)

        if all_selected_accumulator is None:
            domain_summary_rows.append(
                {
                    "target_lat": target_lat,
                    "polarity": "all_polarities",
                    "region": domain_name,
                    "object_count": 0,
                    "selection_class": selection_class,
                    "domain_bbox": format_bbox(domain_bbox),
                    "status": "no_selected_objects",
                }
            )
        else:
            reference_composite = finalize_composite_accumulator(all_selected_accumulator, args)
            reference_composite["region"] = domain_name
            reference_composite["selection_class"] = selection_class
            for polarity, accumulator in sorted(polarity_accumulators.items()):
                composite = finalize_composite_accumulator(accumulator, args)
                composite["region"] = domain_name
                composite["selection_class"] = selection_class
                composite["region_boxes"] = format_bbox(domain_bbox)
                attach_reference_native(composite, reference_composite)
                composite["reference_multipole_class"] = f"{selection_class}_all_polarities"
                domain_summary_rows.append(write_region_composite_outputs(composite, token, domain_name, polarity, selection_class, grids_dir, figures_dir))

        write_csv(composite_root / f"composite_{token}_domain_summary.csv", domain_summary_rows)
        write_json(composite_root / f"composite_{token}_domain_summary.json", domain_summary_rows)
        write_csv(composite_root / f"composite_{token}_domain_objects.csv", domain_detail_rows)
        return

    if str(getattr(args, "composite_region_mode", "none")).lower() == "longitude_bins":
        regions = parse_composite_region_boxes(str(getattr(args, "composite_region_boxes", "")))
        if not regions:
            raise ValueError("--composite-region-boxes did not define any valid longitude bins")
        selection_class = str(getattr(args, "composite_selection_class", "")).strip().lower()
        combine_polarities = bool(getattr(args, "composite_combine_polarities", False))
        region_label = "all_polarities" if combine_polarities else "by_polarity"
        composite_root = output_root / "w_native_meso50_500km_mode1_aligned_by_polarity" / f"composite_{token}_crossing_regions"
        figures_dir = composite_root / "figures"
        grids_dir = composite_root / "grids"
        figures_dir.mkdir(parents=True, exist_ok=True)
        grids_dir.mkdir(parents=True, exist_ok=True)

        object_pool: list[SelectedObject] = []
        for polarity in polarities:
            object_pool.extend(
                select_strict_crossing_objects(
                    structures,
                    target_lat=target_lat,
                    polarity=polarity,
                    intersect_radius_r=float(args.intersect_radius_r),
                    max_objects=int(args.composite_max_objects),
                )
            )
        object_pool.sort(key=lambda item: (item.date, item.hua_object_id))
        print(
            f"[crossing-composite-region] {token}: {len(object_pool)} strict-crossing objects before class/region filters; "
            f"class={selection_class or 'all'}; regions={','.join(regions)}; workers={workers}",
            flush=True,
        )

        all_selected_accumulator: dict[str, object] | None = None
        region_accumulators: dict[tuple[str, str], dict[str, object]] = {}
        region_detail_rows: list[dict[str, object]] = []

        def consume_region_grid(idx: int, obj: SelectedObject, grid: dict[str, object]) -> None:
            nonlocal all_selected_accumulator
            multipole_class = str(grid.get("multipole_class", "")).lower()
            region = assign_composite_region(float(obj.center_lon), regions)
            selected = (not selection_class or multipole_class == selection_class) and region is not None
            distance_km = float(grid["crossing_distance_km"])
            distance_over_r = float(grid["crossing_distance_over_r"])
            row = object_detail_row(target_lat, obj.polarity, obj, grid, distance_km, distance_over_r)
            row.update(
                {
                    "region": region or "",
                    "region_mode": "longitude_bins",
                    "selected_for_region_composite": bool(selected),
                    "selection_class": selection_class or "all",
                    "combine_polarities": combine_polarities,
                }
            )
            region_detail_rows.append(row)
            print(
                f"[crossing-composite-region] {idx}/{len(object_pool)} {obj.hua_object_id} "
                f"{obj.polarity} {multipole_class} region={region or 'none'} selected={selected}",
                flush=True,
            )
            if not selected:
                return
            composite_class = selection_class or "all"
            if all_selected_accumulator is None:
                all_selected_accumulator = init_composite_accumulator(grid, target_lat, region_label, args, multipole_class=composite_class)
                all_selected_accumulator["region"] = "all_regions"
                all_selected_accumulator["region_boxes"] = format_region_boxes(regions)
            update_composite_accumulator(all_selected_accumulator, grid, args)
            group_polarity = "all_polarities" if combine_polarities else obj.polarity
            key = (region, group_polarity)
            if key not in region_accumulators:
                region_accumulators[key] = init_composite_accumulator(grid, target_lat, group_polarity, args, multipole_class=composite_class)
                region_accumulators[key]["region"] = region
                region_accumulators[key]["region_boxes"] = format_region_boxes(regions)
            update_composite_accumulator(region_accumulators[key], grid, args)

        if workers > 1:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [executor.submit(build_object_grid, item) for item in enumerate(object_pool, start=1)]
                for future in as_completed(futures):
                    idx, obj, grid = future.result()
                    consume_region_grid(idx, obj, grid)
        else:
            for item in enumerate(object_pool, start=1):
                idx, obj, grid = build_object_grid(item)
                consume_region_grid(idx, obj, grid)

        if all_selected_accumulator is None:
            summary_rows.append(
                {
                    "target_lat": target_lat,
                    "polarity": region_label,
                    "region": "",
                    "object_count": 0,
                    "selection_class": selection_class or "all",
                    "status": "no_selected_objects",
                }
            )
        else:
            reference_composite = finalize_composite_accumulator(all_selected_accumulator, args)
            reference_composite["region"] = "all_regions"
            reference_composite["selection_class"] = selection_class or "all"
            for (region, group_polarity), accumulator in sorted(region_accumulators.items()):
                composite = finalize_composite_accumulator(accumulator, args)
                composite["region"] = region
                composite["selection_class"] = selection_class or "all"
                composite["region_boxes"] = format_region_boxes(regions)
                attach_reference_native(composite, reference_composite)
                composite["reference_multipole_class"] = f"{selection_class or 'selected'}_all_regions"
                summary_rows.append(write_region_composite_outputs(composite, token, region, group_polarity, selection_class or "all", grids_dir, figures_dir))

        write_csv(composite_root / f"composite_{token}_crossing_region_summary.csv", summary_rows)
        write_json(composite_root / f"composite_{token}_crossing_region_summary.json", summary_rows)
        write_csv(composite_root / f"composite_{token}_crossing_region_objects.csv", region_detail_rows)
        return

    for polarity in polarities:
        objects = select_strict_crossing_objects(
            structures,
            target_lat=target_lat,
            polarity=polarity,
            intersect_radius_r=float(args.intersect_radius_r),
            max_objects=int(args.composite_max_objects),
        )
        if not objects:
            print(f"[crossing-composite] {token} {polarity}: no strict crossing objects", flush=True)
            summary_rows.append({"target_lat": target_lat, "polarity": polarity, "object_count": 0, "status": "no_objects"})
            continue

        print(f"[crossing-composite] {token} {polarity}: {len(objects)} objects; workers={workers}", flush=True)
        accumulator_all: dict[str, object] | None = None
        class_accumulators: dict[str, dict[str, object]] = {}

        if workers > 1:
            futures = []
            with ThreadPoolExecutor(max_workers=workers) as executor:
                for item in enumerate(objects, start=1):
                    futures.append(executor.submit(build_object_grid, item))
                for future in as_completed(futures):
                    idx, obj, grid = future.result()
                    print(f"[crossing-composite] {polarity} {idx}/{len(objects)} {obj.hua_object_id} {grid['multipole_class']}", flush=True)
                    if accumulator_all is None:
                        accumulator_all = init_composite_accumulator(grid, target_lat, polarity, args, multipole_class="all")
                    update_composite_accumulator(accumulator_all, grid, args)
                    multipole_class = str(grid["multipole_class"])
                    if multipole_class not in class_accumulators:
                        class_accumulators[multipole_class] = init_composite_accumulator(grid, target_lat, polarity, args, multipole_class=multipole_class)
                    update_composite_accumulator(class_accumulators[multipole_class], grid, args)
                    distance_km = float(grid["crossing_distance_km"])
                    distance_over_r = float(grid["crossing_distance_over_r"])
                    detail_rows.append(object_detail_row(target_lat, polarity, obj, grid, distance_km, distance_over_r))
        else:
            for idx, obj in enumerate(objects, start=1):
                print(f"[crossing-composite] {polarity} {idx}/{len(objects)} {obj.hua_object_id}", flush=True)
                _, obj, grid = build_object_grid((idx, obj))
                if accumulator_all is None:
                    accumulator_all = init_composite_accumulator(grid, target_lat, polarity, args, multipole_class="all")
                update_composite_accumulator(accumulator_all, grid, args)
                multipole_class = str(grid["multipole_class"])
                if multipole_class not in class_accumulators:
                    class_accumulators[multipole_class] = init_composite_accumulator(grid, target_lat, polarity, args, multipole_class=multipole_class)
                update_composite_accumulator(class_accumulators[multipole_class], grid, args)
                distance_km = float(grid["crossing_distance_km"])
                distance_over_r = float(grid["crossing_distance_over_r"])
                detail_rows.append(object_detail_row(target_lat, polarity, obj, grid, distance_km, distance_over_r))

        if accumulator_all is None:
            continue
        all_composite = finalize_composite_accumulator(accumulator_all, args)
        summary_rows.append(write_composite_outputs(all_composite, token, polarity, "all", grids_dir, figures_dir))
        for multipole_class in sorted(class_accumulators):
            composite = finalize_composite_accumulator(class_accumulators[multipole_class], args)
            attach_reference_native(composite, all_composite)
            summary_rows.append(write_composite_outputs(composite, token, polarity, multipole_class, grids_dir, figures_dir))

    write_csv(composite_root / f"composite_{token}_crossing_summary.csv", summary_rows)
    write_json(composite_root / f"composite_{token}_crossing_summary.json", summary_rows)
    write_csv(composite_root / f"composite_{token}_crossing_objects.csv", detail_rows)


def select_strict_crossing_objects(structures: pd.DataFrame, target_lat: float, polarity: str, intersect_radius_r: float, max_objects: int) -> list[SelectedObject]:
    objects = [summarize_object(part) for _, part in structures.groupby("hua_object_id", sort=False) if len(part) >= 2]
    selected = []
    for obj in objects:
        if obj.polarity != polarity:
            continue
        _, distance_over_r = crossing_distance_for_object(obj, target_lat)
        if distance_over_r <= intersect_radius_r:
            selected.append(obj)
    selected.sort(key=lambda item: (item.pass_layers, -crossing_distance_for_object(item, target_lat)[1], item.radius_km), reverse=True)
    if max_objects > 0:
        selected = selected[:max_objects]
    return selected


def select_domain_objects(structures: pd.DataFrame, bbox: tuple[float, float, float, float], polarity: str, max_objects: int) -> list[SelectedObject]:
    lon_min, lon_max, lat_min, lat_max = bbox
    lat_lo, lat_hi = sorted([lat_min, lat_max])
    objects = [summarize_object(part) for _, part in structures.groupby("hua_object_id", sort=False) if len(part) >= 2]
    selected = []
    for obj in objects:
        if obj.polarity != polarity:
            continue
        if lon_in_region_box(float(obj.center_lon), lon_min, lon_max) and lat_lo <= float(obj.center_lat) <= lat_hi:
            selected.append(obj)
    selected.sort(key=lambda item: (item.pass_layers, item.radius_km, item.date, item.hua_object_id), reverse=True)
    if max_objects > 0:
        selected = selected[:max_objects]
    return selected


def crossing_distance_for_object(obj: SelectedObject, target_lat: float) -> tuple[float, float]:
    distance_km = abs(obj.center_lat - target_lat) * meters_per_degree(target_lat)[1] / 1000.0
    distance_over_r = distance_km / obj.radius_km if obj.radius_km > 0 else math.inf
    return float(distance_km), float(distance_over_r)


def parse_composite_region_boxes(value: str) -> dict[str, list[tuple[float, float]]]:
    regions: dict[str, list[tuple[float, float]]] = {}
    for item in value.split(";"):
        if not item.strip() or ":" not in item:
            continue
        name, raw_boxes = item.split(":", 1)
        name = safe_token(name.strip())
        boxes: list[tuple[float, float]] = []
        for raw_box in raw_boxes.split("|"):
            pieces = [piece.strip() for piece in raw_box.split(",") if piece.strip()]
            if len(pieces) != 2:
                continue
            boxes.append((normalize_lon360(float(pieces[0])), normalize_lon360(float(pieces[1]))))
        if name and boxes:
            regions[name] = boxes
    return regions


def format_region_boxes(regions: dict[str, list[tuple[float, float]]]) -> str:
    parts = []
    for name, boxes in regions.items():
        parts.append(f"{name}:" + "|".join(f"{lon0:g},{lon1:g}" for lon0, lon1 in boxes))
    return ";".join(parts)


def normalize_lon360(lon: float) -> float:
    out = float(lon) % 360.0
    return out + 360.0 if out < 0 else out


def lon_in_region_box(lon: float, lon0: float, lon1: float) -> bool:
    lon = normalize_lon360(lon)
    lon0 = normalize_lon360(lon0)
    lon1 = normalize_lon360(lon1)
    if math.isclose(lon0, lon1):
        return True
    if lon0 < lon1:
        return lon0 <= lon < lon1
    return lon >= lon0 or lon < lon1


def assign_composite_region(lon: float, regions: dict[str, list[tuple[float, float]]]) -> str | None:
    for name, boxes in regions.items():
        if any(lon_in_region_box(lon, lon0, lon1) for lon0, lon1 in boxes):
            return name
    return None


def object_detail_row(target_lat: float, polarity: str, obj: SelectedObject, grid: dict[str, object], distance_km: float, distance_over_r: float) -> dict[str, object]:
    return {
        "target_lat": target_lat,
        "polarity": polarity,
        "hua_object_id": obj.hua_object_id,
        "date": obj.date,
        "pass_layers": obj.pass_layers,
        "center_lon": obj.center_lon,
        "center_lat": obj.center_lat,
        "radius_km": obj.radius_km,
        "crossing_distance_km": distance_km,
        "crossing_distance_over_r": distance_over_r,
        "multipole_class": str(grid.get("multipole_class", "")),
        "multipole_qc_status": str(grid.get("multipole_qc_status", "")),
        "multipole_zero_crossings": int(grid.get("multipole_zero_crossings", -1)),
        "multipole_valid_azimuth_count": int(grid.get("multipole_valid_azimuth_count", 0)),
        "multipole_azimuth_count": int(grid.get("multipole_azimuth_count", 0)),
        "multipole_valid_azimuth_fraction": float(grid.get("multipole_valid_azimuth_fraction", float("nan"))),
        "multipole_boundary_nan_fraction": float(grid.get("multipole_boundary_nan_fraction", float("nan"))),
        "multipole_amp_1e6_m_s": float(grid.get("multipole_amp_1e6_m_s", float("nan"))),
        "multipole_snr": float(grid.get("multipole_snr", float("nan"))),
        "multipole_dominant_mode": int(grid.get("multipole_dominant_mode", -1)),
        "multipole_harmonic_dominance": float(grid.get("multipole_harmonic_dominance", float("nan"))),
        "multipole_depth_range_m": str(grid.get("multipole_depth_range_m", "")),
        "multipole_radius_ring_r": str(grid.get("multipole_radius_ring_r", "")),
        "multipole_native_w_source": str(grid.get("multipole_native_w_source", "")),
        "native_w_meso_filter": str(grid.get("native_w_meso_filter", "")),
        "native_w_phase_alignment": str(grid.get("native_w_phase_alignment", "")),
        "dipole_phase_angle_deg": float(grid.get("dipole_phase_angle_deg", float("nan"))),
        "dipole_phase_valid": bool(grid.get("dipole_phase_valid", False)),
        "corr_rebuild_native": float(grid["corr_rebuild_native"]),
        "q95_abs_rebuild_1e6_m_s": float(grid["q95_abs_rebuild_1e6_m_s"]),
        "q95_abs_native_1e6_m_s": float(grid["q95_abs_native_1e6_m_s"]),
    }


def classify_native_w_multipole(grid: dict[str, object], args: argparse.Namespace) -> dict[str, object]:
    if "ofes_w_native_meso_m_s" in grid:
        native_source = "ofes_w_native_meso_m_s"
    elif "ofes_w_native_raw_m_s" in grid:
        native_source = "ofes_w_native_raw_m_s"
    else:
        native_source = "ofes_w_native_m_s"
    native = np.asarray(grid[native_source], dtype="f4")
    depth = np.asarray(grid["depth_m"], dtype="f4")
    x = np.asarray(grid["x_over_r"], dtype="f4")
    y = np.asarray(grid["y_over_r"], dtype="f4")
    dmin = float(getattr(args, "multipole_depth_min_m", 300.0))
    dmax = float(getattr(args, "multipole_depth_max_m", 500.0))
    r_inner = float(getattr(args, "multipole_radius_inner_r", 0.0))
    r_outer = float(getattr(args, "multipole_radius_outer_r", 1.0))
    n_azimuth = max(12, int(getattr(args, "multipole_azimuth_count", 60)))
    min_valid_azimuth_fraction = float(getattr(args, "multipole_min_valid_azimuth_fraction", 0.80))
    min_sector_valid_fraction = float(getattr(args, "multipole_min_sector_valid_fraction", 0.35))
    boundary_max_nan_fraction = float(getattr(args, "multipole_boundary_max_nan_fraction", 0.35))
    min_amp = float(getattr(args, "multipole_min_amp_1e6_m_s", 0.5)) * 1.0e-6
    min_snr = float(getattr(args, "multipole_min_snr", 2.0))
    min_harmonic_dominance = float(getattr(args, "multipole_min_harmonic_dominance", 1.1))
    r_inner = max(0.0, min(r_inner, r_outer))
    depth_mask = (depth >= min(dmin, dmax)) & (depth <= max(dmin, dmax))
    if not np.any(depth_mask):
        depth_mask = np.isfinite(depth)
    with np.errstate(invalid="ignore"):
        field = np.nanmean(native[depth_mask], axis=0)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    rr = np.hypot(xx, yy)
    boundary_inner = max(r_inner, r_outer * 0.85)
    boundary = (rr >= boundary_inner) & (rr <= r_outer)
    boundary_count = int(np.count_nonzero(boundary))
    boundary_nan_fraction = float(np.count_nonzero(boundary & ~np.isfinite(field)) / boundary_count) if boundary_count else 1.0
    values = np.full(n_azimuth, np.nan, dtype="f4")
    sector_std = np.full(n_azimuth, np.nan, dtype="f4")
    sector_valid_fraction = np.zeros(n_azimuth, dtype="f4")
    sector_valid_count = np.zeros(n_azimuth, dtype="i4")
    width = 2.0 * np.pi / float(n_azimuth)
    radial_count = max(24, int(math.ceil((r_outer - r_inner) / max(float(np.nanmedian(np.abs(np.diff(x)))) if x.size > 1 else 0.1, 1.0e-6))))
    radial_r = np.linspace(r_inner, r_outer, radial_count, dtype="f8")
    radial_length = max(r_outer - r_inner, 1.0e-12)
    for i in range(n_azimuth):
        center = (i + 0.5) * width
        ray_x = radial_r * math.cos(center)
        ray_y = radial_r * math.sin(center)
        if x.size < 2 or y.size < 2:
            continue
        ray_ix = (ray_x - float(x[0])) / float(x[1] - x[0])
        ray_iy = (ray_y - float(y[0])) / float(y[1] - y[0])
        samples = ndimage.map_coordinates(field, np.vstack([ray_iy, ray_ix]), order=1, mode="constant", cval=np.nan)
        finite_ray = np.isfinite(samples)
        valid_count = int(np.count_nonzero(finite_ray))
        sector_valid_count[i] = valid_count
        sector_valid_fraction[i] = valid_count / float(radial_count)
        if sector_valid_fraction[i] >= min_sector_valid_fraction:
            if valid_count >= 2:
                integral = float(np.trapz(samples[finite_ray], radial_r[finite_ray]))
                values[i] = integral / radial_length
            elif valid_count == 1:
                values[i] = float(samples[finite_ray][0])
            sector_std[i] = float(np.nanstd(samples[finite_ray]))
    valid = np.isfinite(values)
    valid_count = int(np.count_nonzero(valid))
    valid_fraction = valid_count / float(n_azimuth)
    qc_status = "pass"
    centered = values.copy()
    centered_smooth = values.copy()
    zero_crossings = -1
    dominant_mode = -1
    harmonic_dominance = float("nan")
    amplitude = float("nan")
    snr = float("nan")
    phase_rad = float("nan")
    phase_valid = False
    if valid_count < max(4, int(math.ceil(n_azimuth * min_valid_azimuth_fraction))):
        label = "qc_sparse_azimuth"
        qc_status = "fail_sparse_azimuth"
    elif boundary_nan_fraction > boundary_max_nan_fraction:
        label = "qc_missing_boundary"
        qc_status = "fail_missing_boundary"
    else:
        centered[valid] = centered[valid] - float(np.nanmean(centered[valid]))
        filled = fill_circular_values(centered)
        centered_smooth = ndimage.gaussian_filter1d(filled.astype("f4"), sigma=1.0, mode="wrap")
        zero_crossings = count_circular_zero_crossings(centered_smooth)
        amplitude = q95_abs(centered_smooth)
        finite_std = sector_std[np.isfinite(sector_std)]
        if finite_std.size:
            noise = float(np.nanmedian(finite_std / np.sqrt(np.maximum(sector_valid_count[np.isfinite(sector_std)], 1))))
        else:
            diff = np.diff(np.r_[centered_smooth, centered_smooth[0]])
            noise = float(1.4826 * np.nanmedian(np.abs(diff - np.nanmedian(diff))) / math.sqrt(2.0))
        noise = max(noise, 1.0e-12)
        snr = amplitude / noise
        harmonics = circular_harmonic_amplitudes(centered_smooth, max_mode=4)
        centers = (np.arange(n_azimuth, dtype="f8") + 0.5) * width
        c1 = np.nanmean(centered_smooth.astype("f8") * np.exp(-1j * centers))
        if np.isfinite(c1.real) and np.isfinite(c1.imag) and np.abs(c1) > 0.0:
            phase_rad = float(-np.angle(c1))
            phase_valid = True
        if harmonics.size:
            dominant_mode = int(np.nanargmax(harmonics) + 1)
            sorted_h = np.sort(harmonics[np.isfinite(harmonics)])
            harmonic_dominance = float(sorted_h[-1] / max(sorted_h[-2], 1.0e-12)) if sorted_h.size >= 2 else float("inf")
        if amplitude < min_amp:
            label = "qc_low_amplitude"
            qc_status = "fail_low_amplitude"
        elif snr < min_snr:
            label = "qc_low_snr"
            qc_status = "fail_low_snr"
        elif harmonic_dominance < min_harmonic_dominance:
            label = "qc_mixed_modes"
            qc_status = "fail_mixed_modes"
        elif zero_crossings <= 1:
            label = "monopole"
        elif zero_crossings == 2 and dominant_mode == 1:
            label = "dipole"
        elif zero_crossings == 4 and dominant_mode == 2:
            label = "quadrupole"
        else:
            label = "other"
    return {
        "multipole_class": label,
        "multipole_qc_status": qc_status,
        "multipole_zero_crossings": int(zero_crossings),
        "multipole_valid_azimuth_count": valid_count,
        "multipole_azimuth_count": int(n_azimuth),
        "multipole_valid_azimuth_fraction": float(valid_fraction),
        "multipole_sector_valid_fraction": sector_valid_fraction,
        "multipole_boundary_nan_fraction": float(boundary_nan_fraction),
        "multipole_amp_1e6_m_s": float(amplitude * 1.0e6) if np.isfinite(amplitude) else float("nan"),
        "multipole_snr": float(snr),
        "multipole_dominant_mode": int(dominant_mode),
        "multipole_harmonic_dominance": float(harmonic_dominance),
        "multipole_azimuth_native_w_1e6_m_s": values * 1.0e6,
        "multipole_azimuth_centered_native_w_1e6_m_s": centered_smooth * 1.0e6,
        "multipole_depth_range_m": f"{min(dmin, dmax):g}-{max(dmin, dmax):g}",
        "multipole_radius_ring_r": f"{r_inner:g}-{r_outer:g}",
        "multipole_classifier": "native_w_60azimuth_true_radial_integral_qc",
        "multipole_radial_sample_count": int(radial_count),
        "multipole_native_w_source": native_source,
        "dipole_phase_angle_rad": float(phase_rad),
        "dipole_phase_angle_deg": float(np.degrees(phase_rad)) if phase_valid else float("nan"),
        "dipole_phase_valid": bool(phase_valid and label == "dipole"),
    }


def fill_circular_values(values: np.ndarray) -> np.ndarray:
    finite = np.asarray(values, dtype="f8")
    if not np.any(np.isfinite(finite)):
        return np.full(finite.shape, np.nan, dtype="f8")
    filled = finite.copy()
    valid_idx = np.flatnonzero(np.isfinite(filled))
    if valid_idx.size == 1:
        filled[~np.isfinite(filled)] = filled[valid_idx[0]]
        return filled
    idx = np.arange(filled.size)
    extended_idx = np.r_[valid_idx, valid_idx[0] + filled.size]
    extended_values = np.r_[filled[valid_idx], filled[valid_idx[0]]]
    return np.interp(idx, extended_idx, extended_values, period=filled.size)


def align_native_w_to_dipole_phase(grid: dict[str, object], args: argparse.Namespace) -> dict[str, object]:
    source_key = "ofes_w_native_meso_m_s" if "ofes_w_native_meso_m_s" in grid else "ofes_w_native_m_s"
    source = np.asarray(grid[source_key], dtype="f4")
    enabled = bool(getattr(args, "native_w_phase_align", True))
    phase = float(grid.get("dipole_phase_angle_rad", float("nan")))
    phase_valid = bool(grid.get("dipole_phase_valid", False))
    is_dipole = str(grid.get("multipole_class", "")).lower() == "dipole"
    if not (enabled and phase_valid and is_dipole and np.isfinite(phase)):
        return {
            "ofes_w_native_meso_aligned_m_s": source.copy(),
            "native_w_phase_alignment": "not_applied",
            "native_w_phase_alignment_reason": "disabled_or_not_valid_dipole",
            "native_w_phase_alignment_source": source_key,
        }

    x = np.asarray(grid["x_over_r"], dtype="f8")
    y = np.asarray(grid["y_over_r"], dtype="f8")
    aligned = rotate_3d_field_by_phase(source, x, y, phase)
    return {
        "ofes_w_native_meso_aligned_m_s": aligned,
        "native_w_phase_alignment": "mode1_positive_lobe_to_plus_x",
        "native_w_phase_alignment_reason": "applied",
        "native_w_phase_alignment_source": source_key,
        "native_w_phase_alignment_angle_rad": phase,
        "native_w_phase_alignment_angle_deg": float(np.degrees(phase)),
    }


def rotate_3d_field_by_phase(values: np.ndarray, x: np.ndarray, y: np.ndarray, phase_rad: float) -> np.ndarray:
    arr = np.asarray(values, dtype="f4")
    if x.size < 2 or y.size < 2:
        return arr.copy()
    xx, yy = np.meshgrid(x, y, indexing="xy")
    cos_p = math.cos(phase_rad)
    sin_p = math.sin(phase_rad)
    x_old = cos_p * xx - sin_p * yy
    y_old = sin_p * xx + cos_p * yy
    ix = (x_old - float(x[0])) / float(x[1] - x[0])
    iy = (y_old - float(y[0])) / float(y[1] - y[0])
    coords = np.vstack([iy.ravel(), ix.ravel()])
    out = np.empty_like(arr, dtype="f4")
    for k in range(arr.shape[0]):
        out[k] = ndimage.map_coordinates(arr[k], coords, order=1, mode="constant", cval=np.nan).reshape(y.size, x.size)
    return out


def circular_harmonic_amplitudes(values: np.ndarray, max_mode: int = 4) -> np.ndarray:
    filled = fill_circular_values(values)
    if not np.any(np.isfinite(filled)):
        return np.full(max_mode, np.nan, dtype="f8")
    filled = filled - float(np.nanmean(filled))
    coeff = np.fft.rfft(filled)
    out = np.full(max_mode, np.nan, dtype="f8")
    for mode in range(1, max_mode + 1):
        if mode < coeff.size:
            out[mode - 1] = float(np.abs(coeff[mode]))
    return out


def count_circular_zero_crossings(values: np.ndarray) -> int:
    finite = np.asarray(values, dtype="f8")
    if not np.any(np.isfinite(finite)):
        return -1
    valid_idx = np.flatnonzero(np.isfinite(finite))
    if valid_idx.size < 2:
        return -1
    filled = fill_circular_values(finite)
    eps = max(float(np.nanpercentile(np.abs(filled), 20)) * 0.05, 1.0e-20)
    signs = np.sign(filled)
    signs[np.abs(filled) <= eps] = 0.0
    for i in range(signs.size):
        if signs[i] == 0:
            prev = signs[(i - 1) % signs.size]
            nxt = signs[(i + 1) % signs.size]
            signs[i] = prev if prev != 0 else nxt
    return int(np.count_nonzero(signs != np.roll(signs, 1)))


def select_crossing_objects(structures: pd.DataFrame, target_lats: list[float], intersect_radius_r: float, strict_only: bool) -> list[dict[str, object]]:
    objects = [summarize_object(part) for _, part in structures.groupby("hua_object_id", sort=False) if len(part) >= 2]
    selections = []
    for target_lat in target_lats:
        candidates = []
        for obj in objects:
            distance_km = abs(obj.center_lat - target_lat) * meters_per_degree(target_lat)[1] / 1000.0
            distance_over_r = distance_km / obj.radius_km if obj.radius_km > 0 else math.inf
            candidates.append((distance_over_r, -obj.pass_layers, distance_km, obj))
        candidates.sort(key=lambda item: (item[0] > intersect_radius_r, item[0], item[1], item[2]))
        strict = [item for item in candidates if item[0] <= intersect_radius_r]
        if strict:
            distance_over_r, _, distance_km, obj = sorted(strict, key=lambda item: (-item[3].pass_layers, item[0]))[0]
            mode = "strict_1r_crossing"
            status = "strict_crossing"
        elif strict_only:
            print(f"[crossing-skip] {lat_label(target_lat)} no strict crossing within {intersect_radius_r:g}R", flush=True)
            continue
        else:
            distance_over_r, _, distance_km, obj = candidates[0]
            mode = "nearest_not_strict"
            status = "nearest_fallback"
        selections.append(
            {
                "target_lat": float(target_lat),
                "object": obj,
                "date": obj.date,
                "distance_km": float(distance_km),
                "distance_over_r": float(distance_over_r),
                "selection_mode": mode,
                "crossing_status": status,
                "strict_crossing": bool(status == "strict_crossing"),
                "nearest_fallback": bool(status == "nearest_fallback"),
                "skipped_no_crossing": False,
            }
        )
    return selections


def select_band_objects(structures: pd.DataFrame, lat_min: float, lat_max: float, max_objects: int) -> list[SelectedObject]:
    lo = min(lat_min, lat_max)
    hi = max(lat_min, lat_max)
    objects = [summarize_object(part) for _, part in structures.groupby("hua_object_id", sort=False) if len(part) >= 2]
    items = [obj for obj in objects if lo <= obj.center_lat <= hi]
    items.sort(key=lambda obj: (obj.pass_layers, obj.max_jump_over_r, obj.max_jump_km), reverse=True)
    if max_objects > 0:
        items = items[:max_objects]
    return items


def select_objects(centers: pd.DataFrame, structures: pd.DataFrame, target_day: date, requested_id: str | None, max_objects: int) -> list[SelectedObject]:
    if requested_id:
        part = structures[structures["hua_object_id"].astype(str).eq(str(requested_id))]
        if part.empty:
            raise ValueError(f"No structure rows for hua_object_id={requested_id}")
        return [summarize_object(part)]
    day_part = structures[structures["date"].astype(str).eq(target_day.isoformat())]
    items = [summarize_object(part) for _, part in day_part.groupby("hua_object_id", sort=False) if len(part) >= 2]
    items.sort(key=lambda obj: (obj.pass_layers, obj.max_jump_over_r, obj.max_jump_km), reverse=True)
    return items[: max(1, max_objects)]


def summarize_object(part: pd.DataFrame) -> SelectedObject:
    part = part.sort_values("depth_index")
    lon_col = "center_lon_refined" if "center_lon_refined" in part.columns else "center_lon"
    lat_col = "center_lat_refined" if "center_lat_refined" in part.columns else "center_lat"
    surface = part.iloc[0]
    mx, my = meters_per_degree(float(surface[lat_col]))
    lon = part[lon_col].to_numpy(dtype="f8")
    lat = part[lat_col].to_numpy(dtype="f8")
    radius_km = float(np.nanmedian(part["radius_km"].to_numpy(dtype="f8")))
    if not np.isfinite(radius_km) or radius_km <= 0:
        radius_km = 100.0
    jumps = []
    for a, b in zip(range(len(part) - 1), range(1, len(part))):
        dist = math.hypot(lon_delta_deg(lon[b], lon[a]) * mx / 1000.0, (lat[b] - lat[a]) * my / 1000.0)
        jumps.append(dist)
    max_jump = max(jumps) if jumps else 0.0
    return SelectedObject(
        hua_object_id=str(surface["hua_object_id"]),
        date=str(surface["date"]),
        polarity=str(surface["polarity"]),
        pass_layers=int(len(part)),
        max_jump_km=float(max_jump),
        max_jump_over_r=float(max_jump / radius_km),
        radius_km=radius_km,
        center_lon=float(surface[lon_col]),
        center_lat=float(surface[lat_col]),
    )


def open_neighbor_center_tables(centers: pd.DataFrame, target_day: date) -> dict[str, pd.DataFrame]:
    out = {}
    for key, day in [("prev", target_day - timedelta(days=1)), ("next", target_day + timedelta(days=1))]:
        part = centers[centers["date"].astype(str).eq(day.isoformat()) & centers["hua_pass"].astype(bool)]
        out[key] = part.copy()
    return out


def rebuild_object_w(raw: dict[str, np.memmap], metas: dict[str, object], centers: pd.DataFrame, obj: SelectedObject, neighbors: dict[str, pd.DataFrame], args: argparse.Namespace) -> dict[str, object]:
    nlev = min(int(args.max_depth_layers), metas["u"].z.count, metas["w"].z.count, metas["prho"].z.count)
    depth = metas["prho"].z.values[:nlev].astype("f8")
    x_over_r = np.linspace(-float(args.extent_r), float(args.extent_r), int(args.grid_n))
    y_over_r = np.linspace(-float(args.extent_r), float(args.extent_r), int(args.grid_n))
    xxr, yyr = np.meshgrid(x_over_r, y_over_r)
    lon_grid, lat_grid = local_lon_lat_grid(xxr, yyr, obj.center_lon, obj.center_lat, obj.radius_km)

    u_raw = sample_stack(raw["u"], metas["u"], lon_grid, lat_grid, nlev) / 100.0
    v_raw = sample_stack(raw["v"], metas["v"], lon_grid, lat_grid, nlev) / 100.0
    native_w_sample_nlev = min(metas["w"].z.count, nlev + 1)
    native_w_raw = sample_stack(raw["w"], metas["w"], lon_grid, lat_grid, native_w_sample_nlev) / 100.0
    native_w, native_w_alignment_info = align_native_w_vertical(
        native_w_raw,
        metas["w"].z.values[:native_w_sample_nlev].astype("f8"),
        depth,
        "layer_center",
    )
    prho_raw = sample_stack(raw["prho"], metas["prho"], lon_grid, lat_grid, nlev)
    prho_raw = normalize_density_units(prho_raw)
    set_thread_scale_cache(args, None)
    if bool(getattr(args, "precompute_object_temporal_blocks", False)):
        set_thread_scale_cache(
            args,
            precompute_object_temporal_blocks(
                prho_raw,
                u_raw,
                v_raw,
                metas,
                lon_grid,
                lat_grid,
                nlev,
                x_over_r,
                obj,
                args,
            ),
        )

    r = np.hypot(xxr, yyr)
    far_mask = (r >= float(args.farfield_inner_r)) & (r <= float(args.farfield_outer_r))
    far_count = max(1, int(np.count_nonzero(far_mask)))
    u_bg_before_filter = np.nanmedian(np.where(far_mask[None, :, :], u_raw, np.nan), axis=(1, 2))
    v_bg_before_filter = np.nanmedian(np.where(far_mask[None, :, :], v_raw, np.nan), axis=(1, 2))
    u_rel_q95_before_filter = q95_abs(np.hypot(u_raw - u_bg_before_filter[:, None, None], v_raw - v_bg_before_filter[:, None, None]))
    native_w, prho, u, v, scale_info = apply_scale_separation(
        native_w,
        prho_raw,
        u_raw,
        v_raw,
        metas,
        lon_grid,
        lat_grid,
        depth,
        nlev,
        x_over_r,
        obj,
        args,
    )
    farfield_valid_fraction = np.isfinite(np.where(far_mask[None, :, :], prho, np.nan)).sum(axis=(1, 2)) / far_count
    farfield_rho_bg_raw = np.nanmedian(np.where(far_mask[None, :, :], prho, np.nan), axis=(1, 2))
    rho_bg_source = str(getattr(args, "rho_bg_source", "regional_box"))
    regional_bg_valid_fraction = np.full(nlev, np.nan, dtype="f4")
    if rho_bg_source == "regional_box":
        if str(getattr(args, "rebuild_density_filter", "none")) == "none":
            rho_bg_raw, regional_bg_valid_fraction = regional_density_background(raw["prho"], metas["prho"], obj, nlev, args)
        else:
            rho_bg_raw, regional_bg_valid_fraction = regional_density_background_temporal(metas["prho"], obj, nlev, args)
        rho_valid_fraction = regional_bg_valid_fraction
        rho_valid_min_fraction = float(getattr(args, "regional_bg_min_valid_fraction", 0.25))
    else:
        rho_bg_raw = farfield_rho_bg_raw
        rho_valid_fraction = farfield_valid_fraction
        rho_valid_min_fraction = float(getattr(args, "farfield_valid_min_fraction", 0.35))
    u_bg = np.nanmedian(np.where(far_mask[None, :, :], u, np.nan), axis=(1, 2))
    v_bg = np.nanmedian(np.where(far_mask[None, :, :], v, np.nan), axis=(1, 2))
    u_rel_q95_after_filter = q95_abs(np.hypot(u - u_bg[:, None, None], v - v_bg[:, None, None]))
    rho_stabilization = not bool(getattr(args, "disable_rho_stabilization", False))
    rho_z_min = float(getattr(args, "rho_z_min", 2.0e-5))
    eta_rho_cap_m = float(getattr(args, "eta_rho_cap_m", 500.0))
    rebuild_density_filter = str(getattr(args, "rebuild_density_filter", "none"))
    rho_prime_q95_before_filter = q95_abs(prho_raw - np.nanmedian(prho_raw, axis=(1, 2))[:, None, None])
    rho_prime_q95_after_filter = float("nan")
    eta_rho_q95_before_horizontal_filter = float("nan")
    eta_rho_q95_after_horizontal_filter = float("nan")
    grad_eta_q95_before_filter = float("nan")
    grad_eta_q95_after_filter = float("nan")
    dx_m = float(np.mean(np.diff(x_over_r))) * obj.radius_km * 1000.0
    dy_m = float(np.mean(np.diff(y_over_r))) * obj.radius_km * 1000.0
    if rho_stabilization:
        valid_bg = rho_valid_fraction >= rho_valid_min_fraction
        rho_bg_input = np.where(valid_bg, rho_bg_raw, np.nan)
        rho_bg = nan_gaussian_smooth_1d(rho_bg_input, float(getattr(args, "rho_bg_smooth_sigma_layers", 2.0)))
        drho_dz_raw = np.gradient(rho_bg, depth)
        good_rho_z = np.isfinite(drho_dz_raw) & (np.abs(drho_dz_raw) >= rho_z_min) & valid_bg
        drho_dz = np.where(good_rho_z, drho_dz_raw, np.nan)
        rho_prime = prho - rho_bg[:, None, None]
        rho_prime_before_detrend = rho_prime.copy()
        if rebuild_density_filter == "bg_perturb_decomp":
            rho_prime = detrend_density_prime(rho_prime, xxr, yyr, args)
        rho_prime_q95_after_filter = q95_abs(rho_prime)
        z_anom = np.divide(-rho_prime, drho_dz[:, None, None], out=np.full_like(rho_prime, np.nan), where=np.isfinite(drho_dz[:, None, None]))
        z_anom = np.clip(z_anom, -eta_rho_cap_m, eta_rho_cap_m)
    else:
        rho_bg = rho_bg_raw
        drho_dz = np.gradient(rho_bg, depth)
        good_rho_z = np.isfinite(drho_dz) & (np.abs(drho_dz) >= 1.0e-8)
        rho_prime = prho - rho_bg[:, None, None]
        rho_prime_before_detrend = rho_prime.copy()
        if rebuild_density_filter == "bg_perturb_decomp":
            rho_prime = detrend_density_prime(rho_prime, xxr, yyr, args)
        rho_prime_q95_after_filter = q95_abs(rho_prime)
        z_anom = np.divide(-rho_prime, drho_dz[:, None, None], out=np.full_like(rho_prime, np.nan), where=good_rho_z[:, None, None])
    eta_rho_q95_before_horizontal_filter = q95_abs(z_anom)
    grad_eta_q95_before_filter = gradient_q95_abs(z_anom, dx_m, dy_m)
    z_anom_pre_horizontal_filter = z_anom.copy()
    if rebuild_density_filter == "none":
        eta_sigma_cells = float(args.smooth_sigma_cells)
        z_anom = nan_gaussian_smooth_3d(z_anom, eta_sigma_cells)
    else:
        eta_sigma_cells = sigma_r_to_cells(float(getattr(args, "eta_horizontal_lowpass_sigma_r", 0.5)), x_over_r)
        z_anom = nan_gaussian_smooth_3d(z_anom, eta_sigma_cells)
    eta_rho_q95_after_horizontal_filter = q95_abs(z_anom)
    grad_eta_q95_after_filter = gradient_q95_abs(z_anom, dx_m, dy_m)

    dzdx = np.empty_like(z_anom)
    dzdy = np.empty_like(z_anom)
    for k in range(nlev):
        dzdy[k], dzdx[k] = np.gradient(z_anom[k], dy_m, dx_m)
    grad_eta_abs = np.hypot(dzdx, dzdy)

    cx_abs, cy_abs, c_valid, c_method = estimate_translation(centers, neighbors, obj)
    u_for_rebuild = u
    v_for_rebuild = v
    cx_profile = np.full(nlev, cx_abs, dtype="f4")
    cy_profile = np.full(nlev, cy_abs, dtype="f4")
    layer_tracking_valid = np.zeros(nlev, dtype=bool)
    c_bg_x = 0.0
    c_bg_y = 0.0
    velocity_reference = str(getattr(args, "velocity_reference", "farfield_relative"))
    translation_profile = str(getattr(args, "translation_profile", "layerwise"))
    if translation_profile == "layer_tracking":
        cx_profile, cy_profile, layer_tracking_valid, c_method = estimate_translation_profile(centers, neighbors, obj, nlev, args)
        c_valid = bool(np.any(layer_tracking_valid))
    cx_profile, cy_profile, layer_tracking_valid, translation_filter_info = filter_translation_profile(cx_profile, cy_profile, layer_tracking_valid, args)
    if velocity_reference == "farfield_relative":
        u_for_rebuild = u - u_bg[:, None, None]
        v_for_rebuild = v - v_bg[:, None, None]
        if not c_valid:
            # A single-date catalog has no temporal center displacement. Keep
            # term1 neutral instead of manufacturing c_rel=-u_bg from c=0.
            cx_profile = np.zeros(nlev, dtype="f4")
            cy_profile = np.zeros(nlev, dtype="f4")
            c_method = f"{c_method}_no_valid_translation"
        elif translation_profile in {"layerwise", "layer_tracking"}:
            cx_profile = cx_profile - np.nan_to_num(u_bg.astype("f4"), nan=0.0)
            cy_profile = cy_profile - np.nan_to_num(v_bg.astype("f4"), nan=0.0)
            c_bg_x = finite_median_or_zero(u_bg)
            c_bg_y = finite_median_or_zero(v_bg)
            c_method = f"{c_method}_minus_farfield_bg_layerwise"
        else:
            bg_nlev = max(1, min(int(getattr(args, "translation_background_layers", 10)), nlev))
            c_bg_x = finite_median_or_zero(u_bg[:bg_nlev])
            c_bg_y = finite_median_or_zero(v_bg[:bg_nlev])
            cx_profile = cx_profile - c_bg_x
            cy_profile = cy_profile - c_bg_y
            c_method = f"{c_method}_minus_farfield_bg{bg_nlev}"
    support = np.isfinite(z_anom) & np.isfinite(u_for_rebuild) & np.isfinite(v_for_rebuild) & np.isfinite(native_w)
    native_w_meso, native_w_meso_info = filter_native_w_meso(
        native_w,
        metas["w"],
        lon_grid,
        lat_grid,
        depth,
        x_over_r,
        y_over_r,
        obj,
        args,
    )
    native_w_meso[~support] = np.nan
    cx3 = cx_profile[:, None, None]
    cy3 = cy_profile[:, None, None]
    rebuild_formula = str(getattr(args, "rebuild_formula", "relative_advection"))
    term1 = cx3 * dzdx + cy3 * dzdy
    if rebuild_formula == "legacy_double_c":
        term2 = -((u_for_rebuild - cx3) * dzdx + (v_for_rebuild - cy3) * dzdy)
    else:
        term2 = -(u_for_rebuild * dzdx + v_for_rebuild * dzdy)
    rebuild_w = term1 + term2
    for arr in [term1, term2, rebuild_w]:
        arr[~support] = np.nan
    native_w[~support] = np.nan

    corr = spatial_corr(rebuild_w, native_w)
    corr_minus = spatial_corr(-rebuild_w, native_w)
    result = {
        **asdict(obj),
        "science_tag": SCIENCE_TAG,
        "depth_m": depth,
        "x_over_r": x_over_r,
        "y_over_r": y_over_r,
        "term1_m_s": term1,
        "term2_m_s": term2,
        "rebuild_w_m_s": rebuild_w,
        "ofes_w_native_m_s": native_w,
        "ofes_w_native_meso_m_s": native_w_meso,
        "ofes_w_native_raw_m_s": native_w_raw,
        "u_for_rebuild_raw_m_s": u_raw,
        "v_for_rebuild_raw_m_s": v_raw,
        "u_for_rebuild_filtered_m_s": u,
        "v_for_rebuild_filtered_m_s": v,
        "prho_for_rebuild": prho,
        "prho_raw_for_rebuild": prho_raw,
        "z_rho_anom_m": z_anom,
        "z_rho_anom_before_horizontal_filter_m": z_anom_pre_horizontal_filter,
        "rho_prime_before_detrend": rho_prime_before_detrend,
        "rho_prime_for_rebuild": rho_prime,
        "rho_bg": rho_bg,
        "rho_bg_raw": rho_bg_raw,
        "farfield_rho_bg_raw": farfield_rho_bg_raw,
        "rho_bg_smooth": rho_bg,
        "drho_dz": drho_dz,
        "rho_z_used": drho_dz,
        "farfield_valid_fraction_by_layer": farfield_valid_fraction.astype("f4"),
        "regional_bg_valid_fraction_by_layer": regional_bg_valid_fraction.astype("f4"),
        "good_rho_z_layer": good_rho_z.astype("u1"),
        "u_bg_m_s": u_bg,
        "v_bg_m_s": v_bg,
        "u_bg_before_filter_m_s": u_bg_before_filter,
        "v_bg_before_filter_m_s": v_bg_before_filter,
        "rho_stabilization": bool(rho_stabilization),
        "rho_bg_source": rho_bg_source,
        "regional_bg_lon_half_width_deg": float(getattr(args, "regional_bg_lon_half_width_deg", 5.0)),
        "regional_bg_lat_half_width_deg": float(getattr(args, "regional_bg_lat_half_width_deg", 2.0)),
        "regional_bg_exclude_r": float(getattr(args, "regional_bg_exclude_r", 1.5)),
        "regional_bg_min_valid_fraction": float(getattr(args, "regional_bg_min_valid_fraction", 0.25)),
        "rho_bg_smooth_sigma_layers": float(getattr(args, "rho_bg_smooth_sigma_layers", 2.0)),
        "rho_z_min": float(rho_z_min),
        "eta_rho_cap_m": float(eta_rho_cap_m),
        "eta_rho_cap_mode": str(getattr(args, "eta_rho_cap_mode", "fixed")),
        "bad_rho_z_mode": str(getattr(args, "bad_rho_z_mode", "mask")),
        "farfield_valid_min_fraction": float(getattr(args, "farfield_valid_min_fraction", 0.35)),
        "bad_rho_z_layer_count": int(nlev - int(np.count_nonzero(good_rho_z))),
        "eta_rho_q95_abs_m": q95_abs(z_anom),
        "eta_rho_max_abs_m": finite_max_abs(z_anom),
        "grad_eta_q95_abs": q95_abs(grad_eta_abs),
        "rebuild_density_filter": rebuild_density_filter,
        "eta_horizontal_lowpass_sigma_r": float(getattr(args, "eta_horizontal_lowpass_sigma_r", 0.5)),
        "eta_horizontal_lowpass_sigma_cells": float(eta_sigma_cells),
        "density_bg_detrend_order": int(getattr(args, "density_bg_detrend_order", 1)),
        "density_bg_detrend_fit_ring": str(getattr(args, "density_bg_detrend_fit_ring", "1.5,4.0")),
        "rho_prime_q95_before_filter": float(rho_prime_q95_before_filter),
        "rho_prime_q95_after_filter": float(rho_prime_q95_after_filter),
        "eta_rho_q95_before_horizontal_filter": float(eta_rho_q95_before_horizontal_filter),
        "eta_rho_q95_after_horizontal_filter": float(eta_rho_q95_after_horizontal_filter),
        "grad_eta_q95_before_filter": float(grad_eta_q95_before_filter),
        "grad_eta_q95_after_filter": float(grad_eta_q95_after_filter),
        **scale_info,
        **native_w_meso_info,
        **translation_filter_info,
        "u_rel_q95_before_filter": float(u_rel_q95_before_filter),
        "u_rel_q95_after_filter": float(u_rel_q95_after_filter),
        "velocity_reference": velocity_reference,
        "translation_profile": translation_profile,
        "rebuild_formula": rebuild_formula,
        **native_w_alignment_info,
        "cx_absolute_m_s": float(cx_abs),
        "cy_absolute_m_s": float(cy_abs),
        "c_bg_x_m_s": float(c_bg_x),
        "c_bg_y_m_s": float(c_bg_y),
        "cx_m_s": finite_median_or_zero(cx_profile),
        "cy_m_s": finite_median_or_zero(cy_profile),
        "cx_profile_m_s": cx_profile,
        "cy_profile_m_s": cy_profile,
        "layer_tracking_valid": layer_tracking_valid.astype("u1"),
        "layer_tracking_valid_count": int(np.count_nonzero(layer_tracking_valid)),
        "layer_tracking_valid_fraction": float(np.count_nonzero(layer_tracking_valid) / max(1, nlev)),
        "c_valid": bool(c_valid),
        "c_method": c_method,
        "grid_n": int(args.grid_n),
        "extent_r": float(args.extent_r),
        "valid_grid_fraction": float(np.isfinite(rebuild_w).sum() / rebuild_w.size),
        "corr_rebuild_native": float(corr),
        "corr_minus_rebuild_native": float(corr_minus),
        "dipole_score_rebuild": float(dipole_score(np.nanmean(rebuild_w, axis=0), xxr, yyr)),
        "dipole_score_native": float(dipole_score(np.nanmean(native_w, axis=0), xxr, yyr)),
        "q95_abs_rebuild_1e6_m_s": q95_abs(rebuild_w) * 1.0e6,
        "q95_abs_native_1e6_m_s": q95_abs(native_w) * 1.0e6,
    }
    set_thread_scale_cache(args, None)
    return result


def sample_stack(raw: np.memmap, meta, lon_grid: np.ndarray, lat_grid: np.ndarray, nlev: int) -> np.ndarray:
    lon = meta.x.values
    lat = meta.y.values
    lon_step = float(np.nanmedian(np.diff(lon)))
    lat_step = float(np.nanmedian(np.diff(lat)))
    ii = ((lon_grid - float(lon[0])) / lon_step) % len(lon)
    jj = np.clip((lat_grid - float(lat[0])) / lat_step, 0, len(lat) - 1)
    out = np.empty((nlev, lon_grid.shape[0], lon_grid.shape[1]), dtype="f4")
    coords = np.vstack([ii.ravel(), jj.ravel()])

    i_floor = np.floor(ii).astype(int)
    i_ceil = np.ceil(ii).astype(int)
    j_floor = np.floor(jj).astype(int)
    j_ceil = np.ceil(jj).astype(int)
    crosses_lon_wrap = bool(np.ptp(ii) > len(lon) / 2)
    if not crosses_lon_wrap:
        i0 = max(0, int(np.nanmin(i_floor)) - 2)
        i1 = min(len(lon) - 1, int(np.nanmax(i_ceil)) + 2)
        j0 = max(0, int(np.nanmin(j_floor)) - 2)
        j1 = min(len(lat) - 1, int(np.nanmax(j_ceil)) + 2)
        if i1 > i0 and j1 > j0:
            local_coords = np.vstack([(ii - i0).ravel(), (jj - j0).ravel()])
            for k in range(nlev):
                layer = np.asarray(raw[i0 : i1 + 1, j0 : j1 + 1, k], dtype="f4")
                layer[np.abs(layer) > 1.0e30] = np.nan
                sampled = ndimage.map_coordinates(layer, local_coords, order=1, mode="nearest", cval=np.nan).reshape(lon_grid.shape)
                out[k] = sampled
            return out

    for k in range(nlev):
        layer = np.asarray(raw[:, :, k], dtype="f4")
        layer[np.abs(layer) > 1.0e30] = np.nan
        sampled = ndimage.map_coordinates(layer, coords, order=1, mode="wrap", cval=np.nan).reshape(lon_grid.shape)
        out[k] = sampled
    return out


def align_native_w_vertical(native_w_raw: np.ndarray, source_depth_m: np.ndarray, target_depth_m: np.ndarray, mode: str) -> tuple[np.ndarray, dict[str, object]]:
    source = np.asarray(source_depth_m, dtype="f8")
    target = np.asarray(target_depth_m, dtype="f8")
    raw = np.asarray(native_w_raw, dtype="f4")
    if source.size == 0 or target.size == 0:
        aligned = np.full((target.size, raw.shape[1], raw.shape[2]), np.nan, dtype="f4")
    else:
        aligned = np.empty((target.size, raw.shape[1], raw.shape[2]), dtype="f4")
        flat_raw = raw.reshape(raw.shape[0], -1)
        flat_out = aligned.reshape(target.size, -1)
        for col in range(flat_raw.shape[1]):
            values = flat_raw[:, col].astype("f8", copy=False)
            good = np.isfinite(values) & np.isfinite(source)
            if np.count_nonzero(good) >= 2:
                flat_out[:, col] = np.interp(target, source[good], values[good], left=values[good][0], right=values[good][-1]).astype("f4")
            elif np.count_nonzero(good) == 1:
                flat_out[:, col] = np.float32(values[good][0])
            else:
                flat_out[:, col] = np.nan

    return aligned, {
        "native_w_vertical_alignment": "layer_center",
        "native_w_ctl_position": "T cell bottom",
        "native_w_comparison_position": "interpolated to prho/u/v layer-center depths",
        "native_w_source_depth_m": source.astype("f4"),
        "native_w_target_depth_m": target.astype("f4"),
        "native_w_alignment_extrapolated_top": bool(source.size and target.size and target[0] < source[0]),
        "native_w_alignment_extrapolated_bottom": bool(source.size and target.size and target[-1] > source[-1]),
    }


def apply_scale_separation(
    native_w: np.ndarray,
    prho_raw: np.ndarray,
    u_raw: np.ndarray,
    v_raw: np.ndarray,
    metas: dict[str, object],
    lon_grid: np.ndarray,
    lat_grid: np.ndarray,
    depth: np.ndarray,
    nlev: int,
    x_over_r: np.ndarray,
    obj: SelectedObject,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    native_w_filtered, native_w_info = filter_native_w_temporal(
        native_w,
        metas["w"],
        lon_grid,
        lat_grid,
        depth,
        obj,
        args,
    )
    prho_filtered, density_info = filter_rebuild_prho_temporal(
        prho_raw,
        metas["prho"],
        lon_grid,
        lat_grid,
        nlev,
        obj,
        args,
    )
    u_filtered, v_filtered, velocity_info = filter_rebuild_velocity_temporal(
        u_raw,
        v_raw,
        metas["u"],
        metas["v"],
        lon_grid,
        lat_grid,
        nlev,
        x_over_r,
        obj,
        args,
    )
    return native_w_filtered, prho_filtered, u_filtered, v_filtered, {
        "scale_separation": "native_w_density_velocity_joint",
        **native_w_info,
        **density_info,
        **velocity_info,
    }


def set_thread_scale_cache(args: argparse.Namespace, cache: dict[str, object] | None) -> None:
    thread_local = getattr(args, "_thread_local", None)
    if thread_local is not None:
        thread_local.scale_cache = cache


def get_thread_scale_cache(args: argparse.Namespace) -> dict[str, object] | None:
    thread_local = getattr(args, "_thread_local", None)
    if thread_local is None:
        return None
    return getattr(thread_local, "scale_cache", None)


def precompute_object_temporal_blocks(
    current_prho: np.ndarray,
    current_u: np.ndarray,
    current_v: np.ndarray,
    metas: dict[str, object],
    lon_grid: np.ndarray,
    lat_grid: np.ndarray,
    nlev: int,
    x_over_r: np.ndarray,
    obj: SelectedObject,
    args: argparse.Namespace,
) -> dict[str, object]:
    cache: dict[str, object] = {"precompute_object_temporal_blocks": True}
    data_root = Path(getattr(args, "data_root", DEFAULT_DATA_ROOT))
    target_day = parse_iso_date(obj.date)
    start_day = parse_iso_date(str(getattr(args, "start", "1991-01-01")))
    end_day = parse_iso_date(str(getattr(args, "end", "1991-01-19")))

    density_mode = str(getattr(args, "rebuild_density_filter", "none"))
    if density_mode != "none":
        window = max(1, int(getattr(args, "rebuild_density_filter_window_days", 10)))
        days = centered_filter_days(target_day, window, start_day, end_day)
        expected = expected_dta_bytes(metas["prho"])
        accum = np.zeros_like(current_prho, dtype="f8")
        counts = np.zeros_like(current_prho, dtype="f4")
        profiles = []
        fractions = []
        used: list[str] = []
        missing: list[str] = []
        for day in days:
            if not cached_daily_exists(data_root, "prho", day, expected):
                missing.append(day.isoformat())
                continue
            raw_prho = open_daily_memmap_cached(args, data_root, "prho", day, metas["prho"], expected)
            sampled = normalize_density_units(sample_stack(raw_prho, metas["prho"], lon_grid, lat_grid, nlev))
            finite = np.isfinite(sampled)
            accum[finite] += sampled[finite]
            counts[finite] += 1.0
            profile, valid_fraction = regional_density_background(raw_prho, metas["prho"], obj, nlev, args)
            profiles.append(profile)
            fractions.append(valid_fraction)
            used.append(day.isoformat())
        if used:
            filtered = np.divide(accum, counts, out=np.full_like(accum, np.nan), where=counts > 0).astype("f4")
            cache["prho_filtered"] = filtered
            cache["density_info"] = {
                "rebuild_density_filter_window_days": window,
                "rebuild_density_filter_days_used": len(used),
                "rebuild_density_filter_dates_used": ",".join(used),
                "rebuild_density_filter_missing_dates": ",".join(missing),
                "scale_precompute_prho_sampled_blocks": True,
            }
            if profiles:
                cache["regional_rho_bg_raw"] = np.nanmean(np.stack(profiles, axis=0), axis=0).astype("f4")
                cache["regional_bg_valid_fraction"] = np.nanmean(np.stack(fractions, axis=0), axis=0).astype("f4")
        else:
            cache["density_info"] = {
                "rebuild_density_filter_window_days": window,
                "rebuild_density_filter_days_used": 0,
                "rebuild_density_filter_dates_used": "",
                "rebuild_density_filter_missing_dates": ",".join(missing),
                "scale_precompute_prho_sampled_blocks": True,
            }

    velocity_mode = str(getattr(args, "rebuild_velocity_filter", "none"))
    if velocity_mode != "none":
        window = max(1, int(getattr(args, "rebuild_velocity_filter_window_days", 10)))
        days = centered_filter_days(target_day, window, start_day, end_day)
        expected_u = expected_dta_bytes(metas["u"])
        expected_v = expected_dta_bytes(metas["v"])
        accum_u = np.zeros_like(current_u, dtype="f8")
        accum_v = np.zeros_like(current_v, dtype="f8")
        counts_u = np.zeros_like(current_u, dtype="f4")
        counts_v = np.zeros_like(current_v, dtype="f4")
        used: list[str] = []
        missing: list[str] = []
        for day in days:
            if not cached_daily_exists(data_root, "u", day, expected_u) or not cached_daily_exists(data_root, "v", day, expected_v):
                missing.append(day.isoformat())
                continue
            raw_u = open_daily_memmap_cached(args, data_root, "u", day, metas["u"], expected_u)
            raw_v = open_daily_memmap_cached(args, data_root, "v", day, metas["v"], expected_v)
            sampled_u = sample_stack(raw_u, metas["u"], lon_grid, lat_grid, nlev) / 100.0
            sampled_v = sample_stack(raw_v, metas["v"], lon_grid, lat_grid, nlev) / 100.0
            finite_u = np.isfinite(sampled_u)
            finite_v = np.isfinite(sampled_v)
            accum_u[finite_u] += sampled_u[finite_u]
            accum_v[finite_v] += sampled_v[finite_v]
            counts_u[finite_u] += 1.0
            counts_v[finite_v] += 1.0
            used.append(day.isoformat())
        if used:
            filtered_u = np.divide(accum_u, counts_u, out=np.full_like(accum_u, np.nan), where=counts_u > 0).astype("f4")
            filtered_v = np.divide(accum_v, counts_v, out=np.full_like(accum_v, np.nan), where=counts_v > 0).astype("f4")
            sigma_r = float(getattr(args, "uv_horizontal_lowpass_sigma_r", 0.5))
            sigma_cells = sigma_r_to_cells(sigma_r, x_over_r)
            cache["u_filtered"] = nan_gaussian_smooth_3d(filtered_u, sigma_cells)
            cache["v_filtered"] = nan_gaussian_smooth_3d(filtered_v, sigma_cells)
            cache["velocity_info"] = {
                "rebuild_velocity_filter": f"{velocity_mode}_highfreq_stop",
                "rebuild_velocity_filter_window_days": window,
                "rebuild_velocity_filter_days_used": len(used),
                "rebuild_velocity_filter_dates_used": ",".join(used),
                "rebuild_velocity_filter_missing_dates": ",".join(missing),
                "uv_horizontal_lowpass_sigma_r": sigma_r,
                "uv_horizontal_lowpass_sigma_cells": float(sigma_cells),
                "scale_precompute_uv_sampled_blocks": True,
            }
        else:
            cache["velocity_info"] = {
                "rebuild_velocity_filter": f"{velocity_mode}_failed_no_days",
                "rebuild_velocity_filter_window_days": window,
                "rebuild_velocity_filter_days_used": 0,
                "rebuild_velocity_filter_dates_used": "",
                "rebuild_velocity_filter_missing_dates": ",".join(missing),
                "uv_horizontal_lowpass_sigma_r": float(getattr(args, "uv_horizontal_lowpass_sigma_r", 0.5)),
                "uv_horizontal_lowpass_sigma_cells": 0.0,
                "scale_precompute_uv_sampled_blocks": True,
            }
    return cache


def open_daily_memmap_cached(args: argparse.Namespace, data_root: Path, variable: str, day: date, meta, expected: int) -> np.memmap:
    cache = getattr(args, "_daily_memmap_cache", None)
    if cache is None:
        path = require_daily_file(data_root, variable, day, expected)
        return open_dta_memmap(path, meta)
    key = (variable, day.isoformat())
    lock = getattr(args, "_daily_memmap_cache_lock", None)
    if lock is None:
        if key not in cache:
            path = require_daily_file(data_root, variable, day, expected)
            cache[key] = open_dta_memmap(path, meta)
        return cache[key]
    with lock:
        if key not in cache:
            path = require_daily_file(data_root, variable, day, expected)
            cache[key] = open_dta_memmap(path, meta)
        return cache[key]


def cached_daily_exists(data_root: Path, variable: str, day: date, expected: int) -> bool:
    try:
        require_daily_file(data_root, variable, day, expected)
    except FileNotFoundError:
        return False
    return True


def filter_native_w_temporal(
    current_native_w: np.ndarray,
    w_meta,
    lon_grid: np.ndarray,
    lat_grid: np.ndarray,
    target_depth_m: np.ndarray,
    obj: SelectedObject,
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, object]]:
    mode = str(getattr(args, "native_w_temporal_filter", "none"))
    if mode == "none":
        return current_native_w, {
            "native_w_temporal_filter": "none",
            "native_w_filter_window_days": 1,
            "native_w_filter_days_used": 1,
            "native_w_filter_dates_used": str(obj.date),
            "native_w_filter_missing_dates": "",
        }

    window = max(1, int(getattr(args, "native_w_filter_window_days", 10)))
    target_day = parse_iso_date(obj.date)
    start_day = parse_iso_date(str(getattr(args, "start", "1991-01-01")))
    end_day = parse_iso_date(str(getattr(args, "end", "1991-01-19")))
    before = (window - 1) // 2
    after = window - 1 - before
    days = []
    for offset in range(-before, after + 1):
        day = target_day + timedelta(days=offset)
        if start_day <= day <= end_day:
            days.append(day)

    data_root = Path(getattr(args, "data_root", DEFAULT_DATA_ROOT))
    expected = expected_dta_bytes(w_meta)
    nlev = len(target_depth_m)
    sample_nlev = min(w_meta.z.count, nlev + 1)
    source_depth = w_meta.z.values[:sample_nlev].astype("f8")
    accum = np.zeros_like(current_native_w, dtype="f8")
    counts = np.zeros_like(current_native_w, dtype="f4")
    used: list[str] = []
    missing: list[str] = []

    for day in days:
        if not cached_daily_exists(data_root, "w", day, expected):
            missing.append(day.isoformat())
            continue
        raw_w = open_daily_memmap_cached(args, data_root, "w", day, w_meta, expected)
        raw_sample = sample_stack(raw_w, w_meta, lon_grid, lat_grid, sample_nlev) / 100.0
        aligned, _ = align_native_w_vertical(raw_sample, source_depth, target_depth_m, "layer_center")
        finite = np.isfinite(aligned)
        accum[finite] += aligned[finite]
        counts[finite] += 1.0
        used.append(day.isoformat())

    if not used:
        return current_native_w, {
            "native_w_temporal_filter": f"{mode}_failed_no_days",
            "native_w_filter_window_days": window,
            "native_w_filter_days_used": 0,
            "native_w_filter_dates_used": "",
            "native_w_filter_missing_dates": ",".join(missing),
        }

    filtered = np.divide(accum, counts, out=np.full_like(accum, np.nan), where=counts > 0).astype("f4")
    return filtered, {
        "native_w_temporal_filter": f"{mode}_highfreq_stop",
        "native_w_filter_window_days": window,
        "native_w_filter_days_used": len(used),
        "native_w_filter_dates_used": ",".join(used),
        "native_w_filter_missing_dates": ",".join(missing),
    }


def filter_native_w_meso(
    current_native_w: np.ndarray,
    w_meta,
    lon_grid: np.ndarray,
    lat_grid: np.ndarray,
    target_depth_m: np.ndarray,
    x_over_r: np.ndarray,
    y_over_r: np.ndarray,
    obj: SelectedObject,
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, object]]:
    mode = str(getattr(args, "native_w_meso_filter", "temporal10d_spatial50_500km"))
    if mode == "none":
        return current_native_w.astype("f4"), {
            "native_w_meso_filter": "none",
            "native_w_meso_time_window_days": 1,
            "native_w_meso_dates_used": str(obj.date),
            "native_w_meso_missing_dates": "",
            "native_w_meso_small_cutoff_km": float("nan"),
            "native_w_meso_large_cutoff_km": float("nan"),
            "native_w_meso_small_sigma_cells": float("nan"),
            "native_w_meso_large_sigma_cells": float("nan"),
        }

    window = max(1, int(getattr(args, "native_w_meso_time_window_days", 10)))
    target_day = parse_iso_date(obj.date)
    start_day = parse_iso_date(str(getattr(args, "start", "1991-01-01")))
    end_day = parse_iso_date(str(getattr(args, "end", "1991-01-19")))
    before = (window - 1) // 2
    after = window - 1 - before
    days = [target_day + timedelta(days=offset) for offset in range(-before, after + 1)]
    days = [day for day in days if start_day <= day <= end_day]

    data_root = Path(getattr(args, "data_root", DEFAULT_DATA_ROOT))
    expected = expected_dta_bytes(w_meta)
    nlev = len(target_depth_m)
    sample_nlev = min(w_meta.z.count, nlev + 1)
    source_depth = w_meta.z.values[:sample_nlev].astype("f8")
    accum = np.zeros_like(current_native_w, dtype="f8")
    counts = np.zeros_like(current_native_w, dtype="f4")
    used: list[str] = []
    missing: list[str] = []

    for day in days:
        if not cached_daily_exists(data_root, "w", day, expected):
            missing.append(day.isoformat())
            continue
        raw_w = open_daily_memmap_cached(args, data_root, "w", day, w_meta, expected)
        raw_sample = sample_stack(raw_w, w_meta, lon_grid, lat_grid, sample_nlev) / 100.0
        aligned, _ = align_native_w_vertical(raw_sample, source_depth, target_depth_m, "layer_center")
        finite = np.isfinite(aligned)
        accum[finite] += aligned[finite]
        counts[finite] += 1.0
        used.append(day.isoformat())

    if used:
        temporal = np.divide(accum, counts, out=np.full_like(accum, np.nan), where=counts > 0).astype("f4")
    else:
        temporal = current_native_w.astype("f4")

    small_cutoff_km = float(getattr(args, "native_w_meso_small_cutoff_km", 50.0))
    large_cutoff_km = float(getattr(args, "native_w_meso_large_cutoff_km", 500.0))
    # Treat the requested cutoff as an approximate Gaussian FWHM. This keeps the
    # field mesoscale-focused without pretending to be a sharp spectral filter.
    fwhm_to_sigma = 1.0 / 2.354820045
    dx_km = max(float(np.nanmedian(np.abs(np.diff(x_over_r)))) * float(obj.radius_km), 1.0e-6)
    dy_km = max(float(np.nanmedian(np.abs(np.diff(y_over_r)))) * float(obj.radius_km), 1.0e-6)
    small_sigma = ((small_cutoff_km * fwhm_to_sigma) / dy_km, (small_cutoff_km * fwhm_to_sigma) / dx_km)
    large_sigma = ((large_cutoff_km * fwhm_to_sigma) / dy_km, (large_cutoff_km * fwhm_to_sigma) / dx_km)
    lp_small = nan_gaussian_smooth_3d(temporal, small_sigma)
    lp_large = nan_gaussian_smooth_3d(temporal, large_sigma)
    meso = (lp_small - lp_large).astype("f4")
    return meso, {
        "native_w_meso_filter": mode,
        "native_w_meso_time_window_days": window,
        "native_w_meso_days_used": len(used),
        "native_w_meso_dates_used": ",".join(used),
        "native_w_meso_missing_dates": ",".join(missing),
        "native_w_meso_small_cutoff_km": small_cutoff_km,
        "native_w_meso_large_cutoff_km": large_cutoff_km,
        "native_w_meso_small_sigma_cells_y": float(small_sigma[0]),
        "native_w_meso_small_sigma_cells_x": float(small_sigma[1]),
        "native_w_meso_large_sigma_cells_y": float(large_sigma[0]),
        "native_w_meso_large_sigma_cells_x": float(large_sigma[1]),
        "native_w_meso_q95_1e6_m_s": q95_abs(meso) * 1.0e6,
    }


def filter_rebuild_prho_temporal(
    current_prho: np.ndarray,
    prho_meta,
    lon_grid: np.ndarray,
    lat_grid: np.ndarray,
    nlev: int,
    obj: SelectedObject,
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, object]]:
    mode = str(getattr(args, "rebuild_density_filter", "none"))
    if mode == "none":
        return current_prho, {
            "rebuild_density_filter_window_days": 1,
            "rebuild_density_filter_days_used": 1,
            "rebuild_density_filter_dates_used": str(obj.date),
            "rebuild_density_filter_missing_dates": "",
        }
    scale_cache = get_thread_scale_cache(args)
    if scale_cache is not None and "prho_filtered" in scale_cache:
        return np.asarray(scale_cache["prho_filtered"], dtype="f4"), dict(scale_cache.get("density_info", {}))

    window = max(1, int(getattr(args, "rebuild_density_filter_window_days", 10)))
    days = centered_filter_days(parse_iso_date(obj.date), window, parse_iso_date(str(getattr(args, "start", "1991-01-01"))), parse_iso_date(str(getattr(args, "end", "1991-01-19"))))
    data_root = Path(getattr(args, "data_root", DEFAULT_DATA_ROOT))
    expected = expected_dta_bytes(prho_meta)
    accum = np.zeros_like(current_prho, dtype="f8")
    counts = np.zeros_like(current_prho, dtype="f4")
    used: list[str] = []
    missing: list[str] = []
    for day in days:
        if not cached_daily_exists(data_root, "prho", day, expected):
            missing.append(day.isoformat())
            continue
        raw_prho = open_daily_memmap_cached(args, data_root, "prho", day, prho_meta, expected)
        sampled = normalize_density_units(sample_stack(raw_prho, prho_meta, lon_grid, lat_grid, nlev))
        finite = np.isfinite(sampled)
        accum[finite] += sampled[finite]
        counts[finite] += 1.0
        used.append(day.isoformat())
    if not used:
        return current_prho, {
            "rebuild_density_filter_window_days": window,
            "rebuild_density_filter_days_used": 0,
            "rebuild_density_filter_dates_used": "",
            "rebuild_density_filter_missing_dates": ",".join(missing),
        }
    filtered = np.divide(accum, counts, out=np.full_like(accum, np.nan), where=counts > 0).astype("f4")
    return filtered, {
        "rebuild_density_filter_window_days": window,
        "rebuild_density_filter_days_used": len(used),
        "rebuild_density_filter_dates_used": ",".join(used),
        "rebuild_density_filter_missing_dates": ",".join(missing),
    }


def filter_rebuild_velocity_temporal(
    current_u: np.ndarray,
    current_v: np.ndarray,
    u_meta,
    v_meta,
    lon_grid: np.ndarray,
    lat_grid: np.ndarray,
    nlev: int,
    x_over_r: np.ndarray,
    obj: SelectedObject,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    mode = str(getattr(args, "rebuild_velocity_filter", "none"))
    if mode == "none":
        return current_u, current_v, {
            "rebuild_velocity_filter": "none",
            "rebuild_velocity_filter_window_days": 1,
            "rebuild_velocity_filter_days_used": 1,
            "rebuild_velocity_filter_dates_used": str(obj.date),
            "rebuild_velocity_filter_missing_dates": "",
            "uv_horizontal_lowpass_sigma_r": 0.0,
            "uv_horizontal_lowpass_sigma_cells": 0.0,
        }
    scale_cache = get_thread_scale_cache(args)
    if scale_cache is not None and "u_filtered" in scale_cache and "v_filtered" in scale_cache:
        return (
            np.asarray(scale_cache["u_filtered"], dtype="f4"),
            np.asarray(scale_cache["v_filtered"], dtype="f4"),
            dict(scale_cache.get("velocity_info", {})),
        )

    window = max(1, int(getattr(args, "rebuild_velocity_filter_window_days", 10)))
    days = centered_filter_days(parse_iso_date(obj.date), window, parse_iso_date(str(getattr(args, "start", "1991-01-01"))), parse_iso_date(str(getattr(args, "end", "1991-01-19"))))
    data_root = Path(getattr(args, "data_root", DEFAULT_DATA_ROOT))
    expected_u = expected_dta_bytes(u_meta)
    expected_v = expected_dta_bytes(v_meta)
    accum_u = np.zeros_like(current_u, dtype="f8")
    accum_v = np.zeros_like(current_v, dtype="f8")
    counts_u = np.zeros_like(current_u, dtype="f4")
    counts_v = np.zeros_like(current_v, dtype="f4")
    used: list[str] = []
    missing: list[str] = []
    for day in days:
        if not cached_daily_exists(data_root, "u", day, expected_u) or not cached_daily_exists(data_root, "v", day, expected_v):
            missing.append(day.isoformat())
            continue
        raw_u = open_daily_memmap_cached(args, data_root, "u", day, u_meta, expected_u)
        raw_v = open_daily_memmap_cached(args, data_root, "v", day, v_meta, expected_v)
        sampled_u = sample_stack(raw_u, u_meta, lon_grid, lat_grid, nlev) / 100.0
        sampled_v = sample_stack(raw_v, v_meta, lon_grid, lat_grid, nlev) / 100.0
        finite_u = np.isfinite(sampled_u)
        finite_v = np.isfinite(sampled_v)
        accum_u[finite_u] += sampled_u[finite_u]
        accum_v[finite_v] += sampled_v[finite_v]
        counts_u[finite_u] += 1.0
        counts_v[finite_v] += 1.0
        used.append(day.isoformat())

    if not used:
        return current_u, current_v, {
            "rebuild_velocity_filter": f"{mode}_failed_no_days",
            "rebuild_velocity_filter_window_days": window,
            "rebuild_velocity_filter_days_used": 0,
            "rebuild_velocity_filter_dates_used": "",
            "rebuild_velocity_filter_missing_dates": ",".join(missing),
            "uv_horizontal_lowpass_sigma_r": float(getattr(args, "uv_horizontal_lowpass_sigma_r", 0.5)),
            "uv_horizontal_lowpass_sigma_cells": 0.0,
        }

    filtered_u = np.divide(accum_u, counts_u, out=np.full_like(accum_u, np.nan), where=counts_u > 0).astype("f4")
    filtered_v = np.divide(accum_v, counts_v, out=np.full_like(accum_v, np.nan), where=counts_v > 0).astype("f4")
    sigma_r = float(getattr(args, "uv_horizontal_lowpass_sigma_r", 0.5))
    sigma_cells = sigma_r_to_cells(sigma_r, x_over_r)
    filtered_u = nan_gaussian_smooth_3d(filtered_u, sigma_cells)
    filtered_v = nan_gaussian_smooth_3d(filtered_v, sigma_cells)
    return filtered_u, filtered_v, {
        "rebuild_velocity_filter": f"{mode}_highfreq_stop",
        "rebuild_velocity_filter_window_days": window,
        "rebuild_velocity_filter_days_used": len(used),
        "rebuild_velocity_filter_dates_used": ",".join(used),
        "rebuild_velocity_filter_missing_dates": ",".join(missing),
        "uv_horizontal_lowpass_sigma_r": sigma_r,
        "uv_horizontal_lowpass_sigma_cells": float(sigma_cells),
    }


def filter_translation_profile(cx: np.ndarray, cy: np.ndarray, valid: np.ndarray, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    speed_before = np.hypot(cx, cy)
    if str(getattr(args, "rebuild_velocity_filter", "none")) == "none":
        return cx, cy, valid, {
            "translation_profile_filter": "none",
            "translation_profile_smooth_sigma_layers": 0.0,
            "translation_profile_max_speed_m_s": float("nan"),
            "c_abs_q95_before_filter": q95_abs(speed_before),
            "c_abs_q95_after_filter": q95_abs(speed_before),
            "c_abs_bad_speed_layer_count": 0,
        }

    max_speed = float(getattr(args, "translation_profile_max_speed_m_s", 0.5))
    sigma = float(getattr(args, "translation_profile_smooth_sigma_layers", 2.0))
    good = np.asarray(valid, dtype=bool) & np.isfinite(cx) & np.isfinite(cy) & (speed_before <= max_speed)
    bad_speed_count = int(np.count_nonzero(np.asarray(valid, dtype=bool) & np.isfinite(speed_before) & (speed_before > max_speed)))
    cx_work = np.where(good, cx, np.nan).astype("f4")
    cy_work = np.where(good, cy, np.nan).astype("f4")
    if np.count_nonzero(good) >= 1:
        cx_smooth = nan_gaussian_smooth_1d(fill_profile_gaps(cx_work, good, finite_median_or_zero(cx_work)), sigma)
        cy_smooth = nan_gaussian_smooth_1d(fill_profile_gaps(cy_work, good, finite_median_or_zero(cy_work)), sigma)
        out_valid = good.copy()
    else:
        cx_smooth = cx.astype("f4")
        cy_smooth = cy.astype("f4")
        out_valid = np.asarray(valid, dtype=bool)
    speed_after = np.hypot(cx_smooth, cy_smooth)
    return cx_smooth.astype("f4"), cy_smooth.astype("f4"), out_valid, {
        "translation_profile_filter": "speed_cap_vertical_smooth",
        "translation_profile_smooth_sigma_layers": sigma,
        "translation_profile_max_speed_m_s": max_speed,
        "c_abs_q95_before_filter": q95_abs(speed_before),
        "c_abs_q95_after_filter": q95_abs(speed_after),
        "c_abs_bad_speed_layer_count": bad_speed_count,
    }


def centered_filter_days(target_day: date, window: int, start_day: date, end_day: date) -> list[date]:
    before = (window - 1) // 2
    after = window - 1 - before
    return [day for day in (target_day + timedelta(days=offset) for offset in range(-before, after + 1)) if start_day <= day <= end_day]


def regional_density_background(raw_prho: np.memmap, meta, obj: SelectedObject, nlev: int, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    lon = np.asarray(meta.x.values, dtype="f8")
    lat = np.asarray(meta.y.values, dtype="f8")
    lon_step = float(np.nanmedian(np.diff(lon)))
    lat_step = float(np.nanmedian(np.diff(lat)))
    lon_half = float(getattr(args, "regional_bg_lon_half_width_deg", 5.0))
    lat_half = float(getattr(args, "regional_bg_lat_half_width_deg", 2.0))
    exclude_r = float(getattr(args, "regional_bg_exclude_r", 1.5))

    lon_offsets = np.arange(-int(math.ceil(lon_half / lon_step)), int(math.ceil(lon_half / lon_step)) + 1)
    lat_min = obj.center_lat - lat_half
    lat_max = obj.center_lat + lat_half
    lat_idx = np.flatnonzero((lat >= lat_min) & (lat <= lat_max))
    if lat_idx.size == 0:
        return np.full(nlev, np.nan, dtype="f4"), np.zeros(nlev, dtype="f4")

    center_i = int(round(((obj.center_lon % 360.0) - float(lon[0])) / lon_step)) % lon.size
    lon_idx = (center_i + lon_offsets) % lon.size
    region_lon = lon[lon_idx]
    region_lat = lat[lat_idx]
    lon2, lat2 = np.meshgrid(region_lon, region_lat, indexing="ij")
    mx, my = meters_per_degree(obj.center_lat)
    dx_km = np.vectorize(lon_delta_deg)(lon2, obj.center_lon) * mx / 1000.0
    dy_km = (lat2 - obj.center_lat) * my / 1000.0
    keep = np.hypot(dx_km, dy_km) >= exclude_r * obj.radius_km
    denom = max(1, int(np.count_nonzero(keep)))

    rho_bg = np.full(nlev, np.nan, dtype="f4")
    valid_fraction = np.zeros(nlev, dtype="f4")
    for k in range(nlev):
        layer = np.asarray(raw_prho[np.ix_(lon_idx, lat_idx, np.array([k], dtype=int))][:, :, 0], dtype="f4")
        layer[np.abs(layer) > 1.0e30] = np.nan
        layer = normalize_density_units(layer)
        values = np.where(keep, layer, np.nan)
        valid = np.isfinite(values)
        valid_fraction[k] = float(np.count_nonzero(valid) / denom)
        if np.any(valid):
            rho_bg[k] = float(np.nanmedian(values))
    return rho_bg, valid_fraction


def regional_density_background_temporal(meta, obj: SelectedObject, nlev: int, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    scale_cache = get_thread_scale_cache(args)
    if scale_cache is not None and "regional_rho_bg_raw" in scale_cache and "regional_bg_valid_fraction" in scale_cache:
        return (
            np.asarray(scale_cache["regional_rho_bg_raw"], dtype="f4"),
            np.asarray(scale_cache["regional_bg_valid_fraction"], dtype="f4"),
        )
    window = max(1, int(getattr(args, "rebuild_density_filter_window_days", 10)))
    days = centered_filter_days(parse_iso_date(obj.date), window, parse_iso_date(str(getattr(args, "start", "1991-01-01"))), parse_iso_date(str(getattr(args, "end", "1991-01-19"))))
    data_root = Path(getattr(args, "data_root", DEFAULT_DATA_ROOT))
    expected = expected_dta_bytes(meta)
    profiles = []
    fractions = []
    for day in days:
        if not cached_daily_exists(data_root, "prho", day, expected):
            continue
        raw_prho = open_daily_memmap_cached(args, data_root, "prho", day, meta, expected)
        profile, valid_fraction = regional_density_background(raw_prho, meta, obj, nlev, args)
        profiles.append(profile)
        fractions.append(valid_fraction)
    if not profiles:
        return np.full(nlev, np.nan, dtype="f4"), np.zeros(nlev, dtype="f4")
    stacked = np.stack(profiles, axis=0)
    frac = np.stack(fractions, axis=0)
    return np.nanmean(stacked, axis=0).astype("f4"), np.nanmean(frac, axis=0).astype("f4")


def sigma_r_to_cells(sigma_r: float, x_over_r: np.ndarray) -> float:
    if sigma_r <= 0 or x_over_r.size < 2:
        return 0.0
    dr = float(np.nanmedian(np.diff(x_over_r)))
    return float(sigma_r / max(abs(dr), 1.0e-12))


def gradient_q95_abs(values: np.ndarray, dx_m: float, dy_m: float) -> float:
    dzdx = np.empty_like(values, dtype="f4")
    dzdy = np.empty_like(values, dtype="f4")
    for k in range(values.shape[0]):
        dzdy[k], dzdx[k] = np.gradient(values[k], dy_m, dx_m)
    return q95_abs(np.hypot(dzdx, dzdy))


def parse_ring(value: str) -> tuple[float, float]:
    parts = [float(item.strip()) for item in value.split(",") if item.strip()]
    if len(parts) != 2:
        raise ValueError("--density-bg-detrend-fit-ring must be inner,outer")
    return min(parts), max(parts)


def detrend_density_prime(rho_prime: np.ndarray, xxr: np.ndarray, yyr: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    order = int(getattr(args, "density_bg_detrend_order", 1))
    inner, outer = parse_ring(str(getattr(args, "density_bg_detrend_fit_ring", "1.5,4.0")))
    r = np.hypot(xxr, yyr)
    fit_mask = (r >= inner) & (r <= outer)
    out = np.asarray(rho_prime, dtype="f4").copy()
    x = xxr[fit_mask].astype("f8")
    y = yyr[fit_mask].astype("f8")
    for k in range(out.shape[0]):
        layer = out[k]
        values = layer[fit_mask].astype("f8")
        good = np.isfinite(values)
        if np.count_nonzero(good) < (3 if order == 1 else 1):
            continue
        if order == 0:
            trend = np.full_like(layer, float(np.nanmedian(values[good])), dtype="f4")
        else:
            vgood = values[good]
            med = float(np.nanmedian(vgood))
            mad = float(np.nanmedian(np.abs(vgood - med)))
            robust = good.copy()
            if np.isfinite(mad) and mad > 0:
                robust[good] = np.abs(vgood - med) <= 4.0 * 1.4826 * mad
            if np.count_nonzero(robust) < 3:
                robust = good
            coef = fit_plane_coefficients_no_lapack(x[robust], y[robust], values[robust])
            if coef is None:
                continue
            trend = (coef[0] + coef[1] * xxr + coef[2] * yyr).astype("f4")
        out[k] = layer - trend
    return out


def fit_plane_coefficients_no_lapack(x: np.ndarray, y: np.ndarray, values: np.ndarray) -> tuple[float, float, float] | None:
    x = np.asarray(x, dtype="f8")
    y = np.asarray(y, dtype="f8")
    v = np.asarray(values, dtype="f8")
    good = np.isfinite(x) & np.isfinite(y) & np.isfinite(v)
    if np.count_nonzero(good) < 3:
        return None
    x = x[good]
    y = y[good]
    v = v[good]
    n = float(v.size)
    sx = float(np.sum(x))
    sy = float(np.sum(y))
    sx2 = float(np.sum(x * x))
    sy2 = float(np.sum(y * y))
    sxy = float(np.sum(x * y))
    sv = float(np.sum(v))
    sxv = float(np.sum(x * v))
    syv = float(np.sum(y * v))
    matrix = [
        [n, sx, sy],
        [sx, sx2, sxy],
        [sy, sxy, sy2],
    ]
    rhs = [sv, sxv, syv]
    return solve_3x3(matrix, rhs)


def solve_3x3(matrix: list[list[float]], rhs: list[float]) -> tuple[float, float, float] | None:
    a = [[float(matrix[i][j]) for j in range(3)] + [float(rhs[i])] for i in range(3)]
    for col in range(3):
        pivot = max(range(col, 3), key=lambda row: abs(a[row][col]))
        if abs(a[pivot][col]) < 1.0e-12:
            return None
        if pivot != col:
            a[col], a[pivot] = a[pivot], a[col]
        scale = a[col][col]
        for j in range(col, 4):
            a[col][j] /= scale
        for row in range(3):
            if row == col:
                continue
            factor = a[row][col]
            for j in range(col, 4):
                a[row][j] -= factor * a[col][j]
    return float(a[0][3]), float(a[1][3]), float(a[2][3])


def estimate_translation(centers: pd.DataFrame, neighbors: dict[str, pd.DataFrame], obj: SelectedObject) -> tuple[float, float, bool, str]:
    prev_row = nearest_neighbor(neighbors.get("prev", pd.DataFrame()), obj)
    next_row = nearest_neighbor(neighbors.get("next", pd.DataFrame()), obj)
    mx, my = meters_per_degree(obj.center_lat)
    if prev_row is not None and next_row is not None:
        cx = lon_delta_deg(float(next_row["center_lon_refined"]), float(prev_row["center_lon_refined"])) * mx / (2.0 * 86400.0)
        cy = (float(next_row["center_lat_refined"]) - float(prev_row["center_lat_refined"])) * my / (2.0 * 86400.0)
        return cx, cy, True, "central_neighbor"
    if next_row is not None:
        cx = lon_delta_deg(float(next_row["center_lon_refined"]), obj.center_lon) * mx / 86400.0
        cy = (float(next_row["center_lat_refined"]) - obj.center_lat) * my / 86400.0
        return cx, cy, True, "forward_neighbor"
    if prev_row is not None:
        cx = lon_delta_deg(obj.center_lon, float(prev_row["center_lon_refined"])) * mx / 86400.0
        cy = (obj.center_lat - float(prev_row["center_lat_refined"])) * my / 86400.0
        return cx, cy, True, "backward_neighbor"
    return 0.0, 0.0, False, "missing_neighbor"


def estimate_translation_profile(
    centers: pd.DataFrame,
    neighbors: dict[str, pd.DataFrame],
    obj: SelectedObject,
    nlev: int,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    obj_rows = centers[centers["hua_object_id"].astype(str).eq(obj.hua_object_id) & centers["hua_pass"].astype(bool)].copy()
    if obj_rows.empty:
        cx_abs, cy_abs, valid, method = estimate_translation(centers, neighbors, obj)
        return np.full(nlev, cx_abs, dtype="f4"), np.full(nlev, cy_abs, dtype="f4"), np.full(nlev, valid, dtype=bool), f"{method}_fallback_surface"

    lon_col = "center_lon_refined" if "center_lon_refined" in obj_rows.columns else "center_lon"
    lat_col = "center_lat_refined" if "center_lat_refined" in obj_rows.columns else "center_lat"
    obj_rows["depth_index"] = obj_rows["depth_index"].astype(int)
    by_level = {int(row["depth_index"]): row for _, row in obj_rows.sort_values("depth_index").iterrows()}

    prev = neighbors.get("prev", pd.DataFrame())
    next_ = neighbors.get("next", pd.DataFrame())
    cx = np.full(nlev, np.nan, dtype="f4")
    cy = np.full(nlev, np.nan, dtype="f4")
    valid = np.zeros(nlev, dtype=bool)
    for k in range(nlev):
        row = by_level.get(k)
        if row is None:
            continue
        layer_obj = SelectedObject(
            hua_object_id=obj.hua_object_id,
            date=obj.date,
            polarity=obj.polarity,
            pass_layers=obj.pass_layers,
            max_jump_km=obj.max_jump_km,
            max_jump_over_r=obj.max_jump_over_r,
            radius_km=obj.radius_km,
            center_lon=float(row[lon_col]),
            center_lat=float(row[lat_col]),
        )
        prev_row = nearest_neighbor_same_depth(prev, layer_obj, k, args)
        next_row = nearest_neighbor_same_depth(next_, layer_obj, k, args)
        mx, my = meters_per_degree(layer_obj.center_lat)
        if prev_row is not None and next_row is not None:
            cx[k] = lon_delta_deg(float(next_row[lon_col]), float(prev_row[lon_col])) * mx / (2.0 * 86400.0)
            cy[k] = (float(next_row[lat_col]) - float(prev_row[lat_col])) * my / (2.0 * 86400.0)
            valid[k] = True
        elif next_row is not None:
            cx[k] = lon_delta_deg(float(next_row[lon_col]), layer_obj.center_lon) * mx / 86400.0
            cy[k] = (float(next_row[lat_col]) - layer_obj.center_lat) * my / 86400.0
            valid[k] = True
        elif prev_row is not None:
            cx[k] = lon_delta_deg(layer_obj.center_lon, float(prev_row[lon_col])) * mx / 86400.0
            cy[k] = (layer_obj.center_lat - float(prev_row[lat_col])) * my / 86400.0
            valid[k] = True

    cx_abs, cy_abs, surface_valid, surface_method = estimate_translation(centers, neighbors, obj)
    cx = fill_profile_gaps(cx, valid, cx_abs)
    cy = fill_profile_gaps(cy, valid, cy_abs)
    method = f"same_depth_center_tracking_{int(np.count_nonzero(valid))}of{nlev}"
    if not np.any(valid):
        method = f"{method}_fallback_{surface_method}"
        valid[:] = surface_valid
    return cx.astype("f4"), cy.astype("f4"), valid, method


def nearest_neighbor_same_depth(rows: pd.DataFrame, obj: SelectedObject, depth_index: int, args: argparse.Namespace) -> pd.Series | None:
    if rows.empty:
        return None
    lon_col = "center_lon_refined" if "center_lon_refined" in rows.columns else "center_lon"
    lat_col = "center_lat_refined" if "center_lat_refined" in rows.columns else "center_lat"
    part = rows[rows["depth_index"].astype(int).eq(int(depth_index)) & rows["hua_pass"].astype(bool)].copy()
    if part.empty:
        return None
    same = part[part["polarity"].astype(str).eq(obj.polarity)]
    if not same.empty:
        part = same
    mx, my = meters_per_degree(obj.center_lat)
    dist = np.asarray(
        np.hypot([lon_delta_deg(v, obj.center_lon) * mx / 1000.0 for v in part[lon_col]], (part[lat_col].astype(float) - obj.center_lat) * my / 1000.0),
        dtype="f8",
    )
    idx = int(np.nanargmin(dist))
    limit = min(float(getattr(args, "layer_tracking_max_distance_km", 250.0)), float(getattr(args, "layer_tracking_max_distance_r", 2.0)) * obj.radius_km)
    if float(dist[idx]) > limit:
        return None
    return part.iloc[idx]


def fill_profile_gaps(values: np.ndarray, valid: np.ndarray, fallback: float) -> np.ndarray:
    out = np.asarray(values, dtype="f8").copy()
    x = np.arange(out.size, dtype="f8")
    good = valid & np.isfinite(out)
    if np.count_nonzero(good) >= 2:
        out[~good] = np.interp(x[~good], x[good], out[good])
    elif np.count_nonzero(good) == 1:
        out[~good] = float(out[good][0])
    else:
        out[:] = float(fallback)
    return out.astype("f4")


def nearest_neighbor(rows: pd.DataFrame, obj: SelectedObject) -> pd.Series | None:
    if rows.empty:
        return None
    lon_col = "center_lon_refined" if "center_lon_refined" in rows.columns else "center_lon"
    lat_col = "center_lat_refined" if "center_lat_refined" in rows.columns else "center_lat"
    surface = rows[rows["depth_index"].astype(int).eq(0)].copy()
    if surface.empty:
        surface = rows.sort_values("depth_index").groupby("hua_object_id", as_index=False).first()
    same = surface[surface["polarity"].astype(str).eq(obj.polarity)]
    if not same.empty:
        surface = same
    mx, my = meters_per_degree(obj.center_lat)
    dist = np.asarray(
        np.hypot([lon_delta_deg(v, obj.center_lon) * mx / 1000.0 for v in surface[lon_col]], (surface[lat_col].astype(float) - obj.center_lat) * my / 1000.0),
        dtype="f8",
    )
    idx = int(np.nanargmin(dist))
    if float(dist[idx]) > max(250.0, 2.0 * obj.radius_km):
        return None
    return surface.iloc[idx]


def local_lon_lat_grid(xxr: np.ndarray, yyr: np.ndarray, lon0: float, lat0: float, radius_km: float) -> tuple[np.ndarray, np.ndarray]:
    mx, my = meters_per_degree(lat0)
    lon = (lon0 + xxr * radius_km * 1000.0 / mx) % 360.0
    lat = lat0 + yyr * radius_km * 1000.0 / my
    return lon, lat


def normalize_density_units(prho: np.ndarray) -> np.ndarray:
    finite = prho[np.isfinite(prho)]
    if finite.size and float(np.nanmedian(finite)) > 1000.0:
        return prho - 1000.0
    return prho


def finite_median_or_zero(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype="f8")
    finite = finite[np.isfinite(finite)]
    return float(np.nanmedian(finite)) if finite.size else 0.0


def finite_max_abs(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype="f8")
    finite = finite[np.isfinite(finite)]
    return float(np.nanmax(np.abs(finite))) if finite.size else float("nan")


def nan_gaussian_smooth_1d(values: np.ndarray, sigma: float) -> np.ndarray:
    arr = np.asarray(values, dtype="f8")
    if sigma <= 0:
        return arr.astype("f4")
    valid = np.isfinite(arr)
    if not np.any(valid):
        return np.full(arr.shape, np.nan, dtype="f4")
    weights = ndimage.gaussian_filter1d(valid.astype("f8"), sigma=sigma, mode="nearest")
    smoothed = ndimage.gaussian_filter1d(np.where(valid, arr, 0.0), sigma=sigma, mode="nearest")
    return np.divide(smoothed, weights, out=np.full_like(smoothed, np.nan), where=weights > 1.0e-8).astype("f4")


def nan_gaussian_smooth_3d(values: np.ndarray, sigma_cells: float | tuple[float, float]) -> np.ndarray:
    sigma_arr = np.asarray(sigma_cells, dtype="f8")
    if np.all(sigma_arr <= 0):
        return values
    out = np.empty_like(values, dtype="f4")
    for k in range(values.shape[0]):
        out[k] = nan_gaussian_smooth_2d(values[k], sigma_cells)
    return out


def nan_gaussian_smooth_2d(values: np.ndarray, sigma_cells: float) -> np.ndarray:
    arr = np.asarray(values, dtype="f8")
    valid = np.isfinite(arr)
    weights = ndimage.gaussian_filter(valid.astype("f8"), sigma=sigma_cells, mode="nearest")
    smoothed = ndimage.gaussian_filter(np.where(valid, arr, 0.0), sigma=sigma_cells, mode="nearest")
    return np.divide(smoothed, weights, out=np.full_like(smoothed, np.nan), where=weights > 1.0e-8).astype("f4")


COMPOSITE_3D_KEYS = [
    "term1_m_s",
    "term2_m_s",
    "rebuild_w_m_s",
    "ofes_w_native_m_s",
    "ofes_w_native_meso_m_s",
    "ofes_w_native_meso_aligned_m_s",
    "prho_for_rebuild",
    "rho_prime_for_rebuild",
    "z_rho_anom_m",
]
COMPOSITE_SECTION_KEYS = COMPOSITE_3D_KEYS
COMPOSITE_PROFILE_KEYS = ["rho_bg", "rho_z_used", "prho_center_profile", "rho_prime_center_profile", "z_rho_center_profile"]


def init_composite_accumulator(grid: dict[str, object], target_lat: float, polarity: str, args: argparse.Namespace, multipole_class: str = "all") -> dict[str, object]:
    depth = np.asarray(grid["depth_m"], dtype="f4")
    x = np.asarray(grid["x_over_r"], dtype="f4")
    y = np.asarray(grid["y_over_r"], dtype="f4")
    shape3 = (depth.size, y.size, x.size)
    shape2 = (depth.size, x.size)
    return {
        "target_lat": float(target_lat),
        "polarity": polarity,
        "depth_m": depth,
        "x_over_r": x,
        "y_over_r": y,
        "cressman_radius_r": float(args.cressman_radius_r),
        "cressman_min_objects": int(args.cressman_min_objects),
        "multipole_class": str(multipole_class),
        "multipole_depth_min_m": float(getattr(args, "multipole_depth_min_m", 300.0)),
        "multipole_depth_max_m": float(getattr(args, "multipole_depth_max_m", 500.0)),
        "multipole_radius_inner_r": float(getattr(args, "multipole_radius_inner_r", 0.0)),
        "multipole_radius_outer_r": float(getattr(args, "multipole_radius_outer_r", 1.0)),
        "multipole_azimuth_count": int(getattr(args, "multipole_azimuth_count", 60)),
        "multipole_min_valid_azimuth_fraction": float(getattr(args, "multipole_min_valid_azimuth_fraction", 0.80)),
        "multipole_min_sector_valid_fraction": float(getattr(args, "multipole_min_sector_valid_fraction", 0.35)),
        "multipole_boundary_max_nan_fraction": float(getattr(args, "multipole_boundary_max_nan_fraction", 0.35)),
        "multipole_min_amp_1e6_m_s": float(getattr(args, "multipole_min_amp_1e6_m_s", 0.5)),
        "multipole_min_snr": float(getattr(args, "multipole_min_snr", 2.0)),
        "multipole_min_harmonic_dominance": float(getattr(args, "multipole_min_harmonic_dominance", 1.1)),
        "native_w_meso_filter": str(getattr(args, "native_w_meso_filter", "temporal10d_spatial50_500km")),
        "native_w_meso_time_window_days": int(getattr(args, "native_w_meso_time_window_days", 10)),
        "native_w_meso_small_cutoff_km": float(getattr(args, "native_w_meso_small_cutoff_km", 50.0)),
        "native_w_meso_large_cutoff_km": float(getattr(args, "native_w_meso_large_cutoff_km", 500.0)),
        "native_w_phase_align": bool(getattr(args, "native_w_phase_align", True)),
        "object_count": 0,
        "object_ids": [],
        "sum3d": {key: np.zeros(shape3, dtype="f8") for key in COMPOSITE_3D_KEYS},
        "count3d": {key: np.zeros(shape3, dtype="u2") for key in COMPOSITE_3D_KEYS},
        "sum_section": {key: np.zeros(shape2, dtype="f8") for key in COMPOSITE_SECTION_KEYS},
        "count_section": {key: np.zeros(shape2, dtype="u2") for key in COMPOSITE_SECTION_KEYS},
        "sum_profile": {key: np.zeros(depth.shape, dtype="f8") for key in COMPOSITE_PROFILE_KEYS},
        "count_profile": {key: np.zeros(depth.shape, dtype="u2") for key in COMPOSITE_PROFILE_KEYS},
    }


def update_composite_accumulator(accumulator: dict[str, object], grid: dict[str, object], args: argparse.Namespace) -> None:
    accumulator["object_count"] = int(accumulator["object_count"]) + 1
    accumulator["object_ids"].append(str(grid["hua_object_id"]))
    radius_r = float(args.cressman_radius_r)
    x = np.asarray(accumulator["x_over_r"], dtype="f8")
    y = np.asarray(accumulator["y_over_r"], dtype="f8")
    kernel2d = cressman_kernel_2d(x, y, radius_r)
    kernel1d = cressman_kernel_1d(x, radius_r)

    for key in COMPOSITE_3D_KEYS:
        mapped, support = cressman_map_3d(np.asarray(grid[key], dtype="f4"), kernel2d)
        add_composite_array(accumulator["sum3d"][key], accumulator["count3d"][key], mapped, support)

    sections = crossing_sections_for_keys(grid, COMPOSITE_SECTION_KEYS)
    for key in COMPOSITE_SECTION_KEYS:
        mapped, support = cressman_map_section(np.asarray(sections[key], dtype="f4"), kernel1d)
        add_composite_array(accumulator["sum_section"][key], accumulator["count_section"][key], mapped, support)

    cy = len(y) // 2
    cx = len(x) // 2
    profile_values = {
        "rho_bg": np.asarray(grid["rho_bg"], dtype="f4"),
        "rho_z_used": np.asarray(grid["rho_z_used"], dtype="f4"),
        "prho_center_profile": np.asarray(grid["prho_for_rebuild"], dtype="f4")[:, cy, cx],
        "rho_prime_center_profile": np.asarray(grid["rho_prime_for_rebuild"], dtype="f4")[:, cy, cx],
        "z_rho_center_profile": np.asarray(grid["z_rho_anom_m"], dtype="f4")[:, cy, cx],
    }
    for key, values in profile_values.items():
        valid = np.isfinite(values)
        accumulator["sum_profile"][key][valid] += values[valid]
        accumulator["count_profile"][key][valid] += 1


def finalize_composite_accumulator(accumulator: dict[str, object], args: argparse.Namespace) -> dict[str, object]:
    min_objects = int(args.cressman_min_objects)
    out: dict[str, object] = {
        "target_lat": float(accumulator["target_lat"]),
        "polarity": str(accumulator["polarity"]),
        "object_count": int(accumulator["object_count"]),
        "source_object_ids": ",".join(accumulator["object_ids"]),
        "depth_m": np.asarray(accumulator["depth_m"], dtype="f4"),
        "x_over_r": np.asarray(accumulator["x_over_r"], dtype="f4"),
        "y_over_r": np.asarray(accumulator["y_over_r"], dtype="f4"),
        "cressman_radius_r": float(accumulator["cressman_radius_r"]),
        "cressman_min_objects": min_objects,
        "multipole_class": str(accumulator["multipole_class"]),
        "multipole_depth_min_m": float(accumulator["multipole_depth_min_m"]),
        "multipole_depth_max_m": float(accumulator["multipole_depth_max_m"]),
        "multipole_radius_inner_r": float(accumulator["multipole_radius_inner_r"]),
        "multipole_radius_outer_r": float(accumulator["multipole_radius_outer_r"]),
        "multipole_azimuth_count": int(accumulator["multipole_azimuth_count"]),
        "multipole_min_valid_azimuth_fraction": float(accumulator["multipole_min_valid_azimuth_fraction"]),
        "multipole_min_sector_valid_fraction": float(accumulator["multipole_min_sector_valid_fraction"]),
        "multipole_boundary_max_nan_fraction": float(accumulator["multipole_boundary_max_nan_fraction"]),
        "multipole_min_amp_1e6_m_s": float(accumulator["multipole_min_amp_1e6_m_s"]),
        "multipole_min_snr": float(accumulator["multipole_min_snr"]),
        "multipole_min_harmonic_dominance": float(accumulator["multipole_min_harmonic_dominance"]),
        "native_w_meso_filter": str(accumulator["native_w_meso_filter"]),
        "native_w_meso_time_window_days": int(accumulator["native_w_meso_time_window_days"]),
        "native_w_meso_small_cutoff_km": float(accumulator["native_w_meso_small_cutoff_km"]),
        "native_w_meso_large_cutoff_km": float(accumulator["native_w_meso_large_cutoff_km"]),
        "native_w_phase_align": bool(accumulator["native_w_phase_align"]),
        "composite_method": "strict_crossing_cressman_object_support",
    }
    for key in COMPOSITE_3D_KEYS:
        values = finalize_sum_count(accumulator["sum3d"][key], accumulator["count3d"][key], min_objects)
        out[f"composite_{key}"] = values
        out[f"support_objects_{key}"] = accumulator["count3d"][key]
    for key in COMPOSITE_SECTION_KEYS:
        values = finalize_sum_count(accumulator["sum_section"][key], accumulator["count_section"][key], min_objects)
        out[f"section_{key}"] = values
        out[f"section_support_objects_{key}"] = accumulator["count_section"][key]
    for key in COMPOSITE_PROFILE_KEYS:
        values = finalize_sum_count(accumulator["sum_profile"][key], accumulator["count_profile"][key], 1)
        out[f"profile_{key}"] = values
        out[f"profile_support_objects_{key}"] = accumulator["count_profile"][key]

    rebuild = out["composite_rebuild_w_m_s"]
    native = out["composite_ofes_w_native_m_s"]
    native_meso_aligned = out["composite_ofes_w_native_meso_aligned_m_s"]
    section_rebuild = out["section_rebuild_w_m_s"]
    section_native = out["section_ofes_w_native_m_s"]
    section_native_meso_aligned = out["section_ofes_w_native_meso_aligned_m_s"]
    support = out["support_objects_rebuild_w_m_s"]
    out.update(
        {
            "corr_rebuild_native_3d": spatial_corr(rebuild, native),
            "corr_rebuild_native_section": spatial_corr(section_rebuild, section_native),
            "q95_abs_rebuild_1e6_m_s": q95_abs(rebuild) * 1.0e6,
            "q95_abs_native_1e6_m_s": q95_abs(native) * 1.0e6,
            "q95_abs_native_meso_aligned_1e6_m_s": q95_abs(native_meso_aligned) * 1.0e6,
            "q95_abs_section_rebuild_1e6_m_s": q95_abs(section_rebuild) * 1.0e6,
            "q95_abs_section_native_1e6_m_s": q95_abs(section_native) * 1.0e6,
            "q95_abs_section_native_meso_aligned_1e6_m_s": q95_abs(section_native_meso_aligned) * 1.0e6,
            "eta_rho_q95_abs_m": q95_abs(out["composite_z_rho_anom_m"]),
            "rho_prime_q95_abs": q95_abs(out["composite_rho_prime_for_rebuild"]),
            "rho_z_q95_abs": q95_abs(out["profile_rho_z_used"]),
            "valid_grid_fraction": float(np.isfinite(rebuild).sum() / rebuild.size),
            "valid_section_fraction": float(np.isfinite(section_rebuild).sum() / section_rebuild.size),
            "mean_support_objects": float(np.nanmean(np.where(support > 0, support, np.nan))),
            "max_support_objects": int(np.nanmax(support)) if support.size else 0,
        }
    )
    return out


def attach_reference_native(composite: dict[str, object], all_composite: dict[str, object]) -> None:
    composite["reference_multipole_class"] = "all"
    composite["reference_composite_ofes_w_native_m_s"] = np.asarray(all_composite["composite_ofes_w_native_m_s"], dtype="f4")
    composite["reference_section_ofes_w_native_m_s"] = np.asarray(all_composite["section_ofes_w_native_m_s"], dtype="f4")
    composite["reference_object_count"] = int(all_composite["object_count"])
    composite["corr_rebuild_reference_native_section"] = spatial_corr(
        np.asarray(composite["section_rebuild_w_m_s"], dtype="f4"),
        np.asarray(composite["reference_section_ofes_w_native_m_s"], dtype="f4"),
    )
    composite["corr_class_native_reference_native_section"] = spatial_corr(
        np.asarray(composite["section_ofes_w_native_m_s"], dtype="f4"),
        np.asarray(composite["reference_section_ofes_w_native_m_s"], dtype="f4"),
    )


def write_composite_outputs(composite: dict[str, object], token: str, polarity: str, multipole_class: str, grids_dir: Path, figures_dir: Path) -> dict[str, object]:
    class_token = safe_token(multipole_class)
    stem = f"{token}_{polarity}_{class_token}_cressman"
    npz_path = grids_dir / f"composite_{stem}.npz"
    json_path = grids_dir / f"composite_{stem}.json"
    np.savez_compressed(npz_path, **{key: value for key, value in composite.items() if isinstance(value, np.ndarray)})
    scalar_payload = {key: json_safe(value) for key, value in composite.items() if not isinstance(value, np.ndarray)}
    write_json(json_path, scalar_payload)

    native_prefix = "native_w_meso_aligned" if "composite_ofes_w_native_meso_aligned_m_s" in composite else "native_w"
    native_section = figures_dir / f"{native_prefix}_cross_section_{stem}.png"
    w_focus = figures_dir / f"{native_prefix}_focus_300_500m_450m_surface_{stem}.png"
    density_profiles = figures_dir / f"density_profiles_{stem}.png"
    density_sections = figures_dir / f"density_sections_{stem}.png"
    w_section = figures_dir / f"ofes_w_cross_section_{stem}.png"
    w_slices = figures_dir / f"ofes_w_slices_{stem}.png"
    plot_composite_native_w_cross_section_pillow(composite, native_section)
    plot_composite_w_cross_section_pillow(composite, w_section)
    plot_composite_w_slices_pillow(composite, w_slices)
    plot_composite_native_w_focus_pillow(composite, w_focus)
    plot_composite_density_profiles_pillow(composite, density_profiles)
    plot_composite_density_sections_pillow(composite, density_sections)
    return {
        **scalar_payload,
        "grid_npz": str(npz_path),
        "grid_json": str(json_path),
        "native_w_cross_section_image": str(native_section),
        "w_cross_section_image": str(w_section),
        "w_slices_image": str(w_slices),
        "native_w_focus_image": str(w_focus),
        "density_profiles_image": str(density_profiles),
        "density_sections_image": str(density_sections),
        "status": "ok",
    }


def write_region_composite_outputs(
    composite: dict[str, object],
    token: str,
    region: str,
    polarity: str,
    multipole_class: str,
    grids_dir: Path,
    figures_dir: Path,
) -> dict[str, object]:
    region_token = safe_token(region)
    class_token = safe_token(multipole_class)
    polarity_token = safe_token(polarity)
    stem = f"{token}_{region_token}_{polarity_token}_{class_token}_cressman"
    npz_path = grids_dir / f"composite_{stem}.npz"
    json_path = grids_dir / f"composite_{stem}.json"
    np.savez_compressed(npz_path, **{key: value for key, value in composite.items() if isinstance(value, np.ndarray)})
    scalar_payload = {key: json_safe(value) for key, value in composite.items() if not isinstance(value, np.ndarray)}
    write_json(json_path, scalar_payload)

    native_prefix = "native_w_meso_aligned" if "composite_ofes_w_native_meso_aligned_m_s" in composite else "native_w"
    w_section = figures_dir / f"{native_prefix}_cross_section_{stem}.png"
    w_focus = figures_dir / f"{native_prefix}_focus_300_500m_450m_surface_{stem}.png"
    plot_composite_native_w_cross_section_pillow(composite, w_section)
    plot_composite_native_w_focus_pillow(composite, w_focus)
    return {
        **scalar_payload,
        "grid_npz": str(npz_path),
        "grid_json": str(json_path),
        "native_w_cross_section_image": str(w_section),
        "native_w_focus_image": str(w_focus),
        "status": "ok",
    }


def cressman_kernel_2d(x: np.ndarray, y: np.ndarray, radius_r: float) -> np.ndarray:
    dx = float(np.nanmedian(np.abs(np.diff(x)))) if x.size > 1 else radius_r
    dy = float(np.nanmedian(np.abs(np.diff(y)))) if y.size > 1 else radius_r
    nx = max(1, int(math.ceil(radius_r / max(dx, 1.0e-12))))
    ny = max(1, int(math.ceil(radius_r / max(dy, 1.0e-12))))
    ox = np.arange(-nx, nx + 1, dtype="f8") * dx
    oy = np.arange(-ny, ny + 1, dtype="f8") * dy
    xx, yy = np.meshgrid(ox, oy, indexing="xy")
    d2 = xx * xx + yy * yy
    r2 = radius_r * radius_r
    weights = np.where(d2 <= r2, (r2 - d2) / np.maximum(r2 + d2, 1.0e-12), 0.0)
    return weights.astype("f4")


def cressman_kernel_1d(x: np.ndarray, radius_r: float) -> np.ndarray:
    dx = float(np.nanmedian(np.abs(np.diff(x)))) if x.size > 1 else radius_r
    nx = max(1, int(math.ceil(radius_r / max(dx, 1.0e-12))))
    ox = np.arange(-nx, nx + 1, dtype="f8") * dx
    d2 = ox * ox
    r2 = radius_r * radius_r
    weights = np.where(d2 <= r2, (r2 - d2) / np.maximum(r2 + d2, 1.0e-12), 0.0)
    return weights.astype("f4")


def cressman_map_3d(values: np.ndarray, kernel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    out = np.full(values.shape, np.nan, dtype="f4")
    support = np.zeros(values.shape, dtype=bool)
    for k in range(values.shape[0]):
        valid = np.isfinite(values[k])
        if not np.any(valid):
            continue
        numerator = ndimage.convolve(np.where(valid, values[k], 0.0).astype("f4"), kernel, mode="constant", cval=0.0)
        denominator = ndimage.convolve(valid.astype("f4"), kernel, mode="constant", cval=0.0)
        layer = np.divide(numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator > 1.0e-8)
        out[k] = layer
        support[k] = np.isfinite(layer)
    return out, support


def cressman_map_section(values: np.ndarray, kernel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    out = np.full(values.shape, np.nan, dtype="f4")
    support = np.zeros(values.shape, dtype=bool)
    for k in range(values.shape[0]):
        valid = np.isfinite(values[k])
        if not np.any(valid):
            continue
        numerator = ndimage.convolve1d(np.where(valid, values[k], 0.0).astype("f4"), kernel, mode="constant", cval=0.0)
        denominator = ndimage.convolve1d(valid.astype("f4"), kernel, mode="constant", cval=0.0)
        line = np.divide(numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator > 1.0e-8)
        out[k] = line
        support[k] = np.isfinite(line)
    return out, support


def add_composite_array(total: np.ndarray, count: np.ndarray, mapped: np.ndarray, support: np.ndarray) -> None:
    valid = support & np.isfinite(mapped)
    total[valid] += mapped[valid]
    count[valid] += 1


def finalize_sum_count(total: np.ndarray, count: np.ndarray, min_count: int) -> np.ndarray:
    threshold = max(1, int(min_count))
    return np.divide(total, count, out=np.full(total.shape, np.nan, dtype="f4"), where=count >= threshold).astype("f4")


def crossing_sections_for_keys(grid: dict[str, object], keys: list[str]) -> dict[str, np.ndarray | float]:
    x = np.asarray(grid["x_over_r"], dtype="f8")
    y = np.asarray(grid["y_over_r"], dtype="f8")
    depth = np.asarray(grid["depth_m"], dtype="f8")
    target_y = (float(grid["target_lat"]) - float(grid["center_lat"])) * meters_per_degree(float(grid["center_lat"]))[1] / (float(grid["radius_km"]) * 1000.0)
    if y.size < 2:
        y_index = 0.0
        center_y_index = 0.0
    else:
        y_index = (target_y - float(y[0])) / float(y[1] - y[0])
        center_y_index = (0.0 - float(y[0])) / float(y[1] - y[0])
    z_index = np.arange(depth.size, dtype="f8")[:, None]
    x_index = np.arange(x.size, dtype="f8")[None, :]
    out: dict[str, np.ndarray | float] = {"x_over_r": x, "depth_m": depth, "target_y_over_r": float(target_y)}
    outside = target_y < float(y[0]) or target_y > float(y[-1])
    for key in keys:
        arr = np.asarray(grid[key], dtype="f4")
        key_y_index = center_y_index if "aligned" in key else y_index
        coords = np.vstack(
            [
                np.broadcast_to(z_index, (depth.size, x.size)).ravel(),
                np.full(depth.size * x.size, key_y_index, dtype="f8"),
                np.broadcast_to(x_index, (depth.size, x.size)).ravel(),
            ]
        )
        sampled = ndimage.map_coordinates(arr, coords, order=1, mode="nearest").reshape(depth.size, x.size)
        if outside and "aligned" not in key:
            sampled[:] = np.nan
        out[key] = sampled.astype("f4")
    return out


def plot_four_panel(grid: dict[str, object], png_path: Path, pdf_path: Path, backend: str) -> None:
    if backend == "matplotlib":
        try:
            plot_four_panel_matplotlib(grid, png_path, pdf_path)
            return
        except BaseException as exc:
            print(f"[plot-warning] matplotlib failed, falling back to pillow: {exc}", flush=True)
    plot_four_panel_pillow(grid, png_path)


def plot_four_panel_matplotlib(grid: dict[str, object], png_path: Path, pdf_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = grid["x_over_r"]
    y = grid["y_over_r"]
    fields = [
        ("term1", np.nanmean(grid["term1_m_s"], axis=0)),
        ("term2", np.nanmean(grid["term2_m_s"], axis=0)),
        ("rebuild W", np.nanmean(grid["rebuild_w_m_s"], axis=0)),
        ("OFES native W", np.nanmean(grid["ofes_w_native_m_s"], axis=0)),
    ]
    vals = np.concatenate([arr[np.isfinite(arr)] for _, arr in fields if np.isfinite(arr).any()])
    lim = float(np.nanpercentile(np.abs(vals), 95)) * 1.0e6 if vals.size else 1.0
    lim = max(lim, 1.0e-12)
    fig, axes = plt.subplots(2, 2, figsize=(13, 11), constrained_layout=True)
    th = np.linspace(0, 2 * np.pi, 361)
    for ax, (title, data) in zip(axes.ravel(), fields):
        mesh = ax.pcolormesh(x, y, data * 1.0e6, shading="auto", cmap="RdBu_r", vmin=-lim, vmax=lim)
        ax.contour(x, y, data * 1.0e6, levels=np.linspace(-lim, lim, 9), colors="0.25", linewidths=0.55, alpha=0.7)
        ax.plot(np.cos(th), np.sin(th), "k-", lw=1.2)
        ax.plot(4 * np.cos(th), 4 * np.sin(th), "k-", lw=1.2)
        ax.plot(0, 0, "k.", ms=16)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(title)
        ax.set_xlabel("x/R")
        ax.set_ylabel("y/R")
        fig.colorbar(mesh, ax=ax, shrink=0.82)
    fig.suptitle(
        f"OFES W rebuild diagnostic | {grid['hua_object_id']} | {grid['date']} | corr={grid['corr_rebuild_native']:.3f} | q95 rebuild/native={grid['q95_abs_rebuild_1e6_m_s']:.2g}/{grid['q95_abs_native_1e6_m_s']:.2g} x10^-6 m/s"
    )
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=180)
    fig.savefig(pdf_path)
    plt.close(fig)


def plot_four_panel_pillow(grid: dict[str, object], png_path: Path) -> None:
    x = grid["x_over_r"]
    y = grid["y_over_r"]
    fields = [
        ("term1", np.nanmean(grid["term1_m_s"], axis=0)),
        ("term2", np.nanmean(grid["term2_m_s"], axis=0)),
        ("rebuild W", np.nanmean(grid["rebuild_w_m_s"], axis=0)),
        ("OFES native W", np.nanmean(grid["ofes_w_native_m_s"], axis=0)),
    ]
    vals = np.concatenate([arr[np.isfinite(arr)] for _, arr in fields if np.isfinite(arr).any()])
    lim = max(float(np.nanpercentile(np.abs(vals), 95)) * 1.0e6 if vals.size else 1.0, 1.0e-12)
    canvas = Image.new("RGB", (1800, 1450), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(30)
    main_font = load_font(20)
    small_font = load_font(15)
    draw.text((40, 30), f"OFES W rebuild diagnostic | {grid['hua_object_id']} | {grid['date']}", fill=(20, 24, 32), font=title_font)
    draw.text((40, 70), f"corr={grid['corr_rebuild_native']:.3f}, corr(-rebuild,W)={grid['corr_minus_rebuild_native']:.3f}, q95 rebuild/native={grid['q95_abs_rebuild_1e6_m_s']:.2g}/{grid['q95_abs_native_1e6_m_s']:.2g} x10^-6 m/s", fill=(80, 88, 100), font=small_font)
    boxes = [(60, 125, 860, 735), (940, 125, 1740, 735), (60, 800, 860, 1410), (940, 800, 1740, 1410)]
    for box, (name, data) in zip(boxes, fields):
        draw_field_pillow(canvas, box, x, y, data * 1.0e6, -lim, lim, name, main_font, small_font)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(png_path)
    canvas.save(png_path.with_suffix(".pdf"), "PDF", resolution=180.0)


def draw_field_pillow(canvas: Image.Image, box: tuple[int, int, int, int], x: np.ndarray, y: np.ndarray, data: np.ndarray, vmin: float, vmax: float, title: str, main_font, small_font) -> None:
    draw = ImageDraw.Draw(canvas)
    draw.rectangle(box, outline=(180, 186, 196), width=1)
    available_w = box[2] - box[0] - 95
    available_h = box[3] - box[1] - 95
    side = min(available_w, available_h)
    left = box[0] + 55 + max(0, available_w - side) // 2
    top = box[1] + 38 + max(0, available_h - side) // 2
    plot = (left, top, left + side, top + side)
    img = array_to_rgb(data, vmin, vmax).resize((plot[2] - plot[0], plot[3] - plot[1]), Image.Resampling.BILINEAR)
    canvas.paste(img, plot[:2])
    th = np.linspace(0, 2 * np.pi, 361)
    for rr in [1.0, 4.0]:
        pts = [map_xy(plot, rr * math.cos(t), rr * math.sin(t), float(x[0]), float(x[-1]), float(y[0]), float(y[-1])) for t in th]
        draw.line(pts, fill=(0, 0, 0), width=2)
    cx, cy = map_xy(plot, 0.0, 0.0, float(x[0]), float(x[-1]), float(y[0]), float(y[-1]))
    draw.ellipse((cx - 4, cy - 4, cx + 4, cy + 4), fill=(0, 0, 0))
    draw.text((box[0] + 10, box[1] + 10), title, fill=(30, 36, 48), font=main_font)
    draw.text((plot[0], box[3] - 30), "x/R", fill=(80, 88, 100), font=small_font)
    draw.text((box[0] + 10, plot[1]), "y/R", fill=(80, 88, 100), font=small_font)
    draw.text((box[0] + 10, box[1] + 35), f"+/- {vmax:.2g} x10^-6 m/s", fill=(80, 88, 100), font=small_font)
    colorbar_box = (plot[2] + 14, plot[1], min(plot[2] + 34, box[2] - 12), plot[3])
    draw_colorbar_pillow(canvas, colorbar_box, vmin, vmax, small_font)


def plot_crossing_four_panel_pillow(grid: dict[str, object], png_path: Path) -> None:
    sections = crossing_sections(grid)
    fields = [
        ("term1", sections["term1_m_s"]),
        ("term2", sections["term2_m_s"]),
        ("rebuild W", sections["rebuild_w_m_s"]),
        ("OFES native W", sections["ofes_w_native_m_s"]),
    ]
    vals = np.concatenate([arr[np.isfinite(arr)] for _, arr in fields if np.isfinite(arr).any()])
    lim = max(float(np.nanpercentile(np.abs(vals), 95)) * 1.0e6 if vals.size else 1.0, 1.0e-12)
    corr = spatial_corr(sections["rebuild_w_m_s"], sections["ofes_w_native_m_s"])
    corr_minus = spatial_corr(-sections["rebuild_w_m_s"], sections["ofes_w_native_m_s"])
    canvas = Image.new("RGB", (1900, 1450), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(30)
    main_font = load_font(20)
    small_font = load_font(15)
    target_lat = float(grid["target_lat"])
    draw.text((45, 30), f"OFES W crossing section | {lat_label(target_lat)} | {grid['hua_object_id']} | {grid['date']}", fill=(20, 24, 32), font=title_font)
    draw.text(
        (45, 70),
        f"{grid['crossing_selection_mode']}; offset={grid['crossing_distance_over_r']:.2f}R; y/R={sections['target_y_over_r']:.2f}; corr={corr:.3f}; corr(-rebuild,W)={corr_minus:.3f}; +/-{lim:.2g} x10^-6 m/s",
        fill=(80, 88, 100),
        font=small_font,
    )
    boxes = [(65, 125, 915, 715), (995, 125, 1845, 715), (65, 790, 915, 1380), (995, 790, 1845, 1380)]
    for box, (name, data) in zip(boxes, fields):
        draw_section_field_pillow(canvas, box, sections["x_over_r"], sections["depth_m"], data * 1.0e6, -lim, lim, name, main_font, small_font)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(png_path)
    canvas.save(png_path.with_suffix(".pdf"), "PDF", resolution=180.0)


def preferred_native_w_composite_key(composite: dict[str, object], section: bool) -> tuple[str, str]:
    prefix = "section_" if section else "composite_"
    candidates = [
        ("ofes_w_native_meso_aligned_m_s", "native W meso 50-500 km, mode-1 aligned"),
        ("ofes_w_native_meso_m_s", "native W meso 50-500 km, unaligned"),
        ("ofes_w_native_m_s", "native W layer-center"),
    ]
    for key, label in candidates:
        full_key = f"{prefix}{key}"
        if full_key in composite:
            return full_key, label
    return f"{prefix}ofes_w_native_m_s", "native W layer-center"


def plot_composite_native_w_cross_section_pillow(composite: dict[str, object], png_path: Path) -> None:
    x = np.asarray(composite["x_over_r"], dtype="f4")
    depth = np.asarray(composite["depth_m"], dtype="f4")
    native_key, native_label = preferred_native_w_composite_key(composite, section=True)
    native = np.asarray(composite[native_key], dtype="f4")
    vals = native[np.isfinite(native)]
    lim = max(float(np.nanpercentile(np.abs(vals), 95)) * 1.0e6 if vals.size else 1.0, 1.0e-12)
    canvas = Image.new("RGB", (1050, 900), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(28)
    main_font = load_font(20)
    small_font = load_font(15)
    class_text = str(composite.get("multipole_class", "all"))
    region_text = str(composite.get("region", "")).strip()
    region_suffix = f" | {region_text}" if region_text else ""
    draw.text(
        (35, 28),
        f"Native W crossing section | {lat_label(float(composite['target_lat']))} | {composite['polarity']} | {class_text}{region_suffix}",
        fill=(20, 24, 32),
        font=title_font,
    )
    draw.text(
        (35, 68),
        f"{native_label}; objects={composite['object_count']}; Cressman R={composite['cressman_radius_r']}R; min objects={composite['cressman_min_objects']}; +/-{lim:.2g} x10^-6 m/s",
        fill=(80, 88, 100),
        font=small_font,
    )
    draw_section_field_pillow(canvas, (70, 125, 980, 830), x, depth, native * 1.0e6, -lim, lim, native_label, main_font, small_font)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(png_path)
    canvas.save(png_path.with_suffix(".pdf"), "PDF", resolution=180.0)


def plot_composite_native_w_focus_pillow(composite: dict[str, object], png_path: Path) -> None:
    x = np.asarray(composite["x_over_r"], dtype="f4")
    y = np.asarray(composite["y_over_r"], dtype="f4")
    depth = np.asarray(composite["depth_m"], dtype="f4")
    native_key, native_label = preferred_native_w_composite_key(composite, section=False)
    native = np.asarray(composite[native_key], dtype="f4")
    rows = [
        ("0-100 m surface", depth_average_indices(depth, 0.0, 100.0)),
        ("300-500 m mean", depth_average_indices(depth, 300.0, 500.0)),
        ("450 m slice", np.array([int(np.nanargmin(np.abs(depth - 450.0)))])),
    ]
    fields2d: list[tuple[str, np.ndarray]] = []
    for row_name, idx in rows:
        with np.errstate(invalid="ignore"):
            fields2d.append((row_name, np.nanmean(native[idx], axis=0)))
    finite_parts = [arr[np.isfinite(arr)] for _, arr in fields2d if np.isfinite(arr).any()]
    vals = np.concatenate(finite_parts) if finite_parts else np.array([], dtype="f4")
    lim = max(float(np.nanpercentile(np.abs(vals), 95)) * 1.0e6 if vals.size else 1.0, 1.0e-12)
    canvas = Image.new("RGB", (760, 1770), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(26)
    main_font = load_font(16)
    small_font = load_font(12)
    region_text = str(composite.get("region", "")).strip()
    region_suffix = f" | {region_text}" if region_text else ""
    draw.text(
        (35, 25),
        f"Native W focus | {lat_label(float(composite['target_lat']))} | {composite['polarity']} | {composite.get('multipole_class', 'all')}{region_suffix}",
        fill=(20, 24, 32),
        font=title_font,
    )
    draw.text((35, 62), f"{native_label}; objects={composite['object_count']}; +/-{lim:.2g} x10^-6 m/s", fill=(80, 88, 100), font=small_font)
    for row_idx, (row_name, data) in enumerate(fields2d):
        box = (70, 130 + row_idx * 535, 680, 630 + row_idx * 535)
        draw.text((20, 105 + row_idx * 535), row_name, fill=(30, 36, 48), font=main_font)
        draw_field_pillow(canvas, box, x, y, data * 1.0e6, -lim, lim, native_label, main_font, small_font)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(png_path)
    canvas.save(png_path.with_suffix(".pdf"), "PDF", resolution=180.0)


def plot_composite_w_cross_section_pillow(composite: dict[str, object], png_path: Path) -> None:
    x = np.asarray(composite["x_over_r"], dtype="f4")
    depth = np.asarray(composite["depth_m"], dtype="f4")
    fields = [
        ("term1", np.asarray(composite["section_term1_m_s"], dtype="f4")),
        ("term2", np.asarray(composite["section_term2_m_s"], dtype="f4")),
        ("rebuild W", np.asarray(composite["section_rebuild_w_m_s"], dtype="f4")),
    ]
    if "reference_section_ofes_w_native_m_s" in composite:
        fields.extend(
            [
                ("native W all", np.asarray(composite["reference_section_ofes_w_native_m_s"], dtype="f4")),
                (f"native W {composite['multipole_class']}", np.asarray(composite["section_ofes_w_native_m_s"], dtype="f4")),
            ]
        )
    else:
        fields.append(("OFES native W", np.asarray(composite["section_ofes_w_native_m_s"], dtype="f4")))
    finite_parts = [arr[np.isfinite(arr)] for _, arr in fields if np.isfinite(arr).any()]
    vals = np.concatenate(finite_parts) if finite_parts else np.array([], dtype="f4")
    lim = max(float(np.nanpercentile(np.abs(vals), 95)) * 1.0e6 if vals.size else 1.0, 1.0e-12)
    if len(fields) == 5:
        canvas = Image.new("RGB", (2500, 1500), "white")
        boxes = [(45, 125, 820, 710), (860, 125, 1635, 710), (1675, 125, 2450, 710), (450, 805, 1225, 1390), (1275, 805, 2050, 1390)]
    else:
        canvas = Image.new("RGB", (1900, 1450), "white")
        boxes = [(65, 125, 915, 715), (995, 125, 1845, 715), (65, 790, 915, 1380), (995, 790, 1845, 1380)]
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(30)
    main_font = load_font(20)
    small_font = load_font(15)
    class_text = str(composite.get("multipole_class", "all"))
    region_text = str(composite.get("region", "")).strip()
    region_suffix = f" | {region_text}" if region_text else ""
    draw.text(
        (45, 30),
        f"OFES W Cressman composite crossing section | {lat_label(float(composite['target_lat']))} | {composite['polarity']} | {class_text}{region_suffix}",
        fill=(20, 24, 32),
        font=title_font,
    )
    draw.text(
        (45, 70),
        f"objects={composite['object_count']}; Cressman R={composite['cressman_radius_r']}R; min objects={composite['cressman_min_objects']}; corr={composite['corr_rebuild_native_section']:.3f}; +/-{lim:.2g} x10^-6 m/s",
        fill=(80, 88, 100),
        font=small_font,
    )
    for box, (name, data) in zip(boxes, fields):
        draw_section_field_pillow(canvas, box, x, depth, data * 1.0e6, -lim, lim, name, main_font, small_font)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(png_path)
    canvas.save(png_path.with_suffix(".pdf"), "PDF", resolution=180.0)


def plot_composite_w_focus_pillow(composite: dict[str, object], png_path: Path) -> None:
    x = np.asarray(composite["x_over_r"], dtype="f4")
    y = np.asarray(composite["y_over_r"], dtype="f4")
    depth = np.asarray(composite["depth_m"], dtype="f4")
    field_defs = [
        ("term1", np.asarray(composite["composite_term1_m_s"], dtype="f4")),
        ("term2", np.asarray(composite["composite_term2_m_s"], dtype="f4")),
        ("rebuild W", np.asarray(composite["composite_rebuild_w_m_s"], dtype="f4")),
    ]
    if "reference_composite_ofes_w_native_m_s" in composite:
        field_defs.extend(
            [
                ("native W all", np.asarray(composite["reference_composite_ofes_w_native_m_s"], dtype="f4")),
                (f"native W {composite['multipole_class']}", np.asarray(composite["composite_ofes_w_native_m_s"], dtype="f4")),
            ]
        )
    else:
        field_defs.append(("OFES native W", np.asarray(composite["composite_ofes_w_native_m_s"], dtype="f4")))
    rows = [
        ("0-100 m surface", depth_average_indices(depth, 0.0, 100.0)),
        ("300-500 m mean", depth_average_indices(depth, 300.0, 500.0)),
        ("450 m slice", np.array([int(np.nanargmin(np.abs(depth - 450.0)))])),
    ]
    fields2d: list[tuple[str, str, np.ndarray]] = []
    for row_name, idx in rows:
        for field_name, arr in field_defs:
            with np.errstate(invalid="ignore"):
                fields2d.append((row_name, field_name, np.nanmean(arr[idx], axis=0)))
    vals = np.concatenate([arr[np.isfinite(arr)] for _, _, arr in fields2d if np.isfinite(arr).any()])
    lim = max(float(np.nanpercentile(np.abs(vals), 95)) * 1.0e6 if vals.size else 1.0, 1.0e-12)
    ncols = len(field_defs)
    canvas = Image.new("RGB", (610 * ncols + 80, 1770), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(30)
    main_font = load_font(16)
    small_font = load_font(12)
    region_text = str(composite.get("region", "")).strip()
    region_suffix = f" | {region_text}" if region_text else ""
    draw.text(
        (35, 25),
        f"OFES W focus slices | {lat_label(float(composite['target_lat']))} | {composite['polarity']} | {composite.get('multipole_class', 'all')}{region_suffix} | objects={composite['object_count']}",
        fill=(20, 24, 32),
        font=title_font,
    )
    for row_idx, (row_name, _) in enumerate(rows):
        draw.text((15, 105 + row_idx * 535), row_name, fill=(30, 36, 48), font=main_font)
        for col_idx, (field_name, arr3d) in enumerate(field_defs):
            idx = row_idx * ncols + col_idx
            _, _, data = fields2d[idx]
            box = (40 + col_idx * 610, 130 + row_idx * 535, 585 + col_idx * 610, 630 + row_idx * 535)
            draw_field_pillow(canvas, box, x, y, data * 1.0e6, -lim, lim, field_name, main_font, small_font)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(png_path)
    canvas.save(png_path.with_suffix(".pdf"), "PDF", resolution=180.0)


def plot_composite_w_slices_pillow(composite: dict[str, object], png_path: Path) -> None:
    x = np.asarray(composite["x_over_r"], dtype="f4")
    y = np.asarray(composite["y_over_r"], dtype="f4")
    depth = np.asarray(composite["depth_m"], dtype="f4")
    fields = [
        ("term1", np.asarray(composite["composite_term1_m_s"], dtype="f4")),
        ("term2", np.asarray(composite["composite_term2_m_s"], dtype="f4")),
        ("rebuild W", np.asarray(composite["composite_rebuild_w_m_s"], dtype="f4")),
        ("OFES native W", np.asarray(composite["composite_ofes_w_native_m_s"], dtype="f4")),
    ]
    desired_depths = [50.0, 200.0, 500.0, 1000.0]
    indices = [int(np.nanargmin(np.abs(depth - target))) for target in desired_depths if depth.size]
    finite_parts = [arr[np.isfinite(arr)] for _, arr in fields if np.isfinite(arr).any()]
    vals = np.concatenate(finite_parts) if finite_parts else np.array([], dtype="f4")
    lim = max(float(np.nanpercentile(np.abs(vals), 95)) * 1.0e6 if vals.size else 1.0, 1.0e-12)
    canvas = Image.new("RGB", (2420, 2300), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(30)
    main_font = load_font(17)
    small_font = load_font(13)
    draw.text(
        (40, 28),
        f"OFES W Cressman composite horizontal slices | {lat_label(float(composite['target_lat']))} | {composite['polarity']} | objects={composite['object_count']}",
        fill=(20, 24, 32),
        font=title_font,
    )
    box_w, box_h = 560, 510
    x0, y0 = 35, 95
    for row, k in enumerate(indices):
        draw.text((20, y0 + row * box_h + 15), f"z={depth[k]:.0f} m", fill=(30, 36, 48), font=main_font)
        for col, (name, arr) in enumerate(fields):
            box = (x0 + col * 590, y0 + row * box_h, x0 + col * 590 + box_w, y0 + row * box_h + box_h - 25)
            draw_field_pillow(canvas, box, x, y, arr[k] * 1.0e6, -lim, lim, name, main_font, small_font)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(png_path)
    canvas.save(png_path.with_suffix(".pdf"), "PDF", resolution=180.0)


def plot_composite_density_sections_pillow(composite: dict[str, object], png_path: Path) -> None:
    x = np.asarray(composite["x_over_r"], dtype="f4")
    depth = np.asarray(composite["depth_m"], dtype="f4")
    fields = [
        ("prho", np.asarray(composite["section_prho_for_rebuild"], dtype="f4"), "kg/m^3"),
        ("rho prime", np.asarray(composite["section_rho_prime_for_rebuild"], dtype="f4"), "kg/m^3"),
        ("z rho anomaly", np.asarray(composite["section_z_rho_anom_m"], dtype="f4"), "m"),
        ("support", np.asarray(composite["section_support_objects_rebuild_w_m_s"], dtype="f4"), "objects"),
    ]
    canvas = Image.new("RGB", (1900, 1450), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(30)
    main_font = load_font(20)
    small_font = load_font(15)
    draw.text(
        (45, 30),
        f"OFES density Cressman composite crossing section | {lat_label(float(composite['target_lat']))} | {composite['polarity']}",
        fill=(20, 24, 32),
        font=title_font,
    )
    draw.text((45, 70), f"objects={composite['object_count']}; support is object count after Cressman mapping", fill=(80, 88, 100), font=small_font)
    boxes = [(65, 125, 915, 715), (995, 125, 1845, 715), (65, 790, 915, 1380), (995, 790, 1845, 1380)]
    for box, (name, data, unit) in zip(boxes, fields):
        finite = data[np.isfinite(data)]
        if finite.size:
            if name in {"rho prime", "z rho anomaly"}:
                lim = max(float(np.nanpercentile(np.abs(finite), 95)), 1.0e-12)
                vmin, vmax = -lim, lim
            else:
                vmin, vmax = float(np.nanpercentile(finite, 2)), float(np.nanpercentile(finite, 98))
                if vmax <= vmin:
                    vmax = vmin + 1.0
        else:
            vmin, vmax = -1.0, 1.0
        draw_section_field_pillow(canvas, box, x, depth, data, vmin, vmax, f"{name} ({unit})", main_font, small_font, unit)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(png_path)
    canvas.save(png_path.with_suffix(".pdf"), "PDF", resolution=180.0)


def plot_composite_density_profiles_pillow(composite: dict[str, object], png_path: Path) -> None:
    depth = np.asarray(composite["depth_m"], dtype="f4")
    profiles = [
        ("rho_bg", np.asarray(composite["profile_rho_bg"], dtype="f4"), "kg/m^3"),
        ("center prho", np.asarray(composite["profile_prho_center_profile"], dtype="f4"), "kg/m^3"),
        ("center rho prime", np.asarray(composite["profile_rho_prime_center_profile"], dtype="f4"), "kg/m^3"),
        ("rho_z used", np.asarray(composite["profile_rho_z_used"], dtype="f4"), "kg/m^4"),
    ]
    canvas = Image.new("RGB", (1650, 1250), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(30)
    main_font = load_font(20)
    small_font = load_font(15)
    draw.text(
        (40, 28),
        f"OFES density profiles from W rebuild composite | {lat_label(float(composite['target_lat']))} | {composite['polarity']} | objects={composite['object_count']}",
        fill=(20, 24, 32),
        font=title_font,
    )
    boxes = [(70, 115, 780, 575), (860, 115, 1570, 575), (70, 690, 780, 1150), (860, 690, 1570, 1150)]
    for box, (name, values, unit) in zip(boxes, profiles):
        draw_profile_panel(canvas, box, values, depth, f"{name} ({unit})", main_font, small_font)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(png_path)
    canvas.save(png_path.with_suffix(".pdf"), "PDF", resolution=180.0)


def crossing_sections(grid: dict[str, object]) -> dict[str, np.ndarray | float]:
    return crossing_sections_for_keys(grid, ["term1_m_s", "term2_m_s", "rebuild_w_m_s", "ofes_w_native_m_s"])


def draw_section_field_pillow(canvas: Image.Image, box: tuple[int, int, int, int], x: np.ndarray, depth: np.ndarray, data: np.ndarray, vmin: float, vmax: float, title: str, main_font, small_font, unit_label: str = "1e-6 m/s") -> None:
    draw = ImageDraw.Draw(canvas)
    draw.rectangle(box, outline=(180, 186, 196), width=1)
    plot = (box[0] + 70, box[1] + 42, box[2] - 40, box[3] - 58)
    img = array_to_rgb(data, vmin, vmax).resize((plot[2] - plot[0], plot[3] - plot[1]), Image.Resampling.BILINEAR)
    canvas.paste(img, plot[:2])
    draw.rectangle(plot, outline=(100, 110, 125), width=1)
    zero_x = section_x_to_px(plot, 0.0, float(x[0]), float(x[-1]))
    draw.line((zero_x, plot[1], zero_x, plot[3]), fill=(30, 30, 30), width=2)
    for rr in [-4, -2, 0, 2, 4]:
        tx = section_x_to_px(plot, float(rr), float(x[0]), float(x[-1]))
        draw.line((tx, plot[3], tx, plot[3] + 5), fill=(80, 88, 100), width=1)
        draw.text((tx - 10, plot[3] + 9), f"{rr:g}", fill=(80, 88, 100), font=small_font)
    for dz in [0, 500, 1000, 1500, 2000, 3000, 4000, 5000]:
        if depth[0] <= dz <= depth[-1]:
            ty = section_depth_to_py(plot, float(dz), float(depth[0]), float(depth[-1]))
            draw.line((plot[0] - 5, ty, plot[0], ty), fill=(80, 88, 100), width=1)
            draw.text((box[0] + 10, ty - 8), f"{dz}", fill=(80, 88, 100), font=small_font)
    draw.text((box[0] + 12, box[1] + 10), title, fill=(30, 36, 48), font=main_font)
    draw.text((plot[0], box[3] - 30), "x/R", fill=(80, 88, 100), font=small_font)
    draw.text((box[0] + 8, plot[1]), "depth m", fill=(80, 88, 100), font=small_font)
    colorbar_box = (plot[2] + 10, plot[1], min(plot[2] + 30, box[2] - 8), plot[3])
    draw_colorbar_pillow(canvas, colorbar_box, vmin, vmax, small_font, unit_label)


def draw_profile_panel(canvas: Image.Image, box: tuple[int, int, int, int], values: np.ndarray, depth: np.ndarray, title: str, main_font, small_font) -> None:
    draw = ImageDraw.Draw(canvas)
    draw.rectangle(box, outline=(180, 186, 196), width=1)
    plot = (box[0] + 80, box[1] + 42, box[2] - 35, box[3] - 58)
    draw.rectangle(plot, outline=(100, 110, 125), width=1)
    finite = np.isfinite(values) & np.isfinite(depth)
    if np.any(finite):
        xmin = float(np.nanpercentile(values[finite], 2))
        xmax = float(np.nanpercentile(values[finite], 98))
        if abs(xmax - xmin) < 1.0e-12:
            xmin -= 1.0
            xmax += 1.0
        dmin = float(np.nanmin(depth[finite]))
        dmax = float(np.nanmax(depth[finite]))
        grid_color = (226, 230, 236)
        for frac in [0.25, 0.5, 0.75]:
            gx = plot[0] + int(round(frac * (plot[2] - plot[0])))
            gy = plot[1] + int(round(frac * (plot[3] - plot[1])))
            draw.line((gx, plot[1], gx, plot[3]), fill=grid_color, width=1)
            draw.line((plot[0], gy, plot[2], gy), fill=grid_color, width=1)
        points = []
        for value, dep in zip(values, depth):
            if not np.isfinite(value) or not np.isfinite(dep):
                if len(points) > 1:
                    draw.line(points, fill=(37, 79, 150), width=3)
                points = []
                continue
            px = plot[0] + int(round((float(value) - xmin) / max(xmax - xmin, 1.0e-12) * (plot[2] - plot[0] - 1)))
            py = plot[1] + int(round((float(dep) - dmin) / max(dmax - dmin, 1.0e-12) * (plot[3] - plot[1] - 1)))
            points.append((px, py))
        if len(points) > 1:
            draw.line(points, fill=(37, 79, 150), width=3)
        draw.text((plot[0], plot[3] + 10), f"{xmin:.3g}", fill=(80, 88, 100), font=small_font)
        draw.text((plot[2] - 55, plot[3] + 10), f"{xmax:.3g}", fill=(80, 88, 100), font=small_font)
        for dz in [0, 500, 1000, 1500, 2000, 3000, 4000, 5000]:
            if dmin <= dz <= dmax:
                ty = section_depth_to_py(plot, float(dz), dmin, dmax)
                draw.line((plot[0] - 5, ty, plot[0], ty), fill=(80, 88, 100), width=1)
                draw.text((box[0] + 10, ty - 8), f"{dz}", fill=(80, 88, 100), font=small_font)
    draw.text((box[0] + 12, box[1] + 10), title, fill=(30, 36, 48), font=main_font)
    draw.text((box[0] + 10, plot[1]), "depth m", fill=(80, 88, 100), font=small_font)


def section_x_to_px(plot: tuple[int, int, int, int], value: float, xmin: float, xmax: float) -> int:
    return plot[0] + int(round((value - xmin) / max(xmax - xmin, 1.0e-12) * (plot[2] - plot[0] - 1)))


def section_depth_to_py(plot: tuple[int, int, int, int], value: float, dmin: float, dmax: float) -> int:
    return plot[1] + int(round((value - dmin) / max(dmax - dmin, 1.0e-12) * (plot[3] - plot[1] - 1)))


def array_to_rgb(values: np.ndarray, vmin: float, vmax: float) -> Image.Image:
    arr = np.asarray(values, dtype="f4")
    nan = ~np.isfinite(arr)
    scaled = np.clip((arr - vmin) / max(vmax - vmin, 1.0e-12), 0.0, 1.0)
    colors = rdbu_r_colors(np.nan_to_num(scaled, nan=0.5))
    colors[nan] = np.array([255, 255, 255], dtype=np.uint8)
    return Image.fromarray(colors, mode="RGB")


def draw_colorbar_pillow(canvas: Image.Image, box: tuple[int, int, int, int], vmin: float, vmax: float, small_font, unit_label: str = "1e-6 m/s") -> None:
    draw = ImageDraw.Draw(canvas)
    if box[2] <= box[0] or box[3] <= box[1]:
        return
    values = np.linspace(vmax, vmin, max(2, box[3] - box[1]), dtype="f4")[:, None]
    bar = array_to_rgb(values, vmin, vmax).resize((box[2] - box[0], box[3] - box[1]), Image.Resampling.BILINEAR)
    canvas.paste(bar, box[:2])
    draw.rectangle(box, outline=(95, 103, 116), width=1)
    tick_x0 = box[2] + 2
    ticks = [(vmax, box[1], f"{vmax:.2g}"), (0.0, (box[1] + box[3]) // 2, "0"), (vmin, box[3], f"{vmin:.2g}")]
    for _, y, label in ticks:
        draw.line((box[2], y, tick_x0 + 4, y), fill=(75, 85, 99), width=1)
        draw.text((tick_x0 + 7, y - 8), label, fill=(75, 85, 99), font=small_font)
    draw.text((box[0] - 2, box[3] + 8), unit_label, fill=(75, 85, 99), font=small_font)


def rdbu_r_colors(scaled: np.ndarray) -> np.ndarray:
    blue = np.array([5, 113, 176], dtype="f4")
    white = np.array([247, 247, 247], dtype="f4")
    red = np.array([202, 0, 32], dtype="f4")
    rgb = np.empty((*scaled.shape, 3), dtype=np.uint8)
    low = scaled <= 0.5
    rgb[low] = (blue * (1.0 - scaled[low, None] / 0.5) + white * (scaled[low, None] / 0.5)).astype(np.uint8)
    high = ~low
    rgb[high] = (white * (1.0 - (scaled[high, None] - 0.5) / 0.5) + red * ((scaled[high, None] - 0.5) / 0.5)).astype(np.uint8)
    return rgb


def spatial_corr(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if int(mask.sum()) < 3:
        return float("nan")
    av = np.asarray(a[mask], dtype="f8")
    bv = np.asarray(b[mask], dtype="f8")
    av -= av.mean()
    bv -= bv.mean()
    denom = math.sqrt(float(np.sum(av * av) * np.sum(bv * bv)))
    return float(np.sum(av * bv) / denom) if denom > 0 else float("nan")


def dipole_score(w: np.ndarray, x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(w) & (np.hypot(x, y) <= 4.0)
    if int(mask.sum()) < 3:
        return float("nan")
    east = np.nanmean(w[mask & (x > 0)])
    west = np.nanmean(w[mask & (x < 0)])
    north = np.nanmean(w[mask & (y > 0)])
    south = np.nanmean(w[mask & (y < 0)])
    scale = np.nanmean(np.abs(w[mask]))
    if not np.isfinite(scale) or scale <= 0:
        return float("nan")
    return float(max(abs(east - west), abs(north - south)) / scale)


def q95_abs(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(np.nanpercentile(np.abs(finite), 95)) if finite.size else float("nan")


def depth_average_indices(depth: np.ndarray, dmin: float, dmax: float) -> np.ndarray:
    arr = np.asarray(depth, dtype="f8")
    mask = (arr >= min(dmin, dmax)) & (arr <= max(dmin, dmax))
    idx = np.flatnonzero(mask)
    if idx.size:
        return idx
    target = 0.5 * (float(dmin) + float(dmax))
    return np.array([int(np.nanargmin(np.abs(arr - target)))], dtype=int)


def meters_per_degree(lat_deg: float) -> tuple[float, float]:
    lat_rad = math.radians(lat_deg)
    return math.pi * EARTH_RADIUS_M * math.cos(lat_rad) / 180.0, math.pi * EARTH_RADIUS_M / 180.0


def lon_delta_deg(lon: float, lon0: float) -> float:
    return ((float(lon) - float(lon0) + 180.0) % 360.0) - 180.0


def date_range(start: date, end: date) -> list[date]:
    current = start
    out = []
    while current <= end:
        out.append(current)
        current += timedelta(days=1)
    return out


def parse_iso_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def rho_stabilization_suffix(args: argparse.Namespace) -> str:
    profile_suffix = "_layer_tracking" if str(getattr(args, "translation_profile", "")) == "layer_tracking" else ""
    w_suffix = "_wcenter"
    if str(getattr(args, "native_w_temporal_filter", "none")) != "none":
        w_suffix = f"{w_suffix}_wlowpass{int(getattr(args, 'native_w_filter_window_days', 10))}d"
    density_filter = str(getattr(args, "rebuild_density_filter", "none"))
    density_suffix = ""
    if density_filter != "none":
        eta_sigma = f"{float(getattr(args, 'eta_horizontal_lowpass_sigma_r', 0.5)):g}".replace(".", "p")
        window = int(getattr(args, "rebuild_density_filter_window_days", 10))
        if density_filter == "joint_lowpass":
            density_suffix = f"_rhojoint{window}d_etaR{eta_sigma}"
        elif density_filter == "bg_perturb_decomp":
            density_suffix = f"_rhodecomp{window}d_etaR{eta_sigma}"
    velocity_filter = str(getattr(args, "rebuild_velocity_filter", "none"))
    velocity_suffix = ""
    if velocity_filter != "none":
        uv_sigma = f"{float(getattr(args, 'uv_horizontal_lowpass_sigma_r', 0.5)):g}".replace(".", "p")
        smooth_sigma = f"{float(getattr(args, 'translation_profile_smooth_sigma_layers', 2.0)):g}".replace(".", "p")
        window = int(getattr(args, "rebuild_velocity_filter_window_days", 10))
        velocity_suffix = f"_uvjoint{window}d_uvR{uv_sigma}_csmooth{smooth_sigma}"
    if bool(getattr(args, "disable_rho_stabilization", False)):
        return f"{profile_suffix}{w_suffix}{density_suffix}{velocity_suffix}"
    if str(getattr(args, "rho_bg_source", "regional_box")) == "regional_box":
        return f"_regional_bg_stable_rhoz{profile_suffix}{w_suffix}{density_suffix}{velocity_suffix}"
    return f"_stable_rhoz{profile_suffix}{w_suffix}{density_suffix}{velocity_suffix}"


def parse_float_list(value: str) -> list[float]:
    out = []
    for item in value.split(","):
        item = item.strip()
        if item:
            out.append(float(item))
    if not out:
        raise ValueError("Expected at least one numeric latitude")
    return out


def parse_bbox(value: str) -> tuple[float, float, float, float]:
    parts = [float(item.strip()) for item in value.split(",") if item.strip()]
    if len(parts) != 4:
        raise ValueError("--composite-domain-bbox must be lon_min,lon_max,lat_min,lat_max")
    lon_min, lon_max, lat_min, lat_max = parts
    if lat_min == lat_max:
        raise ValueError("--composite-domain-bbox latitude bounds must differ")
    return float(lon_min), float(lon_max), float(lat_min), float(lat_max)


def format_bbox(bbox: tuple[float, float, float, float]) -> str:
    lon_min, lon_max, lat_min, lat_max = bbox
    return f"{lon_min:g},{lon_max:g},{lat_min:g},{lat_max:g}"


def parse_string_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def safe_token(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in str(value)).strip("_") or "unknown"


def lat_token(lat: float) -> str:
    hemi = "N" if lat >= 0 else "S"
    value = abs(float(lat))
    if abs(value - round(value)) < 1.0e-9:
        return f"{int(round(value)):02d}{hemi}"
    return f"{value:g}{hemi}".replace(".", "p")


def json_safe(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=True), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def lat_label(lat: float) -> str:
    if lat > 0:
        return f"{abs(lat):g}N"
    if lat < 0:
        return f"{abs(lat):g}S"
    return "EQ"


def write_method_doc(path: Path, args: argparse.Namespace, rows: list[dict[str, object]]) -> None:
    text = f"""# OFES 原生 W vs rebuild W 诊断

- 科学口径：`{SCIENCE_TAG}` 下的等密面异常法。
- 数据根目录：`{args.data_root}`
- 结果根目录：`{args.result_root}`
- 目标日期：`{args.date}`
- 重建公式：`term1 = c_rel(z) · grad(z'_rho)`，`term2 = -U_rel(x,y,z) · grad(z'_rho)`，`rebuild_W = term1 + term2`。
- `z'_rho = -rho' / d rho_bg/dz`；默认 `rho_bg(z)` 来自区域背景盒，排除涡旋核心后取每层中位数，并做垂向平滑、弱层化 masking 和 `eta_rho` 裁剪。
- OFES 原生 `w` 的 `.ctl` 单位为 `cm/s`，runner 读取后转为 `m/s`；`.ctl` 标注 `W at T cell bottom`，当前固定比较模式为 `layer_center`。
- 原生 `w` 会从 bottom 层深度插值到 `prho/u/v` 层中心深度后再计算相关和作图；图中单位为 `10^-6 m/s`，网格文件保存原始 `m/s`。
- native `w` 时间滤波：`{getattr(args, "native_w_temporal_filter", "none")}`，窗口 `{getattr(args, "native_w_filter_window_days", 10)}` 天；当前作为 native W 参照的高频抑制版本，不改变 rebuild W 本身。
- rebuild 密度尺度分离：`{getattr(args, "rebuild_density_filter", "none")}`，窗口 `{getattr(args, "rebuild_density_filter_window_days", 10)}` 天，`eta_rho` 水平低通 sigma=`{getattr(args, "eta_horizontal_lowpass_sigma_r", 0.5)}`R；`bg_perturb_decomp` 额外按层扣除 `{getattr(args, "density_bg_detrend_fit_ring", "1.5,4.0")}R` 环内拟合的大尺度密度斜面。
- rebuild 速度尺度分离：`{getattr(args, "rebuild_velocity_filter", "none")}`，窗口 `{getattr(args, "rebuild_velocity_filter_window_days", 10)}` 天，`u/v` 水平低通 sigma=`{getattr(args, "uv_horizontal_lowpass_sigma_r", 0.5)}`R；启用时 `c(z)` 会先剔除速度超过 `{getattr(args, "translation_profile_max_speed_m_s", 0.5)}` m/s 的层，再做 sigma=`{getattr(args, "translation_profile_smooth_sigma_layers", 2.0)}` 层的垂向平滑。
- 默认传播速度使用逐层 tracking：每个深度层用相邻日同深度 refined center 做 nearest-neighbor 匹配，缺测层再沿垂向插值填补；随后减去该层远场背景流，得到 `c_rel(z)`。
- 输出对象数：`{len(rows)}`
"""
    path.write_text(text, encoding="utf-8")


def load_font(size: int) -> ImageFont.ImageFont:
    for name in ["arial.ttf", "DejaVuSans.ttf"]:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def map_xy(box: tuple[int, int, int, int], x: float, y: float, xmin: float, xmax: float, ymin: float, ymax: float) -> tuple[int, int]:
    px = box[0] + int(round((x - xmin) / max(xmax - xmin, 1.0e-12) * (box[2] - box[0] - 1)))
    py = box[3] - int(round((y - ymin) / max(ymax - ymin, 1.0e-12) * (box[3] - box[1] - 1)))
    return px, py


if __name__ == "__main__":
    main()
