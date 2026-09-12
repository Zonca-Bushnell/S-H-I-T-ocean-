from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
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
    if stages & {"diagnose", "crossing", "band"}:
        run_w_diagnostics(data_root, result_root, output_root, grids_dir, figures_dir, args, stages)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare OFES native W against Dipole-style rebuilt W.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--stages", default="all", help="Comma-separated: extract,diagnose,crossing,band,all. all runs extract plus the default band diagnostic.")
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
    parser.add_argument("--native-w-temporal-filter", choices=["none", "lowpass_running_mean"], default="lowpass_running_mean", help="Optional temporal filter for OFES native W reference only.")
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
    parser.add_argument("--extract-workers", type=int, default=1)
    parser.add_argument("--backend", choices=["matplotlib", "pillow"], default="pillow")
    parser.add_argument("--crossing-lats", default="20,40", help="Comma-separated target latitudes for x-depth W section diagnostics.")
    parser.add_argument("--intersect-radius-r", type=float, default=1.0, help="Strict crossing radius threshold in R.")
    parser.add_argument("--strict-crossing-only", action="store_true", help="Skip a target latitude when no object crosses within intersect-radius-r.")
    parser.add_argument("--band-lat-min", type=float, default=30.0, help="Minimum latitude for match_all band diagnostics.")
    parser.add_argument("--band-lat-max", type=float, default=35.0, help="Maximum latitude for match_all band diagnostics.")
    parser.add_argument("--band-max-objects", type=int, default=8, help="Maximum match_all band objects to plot; use <=0 for all.")
    return parser


def parse_stage_set(value: str) -> set[str]:
    requested = {item.strip().lower() for item in value.split(",") if item.strip()}
    if not requested:
        raise ValueError("--stages must include at least one stage")
    valid = {"extract", "diagnose", "crossing", "band", "all"}
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


def _run_object_diagnostics(data_root: Path, result_root: Path, output_root: Path, grids_dir: Path, figures_dir: Path, args: argparse.Namespace) -> None:
    target_day = parse_iso_date(args.date)
    detection_dir = resolve_detection_table_dir(result_root)
    centers = pd.read_csv(detection_dir / "centers_hua_style.csv")
    structures = pd.read_csv(detection_dir / "structures_hua_style.csv")
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
    centers = pd.read_csv(detection_dir / "centers_hua_style.csv")
    structures = pd.read_csv(detection_dir / "structures_hua_style.csv")
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
    centers = pd.read_csv(detection_dir / "centers_hua_style.csv")
    structures = pd.read_csv(detection_dir / "structures_hua_style.csv")
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
        if (candidate / "centers_hua_style.csv").exists() and (candidate / "structures_hua_style.csv").exists():
            return candidate
    checked = "; ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "Could not find centers_hua_style.csv and structures_hua_style.csv. "
        f"Checked: {checked}"
    )


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
        if translation_profile in {"layerwise", "layer_tracking"}:
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
    return {
        **asdict(obj),
        "science_tag": SCIENCE_TAG,
        "depth_m": depth,
        "x_over_r": x_over_r,
        "y_over_r": y_over_r,
        "term1_m_s": term1,
        "term2_m_s": term2,
        "rebuild_w_m_s": rebuild_w,
        "ofes_w_native_m_s": native_w,
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
        **rebuild_density_temporal_info,
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
        try:
            path = require_daily_file(data_root, "w", day, expected)
        except FileNotFoundError:
            missing.append(day.isoformat())
            continue
        raw_w = open_dta_memmap(path, w_meta)
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

    window = max(1, int(getattr(args, "rebuild_density_filter_window_days", 10)))
    days = centered_filter_days(parse_iso_date(obj.date), window, parse_iso_date(str(getattr(args, "start", "1991-01-01"))), parse_iso_date(str(getattr(args, "end", "1991-01-19"))))
    data_root = Path(getattr(args, "data_root", DEFAULT_DATA_ROOT))
    expected = expected_dta_bytes(prho_meta)
    accum = np.zeros_like(current_prho, dtype="f8")
    counts = np.zeros_like(current_prho, dtype="f4")
    used: list[str] = []
    missing: list[str] = []
    for day in days:
        try:
            path = require_daily_file(data_root, "prho", day, expected)
        except FileNotFoundError:
            missing.append(day.isoformat())
            continue
        raw_prho = open_dta_memmap(path, prho_meta)
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
        try:
            path_u = require_daily_file(data_root, "u", day, expected_u)
            path_v = require_daily_file(data_root, "v", day, expected_v)
        except FileNotFoundError:
            missing.append(day.isoformat())
            continue
        raw_u = open_dta_memmap(path_u, u_meta)
        raw_v = open_dta_memmap(path_v, v_meta)
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
    window = max(1, int(getattr(args, "rebuild_density_filter_window_days", 10)))
    days = centered_filter_days(parse_iso_date(obj.date), window, parse_iso_date(str(getattr(args, "start", "1991-01-01"))), parse_iso_date(str(getattr(args, "end", "1991-01-19"))))
    data_root = Path(getattr(args, "data_root", DEFAULT_DATA_ROOT))
    expected = expected_dta_bytes(meta)
    profiles = []
    fractions = []
    for day in days:
        try:
            path = require_daily_file(data_root, "prho", day, expected)
        except FileNotFoundError:
            continue
        raw_prho = open_dta_memmap(path, meta)
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


def nan_gaussian_smooth_3d(values: np.ndarray, sigma_cells: float) -> np.ndarray:
    if sigma_cells <= 0:
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


def crossing_sections(grid: dict[str, object]) -> dict[str, np.ndarray | float]:
    x = np.asarray(grid["x_over_r"], dtype="f8")
    y = np.asarray(grid["y_over_r"], dtype="f8")
    depth = np.asarray(grid["depth_m"], dtype="f8")
    target_y = (float(grid["target_lat"]) - float(grid["center_lat"])) * meters_per_degree(float(grid["center_lat"]))[1] / (float(grid["radius_km"]) * 1000.0)
    if y.size < 2:
        y_index = 0.0
    else:
        y_index = (target_y - float(y[0])) / float(y[1] - y[0])
    z_index = np.arange(depth.size, dtype="f8")[:, None]
    x_index = np.arange(x.size, dtype="f8")[None, :]
    coords = np.vstack(
        [
            np.broadcast_to(z_index, (depth.size, x.size)).ravel(),
            np.full(depth.size * x.size, y_index, dtype="f8"),
            np.broadcast_to(x_index, (depth.size, x.size)).ravel(),
        ]
    )
    out: dict[str, np.ndarray | float] = {"x_over_r": x, "depth_m": depth, "target_y_over_r": float(target_y)}
    for key in ["term1_m_s", "term2_m_s", "rebuild_w_m_s", "ofes_w_native_m_s"]:
        arr = np.asarray(grid[key], dtype="f4")
        sampled = ndimage.map_coordinates(arr, coords, order=1, mode="nearest").reshape(depth.size, x.size)
        if target_y < float(y[0]) or target_y > float(y[-1]):
            sampled[:] = np.nan
        out[key] = sampled
    return out


def draw_section_field_pillow(canvas: Image.Image, box: tuple[int, int, int, int], x: np.ndarray, depth: np.ndarray, data: np.ndarray, vmin: float, vmax: float, title: str, main_font, small_font) -> None:
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
    draw_colorbar_pillow(canvas, colorbar_box, vmin, vmax, small_font)


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


def draw_colorbar_pillow(canvas: Image.Image, box: tuple[int, int, int, int], vmin: float, vmax: float, small_font) -> None:
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
    draw.text((box[0] - 2, box[3] + 8), "1e-6 m/s", fill=(75, 85, 99), font=small_font)


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
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
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
