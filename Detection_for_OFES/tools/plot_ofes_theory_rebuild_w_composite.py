from __future__ import annotations

import argparse
import math
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from ..ofes_io import ctl_path, expected_dta_bytes, open_dta_memmap, parse_ctl, require_daily_file
from ..run_ofes_rebuild_w import (
    SelectedObject,
    build_parser as build_w_parser,
    finalize_composite_accumulator,
    init_composite_accumulator,
    json_safe,
    load_detection_tables,
    lon_delta_deg,
    meters_per_degree,
    open_neighbor_center_tables,
    parse_iso_date,
    q95_abs,
    rebuild_object_w,
    resolve_detection_table_dir,
    safe_token,
    summarize_object,
    update_composite_accumulator,
    write_csv,
    write_json,
)
from ..w_rebuild_config import DEFAULT_DATA_ROOT


DEFAULT_RESULT_ROOT = Path(r"E:\DATA\01_Eddy_correspond\02_OFES\origin_streamline_cpu_jan01_jan19_life1")
DEFAULT_OUTPUT_ROOT = DEFAULT_RESULT_ROOT / "w_rebuild_diagnostics" / "theory_rebuild_w_coherent_alpha_aligned_by_polarity_hemisphere"
THEORY_KEYS = ["term1_m_s", "term2_m_s", "rebuild_w_m_s"]


def main() -> None:
    cli = parse_args()
    output_root = cli.output_root
    figures_dir = output_root / "figures"
    grids_dir = output_root / "grids"
    figures_dir.mkdir(parents=True, exist_ok=True)
    grids_dir.mkdir(parents=True, exist_ok=True)

    runtime_args = build_runtime_args(cli)
    detection_dir = resolve_detection_table_dir(cli.result_root)
    centers, structures = load_detection_tables(detection_dir)
    objects, alpha_info = load_coherent_selected_objects(cli.result_root, structures)
    if cli.max_objects_per_group > 0:
        objects = limit_objects_per_group(objects, cli.max_objects_per_group)

    metas = {name: parse_ctl(ctl_path(cli.data_root, name)) for name in ["u", "v", "w", "prho"]}
    expected = {name: expected_dta_bytes(meta) for name, meta in metas.items()}
    raw_cache: dict[str, dict[str, np.memmap]] = {}
    summary_rows: list[dict[str, object]] = []

    for group_key in ["NH_cyclonic", "NH_anticyclonic", "SH_cyclonic", "SH_anticyclonic"]:
        group_objects = objects.get(group_key, [])
        if not group_objects:
            summary_rows.append(empty_summary_row(group_key, "no coherent object-days"))
            continue
        composite = rebuild_group_composite(
            group_key,
            group_objects,
            cli.data_root,
            metas,
            expected,
            centers,
            alpha_info,
            runtime_args,
            raw_cache,
        )
        if composite is None:
            summary_rows.append(empty_summary_row(group_key, "all object rebuilds failed"))
            continue
        hemisphere, polarity = group_key.split("_", 1)
        composite["hemisphere"] = hemisphere
        composite["polarity"] = polarity
        composite["shape_class_filter"] = "coherent"
        composite["composite_method"] = "alpha_aligned_center_section_cressman_object_support"
        composite["theory_terms"] = "term1,term2,term1_plus_term2"
        composite["alpha_source"] = "global_ls_centerline"
        composite["alpha_reference"] = "Zhe/composite_3d_lifecycle.py"

        token = f"{hemisphere}_{polarity}"
        npz_path = grids_dir / f"theory_rebuild_w_composite_alpha_{token}.npz"
        json_path = grids_dir / f"theory_rebuild_w_composite_alpha_{token}.json"
        arrays = {key: value for key, value in composite.items() if isinstance(value, np.ndarray)}
        np.savez_compressed(npz_path, **arrays)
        write_json(json_path, {key: json_safe(value) for key, value in composite.items() if not isinstance(value, np.ndarray)})

        section_png = figures_dir / f"theory_rebuild_w_section_alpha_{token}.png"
        slices_png = figures_dir / f"theory_rebuild_w_slices_alpha_{token}.png"
        plot_theory_section(composite, section_png)
        plot_theory_slices(composite, slices_png)

        summary_rows.append(summary_row(group_key, composite, npz_path, json_path, section_png, slices_png, group_objects))

    write_csv(output_root / "theory_rebuild_w_alpha_group_summary.csv", summary_rows)
    write_json(output_root / "theory_rebuild_w_alpha_group_summary.json", summary_rows)
    cache_size = len(getattr(runtime_args, "_daily_memmap_cache", {}))
    print(f"[theory-rebuild-composite] cached daily memmaps: {cache_size}", flush=True)
    print(f"[theory-rebuild-composite] wrote {output_root}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Composite OFES theory rebuild-W terms for coherent velocity-streamline object-days."
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--start", default="1991-01-01")
    parser.add_argument("--end", default="1991-01-19")
    parser.add_argument("--max-depth-layers", type=int, default=105)
    parser.add_argument("--grid-n", type=int, default=81)
    parser.add_argument("--extent-r", type=float, default=4.0)
    parser.add_argument("--cressman-radius-r", type=float, default=1.0)
    parser.add_argument("--cressman-min-objects", type=int, default=8)
    parser.add_argument("--max-objects-per-group", type=int, default=0, help="Use >0 for smoke runs.")
    parser.add_argument("--workers", type=int, default=1, help="Parallel object rebuild workers. Use 4-8 for full OFES runs.")
    return parser.parse_args()


def build_runtime_args(cli: argparse.Namespace) -> argparse.Namespace:
    parser = build_w_parser()
    args = parser.parse_args([])
    args.data_root = cli.data_root
    args.result_root = cli.result_root
    args.output_root = cli.output_root
    args.start = cli.start
    args.end = cli.end
    args.max_depth_layers = cli.max_depth_layers
    args.grid_n = cli.grid_n
    args.extent_r = cli.extent_r
    args.cressman_radius_r = cli.cressman_radius_r
    args.cressman_min_objects = cli.cressman_min_objects
    args.rho_bg_source = "regional_box"
    args.rho_z_min = 2.0e-5
    args.eta_rho_cap_m = 500.0
    args.translation_profile = "layer_tracking"
    args.rebuild_density_filter = "joint_lowpass"
    args.rebuild_velocity_filter = "joint_lowpass"
    args.velocity_reference = "farfield_relative"
    args.rebuild_formula = "relative_advection"
    args.native_w_meso_filter = "none"
    args.precompute_object_temporal_blocks = True
    args.backend = "pillow"
    args.workers = int(cli.workers)
    args._daily_memmap_cache = {}
    args._daily_memmap_cache_lock = threading.RLock()
    args._thread_local = threading.local()
    return args


def load_coherent_selected_objects(result_root: Path, structures: pd.DataFrame) -> tuple[dict[str, list[SelectedObject]], dict[str, dict[str, float]]]:
    shape_path = result_root / "shape_classification_1991_1991_hua_b3_start2_life1" / "shape_tracks.parquet"
    vertical_path = result_root / "catalog" / "vertical_objects.parquet"
    layer_path = result_root / "catalog" / "layer_observations.parquet"
    missing = [path for path in [shape_path, vertical_path, layer_path] if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing coherent catalog input: " + "; ".join(str(path) for path in missing))

    shape_tracks = pd.read_parquet(shape_path)
    coherent_track_ids = set(
        shape_tracks.loc[shape_tracks["shape_class"].astype(str).eq("coherent"), "track3d_id"].astype(int).to_list()
    )
    vertical = pd.read_parquet(vertical_path)
    vertical = vertical[vertical["track3d_id"].astype(int).isin(coherent_track_ids)].copy()

    surface_layers = pd.read_parquet(
        layer_path,
        columns=["layer_detection_id", "depth_index", "eddy3d_object_id"],
    )
    surface_layers = surface_layers[surface_layers["depth_index"].astype(int).eq(0)].copy()
    structures_by_detection_id = structures.reset_index(drop=True).copy()
    structures_by_detection_id["layer_detection_id"] = np.arange(len(structures_by_detection_id), dtype="i8")
    surface_map = surface_layers.merge(
        structures_by_detection_id[["layer_detection_id", "hua_object_id"]],
        on="layer_detection_id",
        how="left",
    )
    vertical = vertical.merge(
        surface_map[["eddy3d_object_id", "hua_object_id"]],
        on="eddy3d_object_id",
        how="left",
    )
    vertical = vertical.dropna(subset=["hua_object_id"]).copy()

    structures_by_id = {str(hid): part.copy() for hid, part in structures.groupby("hua_object_id", sort=False)}
    groups: dict[str, list[SelectedObject]] = {
        "NH_cyclonic": [],
        "NH_anticyclonic": [],
        "SH_cyclonic": [],
        "SH_anticyclonic": [],
    }
    alpha_info: dict[str, dict[str, float]] = {}
    seen: set[str] = set()
    for _, row in vertical.sort_values(["date", "eddy3d_object_id"]).iterrows():
        hua_object_id = str(row["hua_object_id"])
        if hua_object_id in seen:
            continue
        seen.add(hua_object_id)
        part = structures_by_id.get(hua_object_id)
        if part is None or part.empty:
            continue
        obj = summarize_object(part)
        alpha_info[obj.hua_object_id] = alpha_from_structure(part, obj)
        hemisphere = "NH" if obj.center_lat >= 0.0 else "SH"
        polarity = "cyclonic" if obj.polarity == "cyclonic" else "anticyclonic"
        groups[f"{hemisphere}_{polarity}"].append(obj)

    for values in groups.values():
        values.sort(key=lambda obj: (obj.date, obj.hua_object_id))
    return groups, alpha_info


def alpha_from_structure(part: pd.DataFrame, obj: SelectedObject) -> dict[str, float]:
    rows = part.sort_values("depth_index").copy()
    lon_col = "center_lon_refined" if "center_lon_refined" in rows.columns else "center_lon"
    lat_col = "center_lat_refined" if "center_lat_refined" in rows.columns else "center_lat"
    lon = rows[lon_col].to_numpy(dtype="f8")
    lat = rows[lat_col].to_numpy(dtype="f8")
    depth = rows["depth_m"].to_numpy(dtype="f8") if "depth_m" in rows.columns else rows["depth_index"].to_numpy(dtype="f8")
    valid = np.isfinite(lon) & np.isfinite(lat)
    if valid.sum() < 2:
        return alpha_payload(0.0, 0.0, 0.0, 0.0, "insufficient_layers")
    lon = lon[valid]
    lat = lat[valid]
    depth = depth[valid]
    mx, my = meters_per_degree(float(lat[0]))
    radius_m = max(float(obj.radius_km) * 1000.0, 1.0)
    x = np.asarray([lon_delta_deg(float(value), float(lon[0])) * mx / radius_m for value in lon], dtype="f8")
    y = (lat - lat[0]) * my / radius_m
    dx = float(x[-1] - x[0])
    dy = float(y[-1] - y[0])
    tilt = float(math.hypot(dx, dy))
    if tilt < 0.02:
        return alpha_payload(0.0, tilt, dx, dy, "tilt_below_0p02R")
    alpha = float(-math.degrees(math.atan2(dy, dx)))
    x_rot, y_rot = rotate_xy(np.asarray([dx]), np.asarray([dy]), alpha)
    return {
        "alpha_deg": alpha,
        "tilt_distance_r": tilt,
        "tilt_dx_r": dx,
        "tilt_dy_r": dy,
        "tilt_dx_rot_r": float(x_rot[0]),
        "tilt_dy_rot_r": float(y_rot[0]),
        "surface_depth_m": float(depth[0]),
        "deep_depth_m": float(depth[-1]),
        "alpha_status": "ok",
    }


def alpha_payload(alpha: float, tilt: float, dx: float, dy: float, status: str) -> dict[str, float]:
    return {
        "alpha_deg": float(alpha),
        "tilt_distance_r": float(tilt),
        "tilt_dx_r": float(dx),
        "tilt_dy_r": float(dy),
        "tilt_dx_rot_r": float(tilt),
        "tilt_dy_rot_r": 0.0,
        "surface_depth_m": np.nan,
        "deep_depth_m": np.nan,
        "alpha_status": status,
    }


def limit_objects_per_group(groups: dict[str, list[SelectedObject]], limit: int) -> dict[str, list[SelectedObject]]:
    out: dict[str, list[SelectedObject]] = {}
    for key, values in groups.items():
        picked = sorted(values, key=lambda obj: (obj.pass_layers, obj.radius_km, obj.date, obj.hua_object_id), reverse=True)[:limit]
        picked.sort(key=lambda obj: (obj.date, obj.hua_object_id))
        out[key] = picked
    return out


def rebuild_group_composite(
    group_key: str,
    objects: list[SelectedObject],
    data_root: Path,
    metas: dict[str, object],
    expected: dict[str, int],
    centers: pd.DataFrame,
    alpha_info: dict[str, dict[str, float]],
    args: argparse.Namespace,
    raw_cache: dict[str, dict[str, np.memmap]],
) -> dict[str, object] | None:
    accumulator: dict[str, object] | None = None
    failed = 0
    alpha_rows: list[dict[str, float]] = []
    workers = max(1, int(getattr(args, "workers", 1)))
    raw_cache_lock = getattr(args, "_daily_memmap_cache_lock", threading.RLock())

    def build_one(item: tuple[int, SelectedObject]) -> tuple[int, SelectedObject, dict[str, object], dict[str, float]]:
        idx, obj = item
        target_day = parse_iso_date(obj.date)
        day_key = target_day.isoformat()
        if workers <= 1:
            if day_key not in raw_cache:
                paths = {name: require_daily_file(data_root, name, target_day, expected[name]) for name in ["u", "v", "w", "prho"]}
                raw_cache[day_key] = {name: open_dta_memmap(paths[name], metas[name]) for name in ["u", "v", "w", "prho"]}
            raw = raw_cache[day_key]
        else:
            with raw_cache_lock:
                if day_key not in raw_cache:
                    paths = {name: require_daily_file(data_root, name, target_day, expected[name]) for name in ["u", "v", "w", "prho"]}
                    raw_cache[day_key] = {name: open_dta_memmap(paths[name], metas[name]) for name in ["u", "v", "w", "prho"]}
                raw = raw_cache[day_key]
        neighbors = open_neighbor_center_tables(centers, target_day)
        grid = rebuild_object_w(raw, metas, centers, obj, neighbors, args)
        if "ofes_w_native_meso_aligned_m_s" not in grid:
            native_source = "ofes_w_native_meso_m_s" if "ofes_w_native_meso_m_s" in grid else "ofes_w_native_m_s"
            grid["ofes_w_native_meso_aligned_m_s"] = np.asarray(grid[native_source], dtype="f4")
        info = alpha_info.get(obj.hua_object_id, alpha_payload(0.0, 0.0, 0.0, 0.0, "missing_alpha"))
        apply_alpha_alignment(grid, float(info["alpha_deg"]))
        grid.update({f"alpha_{key}": value for key, value in info.items()})
        grid["target_lat"] = float(grid["center_lat"])
        grid["crossing_selection_mode"] = "center_section_theory"
        return idx, obj, grid, info

    def consume(idx: int, obj: SelectedObject, grid: dict[str, object], info: dict[str, float]) -> None:
        nonlocal accumulator
        alpha_rows.append(info)
        if accumulator is None:
            accumulator = init_composite_accumulator(grid, float(grid["center_lat"]), obj.polarity, args, "coherent")
        update_composite_accumulator(accumulator, grid, args)
        print(f"[{group_key}] {idx}/{len(objects)} {obj.hua_object_id} alpha={info['alpha_deg']:.1f} ok", flush=True)

    items = list(enumerate(objects, start=1))
    if workers > 1:
        print(f"[{group_key}] rebuilding {len(objects)} coherent object-days with workers={workers}", flush=True)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(build_one, item): item for item in items}
            for future in as_completed(futures):
                idx, obj = futures[future]
                try:
                    _, _, grid, info = future.result()
                    consume(idx, obj, grid, info)
                except Exception as exc:
                    failed += 1
                    print(f"[{group_key}] {idx}/{len(objects)} {obj.hua_object_id} failed: {exc}", flush=True)
    else:
        for item in items:
            idx, obj = item
            try:
                _, _, grid, info = build_one(item)
                consume(idx, obj, grid, info)
            except Exception as exc:
                failed += 1
                print(f"[{group_key}] {idx}/{len(objects)} {obj.hua_object_id} failed: {exc}", flush=True)
    if accumulator is None:
        return None
    composite = finalize_composite_accumulator(accumulator, args)
    composite["failed_object_count"] = int(failed)
    attach_alpha_summary(composite, alpha_rows)
    return composite


def apply_alpha_alignment(grid: dict[str, object], alpha_deg: float) -> None:
    x = np.asarray(grid["x_over_r"], dtype="f8")
    y = np.asarray(grid["y_over_r"], dtype="f8")
    xx, yy = np.meshgrid(x, y)
    x_orig, y_orig = rotate_xy(xx, yy, -float(alpha_deg))
    for key in THEORY_KEYS:
        grid[key] = rotate_scalar_stack(np.asarray(grid[key], dtype="f4"), x, y, x_orig, y_orig)


def rotate_xy(x: np.ndarray, y: np.ndarray, alpha_deg: float) -> tuple[np.ndarray, np.ndarray]:
    ca = np.cos(np.radians(alpha_deg))
    sa = np.sin(np.radians(alpha_deg))
    return x * ca - y * sa, x * sa + y * ca


def rotate_scalar_stack(stack: np.ndarray, x: np.ndarray, y: np.ndarray, x_query: np.ndarray, y_query: np.ndarray) -> np.ndarray:
    dx = float(x[1] - x[0])
    dy = float(y[1] - y[0])
    xi = (x_query - float(x[0])) / dx
    yi = (y_query - float(y[0])) / dy
    out = np.full_like(stack, np.nan, dtype="f4")
    for k in range(stack.shape[0]):
        out[k] = bilinear_nan_sample(np.asarray(stack[k], dtype="f4"), xi, yi)
    return out


def bilinear_nan_sample(field: np.ndarray, xi: np.ndarray, yi: np.ndarray) -> np.ndarray:
    nx = field.shape[1]
    ny = field.shape[0]
    x0 = np.floor(xi).astype("i4")
    y0 = np.floor(yi).astype("i4")
    fx = xi - x0
    fy = yi - y0
    inside = (x0 >= 0) & (y0 >= 0) & (x0 < nx - 1) & (y0 < ny - 1)
    x0c = np.clip(x0, 0, nx - 2)
    y0c = np.clip(y0, 0, ny - 2)
    x1c = x0c + 1
    y1c = y0c + 1
    v00 = field[y0c, x0c]
    v10 = field[y0c, x1c]
    v01 = field[y1c, x0c]
    v11 = field[y1c, x1c]
    w00 = (1.0 - fx) * (1.0 - fy)
    w10 = fx * (1.0 - fy)
    w01 = (1.0 - fx) * fy
    w11 = fx * fy
    vals = [(v00, w00), (v10, w10), (v01, w01), (v11, w11)]
    num = np.zeros_like(xi, dtype="f8")
    den = np.zeros_like(xi, dtype="f8")
    for value, weight in vals:
        valid = np.isfinite(value) & inside
        num[valid] += value[valid] * weight[valid]
        den[valid] += weight[valid]
    return np.divide(num, den, out=np.full_like(num, np.nan, dtype="f8"), where=den > 1.0e-6).astype("f4")


def attach_alpha_summary(composite: dict[str, object], alpha_rows: list[dict[str, float]]) -> None:
    if not alpha_rows:
        return
    alpha = np.asarray([row["alpha_deg"] for row in alpha_rows], dtype="f8")
    tilt = np.asarray([row["tilt_distance_r"] for row in alpha_rows], dtype="f8")
    yrot = np.asarray([abs(row["tilt_dy_rot_r"]) for row in alpha_rows], dtype="f8")
    composite["alpha_deg_mean"] = float(np.nanmean(alpha))
    composite["alpha_deg_median"] = float(np.nanmedian(alpha))
    composite["tilt_distance_r_mean"] = float(np.nanmean(tilt))
    composite["tilt_distance_r_median"] = float(np.nanmedian(tilt))
    composite["post_rotation_abs_y_tilt_r_mean"] = float(np.nanmean(yrot))
    composite["alpha_source"] = "global_ls_centerline"
    composite["alpha_reference"] = "Zhe/composite_3d_lifecycle.py"


def summary_row(
    group_key: str,
    composite: dict[str, object],
    npz_path: Path,
    json_path: Path,
    section_png: Path,
    slices_png: Path,
    source_objects: list[SelectedObject],
) -> dict[str, object]:
    hemisphere, polarity = group_key.split("_", 1)
    layer_counts = np.asarray([obj.pass_layers for obj in source_objects], dtype="f8")
    radii = np.asarray([obj.radius_km for obj in source_objects], dtype="f8")
    return {
        "group": group_key,
        "hemisphere": hemisphere,
        "polarity": polarity,
        "shape_class": "coherent",
        "source_object_day_count": len(source_objects),
        "composited_object_count": int(composite["object_count"]),
        "failed_object_count": int(composite.get("failed_object_count", 0)),
        "valid_grid_fraction": float(composite.get("valid_grid_fraction", np.nan)),
        "valid_section_fraction": float(composite.get("valid_section_fraction", np.nan)),
        "mean_alpha_deg": float(composite.get("alpha_deg_mean", np.nan)),
        "median_alpha_deg": float(composite.get("alpha_deg_median", np.nan)),
        "mean_tilt_distance_r": float(composite.get("tilt_distance_r_mean", np.nan)),
        "mean_post_rotation_abs_y_tilt_r": float(composite.get("post_rotation_abs_y_tilt_r_mean", np.nan)),
        "q95_abs_term1_1e6_m_s": q95_abs(np.asarray(composite["composite_term1_m_s"])) * 1.0e6,
        "q95_abs_term2_1e6_m_s": q95_abs(np.asarray(composite["composite_term2_m_s"])) * 1.0e6,
        "q95_abs_rebuild_1e6_m_s": q95_abs(np.asarray(composite["composite_rebuild_w_m_s"])) * 1.0e6,
        "q95_abs_section_rebuild_1e6_m_s": q95_abs(np.asarray(composite["section_rebuild_w_m_s"])) * 1.0e6,
        "eta_rho_q95_abs_m": float(composite.get("eta_rho_q95_abs_m", np.nan)),
        "mean_radius_km": float(np.nanmean(radii)) if radii.size else np.nan,
        "mean_layer_count": float(np.nanmean(layer_counts)) if layer_counts.size else np.nan,
        "npz_path": str(npz_path),
        "json_path": str(json_path),
        "section_png": str(section_png),
        "slices_png": str(slices_png),
    }


def empty_summary_row(group_key: str, reason: str) -> dict[str, object]:
    hemisphere, polarity = group_key.split("_", 1)
    return {
        "group": group_key,
        "hemisphere": hemisphere,
        "polarity": polarity,
        "shape_class": "coherent",
        "source_object_day_count": 0,
        "composited_object_count": 0,
        "failed_object_count": 0,
        "skip_reason": reason,
    }


def plot_theory_section(composite: dict[str, object], png_path: Path) -> None:
    depth = np.asarray(composite["depth_m"], dtype="f8")
    x = np.asarray(composite["x_over_r"], dtype="f8")
    fields = [
        ("term1", np.asarray(composite["section_term1_m_s"], dtype="f8") * 1.0e6),
        ("term2", np.asarray(composite["section_term2_m_s"], dtype="f8") * 1.0e6),
        ("term1 + term2", np.asarray(composite["section_rebuild_w_m_s"], dtype="f8") * 1.0e6),
    ]
    vmax = max(q95_abs(field) for _, field in fields)
    vmax = max(vmax, 1.0e-6)
    width, height = 1900, 720
    margin_l, margin_t, panel_w, panel_h = 95, 110, 500, 480
    gap = 55
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    title = f"Alpha-aligned theory rebuild-W center sections | {composite['hemisphere']} {composite['polarity']} coherent | n={composite['object_count']}"
    draw.text((margin_l, 30), title, fill=(20, 30, 45), font=font)
    draw.text((margin_l, 55), "Units: 10^-6 m/s. Section is y_rot/R=0 after global-ls-alpha centerline alignment.", fill=(70, 80, 95), font=font)
    for col, (label, field) in enumerate(fields):
        x0 = margin_l + col * (panel_w + gap)
        y0 = margin_t
        draw_field_panel(image, draw, field, x, depth, (x0, y0, panel_w, panel_h), -vmax, vmax, "x/R", "depth (m)", label)
    draw_colorbar(image, draw, width - 95, margin_t, 24, panel_h, -vmax, vmax, "10^-6 m/s")
    png_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(png_path)


def plot_theory_slices(composite: dict[str, object], png_path: Path) -> None:
    depth = np.asarray(composite["depth_m"], dtype="f8")
    x = np.asarray(composite["x_over_r"], dtype="f8")
    y = np.asarray(composite["y_over_r"], dtype="f8")
    target_depths = [100.0, 300.0, 500.0, 800.0]
    depth_indices = [int(np.nanargmin(np.abs(depth - z))) for z in target_depths]
    keys = [("term1", "composite_term1_m_s"), ("term2", "composite_term2_m_s"), ("term1 + term2", "composite_rebuild_w_m_s")]
    arrays = [np.asarray(composite[key], dtype="f8") * 1.0e6 for _, key in keys]
    vmax = max(q95_abs(arr[idx]) for arr in arrays for idx in depth_indices)
    vmax = max(vmax, 1.0e-6)
    panel = 250
    margin_l, margin_t = 90, 105
    gap_x, gap_y = 38, 32
    width = margin_l + len(keys) * panel + (len(keys) - 1) * gap_x + 150
    height = margin_t + len(depth_indices) * panel + (len(depth_indices) - 1) * gap_y + 70
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    title = f"Alpha-aligned theory rebuild-W horizontal slices | {composite['hemisphere']} {composite['polarity']} coherent | n={composite['object_count']}"
    draw.text((margin_l, 28), title, fill=(20, 30, 45), font=font)
    draw.text((margin_l, 52), "Object-centered x_rot/R,y_rot/R Cressman composite; white = insufficient support/NaN.", fill=(70, 80, 95), font=font)
    for row, idx in enumerate(depth_indices):
        for col, (label, key) in enumerate(keys):
            x0 = margin_l + col * (panel + gap_x)
            y0 = margin_t + row * (panel + gap_y)
            draw_field_panel(
                image,
                draw,
                np.asarray(composite[key], dtype="f8")[idx] * 1.0e6,
                x,
                y,
                (x0, y0, panel, panel),
                -vmax,
                vmax,
                "x/R",
                "y/R",
                f"{label}, z={depth[idx]:.0f} m",
                circle=True,
            )
    draw_colorbar(image, draw, width - 85, margin_t, 22, len(depth_indices) * panel + (len(depth_indices) - 1) * gap_y, -vmax, vmax, "10^-6 m/s")
    png_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(png_path)


def draw_field_panel(
    image: Image.Image,
    draw: ImageDraw.ImageDraw,
    field: np.ndarray,
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    box: tuple[int, int, int, int],
    vmin: float,
    vmax: float,
    xlabel: str,
    ylabel: str,
    title: str,
    circle: bool = False,
) -> None:
    x0, y0, w, h = box
    rgb = field_to_rgb(field, vmin, vmax)
    panel = Image.fromarray(rgb).resize((w, h), Image.Resampling.BILINEAR)
    image.paste(panel, (x0, y0))
    draw.rectangle([x0, y0, x0 + w, y0 + h], outline=(80, 85, 92), width=1)
    font = ImageFont.load_default()
    draw.text((x0, y0 - 18), title, fill=(20, 30, 45), font=font)
    draw.text((x0 + w // 2 - 16, y0 + h + 8), xlabel, fill=(40, 45, 55), font=font)
    draw.text((x0 - 58, y0 + h // 2), ylabel, fill=(40, 45, 55), font=font)
    for frac in [0.25, 0.5, 0.75]:
        xx = x0 + int(frac * w)
        yy = y0 + int(frac * h)
        draw.line([xx, y0, xx, y0 + h], fill=(0, 0, 0), width=1)
        draw.line([x0, yy, x0 + w, yy], fill=(0, 0, 0), width=1)
    if circle:
        cx = x0 + w // 2
        cy = y0 + h // 2
        r1 = int(w / (x_axis[-1] - x_axis[0]) * 1.0)
        draw.ellipse([cx - r1, cy - r1, cx + r1, cy + r1], outline=(10, 10, 10), width=2)
        draw.ellipse([cx - 2, cy - 2, cx + 2, cy + 2], fill=(10, 10, 10))


def field_to_rgb(values: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    arr = np.asarray(values, dtype="f8")
    norm = (arr - vmin) / (vmax - vmin) if vmax > vmin else np.full_like(arr, 0.5)
    norm = np.clip(norm, 0.0, 1.0)
    valid = np.isfinite(arr)
    rgb = np.full(arr.shape + (3,), 255, dtype="u1")
    cold = np.array([37, 94, 159], dtype="f8")
    mid = np.array([247, 247, 247], dtype="f8")
    warm = np.array([178, 24, 43], dtype="f8")
    lower = norm <= 0.5
    upper = ~lower
    t = np.zeros_like(norm)
    t[lower] = norm[lower] / 0.5
    t[upper] = (norm[upper] - 0.5) / 0.5
    colors = np.empty_like(rgb, dtype="f8")
    colors[lower] = cold + (mid - cold) * t[lower][..., None]
    colors[upper] = mid + (warm - mid) * t[upper][..., None]
    rgb[valid] = np.clip(colors[valid], 0, 255).astype("u1")
    return rgb


def draw_colorbar(
    image: Image.Image,
    draw: ImageDraw.ImageDraw,
    x0: int,
    y0: int,
    w: int,
    h: int,
    vmin: float,
    vmax: float,
    label: str,
) -> None:
    grad = np.linspace(vmax, vmin, h, dtype="f8")[:, None]
    rgb = field_to_rgb(grad, vmin, vmax)
    bar = Image.fromarray(rgb).resize((w, h), Image.Resampling.NEAREST)
    image.paste(bar, (x0, y0))
    draw.rectangle([x0, y0, x0 + w, y0 + h], outline=(80, 85, 92), width=1)
    font = ImageFont.load_default()
    for frac, value in [(0.0, vmax), (0.5, 0.0), (1.0, vmin)]:
        yy = y0 + int(frac * h)
        draw.line([x0 + w, yy, x0 + w + 5, yy], fill=(30, 35, 45), width=1)
        draw.text((x0 + w + 8, yy - 6), f"{value:.2g}", fill=(30, 35, 45), font=font)
    draw.text((x0 - 10, y0 + h + 8), label, fill=(30, 35, 45), font=font)


if __name__ == "__main__":
    main()
