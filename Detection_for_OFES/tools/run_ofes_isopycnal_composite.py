"""Geometry-first OFES Jan-1 composites.

This entry point deliberately does not call ``rebuild_object_w``.  It first
composites native OFES potential density and native vertical velocity, then
obtains isopycnal depth by direct bracket interpolation in the *composite*
density columns.  It therefore cannot accidentally use the legacy
``-rho_prime / d(rho_bg)/dD`` eta approximation.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import savemat

from ..ofes_io import ctl_path, expected_dta_bytes, open_dta_memmap, parse_ctl, require_daily_file
from ..run_ofes_rebuild_w import (
    SelectedObject,
    align_native_w_vertical,
    classify_native_w_multipole,
    cressman_kernel_2d,
    cressman_map_3d,
    finalize_sum_count,
    local_lon_lat_grid,
    normalize_density_units,
    plot_composite_native_w_cross_section_pillow,
    plot_composite_native_w_focus_pillow,
    sample_stack,
)


DEFAULT_EXPERIMENT_ROOT = Path(
    r"E:\DATA\01_Eddy_correspond\02_OFES"
    r"\origin_unified_eta_mss_three_kernel_surface_jan01_jan19"
)
DEFAULT_OUTPUT_ROOT = DEFAULT_EXPERIMENT_ROOT / "comparison" / "isopycnal_composite_jan01_hemisphere_polarity"
DEFAULT_DATA_ROOT = Path(r"F:\OFES\external_OFES2")
KERNELS = ("gaussian", "lanczos", "bessel")
GROUPS = ("NH_cyclonic", "NH_anticyclonic", "SH_cyclonic", "SH_anticyclonic")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, default=DEFAULT_EXPERIMENT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--day", default="1991-01-01")
    parser.add_argument("--kernels", default=",".join(KERNELS))
    parser.add_argument("--groups", default=",".join(GROUPS))
    parser.add_argument("--grid-n", type=int, default=81)
    parser.add_argument("--extent-r", type=float, default=4.0)
    parser.add_argument("--max-depth-layers", type=int, default=105)
    parser.add_argument("--cressman-radius-r", type=float, default=1.0)
    parser.add_argument("--cressman-min-objects", type=int, default=8)
    parser.add_argument(
        "--composite-method", choices=["cressman", "pointwise_mean"], default="cressman",
        help="Cressman smoothing or direct normalized-grid pointwise averaging.",
    )
    parser.add_argument("--max-geometry-depth-m", type=float, default=2000.0,
                        help="Maximum depth retained in QC geometry figures and scientific summaries.")
    parser.add_argument("--background-ring-min-r", type=float, default=2.0,
                        help="Inner radius of the normalized outer-ring density reference.")
    parser.add_argument("--background-ring-max-r", type=float, default=4.0,
                        help="Outer radius of the normalized outer-ring density reference.")
    parser.add_argument("--min-bracket-stratification", type=float, default=1.0e-4,
                        help="Minimum absolute density slope |delta rho/delta D| (kg m-4) for a valid inversion bracket.")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-objects-per-group", type=int, default=0, help="Positive values are smoke-only limits.")
    parser.add_argument("--object-table", type=Path, help="Optional existing object table; replaces the three-kernel catalog input.")
    parser.add_argument("--composite-label", default="", help="Output label used with --object-table.")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def csv_values(value: str, allowed: tuple[str, ...], label: str) -> list[str]:
    result = [part.strip() for part in value.split(",") if part.strip()]
    unknown = sorted(set(result) - set(allowed))
    if not result or unknown:
        raise ValueError(f"Invalid {label}: {', '.join(unknown) or '<empty>'}; allowed: {', '.join(allowed)}")
    return result


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_surface_objects(root: Path, kernel: str, day: str) -> dict[str, list[SelectedObject]]:
    table = root / kernel / "catalog" / "final_catalog" / "daily_runs" / day.replace("-", "") / "centers_hua_style.csv"
    if not table.exists():
        raise FileNotFoundError(table)
    centers = pd.read_csv(table)
    lon_col = "center_lon_refined" if "center_lon_refined" in centers.columns else "center_lon"
    lat_col = "center_lat_refined" if "center_lat_refined" in centers.columns else "center_lat"
    groups = {key: [] for key in GROUPS}
    for _, row in centers.iterrows():
        lat = float(row[lat_col])
        radius = float(row["radius_km"])
        if not (np.isfinite(lat) and np.isfinite(radius) and radius > 0):
            continue
        polarity = str(row["polarity"]).lower()
        if polarity not in {"cyclonic", "anticyclonic"}:
            continue
        hemisphere = "NH" if lat >= 0 else "SH"
        groups[f"{hemisphere}_{polarity}"].append(
            SelectedObject(
                hua_object_id=str(row["hua_object_id"]), date=day, polarity=polarity,
                pass_layers=1, max_jump_km=0.0, max_jump_over_r=0.0, radius_km=radius,
                center_lon=float(row[lon_col]), center_lat=lat,
            )
        )
    for values in groups.values():
        values.sort(key=lambda obj: (obj.radius_km, obj.hua_object_id), reverse=True)
    return groups


def load_table_objects(table: Path, day: str) -> dict[str, list[SelectedObject]]:
    """Use an already-selected object table without changing composite physics."""
    if not table.exists():
        raise FileNotFoundError(table)
    rows = pd.read_csv(table)
    if "depth_index" in rows.columns:
        rows = rows.loc[rows["depth_index"].eq(0)].copy()
    lon_col = "center_lon_refined" if "center_lon_refined" in rows.columns else "center_lon"
    lat_col = "center_lat_refined" if "center_lat_refined" in rows.columns else "center_lat"
    required = {"hua_object_id", "polarity", "radius_km", lon_col, lat_col}
    missing = required - set(rows.columns)
    if missing:
        raise ValueError(f"Object table is missing required columns: {sorted(missing)}")
    groups = {key: [] for key in GROUPS}
    for _, row in rows.iterrows():
        lon, lat, radius = float(row[lon_col]), float(row[lat_col]), float(row["radius_km"])
        polarity = str(row["polarity"]).lower()
        if not (np.isfinite(lon) and np.isfinite(lat) and np.isfinite(radius) and radius > 0):
            continue
        if polarity not in {"cyclonic", "anticyclonic"}:
            continue
        hemisphere = "NH" if lat >= 0.0 else "SH"
        groups[f"{hemisphere}_{polarity}"].append(
            SelectedObject(str(row["hua_object_id"]), day, polarity, 1, 0.0, 0.0, radius, lon, lat)
        )
    for values in groups.values():
        values.sort(key=lambda obj: (obj.radius_km, obj.hua_object_id), reverse=True)
    return groups


def sample_object_geometry(raw: dict[str, np.memmap], metas: dict[str, object], obj: SelectedObject, args: argparse.Namespace) -> dict[str, object]:
    nlev = min(int(args.max_depth_layers), metas["prho"].z.count, metas["w"].z.count)
    depth = np.asarray(metas["prho"].z.values[:nlev], dtype="f4")
    x = np.linspace(-float(args.extent_r), float(args.extent_r), int(args.grid_n), dtype="f4")
    y = np.linspace(-float(args.extent_r), float(args.extent_r), int(args.grid_n), dtype="f4")
    xx, yy = np.meshgrid(x, y, indexing="xy")
    lon, lat = local_lon_lat_grid(xx, yy, obj.center_lon, obj.center_lat, obj.radius_km)
    prho = normalize_density_units(sample_stack(raw["prho"], metas["prho"], lon, lat, nlev)).astype("f4")
    w_source_depth = np.asarray(metas["w"].z.values[:nlev], dtype="f8")
    w_raw = sample_stack(raw["w"], metas["w"], lon, lat, nlev) / 100.0
    native_w, _ = align_native_w_vertical(w_raw, w_source_depth, depth.astype("f8"), "layer_center")
    return {"object": obj, "depth_m": depth, "x_over_r": x, "y_over_r": y, "prho": prho, "native_w": native_w.astype("f4")}


def init_accumulator(template: dict[str, object], args: argparse.Namespace) -> dict[str, object]:
    depth = np.asarray(template["depth_m"], dtype="f4")
    x = np.asarray(template["x_over_r"], dtype="f4")
    y = np.asarray(template["y_over_r"], dtype="f4")
    shape = (depth.size, y.size, x.size)
    return {
        "depth_m": depth, "x_over_r": x, "y_over_r": y,
        "sum": {key: np.zeros(shape, dtype="f8") for key in ("prho", "native_w")},
        "count": {key: np.zeros(shape, dtype="u2") for key in ("prho", "native_w")},
        "object_ids": [], "radii_m": [], "center_lats": [], "object_count": 0,
    }


def update_accumulator(acc: dict[str, object], sampled: dict[str, object], kernel2d: np.ndarray | None) -> None:
    obj = sampled["object"]
    for key in ("prho", "native_w"):
        values = np.asarray(sampled[key], dtype="f4")
        if kernel2d is None:
            valid = np.isfinite(values)
            acc["sum"][key][valid] += values[valid]
        else:
            mapped, support = cressman_map_3d(values, kernel2d)
            valid = support & np.isfinite(mapped)
            acc["sum"][key][valid] += mapped[valid]
        acc["count"][key][valid] += 1
    acc["object_ids"].append(obj.hua_object_id)
    acc["radii_m"].append(float(obj.radius_km) * 1000.0)
    acc["center_lats"].append(float(obj.center_lat))
    acc["object_count"] += 1


def direct_isopycnal_depth(
    prho: np.ndarray,
    depth: np.ndarray,
    center_profile: np.ndarray,
    support: np.ndarray,
    min_objects: int,
    min_bracket_stratification: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Invert composite density columns; choose the bracket nearest each reference depth."""
    nz, ny, nx = prho.shape
    out = np.full((nz, ny, nx), np.nan, dtype="f4")
    out_support = np.zeros((nz, ny, nx), dtype="u2")
    crossing_count = np.zeros((nz, ny, nx), dtype="u2")
    bracket_stratification = np.full((nz, ny, nx), np.nan, dtype="f4")
    levels = np.asarray(center_profile, dtype="f8")
    lower = np.asarray(prho[:-1], dtype="f8")
    upper = np.asarray(prho[1:], dtype="f8")
    pair_support = np.minimum(np.asarray(support[:-1]), np.asarray(support[1:]))
    mid_depth = 0.5 * (np.asarray(depth[:-1], dtype="f8") + np.asarray(depth[1:], dtype="f8"))[:, None, None]
    for zz, level in enumerate(levels):
        if not np.isfinite(level):
            continue
        crosses = np.isfinite(lower) & np.isfinite(upper) & (np.abs(upper - lower) > 1.0e-10)
        crosses &= ((lower <= level) & (level <= upper)) | ((upper <= level) & (level <= lower))
        crossing_count[zz] = np.minimum(np.sum(crosses, axis=0), np.iinfo(np.uint16).max).astype("u2")
        costs = np.where(crosses, np.abs(mid_depth - float(depth[zz])), np.inf)
        index = np.argmin(costs, axis=0)
        any_cross = np.isfinite(np.take_along_axis(costs, index[None, :, :], axis=0)[0])
        p0 = np.take_along_axis(lower, index[None, :, :], axis=0)[0]
        p1 = np.take_along_axis(upper, index[None, :, :], axis=0)[0]
        s0 = np.take_along_axis(pair_support, index[None, :, :], axis=0)[0]
        d0 = np.asarray(depth, dtype="f8")[index]
        d1 = np.asarray(depth, dtype="f8")[index + 1]
        local_stratification = np.abs((p1 - p0) / (d1 - d0))
        good = any_cross & (s0 >= int(min_objects)) & (local_stratification >= float(min_bracket_stratification))
        fraction = np.divide(
            float(level) - p0,
            p1 - p0,
            out=np.full_like(p0, np.nan, dtype="f8"),
            where=np.abs(p1 - p0) > 1.0e-10,
        )
        values = d0 + fraction * (d1 - d0)
        out[zz, good] = values[good].astype("f4")
        out_support[zz, good] = s0[good]
        bracket_stratification[zz, good] = local_stratification[good].astype("f4")
    anomaly = out - np.asarray(depth, dtype="f4")[:, None, None]
    return out, anomaly, out_support, crossing_count, bracket_stratification


def geometry_gradients(d_rho: np.ndarray, x: np.ndarray, y: np.ndarray, radius_m: float) -> tuple[np.ndarray, np.ndarray]:
    dx = float(np.nanmedian(np.diff(x))) * radius_m
    dy = float(np.nanmedian(np.diff(y))) * radius_m
    d_dy, d_dx = np.gradient(np.asarray(d_rho, dtype="f4"), dy, dx, axis=(1, 2))
    return d_dx.astype("f4"), d_dy.astype("f4")


def render_qc_section_matlab(
    group_root: Path,
    stem: str,
    x: np.ndarray,
    depth: np.ndarray,
    section: np.ndarray,
    title: str,
    colorbar_label: str,
) -> Path | None:
    """Use MATLAB for contours because this host's matplotlib renderer is unstable."""
    input_mat = group_root / f"{stem}.mat"
    output_png = group_root / "figures" / f"{stem}.png"
    output_png.parent.mkdir(parents=True, exist_ok=True)
    savemat(input_mat, {"x_over_r": x, "depth_m": depth, "section_m": section})
    matlab = os.environ.get("MATLAB_EXE") or shutil.which("matlab") or r"D:\Util\Ma\01_Matlab\bin\matlab.exe"
    if not Path(matlab).exists():
        return None
    helper_dir = Path(__file__).resolve().parent
    escaped = lambda value: str(value).replace("'", "''")
    command = (
        f"addpath('{escaped(helper_dir)}'); "
        f"plot_isopycnal_qc_section('{escaped(input_mat)}','{escaped(output_png)}','{escaped(title)}','{escaped(colorbar_label)}');"
    )
    completed = subprocess.run([matlab, "-batch", command], capture_output=True, text=True, timeout=180, check=False)
    if completed.returncode != 0 or not output_png.exists():
        (group_root / "MATLAB_RENDER_WARNING.txt").write_text(
            "MATLAB QC section rendering failed.\n" + completed.stdout + "\n" + completed.stderr,
            encoding="utf-8",
        )
        return None
    return output_png


def write_group_outputs(
    root: Path, kernel: str, group: str, payload: dict[str, object], *, render_geometry_sections: bool = True,
) -> dict[str, object]:
    group_root = root / kernel / group
    figures = group_root / "figures"
    group_root.mkdir(parents=True, exist_ok=True)
    arrays = {key: value for key, value in payload.items() if isinstance(value, np.ndarray)}
    np.savez_compressed(group_root / "isopycnal_composite.npz", **arrays)
    metadata = {key: value for key, value in payload.items() if not isinstance(value, np.ndarray)}
    (group_root / "isopycnal_composite.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    x, y, depth = (np.asarray(payload[key]) for key in ("x_over_r", "y_over_r", "depth_m"))
    center_y = y.size // 2
    iso_anom = np.asarray(payload["d_rho_anom_qc_m"])
    native_w = np.asarray(payload["native_w_m_s"])
    max_depth = float(payload["max_geometry_depth_m"])
    keep = depth <= max_depth
    support = np.asarray(payload["support_d_rho_objects"])
    crossings = np.asarray(payload["bracket_crossing_count"])
    stratification = np.asarray(payload["bracket_stratification_kg_m4"])
    layer_rows: list[dict[str, float]] = []
    for index, nominal_depth in enumerate(depth):
        layer = iso_anom[index]
        valid = np.isfinite(layer)
        layer_rows.append({
            "nominal_depth_m": float(nominal_depth),
            "within_display_depth": bool(nominal_depth <= max_depth),
            "valid_cells": int(np.sum(valid)),
            "valid_fraction": float(np.mean(valid)),
            "median_object_support": float(np.nanmedian(np.where(valid, support[index], np.nan))) if np.any(valid) else np.nan,
            "median_crossing_count": float(np.nanmedian(np.where(valid, crossings[index], np.nan))) if np.any(valid) else np.nan,
            "median_bracket_stratification_kg_m4": float(np.nanmedian(np.where(valid, stratification[index], np.nan))) if np.any(valid) else np.nan,
        })
    pd.DataFrame(layer_rows).to_csv(group_root / "isopycnal_qc_by_depth.csv", index=False, encoding="utf-8-sig")
    section_path = None
    if render_geometry_sections:
        section_path = render_qc_section_matlab(
            group_root, "isopycnal_section_x_qc", x, depth[keep], iso_anom[keep, center_y, :],
            f"{kernel} {group}: QC direct-isopycnal displacement (0-{max_depth:.0f} m)", "D_{rho} - D0 (m)",
        )
    rho_anom = np.asarray(payload["rho_anom_ring_qc_kg_m3"])
    density_section_path = None
    if render_geometry_sections:
        density_section_path = render_qc_section_matlab(
            group_root, "density_anomaly_section_x_qc", x, depth[keep], rho_anom[keep, center_y, :],
            f"{kernel} {group}: QC fixed-depth density anomaly (0-{max_depth:.0f} m)", "rho prime (kg m^{-3})",
        )
    # Reuse the established Pillow W composite figures.  This geometry-first
    # pipeline stores native W under a shorter key, so expose the aliases the
    # renderer already consumes instead of duplicating a plotting method.
    w_plot_payload = dict(payload)
    w_plot_payload["composite_ofes_w_native_m_s"] = native_w
    w_plot_payload["section_ofes_w_native_m_s"] = native_w[:, center_y, :]
    w_plot_payload["target_lat"] = float(payload["mean_center_lat"])
    w_plot_payload["polarity"] = group
    w_plot_payload["multipole_class"] = str(payload.get("multipole_class", "all_selected"))
    w_plot_payload["region"] = "strict-core object table" if "strict" in kernel else ""
    w_section = figures / "native_w_cross_section.png"
    w_focus = figures / "native_w_focus.png"
    plot_composite_native_w_cross_section_pillow(w_plot_payload, w_section)
    plot_composite_native_w_focus_pillow(w_plot_payload, w_focus)
    return {
        "kernel": kernel, "group": group, "object_count": payload["object_count"],
        "mean_radius_km": payload["mean_radius_m"] / 1000.0,
        "valid_isopycnal_fraction": float(np.isfinite(iso_anom).sum() / iso_anom.size),
        "valid_native_w_fraction": float(np.isfinite(native_w).sum() / native_w.size),
        "npz": str(group_root / "isopycnal_composite.npz"),
        "isopycnal_section": str(section_path) if section_path else None,
        "density_anomaly_section": str(density_section_path) if density_section_path else None,
        "isopycnal_qc_by_depth": str(group_root / "isopycnal_qc_by_depth.csv"),
        "native_w_section": str(w_section),
        "native_w_focus": str(w_focus),
    }


def run_group(raw: dict[str, np.memmap], metas: dict[str, object], objects: list[SelectedObject], args: argparse.Namespace) -> dict[str, object] | None:
    if args.max_objects_per_group > 0:
        objects = objects[: int(args.max_objects_per_group)]
    if not objects:
        return None
    acc = None
    failures = 0
    workers = max(1, int(args.workers))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(sample_object_geometry, raw, metas, obj, args) for obj in objects]
        for future in as_completed(futures):
            try:
                sampled = future.result()
            except Exception as exc:
                failures += 1
                print(f"[geometry] object failed: {exc}", flush=True)
                continue
            if acc is None:
                acc = init_accumulator(sampled, args)
                kernel2d = (
                    cressman_kernel_2d(acc["x_over_r"], acc["y_over_r"], float(args.cressman_radius_r))
                    if args.composite_method == "cressman" else None
                )
            update_accumulator(acc, sampled, kernel2d)
    if acc is None:
        return None
    return finalize_accumulator(acc, failures, args)


def finalize_accumulator(acc: dict[str, object], failures: int, args: argparse.Namespace) -> dict[str, object]:
    """Finalize a shared object accumulator without changing composite physics."""
    prho = finalize_sum_count(acc["sum"]["prho"], acc["count"]["prho"], int(args.cressman_min_objects))
    native_w = finalize_sum_count(acc["sum"]["native_w"], acc["count"]["native_w"], int(args.cressman_min_objects))
    depth = np.asarray(acc["depth_m"], dtype="f4")
    x, y = np.asarray(acc["x_over_r"], dtype="f4"), np.asarray(acc["y_over_r"], dtype="f4")
    center_profile = prho[:, y.size // 2, x.size // 2]
    d_rho, d_anom, d_support, crossing_count, bracket_stratification = direct_isopycnal_depth(
        prho, depth, center_profile, acc["count"]["prho"], int(args.cressman_min_objects),
        float(args.min_bracket_stratification),
    )
    shallow = depth <= float(args.max_geometry_depth_m)
    d_rho_qc = d_rho.copy()
    d_anom_qc = d_anom.copy()
    d_rho_qc[~shallow, :, :] = np.nan
    d_anom_qc[~shallow, :, :] = np.nan
    xx, yy = np.meshgrid(x, y, indexing="xy")
    radius = np.hypot(xx, yy)
    ring = (radius >= float(args.background_ring_min_r)) & (radius <= float(args.background_ring_max_r))
    ring_profile = np.nanmedian(np.where(ring[None, :, :], prho, np.nan), axis=(1, 2)).astype("f4")
    rho_anom_ring = prho - ring_profile[:, None, None]
    rho_anom_ring_qc = rho_anom_ring.copy()
    rho_anom_ring_qc[~shallow, :, :] = np.nan
    rho_anom_ring_qc[np.asarray(acc["count"]["prho"]) < int(args.cressman_min_objects)] = np.nan
    radius_m = float(np.nanmedian(acc["radii_m"]))
    d_dx, d_dy = geometry_gradients(d_rho, x, y, radius_m)
    d_dx[~np.isfinite(d_rho)] = np.nan
    d_dy[~np.isfinite(d_rho)] = np.nan
    return {
        "depth_m": depth, "x_over_r": x, "y_over_r": y, "composite_prho": prho,
        "native_w_m_s": native_w, "support_prho_objects": acc["count"]["prho"],
        "support_native_w_objects": acc["count"]["native_w"], "rho0_center_profile": center_profile,
        "d_rho_m": d_rho, "d_rho_anom_m": d_anom, "support_d_rho_objects": d_support,
        "d_rho_qc_m": d_rho_qc, "d_rho_anom_qc_m": d_anom_qc,
        "bracket_crossing_count": crossing_count, "bracket_stratification_kg_m4": bracket_stratification,
        "rho_reference_ring_profile": ring_profile,
        "rho_anom_ring_kg_m3": rho_anom_ring.astype("f4"), "rho_anom_ring_qc_kg_m3": rho_anom_ring_qc.astype("f4"),
        "dDdx_m_per_m": d_dx, "dDdy_m_per_m": d_dy,
        "object_count": int(acc["object_count"]), "failed_object_count": int(failures),
        "mean_radius_m": radius_m, "mean_center_lat": float(np.nanmean(acc["center_lats"])),
        "source_object_ids": ",".join(acc["object_ids"]),
        "day": str(args.day), "geometry_method": "composite_prho_then_direct_bracket_isopycnal_inversion",
        "rho0_policy": "group_composite_center_profile_at_each_nominal_depth",
        "multiple_crossing_policy": "valid_bracket_nearest_nominal_depth",
        "max_geometry_depth_m": float(args.max_geometry_depth_m),
        "min_bracket_stratification_kg_m4": float(args.min_bracket_stratification),
        "density_anomaly_reference": "group_composite_prho_outer_ring_median_at_fixed_depth",
        "background_ring_r": f"{float(args.background_ring_min_r):g}-{float(args.background_ring_max_r):g}",
        "density_derivative_used_for_geometry": False,
        "native_w_policy": (
            "OFES_native_w_cressman_composite_only_not_rebuild_w"
            if args.composite_method == "cressman"
            else "OFES_native_w_direct_pointwise_mean_only_not_rebuild_w"
        ),
        "composite_method": str(args.composite_method),
        "cressman_radius_r": float(args.cressman_radius_r), "cressman_min_objects": int(args.cressman_min_objects),
        "gradient_metric_radius_m": radius_m,
    }


def main() -> None:
    args = parse_args()
    kernels = csv_values(args.kernels, KERNELS, "--kernels")
    groups = csv_values(args.groups, GROUPS, "--groups")
    metas = {name: parse_ctl(ctl_path(args.data_root, name)) for name in ("prho", "w")}
    day = datetime.strptime(args.day, "%Y-%m-%d").date()
    raw = {name: open_dta_memmap(require_daily_file(args.data_root, name, day, expected_dta_bytes(meta)), meta) for name, meta in metas.items()}
    args.output_root.mkdir(parents=True, exist_ok=True)
    if args.object_table:
        labels = [str(args.composite_label).strip() or "object_table"]
        object_groups_by_label = {labels[0]: load_table_objects(args.object_table, args.day)}
    else:
        labels = kernels
        object_groups_by_label = {
            kernel: load_surface_objects(args.experiment_root, kernel, args.day)
            for kernel in labels
        }
    summaries: list[dict[str, object]] = []
    for kernel in labels:
        object_groups = object_groups_by_label[kernel]
        for group in groups:
            group_root = args.output_root / kernel / group
            if args.resume and (group_root / "isopycnal_composite.npz").exists() and (group_root / "isopycnal_composite.json").exists():
                print(f"[{kernel}/{group}] resume existing output", flush=True)
                continue
            objects = object_groups[group]
            print(f"[{kernel}/{group}] sampling {len(objects)} Jan-1 surface objects", flush=True)
            payload = run_group(raw, metas, objects, args)
            if payload is None:
                summaries.append({"kernel": kernel, "group": group, "status": "no_successful_objects"})
                continue
            summaries.append({"status": "ok", **write_group_outputs(args.output_root, kernel, group, payload)})
    pd.DataFrame(summaries).to_csv(args.output_root / "SUMMARY.csv", index=False, encoding="utf-8-sig")
    (args.output_root / "SUMMARY.json").write_text(json.dumps({"created_utc": utc_now(), "rows": summaries}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[geometry] wrote {args.output_root}", flush=True)


if __name__ == "__main__":
    main()
