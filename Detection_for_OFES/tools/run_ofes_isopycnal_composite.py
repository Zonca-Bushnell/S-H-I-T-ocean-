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
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..ofes_io import ctl_path, expected_dta_bytes, open_dta_memmap, parse_ctl, require_daily_file
from ..run_ofes_rebuild_w import (
    SelectedObject,
    align_native_w_vertical,
    cressman_kernel_2d,
    cressman_map_3d,
    finalize_sum_count,
    local_lon_lat_grid,
    normalize_density_units,
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
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-objects-per-group", type=int, default=0, help="Positive values are smoke-only limits.")
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
        "object_ids": [], "radii_m": [], "object_count": 0,
    }


def update_accumulator(acc: dict[str, object], sampled: dict[str, object], kernel2d: np.ndarray) -> None:
    obj = sampled["object"]
    for key in ("prho", "native_w"):
        mapped, support = cressman_map_3d(np.asarray(sampled[key], dtype="f4"), kernel2d)
        valid = support & np.isfinite(mapped)
        acc["sum"][key][valid] += mapped[valid]
        acc["count"][key][valid] += 1
    acc["object_ids"].append(obj.hua_object_id)
    acc["radii_m"].append(float(obj.radius_km) * 1000.0)
    acc["object_count"] += 1


def direct_isopycnal_depth(prho: np.ndarray, depth: np.ndarray, center_profile: np.ndarray, support: np.ndarray, min_objects: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Invert composite density columns; choose the bracket nearest each reference depth."""
    nz, ny, nx = prho.shape
    out = np.full((nz, ny, nx), np.nan, dtype="f4")
    out_support = np.zeros((nz, ny, nx), dtype="u2")
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
        costs = np.where(crosses, np.abs(mid_depth - float(depth[zz])), np.inf)
        index = np.argmin(costs, axis=0)
        any_cross = np.isfinite(np.take_along_axis(costs, index[None, :, :], axis=0)[0])
        p0 = np.take_along_axis(lower, index[None, :, :], axis=0)[0]
        p1 = np.take_along_axis(upper, index[None, :, :], axis=0)[0]
        s0 = np.take_along_axis(pair_support, index[None, :, :], axis=0)[0]
        d0 = np.asarray(depth, dtype="f8")[index]
        d1 = np.asarray(depth, dtype="f8")[index + 1]
        good = any_cross & (s0 >= int(min_objects))
        fraction = np.divide(
            float(level) - p0,
            p1 - p0,
            out=np.full_like(p0, np.nan, dtype="f8"),
            where=np.abs(p1 - p0) > 1.0e-10,
        )
        values = d0 + fraction * (d1 - d0)
        out[zz, good] = values[good].astype("f4")
        out_support[zz, good] = s0[good]
    anomaly = out - np.asarray(depth, dtype="f4")[:, None, None]
    return out, anomaly, out_support


def geometry_gradients(d_rho: np.ndarray, x: np.ndarray, y: np.ndarray, radius_m: float) -> tuple[np.ndarray, np.ndarray]:
    dx = float(np.nanmedian(np.diff(x))) * radius_m
    dy = float(np.nanmedian(np.diff(y))) * radius_m
    d_dy, d_dx = np.gradient(np.asarray(d_rho, dtype="f4"), dy, dx, axis=(1, 2))
    return d_dx.astype("f4"), d_dy.astype("f4")


def finite_limit(values: np.ndarray, fallback: float = 1.0) -> float:
    finite = np.abs(values[np.isfinite(values)])
    return max(float(np.nanpercentile(finite, 95)), fallback) if finite.size else fallback


def plot_section(path: Path, data: np.ndarray, x: np.ndarray, depth: np.ndarray, title: str, label: str, scale: float = 1.0) -> None:
    lim = finite_limit(data * scale)
    fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
    image = ax.imshow(data * scale, origin="upper", aspect="auto", extent=[x[0], x[-1], depth[-1], depth[0]], cmap="RdBu_r", vmin=-lim, vmax=lim, interpolation="nearest")
    ax.set(title=title, xlabel="x/R (east-west)", ylabel="Depth (m)")
    ax.axvline(0, color="black", linewidth=0.7)
    fig.colorbar(image, ax=ax, label=label)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_slices(path: Path, data: np.ndarray, x: np.ndarray, y: np.ndarray, depth: np.ndarray, targets: list[float], title: str, label: str, scale: float = 1.0) -> None:
    lim = finite_limit(data * scale)
    fig, axes = plt.subplots(2, 2, figsize=(10, 9), constrained_layout=True)
    image = None
    for ax, target in zip(axes.flat, targets):
        index = int(np.nanargmin(np.abs(depth - target)))
        image = ax.imshow(data[index] * scale, origin="lower", extent=[x[0], x[-1], y[0], y[-1]], cmap="RdBu_r", vmin=-lim, vmax=lim, interpolation="nearest")
        ax.set(title=f"D0={depth[index]:.0f} m", xlabel="x/R (east-west)", ylabel="y/R (north-south)")
        ax.axhline(0, color="black", linewidth=0.5)
        ax.axvline(0, color="black", linewidth=0.5)
    fig.suptitle(title)
    if image is not None:
        fig.colorbar(image, ax=axes.ravel().tolist(), label=label, shrink=0.85)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_group_outputs(root: Path, kernel: str, group: str, payload: dict[str, object]) -> dict[str, object]:
    group_root = root / kernel / group
    figures = group_root / "figures"
    group_root.mkdir(parents=True, exist_ok=True)
    arrays = {key: value for key, value in payload.items() if isinstance(value, np.ndarray)}
    np.savez_compressed(group_root / "isopycnal_composite.npz", **arrays)
    metadata = {key: value for key, value in payload.items() if not isinstance(value, np.ndarray)}
    (group_root / "isopycnal_composite.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    x, y, depth = (np.asarray(payload[key]) for key in ("x_over_r", "y_over_r", "depth_m"))
    center_y = y.size // 2
    iso_anom = np.asarray(payload["d_rho_anom_m"])
    native_w = np.asarray(payload["native_w_m_s"])
    plot_section(figures / "isopycnal_section_x.png", iso_anom[:, center_y, :], x, depth, f"{kernel} {group}: direct-isopycnal displacement", "D_rho - D0 (m)")
    plot_slices(figures / "isopycnal_slices.png", iso_anom, x, y, depth, [100, 500, 1000, 1500], f"{kernel} {group}: direct-isopycnal displacement", "D_rho - D0 (m)")
    plot_section(figures / "native_w_section_x.png", native_w[:, center_y, :], x, depth, f"{kernel} {group}: OFES native W", "W (10^-6 m s^-1)", 1.0e6)
    plot_slices(figures / "native_w_slices.png", native_w, x, y, depth, [100, 500, 1000, 1500], f"{kernel} {group}: OFES native W", "W (10^-6 m s^-1)", 1.0e6)
    return {
        "kernel": kernel, "group": group, "object_count": payload["object_count"],
        "mean_radius_km": payload["mean_radius_m"] / 1000.0,
        "valid_isopycnal_fraction": float(np.isfinite(iso_anom).sum() / iso_anom.size),
        "valid_native_w_fraction": float(np.isfinite(native_w).sum() / native_w.size),
        "npz": str(group_root / "isopycnal_composite.npz"),
        "isopycnal_section": str(figures / "isopycnal_section_x.png"),
        "native_w_section": str(figures / "native_w_section_x.png"),
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
                kernel2d = cressman_kernel_2d(acc["x_over_r"], acc["y_over_r"], float(args.cressman_radius_r))
            update_accumulator(acc, sampled, kernel2d)
    if acc is None:
        return None
    prho = finalize_sum_count(acc["sum"]["prho"], acc["count"]["prho"], int(args.cressman_min_objects))
    native_w = finalize_sum_count(acc["sum"]["native_w"], acc["count"]["native_w"], int(args.cressman_min_objects))
    depth = np.asarray(acc["depth_m"], dtype="f4")
    x, y = np.asarray(acc["x_over_r"], dtype="f4"), np.asarray(acc["y_over_r"], dtype="f4")
    center_profile = prho[:, y.size // 2, x.size // 2]
    d_rho, d_anom, d_support = direct_isopycnal_depth(prho, depth, center_profile, acc["count"]["prho"], int(args.cressman_min_objects))
    radius_m = float(np.nanmedian(acc["radii_m"]))
    d_dx, d_dy = geometry_gradients(d_rho, x, y, radius_m)
    d_dx[~np.isfinite(d_rho)] = np.nan
    d_dy[~np.isfinite(d_rho)] = np.nan
    return {
        "depth_m": depth, "x_over_r": x, "y_over_r": y, "composite_prho": prho,
        "native_w_m_s": native_w, "support_prho_objects": acc["count"]["prho"],
        "support_native_w_objects": acc["count"]["native_w"], "rho0_center_profile": center_profile,
        "d_rho_m": d_rho, "d_rho_anom_m": d_anom, "support_d_rho_objects": d_support,
        "dDdx_m_per_m": d_dx, "dDdy_m_per_m": d_dy,
        "object_count": int(acc["object_count"]), "failed_object_count": int(failures),
        "mean_radius_m": radius_m, "source_object_ids": ",".join(acc["object_ids"]),
        "day": str(args.day), "geometry_method": "composite_prho_then_direct_bracket_isopycnal_inversion",
        "rho0_policy": "group_composite_center_profile_at_each_nominal_depth",
        "multiple_crossing_policy": "valid_bracket_nearest_nominal_depth",
        "density_derivative_used_for_geometry": False,
        "native_w_policy": "OFES_native_w_cressman_composite_only_not_rebuild_w",
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
    summaries: list[dict[str, object]] = []
    for kernel in kernels:
        object_groups = load_surface_objects(args.experiment_root, kernel, args.day)
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
