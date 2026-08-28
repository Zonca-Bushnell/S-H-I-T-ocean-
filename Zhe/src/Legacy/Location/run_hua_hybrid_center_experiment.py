from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from netCDF4 import Dataset, num2date


TARGET_OBJECTS = (1120461, 1076249)
EARTH_RADIUS_M = 6_371_000.0


@dataclass
class GridSlice:
    depth_index: int
    depth_m: float
    x_km: np.ndarray
    y_km: np.ndarray
    u_anom: np.ndarray
    v_anom: np.ndarray
    speed: np.ndarray


def _date(value: str) -> date:
    return datetime.strptime(str(value)[:10], "%Y-%m-%d").date()


def _wrap_lon_delta_deg(lon: np.ndarray, lon0: float) -> np.ndarray:
    return (lon - lon0 + 180.0) % 360.0 - 180.0


def _axis_rows(axis_path: Path, object_ids: tuple[int, ...]) -> pd.DataFrame:
    filters = [("eddy3d_object_id", "in", list(object_ids))]
    try:
        return pd.read_parquet(axis_path, filters=filters)
    except Exception:
        df = pd.read_parquet(axis_path)
        return df[df["eddy3d_object_id"].isin(object_ids)].copy()


def _nearest_time_index(ds: Dataset, wanted: date) -> int:
    tvar = ds.variables["time"]
    times = num2date(tvar[:], units=tvar.units, calendar=getattr(tvar, "calendar", "standard"))
    keys = [date(int(t.year), int(t.month), int(t.day)) for t in times]
    try:
        return keys.index(wanted)
    except ValueError as exc:
        raise ValueError(f"Date {wanted} not found in {ds.filepath()}") from exc


def _doy_index(clim: Dataset, wanted: date) -> int:
    doy = wanted.timetuple().tm_yday
    vals = np.asarray(clim.variables["doy"][:], dtype=int)
    where = np.where(vals == doy)[0]
    if len(where):
        return int(where[0])
    return doy - 1


def _subset_indices(lon: np.ndarray, lat: np.ndarray, lon0: float, lat0: float, half_width_km: float) -> tuple[np.ndarray, np.ndarray]:
    dlon_m = np.deg2rad(_wrap_lon_delta_deg(lon, lon0)) * EARTH_RADIUS_M * math.cos(math.radians(lat0))
    dlat_m = np.deg2rad(lat - lat0) * EARTH_RADIUS_M
    lon_idx = np.where(np.abs(dlon_m) <= half_width_km * 1000.0)[0]
    lat_idx = np.where(np.abs(dlat_m) <= half_width_km * 1000.0)[0]
    if len(lon_idx) < 5 or len(lat_idx) < 5:
        raise ValueError("Velocity subset too small; increase window or check center coordinates")
    return lon_idx, lat_idx


def _interp2(x: np.ndarray, y: np.ndarray, arr: np.ndarray, xp: np.ndarray, yp: np.ndarray) -> np.ndarray:
    out = np.full(np.shape(xp), np.nan, dtype="float64")
    xp_flat = np.asarray(xp, dtype="float64").ravel()
    yp_flat = np.asarray(yp, dtype="float64").ravel()
    vals = out.ravel()
    for n, (xx, yy) in enumerate(zip(xp_flat, yp_flat)):
        if not (x[0] <= xx <= x[-1] and y[0] <= yy <= y[-1]):
            continue
        ix = int(np.searchsorted(x, xx) - 1)
        iy = int(np.searchsorted(y, yy) - 1)
        ix = max(0, min(ix, len(x) - 2))
        iy = max(0, min(iy, len(y) - 2))
        x0, x1 = x[ix], x[ix + 1]
        y0, y1 = y[iy], y[iy + 1]
        if x1 == x0 or y1 == y0:
            continue
        q11, q12 = arr[iy, ix], arr[iy + 1, ix]
        q21, q22 = arr[iy, ix + 1], arr[iy + 1, ix + 1]
        if not np.all(np.isfinite([q11, q12, q21, q22])):
            continue
        wx = (xx - x0) / (x1 - x0)
        wy = (yy - y0) / (y1 - y0)
        vals[n] = (1 - wx) * (1 - wy) * q11 + wx * (1 - wy) * q21 + (1 - wx) * wy * q12 + wx * wy * q22
    return out


def _angle_diff(a: float, b: float) -> float:
    diff = a - b
    if diff > math.pi:
        diff -= 2.0 * math.pi
    elif diff < -math.pi:
        diff += 2.0 * math.pi
    return diff


def _circle_points(radius_cells: int) -> list[tuple[int, int]]:
    points = []
    n = max(16, int(round(8 * radius_cells)))
    for theta in np.linspace(-math.pi / 2.0, 3.0 * math.pi / 2.0, n, endpoint=False):
        ix = int(round(radius_cells * math.cos(theta)))
        iy = int(round(radius_cells * math.sin(theta)))
        if not points or points[-1] != (ix, iy):
            points.append((ix, iy))
    if len(points) > 1 and points[0] == points[-1]:
        points.pop()
    return points


def _grid_spacing_km(sl: GridSlice) -> float:
    dx = float(np.nanmedian(np.abs(np.diff(sl.x_km)))) if len(sl.x_km) > 1 else np.nan
    dy = float(np.nanmedian(np.abs(np.diff(sl.y_km)))) if len(sl.y_km) > 1 else np.nan
    vals = [v for v in (dx, dy) if np.isfinite(v) and v > 0]
    return float(np.nanmedian(vals)) if vals else 25.0


def _source_aligned_circle_check(
    sl: GridSlice,
    cx_km: float,
    cy_km: float,
    radius_cells: int,
    *,
    vel_mag_ratio: float = 4.0,
    vel_angle_diff_thresh_pi: float = 0.75,
    deviation_max_degree_pi: float = 0.1,
    dead_zone_angle_pi: float = 0.1,
    symmetric_angle_pi: float = 0.75,
) -> dict[str, float | bool]:
    spacing = _grid_spacing_km(sl)
    offsets = _circle_points(radius_cells)
    x = cx_km + np.asarray([p[0] for p in offsets], dtype="float64") * spacing
    y = cy_km + np.asarray([p[1] for p in offsets], dtype="float64") * spacing
    u = _interp2(sl.x_km, sl.y_km, sl.u_anom, x, y)
    v = _interp2(sl.x_km, sl.y_km, sl.v_anom, x, y)
    sp = np.hypot(u, v)
    finite = np.isfinite(sp) & (sp > 1e-10)
    failure_counts = {i: 0 for i in range(7)}
    if finite.mean() < 1.0:
        failure_counts[0] += int((~finite).sum())

    angles = np.arctan2(v, u)
    max_velocity_ratio = 0.0
    max_angle_diff = 0.0
    positive_direct_points = 0
    deviation_flag = 0
    angle_previous = 0.0
    rotation_failed = False
    angle_diffs: list[float] = []

    for i in range(len(offsets)):
        j = (i + 1) % len(offsets)
        if not (finite[i] and finite[j]):
            rotation_failed = True
            failure_counts[0] += 1
            continue
        ratio = abs(sp[j] / sp[i])
        max_velocity_ratio = max(max_velocity_ratio, ratio)
        if ratio > vel_mag_ratio or ratio < 1.0 / vel_mag_ratio:
            rotation_failed = True
            failure_counts[1] += 1

        angle_current = _angle_diff(float(angles[i]), float(angles[j]))
        angle_diffs.append(angle_current)
        max_angle_diff = max(max_angle_diff, abs(angle_current))
        if abs(angle_current) > math.pi * vel_angle_diff_thresh_pi:
            rotation_failed = True
            failure_counts[2] += 1

        if i == 0:
            if angle_current > 0:
                deviation_flag += 1
                positive_direct_points += 1
            angle_previous = angle_current
            continue

        if deviation_flag == 0:
            if angle_current * angle_previous < 0:
                positive_direct_points += 1
                if abs(angle_current) < deviation_max_degree_pi * math.pi:
                    deviation_flag += 1
                else:
                    rotation_failed = True
                    failure_counts[3] += 1
        elif deviation_flag <= int(math.floor(radius_cells / 5.0) + 1):
            if angle_current * angle_previous < 0:
                deviation_flag = 0
            else:
                positive_direct_points += 1
                if abs(angle_current) < deviation_max_degree_pi * math.pi:
                    deviation_flag += 1
                else:
                    rotation_failed = True
                    failure_counts[3] += 1
        else:
            if angle_current * angle_previous < 0:
                deviation_flag = 0
            else:
                positive_direct_points += 1
                rotation_failed = True
                failure_counts[4] += 1
        angle_previous = angle_current

    dead_zone_indices = [
        len(offsets) // 8,
        len(offsets) // 8 + len(offsets) // 4,
        len(offsets) // 8 + 2 * len(offsets) // 4,
        len(offsets) // 8 + 3 * len(offsets) // 4,
    ]
    dead_zone_targets = [-math.pi / 4.0, math.pi / 4.0, 3.0 * math.pi / 4.0, -3.0 * math.pi / 4.0]
    for idx, target in zip(dead_zone_indices, dead_zone_targets):
        idx = idx % len(offsets)
        if finite[idx] and abs(_angle_diff(float(angles[idx]), target)) < dead_zone_angle_pi * math.pi:
            rotation_failed = True
            failure_counts[5] += 1

    symmetry_failed = False
    half = len(offsets) // 2
    symmetry_angle = symmetric_angle_pi * math.pi
    symmetry_pass = 0
    symmetry_total = 0
    for i in range(half):
        j = (i + half) % len(offsets)
        if not (finite[i] and finite[j]):
            continue
        diff = abs(_angle_diff(float(angles[i]), float(angles[j])))
        symmetry_total += 1
        if (math.pi - symmetry_angle) < diff < (math.pi + symmetry_angle):
            symmetry_pass += 1
        else:
            symmetry_failed = True
            failure_counts[6] += 1
    if symmetry_failed:
        rotation_failed = True

    # Diagnostic tangent alignment is not a hard source criterion in the C++
    # implementation; keep it for plotting/interpretation because the paper
    # describes the same geometric idea in tangent-language.
    th = np.arctan2(y - cy_km, x - cx_km)
    tangent_x = -np.sin(th)
    tangent_y = np.cos(th)
    tang = u * tangent_x + v * tangent_y
    tangent_alignment = np.abs(tang) / np.maximum(sp, 1e-12)
    tangent_pass_fraction = float(np.nanmean(tangent_alignment[finite] >= math.cos(math.radians(24.0)))) if finite.any() else 0.0

    mean_speed = float(np.nanmean(sp[finite])) if finite.any() else np.nan
    passed = not rotation_failed
    score = (
        float(passed) * 10.0
        + radius_cells
        + tangent_pass_fraction
        - 0.1 * sum(failure_counts.values())
        - (mean_speed if np.isfinite(mean_speed) else 1.0)
    )
    dominant_failure = max(failure_counts.items(), key=lambda kv: kv[1])[0] if sum(failure_counts.values()) else -1
    return {
        "passed": bool(passed),
        "score": float(score),
        "radius_cells": float(radius_cells),
        "radius_km": float(radius_cells * spacing),
        "circulation_sign": float(-1.0 if np.nanmedian(angle_diffs) < 0 else 1.0) if angle_diffs else np.nan,
        "mean_circle_speed": mean_speed,
        "max_velocity_ratio": float(max_velocity_ratio),
        "max_angle_diff_deg": float(math.degrees(max_angle_diff)),
        "positive_direct_points": float(positive_direct_points),
        "tangent_pass_fraction": tangent_pass_fraction,
        "symmetry_pass_fraction": float(symmetry_pass / symmetry_total) if symmetry_total else 0.0,
        "finite_fraction": float(finite.mean()),
        "dominant_failure": float(dominant_failure),
        **{f"failure_{k}_count": float(v) for k, v in failure_counts.items()},
    }


def _nearest_speed(sl: GridSlice, x_km: float, y_km: float) -> float:
    return float(_interp2(sl.x_km, sl.y_km, sl.speed, np.asarray([x_km]), np.asarray([y_km]))[0])


def _speed_min_candidates(sl: GridSlice, prev_x: float, prev_y: float, search_km: float, max_candidates: int) -> list[tuple[float, float, float]]:
    xx, yy = np.meshgrid(sl.x_km, sl.y_km)
    dist = np.hypot(xx - prev_x, yy - prev_y)
    mask = (dist <= search_km) & np.isfinite(sl.speed)
    if not np.any(mask):
        return [(prev_x, prev_y, _nearest_speed(sl, prev_x, prev_y))]
    flat = np.where(mask.ravel())[0]
    order = flat[np.argsort(sl.speed.ravel()[flat])[:max_candidates]]
    cols = order % sl.speed.shape[1]
    rows = order // sl.speed.shape[1]
    return [(float(sl.x_km[c]), float(sl.y_km[r]), float(sl.speed[r, c])) for r, c in zip(rows, cols)]


def _iterative_local_speed_min(sl: GridSlice, start_x: float, start_y: float, max_iter: int = 40) -> tuple[float, float, float, int]:
    ix = int(np.argmin(np.abs(sl.x_km - start_x)))
    iy = int(np.argmin(np.abs(sl.y_km - start_y)))
    last = (-1, -1)
    steps = 0
    while (ix, iy) != last and steps < max_iter:
        last = (ix, iy)
        x0, x1 = max(0, ix - 2), min(len(sl.x_km), ix + 3)
        y0, y1 = max(0, iy - 2), min(len(sl.y_km), iy + 3)
        window = sl.speed[y0:y1, x0:x1]
        if not np.isfinite(window).any():
            break
        local = int(np.nanargmin(window))
        wy, wx = np.unravel_index(local, window.shape)
        ix = x0 + wx
        iy = y0 + wy
        steps += 1
    return float(sl.x_km[ix]), float(sl.y_km[iy]), float(sl.speed[iy, ix]), steps


def _hua_center_source_aligned(
    sl: GridSlice,
    prev_x: float,
    prev_y: float,
    *,
    start_radius_cells: int,
    max_radius_cells: int,
) -> dict[str, float | bool | str]:
    cx, cy, center_speed, min_steps = _iterative_local_speed_min(sl, prev_x, prev_y)
    best_pass: dict[str, float | bool | str] | None = None
    first_fail: dict[str, float | bool | str] | None = None
    for radius_cells in range(start_radius_cells, max_radius_cells + 1):
        score = _source_aligned_circle_check(sl, cx, cy, radius_cells)
        row: dict[str, float | bool | str] = {
            "hua_x_km": cx,
            "hua_y_km": cy,
            "hua_center_speed_ms": float(center_speed),
            "hua_min_search_steps": float(min_steps),
            **{f"hua_{k}": v for k, v in score.items()},
        }
        if bool(score["passed"]):
            best_pass = row
            continue
        first_fail = row
        break
    if best_pass is not None:
        best_pass["hua_mode"] = "source_aligned_hybrid_pass"
        best_pass["hua_accepted_radius_cells"] = max(float(best_pass["hua_radius_cells"]) - 1.0, 0.0)
        return best_pass
    if first_fail is not None:
        first_fail["hua_mode"] = "source_aligned_circle_failed"
        first_fail["hua_accepted_radius_cells"] = 0.0
        return first_fail
    return {
        "hua_x_km": cx,
        "hua_y_km": cy,
        "hua_center_speed_ms": float(center_speed),
        "hua_min_search_steps": float(min_steps),
        "hua_radius_cells": np.nan,
        "hua_radius_km": np.nan,
        "hua_passed": False,
        "hua_score": -np.inf,
        "hua_mode": "source_aligned_no_circle_test",
        "hua_accepted_radius_cells": 0.0,
    }


def _load_object_slices(
    annual_root: Path,
    clim_path: Path,
    obj: pd.DataFrame,
    *,
    half_width_km: float,
    filter_root: Path | None = None,
) -> tuple[list[GridSlice], dict[str, float | str]]:
    first = obj.sort_values("depth_index").iloc[0]
    wanted = _date(str(first["date"]))
    nc_path = annual_root / f"global_phy_{wanted.year}.nc"
    filter_path = filter_root / f"global_phy_{wanted.year}_bandpass_30_180d.nc" if filter_root else None
    filter_ds = Dataset(filter_path) if filter_path and filter_path.exists() else None
    with Dataset(nc_path) as ds, Dataset(clim_path) as clim:
        ti = _nearest_time_index(ds, wanted)
        fti = _nearest_time_index(filter_ds, wanted) if filter_ds is not None else None
        di = _doy_index(clim, wanted)
        lon = np.asarray(ds.variables["longitude"][:], dtype="float64")
        lat = np.asarray(ds.variables["latitude"][:], dtype="float64")
        depths = np.asarray(ds.variables["depth"][:], dtype="float64")
        lon0 = float(first["longitude"])
        lat0 = float(first["latitude"])
        lon_idx, lat_idx = _subset_indices(lon, lat, lon0, lat0, half_width_km)
        lon_sub = lon[lon_idx]
        lat_sub = lat[lat_idx]
        x_km = np.deg2rad(_wrap_lon_delta_deg(lon_sub, lon0)) * EARTH_RADIUS_M * math.cos(math.radians(lat0)) / 1000.0
        y_km = np.deg2rad(lat_sub - lat0) * EARTH_RADIUS_M / 1000.0
        slices: list[GridSlice] = []
        try:
            for depth_index in obj.sort_values("depth_index")["depth_index"].astype(int).unique():
                if filter_ds is not None:
                    ua = np.asarray(filter_ds.variables["uo_glor"][fti, depth_index, lat_idx.min() : lat_idx.max() + 1, lon_idx.min() : lon_idx.max() + 1], dtype="float64")
                    va = np.asarray(filter_ds.variables["vo_glor"][fti, depth_index, lat_idx.min() : lat_idx.max() + 1, lon_idx.min() : lon_idx.max() + 1], dtype="float64")
                else:
                    u = np.asarray(ds.variables["uo_glor"][ti, depth_index, lat_idx.min() : lat_idx.max() + 1, lon_idx.min() : lon_idx.max() + 1], dtype="float64")
                    v = np.asarray(ds.variables["vo_glor"][ti, depth_index, lat_idx.min() : lat_idx.max() + 1, lon_idx.min() : lon_idx.max() + 1], dtype="float64")
                    uc = np.asarray(clim.variables["u_clim"][di, depth_index, lat_idx.min() : lat_idx.max() + 1, lon_idx.min() : lon_idx.max() + 1], dtype="float64")
                    vc = np.asarray(clim.variables["v_clim"][di, depth_index, lat_idx.min() : lat_idx.max() + 1, lon_idx.min() : lon_idx.max() + 1], dtype="float64")
                    ua = u - uc
                    va = v - vc
                speed = np.hypot(ua, va)
                slices.append(GridSlice(depth_index, float(depths[depth_index]), x_km, y_km, ua, va, speed))
        finally:
            if filter_ds is not None:
                filter_ds.close()
    meta = {
        "date": wanted.isoformat(),
        "surface_longitude": float(first["longitude"]),
        "surface_latitude": float(first["latitude"]),
        "annual_file": str(nc_path),
        "climatology_file": str(clim_path),
        "filter_file": str(filter_path) if filter_path and filter_path.exists() else "",
        "velocity_anomaly_definition": "bandpass_30_180d_filter" if filter_path and filter_path.exists() else "raw_minus_doy_climatology",
    }
    return slices, meta


def _plot_pages(df: pd.DataFrame, slices: list[GridSlice], out_pdf: Path, out_pages_prefix: Path, title: str, panels_per_page: int = 12) -> None:
    by_depth = {sl.depth_index: sl for sl in slices}
    depths = list(df["depth_index"].astype(int))
    vmax = float(np.nanpercentile([np.nanpercentile(sl.speed, 97) for sl in slices], 95))
    vmax = max(vmax, 1e-6)
    with PdfPages(out_pdf) as pdf:
        for page, start in enumerate(range(0, len(depths), panels_per_page), start=1):
            part_depths = depths[start : start + panels_per_page]
            ncols = 4
            nrows = int(math.ceil(len(part_depths) / ncols))
            fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4.2 * nrows), squeeze=False)
            for ax in axes.ravel():
                ax.axis("off")
            for ax, depth_index in zip(axes.ravel(), part_depths):
                row = df[df["depth_index"].eq(depth_index)].iloc[0]
                sl = by_depth[depth_index]
                ax.axis("on")
                im = ax.pcolormesh(sl.x_km, sl.y_km, sl.speed, shading="auto", cmap="magma", vmin=0.0, vmax=vmax)
                step = max(1, len(sl.x_km) // 18)
                ax.quiver(sl.x_km[::step], sl.y_km[::step], sl.u_anom[::step, ::step], sl.v_anom[::step, ::step], color="white", alpha=0.55, scale=2.2, width=0.002)
                ax.scatter([0], [0], marker="+", s=120, linewidth=2.5, color="#06b6d4", label="SLA seed")
                ax.scatter(row["current_x_km"], row["current_y_km"], marker="x", s=75, linewidth=2.0, color="#22d3ee", label="current")
                ax.scatter(row["speed_x_km"], row["speed_y_km"], marker="o", s=58, facecolor="#f97316", edgecolor="black", linewidth=1.2, label="speed min")
                edge = "#22c55e" if bool(row["hua_passed"]) else "#ef4444"
                ax.scatter(row["hua_x_km"], row["hua_y_km"], marker="*", s=115, facecolor="#fde047", edgecolor=edge, linewidth=1.4, label="Hua hybrid")
                ax.scatter(row["recommended_x_km"], row["recommended_y_km"], marker="s", s=56, facecolor="none", edgecolor="#a855f7", linewidth=1.6, label="recommended")
                circ = plt.Circle((row["hua_x_km"], row["hua_y_km"]), row["hua_radius_km"], color=edge, fill=False, linewidth=1.3, alpha=0.9)
                ax.add_patch(circ)
                ax.plot([row["current_x_km"], row["hua_x_km"]], [row["current_y_km"], row["hua_y_km"]], color="white", linewidth=1.0, alpha=0.55)
                ax.plot([row["hua_x_km"], row["recommended_x_km"]], [row["hua_y_km"], row["recommended_y_km"]], color="#a855f7", linewidth=1.0, alpha=0.85)
                ax.set_xlim(float(np.nanmin(sl.x_km)), float(np.nanmax(sl.x_km)))
                ax.set_ylim(float(np.nanmin(sl.y_km)), float(np.nanmax(sl.y_km)))
                ax.set_aspect("equal")
                ax.grid(color="white", alpha=0.18, linewidth=0.5)
                ax.set_title(
                    f"L{depth_index:02d} {sl.depth_m:.0f} m | Hua {row['hua_mode']}\n"
                    f"cur {row['current_speed_ms']:.3f}, speed {row['speed_center_speed_ms']:.3f}, "
                    f"Hua {row['hua_center_speed_ms']:.3f}; pass={bool(row['hua_passed'])}; {row['production_action']}",
                    fontsize=8,
                )
            handles, labels = axes.ravel()[0].get_legend_handles_labels()
            fig.legend(handles, labels, loc="upper right", fontsize=8)
            cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.72, pad=0.015)
            cbar.set_label("|u',v'| (m/s)")
            fig.suptitle(f"{title} - layer audit page {page}", fontsize=14)
            pdf.savefig(fig, bbox_inches="tight")
            fig.savefig(out_pages_prefix.with_name(f"{out_pages_prefix.name}_page_{page:02d}.png"), dpi=190, bbox_inches="tight")
            plt.close(fig)


def _plot_axis_3d(df_all: pd.DataFrame, output: Path) -> None:
    fig = plt.figure(figsize=(15, 7))
    for panel, (obj_id, df) in enumerate(df_all.groupby("eddy3d_object_id"), start=1):
        ax = fig.add_subplot(1, 2, panel, projection="3d")
        z = -df["depth_m"].to_numpy(dtype="float64") / 1000.0
        for prefix, color, label, marker in [
            ("current", "#06b6d4", "current centers", "x"),
            ("speed", "#f97316", "u',v' speed minima", "o"),
            ("hua", "#eab308", "Hua hybrid centers", "*"),
            ("recommended", "#a855f7", "production recommended", "s"),
        ]:
            x = df[f"{prefix}_x_km"].to_numpy(dtype="float64")
            y = df[f"{prefix}_y_km"].to_numpy(dtype="float64")
            ax.plot(x, y, z, color=color, linewidth=2.2, label=label)
            ax.scatter(x, y, z, color=color, marker=marker, s=28)
        row = df.iloc[0]
        ax.scatter([0], [0], [z[0]], marker="+", s=140, color="red", linewidth=3, label="SLA seed")
        ax.set_title(f"{row['shape_class']} id {obj_id}\n{row['polarity']}, {row['date']}")
        ax.set_xlabel("east x from SLA seed (km)")
        ax.set_ylabel("north y from SLA seed (km)")
        ax.set_zlabel("depth (km, down)")
        ax.set_xlim(-230, 230)
        ax.set_ylim(-230, 230)
        ax.set_zlim(-3.1, 0.05)
        ax.view_init(elev=22, azim=-55)
        ax.legend(fontsize=8, loc="upper left")
    fig.suptitle("ACC current vs anomaly speed-leading vs Hua-style hybrid axes", fontsize=15)
    fig.savefig(output.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _object_experiment(args: argparse.Namespace, obj: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float | str]]:
    obj = obj.sort_values("depth_index").copy()
    filter_root = Path(args.filter_root) if args.filter_root else None
    slices, meta = _load_object_slices(Path(args.annual_root), Path(args.climatology), obj, half_width_km=args.window_km, filter_root=filter_root)
    rows = []
    prev_hua_x = 0.0
    prev_hua_y = 0.0
    prev_recommended_x = 0.0
    prev_recommended_y = 0.0
    for sl in slices:
        current = obj[obj["depth_index"].eq(sl.depth_index)].iloc[0]
        current_x = float(current["x_m"]) / 1000.0
        current_y = float(current["y_m"]) / 1000.0
        speed_candidates = _speed_min_candidates(sl, prev_hua_x, prev_hua_y, args.deep_search_km if sl.depth_index > 0 else args.surface_search_km, 1)
        speed_x, speed_y, speed_center_speed = speed_candidates[0]
        hua = _hua_center_source_aligned(
            sl,
            prev_recommended_x,
            prev_recommended_y,
            start_radius_cells=args.start_radius_cells,
            max_radius_cells=args.max_radius_cells,
        )
        if bool(hua.get("hua_passed", False)):
            recommended_x = float(hua["hua_x_km"])
            recommended_y = float(hua["hua_y_km"])
            production_action = "accept_hua_center"
            production_confidence = 1.0
        else:
            candidate_jump = float(np.hypot(float(hua["hua_x_km"]) - prev_recommended_x, float(hua["hua_y_km"]) - prev_recommended_y))
            current_jump = float(np.hypot(current_x - prev_recommended_x, current_y - prev_recommended_y))
            if sl.depth_index == 0 or current_jump <= args.continuity_keep_km:
                recommended_x = current_x
                recommended_y = current_y
                production_action = "keep_current_center_low_confidence"
            else:
                recommended_x = prev_recommended_x
                recommended_y = prev_recommended_y
                production_action = "carry_previous_center_low_confidence"
            production_confidence = max(0.0, 0.45 - 0.002 * candidate_jump)
        if bool(hua["hua_passed"]):
            prev_hua_x = float(hua["hua_x_km"])
            prev_hua_y = float(hua["hua_y_km"])
        else:
            # Keep continuity, but still report the best failed candidate.
            prev_hua_x = float(hua["hua_x_km"])
            prev_hua_y = float(hua["hua_y_km"])
        prev_recommended_x = recommended_x
        prev_recommended_y = recommended_y
        rows.append(
            {
                "eddy3d_object_id": int(current["eddy3d_object_id"]),
                "track3d_id": int(current["track3d_id"]),
                "shape_class": str(current["shape_class"]),
                "polarity": str(current["polarity"]),
                "date": str(current["date"])[:10],
                "depth_index": int(sl.depth_index),
                "depth_m": float(sl.depth_m),
                "current_x_km": current_x,
                "current_y_km": current_y,
                "current_speed_ms": _nearest_speed(sl, current_x, current_y),
                "speed_x_km": speed_x,
                "speed_y_km": speed_y,
                "speed_center_speed_ms": speed_center_speed,
                "recommended_x_km": recommended_x,
                "recommended_y_km": recommended_y,
                "recommended_speed_ms": _nearest_speed(sl, recommended_x, recommended_y),
                "production_action": production_action,
                "production_confidence": production_confidence,
                **hua,
            }
        )
    out = pd.DataFrame(rows)
    out["current_to_hua_km"] = np.hypot(out["current_x_km"] - out["hua_x_km"], out["current_y_km"] - out["hua_y_km"])
    out["speed_to_hua_km"] = np.hypot(out["speed_x_km"] - out["hua_x_km"], out["speed_y_km"] - out["hua_y_km"])
    out["current_to_speed_km"] = np.hypot(out["current_x_km"] - out["speed_x_km"], out["current_y_km"] - out["speed_y_km"])
    out["current_to_recommended_km"] = np.hypot(out["current_x_km"] - out["recommended_x_km"], out["current_y_km"] - out["recommended_y_km"])
    out["hua_to_recommended_km"] = np.hypot(out["hua_x_km"] - out["recommended_x_km"], out["hua_y_km"] - out["recommended_y_km"])
    return out, meta


def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    filter_root = Path(args.filter_root) if args.filter_root else None
    axis = _axis_rows(Path(args.axis_path), tuple(int(x) for x in args.object_ids.split(",")))
    if axis.empty:
        raise ValueError("No requested objects found in axis file")
    summaries = []
    all_rows = []
    for object_id, obj in axis.groupby("eddy3d_object_id"):
        result, meta = _object_experiment(args, obj)
        all_rows.append(result)
        result.to_parquet(output_dir / f"hua_hybrid_center_audit_id_{int(object_id)}.parquet", index=False)
        result.to_csv(output_dir / f"hua_hybrid_center_audit_id_{int(object_id)}.csv", index=False)
        title = f"Hua-style hybrid velocity-path audit: {result.iloc[0]['shape_class']} id {int(object_id)}"
        _plot_pages(
            result,
            _load_object_slices(Path(args.annual_root), Path(args.climatology), obj, half_width_km=args.window_km, filter_root=filter_root)[0],
            output_dir / f"hua_hybrid_layer_audit_id_{int(object_id)}.pdf",
            output_dir / f"hua_hybrid_layer_audit_id_{int(object_id)}",
            title,
        )
        summaries.append(
            {
                "eddy3d_object_id": int(object_id),
                **meta,
                "shape_class": str(result.iloc[0]["shape_class"]),
                "polarity": str(result.iloc[0]["polarity"]),
                "n_layers": int(len(result)),
                "hua_pass_layers": int(result["hua_passed"].sum()),
                "hua_pass_fraction": float(result["hua_passed"].mean()),
                "current_speed_median_ms": float(result["current_speed_ms"].median()),
                "speed_min_median_ms": float(result["speed_center_speed_ms"].median()),
                "hua_speed_median_ms": float(result["hua_center_speed_ms"].median()),
                "recommended_speed_median_ms": float(result["recommended_speed_ms"].median()),
                "current_to_hua_median_km": float(result["current_to_hua_km"].median()),
                "current_to_hua_p90_km": float(result["current_to_hua_km"].quantile(0.9)),
                "current_to_speed_median_km": float(result["current_to_speed_km"].median()),
                "current_to_recommended_median_km": float(result["current_to_recommended_km"].median()),
                "accept_hua_layers": int(result["production_action"].eq("accept_hua_center").sum()),
                "keep_current_low_conf_layers": int(result["production_action"].eq("keep_current_center_low_confidence").sum()),
                "carry_previous_low_conf_layers": int(result["production_action"].eq("carry_previous_center_low_confidence").sum()),
                "failure_0_invalid_velocity": int(result.get("hua_failure_0_count", pd.Series(dtype=float)).sum()),
                "failure_1_velocity_ratio": int(result.get("hua_failure_1_count", pd.Series(dtype=float)).sum()),
                "failure_2_angle_jump": int(result.get("hua_failure_2_count", pd.Series(dtype=float)).sum()),
                "failure_3_large_direction_deviation": int(result.get("hua_failure_3_count", pd.Series(dtype=float)).sum()),
                "failure_4_too_many_deviations": int(result.get("hua_failure_4_count", pd.Series(dtype=float)).sum()),
                "failure_5_dead_zone": int(result.get("hua_failure_5_count", pd.Series(dtype=float)).sum()),
                "failure_6_symmetry": int(result.get("hua_failure_6_count", pd.Series(dtype=float)).sum()),
            }
        )
    combined = pd.concat(all_rows, ignore_index=True)
    combined.to_csv(output_dir / "hua_hybrid_center_audit_all.csv", index=False)
    combined.to_parquet(output_dir / "hua_hybrid_center_audit_all.parquet", index=False)
    pd.DataFrame(summaries).to_csv(output_dir / "hua_hybrid_center_summary.csv", index=False)
    (output_dir / "hua_hybrid_center_summary.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    _plot_axis_3d(combined, output_dir / "hua_hybrid_axis_comparison_3d")
    report = [
        "# Hua SSH-velocity hybrid 对 ACC 单对象中心的实验性影响",
        "",
        "本实验不修改主识别流程。它以现有 SLA/SSH 表层中心为种子，在每个深度层使用 30-180 天带通后的扰动速度 `u'=u_{30-180d}, v'=v_{30-180d}`，先找局部速度低值，再按 Hua et al. 2023 的思想在候选中心周围圆周路径上检查旋转一致性、相邻速度比、切向性和对心对称性。",
        "",
        "图中 `Hua hybrid` 星号外圈为绿色表示圆周检验通过，红色表示该层速度低值存在但圆周旋转/对称性检验未通过。该实验用于判断 Hua 规则会如何改变深层中心，不代表最终生产结果。",
        "",
        "## 数值摘要",
    ]
    for item in summaries:
        report.append(
            f"- `{item['shape_class']}` object `{item['eddy3d_object_id']}`："
            f"Hua 通过 `{item['hua_pass_layers']}/{item['n_layers']}` 层；"
            f"当前中心 median |u'|={item['current_speed_median_ms']:.4f} m/s，"
            f"纯速度低值 median |u'|={item['speed_min_median_ms']:.4f} m/s，"
            f"Hua 候选 median |u'|={item['hua_speed_median_ms']:.4f} m/s；"
            f"生产推荐采纳 Hua `{item['accept_hua_layers']}` 层，"
            f"当前-Hua median 距离={item['current_to_hua_median_km']:.1f} km。"
        )
    report.extend(
        [
            "",
            "## 技术口径",
            "",
            "- Hua 原文是 `SSH extrema -> nearby velocity minimum -> circular velocity-path verification -> outward/deeper expansion`。",
            "- 本版按开源 C++ 的核心逻辑移植：5x5 迭代局部速度极小搜索，从 `STARTRADIUS=3` 开始逐圈圆周检查，首次失败时以上一圈为可接受边界。",
            "- 圆周失败原因按源码语义记录：0 无效/零速度，1 相邻速度比过大，2 速度方向角跳变过大，3 单次方向偏离过大，4 方向偏离次数过多，5 dead-zone，6 对心速度对称失败。",
            "- 这里没有直接编译 C++/VTK 程序，而是在 ACC 的 annual NetCDF + climatology 口径下做 Python 审计移植。",
            "- 若 Hua 通过，生产建议采用 Hua center；若不通过，生产建议不会跳到候选速度低值，而是保留当前中心或上一层连续中心，并降低置信度。",
        ]
    )
    (output_dir / "hua_hybrid_center_interpretation_zh.md").write_text("\n".join(report), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Experimental Hua 2023 SSH-velocity hybrid center audit for ACC objects.")
    parser.add_argument("--axis-path", default="/root/autodl-fs/2020_2022_acc/result/representative_vortex/axis/rotated_points.parquet")
    parser.add_argument("--annual-root", default="/root/autodl-fs/2020_2022_acc")
    parser.add_argument("--climatology", default="/root/autodl-fs/2020_2022_acc/result/climatology/cmems_doy_climatology_2020_2022_31d.nc")
    parser.add_argument("--filter-root", default="/root/autodl-fs/2020_2022_acc/Filter")
    parser.add_argument("--output-dir", default="/root/autodl-fs/2020_2022_acc/result/hua_hybrid_center_experiment")
    parser.add_argument("--object-ids", default="1120461,1076249")
    parser.add_argument("--window-km", type=float, default=230.0)
    parser.add_argument("--surface-search-km", type=float, default=95.0)
    parser.add_argument("--deep-search-km", type=float, default=65.0)
    parser.add_argument("--start-radius-cells", type=int, default=3)
    parser.add_argument("--max-radius-cells", type=int, default=8)
    parser.add_argument("--continuity-keep-km", type=float, default=45.0)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
