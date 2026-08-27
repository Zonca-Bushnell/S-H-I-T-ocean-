from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from netCDF4 import Dataset, num2date
from scipy import ndimage


EARTH_RADIUS_M = 6_371_000.0
FAILURE_LABELS = {
    0: "invalid_velocity",
    1: "velocity_ratio",
    2: "angle_jump",
    3: "rotation_direction",
    4: "too_many_direction_exceptions",
    5: "dead_zone",
    6: "symmetry",
    7: "tangent_alignment",
    8: "opposite_reversal",
    9: "boundary_monotonic_rotation",
}
OBJECT_VOXEL_COLUMNS = [
    "date",
    "hua_object_id",
    "depth_index",
    "i",
    "j",
    "lon",
    "lat",
    "depth_m",
    "polarity",
    "accepted_radius_cells",
    "node_key_3d",
    "node_key_2d",
]


@dataclass(frozen=True)
class DetectionParams:
    ssh_window_cells: int
    start_radius_cells: int
    max_radius_cells: int
    speed_ratio_max: float
    angle_jump_max_deg: float
    tangent_tolerance_deg: float
    symmetry_tolerance_deg: float
    min_tangent_fraction: float
    min_reversal_fraction: float
    min_finite_fraction: float
    direction_exception_extra: int
    surface_search_cells: int
    deep_search_cells: int
    require_boundary_monotonic_rotation: bool = False
    boundary_monotonic_exception_limit: int = 0


def _parse_date(value: str) -> date:
    return datetime.strptime(str(value)[:10], "%Y-%m-%d").date()


def _date_range(start: date, end: date) -> list[date]:
    days: list[date] = []
    current = start
    while current <= end:
        days.append(current)
        current += timedelta(days=1)
    return days


def _wrap_lon_delta_deg(lon: np.ndarray, lon0: float) -> np.ndarray:
    return (lon - lon0 + 180.0) % 360.0 - 180.0


def _time_lookup(ds: Dataset) -> dict[date, int]:
    tvar = ds.variables["time"]
    times = num2date(tvar[:], units=tvar.units, calendar=getattr(tvar, "calendar", "standard"))
    return {date(int(t.year), int(t.month), int(t.day)): i for i, t in enumerate(times)}


def _grid_spacing_km(lon: np.ndarray, lat: np.ndarray) -> tuple[float, float]:
    mid_lat = float(np.nanmedian(lat))
    dx = np.deg2rad(float(np.nanmedian(np.abs(np.diff(lon))))) * EARTH_RADIUS_M * math.cos(math.radians(mid_lat)) / 1000.0
    dy = np.deg2rad(float(np.nanmedian(np.abs(np.diff(lat))))) * EARTH_RADIUS_M / 1000.0
    return abs(dx), abs(dy)


def _local_extrema(zos: np.ndarray, window: int, *, max_candidates: int) -> pd.DataFrame:
    finite = np.isfinite(zos)
    fill_max = np.where(finite, zos, -np.inf)
    fill_min = np.where(finite, zos, np.inf)
    max_mask = finite & (fill_max == ndimage.maximum_filter(fill_max, size=window, mode="nearest"))
    min_mask = finite & (fill_min == ndimage.minimum_filter(fill_min, size=window, mode="nearest"))
    rows = []
    structure = np.ones((3, 3), dtype=bool)
    for kind, mask in (("ssh_max", max_mask), ("ssh_min", min_mask)):
        labels, count = ndimage.label(mask, structure=structure)
        for label in range(1, count + 1):
            yy, xx = np.where(labels == label)
            n = len(xx)
            if n == 0 or n > 100:
                continue
            values = zos[yy, xx]
            if kind == "ssh_max":
                pick = int(np.nanargmax(values))
            else:
                pick = int(np.nanargmin(values))
            rows.append(
                {
                    "ssh_extremum_type": kind,
                    "seed_i": int(xx[pick]),
                    "seed_j": int(yy[pick]),
                    "ssh_value_m": float(values[pick]),
                    "component_pixels": int(n),
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["abs_ssh_value_m"] = out["ssh_value_m"].abs()
    out = out.sort_values("abs_ssh_value_m", ascending=False).reset_index(drop=True)
    if max_candidates > 0:
        out = out.head(max_candidates).copy()
    return out


def _circle_offsets(radius_cells: int) -> list[tuple[int, int]]:
    points: list[tuple[int, int]] = []
    n = max(16, int(round(8 * radius_cells)))
    for theta in np.linspace(-math.pi / 2.0, 3.0 * math.pi / 2.0, n, endpoint=False):
        point = (int(round(radius_cells * math.cos(theta))), int(round(radius_cells * math.sin(theta))))
        if not points or points[-1] != point:
            points.append(point)
    if len(points) > 1 and points[0] == points[-1]:
        points.pop()
    return points


def _angle_diff(a: np.ndarray | float, b: np.ndarray | float) -> np.ndarray | float:
    return (np.asarray(a) - np.asarray(b) + math.pi) % (2.0 * math.pi) - math.pi


def _iterative_speed_min(speed: np.ndarray, start_i: int, start_j: int, *, max_steps: int = 40) -> tuple[int, int, float, int]:
    ii = int(np.clip(start_i, 0, speed.shape[1] - 1))
    jj = int(np.clip(start_j, 0, speed.shape[0] - 1))
    last = (-1, -1)
    steps = 0
    while (ii, jj) != last and steps < max_steps:
        last = (ii, jj)
        x0, x1 = max(0, ii - 2), min(speed.shape[1], ii + 3)
        y0, y1 = max(0, jj - 2), min(speed.shape[0], jj + 3)
        window = speed[y0:y1, x0:x1]
        if not np.isfinite(window).any():
            break
        local = int(np.nanargmin(window))
        wy, wx = np.unravel_index(local, window.shape)
        ii = x0 + wx
        jj = y0 + wy
        steps += 1
    val = float(speed[jj, ii]) if np.isfinite(speed[jj, ii]) else np.nan
    return ii, jj, val, steps


def _seeded_speed_min(speed: np.ndarray, seed_i: int, seed_j: int, radius_cells: int) -> tuple[int, int, float, int]:
    yy, xx = np.ogrid[: speed.shape[0], : speed.shape[1]]
    mask = (xx - seed_i) ** 2 + (yy - seed_j) ** 2 <= radius_cells**2
    mask &= np.isfinite(speed)
    if not np.any(mask):
        return _iterative_speed_min(speed, seed_i, seed_j)
    flat = np.where(mask.ravel())[0]
    pick = int(flat[np.nanargmin(speed.ravel()[flat])])
    jj, ii = np.unravel_index(pick, speed.shape)
    return _iterative_speed_min(speed, int(ii), int(jj))


def _circle_check(
    u: np.ndarray,
    v: np.ndarray,
    center_i: int,
    center_j: int,
    radius_cells: int,
    params: DetectionParams,
) -> dict[str, float | bool | str]:
    offsets = _circle_offsets(radius_cells)
    ii = np.asarray([center_i + dx for dx, _ in offsets], dtype=int)
    jj = np.asarray([center_j + dy for _, dy in offsets], dtype=int)
    inside = (ii >= 0) & (ii < u.shape[1]) & (jj >= 0) & (jj < u.shape[0])
    uu = np.full(len(offsets), np.nan, dtype="float64")
    vv = np.full(len(offsets), np.nan, dtype="float64")
    uu[inside] = u[jj[inside], ii[inside]]
    vv[inside] = v[jj[inside], ii[inside]]
    sp = np.hypot(uu, vv)
    finite = np.isfinite(sp) & (sp > 1e-10)
    failure_counts = {k: 0 for k in FAILURE_LABELS}
    if finite.mean() < params.min_finite_fraction:
        failure_counts[0] += int((~finite).sum())

    angles = np.arctan2(vv, uu)
    angle_diffs: list[float] = []
    max_ratio = 0.0
    max_angle = 0.0
    positive_diffs = 0
    negative_diffs = 0
    rotation_failed = False
    for n in range(len(offsets)):
        m = (n + 1) % len(offsets)
        if not (finite[n] and finite[m]):
            rotation_failed = True
            failure_counts[0] += 1
            continue
        ratio = float(sp[m] / sp[n])
        max_ratio = max(max_ratio, ratio, 1.0 / ratio if ratio > 0 else np.inf)
        if ratio > params.speed_ratio_max or ratio < 1.0 / params.speed_ratio_max:
            rotation_failed = True
            failure_counts[1] += 1
        dtheta = float(_angle_diff(angles[n], angles[m]))
        angle_diffs.append(dtheta)
        max_angle = max(max_angle, abs(math.degrees(dtheta)))
        if abs(math.degrees(dtheta)) > params.angle_jump_max_deg:
            rotation_failed = True
            failure_counts[2] += 1
        if dtheta > 0:
            positive_diffs += 1
        elif dtheta < 0:
            negative_diffs += 1
    max_exceptions = int(math.floor(radius_cells / 5.0) + 1 + params.direction_exception_extra)
    direction_exceptions = min(positive_diffs, negative_diffs)
    monotonic_exception_limit = (
        int(params.boundary_monotonic_exception_limit)
        if params.require_boundary_monotonic_rotation
        else max_exceptions
    )
    boundary_monotonic_passed = direction_exceptions <= monotonic_exception_limit
    if direction_exceptions > max_exceptions:
        rotation_failed = True
        failure_counts[4] += int(direction_exceptions - max_exceptions)
    if params.require_boundary_monotonic_rotation and not boundary_monotonic_passed:
        rotation_failed = True
        failure_counts[9] += int(direction_exceptions - monotonic_exception_limit)

    dx = np.asarray([p[0] for p in offsets], dtype="float64")
    dy = np.asarray([p[1] for p in offsets], dtype="float64")
    th = np.arctan2(dy, dx)
    tx = -np.sin(th)
    ty = np.cos(th)
    tangent_cos = np.abs((uu * tx + vv * ty) / np.maximum(sp, 1e-12))
    tangent_ok = finite & (tangent_cos >= math.cos(math.radians(params.tangent_tolerance_deg)))
    tangent_fraction = float(tangent_ok.sum() / finite.sum()) if finite.any() else 0.0
    if tangent_fraction < params.min_tangent_fraction:
        rotation_failed = True
        failure_counts[7] += int(max(1, round((params.min_tangent_fraction - tangent_fraction) * len(offsets))))

    half = len(offsets) // 2
    symmetry_ok = 0
    symmetry_total = 0
    reversal_ok = 0
    reversal_total = 0
    for n in range(half):
        m = (n + half) % len(offsets)
        if not (finite[n] and finite[m]):
            continue
        diff = abs(float(_angle_diff(angles[n], angles[m])))
        symmetry_total += 1
        if abs(diff - math.pi) <= math.radians(params.symmetry_tolerance_deg):
            symmetry_ok += 1
        reversal_total += 1
        if uu[n] * uu[m] + vv[n] * vv[m] < 0:
            reversal_ok += 1
    symmetry_fraction = float(symmetry_ok / symmetry_total) if symmetry_total else 0.0
    reversal_fraction = float(reversal_ok / reversal_total) if reversal_total else 0.0
    if symmetry_total and symmetry_ok < symmetry_total:
        failure_counts[6] += int(symmetry_total - symmetry_ok)
    if reversal_fraction < params.min_reversal_fraction:
        rotation_failed = True
        failure_counts[8] += int(max(1, round((params.min_reversal_fraction - reversal_fraction) * max(reversal_total, 1))))

    tangential = uu * tx + vv * ty
    circulation_sign = float(np.sign(np.nanmedian(tangential[finite]))) if finite.any() else np.nan
    dominant = max(failure_counts.items(), key=lambda kv: kv[1])[0] if sum(failure_counts.values()) else -1
    return {
        "circle_passed": bool(not rotation_failed),
        "radius_cells": float(radius_cells),
        "finite_fraction": float(finite.mean()),
        "mean_circle_speed_ms": float(np.nanmean(sp[finite])) if finite.any() else np.nan,
        "max_velocity_ratio": float(max_ratio),
        "max_angle_jump_deg": float(max_angle),
        "direction_exception_count": float(direction_exceptions),
        "positive_angle_diff_count": float(positive_diffs),
        "negative_angle_diff_count": float(negative_diffs),
        "direction_exception_limit": float(max_exceptions),
        "boundary_monotonic_required": bool(params.require_boundary_monotonic_rotation),
        "boundary_monotonic_passed": bool(boundary_monotonic_passed),
        "boundary_monotonic_exception_limit": float(monotonic_exception_limit),
        "tangent_pass_fraction": tangent_fraction,
        "symmetry_pass_fraction": symmetry_fraction,
        "opposite_reversal_fraction": reversal_fraction,
        "circulation_sign": circulation_sign,
        "dominant_failure_code": float(dominant),
        "dominant_failure": FAILURE_LABELS.get(int(dominant), "none") if dominant >= 0 else "none",
        **{f"failure_{k}_{label}_count": float(failure_counts[k]) for k, label in FAILURE_LABELS.items()},
    }


def _hua_verify_radius(
    u: np.ndarray,
    v: np.ndarray,
    center_i: int,
    center_j: int,
    params: DetectionParams,
) -> dict[str, float | bool | str]:
    best: dict[str, float | bool | str] | None = None
    first_fail: dict[str, float | bool | str] | None = None
    for radius in range(params.start_radius_cells, params.max_radius_cells + 1):
        row = _circle_check(u, v, center_i, center_j, radius, params)
        if bool(row["circle_passed"]):
            best = row
        else:
            first_fail = row
            break
    source_row = best if best is not None else first_fail
    if source_row is None:
        source_row = {"circle_passed": False, "radius_cells": np.nan, "dominant_failure": "no_circle"}
    source_row = dict(source_row)
    source_row["hua_pass"] = bool(best is not None)
    source_row["accepted_radius_cells"] = float(source_row["radius_cells"]) if best is not None else 0.0
    return source_row


def _object_voxels_for_layer(
    u: np.ndarray,
    v: np.ndarray,
    lon: np.ndarray,
    lat: np.ndarray,
    depth_m: float,
    *,
    day: date,
    object_id: str,
    depth_index: int,
    center_i: int,
    center_j: int,
    radius_cells: float,
    polarity: str,
) -> list[dict[str, object]]:
    """Return the finite component connected to the Hua center inside its accepted circle."""
    if not np.isfinite(radius_cells) or radius_cells <= 0:
        return []
    radius = int(math.ceil(float(radius_cells)))
    x0, x1 = max(0, center_i - radius), min(u.shape[1], center_i + radius + 1)
    y0, y1 = max(0, center_j - radius), min(u.shape[0], center_j + radius + 1)
    if x0 >= x1 or y0 >= y1:
        return []

    yy, xx = np.ogrid[y0:y1, x0:x1]
    circle = (xx - center_i) ** 2 + (yy - center_j) ** 2 <= float(radius_cells) ** 2
    finite = np.isfinite(u[y0:y1, x0:x1]) & np.isfinite(v[y0:y1, x0:x1])
    mask = circle & finite
    cj = center_j - y0
    ci = center_i - x0
    if cj < 0 or cj >= mask.shape[0] or ci < 0 or ci >= mask.shape[1] or not mask[cj, ci]:
        return []
    labels, _ = ndimage.label(mask, structure=np.ones((3, 3), dtype=bool))
    component_label = int(labels[cj, ci])
    if component_label <= 0:
        return []
    comp_y, comp_x = np.where(labels == component_label)
    rows: list[dict[str, object]] = []
    ny, nx = u.shape
    for ly, lx in zip(comp_y.tolist(), comp_x.tolist()):
        jj = int(y0 + ly)
        ii = int(x0 + lx)
        rows.append(
            {
                "date": day.isoformat(),
                "hua_object_id": object_id,
                "depth_index": int(depth_index),
                "i": ii,
                "j": jj,
                "lon": float(lon[ii]),
                "lat": float(lat[jj]),
                "depth_m": float(depth_m),
                "polarity": polarity,
                "accepted_radius_cells": float(radius_cells),
                "node_key_3d": int(depth_index * ny * nx + jj * nx + ii),
                "node_key_2d": int(jj * nx + ii),
            }
        )
    return rows


def _extremum_polarity(extremum: str, lat_value: float, circulation_sign: float) -> str:
    if np.isfinite(circulation_sign) and circulation_sign != 0:
        # Positive tangential circulation is counterclockwise. In the Southern
        # Hemisphere cyclonic rotation is clockwise, so use f sign.
        f_sign = 1.0 if lat_value >= 0 else -1.0
        return "cyclonic" if circulation_sign == f_sign else "anticyclonic"
    if lat_value < 0:
        return "cyclonic" if extremum == "ssh_min" else "anticyclonic"
    return "cyclonic" if extremum == "ssh_min" else "anticyclonic"


def _format_year_template(template: str, year: int) -> str:
    return str(template).format(year=year)


def _load_year_arrays(args: argparse.Namespace, year: int) -> tuple[Dataset, Dataset | None, np.ndarray, np.ndarray, np.ndarray]:
    filter_root = Path(args.filter_root)
    raw_root = Path(args.raw_root)
    filt = Dataset(filter_root / _format_year_template(args.filter_template, year))
    raw_path = raw_root / _format_year_template(args.raw_template, year)
    raw = Dataset(raw_path) if raw_path.exists() else None
    lon = np.asarray(filt.variables["longitude"][:], dtype="float64")
    lat = np.asarray(filt.variables["latitude"][:], dtype="float64")
    depth = np.asarray(filt.variables["depth"][:], dtype="float64")
    return filt, raw, lon, lat, depth


def _detect_day(
    day: date,
    filt: Dataset,
    raw: Dataset | None,
    lon: np.ndarray,
    lat: np.ndarray,
    depth: np.ndarray,
    params: DetectionParams,
    args: argparse.Namespace,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    time_index = _time_lookup(filt)[day]
    zos = np.asarray(filt.variables["zos_glor"][time_index, :, :], dtype="float64")
    extrema = _local_extrema(zos, params.ssh_window_cells, max_candidates=args.max_candidates_per_day)
    if extrema.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(columns=OBJECT_VOXEL_COLUMNS)

    max_depth_idx = int(np.searchsorted(depth, args.max_depth_m, side="right"))
    max_depth_idx = min(max_depth_idx, len(depth))
    depth_indices = np.arange(max_depth_idx, dtype=int)
    dx_km, dy_km = _grid_spacing_km(lon, lat)
    centers_rows: list[dict[str, object]] = []
    circle_rows: list[dict[str, object]] = []
    structure_rows: list[dict[str, object]] = []
    voxel_rows: list[dict[str, object]] = []

    u_day = None
    v_day = None
    if args.preload_day_uv:
        # ACC windows are small enough that loading one day of u/v once is
        # cheaper than thousands of HDF5 layer reads during candidate checks.
        u_day = np.asarray(filt.variables["uo_glor"][time_index, :max_depth_idx, :, :], dtype="float32")
        v_day = np.asarray(filt.variables["vo_glor"][time_index, :max_depth_idx, :, :], dtype="float32")

    if u_day is None:
        u0 = np.asarray(filt.variables["uo_glor"][time_index, 0, :, :], dtype="float32")
        v0 = np.asarray(filt.variables["vo_glor"][time_index, 0, :, :], dtype="float32")
    else:
        u0 = u_day[0]
        v0 = v_day[0]
    speed0 = np.hypot(u0, v0)
    for seed_order, seed in extrema.iterrows():
        seed_i = int(seed["seed_i"])
        seed_j = int(seed["seed_j"])
        center_i, center_j, center_speed, min_steps = _seeded_speed_min(
            speed0,
            seed_i,
            seed_j,
            params.surface_search_cells,
        )
        prev_i, prev_j = center_i, center_j
        object_id = f"{day:%Y%m%d}_{int(seed_order):05d}"
        stopped = False
        for depth_index in depth_indices:
            if u_day is None:
                u = np.asarray(filt.variables["uo_glor"][time_index, depth_index, :, :], dtype="float32")
                v = np.asarray(filt.variables["vo_glor"][time_index, depth_index, :, :], dtype="float32")
            else:
                u = u_day[depth_index]
                v = v_day[depth_index]
            speed = np.hypot(u, v)
            if depth_index > 0:
                center_i, center_j, center_speed, min_steps = _seeded_speed_min(speed, prev_i, prev_j, params.deep_search_cells)
            check = _hua_verify_radius(u, v, center_i, center_j, params)
            if bool(check["hua_pass"]):
                prev_i, prev_j = center_i, center_j
            else:
                stopped = True
            lat_value = float(lat[center_j])
            lon_value = float(lon[center_i])
            polarity = _extremum_polarity(str(seed["ssh_extremum_type"]), lat_value, float(check.get("circulation_sign", np.nan)))
            row = {
                "date": day.isoformat(),
                "hua_object_id": object_id,
                "seed_order": int(seed_order),
                "ssh_extremum_type": str(seed["ssh_extremum_type"]),
                "polarity": polarity,
                "time_index": int(time_index),
                "depth_index": int(depth_index),
                "depth_m": float(depth[depth_index]),
                "seed_i": seed_i,
                "seed_j": seed_j,
                "seed_lon": float(lon[seed_i]),
                "seed_lat": float(lat[seed_j]),
                "ssh_value_m": float(seed["ssh_value_m"]),
                "speed_min_i": int(center_i),
                "speed_min_j": int(center_j),
                "center_lon": lon_value,
                "center_lat": lat_value,
                "center_x_from_seed_km": float((center_i - seed_i) * dx_km),
                "center_y_from_seed_km": float((center_j - seed_j) * dy_km),
                "center_speed_ms": float(center_speed),
                "local_min_steps": int(min_steps),
                "stopped_after_failure": bool(stopped),
                **check,
            }
            centers_rows.append(row)
            circle_rows.append({k: v for k, v in row.items() if k not in {"center_lon", "center_lat"}})
            if bool(check["hua_pass"]):
                structure_rows.append(
                    {
                        "date": day.isoformat(),
                        "hua_object_id": object_id,
                        "depth_index": int(depth_index),
                        "depth_m": float(depth[depth_index]),
                        "center_lon": lon_value,
                        "center_lat": lat_value,
                        "radius_km": float(check["accepted_radius_cells"]) * float(np.nanmean([dx_km, dy_km])),
                        "polarity": polarity,
                    }
                )
                if args.write_object_voxels:
                    voxel_rows.extend(
                        _object_voxels_for_layer(
                            u,
                            v,
                            lon,
                            lat,
                            float(depth[depth_index]),
                            day=day,
                            object_id=object_id,
                            depth_index=int(depth_index),
                            center_i=int(center_i),
                            center_j=int(center_j),
                            radius_cells=float(check["accepted_radius_cells"]),
                            polarity=polarity,
                        )
                    )
            if args.stop_at_first_failed_layer and stopped:
                break

    centers = pd.DataFrame(centers_rows)
    circle = pd.DataFrame(circle_rows)
    structures = pd.DataFrame(structure_rows)
    voxels = pd.DataFrame(voxel_rows, columns=OBJECT_VOXEL_COLUMNS)
    if args.write_day_figures and not centers.empty:
        _plot_day_summary(day, centers, zos, lon, lat, output_dir / "figures")
    return centers, circle, structures, voxels


def _plot_day_summary(day: date, centers: pd.DataFrame, zos: np.ndarray, lon: np.ndarray, lat: np.ndarray, figure_dir: Path) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    surface = centers[centers["depth_index"].eq(0)].copy()
    if surface.empty:
        return
    fig, ax = plt.subplots(figsize=(13, 4.8))
    vmax = float(np.nanpercentile(np.abs(zos), 98))
    vmax = max(vmax, 1e-6)
    im = ax.pcolormesh(lon, lat, zos, shading="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    passed = surface["hua_pass"].astype(bool)
    ax.scatter(surface.loc[~passed, "seed_lon"], surface.loc[~passed, "seed_lat"], s=12, c="#9ca3af", label="SSH seed failed")
    ax.scatter(surface.loc[passed, "center_lon"], surface.loc[passed, "center_lat"], s=18, c="#22c55e", label="Hua passed center")
    ax.set_title(f"Hua SSH+velocity surface candidates {day:%Y-%m-%d}")
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.legend(loc="upper right", fontsize=8)
    cbar = fig.colorbar(im, ax=ax, pad=0.01)
    cbar.set_label("30-180d bandpass zos (m)")
    fig.savefig(figure_dir / f"surface_candidates_{day:%Y%m%d}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _write_parts(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, index=False, engine="fastparquet")
    tmp.replace(path)


def _read_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path, engine="fastparquet")


def _write_parquet(df: pd.DataFrame, path: Path, index: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=index, engine="fastparquet")


def _merge_parts(parts_dir: Path, name: str, output_dir: Path) -> pd.DataFrame:
    existing = output_dir / f"{name}.parquet"
    if existing.exists():
        return _read_parquet(existing)
    parts = sorted(parts_dir.rglob("*.parquet"))
    if not parts:
        return pd.DataFrame()
    frames = [_read_parquet(path) for path in parts]
    merged = pd.concat(frames, ignore_index=True)
    _write_parquet(merged, output_dir / f"{name}.parquet", index=False)
    merged.to_csv(output_dir / f"{name}.csv", index=False)
    return merged


def _voxel_stats_from_parts(parts_dir: Path) -> pd.DataFrame:
    parts = sorted(parts_dir.rglob("*.parquet"))
    if not parts:
        return pd.DataFrame()
    cols = ["hua_object_id", "depth_index", "i", "j", "node_key_2d", "node_key_3d"]
    stats = []
    for path in parts:
        voxels = pd.read_parquet(path, columns=cols, engine="fastparquet")
        if voxels.empty:
            continue
        surface = voxels[voxels["depth_index"].eq(0)]
        total = (
            voxels.groupby("hua_object_id")
            .agg(
                voxel_count_3d=("node_key_3d", "nunique"),
                min_i=("i", "min"),
                max_i=("i", "max"),
                min_j=("j", "min"),
                max_j=("j", "max"),
                min_depth_index=("depth_index", "min"),
                max_depth_index=("depth_index", "max"),
            )
            .reset_index()
        )
        surf = surface.groupby("hua_object_id")["node_key_2d"].nunique().rename("surface_voxel_count_2d").reset_index()
        stats.append(total.merge(surf, on="hua_object_id", how="left"))
    if not stats:
        return pd.DataFrame()
    merged = pd.concat(stats, ignore_index=True)
    reduced = (
        merged.groupby("hua_object_id")
        .agg(
            voxel_count_3d=("voxel_count_3d", "sum"),
            surface_voxel_count_2d=("surface_voxel_count_2d", "sum"),
            min_i=("min_i", "min"),
            max_i=("max_i", "max"),
            min_j=("min_j", "min"),
            max_j=("max_j", "max"),
            min_depth_index=("min_depth_index", "min"),
            max_depth_index=("max_depth_index", "max"),
        )
        .reset_index()
    )
    reduced["surface_voxel_count_2d"] = reduced["surface_voxel_count_2d"].fillna(0).astype("int64")
    return reduced


def _write_frame_object_summary(centers: pd.DataFrame, voxels: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    if centers.empty:
        summary = pd.DataFrame()
    else:
        passed = centers[centers["hua_pass"].astype(bool)].copy()
        if passed.empty:
            summary = pd.DataFrame()
        else:
            layer_stats = (
                passed.groupby("hua_object_id")
                .agg(
                    date=("date", "first"),
                    polarity=("polarity", "first"),
                    surface_seed_i=("seed_i", "first"),
                    surface_seed_j=("seed_j", "first"),
                    surface_seed_lon=("seed_lon", "first"),
                    surface_seed_lat=("seed_lat", "first"),
                    surface_center_i=("speed_min_i", "first"),
                    surface_center_j=("speed_min_j", "first"),
                    surface_center_lon=("center_lon", "first"),
                    surface_center_lat=("center_lat", "first"),
                    ssh_value_m=("ssh_value_m", "first"),
                    pass_layers=("depth_index", "size"),
                    max_depth_m=("depth_m", "max"),
                    mean_radius_cells=("accepted_radius_cells", "mean"),
                    min_center_speed_ms=("center_speed_ms", "min"),
                    mean_center_speed_ms=("center_speed_ms", "mean"),
                )
                .reset_index()
            )
            voxel_stats = _voxel_stats_from_parts(output_dir / "object_voxels_parts") if voxels.empty else pd.DataFrame()
            if voxels.empty and voxel_stats.empty:
                summary = layer_stats
                summary["voxel_count_3d"] = 0
                summary["surface_voxel_count_2d"] = 0
            else:
                if voxel_stats.empty:
                    surface = voxels[voxels["depth_index"].eq(0)]
                    voxel_stats = (
                        voxels.groupby("hua_object_id")
                        .agg(
                            voxel_count_3d=("node_key_3d", "nunique"),
                            min_i=("i", "min"),
                            max_i=("i", "max"),
                            min_j=("j", "min"),
                            max_j=("j", "max"),
                            min_depth_index=("depth_index", "min"),
                            max_depth_index=("depth_index", "max"),
                        )
                        .reset_index()
                    )
                    surf = surface.groupby("hua_object_id")["node_key_2d"].nunique().rename("surface_voxel_count_2d").reset_index()
                    voxel_stats = voxel_stats.merge(surf, on="hua_object_id", how="left")
                summary = layer_stats.merge(voxel_stats, on="hua_object_id", how="left")
    _write_parquet(summary, output_dir / "frame_object_summary.parquet", index=False)
    summary.to_csv(output_dir / "frame_object_summary.csv", index=False)
    return summary


def _plot_axis_examples(centers: pd.DataFrame, output_dir: Path, max_examples: int = 8) -> None:
    if centers.empty:
        return
    figure_dir = output_dir / "axis_velocity_stack_examples"
    figure_dir.mkdir(parents=True, exist_ok=True)
    ranked = (
        centers.groupby("hua_object_id")
        .agg(n_layers=("depth_index", "size"), n_pass=("hua_pass", "sum"), date=("date", "first"), polarity=("polarity", "first"))
        .sort_values(["n_pass", "n_layers"], ascending=False)
        .head(max_examples)
        .reset_index()
    )
    for _, item in ranked.iterrows():
        obj = centers[centers["hua_object_id"].eq(item["hua_object_id"])].sort_values("depth_index")
        z = -obj["depth_m"].to_numpy(dtype="float64") / 1000.0
        x = obj["center_x_from_seed_km"].to_numpy(dtype="float64")
        y = obj["center_y_from_seed_km"].to_numpy(dtype="float64")
        passed = obj["hua_pass"].astype(bool).to_numpy()
        fig = plt.figure(figsize=(8, 6.5))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot(x, y, z, color="#334155", linewidth=2.0, label="Hua candidate axis")
        ax.scatter(x[passed], y[passed], z[passed], c="#22c55e", s=36, label="pass")
        ax.scatter(x[~passed], y[~passed], z[~passed], c="#ef4444", s=32, marker="x", label="fail")
        ax.scatter([0], [0], [z[0] if len(z) else 0], marker="+", c="red", s=150, linewidth=3, label="SSH seed")
        ax.set_xlabel("east from SSH seed (km)")
        ax.set_ylabel("north from SSH seed (km)")
        ax.set_zlabel("depth (km, down)")
        ax.set_title(f"Hua replicated axis {item['hua_object_id']}\n{item['polarity']}, {item['date']}, pass {int(item['n_pass'])}/{int(item['n_layers'])}")
        ax.view_init(elev=22, azim=-55)
        ax.legend(fontsize=8)
        stem = figure_dir / f"axis_{item['hua_object_id']}"
        fig.savefig(stem.with_suffix(".png"), dpi=180, bbox_inches="tight")
        fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)


def _write_docs(output_dir: Path, args: argparse.Namespace, summary: dict[str, object], rejection: pd.DataFrame) -> None:
    lines = [
        "# Hua 2023 SSH+Velocity Hybrid 方法对齐说明",
        "",
        "本目录是 Hua et al. 2023 方法在 ACC 30-180 天带通场上的论文复刻实验，不读取现有 catalog/tracks/completed centers。",
        "",
        "## 方法链条",
        "",
        "1. 用 `zos_glor` 带通信号在表层做局地极大/极小搜索，对应论文的 SSH minima/maxima candidate centers。",
        "2. 在 SSH candidate 附近寻找 `sqrt(u'^2+v'^2)` 局部低值，得到速度中心候选。",
        "3. 从 `STARTRADIUS=3` 个网格点开始沿圆周路径检查速度模连续性、方向连续性、切向性、对心对称性和两侧反转。",
        "4. 表层通过后，以下一层上方中心为 seed 向深层扩展；失败层记录原因，不硬跳到远处中心。",
        "",
        "## ACC 适配",
        "",
        "- 主速度口径为 `u'=u_{30-180d}, v'=v_{30-180d}`；SSH 口径为 `zos_{30-180d}`。",
        "- 未复制 raw 年文件，未写 `input_daily/`。",
        "- SSH 搜索窗口、深层搜索半径是 ACC 网格适配参数，已写入 `run_summary.json`。",
        "",
        "## 运行摘要",
        "",
        f"- 日期范围：`{args.start}` 到 `{args.end}`。",
        f"- 总中心记录：`{summary.get('n_center_rows', 0)}`。",
        f"- 表层候选数：`{summary.get('n_surface_candidates', 0)}`。",
        f"- Hua pass 层数：`{summary.get('n_pass_layers', 0)}`。",
        f"- Hua pass fraction：`{summary.get('pass_fraction', 0.0):.4f}`。",
        "",
        "## 失败原因",
        "",
    ]
    if rejection.empty:
        lines.append("没有失败原因统计。")
    else:
        for row in rejection.to_dict("records"):
            lines.append(f"- `{row['failure_reason']}`：`{row['count']}`")
    (output_dir / "method_alignment_zh.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output_dir / "hua_acc_replication_summary_zh.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_synthetic_tests(output_dir: Path, params: DetectionParams) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    yy, xx = np.mgrid[-40:41, -40:41]
    r = np.hypot(xx, yy)
    tang_u = -yy / np.maximum(r, 1)
    tang_v = xx / np.maximum(r, 1)
    amp = np.exp(-(r / 15) ** 2)
    cases = []
    saddle_u = xx * np.exp(-(r / 18) ** 2)
    saddle_v = -yy * np.exp(-(r / 18) ** 2)
    for name, u, v, expected in [
        ("gaussian_vortex", tang_u * amp, tang_v * amp, True),
        ("pure_shear", np.ones_like(xx, dtype=float) * 0.05, yy * 0.0, False),
        ("double_core_saddle", saddle_u, saddle_v, False),
    ]:
        check = _hua_verify_radius(u, v, 40, 40, params)
        cases.append({"case": name, "expected_pass": expected, **check})
    df = pd.DataFrame(cases)
    df.to_csv(output_dir / "synthetic_hua_tests.csv", index=False)
    _write_parquet(df, output_dir / "synthetic_hua_tests.parquet", index=False)
    return df


def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    params = DetectionParams(
        ssh_window_cells=args.ssh_window_cells,
        start_radius_cells=args.start_radius_cells,
        max_radius_cells=args.max_radius_cells,
        speed_ratio_max=args.speed_ratio_max,
        angle_jump_max_deg=args.angle_jump_max_deg,
        tangent_tolerance_deg=args.tangent_tolerance_deg,
        symmetry_tolerance_deg=args.symmetry_tolerance_deg,
        min_tangent_fraction=args.min_tangent_fraction,
        min_reversal_fraction=args.min_reversal_fraction,
        min_finite_fraction=args.min_finite_fraction,
        direction_exception_extra=args.direction_exception_extra,
        surface_search_cells=args.surface_search_cells,
        deep_search_cells=args.deep_search_cells,
        require_boundary_monotonic_rotation=args.require_boundary_monotonic_rotation,
        boundary_monotonic_exception_limit=args.boundary_monotonic_exception_limit,
    )
    if args.synthetic_tests:
        run_synthetic_tests(output_dir / "synthetic_tests", params)

    days = _date_range(_parse_date(args.start), _parse_date(args.end))
    part_root = output_dir / "parts"
    if not args.finalize_only:
        years = sorted({d.year for d in days})
        for year in years:
            filt, raw, lon, lat, depth = _load_year_arrays(args, year)
            try:
                year_days = [d for d in days if d.year == year]
                for day in year_days:
                    centers_path = part_root / "centers" / f"date={day:%Y%m%d}.parquet"
                    circle_path = part_root / "circle" / f"date={day:%Y%m%d}.parquet"
                    structures_path = part_root / "structures" / f"date={day:%Y%m%d}.parquet"
                    voxels_path = output_dir / "object_voxels_parts" / f"year={day.year}" / f"date={day:%Y%m%d}.parquet"
                    voxel_ready = (not args.write_object_voxels) or voxels_path.exists()
                    if args.resume and centers_path.exists() and circle_path.exists() and structures_path.exists() and voxel_ready:
                        continue
                    centers, circle, structures, voxels = _detect_day(day, filt, raw, lon, lat, depth, params, args, output_dir)
                    _write_parts(centers, centers_path)
                    _write_parts(circle, circle_path)
                    _write_parts(structures, structures_path)
                    if args.write_object_voxels:
                        _write_parts(voxels, voxels_path)
                    print(f"[hua] {day} centers={len(centers)} pass={int(centers['hua_pass'].sum()) if not centers.empty else 0}", flush=True)
            finally:
                filt.close()
                if raw is not None:
                    raw.close()
    if args.partial_only:
        return

    centers = _merge_parts(part_root / "centers", "centers_hua_style", output_dir)
    _merge_parts(part_root / "circle", "circle_check_diagnostics", output_dir)
    _merge_parts(part_root / "structures", "structures_hua_style", output_dir)
    if args.write_object_voxels:
        voxels = pd.DataFrame(columns=OBJECT_VOXEL_COLUMNS)
    else:
        voxels = pd.DataFrame(columns=OBJECT_VOXEL_COLUMNS)
    _write_frame_object_summary(centers, voxels, output_dir)
    if centers.empty:
        rejection = pd.DataFrame(columns=["failure_reason", "count"])
        summary = {"n_center_rows": 0, "n_surface_candidates": 0, "n_pass_layers": 0, "pass_fraction": 0.0}
    else:
        rejection = centers.loc[~centers["hua_pass"].astype(bool), "dominant_failure"].value_counts().rename_axis("failure_reason").reset_index(name="count")
        rejection.to_csv(output_dir / "rejection_reasons.csv", index=False)
        _plot_axis_examples(centers, output_dir)
        summary = {
            "n_center_rows": int(len(centers)),
            "n_surface_candidates": int(centers[centers["depth_index"].eq(0)]["hua_object_id"].nunique()),
            "n_pass_layers": int(centers["hua_pass"].sum()),
            "pass_fraction": float(centers["hua_pass"].mean()),
            "n_days": int(centers["date"].nunique()),
            "n_objects": int(centers["hua_object_id"].nunique()),
            "parameters": {**vars(args), "detection_params": params.__dict__},
        }
    (output_dir / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    if rejection.empty:
        rejection.to_csv(output_dir / "rejection_reasons.csv", index=False)
    _write_docs(output_dir, args, summary, rejection)


def main() -> None:
    parser = argparse.ArgumentParser(description="Replicate Hua 2023 SSH+velocity hybrid eddy detection on bandpass fields.")
    parser.add_argument("--filter-root", default="/root/autodl-fs/2020_2022_acc/Filter")
    parser.add_argument("--raw-root", default="/root/autodl-fs/2020_2022_acc")
    parser.add_argument("--filter-template", default="global_phy_{year}_bandpass_30_180d.nc")
    parser.add_argument("--raw-template", default="global_phy_{year}.nc")
    parser.add_argument("--output-dir", default="/root/autodl-fs/2020_2022_acc/hua_paper_replication/smoke_20200101_20200107")
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2020-01-07")
    parser.add_argument("--max-depth-m", type=float, default=3000.0)
    parser.add_argument("--ssh-window-cells", type=int, default=7)
    parser.add_argument("--max-candidates-per-day", type=int, default=0)
    parser.add_argument("--surface-search-cells", type=int, default=8)
    parser.add_argument("--deep-search-cells", type=int, default=6)
    parser.add_argument("--start-radius-cells", type=int, default=3)
    parser.add_argument("--max-radius-cells", type=int, default=8)
    parser.add_argument("--speed-ratio-max", type=float, default=3.0)
    parser.add_argument("--angle-jump-max-deg", type=float, default=150.0)
    parser.add_argument("--tangent-tolerance-deg", type=float, default=24.0)
    parser.add_argument("--symmetry-tolerance-deg", type=float, default=120.0)
    parser.add_argument("--min-tangent-fraction", type=float, default=0.70)
    parser.add_argument("--min-reversal-fraction", type=float, default=0.70)
    parser.add_argument("--min-finite-fraction", type=float, default=0.95)
    parser.add_argument("--direction-exception-extra", type=int, default=0)
    parser.add_argument("--require-boundary-monotonic-rotation", action="store_true")
    parser.add_argument("--boundary-monotonic-exception-limit", type=int, default=0)
    parser.add_argument("--preload-day-uv", action="store_true")
    parser.add_argument("--write-object-voxels", action="store_true")
    parser.add_argument("--partial-only", action="store_true")
    parser.add_argument("--finalize-only", action="store_true")
    parser.add_argument("--stop-at-first-failed-layer", action="store_true")
    parser.add_argument("--write-day-figures", action="store_true")
    parser.add_argument("--synthetic-tests", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
