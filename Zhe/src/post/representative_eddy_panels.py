from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.fft import dstn, idstn
from scipy.interpolate import RegularGridInterpolator, griddata
from scipy.ndimage import gaussian_filter, minimum_filter


RHO0 = 1025.0
OMEGA = 7.2921159e-5


@dataclass(frozen=True)
class AxisStep:
    rank: int
    from_depth_index: int
    to_depth_index: int
    from_depth_m: float
    to_depth_m: float
    distance_km: float
    distance_over_R: float
    dx_km: float
    dy_km: float
    mid_x_km: float
    mid_y_km: float


def _parse_depths(text: str) -> list[float]:
    return [float(part.strip()) for part in text.split(",") if part.strip()]


def _load_npz(root: Path) -> dict[str, np.ndarray]:
    path = root / "azimuthal_representative_velocity.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def _load_radius_by_polarity(radial_seed_root: Path) -> dict[str, float]:
    path = radial_seed_root / "representative_radii.csv"
    if not path.exists():
        return {}
    table = pd.read_csv(path)
    return {str(row["polarity"]): float(row["representative_radius_m"]) for _, row in table.iterrows()}


def _axis_by_tau(
    radial_seed_root: Path,
    polarity: str,
    tau: float,
    bandwidth: float,
    orientation: str,
) -> pd.DataFrame:
    points = pd.read_parquet(radial_seed_root / "axis" / "rotated_points.parquet")
    selected = pd.read_parquet(radial_seed_root / "object_cache" / "selected_lifecycle_objects.parquet")
    selected = selected[["eddy3d_object_id", "track3d_id", "date", "life_phase"]].copy()
    merged = points.merge(selected, on=["eddy3d_object_id", "track3d_id", "date"], how="inner")
    merged = merged[merged["polarity"].astype(str).eq(polarity)].copy()
    if merged.empty:
        raise ValueError(f"No representative axis points for polarity={polarity}")

    x_col, y_col = ("x_rot_m", "y_rot_m") if orientation == "turned" else ("x_m", "y_m")
    merged["tau_weight"] = np.exp(-0.5 * ((merged["life_phase"].astype(float) - tau) / float(bandwidth)) ** 2)
    rows = []
    for depth_index, part in merged.groupby("depth_index", sort=True):
        w = part["tau_weight"].to_numpy(dtype="f8")
        w_sum = float(np.nansum(w))
        if not np.isfinite(w_sum) or w_sum <= 0.0:
            continue
        rows.append(
            {
                "depth_index": int(depth_index),
                "depth_m": float(np.nanmedian(part["depth_m"].to_numpy(dtype="f8"))),
                "x_km": float(np.nansum(part[x_col].to_numpy(dtype="f8") * w) / w_sum / 1000.0),
                "y_km": float(np.nansum(part[y_col].to_numpy(dtype="f8") * w) / w_sum / 1000.0),
                "effective_weight": w_sum,
            }
        )
    axis = pd.DataFrame(rows).sort_values("depth_index").reset_index(drop=True)
    if axis.empty:
        raise ValueError(f"No weighted representative axis for polarity={polarity}")
    return axis


def _sample_cartesian_field(field: np.ndarray, x_grid: np.ndarray, y_grid: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    interp = RegularGridInterpolator(
        (y_grid[:, 0], x_grid[0, :]),
        field,
        bounds_error=False,
        fill_value=np.nan,
    )
    return interp(np.column_stack([y, x]))


def _composite_hua_ring_check(
    *,
    u_grid: np.ndarray,
    v_grid: np.ndarray,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    center_x_km: float,
    center_y_km: float,
    radius_km: float,
) -> dict[str, float | bool | str]:
    angles = np.linspace(0.0, 2.0 * np.pi, 72, endpoint=False)
    x = center_x_km + radius_km * np.cos(angles)
    y = center_y_km + radius_km * np.sin(angles)
    uu = _sample_cartesian_field(u_grid, x_grid, y_grid, x, y)
    vv = _sample_cartesian_field(v_grid, x_grid, y_grid, x, y)
    speed = np.hypot(uu, vv)
    finite = np.isfinite(speed) & (speed > 1.0e-12)
    finite_fraction = float(finite.mean())
    if finite.sum() < 12:
        return {
            "composite_hua_pass": False,
            "composite_hua_failure": "insufficient_ring_data",
            "finite_fraction": finite_fraction,
            "tangent_pass_fraction": np.nan,
            "opposite_reversal_fraction": np.nan,
            "direction_exception_count": np.nan,
        }

    tx = -np.sin(angles)
    ty = np.cos(angles)
    tangent_cos = np.abs((uu * tx + vv * ty) / np.maximum(speed, 1.0e-12))
    tangent_fraction = float(np.nanmean((tangent_cos >= np.cos(np.deg2rad(30.0)))[finite]))

    half = len(angles) // 2
    reversal_ok = 0
    reversal_total = 0
    for n in range(half):
        m = (n + half) % len(angles)
        if finite[n] and finite[m]:
            reversal_total += 1
            reversal_ok += int(uu[n] * uu[m] + vv[n] * vv[m] < 0.0)
    reversal_fraction = float(reversal_ok / reversal_total) if reversal_total else 0.0

    velocity_angles = np.arctan2(vv, uu)
    diffs = []
    for n in range(len(angles)):
        m = (n + 1) % len(angles)
        if finite[n] and finite[m]:
            diffs.append(float((velocity_angles[m] - velocity_angles[n] + np.pi) % (2.0 * np.pi) - np.pi))
    diffs_arr = np.asarray(diffs, dtype="f8")
    positive = int(np.sum(diffs_arr > 0.0))
    negative = int(np.sum(diffs_arr < 0.0))
    direction_exceptions = float(min(positive, negative))
    boundary_monotonic_passed = direction_exceptions <= 2.0

    passed = bool(finite_fraction >= 0.70 and tangent_fraction >= 0.55 and reversal_fraction >= 0.55 and boundary_monotonic_passed)
    failure = "none"
    if not passed:
        if finite_fraction < 0.70:
            failure = "finite_fraction"
        elif tangent_fraction < 0.55:
            failure = "tangent_alignment"
        elif reversal_fraction < 0.55:
            failure = "opposite_reversal"
        else:
            failure = "boundary_monotonic_rotation"
    return {
        "composite_hua_pass": passed,
        "composite_hua_failure": failure,
        "finite_fraction": finite_fraction,
        "tangent_pass_fraction": tangent_fraction,
        "opposite_reversal_fraction": reversal_fraction,
        "direction_exception_count": direction_exceptions,
    }


def _speed_min_candidates(
    speed_grid: np.ndarray,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    radius_km: float,
    search_rmax: float,
    max_candidates: int = 40,
) -> list[tuple[int, int]]:
    search = np.isfinite(speed_grid) & (np.hypot(x_grid, y_grid) <= radius_km * float(search_rmax))
    if not search.any():
        return []
    filled = np.where(search, speed_grid, np.inf)
    local_min = filled == minimum_filter(filled, size=5, mode="nearest")
    finite_values = speed_grid[search]
    threshold = float(np.nanpercentile(finite_values, 35.0))
    jj, ii = np.where(search & local_min & (speed_grid <= threshold))
    candidates = [(int(j), int(i)) for j, i in zip(jj, ii)]
    global_j, global_i = np.unravel_index(int(np.nanargmin(filled)), filled.shape)
    candidates.append((int(global_j), int(global_i)))
    candidates = list(dict.fromkeys(candidates))
    candidates.sort(key=lambda item: float(speed_grid[item[0], item[1]]))
    return candidates[:max_candidates]


def _composite_hua_axis(
    *,
    radial: np.ndarray,
    theta: np.ndarray,
    depth: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    speed: np.ndarray,
    radius_m: float,
    grid_size: int,
    search_rmax: float,
) -> pd.DataFrame:
    radius_km = float(radius_m) / 1000.0
    rows = []
    previous_xy: tuple[float, float] | None = None
    for depth_index in range(len(depth)):
        x_grid, y_grid, speed_grid = _regular_grid_from_polar(radial, theta, speed[depth_index], radius_m, grid_size)
        _, _, u_grid = _regular_grid_from_polar(radial, theta, u[depth_index], radius_m, grid_size)
        _, _, v_grid = _regular_grid_from_polar(radial, theta, v[depth_index], radius_m, grid_size)
        candidates = _speed_min_candidates(speed_grid, x_grid, y_grid, radius_km, search_rmax)
        if not candidates:
            rows.append(
                {
                    "depth_index": int(depth_index),
                    "depth_m": float(depth[depth_index]),
                    "x_km": np.nan,
                    "y_km": np.nan,
                    "effective_weight": np.nan,
                    "center_speed_ms": np.nan,
                    "axis_source": "composite_hua",
                    "composite_hua_pass": False,
                    "composite_hua_failure": "no_search_data",
                }
            )
            continue

        finite_speed = speed_grid[np.isfinite(speed_grid)]
        speed_scale = max(float(np.nanpercentile(finite_speed, 75.0)), 1.0e-9) if finite_speed.size else 1.0
        scored = []
        for j, i in candidates:
            center_x = float(x_grid[j, i])
            center_y = float(y_grid[j, i])
            check = _composite_hua_ring_check(
                u_grid=u_grid,
                v_grid=v_grid,
                x_grid=x_grid,
                y_grid=y_grid,
                center_x_km=center_x,
                center_y_km=center_y,
                radius_km=max(radius_km * 0.55, 20.0),
            )
            continuity = 0.0
            if previous_xy is not None:
                continuity = np.hypot(center_x - previous_xy[0], center_y - previous_xy[1]) / max(radius_km, 1.0)
            radial_penalty = np.hypot(center_x, center_y) / max(radius_km, 1.0)
            score = (
                float(speed_grid[j, i]) / speed_scale
                + 0.20 * radial_penalty**2
                + 0.65 * continuity**2
                + (0.0 if bool(check["composite_hua_pass"]) else 2.0)
            )
            scored.append((score, bool(check["composite_hua_pass"]), j, i, check))

        scored.sort(key=lambda item: item[0])
        pass_scored = [item for item in scored if item[1]]
        _, _, j, i, check = (pass_scored[0] if pass_scored else scored[0])
        center_x = float(x_grid[j, i])
        center_y = float(y_grid[j, i])
        previous_xy = (center_x, center_y)
        rows.append(
            {
                "depth_index": int(depth_index),
                "depth_m": float(depth[depth_index]),
                "x_km": center_x,
                "y_km": center_y,
                "effective_weight": float(np.isfinite(speed_grid).sum()),
                "center_speed_ms": float(speed_grid[j, i]),
                "axis_source": "composite_hua",
                "composite_hua_fallback": not bool(check["composite_hua_pass"]),
                "candidate_count": int(len(candidates)),
                **check,
            }
        )
    axis = pd.DataFrame(rows)
    axis = axis[np.isfinite(axis["x_km"]) & np.isfinite(axis["y_km"])].copy()
    if axis.empty:
        raise ValueError("No composite-Hua representative axis could be extracted")
    return axis.sort_values("depth_index").reset_index(drop=True)


def _axis_source_comparison(radial_axis: pd.DataFrame, composite_axis: pd.DataFrame) -> pd.DataFrame:
    merged = radial_axis[["depth_index", "depth_m", "x_km", "y_km"]].merge(
        composite_axis[["depth_index", "x_km", "y_km", "composite_hua_pass", "composite_hua_failure"]],
        on="depth_index",
        how="inner",
        suffixes=("_radial_seed", "_composite_hua"),
    )
    merged["axis_source_offset_km"] = np.hypot(
        merged["x_km_composite_hua"] - merged["x_km_radial_seed"],
        merged["y_km_composite_hua"] - merged["y_km_radial_seed"],
    )
    return merged


def _strongest_axis_steps(axis: pd.DataFrame, radius_m: float, max_steps: int = 2) -> list[AxisStep]:
    axis = axis.sort_values("depth_index").reset_index(drop=True)
    x = axis["x_km"].to_numpy(dtype="f8")
    y = axis["y_km"].to_numpy(dtype="f8")
    depth = axis["depth_m"].to_numpy(dtype="f8")
    depth_index = axis["depth_index"].to_numpy(dtype="i8")
    dx = np.diff(x)
    dy = np.diff(y)
    dist = np.hypot(dx, dy)
    valid = np.isfinite(dist)
    order = np.argsort(np.where(valid, dist, -np.inf))[::-1]
    steps: list[AxisStep] = []
    used_pairs: set[int] = set()
    radius_km = max(float(radius_m) / 1000.0, 1.0)
    for pair_i in order:
        pair_i = int(pair_i)
        if not valid[pair_i] or pair_i in used_pairs:
            continue
        steps.append(
            AxisStep(
                rank=len(steps) + 1,
                from_depth_index=int(depth_index[pair_i]),
                to_depth_index=int(depth_index[pair_i + 1]),
                from_depth_m=float(depth[pair_i]),
                to_depth_m=float(depth[pair_i + 1]),
                distance_km=float(dist[pair_i]),
                distance_over_R=float(dist[pair_i] / radius_km),
                dx_km=float(dx[pair_i]),
                dy_km=float(dy[pair_i]),
                mid_x_km=float(0.5 * (x[pair_i] + x[pair_i + 1])),
                mid_y_km=float(0.5 * (y[pair_i] + y[pair_i + 1])),
            )
        )
        used_pairs.update({pair_i - 1, pair_i, pair_i + 1})
        if len(steps) >= max_steps:
            break
    return steps


def _regular_grid_from_polar(
    radial: np.ndarray,
    theta: np.ndarray,
    values: np.ndarray,
    radius_m: float,
    grid_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    radius_km = float(radius_m) / 1000.0
    rr, tt = np.meshgrid(radial * radius_km, theta, indexing="ij")
    x_src = (rr * np.cos(tt)).ravel()
    y_src = (rr * np.sin(tt)).ravel()
    val_src = np.asarray(values, dtype="f8").ravel()
    valid = np.isfinite(val_src)
    extent = float(np.nanmax(radial) * radius_km)
    axis = np.linspace(-extent, extent, int(grid_size), dtype="f8")
    x_grid, y_grid = np.meshgrid(axis, axis)
    if valid.sum() < 4:
        return x_grid, y_grid, np.full_like(x_grid, np.nan)
    grid = griddata((x_src[valid], y_src[valid]), val_src[valid], (x_grid, y_grid), method="linear")
    fill = griddata((x_src[valid], y_src[valid]), val_src[valid], (x_grid, y_grid), method="nearest")
    grid = np.where(np.isfinite(grid), grid, fill)
    grid[np.hypot(x_grid, y_grid) > extent] = np.nan
    return x_grid, y_grid, grid


def _nan_gaussian_smooth(field: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return np.asarray(field, dtype="f8")
    values = np.asarray(field, dtype="f8")
    valid = np.isfinite(values)
    if not valid.any():
        return values.copy()
    weighted = gaussian_filter(np.where(valid, values, 0.0), sigma=sigma, mode="nearest")
    weights = gaussian_filter(valid.astype("f8"), sigma=sigma, mode="nearest")
    out = np.divide(weighted, weights, out=np.full_like(values, np.nan), where=weights > 1e-12)
    out[~valid] = np.nan
    return out


def _streamfunction_proxy(u_grid: np.ndarray, v_grid: np.ndarray, x_km: np.ndarray, y_km: np.ndarray) -> np.ndarray:
    dx = float(np.nanmedian(np.diff(x_km[0, :]))) * 1000.0
    dy = float(np.nanmedian(np.diff(y_km[:, 0]))) * 1000.0
    u = np.nan_to_num(u_grid, nan=0.0)
    v = np.nan_to_num(v_grid, nan=0.0)
    rhs = np.gradient(v, dx, axis=1) - np.gradient(u, dy, axis=0)
    interior = rhs[1:-1, 1:-1]
    if interior.size == 0:
        return np.full_like(rhs, np.nan)
    ny, nx = interior.shape
    rhs_hat = dstn(interior, type=1, norm="ortho")
    jj = np.arange(1, ny + 1, dtype="f8")[:, None]
    ii = np.arange(1, nx + 1, dtype="f8")[None, :]
    denom_x = 2.0 * (np.cos(np.pi * ii / (nx + 1.0)) - 1.0) / dx**2
    denom_y = 2.0 * (np.cos(np.pi * jj / (ny + 1.0)) - 1.0) / dy**2
    psi_inner = idstn(rhs_hat / (denom_x + denom_y), type=1, norm="ortho")
    psi = np.full_like(rhs, np.nan, dtype="f8")
    psi[1:-1, 1:-1] = psi_inner
    psi[np.isnan(u_grid) | np.isnan(v_grid)] = np.nan
    return psi


def _layer_grids(
    *,
    radial: np.ndarray,
    theta: np.ndarray,
    u_layer: np.ndarray,
    v_layer: np.ndarray,
    speed_layer: np.ndarray,
    radius_m: float,
    grid_size: int,
    f0: float,
    smooth_sigma: float,
) -> dict[str, np.ndarray]:
    x, y, speed = _regular_grid_from_polar(radial, theta, speed_layer, radius_m, grid_size)
    _, _, u = _regular_grid_from_polar(radial, theta, u_layer, radius_m, grid_size)
    _, _, v = _regular_grid_from_polar(radial, theta, v_layer, radius_m, grid_size)
    psi = _streamfunction_proxy(u, v, x, y)
    pressure = RHO0 * f0 * psi
    return {
        "x": x,
        "y": y,
        "u": u,
        "v": v,
        "speed": _nan_gaussian_smooth(speed, smooth_sigma),
        "pressure": _nan_gaussian_smooth(pressure, smooth_sigma),
    }


def _nearest_depth_slot(depth: np.ndarray, depth_index: int) -> int:
    matches = np.where(np.arange(len(depth)) == int(depth_index))[0]
    if matches.size:
        return int(matches[0])
    return int(np.clip(depth_index, 0, len(depth) - 1))


def _section_from_grid(
    field: np.ndarray,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    x_line: np.ndarray,
    y_line: np.ndarray,
) -> np.ndarray:
    x_axis = x_grid[0, :]
    y_axis = y_grid[:, 0]
    interp = RegularGridInterpolator(
        (y_axis, x_axis),
        field,
        bounds_error=False,
        fill_value=np.nan,
    )
    return interp(np.column_stack([y_line, x_line]))


def _normal_velocity_section(
    *,
    step: AxisStep,
    axis: pd.DataFrame,
    depth: np.ndarray,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    u_stack: np.ndarray,
    v_stack: np.ndarray,
    radius_m: float,
    section_mode: str,
    depth_padding_layers: int,
    half_width_r: float,
    min_half_width_km: float,
    anchor_depth_index: int | None = None,
) -> dict[str, np.ndarray]:
    def axis_curved_centerline() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
        sorted_axis = axis.sort_values("depth_index").reset_index(drop=True)
        axis_depth = sorted_axis["depth_m"].to_numpy(dtype="f8")
        axis_x = sorted_axis["x_km"].to_numpy(dtype="f8")
        axis_y = sorted_axis["y_km"].to_numpy(dtype="f8")
        if len(sorted_axis) >= 2:
            center_x_full = np.interp(depth, axis_depth, axis_x)
            center_y_full = np.interp(depth, axis_depth, axis_y)
        else:
            cx = float(axis_x[0]) if len(axis_x) else 0.0
            cy = float(axis_y[0]) if len(axis_y) else 0.0
            center_x_full = np.full_like(depth, cx, dtype="f8")
            center_y_full = np.full_like(depth, cy, dtype="f8")
        dx_dz = np.gradient(center_x_full, depth, edge_order=1)
        dy_dz = np.gradient(center_y_full, depth, edge_order=1)
        mag = np.hypot(dx_dz, dy_dz)
        tx = np.full_like(depth, np.nan, dtype="f8")
        ty = np.full_like(depth, np.nan, dtype="f8")
        valid = np.isfinite(mag) & (mag > 1.0e-8)
        tx[valid] = dx_dz[valid] / mag[valid]
        ty[valid] = dy_dz[valid] / mag[valid]
        fallback_count = int((~valid).sum())
        last = (1.0, 0.0)
        for i in range(len(depth)):
            if np.isfinite(tx[i]) and np.isfinite(ty[i]):
                last = (float(tx[i]), float(ty[i]))
            else:
                tx[i], ty[i] = last
        last = (1.0, 0.0)
        for i in range(len(depth) - 1, -1, -1):
            if np.isfinite(tx[i]) and np.isfinite(ty[i]):
                last = (float(tx[i]), float(ty[i]))
            else:
                tx[i], ty[i] = last
        nx_axis = -ty
        ny_axis = tx
        return center_x_full, center_y_full, tx, ty, nx_axis, ny_axis, fallback_count

    norm = float(np.hypot(step.dx_km, step.dy_km))
    if not np.isfinite(norm) or norm <= 1e-9:
        jump_ex, jump_ey = 1.0, 0.0
    else:
        jump_ex, jump_ey = step.dx_km / norm, step.dy_km / norm
    jump_nx, jump_ny = -jump_ey, jump_ex

    axis_fallback_count = 0
    if section_mode == "axis_curved":
        center_x_full, center_y_full, tx, ty, nx_axis, ny_axis, axis_fallback_count = axis_curved_centerline()
        center_s_full = center_x_full * nx_axis + center_y_full * ny_axis
        axis_nx = np.interp(axis["depth_m"].to_numpy(dtype="f8"), depth, nx_axis)
        axis_ny = np.interp(axis["depth_m"].to_numpy(dtype="f8"), depth, ny_axis)
        center_coord = axis["x_km"].to_numpy(dtype="f8") * axis_nx + axis["y_km"].to_numpy(dtype="f8") * axis_ny
        axis_label = "axis-following curved section, tilt-preserving coordinate"
        coord_label = "distance along local axis-normal section from surface center projection (km)"
        velocity_label = "horizontal velocity normal to curved section, u_axis"
    elif section_mode == "normal":
        section_ex, section_ey = jump_nx, jump_ny
        velocity_nx, velocity_ny = jump_ex, jump_ey
        anchor_x, anchor_y = step.mid_x_km, step.mid_y_km
        axis_label = "jump-normal section through center-pair midpoint"
        coord_label = "distance along jump-normal section from midpoint (km)"
        velocity_label = "horizontal velocity normal to section, u_parallel"
    else:
        section_ex, section_ey = jump_ex, jump_ey
        velocity_nx, velocity_ny = jump_nx, jump_ny
        anchor_index = step.from_depth_index if anchor_depth_index is None else int(anchor_depth_index)
        row = axis[axis["depth_index"].astype(int).eq(anchor_index)]
        anchor_x = float(row.iloc[0]["x_km"]) if not row.empty else step.mid_x_km
        anchor_y = float(row.iloc[0]["y_km"]) if not row.empty else step.mid_y_km
        axis_label = "jump-parallel section"
        coord_label = "distance along jump direction from layer center (km)"
        velocity_label = "horizontal velocity normal to section, u_perp"

    x_half = max(float(radius_m) / 1000.0 * float(half_width_r), float(min_half_width_km))
    max_half = max(float(radius_m) / 1000.0 * 2.5, 150.0, x_half)
    if section_mode == "axis_curved" and np.isfinite(center_coord).any():
        max_half = max(max_half, float(np.nanmax(np.abs(center_coord))) + x_half)
    coord = np.linspace(-max_half, max_half, 181, dtype="f8")
    speed_stack = np.hypot(u_stack, v_stack)
    if section_mode == "axis_curved":
        section = np.full((len(depth), len(coord)), np.nan, dtype="f8")
        speed_section = np.full_like(section, np.nan)
        for iz in range(len(depth)):
            offset = coord - center_s_full[iz]
            x_line = center_x_full[iz] + offset * nx_axis[iz]
            y_line = center_y_full[iz] + offset * ny_axis[iz]
            normal_velocity_layer = u_stack[iz] * tx[iz] + v_stack[iz] * ty[iz]
            section[iz] = _section_from_grid(normal_velocity_layer, x_grid, y_grid, x_line, y_line)
            speed_section[iz] = _section_from_grid(speed_stack[iz], x_grid, y_grid, x_line, y_line)
    else:
        x_line = anchor_x + coord * section_ex
        y_line = anchor_y + coord * section_ey
        normal_velocity = u_stack * velocity_nx + v_stack * velocity_ny
        section = np.vstack([_section_from_grid(layer, x_grid, y_grid, x_line, y_line) for layer in normal_velocity])
        speed_section = np.vstack([_section_from_grid(layer, x_grid, y_grid, x_line, y_line) for layer in speed_stack])
        center_coord = (
            (axis["x_km"].to_numpy(dtype="f8") - anchor_x) * section_ex
            + (axis["y_km"].to_numpy(dtype="f8") - anchor_y) * section_ey
        )
    signed_speed_section = np.sign(section) * speed_section

    k_min = max(0, min(step.from_depth_index, step.to_depth_index) - max(0, depth_padding_layers))
    k_max = min(len(depth) - 1, max(step.from_depth_index, step.to_depth_index) + max(0, depth_padding_layers))
    if section_mode == "axis_curved" and np.isfinite(center_coord).any():
        center_z = axis["depth_m"].to_numpy(dtype="f8")
        in_depth = (center_z >= float(depth[k_min])) & (center_z <= float(depth[k_max]))
        visible_centers = center_coord[in_depth]
        if not np.isfinite(visible_centers).any():
            visible_centers = center_coord[np.isfinite(center_coord)]
        xlim = np.array(
            [
                min(0.0, float(np.nanmin(visible_centers))) - x_half,
                max(0.0, float(np.nanmax(visible_centers))) + x_half,
            ],
            dtype="f8",
        )
    else:
        xlim = np.array([-x_half, x_half], dtype="f8")
    return {
        "section_coord_km": coord,
        "depth": depth,
        "normal_horizontal_velocity_section": section,
        "horizontal_speed_section": speed_section,
        "signed_horizontal_speed_section": signed_speed_section,
        "center_section_coord_km": center_coord,
        "center_depth_m": axis["depth_m"].to_numpy(dtype="f8"),
        "xlim_km": xlim,
        "zlim_m": np.array([float(depth[k_min]), float(depth[k_max])], dtype="f8"),
        "section_axis": axis_label,
        "coordinate_label": coord_label,
        "velocity_label": velocity_label,
        "axis_curved_direction_fallback_count": int(axis_fallback_count),
        "axis_curved_centerline_forced_to_zero": False,
        "axis_curved_reference": "surface_center" if section_mode == "axis_curved" else "",
        "axis_curved_preserves_tilt_projection": bool(section_mode == "axis_curved"),
        "axis_curved_interpretation": (
            "axis-following curved section with tilt-preserving surface-center reference"
            if section_mode == "axis_curved"
            else ""
        ),
    }


def _field_limits(fields: list[np.ndarray], symmetric: bool = False) -> tuple[float, float]:
    chunks = [np.asarray(field, dtype="f8")[np.isfinite(field)] for field in fields if np.isfinite(field).any()]
    if not chunks:
        return (-1.0, 1.0) if symmetric else (0.0, 1.0)
    values = np.concatenate(chunks)
    if values.size == 0:
        return (-1.0, 1.0) if symmetric else (0.0, 1.0)
    if symmetric or (np.nanmin(values) < 0 < np.nanmax(values)):
        vmax = float(np.nanpercentile(np.abs(values), 98.0))
        vmax = max(vmax, 1e-12)
        return -vmax, vmax
    vmin = float(np.nanpercentile(values, 2.0))
    vmax = float(np.nanpercentile(values, 98.0))
    if not vmax > vmin:
        vmax = vmin + 1.0
    return vmin, vmax


def _plot_axis_panel(ax, axis: pd.DataFrame, key: str, title: str, steps: list[AxisStep], xlim: float) -> None:
    ax.plot(axis[key], axis["depth_m"], "-o", color="#244a9b", lw=1.8, ms=3.5)
    ax.axvline(0, color="0.72", lw=0.9)
    for step in steps:
        color = "tab:red" if step.rank == 1 else "tab:green"
        ax.axhline(step.from_depth_m, color=color, ls="--", lw=1.0, alpha=0.75)
        ax.axhline(step.to_depth_m, color=color, ls=":", lw=1.0, alpha=0.75)
    ax.set_xlim(-xlim, xlim)
    ax.set_ylim(float(axis["depth_m"].max()), float(axis["depth_m"].min()))
    ax.set_xlabel("offset (km)")
    ax.set_ylabel("depth (m)")
    ax.set_title(title)
    ax.grid(alpha=0.22)


def _plot_horizontal(
    ax,
    *,
    grid: dict[str, np.ndarray],
    field_name: str,
    title: str,
    label: str,
    cmap: str,
    symmetric: bool,
    center_xy: tuple[float, float],
) -> plt.cm.ScalarMappable:
    field = grid[field_name]
    vmin, vmax = _field_limits([field], symmetric=symmetric)
    mesh = ax.pcolormesh(grid["x"], grid["y"], field, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    if field_name == "speed":
        step = max(1, int(grid["x"].shape[0] // 18))
        ax.quiver(
            grid["x"][::step, ::step],
            grid["y"][::step, ::step],
            grid["u"][::step, ::step],
            grid["v"][::step, ::step],
            color="white",
            alpha=0.65,
            scale=4.0,
            width=0.0022,
        )
    finite = field[np.isfinite(field)]
    if finite.size > 10:
        levels = np.linspace(float(np.nanpercentile(finite, 8)), float(np.nanpercentile(finite, 92)), 8)
        if np.unique(levels).size > 2:
            ax.contour(grid["x"], grid["y"], field, levels=levels, colors="0.28", linewidths=0.45, alpha=0.45)
    ax.scatter([0.0], [0.0], marker="+", c="red", s=70, lw=1.6, label="surface")
    ax.scatter([center_xy[0]], [center_xy[1]], marker="x", c="cyan", s=65, lw=1.8, label="layer axis")
    ax.axhline(0, color="0.75", lw=0.8)
    ax.axvline(0, color="0.75", lw=0.8)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=8)
    ax.set_xlabel("x (km)")
    ax.set_ylabel("y (km)")
    ax.legend(loc="upper right", fontsize=6.8)
    plt.colorbar(mesh, ax=ax, shrink=0.82, label=label)
    return mesh


def _plot_section(
    ax,
    section: dict[str, np.ndarray],
    title: str,
    value_limits: tuple[float, float],
    *,
    field_key: str,
    label: str,
    cmap: str,
    draw_zero: bool,
    zero_field_key: str | None = None,
) -> plt.cm.ScalarMappable:
    coord = section["section_coord_km"]
    depth = section["depth"]
    field = section[field_key]
    mesh = ax.pcolormesh(coord, depth, field, shading="auto", cmap=cmap, vmin=value_limits[0], vmax=value_limits[1])
    finite = field[np.isfinite(field)]
    if finite.size > 10:
        levels = np.linspace(float(value_limits[0]), float(value_limits[1]), 9)
        levels = levels[np.isfinite(levels)]
        if draw_zero:
            span = max(abs(float(value_limits[0])), abs(float(value_limits[1])))
            levels = levels[np.abs(levels) > span * 0.08]
        if levels.size:
            ax.contour(coord, depth, field, levels=levels, colors="0.35", linewidths=0.5, alpha=0.55)
        if draw_zero:
            zero_field = section[zero_field_key] if zero_field_key is not None else field
            zero_finite = zero_field[np.isfinite(zero_field)]
            if zero_finite.size and float(np.nanmin(zero_finite)) < 0.0 < float(np.nanmax(zero_finite)):
                ax.contour(coord, depth, zero_field, levels=[0.0], colors="black", linewidths=2.2)
    ax.plot(section["center_section_coord_km"], section["center_depth_m"], "-o", color="0.12", lw=1.3, ms=2.6, label="axis centers")
    ax.set_xlim(float(section["xlim_km"][0]), float(section["xlim_km"][1]))
    ax.set_ylim(float(section["zlim_m"][1]), float(section["zlim_m"][0]))
    ax.set_title(f"{title}\n{section['section_axis']}", fontsize=8)
    ax.set_xlabel(section["coordinate_label"])
    ax.set_ylabel("depth (m)")
    ax.grid(alpha=0.16)
    ax.legend(loc="upper right", fontsize=6.6)
    mesh.set_label(label)
    return mesh


def _plot_support(ax, count: np.ndarray, tau_grid: np.ndarray, depth: np.ndarray, title: str) -> None:
    support = np.nanmean(count, axis=(2, 3)).T
    mesh = ax.pcolormesh(tau_grid, depth, support, shading="auto", cmap="viridis")
    ax.set_ylim(float(np.nanmax(depth)), float(np.nanmin(depth)))
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("life phase tau")
    ax.set_ylabel("depth (m)")
    plt.colorbar(mesh, ax=ax, shrink=0.78, label="mean bin weight")


def _plot_unavailable(ax, message: str) -> None:
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes)
    ax.set_axis_off()


def _build_layer_cache(
    *,
    steps: list[AxisStep],
    radial: np.ndarray,
    theta: np.ndarray,
    depth: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    speed: np.ndarray,
    radius_m: float,
    grid_size: int,
    f0: float,
    smooth_sigma: float,
) -> dict[int, dict[str, np.ndarray]]:
    cache: dict[int, dict[str, np.ndarray]] = {}
    for step in steps:
        for depth_index in (step.from_depth_index, step.to_depth_index):
            if depth_index in cache:
                continue
            slot = _nearest_depth_slot(depth, depth_index)
            cache[depth_index] = _layer_grids(
                radial=radial,
                theta=theta,
                u_layer=u[slot],
                v_layer=v[slot],
                speed_layer=speed[slot],
                radius_m=radius_m,
                grid_size=grid_size,
                f0=f0,
                smooth_sigma=smooth_sigma,
            )
    return cache


def _build_uv_stacks(
    radial: np.ndarray,
    theta: np.ndarray,
    depth: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    radius_m: float,
    grid_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    u_layers = []
    v_layers = []
    x_ref: np.ndarray | None = None
    y_ref: np.ndarray | None = None
    for slot in range(len(depth)):
        x, y, u_grid = _regular_grid_from_polar(radial, theta, u[slot], radius_m, grid_size)
        _, _, v_grid = _regular_grid_from_polar(radial, theta, v[slot], radius_m, grid_size)
        if x_ref is None:
            x_ref, y_ref = x, y
        u_layers.append(u_grid)
        v_layers.append(v_grid)
    if x_ref is None or y_ref is None:
        raise ValueError("No depth layers available for representative velocity stack")
    return x_ref, y_ref, np.stack(u_layers), np.stack(v_layers)


def _plot_step_group(
    *,
    step: AxisStep,
    axis: pd.DataFrame,
    layer_cache: dict[int, dict[str, np.ndarray]],
    speed_u_ax,
    pressure_u_ax,
    speed_l_ax,
    pressure_l_ax,
    section_u_ax,
    section_l_ax,
    section_upper: dict[str, np.ndarray],
    section_lower: dict[str, np.ndarray],
    section_limits: tuple[float, float],
    right_panel_mode: str,
    label: str,
    speed_no: str,
    pressure_no: str,
    section_upper_no: str,
    section_lower_no: str,
) -> plt.cm.ScalarMappable:
    def center_xy(depth_index: int) -> tuple[float, float]:
        row = axis[axis["depth_index"].astype(int).eq(int(depth_index))]
        if row.empty:
            return 0.0, 0.0
        return float(row.iloc[0]["x_km"]), float(row.iloc[0]["y_km"])

    upper_grid = layer_cache[step.from_depth_index]
    lower_grid = layer_cache[step.to_depth_index]
    step_text = f"{label} representative step {step.from_depth_index}->{step.to_depth_index}"
    _plot_horizontal(
        speed_u_ax,
        grid=upper_grid,
        field_name="speed",
        title=f"{speed_no}U  {step_text} upper/from: speed |u',v'|\nk={step.from_depth_index}, z={step.from_depth_m:.0f} m",
        label="m/s",
        cmap="coolwarm",
        symmetric=False,
        center_xy=center_xy(step.from_depth_index),
    )
    _plot_horizontal(
        pressure_u_ax,
        grid=upper_grid,
        field_name="pressure",
        title=f"{pressure_no}U  {step_text} upper/from: geostrophic p' proxy\nk={step.from_depth_index}, z={step.from_depth_m:.0f} m",
        label="Pa proxy",
        cmap="RdBu_r",
        symmetric=True,
        center_xy=center_xy(step.from_depth_index),
    )
    _plot_horizontal(
        speed_l_ax,
        grid=lower_grid,
        field_name="speed",
        title=f"{speed_no}L  {step_text} lower/to: speed |u',v'|\nk={step.to_depth_index}, z={step.to_depth_m:.0f} m",
        label="m/s",
        cmap="coolwarm",
        symmetric=False,
        center_xy=center_xy(step.to_depth_index),
    )
    _plot_horizontal(
        pressure_l_ax,
        grid=lower_grid,
        field_name="pressure",
        title=f"{pressure_no}L  {step_text} lower/to: geostrophic p' proxy\nk={step.to_depth_index}, z={step.to_depth_m:.0f} m",
        label="Pa proxy",
        cmap="RdBu_r",
        symmetric=True,
        center_xy=center_xy(step.to_depth_index),
    )
    if right_panel_mode == "horizontal_speed":
        field_key = "horizontal_speed_section"
        title_suffix = "horizontal speed |u_h|"
        cmap = "coolwarm"
        draw_zero = False
        zero_field_key = None
    elif right_panel_mode == "signed_horizontal_speed":
        field_key = "signed_horizontal_speed_section"
        title_suffix = "signed horizontal speed sign(u_perp)|u_h|"
        cmap = "RdBu_r"
        draw_zero = True
        zero_field_key = "normal_horizontal_velocity_section"
    else:
        field_key = "normal_horizontal_velocity_section"
        title_suffix = "normal horizontal velocity"
        cmap = "RdBu_r"
        draw_zero = True
        zero_field_key = None
    mesh_u = _plot_section(
        section_u_ax,
        section_upper,
        f"{section_upper_no}  J{step.rank} {step.from_depth_index}->{step.to_depth_index} upper/from: {title_suffix}",
        section_limits,
        field_key=field_key,
        label="m/s",
        cmap=cmap,
        draw_zero=draw_zero,
        zero_field_key=zero_field_key,
    )
    _plot_section(
        section_l_ax,
        section_lower,
        f"{section_lower_no}  J{step.rank} {step.from_depth_index}->{step.to_depth_index} lower/to: {title_suffix}",
        section_limits,
        field_key=field_key,
        label="m/s",
        cmap=cmap,
        draw_zero=draw_zero,
        zero_field_key=zero_field_key,
    )
    return mesh_u


def plot_representative_eddy_panels(
    *,
    me_liutex_root: Path,
    radial_seed_root: Path,
    output_dir: Path,
    orientation: str,
    tau: float,
    axis_bandwidth: float,
    grid_size: int,
    reference_lat: float,
    section_mode: str = "normal",
    horizontal_smooth_sigma_cells: float = 0.8,
    section_depth_padding_layers: int = 6,
    section_half_width_r: float = 1.2,
    section_min_half_width_km: float = 75.0,
    right_panel_mode: str = "normal_horizontal_velocity",
    axis_source: str = "radial_seed",
    composite_hua_search_rmax: float = 1.5,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    data = _load_npz(me_liutex_root)
    polarities = [str(x) for x in np.asarray(data["polarities"])]
    tau_grid = np.asarray(data["tau_grid"], dtype="f8")
    depth = np.asarray(data["depth"], dtype="f8")
    radial = np.asarray(data["radial"], dtype="f8")
    theta = np.asarray(data["theta"], dtype="f8")
    u_mean = np.asarray(data["u_mean"], dtype="f8")
    v_mean = np.asarray(data["v_mean"], dtype="f8")
    speed_mean = np.asarray(data["speed_mean"], dtype="f8")
    count = np.asarray(data["count"], dtype="f8")
    n_objects = np.asarray(data["n_objects"])
    n_tracks = np.asarray(data["n_tracks"])
    tau_i = int(np.nanargmin(np.abs(tau_grid - float(tau))))
    radius_by_polarity = _load_radius_by_polarity(radial_seed_root)
    f0 = 2.0 * OMEGA * np.sin(np.deg2rad(float(reference_lat)))
    orientation_label = "global-alpha TURN" if orientation == "turned" else "UNTURN east/north"
    written: list[Path] = []
    manifests: list[dict] = []
    axis_comparisons: list[pd.DataFrame] = []

    for ip, polarity in enumerate(polarities):
        radius_m = radius_by_polarity.get(polarity, 80000.0)
        radial_axis = _axis_by_tau(radial_seed_root, polarity, float(tau_grid[tau_i]), axis_bandwidth, orientation)
        if axis_source == "composite_hua":
            axis = _composite_hua_axis(
                radial=radial,
                theta=theta,
                depth=depth,
                u=u_mean[ip, tau_i],
                v=v_mean[ip, tau_i],
                speed=speed_mean[ip, tau_i],
                radius_m=radius_m,
                grid_size=grid_size,
                search_rmax=composite_hua_search_rmax,
            )
            axis_path = output_dir / (
                f"representative_composite_hua_axis_{orientation}_{polarity}_"
                f"tau{int(round(float(tau_grid[tau_i]) * 100)):03d}.csv"
            )
            axis.to_csv(axis_path, index=False)
            comparison = _axis_source_comparison(radial_axis, axis)
            comparison.insert(0, "polarity", polarity)
            comparison.insert(1, "orientation", orientation)
            comparison.insert(2, "tau", float(tau_grid[tau_i]))
            axis_comparisons.append(comparison)
        else:
            axis = radial_axis
        steps = _strongest_axis_steps(axis, radius_m, max_steps=2)
        if not steps:
            continue
        layer_cache = _build_layer_cache(
            steps=steps,
            radial=radial,
            theta=theta,
            depth=depth,
            u=u_mean[ip, tau_i],
            v=v_mean[ip, tau_i],
            speed=speed_mean[ip, tau_i],
            radius_m=radius_m,
            grid_size=grid_size,
            f0=f0,
            smooth_sigma=horizontal_smooth_sigma_cells,
        )
        x_grid, y_grid, u_stack, v_stack = _build_uv_stacks(
            radial, theta, depth, u_mean[ip, tau_i], v_mean[ip, tau_i], radius_m, grid_size
        )
        section_pairs = [
            (
                _normal_velocity_section(
                    step=step,
                    axis=axis,
                    depth=depth,
                    x_grid=x_grid,
                    y_grid=y_grid,
                    u_stack=u_stack,
                    v_stack=v_stack,
                    radius_m=radius_m,
                    section_mode=section_mode,
                    depth_padding_layers=section_depth_padding_layers,
                    half_width_r=section_half_width_r,
                    min_half_width_km=section_min_half_width_km,
                    anchor_depth_index=step.from_depth_index,
                ),
                _normal_velocity_section(
                    step=step,
                    axis=axis,
                    depth=depth,
                    x_grid=x_grid,
                    y_grid=y_grid,
                    u_stack=u_stack,
                    v_stack=v_stack,
                    radius_m=radius_m,
                    section_mode=section_mode,
                    depth_padding_layers=section_depth_padding_layers,
                    half_width_r=section_half_width_r,
                    min_half_width_km=section_min_half_width_km,
                    anchor_depth_index=step.to_depth_index,
                ),
            )
            for step in steps
        ]
        if right_panel_mode == "horizontal_speed":
            section_field_key = "horizontal_speed_section"
            section_limits = _field_limits([part[section_field_key] for pair in section_pairs for part in pair], symmetric=False)
        elif right_panel_mode == "signed_horizontal_speed":
            section_field_key = "signed_horizontal_speed_section"
            section_limits = _field_limits([part[section_field_key] for pair in section_pairs for part in pair], symmetric=True)
        else:
            section_field_key = "normal_horizontal_velocity_section"
            section_limits = _field_limits([part[section_field_key] for pair in section_pairs for part in pair], symmetric=True)

        fig = plt.figure(figsize=(32, 18), constrained_layout=True)
        gs = fig.add_gridspec(
            5,
            6,
            height_ratios=[1.0, 1.0, 1.0, 1.0, 0.85],
            width_ratios=[0.9, 0.9, 1.05, 1.05, 1.05, 1.05],
        )
        axes = {
            "x": fig.add_subplot(gs[0:4, 0]),
            "y": fig.add_subplot(gs[0:4, 1]),
            "3u": fig.add_subplot(gs[0, 2]),
            "4u": fig.add_subplot(gs[0, 3]),
            "3l": fig.add_subplot(gs[1, 2]),
            "4l": fig.add_subplot(gs[1, 3]),
            "5u": fig.add_subplot(gs[2, 2]),
            "6u": fig.add_subplot(gs[2, 3]),
            "5l": fig.add_subplot(gs[3, 2]),
            "6l": fig.add_subplot(gs[3, 3]),
            "8": fig.add_subplot(gs[0:2, 4]),
            "10": fig.add_subplot(gs[0:2, 5]),
            "9": fig.add_subplot(gs[2:4, 4]),
            "11": fig.add_subplot(gs[2:4, 5]),
            "7": fig.add_subplot(gs[4, :]),
        }
        offsets = np.abs(axis[["x_km", "y_km"]].to_numpy(dtype="f8"))
        offset_xlim = max(float(np.nanmax(offsets)) * 1.08 if np.isfinite(offsets).any() else 1.0, 5.0)
        _plot_axis_panel(axes["x"], axis, "x_km", "1  representative axis delta x", steps, offset_xlim)
        _plot_axis_panel(axes["y"], axis, "y_km", "2  representative axis delta y", steps, offset_xlim)

        section_mesh = _plot_step_group(
            step=steps[0],
            axis=axis,
            layer_cache=layer_cache,
            speed_u_ax=axes["3u"],
            pressure_u_ax=axes["4u"],
            speed_l_ax=axes["3l"],
            pressure_l_ax=axes["4l"],
            section_u_ax=axes["8"],
            section_l_ax=axes["10"],
            section_upper=section_pairs[0][0],
            section_lower=section_pairs[0][1],
            section_limits=section_limits,
            right_panel_mode=right_panel_mode,
            label="first",
            speed_no="3",
            pressure_no="4",
            section_upper_no="8",
            section_lower_no="10",
        )
        if len(steps) > 1:
            _plot_step_group(
                step=steps[1],
                axis=axis,
                layer_cache=layer_cache,
                speed_u_ax=axes["5u"],
                pressure_u_ax=axes["6u"],
                speed_l_ax=axes["5l"],
                pressure_l_ax=axes["6l"],
                section_u_ax=axes["9"],
                section_l_ax=axes["11"],
                section_upper=section_pairs[1][0],
                section_lower=section_pairs[1][1],
                section_limits=section_limits,
                right_panel_mode=right_panel_mode,
                label="second",
                speed_no="5",
                pressure_no="6",
                section_upper_no="9",
                section_lower_no="11",
            )
        else:
            for key in ("5u", "6u", "5l", "6l", "9", "11"):
                _plot_unavailable(axes[key], "no second representative axis step")

        fig.colorbar(section_mesh, ax=[axes["8"], axes["10"], axes["9"], axes["11"]], shrink=0.82, label="m/s")
        _plot_support(axes["7"], count[ip], tau_grid, depth, "7  composite support, no trajectory")
        fig.suptitle(
            f"Representative eddy latest panel family: {polarity}, tau={float(tau_grid[tau_i]):.2f}, "
            f"{orientation_label}, axis-source={axis_source}, {section_mode} section, {right_panel_mode}; "
            f"n_objects={int(n_objects[ip, tau_i])}, "
            f"n_tracks={int(n_tracks[ip, tau_i])}; J1={steps[0].distance_km:.1f} km"
            + (f", J2={steps[1].distance_km:.1f} km" if len(steps) > 1 else ""),
            fontsize=15,
        )
        axis_tag = "" if axis_source == "radial_seed" else f"_{axis_source}_axis"
        stem = (
            f"representative_latest_panel_{right_panel_mode}_{section_mode}{axis_tag}_"
            f"{orientation}_{polarity}_tau{int(round(float(tau_grid[tau_i]) * 100)):03d}"
        )
        png = output_dir / f"{stem}.png"
        pdf = output_dir / f"{stem}.pdf"
        fig.savefig(png, dpi=220)
        fig.savefig(pdf)
        plt.close(fig)
        written.extend([png, pdf])
        manifests.append(
            {
                "polarity": polarity,
                "n_objects": int(n_objects[ip, tau_i]),
                "n_tracks": int(n_tracks[ip, tau_i]),
                "axis_source": axis_source,
                "axis_steps": [step.__dict__ for step in steps],
                "composite_hua_pass_count": (
                    int(axis["composite_hua_pass"].sum()) if "composite_hua_pass" in axis.columns else None
                ),
                "composite_hua_layer_count": int(len(axis)),
                "axis_curved_direction_fallback_count": int(
                    sum(
                        part.get("axis_curved_direction_fallback_count", 0)
                        for pair in section_pairs
                        for part in pair
                    )
                ),
                "axis_curved_centerline_forced_to_zero": False,
                "axis_curved_reference": "surface_center" if section_mode == "axis_curved" else "",
                "axis_curved_preserves_tilt_projection": bool(section_mode == "axis_curved"),
                "axis_curved_interpretation": (
                    "axis-following curved section with tilt-preserving surface-center reference"
                    if section_mode == "axis_curved"
                    else ""
                ),
            }
        )

    manifest = {
        "me_liutex_root": str(me_liutex_root),
        "radial_seed_root": str(radial_seed_root),
        "orientation": orientation,
        "section_mode": section_mode,
        "right_panel_mode": right_panel_mode,
        "axis_source": axis_source,
        "composite_hua_search_rmax": composite_hua_search_rmax,
        "tau": float(tau_grid[tau_i]),
        "axis_step_definition": "top two adjacent-depth representative axis displacements at selected tau",
        "horizontal_smooth_sigma_cells": horizontal_smooth_sigma_cells,
        "reference_lat": reference_lat,
        "polarity_summaries": manifests,
        "figures": [str(path) for path in written],
    }
    axis_tag = "" if axis_source == "radial_seed" else f"_{axis_source}_axis"
    (output_dir / f"representative_latest_panel_{right_panel_mode}_{section_mode}{axis_tag}_{orientation}_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if axis_comparisons:
        pd.concat(axis_comparisons, ignore_index=True).to_csv(
            output_dir
            / f"representative_axis_source_comparison_{right_panel_mode}_{section_mode}_{orientation}_tau{int(round(float(tau_grid[tau_i]) * 100)):03d}.csv",
            index=False,
        )
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot latest representative eddy panel family without object trajectory.")
    parser.add_argument("--me-liutex-root", type=Path, required=True)
    parser.add_argument("--radial-seed-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--orientation", choices=["turned", "unturned"], default="turned")
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument("--axis-bandwidth", type=float, default=0.075)
    parser.add_argument("--grid-size", type=int, default=121)
    parser.add_argument("--reference-lat", type=float, default=28.0)
    parser.add_argument("--section-mode", choices=["parallel", "normal", "axis_curved"], default="normal")
    parser.add_argument(
        "--right-panel-mode",
        choices=["normal_horizontal_velocity", "horizontal_speed", "signed_horizontal_speed"],
        default="normal_horizontal_velocity",
    )
    parser.add_argument("--axis-source", choices=["radial_seed", "composite_hua"], default="radial_seed")
    parser.add_argument("--composite-hua-search-rmax", type=float, default=1.5)
    parser.add_argument("--horizontal-smooth-sigma-cells", type=float, default=0.8)
    parser.add_argument("--section-depth-padding-layers", type=int, default=6)
    parser.add_argument("--section-half-width-r", type=float, default=1.2)
    parser.add_argument("--section-min-half-width-km", type=float, default=75.0)
    args = parser.parse_args()
    written = plot_representative_eddy_panels(
        me_liutex_root=args.me_liutex_root,
        radial_seed_root=args.radial_seed_root,
        output_dir=args.output_dir,
        orientation=args.orientation,
        tau=args.tau,
        axis_bandwidth=args.axis_bandwidth,
        grid_size=args.grid_size,
        reference_lat=args.reference_lat,
        section_mode=args.section_mode,
        horizontal_smooth_sigma_cells=args.horizontal_smooth_sigma_cells,
        section_depth_padding_layers=args.section_depth_padding_layers,
        section_half_width_r=args.section_half_width_r,
        section_min_half_width_km=args.section_min_half_width_km,
        right_panel_mode=args.right_panel_mode,
        axis_source=args.axis_source,
        composite_hua_search_rmax=args.composite_hua_search_rmax,
    )
    print(json.dumps({"figures": [str(path) for path in written]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
