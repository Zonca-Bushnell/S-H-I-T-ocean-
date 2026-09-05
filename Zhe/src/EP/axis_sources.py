from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator, griddata
from scipy.ndimage import minimum_filter

from .contracts import EARTH_OMEGA, RHO0, axis_source_filename
from .fields import RepresentativeVortexDataset
from .numerics import streamfunction_from_zeta


def tau_tag(tau: float) -> str:
    return f"tau{int(round(float(tau) * 100)):03d}"


def axis_sources_dir(me_liutex_root: Path) -> Path:
    return Path(me_liutex_root) / "axis_sources"


def axis_source_path(me_liutex_root: Path, axis_source: str, tau: float) -> Path:
    return axis_sources_dir(me_liutex_root) / axis_source_filename(axis_source, tau)


def load_persisted_axis_source(
    me_liutex_root: Path,
    axis_source: str,
    polarity: str,
    tau: float,
) -> pd.DataFrame:
    path = axis_source_path(me_liutex_root, axis_source, tau)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing persisted representative axis source: {path}. "
            "Run `python -m src.EP.cli build-representative-axis-sources` first."
        )
    table = pd.read_csv(path)
    table = table[table["polarity"].astype(str).eq(str(polarity))].copy()
    if table.empty:
        raise ValueError(f"No {axis_source} axis rows for polarity={polarity} in {path}")
    return table.sort_values("depth_index").reset_index(drop=True)


def _load_radius_by_polarity(radial_seed_root: Path) -> dict[str, float]:
    path = Path(radial_seed_root) / "representative_radii.csv"
    if not path.exists():
        return {}
    table = pd.read_csv(path)
    radius_col = "representative_radius_m" if "representative_radius_m" in table.columns else "radius_m"
    return {str(row["polarity"]): float(row[radius_col]) for _, row in table.iterrows() if radius_col in row}


def radial_seed_axis_by_tau(
    radial_seed_root: Path,
    polarity: str,
    tau: float,
    bandwidth: float,
    orientation: str,
) -> pd.DataFrame:
    points = pd.read_parquet(Path(radial_seed_root) / "axis" / "rotated_points.parquet")
    selected = pd.read_parquet(Path(radial_seed_root) / "object_cache" / "selected_lifecycle_objects.parquet")
    selected = selected[["eddy3d_object_id", "track3d_id", "date", "life_phase"]].copy()
    merged = points.merge(selected, on=["eddy3d_object_id", "track3d_id", "date"], how="inner")
    merged = merged[merged["polarity"].astype(str).eq(str(polarity))].copy()
    if merged.empty:
        raise ValueError(f"No representative axis points for polarity={polarity}")
    x_col, y_col = ("x_rot_m", "y_rot_m") if orientation == "turned" else ("x_m", "y_m")
    merged["tau_weight"] = np.exp(-0.5 * ((merged["life_phase"].astype(float) - tau) / float(bandwidth)) ** 2)
    rows = []
    for depth_index, part in merged.groupby("depth_index", sort=True):
        w = part["tau_weight"].to_numpy(dtype="float64")
        w_sum = float(np.nansum(w))
        if not np.isfinite(w_sum) or w_sum <= 0.0:
            continue
        rows.append(
            {
                "depth_index": int(depth_index),
                "depth_m": float(np.nanmedian(part["depth_m"].to_numpy(dtype="float64"))),
                "x_km": float(np.nansum(part[x_col].to_numpy(dtype="float64") * w) / w_sum / 1000.0),
                "y_km": float(np.nansum(part[y_col].to_numpy(dtype="float64") * w) / w_sum / 1000.0),
                "effective_weight": w_sum,
            }
        )
    axis = pd.DataFrame(rows).sort_values("depth_index").reset_index(drop=True)
    if axis.empty:
        raise ValueError(f"No weighted representative axis for polarity={polarity}")
    return axis


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
    val_src = np.asarray(values, dtype="float64").ravel()
    valid = np.isfinite(val_src)
    extent = float(np.nanmax(radial) * radius_km)
    axis = np.linspace(-extent, extent, int(grid_size), dtype="float64")
    x_grid, y_grid = np.meshgrid(axis, axis)
    if valid.sum() < 4:
        return x_grid, y_grid, np.full_like(x_grid, np.nan)
    grid = griddata((x_src[valid], y_src[valid]), val_src[valid], (x_grid, y_grid), method="linear")
    fill = griddata((x_src[valid], y_src[valid]), val_src[valid], (x_grid, y_grid), method="nearest")
    grid = np.where(np.isfinite(grid), grid, fill)
    grid[np.hypot(x_grid, y_grid) > extent] = np.nan
    return x_grid, y_grid, grid


def _sample_cartesian_field(field: np.ndarray, x_grid: np.ndarray, y_grid: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    interp = RegularGridInterpolator((y_grid[:, 0], x_grid[0, :]), field, bounds_error=False, fill_value=np.nan)
    return interp(np.column_stack([y, x]))


def _streamfunction_proxy(u_grid: np.ndarray, v_grid: np.ndarray, x_km: np.ndarray, y_km: np.ndarray) -> np.ndarray:
    dx = float(np.nanmedian(np.diff(x_km[0, :]))) * 1000.0
    dy = float(np.nanmedian(np.diff(y_km[:, 0]))) * 1000.0
    zeta = np.gradient(np.nan_to_num(v_grid, nan=0.0), dx, axis=1) - np.gradient(np.nan_to_num(u_grid, nan=0.0), dy, axis=0)
    psi = streamfunction_from_zeta(zeta, dx=dx, dy=dy)
    psi[np.isnan(u_grid) | np.isnan(v_grid)] = np.nan
    return psi


def _pressure_extreme(pressure_grid: np.ndarray, x_grid: np.ndarray, y_grid: np.ndarray, radius_km: float, search_rmax: float) -> dict[str, float]:
    search = np.isfinite(pressure_grid) & (np.hypot(x_grid, y_grid) <= radius_km * float(search_rmax))
    if not search.any():
        return {"pressure_extreme_x_km": np.nan, "pressure_extreme_y_km": np.nan, "pressure_extreme_value": np.nan}
    masked = np.where(search, np.abs(pressure_grid), np.nan)
    j, i = np.unravel_index(int(np.nanargmax(masked)), masked.shape)
    return {
        "pressure_extreme_x_km": float(x_grid[j, i]),
        "pressure_extreme_y_km": float(y_grid[j, i]),
        "pressure_extreme_value": float(pressure_grid[j, i]),
    }


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
    diffs_arr = np.asarray(diffs, dtype="float64")
    direction_exceptions = float(min(int(np.sum(diffs_arr > 0.0)), int(np.sum(diffs_arr < 0.0))))
    passed = bool(finite_fraction >= 0.70 and tangent_fraction >= 0.55 and reversal_fraction >= 0.55 and direction_exceptions <= 2.0)
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


def _refine_speed_minimum(
    speed_grid: np.ndarray,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    center_x_km: float,
    center_y_km: float,
    window_km: float,
    refine_factor: int,
) -> dict[str, float | bool | str]:
    if refine_factor <= 1 or window_km <= 0.0:
        return {
            "refined_ok": False,
            "x_km_refined": center_x_km,
            "y_km_refined": center_y_km,
            "refined_center_speed_ms": np.nan,
            "refined_offset_km": 0.0,
            "refined_failure": "disabled",
        }
    x_axis = x_grid[0, :]
    y_axis = y_grid[:, 0]
    step = max(min(abs(float(np.nanmedian(np.diff(x_axis)))), abs(float(np.nanmedian(np.diff(y_axis))))) / float(refine_factor), 1.0e-3)
    x_fine = np.arange(center_x_km - window_km, center_x_km + window_km + 0.5 * step, step)
    y_fine = np.arange(center_y_km - window_km, center_y_km + window_km + 0.5 * step, step)
    xx, yy = np.meshgrid(x_fine, y_fine)
    interp = RegularGridInterpolator((y_axis, x_axis), speed_grid, bounds_error=False, fill_value=np.nan)
    fine = interp(np.column_stack([yy.ravel(), xx.ravel()])).reshape(xx.shape)
    finite = np.isfinite(fine)
    if finite.sum() < 4:
        return {
            "refined_ok": False,
            "x_km_refined": center_x_km,
            "y_km_refined": center_y_km,
            "refined_center_speed_ms": np.nan,
            "refined_offset_km": 0.0,
            "refined_failure": "insufficient_window_data",
        }
    edge = np.zeros(fine.shape, dtype=bool)
    edge[[0, -1], :] = True
    edge[:, [0, -1]] = True
    j, i = np.unravel_index(int(np.nanargmin(np.where(finite, fine, np.inf))), fine.shape)
    return {
        "refined_ok": not bool(edge[j, i]),
        "x_km_refined": float(xx[j, i]),
        "y_km_refined": float(yy[j, i]),
        "refined_center_speed_ms": float(fine[j, i]),
        "refined_offset_km": float(np.hypot(float(xx[j, i]) - center_x_km, float(yy[j, i]) - center_y_km)),
        "refined_failure": "minimum_on_refined_boundary" if bool(edge[j, i]) else "none",
    }


def _speed_min_candidates(speed_grid: np.ndarray, x_grid: np.ndarray, y_grid: np.ndarray, radius_km: float, search_rmax: float) -> list[tuple[int, int]]:
    search = np.isfinite(speed_grid) & (np.hypot(x_grid, y_grid) <= radius_km * float(search_rmax))
    if not search.any():
        return []
    filled = np.where(search, speed_grid, np.inf)
    local_min = filled == minimum_filter(filled, size=5, mode="nearest")
    threshold = float(np.nanpercentile(speed_grid[search], 35.0))
    jj, ii = np.where(search & local_min & (speed_grid <= threshold))
    candidates = [(int(j), int(i)) for j, i in zip(jj, ii)]
    global_j, global_i = np.unravel_index(int(np.nanargmin(filled)), filled.shape)
    candidates.append((int(global_j), int(global_i)))
    candidates = list(dict.fromkeys(candidates))
    candidates.sort(key=lambda item: float(speed_grid[item[0], item[1]]))
    return candidates[:40]


def composite_hua_refined_axis(
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
    f0: float,
    refine_factor: int,
    refine_window_km: float,
) -> pd.DataFrame:
    radius_km = float(radius_m) / 1000.0
    rows = []
    previous_xy: tuple[float, float] | None = None
    for depth_index in range(len(depth)):
        x_grid, y_grid, speed_grid = _regular_grid_from_polar(radial, theta, speed[depth_index], radius_m, grid_size)
        _, _, u_grid = _regular_grid_from_polar(radial, theta, u[depth_index], radius_m, grid_size)
        _, _, v_grid = _regular_grid_from_polar(radial, theta, v[depth_index], radius_m, grid_size)
        pressure_grid = RHO0 * f0 * _streamfunction_proxy(u_grid, v_grid, x_grid, y_grid)
        pressure_info = _pressure_extreme(pressure_grid, x_grid, y_grid, radius_km, search_rmax)
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
                    "axis_source": "composite_hua_refined",
                    "composite_hua_pass": False,
                    "composite_hua_failure": "no_search_data",
                    **pressure_info,
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
            score = float(speed_grid[j, i]) / speed_scale + 0.20 * radial_penalty**2 + 0.65 * continuity**2
            score += 0.0 if bool(check["composite_hua_pass"]) else 2.0
            scored.append((score, bool(check["composite_hua_pass"]), j, i, check))
        scored.sort(key=lambda item: item[0])
        _, _, j, i, check = ([item for item in scored if item[1]] or scored)[0]
        center_x_grid = float(x_grid[j, i])
        center_y_grid = float(y_grid[j, i])
        refine = _refine_speed_minimum(speed_grid, x_grid, y_grid, center_x_grid, center_y_grid, refine_window_km, refine_factor)
        center_x = float(refine["x_km_refined"]) if bool(refine["refined_ok"]) else center_x_grid
        center_y = float(refine["y_km_refined"]) if bool(refine["refined_ok"]) else center_y_grid
        refined_check = _composite_hua_ring_check(
            u_grid=u_grid,
            v_grid=v_grid,
            x_grid=x_grid,
            y_grid=y_grid,
            center_x_km=center_x,
            center_y_km=center_y,
            radius_km=max(radius_km * 0.55, 20.0),
        )
        previous_xy = (center_x, center_y)
        rows.append(
            {
                "depth_index": int(depth_index),
                "depth_m": float(depth[depth_index]),
                "x_km": center_x,
                "y_km": center_y,
                "x_km_grid": center_x_grid,
                "y_km_grid": center_y_grid,
                "effective_weight": float(np.isfinite(speed_grid).sum()),
                "center_speed_ms": float(refine["refined_center_speed_ms"])
                if bool(refine["refined_ok"]) and np.isfinite(float(refine["refined_center_speed_ms"]))
                else float(speed_grid[j, i]),
                "axis_source": "composite_hua_refined",
                "composite_hua_fallback": not bool(refined_check["composite_hua_pass"]),
                "candidate_count": int(len(candidates)),
                "grid_composite_hua_pass": bool(check["composite_hua_pass"]),
                "grid_composite_hua_failure": check["composite_hua_failure"],
                **refine,
                **pressure_info,
                "distance_to_pressure_extreme_km": float(
                    np.hypot(center_x - pressure_info["pressure_extreme_x_km"], center_y - pressure_info["pressure_extreme_y_km"])
                ),
                **refined_check,
            }
        )
    axis = pd.DataFrame(rows)
    axis = axis[np.isfinite(axis["x_km"]) & np.isfinite(axis["y_km"])].copy()
    if axis.empty:
        raise ValueError("No composite-Hua representative axis could be extracted")
    return axis.sort_values("depth_index").reset_index(drop=True)


def axis_source_comparison(radial_axis: pd.DataFrame, composite_axis: pd.DataFrame) -> pd.DataFrame:
    composite_cols = ["depth_index", "x_km", "y_km", "composite_hua_pass", "composite_hua_failure"]
    for col in ("x_km_grid", "y_km_grid", "refined_offset_km", "refined_ok", "pressure_extreme_x_km", "pressure_extreme_y_km", "distance_to_pressure_extreme_km"):
        if col in composite_axis.columns:
            composite_cols.append(col)
    merged = radial_axis[["depth_index", "depth_m", "x_km", "y_km"]].merge(
        composite_axis[composite_cols],
        on="depth_index",
        how="inner",
        suffixes=("_radial_seed", "_composite_hua_refined"),
    )
    merged["axis_source_offset_km"] = np.hypot(
        merged["x_km_composite_hua_refined"] - merged["x_km_radial_seed"],
        merged["y_km_composite_hua_refined"] - merged["y_km_radial_seed"],
    )
    return merged


def build_representative_axis_sources_for_root(
    *,
    me_liutex_root: Path,
    radial_seed_root: Path,
    orientation: str,
    tau: float,
    axis_bandwidth: float = 0.075,
    grid_size: int = 121,
    reference_lat: float = 28.0,
    composite_hua_search_rmax: float = 1.5,
    composite_hua_refine_factor: int = 4,
    composite_hua_refine_window_km: float = 20.0,
) -> list[Path]:
    dataset = RepresentativeVortexDataset.load(
        Path(me_liutex_root) / "azimuthal_representative_velocity.npz",
        Path(radial_seed_root),
    )
    tau_i = dataset.nearest_tau_index(float(tau))
    tau_value = float(dataset.tau_grid[tau_i])
    radius_by_polarity = _load_radius_by_polarity(radial_seed_root)
    f0 = 2.0 * EARTH_OMEGA * np.sin(np.deg2rad(float(reference_lat)))
    out_dir = axis_sources_dir(me_liutex_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    radial_rows: list[pd.DataFrame] = []
    composite_rows: list[pd.DataFrame] = []
    comparison_rows: list[pd.DataFrame] = []
    polarity_summaries: list[dict] = []
    for ip, polarity in enumerate(dataset.polarities):
        radius_m = radius_by_polarity.get(polarity, dataset.radius_by_polarity_m.get(polarity, 80000.0))
        radial_axis = radial_seed_axis_by_tau(radial_seed_root, polarity, tau_value, axis_bandwidth, orientation).copy()
        radial_axis.insert(0, "polarity", polarity)
        radial_axis.insert(1, "orientation", orientation)
        radial_axis.insert(2, "tau", tau_value)
        radial_axis["axis_source"] = "radial_seed"
        radial_rows.append(radial_axis)

        composite_axis = composite_hua_refined_axis(
            radial=dataset.radius_coord,
            theta=dataset.theta_rad,
            depth=dataset.depth_m,
            u=dataset.u_mean[ip, tau_i],
            v=dataset.v_mean[ip, tau_i],
            speed=dataset.speed_mean[ip, tau_i],
            radius_m=radius_m,
            grid_size=grid_size,
            search_rmax=composite_hua_search_rmax,
            f0=f0,
            refine_factor=composite_hua_refine_factor,
            refine_window_km=composite_hua_refine_window_km,
        ).copy()
        composite_axis.insert(0, "polarity", polarity)
        composite_axis.insert(1, "orientation", orientation)
        composite_axis.insert(2, "tau", tau_value)
        composite_rows.append(composite_axis)

        comparison = axis_source_comparison(
            radial_axis.drop(columns=["polarity", "orientation", "tau"], errors="ignore"),
            composite_axis.drop(columns=["polarity", "orientation", "tau"], errors="ignore"),
        )
        comparison.insert(0, "polarity", polarity)
        comparison.insert(1, "orientation", orientation)
        comparison.insert(2, "tau", tau_value)
        comparison_rows.append(comparison)
        polarity_summaries.append(
            {
                "polarity": polarity,
                "radial_seed_layers": int(len(radial_axis)),
                "composite_hua_refined_layers": int(len(composite_axis)),
                "composite_hua_pass_count": int(composite_axis["composite_hua_pass"].sum()),
                "median_axis_source_offset_km": float(np.nanmedian(comparison["axis_source_offset_km"])),
                "p90_axis_source_offset_km": float(np.nanpercentile(comparison["axis_source_offset_km"], 90)),
            }
        )

    tag = tau_tag(tau_value)
    radial_path = out_dir / f"radial_seed_axis_{tag}.csv"
    composite_path = out_dir / f"composite_hua_refined_axis_{tag}.csv"
    comparison_path = out_dir / f"axis_source_comparison_{tag}.csv"
    manifest_path = out_dir / f"axis_sources_manifest_{tag}.json"
    pd.concat(radial_rows, ignore_index=True).to_csv(radial_path, index=False)
    pd.concat(composite_rows, ignore_index=True).to_csv(composite_path, index=False)
    pd.concat(comparison_rows, ignore_index=True).to_csv(comparison_path, index=False)
    manifest = {
        "me_liutex_root": str(me_liutex_root),
        "radial_seed_root": str(radial_seed_root),
        "orientation": orientation,
        "tau": tau_value,
        "axis_sources": ["radial_seed", "composite_hua_refined"],
        "default_axis_source": "radial_seed",
        "composite_hua_refined": {
            "search_rmax": composite_hua_search_rmax,
            "refine_factor": composite_hua_refine_factor,
            "refine_window_km": composite_hua_refine_window_km,
        },
        "files": {
            "radial_seed_axis": str(radial_path),
            "composite_hua_refined_axis": str(composite_path),
            "axis_source_comparison": str(comparison_path),
        },
        "polarity_summaries": polarity_summaries,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return [radial_path, composite_path, comparison_path, manifest_path]
