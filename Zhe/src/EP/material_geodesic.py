from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np

try:
    import pandas as pd
except ModuleNotFoundError:  # pragma: no cover - dependency-light dry-run support.
    pd = None

try:
    from scipy import ndimage
except ModuleNotFoundError:  # pragma: no cover - checked at runtime.
    ndimage = None

from .contracts import DEFAULT_RESULT_ROOT
from .boundary_strategy import resolve_boundary_strategy
from .dynamic_boundary import boundary_flux_metrics, connected_component, edge_mask, neighbors4


GEODESIC_BOUNDARY_MODES = (
    "cauchy_green_geodesic_v1",
    "lavd_material_v1",
    "hybrid_geodesic_lavd_v1",
    "pv_retention_geodesic_v1",
    "pv_retention_lavd_v1",
    "pv_retention_hybrid_v1",
)
BOUNDARY_BUDGETS = ("edge_proxy", "full_3d")
DEFAULT_GEODESIC_OUTPUT_ROOT = Path(
    "/root/autodl-fs/kuroshiou/EP-FLUX/object_material_geodesic_ep_validation"
)


@dataclass(frozen=True)
class MaterialGeodesicRequest:
    result_root: Path = DEFAULT_RESULT_ROOT
    filter_root: Path = Path("/root/autodl-fs/kuroshiou/Filter")
    output_root: Path = DEFAULT_GEODESIC_OUTPUT_ROOT
    shapes: tuple[str, ...] = ("coherent", "upright_like")
    orientations: tuple[str, ...] = ("turned",)
    buoyancy_sources: tuple[str, ...] = ("thermal_wind",)
    boundary_modes: tuple[str, ...] = GEODESIC_BOUNDARY_MODES
    boundary_budget: str = "full_3d"
    filter_template: str = "global_phy_{year}_bandpass_30_180d.nc"
    radial_bins: int = 18
    azimuth_bins: int = 36
    rmax: float = 1.5
    reference_lat: float = 30.0
    constant_n2: float = 2.0e-5
    core_radius_over_R: float = 1.5
    speed_core_quantile: float = 0.45
    pv_core_quantile: float = 0.70
    min_mask_fraction: float = 0.01
    min_core_retention: float = 0.75
    min_pv_retention: float = 0.75
    pv_retention_weight: float = 0.80
    weak_retention_weight: float = 0.40
    particle_retention_weight: float = 0.20
    require_pv_retention: bool = False
    min_area_fraction: float = 0.10
    max_area_fraction: float = 0.75
    trajectory_window_days: int = 7
    particle_spacing_km: float = 5.0
    advection_step_hours: float = 6.0
    max_tracks_per_shape: int = 0
    max_objectdays: int = 0
    max_depth_layers: int = 0
    skip_missing: bool = False
    dry_run: bool = False


def _split_csv(value: str | tuple[str, ...] | list[str]) -> tuple[str, ...]:
    if isinstance(value, (tuple, list)):
        return tuple(str(v).strip() for v in value if str(v).strip())
    return tuple(part.strip() for part in str(value).split(",") if part.strip())


def _validate_request(request: MaterialGeodesicRequest) -> None:
    unknown = [mode for mode in request.boundary_modes if mode not in GEODESIC_BOUNDARY_MODES]
    if unknown:
        raise ValueError(f"boundary modes must be in {GEODESIC_BOUNDARY_MODES}: {unknown}")
    for mode in request.boundary_modes:
        resolve_boundary_strategy(mode)
    if request.boundary_budget not in BOUNDARY_BUDGETS:
        raise ValueError(f"boundary budget must be one of {BOUNDARY_BUDGETS}")
    if request.trajectory_window_days < 1:
        raise ValueError("trajectory_window_days must be positive")
    if request.advection_step_hours <= 0:
        raise ValueError("advection_step_hours must be positive")
    if not 0.0 < request.min_pv_retention <= 1.0:
        raise ValueError("min_pv_retention must be in (0, 1]")
    for name, value in {
        "pv_retention_weight": request.pv_retention_weight,
        "weak_retention_weight": request.weak_retention_weight,
        "particle_retention_weight": request.particle_retention_weight,
    }.items():
        if value < 0:
            raise ValueError(f"{name} must be non-negative")


def _require_runtime() -> None:
    if pd is None:
        raise ModuleNotFoundError("pandas is required for material geodesic validation")
    if ndimage is None:
        raise ModuleNotFoundError("scipy is required for material geodesic validation")


def _load_runtime_helpers() -> None:
    global MaterialCoherenceRequest
    global _base_material_request
    global _closure_residual_table
    global _compute_base_record
    global _core_domain_from_row
    global _finite_mean
    global _json_ready
    global _load_shape_objects
    global _polar_grid
    global _vorticity_proxy
    global _write_table
    global _full_boundary_flux_budget
    global _weighted_mean

    from .material_coherence import (
        MaterialCoherenceRequest,
        _load_runtime_helpers as _load_coherence_runtime_helpers,
        _base_material_request,
        _closure_residual_table,
        _compute_base_record,
        _finite_mean,
        _vorticity_proxy,
    )
    from .object_material_boundary import _load_shape_objects, _polar_grid
    from .material_volume import _full_boundary_flux_budget, _json_ready, _weighted_mean, _write_table
    _load_coherence_runtime_helpers()


def _coherence_request(request: MaterialGeodesicRequest, shape: str, orientation: str, buoyancy_source: str):
    return MaterialCoherenceRequest(
        result_root=request.result_root,
        filter_root=request.filter_root,
        output_root=request.output_root,
        shapes=(shape,),
        orientations=(orientation,),
        buoyancy_sources=(buoyancy_source,),
        boundary_modes=("particle_retention_v1",),
        boundary_budget=request.boundary_budget,
        filter_template=request.filter_template,
        radial_bins=request.radial_bins,
        azimuth_bins=request.azimuth_bins,
        rmax=request.rmax,
        reference_lat=request.reference_lat,
        constant_n2=request.constant_n2,
        core_radius_over_R=request.core_radius_over_R,
        speed_core_quantile=request.speed_core_quantile,
        pv_core_quantile=request.pv_core_quantile,
        min_mask_fraction=request.min_mask_fraction,
        min_core_retention=request.min_core_retention,
        min_area_fraction=request.min_area_fraction,
        max_area_fraction=request.max_area_fraction,
        trajectory_window_days=request.trajectory_window_days,
        particle_spacing_km=request.particle_spacing_km,
        advection_step_hours=request.advection_step_hours,
        max_tracks_per_shape=request.max_tracks_per_shape,
        max_objectdays=request.max_objectdays,
        skip_missing=request.skip_missing,
        dry_run=request.dry_run,
    )


def _mesh_spacing_km(x_km: np.ndarray, y_km: np.ndarray) -> float:
    values: list[float] = []
    if x_km.shape[0] > 1:
        values.append(float(np.nanmedian(np.hypot(np.diff(x_km, axis=0), np.diff(y_km, axis=0)))))
    if x_km.shape[1] > 1:
        values.append(float(np.nanmedian(np.hypot(np.diff(x_km, axis=1), np.diff(y_km, axis=1)))))
    values = [value for value in values if np.isfinite(value) and value > 0]
    return float(np.nanmedian(values)) if values else 5.0


def _safe_nanmedian(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float("nan")
    return float(np.median(finite))


def _sample_polar_scalar(field: np.ndarray, x_km: np.ndarray, y_km: np.ndarray, radius_m: float, radial_coord: np.ndarray) -> np.ndarray:
    if ndimage is None:
        raise ModuleNotFoundError("scipy is required for material geodesic validation")
    radial = np.asarray(radial_coord, dtype=float)
    r_over_r = np.hypot(x_km, y_km) * 1000.0 / max(float(radius_m), 1.0)
    theta = np.mod(np.arctan2(y_km, x_km), 2.0 * np.pi)
    dr = float(np.nanmedian(np.diff(radial))) if radial.size > 1 else 1.0
    dtheta = 2.0 * np.pi / field.shape[1]
    coords = np.vstack(
        [
            ((r_over_r - radial[0]) / dr).ravel(),
            (theta / dtheta).ravel(),
        ]
    )
    valid = np.isfinite(coords).all(axis=0) & (coords[0] >= 0) & (coords[0] <= field.shape[0] - 1)
    safe = np.where(np.isfinite(field), field, np.nan)
    filled = np.where(np.isfinite(safe), safe, np.nanmedian(safe[np.isfinite(safe)]) if np.any(np.isfinite(safe)) else 0.0)
    out = np.full(coords.shape[1], np.nan, dtype=float)
    if np.any(valid):
        out[valid] = ndimage.map_coordinates(filled, coords[:, valid], order=1, mode="wrap")
    return out.reshape(x_km.shape)


def _time_interpolated_layer(records: list[dict[str, object]], center_index: int, t_days: float, depth_index: int):
    center_day = records[center_index]["day"]
    target_ord = center_day.toordinal() + float(t_days)
    ords = np.asarray([record["day"].toordinal() for record in records], dtype=float)
    if target_ord < ords[0] or target_ord > ords[-1]:
        return None
    right = int(np.searchsorted(ords, target_ord, side="left"))
    if right <= 0:
        left = right = 0
    elif right >= len(records):
        left = right = len(records) - 1
    else:
        left = right - 1
    if left == right:
        weight = 0.0
    else:
        weight = float((target_ord - ords[left]) / max(ords[right] - ords[left], 1e-12))
    rep_l = records[left]["rep"]
    rep_r = records[right]["rep"]
    if depth_index >= rep_l.u.shape[0] or depth_index >= rep_r.u.shape[0]:
        return None
    u = (1.0 - weight) * rep_l.u[depth_index] + weight * rep_r.u[depth_index]
    v = (1.0 - weight) * rep_l.v[depth_index] + weight * rep_r.v[depth_index]
    return u, v, rep_l.radius_m, rep_l.radius_coord


def _velocity_at(
    records: list[dict[str, object]],
    center_index: int,
    t_days: float,
    depth_index: int,
    x_km: np.ndarray,
    y_km: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    layer = _time_interpolated_layer(records, center_index, t_days, depth_index)
    if layer is None:
        nan = np.full_like(x_km, np.nan, dtype=float)
        return nan, nan, np.zeros_like(x_km, dtype=bool)
    u, v, radius_m, radial = layer
    us = _sample_polar_scalar(u, x_km, y_km, radius_m, radial)
    vs = _sample_polar_scalar(v, x_km, y_km, radius_m, radial)
    valid = np.isfinite(us) & np.isfinite(vs)
    return us, vs, valid


def _rk4_step(
    records: list[dict[str, object]],
    center_index: int,
    t_days: float,
    dt_days: float,
    depth_index: int,
    x_km: np.ndarray,
    y_km: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dt_seconds = dt_days * 86400.0

    def rhs(t: float, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        u, v, ok = _velocity_at(records, center_index, t, depth_index, x, y)
        return u / 1000.0, v / 1000.0, ok

    k1x, k1y, ok1 = rhs(t_days, x_km, y_km)
    k2x, k2y, ok2 = rhs(t_days + 0.5 * dt_days, x_km + 0.5 * dt_seconds * k1x, y_km + 0.5 * dt_seconds * k1y)
    k3x, k3y, ok3 = rhs(t_days + 0.5 * dt_days, x_km + 0.5 * dt_seconds * k2x, y_km + 0.5 * dt_seconds * k2y)
    k4x, k4y, ok4 = rhs(t_days + dt_days, x_km + dt_seconds * k3x, y_km + dt_seconds * k3y)
    valid = ok1 & ok2 & ok3 & ok4
    x_new = x_km + dt_seconds * (k1x + 2.0 * k2x + 2.0 * k3x + k4x) / 6.0
    y_new = y_km + dt_seconds * (k1y + 2.0 * k2y + 2.0 * k3y + k4y) / 6.0
    return x_new, y_new, valid & np.isfinite(x_new) & np.isfinite(y_new)


def _integrate_flow_map(
    records: list[dict[str, object]],
    center_index: int,
    depth_index: int,
    x0_km: np.ndarray,
    y0_km: np.ndarray,
    *,
    direction: int,
    window_days: int,
    step_hours: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    dt_days = math.copysign(float(step_hours) / 24.0, float(direction))
    steps = int(math.ceil(float(window_days) * 24.0 / float(step_hours)))
    x = np.asarray(x0_km, dtype=float).copy()
    y = np.asarray(y0_km, dtype=float).copy()
    valid = np.isfinite(x) & np.isfinite(y)
    t = 0.0
    completed = 0.0
    for _ in range(steps):
        x_next, y_next, ok = _rk4_step(records, center_index, t, dt_days, depth_index, x, y)
        valid &= ok
        x = np.where(valid, x_next, np.nan)
        y = np.where(valid, y_next, np.nan)
        t += dt_days
        completed = abs(t)
        if not np.any(valid):
            break
    return x, y, valid, completed


def _integrate_lavd(
    records: list[dict[str, object]],
    center_index: int,
    depth_index: int,
    x0_km: np.ndarray,
    y0_km: np.ndarray,
    *,
    window_days: int,
    step_hours: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    rep0 = records[center_index]["rep"]
    x = np.asarray(x0_km, dtype=float).copy()
    y = np.asarray(y0_km, dtype=float).copy()
    valid = np.isfinite(x) & np.isfinite(y)
    valid_total = valid.copy()
    lavd = np.zeros_like(x, dtype=float)
    dt_days = float(step_hours) / 24.0
    steps_each = int(math.ceil(float(window_days) * 24.0 / float(step_hours)))
    completed = 0.0
    for direction in (-1, 1):
        x_dir = x.copy()
        y_dir = y.copy()
        valid_dir = valid.copy()
        t = 0.0
        for _ in range(steps_each):
            layer = _time_interpolated_layer(records, center_index, t, depth_index)
            if layer is None:
                break
            u, v, radius_m, radial = layer
            zeta = _vorticity_single_layer(u, v, rep0.radial_m, rep0.theta_rad)
            bg = np.nanmedian(zeta[np.isfinite(zeta)])
            zeta_s = _sample_polar_scalar(zeta, x_dir, y_dir, radius_m, radial)
            valid_dir &= np.isfinite(zeta_s)
            valid_total &= valid_dir
            lavd += np.where(valid_dir, np.abs(zeta_s - bg) * dt_days * 86400.0, 0.0)
            x_dir, y_dir, ok = _rk4_step(records, center_index, t, math.copysign(dt_days, direction), depth_index, x_dir, y_dir)
            valid_dir &= ok
            valid_total &= valid_dir
            t += math.copysign(dt_days, direction)
            completed = max(completed, abs(t))
    return lavd, valid_total, completed


def _vorticity_single_layer(u: np.ndarray, v: np.ndarray, radial_m: np.ndarray, theta: np.ndarray) -> np.ndarray:
    radial = np.asarray(radial_m, dtype=float).copy()
    if radial.size > 1 and radial[0] <= 0:
        radial[0] = radial[1] * 0.5
    d_r_v_dr = np.gradient(radial[:, None] * v, radial, axis=0, edge_order=1)
    d_u_dtheta = (np.roll(u, -1, axis=1) - np.roll(u, 1, axis=1)) / (2.0 * (2.0 * np.pi / u.shape[1]))
    return (d_r_v_dr - d_u_dtheta) / radial[:, None]


def _flow_map_cauchy_green(
    x0: np.ndarray,
    y0: np.ndarray,
    x1: np.ndarray,
    y1: np.ndarray,
    valid: np.ndarray,
) -> dict[str, np.ndarray]:
    x0r = np.asarray(x0, dtype=float)
    y0r = np.asarray(y0, dtype=float)
    x1r = np.asarray(x1, dtype=float)
    y1r = np.asarray(y1, dtype=float)
    d_x0_a = np.gradient(x0r, axis=0, edge_order=1)
    d_x0_b = np.gradient(x0r, axis=1, edge_order=1)
    d_y0_a = np.gradient(y0r, axis=0, edge_order=1)
    d_y0_b = np.gradient(y0r, axis=1, edge_order=1)
    d_x1_a = np.gradient(x1r, axis=0, edge_order=1)
    d_x1_b = np.gradient(x1r, axis=1, edge_order=1)
    d_y1_a = np.gradient(y1r, axis=0, edge_order=1)
    d_y1_b = np.gradient(y1r, axis=1, edge_order=1)
    det0 = d_x0_a * d_y0_b - d_x0_b * d_y0_a
    good = valid & np.isfinite(det0) & (np.abs(det0) > 1e-10)
    f11 = np.full_like(x0r, np.nan, dtype=float)
    f12 = np.full_like(x0r, np.nan, dtype=float)
    f21 = np.full_like(x0r, np.nan, dtype=float)
    f22 = np.full_like(x0r, np.nan, dtype=float)
    f11[good] = (d_x1_a[good] * d_y0_b[good] - d_x1_b[good] * d_y0_a[good]) / det0[good]
    f12[good] = (-d_x1_a[good] * d_x0_b[good] + d_x1_b[good] * d_x0_a[good]) / det0[good]
    f21[good] = (d_y1_a[good] * d_y0_b[good] - d_y1_b[good] * d_y0_a[good]) / det0[good]
    f22[good] = (-d_y1_a[good] * d_x0_b[good] + d_y1_b[good] * d_x0_a[good]) / det0[good]
    c11 = f11 * f11 + f21 * f21
    c12 = f11 * f12 + f21 * f22
    c22 = f12 * f12 + f22 * f22
    trace = c11 + c22
    discr = np.maximum(0.0, (c11 - c22) ** 2 + 4.0 * c12 * c12)
    root = np.sqrt(discr)
    lam1 = 0.5 * (trace - root)
    lam2 = 0.5 * (trace + root)
    for arr in (f11, f12, f21, f22, c11, c12, c22, lam1, lam2):
        arr[~good] = np.nan
    return {
        "F11": f11,
        "F12": f12,
        "F21": f21,
        "F22": f22,
        "C11": c11,
        "C12": c12,
        "C22": c22,
        "lambda1": lam1,
        "lambda2": lam2,
        "valid_cg": good,
    }


def _seed_from_axis(x_km: np.ndarray, y_km: np.ndarray, candidate: np.ndarray) -> tuple[int, int] | None:
    distance = np.where(candidate, np.hypot(x_km, y_km), np.inf)
    if not np.any(np.isfinite(distance)):
        return None
    return tuple(int(i) for i in np.unravel_index(int(np.nanargmin(distance)), distance.shape))


def _mask_area_fraction(mask: np.ndarray, domain: np.ndarray) -> float:
    denom = max(1, int(np.count_nonzero(domain)))
    return float(np.count_nonzero(mask) / denom)


def _closed_component(candidate: np.ndarray, seed: tuple[int, int] | None) -> tuple[np.ndarray, str]:
    if seed is None:
        return np.zeros_like(candidate, dtype=bool), "missing_seed"
    component = connected_component(candidate.astype(bool), seed)
    if not np.any(component):
        return component, "empty_component"
    if np.any(component[-1, :]):
        return component, "touches_outer_boundary"
    return component, "closed_component"


def _retention_for_mask(mask: np.ndarray, x_end: np.ndarray, y_end: np.ndarray, x_km: np.ndarray, y_km: np.ndarray, radius_m: float, radial_coord: np.ndarray) -> float:
    if not np.any(mask):
        return np.nan
    sampled = _sample_polar_scalar(mask.astype(float), x_end, y_end, radius_m, radial_coord)
    inside = sampled >= 0.5
    valid = mask & np.isfinite(sampled)
    if not np.any(valid):
        return np.nan
    return float(np.count_nonzero(inside & valid) / np.count_nonzero(valid))


def _candidate_from_stretch(
    lambda2: np.ndarray,
    core_domain: np.ndarray,
    x_km: np.ndarray,
    y_km: np.ndarray,
    *,
    min_area_fraction: float,
    max_area_fraction: float,
) -> list[tuple[np.ndarray, dict[str, float | str]]]:
    valid = core_domain & np.isfinite(lambda2) & (lambda2 > 0)
    if not np.any(valid):
        return []
    stretch_error = np.abs(np.log(lambda2))
    seed = _seed_from_axis(x_km, y_km, valid)
    candidates: list[tuple[np.ndarray, dict[str, float | str]]] = []
    values = stretch_error[valid]
    for q in (0.20, 0.30, 0.40, 0.50, 0.60):
        threshold = float(np.nanquantile(values, q))
        component, status = _closed_component(valid & (stretch_error <= threshold), seed)
        area_fraction = _mask_area_fraction(component, valid)
        if area_fraction < min_area_fraction or area_fraction > max_area_fraction:
            continue
        candidates.append(
            (
                component,
                {
                    "geodesic_status": status,
                    "geodesic_threshold_quantile": q,
                    "mean_lambda2": _finite_mean(lambda2[component]),
                    "mean_log_lambda2_abs": _finite_mean(stretch_error[component]),
                    "closed_geodesic_found": status == "closed_component",
                    "closed_geodesic_area_fraction": area_fraction,
                    "lambda_line_solver": "closed_low_stretch_candidate_from_cauchy_green_v1",
                },
            )
        )
    return candidates


def _candidate_from_lavd(
    lavd: np.ndarray,
    core_domain: np.ndarray,
    x_km: np.ndarray,
    y_km: np.ndarray,
    *,
    min_area_fraction: float,
    max_area_fraction: float,
) -> list[tuple[np.ndarray, dict[str, float | str]]]:
    valid = core_domain & np.isfinite(lavd)
    if not np.any(valid):
        return []
    seed = _seed_from_axis(x_km, y_km, valid)
    candidates: list[tuple[np.ndarray, dict[str, float | str]]] = []
    values = lavd[valid]
    for q in (0.50, 0.60, 0.70, 0.80):
        threshold = float(np.nanquantile(values, q))
        component, status = _closed_component(valid & (lavd >= threshold), seed)
        area_fraction = _mask_area_fraction(component, valid)
        if area_fraction < min_area_fraction or area_fraction > max_area_fraction:
            continue
        candidates.append(
            (
                component,
                {
                    "lavd_boundary_status": status,
                    "lavd_threshold_quantile": q,
                    "lavd_mean": _finite_mean(lavd[component]),
                    "lavd_p90": float(np.nanpercentile(lavd[component], 90)) if np.any(component) else np.nan,
                    "lavd_closed_contour_found": status == "closed_component",
                    "lavd_area_fraction": area_fraction,
                    "lavd_integral_model": "particle_trajectory_relative_vorticity_deviation_v1",
                },
            )
        )
    return candidates


def _mask_flux_score(
    mask: np.ndarray,
    rep,
    q_proxy: np.ndarray,
    buoyancy: np.ndarray,
    depth_index: int,
    x_km: np.ndarray,
    y_km: np.ndarray,
) -> dict[str, float]:
    theta_prime = rep.theta_prime[depth_index] if rep.theta_prime is not None else np.full_like(rep.u[depth_index], np.nan)
    mean_u = _weighted_mean(rep.u[depth_index], mask)
    mean_v = _weighted_mean(rep.v[depth_index], mask)
    internal = float(
        np.nanmedian(
            np.abs(rep.u[depth_index][mask] * q_proxy[depth_index][mask])
            + np.abs(rep.v[depth_index][mask] * q_proxy[depth_index][mask])
        )
    ) if np.any(mask) else np.nan
    flux = boundary_flux_metrics(
        mask=mask,
        u=rep.u[depth_index],
        v=rep.v[depth_index],
        buoyancy=buoyancy[depth_index],
        q_proxy=q_proxy[depth_index],
        theta_prime=theta_prime,
        x_km=x_km,
        y_km=y_km,
        mean_u=mean_u,
        mean_v=mean_v,
        internal_flux_scale=internal if np.isfinite(internal) and internal > 0 else 1.0,
    )
    return {
        "leakage_mean_abs_ms": float(flux.get("leakage_mean_abs_ms", np.nan)),
        "boundary_flux_over_internal_flux": float(flux.get("boundary_flux_over_internal_flux", np.nan)),
        "edge_cell_count": float(flux.get("edge_cell_count", np.nan)),
    }


def _core_retention(mask: np.ndarray, core_domain: np.ndarray, weak_core: np.ndarray, pv_core: np.ndarray) -> tuple[float, float]:
    weak_total = max(1, int(np.count_nonzero(weak_core & core_domain)))
    pv_total = max(1, int(np.count_nonzero(pv_core & core_domain)))
    return (
        float(np.count_nonzero(mask & weak_core & core_domain) / weak_total),
        float(np.count_nonzero(mask & pv_core & core_domain) / pv_total),
    )


def _pv_retention_audit(
    mask: np.ndarray,
    q_layer: np.ndarray,
    core_domain: np.ndarray,
    pv_core: np.ndarray,
    x_km: np.ndarray,
    y_km: np.ndarray,
) -> dict[str, float | str]:
    q_abs = np.abs(q_layer)
    finite_core = core_domain & np.isfinite(q_abs)
    finite_pv_core = pv_core & np.isfinite(q_abs)
    pv_abs_total = float(np.nansum(q_abs[finite_core]))
    pv_high_total = float(np.nansum(q_abs[finite_pv_core]))
    pv_abs_retention = float(np.nansum(q_abs[mask & finite_core]) / (pv_abs_total + 1e-12))
    pv_high_retention = float(np.nansum(q_abs[mask & finite_pv_core]) / (pv_high_total + 1e-12))

    if not np.any(finite_pv_core):
        return {
            "pv_abs_retention": pv_abs_retention,
            "pv_high_quantile_retention": pv_high_retention,
            "pv_centroid_inside_mask": "false",
            "pv_centroid_x_km": np.nan,
            "pv_centroid_y_km": np.nan,
            "distance_mask_to_pv_centroid_km": np.nan,
        }

    weights = q_abs[finite_pv_core]
    px = float(np.nansum(x_km[finite_pv_core] * weights) / (np.nansum(weights) + 1e-12))
    py = float(np.nansum(y_km[finite_pv_core] * weights) / (np.nansum(weights) + 1e-12))
    nearest = np.nanargmin((x_km - px) ** 2 + (y_km - py) ** 2)
    nearest_ij = np.unravel_index(int(nearest), x_km.shape)
    inside = bool(mask[nearest_ij])
    if np.any(mask):
        distance = float(np.nanmin(np.hypot(x_km[mask] - px, y_km[mask] - py)))
    else:
        distance = np.nan
    return {
        "pv_abs_retention": pv_abs_retention,
        "pv_high_quantile_retention": pv_high_retention,
        "pv_centroid_inside_mask": "true" if inside else "false",
        "pv_centroid_x_km": px,
        "pv_centroid_y_km": py,
        "distance_mask_to_pv_centroid_km": distance,
    }


def _select_best_candidate(
    candidates: list[tuple[np.ndarray, dict[str, float | str]]],
    *,
    rep,
    q_proxy: np.ndarray,
    buoyancy: np.ndarray,
    depth_index: int,
    x_km: np.ndarray,
    y_km: np.ndarray,
    core_domain: np.ndarray,
    weak_core: np.ndarray,
    pv_core: np.ndarray,
    forward_map: tuple[np.ndarray, np.ndarray, np.ndarray, float],
    backward_map: tuple[np.ndarray, np.ndarray, np.ndarray, float],
    request: MaterialGeodesicRequest,
) -> tuple[np.ndarray, dict[str, float | str]]:
    if not candidates:
        return np.zeros_like(core_domain, dtype=bool), {"boundary_status": "no_candidate"}
    best_mask = np.zeros_like(core_domain, dtype=bool)
    best_meta: dict[str, float | str] = {}
    best_score = np.inf
    xf, yf, vf, days_f = forward_map
    xb, yb, vb, days_b = backward_map
    for mask, meta in candidates:
        if not np.any(mask):
            continue
        weak_ret, pv_ret = _core_retention(mask, core_domain, weak_core, pv_core)
        pv_audit = _pv_retention_audit(mask, q_proxy[depth_index], core_domain, pv_core, x_km, y_km)
        ret_f = _retention_for_mask(mask, xf, yf, x_km, y_km, rep.radius_m, rep.radius_coord)
        ret_b = _retention_for_mask(mask, xb, yb, x_km, y_km, rep.radius_m, rep.radius_coord)
        ret_values = np.asarray([ret_f, ret_b], dtype=float)
        ret_values = ret_values[np.isfinite(ret_values)]
        particle_ret = float(np.mean(ret_values)) if ret_values.size else np.nan
        flux = _mask_flux_score(mask, rep, q_proxy, buoyancy, depth_index, x_km, y_km)
        stretch_penalty = float(meta.get("mean_log_lambda2_abs", 0.0) or 0.0)
        closure_penalty = 0.0 if bool(meta.get("closed_geodesic_found", meta.get("lavd_closed_contour_found", False))) else 0.25
        retention_penalty = 1.0 - particle_ret if np.isfinite(particle_ret) else 1.0
        if bool(meta.get("pv_retention_focused", False)):
            pv_loss = max(0.0, request.min_pv_retention - pv_ret)
            weak_loss = max(0.0, request.min_core_retention - weak_ret)
            if request.require_pv_retention and pv_ret < request.min_pv_retention:
                pv_loss += 10.0 * (request.min_pv_retention - pv_ret)
            score = (
                flux["leakage_mean_abs_ms"]
                + request.particle_retention_weight * retention_penalty
                + request.pv_retention_weight * pv_loss
                + request.weak_retention_weight * weak_loss
                + 0.05 * stretch_penalty
                + closure_penalty
            )
        else:
            core_penalty = max(0.0, request.min_core_retention - min(weak_ret, pv_ret))
            score = (
                flux["leakage_mean_abs_ms"]
                + 0.20 * retention_penalty
                + 0.15 * core_penalty
                + 0.05 * stretch_penalty
                + closure_penalty
            )
        if score < best_score:
            best_score = float(score)
            best_mask = mask
            best_meta = {
                **meta,
                **flux,
                "boundary_status": "ok",
                "selection_score": float(score),
                "particle_retention_forward": ret_f,
                "particle_retention_backward": ret_b,
                "particle_retention_mean": particle_ret,
                "particle_escape_fraction": float(1.0 - particle_ret) if np.isfinite(particle_ret) else np.nan,
                "forward_days_integrated": float(days_f),
                "backward_days_integrated": float(days_b),
                "forward_valid_fraction": float(np.nanmean(vf)),
                "backward_valid_fraction": float(np.nanmean(vb)),
                "weak_core_retention": weak_ret,
                "pv_core_retention": pv_ret,
                **pv_audit,
                "mask_fraction": _mask_area_fraction(mask, core_domain),
            }
    return best_mask, best_meta if best_meta else {"boundary_status": "no_scored_candidate"}


def _layer_geodesic_boundary(
    *,
    records: list[dict[str, object]],
    center_index: int,
    depth_index: int,
    mode: str,
    request: MaterialGeodesicRequest,
) -> tuple[np.ndarray, dict[str, float | str]]:
    record = records[center_index]
    rep = record["rep"]
    q_proxy = record["debug"]["q_proxy"]
    buoyancy = record["debug"]["buoyancy"]
    base_mask = record["debug"]["mask"][depth_index].astype(bool)
    x_km, y_km = rep.mesh_xy_km
    finite = np.isfinite(rep.speed[depth_index]) & np.isfinite(q_proxy[depth_index])
    core_domain = finite & ((rep.radial_m[:, None] / max(float(rep.radius_m), 1.0)) <= request.core_radius_over_R)
    if not np.any(core_domain):
        return np.zeros_like(finite, dtype=bool), {"boundary_status": "empty_core_domain"}

    speed_values = rep.speed[depth_index][core_domain]
    pv_values = np.abs(q_proxy[depth_index][core_domain])
    speed_threshold = float(np.nanquantile(speed_values, request.speed_core_quantile))
    pv_threshold = float(np.nanquantile(pv_values, request.pv_core_quantile))
    weak_core = core_domain & (rep.speed[depth_index] <= speed_threshold)
    pv_core = core_domain & (np.abs(q_proxy[depth_index]) >= pv_threshold)

    forward = _integrate_flow_map(
        records,
        center_index,
        depth_index,
        x_km,
        y_km,
        direction=1,
        window_days=request.trajectory_window_days,
        step_hours=request.advection_step_hours,
    )
    backward = _integrate_flow_map(
        records,
        center_index,
        depth_index,
        x_km,
        y_km,
        direction=-1,
        window_days=request.trajectory_window_days,
        step_hours=request.advection_step_hours,
    )
    cg = _flow_map_cauchy_green(x_km, y_km, forward[0], forward[1], forward[2])
    lavd, lavd_valid, lavd_days = _integrate_lavd(
        records,
        center_index,
        depth_index,
        x_km,
        y_km,
        window_days=request.trajectory_window_days,
        step_hours=request.advection_step_hours,
    )

    candidate_family = mode.replace("pv_retention_", "")
    if candidate_family == "geodesic_v1":
        candidate_family = "cauchy_green_geodesic_v1"
    if candidate_family == "lavd_v1":
        candidate_family = "lavd_material_v1"
    if candidate_family == "hybrid_v1":
        candidate_family = "hybrid_geodesic_lavd_v1"
    pv_retention_focused = mode.startswith("pv_retention_")

    candidates: list[tuple[np.ndarray, dict[str, float | str]]] = []
    if candidate_family in {"cauchy_green_geodesic_v1", "hybrid_geodesic_lavd_v1"}:
        candidates.extend(
            _candidate_from_stretch(
                cg["lambda2"],
                core_domain & cg["valid_cg"],
                x_km,
                y_km,
                min_area_fraction=request.min_area_fraction,
                max_area_fraction=request.max_area_fraction,
            )
        )
    if candidate_family in {"lavd_material_v1", "hybrid_geodesic_lavd_v1"}:
        candidates.extend(
            _candidate_from_lavd(
                lavd,
                core_domain & lavd_valid,
                x_km,
                y_km,
                min_area_fraction=request.min_area_fraction,
                max_area_fraction=request.max_area_fraction,
            )
        )
    if candidate_family == "hybrid_geodesic_lavd_v1" and np.any(base_mask):
        hybrid_domain = core_domain & cg["valid_cg"] & lavd_valid
        if np.any(hybrid_domain):
            stretch_error = np.abs(np.log(cg["lambda2"]))
            lavd_threshold = float(np.nanquantile(lavd[hybrid_domain], 0.60))
            stretch_threshold = float(np.nanquantile(stretch_error[hybrid_domain], 0.45))
            seed = _seed_from_axis(x_km, y_km, hybrid_domain)
            component, status = _closed_component(
                hybrid_domain & (lavd >= lavd_threshold) & (stretch_error <= stretch_threshold),
                seed,
            )
            if np.any(component):
                candidates.append(
                    (
                        component,
                        {
                            "geodesic_status": status,
                            "lavd_boundary_status": status,
                            "closed_geodesic_found": status == "closed_component",
                            "lavd_closed_contour_found": status == "closed_component",
                            "mean_lambda2": _finite_mean(cg["lambda2"][component]),
                            "mean_log_lambda2_abs": _finite_mean(stretch_error[component]),
                            "lavd_mean": _finite_mean(lavd[component]),
                            "lavd_p90": float(np.nanpercentile(lavd[component], 90)),
                            "hybrid_rule": "low_cauchy_green_stretch_and_high_lavd_v1",
                        },
                    )
                )

    if pv_retention_focused:
        candidates = [
            (mask, {**meta, "pv_retention_focused": "true"})
            for mask, meta in candidates
        ]

    mask, meta = _select_best_candidate(
        candidates,
        rep=rep,
        q_proxy=q_proxy,
        buoyancy=buoyancy,
        depth_index=depth_index,
        x_km=x_km,
        y_km=y_km,
        core_domain=core_domain,
        weak_core=weak_core,
        pv_core=pv_core,
        forward_map=forward,
        backward_map=backward,
        request=request,
    )
    if pv_retention_focused and float(meta.get("pv_core_retention", 0.0) or 0.0) < request.min_pv_retention:
        meta["boundary_status"] = "pv_retention_below_target"
    meta.update(
        {
            "boundary_mode": mode,
            "depth_index": int(depth_index),
            "depth_m": float(rep.depth_m[depth_index]),
            "flow_map_valid_fraction": float(np.count_nonzero(cg["valid_cg"] & core_domain) / max(1, np.count_nonzero(core_domain))),
            "lambda1_median": _safe_nanmedian(cg["lambda1"]),
            "lambda2_median": _safe_nanmedian(cg["lambda2"]),
            "lavd_valid_fraction": float(np.count_nonzero(lavd_valid & core_domain) / max(1, np.count_nonzero(core_domain))),
            "lavd_days_integrated": float(lavd_days),
            "lavd_domain_median": float(np.nanmedian(lavd[lavd_valid])) if np.any(lavd_valid) else np.nan,
            "speed_threshold_ms": speed_threshold,
            "pv_abs_threshold": pv_threshold,
            "geodesic_boundary_family": "particle_flow_map_cauchy_green_lavd_v1",
        }
    )
    return mask, meta


def _vertical_stitch_masks(masks: np.ndarray) -> tuple[np.ndarray, int]:
    if masks.ndim != 3 or masks.shape[0] < 3:
        return masks, 0
    out = masks.astype(bool).copy()
    changed = 0
    areas = np.asarray([np.count_nonzero(mask) for mask in out], dtype=float)
    for iz in range(1, out.shape[0] - 1):
        if areas[iz] > 0:
            left = areas[iz - 1] if areas[iz - 1] > 0 else areas[iz]
            right = areas[iz + 1] if areas[iz + 1] > 0 else areas[iz]
            ref = 0.5 * (left + right)
            if ref <= 0 or abs(areas[iz] - ref) / ref < 0.85:
                continue
        candidate = (neighbors4(out[iz - 1]) | out[iz - 1] | neighbors4(out[iz + 1]) | out[iz + 1])
        if np.any(candidate):
            out[iz] = candidate
            changed += 1
    return out, changed


def _compute_geodesic_record(
    *,
    records: list[dict[str, object]],
    center_index: int,
    mode: str,
    request: MaterialGeodesicRequest,
) -> tuple[pd.DataFrame, np.ndarray]:
    record = records[center_index]
    rep = record["rep"]
    rows: list[dict[str, float | str]] = []
    masks: list[np.ndarray] = []
    depth_count = int(rep.depth_m.size)
    if request.max_depth_layers > 0:
        depth_count = min(depth_count, int(request.max_depth_layers))
    for iz in range(depth_count):
        mask, meta = _layer_geodesic_boundary(
            records=records,
            center_index=center_index,
            depth_index=iz,
            mode=mode,
            request=request,
        )
        masks.append(mask)
        rows.append(meta)
    mask_array, stitched_count = _vertical_stitch_masks(np.asarray(masks, dtype=bool))
    table = pd.DataFrame(rows)
    if not table.empty:
        table["vertical_stitch_adjusted_layer_count"] = int(stitched_count)
    if request.boundary_budget == "full_3d" and mask_array.size:
        rep = record["rep"]
        x_km, y_km = rep.mesh_xy_km
        budget = _full_boundary_flux_budget(
            rep=rep,
            masks=mask_array,
            q_proxy=record["debug"]["q_proxy"],
            buoyancy=record["debug"]["buoyancy"],
            x_km=x_km,
            y_km=y_km,
        )
        if request.max_depth_layers > 0:
            budget = budget.iloc[:depth_count].copy()
        table = table.merge(budget, on=["depth_index", "depth_m"], how="left")
    return table, mask_array


def _enrich_geodesic_table(
    table: pd.DataFrame,
    *,
    record,
    shape: str,
    orientation: str,
    buoyancy_source: str,
    mode: str,
    request: MaterialGeodesicRequest,
) -> pd.DataFrame:
    out = table.copy()
    obj = record["obj"]
    base = record["base_table"].copy()
    if "leakage_mean_abs_ms" in base.columns:
        out["levelset_v2_leakage_mean_abs_ms"] = base["leakage_mean_abs_ms"].to_numpy(float)[: out.shape[0]]
    if "boundary_flux_over_internal_flux" in base.columns:
        out["levelset_v2_boundary_flux_over_internal_flux"] = base["boundary_flux_over_internal_flux"].to_numpy(float)[: out.shape[0]]
    for col in (
        "pv_flux_magnitude",
        "heat_flux_magnitude",
        "G_magnitude_proxy",
        "pv_centroid_distance_from_axis_km",
        "weak_speed_centroid_distance_from_axis_km",
    ):
        if col in base.columns and col not in out.columns:
            out[col] = base[col].to_numpy(float)[: out.shape[0]]
    if "levelset_v2_leakage_mean_abs_ms" in out.columns and "leakage_mean_abs_ms" in out.columns:
        baseline = out["levelset_v2_leakage_mean_abs_ms"].to_numpy(float)
        selected = out["leakage_mean_abs_ms"].to_numpy(float)
        out["geodesic_leakage_change_vs_levelset_v2"] = (baseline - selected) / (np.abs(baseline) + 1e-12)
    out["shape"] = shape
    out["orientation"] = orientation
    out["buoyancy_source"] = buoyancy_source
    out["boundary_mode"] = mode
    out["boundary_budget"] = request.boundary_budget
    out["polarity"] = str(obj.polarity)
    out["tau"] = float(record["rep"].tau) if np.isfinite(record["rep"].tau) else np.nan
    out["date"] = str(obj.date)
    out["track3d_id"] = int(obj.track3d_id)
    out["eddy3d_object_id"] = int(obj.eddy3d_object_id)
    out["mean_radius_m"] = float(obj.mean_radius_m)
    out["trajectory_window_days"] = int(request.trajectory_window_days)
    out["particle_spacing_km"] = float(request.particle_spacing_km)
    out["advection_step_hours"] = float(request.advection_step_hours)
    out["particle_advection_model"] = "true_rk4_particle_flow_map_in_track_local_coordinates"
    out["material_boundary_claim_level"] = "finite_time_lagrangian_boundary_v1"
    closure = _closure_residual_table(out)
    for col in (
        "pv_boundary_over_internal_proxy",
        "heat_boundary_abs_proxy",
        "momentum_boundary_abs_proxy",
        "closure_residual_proxy",
    ):
        out[col] = closure[col].to_numpy() if col in closure else np.nan
    return out


def _summary_table(profiles: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["shape", "boundary_mode", "polarity", "track3d_id"]
    for keys, sub in profiles.groupby(group_cols, sort=True):
        shape, mode, polarity, track_id = keys
        def median_col(name: str) -> float:
            if name not in sub.columns:
                return float("nan")
            return _safe_nanmedian(sub[name].to_numpy(float))

        def fraction_col(name: str) -> float:
            if name not in sub.columns:
                return float("nan")
            values = sub[name].astype(float).to_numpy()
            finite = values[np.isfinite(values)]
            if finite.size == 0:
                return float("nan")
            return float(np.mean(finite))

        rows.append(
            {
                "shape": shape,
                "boundary_mode": mode,
                "polarity": polarity,
                "track3d_id": int(track_id),
                "n_objectdays": int(sub["eddy3d_object_id"].nunique()),
                "n_rows": int(sub.shape[0]),
                "closed_geodesic_found_fraction": fraction_col("closed_geodesic_found"),
                "lavd_closed_contour_found_fraction": fraction_col("lavd_closed_contour_found"),
                "flow_map_valid_fraction_median": median_col("flow_map_valid_fraction"),
                "particle_retention_median": median_col("particle_retention_mean"),
                "leakage_median_ms": median_col("leakage_mean_abs_ms"),
                "leakage_change_vs_levelset_v2_median": median_col("geodesic_leakage_change_vs_levelset_v2"),
                "boundary_flux_over_internal_flux_median": median_col("boundary_flux_over_internal_flux"),
                "closure_residual_proxy_median": median_col("closure_residual_proxy"),
                "weak_core_retention_median": median_col("weak_core_retention"),
                "pv_core_retention_median": median_col("pv_core_retention"),
                "pv_abs_retention_median": median_col("pv_abs_retention"),
                "pv_high_quantile_retention_median": median_col("pv_high_quantile_retention"),
                "distance_mask_to_pv_centroid_median_km": median_col("distance_mask_to_pv_centroid_km"),
            }
        )
    return pd.DataFrame(rows)


def _write_subset_tables(root: Path, profiles: pd.DataFrame) -> None:
    trajectory_cols = [
        col
        for col in profiles.columns
        if col
        in {
            "shape",
            "orientation",
            "buoyancy_source",
            "boundary_mode",
            "polarity",
            "tau",
            "date",
            "track3d_id",
            "eddy3d_object_id",
            "depth_index",
            "depth_m",
            "flow_map_valid_fraction",
            "forward_valid_fraction",
            "backward_valid_fraction",
            "forward_days_integrated",
            "backward_days_integrated",
            "particle_retention_forward",
            "particle_retention_backward",
            "particle_retention_mean",
            "particle_escape_fraction",
            "particle_advection_model",
        }
    ]
    cg_cols = [
        col
        for col in profiles.columns
        if col
        in {
            "shape",
            "boundary_mode",
            "polarity",
            "date",
            "track3d_id",
            "eddy3d_object_id",
            "depth_index",
            "depth_m",
            "lambda1_median",
            "lambda2_median",
            "mean_lambda2",
            "mean_log_lambda2_abs",
            "closed_geodesic_found",
            "closed_geodesic_area_fraction",
            "geodesic_status",
            "lambda_line_solver",
        }
    ]
    lavd_cols = [
        col
        for col in profiles.columns
        if col
        in {
            "shape",
            "boundary_mode",
            "polarity",
            "date",
            "track3d_id",
            "eddy3d_object_id",
            "depth_index",
            "depth_m",
            "lavd_valid_fraction",
            "lavd_days_integrated",
            "lavd_domain_median",
            "lavd_mean",
            "lavd_p90",
            "lavd_closed_contour_found",
            "lavd_area_fraction",
            "lavd_boundary_status",
            "lavd_integral_model",
        }
    ]
    budget_cols = [
        col
        for col in profiles.columns
        if col.startswith(("lateral_", "top_", "bottom_", "total_"))
        or col
        in {
            "shape",
            "boundary_mode",
            "polarity",
            "date",
            "track3d_id",
            "eddy3d_object_id",
            "depth_index",
            "depth_m",
            "boundary_budget",
            "vertical_velocity_source",
        }
    ]
    closure_cols = [
        col
        for col in profiles.columns
        if col
        in {
            "shape",
            "boundary_mode",
            "polarity",
            "date",
            "track3d_id",
            "eddy3d_object_id",
            "depth_index",
            "depth_m",
            "boundary_flux_over_internal_flux",
            "pv_boundary_over_internal_proxy",
            "heat_boundary_abs_proxy",
            "momentum_boundary_abs_proxy",
            "closure_residual_proxy",
            "weak_core_retention",
            "pv_core_retention",
            "pv_abs_retention",
            "pv_high_quantile_retention",
            "pv_centroid_inside_mask",
            "pv_centroid_x_km",
            "pv_centroid_y_km",
            "distance_mask_to_pv_centroid_km",
        }
    ]
    _write_table(profiles[trajectory_cols], root / "particle_trajectory_audit.csv")
    _write_table(profiles[cg_cols], root / "cauchy_green_metrics.csv")
    _write_table(profiles[cg_cols], root / "closed_geodesic_boundary_metrics.csv")
    _write_table(profiles[lavd_cols], root / "lavd_material_boundary_metrics.csv")
    _write_table(profiles[budget_cols], root / "full_boundary_flux_budget.csv")
    _write_table(profiles[closure_cols], root / "ep_material_closure_residual.csv")


def _plot_outputs(output_root: Path, profiles: pd.DataFrame, summary: pd.DataFrame) -> list[Path]:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return []
    figures = output_root / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    clean = profiles.replace([np.inf, -np.inf], np.nan)

    if {"boundary_mode", "flow_map_valid_fraction"}.issubset(clean.columns):
        fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
        groups = [part["flow_map_valid_fraction"].dropna().to_numpy(float) for _, part in clean.groupby("boundary_mode", sort=True)]
        labels = [str(key) for key, _ in clean.groupby("boundary_mode", sort=True)]
        ax.boxplot(groups, labels=labels, showfliers=False)
        ax.set_ylabel("valid particle flow-map fraction")
        ax.set_title("Flow-map validity by material-boundary mode")
        ax.tick_params(axis="x", rotation=20)
        ax.grid(True, axis="y", alpha=0.25)
        path = figures / "flow_map_valid_fraction_by_depth.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        written.append(path)

    if {"boundary_mode", "leakage_mean_abs_ms"}.issubset(clean.columns):
        fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
        groups = [part["leakage_mean_abs_ms"].dropna().to_numpy(float) for _, part in clean.groupby("boundary_mode", sort=True)]
        labels = [str(key) for key, _ in clean.groupby("boundary_mode", sort=True)]
        ax.boxplot(groups, labels=labels, showfliers=False)
        ax.set_ylabel("boundary leakage |u_n| (m/s)")
        ax.set_title("Boundary leakage method comparison")
        ax.tick_params(axis="x", rotation=20)
        ax.grid(True, axis="y", alpha=0.25)
        path = figures / "boundary_leakage_method_comparison.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        written.append(path)

    if {"shape", "boundary_mode", "closure_residual_proxy"}.issubset(clean.columns):
        grouped = clean.groupby(["shape", "boundary_mode"], sort=True)["closure_residual_proxy"].median().reset_index()
        labels = grouped["shape"].astype(str) + "\n" + grouped["boundary_mode"].astype(str)
        fig, ax = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
        ax.bar(np.arange(grouped.shape[0]), grouped["closure_residual_proxy"].to_numpy(float), color="#4c78a8")
        ax.set_xticks(np.arange(grouped.shape[0]))
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.set_ylabel("median closure residual proxy")
        ax.set_title("EP material closure residual by shape and boundary mode")
        ax.grid(True, axis="y", alpha=0.25)
        path = figures / "closure_residual_by_shape.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        written.append(path)

    if {"pv_core_retention", "leakage_mean_abs_ms", "boundary_mode"}.issubset(clean.columns):
        fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
        for mode, part in clean.groupby("boundary_mode", sort=True):
            ax.scatter(part["pv_core_retention"], part["leakage_mean_abs_ms"], s=10, alpha=0.35, label=str(mode))
        ax.set_xlabel("PV-core retention fraction")
        ax.set_ylabel("boundary leakage |u_n| (m/s)")
        ax.set_title("PV retention versus boundary leakage")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7)
        path = figures / "pv_retention_vs_leakage.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        written.append(path)

    if {"pv_core_retention", "closure_residual_proxy", "boundary_mode"}.issubset(clean.columns):
        fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
        for mode, part in clean.groupby("boundary_mode", sort=True):
            ax.scatter(part["pv_core_retention"], part["closure_residual_proxy"], s=10, alpha=0.35, label=str(mode))
        ax.set_xlabel("PV-core retention fraction")
        ax.set_ylabel("closure residual proxy")
        ax.set_title("Closure residual versus PV retention")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7)
        path = figures / "closure_residual_vs_pv_retention.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        written.append(path)

    if {"distance_mask_to_pv_centroid_km", "leakage_mean_abs_ms", "boundary_mode"}.issubset(clean.columns):
        fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
        for mode, part in clean.groupby("boundary_mode", sort=True):
            ax.scatter(part["distance_mask_to_pv_centroid_km"], part["leakage_mean_abs_ms"], s=10, alpha=0.35, label=str(mode))
        ax.set_xlabel("mask distance to PV centroid (km)")
        ax.set_ylabel("boundary leakage |u_n| (m/s)")
        ax.set_title("PV core separation audit")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7)
        path = figures / "hua_lavd_pv_core_separation.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        written.append(path)

    if {"total_heat_flux_watt_proxy", "total_pv_flux_proxy", "total_momentum_x_flux_proxy", "boundary_mode"}.issubset(clean.columns):
        grouped = clean.groupby("boundary_mode", sort=True)[
            ["total_heat_flux_watt_proxy", "total_pv_flux_proxy", "total_momentum_x_flux_proxy"]
        ].median()
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
        for ax, col, title in zip(axes, grouped.columns, ("heat", "PV", "momentum-x")):
            ax.bar(np.arange(grouped.shape[0]), grouped[col].to_numpy(float), color="#f58518")
            ax.set_xticks(np.arange(grouped.shape[0]))
            ax.set_xticklabels(grouped.index.astype(str), rotation=35, ha="right")
            ax.set_title(title)
            ax.grid(True, axis="y", alpha=0.25)
        path = figures / "heat_pv_momentum_boundary_budget.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        written.append(path)
    return written


def _write_summary(path: Path, summary: pd.DataFrame, request: MaterialGeodesicRequest) -> None:
    lines = [
        "# Object Material Geodesic EP Validation",
        "",
        "## 口径",
        f"- shapes: `{','.join(request.shapes)}`",
        f"- boundary modes: `{','.join(request.boundary_modes)}`",
        f"- trajectory window: `{request.trajectory_window_days}` days forward/backward",
        f"- advection step: `{request.advection_step_hours}` hours",
        "- particle advection: RK4 on 30-180d bandpass velocity in object-following local coordinates",
        "- vertical velocity is not used for particle trajectories; top/bottom flux still uses continuity `w_proxy`.",
        f"- PV retention target: `{request.min_pv_retention}`; require PV retention: `{request.require_pv_retention}`",
        f"- PV/weak/particle weights: `{request.pv_retention_weight}` / `{request.weak_retention_weight}` / `{request.particle_retention_weight}`",
        "",
        "## 结果摘要",
        "```text",
        summary.to_string(index=False) if not summary.empty else "empty",
        "```",
        "",
        "## 判读边界",
        "- `closed_geodesic_found_fraction` 和 `lavd_closed_contour_found_fraction` 是能否形成强材料边界结论的第一门槛。",
        "- 若 leakage、boundary exchange、closure residual 未低于 proxy 边界，不能强行宣称严格材料体闭合成立。",
        "- `pv_retention_*` 模式把 PV core retention 从弱惩罚升级为主目标，用于检查闭合失败是否来自边界没有围住 PV anomaly 核心。",
        "- 若 PV retention 提高但 leakage 明显变差，说明 PV 动力核心可能不属于低泄漏旋转材料核，应解释为双核心或核心-剪切环分裂。",
        "- 该版本已使用真实粒子 flow map 与轨迹 LAVD，但 Cauchy-Green 闭合曲线搜索仍是 v1 数值实现，需用成功率和失败原因审计。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _print_dry_run(request: MaterialGeodesicRequest) -> None:
    print("Object material geodesic EP validation dry-run")
    print(f"result_root: {request.result_root}")
    print(f"filter_root: {request.filter_root}")
    print(f"output_root: {request.output_root}")
    print(f"shapes: {','.join(request.shapes)}")
    print(f"orientations: {','.join(request.orientations)}")
    print(f"buoyancy_sources: {','.join(request.buoyancy_sources)}")
    print(f"boundary_modes: {','.join(request.boundary_modes)}")
    print(f"boundary_budget: {request.boundary_budget}")
    print(f"trajectory_window_days: {request.trajectory_window_days}")
    print(f"particle_spacing_km: {request.particle_spacing_km}")
    print(f"advection_step_hours: {request.advection_step_hours}")
    print(f"max_depth_layers: {request.max_depth_layers or 'all'}")
    print(f"min_pv_retention: {request.min_pv_retention}")
    print(f"require_pv_retention: {request.require_pv_retention}")
    print(f"pv_retention_weight: {request.pv_retention_weight}")
    print(f"weak_retention_weight: {request.weak_retention_weight}")
    print(f"particle_retention_weight: {request.particle_retention_weight}")
    print("advection_velocity: 30-180d bandpass uo_glor/vo_glor")
    print("method: RK4 particles + Cauchy-Green flow map + trajectory LAVD + closed boundary search")


def run_object_material_geodesic_validation(request: MaterialGeodesicRequest) -> dict[str, Path]:
    _validate_request(request)
    if request.dry_run:
        _print_dry_run(request)
        return {}
    _load_runtime_helpers()
    _require_runtime()
    f0 = 2.0 * 7.2921159e-5 * math.sin(math.radians(request.reference_lat))
    radial, theta, radial_mesh, theta_mesh = _polar_grid(request.rmax, request.radial_bins, request.azimuth_bins)
    outputs: dict[str, Path] = {}
    all_profiles: list[pd.DataFrame] = []
    all_summaries: list[pd.DataFrame] = []

    for shape in request.shapes:
        try:
            rv_root, objects, points = _load_shape_objects(
                request.result_root,
                shape,
                request.max_tracks_per_shape,
                request.max_objectdays,
            )
        except FileNotFoundError:
            if request.skip_missing:
                continue
            raise
        if objects.empty:
            continue
        objects = objects.sort_values(["track3d_id", "date", "eddy3d_object_id"]).copy()
        for orientation in request.orientations:
            for buoyancy_source in request.buoyancy_sources:
                coherence_request = _coherence_request(request, shape, orientation, buoyancy_source)
                day_cache: dict[date, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
                records_by_track: dict[int, list[dict[str, object]]] = {}
                for obj in objects.itertuples(index=False):
                    record = _compute_base_record(
                        obj=obj,
                        points=points,
                        day_cache=day_cache,
                        request=coherence_request,
                        radial=radial,
                        theta=theta,
                        radial_mesh=radial_mesh,
                        theta_mesh=theta_mesh,
                        orientation=orientation,
                        buoyancy_source=buoyancy_source,
                        f0=f0,
                    )
                    if record is not None:
                        records_by_track.setdefault(int(obj.track3d_id), []).append(record)

                for mode in request.boundary_modes:
                    combo_dir = request.output_root / shape / orientation / buoyancy_source / mode
                    combo_dir.mkdir(parents=True, exist_ok=True)
                    rows: list[pd.DataFrame] = []
                    for records in records_by_track.values():
                        records.sort(key=lambda item: (item["day"], int(item["obj"].eddy3d_object_id)))
                        for idx, record in enumerate(records):
                            obj = record["obj"]
                            print(
                                "geodesic object "
                                f"shape={shape} mode={mode} track={int(obj.track3d_id)} "
                                f"object={int(obj.eddy3d_object_id)} date={obj.date}",
                                flush=True,
                            )
                            try:
                                table, _ = _compute_geodesic_record(
                                    records=records,
                                    center_index=idx,
                                    mode=mode,
                                    request=request,
                                )
                            except Exception:
                                if request.skip_missing:
                                    continue
                                raise
                            rows.append(
                                _enrich_geodesic_table(
                                    table,
                                    record=record,
                                    shape=shape,
                                    orientation=orientation,
                                    buoyancy_source=buoyancy_source,
                                    mode=mode,
                                    request=request,
                                )
                            )
                    if not rows:
                        continue
                    profiles = pd.concat(rows, ignore_index=True)
                    summary = _summary_table(profiles)
                    _write_table(profiles, combo_dir / "hybrid_material_boundary_profiles.csv")
                    _write_table(summary, combo_dir / "shape_materiality_comparison.csv")
                    _write_subset_tables(combo_dir, profiles)
                    (combo_dir / "material_geodesic_manifest.json").write_text(
                        json.dumps(
                            _json_ready(
                                {
                                    "shape": shape,
                                    "orientation": orientation,
                                    "buoyancy_source": buoyancy_source,
                                    "boundary_mode": mode,
                                    "rv_root": rv_root,
                                    "request": request.__dict__,
                                    "method_status": "finite-time particle Cauchy-Green/LAVD material-boundary validation v1",
                                }
                            ),
                            ensure_ascii=False,
                            indent=2,
                        ),
                        encoding="utf-8",
                    )
                    all_profiles.append(profiles)
                    all_summaries.append(summary)
                    outputs[str(combo_dir)] = combo_dir

    if all_profiles:
        request.output_root.mkdir(parents=True, exist_ok=True)
        root_profiles = pd.concat(all_profiles, ignore_index=True)
        root_summary = pd.concat(all_summaries, ignore_index=True)
        _write_table(root_profiles, request.output_root / "hybrid_material_boundary_profiles.csv")
        _write_table(root_summary, request.output_root / "shape_materiality_comparison.csv")
        if any(str(mode).startswith("pv_retention_") for mode in request.boundary_modes):
            _write_table(root_profiles, request.output_root / "pv_retention_boundary_profiles.csv")
            _write_table(root_summary, request.output_root / "pv_retention_boundary_summary.csv")
            (request.output_root / "pv_retention_boundary_summary.json").write_text(
                json.dumps(_json_ready(root_summary.to_dict(orient="records")), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        _write_subset_tables(request.output_root, root_profiles)
        for plot_path in _plot_outputs(request.output_root, root_profiles, root_summary):
            outputs[f"figure:{plot_path.name}"] = plot_path
        _write_summary(request.output_root / "object_material_geodesic_ep_validation_summary_zh.md", root_summary, request)
        if any(str(mode).startswith("pv_retention_") for mode in request.boundary_modes):
            _write_summary(request.output_root / "pv_retention_ep_validation_summary_zh.md", root_summary, request)
        (request.output_root / "material_geodesic_manifest.json").write_text(
            json.dumps(_json_ready({"request": request.__dict__}), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        outputs["root_summary"] = request.output_root / "shape_materiality_comparison.csv"
    return outputs


def request_from_args(args) -> MaterialGeodesicRequest:
    return MaterialGeodesicRequest(
        result_root=Path(args.result_root),
        filter_root=Path(args.filter_root),
        output_root=Path(args.output_root),
        shapes=_split_csv(args.shapes),
        orientations=_split_csv(args.orientations),
        buoyancy_sources=_split_csv(args.buoyancy_sources),
        boundary_modes=_split_csv(args.boundary_mode),
        boundary_budget=str(args.boundary_budget),
        filter_template=str(args.filter_template),
        radial_bins=int(args.radial_bins),
        azimuth_bins=int(args.azimuth_bins),
        rmax=float(args.rmax),
        reference_lat=float(args.reference_lat),
        constant_n2=float(args.constant_n2),
        core_radius_over_R=float(args.core_radius_over_R),
        speed_core_quantile=float(args.speed_core_quantile),
        pv_core_quantile=float(args.pv_core_quantile),
        min_mask_fraction=float(args.min_mask_fraction),
        min_core_retention=float(args.min_core_retention),
        min_pv_retention=float(args.min_pv_retention),
        pv_retention_weight=float(args.pv_retention_weight),
        weak_retention_weight=float(args.weak_retention_weight),
        particle_retention_weight=float(args.particle_retention_weight),
        require_pv_retention=bool(args.require_pv_retention),
        min_area_fraction=float(args.min_area_fraction),
        max_area_fraction=float(args.max_area_fraction),
        trajectory_window_days=int(args.trajectory_window_days),
        particle_spacing_km=float(args.particle_spacing_km),
        advection_step_hours=float(args.advection_step_hours),
        max_tracks_per_shape=int(args.max_tracks_per_shape),
        max_objectdays=int(args.max_objectdays),
        max_depth_layers=int(args.max_depth_layers),
        skip_missing=bool(args.skip_missing),
        dry_run=bool(args.dry_run),
    )
