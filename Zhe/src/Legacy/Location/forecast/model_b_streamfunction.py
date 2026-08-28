from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter, gaussian_filter1d

from ..validation.isopycnal_projection import (
    build_isopycnal_surfaces,
    interpolate_to_surfaces,
    project_surface_field_to_depth,
)

from .common import G, finite_or_nan, geo_params, gradient
from .models import ForecastState, filled_centers


NETWORK_KAPPA2 = 4.0
DRIVER_SMOOTH_SIGMA = 1.0
PSI_LS_REGULARIZATION = 1.0e-3
ETA_COUPLING = 0.30
SIGMA_COUPLING = 0.18
MAX_PHASE_SUBSTEPS = 4


@dataclass
class ModelBState:
    base: ForecastState
    rho_levels: np.ndarray
    psi_initial_from_birth_velocity: np.ndarray
    psi_on_isopycnal: np.ndarray
    q_model_b: np.ndarray
    surface_internal_coupling: np.ndarray
    eta_increment: np.ndarray


def build_model_b_state(
    sigma_birth: np.ndarray,
    adt_birth: np.ndarray,
    u_birth: np.ndarray,
    v_birth: np.ndarray,
    sigma_clim: np.ndarray,
    depth: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    radius_m: float,
    latitude_ref: float,
) -> ModelBState:
    gp = geo_params(latitude_ref)
    sigma_total = np.asarray(sigma_clim, dtype="f8")[:, None, None] + finite_or_nan(sigma_birth)
    surfaces = build_isopycnal_surfaces(sigma_total, depth, smooth_sigma_grid=1.0)
    psi = build_initial_psi_from_birth_velocity(
        finite_or_nan(u_birth),
        finite_or_nan(v_birth),
        surfaces.z_m,
        np.asarray(depth, dtype="f8"),
        np.asarray(x, dtype="f8"),
        np.asarray(y, dtype="f8"),
        float(radius_m),
        regularization=PSI_LS_REGULARIZATION,
    )
    q_b = laplace_beltrami(psi, surfaces.z_m, x, y, radius_m)
    coupling = surface_internal_coupling(finite_or_nan(adt_birth), psi, float(gp.f0))
    eta_increment = np.zeros_like(finite_or_nan(adt_birth), dtype="f8")
    cx, cy = _streamfunction_driver_profile(psi, x, y)
    center_x = _regularize_driver_profile(cx)
    center_y = _regularize_driver_profile(cy)
    source = np.full(center_x.shape, "model_B_birth_velocity_initialized_pv_eta_network", dtype="U64")
    confidence = _streamfunction_confidence(psi, surfaces.valid_fraction)
    state = ForecastState(
        finite_or_nan(sigma_birth),
        finite_or_nan(adt_birth),
        np.asarray(sigma_clim, dtype="f8"),
        np.asarray(depth, dtype="f8"),
        np.asarray(x, dtype="f8"),
        np.asarray(y, dtype="f8"),
        float(radius_m),
        float(gp.f0),
        center_x,
        center_y,
        source,
        confidence,
    )
    return ModelBState(state, surfaces.rho_levels, psi.copy(), psi, q_b, coupling, eta_increment)


def predict_model_b_state(state: ModelBState, tau: float, strength: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    sigma_pred, adt_pred, *_ = step_model_b_pv_eta_network(state, tau, strength)
    return sigma_pred, adt_pred


def forecast_model_b_phase(
    state: ModelBState,
    tau: float,
    strength: float,
    shape: str,
    polarity: str,
    phase: str,
    phase_index: int,
    model: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    sigma_pred, adt_pred, psi, q_b, coupling, eta_increment, surfaces = step_model_b_pv_eta_network(state, tau, strength)
    u_depth, v_depth = project_model_b_to_fixed_depth(psi, surfaces.z_m, state.base)
    table = _diagnostic_table(
        surfaces.valid_fraction,
        psi,
        q_b,
        coupling,
        eta_increment,
        shape,
        polarity,
        phase,
        phase_index,
        model,
    )
    return sigma_pred, adt_pred, u_depth, v_depth, surfaces.z_anom_m, psi, q_b, coupling, eta_increment, table


def diagnose_model_b_streamfunction(
    state: ModelBState,
    sigma_pred: np.ndarray,
    adt_pred: np.ndarray,
    shape: str,
    polarity: str,
    phase: str,
    phase_index: int,
    model: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    sigma_total = state.base.sigma_clim[:, None, None] + finite_or_nan(sigma_pred)
    surfaces = build_isopycnal_surfaces(sigma_total, state.base.depth, rho_levels=state.rho_levels, smooth_sigma_grid=1.0)
    psi, q_b, coupling = solve_surface_internal_pv_network(
        state.psi_initial_from_birth_velocity,
        finite_or_nan(adt_pred),
        surfaces.z_m,
        state.base.depth,
        state.base.x,
        state.base.y,
        state.base.radius_m,
        state.base.f0,
    )
    u_s, v_s = velocity_on_isopycnals(psi, surfaces.z_m, state.base.x, state.base.y, state.base.radius_m)
    u_depth = project_surface_field_to_depth(u_s, surfaces.z_m, state.base.depth)
    v_depth = project_surface_field_to_depth(v_s, surfaces.z_m, state.base.depth)
    table = _diagnostic_table(surfaces.valid_fraction, psi, q_b, coupling, np.zeros_like(adt_pred), shape, polarity, phase, phase_index, model)
    return u_depth, v_depth, surfaces.z_anom_m, psi, q_b, coupling, table


def build_initial_psi_from_birth_velocity(
    u_birth: np.ndarray,
    v_birth: np.ndarray,
    z_surface_m: np.ndarray,
    depth_m: np.ndarray,
    x_R: np.ndarray,
    y_R: np.ndarray,
    radius_m: float,
    regularization: float = PSI_LS_REGULARIZATION,
) -> np.ndarray:
    """Infer initial isopycnal streamfunction from the birth velocity condition."""
    u_s = interpolate_to_surfaces(finite_or_nan(u_birth), np.asarray(depth_m, dtype="f8"), finite_or_nan(z_surface_m))
    v_s = interpolate_to_surfaces(finite_or_nan(v_birth), np.asarray(depth_m, dtype="f8"), finite_or_nan(z_surface_m))
    x_m = np.asarray(x_R, dtype="f8") * max(float(radius_m), 1.0)
    y_m = np.asarray(y_R, dtype="f8") * max(float(radius_m), 1.0)
    psi = np.full_like(u_s, np.nan, dtype="f8")
    dx = float(np.nanmedian(np.diff(x_m))) if x_m.size > 1 else 1.0
    dy = float(np.nanmedian(np.diff(y_m))) if y_m.size > 1 else 1.0
    for k in range(u_s.shape[0]):
        uu = _nearest_fill_2d(u_s[k])
        vv = _nearest_fill_2d(v_s[k])
        valid = np.isfinite(u_s[k]) | np.isfinite(v_s[k])
        if np.count_nonzero(valid) < 9:
            continue
        zeta = gradient(vv, x_m, axis=1) - gradient(uu, y_m, axis=0)
        layer_psi = _regularized_poisson(zeta, dx, dy, regularization=regularization)
        layer_psi -= np.nanmedian(layer_psi)
        layer_psi = _fit_birth_streamfunction_amplitude(layer_psi, uu, vv, x_m, y_m, valid)
        psi[k] = np.where(valid, layer_psi, np.nan)
    return _fill_missing_vertical(psi)


def step_model_b_pv_eta_network(
    state: ModelBState,
    tau: float,
    strength: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, object]:
    """Advance sigma, eta and streamfunction from the birth state using the Model B network."""
    tau = float(tau)
    sigma = finite_or_nan(state.base.sigma_birth).copy()
    adt = finite_or_nan(state.base.adt_birth).copy()
    psi = finite_or_nan(state.psi_initial_from_birth_velocity).copy()
    eta_total_increment = np.zeros_like(adt, dtype="f8")
    if abs(tau) > 1e-12:
        nsteps = max(1, min(MAX_PHASE_SUBSTEPS, int(np.ceil(abs(tau) * MAX_PHASE_SUBSTEPS))))
        dtau = tau * float(strength) / nsteps
        for _ in range(nsteps):
            surfaces = build_isopycnal_surfaces(
                state.base.sigma_clim[:, None, None] + sigma,
                state.base.depth,
                rho_levels=state.rho_levels,
                smooth_sigma_grid=1.0,
            )
            psi, q_b, coupling = solve_surface_internal_pv_network(
                psi,
                adt,
                surfaces.z_m,
                state.base.depth,
                state.base.x,
                state.base.y,
                state.base.radius_m,
                state.base.f0,
            )
            eta_step = _eta_increment_from_pv_network(q_b, coupling, adt, dtau)
            sigma_step = _sigma_increment_from_pv_network(q_b, sigma, dtau)
            adt = _bounded_add(adt, eta_step, state.base.adt_birth, max_fraction=0.45)
            sigma = _bounded_add(sigma, sigma_step, state.base.sigma_birth, max_fraction=0.35)
            eta_total_increment = eta_total_increment + eta_step
    surfaces = build_isopycnal_surfaces(
        state.base.sigma_clim[:, None, None] + sigma,
        state.base.depth,
        rho_levels=state.rho_levels,
        smooth_sigma_grid=1.0,
    )
    psi, q_b, coupling = solve_surface_internal_pv_network(
        psi,
        adt,
        surfaces.z_m,
        state.base.depth,
        state.base.x,
        state.base.y,
        state.base.radius_m,
        state.base.f0,
    )
    return sigma, adt, psi, q_b, coupling, eta_total_increment, surfaces


def solve_surface_internal_pv_network(
    psi_current: np.ndarray,
    adt_m: np.ndarray,
    z_surface_m: np.ndarray,
    depth_m: np.ndarray,
    x_R: np.ndarray,
    y_R: np.ndarray,
    radius_m: float,
    f0: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One relaxed solve of the coupled free-surface / isopycnal PV-streamfunction network."""
    psi = _fill_missing_vertical(finite_or_nan(psi_current))
    q_prev = laplace_beltrami(psi, z_surface_m, x_R, y_R, radius_m)
    coupling = surface_internal_coupling(finite_or_nan(adt_m), psi, f0)
    vertical_target = _vertical_neighbor_average(psi)
    psi_scale = _safe_p95(psi, fallback=1.0)
    q_source = _normalize_field(q_prev) * psi_scale
    surface_pull = np.zeros_like(psi, dtype="f8")
    if psi.shape[0]:
        decay = np.exp(-np.linspace(0.0, 4.0, psi.shape[0]))[:, None, None]
        surface_pull = decay * np.where(np.isfinite(coupling[0]), coupling[0], 0.0)[None, :, :]
    psi_next = psi + 0.28 * (vertical_target - psi) + 0.18 * surface_pull + 0.06 * q_source
    psi_next = _smooth_isopycnal_stack(psi_next, sigma=0.75)
    q_next = laplace_beltrami(psi_next, z_surface_m, x_R, y_R, radius_m)
    coupling_next = surface_internal_coupling(finite_or_nan(adt_m), psi_next, f0)
    return psi_next, q_next, coupling_next


def project_model_b_to_fixed_depth(
    psi: np.ndarray,
    z_surface_m: np.ndarray,
    state: ForecastState,
) -> tuple[np.ndarray, np.ndarray]:
    u_s, v_s = velocity_on_isopycnals(psi, z_surface_m, state.x, state.y, state.radius_m)
    return (
        project_surface_field_to_depth(u_s, z_surface_m, state.depth),
        project_surface_field_to_depth(v_s, z_surface_m, state.depth),
    )


def surface_internal_coupling(adt_m: np.ndarray, psi: np.ndarray, f0: float) -> np.ndarray:
    eta = finite_or_nan(adt_m)
    psi_eta = G * np.where(np.isfinite(eta), eta, 0.0) / max(abs(float(f0)), 1e-12)
    coupling = np.zeros_like(psi, dtype="f8")
    if psi.shape[0]:
        coupling[0] = psi_eta - psi[0]
    return coupling


def velocity_on_isopycnals(
    psi: np.ndarray,
    z_surface_m: np.ndarray,
    x_R: np.ndarray,
    y_R: np.ndarray,
    radius_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    x_m = np.asarray(x_R, dtype="f8") * max(float(radius_m), 1.0)
    y_m = np.asarray(y_R, dtype="f8") * max(float(radius_m), 1.0)
    z = finite_or_nan(z_surface_m)
    zx = gradient(z, x_m, axis=2)
    zy = gradient(z, y_m, axis=1)
    denom = 1.0 + zx * zx + zy * zy
    gxx = 1.0 - zx * zx / denom
    gyy = 1.0 - zy * zy / denom
    gxy = -zx * zy / denom
    psix = gradient(psi, x_m, axis=2)
    psiy = gradient(psi, y_m, axis=1)
    ax = gxx * psix + gxy * psiy
    ay = gxy * psix + gyy * psiy
    return -ay, ax


def laplace_beltrami(
    psi: np.ndarray,
    z_surface_m: np.ndarray,
    x_R: np.ndarray,
    y_R: np.ndarray,
    radius_m: float,
) -> np.ndarray:
    x_m = np.asarray(x_R, dtype="f8") * max(float(radius_m), 1.0)
    y_m = np.asarray(y_R, dtype="f8") * max(float(radius_m), 1.0)
    z = finite_or_nan(z_surface_m)
    zx = gradient(z, x_m, axis=2)
    zy = gradient(z, y_m, axis=1)
    denom = 1.0 + zx * zx + zy * zy
    sqrt_g = np.sqrt(denom)
    gxx = 1.0 - zx * zx / denom
    gyy = 1.0 - zy * zy / denom
    gxy = -zx * zy / denom
    psix = gradient(psi, x_m, axis=2)
    psiy = gradient(psi, y_m, axis=1)
    flux_x = sqrt_g * (gxx * psix + gxy * psiy)
    flux_y = sqrt_g * (gxy * psix + gyy * psiy)
    div = gradient(flux_x, x_m, axis=2) + gradient(flux_y, y_m, axis=1)
    return np.divide(div, sqrt_g, out=np.full_like(div, np.nan), where=sqrt_g > 0)


def _regularized_poisson(rhs: np.ndarray, dx: float, dy: float, regularization: float, iterations: int = 260) -> np.ndarray:
    source = np.asarray(rhs, dtype="f8")
    source = np.where(np.isfinite(source), source, 0.0)
    psi = np.zeros_like(source, dtype="f8")
    dx2 = max(float(dx) * float(dx), 1.0)
    dy2 = max(float(dy) * float(dy), 1.0)
    denom = 2.0 * (dx2 + dy2)
    omega = 0.80
    for _ in range(iterations):
        update = (
            (psi[1:-1, 2:] + psi[1:-1, :-2]) * dy2
            + (psi[2:, 1:-1] + psi[:-2, 1:-1]) * dx2
            - source[1:-1, 1:-1] * dx2 * dy2
        ) / denom
        psi[1:-1, 1:-1] = (1.0 - omega) * psi[1:-1, 1:-1] + omega * update
    if regularization > 0:
        psi = gaussian_filter(psi, sigma=min(max(float(regularization) * 12.0, 0.0), 1.5), mode="nearest")
    return psi


def _fit_birth_streamfunction_amplitude(
    psi: np.ndarray,
    u_target: np.ndarray,
    v_target: np.ndarray,
    x_m: np.ndarray,
    y_m: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    u_hat = -gradient(psi, y_m, axis=0)
    v_hat = gradient(psi, x_m, axis=1)
    good = valid & np.isfinite(u_target) & np.isfinite(v_target) & np.isfinite(u_hat) & np.isfinite(v_hat)
    if np.count_nonzero(good) < 20:
        return psi
    pred = np.concatenate([u_hat[good].ravel(), v_hat[good].ravel()])
    targ = np.concatenate([u_target[good].ravel(), v_target[good].ravel()])
    denom = float(np.dot(pred, pred))
    if denom <= 1.0e-30:
        return psi
    scale = float(np.dot(pred, targ) / denom)
    if not np.isfinite(scale):
        return psi
    scale = float(np.clip(scale, 0.1, 20.0))
    return psi * scale


def _eta_increment_from_pv_network(q_b: np.ndarray, coupling: np.ndarray, adt: np.ndarray, dtau: float) -> np.ndarray:
    eta_scale = _safe_p95(adt, fallback=0.02)
    q_top = _normalize_field(q_b[0]) if q_b.shape[0] else np.zeros_like(adt, dtype="f8")
    c_top = _normalize_field(coupling[0]) if coupling.shape[0] else np.zeros_like(adt, dtype="f8")
    driver = 0.55 * q_top + 0.45 * c_top
    driver = gaussian_filter(np.where(np.isfinite(driver), driver, 0.0), sigma=1.0, mode="nearest")
    return float(dtau) * ETA_COUPLING * eta_scale * driver


def _sigma_increment_from_pv_network(q_b: np.ndarray, sigma: np.ndarray, dtau: float) -> np.ndarray:
    sigma_scale = _safe_p95(sigma, fallback=0.05)
    q_norm = _normalize_field(q_b)
    q_norm = _smooth_isopycnal_stack(q_norm, sigma=0.75)
    return -float(dtau) * SIGMA_COUPLING * sigma_scale * q_norm


def _bounded_add(base: np.ndarray, delta: np.ndarray, reference: np.ndarray, max_fraction: float) -> np.ndarray:
    base_arr = np.asarray(base, dtype="f8")
    step = np.asarray(delta, dtype="f8")
    limit = max(float(max_fraction) * _safe_p95(reference, fallback=_safe_p95(base_arr, fallback=1.0)), 1.0e-12)
    step = np.clip(step, -limit, limit)
    out = base_arr + step
    return np.where(np.isfinite(base_arr), out, np.nan)


def _vertical_neighbor_average(psi: np.ndarray) -> np.ndarray:
    arr = np.asarray(psi, dtype="f8")
    out = arr.copy()
    n = arr.shape[0]
    if n <= 1:
        return out
    out[0] = 0.65 * arr[0] + 0.35 * arr[1]
    out[-1] = 0.65 * arr[-1] + 0.35 * arr[-2]
    if n > 2:
        out[1:-1] = 0.50 * arr[1:-1] + 0.25 * (arr[:-2] + arr[2:])
    return out


def _normalize_field(field: np.ndarray) -> np.ndarray:
    arr = np.asarray(field, dtype="f8")
    med = np.nanmedian(arr) if np.isfinite(arr).any() else 0.0
    centered = arr - med
    scale = _safe_p95(centered, fallback=1.0)
    return np.divide(centered, scale, out=np.zeros_like(centered, dtype="f8"), where=np.isfinite(centered))


def _safe_p95(values: np.ndarray, fallback: float) -> float:
    arr = np.asarray(values, dtype="f8")
    good = np.isfinite(arr)
    if not np.any(good):
        return float(fallback)
    val = float(np.nanpercentile(np.abs(arr[good]), 95))
    if not np.isfinite(val) or val <= 1.0e-30:
        return float(fallback)
    return val


def _smooth_isopycnal_stack(field: np.ndarray, sigma: float) -> np.ndarray:
    arr = np.asarray(field, dtype="f8")
    filled = np.where(np.isfinite(arr), arr, 0.0)
    return gaussian_filter(filled, sigma=(0.0, float(sigma), float(sigma)), mode="nearest")


def _fill_missing_vertical(field: np.ndarray) -> np.ndarray:
    arr = np.asarray(field, dtype="f8").copy()
    if arr.ndim != 3:
        return arr
    for k in range(arr.shape[0]):
        arr[k] = _nearest_fill_2d(arr[k])
    return arr


def _streamfunction_driver_profile(psi: np.ndarray, x: np.ndarray, y: np.ndarray, search_radius_R: float = 1.5) -> tuple[np.ndarray, np.ndarray]:
    xx, yy = np.meshgrid(np.asarray(x, dtype="f8"), np.asarray(y, dtype="f8"))
    central = (np.abs(xx) <= search_radius_R) & (np.abs(yy) <= search_radius_R)
    cx = np.full(psi.shape[0], np.nan, dtype="f8")
    cy = np.full(psi.shape[0], np.nan, dtype="f8")
    for k in range(psi.shape[0]):
        layer = finite_or_nan(psi[k])
        edge = np.nanmedian(layer[~central]) if np.any(np.isfinite(layer[~central])) else np.nanmedian(layer)
        weights = np.where(central & np.isfinite(layer), np.abs(layer - edge), 0.0)
        total = float(np.nansum(weights))
        if total <= 0:
            continue
        cx[k] = float(np.nansum(weights * xx) / total)
        cy[k] = float(np.nansum(weights * yy) / total)
    return cx, cy


def _regularize_driver_profile(values: np.ndarray) -> np.ndarray:
    filled = filled_centers(values)
    if filled.size >= 4:
        filled = gaussian_filter1d(filled, sigma=DRIVER_SMOOTH_SIGMA, mode="nearest")
        filled -= filled[0]
    return filled


def _streamfunction_confidence(psi: np.ndarray, valid_fraction: np.ndarray) -> np.ndarray:
    amp = np.nanpercentile(np.abs(psi), 90, axis=(1, 2))
    scale = float(np.nanmax(amp)) if np.isfinite(amp).any() else 1.0
    conf = np.asarray(valid_fraction, dtype="f8") * np.divide(amp, max(scale, 1e-30), out=np.zeros_like(amp), where=np.isfinite(amp))
    return np.clip(conf, 0.0, 1.0)


def _nearest_fill_2d(layer: np.ndarray) -> np.ndarray:
    arr = np.asarray(layer, dtype="f8")
    valid = np.isfinite(arr)
    if valid.all():
        return arr.copy()
    if not valid.any():
        return np.zeros_like(arr, dtype="f8")
    from scipy.ndimage import distance_transform_edt

    _, indices = distance_transform_edt(~valid, return_indices=True)
    return arr[tuple(indices)]


def _diagnostic_table(
    valid_fraction: np.ndarray,
    psi: np.ndarray,
    q_b: np.ndarray,
    coupling: np.ndarray,
    eta_increment: np.ndarray,
    shape: str,
    polarity: str,
    phase: str,
    phase_index: int,
    model: str,
) -> pd.DataFrame:
    rows = []
    for k in range(psi.shape[0]):
        rows.append(
            {
                "shape": shape,
                "polarity": polarity,
                "phase": phase,
                "phase_index": int(phase_index),
                "model": model,
                "rho_bin": int(k),
                "depth_index": int(k),
                "TD_PV_star": np.nan,
                "TD_PV_adjacent_star": np.nan,
                "pva_volume_integral": np.nan,
                "pva_boundary_integral": np.nan,
                "pv_balance_residual": np.nan,
                "valid_fraction": float(valid_fraction[k]) if k < len(valid_fraction) else np.nan,
                "psi_p95": float(np.nanpercentile(np.abs(psi[k]), 95)) if np.isfinite(psi[k]).any() else np.nan,
                "q_model_B_p95": float(np.nanpercentile(np.abs(q_b[k]), 95)) if np.isfinite(q_b[k]).any() else np.nan,
                "surface_internal_coupling_p95": float(np.nanpercentile(np.abs(coupling[k]), 95)) if np.isfinite(coupling[k]).any() else np.nan,
                "eta_increment_p95": float(np.nanpercentile(np.abs(eta_increment), 95)) if np.isfinite(eta_increment).any() else np.nan,
                "eta_update_source": "pv_network_linearized",
            }
        )
    return pd.DataFrame(rows)
