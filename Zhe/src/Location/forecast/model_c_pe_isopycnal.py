from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

from ..validation.isopycnal_projection import (
    build_isopycnal_surfaces,
    interpolate_to_surfaces,
    project_surface_field_to_depth,
)
from ..validation.isopycnal_pv import compute_isopycnal_control_volume_pv

from .common import G, finite_or_nan, geo_params, gradient
from .model_b_streamfunction import (
    build_initial_psi_from_birth_velocity,
    laplace_beltrami,
    _regularized_poisson,
    velocity_on_isopycnals,
)
from .models import ForecastState, filled_centers


PV_RESIDUAL_MAX = 0.75
MAX_PHASE_SUBSTEPS = 4
LIFECYCLE_TIME_SCALE_S = 70.0 * 86400.0
ETA_CONTINUITY_LIMIT_FRACTION = 0.35
H_THICKNESS_LIMIT_FRACTION = 0.30
PSI_INVERSION_RELAXATION = 0.35
PSI_POISSON_ITERATIONS = 45
MONTGOMERY_CONSTRAINT_WEIGHT = 0.22
PV_INVERSION_WEIGHT = 0.62
THICKNESS_PSI_WEIGHT = 0.16
RHO0 = 1025.0


@dataclass
class ModelCState:
    base: ForecastState
    rho_levels: np.ndarray
    psi_initial_from_birth_velocity: np.ndarray
    psi_on_isopycnal: np.ndarray
    q_model_c: np.ndarray
    normal_thickness_m: np.ndarray
    surface_area_factor: np.ndarray
    control_volume_dv: np.ndarray
    eta_tendency: np.ndarray
    h_tendency: np.ndarray
    q_closed_control_volume: np.ndarray
    pv_inversion_rhs: np.ndarray
    pv_closure_used_mask: np.ndarray
    eta_mass_residual: np.ndarray
    h_pred_m: np.ndarray
    h_repaired_fraction: np.ndarray
    surface_pressure_pa: np.ndarray
    pressure_on_isopycnal_pa: np.ndarray
    hydrostatic_pressure_increment_pa: np.ndarray
    montgomery_potential: np.ndarray
    montgomery_pressure_gradient_residual: np.ndarray
    montgomery_streamfunction_residual: np.ndarray
    psi_initial_reconstruction_rmse: np.ndarray
    psi_initial_divergent_residual: np.ndarray
    pv_closure_residual_c: np.ndarray
    pv_gate_closed: np.ndarray
    model_c_driver_confidence: np.ndarray


def build_model_c_state(
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
) -> ModelCState:
    """Build the PE-isopycnal Model C birth state.

    Birth velocity is used once to initialize Psi_i. Later phase steps use only
    the evolving surface/isopycnal PV-closure state.
    """
    gp = geo_params(latitude_ref)
    sigma_total = np.asarray(sigma_clim, dtype="f8")[:, None, None] + finite_or_nan(sigma_birth)
    surfaces = build_isopycnal_surfaces(sigma_total, depth, smooth_sigma_grid=1.0)
    psi0 = build_initial_psi_from_birth_velocity(
        finite_or_nan(u_birth),
        finite_or_nan(v_birth),
        surfaces.z_m,
        np.asarray(depth, dtype="f8"),
        np.asarray(x, dtype="f8"),
        np.asarray(y, dtype="f8"),
        float(radius_m),
    )
    u_s, v_s = velocity_on_isopycnals(psi0, surfaces.z_m, x, y, radius_m)
    init_rmse, init_div = _initial_psi_reconstruction_metrics(
        psi0,
        finite_or_nan(u_birth),
        finite_or_nan(v_birth),
        surfaces.z_m,
        depth,
        x,
        y,
        radius_m,
    )
    u_depth = project_surface_field_to_depth(u_s, surfaces.z_m, depth)
    v_depth = project_surface_field_to_depth(v_s, surfaces.z_m, depth)
    pv = compute_isopycnal_control_volume_pv(
        u_depth,
        v_depth,
        sigma_total,
        surfaces.rho_levels,
        depth,
        x,
        y,
        radius_m,
        float(gp.f0),
        {"model": "model_C_PE_isopycnal_PV_closure"},
    )
    metrics = compute_pe_isopycnal_metrics(surfaces.z_m, finite_or_nan(adt_birth), x, y, radius_m)
    gate = _pv_gate_to_layers(pv.pv_balance_residual, surfaces.rho_levels.size)
    q_closed = closed_pv_rhs_to_surfaces(pv.q_star, surfaces.z_m, depth, gate)
    pressure = compute_montgomery_pressure_closure(surfaces.rho_levels, surfaces.z_m, finite_or_nan(adt_birth), metrics["normal_thickness_m"], x, y, radius_m)
    montgomery = pressure["montgomery_potential"]
    pv_rhs = build_pv_inversion_rhs(q_closed, metrics["normal_thickness_m"], montgomery, psi0, gate, x, y, radius_m, float(gp.f0))
    montgomery_residual = _montgomery_streamfunction_residual(psi0, montgomery, float(gp.f0))
    confidence = _model_c_confidence(gate, surfaces.valid_fraction)
    cx, cy = _gated_pv_driver_profile(pv.pv_centroid_x_R, pv.pv_centroid_y_R, pv.pv_balance_residual, surfaces.rho_levels.size)
    source = np.full(cx.shape, "model_C_PE_isopycnal_PV_closure_network", dtype="U64")
    base = ForecastState(
        finite_or_nan(sigma_birth),
        finite_or_nan(adt_birth),
        np.asarray(sigma_clim, dtype="f8"),
        np.asarray(depth, dtype="f8"),
        np.asarray(x, dtype="f8"),
        np.asarray(y, dtype="f8"),
        float(radius_m),
        float(gp.f0),
        cx,
        cy,
        source,
        confidence,
    )
    q_c = laplace_beltrami(psi0, surfaces.z_m, x, y, radius_m)
    return ModelCState(
        base=base,
        rho_levels=surfaces.rho_levels,
        psi_initial_from_birth_velocity=psi0.copy(),
        psi_on_isopycnal=psi0.copy(),
        q_model_c=q_c,
        normal_thickness_m=metrics["normal_thickness_m"],
        surface_area_factor=metrics["surface_area_factor"],
        control_volume_dv=metrics["control_volume_dv"],
        eta_tendency=np.zeros_like(finite_or_nan(adt_birth), dtype="f8"),
        h_tendency=np.zeros_like(psi0, dtype="f8"),
        q_closed_control_volume=q_closed,
        pv_inversion_rhs=pv_rhs,
        pv_closure_used_mask=gate,
        eta_mass_residual=np.zeros_like(finite_or_nan(adt_birth), dtype="f8"),
        h_pred_m=metrics["normal_thickness_m"],
        h_repaired_fraction=np.zeros(surfaces.rho_levels.size, dtype="f8"),
        surface_pressure_pa=pressure["surface_pressure_pa"],
        pressure_on_isopycnal_pa=pressure["pressure_on_isopycnal_pa"],
        hydrostatic_pressure_increment_pa=pressure["hydrostatic_pressure_increment_pa"],
        montgomery_potential=montgomery,
        montgomery_pressure_gradient_residual=pressure["montgomery_pressure_gradient_residual"],
        montgomery_streamfunction_residual=montgomery_residual,
        psi_initial_reconstruction_rmse=init_rmse,
        psi_initial_divergent_residual=init_div,
        pv_closure_residual_c=_pv_residual_to_layers(pv.pv_balance_residual, surfaces.rho_levels.size),
        pv_gate_closed=gate,
        model_c_driver_confidence=confidence,
    )


def forecast_model_c_phase(
    state: ModelCState,
    tau: float,
    strength: float,
    shape: str,
    polarity: str,
    phase: str,
    phase_index: int,
    model: str,
) -> tuple[np.ndarray, ...]:
    (
        sigma_pred,
        adt_pred,
        psi,
        q_c,
        eta_t,
        h_t,
        surfaces,
        metrics,
        pv,
        q_closed,
        pv_rhs,
        used_mask,
        eta_mass_resid,
        h_repaired,
        pressure,
        montgomery,
        montgomery_grad_resid,
        montgomery_resid,
    ) = step_model_c_pe_isopycnal(state, tau, strength)
    u_s, v_s = velocity_on_isopycnals(psi, surfaces.z_m, state.base.x, state.base.y, state.base.radius_m)
    u_depth = project_surface_field_to_depth(u_s, surfaces.z_m, state.base.depth)
    v_depth = project_surface_field_to_depth(v_s, surfaces.z_m, state.base.depth)
    table = _diagnostic_table(
        pv.table,
        surfaces.valid_fraction,
        metrics,
        psi,
        q_c,
        eta_t,
        h_t,
        q_closed,
        pv_rhs,
        used_mask,
        eta_mass_resid,
        h_repaired,
        pressure,
        montgomery,
        montgomery_grad_resid,
        montgomery_resid,
        shape,
        polarity,
        phase,
        phase_index,
        model,
    )
    return (
        sigma_pred,
        adt_pred,
        u_depth,
        v_depth,
        surfaces.z_anom_m,
        psi,
        q_c,
        metrics["normal_thickness_m"],
        metrics["surface_area_factor"],
        metrics["control_volume_dv"],
        eta_t,
        h_t,
        q_closed,
        pv_rhs,
        used_mask,
        eta_mass_resid,
        metrics["normal_thickness_m"],
        h_repaired,
        pressure["surface_pressure_pa"],
        pressure["pressure_on_isopycnal_pa"],
        pressure["hydrostatic_pressure_increment_pa"],
        montgomery,
        montgomery_grad_resid,
        montgomery_resid,
        table,
    )


def step_model_c_pe_isopycnal(
    state: ModelCState,
    tau: float,
    strength: float = 1.0,
):
    """Advance Model C with PE-isopycnal thickness/PV-closure diagnostics."""
    sigma = finite_or_nan(state.base.sigma_birth).copy()
    adt = finite_or_nan(state.base.adt_birth).copy()
    psi = finite_or_nan(state.psi_initial_from_birth_velocity).copy()
    h_state = finite_or_nan(state.h_pred_m).copy()
    eta_total = np.zeros_like(adt, dtype="f8")
    h_total = np.zeros_like(psi, dtype="f8")
    eta_mass_total = np.zeros_like(adt, dtype="f8")
    h_repaired_total = np.zeros(psi.shape[0], dtype="f8")
    step_count = 0
    q_closed = finite_or_nan(state.q_closed_control_volume).copy()
    pv_rhs = finite_or_nan(state.pv_inversion_rhs).copy()
    used_mask = finite_or_nan(state.pv_closure_used_mask).copy()
    pressure = {
        "surface_pressure_pa": finite_or_nan(state.surface_pressure_pa).copy(),
        "pressure_on_isopycnal_pa": finite_or_nan(state.pressure_on_isopycnal_pa).copy(),
        "hydrostatic_pressure_increment_pa": finite_or_nan(state.hydrostatic_pressure_increment_pa).copy(),
        "montgomery_pressure_gradient_residual": finite_or_nan(state.montgomery_pressure_gradient_residual).copy(),
        "montgomery_potential": finite_or_nan(state.montgomery_potential).copy(),
    }
    montgomery = finite_or_nan(state.montgomery_potential).copy()
    montgomery_grad_resid = finite_or_nan(state.montgomery_pressure_gradient_residual).copy()
    montgomery_resid = finite_or_nan(state.montgomery_streamfunction_residual).copy()
    tau = float(tau)
    if abs(tau) > 1.0e-12:
        nsteps = max(1, min(MAX_PHASE_SUBSTEPS, int(np.ceil(abs(tau) * MAX_PHASE_SUBSTEPS))))
        dtau = tau * float(strength) / nsteps
        dt_seconds = abs(dtau) * LIFECYCLE_TIME_SCALE_S
        for _ in range(nsteps):
            surfaces = build_isopycnal_surfaces(
                state.base.sigma_clim[:, None, None] + sigma,
                state.base.depth,
                rho_levels=state.rho_levels,
                smooth_sigma_grid=1.0,
            )
            metrics = compute_pe_isopycnal_metrics(surfaces.z_m, adt, state.base.x, state.base.y, state.base.radius_m)
            h_state = _blend_thickness_state(h_state, metrics["normal_thickness_m"])
            u_s, v_s = velocity_on_isopycnals(psi, surfaces.z_m, state.base.x, state.base.y, state.base.radius_m)
            h_t = update_normal_thickness_continuity(h_state, u_s, v_s, state.base.x, state.base.y, state.base.radius_m)
            h_state, h_delta, repaired = _advance_thickness_state(h_state, h_t, dt_seconds)
            u_depth = project_surface_field_to_depth(u_s, surfaces.z_m, state.base.depth)
            v_depth = project_surface_field_to_depth(v_s, surfaces.z_m, state.base.depth)
            pv = compute_isopycnal_control_volume_pv(
                u_depth,
                v_depth,
                state.base.sigma_clim[:, None, None] + sigma,
                surfaces.rho_levels,
                state.base.depth,
                state.base.x,
                state.base.y,
                state.base.radius_m,
                state.base.f0,
                {"model": "model_C_PE_isopycnal_PV_closure"},
            )
            gate = _pv_gate_to_layers(pv.pv_balance_residual, surfaces.rho_levels.size)
            q_closed = closed_pv_rhs_to_surfaces(pv.q_star, surfaces.z_m, state.base.depth, gate)
            pressure = compute_montgomery_pressure_closure(surfaces.rho_levels, surfaces.z_m, adt, h_state, state.base.x, state.base.y, state.base.radius_m)
            montgomery = pressure["montgomery_potential"]
            montgomery_grad_resid = pressure["montgomery_pressure_gradient_residual"]
            pv_rhs = build_pv_inversion_rhs(q_closed, h_state, montgomery, psi, gate, state.base.x, state.base.y, state.base.radius_m, state.base.f0)
            psi, q_c, montgomery_resid = solve_closed_pv_streamfunction_step(
                psi,
                pv_rhs,
                h_state,
                montgomery,
                surfaces.z_m,
                state.base.x,
                state.base.y,
                state.base.radius_m,
                state.base.f0,
                gate,
            )
            eta_t, eta_mass_resid = update_eta_continuity(h_state, u_s, v_s, adt, state.base.x, state.base.y, state.base.radius_m)
            eta_step = eta_t * dt_seconds
            sigma_step = _sigma_increment_from_thickness(h_delta, sigma, state.base.sigma_clim, state.base.depth)
            adt = _bounded_add(adt, eta_step, state.base.adt_birth, max_fraction=ETA_CONTINUITY_LIMIT_FRACTION)
            sigma = _bounded_add(sigma, sigma_step, state.base.sigma_birth, max_fraction=H_THICKNESS_LIMIT_FRACTION)
            eta_total += eta_t
            eta_mass_total += eta_mass_resid
            h_total += h_t
            h_repaired_total += repaired
            step_count += 1
            used_mask = gate
        eta_total /= max(step_count, 1)
        eta_mass_total /= max(step_count, 1)
        h_total /= max(step_count, 1)
        h_repaired_total /= max(step_count, 1)
    surfaces = build_isopycnal_surfaces(
        state.base.sigma_clim[:, None, None] + sigma,
        state.base.depth,
        rho_levels=state.rho_levels,
        smooth_sigma_grid=1.0,
    )
    metrics = compute_pe_isopycnal_metrics(surfaces.z_m, adt, state.base.x, state.base.y, state.base.radius_m)
    h_state = _blend_thickness_state(h_state, metrics["normal_thickness_m"])
    u_s, v_s = velocity_on_isopycnals(psi, surfaces.z_m, state.base.x, state.base.y, state.base.radius_m)
    u_depth = project_surface_field_to_depth(u_s, surfaces.z_m, state.base.depth)
    v_depth = project_surface_field_to_depth(v_s, surfaces.z_m, state.base.depth)
    pv = compute_isopycnal_control_volume_pv(
        u_depth,
        v_depth,
        state.base.sigma_clim[:, None, None] + sigma,
        surfaces.rho_levels,
        state.base.depth,
        state.base.x,
        state.base.y,
        state.base.radius_m,
        state.base.f0,
        {"model": "model_C_PE_isopycnal_PV_closure"},
    )
    q_c = laplace_beltrami(psi, surfaces.z_m, state.base.x, state.base.y, state.base.radius_m)
    if not np.any(np.isfinite(h_total)):
        h_total = update_normal_thickness_continuity(h_state, u_s, v_s, state.base.x, state.base.y, state.base.radius_m)
    gate = _pv_gate_to_layers(pv.pv_balance_residual, surfaces.rho_levels.size)
    q_closed = closed_pv_rhs_to_surfaces(pv.q_star, surfaces.z_m, state.base.depth, gate)
    pressure = compute_montgomery_pressure_closure(surfaces.rho_levels, surfaces.z_m, adt, h_state, state.base.x, state.base.y, state.base.radius_m)
    montgomery = pressure["montgomery_potential"]
    montgomery_grad_resid = pressure["montgomery_pressure_gradient_residual"]
    pv_rhs = build_pv_inversion_rhs(q_closed, h_state, montgomery, psi, gate, state.base.x, state.base.y, state.base.radius_m, state.base.f0)
    montgomery_resid = _montgomery_streamfunction_residual(psi, montgomery, state.base.f0)
    metrics["normal_thickness_m"] = h_state
    h_repaired_fraction = h_repaired_total
    return sigma, adt, psi, q_c, eta_total, h_total, surfaces, metrics, pv, q_closed, pv_rhs, gate, eta_mass_total, h_repaired_fraction, pressure, montgomery, montgomery_grad_resid, montgomery_resid


def compute_pe_isopycnal_metrics(
    z_surface_m: np.ndarray,
    adt_m: np.ndarray,
    x_R: np.ndarray,
    y_R: np.ndarray,
    radius_m: float,
) -> dict[str, np.ndarray]:
    x_m = np.asarray(x_R, dtype="f8") * max(float(radius_m), 1.0)
    y_m = np.asarray(y_R, dtype="f8") * max(float(radius_m), 1.0)
    z = finite_or_nan(z_surface_m)
    zx = gradient(z, x_m, axis=2)
    zy = gradient(z, y_m, axis=1)
    sqrt_g = np.sqrt(1.0 + zx * zx + zy * zy)
    bins = np.maximum(np.diff(z, axis=0), 0.0)
    area_bins = 0.5 * (sqrt_g[:-1] + sqrt_g[1:])
    thickness_bins = np.divide(bins, area_bins, out=np.full_like(bins, np.nan), where=area_bins > 0)
    thickness_layers = _control_volume_centers_to_layers(thickness_bins, z.shape[0])
    top_gap = np.maximum(z[0] - finite_or_nan(adt_m), 0.0) if z.shape[0] else np.empty_like(adt_m)
    if thickness_layers.shape[0]:
        thickness_layers[0] = np.where(np.isfinite(top_gap), 0.5 * (thickness_layers[0] + top_gap), thickness_layers[0])
    dx = abs(float(np.nanmedian(np.diff(x_m)))) if x_m.size > 1 else 1.0
    dy = abs(float(np.nanmedian(np.diff(y_m)))) if y_m.size > 1 else 1.0
    control_volume_dv = thickness_bins * area_bins * dx * dy
    return {
        "normal_thickness_m": thickness_layers,
        "surface_area_factor": sqrt_g,
        "control_volume_dv": control_volume_dv,
    }


def update_normal_thickness_continuity(
    h_m: np.ndarray,
    u_s: np.ndarray,
    v_s: np.ndarray,
    x_R: np.ndarray,
    y_R: np.ndarray,
    radius_m: float,
) -> np.ndarray:
    """Adiabatic isopycnal thickness tendency: dh/dt = -div_S(h u)."""
    x_m = np.asarray(x_R, dtype="f8") * max(float(radius_m), 1.0)
    y_m = np.asarray(y_R, dtype="f8") * max(float(radius_m), 1.0)
    h = np.where(np.isfinite(h_m), h_m, 0.0)
    u = np.where(np.isfinite(u_s), u_s, 0.0)
    v = np.where(np.isfinite(v_s), v_s, 0.0)
    tendency = -(gradient(h * u, x_m, axis=2) + gradient(h * v, y_m, axis=1))
    return gaussian_filter(np.where(np.isfinite(tendency), tendency, 0.0), sigma=(0.0, 0.65, 0.65), mode="nearest")


def update_eta_continuity(
    h_m: np.ndarray,
    u_s: np.ndarray,
    v_s: np.ndarray,
    adt_m: np.ndarray,
    x_R: np.ndarray,
    y_R: np.ndarray,
    radius_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Free-surface tendency with zero external mass flux."""
    if h_m.size == 0:
        zero = np.zeros_like(adt_m, dtype="f8")
        return zero, zero
    x_m = np.asarray(x_R, dtype="f8") * max(float(radius_m), 1.0)
    y_m = np.asarray(y_R, dtype="f8") * max(float(radius_m), 1.0)
    h0 = np.where(np.isfinite(h_m[0]), h_m[0], 0.0)
    u0 = np.where(np.isfinite(u_s[0]), u_s[0], 0.0)
    v0 = np.where(np.isfinite(v_s[0]), v_s[0], 0.0)
    raw = -(gradient(h0 * u0, x_m, axis=1) + gradient(h0 * v0, y_m, axis=0))
    mass_residual = np.full_like(raw, np.nanmean(raw) if np.isfinite(raw).any() else 0.0, dtype="f8")
    tendency = raw - mass_residual
    tendency = gaussian_filter(np.where(np.isfinite(tendency), tendency, 0.0), sigma=1.0, mode="nearest")
    return tendency, mass_residual


def compute_montgomery_pressure_closure(
    rho_levels: np.ndarray,
    z_surface_m: np.ndarray,
    adt_m: np.ndarray,
    h_m: np.ndarray,
    x_R: np.ndarray,
    y_R: np.ndarray,
    radius_m: float,
) -> dict[str, np.ndarray]:
    """Pressure-gradient closure for the isopycnal Montgomery potential.

    CMEMS depth is positive downward here. Only horizontal pressure-gradient
    anomalies enter the forecast constraint, so each pressure and height layer
    is de-meaned by its horizontal median before forming M_i.
    """
    z = finite_or_nan(z_surface_m)
    h = finite_or_nan(h_m)
    rho = _rho_levels_to_absolute_density(rho_levels, z.shape[0])
    surface_pressure = RHO0 * G * np.where(np.isfinite(adt_m), adt_m, 0.0)
    hydro_increment = rho[:, None, None] * G * np.where(np.isfinite(h), h, 0.0)
    pressure_raw = surface_pressure[None, :, :] + np.cumsum(hydro_increment, axis=0)
    pressure_anom = pressure_raw - np.nanmedian(pressure_raw, axis=(1, 2), keepdims=True)
    z_anom = z - np.nanmedian(z, axis=(1, 2), keepdims=True)
    mont_raw = pressure_anom / RHO0 + G * z_anom
    montgomery = _smooth_stack(np.where(np.isfinite(mont_raw), mont_raw, 0.0), sigma=0.65)
    montgomery -= np.nanmedian(montgomery, axis=(1, 2), keepdims=True)
    grad_resid = _pressure_gradient_residual(montgomery, mont_raw, x_R, y_R, radius_m)
    return {
        "surface_pressure_pa": surface_pressure,
        "pressure_on_isopycnal_pa": pressure_anom,
        "hydrostatic_pressure_increment_pa": hydro_increment,
        "montgomery_potential": montgomery,
        "montgomery_pressure_gradient_residual": grad_resid,
    }


def compute_montgomery_potential(
    rho_levels: np.ndarray,
    z_surface_m: np.ndarray,
    adt_m: np.ndarray,
    h_m: np.ndarray,
    f0: float,
) -> np.ndarray:
    """Backward-compatible wrapper for pressure-closure Montgomery potential."""
    _ = f0
    dummy_x = np.arange(z_surface_m.shape[2], dtype="f8") if np.ndim(z_surface_m) == 3 else np.arange(1, dtype="f8")
    dummy_y = np.arange(z_surface_m.shape[1], dtype="f8") if np.ndim(z_surface_m) == 3 else np.arange(1, dtype="f8")
    return compute_montgomery_pressure_closure(rho_levels, z_surface_m, adt_m, h_m, dummy_x, dummy_y, 1.0)["montgomery_potential"]


def closed_pv_rhs_to_surfaces(q_star: np.ndarray, z_surface_m: np.ndarray, depth_m: np.ndarray, gate_layers: np.ndarray) -> np.ndarray:
    q_s = interpolate_to_surfaces(finite_or_nan(q_star), np.asarray(depth_m, dtype="f8"), finite_or_nan(z_surface_m))
    q_s = _smooth_stack(np.where(np.isfinite(q_s), q_s, 0.0), sigma=0.65)
    return q_s * np.asarray(gate_layers, dtype="f8")[:, None, None]


def _rho_levels_to_absolute_density(rho_levels: np.ndarray, n_layers: int) -> np.ndarray:
    rho = np.asarray(rho_levels, dtype="f8")
    if rho.size == 0:
        return np.full(n_layers, RHO0, dtype="f8")
    if rho.size != n_layers:
        idx_src = np.linspace(0.0, 1.0, rho.size)
        idx_dst = np.linspace(0.0, 1.0, n_layers)
        rho = np.interp(idx_dst, idx_src, rho)
    # sigma0 is normally density anomaly relative to 1000 kg/m^3.
    rho_abs = np.where(np.nanmedian(rho) < 200.0, 1000.0 + rho, rho)
    rho_abs = np.where(np.isfinite(rho_abs) & (rho_abs > 900.0), rho_abs, RHO0)
    return rho_abs.astype("f8")


def _pressure_gradient_residual(
    montgomery: np.ndarray,
    raw_montgomery: np.ndarray,
    x_R: np.ndarray,
    y_R: np.ndarray,
    radius_m: float,
) -> np.ndarray:
    x_m = np.asarray(x_R, dtype="f8") * max(float(radius_m), 1.0)
    y_m = np.asarray(y_R, dtype="f8") * max(float(radius_m), 1.0)
    diff = finite_or_nan(montgomery) - finite_or_nan(raw_montgomery)
    gx = gradient(np.where(np.isfinite(diff), diff, 0.0), x_m, axis=2)
    gy = gradient(np.where(np.isfinite(diff), diff, 0.0), y_m, axis=1)
    return np.sqrt(gx * gx + gy * gy)


def build_pv_inversion_rhs(
    q_closed: np.ndarray,
    h_m: np.ndarray,
    montgomery_potential: np.ndarray,
    psi: np.ndarray,
    gate_layers: np.ndarray,
    x_R: np.ndarray,
    y_R: np.ndarray,
    radius_m: float,
    f0: float,
) -> np.ndarray:
    length_scale = max(float(radius_m), 1.0)
    psi_scale = _safe_p95(psi, fallback=1.0)
    h_anom = h_m - np.nanmedian(h_m, axis=(1, 2), keepdims=True)
    mont_psi = _scaled_montgomery_streamfunction(montgomery_potential, psi, f0)
    rhs = (
        PV_INVERSION_WEIGHT * _normalize(q_closed)
        + THICKNESS_PSI_WEIGHT * _normalize(h_anom)
        + MONTGOMERY_CONSTRAINT_WEIGHT * _normalize(mont_psi - psi)
    )
    rhs *= psi_scale / (length_scale * length_scale)
    return _smooth_stack(rhs * np.asarray(gate_layers, dtype="f8")[:, None, None], sigma=0.55)


def solve_closed_pv_streamfunction_step(
    psi: np.ndarray,
    pv_inversion_rhs: np.ndarray,
    h_m: np.ndarray,
    montgomery_potential: np.ndarray,
    z_surface_m: np.ndarray,
    x_R: np.ndarray,
    y_R: np.ndarray,
    radius_m: float,
    f0: float,
    gate_layers: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_m = np.asarray(x_R, dtype="f8") * max(float(radius_m), 1.0)
    y_m = np.asarray(y_R, dtype="f8") * max(float(radius_m), 1.0)
    dx = abs(float(np.nanmedian(np.diff(x_m)))) if x_m.size > 1 else 1.0
    dy = abs(float(np.nanmedian(np.diff(y_m)))) if y_m.size > 1 else 1.0
    psi_arr = finite_or_nan(psi)
    mont_target = _scaled_montgomery_streamfunction(montgomery_potential, psi_arr, f0)
    vertical_target = _vertical_neighbor_average(psi_arr)
    solved = np.full_like(psi_arr, np.nan, dtype="f8")
    for k in range(psi_arr.shape[0]):
        if k < len(gate_layers) and not (np.isfinite(gate_layers[k]) and gate_layers[k] > 0.5):
            solved[k] = psi_arr[k]
            continue
        rhs = np.where(np.isfinite(pv_inversion_rhs[k]), pv_inversion_rhs[k], 0.0)
        inv = _regularized_poisson(rhs, dx, dy, regularization=1.0e-3, iterations=PSI_POISSON_ITERATIONS)
        inv -= np.nanmedian(inv)
        solved[k] = inv
    target = (
        PV_INVERSION_WEIGHT * _scale_field_to(solved, psi_arr)
        + MONTGOMERY_CONSTRAINT_WEIGHT * mont_target
        + THICKNESS_PSI_WEIGHT * vertical_target
    )
    gate = np.asarray(gate_layers, dtype="f8")[:, None, None]
    alpha = PSI_INVERSION_RELAXATION * (0.20 + 0.80 * gate)
    psi_next = (1.0 - alpha) * psi_arr + alpha * target
    psi_next = _smooth_stack(np.where(np.isfinite(psi_next), psi_next, psi_arr), sigma=0.55)
    q_next = laplace_beltrami(psi_next, z_surface_m, x_R, y_R, radius_m)
    mont_resid = _montgomery_streamfunction_residual(psi_next, montgomery_potential, f0)
    return psi_next, q_next, mont_resid


def _blend_thickness_state(current: np.ndarray, geometry: np.ndarray) -> np.ndarray:
    cur = finite_or_nan(current)
    geo = finite_or_nan(geometry)
    if cur.shape != geo.shape or not np.isfinite(cur).any():
        return geo
    return np.where(np.isfinite(geo), 0.85 * cur + 0.15 * geo, cur)


def _advance_thickness_state(h_state: np.ndarray, h_tendency: np.ndarray, dt_seconds: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h = np.where(np.isfinite(h_state), h_state, np.nan)
    delta = np.where(np.isfinite(h_tendency), h_tendency * float(dt_seconds), 0.0)
    limit = np.maximum(0.05, 0.25 * np.nanpercentile(np.abs(h), 95, axis=(1, 2))[:, None, None])
    delta = np.clip(delta, -limit, limit)
    h_next = h + delta
    positive = np.where(h > 0, h, np.nan)
    floor = np.nanpercentile(positive, 5, axis=(1, 2))[:, None, None]
    floor = np.where(np.isfinite(floor) & (floor > 0), 0.05 * floor, 0.01)
    repaired_mask = ~np.isfinite(h_next) | (h_next < floor)
    h_next = np.where(repaired_mask, floor, h_next)
    col0 = np.nansum(h, axis=0, keepdims=True)
    col1 = np.nansum(h_next, axis=0, keepdims=True)
    h_next -= (col1 - col0) / max(h.shape[0], 1)
    repaired_mask |= h_next < floor
    h_next = np.where(h_next < floor, floor, h_next)
    repaired_fraction = np.nanmean(repaired_mask, axis=(1, 2))
    return h_next, h_next - h, np.where(np.isfinite(repaired_fraction), repaired_fraction, 0.0)


def _sigma_increment_from_thickness(h_delta: np.ndarray, sigma: np.ndarray, sigma_clim: np.ndarray, depth: np.ndarray) -> np.ndarray:
    total_sigma = np.asarray(sigma_clim, dtype="f8")[:, None, None] + finite_or_nan(sigma)
    ds_dz = gradient(total_sigma, np.asarray(depth, dtype="f8"), axis=0)
    displacement = np.cumsum(np.where(np.isfinite(h_delta), h_delta, 0.0), axis=0)
    displacement -= np.nanmedian(displacement, axis=0, keepdims=True)
    return _smooth_stack(-displacement * ds_dz, sigma=0.65)


def _scaled_montgomery_streamfunction(montgomery_potential: np.ndarray, psi: np.ndarray, f0: float) -> np.ndarray:
    scale_f = max(abs(float(f0)), 1.0e-12)
    raw = finite_or_nan(montgomery_potential) / scale_f
    raw -= np.nanmedian(raw, axis=(1, 2), keepdims=True)
    return _scale_field_to(raw, psi)


def _montgomery_streamfunction_residual(psi: np.ndarray, montgomery_potential: np.ndarray, f0: float) -> np.ndarray:
    target = _scaled_montgomery_streamfunction(montgomery_potential, psi, f0)
    return np.where(np.isfinite(psi), finite_or_nan(psi) - target, np.nan)


def _vertical_neighbor_average(field: np.ndarray) -> np.ndarray:
    arr = finite_or_nan(field)
    out = arr.copy()
    if arr.shape[0] == 1:
        return out
    out[0] = 0.67 * arr[0] + 0.33 * arr[1]
    out[-1] = 0.67 * arr[-1] + 0.33 * arr[-2]
    if arr.shape[0] > 2:
        out[1:-1] = 0.25 * arr[:-2] + 0.50 * arr[1:-1] + 0.25 * arr[2:]
    return out


def _scale_field_to(field: np.ndarray, reference: np.ndarray) -> np.ndarray:
    arr = finite_or_nan(field)
    ref = finite_or_nan(reference)
    arr -= np.nanmedian(arr, axis=(1, 2), keepdims=True)
    src = np.nanpercentile(np.abs(arr), 95, axis=(1, 2))[:, None, None]
    dst = np.nanpercentile(np.abs(ref), 95, axis=(1, 2))[:, None, None]
    src = np.where(np.isfinite(src) & (src > 1.0e-30), src, 1.0)
    dst = np.where(np.isfinite(dst) & (dst > 1.0e-30), dst, 1.0)
    return arr / src * dst


def _initial_psi_reconstruction_metrics(
    psi: np.ndarray,
    u_birth: np.ndarray,
    v_birth: np.ndarray,
    z_surface_m: np.ndarray,
    depth_m: np.ndarray,
    x_R: np.ndarray,
    y_R: np.ndarray,
    radius_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    u_s_obs = interpolate_to_surfaces(finite_or_nan(u_birth), np.asarray(depth_m, dtype="f8"), finite_or_nan(z_surface_m))
    v_s_obs = interpolate_to_surfaces(finite_or_nan(v_birth), np.asarray(depth_m, dtype="f8"), finite_or_nan(z_surface_m))
    u_s_fit, v_s_fit = velocity_on_isopycnals(psi, z_surface_m, x_R, y_R, radius_m)
    diff2 = (u_s_fit - u_s_obs) ** 2 + (v_s_fit - v_s_obs) ** 2
    rmse = np.sqrt(np.nanmean(diff2, axis=(1, 2)))
    x_m = np.asarray(x_R, dtype="f8") * max(float(radius_m), 1.0)
    y_m = np.asarray(y_R, dtype="f8") * max(float(radius_m), 1.0)
    div = gradient(np.where(np.isfinite(u_s_fit), u_s_fit, 0.0), x_m, axis=2) + gradient(np.where(np.isfinite(v_s_fit), v_s_fit, 0.0), y_m, axis=1)
    div_resid = np.nanpercentile(np.abs(div), 95, axis=(1, 2))
    return np.where(np.isfinite(rmse), rmse, np.nan), np.where(np.isfinite(div_resid), div_resid, np.nan)


def _diagnostic_table(
    pv_table: pd.DataFrame,
    valid_fraction: np.ndarray,
    metrics: dict[str, np.ndarray],
    psi: np.ndarray,
    q_c: np.ndarray,
    eta_t: np.ndarray,
    h_t: np.ndarray,
    q_closed: np.ndarray,
    pv_rhs: np.ndarray,
    used_mask: np.ndarray,
    eta_mass_residual: np.ndarray,
    h_repaired_fraction: np.ndarray,
    pressure_closure: dict[str, np.ndarray],
    montgomery_potential: np.ndarray,
    montgomery_gradient_residual: np.ndarray,
    montgomery_residual: np.ndarray,
    shape: str,
    polarity: str,
    phase: str,
    phase_index: int,
    model: str,
) -> pd.DataFrame:
    rows = []
    residual_layers = _pv_residual_to_layers(
        pv_table["pv_balance_residual"].to_numpy(dtype="f8") if "pv_balance_residual" in pv_table else np.asarray([]),
        psi.shape[0],
    )
    gate_layers = _pv_gate_to_layers(
        pv_table["pv_balance_residual"].to_numpy(dtype="f8") if "pv_balance_residual" in pv_table else np.asarray([]),
        psi.shape[0],
    )
    h = metrics["normal_thickness_m"]
    area = metrics["surface_area_factor"]
    dvol = metrics["control_volume_dv"]
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
                "pv_balance_residual": float(residual_layers[k]) if k < residual_layers.size else np.nan,
                "pv_gate_closed": bool(gate_layers[k] > 0.5) if k < gate_layers.size else False,
                "valid_fraction": float(valid_fraction[k]) if k < len(valid_fraction) else np.nan,
                "normal_thickness_p50_m": _nanpercentile(h[k], 50),
                "normal_thickness_p95_m": _nanpercentile(h[k], 95),
                "surface_area_factor_p95": _nanpercentile(area[k], 95),
                "control_volume_dv_p95": _nanpercentile(dvol[min(k, max(dvol.shape[0] - 1, 0))], 95) if dvol.size else np.nan,
                "psi_p95": _nanpercentile(np.abs(psi[k]), 95),
                "q_model_C_p95": _nanpercentile(np.abs(q_c[k]), 95),
                "q_closed_control_volume_p95": _nanpercentile(np.abs(q_closed[k]), 95),
                "pv_inversion_rhs_p95": _nanpercentile(np.abs(pv_rhs[k]), 95),
                "pv_closure_used_mask": bool(used_mask[k] > 0.5) if k < len(used_mask) and np.isfinite(used_mask[k]) else False,
                "eta_tendency_p95": _nanpercentile(np.abs(eta_t), 95),
                "eta_mass_residual_p95": _nanpercentile(np.abs(eta_mass_residual), 95),
                "h_tendency_p95": _nanpercentile(np.abs(h_t[k]), 95),
                "h_repaired_fraction": float(h_repaired_fraction[k]) if k < len(h_repaired_fraction) else np.nan,
                "surface_pressure_p95_pa": _nanpercentile(np.abs(pressure_closure["surface_pressure_pa"]), 95),
                "pressure_on_isopycnal_p95_pa": _nanpercentile(np.abs(pressure_closure["pressure_on_isopycnal_pa"][k]), 95),
                "hydrostatic_pressure_increment_p95_pa": _nanpercentile(np.abs(pressure_closure["hydrostatic_pressure_increment_pa"][k]), 95),
                "montgomery_potential_p95": _nanpercentile(np.abs(montgomery_potential[k]), 95),
                "montgomery_pressure_gradient_residual_p95": _nanpercentile(np.abs(montgomery_gradient_residual[k]), 95),
                "montgomery_streamfunction_residual_p95": _nanpercentile(np.abs(montgomery_residual[k]), 95),
                "eta_update_source": "free_surface_continuity_zero_external_mass_flux",
                "h_update_source": "isopycnal_thickness_continuity",
                "psi_update_source": "closed_pv_rhs_plus_montgomery_constraint",
            }
        )
    if not pv_table.empty:
        model_rows = pv_table.copy()
        model_rows["model"] = model
        for col in rows[0].keys() if rows else []:
            if col not in model_rows:
                model_rows[col] = np.nan
        return pd.concat([model_rows, pd.DataFrame(rows)], ignore_index=True, sort=False)
    return pd.DataFrame(rows)


def _gated_pv_driver_profile(cx_bin: np.ndarray, cy_bin: np.ndarray, residual_bin: np.ndarray, n_layers: int) -> tuple[np.ndarray, np.ndarray]:
    gate = np.asarray(residual_bin, dtype="f8") <= PV_RESIDUAL_MAX
    cx = np.where(gate, cx_bin, np.nan)
    cy = np.where(gate, cy_bin, np.nan)
    return filled_centers(_control_volume_centers_to_layers(cx, n_layers)), filled_centers(_control_volume_centers_to_layers(cy, n_layers))


def _model_c_confidence(gate_layers: np.ndarray, valid_fraction: np.ndarray) -> np.ndarray:
    valid = np.asarray(valid_fraction, dtype="f8")
    gate = np.asarray(gate_layers, dtype="f8")
    if valid.size != gate.size:
        valid = np.resize(valid, gate.size)
    return np.clip(gate * valid, 0.0, 1.0)


def _pv_gate_to_layers(residual_bin: np.ndarray, n_layers: int) -> np.ndarray:
    residual = _pv_residual_to_layers(residual_bin, n_layers)
    return np.where(np.isfinite(residual) & (residual <= PV_RESIDUAL_MAX), 1.0, 0.0)


def _pv_residual_to_layers(residual_bin: np.ndarray, n_layers: int) -> np.ndarray:
    arr = np.asarray(residual_bin, dtype="f8")
    if arr.size == n_layers:
        return arr.copy()
    if arr.size == 0:
        return np.full(n_layers, np.nan, dtype="f8")
    if arr.size == n_layers - 1:
        return _control_volume_centers_to_layers(arr, n_layers)
    x_old = np.linspace(0.0, 1.0, arr.size)
    x_new = np.linspace(0.0, 1.0, n_layers)
    good = np.isfinite(arr)
    if np.count_nonzero(good) < 2:
        return np.full(n_layers, np.nan, dtype="f8")
    return np.interp(x_new, x_old[good], arr[good])


def _control_volume_centers_to_layers(values: np.ndarray, n_layers: int) -> np.ndarray:
    arr = np.asarray(values, dtype="f8")
    if arr.ndim == 1:
        out = np.full(n_layers, np.nan, dtype="f8")
        if arr.size == n_layers:
            return arr.copy()
        if arr.size == n_layers - 1 and arr.size:
            out[0] = arr[0]
            out[-1] = arr[-1]
            if n_layers > 2:
                out[1:-1] = 0.5 * (arr[:-1] + arr[1:])
            return out
        return out
    out_shape = (n_layers,) + arr.shape[1:]
    out = np.full(out_shape, np.nan, dtype="f8")
    if arr.shape[0] == n_layers:
        return arr.copy()
    if arr.shape[0] == n_layers - 1 and arr.shape[0]:
        out[0] = arr[0]
        out[-1] = arr[-1]
        if n_layers > 2:
            out[1:-1] = 0.5 * (arr[:-1] + arr[1:])
    return out


def _bounded_add(base: np.ndarray, delta: np.ndarray, reference: np.ndarray, max_fraction: float) -> np.ndarray:
    base_arr = np.asarray(base, dtype="f8")
    step = np.asarray(delta, dtype="f8")
    limit = max(float(max_fraction) * _safe_p95(reference, fallback=_safe_p95(base_arr, fallback=1.0)), 1.0e-12)
    step = np.clip(step, -limit, limit)
    out = base_arr + step
    return np.where(np.isfinite(base_arr), out, np.nan)


def _normalize(field: np.ndarray) -> np.ndarray:
    arr = np.asarray(field, dtype="f8")
    med = np.nanmedian(arr) if np.isfinite(arr).any() else 0.0
    centered = arr - med
    scale = _safe_p95(centered, fallback=1.0)
    return np.divide(centered, scale, out=np.zeros_like(centered, dtype="f8"), where=np.isfinite(centered))


def _smooth_stack(field: np.ndarray, sigma: float) -> np.ndarray:
    arr = np.asarray(field, dtype="f8")
    return gaussian_filter(np.where(np.isfinite(arr), arr, 0.0), sigma=(0.0, float(sigma), float(sigma)), mode="nearest")


def _safe_p95(values: np.ndarray, fallback: float) -> float:
    arr = np.asarray(values, dtype="f8")
    good = np.isfinite(arr)
    if not np.any(good):
        return float(fallback)
    val = float(np.nanpercentile(np.abs(arr[good]), 95))
    if not np.isfinite(val) or val <= 1.0e-30:
        return float(fallback)
    return val


def _nanpercentile(values: np.ndarray, q: float) -> float:
    arr = np.asarray(values, dtype="f8")
    if not np.isfinite(arr).any():
        return np.nan
    return float(np.nanpercentile(arr, q))
