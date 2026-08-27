from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter1d, shift

from ..validation.isopycnal_projection import build_isopycnal_surfaces
from ..validation.isopycnal_pv import compute_isopycnal_control_volume_pv
from .common import finite_or_nan, geo_params, thermal_wind_velocity

from .models import ForecastState, filled_centers


PV_CLOSURE_DRIVER_MAX = 0.75
DRIVER_SMOOTH_SIGMA = 1.0


def build_model_a_state(
    sigma_birth: np.ndarray,
    adt_birth: np.ndarray,
    sigma_clim: np.ndarray,
    depth: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    radius_m: float,
    latitude_ref: float,
) -> ForecastState:
    gp = geo_params(latitude_ref)
    sigma_total = np.asarray(sigma_clim, dtype="f8")[:, None, None] + finite_or_nan(sigma_birth)
    u0, v0 = thermal_wind_velocity(sigma_birth, adt_birth, depth, x, y, radius_m, gp.f0)
    surfaces = build_isopycnal_surfaces(sigma_total, depth, smooth_sigma_grid=1.0)
    pv = compute_isopycnal_control_volume_pv(
        u0,
        v0,
        sigma_total,
        surfaces.rho_levels,
        depth,
        x,
        y,
        radius_m,
        gp.f0,
        {"model": "birth_initial_model_A_isopycnal"},
    )
    layer_count = int(np.asarray(depth).size)
    pv_weight = _pv_driver_weights(pv.pva_volume_integral, pv.pv_balance_residual)
    surface_x, surface_y = _surface_node_centroid(adt_birth, x, y)
    surface_weight = _surface_node_weight(adt_birth, x, y, pv_weight)
    cx, conf, source = _control_volume_driver_to_layers(
        pv.pv_centroid_x_R,
        pv_weight,
        layer_count,
        surface_x,
        surface_weight,
    )
    cy, _, _ = _control_volume_driver_to_layers(
        pv.pv_centroid_y_R,
        pv_weight,
        layer_count,
        surface_y,
        surface_weight,
    )
    gx, gy = _isopycnal_geometry_centroid_profile(surfaces.z_anom_m, x, y)
    missing = ~np.isfinite(cx) | ~np.isfinite(cy)
    cx = np.where(missing, gx, cx)
    cy = np.where(missing, gy, cy)
    source = np.where(missing, "isopycnal_geometry_fill", source).astype("U48")
    conf = np.where(missing & np.isfinite(gx) & np.isfinite(gy), 0.25, conf)
    center_x = _regularize_driver_profile(cx)
    center_y = _regularize_driver_profile(cy)
    return ForecastState(
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
        _driver_low_confidence(center_x, center_y, conf),
    )


def predict_model_a_state(state: ForecastState, tau: float, strength: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    return _shift_with_isopycnal_centers(state, tau, strength)


def _control_volume_centers_to_layers(values: np.ndarray, layer_count: int) -> np.ndarray:
    """Map N-1 isopycnal control-volume centroids back to N surface/depth nodes."""
    n = int(layer_count)
    if n <= 0:
        return np.asarray([], dtype="f8")
    arr = np.asarray(values, dtype="f8").reshape(-1)
    if arr.size == n:
        return arr.copy()
    if arr.size == 0:
        return np.full(n, np.nan, dtype="f8")
    if arr.size == n - 1:
        out = np.full(n, np.nan, dtype="f8")
        out[0] = arr[0]
        out[-1] = arr[-1]
        if n > 2:
            out[1:-1] = 0.5 * (arr[:-1] + arr[1:])
        return out
    src = np.linspace(0.0, 1.0, arr.size)
    dst = np.linspace(0.0, 1.0, n)
    good = np.isfinite(arr)
    if not np.any(good):
        return np.full(n, np.nan, dtype="f8")
    if np.count_nonzero(good) == 1:
        return np.full(n, float(arr[good][0]), dtype="f8")
    return np.interp(dst, src[good], arr[good])


def _pv_driver_weights(volume_integral: np.ndarray, residual: np.ndarray) -> np.ndarray:
    volume = np.asarray(volume_integral, dtype="f8").reshape(-1)
    res = np.asarray(residual, dtype="f8").reshape(-1)
    good = np.isfinite(volume) & np.isfinite(res) & (res <= PV_CLOSURE_DRIVER_MAX)
    weight = np.zeros_like(volume, dtype="f8")
    weight[good] = np.abs(volume[good]) / (1.0 + np.maximum(res[good], 0.0))
    return weight


def _control_volume_driver_to_layers(
    values: np.ndarray,
    weights: np.ndarray,
    layer_count: int,
    surface_value: float,
    surface_weight: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = int(layer_count)
    out = np.full(n, np.nan, dtype="f8")
    confidence = np.zeros(n, dtype="f8")
    source = np.full(n, "control_volume_missing", dtype="U48")
    vals = np.asarray(values, dtype="f8").reshape(-1)
    w = np.asarray(weights, dtype="f8").reshape(-1)
    if n <= 0:
        return out, confidence, source
    if vals.size != max(n - 1, 0):
        vals = _control_volume_centers_to_layers(vals, n)
        w = np.ones(n, dtype="f8")
        good = np.isfinite(vals)
        out[good] = vals[good]
        confidence[good] = 1.0
        source[good] = "isopycnal_control_volume_network"
        return out, confidence, source
    scale = float(np.nanmedian(w[w > 0])) if np.any(w > 0) else 1.0
    for k in range(n):
        candidates: list[tuple[float, float, str]] = []
        if k == 0 and np.isfinite(surface_value) and surface_weight > 0:
            candidates.append((float(surface_value), float(surface_weight), "surface_node"))
        if vals.size:
            if k == 0:
                bins = [0]
            elif k == n - 1:
                bins = [vals.size - 1]
            else:
                bins = [k - 1, k]
            for b in bins:
                if 0 <= b < vals.size and np.isfinite(vals[b]) and b < w.size and w[b] > 0:
                    candidates.append((float(vals[b]), float(w[b]), "control_volume"))
        if not candidates:
            continue
        total_w = sum(item[1] for item in candidates)
        out[k] = sum(value * weight for value, weight, _ in candidates) / total_w
        confidence[k] = min(total_w / max(scale, 1e-30), 1.0)
        tags = {item[2] for item in candidates}
        source[k] = "surface_isopycnal_control_volume_network" if "surface_node" in tags else "isopycnal_control_volume_network"
    return out, confidence, source


def _surface_node_centroid(field: np.ndarray, x: np.ndarray, y: np.ndarray, search_radius_R: float = 1.5) -> tuple[float, float]:
    arr = finite_or_nan(field)
    xx, yy = np.meshgrid(np.asarray(x, dtype="f8"), np.asarray(y, dtype="f8"))
    central = (np.abs(xx) <= search_radius_R) & (np.abs(yy) <= search_radius_R)
    weights = np.where(central & np.isfinite(arr), np.abs(arr), 0.0)
    total = float(np.nansum(weights))
    if total <= 0:
        return np.nan, np.nan
    return float(np.nansum(weights * xx) / total), float(np.nansum(weights * yy) / total)


def _surface_node_weight(field: np.ndarray, x: np.ndarray, y: np.ndarray, pv_weight: np.ndarray, search_radius_R: float = 1.5) -> float:
    arr = finite_or_nan(field)
    xx, yy = np.meshgrid(np.asarray(x, dtype="f8"), np.asarray(y, dtype="f8"))
    central = (np.abs(xx) <= search_radius_R) & (np.abs(yy) <= search_radius_R)
    has_surface_signal = float(np.nansum(np.where(central & np.isfinite(arr), np.abs(arr), 0.0))) > 0
    if not has_surface_signal:
        return 0.0
    good = np.asarray(pv_weight, dtype="f8")
    return float(np.nanmedian(good[good > 0])) if np.any(good > 0) else 1.0


def _isopycnal_geometry_centroid_profile(z_anom_m: np.ndarray, x: np.ndarray, y: np.ndarray, search_radius_R: float = 1.5) -> tuple[np.ndarray, np.ndarray]:
    z = finite_or_nan(z_anom_m)
    xx, yy = np.meshgrid(np.asarray(x, dtype="f8"), np.asarray(y, dtype="f8"))
    central = (np.abs(xx) <= search_radius_R) & (np.abs(yy) <= search_radius_R)
    cx = np.full(z.shape[0], np.nan, dtype="f8")
    cy = np.full(z.shape[0], np.nan, dtype="f8")
    for k in range(z.shape[0]):
        weights = np.where(central & np.isfinite(z[k]), np.abs(z[k]), 0.0)
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


def _driver_low_confidence(center_x: np.ndarray, center_y: np.ndarray, confidence: np.ndarray) -> np.ndarray:
    conf = np.asarray(confidence, dtype="f8").copy()
    if conf.size < 2:
        return conf
    dx = np.diff(center_x)
    dy = np.diff(center_y)
    step = np.hypot(dx, dy)
    if np.isfinite(step).any():
        large_jump = np.r_[False, step > np.nanpercentile(step[np.isfinite(step)], 90)]
    else:
        large_jump = np.zeros(conf.shape, dtype=bool)
    conf[large_jump & (conf < 0.5)] *= 0.25
    return np.clip(conf, 0.0, 1.0)


def diagnose_model_a_pv(
    state: ForecastState,
    sigma_pred: np.ndarray,
    adt_pred: np.ndarray,
    shape: str,
    polarity: str,
    phase: str,
    phase_index: int,
    model: str,
):
    sigma_total = state.sigma_clim[:, None, None] + sigma_pred
    surfaces = build_isopycnal_surfaces(sigma_total, state.depth, rho_levels=None, smooth_sigma_grid=1.0)
    u_pred, v_pred = thermal_wind_velocity(sigma_pred, adt_pred, state.depth, state.x, state.y, state.radius_m, state.f0)
    pv = compute_isopycnal_control_volume_pv(
        u_pred,
        v_pred,
        sigma_total,
        surfaces.rho_levels,
        state.depth,
        state.x,
        state.y,
        state.radius_m,
        state.f0,
        {"shape": shape, "polarity": polarity, "phase": phase, "phase_index": phase_index, "model": model},
    )
    return u_pred, v_pred, surfaces.z_anom_m, pv.table


def _shift_with_isopycnal_centers(state: ForecastState, tau: float, strength: float) -> tuple[np.ndarray, np.ndarray]:
    dx = float(np.nanmedian(np.diff(state.x))) if len(state.x) > 1 else 1.0
    dy = float(np.nanmedian(np.diff(state.y))) if len(state.y) > 1 else 1.0
    sigma_out = np.full_like(state.sigma_birth, np.nan, dtype="f8")
    for k in range(state.sigma_birth.shape[0]):
        sx = float(tau * strength * state.center_x_R[k] / max(dx, 1e-9))
        sy = float(tau * strength * state.center_y_R[k] / max(dy, 1e-9))
        layer = finite_or_nan(state.sigma_birth[k])
        mask = np.isfinite(layer)
        shifted = shift(_nearest_fill_2d(layer), shift=(sy, sx), order=1, mode="nearest", prefilter=False)
        sigma_out[k] = np.where(mask, shifted, np.nan)
    sx0 = float(tau * strength * state.center_x_R[0] / max(dx, 1e-9))
    sy0 = float(tau * strength * state.center_y_R[0] / max(dy, 1e-9))
    adt = finite_or_nan(state.adt_birth)
    mask = np.isfinite(adt)
    shifted_adt = shift(_nearest_fill_2d(adt), shift=(sy0, sx0), order=1, mode="nearest", prefilter=False)
    return sigma_out, np.where(mask, shifted_adt, np.nan)


def _nearest_fill_2d(layer: np.ndarray) -> np.ndarray:
    arr = np.asarray(layer, dtype="f8")
    valid = np.isfinite(arr)
    if valid.all():
        return arr.copy()
    if not valid.any():
        return np.zeros_like(arr, dtype="f8")
    _, indices = distance_transform_edt(~valid, return_indices=True)
    return arr[tuple(indices)]
