from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter1d, shift

from .common import finite_or_nan, geo_params, thermal_wind_velocity

from .models import ForecastState, filled_centers


def build_baseline_state(
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
    u0, v0 = thermal_wind_velocity(sigma_birth, adt_birth, depth, x, y, radius_m, gp.f0)
    cx, cy = velocity_centroid_profile(u0, v0, x, y)
    source = np.full(cx.shape, "li_depth_velocity_centroid", dtype="U48")
    confidence = np.ones(cx.shape, dtype="f8")
    return ForecastState(
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


def predict_baseline_state(state: ForecastState, tau: float, strength: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Fixed-depth Li-style phase step.

    This deliberately stays on the original depth grid. The vertical tilt is a
    smooth depth-layer displacement inferred from the birth fixed-depth velocity
    structure; no isopycnal surfaces are constructed here.
    """
    smooth_x = _smooth_depth_centers(state.center_x_R)
    smooth_y = _smooth_depth_centers(state.center_y_R)
    return _shift_fixed_depth_layers(state, smooth_x, smooth_y, tau, strength)


def baseline_velocity(state: ForecastState, sigma: np.ndarray, adt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return thermal_wind_velocity(sigma, adt, state.depth, state.x, state.y, state.radius_m, state.f0)


def velocity_centroid_profile(u: np.ndarray, v: np.ndarray, x: np.ndarray, y: np.ndarray, search_radius_R: float = 1.5) -> tuple[np.ndarray, np.ndarray]:
    xx, yy = np.meshgrid(np.asarray(x, dtype="f8"), np.asarray(y, dtype="f8"))
    central = (np.abs(xx) <= search_radius_R) & (np.abs(yy) <= search_radius_R)
    cx = np.full(u.shape[0], np.nan, dtype="f8")
    cy = np.full(u.shape[0], np.nan, dtype="f8")
    for k in range(u.shape[0]):
        speed = np.hypot(u[k], v[k])
        valid = central & np.isfinite(speed)
        if not np.any(valid):
            continue
        vals = speed[valid]
        cutoff = np.nanpercentile(vals, 25)
        weights = np.where(valid, np.maximum(cutoff - speed, 0.0), 0.0)
        if np.nansum(weights) <= 0:
            iy, ix = np.unravel_index(np.nanargmin(np.where(valid, speed, np.nan)), speed.shape)
            cx[k] = xx[iy, ix]
            cy[k] = yy[iy, ix]
        else:
            cx[k] = float(np.nansum(weights * xx) / np.nansum(weights))
            cy[k] = float(np.nansum(weights * yy) / np.nansum(weights))
    return filled_centers(cx), filled_centers(cy)


def recenter_3d_by_velocity(field: np.ndarray, u: np.ndarray, v: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    cx, cy = velocity_centroid_profile(u, v, x, y)
    dx = float(np.nanmedian(np.diff(x))) if len(x) > 1 else 1.0
    dy = float(np.nanmedian(np.diff(y))) if len(y) > 1 else 1.0
    out = np.full_like(field, np.nan, dtype="f8")
    for k in range(field.shape[0]):
        layer = finite_or_nan(field[k])
        mask = np.isfinite(layer)
        sx = -float(cx[k]) / max(dx, 1e-9)
        sy = -float(cy[k]) / max(dy, 1e-9)
        shifted = shift(_nearest_fill_2d(layer), shift=(sy, sx), order=1, mode="nearest", prefilter=False)
        out[k] = np.where(mask, shifted, np.nan)
    return out


def recenter_2d_by_surface_velocity(field: np.ndarray, u: np.ndarray, v: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    cx, cy = velocity_centroid_profile(u[:1], v[:1], x, y)
    dx = float(np.nanmedian(np.diff(x))) if len(x) > 1 else 1.0
    dy = float(np.nanmedian(np.diff(y))) if len(y) > 1 else 1.0
    arr = finite_or_nan(field)
    mask = np.isfinite(arr)
    sx = -float(cx[0]) / max(dx, 1e-9)
    sy = -float(cy[0]) / max(dy, 1e-9)
    shifted = shift(_nearest_fill_2d(arr), shift=(sy, sx), order=1, mode="nearest", prefilter=False)
    return np.where(mask, shifted, np.nan)


def _shift_fixed_depth_layers(state: ForecastState, center_x: np.ndarray, center_y: np.ndarray, tau: float, strength: float) -> tuple[np.ndarray, np.ndarray]:
    dx = float(np.nanmedian(np.diff(state.x))) if len(state.x) > 1 else 1.0
    dy = float(np.nanmedian(np.diff(state.y))) if len(state.y) > 1 else 1.0
    sigma_out = np.full_like(state.sigma_birth, np.nan, dtype="f8")
    for k in range(state.sigma_birth.shape[0]):
        sx = float(tau * strength * center_x[k] / max(dx, 1e-9))
        sy = float(tau * strength * center_y[k] / max(dy, 1e-9))
        layer = finite_or_nan(state.sigma_birth[k])
        mask = np.isfinite(layer)
        shifted = shift(_nearest_fill_2d(layer), shift=(sy, sx), order=1, mode="nearest", prefilter=False)
        sigma_out[k] = np.where(mask, shifted, np.nan)
    sx0 = float(tau * strength * center_x[0] / max(dx, 1e-9))
    sy0 = float(tau * strength * center_y[0] / max(dy, 1e-9))
    adt = finite_or_nan(state.adt_birth)
    mask = np.isfinite(adt)
    shifted_adt = shift(_nearest_fill_2d(adt), shift=(sy0, sx0), order=1, mode="nearest", prefilter=False)
    return sigma_out, np.where(mask, shifted_adt, np.nan)


def _smooth_depth_centers(values: np.ndarray) -> np.ndarray:
    if values.size < 4:
        return values.copy()
    return gaussian_filter1d(np.asarray(values, dtype="f8"), sigma=1.0, mode="nearest")


def _nearest_fill_2d(layer: np.ndarray) -> np.ndarray:
    arr = np.asarray(layer, dtype="f8")
    valid = np.isfinite(arr)
    if valid.all():
        return arr.copy()
    if not valid.any():
        return np.zeros_like(arr, dtype="f8")
    _, indices = distance_transform_edt(~valid, return_indices=True)
    return arr[tuple(indices)]
