from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .unified_math import G, finite_or_nan, gradient, vertical_weights


@dataclass
class MorelFluxPV:
    q_star: np.ndarray
    pv_flux_x: np.ndarray
    pv_flux_y: np.ndarray
    pv_flux_z: np.ndarray
    pva_volume_integral: np.ndarray
    pva_boundary_integral: np.ndarray
    pv_balance_residual: np.ndarray
    pv_centroid_x_R: np.ndarray
    pv_centroid_y_R: np.ndarray
    td_pv_star: np.ndarray
    td_pv_adjacent_star: np.ndarray
    diagnostics: dict[str, float]


def _cell_to_face(arr: np.ndarray, axis: int) -> np.ndarray:
    arr = np.asarray(arr, dtype="f8")
    shape = list(arr.shape)
    shape[axis] += 1
    face = np.zeros(shape, dtype="f8")
    face = np.moveaxis(face, axis, -1)
    src = np.moveaxis(arr, axis, -1)
    face[..., 1:-1] = 0.5 * (src[..., :-1] + src[..., 1:])
    face[..., 0] = src[..., 0]
    face[..., -1] = src[..., -1]
    return np.moveaxis(face, -1, axis)


def _face_divergence(
    flux_x: np.ndarray,
    flux_y: np.ndarray,
    flux_z: np.ndarray,
    x_m: np.ndarray,
    y_m: np.ndarray,
    depth_m: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dx = float(np.nanmedian(np.diff(x_m))) if x_m.size > 1 else 1.0
    dy = float(np.nanmedian(np.diff(y_m))) if y_m.size > 1 else 1.0
    fx_face = _cell_to_face(flux_x, axis=2)
    fy_face = _cell_to_face(flux_y, axis=1)
    fz_face = _cell_to_face(flux_z, axis=0)
    z = np.asarray(depth_m, dtype="f8")
    if z.size > 1:
        z_edges = np.empty(z.size + 1, dtype="f8")
        z_edges[1:-1] = 0.5 * (z[:-1] + z[1:])
        z_edges[0] = z[0] - 0.5 * (z[1] - z[0])
        z_edges[-1] = z[-1] + 0.5 * (z[-1] - z[-2])
        dz = np.diff(z_edges)
    else:
        dz = np.ones(1, dtype="f8")
    dz_safe = np.maximum(dz, 1e-9)
    div = (
        (fx_face[:, :, 1:] - fx_face[:, :, :-1]) / max(dx, 1e-12)
        + (fy_face[:, 1:, :] - fy_face[:, :-1, :]) / max(dy, 1e-12)
        + (fz_face[1:, :, :] - fz_face[:-1, :, :]) / dz_safe[:, None, None]
    )
    return div, fx_face, fy_face, fz_face


def _boundary_integrals(
    div: np.ndarray,
    fx_face: np.ndarray,
    fy_face: np.ndarray,
    fz_face: np.ndarray,
    x_m: np.ndarray,
    y_m: np.ndarray,
    depth_m: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dx = float(np.nanmedian(np.diff(x_m))) if x_m.size > 1 else 1.0
    dy = float(np.nanmedian(np.diff(y_m))) if y_m.size > 1 else 1.0
    weights_z = vertical_weights(np.asarray(depth_m, dtype="f8"))
    volume = np.nansum(div, axis=(1, 2)) * dx * dy * weights_z
    side_x = (np.nansum(fx_face[:, :, -1], axis=1) - np.nansum(fx_face[:, :, 0], axis=1)) * dy * weights_z
    side_y = (np.nansum(fy_face[:, -1, :], axis=1) - np.nansum(fy_face[:, 0, :], axis=1)) * dx * weights_z
    side_z = np.nansum(fz_face[1:] - fz_face[:-1], axis=(1, 2)) * dx * dy
    boundary = side_x + side_y + side_z
    residual = np.divide(np.abs(volume - boundary), np.abs(volume) + np.abs(boundary) + 1e-18)
    return volume, boundary, residual


def _pv_centroids(q_star: np.ndarray, x_R: np.ndarray, y_R: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    q = np.asarray(q_star, dtype="f8")
    xx = np.asarray(x_R, dtype="f8")[None, :]
    yy = np.asarray(y_R, dtype="f8")[:, None]
    weights = np.abs(q)
    weights = np.where(np.isfinite(weights), weights, 0.0)
    denom = np.nansum(weights, axis=(1, 2))
    cx = np.divide(np.nansum(weights * xx[None, :, :], axis=(1, 2)), denom, out=np.full(q.shape[0], np.nan), where=denom > 0)
    cy = np.divide(np.nansum(weights * yy[None, :, :], axis=(1, 2)), denom, out=np.full(q.shape[0], np.nan), where=denom > 0)
    valid = np.isfinite(cx) & np.isfinite(cy)
    if valid.any():
        ref = int(np.flatnonzero(valid)[0])
        td = np.sqrt((cx - cx[ref]) ** 2 + (cy - cy[ref]) ** 2)
    else:
        td = np.full(q.shape[0], np.nan)
    td_adj = np.full(q.shape[0], np.nan)
    prev = None
    for k in range(q.shape[0]):
        if not valid[k]:
            continue
        if prev is not None:
            td_adj[k] = float(np.sqrt((cx[k] - cx[prev]) ** 2 + (cy[k] - cy[prev]) ** 2))
        prev = k
    return cx, cy, td, td_adj


def project_velocity_to_graph_tangent(
    u_m_s: np.ndarray,
    v_m_s: np.ndarray,
    eta_star: np.ndarray,
    x_R: np.ndarray,
    y_R: np.ndarray,
    radius_m: float,
    depth_m: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    h_scale = max(float(np.nanmax(depth_m) - np.nanmin(depth_m)), 1.0)
    eta_x = gradient(np.asarray(eta_star, dtype="f8"), np.asarray(x_R, dtype="f8"), axis=2) * h_scale / max(radius_m, 1.0)
    eta_y = gradient(np.asarray(eta_star, dtype="f8"), np.asarray(y_R, dtype="f8"), axis=1) * h_scale / max(radius_m, 1.0)
    slope2 = np.where(np.isfinite(eta_x * eta_x + eta_y * eta_y), eta_x * eta_x + eta_y * eta_y, 0.0)
    inv_norm = 1.0 / np.sqrt(1.0 + slope2)
    nx = -eta_x * inv_norm
    ny = -eta_y * inv_norm
    ndotu = nx * u_m_s + ny * v_m_s
    u_t = u_m_s - ndotu * nx
    v_t = v_m_s - ndotu * ny
    diagnostics = {
        "surface_tangent_slope_p95": float(np.nanpercentile(np.sqrt(slope2), 95)),
        "surface_tangent_speed_change_p95_m_s": float(np.nanpercentile(np.sqrt((u_t - u_m_s) ** 2 + (v_t - v_m_s) ** 2), 95)),
    }
    return u_t, v_t, diagnostics


def compute_morel_flux_pv(
    u_m_s: np.ndarray,
    v_m_s: np.ndarray,
    sigma0_anom: np.ndarray,
    dsigma0_clim_dz: np.ndarray,
    depth_m: np.ndarray,
    x_R: np.ndarray,
    y_R: np.ndarray,
    radius_m: float,
    f0: float,
) -> MorelFluxPV:
    u_raw = finite_or_nan(u_m_s)
    v_raw = finite_or_nan(v_m_s)
    sigma_raw = finite_or_nan(sigma0_anom)
    u = np.where(np.isfinite(u_raw), u_raw, 0.0)
    v = np.where(np.isfinite(v_raw), v_raw, 0.0)
    sigma_anom = np.where(np.isfinite(sigma_raw), sigma_raw, 0.0)
    depth = np.asarray(depth_m, dtype="f8")
    x_m = np.asarray(x_R, dtype="f8") * max(radius_m, 1.0)
    y_m = np.asarray(y_R, dtype="f8") * max(radius_m, 1.0)
    dsigma_clim = np.asarray(dsigma0_clim_dz, dtype="f8")
    sigma_z_anom = gradient(sigma_anom, depth, axis=0)
    sigma_total_z_raw = sigma_z_anom + dsigma_clim[:, None, None]
    sigma_total_z = np.where(np.isfinite(sigma_total_z_raw), sigma_total_z_raw, dsigma_clim[:, None, None])
    sigma_x_raw = gradient(sigma_anom, x_m, axis=2)
    sigma_y_raw = gradient(sigma_anom, y_m, axis=1)
    sigma_x = np.where(np.isfinite(sigma_x_raw), sigma_x_raw, 0.0)
    sigma_y = np.where(np.isfinite(sigma_y_raw), sigma_y_raw, 0.0)
    flux_x = v * sigma_total_z
    flux_y = -u * sigma_total_z
    flux_z = u * sigma_y - v * sigma_x
    div_flux, fx_face, fy_face, fz_face = _face_divergence(flux_x, flux_y, flux_z, x_m, y_m, depth)
    planetary_anom = f0 * sigma_z_anom
    pva = div_flux + planetary_anom
    pv_scale = max(abs(float(f0)) * float(np.nanmedian(np.abs(dsigma_clim[np.isfinite(dsigma_clim)]))), 1e-14)
    q_star = np.where(np.isfinite(pva / pv_scale), pva / pv_scale, 0.0)
    volume, boundary, residual = _boundary_integrals(div_flux, fx_face, fy_face, fz_face, x_m, y_m, depth)
    cx, cy, td, td_adj = _pv_centroids(q_star, np.asarray(x_R, dtype="f8"), np.asarray(y_R, dtype="f8"))
    diagnostics = {
        "pv_scale": float(pv_scale),
        "pv_flux_q_p95": float(np.nanpercentile(np.abs(q_star), 95)),
        "pv_balance_residual_p50": float(np.nanpercentile(residual, 50)),
        "pv_balance_residual_p95": float(np.nanpercentile(residual, 95)),
        "TD_PV_star_p95": float(np.nanpercentile(td[np.isfinite(td)], 95)) if np.isfinite(td).any() else np.nan,
        "TD_PV_adjacent_star_p95": float(np.nanpercentile(td_adj[np.isfinite(td_adj)], 95)) if np.isfinite(td_adj).any() else np.nan,
    }
    return MorelFluxPV(
        q_star=q_star,
        pv_flux_x=np.where(np.isfinite(flux_x / pv_scale), flux_x / pv_scale, 0.0),
        pv_flux_y=np.where(np.isfinite(flux_y / pv_scale), flux_y / pv_scale, 0.0),
        pv_flux_z=np.where(np.isfinite(flux_z / pv_scale), flux_z / pv_scale, 0.0),
        pva_volume_integral=volume,
        pva_boundary_integral=boundary,
        pv_balance_residual=residual,
        pv_centroid_x_R=cx,
        pv_centroid_y_R=cy,
        td_pv_star=td,
        td_pv_adjacent_star=td_adj,
        diagnostics=diagnostics,
    )
