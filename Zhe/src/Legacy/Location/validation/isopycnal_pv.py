from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .unified_math import finite_or_nan, gradient, vertical_weights


@dataclass
class IsopycnalPVResult:
    q_star: np.ndarray
    pva_volume_integral: np.ndarray
    pva_boundary_integral: np.ndarray
    pv_balance_residual: np.ndarray
    pv_centroid_x_R: np.ndarray
    pv_centroid_y_R: np.ndarray
    td_pv_star: np.ndarray
    td_pv_adjacent_star: np.ndarray
    table: pd.DataFrame


def compute_isopycnal_control_volume_pv(
    u_m_s: np.ndarray,
    v_m_s: np.ndarray,
    sigma_total: np.ndarray,
    rho_levels: np.ndarray,
    depth_m: np.ndarray,
    x_R: np.ndarray,
    y_R: np.ndarray,
    radius_m: float,
    f0: float,
    label: dict,
) -> IsopycnalPVResult:
    depth = np.asarray(depth_m, dtype="f8")
    x_m = np.asarray(x_R, dtype="f8") * max(float(radius_m), 1.0)
    y_m = np.asarray(y_R, dtype="f8") * max(float(radius_m), 1.0)
    sigma = finite_or_nan(sigma_total)
    u = np.where(np.isfinite(u_m_s), u_m_s, 0.0)
    v = np.where(np.isfinite(v_m_s), v_m_s, 0.0)
    sx = np.where(np.isfinite(gradient(sigma, x_m, axis=2)), gradient(sigma, x_m, axis=2), 0.0)
    sy = np.where(np.isfinite(gradient(sigma, y_m, axis=1)), gradient(sigma, y_m, axis=1), 0.0)
    sz = np.where(np.isfinite(gradient(sigma, depth, axis=0)), gradient(sigma, depth, axis=0), 0.0)
    flux_x = v * sz
    flux_y = -u * sz
    flux_z = u * sy - v * sx
    div = _divergence_centered(flux_x, flux_y, flux_z, x_m, y_m, depth)
    pv_scale = max(abs(float(f0)) * float(np.nanmedian(np.abs(sz[np.isfinite(sz)]))), 1e-14)
    q_star = np.where(np.isfinite(div / pv_scale), div / pv_scale, 0.0)

    rows: list[dict] = []
    nbin = max(len(rho_levels) - 1, 0)
    volume_int = np.full(nbin, np.nan, dtype="f8")
    boundary_int = np.full(nbin, np.nan, dtype="f8")
    residual = np.full(nbin, np.nan, dtype="f8")
    cx = np.full(nbin, np.nan, dtype="f8")
    cy = np.full(nbin, np.nan, dtype="f8")
    td = np.full(nbin, np.nan, dtype="f8")
    td_adj = np.full(nbin, np.nan, dtype="f8")
    dx = float(np.nanmedian(np.diff(x_m))) if len(x_m) > 1 else 1.0
    dy = float(np.nanmedian(np.diff(y_m))) if len(y_m) > 1 else 1.0
    wz = vertical_weights(depth)
    cell_vol = wz[:, None, None] * dx * dy
    xx = np.asarray(x_R, dtype="f8")[None, None, :]
    yy = np.asarray(y_R, dtype="f8")[None, :, None]
    prev = None
    for b in range(nbin):
        lo, hi = float(rho_levels[b]), float(rho_levels[b + 1])
        mask = np.isfinite(sigma) & (sigma >= lo) & (sigma < hi)
        if not np.any(mask):
            continue
        weights = np.abs(q_star) * cell_vol * mask
        mass = float(np.nansum(weights))
        if mass > 0:
            cx[b] = float(np.nansum(weights * xx) / mass)
            cy[b] = float(np.nansum(weights * yy) / mass)
        volume_int[b] = float(np.nansum(div * cell_vol * mask))
        boundary_int[b] = _mask_boundary_flux(mask, flux_x, flux_y, flux_z, dx, dy, wz)
        residual[b] = abs(volume_int[b] - boundary_int[b]) / (abs(volume_int[b]) + abs(boundary_int[b]) + 1e-18)
        if np.isfinite(cx[b]) and np.isfinite(cy[b]):
            if prev is None:
                ref = b
            td[b] = float(np.hypot(cx[b] - cx[ref], cy[b] - cy[ref]))
            if prev is not None:
                td_adj[b] = float(np.hypot(cx[b] - cx[prev], cy[b] - cy[prev]))
            prev = b
        rows.append(
            {
                **label,
                "rho_bin": b,
                "rho_lower": lo,
                "rho_upper": hi,
                "pva_volume_integral": volume_int[b],
                "pva_boundary_integral": boundary_int[b],
                "pv_balance_residual": residual[b],
                "pv_centroid_x_R": cx[b],
                "pv_centroid_y_R": cy[b],
                "TD_PV_star": td[b],
                "TD_PV_adjacent_star": td_adj[b],
                "n_cells": int(np.count_nonzero(mask)),
            }
        )
    return IsopycnalPVResult(q_star, volume_int, boundary_int, residual, cx, cy, td, td_adj, pd.DataFrame(rows))


def _divergence_centered(fx: np.ndarray, fy: np.ndarray, fz: np.ndarray, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    return gradient(fx, x, axis=2) + gradient(fy, y, axis=1) + gradient(fz, z, axis=0)


def _mask_boundary_flux(mask: np.ndarray, fx: np.ndarray, fy: np.ndarray, fz: np.ndarray, dx: float, dy: float, wz: np.ndarray) -> float:
    total = 0.0
    area_x = dy * wz[:, None, None]
    area_y = dx * wz[:, None, None]
    # x faces
    change = mask[:, :, :-1] != mask[:, :, 1:]
    face_flux = 0.5 * (fx[:, :, :-1] + fx[:, :, 1:])
    sign = np.where(mask[:, :, :-1] & ~mask[:, :, 1:], 1.0, -1.0)
    total += float(np.nansum(face_flux * sign * area_x * change))
    # y faces
    change = mask[:, :-1, :] != mask[:, 1:, :]
    face_flux = 0.5 * (fy[:, :-1, :] + fy[:, 1:, :])
    sign = np.where(mask[:, :-1, :] & ~mask[:, 1:, :], 1.0, -1.0)
    total += float(np.nansum(face_flux * sign * area_y * change))
    # z faces
    change = mask[:-1, :, :] != mask[1:, :, :]
    face_flux = 0.5 * (fz[:-1, :, :] + fz[1:, :, :])
    sign = np.where(mask[:-1, :, :] & ~mask[1:, :, :], 1.0, -1.0)
    total += float(np.nansum(face_flux[change] * sign[change] * dx * dy))
    return total
