from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter

from .unified_math import finite_or_nan, gradient


@dataclass
class IsopycnalSurfaces:
    rho_levels: np.ndarray
    z_m: np.ndarray
    z_anom_m: np.ndarray
    valid_fraction: np.ndarray


def reference_density_levels(sigma_total: np.ndarray, depth_m: np.ndarray) -> np.ndarray:
    """Choose one density value per source depth from the horizontal median profile."""
    profile = np.nanmedian(np.asarray(sigma_total, dtype="f8"), axis=(1, 2))
    if not np.all(np.isfinite(profile)):
        profile = np.where(np.isfinite(profile), profile, np.interp(depth_m, depth_m[np.isfinite(profile)], profile[np.isfinite(profile)]))
    return _strictly_increasing(profile)


def build_isopycnal_surfaces(
    sigma_total: np.ndarray,
    depth_m: np.ndarray,
    rho_levels: np.ndarray | None = None,
    smooth_sigma_grid: float = 1.0,
) -> IsopycnalSurfaces:
    sigma = finite_or_nan(sigma_total)
    depth = np.asarray(depth_m, dtype="f8")
    if rho_levels is None:
        rho_levels = reference_density_levels(sigma, depth)
    rho = np.asarray(rho_levels, dtype="f8")
    nz, ny, nx = sigma.shape
    z_iso = np.full((rho.size, ny, nx), np.nan, dtype="f8")
    for j in range(ny):
        for i in range(nx):
            col = sigma[:, j, i]
            good = np.isfinite(col) & np.isfinite(depth)
            if np.count_nonzero(good) < 2:
                continue
            sig_col = _strictly_increasing(col[good])
            dep_col = depth[good]
            inside = (rho >= sig_col[0]) & (rho <= sig_col[-1])
            if np.any(inside):
                z_iso[inside, j, i] = np.interp(rho[inside], sig_col, dep_col)
    if smooth_sigma_grid > 0:
        z_iso = _smooth_preserving_nan(z_iso, smooth_sigma_grid)
    z_ref = np.interp(rho, _strictly_increasing(np.nanmedian(sigma, axis=(1, 2))), depth)
    z_anom = z_iso - z_ref[:, None, None]
    valid_fraction = np.nanmean(np.isfinite(z_iso), axis=(1, 2))
    return IsopycnalSurfaces(rho_levels=rho, z_m=z_iso, z_anom_m=z_anom, valid_fraction=valid_fraction)


def interpolate_to_surfaces(field: np.ndarray, depth_m: np.ndarray, z_surface_m: np.ndarray) -> np.ndarray:
    arr = finite_or_nan(field)
    depth = np.asarray(depth_m, dtype="f8")
    out = np.full_like(z_surface_m, np.nan, dtype="f8")
    for j in range(arr.shape[1]):
        for i in range(arr.shape[2]):
            col = arr[:, j, i]
            good = np.isfinite(col) & np.isfinite(depth)
            if np.count_nonzero(good) < 2:
                continue
            zq = z_surface_m[:, j, i]
            inside = np.isfinite(zq) & (zq >= depth[good][0]) & (zq <= depth[good][-1])
            if np.any(inside):
                out[inside, j, i] = np.interp(zq[inside], depth[good], col[good])
    return out


def surface_geometry(z_surface_m: np.ndarray, x_R: np.ndarray, y_R: np.ndarray, radius_m: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_m = np.asarray(x_R, dtype="f8") * max(float(radius_m), 1.0)
    y_m = np.asarray(y_R, dtype="f8") * max(float(radius_m), 1.0)
    z = _smooth_preserving_nan(z_surface_m, 1.0)
    zx = gradient(z, x_m, axis=2)
    zy = gradient(z, y_m, axis=1)
    norm = np.sqrt(1.0 + zx * zx + zy * zy)
    nx = -zx / norm
    ny = -zy / norm
    curvature_h = 0.5 * (gradient(nx, x_m, axis=2) + gradient(ny, y_m, axis=1))
    slope = np.sqrt(zx * zx + zy * zy)
    asym = np.nanmean(np.where(np.asarray(x_R)[None, None, :] > 0, curvature_h, np.nan), axis=(1, 2)) - np.nanmean(
        np.where(np.asarray(x_R)[None, None, :] < 0, curvature_h, np.nan), axis=(1, 2)
    )
    return slope, curvature_h, asym


def project_surface_field_to_depth(surface_field: np.ndarray, z_surface_m: np.ndarray, depth_m: np.ndarray) -> np.ndarray:
    """Project values carried by rho-surfaces back to fixed depth for visual comparison."""
    rho_n, ny, nx = surface_field.shape
    depth = np.asarray(depth_m, dtype="f8")
    out = np.full((depth.size, ny, nx), np.nan, dtype="f8")
    for j in range(ny):
        for i in range(nx):
            zc = z_surface_m[:, j, i]
            fc = surface_field[:, j, i]
            good = np.isfinite(zc) & np.isfinite(fc)
            if np.count_nonzero(good) < 2:
                continue
            order = np.argsort(zc[good])
            zg = zc[good][order]
            fg = fc[good][order]
            inside = (depth >= zg[0]) & (depth <= zg[-1])
            out[inside, j, i] = np.interp(depth[inside], zg, fg)
    return out


def _strictly_increasing(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype="f8").copy()
    if arr.size == 0:
        return arr
    eps = max(float(np.nanstd(arr)), 1.0) * 1e-9
    for k in range(1, arr.size):
        if not np.isfinite(arr[k]):
            arr[k] = arr[k - 1] + eps
        if arr[k] <= arr[k - 1]:
            arr[k] = arr[k - 1] + eps
    return arr


def _smooth_preserving_nan(arr: np.ndarray, sigma: float) -> np.ndarray:
    vals = np.asarray(arr, dtype="f8")
    if sigma <= 0:
        return vals
    finite = np.isfinite(vals)
    filled = np.where(finite, vals, 0.0)
    weights = finite.astype("f8")
    smooth = gaussian_filter(filled, sigma=(0, sigma, sigma), mode="nearest")
    w_smooth = gaussian_filter(weights, sigma=(0, sigma, sigma), mode="nearest")
    return np.divide(smooth, w_smooth, out=np.full_like(vals, np.nan), where=w_smooth > 1e-6)
