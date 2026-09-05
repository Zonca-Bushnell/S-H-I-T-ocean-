from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    from scipy.fft import dstn, idstn
except Exception:  # pragma: no cover
    dstn = None
    idstn = None


EARTH_RADIUS_M = 6_371_000.0
G = 9.81
RHO0 = 1025.0


def sanitize_ocean_field(values: np.ndarray) -> np.ndarray:
    cleaned = np.asarray(values, dtype="f8").copy()
    cleaned[np.abs(cleaned) > 1.0e20] = np.nan
    return cleaned


def xy_to_lonlat(
    x: np.ndarray,
    y: np.ndarray,
    lon0: float,
    lat0: float,
) -> tuple[np.ndarray, np.ndarray]:
    lat = lat0 + np.degrees(y / EARTH_RADIUS_M)
    lon = lon0 + np.degrees(x / (EARTH_RADIUS_M * np.cos(np.radians(lat0))))
    return lon, lat


def bilinear_sample(
    lon: np.ndarray,
    lat: np.ndarray,
    field: np.ndarray,
    target_lon: np.ndarray,
    target_lat: np.ndarray,
) -> np.ndarray:
    lon_step = float(np.nanmedian(np.diff(lon)))
    lat_step = float(np.nanmedian(np.diff(lat)))
    x = (target_lon - float(lon[0])) / lon_step
    y = (target_lat - float(lat[0])) / lat_step
    i0 = np.floor(y).astype("i8")
    j0 = np.floor(x).astype("i8")
    wy = y - i0
    wx = x - j0
    good = (i0 >= 0) & (j0 >= 0) & (i0 < len(lat) - 1) & (j0 < len(lon) - 1)
    out = np.full(target_lon.shape, np.nan, dtype="f8")
    if not np.any(good):
        return out
    i = i0[good]
    j = j0[good]
    out[good] = (
        field[i, j] * (1 - wy[good]) * (1 - wx[good])
        + field[i + 1, j] * wy[good] * (1 - wx[good])
        + field[i, j + 1] * (1 - wy[good]) * wx[good]
        + field[i + 1, j + 1] * wy[good] * wx[good]
    )
    return out


def load_n2(path: Path, depth: np.ndarray) -> np.ndarray:
    data = np.load(path)
    n2 = G / RHO0 * np.asarray(data["dsigma0_dz"], dtype="f8")
    source_depth = np.asarray(data["depth"], dtype="f8")
    values = np.interp(depth, source_depth, n2)
    return np.clip(values, 1.0e-7, np.inf)


def grid_spacing_m(lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, float, float]:
    dlon = float(np.nanmedian(np.diff(lon)))
    dlat = float(np.nanmedian(np.diff(lat)))
    dx_by_lat = EARTH_RADIUS_M * np.cos(np.radians(lat)) * np.radians(dlon)
    dy = EARTH_RADIUS_M * np.radians(dlat)
    dx = float(np.nanmedian(np.abs(dx_by_lat[np.isfinite(dx_by_lat)])))
    return dx_by_lat.astype("f8"), abs(float(dy)), dx


def relative_vorticity(lon: np.ndarray, lat: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    dx_by_lat, dy, _ = grid_spacing_m(lon, lat)
    dvdx = np.gradient(v, axis=2) / dx_by_lat[None, :, None]
    dudy = np.gradient(u, axis=1) / dy
    return dvdx - dudy


def streamfunction_from_zeta(zeta: np.ndarray, dx: float, dy: float) -> np.ndarray:
    rhs = np.nan_to_num(zeta, nan=0.0, posinf=0.0, neginf=0.0)
    nz, ny, nx = rhs.shape
    psi = np.zeros_like(rhs, dtype="f8")
    if dstn is None or idstn is None or ny < 3 or nx < 3:
        return psi
    rhs_i = rhs[:, 1:-1, 1:-1]
    rhs_hat = dstn(rhs_i, type=1, axes=(1, 2), norm="ortho")
    nx_i = rhs_i.shape[2]
    ny_i = rhs_i.shape[1]
    mx = np.arange(1, nx_i + 1, dtype="f8")
    my = np.arange(1, ny_i + 1, dtype="f8")
    lam_x = -4.0 * np.sin(np.pi * mx / (2.0 * (nx_i + 1))) ** 2 / max(dx * dx, 1.0e-12)
    lam_y = -4.0 * np.sin(np.pi * my / (2.0 * (ny_i + 1))) ** 2 / max(dy * dy, 1.0e-12)
    denom = lam_y[None, :, None] + lam_x[None, None, :]
    psi_hat = np.divide(rhs_hat, denom, out=np.zeros_like(rhs_hat), where=np.abs(denom) > 1.0e-12)
    psi[:, 1:-1, 1:-1] = idstn(psi_hat, type=1, axes=(1, 2), norm="ortho")
    return psi


def ddz(values: np.ndarray, depth: np.ndarray) -> np.ndarray:
    return np.gradient(values, depth, axis=0, edge_order=1)


def radial_derivative(values: np.ndarray, r_m: np.ndarray) -> np.ndarray:
    return np.gradient(values, r_m, axis=1, edge_order=1)


def azimuth_second_derivative(values: np.ndarray, theta: np.ndarray) -> np.ndarray:
    dtheta = float(theta[1] - theta[0]) if len(theta) > 1 else 2.0 * np.pi
    return (np.roll(values, -1, axis=2) - 2.0 * values + np.roll(values, 1, axis=2)) / (dtheta * dtheta)
