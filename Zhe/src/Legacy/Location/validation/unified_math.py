from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.fft import dstn, idstn
from scipy.linalg import eigh
from scipy.linalg import solve_banded
from scipy.ndimage import distance_transform_edt


G = 9.81
OMEGA = 7.2921159e-5
EARTH_RADIUS_M = 6_371_000.0


@dataclass(frozen=True)
class GeoParams:
    f0: float
    beta: float


def geo_params(latitude_ref: float = 30.0) -> GeoParams:
    lat = np.deg2rad(latitude_ref)
    return GeoParams(
        f0=float(2.0 * OMEGA * np.sin(lat)),
        beta=float(2.0 * OMEGA * np.cos(lat) / EARTH_RADIUS_M),
    )


def finite_or_nan(array: np.ndarray) -> np.ndarray:
    arr = np.asarray(array, dtype="f8")
    return np.where(np.isfinite(arr), arr, np.nan)


def gradient(array: np.ndarray, coord: np.ndarray, axis: int) -> np.ndarray:
    arr = np.asarray(array, dtype="f8")
    c = np.asarray(coord, dtype="f8")
    if c.size < 2:
        return np.zeros_like(arr, dtype="f8")
    return np.gradient(arr, c, axis=axis, edge_order=1)


def nanfill_zero(array: np.ndarray) -> np.ndarray:
    return np.where(np.isfinite(array), array, 0.0)


def component_scale(u: np.ndarray, v: np.ndarray) -> float:
    vals = np.concatenate([np.asarray(u, dtype="f8").ravel(), np.asarray(v, dtype="f8").ravel()])
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 1.0
    scale = float(np.nanpercentile(np.abs(vals), 95))
    return scale if np.isfinite(scale) and scale > 0 else 1.0


def velocity_from_psi(psi: np.ndarray, y: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return -gradient(psi, y, axis=1), gradient(psi, x, axis=2)


def streamfunction_from_zeta(zeta: np.ndarray, y: np.ndarray, x: np.ndarray) -> np.ndarray:
    rhs = nanfill_zero(zeta)
    nz, ny, nx = rhs.shape
    if ny < 3 or nx < 3:
        return np.zeros_like(rhs, dtype="f8")
    dx = float(np.nanmedian(np.diff(x)))
    dy = float(np.nanmedian(np.diff(y)))
    rhs_i = rhs[:, 1:-1, 1:-1]
    rhs_hat = dstn(rhs_i, type=1, axes=(1, 2), norm="ortho")
    nx_i = rhs_i.shape[2]
    ny_i = rhs_i.shape[1]
    mx = np.arange(1, nx_i + 1, dtype="f8")
    my = np.arange(1, ny_i + 1, dtype="f8")
    lam_x = -4.0 * np.sin(np.pi * mx / (2.0 * (nx_i + 1))) ** 2 / max(dx * dx, 1e-12)
    lam_y = -4.0 * np.sin(np.pi * my / (2.0 * (ny_i + 1))) ** 2 / max(dy * dy, 1e-12)
    denom = lam_y[None, :, None] + lam_x[None, None, :]
    psi_hat = np.divide(rhs_hat, denom, out=np.zeros_like(rhs_hat), where=np.abs(denom) > 1e-12)
    psi_i = idstn(psi_hat, type=1, axes=(1, 2), norm="ortho")
    psi = np.zeros((nz, ny, nx), dtype="f8")
    psi[:, 1:-1, 1:-1] = psi_i
    return psi


def vertical_coefficients(z_star: np.ndarray, f_profile: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    z = np.asarray(z_star, dtype="f8")
    f = np.asarray(f_profile, dtype="f8")
    nz = z.size
    lower = np.zeros(nz, dtype="f8")
    diag = np.zeros(nz, dtype="f8")
    upper = np.zeros(nz, dtype="f8")
    for k in range(nz):
        if k > 0:
            dzm = max(float(z[k] - z[k - 1]), 1e-9)
            dzc = max(float((z[min(k + 1, nz - 1)] - z[k - 1]) * 0.5) if k < nz - 1 else dzm, 1e-9)
            lower[k] = 0.5 * (f[k] + f[k - 1]) / (dzm * dzc)
        if k < nz - 1:
            dzp = max(float(z[k + 1] - z[k]), 1e-9)
            dzc = max(float((z[k + 1] - z[max(k - 1, 0)]) * 0.5) if k > 0 else dzp, 1e-9)
            upper[k] = 0.5 * (f[k] + f[k + 1]) / (dzp * dzc)
        diag[k] = -(lower[k] + upper[k])
    return lower, diag, upper


def vertical_weights(z_star: np.ndarray) -> np.ndarray:
    z = np.asarray(z_star, dtype="f8")
    if z.size == 1:
        return np.ones(1, dtype="f8")
    edges = np.empty(z.size + 1, dtype="f8")
    edges[1:-1] = 0.5 * (z[:-1] + z[1:])
    edges[0] = z[0] - 0.5 * (z[1] - z[0])
    edges[-1] = z[-1] + 0.5 * (z[-1] - z[-2])
    weights = np.diff(edges)
    good = np.isfinite(weights) & (weights > 0)
    fill = float(np.nanmedian(weights[good])) if good.any() else 1.0
    return np.where(good, weights, fill)


def vertical_mode_decomposition(f_profile: np.ndarray, z_star: np.ndarray, mode_count: int = 3, bc: str = "qgpv-neumann") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if bc != "qgpv-neumann":
        raise ValueError(f"Unsupported vertical mode boundary condition: {bc!r}")
    z = np.asarray(z_star, dtype="f8")
    f = np.asarray(f_profile, dtype="f8")
    nz = z.size
    operator = np.zeros((nz, nz), dtype="f8")
    for k in range(nz - 1):
        dz = max(float(z[k + 1] - z[k]), 1e-9)
        c = 0.5 * (f[k] + f[k + 1]) / dz
        operator[k, k] += c
        operator[k + 1, k + 1] += c
        operator[k, k + 1] -= c
        operator[k + 1, k] -= c
    weights = vertical_weights(z_star)
    mass_sqrt = np.sqrt(np.maximum(weights, 1e-12))
    sym = (operator / mass_sqrt[:, None]) / mass_sqrt[None, :]
    sym = 0.5 * (sym + sym.T)
    values, vectors = eigh(sym, check_finite=False)
    order = np.argsort(values)
    values = np.maximum(values[order], 0.0)
    vectors = vectors[:, order]
    max_modes = max(1, min(int(mode_count) + 1, nz))
    profiles = vectors[:, :max_modes] / mass_sqrt[:, None]
    for n in range(profiles.shape[1]):
        norm = np.sqrt(np.nansum(weights * profiles[:, n] * profiles[:, n]))
        if np.isfinite(norm) and norm > 0:
            profiles[:, n] /= norm
        if np.nanmean(profiles[:, n]) < 0:
            profiles[:, n] *= -1.0
    radius_like = np.divide(1.0, np.sqrt(values[:max_modes]), out=np.full(max_modes, np.inf), where=values[:max_modes] > 1e-12)
    return values[:max_modes], profiles, radius_like


def project_vertical_modes(field: np.ndarray, profiles: np.ndarray, z_star: np.ndarray) -> np.ndarray:
    arr = np.asarray(field, dtype="f8")
    weights = vertical_weights(z_star)
    coeff = np.einsum("k,kxy,kn->nxy", weights, arr, profiles, optimize=True)
    return coeff[:, None, :, :] * profiles.T[:, :, None, None]


def qgpv_operator(psi: np.ndarray, f_profile: np.ndarray, z_star: np.ndarray, y: np.ndarray, x: np.ndarray) -> np.ndarray:
    dx = float(np.nanmedian(np.diff(x)))
    dy = float(np.nanmedian(np.diff(y)))
    dx2 = max(dx * dx, 1e-12)
    dy2 = max(dy * dy, 1e-12)
    lower_z, diag_z, upper_z = vertical_coefficients(z_star, f_profile)
    out = np.zeros_like(psi, dtype="f8")
    core = (
        (psi[:, 1:-1, 2:] - 2.0 * psi[:, 1:-1, 1:-1] + psi[:, 1:-1, :-2]) / dx2
        + (psi[:, 2:, 1:-1] - 2.0 * psi[:, 1:-1, 1:-1] + psi[:, :-2, 1:-1]) / dy2
    )
    for k in range(psi.shape[0]):
        vert = diag_z[k] * psi[k, 1:-1, 1:-1]
        if k > 0:
            vert = vert + lower_z[k] * psi[k - 1, 1:-1, 1:-1]
        if k < psi.shape[0] - 1:
            vert = vert + upper_z[k] * psi[k + 1, 1:-1, 1:-1]
        out[k, 1:-1, 1:-1] = core[k] + vert
    return out


def invert_qgpv(forcing: np.ndarray, f_profile: np.ndarray, z_star: np.ndarray, y: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    rhs = nanfill_zero(forcing)
    nz, ny, nx = rhs.shape
    dx = float(np.nanmedian(np.diff(x)))
    dy = float(np.nanmedian(np.diff(y)))
    rhs_i = rhs[:, 1:-1, 1:-1]
    rhs_hat = dstn(rhs_i, type=1, axes=(1, 2), norm="ortho")
    ny_i, nx_i = rhs_i.shape[1], rhs_i.shape[2]
    lower_z, diag_z, upper_z = vertical_coefficients(z_star, f_profile)
    mx = np.arange(1, nx_i + 1, dtype="f8")
    my = np.arange(1, ny_i + 1, dtype="f8")
    lam_x = -4.0 * np.sin(np.pi * mx / (2.0 * (nx_i + 1))) ** 2 / max(dx * dx, 1e-12)
    lam_y = -4.0 * np.sin(np.pi * my / (2.0 * (ny_i + 1))) ** 2 / max(dy * dy, 1e-12)
    psi_hat = np.zeros_like(rhs_hat, dtype="f8")
    ab = np.zeros((3, nz), dtype="f8")
    for j in range(ny_i):
        for i in range(nx_i):
            ab.fill(0.0)
            ab[0, 1:] = upper_z[:-1]
            ab[1, :] = diag_z + lam_x[i] + lam_y[j]
            ab[2, :-1] = lower_z[1:]
            psi_hat[:, j, i] = solve_banded((1, 1), ab, rhs_hat[:, j, i], check_finite=False)
    sol_i = idstn(psi_hat, type=1, axes=(1, 2), norm="ortho")
    psi = np.zeros_like(rhs, dtype="f8")
    psi[:, 1:-1, 1:-1] = sol_i
    residual = qgpv_operator(psi, f_profile, z_star, y, x) - rhs
    rhs_p95 = float(np.nanpercentile(np.abs(rhs[:, 1:-1, 1:-1]), 95))
    res_p95 = float(np.nanpercentile(np.abs(residual[:, 1:-1, 1:-1]), 95))
    return psi, residual, res_p95 / rhs_p95 if rhs_p95 > 0 else np.nan


def taper_from_valid_mask(valid2d: np.ndarray, width: float = 5.0) -> np.ndarray:
    valid = np.asarray(valid2d, dtype=bool)
    if not np.any(valid):
        return np.zeros_like(valid, dtype="f8")
    dist = distance_transform_edt(valid)
    taper = np.ones_like(dist, dtype="f8")
    ramp = (dist > 0) & (dist < width)
    taper[~valid] = 0.0
    taper[ramp] = 0.5 * (1.0 - np.cos(np.pi * dist[ramp] / max(width, 1e-6)))
    return np.clip(taper, 0.0, 1.0)


def prepare_forcing(forcing: np.ndarray, taper_width: float = 5.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw = np.asarray(forcing, dtype="f8")
    finite = np.isfinite(raw)
    coverage2d = np.nanmean(finite.astype("f8"), axis=0)
    valid2d = coverage2d >= 0.45
    if not np.any(valid2d):
        valid2d = coverage2d > 0
    taper2d = taper_from_valid_mask(valid2d, width=taper_width)
    anomaly = np.zeros_like(raw, dtype="f8")
    for k in range(raw.shape[0]):
        layer = raw[k]
        edge2d = (taper2d > 0) & (taper2d < 0.98)
        edge_vals = layer[edge2d & np.isfinite(layer)]
        if edge_vals.size < 10:
            edge_vals = layer[valid2d & np.isfinite(layer)]
        bg = float(np.nanmean(edge_vals)) if edge_vals.size else 0.0
        anom = np.where(np.isfinite(layer), layer - bg, 0.0)
        anom *= taper2d
        vals = anom[np.isfinite(layer) & valid2d]
        weights = taper2d[np.isfinite(layer) & valid2d]
        if vals.size and np.nansum(weights) > 0:
            anom -= float(np.nansum(vals * weights) / np.nansum(weights)) * taper2d
        anomaly[k] = anom
    return anomaly, np.broadcast_to(taper2d[None, :, :], raw.shape).astype("f4"), valid2d.astype("f4")


def fit_vector_scale(u_pred: np.ndarray, v_pred: np.ndarray, u_obs: np.ndarray, v_obs: np.ndarray) -> float:
    good = np.isfinite(u_pred) & np.isfinite(v_pred) & np.isfinite(u_obs) & np.isfinite(v_obs)
    if int(good.sum()) < 20:
        return 1.0
    pred = np.concatenate([u_pred[good].ravel(), v_pred[good].ravel()])
    obs = np.concatenate([u_obs[good].ravel(), v_obs[good].ravel()])
    denom = float(np.dot(pred, pred))
    if denom <= 0 or not np.isfinite(denom):
        return 1.0
    scale = float(np.dot(pred, obs) / denom)
    return scale if np.isfinite(scale) else 1.0


def boundary_energy_ratio(u: np.ndarray, v: np.ndarray, width: int = 3) -> float:
    speed2 = np.asarray(u, dtype="f8") ** 2 + np.asarray(v, dtype="f8") ** 2
    if not np.any(np.isfinite(speed2)):
        return np.nan
    band = np.zeros(speed2.shape[-2:], dtype=bool)
    band[:width, :] = True
    band[-width:, :] = True
    band[:, :width] = True
    band[:, -width:] = True
    band3 = np.broadcast_to(band[None, :, :], speed2.shape)
    total = float(np.nansum(speed2))
    edge = float(np.nansum(np.where(band3, speed2, 0.0)))
    return edge / total if total > 0 else np.nan
