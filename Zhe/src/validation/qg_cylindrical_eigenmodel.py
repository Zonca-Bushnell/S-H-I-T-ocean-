"""QG cylindrical PV eigenmodel utilities.

This module implements the first-version model described in
``qg_cylindrical_eigenmodel.md``.  It uses only NumPy by default.  If SciPy is
available, SciPy's Bessel functions are used; otherwise small pure-Python
fallbacks cover the integer orders used by the validation examples.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
from typing import Callable, Iterable

import numpy as np

try:  # pragma: no cover - optional dependency path
    from scipy import special as _scipy_special
except Exception:  # pragma: no cover - SciPy is optional in this repo
    _scipy_special = None


ArrayLike = float | np.ndarray


@dataclass(frozen=True)
class QGColumnParams:
    """Parameters for a constant-stratification QG cylindrical mode."""

    radius: float
    pv_wavenumber: float
    stretching: float

    def __post_init__(self) -> None:
        if self.radius <= 0:
            raise ValueError("radius must be positive")
        if self.pv_wavenumber <= 0:
            raise ValueError("pv_wavenumber must be positive")
        if self.stretching <= 0:
            raise ValueError("stretching must be positive")


def radial_wavenumbers(params: QGColumnParams, kz: float) -> tuple[float, float]:
    """Return internal Helmholtz wavenumber kappa and external decay gamma."""

    gamma = math.sqrt(params.stretching) * abs(kz)
    kappa_sq = params.pv_wavenumber**2 - gamma**2
    if kappa_sq <= 0:
        raise ValueError(
            "internal kappa is imaginary; choose K^2 > S*kz^2 for this v1 model"
        )
    if gamma == 0:
        raise ValueError("kz=0 has no decaying K_m exterior in an unbounded domain")
    return math.sqrt(kappa_sq), gamma


def bessel_j(m: int, x: ArrayLike) -> ArrayLike:
    """Integer-order Bessel J_m."""

    if m < 0:
        return ((-1) ** abs(m)) * bessel_j(abs(m), x)
    if _scipy_special is not None:  # pragma: no cover
        return _scipy_special.jv(m, x)
    return _vectorize_scalar(lambda value: _bessel_j_scalar(m, value), x)


def bessel_j_derivative(m: int, x: ArrayLike) -> ArrayLike:
    """Derivative dJ_m(x)/dx."""

    return 0.5 * (bessel_j(m - 1, x) - bessel_j(m + 1, x))


def modified_bessel_k(m: int, x: ArrayLike) -> ArrayLike:
    """Integer-order modified Bessel K_m for x > 0."""

    if m < 0:
        return modified_bessel_k(abs(m), x)
    if _scipy_special is not None:  # pragma: no cover
        return _scipy_special.kv(m, x)
    return _vectorize_scalar(lambda value: _modified_bessel_k_scalar(m, value), x)


def modified_bessel_k_derivative(m: int, x: ArrayLike) -> ArrayLike:
    """Derivative dK_m(x)/dx."""

    return -0.5 * (modified_bessel_k(m - 1, x) + modified_bessel_k(m + 1, x))


def interior_radial(m: int, r: ArrayLike, kappa: float) -> ArrayLike:
    """Internal radial structure J_m(kappa*r)."""

    return bessel_j(m, kappa * np.asarray(r))


def interior_radial_derivative(m: int, r: ArrayLike, kappa: float) -> ArrayLike:
    """Derivative d/dr J_m(kappa*r)."""

    return kappa * bessel_j_derivative(m, kappa * np.asarray(r))


def exterior_amplitude(m: int, params: QGColumnParams, kappa: float, gamma: float) -> float:
    """Amplitude C that makes psi continuous at r=a."""

    numerator = bessel_j(m, kappa * params.radius)
    denominator = modified_bessel_k(m, gamma * params.radius)
    return float(numerator / denominator)


def exterior_radial(
    m: int, r: ArrayLike, gamma: float, amplitude: float
) -> ArrayLike:
    """External decaying radial structure C*K_m(gamma*r)."""

    return amplitude * modified_bessel_k(m, gamma * np.asarray(r))


def exterior_radial_derivative(
    m: int, r: ArrayLike, gamma: float, amplitude: float
) -> ArrayLike:
    """Derivative d/dr C*K_m(gamma*r)."""

    return amplitude * gamma * modified_bessel_k_derivative(m, gamma * np.asarray(r))


def matched_radial(m: int, r: ArrayLike, params: QGColumnParams, kz: float) -> np.ndarray:
    """Piecewise radial structure with no eager evaluation at r=0."""

    r_arr = np.asarray(r, dtype=float)
    kappa, gamma = radial_wavenumbers(params, kz)
    amplitude = exterior_amplitude(m, params, kappa, gamma)
    radial = np.empty_like(r_arr, dtype=float)
    inside = r_arr <= params.radius
    radial[inside] = interior_radial(m, r_arr[inside], kappa)
    radial[~inside] = exterior_radial(m, r_arr[~inside], gamma, amplitude)
    return radial


def matched_radial_derivative(
    m: int, r: ArrayLike, params: QGColumnParams, kz: float
) -> np.ndarray:
    """Piecewise radial derivative with no eager evaluation at r=0."""

    r_arr = np.asarray(r, dtype=float)
    kappa, gamma = radial_wavenumbers(params, kz)
    amplitude = exterior_amplitude(m, params, kappa, gamma)
    radial_dr = np.empty_like(r_arr, dtype=float)
    inside = r_arr <= params.radius
    radial_dr[inside] = interior_radial_derivative(m, r_arr[inside], kappa)
    radial_dr[~inside] = exterior_radial_derivative(
        m, r_arr[~inside], gamma, amplitude
    )
    return radial_dr


def eigencondition_residual(m: int, params: QGColumnParams, kz: float) -> float:
    """Boundary derivative mismatch for the matched piecewise mode.

    The residual is zero when

        kappa J_m'(kappa*a)/J_m(kappa*a)
        = gamma K_m'(gamma*a)/K_m(gamma*a).
    """

    kappa, gamma = radial_wavenumbers(params, kz)
    a = params.radius
    left = kappa * bessel_j_derivative(m, kappa * a) / bessel_j(m, kappa * a)
    right = (
        gamma
        * modified_bessel_k_derivative(m, gamma * a)
        / modified_bessel_k(m, gamma * a)
    )
    return float(left - right)


def boundary_jumps(m: int, params: QGColumnParams, kz: float) -> tuple[float, float]:
    """Return [psi] and [d psi/dr] at r=a for the continuity-matched mode."""

    kappa, gamma = radial_wavenumbers(params, kz)
    amplitude = exterior_amplitude(m, params, kappa, gamma)
    a = params.radius
    psi_jump = interior_radial(m, a, kappa) - exterior_radial(m, a, gamma, amplitude)
    dr_jump = interior_radial_derivative(m, a, kappa) - exterior_radial_derivative(
        m, a, gamma, amplitude
    )
    return float(psi_jump), float(dr_jump)


def piecewise_streamfunction(
    x: ArrayLike,
    y: ArrayLike,
    z: ArrayLike,
    *,
    m: int,
    kz: float,
    params: QGColumnParams,
    phase: float = 0.0,
) -> np.ndarray:
    """Real streamfunction Re[R(r) exp(i(m*theta+kz*z+phase))]."""

    x_arr, y_arr, z_arr = np.broadcast_arrays(x, y, z)
    r = np.hypot(x_arr, y_arr)
    theta = np.arctan2(y_arr, x_arr)
    radial = matched_radial(m, r, params, kz)
    return np.asarray(radial * np.cos(m * theta + kz * z_arr + phase), dtype=float)


def helical_tilt_streamfunction(
    x: ArrayLike,
    y: ArrayLike,
    z: ArrayLike,
    *,
    kz: float,
    params: QGColumnParams,
    phase: float = 0.0,
) -> np.ndarray:
    """m=1 helical tilt mode cos(theta+kz*z+phase)."""

    return piecewise_streamfunction(x, y, z, m=1, kz=kz, params=params, phase=phase)


def geostrophic_velocity(
    x: ArrayLike,
    y: ArrayLike,
    z: ArrayLike,
    *,
    m: int,
    kz: float,
    params: QGColumnParams,
    phase: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return geostrophic velocity (u, v)=(-psi_y, psi_x)."""

    x_arr, y_arr, z_arr = np.broadcast_arrays(x, y, z)
    r = np.hypot(x_arr, y_arr)
    theta = np.arctan2(y_arr, x_arr)
    safe_r = np.where(r == 0.0, np.nan, r)
    radial = matched_radial(m, r, params, kz)
    radial_dr = matched_radial_derivative(m, r, params, kz)
    total_phase = m * theta + kz * z_arr + phase
    dpsi_dr = radial_dr * np.cos(total_phase)
    dpsi_dtheta = -m * radial * np.sin(total_phase)
    psi_x = np.cos(theta) * dpsi_dr - np.sin(theta) * dpsi_dtheta / safe_r
    psi_y = np.sin(theta) * dpsi_dr + np.cos(theta) * dpsi_dtheta / safe_r
    return -psi_y, psi_x


def linear_tilt_from_axisymmetric(
    x: ArrayLike,
    y: ArrayLike,
    z: ArrayLike,
    *,
    kz: float,
    params: QGColumnParams,
    x_offset: Callable[[np.ndarray], np.ndarray],
    y_offset: Callable[[np.ndarray], np.ndarray],
    phase: float = 0.0,
) -> np.ndarray:
    """Small-displacement, straight-tilt approximation of an m=0 base vortex."""

    x_arr, y_arr, z_arr = np.broadcast_arrays(x, y, z)
    r = np.hypot(x_arr, y_arr)
    safe_r = np.where(r == 0.0, np.nan, r)
    theta = np.arctan2(y_arr, x_arr)
    base = piecewise_streamfunction(
        x_arr, y_arr, z_arr, m=0, kz=kz, params=params, phase=phase
    )
    u_base, v_base = geostrophic_velocity(
        x_arr, y_arr, z_arr, m=0, kz=kz, params=params, phase=phase
    )
    # psi_x=v_g and psi_y=-u_g by geostrophic definition.
    psi_x = v_base
    psi_y = -u_base
    shifted = base - x_offset(z_arr) * psi_x - y_offset(z_arr) * psi_y
    return np.where(r == 0.0, base, shifted)


def wave_packet_streamfunction(
    x: ArrayLike,
    y: ArrayLike,
    z: ArrayLike,
    *,
    m: int,
    kz_values: Iterable[float],
    weights: Iterable[complex],
    params: QGColumnParams,
    phase: float = 0.0,
) -> np.ndarray:
    """Finite quadrature approximation to a continuous vertical wave packet."""

    x_arr, y_arr, z_arr = np.broadcast_arrays(x, y, z)
    total = np.zeros_like(x_arr, dtype=complex)
    theta = np.arctan2(y_arr, x_arr)
    r = np.hypot(x_arr, y_arr)
    for kz, weight in zip(kz_values, weights, strict=True):
        radial = matched_radial(m, r, params, float(kz))
        total += weight * radial * np.exp(1j * (m * theta + kz * z_arr + phase))
    return total.real


def radial_ode_residual(
    r: np.ndarray,
    radial: np.ndarray,
    *,
    m: int,
    coefficient: float,
) -> np.ndarray:
    """Finite-difference residual for R''+R'/r-m^2*R/r^2+coefficient*R=0."""

    r = np.asarray(r, dtype=float)
    radial = np.asarray(radial, dtype=float)
    dr = r[1] - r[0]
    first = np.gradient(radial, dr, edge_order=2)
    second = np.gradient(first, dr, edge_order=2)
    return second + first / r - (m**2) * radial / (r**2) + coefficient * radial


def _vectorize_scalar(function: Callable[[float], float], x: ArrayLike) -> ArrayLike:
    arr = np.asarray(x, dtype=float)
    values = np.vectorize(function, otypes=[float])(arr)
    if np.isscalar(x):
        return float(values)
    return values


def _bessel_j_scalar(m: int, x: float) -> float:
    if x == 0:
        return 1.0 if m == 0 else 0.0
    total = 0.0
    for s in range(120):
        log_abs = (2 * s + m) * math.log(abs(x) / 2.0)
        log_abs -= math.lgamma(s + 1) + math.lgamma(s + m + 1)
        term = ((-1) ** s) * math.exp(log_abs)
        if x < 0 and m % 2:
            term = -term
        total += term
        if abs(term) < 1e-15 * max(1.0, abs(total)):
            break
    return total


@lru_cache(maxsize=4096)
def _modified_bessel_k_scalar(m: int, x: float) -> float:
    if x <= 0:
        raise ValueError("modified Bessel K_m requires x > 0")
    # K_m(x)=integral_0^infty exp(-x cosh(t))*cosh(m t) dt.
    # t=12 is enough for the moderate validation values used here because the
    # integrand decays double-exponentially for large t.
    n = 4096
    t_max = 12.0
    t = np.linspace(0.0, t_max, n + 1)
    integrand = np.exp(-x * np.cosh(t)) * np.cosh(m * t)
    return float(np.trapezoid(integrand, t))
