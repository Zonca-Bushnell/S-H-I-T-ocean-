from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import AxisLine


def _safe_gradient(values: np.ndarray, coord: np.ndarray, axis: int) -> np.ndarray:
    coord = np.asarray(coord, dtype=float)
    if coord.size < 2:
        return np.zeros_like(values, dtype=float)
    return np.gradient(values, coord, axis=axis, edge_order=1)


@dataclass(frozen=True)
class CurvedTubeMetric:
    """First-order curved-tube metric audit on a representative polar grid."""

    depth_m: np.ndarray
    radial_m: np.ndarray
    theta_rad: np.ndarray
    kappa_1_per_m: np.ndarray
    kappa_2_per_m: np.ndarray
    jacobian: np.ndarray
    sqrt_g: np.ndarray
    g_ss: np.ndarray
    g_xixi: np.ndarray
    g_etaeta: np.ndarray
    valid_metric_mask: np.ndarray
    epsilon_tilt: np.ndarray
    epsilon_curvature: np.ndarray
    large_curvature_threshold: float

    @classmethod
    def from_axis(
        cls,
        axis: AxisLine,
        radial_m: np.ndarray,
        theta_rad: np.ndarray,
        *,
        large_curvature_threshold: float = 1.0,
    ) -> "CurvedTubeMetric":
        depth = np.asarray(axis.depth_m, dtype=float)
        radial = np.asarray(radial_m, dtype=float)
        theta = np.asarray(theta_rad, dtype=float)

        dx_dz, dy_dz = axis.slopes_m_per_m()
        epsilon_tilt = np.hypot(dx_dz, dy_dz)
        kappa_x, kappa_y = _curvature_components_xy(axis)
        kappa_mag = np.hypot(kappa_x, kappa_y)
        epsilon_curvature = kappa_mag[:, None] * radial[None, :]

        rr, tt = np.meshgrid(radial, theta, indexing="ij")
        x = rr * np.cos(tt)
        y = rr * np.sin(tt)
        jacobian = 1.0 - kappa_x[:, None, None] * x[None, :, :] - kappa_y[:, None, None] * y[None, :, :]
        sqrt_g = jacobian.copy()
        g_ss = jacobian**2
        g_xixi = np.ones_like(jacobian)
        g_etaeta = np.ones_like(jacobian)
        valid_metric_mask = (jacobian > 0.0) & (
            epsilon_curvature[:, :, None] <= large_curvature_threshold
        )

        return cls(
            depth_m=depth,
            radial_m=radial,
            theta_rad=theta,
            kappa_1_per_m=kappa_x,
            kappa_2_per_m=kappa_y,
            jacobian=jacobian,
            sqrt_g=sqrt_g,
            g_ss=g_ss,
            g_xixi=g_xixi,
            g_etaeta=g_etaeta,
            valid_metric_mask=valid_metric_mask,
            epsilon_tilt=epsilon_tilt,
            epsilon_curvature=epsilon_curvature,
            large_curvature_threshold=float(large_curvature_threshold),
        )

    @property
    def kappa_magnitude_per_m(self) -> np.ndarray:
        return np.hypot(self.kappa_1_per_m, self.kappa_2_per_m)

    def jacobian_mean(self, support_mask: np.ndarray | None = None) -> np.ndarray:
        values = self.jacobian
        if support_mask is not None:
            values = np.where(support_mask, values, np.nan)
        with np.errstate(invalid="ignore"):
            out = np.nanmean(values, axis=2)
        return np.where(np.isfinite(out), out, 1.0)

    def valid_fraction(self, support_mask: np.ndarray | None = None) -> np.ndarray:
        mask = self.valid_metric_mask
        if support_mask is not None:
            mask = mask & support_mask
        with np.errstate(invalid="ignore"):
            return np.nanmean(mask.astype(float), axis=2)

    def jacobian_min(self, support_mask: np.ndarray | None = None) -> np.ndarray:
        values = self.jacobian
        if support_mask is not None:
            values = np.where(support_mask, values, np.nan)
        with np.errstate(invalid="ignore"):
            out = np.nanmin(values, axis=2)
        return np.where(np.isfinite(out), out, np.nan)

    def jacobian_max(self, support_mask: np.ndarray | None = None) -> np.ndarray:
        values = self.jacobian
        if support_mask is not None:
            values = np.where(support_mask, values, np.nan)
        with np.errstate(invalid="ignore"):
            out = np.nanmax(values, axis=2)
        return np.where(np.isfinite(out), out, np.nan)


def _curvature_components_xy(axis: AxisLine) -> tuple[np.ndarray, np.ndarray]:
    depth = np.asarray(axis.depth_m, dtype=float)
    if depth.size < 3:
        zeros = np.zeros_like(depth)
        return zeros, zeros

    x = axis.x_km * 1000.0
    y = axis.y_km * 1000.0
    z = -depth
    curve = np.stack([x, y, z], axis=1)
    dr_dp = np.stack(
        [
            _safe_gradient(curve[:, 0], depth, axis=0),
            _safe_gradient(curve[:, 1], depth, axis=0),
            _safe_gradient(curve[:, 2], depth, axis=0),
        ],
        axis=1,
    )
    ds_dp = np.linalg.norm(dr_dp, axis=1)
    ds_dp[ds_dp == 0.0] = 1.0
    tangent = dr_dp / ds_dp[:, None]
    dt_dp = np.stack(
        [
            _safe_gradient(tangent[:, 0], depth, axis=0),
            _safe_gradient(tangent[:, 1], depth, axis=0),
            _safe_gradient(tangent[:, 2], depth, axis=0),
        ],
        axis=1,
    )
    curvature_vector = dt_dp / ds_dp[:, None]
    kappa_x = np.where(np.isfinite(curvature_vector[:, 0]), curvature_vector[:, 0], 0.0)
    kappa_y = np.where(np.isfinite(curvature_vector[:, 1]), curvature_vector[:, 1], 0.0)
    return kappa_x, kappa_y


def jacobian_weighted_divergence(
    fn: np.ndarray,
    fz: np.ndarray,
    radial_m: np.ndarray,
    depth_m: np.ndarray,
    jacobian_mean: np.ndarray,
) -> np.ndarray:
    radial = np.asarray(radial_m, dtype=float).copy()
    if radial.size > 1 and radial[0] <= 0:
        radial[0] = radial[1] * 0.5
    depth = np.asarray(depth_m, dtype=float)
    jbar = np.where(np.abs(jacobian_mean) > 1e-12, jacobian_mean, np.nan)
    radial_flux = radial[None, :] * jbar * fn
    div_r = _safe_gradient(radial_flux, radial, axis=1) / (radial[None, :] * jbar)
    div_z = _safe_gradient(jbar * fz, depth, axis=0) / jbar
    return div_r + div_z
