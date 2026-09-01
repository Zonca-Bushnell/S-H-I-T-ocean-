from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .contracts import RHO0
from .fields import RepresentativeSlice
from .geometry import AxisLine, build_bishop_frame
from .metric import CurvedTubeMetric, jacobian_weighted_divergence


def _azimuthal_anomaly(values: np.ndarray) -> np.ndarray:
    mean = np.nanmean(values, axis=2, keepdims=True)
    return values - mean


def _finite_mean(values: np.ndarray, axis: int | tuple[int, ...]) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        return np.nanmean(values, axis=axis)


def _safe_gradient(values: np.ndarray, coord: np.ndarray, axis: int) -> np.ndarray:
    coord = np.asarray(coord, dtype=float)
    if coord.size < 2:
        return np.zeros_like(values)
    return np.gradient(values, coord, axis=axis, edge_order=1)


def _periodic_gradient(values: np.ndarray, theta: np.ndarray) -> np.ndarray:
    dtheta = float(np.nanmedian(np.diff(np.unwrap(theta))))
    if not np.isfinite(dtheta) or dtheta == 0:
        dtheta = 2.0 * np.pi / values.shape[-1]
    return (np.roll(values, -1, axis=-1) - np.roll(values, 1, axis=-1)) / (2.0 * dtheta)


def _positive_radial_coord(radial: np.ndarray) -> np.ndarray:
    coord = np.asarray(radial, dtype=float).copy()
    if coord.size > 1 and coord[0] <= 0:
        coord[0] = coord[1] * 0.5
    return coord


@dataclass(frozen=True)
class EPFluxResult:
    profiles: pd.DataFrame
    metrics: dict[str, float]


class EPFluxCalculator:
    """Classic, tilted, and curved-tube EP diagnostics for one composite slice."""

    def __init__(
        self,
        representative: RepresentativeSlice,
        axis: AxisLine,
        *,
        f0: float,
        n2: np.ndarray | float,
        buoyancy_source: str = "thermal_wind",
        curved_tube_mode: str = "scale_audit",
        large_curvature_threshold: float = 1.0,
    ) -> None:
        self.rep = representative
        self.axis = axis.interpolate_to(representative.depth_m)
        self.f0 = float(f0)
        self.n2 = self._coerce_n2(n2)
        if buoyancy_source not in ("thermal_wind", "streamfunction_dz"):
            raise ValueError("buoyancy_source must be thermal_wind or streamfunction_dz")
        if curved_tube_mode not in ("scale_audit", "jacobian_only", "jacobian_christoffel"):
            raise ValueError("curved_tube_mode must be scale_audit, jacobian_only, or jacobian_christoffel")
        self.buoyancy_source = buoyancy_source
        self.curved_tube_mode = curved_tube_mode
        self.large_curvature_threshold = float(large_curvature_threshold)

    def compute(self) -> EPFluxResult:
        rep = self.rep
        depth = rep.depth_m
        radial = rep.radial_m
        theta = rep.theta_rad
        ur, ut = rep.polar_velocity()
        psi = self._streamfunction_from_tangential(ut, radial)
        if self.buoyancy_source == "thermal_wind":
            b_tilted, b_ordinary, b_tilt_correction = self._thermal_wind_buoyancy_parts(
                rep.u, rep.v, radial, theta
            )
        else:
            b_tilted, b_ordinary, b_tilt_correction = self._streamfunction_buoyancy_parts(
                psi, radial, theta
            )

        ur_p = _azimuthal_anomaly(ur)
        ut_p = _azimuthal_anomaly(ut)
        bt_p = _azimuthal_anomaly(b_tilted)
        bo_p = _azimuthal_anomaly(b_ordinary)
        bc_p = _azimuthal_anomaly(b_tilt_correction)

        fn_classic = -RHO0 * _finite_mean(ut_p * ur_p, axis=2)
        fz_ordinary = RHO0 * self.f0 * _finite_mean(ur_p * bo_p, axis=2) / self.n2[:, None]
        fz_tilted = RHO0 * self.f0 * _finite_mean(ur_p * bt_p, axis=2) / self.n2[:, None]
        fz_tilt_correction = RHO0 * self.f0 * _finite_mean(ur_p * bc_p, axis=2) / self.n2[:, None]

        div_classic = self._divergence(fn_classic, fz_ordinary, radial, depth)
        div_tilted = self._divergence(fn_classic, fz_tilted, radial, depth)
        frame = build_bishop_frame(self.axis)
        metric = CurvedTubeMetric.from_axis(
            self.axis,
            radial,
            theta,
            large_curvature_threshold=self.large_curvature_threshold,
        )
        support_mask = np.isfinite(rep.speed)
        jacobian_mean = metric.jacobian_mean(support_mask)
        jacobian_min = metric.jacobian_min(support_mask)
        jacobian_max = metric.jacobian_max(support_mask)
        metric_valid_fraction = metric.valid_fraction(support_mask)
        epsilon_tilt = metric.epsilon_tilt[:, None] * np.ones_like(fn_classic)
        epsilon_curvature = metric.epsilon_curvature
        invalid_or_large = metric_valid_fraction < 0.999

        div_jacobian = jacobian_weighted_divergence(fn_classic, fz_tilted, radial, depth, jacobian_mean)
        div_jacobian_correction = div_jacobian - div_tilted
        curvature_per_m = metric.kappa_magnitude_per_m[:, None]
        div_christoffel = curvature_per_m * fn_classic
        div_curved_total = div_jacobian + div_christoffel
        scale_upper_bound = frame.curvature_proxy_per_m[:, None] * fn_classic
        if self.curved_tube_mode == "scale_audit":
            curved_div = div_tilted
        elif self.curved_tube_mode == "jacobian_only":
            curved_div = div_jacobian
        else:
            curved_div = div_curved_total

        pv_flux = self._pv_flux_proxy(psi, ur_p, radial, theta, depth)
        profiles = self._profiles_table(
            fn_classic,
            fz_ordinary,
            fz_tilted,
            fz_tilt_correction,
            div_classic,
            div_tilted,
            div_jacobian,
            div_jacobian_correction,
            div_christoffel,
            div_curved_total,
            curved_div,
            scale_upper_bound,
            epsilon_tilt,
            epsilon_curvature,
            jacobian_mean,
            jacobian_min,
            jacobian_max,
            metric_valid_fraction,
            invalid_or_large,
            pv_flux,
        )
        return EPFluxResult(profiles=profiles, metrics=self._metrics(profiles))

    def _coerce_n2(self, n2: np.ndarray | float) -> np.ndarray:
        if np.isscalar(n2):
            return np.full_like(self.rep.depth_m, float(n2), dtype=float)
        values = np.asarray(n2, dtype=float)
        if values.shape == self.rep.depth_m.shape:
            return values
        return np.interp(self.rep.depth_m, np.linspace(self.rep.depth_m.min(), self.rep.depth_m.max(), values.size), values)

    def _streamfunction_from_tangential(self, ut: np.ndarray, radial: np.ndarray) -> np.ndarray:
        psi = np.zeros_like(ut, dtype=float)
        dr = np.diff(radial)
        for idx in range(1, radial.size):
            psi[:, idx, :] = psi[:, idx - 1, :] + 0.5 * (ut[:, idx, :] + ut[:, idx - 1, :]) * dr[idx - 1]
        return psi - np.nanmean(psi, axis=(1, 2), keepdims=True)

    def _streamfunction_buoyancy_parts(
        self, psi: np.ndarray, radial: np.ndarray, theta: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        depth = self.rep.depth_m
        dpsi_dz_tilted = _safe_gradient(psi, depth, axis=0)
        dpsi_dr = _safe_gradient(psi, radial, axis=1)
        dpsi_dtheta = _periodic_gradient(psi, theta)
        radial_positive = _positive_radial_coord(radial)
        rr = radial_positive[None, :, None]
        grad_x = dpsi_dr * np.cos(theta)[None, None, :] - dpsi_dtheta * np.sin(theta)[None, None, :] / rr
        grad_y = dpsi_dr * np.sin(theta)[None, None, :] + dpsi_dtheta * np.cos(theta)[None, None, :] / rr
        dx_dz, dy_dz = self.axis.slopes_m_per_m()
        tilt_projection = dx_dz[:, None, None] * grad_x + dy_dz[:, None, None] * grad_y
        dpsi_dz_ordinary = dpsi_dz_tilted + tilt_projection
        return (
            self.f0 * dpsi_dz_tilted,
            self.f0 * dpsi_dz_ordinary,
            self.f0 * tilt_projection,
        )

    def _thermal_wind_buoyancy_parts(
        self, u_g: np.ndarray, v_g: np.ndarray, radial: np.ndarray, theta: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        depth = self.rep.depth_m
        du_dz_tilted = _safe_gradient(u_g, depth, axis=0)
        dv_dz_tilted = _safe_gradient(v_g, depth, axis=0)

        du_dx, du_dy = self._cartesian_gradient(u_g, radial, theta)
        dv_dx, dv_dy = self._cartesian_gradient(v_g, radial, theta)
        dx_dz, dy_dz = self.axis.slopes_m_per_m()
        du_dz_correction = dx_dz[:, None, None] * du_dx + dy_dz[:, None, None] * du_dy
        dv_dz_correction = dx_dz[:, None, None] * dv_dx + dy_dz[:, None, None] * dv_dy

        b_tilted = self._buoyancy_from_thermal_wind_shear(du_dz_tilted, dv_dz_tilted, radial, theta)
        b_correction = self._buoyancy_from_thermal_wind_shear(
            du_dz_correction, dv_dz_correction, radial, theta
        )
        return b_tilted, b_tilted + b_correction, b_correction

    def _cartesian_gradient(
        self, values: np.ndarray, radial: np.ndarray, theta: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        radial_positive = _positive_radial_coord(radial)
        d_dr = _safe_gradient(values, radial, axis=1)
        d_dtheta = _periodic_gradient(values, theta)
        rr = radial_positive[None, :, None]
        cos_t = np.cos(theta)[None, None, :]
        sin_t = np.sin(theta)[None, None, :]
        grad_x = d_dr * cos_t - d_dtheta * sin_t / rr
        grad_y = d_dr * sin_t + d_dtheta * cos_t / rr
        return grad_x, grad_y

    def _buoyancy_from_thermal_wind_shear(
        self, du_dz: np.ndarray, dv_dz: np.ndarray, radial: np.ndarray, theta: np.ndarray
    ) -> np.ndarray:
        db_dx = self.f0 * dv_dz
        db_dy = -self.f0 * du_dz
        cos_t = np.cos(theta)[None, None, :]
        sin_t = np.sin(theta)[None, None, :]
        db_dr = db_dx * cos_t + db_dy * sin_t
        b = np.zeros_like(db_dr, dtype=float)
        dr = np.diff(radial)
        for idx in range(1, radial.size):
            b[:, idx, :] = b[:, idx - 1, :] + 0.5 * (db_dr[:, idx, :] + db_dr[:, idx - 1, :]) * dr[idx - 1]
        return b - np.nanmean(b, axis=(1, 2), keepdims=True)

    def _divergence(self, fn: np.ndarray, fz: np.ndarray, radial: np.ndarray, depth: np.ndarray) -> np.ndarray:
        radial_positive = _positive_radial_coord(radial)
        radial_flux = radial_positive[None, :] * fn
        div_r = _safe_gradient(radial_flux, radial_positive, axis=1) / radial_positive[None, :]
        div_z = _safe_gradient(fz, depth, axis=0)
        return div_r + div_z

    def _pv_flux_proxy(
        self,
        psi: np.ndarray,
        ur_prime: np.ndarray,
        radial: np.ndarray,
        theta: np.ndarray,
        depth: np.ndarray,
    ) -> np.ndarray:
        dpsi_dr = _safe_gradient(psi, radial, axis=1)
        radial_positive = _positive_radial_coord(radial)
        lap_r = _safe_gradient(radial_positive[None, :, None] * dpsi_dr, radial_positive, axis=1) / radial_positive[None, :, None]
        lap_t = _periodic_gradient(_periodic_gradient(psi, theta), theta) / (radial_positive[None, :, None] ** 2)
        dpsi_dz = _safe_gradient(psi, depth, axis=0)
        strat = (self.f0**2 / self.n2)[:, None, None] * dpsi_dz
        pv = lap_r + lap_t + _safe_gradient(strat, depth, axis=0)
        return _finite_mean(ur_prime * _azimuthal_anomaly(pv), axis=2)

    def _profiles_table(self, *arrays: np.ndarray) -> pd.DataFrame:
        names = [
            "F_n_classic",
            "F_z_ordinary",
            "F_z_tilted",
            "F_z_tilt_correction",
            "divF_classic",
            "divF_tilted",
            "divF_jacobian",
            "divF_jacobian_correction",
            "divF_christoffel_qg_approx",
            "divF_curved_total",
            "divF_curved_tube_qg_approx",
            "divF_scale_upper_bound",
            "epsilon_tilt",
            "epsilon_curvature",
            "jacobian_mean",
            "jacobian_min",
            "jacobian_max",
            "metric_valid_fraction",
            "metric_invalid_or_large_curvature",
            "pv_flux_proxy",
        ]
        rows: list[dict[str, float | str]] = []
        for iz, depth in enumerate(self.rep.depth_m):
            for ir, radius_m in enumerate(self.rep.radial_m):
                row: dict[str, float | str] = {
                    "polarity": self.rep.polarity,
                    "tau": self.rep.tau,
                    "depth_m": float(depth),
                    "radius_m": float(radius_m),
                    "radius_over_R": float(radius_m / self.rep.radius_m),
                    "axis_x_km": float(self.axis.x_km[iz]),
                    "axis_y_km": float(self.axis.y_km[iz]),
                    "axis_tilt_km": float(self.axis.tilt_km[iz]),
                    "buoyancy_source": self.buoyancy_source,
                    "curved_tube_mode": self.curved_tube_mode,
                }
                for name, values in zip(names, arrays):
                    value = values[iz, ir]
                    if isinstance(value, (bool, np.bool_)):
                        row[name] = bool(value)
                    else:
                        row[name] = float(value)
                rows.append(row)
        return pd.DataFrame(rows)

    def _metrics(self, profiles: pd.DataFrame) -> dict[str, float]:
        core = profiles[profiles["radius_over_R"] <= 1.5]
        out: dict[str, float] = {
            "rows": float(len(profiles)),
            "core_rows": float(len(core)),
            "finite_divF_tilted_fraction": float(np.isfinite(profiles["divF_tilted"]).mean()),
            "finite_pv_flux_fraction": float(np.isfinite(profiles["pv_flux_proxy"]).mean()),
            "median_axis_tilt_km": float(np.nanmedian(profiles["axis_tilt_km"])),
        }
        ordinary = core["F_z_ordinary"].to_numpy(float)
        correction = core["F_z_tilt_correction"].to_numpy(float)
        denom = np.nanmedian(np.abs(ordinary)) + 1e-30
        out["median_abs_tilt_correction_over_ordinary"] = float(np.nanmedian(np.abs(correction)) / denom)
        div = core["divF_tilted"].to_numpy(float)
        pv = core["pv_flux_proxy"].to_numpy(float)
        mask = np.isfinite(div) & np.isfinite(pv)
        out["divF_pv_flux_corr_core"] = float(np.corrcoef(div[mask], pv[mask])[0, 1]) if mask.sum() > 2 else float("nan")
        curved = core["divF_curved_tube_qg_approx"].to_numpy(float)
        curved_total = core["divF_curved_total"].to_numpy(float)
        jacobian_correction = core["divF_jacobian_correction"].to_numpy(float)
        christoffel = core["divF_christoffel_qg_approx"].to_numpy(float)
        scale_upper = core["divF_scale_upper_bound"].to_numpy(float)
        denom_div = np.nanmedian(np.abs(div)) + 1e-30
        out["median_abs_curved_minus_tilted_over_tilted"] = float(np.nanmedian(np.abs(curved - div)) / denom_div)
        out["median_abs_curved_total_minus_tilted_over_tilted"] = float(
            np.nanmedian(np.abs(curved_total - div)) / denom_div
        )
        out["median_abs_jacobian_correction_over_tilted"] = float(
            np.nanmedian(np.abs(jacobian_correction)) / denom_div
        )
        out["median_abs_christoffel_over_tilted"] = float(np.nanmedian(np.abs(christoffel)) / denom_div)
        out["median_abs_scale_upper_bound_over_tilted"] = float(np.nanmedian(np.abs(scale_upper)) / denom_div)
        out["median_epsilon_tilt"] = float(np.nanmedian(np.abs(core["epsilon_tilt"])))
        out["median_epsilon_curvature"] = float(np.nanmedian(np.abs(core["epsilon_curvature"])))
        out["p90_epsilon_curvature"] = float(np.nanpercentile(np.abs(core["epsilon_curvature"]), 90))
        out["metric_valid_fraction_core"] = float(np.nanmean(core["metric_valid_fraction"]))
        out["jacobian_min_core"] = float(np.nanmin(core["jacobian_min"]))
        out["jacobian_max_core"] = float(np.nanmax(core["jacobian_max"]))
        return out
