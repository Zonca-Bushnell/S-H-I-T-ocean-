from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap, Normalize, SymLogNorm
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from tqdm import tqdm

from .axis_streamfunction_separation import DEFAULT_AXIS_DIR, DEFAULT_CATALOG, DEFAULT_INPUT_DAILY, grid_spacing_m, parse_csv_list, read_daily_uv, relative_vorticity, streamfunction_from_zeta
from .lifecycle_common import (
    DEFAULT_LIFECYCLE_ROOT,
    DEFAULT_POLARITIES,
    DEFAULT_SHAPE_BY_SHAPE_DIR,
    DEFAULT_SHAPES,
    PHASE_NAMES,
    apply_lifecycle_limits,
    load_center_lines,
    load_lifecycle_objects,
)
from .tilted_ep_flux_validation import (
    DEFAULT_CLIMATOLOGY,
    DEFAULT_CLIMATOLOGY_NC,
    OMEGA,
    azimuth_second_derivative,
    bilinear_sample,
    ddz,
    divergence,
    load_n2,
    make_polar_grid,
    radial_derivative,
    read_climatology_uv,
    sample_object_fields,
)
from .representative_velocity_stack_tilted import axis_xy_m, fit_pooled_axis
from .representative_velocity_stack import make_xy_grid, nearest_depth_indices, parse_float_list, velocity_from_psi


DEFAULT_OUTPUT = DEFAULT_LIFECYCLE_ROOT / "nondim_theory_validation"
TAU_CENTERS = {"birth": 0.1, "growth": 0.3, "mature": 0.5, "decay": 0.7, "death": 0.9}


def zero_center_line(center_line: pd.DataFrame) -> pd.DataFrame:
    out = center_line.copy()
    out["x_rot_m"] = 0.0
    out["y_rot_m"] = 0.0
    return out


def axis_slopes(center_line: pd.DataFrame, depth: np.ndarray, mode: str) -> tuple[np.ndarray, np.ndarray]:
    if mode == "upright":
        return np.zeros(len(depth), dtype="f8"), np.zeros(len(depth), dtype="f8")
    x = center_line["x_rot_m"].to_numpy(dtype="f8")
    y = center_line["y_rot_m"].to_numpy(dtype="f8")
    return np.gradient(x, depth, edge_order=1), np.gradient(y, depth, edge_order=1)


def polar_gradients(psi: np.ndarray, radial: np.ndarray, theta: np.ndarray, radius_m: float) -> tuple[np.ndarray, np.ndarray]:
    r_m = np.maximum(radial * radius_m, 1.0)
    dpsi_dr = radial_derivative(psi, r_m)
    dtheta = float(theta[1] - theta[0]) if len(theta) > 1 else 2.0 * np.pi
    dpsi_dtheta = (np.roll(psi, -1, axis=2) - np.roll(psi, 1, axis=2)) / (2.0 * dtheta)
    tt = theta[None, None, :]
    rr = r_m[None, :, None]
    grad_x = np.cos(tt) * dpsi_dr - np.sin(tt) * dpsi_dtheta / rr
    grad_y = np.sin(tt) * dpsi_dr + np.cos(tt) * dpsi_dtheta / rr
    return grad_x, grad_y


def compute_nondim_terms(
    fields: dict[str, np.ndarray],
    center_line: pd.DataFrame,
    depth: np.ndarray,
    radial: np.ndarray,
    theta: np.ndarray,
    radius_m: float,
    n2: np.ndarray,
    f0: float,
    *,
    axis_mode: str,
) -> dict[str, np.ndarray]:
    psi = np.where(np.abs(fields["psi_prime"]) > 1e20, np.nan, fields["psi_prime"])
    un = fields["u_n_prime"]
    us = fields["u_s_prime"]
    un_clim = fields.get("u_n_clim")
    us_clim = fields["u_s_clim"]

    psi_prime = psi - np.nanmean(psi, axis=2, keepdims=True)
    r_m = np.maximum(radial * radius_m, 1.0)
    dpsi_dr = radial_derivative(psi_prime, r_m)
    radial_lap = radial_derivative(dpsi_dr * r_m[None, :, None], r_m) / r_m[None, :, None]
    az_lap = azimuth_second_derivative(psi_prime, theta) / (r_m[None, :, None] ** 2)
    dpsi_dz_tilted = ddz(psi_prime, depth)
    grad_x, grad_y = polar_gradients(psi_prime, radial, theta, radius_m)
    rc_z_x, rc_z_y = axis_slopes(center_line, depth, axis_mode)
    tilt_projection = rc_z_x[:, None, None] * grad_x + rc_z_y[:, None, None] * grad_y
    dpsi_dz_ordinary = dpsi_dz_tilted + tilt_projection

    strat = (f0 * f0 / n2)[:, None, None] * dpsi_dz_tilted
    q_total = radial_lap + az_lap + ddz(strat, depth)
    q_mean = np.nanmean(q_total, axis=2)
    q_prime = q_total - q_mean[:, :, None]

    un_prime = un - np.nanmean(un, axis=2, keepdims=True)
    us_prime = us - np.nanmean(us, axis=2, keepdims=True)
    coef = (f0 * f0 / n2)[:, None]
    fz_valid = np.isfinite(un_prime) & np.isfinite(dpsi_dz_tilted) & np.isfinite(dpsi_dz_ordinary) & np.isfinite(tilt_projection)
    valid_count = np.sum(fz_valid, axis=2)
    mean_tilted = np.divide(
        np.nansum(np.where(fz_valid, un_prime * dpsi_dz_tilted, 0.0), axis=2),
        valid_count,
        out=np.full(valid_count.shape, np.nan, dtype="f8"),
        where=valid_count > 0,
    )
    mean_ordinary = np.divide(
        np.nansum(np.where(fz_valid, un_prime * dpsi_dz_ordinary, 0.0), axis=2),
        valid_count,
        out=np.full(valid_count.shape, np.nan, dtype="f8"),
        where=valid_count > 0,
    )
    mean_projection = np.divide(
        np.nansum(np.where(fz_valid, un_prime * tilt_projection, 0.0), axis=2),
        valid_count,
        out=np.full(valid_count.shape, np.nan, dtype="f8"),
        where=valid_count > 0,
    )
    tilt_x = rc_z_x[:, None, None] * grad_x
    tilt_y = rc_z_y[:, None, None] * grad_y
    mean_projection_x = np.divide(
        np.nansum(np.where(fz_valid, un_prime * tilt_x, 0.0), axis=2),
        valid_count,
        out=np.full(valid_count.shape, np.nan, dtype="f8"),
        where=valid_count > 0,
    )
    mean_projection_y = np.divide(
        np.nansum(np.where(fz_valid, un_prime * tilt_y, 0.0), axis=2),
        valid_count,
        out=np.full(valid_count.shape, np.nan, dtype="f8"),
        where=valid_count > 0,
    )
    projection_mean = np.divide(
        np.nansum(np.where(fz_valid, tilt_projection, 0.0), axis=2),
        valid_count,
        out=np.full(valid_count.shape, np.nan, dtype="f8"),
        where=valid_count > 0,
    )
    projection_rms = np.sqrt(
        np.divide(
            np.nansum(np.where(fz_valid, tilt_projection * tilt_projection, 0.0), axis=2),
            valid_count,
            out=np.full(valid_count.shape, np.nan, dtype="f8"),
            where=valid_count > 0,
        )
    )
    un_prime_rms = np.sqrt(
        np.divide(
            np.nansum(np.where(fz_valid, un_prime * un_prime, 0.0), axis=2),
            valid_count,
            out=np.full(valid_count.shape, np.nan, dtype="f8"),
            where=valid_count > 0,
        )
    )
    raw_tilt_flux = -mean_projection
    fz_tilted = coef * mean_tilted
    fz_ordinary = coef * mean_ordinary
    fz_tilt_correction = coef * raw_tilt_flux
    const_coef = float(f0 * f0 / np.nanmedian(n2))
    return {
        "F_n": -np.nanmean(us_prime * un_prime, axis=2),
        "F_z_tilted": fz_tilted,
        "F_z_ordinary": fz_ordinary,
        "F_z_tilt_correction": fz_tilt_correction,
        "axis_slope_x": rc_z_x[:, None] + np.zeros_like(fz_tilted),
        "axis_slope_y": rc_z_y[:, None] + np.zeros_like(fz_tilted),
        "axis_slope_mag": np.hypot(rc_z_x, rc_z_y)[:, None] + np.zeros_like(fz_tilted),
        "N2": n2[:, None] + np.zeros_like(fz_tilted),
        "stratification_factor": coef + np.zeros_like(fz_tilted),
        "raw_tilt_flux": raw_tilt_flux,
        "raw_tilt_flux_x": -mean_projection_x,
        "raw_tilt_flux_y": -mean_projection_y,
        "F_z_tilt_correction_const_N2": const_coef * raw_tilt_flux,
        "tilt_projection_mean": projection_mean,
        "tilt_projection_rms": projection_rms,
        "un_prime_rms": un_prime_rms,
        "un_tilt_projection_cov": mean_projection,
        "pv_flux": np.nanmean(un_prime * q_prime, axis=2),
        "q_mean": q_mean,
        "q_prime_variance": np.nanmean(q_prime * q_prime, axis=2),
        "Unbar": np.nanmean(un_clim, axis=2) if un_clim is not None else np.full_like(fz_tilted, np.nan),
        "Ubar": np.nanmean(us_clim, axis=2),
        "valid": np.mean(np.isfinite(psi), axis=2),
    }


def make_accum(terms: dict[str, np.ndarray]) -> dict:
    out = {name: np.zeros_like(value, dtype="f8") for name, value in terms.items() if name != "valid"}
    out["count"] = np.zeros_like(terms["valid"], dtype="f8")
    out["objects"] = set()
    out["dates"] = set()
    return out


def add_terms(accum: dict, key: tuple[str, int, str, str], terms: dict[str, np.ndarray], object_id: int, date: str) -> None:
    if key not in accum:
        accum[key] = make_accum(terms)
    valid = np.isfinite(terms["valid"]) & (terms["valid"] > 0)
    for name, value in terms.items():
        if name == "valid":
            continue
        accum[key][name] += np.nan_to_num(value, nan=0.0) * valid
    accum[key]["count"] += valid.astype("f8")
    accum[key]["objects"].add(int(object_id))
    accum[key]["dates"].add(str(date))


def finalize(accum: dict) -> dict:
    final = {}
    for key, item in accum.items():
        count = item["count"]
        out = {"count": count, "objects": item["objects"], "dates": item["dates"]}
        for name, value in item.items():
            if name in {"count", "objects", "dates"}:
                continue
            out[name] = np.divide(value, count, out=np.full_like(value, np.nan), where=count > 0)
        final[key] = out
    return final


def rms(values: np.ndarray) -> float:
    good = np.isfinite(values)
    return float(np.sqrt(np.nanmean(values[good] ** 2))) if np.any(good) else np.nan


def rows_from_final(final: dict, radial: np.ndarray, depth: np.ndarray, radii: dict[str, float]) -> pd.DataFrame:
    rows = []
    mechanism_columns = (
        "axis_slope_x",
        "axis_slope_y",
        "axis_slope_mag",
        "N2",
        "stratification_factor",
        "raw_tilt_flux",
        "raw_tilt_flux_x",
        "raw_tilt_flux_y",
        "F_z_tilt_correction_const_N2",
        "tilt_projection_mean",
        "tilt_projection_rms",
        "un_prime_rms",
        "un_tilt_projection_cov",
    )
    for (mode, polarity, phase_index, phase_name), item in sorted(final.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2])):
        radius = radii.get(polarity)
        if radius is None:
            continue
        divf = divergence(item["F_n"], item["F_z_tilted"], radial, depth, radius)
        divf_ordinary = divergence(item["F_n"], item["F_z_ordinary"], radial, depth, radius)
        for k, depth_m in enumerate(depth):
            for j, r in enumerate(radial):
                fzt = item["F_z_tilted"][k, j]
                fz0 = item["F_z_ordinary"][k, j]
                fzc = item["F_z_tilt_correction"][k, j]
                row = {
                    "axis_mode": mode,
                    "shape_class": "all_shapes",
                    "polarity": polarity,
                    "phase_index": int(phase_index),
                    "phase_name": phase_name,
                    "tau_center": TAU_CENTERS[phase_name],
                    "depth_index": k,
                    "depth_m": float(depth_m),
                    "r_over_R": float(r),
                    "F_n": float(item["F_n"][k, j]),
                    "F_z_tilted": float(fzt),
                    "F_z_ordinary": float(fz0),
                    "F_z_tilt_correction": float(fzc),
                    "tilt_fraction": float(fzc / fzt) if np.isfinite(fzc) and np.isfinite(fzt) and abs(fzt) > 1e-14 else np.nan,
                    "ordinary_fraction": float(fz0 / fzt) if np.isfinite(fz0) and np.isfinite(fzt) and abs(fzt) > 1e-14 else np.nan,
                    "tilt_to_ordinary_ratio": float(fzc / fz0) if np.isfinite(fzc) and np.isfinite(fz0) and abs(fz0) > 1e-14 else np.nan,
                    "F_z_decomposition_residual": float(fzt - fz0 - fzc),
                    "divF": float(divf[k, j]),
                    "divF_ordinary": float(divf_ordinary[k, j]),
                    "pv_flux": float(item["pv_flux"][k, j]),
                        "q_mean": float(item["q_mean"][k, j]),
                        "q_prime_variance": float(item["q_prime_variance"][k, j]),
                        "Unbar": float(item["Unbar"][k, j]) if "Unbar" in item else np.nan,
                        "Ubar": float(item["Ubar"][k, j]),
                    "count": float(item["count"][k, j]),
                    "n_objects": len(item["objects"]),
                    "n_dates": len(item["dates"]),
                }
                for name in mechanism_columns:
                    if name in item:
                        row[name] = float(item[name][k, j])
                rows.append(row)
    return pd.DataFrame.from_records(rows)


def add_partial_tau(profiles: pd.DataFrame) -> pd.DataFrame:
    out = profiles.copy()
    out["partial_tau_Ubar"] = np.nan
    out["partial_tau_A_T"] = np.nan
    for (mode, polarity, depth_index, r_value), idx in out.groupby(["axis_mode", "polarity", "depth_index", "r_over_R"], sort=False).groups.items():
        locs = list(idx)
        part = out.loc[locs].sort_values("tau_center")
        tau = part["tau_center"].to_numpy(dtype="f8")
        if len(tau) < 2:
            continue
        out.loc[part.index, "partial_tau_Ubar"] = np.gradient(part["Ubar"].to_numpy(dtype="f8"), tau, edge_order=1)
        if "A_T" in part:
            out.loc[part.index, "partial_tau_A_T"] = np.gradient(part["A_T"].to_numpy(dtype="f8"), tau, edge_order=1)
    return out


def add_wave_activity(profiles: pd.DataFrame, radii: dict[str, float]) -> pd.DataFrame:
    parts = []
    for (mode, polarity, phase_name), part in profiles.groupby(["axis_mode", "polarity", "phase_name"], sort=False):
        part = part.sort_values(["depth_index", "r_over_R"]).copy()
        radius = radii.get(polarity, np.nan)
        qn_values = []
        for _, depth_part in part.groupby("depth_index", sort=False):
            r_m = depth_part["r_over_R"].to_numpy(dtype="f8") * radius
            qn = np.gradient(depth_part["q_mean"].to_numpy(dtype="f8"), np.maximum(r_m, 1.0), edge_order=1)
            qn_values.extend(qn.tolist())
        part["Q_n"] = qn_values
        qn_abs = np.abs(part["Q_n"].to_numpy(dtype="f8"))
        threshold = 0.05 * np.nanpercentile(qn_abs, 95) if np.isfinite(qn_abs).any() else np.nan
        part["Q_n_valid"] = np.isfinite(part["Q_n"]) & (np.abs(part["Q_n"]) > threshold)
        part["A_T"] = np.where(part["Q_n_valid"], part["q_prime_variance"] / (2.0 * part["Q_n"]), np.nan)
        parts.append(part)
    out = pd.concat(parts, ignore_index=True) if parts else profiles.copy()
    out = add_partial_tau(out)
    out["divF_nd"] = np.nan
    out["divJ"] = np.nan
    out["divJ_nd"] = np.nan
    out["A_T_advective_flux_n"] = out["Unbar"] * out["A_T"] if "Unbar" in out else np.nan
    out["J_n"] = out["F_n"] + out["A_T_advective_flux_n"]
    out["J_z"] = out["F_z_tilted"]
    out["partial_tau_A_T_nd"] = np.nan
    out["wave_activity_residual_nd"] = np.nan
    out["wave_activity_residual_F_nd"] = np.nan
    out["wave_activity_residual_J_nd"] = np.nan
    for (_, polarity, phase_name), idx in out.groupby(["axis_mode", "polarity", "phase_name"], sort=False).groups.items():
        part = out.loc[idx].sort_values(["depth_index", "r_over_R"])
        radius = radii.get(polarity, np.nan)
        if np.isfinite(radius):
            depth = np.sort(part["depth_m"].unique().astype("f8"))
            radial = np.sort(part["r_over_R"].unique().astype("f8"))
            jn = part.pivot(index="depth_m", columns="r_over_R", values="J_n").sort_index().to_numpy(dtype="f8")
            jz = part.pivot(index="depth_m", columns="r_over_R", values="J_z").sort_index().to_numpy(dtype="f8")
            divj = divergence(jn, jz, radial, depth, float(radius)).ravel()
            ordered_index = part.index.to_numpy()
            out.loc[ordered_index, "divJ"] = divj
        div = out.loc[idx, "divF"].to_numpy(dtype="f8")
        divj_values = out.loc[idx, "divJ"].to_numpy(dtype="f8")
        da = out.loc[idx, "partial_tau_A_T"].to_numpy(dtype="f8")
        div_scale = rms(div)
        divj_scale = rms(divj_values)
        da_scale = rms(da)
        if np.isfinite(div_scale) and div_scale > 0:
            out.loc[idx, "divF_nd"] = div / div_scale
        if np.isfinite(divj_scale) and divj_scale > 0:
            out.loc[idx, "divJ_nd"] = divj_values / divj_scale
        if np.isfinite(da_scale) and da_scale > 0:
            out.loc[idx, "partial_tau_A_T_nd"] = da / da_scale
        out.loc[idx, "wave_activity_residual_nd"] = out.loc[idx, "partial_tau_A_T_nd"] + out.loc[idx, "divF_nd"]
        out.loc[idx, "wave_activity_residual_F_nd"] = out.loc[idx, "partial_tau_A_T_nd"] + out.loc[idx, "divF_nd"]
        out.loc[idx, "wave_activity_residual_J_nd"] = out.loc[idx, "partial_tau_A_T_nd"] + out.loc[idx, "divJ_nd"]
    return out


def metric_corr(x: np.ndarray, y: np.ndarray) -> float:
    good = np.isfinite(x) & np.isfinite(y)
    return float(np.corrcoef(x[good], y[good])[0, 1]) if np.sum(good) > 2 else np.nan


def metric_slope(x: np.ndarray, y: np.ndarray) -> float:
    good = np.isfinite(x) & np.isfinite(y)
    if np.sum(good) <= 2:
        return np.nan
    xs = x[good] / rms(x[good])
    ys = y[good] / rms(y[good])
    return float(np.polyfit(xs, ys, deg=1)[0])


def build_metrics(profiles: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tilted = profiles[profiles["axis_mode"] == "tilted"].copy()
    flux_rows = []
    feedback_rows = []
    wave_rows = []
    for (polarity, phase_name), part in tilted.groupby(["polarity", "phase_name"], sort=True):
        fzt = part["F_z_tilted"].to_numpy(dtype="f8")
        fz0 = part["F_z_ordinary"].to_numpy(dtype="f8")
        fzc = part["F_z_tilt_correction"].to_numpy(dtype="f8")
        resid = fzt - fz0 - fzc
        flux_rows.append(
            {
                "polarity": polarity,
                "phase_name": phase_name,
                "tau_center": TAU_CENTERS[phase_name],
                "median_tilt_fraction": float(np.nanmedian(part["tilt_fraction"])),
                "median_ordinary_fraction": float(np.nanmedian(part["ordinary_fraction"])),
                "rms_Fz_tilted": rms(fzt),
                "rms_Fz_ordinary": rms(fz0),
                "rms_Fz_tilt_correction": rms(fzc),
                "decomposition_relative_residual": rms(resid) / rms(fzt) if rms(fzt) and np.isfinite(rms(fzt)) else np.nan,
                "valid_fraction": float(np.mean(np.isfinite(part["tilt_fraction"]))),
            }
        )
        div = part["divF"].to_numpy(dtype="f8")
        dutau = part["partial_tau_Ubar"].to_numpy(dtype="f8")
        good = np.isfinite(div) & np.isfinite(dutau)
        feedback_rows.append(
            {
                "polarity": polarity,
                "phase_name": phase_name,
                "tau_center": TAU_CENTERS[phase_name],
                "corr_divF_partial_tau_Ubar": metric_corr(div, dutau),
                "same_sign_fraction": float(np.mean(np.sign(div[good]) == np.sign(dutau[good]))) if np.any(good) else np.nan,
                "normalized_slope": metric_slope(div, dutau),
                "valid_bins": int(np.sum(good)),
            }
        )
        valid_wave = part["Q_n_valid"].to_numpy(dtype=bool) & np.isfinite(part["wave_activity_residual_nd"])
        wave_rows.append(
            {
                "polarity": polarity,
                "phase_name": phase_name,
                "tau_center": TAU_CENTERS[phase_name],
                "Q_n_valid_fraction": float(np.mean(part["Q_n_valid"])),
                "wave_activity_residual_nd_rms": rms(part.loc[valid_wave, "wave_activity_residual_nd"].to_numpy(dtype="f8")),
                "corr_partial_tau_A_T_nd_divF_nd": metric_corr(part["partial_tau_A_T_nd"].to_numpy(dtype="f8"), -part["divF_nd"].to_numpy(dtype="f8")),
                "valid_bins": int(np.sum(valid_wave)),
            }
        )
    upright_rows = []
    for (polarity, phase_name), tilt_part in tilted.groupby(["polarity", "phase_name"], sort=True):
        up_part = profiles[(profiles["axis_mode"] == "upright") & (profiles["polarity"] == polarity) & (profiles["phase_name"] == phase_name)].copy()
        if up_part.empty:
            continue
        merged = tilt_part.merge(
            up_part,
            on=["polarity", "phase_index", "phase_name", "depth_index", "r_over_R"],
            suffixes=("_tilted", "_upright"),
        )
        upright_rows.append(
            {
                "polarity": polarity,
                "phase_name": phase_name,
                "tau_center": TAU_CENTERS[phase_name],
                "upright_median_abs_tilt_fraction": float(np.nanmedian(np.abs(merged["tilt_fraction_upright"]))),
                "normalized_Fz_difference": rms((merged["F_z_tilted_tilted"] - merged["F_z_tilted_upright"]).to_numpy(dtype="f8")) / rms(merged["F_z_tilted_tilted"].to_numpy(dtype="f8")),
                "normalized_divF_difference": rms((merged["divF_tilted"] - merged["divF_upright"]).to_numpy(dtype="f8")) / rms(merged["divF_tilted"].to_numpy(dtype="f8")),
                "closure_corr_tilted": metric_corr(merged["divF_tilted"].to_numpy(dtype="f8"), merged["pv_flux_tilted"].to_numpy(dtype="f8")),
                "closure_corr_upright": metric_corr(merged["divF_upright"].to_numpy(dtype="f8"), merged["pv_flux_upright"].to_numpy(dtype="f8")),
            }
        )
    checklist = pd.DataFrame(
        [
            {"check": "F_z_decomposition_identity", "status": "pass" if pd.DataFrame(flux_rows)["decomposition_relative_residual"].max() < 1e-8 else "review"},
            {"check": "nondim_feedback_uses_partial_tau", "status": "pass"},
            {"check": "wave_activity_valid_Qn_mask", "status": "pass" if pd.DataFrame(wave_rows)["Q_n_valid_fraction"].between(0, 1).all() else "review"},
            {"check": "upright_tilt_fraction_near_zero", "status": "pass" if pd.DataFrame(upright_rows)["upright_median_abs_tilt_fraction"].max() < 1e-8 else "review"},
        ]
    )
    return pd.DataFrame(flux_rows), pd.DataFrame(feedback_rows), pd.DataFrame(wave_rows), pd.DataFrame(upright_rows), checklist


FZ_COMPONENTS = (
    ("F_z_tilted", "Fz tilted"),
    ("F_z_ordinary", "Fz ordinary"),
    ("F_z_tilt_correction", "Fz tilt correction"),
)


def component_norm(matrices: list[np.ndarray]) -> SymLogNorm:
    finite_abs = np.concatenate([np.abs(values[np.isfinite(values)]).ravel() for values in matrices if np.isfinite(values).any()])
    if finite_abs.size:
        vmax = np.nanpercentile(finite_abs, 99)
        if not np.isfinite(vmax) or vmax <= 0:
            vmax = float(np.nanmax(finite_abs))
    else:
        vmax = 1.0
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0
    nonzero = finite_abs[finite_abs > 0]
    linthresh = np.nanpercentile(nonzero, 50) if nonzero.size else vmax * 0.02
    if not np.isfinite(linthresh) or linthresh <= 0 or linthresh >= vmax:
        linthresh = vmax * 0.02
    return SymLogNorm(linthresh=linthresh, linscale=0.8, vmin=-vmax, vmax=vmax, base=10)


def tilt_fraction_norm(values: np.ndarray) -> tuple[Normalize, float]:
    finite = values[np.isfinite(values)]
    if finite.size:
        robust = float(np.nanpercentile(np.abs(finite), 98))
        limit = min(2.0, robust) if np.isfinite(robust) and robust > 0 else 1.0
    else:
        limit = 1.0
    if not np.isfinite(limit) or limit <= 0:
        limit = 1.0
    limit = max(float(limit), 0.05)
    return Normalize(vmin=-limit, vmax=limit, clip=True), limit


def plot_tilt_fraction_rz(
    figure_dir: Path,
    polarity: str,
    phase_name: str,
    part: pd.DataFrame,
) -> None:
    data = pivot_values(part, "tilt_fraction")
    values = data.to_numpy(dtype="f8")
    finite = values[np.isfinite(values)]
    norm, limit = tilt_fraction_norm(values)
    cmap = plt.get_cmap("RdBu_r")

    fig, ax = plt.subplots(figsize=(6.0, 4.8), dpi=150)
    mesh = ax.pcolormesh(data.columns.to_numpy(dtype="f8"), data.index.to_numpy(dtype="f8"), values, shading="auto", cmap=cmap, norm=norm)
    ax.invert_yaxis()
    ax.set_xlabel("r/R")
    ax.set_ylabel("depth m")
    ax.set_title(f"{polarity} {phase_name}: Fz tilt fraction")
    cbar = fig.colorbar(mesh, ax=ax, pad=0.02)
    cbar.set_label("Fz tilt correction / Fz tilted")

    if finite.size:
        median = float(np.nanmedian(finite))
        p10 = float(np.nanpercentile(finite, 10))
        p90 = float(np.nanpercentile(finite, 90))
        text = f"median={median:.3g}\nP10/P90={p10:.3g}/{p90:.3g}\nvalid bins={finite.size}\ncolor clipped to +/-{limit:.3g}"
    else:
        text = "no finite tilt_fraction bins"
    ax.text(
        0.02,
        0.02,
        text,
        transform=ax.transAxes,
        va="bottom",
        ha="left",
        fontsize=8.5,
        bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "0.7", "linewidth": 0.6},
    )
    fig.tight_layout()
    fig.savefig(figure_dir / f"{polarity}_{phase_name}_tilt_fraction_rz.png", bbox_inches="tight")
    plt.close(fig)


def plot_tilt_fraction_tilted_structure_3d(
    figure_dir: Path,
    polarity: str,
    phase_name: str,
    part: pd.DataFrame,
    axis_x_over_r: np.ndarray,
    axis_y_over_r: np.ndarray,
    *,
    stride: int,
) -> None:
    data = pivot_values(part, "tilt_fraction")
    depth = data.index.to_numpy(dtype="f8")
    r = data.columns.to_numpy(dtype="f8")
    values = data.to_numpy(dtype="f8")
    norm, limit = tilt_fraction_norm(values)
    cmap = plt.get_cmap("RdBu_r")
    step = max(int(stride), 1)
    depth_slice = slice(None, None, step)
    r_slice = slice(None, None, step)
    x_axis = axis_x_over_r[depth_slice]
    y_axis = axis_y_over_r[depth_slice]
    r_sample = r[r_slice]
    sampled = values[depth_slice, r_slice]
    finite = np.isfinite(sampled)
    x = x_axis[:, None] + r_sample[None, :]
    y = y_axis[:, None] + np.zeros((len(x_axis), len(r_sample)), dtype="f8")
    z = -depth[depth_slice][:, None] + np.zeros((len(x_axis), len(r_sample)), dtype="f8")

    colors = cmap(norm(np.where(finite, sampled, 0.0)))
    colors[~finite] = (0.72, 0.72, 0.72, 0.18)

    fig = plt.figure(figsize=(7.2, 5.6), dpi=150)
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(x, y, z, facecolors=colors, rstride=1, cstride=1, linewidth=0, antialiased=False, shade=False, alpha=0.95)
    ax.plot(axis_x_over_r, axis_y_over_r, -depth, color="black", linewidth=2.2)
    ax.scatter([axis_x_over_r[0]], [axis_y_over_r[0]], [-depth[0]], color="black", s=26, label="surface axis")
    ax.scatter([axis_x_over_r[-1]], [axis_y_over_r[-1]], [-depth[-1]], color="#d62728", s=38, label="deep axis end")
    ax.set_title(f"{polarity} {phase_name}: tilt fraction on tilted axis")
    ax.set_xlabel("x/R")
    ax.set_ylabel("y/R")
    ax.set_zlabel("-depth m")
    ax.view_init(elev=24, azim=-58)
    ax.set_box_aspect((1.45, 0.45, 1.0))
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.72, pad=0.05)
    cbar.set_label(f"Fz tilt / Fz tilted (clipped +/-{limit:.3g})")
    fig.savefig(figure_dir / f"{polarity}_{phase_name}_tilt_fraction_tilted_structure_3d.png", bbox_inches="tight")
    plt.close(fig)


def plot_fz_tilted_structure_png(
    figure_dir: Path,
    polarity: str,
    phase_name: str,
    part: pd.DataFrame,
    axis_x_over_r: np.ndarray,
    axis_y_over_r: np.ndarray,
    *,
    stride: int,
) -> None:
    pivots = [part.pivot(index="depth_m", columns="r_over_R", values=name).sort_index() for name, _ in FZ_COMPONENTS]
    depth = pivots[0].index.to_numpy(dtype="f8")
    r = pivots[0].columns.to_numpy(dtype="f8")
    matrices = [pivot.to_numpy(dtype="f8") for pivot in pivots]
    norm = component_norm(matrices)
    cmap = plt.get_cmap("RdBu_r")
    step = max(int(stride), 1)
    depth_slice = slice(None, None, step)
    r_slice = slice(None, None, step)
    x_axis = axis_x_over_r[depth_slice]
    y_axis = axis_y_over_r[depth_slice]
    x = x_axis[:, None] + r[r_slice][None, :]
    y = y_axis[:, None] + np.zeros((len(x_axis), len(r[r_slice])), dtype="f8")
    z = -depth[depth_slice][:, None] + np.zeros((len(x_axis), len(r[r_slice])), dtype="f8")

    fig = plt.figure(figsize=(15, 5.2), dpi=150)
    for index, ((_, title), values) in enumerate(zip(FZ_COMPONENTS, matrices), start=1):
        ax = fig.add_subplot(1, 3, index, projection="3d")
        sampled = values[depth_slice, r_slice]
        colors = cmap(norm(sampled))
        ax.plot_surface(x, y, z, facecolors=colors, rstride=1, cstride=1, linewidth=0, antialiased=False, shade=False, alpha=0.94)
        ax.plot(axis_x_over_r, axis_y_over_r, -depth, color="black", linewidth=2.0)
        ax.scatter([axis_x_over_r[0]], [axis_y_over_r[0]], [-depth[0]], color="black", s=18)
        ax.scatter([axis_x_over_r[-1]], [axis_y_over_r[-1]], [-depth[-1]], color="#d62728", s=30)
        ax.set_title(title)
        ax.set_xlabel("x/R")
        ax.set_ylabel("y/R")
        ax.set_zlabel("-depth m")
        ax.view_init(elev=24, azim=-58)
        ax.set_box_aspect((1.4, 0.45, 1.0))
    fig.suptitle(f"{polarity} {phase_name}: Fz components on east-aligned tilted vortex axis")
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=fig.axes, shrink=0.72, pad=0.03)
    cbar.set_label("Fz component")
    fig.savefig(figure_dir / f"{polarity}_{phase_name}_Fz_components_tilted_structure_3d.png", bbox_inches="tight")
    plt.close(fig)


def plot_fz_tilted_structure_html(
    figure_dir: Path,
    polarity: str,
    phase_name: str,
    part: pd.DataFrame,
    axis_x_over_r: np.ndarray,
    axis_y_over_r: np.ndarray,
    *,
    stride: int,
) -> None:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    pivots = [part.pivot(index="depth_m", columns="r_over_R", values=name).sort_index() for name, _ in FZ_COMPONENTS]
    depth = pivots[0].index.to_numpy(dtype="f8")
    r = pivots[0].columns.to_numpy(dtype="f8")
    matrices = [pivot.to_numpy(dtype="f8") for pivot in pivots]
    finite_abs = np.concatenate([np.abs(values[np.isfinite(values)]).ravel() for values in matrices if np.isfinite(values).any()])
    vmax = float(np.nanpercentile(finite_abs, 99)) if finite_abs.size else 1.0
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0
    step = max(int(stride), 1)
    depth_slice = slice(None, None, step)
    r_slice = slice(None, None, step)
    x_axis = axis_x_over_r[depth_slice]
    y_axis = axis_y_over_r[depth_slice]
    x = x_axis[:, None] + r[r_slice][None, :]
    y = y_axis[:, None] + np.zeros((len(x_axis), len(r[r_slice])), dtype="f8")
    z = -depth[depth_slice][:, None] + np.zeros((len(x_axis), len(r[r_slice])), dtype="f8")

    fig = make_subplots(
        rows=1,
        cols=3,
        specs=[[{"type": "surface"}, {"type": "surface"}, {"type": "surface"}]],
        subplot_titles=[title for _, title in FZ_COMPONENTS],
    )
    for col, ((_, title), values) in enumerate(zip(FZ_COMPONENTS, matrices), start=1):
        fig.add_trace(
            go.Surface(
                x=x,
                y=y,
                z=z,
                surfacecolor=values[depth_slice, r_slice],
                colorscale="RdBu_r",
                cmin=-vmax,
                cmax=vmax,
                showscale=col == 3,
                colorbar={"title": "Fz"} if col == 3 else None,
                name=title,
            ),
            row=1,
            col=col,
        )
        fig.add_trace(
            go.Scatter3d(x=axis_x_over_r, y=axis_y_over_r, z=-depth, mode="lines", line={"color": "black", "width": 6}, name="tilted axis", showlegend=col == 1),
            row=1,
            col=col,
        )
        fig.add_trace(
            go.Scatter3d(
                x=[axis_x_over_r[-1]],
                y=[axis_y_over_r[-1]],
                z=[-depth[-1]],
                mode="markers",
                marker={"color": "red", "size": 5},
                name="deep axis end",
                showlegend=col == 1,
            ),
            row=1,
            col=col,
        )
    fig.update_layout(
        title=f"{polarity} {phase_name}: Fz components on east-aligned tilted vortex axis",
        height=620,
        width=1500,
        scene={"xaxis_title": "x/R", "yaxis_title": "y/R", "zaxis_title": "-depth m", "aspectmode": "data"},
        scene2={"xaxis_title": "x/R", "yaxis_title": "y/R", "zaxis_title": "-depth m", "aspectmode": "data"},
        scene3={"xaxis_title": "x/R", "yaxis_title": "y/R", "zaxis_title": "-depth m", "aspectmode": "data"},
    )
    fig.write_html(figure_dir / f"{polarity}_{phase_name}_Fz_components_tilted_structure_3d.html", include_plotlyjs="cdn")


def plot_tilted_structure_outputs(
    fig_dir: Path,
    profiles: pd.DataFrame,
    axis_dir: Path,
    radii: dict[str, float],
    *,
    make_png: bool,
    make_html: bool,
    stride: int,
) -> None:
    if not make_png and not make_html:
        return
    tilted = profiles[profiles["axis_mode"] == "tilted"]
    axis_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for (polarity, phase_name), part in tilted.groupby(["polarity", "phase_name"], sort=True):
        radius = radii.get(str(polarity))
        if radius is None:
            continue
        depth = np.sort(part["depth_m"].unique().astype("f8"))
        if str(polarity) not in axis_cache:
            axis = fit_pooled_axis(axis_dir, str(polarity))
            axis_x_m, axis_y_m = axis_xy_m(axis, depth)
            axis_cache[str(polarity)] = (axis_x_m / radius, axis_y_m / radius)
        axis_x_over_r, axis_y_over_r = axis_cache[str(polarity)]
        if make_png:
            plot_fz_tilted_structure_png(fig_dir, str(polarity), str(phase_name), part, axis_x_over_r, axis_y_over_r, stride=stride)
            plot_tilt_fraction_rz(fig_dir, str(polarity), str(phase_name), part)
            plot_tilt_fraction_tilted_structure_3d(fig_dir, str(polarity), str(phase_name), part, axis_x_over_r, axis_y_over_r, stride=stride)
        if make_html:
            plot_fz_tilted_structure_html(fig_dir, str(polarity), str(phase_name), part, axis_x_over_r, axis_y_over_r, stride=stride)


def pivot_values(part: pd.DataFrame, name: str) -> pd.DataFrame:
    return part.pivot(index="depth_m", columns="r_over_R", values=name).sort_index()


def plot_profile_panels(
    path: Path,
    part: pd.DataFrame,
    fields: tuple[tuple[str, str, str], ...],
    *,
    title: str,
    mask_name: str | None = None,
    annotate_valid: bool = False,
) -> None:
    fig, axes = plt.subplots(1, len(fields), figsize=(4.2 * len(fields), 4.6), dpi=145)
    if len(fields) == 1:
        axes = [axes]
    for ax, (name, label, cmap) in zip(axes, fields):
        data = pivot_values(part, name)
        values = data.to_numpy(dtype="f8")
        mask = None
        if mask_name and mask_name in part.columns:
            mask_data = pivot_values(part, mask_name).reindex(index=data.index, columns=data.columns)
            mask = mask_data.to_numpy(dtype=bool)
        finite = values[np.isfinite(values)]
        kwargs = {}
        if finite.size and np.nanmin(finite) < 0 < np.nanmax(finite):
            vmax = np.nanpercentile(np.abs(finite), 98)
            if np.isfinite(vmax) and vmax > 0:
                nonzero = np.abs(finite[np.abs(finite) > 0])
                linthresh = np.nanpercentile(nonzero, 35) if nonzero.size else vmax * 0.02
                if not np.isfinite(linthresh) or linthresh <= 0 or linthresh >= vmax:
                    linthresh = vmax * 0.02
                kwargs.update({"norm": SymLogNorm(linthresh=linthresh, linscale=0.8, vmin=-vmax, vmax=vmax, base=10)})
        mesh = ax.pcolormesh(data.columns.to_numpy(dtype="f8"), data.index.to_numpy(dtype="f8"), values, shading="auto", cmap=cmap, **kwargs)
        if mask is not None:
            ax.contour(data.columns.to_numpy(dtype="f8"), data.index.to_numpy(dtype="f8"), mask.astype(float), levels=[0.5], colors="black", linewidths=0.65)
            invalid = np.ma.masked_where(mask, np.ones_like(mask, dtype="f8"))
            ax.pcolor(data.columns.to_numpy(dtype="f8"), data.index.to_numpy(dtype="f8"), invalid, hatch="///", alpha=0.0, shading="auto")
        ax.invert_yaxis()
        ax.set_xlabel("r/R")
        label_text = label
        if annotate_valid:
            valid_fraction = float(np.mean(np.isfinite(values)))
            label_text = f"{label}\nvalid={valid_fraction:.1%}"
        ax.set_title(label_text)
        fig.colorbar(mesh, ax=ax, shrink=0.82)
    axes[0].set_ylabel("depth m")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def build_mechanism_tables(profiles: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    mechanism = profiles[
        [
            "axis_mode",
            "shape_class",
            "polarity",
            "phase_index",
            "phase_name",
            "tau_center",
            "depth_index",
            "depth_m",
            "r_over_R",
            "F_z_tilted",
            "F_z_ordinary",
            "F_z_tilt_correction",
            "tilt_fraction",
            "count",
            "n_objects",
            "n_dates",
            "axis_slope_x",
            "axis_slope_y",
            "axis_slope_mag",
            "N2",
            "stratification_factor",
            "raw_tilt_flux",
            "raw_tilt_flux_x",
            "raw_tilt_flux_y",
            "F_z_tilt_correction_const_N2",
            "tilt_projection_mean",
            "tilt_projection_rms",
            "un_prime_rms",
            "un_tilt_projection_cov",
        ]
    ].copy()
    tilted = mechanism[mechanism["axis_mode"] == "tilted"].copy()
    metric_rows = []
    n2_rows = []
    alpha_rows = []
    for (polarity, phase_name), part in tilted.groupby(["polarity", "phase_name"], sort=True):
        raw = part["raw_tilt_flux"].to_numpy(dtype="f8")
        fzc = part["F_z_tilt_correction"].to_numpy(dtype="f8")
        coef = part["stratification_factor"].to_numpy(dtype="f8")
        x = part["raw_tilt_flux_x"].to_numpy(dtype="f8")
        y = part["raw_tilt_flux_y"].to_numpy(dtype="f8")
        const = part["F_z_tilt_correction_const_N2"].to_numpy(dtype="f8")
        metric_rows.append(
            {
                "polarity": polarity,
                "phase_name": phase_name,
                "tau_center": TAU_CENTERS[phase_name],
                "raw_identity_relative_residual": rms(fzc - coef * raw) / rms(fzc) if np.isfinite(rms(fzc)) and rms(fzc) > 0 else np.nan,
                "xy_split_relative_residual": rms(raw - x - y) / rms(raw) if np.isfinite(rms(raw)) and rms(raw) > 0 else np.nan,
                "rms_raw_tilt_flux": rms(raw),
                "rms_Fz_tilt_correction": rms(fzc),
                "rms_const_N2_Fz_tilt_correction": rms(const),
                "median_axis_slope_mag": float(np.nanmedian(part["axis_slope_mag"])),
                "median_N2": float(np.nanmedian(part["N2"])),
                "median_stratification_factor": float(np.nanmedian(part["stratification_factor"])),
                "median_tilt_fraction": float(np.nanmedian(part["tilt_fraction"])),
                "n_objects": int(part["n_objects"].max()),
                "n_dates": int(part["n_dates"].max()),
            }
        )
        for depth_m, depth_part in part.groupby("depth_m", sort=True):
            n2_rows.append(
                {
                    "polarity": polarity,
                    "phase_name": phase_name,
                    "tau_center": TAU_CENTERS[phase_name],
                    "depth_m": float(depth_m),
                    "N2": float(np.nanmedian(depth_part["N2"])),
                    "stratification_factor": float(np.nanmedian(depth_part["stratification_factor"])),
                    "rms_raw_tilt_flux": rms(depth_part["raw_tilt_flux"].to_numpy(dtype="f8")),
                    "rms_Fz_tilt_correction": rms(depth_part["F_z_tilt_correction"].to_numpy(dtype="f8")),
                    "rms_const_N2_Fz_tilt_correction": rms(depth_part["F_z_tilt_correction_const_N2"].to_numpy(dtype="f8")),
                }
            )
        for alpha in (0.0, 0.25, 0.5, 0.75, 1.0):
            scaled = alpha * fzc
            alpha_rows.append(
                {
                    "polarity": polarity,
                    "phase_name": phase_name,
                    "tau_center": TAU_CENTERS[phase_name],
                    "axis_scale": alpha,
                    "rms_scaled_Fz_tilt_correction": rms(scaled),
                    "median_scaled_tilt_fraction": float(np.nanmedian(alpha * part["tilt_fraction"].to_numpy(dtype="f8"))),
                }
            )
    return mechanism, pd.DataFrame(metric_rows), pd.DataFrame(n2_rows), pd.DataFrame(alpha_rows)


def plot_mechanism_outputs(mechanism_dir: Path, mechanism: pd.DataFrame, alpha: pd.DataFrame) -> None:
    fig_dir = mechanism_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tilted = mechanism[mechanism["axis_mode"] == "tilted"]
    for (polarity, phase_name), part in tilted.groupby(["polarity", "phase_name"], sort=True):
        prefix = f"{polarity}_{phase_name}"
        title_prefix = f"{polarity} {phase_name}"
        plot_profile_panels(
            fig_dir / f"{prefix}_tilt_term_factor_chain.png",
            part,
            (
                ("axis_slope_mag", "|Rc,z|", "viridis"),
                ("raw_tilt_flux", "raw tilt flux", "RdBu_r"),
                ("stratification_factor", "f0^2/N^2", "viridis"),
                ("F_z_tilt_correction", "Fz tilt correction", "RdBu_r"),
                ("tilt_fraction", "tilt fraction", "RdBu_r"),
            ),
            title=f"{title_prefix}: tilt term factor chain",
        )
        plot_profile_panels(
            fig_dir / f"{prefix}_tilt_projection_product.png",
            part,
            (
                ("tilt_projection_mean", "mean Rc,z dot grad psi", "RdBu_r"),
                ("tilt_projection_rms", "RMS Rc,z dot grad psi", "magma"),
                ("un_prime_rms", "RMS u_n prime", "magma"),
                ("un_tilt_projection_cov", "mean u_n prime projection", "RdBu_r"),
            ),
            title=f"{title_prefix}: projection product",
        )
        plot_profile_panels(
            fig_dir / f"{prefix}_n2_weighting_effect.png",
            part,
            (
                ("raw_tilt_flux", "raw tilt flux", "RdBu_r"),
                ("F_z_tilt_correction_const_N2", "constant N2 weighted", "RdBu_r"),
                ("F_z_tilt_correction", "real N2 weighted", "RdBu_r"),
                ("stratification_factor", "f0^2/N^2", "viridis"),
            ),
            title=f"{title_prefix}: N2 weighting effect",
        )
        plot_profile_panels(
            fig_dir / f"{prefix}_xy_component_split.png",
            part,
            (
                ("raw_tilt_flux_x", "x tilt contribution", "RdBu_r"),
                ("raw_tilt_flux_y", "y tilt contribution", "RdBu_r"),
                ("raw_tilt_flux", "x+y raw tilt flux", "RdBu_r"),
            ),
            title=f"{title_prefix}: x/y tilt contribution split",
        )
        alpha_part = alpha[(alpha["polarity"] == polarity) & (alpha["phase_name"] == phase_name)].sort_values("axis_scale")
        fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.8), dpi=150)
        axes[0].plot(alpha_part["axis_scale"], alpha_part["rms_scaled_Fz_tilt_correction"], marker="o")
        axes[0].set_xlabel("axis scale alpha")
        axes[0].set_ylabel("RMS scaled Fz tilt")
        axes[0].grid(True, color="0.9")
        axes[1].plot(alpha_part["axis_scale"], alpha_part["median_scaled_tilt_fraction"], marker="o", color="#d62728")
        axes[1].set_xlabel("axis scale alpha")
        axes[1].set_ylabel("median scaled tilt fraction")
        axes[1].grid(True, color="0.9")
        fig.suptitle(f"{title_prefix}: axis offset sensitivity")
        fig.tight_layout()
        fig.savefig(fig_dir / f"{prefix}_axis_offset_sensitivity.png")
        plt.close(fig)


def write_mechanism_summary(mechanism_dir: Path, metrics: pd.DataFrame, n2: pd.DataFrame, alpha: pd.DataFrame) -> None:
    lines = [
        "# Fz tilt term mechanism diagnostics",
        "",
        "These diagnostics decompose F_z^(tilt) into axis slope, horizontal streamfunction-gradient projection, u_n prime covariance, and N2 weighting.",
        "",
        "## Mechanism Metrics",
        "```csv",
        metrics.to_csv(index=False).strip(),
        "```",
        "",
        "## N2 Weighting Diagnostics",
        "```csv",
        n2.groupby(["polarity", "phase_name"], sort=True)
        .agg(
            median_N2=("N2", "median"),
            median_stratification_factor=("stratification_factor", "median"),
            max_rms_raw_tilt_flux=("rms_raw_tilt_flux", "max"),
            max_rms_Fz_tilt_correction=("rms_Fz_tilt_correction", "max"),
        )
        .reset_index()
        .to_csv(index=False)
        .strip(),
        "```",
        "",
        "## Axis Offset Sensitivity",
        "```csv",
        alpha.to_csv(index=False).strip(),
        "```",
    ]
    (mechanism_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_mechanism_outputs(output_dir: Path, profiles: pd.DataFrame) -> pd.DataFrame:
    mechanism_dir = output_dir / "tilt_term_mechanism"
    mechanism_dir.mkdir(parents=True, exist_ok=True)
    mechanism, metrics, n2, alpha = build_mechanism_tables(profiles)
    mechanism.to_parquet(mechanism_dir / "tilt_term_mechanism_profiles.parquet", index=False)
    mechanism.to_csv(mechanism_dir / "tilt_term_mechanism_profiles.csv", index=False)
    metrics.to_csv(mechanism_dir / "tilt_term_mechanism_metrics.csv", index=False)
    n2.to_csv(mechanism_dir / "n2_weighting_diagnostics.csv", index=False)
    alpha.to_csv(mechanism_dir / "axis_offset_sensitivity.csv", index=False)
    plot_mechanism_outputs(mechanism_dir, mechanism, alpha)
    write_mechanism_summary(mechanism_dir, metrics, n2, alpha)
    return metrics


def build_total_flux_tables(profiles: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    columns = [
        "axis_mode",
        "shape_class",
        "polarity",
        "phase_index",
        "phase_name",
        "tau_center",
        "depth_index",
        "depth_m",
        "r_over_R",
        "Q_n",
        "Q_n_valid",
        "A_T",
        "partial_tau_A_T",
        "partial_tau_A_T_nd",
        "F_n",
        "F_z_tilted",
        "divF",
        "divF_nd",
        "Unbar",
        "A_T_advective_flux_n",
        "J_n",
        "J_z",
        "divJ",
        "divJ_nd",
        "wave_activity_residual_F_nd",
        "wave_activity_residual_J_nd",
        "count",
        "n_objects",
        "n_dates",
    ]
    available = [name for name in columns if name in profiles.columns]
    total = profiles[available].copy()
    tilted = total[total["axis_mode"] == "tilted"].copy()
    metric_rows = []
    residual_rows = []
    for (polarity, phase_name), part in tilted.groupby(["polarity", "phase_name"], sort=True):
        valid = part["Q_n_valid"].to_numpy(dtype=bool) & np.isfinite(part["A_T"].to_numpy(dtype="f8"))
        valid &= np.isfinite(part["wave_activity_residual_F_nd"].to_numpy(dtype="f8"))
        valid &= np.isfinite(part["wave_activity_residual_J_nd"].to_numpy(dtype="f8"))
        residual_f = part.loc[valid, "wave_activity_residual_F_nd"].to_numpy(dtype="f8")
        residual_j = part.loc[valid, "wave_activity_residual_J_nd"].to_numpy(dtype="f8")
        residual_f_rms = rms(residual_f)
        residual_j_rms = rms(residual_j)
        improvement = 1.0 - residual_j_rms / residual_f_rms if np.isfinite(residual_f_rms) and residual_f_rms > 0 and np.isfinite(residual_j_rms) else np.nan
        da = part.loc[valid, "partial_tau_A_T_nd"].to_numpy(dtype="f8")
        divf = part.loc[valid, "divF_nd"].to_numpy(dtype="f8")
        divj = part.loc[valid, "divJ_nd"].to_numpy(dtype="f8")
        metric_rows.append(
            {
                "polarity": polarity,
                "phase_name": phase_name,
                "tau_center": TAU_CENTERS[phase_name],
                "residual_F_rms": residual_f_rms,
                "residual_J_rms": residual_j_rms,
                "improvement_fraction": improvement,
                "corr_partial_tau_A_T_minus_divF": metric_corr(da, -divf),
                "corr_partial_tau_A_T_minus_divJ": metric_corr(da, -divj),
                "valid_bins": int(np.sum(valid)),
                "Q_n_valid_fraction": float(np.mean(part["Q_n_valid"].to_numpy(dtype=bool))),
                "rms_F_n": rms(part["F_n"].to_numpy(dtype="f8")),
                "rms_advective_flux_n": rms(part["A_T_advective_flux_n"].to_numpy(dtype="f8")),
                "rms_J_n": rms(part["J_n"].to_numpy(dtype="f8")),
                "n_objects": int(part["n_objects"].max()) if "n_objects" in part else 0,
                "n_dates": int(part["n_dates"].max()) if "n_dates" in part else 0,
            }
        )
        residual_rows.append(
            {
                "polarity": polarity,
                "phase_name": phase_name,
                "tau_center": TAU_CENTERS[phase_name],
                "residual_F_rms": residual_f_rms,
                "residual_J_rms": residual_j_rms,
                "improvement_fraction": improvement,
                "mean_residual_F_nd": float(np.nanmean(residual_f)) if residual_f.size else np.nan,
                "mean_residual_J_nd": float(np.nanmean(residual_j)) if residual_j.size else np.nan,
                "valid_bins": int(np.sum(valid)),
            }
        )
    return total, pd.DataFrame(metric_rows), pd.DataFrame(residual_rows)


def plot_jt_validity_mask(fig_dir: Path, prefix: str, title_prefix: str, part: pd.DataFrame) -> None:
    masks = []
    for name, label in (
        ("Q_n_valid", "Q_n valid"),
        ("A_T", "A_T finite"),
        ("J_n", "J_n finite"),
    ):
        data = pivot_values(part, name)
        values = data.to_numpy()
        if name != "Q_n_valid":
            values = np.isfinite(values)
        else:
            values = values.astype(bool)
        masks.append((data, values.astype(float), label))
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.0), dpi=145)
    for ax, (data, values, label) in zip(axes, masks):
        mesh = ax.pcolormesh(data.columns.to_numpy(dtype="f8"), data.index.to_numpy(dtype="f8"), values, shading="auto", cmap="Greys", vmin=0, vmax=1)
        ax.contour(data.columns.to_numpy(dtype="f8"), data.index.to_numpy(dtype="f8"), values, levels=[0.5], colors="#d62728", linewidths=0.8)
        ax.invert_yaxis()
        ax.set_xlabel("r/R")
        ax.set_title(f"{label}\nvalid={np.mean(values > 0.5):.1%}")
        fig.colorbar(mesh, ax=ax, shrink=0.82)
    axes[0].set_ylabel("depth m")
    fig.suptitle(f"{title_prefix}: J_T validity masks")
    fig.tight_layout()
    fig.savefig(fig_dir / f"{prefix}_JT_validity_mask.png")
    plt.close(fig)


def plot_jt_validity_fraction(fig_dir: Path, total: pd.DataFrame) -> None:
    tilted = total[total["axis_mode"] == "tilted"]
    rows = []
    for (polarity, phase_name), part in tilted.groupby(["polarity", "phase_name"], sort=True):
        rows.append(
            {
                "polarity": polarity,
                "phase_name": phase_name,
                "tau_center": TAU_CENTERS[phase_name],
                "Q_n_valid_fraction": float(np.mean(part["Q_n_valid"].to_numpy(dtype=bool))),
                "J_n_valid_fraction": float(np.mean(np.isfinite(part["J_n"].to_numpy(dtype="f8")))),
            }
        )
    table = pd.DataFrame(rows)
    if table.empty:
        return
    fig, ax = plt.subplots(figsize=(8.2, 4.8), dpi=150)
    for polarity, part in table.groupby("polarity", sort=True):
        part = part.sort_values("tau_center")
        ax.plot(part["tau_center"], part["Q_n_valid_fraction"], marker="o", label=f"{polarity} Q_n")
        ax.plot(part["tau_center"], part["J_n_valid_fraction"], marker="s", linestyle="--", label=f"{polarity} J_n")
    ax.set_xlabel("tau")
    ax.set_ylabel("valid fraction")
    ax.set_title("J_T validity fraction by lifecycle phase")
    ax.grid(True, color="0.9")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "JT_validity_fraction_by_phase.png")
    plt.close(fig)


def tilted_axis_for_part(axis_dir: Path, radii: dict[str, float], polarity: str, depth: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    radius = radii.get(str(polarity), np.nan)
    axis = fit_pooled_axis(axis_dir, str(polarity))
    axis_x_m, axis_y_m = axis_xy_m(axis, depth)
    return axis_x_m / radius, axis_y_m / radius


def plot_jt_3d_png(
    fig_dir: Path,
    prefix: str,
    title_prefix: str,
    part: pd.DataFrame,
    axis_x: np.ndarray,
    axis_y: np.ndarray,
    *,
    value_name: str,
    filename_suffix: str,
    stride: int,
) -> None:
    value_data = pivot_values(part, value_name)
    valid_data = pivot_values(part, "Q_n_valid").reindex(index=value_data.index, columns=value_data.columns)
    depth = value_data.index.to_numpy(dtype="f8")
    radial = value_data.columns.to_numpy(dtype="f8")
    values = value_data.to_numpy(dtype="f8")
    valid = valid_data.to_numpy(dtype=bool)
    step = max(int(stride), 1)
    ds = slice(None, None, step)
    rs = slice(None, None, step)
    x = axis_x[ds][:, None] + radial[rs][None, :]
    y = axis_y[ds][:, None] + np.zeros((len(depth[ds]), len(radial[rs])), dtype="f8")
    z = -depth[ds][:, None] + np.zeros((len(depth[ds]), len(radial[rs])), dtype="f8")
    sample_values = values[ds, rs]
    sample_valid = valid[ds, rs]

    fig = plt.figure(figsize=(8.4, 6.0), dpi=150)
    ax = fig.add_subplot(111, projection="3d")
    if value_name == "r_over_R":
        surface_color = radial[rs][None, :] + np.zeros_like(x)
        cmap = "viridis"
        norm = None
        label = "r/R"
    else:
        surface_color = np.where(sample_valid & np.isfinite(sample_values), sample_values, np.nan)
        cmap = "RdBu_r"
        finite = surface_color[np.isfinite(surface_color)]
        vmax = np.nanpercentile(np.abs(finite), 98) if finite.size else 1.0
        if not np.isfinite(vmax) or vmax <= 0:
            vmax = 1.0
        norm = SymLogNorm(linthresh=max(vmax * 0.02, 1e-12), linscale=0.8, vmin=-vmax, vmax=vmax, base=10)
        label = value_name
    cmap_obj = plt.get_cmap(cmap)
    if norm is None:
        facecolors = cmap_obj((surface_color - np.nanmin(surface_color)) / max(np.nanmax(surface_color) - np.nanmin(surface_color), 1e-12))
    else:
        facecolors = cmap_obj(norm(surface_color))
    if value_name == "r_over_R":
        facecolors[..., 3] = np.where(sample_valid, 0.92, 0.18)
    else:
        facecolors[..., 3] = np.where(sample_valid & np.isfinite(sample_values), 0.92, 0.16)
    ax.plot_surface(x, y, z, facecolors=facecolors, rstride=1, cstride=1, linewidth=0, antialiased=False, shade=False)
    ax.plot(axis_x, axis_y, -depth, color="black", linewidth=2.0)
    ax.scatter([axis_x[-1]], [axis_y[-1]], [-depth[-1]], color="#d62728", s=36)
    ax.set_xlabel("x/R")
    ax.set_ylabel("y/R")
    ax.set_zlabel("-depth m")
    ax.set_title(f"{title_prefix}: {label} on tilted r/R curtain")
    ax.view_init(elev=24, azim=-58)
    sm = ScalarMappable(norm=norm, cmap=cmap_obj)
    sm.set_array(surface_color[np.isfinite(surface_color)])
    fig.colorbar(sm, ax=ax, shrink=0.72, pad=0.08, label=label)
    fig.tight_layout()
    fig.savefig(fig_dir / f"{prefix}_{filename_suffix}.png")
    plt.close(fig)


def plot_jt_3d_html(
    fig_dir: Path,
    prefix: str,
    title_prefix: str,
    part: pd.DataFrame,
    axis_x: np.ndarray,
    axis_y: np.ndarray,
    *,
    value_name: str,
    filename_suffix: str,
    stride: int,
) -> None:
    import plotly.graph_objects as go

    value_data = pivot_values(part, value_name)
    valid_data = pivot_values(part, "Q_n_valid").reindex(index=value_data.index, columns=value_data.columns)
    depth = value_data.index.to_numpy(dtype="f8")
    radial = value_data.columns.to_numpy(dtype="f8")
    values = value_data.to_numpy(dtype="f8")
    valid = valid_data.to_numpy(dtype=bool)
    step = max(int(stride), 1)
    ds = slice(None, None, step)
    rs = slice(None, None, step)
    x = axis_x[ds][:, None] + radial[rs][None, :]
    y = axis_y[ds][:, None] + np.zeros((len(depth[ds]), len(radial[rs])), dtype="f8")
    z = -depth[ds][:, None] + np.zeros((len(depth[ds]), len(radial[rs])), dtype="f8")
    if value_name == "r_over_R":
        surface_color = radial[rs][None, :] + np.zeros_like(x)
        title = "r/R"
    else:
        surface_color = np.where(valid[ds, rs] & np.isfinite(values[ds, rs]), values[ds, rs], np.nan)
        title = value_name
    finite = surface_color[np.isfinite(surface_color)]
    cmax = float(np.nanpercentile(np.abs(finite), 98)) if finite.size and value_name != "r_over_R" else float(np.nanmax(finite)) if finite.size else 1.0
    cmin = -cmax if value_name != "r_over_R" else float(np.nanmin(finite)) if finite.size else 0.0
    fig = go.Figure()
    fig.add_trace(
        go.Surface(
            x=x,
            y=y,
            z=z,
            surfacecolor=surface_color,
            colorscale="RdBu" if value_name != "r_over_R" else "Viridis",
            cmin=cmin,
            cmax=cmax,
            colorbar={"title": title},
            opacity=0.9,
            name=title,
        )
    )
    fig.add_trace(go.Scatter3d(x=axis_x, y=axis_y, z=-depth, mode="lines", line={"color": "black", "width": 6}, name="tilted axis"))
    fig.add_trace(go.Scatter3d(x=[axis_x[-1]], y=[axis_y[-1]], z=[-depth[-1]], mode="markers", marker={"color": "red", "size": 5}, name="deep axis end"))
    fig.update_layout(
        title=f"{title_prefix}: {title} on tilted r/R curtain",
        scene={"xaxis_title": "x/R", "yaxis_title": "y/R", "zaxis_title": "-depth m", "aspectmode": "data"},
        height=720,
        width=900,
    )
    fig.write_html(fig_dir / f"{prefix}_{filename_suffix}.html", include_plotlyjs="cdn")


def plot_total_flux_outputs(total_dir: Path, total: pd.DataFrame, metrics: pd.DataFrame, axis_dir: Path, radii: dict[str, float], args: argparse.Namespace) -> None:
    fig_dir = total_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tilted = total[total["axis_mode"] == "tilted"]
    for (polarity, phase_name), part in tilted.groupby(["polarity", "phase_name"], sort=True):
        prefix = f"{polarity}_{phase_name}"
        title_prefix = f"{polarity} {phase_name}"
        plot_profile_panels(
            fig_dir / f"{prefix}_JT_components.png",
            part,
            (
                ("F_n", "F_n", "RdBu_r"),
                ("A_T_advective_flux_n", "Unbar A_T", "RdBu_r"),
                ("J_n", "J_n", "RdBu_r"),
                ("J_z", "J_z = F_z", "RdBu_r"),
            ),
            title=f"{title_prefix}: total wave-action flux components",
            mask_name="Q_n_valid",
            annotate_valid=True,
        )
        plot_jt_validity_mask(fig_dir, prefix, title_prefix, part)
        if bool(args.jt_3d_structure):
            depth = np.sort(part["depth_m"].unique().astype("f8"))
            axis_x, axis_y = tilted_axis_for_part(axis_dir, radii, str(polarity), depth)
            plot_jt_3d_png(
                fig_dir,
                prefix,
                title_prefix,
                part,
                axis_x,
                axis_y,
                value_name="r_over_R",
                filename_suffix="r_over_R_tilted_validity_3d",
                stride=int(args.jt_3d_stride),
            )
            plot_jt_3d_png(
                fig_dir,
                prefix,
                title_prefix,
                part,
                axis_x,
                axis_y,
                value_name="J_n",
                filename_suffix="JT_on_tilted_r_over_R_3d",
                stride=int(args.jt_3d_stride),
            )
            if bool(args.jt_3d_html):
                plot_jt_3d_html(
                    fig_dir,
                    prefix,
                    title_prefix,
                    part,
                    axis_x,
                    axis_y,
                    value_name="r_over_R",
                    filename_suffix="r_over_R_tilted_validity_3d",
                    stride=int(args.jt_3d_stride),
                )
                plot_jt_3d_html(
                    fig_dir,
                    prefix,
                    title_prefix,
                    part,
                    axis_x,
                    axis_y,
                    value_name="J_n",
                    filename_suffix="JT_on_tilted_r_over_R_3d",
                    stride=int(args.jt_3d_stride),
                )
        plot_profile_panels(
            fig_dir / f"{prefix}_divF_vs_divJ.png",
            part,
            (
                ("divF", "div F_T", "RdBu_r"),
                ("divJ", "div J_T", "RdBu_r"),
            ),
            title=f"{title_prefix}: E-P flux divergence vs total flux divergence",
        )
        plot_profile_panels(
            fig_dir / f"{prefix}_wave_action_budget_comparison.png",
            part,
            (
                ("partial_tau_A_T_nd", "partial_tau A_T nd", "RdBu_r"),
                ("wave_activity_residual_F_nd", "partial_tau A_T + divF", "RdBu_r"),
                ("wave_activity_residual_J_nd", "partial_tau A_T + divJ", "RdBu_r"),
            ),
            title=f"{title_prefix}: wave-action budget residual comparison",
        )
    if not metrics.empty:
        fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=150)
        for polarity, part in metrics.groupby("polarity", sort=True):
            part = part.sort_values("tau_center")
            ax.plot(part["tau_center"], part["improvement_fraction"], marker="o", label=polarity)
        ax.axhline(0.0, color="0.4", linewidth=1.0)
        ax.set_xlabel("tau")
        ax.set_ylabel("1 - residual_J_rms / residual_F_rms")
        ax.set_title("Wave-action residual improvement by lifecycle phase")
        ax.grid(True, color="0.9")
        ax.legend()
        fig.tight_layout()
        fig.savefig(fig_dir / "wave_action_residual_improvement_by_phase.png")
        plt.close(fig)
    plot_jt_validity_fraction(fig_dir, total)


def build_jt_tilt_context(metrics: pd.DataFrame, tilt_metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if metrics.empty or tilt_metrics.empty:
        return pd.DataFrame(), pd.DataFrame()
    keep = [
        "polarity",
        "phase_name",
        "median_axis_slope_mag",
        "rms_Fz_tilt_correction",
        "median_tilt_fraction",
        "rms_raw_tilt_flux",
        "median_N2",
        "median_stratification_factor",
    ]
    available = [name for name in keep if name in tilt_metrics.columns]
    context = metrics.merge(tilt_metrics[available], on=["polarity", "phase_name"], how="left")
    context["tilt_context_score"] = np.abs(context["median_tilt_fraction"]) * context["Q_n_valid_fraction"]
    context["advective_to_EP_ratio"] = np.divide(
        context["rms_advective_flux_n"],
        context["rms_F_n"],
        out=np.full(len(context), np.nan, dtype="f8"),
        where=np.isfinite(context["rms_F_n"]) & (np.abs(context["rms_F_n"]) > 0),
    )
    context["JT_improved"] = context["improvement_fraction"] > 0
    summary_rows = []
    for polarity, part in context.groupby("polarity", sort=True):
        summary_rows.append(
            {
                "polarity": polarity,
                "n_phase_groups": int(len(part)),
                "n_improved": int(part["JT_improved"].sum()),
                "mean_improvement_fraction": float(np.nanmean(part["improvement_fraction"])),
                "mean_abs_tilt_fraction": float(np.nanmean(np.abs(part["median_tilt_fraction"]))),
                "mean_axis_slope_mag": float(np.nanmean(part["median_axis_slope_mag"])),
                "mean_Q_n_valid_fraction": float(np.nanmean(part["Q_n_valid_fraction"])),
                "corr_improvement_abs_tilt_fraction": metric_corr(part["improvement_fraction"].to_numpy(dtype="f8"), np.abs(part["median_tilt_fraction"].to_numpy(dtype="f8"))),
                "corr_improvement_axis_slope": metric_corr(part["improvement_fraction"].to_numpy(dtype="f8"), part["median_axis_slope_mag"].to_numpy(dtype="f8")),
                "corr_improvement_Q_n_valid_fraction": metric_corr(part["improvement_fraction"].to_numpy(dtype="f8"), part["Q_n_valid_fraction"].to_numpy(dtype="f8")),
            }
        )
    return context, pd.DataFrame(summary_rows)


def plot_jt_tilt_context(total_dir: Path, context: pd.DataFrame) -> None:
    if context.empty:
        return
    fig_dir = total_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plots = (
        ("median_tilt_fraction", "tilt fraction", "JT_improvement_vs_tilt_fraction.png"),
        ("median_axis_slope_mag", "|Rc,z|", "JT_improvement_vs_axis_slope.png"),
        ("Q_n_valid_fraction", "Q_n valid fraction", "JT_improvement_vs_Qn_valid_fraction.png"),
    )
    for x_name, x_label, filename in plots:
        fig, ax = plt.subplots(figsize=(6.5, 4.8), dpi=150)
        for polarity, part in context.groupby("polarity", sort=True):
            ax.scatter(part[x_name], part["improvement_fraction"], label=polarity, s=48)
            for row in part.itertuples(index=False):
                ax.annotate(str(row.phase_name), (getattr(row, x_name), row.improvement_fraction), fontsize=7, xytext=(3, 3), textcoords="offset points")
        ax.axhline(0.0, color="0.4", linewidth=1.0)
        ax.set_xlabel(x_label)
        ax.set_ylabel("J_T improvement fraction")
        ax.set_title(f"J_T improvement vs {x_label}")
        ax.grid(True, color="0.9")
        ax.legend()
        fig.tight_layout()
        fig.savefig(fig_dir / filename)
        plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(8.5, 8.5), dpi=150, sharex=True)
    for polarity, part in context.groupby("polarity", sort=True):
        part = part.sort_values("tau_center")
        axes[0].plot(part["tau_center"], part["improvement_fraction"], marker="o", label=polarity)
        axes[1].plot(part["tau_center"], np.abs(part["median_tilt_fraction"]), marker="o", label=polarity)
        axes[2].plot(part["tau_center"], part["Q_n_valid_fraction"], marker="o", label=polarity)
    axes[0].axhline(0.0, color="0.4", linewidth=1.0)
    axes[0].set_ylabel("J_T improvement")
    axes[1].set_ylabel("|tilt fraction|")
    axes[2].set_ylabel("Q_n valid")
    axes[2].set_xlabel("tau")
    for ax in axes:
        ax.grid(True, color="0.9")
        ax.legend()
    fig.suptitle("J_T improvement with tilt context by phase")
    fig.tight_layout()
    fig.savefig(fig_dir / "JT_tilt_context_by_phase.png")
    plt.close(fig)


def write_total_flux_summary(total_dir: Path, metrics: pd.DataFrame, residuals: pd.DataFrame, context: pd.DataFrame, context_summary: pd.DataFrame) -> None:
    if metrics.empty:
        verdict = "No valid bins were available for total wave-action flux diagnostics."
    else:
        improved = int((metrics["improvement_fraction"] > 0).sum())
        total_count = int(metrics["improvement_fraction"].notna().sum())
        verdict = (
            f"J_T improves the nondimensional wave-action residual in {improved}/{total_count} polarity-phase groups. "
            "If improvement is weak or negative, likely causes include limited Q_n-valid area, five-phase coarse differencing, representativeness averaging, or neglected w_T A_T."
        )
    lines = [
        "# Wave-action total flux diagnostics",
        "",
        "This compares the E-P-only budget partial_tau A_T + div(F_T) with the total-flux budget partial_tau A_T + div(J_T).",
        "Here J_T = (Unbar A_T + F_n, F_z); w_T A_T is not included.",
        "A_T, Unbar*A_T, and J_n are only shown where Q_n_valid=True and A_T is finite; hatched or blank regions mark physically unreliable wave-action bins, not zero signal.",
        "The r/R tilted 3D plots are geometric curtains in east-aligned tilted coordinates, not full non-axisymmetric 3D volume fields.",
        "",
        verdict,
        "",
        "## Metrics",
        "```csv",
        metrics.to_csv(index=False).strip(),
        "```",
        "",
        "## Residual Comparison",
        "```csv",
        residuals.to_csv(index=False).strip(),
        "```",
        "",
        "## JT and Tilt Context",
        "",
        "This section joins the J_T residual improvement with the tilted-axis mechanism diagnostics. Positive improvement with low Q_n_valid_fraction should be treated cautiously. If improvement tracks abs(tilt_fraction) or axis_slope, the total-flux result is likely linked to the tilted structure; otherwise it is more likely dominated by the mean-flow advection term.",
        "",
        "```csv",
        context_summary.to_csv(index=False).strip() if not context_summary.empty else "no_context_available",
        "```",
        "",
        "## JT Tilt Context Metrics",
        "```csv",
        context.to_csv(index=False).strip() if not context.empty else "no_context_available",
        "```",
    ]
    (total_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_total_flux_outputs(output_dir: Path, profiles: pd.DataFrame, tilt_metrics: pd.DataFrame, axis_dir: Path, radii: dict[str, float], args: argparse.Namespace) -> None:
    total_dir = output_dir / "wave_action_total_flux"
    total_dir.mkdir(parents=True, exist_ok=True)
    total, metrics, residuals = build_total_flux_tables(profiles)
    context, context_summary = build_jt_tilt_context(metrics, tilt_metrics)
    total.to_parquet(total_dir / "wave_action_total_flux_profiles.parquet", index=False)
    total.to_csv(total_dir / "wave_action_total_flux_profiles.csv", index=False)
    metrics.to_csv(total_dir / "wave_action_total_flux_metrics.csv", index=False)
    residuals.to_csv(total_dir / "wave_action_residual_comparison.csv", index=False)
    context.to_csv(total_dir / "JT_tilt_context_metrics.csv", index=False)
    context_summary.to_csv(total_dir / "JT_tilt_context_summary.csv", index=False)
    plot_total_flux_outputs(total_dir, total, metrics, axis_dir, radii, args)
    plot_jt_tilt_context(total_dir, context)
    write_total_flux_summary(total_dir, metrics, residuals, context, context_summary)


NONLINEARITY_CATEGORY_LABELS = {
    0: "weak_gradient",
    1: "linear_wave_like",
    2: "finite_amplitude",
    3: "budget_nonclosure",
    4: "finite_and_budget",
    5: "intermediate_valid",
}


def build_nonlinearity_tables(profiles: pd.DataFrame, radii: dict[str, float]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tilted = profiles[profiles["axis_mode"] == "tilted"].copy()
    if tilted.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    tilted["representative_radius_m"] = tilted["polarity"].astype(str).map(radii).astype("f8")
    tilted["q_prime_rms"] = np.sqrt(np.maximum(tilted["q_prime_variance"].to_numpy(dtype="f8"), 0.0))
    tilted["linear_pv_scale"] = np.abs(tilted["Q_n"].to_numpy(dtype="f8")) * tilted["representative_radius_m"].to_numpy(dtype="f8")
    median_qprime_by_polarity = tilted.groupby("polarity")["q_prime_rms"].transform(lambda s: float(np.nanmedian(s.to_numpy(dtype="f8"))))
    median_qprime = median_qprime_by_polarity.to_numpy(dtype="f8")
    safe_qprime = np.maximum(tilted["q_prime_rms"].to_numpy(dtype="f8"), 1e-30)
    safe_linear_scale = np.maximum(tilted["linear_pv_scale"].to_numpy(dtype="f8"), 1e-30)
    tilted["recovery_ratio"] = safe_linear_scale / safe_qprime
    tilted["weak_gradient_score"] = -np.log10(safe_linear_scale / np.maximum(median_qprime, 1e-30))
    tilted["budget_residual_score"] = np.abs(tilted["wave_activity_residual_J_nd"].to_numpy(dtype="f8"))
    qn_valid = tilted["Q_n_valid"].to_numpy(dtype=bool)
    denominator = tilted["linear_pv_scale"].to_numpy(dtype="f8")
    numerator = tilted["q_prime_rms"].to_numpy(dtype="f8")
    nlq = np.divide(numerator, denominator, out=np.full(len(tilted), np.nan, dtype="f8"), where=qn_valid & np.isfinite(denominator) & (denominator > 0))
    tilted["NL_q"] = nlq
    tilted["weak_gradient_or_singular"] = ~qn_valid
    tilted["finite_amplitude_nonlinear"] = qn_valid & np.isfinite(nlq) & (nlq > 1.0)
    residual = tilted["wave_activity_residual_J_nd"].to_numpy(dtype="f8")
    tilted["budget_nonclosure"] = qn_valid & np.isfinite(residual) & (np.abs(residual) > 2.0)
    tilted["linear_wave_like"] = qn_valid & np.isfinite(nlq) & (nlq <= 1.0) & np.isfinite(residual) & (np.abs(residual) <= 1.0)
    tilted["ambiguous_intermediate"] = qn_valid & ~tilted["finite_amplitude_nonlinear"] & ~tilted["budget_nonclosure"] & ~tilted["linear_wave_like"]
    category = np.full(len(tilted), 5, dtype="i4")
    category[tilted["weak_gradient_or_singular"].to_numpy(dtype=bool)] = 0
    category[tilted["linear_wave_like"].to_numpy(dtype=bool)] = 1
    finite = tilted["finite_amplitude_nonlinear"].to_numpy(dtype=bool)
    budget = tilted["budget_nonclosure"].to_numpy(dtype=bool)
    category[finite & ~budget] = 2
    category[~finite & budget] = 3
    category[finite & budget] = 4
    tilted["nonlinearity_category_code"] = category
    tilted["nonlinearity_category"] = [NONLINEARITY_CATEGORY_LABELS[int(code)] for code in category]

    metric_rows = []
    for (polarity, phase_name), part in tilted.groupby(["polarity", "phase_name"], sort=True):
        values = part["NL_q"].to_numpy(dtype="f8")
        finite_values = values[np.isfinite(values)]
        n_total = len(part)
        n_valid = int(part["Q_n_valid"].sum())
        metric_rows.append(
            {
                "polarity": polarity,
                "phase_name": phase_name,
                "tau_center": TAU_CENTERS.get(phase_name, float(part["tau_center"].iloc[0]) if "tau_center" in part else np.nan),
                "total_bins": int(n_total),
                "Q_n_valid_fraction": float(np.mean(part["Q_n_valid"].to_numpy(dtype=bool))),
                "Q_n_invalid_fraction": float(np.mean(part["weak_gradient_or_singular"].to_numpy(dtype=bool))),
                "NL_q_median": float(np.nanmedian(finite_values)) if finite_values.size else np.nan,
                "NL_q_p75": float(np.nanpercentile(finite_values, 75)) if finite_values.size else np.nan,
                "NL_q_p90": float(np.nanpercentile(finite_values, 90)) if finite_values.size else np.nan,
                "fraction_NLq_gt1": float(np.mean(part["finite_amplitude_nonlinear"].to_numpy(dtype=bool))),
                "fraction_NLq_gt1_among_Qn_valid": float(np.sum(part["finite_amplitude_nonlinear"]) / n_valid) if n_valid else np.nan,
                "fraction_budget_nonclosure": float(np.mean(part["budget_nonclosure"].to_numpy(dtype=bool))),
                "fraction_budget_nonclosure_among_Qn_valid": float(np.sum(part["budget_nonclosure"]) / n_valid) if n_valid else np.nan,
                "linear_wave_like_fraction": float(np.mean(part["linear_wave_like"].to_numpy(dtype=bool))),
                "ambiguous_intermediate_fraction": float(np.mean(part["ambiguous_intermediate"].to_numpy(dtype=bool))),
                "strong_nonlinear_or_singular_fraction": float(
                    np.mean(
                        part["weak_gradient_or_singular"].to_numpy(dtype=bool)
                        | part["finite_amplitude_nonlinear"].to_numpy(dtype=bool)
                        | part["budget_nonclosure"].to_numpy(dtype=bool)
                    )
                ),
                "valid_bins": n_valid,
            }
        )
    metrics = pd.DataFrame(metric_rows)

    summary_rows = []
    for polarity, part in tilted.groupby("polarity", sort=True):
        values = part["NL_q"].to_numpy(dtype="f8")
        finite_values = values[np.isfinite(values)]
        n_valid = int(part["Q_n_valid"].sum())
        summary_rows.append(
            {
                "polarity": polarity,
                "total_bins": int(len(part)),
                "Q_n_valid_fraction": float(np.mean(part["Q_n_valid"].to_numpy(dtype=bool))),
                "Q_n_invalid_fraction": float(np.mean(part["weak_gradient_or_singular"].to_numpy(dtype=bool))),
                "NL_q_median": float(np.nanmedian(finite_values)) if finite_values.size else np.nan,
                "NL_q_p90": float(np.nanpercentile(finite_values, 90)) if finite_values.size else np.nan,
                "fraction_NLq_gt1": float(np.mean(part["finite_amplitude_nonlinear"].to_numpy(dtype=bool))),
                "fraction_NLq_gt1_among_Qn_valid": float(np.sum(part["finite_amplitude_nonlinear"]) / n_valid) if n_valid else np.nan,
                "fraction_budget_nonclosure": float(np.mean(part["budget_nonclosure"].to_numpy(dtype=bool))),
                "linear_wave_like_fraction": float(np.mean(part["linear_wave_like"].to_numpy(dtype=bool))),
                "strong_nonlinear_or_singular_fraction": float(
                    np.mean(
                        part["weak_gradient_or_singular"].to_numpy(dtype=bool)
                        | part["finite_amplitude_nonlinear"].to_numpy(dtype=bool)
                        | part["budget_nonclosure"].to_numpy(dtype=bool)
                    )
                ),
            }
        )
    summary = pd.DataFrame(summary_rows)
    columns = [
        "polarity",
        "phase_index",
        "phase_name",
        "tau_center",
        "depth_index",
        "depth_m",
        "r_over_R",
        "representative_radius_m",
        "Q_n",
        "Q_n_valid",
        "q_prime_variance",
        "q_prime_rms",
        "linear_pv_scale",
        "recovery_ratio",
        "weak_gradient_score",
        "budget_residual_score",
        "NL_q",
        "A_T",
        "wave_activity_residual_J_nd",
        "weak_gradient_or_singular",
        "finite_amplitude_nonlinear",
        "budget_nonclosure",
        "linear_wave_like",
        "ambiguous_intermediate",
        "nonlinearity_category_code",
        "nonlinearity_category",
        "n_objects",
        "n_dates",
    ]
    return tilted[[name for name in columns if name in tilted.columns]].copy(), metrics, summary


def plot_nlq_rz(fig_dir: Path, prefix: str, title_prefix: str, part: pd.DataFrame) -> None:
    data = pivot_values(part, "NL_q")
    values = data.to_numpy(dtype="f8")
    plot_values = np.log10(1.0 + values)
    finite = plot_values[np.isfinite(plot_values)]
    vmax = float(np.nanpercentile(finite, 98)) if finite.size else 1.0
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0
    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad("0.82")
    valid = pivot_values(part, "Q_n_valid").reindex(index=data.index, columns=data.columns).to_numpy(dtype=bool)
    fig, axes = plt.subplots(3, 1, figsize=(7.0, 9.2), dpi=150, sharex=True)
    mesh = None
    for ax, (view_title, depth_max) in zip(axes, (("full depth", None), ("upper 100 m", 100.0), ("shallow 20 m", 20.0))):
        mesh = ax.pcolormesh(data.columns.to_numpy(dtype="f8"), data.index.to_numpy(dtype="f8"), plot_values, shading="auto", cmap=cmap, vmin=0.0, vmax=vmax)
        try:
            ax.contour(data.columns.to_numpy(dtype="f8"), data.index.to_numpy(dtype="f8"), valid.astype(float), levels=[0.5], colors="#d62728", linewidths=0.8)
            if np.isfinite(values).any():
                ax.contour(data.columns.to_numpy(dtype="f8"), data.index.to_numpy(dtype="f8"), values, levels=[1.0], colors="cyan", linewidths=0.9)
        except ValueError:
            pass
        ax.invert_yaxis()
        if depth_max is not None:
            ax.set_ylim(depth_max, 0.0)
        ax.set_ylabel("depth m")
        ax.set_title(view_title)
    axes[-1].set_xlabel("r/R")
    fig.suptitle(f"{title_prefix}: NLq = sqrt(q'^2)/(|Qn|R)")
    cbar = fig.colorbar(mesh, ax=axes, pad=0.02, shrink=0.88)
    cbar.set_label("log10(1 + NLq); gray = Qn invalid")
    finite_nlq = values[np.isfinite(values)]
    if finite_nlq.size:
        text = f"median={np.nanmedian(finite_nlq):.3g}\nP90={np.nanpercentile(finite_nlq, 90):.3g}\nNLq>1={np.mean(part['finite_amplitude_nonlinear']):.1%}"
        axes[-1].text(0.02, 0.03, text, transform=axes[-1].transAxes, va="bottom", ha="left", fontsize=8.5, bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "0.7", "linewidth": 0.6})
    fig.savefig(fig_dir / f"{prefix}_NLq_rz.png", bbox_inches="tight")
    plt.close(fig)


def plot_nonlinearity_category_rz(fig_dir: Path, prefix: str, title_prefix: str, part: pd.DataFrame) -> None:
    data = pivot_values(part, "nonlinearity_category_code")
    values = data.to_numpy(dtype="f8")
    colors = ["#bdbdbd", "#2ca25f", "#fdae61", "#de2d26", "#7b3294", "#80b1d3"]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(-0.5, 6.5, 1.0), cmap.N)
    fig, axes = plt.subplots(3, 1, figsize=(7.0, 9.2), dpi=150, sharex=True)
    for ax, (view_title, depth_max) in zip(axes, (("full depth", None), ("upper 100 m", 100.0), ("shallow 20 m", 20.0))):
        ax.pcolormesh(data.columns.to_numpy(dtype="f8"), data.index.to_numpy(dtype="f8"), values, shading="auto", cmap=cmap, norm=norm)
        ax.invert_yaxis()
        if depth_max is not None:
            ax.set_ylim(depth_max, 0.0)
        ax.set_ylabel("depth m")
        ax.set_title(view_title)
    axes[-1].set_xlabel("r/R")
    fig.suptitle(f"{title_prefix}: nonlinearity categories")
    handles = [Patch(facecolor=colors[code], label=NONLINEARITY_CATEGORY_LABELS[code]) for code in range(6)]
    axes[1].legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8)
    fig.savefig(fig_dir / f"{prefix}_nonlinearity_category_rz.png", bbox_inches="tight")
    plt.close(fig)


def plot_qn_vs_qprime_scatter(fig_dir: Path, prefix: str, title_prefix: str, part: pd.DataFrame) -> None:
    x = part["linear_pv_scale"].to_numpy(dtype="f8")
    y = part["q_prime_rms"].to_numpy(dtype="f8")
    valid = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    qn_valid = part["Q_n_valid"].to_numpy(dtype=bool)
    fig, ax = plt.subplots(figsize=(5.8, 5.2), dpi=150)
    ax.scatter(x[valid & ~qn_valid], y[valid & ~qn_valid], s=9, alpha=0.25, color="0.55", label="Qn invalid")
    ax.scatter(x[valid & qn_valid], y[valid & qn_valid], s=12, alpha=0.55, color="#2b8cbe", label="Qn valid")
    if np.any(valid):
        lo = float(np.nanmin([np.nanmin(x[valid]), np.nanmin(y[valid])]))
        hi = float(np.nanmax([np.nanmax(x[valid]), np.nanmax(y[valid])]))
        if np.isfinite(lo) and np.isfinite(hi) and lo > 0 and hi > lo:
            grid = np.logspace(np.log10(lo), np.log10(hi), 100)
            ax.plot(grid, grid, color="#d62728", linewidth=1.2, label="NLq = 1")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("|Qn| R")
    ax.set_ylabel("sqrt(q_prime_variance)")
    ax.set_title(f"{title_prefix}: finite-amplitude PV test")
    ax.grid(True, color="0.9", which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / f"{prefix}_Qn_vs_qprime_scatter.png", bbox_inches="tight")
    plt.close(fig)


def robust_log_values(values: np.ndarray) -> tuple[np.ndarray, float, float]:
    plot_values = np.log10(np.maximum(values, 1e-30))
    finite = plot_values[np.isfinite(plot_values)]
    if finite.size:
        vmin, vmax = np.nanpercentile(finite, [2, 98])
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
            vmin, vmax = float(np.nanmin(finite)), float(np.nanmax(finite))
    else:
        vmin, vmax = -1.0, 1.0
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin, vmax = -1.0, 1.0
    return plot_values, float(vmin), float(vmax)


def plot_with_valid_contour(ax: plt.Axes, part: pd.DataFrame, field: str, *, title: str, cmap: str, log_scale: bool = False, depth_max: float | None = None) -> None:
    data = pivot_values(part, field)
    values = data.to_numpy(dtype="f8")
    if log_scale:
        plot_values, vmin, vmax = robust_log_values(values)
        label = f"log10({field})"
    else:
        plot_values = values
        finite = plot_values[np.isfinite(plot_values)]
        if finite.size:
            vmin, vmax = np.nanpercentile(finite, [2, 98])
            if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
                vmin, vmax = float(np.nanmin(finite)), float(np.nanmax(finite))
        else:
            vmin, vmax = 0.0, 1.0
        label = field
    mesh = ax.pcolormesh(data.columns.to_numpy(dtype="f8"), data.index.to_numpy(dtype="f8"), plot_values, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    valid = pivot_values(part, "Q_n_valid").reindex(index=data.index, columns=data.columns).to_numpy(dtype=bool)
    try:
        ax.contour(data.columns.to_numpy(dtype="f8"), data.index.to_numpy(dtype="f8"), valid.astype(float), levels=[0.5], colors="cyan", linewidths=0.8)
    except ValueError:
        pass
    ax.invert_yaxis()
    if depth_max is not None:
        ax.set_ylim(depth_max, 0.0)
    ax.set_xlabel("r/R")
    ax.set_ylabel("depth m")
    ax.set_title(title)
    return mesh, label


def plot_nonlinearity_mechanism_components(fig_dir: Path, prefix: str, title_prefix: str, part: pd.DataFrame) -> None:
    fields = (
        ("linear_pv_scale", "|Qn| R", "viridis", True),
        ("q_prime_rms", "sqrt(q prime variance)", "magma", True),
        ("recovery_ratio", "|Qn|R / q_rms", "cividis", True),
        ("budget_residual_score", "|J_T residual|", "inferno", False),
    )
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.6), dpi=145)
    for ax, (field, title, cmap, log_scale) in zip(axes.ravel(), fields):
        mesh, label = plot_with_valid_contour(ax, part, field, title=title, cmap=cmap, log_scale=log_scale, depth_max=100.0)
        fig.colorbar(mesh, ax=ax, shrink=0.82, label=label)
    fig.suptitle(f"{title_prefix}: nonlinearity mechanism components (upper 100 m)")
    fig.tight_layout()
    fig.savefig(fig_dir / f"{prefix}_nonlinearity_mechanism_components.png", bbox_inches="tight")
    plt.close(fig)


def plot_nonlinearity_zoom_stack(fig_dir: Path, prefix: str, title_prefix: str, part: pd.DataFrame) -> None:
    data = pivot_values(part, "weak_gradient_score")
    valid_data = pivot_values(part, "Q_n_valid").reindex(index=data.index, columns=data.columns)
    values = data.to_numpy(dtype="f8")
    finite = values[np.isfinite(values)]
    vmin, vmax = (np.nanpercentile(finite, [2, 98]) if finite.size else (0.0, 1.0))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin, vmax = 0.0, 1.0
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 9.0), dpi=145, sharex=True)
    for ax, (title, depth_max) in zip(axes, (("full depth", None), ("upper 100 m", 100.0), ("shallow 20 m", 20.0))):
        mesh = ax.pcolormesh(data.columns.to_numpy(dtype="f8"), data.index.to_numpy(dtype="f8"), values, shading="auto", cmap="Blues", vmin=vmin, vmax=vmax)
        valid = valid_data.to_numpy(dtype=bool)
        try:
            ax.contour(data.columns.to_numpy(dtype="f8"), data.index.to_numpy(dtype="f8"), valid.astype(float), levels=[0.5], colors="#d62728", linewidths=0.8)
        except ValueError:
            pass
        ax.invert_yaxis()
        if depth_max is not None:
            ax.set_ylim(depth_max, 0.0)
        ax.set_ylabel("depth m")
        ax.set_title(title)
    axes[-1].set_xlabel("r/R")
    cbar = fig.colorbar(mesh, ax=axes, shrink=0.88, pad=0.02)
    cbar.set_label("weak gradient score = -log10(|Qn|R / median q_rms)")
    fig.suptitle(f"{title_prefix}: weak-gradient structure with shallow zooms")
    fig.savefig(fig_dir / f"{prefix}_nonlinearity_zoom_stack.png", bbox_inches="tight")
    plt.close(fig)


def composite_rgb_array(part: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    base = pivot_values(part, "weak_gradient_score")
    weak = base.to_numpy(dtype="f8")
    finite = pivot_values(part, "NL_q").reindex(index=base.index, columns=base.columns).to_numpy(dtype="f8")
    residual = pivot_values(part, "budget_residual_score").reindex(index=base.index, columns=base.columns).to_numpy(dtype="f8")

    def normalize(values: np.ndarray, *, log: bool = False) -> np.ndarray:
        vals = np.log10(1.0 + values) if log else values.copy()
        finite_vals = vals[np.isfinite(vals)]
        if not finite_vals.size:
            return np.zeros_like(vals, dtype="f8")
        lo, hi = np.nanpercentile(finite_vals, [5, 95])
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = float(np.nanmin(finite_vals)), float(np.nanmax(finite_vals))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            return np.zeros_like(vals, dtype="f8")
        return np.clip((vals - lo) / (hi - lo), 0.0, 1.0)

    red = normalize(finite, log=True)
    green = normalize(residual, log=False)
    blue = normalize(weak, log=False)
    rgb = np.dstack([np.nan_to_num(red), np.nan_to_num(green), np.nan_to_num(blue)])
    return base, rgb


def plot_nonlinearity_composite_rgb(fig_dir: Path, prefix: str, title_prefix: str, part: pd.DataFrame) -> None:
    data, rgb = composite_rgb_array(part)
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), dpi=145)
    for ax, depth_max, title in zip(axes, (100.0, 20.0), ("upper 100 m", "shallow 20 m")):
        ax.imshow(rgb, origin="upper", extent=(float(data.columns.min()), float(data.columns.max()), float(data.index.max()), float(data.index.min())), aspect="auto")
        ax.set_ylim(depth_max, 0.0)
        ax.set_xlabel("r/R")
        ax.set_ylabel("depth m")
        ax.set_title(title)
    handles = [
        Patch(facecolor=(1, 0, 0), label="red: finite amplitude NLq"),
        Patch(facecolor=(0, 0.8, 0), label="green: budget nonclosure"),
        Patch(facecolor=(0, 0, 1), label="blue: weak PV gradient"),
    ]
    axes[1].legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8)
    fig.suptitle(f"{title_prefix}: nonlinear spatial structure RGB composite")
    fig.tight_layout()
    fig.savefig(fig_dir / f"{prefix}_nonlinearity_composite_rgb.png", bbox_inches="tight")
    plt.close(fig)


def plot_nonlinearity_tilted_structure_3d(fig_dir: Path, prefix: str, title_prefix: str, part: pd.DataFrame, axis_dir: Path, radii: dict[str, float], *, stride: int = 2) -> None:
    data = pivot_values(part, "weak_gradient_score")
    depth = data.index.to_numpy(dtype="f8")
    radial = data.columns.to_numpy(dtype="f8")
    axis_x, axis_y = tilted_axis_for_part(axis_dir, radii, str(part["polarity"].iloc[0]), depth)
    values = data.to_numpy(dtype="f8")
    finite = values[np.isfinite(values)]
    vmin, vmax = (np.nanpercentile(finite, [2, 98]) if finite.size else (0.0, 1.0))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin, vmax = 0.0, 1.0
    step = max(int(stride), 1)
    ds = slice(None, None, step)
    rs = slice(None, None, step)
    x = axis_x[ds][:, None] + radial[rs][None, :]
    y = axis_y[ds][:, None] + np.zeros((len(depth[ds]), len(radial[rs])), dtype="f8")
    z = -depth[ds][:, None] + np.zeros((len(depth[ds]), len(radial[rs])), dtype="f8")
    sampled = values[ds, rs]
    cmap = plt.get_cmap("Blues")
    norm = Normalize(vmin=vmin, vmax=vmax, clip=True)
    colors = cmap(norm(sampled))
    colors[~np.isfinite(sampled)] = (0.7, 0.7, 0.7, 0.2)
    fig = plt.figure(figsize=(7.6, 5.8), dpi=150)
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(x, y, z, facecolors=colors, rstride=1, cstride=1, linewidth=0, antialiased=False, shade=False, alpha=0.94)
    ax.plot(axis_x, axis_y, -depth, color="black", linewidth=2.0)
    ax.scatter([axis_x[-1]], [axis_y[-1]], [-depth[-1]], color="#d62728", s=36)
    ax.set_xlabel("x/R")
    ax.set_ylabel("y/R")
    ax.set_zlabel("-depth m")
    ax.set_title(f"{title_prefix}: weak-gradient structure on tilted axis")
    ax.view_init(elev=24, azim=-58)
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, shrink=0.72, pad=0.06, label="weak gradient score")
    fig.savefig(fig_dir / f"{prefix}_nonlinearity_tilted_structure_3d.png", bbox_inches="tight")
    plt.close(fig)


def interpolate_rz_to_xy_layers(part: pd.DataFrame, field: str, depth: np.ndarray, radial_grid: np.ndarray, depth_indices: list[int], r_norm: np.ndarray) -> np.ndarray:
    data = pivot_values(part, field).reindex(index=depth)
    r = data.columns.to_numpy(dtype="f8")
    matrix = data.to_numpy(dtype="f8")
    layers = []
    for depth_index in depth_indices:
        values = matrix[depth_index]
        good = np.isfinite(values)
        if np.sum(good) < 2:
            layer = np.full_like(r_norm, np.nan, dtype="f8")
        else:
            layer = np.interp(np.clip(r_norm.ravel(), r[good].min(), r[good].max()), r[good], values[good]).reshape(r_norm.shape)
            layer = np.where(r_norm <= np.nanmax(radial_grid), layer, np.nan)
        layers.append(layer)
    return np.asarray(layers)


def nonlinearity_axis_layers(part: pd.DataFrame, axis_dir: Path, radii: dict[str, float], depth: np.ndarray, depth_indices: list[int], x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    polarity = str(part["polarity"].iloc[0])
    axis_x, axis_y = tilted_axis_for_part(axis_dir, radii, polarity, depth)
    x_layers = []
    y_layers = []
    for depth_index in depth_indices:
        x_layers.append(x + axis_x[depth_index])
        y_layers.append(y + axis_y[depth_index])
    return np.asarray(x_layers), np.asarray(y_layers), axis_x, axis_y


def rgba_from_scalar_layers(layers: np.ndarray, cmap_name: str, *, log1p: bool = False, mask: np.ndarray | None = None, invalid_color: tuple[float, float, float, float] = (0.72, 0.72, 0.72, 0.16)) -> tuple[np.ndarray, Normalize, str]:
    values = np.log10(1.0 + layers) if log1p else layers.copy()
    finite_mask = np.isfinite(values)
    if mask is not None:
        finite_mask &= mask
    finite = values[finite_mask]
    if finite.size:
        vmin, vmax = np.nanpercentile(finite, [2, 98])
    else:
        vmin, vmax = 0.0, 1.0
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin, vmax = 0.0, 1.0
    norm = Normalize(vmin=float(vmin), vmax=float(vmax), clip=True)
    cmap = plt.get_cmap(cmap_name)
    colors = cmap(norm(np.where(finite_mask, values, vmin)))
    colors[~finite_mask] = invalid_color
    label = f"log10(1+value)" if log1p else "value"
    return colors, norm, label


def rgb_stack_layers(part: pd.DataFrame, depth: np.ndarray, radial_grid: np.ndarray, depth_indices: list[int], r_norm: np.ndarray) -> np.ndarray:
    weak = interpolate_rz_to_xy_layers(part, "weak_gradient_score", depth, radial_grid, depth_indices, r_norm)
    nlq = interpolate_rz_to_xy_layers(part, "NL_q", depth, radial_grid, depth_indices, r_norm)
    residual = interpolate_rz_to_xy_layers(part, "budget_residual_score", depth, radial_grid, depth_indices, r_norm)

    def norm01(values: np.ndarray, *, log1p: bool = False) -> np.ndarray:
        vals = np.log10(1.0 + values) if log1p else values.copy()
        finite = vals[np.isfinite(vals)]
        if not finite.size:
            return np.zeros_like(vals, dtype="f8")
        lo, hi = np.nanpercentile(finite, [5, 95])
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = float(np.nanmin(finite)), float(np.nanmax(finite))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            return np.zeros_like(vals, dtype="f8")
        return np.clip((vals - lo) / (hi - lo), 0.0, 1.0)

    red = norm01(nlq, log1p=True)
    green = norm01(residual, log1p=False)
    blue = norm01(weak, log1p=False)
    alpha = np.where(np.isfinite(weak) | np.isfinite(nlq) | np.isfinite(residual), 0.44, 0.0)
    return np.stack([np.nan_to_num(red), np.nan_to_num(green), np.nan_to_num(blue), alpha], axis=-1)


def plot_nonlinearity_stack_png(
    output_dir: Path,
    prefix: str,
    title_prefix: str,
    depth: np.ndarray,
    depth_indices: list[int],
    x_layers: np.ndarray,
    y_layers: np.ndarray,
    axis_x: np.ndarray,
    axis_y: np.ndarray,
    facecolors: np.ndarray,
    *,
    colorbar_norm: Normalize | None,
    colorbar_cmap: str | None,
    colorbar_label: str | None,
    suffix: str,
    colorbar_ticks: list[float] | None = None,
    colorbar_ticklabels: list[str] | None = None,
) -> None:
    fig = plt.figure(figsize=(10.5, 8.5), dpi=170)
    ax = fig.add_subplot(111, projection="3d")
    for layer_idx, depth_index in enumerate(depth_indices):
        x_plot = x_layers[layer_idx]
        y_plot = y_layers[layer_idx]
        z_plot = np.full_like(x_plot, -float(depth[depth_index]))
        ax.plot_surface(x_plot, y_plot, z_plot, facecolors=facecolors[layer_idx], linewidth=0, antialiased=False, shade=False)
        angle = np.linspace(0, 2.0 * np.pi, 240)
        edge_radius = np.nanmax(np.hypot(x_plot - axis_x[depth_index], y_plot - axis_y[depth_index]))
        ax.plot(axis_x[depth_index] + edge_radius * np.cos(angle), axis_y[depth_index] + edge_radius * np.sin(angle), np.full_like(angle, -float(depth[depth_index])), color="0.25", alpha=0.30, linewidth=0.55)
        ax.text(float(np.nanmax(x_plot)) + 0.12, float(np.nanmax(y_plot)) + 0.12, -float(depth[depth_index]), f"{depth[depth_index]:.0f} m", fontsize=8)
    ax.plot(axis_x, axis_y, -depth, color="black", linewidth=2.4)
    ax.scatter([axis_x[0]], [axis_y[0]], [-depth[0]], color="black", s=24)
    ax.scatter([axis_x[-1]], [axis_y[-1]], [-depth[-1]], color="#d62728", s=42)
    if colorbar_norm is not None and colorbar_cmap is not None and colorbar_label is not None:
        cmap_obj = plt.get_cmap(colorbar_cmap) if isinstance(colorbar_cmap, str) else colorbar_cmap
        scalar = ScalarMappable(norm=colorbar_norm, cmap=cmap_obj)
        scalar.set_array([])
        cbar = fig.colorbar(scalar, ax=ax, shrink=0.65, pad=0.08, label=colorbar_label)
        if colorbar_ticks is not None:
            cbar.set_ticks(colorbar_ticks)
        if colorbar_ticklabels is not None:
            cbar.set_ticklabels(colorbar_ticklabels)
    else:
        handles = [
            Patch(facecolor=(1, 0, 0, 0.7), label="red: finite amplitude NLq"),
            Patch(facecolor=(0, 0.8, 0, 0.7), label="green: J_T residual"),
            Patch(facecolor=(0, 0, 1, 0.7), label="blue: weak PV gradient"),
        ]
        ax.legend(handles=handles, loc="upper left", fontsize=8)
    ax.set_xlabel("x/R")
    ax.set_ylabel("y/R")
    ax.set_zlabel("-depth m")
    ax.set_title(f"{title_prefix}: {suffix.replace('_', ' ')}")
    ax.view_init(elev=23, azim=-52)
    ax.set_box_aspect((1.25, 1, 0.82))
    fig.tight_layout()
    fig.savefig(output_dir / f"{prefix}_{suffix}.png")
    plt.close(fig)


def plot_nonlinearity_stack_html(
    output_dir: Path,
    prefix: str,
    title_prefix: str,
    depth: np.ndarray,
    depth_indices: list[int],
    x_layers: np.ndarray,
    y_layers: np.ndarray,
    axis_x: np.ndarray,
    axis_y: np.ndarray,
    surface_values: np.ndarray,
    *,
    suffix: str,
    colorscale: str,
    cmin: float,
    cmax: float,
) -> None:
    import plotly.graph_objects as go

    fig = go.Figure()
    for layer_idx, depth_index in enumerate(depth_indices):
        fig.add_trace(
            go.Surface(
                x=x_layers[layer_idx],
                y=y_layers[layer_idx],
                z=np.full_like(x_layers[layer_idx], -float(depth[depth_index])),
                surfacecolor=surface_values[layer_idx],
                colorscale=colorscale,
                cmin=cmin,
                cmax=cmax,
                opacity=0.52,
                showscale=layer_idx == 0,
                colorbar={"title": suffix} if layer_idx == 0 else None,
                name=f"{depth[depth_index]:.0f} m",
            )
        )
    fig.add_trace(go.Scatter3d(x=axis_x, y=axis_y, z=-depth, mode="lines", line={"color": "black", "width": 8}, name="tilted axis"))
    fig.add_trace(go.Scatter3d(x=[axis_x[-1]], y=[axis_y[-1]], z=[-depth[-1]], mode="markers", marker={"color": "red", "size": 5}, name="deep axis end"))
    fig.update_layout(
        title=f"{title_prefix}: {suffix.replace('_', ' ')}",
        scene={"xaxis_title": "x/R", "yaxis_title": "y/R", "zaxis_title": "-depth m", "aspectmode": "data"},
        height=820,
        width=1000,
    )
    fig.write_html(output_dir / f"{prefix}_{suffix}.html", include_plotlyjs="cdn")


def write_nonlinearity_stack_3d_outputs(output_dir: Path, nonlinear: pd.DataFrame, axis_dir: Path, radii: dict[str, float], args: argparse.Namespace) -> None:
    if not bool(args.nonlinearity_stack_3d):
        return
    stack_dir = output_dir / "strong_nonlinearity_diagnostics" / "nonlinearity_stack_3d"
    stack_dir.mkdir(parents=True, exist_ok=True)
    x, y, r_norm = make_xy_grid(float(args.nonlinearity_stack_xy_extent), int(args.nonlinearity_stack_grid_size))
    requested_depths = parse_float_list(args.nonlinearity_stack_depth_levels)
    for (polarity, phase_name), part in nonlinear.groupby(["polarity", "phase_name"], sort=True):
        prefix = f"{polarity}_{phase_name}"
        title_prefix = f"{polarity} {phase_name}"
        depth = np.sort(part["depth_m"].unique().astype("f8"))
        depth_indices = nearest_depth_indices(depth, requested_depths)
        radial_grid = np.sort(part["r_over_R"].unique().astype("f8"))
        x_layers, y_layers, axis_x, axis_y = nonlinearity_axis_layers(part, axis_dir, radii, depth, depth_indices, x, y)

        weak_layers = interpolate_rz_to_xy_layers(part, "weak_gradient_score", depth, radial_grid, depth_indices, r_norm)
        weak_colors, weak_norm, _ = rgba_from_scalar_layers(weak_layers, "Blues")
        weak_colors[..., 3] = np.where(np.isfinite(weak_layers), 0.38, 0.0)
        plot_nonlinearity_stack_png(stack_dir, prefix, title_prefix, depth, depth_indices, x_layers, y_layers, axis_x, axis_y, weak_colors, colorbar_norm=weak_norm, colorbar_cmap="Blues", colorbar_label="weak gradient score", suffix="weak_gradient_stack_3d")

        nlq_layers = interpolate_rz_to_xy_layers(part, "NL_q", depth, radial_grid, depth_indices, r_norm)
        valid_layers = interpolate_rz_to_xy_layers(part, "Q_n_valid", depth, radial_grid, depth_indices, r_norm) > 0.5
        nlq_colors, nlq_norm, _ = rgba_from_scalar_layers(nlq_layers, "magma", log1p=True, mask=valid_layers)
        nlq_colors[..., 3] = np.where(valid_layers & np.isfinite(nlq_layers), 0.48, 0.13)
        plot_nonlinearity_stack_png(stack_dir, prefix, title_prefix, depth, depth_indices, x_layers, y_layers, axis_x, axis_y, nlq_colors, colorbar_norm=nlq_norm, colorbar_cmap="magma", colorbar_label="log10(1 + NLq), Qn valid only", suffix="NLq_stack_3d")

        rgb_colors = rgb_stack_layers(part, depth, radial_grid, depth_indices, r_norm)
        plot_nonlinearity_stack_png(stack_dir, prefix, title_prefix, depth, depth_indices, x_layers, y_layers, axis_x, axis_y, rgb_colors, colorbar_norm=None, colorbar_cmap=None, colorbar_label=None, suffix="mechanism_rgb_stack_3d")

        if bool(args.nonlinearity_stack_html):
            weak_values = weak_layers.copy()
            finite = weak_values[np.isfinite(weak_values)]
            cmin, cmax = (np.nanpercentile(finite, [2, 98]) if finite.size else (0.0, 1.0))
            plot_nonlinearity_stack_html(stack_dir, prefix, title_prefix, depth, depth_indices, x_layers, y_layers, axis_x, axis_y, weak_values, suffix="weak_gradient_stack_3d", colorscale="Blues", cmin=float(cmin), cmax=float(cmax))
            nlq_values = np.where(valid_layers, np.log10(1.0 + nlq_layers), np.nan)
            finite = nlq_values[np.isfinite(nlq_values)]
            cmin, cmax = (np.nanpercentile(finite, [2, 98]) if finite.size else (0.0, 1.0))
            plot_nonlinearity_stack_html(stack_dir, prefix, title_prefix, depth, depth_indices, x_layers, y_layers, axis_x, axis_y, nlq_values, suffix="NLq_stack_3d", colorscale="Magma", cmin=float(cmin), cmax=float(cmax))


def plot_nonlinearity_summary(fig_dir: Path, metrics: pd.DataFrame) -> None:
    if metrics.empty:
        return
    phase_order = [name for name in PHASE_NAMES if name in set(metrics["phase_name"])]
    stack_fields = [
        ("Q_n_invalid_fraction", "weak gradient/singular", "#bdbdbd"),
        ("fraction_NLq_gt1", "NLq > 1", "#fdae61"),
        ("fraction_budget_nonclosure", "|J residual| > 2", "#de2d26"),
        ("linear_wave_like_fraction", "linear wave-like", "#2ca25f"),
    ]
    polarities = list(metrics["polarity"].drop_duplicates())
    fig, axes = plt.subplots(len(polarities), 1, figsize=(8.5, 3.3 * max(len(polarities), 1)), dpi=150, sharex=True)
    if len(polarities) == 1:
        axes = [axes]
    for ax, polarity in zip(axes, polarities):
        part = metrics[metrics["polarity"] == polarity].set_index("phase_name").reindex(phase_order)
        x = np.arange(len(part))
        bottom = np.zeros(len(part), dtype="f8")
        for field, label, color in stack_fields:
            values = part[field].to_numpy(dtype="f8")
            ax.bar(x, values, bottom=bottom, label=label, color=color, alpha=0.86)
            bottom += np.nan_to_num(values, nan=0.0)
        ax.set_title(f"{polarity}: nonlinearity fractions by phase")
        ax.set_ylabel("fraction")
        ax.set_ylim(0, max(1.0, float(np.nanmax(bottom)) if np.isfinite(bottom).any() else 1.0))
        ax.grid(True, color="0.92", axis="y")
        ax.legend(fontsize=8, ncol=2)
    axes[-1].set_xticks(np.arange(len(phase_order)), phase_order)
    fig.tight_layout()
    fig.savefig(fig_dir / "nonlinearity_by_phase.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.2, 4.8), dpi=150)
    for polarity, part in metrics.groupby("polarity", sort=True):
        part = part.sort_values("tau_center")
        ax.plot(part["tau_center"], part["linear_wave_like_fraction"], marker="o", label=polarity)
    ax.set_xlabel("tau")
    ax.set_ylabel("linear wave-like fraction")
    ax.set_title("Fraction still interpretable by linear wave-action theory")
    ax.grid(True, color="0.9")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "linear_wave_like_fraction_by_phase.png", bbox_inches="tight")
    plt.close(fig)


def plot_spatial_profile_overviews(fig_dir: Path, nonlinear: pd.DataFrame) -> None:
    if nonlinear.empty:
        return

    def summarize(group_cols: list[str]) -> pd.DataFrame:
        rows = []
        for keys, part in nonlinear.groupby(group_cols, sort=True):
            if not isinstance(keys, tuple):
                keys = (keys,)
            row = dict(zip(group_cols, keys))
            valid_nlq = part.loc[part["Q_n_valid"], "NL_q"].to_numpy(dtype="f8")
            row.update(
                {
                    "weak_gradient_fraction": float(np.mean(part["weak_gradient_or_singular"].to_numpy(dtype=bool))),
                    "NL_q_median": float(np.nanmedian(valid_nlq)) if np.isfinite(valid_nlq).any() else np.nan,
                    "budget_residual_median": float(np.nanmedian(part["budget_residual_score"].to_numpy(dtype="f8"))),
                }
            )
            rows.append(row)
        return pd.DataFrame(rows)

    for group_axis, x_name, filename, xlabel in (
        ("depth_m", "depth_m", "nonlinearity_depth_profile_by_phase.png", "depth m"),
        ("r_over_R", "r_over_R", "nonlinearity_radial_profile_by_phase.png", "r/R"),
    ):
        table = summarize(["polarity", "phase_name", "tau_center", group_axis])
        fig, axes = plt.subplots(3, 1, figsize=(10.5, 9.2), dpi=145, sharex=True)
        for polarity, pol_part in table.groupby("polarity", sort=True):
            for phase_name, part in pol_part.groupby("phase_name", sort=True):
                part = part.sort_values(x_name)
                label = f"{polarity} {phase_name}"
                alpha = 0.68 if polarity == "cyclonic" else 0.48
                axes[0].plot(part[x_name], part["weak_gradient_fraction"], label=label, alpha=alpha)
                axes[1].plot(part[x_name], part["NL_q_median"], label=label, alpha=alpha)
                axes[2].plot(part[x_name], part["budget_residual_median"], label=label, alpha=alpha)
        axes[0].set_ylabel("weak-gradient fraction")
        axes[1].set_ylabel("median NLq")
        axes[1].set_yscale("log")
        axes[2].set_ylabel("median |J residual|")
        axes[2].set_xlabel(xlabel)
        if group_axis == "depth_m":
            axes[2].set_xlim(0, 120.0)
        for ax in axes:
            ax.grid(True, color="0.9")
        axes[0].legend(fontsize=7, ncol=2, bbox_to_anchor=(1.02, 1.0), loc="upper left")
        fig.suptitle(filename.replace("_", " ").replace(".png", ""))
        fig.tight_layout()
        fig.savefig(fig_dir / filename, bbox_inches="tight")
        plt.close(fig)


def plot_nonlinearity_structure_mosaic(fig_dir: Path, nonlinear: pd.DataFrame) -> None:
    phase_order = [name for name in PHASE_NAMES if name in set(nonlinear["phase_name"])]
    polarities = list(nonlinear["polarity"].drop_duplicates())
    if not phase_order or not polarities:
        return
    fig, axes = plt.subplots(len(polarities), len(phase_order), figsize=(3.0 * len(phase_order), 2.8 * len(polarities)), dpi=145, squeeze=False)
    for row, polarity in enumerate(polarities):
        for col, phase_name in enumerate(phase_order):
            ax = axes[row][col]
            part = nonlinear[(nonlinear["polarity"] == polarity) & (nonlinear["phase_name"] == phase_name)]
            if part.empty:
                ax.axis("off")
                continue
            data, rgb = composite_rgb_array(part)
            ax.imshow(rgb, origin="upper", extent=(float(data.columns.min()), float(data.columns.max()), float(data.index.max()), float(data.index.min())), aspect="auto")
            ax.set_ylim(100.0, 0.0)
            ax.set_title(f"{polarity}\n{phase_name}", fontsize=9)
            if col == 0:
                ax.set_ylabel("depth m")
            if row == len(polarities) - 1:
                ax.set_xlabel("r/R")
    handles = [
        Patch(facecolor=(1, 0, 0), label="finite amplitude"),
        Patch(facecolor=(0, 0.8, 0), label="budget nonclosure"),
        Patch(facecolor=(0, 0, 1), label="weak PV gradient"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3)
    fig.suptitle("Nonlinearity structure mosaic, upper 100 m", y=1.04)
    fig.tight_layout()
    fig.savefig(fig_dir / "nonlinearity_structure_mosaic.png", bbox_inches="tight")
    plt.close(fig)


def plot_linear_recovery_zone_zoom(fig_dir: Path, nonlinear: pd.DataFrame) -> None:
    phase_order = [name for name in PHASE_NAMES if name in set(nonlinear["phase_name"])]
    polarities = list(nonlinear["polarity"].drop_duplicates())
    fig, axes = plt.subplots(len(polarities), len(phase_order), figsize=(3.0 * max(len(phase_order), 1), 2.8 * max(len(polarities), 1)), dpi=145, squeeze=False)
    cmap = ListedColormap(["#f0f0f0", "#3182bd"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5], cmap.N)
    for row, polarity in enumerate(polarities):
        for col, phase_name in enumerate(phase_order):
            ax = axes[row][col]
            part = nonlinear[(nonlinear["polarity"] == polarity) & (nonlinear["phase_name"] == phase_name)]
            if part.empty:
                ax.axis("off")
                continue
            data = pivot_values(part, "Q_n_valid")
            values = data.to_numpy(dtype=bool).astype(float)
            ax.pcolormesh(data.columns.to_numpy(dtype="f8"), data.index.to_numpy(dtype="f8"), values, shading="auto", cmap=cmap, norm=norm)
            ax.invert_yaxis()
            ax.set_ylim(30.0, 0.0)
            ax.set_title(f"{polarity}\n{phase_name}", fontsize=9)
            if col == 0:
                ax.set_ylabel("depth m")
            if row == len(polarities) - 1:
                ax.set_xlabel("r/R")
    fig.suptitle("Linear recovery zone: Q_n_valid=True, shallow 30 m", y=1.04)
    fig.tight_layout()
    fig.savefig(fig_dir / "linear_recovery_zone_zoom.png", bbox_inches="tight")
    plt.close(fig)


def write_nonlinearity_summary(nonlinear_dir: Path, metrics: pd.DataFrame, summary: pd.DataFrame) -> None:
    lines = [
        "# Strong nonlinearity diagnostics",
        "",
        "Strong nonlinearity is split into weak/singular mean PV gradient, finite-amplitude PV perturbation, and total wave-action budget nonclosure.",
        "NL_q = sqrt(q_prime_variance) / (abs(Q_n) R) is evaluated only where Q_n_valid=True; Q_n_valid=False is kept as weak_gradient_or_singular rather than filled.",
        "Spatial-structure figures are written under `spatial_structure_figures/`. They include shallow zooms because most Q_n-valid recovery zones are compressed into the upper tens of meters on the full-depth axis.",
        "Stacked 3D nonlinearity figures are written under `nonlinearity_stack_3d/`. They rotate the azimuthal-mean r/R-depth structure into axisymmetric horizontal disks along the pooled tilted axis; they are representative geometry, not object-level asymmetric 3D fields.",
        "Blue in the RGB composites marks weak PV-gradient structure, red marks finite-amplitude NL_q, and green marks J_T budget-residual hotspots.",
        "The current diagnosis is that strong nonlinearity mainly appears as weak PV recovery through most of the column, with finite-amplitude behavior in the shallow Q_n-valid recovery zone.",
        "",
        "## Summary By Polarity",
        "```csv",
        summary.to_csv(index=False).strip(),
        "```",
        "",
        "## Metrics By Polarity And Phase",
        "```csv",
        metrics.to_csv(index=False).strip(),
        "```",
    ]
    (nonlinear_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_nonlinearity_outputs(output_dir: Path, profiles: pd.DataFrame, radii: dict[str, float], axis_dir: Path, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    nonlinear_dir = output_dir / "strong_nonlinearity_diagnostics"
    fig_dir = nonlinear_dir / "figures"
    spatial_dir = nonlinear_dir / "spatial_structure_figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    spatial_dir.mkdir(parents=True, exist_ok=True)
    nonlinear, metrics, summary = build_nonlinearity_tables(profiles, radii)
    nonlinear.to_parquet(nonlinear_dir / "nonlinearity_profiles.parquet", index=False)
    nonlinear.to_csv(nonlinear_dir / "nonlinearity_profiles.csv", index=False)
    metrics.to_csv(nonlinear_dir / "nonlinearity_metrics.csv", index=False)
    summary.to_csv(nonlinear_dir / "nonlinearity_summary_by_polarity.csv", index=False)
    for (polarity, phase_name), part in nonlinear.groupby(["polarity", "phase_name"], sort=True):
        prefix = f"{polarity}_{phase_name}"
        title_prefix = f"{polarity} {phase_name}"
        plot_nlq_rz(fig_dir, prefix, title_prefix, part)
        plot_nonlinearity_category_rz(fig_dir, prefix, title_prefix, part)
        plot_qn_vs_qprime_scatter(fig_dir, prefix, title_prefix, part)
        plot_nonlinearity_mechanism_components(spatial_dir, prefix, title_prefix, part)
        plot_nonlinearity_zoom_stack(spatial_dir, prefix, title_prefix, part)
        plot_nonlinearity_composite_rgb(spatial_dir, prefix, title_prefix, part)
        plot_nonlinearity_tilted_structure_3d(spatial_dir, prefix, title_prefix, part, axis_dir, radii, stride=2)
    plot_nonlinearity_summary(fig_dir, metrics)
    plot_spatial_profile_overviews(spatial_dir, nonlinear)
    plot_nonlinearity_structure_mosaic(spatial_dir, nonlinear)
    plot_linear_recovery_zone_zoom(spatial_dir, nonlinear)
    write_nonlinearity_stack_3d_outputs(output_dir, nonlinear, axis_dir, radii, args)
    write_nonlinearity_summary(nonlinear_dir, metrics, summary)
    return metrics, summary


def lifecycle_radii_from_args(args: argparse.Namespace) -> dict[str, float]:
    shapes = parse_csv_list(args.shapes, DEFAULT_SHAPES)
    polarities = parse_csv_list(args.polarities, DEFAULT_POLARITIES)
    objects = load_lifecycle_objects(Path(args.axis_dir), Path(args.catalog_dir), Path(args.shape_dir), shapes, polarities)
    objects = apply_lifecycle_limits(objects, int(args.max_days), int(args.max_objects_per_polarity), int(args.random_seed))
    if objects.empty:
        raise RuntimeError("No lifecycle objects selected for representative radii.")
    return {str(p): float(part["mean_radius_m"].median()) for p, part in objects.groupby("polarity")}


def run_nonlinearity_diagnostics_only(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    profiles_path = output_dir / "nondim_vertical_flux_decomposition.parquet"
    if not profiles_path.exists():
        raise FileNotFoundError(f"Missing required profile parquet: {profiles_path}")
    profiles = pd.read_parquet(profiles_path)
    radii = lifecycle_radii_from_args(args)
    write_nonlinearity_outputs(output_dir, profiles, radii, Path(args.axis_dir), args)
    print(f"Strong nonlinearity diagnostics: {output_dir / 'strong_nonlinearity_diagnostics'}")


REGION_ORDER = ("core", "near_field", "outer_ring", "diagnostic_window", "layer_body")


def load_lifecycle_objects_for_args(args: argparse.Namespace) -> pd.DataFrame:
    shapes = parse_csv_list(args.shapes, DEFAULT_SHAPES)
    polarities = parse_csv_list(args.polarities, DEFAULT_POLARITIES)
    objects = load_lifecycle_objects(Path(args.axis_dir), Path(args.catalog_dir), Path(args.shape_dir), shapes, polarities)
    return apply_lifecycle_limits(objects, int(args.max_days), int(args.max_objects_per_polarity), int(args.random_seed))


def load_layer_radius_context(args: argparse.Namespace, objects: pd.DataFrame) -> pd.DataFrame:
    object_cols = ["eddy3d_object_id", "polarity", "phase_index", "phase_name", "mean_radius_m"]
    context = objects[object_cols].drop_duplicates("eddy3d_object_id").copy()
    centers = pd.read_parquet(
        Path(args.catalog_dir) / "layer_centers_completed.parquet",
        columns=["eddy3d_object_id", "depth_index", "depth_m", "latitude", "radius_m", "radius_source", "speed_at_core"],
    )
    centers = centers.merge(context, on="eddy3d_object_id", how="inner")
    centers["radius_over_object_R"] = centers["radius_m"].astype("f8") / centers["mean_radius_m"].astype("f8")
    centers["speed_for_Ro_m_s"] = centers["speed_at_core"].astype("f8")

    need_speed = ~np.isfinite(centers["speed_for_Ro_m_s"].to_numpy(dtype="f8"))
    if bool(np.any(need_speed)):
        obs = pd.read_parquet(
            Path(args.catalog_dir) / "layer_observations.parquet",
            columns=["eddy3d_object_id", "depth_index", "core_speed"],
        )
        obs = obs.groupby(["eddy3d_object_id", "depth_index"], as_index=False).agg(core_speed=("core_speed", "mean"))
        centers = centers.merge(obs, on=["eddy3d_object_id", "depth_index"], how="left")
        centers["speed_for_Ro_m_s"] = centers["speed_for_Ro_m_s"].where(np.isfinite(centers["speed_for_Ro_m_s"]), centers["core_speed"])
    else:
        centers["core_speed"] = np.nan

    lat = centers["latitude"].astype("f8").to_numpy()
    f0 = 2.0 * OMEGA * np.sin(np.radians(lat))
    denom = np.abs(f0) * centers["radius_m"].astype("f8").to_numpy()
    centers["Ro_layer"] = np.divide(
        centers["speed_for_Ro_m_s"].astype("f8").to_numpy(),
        denom,
        out=np.full(len(centers), np.nan, dtype="f8"),
        where=np.isfinite(denom) & (denom > 0),
    )
    centers = centers[np.isfinite(centers["radius_over_object_R"]) & (centers["radius_over_object_R"] > 0)].copy()
    return centers


def build_layer_body_radius_profile(layer_context: pd.DataFrame) -> pd.DataFrame:
    if layer_context.empty:
        return pd.DataFrame()
    rows = []
    for (polarity, phase_name, depth_index, depth_m), part in layer_context.groupby(["polarity", "phase_name", "depth_index", "depth_m"], sort=True):
        source_counts = part["radius_source"].value_counts(dropna=False).to_dict()
        rows.append(
            {
                "polarity": polarity,
                "phase_name": phase_name,
                "tau_center": TAU_CENTERS[str(phase_name)],
                "depth_index": int(depth_index),
                "depth_m": float(depth_m),
                "radius_over_object_R_median": float(part["radius_over_object_R"].median()),
                "radius_over_object_R_p25": float(part["radius_over_object_R"].quantile(0.25)),
                "radius_over_object_R_p75": float(part["radius_over_object_R"].quantile(0.75)),
                "n_objects": int(part["eddy3d_object_id"].nunique()),
                "detected_layer_radius_fraction": float(np.mean(part["radius_source"].astype(str) == "detected_layer_radius")),
                "radius_source_counts": ";".join(f"{key}:{value}" for key, value in source_counts.items()),
            }
        )
    return pd.DataFrame(rows)


def build_ro_tables(layer_context: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if layer_context.empty:
        return pd.DataFrame(), pd.DataFrame()
    rows_depth = []
    for (polarity, phase_name, depth_index, depth_m), part in layer_context.groupby(["polarity", "phase_name", "depth_index", "depth_m"], sort=True):
        ro = part["Ro_layer"].to_numpy(dtype="f8")
        ro = ro[np.isfinite(ro)]
        rows_depth.append(
            {
                "polarity": polarity,
                "phase_name": phase_name,
                "tau_center": TAU_CENTERS[str(phase_name)],
                "depth_index": int(depth_index),
                "depth_m": float(depth_m),
                "Ro_median": float(np.nanmedian(ro)) if ro.size else np.nan,
                "Ro_p75": float(np.nanpercentile(ro, 75)) if ro.size else np.nan,
                "Ro_p90": float(np.nanpercentile(ro, 90)) if ro.size else np.nan,
                "Ro_p95": float(np.nanpercentile(ro, 95)) if ro.size else np.nan,
                "n_samples": int(ro.size),
            }
        )
    rows_phase = []
    for (polarity, phase_name), part in layer_context.groupby(["polarity", "phase_name"], sort=True):
        ro = part["Ro_layer"].to_numpy(dtype="f8")
        ro = ro[np.isfinite(ro)]
        rows_phase.append(
            {
                "polarity": polarity,
                "phase_name": phase_name,
                "tau_center": TAU_CENTERS[str(phase_name)],
                "Ro_median": float(np.nanmedian(ro)) if ro.size else np.nan,
                "Ro_p75": float(np.nanpercentile(ro, 75)) if ro.size else np.nan,
                "Ro_p90": float(np.nanpercentile(ro, 90)) if ro.size else np.nan,
                "Ro_p95": float(np.nanpercentile(ro, 95)) if ro.size else np.nan,
                "fraction_Ro_lt_0p1": float(np.mean(ro < 0.1)) if ro.size else np.nan,
                "fraction_Ro_lt_0p2": float(np.mean(ro < 0.2)) if ro.size else np.nan,
                "n_samples": int(ro.size),
            }
        )
    return pd.DataFrame(rows_depth), pd.DataFrame(rows_phase)


def region_masks(part: pd.DataFrame, layer_body_profile: pd.DataFrame) -> dict[str, np.ndarray]:
    r = part["r_over_R"].to_numpy(dtype="f8")
    masks = {
        "core": (r >= 0.0) & (r <= 1.0),
        "near_field": (r > 1.0) & (r <= 1.5),
        "outer_ring": (r > 1.5) & (r <= 2.5),
        "diagnostic_window": (r >= 0.0) & (r <= 2.5),
    }
    body = part[["polarity", "phase_name", "depth_index", "r_over_R"]].merge(
        layer_body_profile[["polarity", "phase_name", "depth_index", "radius_over_object_R_median"]],
        on=["polarity", "phase_name", "depth_index"],
        how="left",
    )
    limit = body["radius_over_object_R_median"].to_numpy(dtype="f8")
    masks["layer_body"] = np.isfinite(limit) & (r >= 0.0) & (r <= limit)
    return masks


def build_fj_tilt_region_sensitivity(profiles: pd.DataFrame, layer_body_profile: pd.DataFrame) -> pd.DataFrame:
    tilted = profiles[profiles["axis_mode"] == "tilted"].copy()
    rows = []
    for (polarity, phase_name), part in tilted.groupby(["polarity", "phase_name"], sort=True):
        masks = region_masks(part, layer_body_profile)
        for region in REGION_ORDER:
            use = masks[region]
            sub = part.loc[use]
            rows.append(
                {
                    "polarity": polarity,
                    "phase_name": phase_name,
                    "tau_center": TAU_CENTERS[str(phase_name)],
                    "region": region,
                    "n_bins": int(len(sub)),
                    "median_tilt_fraction": float(np.nanmedian(sub["tilt_fraction"])) if len(sub) else np.nan,
                    "rms_Fz_tilt_correction": rms(sub["F_z_tilt_correction"].to_numpy(dtype="f8")) if len(sub) else np.nan,
                    "rms_Fn": rms(sub["F_n"].to_numpy(dtype="f8")) if len(sub) else np.nan,
                    "rms_divF": rms(sub["divF"].to_numpy(dtype="f8")) if len(sub) else np.nan,
                    "rms_divJ": rms(sub["divJ"].to_numpy(dtype="f8")) if "divJ" in sub and len(sub) else np.nan,
                    "J_residual_rms": rms(sub["wave_activity_residual_J_nd"].to_numpy(dtype="f8")) if "wave_activity_residual_J_nd" in sub and len(sub) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def build_nonlinearity_region_sensitivity(nonlinear: pd.DataFrame, layer_body_profile: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (polarity, phase_name), part in nonlinear.groupby(["polarity", "phase_name"], sort=True):
        masks = region_masks(part, layer_body_profile)
        for region in REGION_ORDER:
            sub = part.loc[masks[region]]
            nlq = sub["NL_q"].to_numpy(dtype="f8") if len(sub) else np.asarray([], dtype="f8")
            nlq = nlq[np.isfinite(nlq)]
            rows.append(
                {
                    "polarity": polarity,
                    "phase_name": phase_name,
                    "tau_center": TAU_CENTERS[str(phase_name)],
                    "region": region,
                    "n_bins": int(len(sub)),
                    "Q_n_valid_fraction": float(np.mean(sub["Q_n_valid"].to_numpy(dtype=bool))) if len(sub) else np.nan,
                    "NL_q_median": float(np.nanmedian(nlq)) if nlq.size else np.nan,
                    "NL_q_p90": float(np.nanpercentile(nlq, 90)) if nlq.size else np.nan,
                    "fraction_NLq_gt1": float(np.mean(sub["finite_amplitude_nonlinear"].to_numpy(dtype=bool))) if len(sub) else np.nan,
                    "linear_wave_like_fraction": float(np.mean(sub["linear_wave_like"].to_numpy(dtype=bool))) if len(sub) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def build_representative_velocity_region_sensitivity(output_dir: Path, radii: dict[str, float], layer_body_profile: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    profile_path = output_dir / "streamfunction_templates" / "lifecycle_radial_psi_profiles.parquet"
    if not profile_path.exists():
        return pd.DataFrame(), "representative velocity sensitivity skipped: template profile not found"
    profiles = pd.read_parquet(profile_path)
    rows = []
    for (polarity, phase_name), part in profiles.groupby(["polarity", "phase_name"], sort=True):
        radius = radii.get(str(polarity), np.nan)
        if not np.isfinite(radius):
            continue
        depth = np.sort(part["depth_m"].unique().astype("f8"))
        r = np.sort(part["r_over_R"].unique().astype("f8"))
        matrix = part.pivot(index="depth_m", columns="r_over_R", values="psi_mean").sort_index().to_numpy(dtype="f8")
        speed = np.abs(velocity_from_psi(matrix, r, float(radius)))
        grid = pd.DataFrame(
            [
                {"polarity": polarity, "phase_name": phase_name, "depth_index": i, "depth_m": float(depth_m), "r_over_R": float(rv), "speed_m_s": float(speed[i, j])}
                for i, depth_m in enumerate(depth)
                for j, rv in enumerate(r)
            ]
        )
        masks = region_masks(grid, layer_body_profile)
        for region in REGION_ORDER:
            values = grid.loc[masks[region], "speed_m_s"].to_numpy(dtype="f8")
            values = values[np.isfinite(values)]
            rows.append(
                {
                    "polarity": polarity,
                    "phase_name": phase_name,
                    "tau_center": TAU_CENTERS[str(phase_name)],
                    "region": region,
                    "speed_max_m_s": float(np.nanmax(values)) if values.size else np.nan,
                    "speed_p95_m_s": float(np.nanpercentile(values, 95)) if values.size else np.nan,
                    "speed_rms_m_s": rms(values),
                    "n_bins": int(values.size),
                }
            )
    return pd.DataFrame(rows), "representative velocity sensitivity computed"


def build_qg_pv_summary(ro_phase: pd.DataFrame, nonlinear_region: pd.DataFrame) -> pd.DataFrame:
    rows = []
    core = nonlinear_region[nonlinear_region["region"] == "core"].copy()
    for polarity, ro_part in ro_phase.groupby("polarity", sort=True):
        nl_part = core[core["polarity"] == polarity]
        ro_p90 = float(np.nanmedian(ro_part["Ro_p90"].to_numpy(dtype="f8"))) if not ro_part.empty else np.nan
        ro_p95 = float(np.nanmedian(ro_part["Ro_p95"].to_numpy(dtype="f8"))) if not ro_part.empty else np.nan
        nlq_median = float(np.nanmedian(nl_part["NL_q_median"].to_numpy(dtype="f8"))) if not nl_part.empty else np.nan
        qn_valid = float(np.nanmedian(nl_part["Q_n_valid_fraction"].to_numpy(dtype="f8"))) if not nl_part.empty else np.nan
        qg_yes = bool(np.isfinite(ro_p90) and ro_p90 < 0.2)
        finite_pv_yes = bool(np.isfinite(nlq_median) and nlq_median > 1.0)
        rows.append(
            {
                "polarity": polarity,
                "Ro_p90_median_across_phases": ro_p90,
                "Ro_p95_median_across_phases": ro_p95,
                "core_NL_q_median_across_phases": nlq_median,
                "core_Q_n_valid_fraction_median": qn_valid,
                "quasi_geostrophic_yes_no": "yes" if qg_yes else "review",
                "finite_amplitude_PV_yes_no": "yes" if finite_pv_yes else "no_or_review",
                "interpretation": "Ro small but PV finite-amplitude nonlinear" if qg_yes and finite_pv_yes else "review thresholds or region sensitivity",
            }
        )
    return pd.DataFrame(rows)


def plot_region_metric(fig_dir: Path, table: pd.DataFrame, value: str, ylabel: str, title: str, filename: str) -> None:
    if table.empty or value not in table:
        return
    regions = list(REGION_ORDER)
    polarities = list(table["polarity"].drop_duplicates())
    fig, axes = plt.subplots(1, len(polarities), figsize=(6.5 * max(len(polarities), 1), 4.8), dpi=150, sharey=True)
    if len(polarities) == 1:
        axes = [axes]
    for ax, polarity in zip(axes, polarities):
        part = table[table["polarity"] == polarity].copy()
        for phase_name, phase_part in part.groupby("phase_name", sort=True):
            phase_part = phase_part.set_index("region").reindex(regions)
            ax.plot(regions, phase_part[value], marker="o", label=phase_name)
        ax.set_title(str(polarity))
        ax.set_xlabel("region")
        ax.tick_params(axis="x", rotation=25)
        ax.grid(True, color="0.9")
    axes[0].set_ylabel(ylabel)
    axes[-1].legend(fontsize=8, bbox_to_anchor=(1.02, 1.0), loc="upper left")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(fig_dir / filename, bbox_inches="tight")
    plt.close(fig)


def plot_ro_by_phase(fig_dir: Path, ro_phase: pd.DataFrame) -> None:
    if ro_phase.empty:
        return
    fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=150)
    for polarity, part in ro_phase.groupby("polarity", sort=True):
        part = part.sort_values("tau_center")
        ax.plot(part["tau_center"], part["Ro_median"], marker="o", label=f"{polarity} median")
        ax.plot(part["tau_center"], part["Ro_p90"], marker="s", linestyle="--", label=f"{polarity} p90")
        ax.plot(part["tau_center"], part["Ro_p95"], marker="^", linestyle=":", label=f"{polarity} p95")
    ax.axhline(0.1, color="0.35", linewidth=1.0, linestyle="--")
    ax.axhline(0.2, color="0.35", linewidth=1.0, linestyle=":")
    ax.set_xlabel("tau")
    ax.set_ylabel("Ro")
    ax.set_title("Rossby number by lifecycle phase")
    ax.grid(True, color="0.9")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(fig_dir / "Ro_by_phase.png", bbox_inches="tight")
    plt.close(fig)


def plot_ro_vs_nlq(fig_dir: Path, ro_phase: pd.DataFrame, nonlinear_region: pd.DataFrame) -> None:
    if ro_phase.empty or nonlinear_region.empty:
        return
    merged = nonlinear_region.merge(ro_phase[["polarity", "phase_name", "Ro_p90", "Ro_p95"]], on=["polarity", "phase_name"], how="left")
    fig, ax = plt.subplots(figsize=(7.4, 5.4), dpi=150)
    for region, part in merged.groupby("region", sort=False):
        ax.scatter(part["Ro_p90"], part["NL_q_median"], s=48, label=region, alpha=0.82)
    ax.axvline(0.1, color="0.35", linewidth=1.0, linestyle="--")
    ax.axvline(0.2, color="0.35", linewidth=1.0, linestyle=":")
    ax.axhline(1.0, color="#d62728", linewidth=1.0)
    ax.set_yscale("log")
    ax.set_xlabel("Ro p90")
    ax.set_ylabel("NL_q median")
    ax.set_title("QG scale vs finite-amplitude PV nonlinearity")
    ax.grid(True, color="0.9", which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "Ro_vs_NLq_by_region.png", bbox_inches="tight")
    plt.close(fig)


def plot_layer_body_mask(fig_dir: Path, nonlinear: pd.DataFrame, layer_body_profile: pd.DataFrame) -> None:
    if nonlinear.empty or layer_body_profile.empty:
        return
    for (polarity, phase_name), part in nonlinear.groupby(["polarity", "phase_name"], sort=True):
        data = pivot_values(part, "weak_gradient_score")
        values = data.to_numpy(dtype="f8")
        finite = values[np.isfinite(values)]
        vmin, vmax = (np.nanpercentile(finite, [2, 98]) if finite.size else (0.0, 1.0))
        boundary = layer_body_profile[(layer_body_profile["polarity"] == polarity) & (layer_body_profile["phase_name"] == phase_name)].sort_values("depth_m")
        fig, ax = plt.subplots(figsize=(6.4, 4.9), dpi=150)
        mesh = ax.pcolormesh(data.columns.to_numpy(dtype="f8"), data.index.to_numpy(dtype="f8"), values, shading="auto", cmap="Blues", vmin=vmin, vmax=vmax)
        if not boundary.empty:
            ax.plot(boundary["radius_over_object_R_median"], boundary["depth_m"], color="#d62728", linewidth=1.8, label="layer-body median radius")
            ax.plot(boundary["radius_over_object_R_p25"], boundary["depth_m"], color="#d62728", linewidth=0.8, linestyle="--", alpha=0.7)
            ax.plot(boundary["radius_over_object_R_p75"], boundary["depth_m"], color="#d62728", linewidth=0.8, linestyle="--", alpha=0.7)
        ax.invert_yaxis()
        ax.set_xlabel("r/R")
        ax.set_ylabel("depth m")
        ax.set_title(f"{polarity} {phase_name}: layer-body mask over weak-gradient field")
        ax.legend(fontsize=8)
        fig.colorbar(mesh, ax=ax, label="weak gradient score")
        fig.tight_layout()
        fig.savefig(fig_dir / f"{polarity}_{phase_name}_layer_body_mask_rz.png", bbox_inches="tight")
        plt.close(fig)


def plot_core_environment_summary_panel(fig_dir: Path, ro_phase: pd.DataFrame, fj: pd.DataFrame, nonlinear: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=150)
    if not ro_phase.empty:
        for polarity, part in ro_phase.groupby("polarity", sort=True):
            part = part.sort_values("tau_center")
            axes[0, 0].plot(part["tau_center"], part["Ro_p90"], marker="o", label=polarity)
        axes[0, 0].axhline(0.2, color="0.4", linestyle=":")
    axes[0, 0].set_title("Ro p90")
    if not fj.empty:
        use = fj[fj["region"].isin(["core", "outer_ring", "layer_body"])]
        for region, part in use.groupby("region", sort=False):
            axes[0, 1].plot(part["tau_center"], part["median_tilt_fraction"], marker="o", label=region)
    axes[0, 1].set_title("median tilt fraction")
    if not nonlinear.empty:
        for region, part in nonlinear[nonlinear["region"].isin(["core", "outer_ring", "layer_body"])].groupby("region", sort=False):
            axes[1, 0].plot(part["tau_center"], part["Q_n_valid_fraction"], marker="o", label=region)
            axes[1, 1].plot(part["tau_center"], part["NL_q_median"], marker="o", label=region)
    axes[1, 0].set_title("Q_n valid fraction")
    axes[1, 1].set_title("NL_q median")
    axes[1, 1].set_yscale("log")
    for ax in axes.ravel():
        ax.set_xlabel("tau")
        ax.grid(True, color="0.9")
        ax.legend(fontsize=8)
    fig.suptitle("Core-vs-environment sensitivity summary")
    fig.tight_layout()
    fig.savefig(fig_dir / "core_environment_summary_panel.png", bbox_inches="tight")
    plt.close(fig)


def write_core_environment_summary(sens_dir: Path, qg_summary: pd.DataFrame, velocity_status: str) -> None:
    lines = [
        "# Core-vs-environment sensitivity and Ro/QG check",
        "",
        "Old diagnostics are retained as diagnostic_window results over 0 <= r/R <= 2.5.",
        "Core/near/outer/layer_body rows separate vortex-core and surrounding-environment contributions.",
        f"- Representative velocity sensitivity: {velocity_status}",
        "",
        "## QG vs PV Nonlinearity",
        "```csv",
        qg_summary.to_csv(index=False).strip() if not qg_summary.empty else "no_qg_summary",
        "```",
    ]
    (sens_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_core_environment_sensitivity_outputs(output_dir: Path, profiles: pd.DataFrame, nonlinear: pd.DataFrame, radii: dict[str, float], args: argparse.Namespace) -> None:
    sens_dir = output_dir / "core_environment_sensitivity"
    fig_dir = sens_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    objects = load_lifecycle_objects_for_args(args)
    layer_context = load_layer_radius_context(args, objects)
    layer_body = build_layer_body_radius_profile(layer_context)
    ro_depth, ro_phase = build_ro_tables(layer_context)
    fj = build_fj_tilt_region_sensitivity(profiles, layer_body)
    nonlin_region = build_nonlinearity_region_sensitivity(nonlinear, layer_body)
    velocity_region, velocity_status = build_representative_velocity_region_sensitivity(output_dir, radii, layer_body)
    qg_summary = build_qg_pv_summary(ro_phase, nonlin_region)

    layer_body.to_csv(sens_dir / "layer_body_radius_profile.csv", index=False)
    ro_depth.to_csv(sens_dir / "Ro_by_depth.csv", index=False)
    ro_phase.to_csv(sens_dir / "Ro_by_phase.csv", index=False)
    fj.to_csv(sens_dir / "FJ_tilt_region_sensitivity.csv", index=False)
    nonlin_region.to_csv(sens_dir / "nonlinearity_region_sensitivity.csv", index=False)
    velocity_region.to_csv(sens_dir / "representative_velocity_region_sensitivity.csv", index=False)
    qg_summary.to_csv(sens_dir / "QG_vs_PV_nonlinearity_summary.csv", index=False)

    plot_ro_by_phase(fig_dir, ro_phase)
    plot_ro_vs_nlq(fig_dir, ro_phase, nonlin_region)
    plot_region_metric(fig_dir, fj, "median_tilt_fraction", "median tilt fraction", "Tilt fraction by region", "tilt_fraction_by_region.png")
    plot_region_metric(fig_dir, nonlin_region, "Q_n_valid_fraction", "Q_n valid fraction", "Q_n valid by region", "Qn_valid_by_region.png")
    plot_region_metric(fig_dir, nonlin_region, "NL_q_median", "NL_q median", "NL_q by region", "NLq_by_region.png")
    plot_layer_body_mask(fig_dir, nonlinear, layer_body)
    plot_core_environment_summary_panel(fig_dir, ro_phase, fj, nonlin_region)
    write_core_environment_summary(sens_dir, qg_summary, velocity_status)


def run_core_environment_sensitivity_only(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    profiles_path = output_dir / "nondim_vertical_flux_decomposition.parquet"
    nonlinear_path = output_dir / "strong_nonlinearity_diagnostics" / "nonlinearity_profiles.parquet"
    if not profiles_path.exists():
        raise FileNotFoundError(f"Missing required profile parquet: {profiles_path}")
    if not nonlinear_path.exists():
        raise FileNotFoundError(f"Missing required nonlinearity parquet: {nonlinear_path}")
    profiles = pd.read_parquet(profiles_path)
    nonlinear = pd.read_parquet(nonlinear_path)
    radii = lifecycle_radii_from_args(args)
    write_core_environment_sensitivity_outputs(output_dir, profiles, nonlinear, radii, args)
    print(f"Core/environment sensitivity: {output_dir / 'core_environment_sensitivity'}")


MLRW_CATEGORY_ORDER = (
    "strict_mlrw",
    "marginal_mlrw",
    "finite_amplitude_invalid",
    "budget_nonclosure_invalid",
    "weak_gradient_invalid",
)
MLRW_CATEGORY_CODE = {name: i for i, name in enumerate(MLRW_CATEGORY_ORDER)}
MLRW_CATEGORY_COLORS = {
    "strict_mlrw": "#1a9850",
    "marginal_mlrw": "#fee08b",
    "finite_amplitude_invalid": "#d73027",
    "budget_nonclosure_invalid": "#7b3294",
    "weak_gradient_invalid": "#bdbdbd",
}


def build_mlrw_applicability_profiles(output_dir: Path) -> pd.DataFrame:
    nonlinear_path = output_dir / "strong_nonlinearity_diagnostics" / "nonlinearity_profiles.parquet"
    ro_path = output_dir / "core_environment_sensitivity" / "Ro_by_phase.csv"
    if not nonlinear_path.exists():
        raise FileNotFoundError(f"Missing nonlinearity profile parquet: {nonlinear_path}")
    if not ro_path.exists():
        raise FileNotFoundError(f"Missing Ro phase summary: {ro_path}")

    profiles = pd.read_parquet(nonlinear_path).copy()
    ro = pd.read_csv(ro_path)[["polarity", "phase_name", "Ro_p90", "fraction_Ro_lt_0p1"]]
    profiles = profiles.merge(ro, on=["polarity", "phase_name"], how="left")
    qn_valid = profiles["Q_n_valid"].to_numpy(dtype=bool)
    nlq = profiles["NL_q"].to_numpy(dtype="f8")
    residual = np.abs(profiles["wave_activity_residual_J_nd"].to_numpy(dtype="f8"))

    category = np.full(len(profiles), "weak_gradient_invalid", dtype=object)
    strict = qn_valid & np.isfinite(nlq) & np.isfinite(residual) & (nlq <= 1.0) & (residual <= 1.0)
    marginal = qn_valid & ~strict & np.isfinite(nlq) & np.isfinite(residual) & (nlq <= 10.0) & (residual <= 2.0)
    finite_amp = qn_valid & np.isfinite(nlq) & (nlq > 10.0)
    budget_bad = qn_valid & ~strict & ~marginal & ~finite_amp
    category[strict] = "strict_mlrw"
    category[marginal] = "marginal_mlrw"
    category[finite_amp] = "finite_amplitude_invalid"
    category[budget_bad] = "budget_nonclosure_invalid"

    nlq_factor = np.where(np.isfinite(nlq) & (nlq > 0), np.minimum(1.0, 1.0 / nlq), 0.0)
    residual_factor = np.where(np.isfinite(residual) & (residual > 0), np.minimum(1.0, 1.0 / residual), 1.0)
    score = np.where(qn_valid, nlq_factor * residual_factor, 0.0)

    profiles["mlrw_category"] = category
    profiles["mlrw_category_code"] = [MLRW_CATEGORY_CODE[str(name)] for name in category]
    profiles["mlrw_score"] = score
    profiles["abs_wave_activity_residual_J_nd"] = residual
    profiles["qg_invalid"] = profiles["Ro_p90"].to_numpy(dtype="f8") >= 0.1
    return profiles


def mlrw_fraction_row(part: pd.DataFrame) -> dict[str, float]:
    total = int(len(part))
    row: dict[str, float] = {"n_bins": total}
    if total == 0:
        for category in MLRW_CATEGORY_ORDER:
            row[f"fraction_{category}"] = np.nan
        row["fraction_strict_plus_marginal"] = np.nan
        row["median_mlrw_score"] = np.nan
        return row
    for category in MLRW_CATEGORY_ORDER:
        row[f"fraction_{category}"] = float(np.mean(part["mlrw_category"].to_numpy(dtype=object) == category))
    row["fraction_strict_plus_marginal"] = row["fraction_strict_mlrw"] + row["fraction_marginal_mlrw"]
    row["median_mlrw_score"] = float(np.nanmedian(part["mlrw_score"].to_numpy(dtype="f8")))
    return row


def build_mlrw_metrics(profiles: pd.DataFrame, layer_body: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows_phase = []
    rows_depth = []
    rows_region = []
    rows_pol = []

    for (polarity, phase_name), part in profiles.groupby(["polarity", "phase_name"], sort=True):
        row = {
            "polarity": polarity,
            "phase_name": phase_name,
            "tau_center": TAU_CENTERS[str(phase_name)],
            "Ro_p90": float(np.nanmedian(part["Ro_p90"].to_numpy(dtype="f8"))),
            "qg_invalid": bool(np.nanmedian(part["Ro_p90"].to_numpy(dtype="f8")) >= 0.1),
        }
        row.update(mlrw_fraction_row(part))
        rows_phase.append(row)

        masks = region_masks(part, layer_body)
        for region in REGION_ORDER:
            sub = part.loc[masks[region]]
            row_region = {"polarity": polarity, "phase_name": phase_name, "tau_center": TAU_CENTERS[str(phase_name)], "region": region}
            row_region.update(mlrw_fraction_row(sub))
            rows_region.append(row_region)

        for depth_m, depth_part in part.groupby("depth_m", sort=True):
            row_depth = {"polarity": polarity, "phase_name": phase_name, "tau_center": TAU_CENTERS[str(phase_name)], "depth_m": float(depth_m)}
            row_depth.update(mlrw_fraction_row(depth_part))
            rows_depth.append(row_depth)

    for polarity, part in profiles.groupby("polarity", sort=True):
        row_pol = {"polarity": polarity}
        row_pol.update(mlrw_fraction_row(part))
        row_pol["Ro_p90_median"] = float(np.nanmedian(part["Ro_p90"].to_numpy(dtype="f8")))
        row_pol["qg_invalid_any_phase"] = bool(np.nanmax(part["Ro_p90"].to_numpy(dtype="f8")) >= 0.1)
        rows_pol.append(row_pol)

    return pd.DataFrame(rows_phase), pd.DataFrame(rows_region), pd.DataFrame(rows_depth), pd.DataFrame(rows_pol)


def mlrw_category_cmap() -> tuple[ListedColormap, BoundaryNorm]:
    colors = [MLRW_CATEGORY_COLORS[name] for name in MLRW_CATEGORY_ORDER]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(-0.5, len(colors) + 0.5, 1.0), cmap.N)
    return cmap, norm


def plot_mlrw_applicability_rz(fig_dir: Path, prefix: str, title_prefix: str, part: pd.DataFrame) -> None:
    data = pivot_values(part, "mlrw_category_code")
    cmap, norm = mlrw_category_cmap()
    fig, ax = plt.subplots(figsize=(7.6, 6.0), dpi=150)
    mesh = ax.pcolormesh(data.columns.to_numpy(dtype="f8"), data.index.to_numpy(dtype="f8"), data.to_numpy(dtype="f8"), shading="auto", cmap=cmap, norm=norm)
    ax.invert_yaxis()
    ax.set_xlabel("r/R")
    ax.set_ylabel("depth m")
    ax.set_title(f"{title_prefix}: MLRW applicability")
    handles = [Patch(facecolor=MLRW_CATEGORY_COLORS[name], label=name) for name in MLRW_CATEGORY_ORDER]
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / f"{prefix}_mlrw_applicability_rz.png", bbox_inches="tight")
    plt.close(fig)


def plot_mlrw_score_rz(fig_dir: Path, prefix: str, title_prefix: str, part: pd.DataFrame) -> None:
    data = pivot_values(part, "mlrw_score")
    fig, ax = plt.subplots(figsize=(7.2, 5.8), dpi=150)
    mesh = ax.pcolormesh(data.columns.to_numpy(dtype="f8"), data.index.to_numpy(dtype="f8"), data.to_numpy(dtype="f8"), shading="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    ax.invert_yaxis()
    ax.set_xlabel("r/R")
    ax.set_ylabel("depth m")
    ax.set_title(f"{title_prefix}: MLRW applicability score")
    fig.colorbar(mesh, ax=ax, label="S_MLRW")
    fig.tight_layout()
    fig.savefig(fig_dir / f"{prefix}_mlrw_score_rz.png", bbox_inches="tight")
    plt.close(fig)


def plot_mlrw_terms_rz_panel(fig_dir: Path, prefix: str, title_prefix: str, part: pd.DataFrame) -> None:
    fields = [
        ("Q_n_valid", "Q_n valid", "Blues", None, None),
        ("NL_q", "log10(1+NL_q)", "magma", None, None),
        ("abs_wave_activity_residual_J_nd", "|J_T residual|", "inferno", None, None),
        ("mlrw_score", "S_MLRW", "viridis", 0.0, 1.0),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.2), dpi=145, sharex=True, sharey=True)
    for ax, (field, title, cmap, vmin, vmax) in zip(axes.ravel(), fields):
        data = pivot_values(part, field)
        values = data.to_numpy(dtype="f8")
        if field == "Q_n_valid":
            values = values.astype(bool).astype(float)
            vmin, vmax = 0.0, 1.0
        elif field == "NL_q":
            values = np.log10(1.0 + values)
            finite = values[np.isfinite(values)]
            if finite.size:
                vmin, vmax = np.nanpercentile(finite, [2, 98])
        elif field == "abs_wave_activity_residual_J_nd":
            finite = values[np.isfinite(values)]
            if finite.size:
                vmin, vmax = 0.0, np.nanpercentile(finite, 98)
        mesh = ax.pcolormesh(data.columns.to_numpy(dtype="f8"), data.index.to_numpy(dtype="f8"), values, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.invert_yaxis()
        ax.set_title(title)
        fig.colorbar(mesh, ax=ax, shrink=0.82)
    for ax in axes[:, 0]:
        ax.set_ylabel("depth m")
    for ax in axes[-1, :]:
        ax.set_xlabel("r/R")
    fig.suptitle(f"{title_prefix}: MLRW prerequisite terms")
    fig.tight_layout()
    fig.savefig(fig_dir / f"{prefix}_mlrw_terms_rz_panel.png", bbox_inches="tight")
    plt.close(fig)


def plot_mlrw_applicability_by_phase(fig_dir: Path, metrics: pd.DataFrame) -> None:
    phase_order = [name for name in PHASE_NAMES if name in set(metrics["phase_name"])]
    polarities = list(metrics["polarity"].drop_duplicates())
    fig, axes = plt.subplots(len(polarities), 1, figsize=(9, 3.6 * max(len(polarities), 1)), dpi=150, sharex=True, squeeze=False)
    for ax, polarity in zip(axes.ravel(), polarities):
        part = metrics[metrics["polarity"] == polarity].set_index("phase_name").reindex(phase_order)
        x = np.arange(len(part))
        bottom = np.zeros(len(part), dtype="f8")
        for category in MLRW_CATEGORY_ORDER:
            values = part[f"fraction_{category}"].to_numpy(dtype="f8")
            ax.bar(x, values, bottom=bottom, label=category, color=MLRW_CATEGORY_COLORS[category], alpha=0.9)
            bottom += np.nan_to_num(values, nan=0.0)
        ax.set_title(f"{polarity}: MLRW applicability by phase")
        ax.set_ylabel("fraction")
        ax.set_ylim(0, 1)
        ax.grid(True, color="0.92", axis="y")
        ax.legend(fontsize=7, ncol=2)
    axes[-1, 0].set_xticks(np.arange(len(phase_order)), phase_order)
    fig.tight_layout()
    fig.savefig(fig_dir / "mlrw_applicability_by_phase.png", bbox_inches="tight")
    plt.close(fig)


def plot_mlrw_applicability_by_region(fig_dir: Path, region: pd.DataFrame) -> None:
    use = region[region["region"].isin(REGION_ORDER)].copy()
    if use.empty:
        return
    grouped = use.groupby(["polarity", "region"], as_index=False)[[f"fraction_{cat}" for cat in MLRW_CATEGORY_ORDER]].mean()
    polarities = list(grouped["polarity"].drop_duplicates())
    fig, axes = plt.subplots(1, len(polarities), figsize=(6.3 * max(len(polarities), 1), 4.8), dpi=150, sharey=True, squeeze=False)
    for ax, polarity in zip(axes.ravel(), polarities):
        part = grouped[grouped["polarity"] == polarity].set_index("region").reindex(REGION_ORDER)
        x = np.arange(len(part))
        bottom = np.zeros(len(part), dtype="f8")
        for category in MLRW_CATEGORY_ORDER:
            values = part[f"fraction_{category}"].to_numpy(dtype="f8")
            ax.bar(x, values, bottom=bottom, label=category, color=MLRW_CATEGORY_COLORS[category], alpha=0.9)
            bottom += np.nan_to_num(values, nan=0.0)
        ax.set_title(polarity)
        ax.set_xticks(x, REGION_ORDER, rotation=25, ha="right")
        ax.grid(True, color="0.92", axis="y")
    axes[0, 0].set_ylabel("fraction")
    axes[0, -1].legend(fontsize=7, bbox_to_anchor=(1.02, 1.0), loc="upper left")
    fig.suptitle("MLRW applicability by core/environment region")
    fig.tight_layout()
    fig.savefig(fig_dir / "mlrw_applicability_by_region.png", bbox_inches="tight")
    plt.close(fig)


def plot_mlrw_depth_profile(fig_dir: Path, depth_table: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 6.0), dpi=150)
    for polarity, part in depth_table.groupby("polarity", sort=True):
        prof = part.groupby("depth_m", as_index=False)[["fraction_strict_plus_marginal", "fraction_weak_gradient_invalid", "fraction_finite_amplitude_invalid"]].mean().sort_values("depth_m")
        ax.plot(prof["fraction_strict_plus_marginal"], prof["depth_m"], marker="o", markersize=2, label=f"{polarity} strict+marginal")
        ax.plot(prof["fraction_weak_gradient_invalid"], prof["depth_m"], linestyle="--", label=f"{polarity} weak gradient")
    ax.invert_yaxis()
    ax.set_ylim(200, 0)
    ax.set_xlabel("fraction")
    ax.set_ylabel("depth m")
    ax.set_title("MLRW applicability depth profile, upper 200 m")
    ax.grid(True, color="0.9")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "mlrw_applicability_depth_profile.png", bbox_inches="tight")
    plt.close(fig)


def plot_mlrw_valid_zone_vs_nonlinear_zone(fig_dir: Path, metrics: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 5.2), dpi=150)
    for polarity, part in metrics.groupby("polarity", sort=True):
        part = part.sort_values("tau_center")
        ax.plot(part["tau_center"], part["fraction_strict_plus_marginal"], marker="o", label=f"{polarity} MLRW usable")
        ax.plot(part["tau_center"], part["fraction_finite_amplitude_invalid"], marker="s", linestyle="--", label=f"{polarity} finite amplitude invalid")
        ax.plot(part["tau_center"], part["fraction_weak_gradient_invalid"], marker="^", linestyle=":", label=f"{polarity} weak gradient invalid")
    ax.set_xlabel("tau")
    ax.set_ylabel("fraction")
    ax.set_ylim(0, 1)
    ax.set_title("MLRW valid zone vs invalid mechanisms")
    ax.grid(True, color="0.9")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(fig_dir / "mlrw_valid_zone_vs_nonlinear_zone.png", bbox_inches="tight")
    plt.close(fig)


def nearest_rz_to_xy_layers(part: pd.DataFrame, field: str, depth: np.ndarray, radial_grid: np.ndarray, depth_indices: list[int], r_norm: np.ndarray) -> np.ndarray:
    data = pivot_values(part, field).reindex(index=depth)
    matrix = data.to_numpy(dtype="f8")
    nearest = np.abs(radial_grid[:, None] - r_norm.ravel()[None, :]).argmin(axis=0)
    nearest = nearest.reshape(r_norm.shape)
    layers = []
    for depth_index in depth_indices:
        row = matrix[depth_index]
        values = row[nearest]
        values = np.where(r_norm <= float(np.nanmax(radial_grid)), values, np.nan)
        layers.append(values)
    return np.asarray(layers)


def mlrw_category_facecolors(code_layers: np.ndarray) -> np.ndarray:
    rgba_lookup = np.asarray([matplotlib.colors.to_rgba(MLRW_CATEGORY_COLORS[name], alpha=0.46) for name in MLRW_CATEGORY_ORDER])
    colors = np.zeros(code_layers.shape + (4,), dtype="f8")
    valid = np.isfinite(code_layers)
    codes = np.clip(np.rint(np.nan_to_num(code_layers, nan=0.0)).astype(int), 0, len(MLRW_CATEGORY_ORDER) - 1)
    colors[valid] = rgba_lookup[codes[valid]]
    colors[~valid] = (0.0, 0.0, 0.0, 0.0)
    return colors


def plot_mlrw_stack_outputs(output_dir: Path, profiles: pd.DataFrame, axis_dir: Path, radii: dict[str, float], args: argparse.Namespace) -> None:
    stack_dir = output_dir / "mlrw_applicability" / "stack_3d"
    stack_dir.mkdir(parents=True, exist_ok=True)
    x, y, r_norm = make_xy_grid(float(args.nonlinearity_stack_xy_extent), int(args.nonlinearity_stack_grid_size))
    requested_depths = parse_float_list(args.nonlinearity_stack_depth_levels)
    category_cmap, category_norm = mlrw_category_cmap()
    for (polarity, phase_name), part in profiles.groupby(["polarity", "phase_name"], sort=True):
        prefix = f"{polarity}_{phase_name}"
        title_prefix = f"{polarity} {phase_name}"
        depth = np.sort(part["depth_m"].unique().astype("f8"))
        depth_indices = nearest_depth_indices(depth, requested_depths)
        radial_grid = np.sort(part["r_over_R"].unique().astype("f8"))
        x_layers, y_layers, axis_x, axis_y = nonlinearity_axis_layers(part, axis_dir, radii, depth, depth_indices, x, y)

        code_layers = nearest_rz_to_xy_layers(part, "mlrw_category_code", depth, radial_grid, depth_indices, r_norm)
        category_colors = mlrw_category_facecolors(code_layers)
        plot_nonlinearity_stack_png(
            stack_dir,
            prefix,
            title_prefix,
            depth,
            depth_indices,
            x_layers,
            y_layers,
            axis_x,
            axis_y,
            category_colors,
            colorbar_norm=category_norm,
            colorbar_cmap=category_cmap,
            colorbar_label="MLRW category",
            suffix="mlrw_applicability_stack_3d",
            colorbar_ticks=list(range(len(MLRW_CATEGORY_ORDER))),
            colorbar_ticklabels=list(MLRW_CATEGORY_ORDER),
        )

        score_layers = interpolate_rz_to_xy_layers(part, "mlrw_score", depth, radial_grid, depth_indices, r_norm)
        score_colors, score_norm, _ = rgba_from_scalar_layers(score_layers, "viridis")
        score_colors[..., 3] = np.where(np.isfinite(score_layers), 0.46, 0.0)
        plot_nonlinearity_stack_png(stack_dir, prefix, title_prefix, depth, depth_indices, x_layers, y_layers, axis_x, axis_y, score_colors, colorbar_norm=score_norm, colorbar_cmap="viridis", colorbar_label="S_MLRW", suffix="mlrw_score_stack_3d")


def plot_mlrw_structure_mosaic(fig_dir: Path, profiles: pd.DataFrame) -> None:
    phase_order = [name for name in PHASE_NAMES if name in set(profiles["phase_name"])]
    polarities = list(profiles["polarity"].drop_duplicates())
    cmap, norm = mlrw_category_cmap()
    fig, axes = plt.subplots(len(polarities), len(phase_order), figsize=(3.0 * len(phase_order), 2.8 * len(polarities)), dpi=145, squeeze=False)
    for row, polarity in enumerate(polarities):
        for col, phase_name in enumerate(phase_order):
            ax = axes[row, col]
            part = profiles[(profiles["polarity"] == polarity) & (profiles["phase_name"] == phase_name)]
            if part.empty:
                ax.axis("off")
                continue
            data = pivot_values(part, "mlrw_category_code")
            ax.pcolormesh(data.columns.to_numpy(dtype="f8"), data.index.to_numpy(dtype="f8"), data.to_numpy(dtype="f8"), shading="auto", cmap=cmap, norm=norm)
            ax.invert_yaxis()
            ax.set_ylim(120, 0)
            ax.set_title(f"{polarity}\n{phase_name}", fontsize=9)
            if col == 0:
                ax.set_ylabel("depth m")
            if row == len(polarities) - 1:
                ax.set_xlabel("r/R")
    handles = [Patch(facecolor=MLRW_CATEGORY_COLORS[name], label=name) for name in MLRW_CATEGORY_ORDER]
    fig.legend(handles=handles, loc="upper center", ncol=3, fontsize=8)
    fig.suptitle("MLRW applicability structure mosaic, upper 120 m", y=1.05)
    fig.tight_layout()
    fig.savefig(fig_dir / "mlrw_applicability_structure_mosaic.png", bbox_inches="tight")
    plt.close(fig)


def write_mlrw_applicability_summary(mlrw_dir: Path, metrics: pd.DataFrame, by_region: pd.DataFrame, by_polarity: pd.DataFrame) -> None:
    lines = [
        "# MLRW applicability diagnostics",
        "",
        "This post-processing diagnostic locates where a small-amplitude linear Rossby-wave/MLRW interpretation remains locally applicable inside the representative tilted vortex.",
        "The classification is based on Q_n validity, NL_q amplitude, and the nondimensional J_T wave-action residual. Ro is retained as a phase-level QG prerequisite check.",
        "",
        "## Main result",
    ]
    for _, row in by_polarity.iterrows():
        lines.append(
            f"- {row['polarity']}: strict={row['fraction_strict_mlrw']:.3g}, marginal={row['fraction_marginal_mlrw']:.3g}, "
            f"finite-amplitude invalid={row['fraction_finite_amplitude_invalid']:.3g}, weak-gradient invalid={row['fraction_weak_gradient_invalid']:.3g}, "
            f"Ro_p90_median={row['Ro_p90_median']:.3g}."
        )
    lines.extend(
        [
            "",
            "Interpretation: strict_mlrw is the only region where the small-amplitude linear wave-action premise is satisfied. marginal_mlrw keeps a Q_n recovery mechanism but is already finite-amplitude enough that MLRW should be treated as qualitative guidance. finite_amplitude_invalid and weak_gradient_invalid mark where the whole-vortex linear MLRW premise fails.",
            "",
            "## Metrics By Phase",
            "```csv",
            metrics.to_csv(index=False).strip(),
            "```",
            "",
            "## Metrics By Region",
            "```csv",
            by_region.to_csv(index=False).strip(),
            "```",
            "",
            "## Figures",
            "- `figures/*_mlrw_applicability_rz.png`",
            "- `figures/*_mlrw_score_rz.png`",
            "- `figures/*_mlrw_terms_rz_panel.png`",
            "- `stack_3d/*_mlrw_applicability_stack_3d.png`",
            "- `stack_3d/*_mlrw_score_stack_3d.png`",
        ]
    )
    (mlrw_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_mlrw_applicability_outputs(output_dir: Path, axis_dir: Path, radii: dict[str, float], args: argparse.Namespace) -> None:
    mlrw_dir = output_dir / "mlrw_applicability"
    fig_dir = mlrw_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    profiles = build_mlrw_applicability_profiles(output_dir)
    layer_body_path = output_dir / "core_environment_sensitivity" / "layer_body_radius_profile.csv"
    if not layer_body_path.exists():
        raise FileNotFoundError(f"Missing layer-body radius profile: {layer_body_path}")
    layer_body = pd.read_csv(layer_body_path)
    metrics, by_region, by_depth, by_polarity = build_mlrw_metrics(profiles, layer_body)

    profiles.to_parquet(mlrw_dir / "mlrw_applicability_profiles.parquet", index=False)
    profiles.to_csv(mlrw_dir / "mlrw_applicability_profiles.csv", index=False)
    metrics.to_csv(mlrw_dir / "mlrw_applicability_metrics.csv", index=False)
    by_region.to_csv(mlrw_dir / "mlrw_applicability_by_region.csv", index=False)
    by_depth.to_csv(mlrw_dir / "mlrw_applicability_by_depth.csv", index=False)
    by_polarity.to_csv(mlrw_dir / "mlrw_applicability_by_polarity.csv", index=False)

    for (polarity, phase_name), part in profiles.groupby(["polarity", "phase_name"], sort=True):
        prefix = f"{polarity}_{phase_name}"
        title_prefix = f"{polarity} {phase_name}"
        plot_mlrw_applicability_rz(fig_dir, prefix, title_prefix, part)
        plot_mlrw_score_rz(fig_dir, prefix, title_prefix, part)
        plot_mlrw_terms_rz_panel(fig_dir, prefix, title_prefix, part)
    plot_mlrw_applicability_by_phase(fig_dir, metrics)
    plot_mlrw_applicability_by_region(fig_dir, by_region)
    plot_mlrw_depth_profile(fig_dir, by_depth)
    plot_mlrw_valid_zone_vs_nonlinear_zone(fig_dir, metrics)
    plot_mlrw_structure_mosaic(fig_dir, profiles)
    plot_mlrw_stack_outputs(output_dir, profiles, axis_dir, radii, args)
    write_mlrw_applicability_summary(mlrw_dir, metrics, by_region, by_polarity)


def run_mlrw_applicability_only(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    radii = lifecycle_radii_from_args(args)
    write_mlrw_applicability_outputs(output_dir, Path(args.axis_dir), radii, args)
    print(f"MLRW applicability diagnostics: {output_dir / 'mlrw_applicability'}")


PHASE_ORDER = tuple(TAU_CENTERS.keys())


def sort_lifecycle_table(table: pd.DataFrame) -> pd.DataFrame:
    if table.empty:
        return table
    out = table.copy()
    out["phase_order"] = out["phase_name"].map({name: i for i, name in enumerate(PHASE_ORDER)})
    return out.sort_values(["polarity", "phase_order"]).drop(columns=["phase_order"])


def phase_derivative(values: np.ndarray, tau: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype="f8")
    tau = np.asarray(tau, dtype="f8")
    if values.size < 2 or tau.size < 2:
        return np.full_like(values, np.nan, dtype="f8")
    return np.gradient(values, tau, edge_order=1)


def peak_phase(rowset: pd.DataFrame, column: str, use_abs: bool = False) -> tuple[str, float, float]:
    if rowset.empty or column not in rowset.columns:
        return "", np.nan, np.nan
    values = rowset[column].to_numpy(dtype="f8")
    if use_abs:
        values = np.abs(values)
    finite = np.isfinite(values)
    if not np.any(finite):
        return "", np.nan, np.nan
    idx = np.where(finite)[0][int(np.nanargmax(values[finite]))]
    row = rowset.iloc[idx]
    raw_value = float(row[column])
    return str(row["phase_name"]), float(row["tau_center"]), raw_value


def build_tilt_evolution_phase_metrics(output_dir: Path) -> pd.DataFrame:
    tilt_path = output_dir / "tilt_term_mechanism" / "tilt_term_mechanism_metrics.csv"
    jt_path = output_dir / "wave_action_total_flux" / "wave_action_total_flux_metrics.csv"
    if not tilt_path.exists():
        raise FileNotFoundError(f"Missing tilt mechanism metrics: {tilt_path}")
    if not jt_path.exists():
        raise FileNotFoundError(f"Missing wave-action total-flux metrics: {jt_path}")
    tilt = pd.read_csv(tilt_path)
    jt = pd.read_csv(jt_path)
    tilt_cols = [
        "polarity",
        "phase_name",
        "tau_center",
        "median_axis_slope_mag",
        "rms_Fz_tilt_correction",
        "median_tilt_fraction",
        "rms_raw_tilt_flux",
        "n_objects",
        "n_dates",
    ]
    jt_cols = [
        "polarity",
        "phase_name",
        "tau_center",
        "improvement_fraction",
        "residual_F_rms",
        "residual_J_rms",
        "Q_n_valid_fraction",
        "valid_bins",
    ]
    missing = [c for c in tilt_cols if c not in tilt.columns] + [c for c in jt_cols if c not in jt.columns]
    if missing:
        raise KeyError(f"Missing required tilt-evolution columns: {missing}")
    phase = tilt[tilt_cols].merge(jt[jt_cols], on=["polarity", "phase_name", "tau_center"], how="inner")
    phase = sort_lifecycle_table(phase)
    rows = []
    for polarity, part in phase.groupby("polarity", sort=False):
        part = sort_lifecycle_table(part).copy()
        part["partial_tau_axis_slope"] = phase_derivative(part["median_axis_slope_mag"].to_numpy(dtype="f8"), part["tau_center"].to_numpy(dtype="f8"))
        part["partial_tau_rms_Fz_tilt_correction"] = phase_derivative(part["rms_Fz_tilt_correction"].to_numpy(dtype="f8"), part["tau_center"].to_numpy(dtype="f8"))
        part["abs_median_tilt_fraction"] = np.abs(part["median_tilt_fraction"].to_numpy(dtype="f8"))
        part["partial_tau_abs_tilt_fraction"] = phase_derivative(part["abs_median_tilt_fraction"].to_numpy(dtype="f8"), part["tau_center"].to_numpy(dtype="f8"))
        rows.append(part)
    return sort_lifecycle_table(pd.concat(rows, ignore_index=True) if rows else phase)


def interpret_tilt_evolution_row(corr_daxis_jt: float, corr_axis_abs_tilt: float, corr_daxis_fz: float, mean_qn: float) -> str:
    labels = []
    if np.isfinite(corr_daxis_jt) and corr_daxis_jt >= 0.4:
        labels.append("JT_improvement_tracks_tilt_tendency")
    elif np.isfinite(corr_daxis_jt) and corr_daxis_jt <= -0.4:
        labels.append("JT_improvement_opposes_tilt_tendency")
    else:
        labels.append("JT_improvement_mixed")
    if np.isfinite(corr_axis_abs_tilt) and corr_axis_abs_tilt >= 0.4:
        labels.append("tilt_cancellation_tracks_tilt_amplitude")
    elif np.isfinite(corr_axis_abs_tilt) and corr_axis_abs_tilt <= -0.4:
        labels.append("tilt_cancellation_opposes_tilt_amplitude")
    else:
        labels.append("tilt_cancellation_not_amplitude_proxy")
    if np.isfinite(corr_daxis_fz) and corr_daxis_fz <= -0.4:
        labels.append("Fz_tilt_negative_feedback_like")
    elif np.isfinite(corr_daxis_fz) and corr_daxis_fz >= 0.4:
        labels.append("Fz_tilt_in_phase_with_tilt_growth")
    else:
        labels.append("Fz_tilt_phase_mixed")
    if np.isfinite(mean_qn) and mean_qn < 0.15:
        labels.append("low_Qn_valid_caution")
    return ";".join(labels)


def build_tilt_evolution_correlation_metrics(phase: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for polarity, part in phase.groupby("polarity", sort=False):
        part = sort_lifecycle_table(part)
        axis = part["median_axis_slope_mag"].to_numpy(dtype="f8")
        daxis = part["partial_tau_axis_slope"].to_numpy(dtype="f8")
        fz = part["rms_Fz_tilt_correction"].to_numpy(dtype="f8")
        abs_tilt = part["abs_median_tilt_fraction"].to_numpy(dtype="f8")
        jt = part["improvement_fraction"].to_numpy(dtype="f8")
        axis_peak_phase, axis_peak_tau, axis_peak_value = peak_phase(part, "median_axis_slope_mag")
        fz_peak_phase, fz_peak_tau, fz_peak_value = peak_phase(part, "rms_Fz_tilt_correction")
        cancel_peak_phase, cancel_peak_tau, cancel_peak_value = peak_phase(part, "median_tilt_fraction", use_abs=True)
        jt_peak_phase, jt_peak_tau, jt_peak_value = peak_phase(part, "improvement_fraction")
        corr_daxis_jt = metric_corr(daxis, jt)
        corr_axis_abs = metric_corr(axis, abs_tilt)
        corr_daxis_fz = metric_corr(daxis, fz)
        mean_qn = float(np.nanmean(part["Q_n_valid_fraction"].to_numpy(dtype="f8")))
        rows.append(
            {
                "polarity": polarity,
                "n_phase": int(len(part)),
                "mean_Q_n_valid_fraction": mean_qn,
                "corr_axis_slope_rms_Fz_tilt_correction": metric_corr(axis, fz),
                "corr_partial_tau_axis_slope_rms_Fz_tilt_correction": corr_daxis_fz,
                "corr_axis_slope_abs_tilt_fraction": corr_axis_abs,
                "corr_partial_tau_axis_slope_abs_tilt_fraction": metric_corr(daxis, abs_tilt),
                "corr_axis_slope_improvement_fraction": metric_corr(axis, jt),
                "corr_partial_tau_axis_slope_improvement_fraction": corr_daxis_jt,
                "axis_slope_peak_phase": axis_peak_phase,
                "axis_slope_peak_tau": axis_peak_tau,
                "axis_slope_peak_value": axis_peak_value,
                "Fz_tilt_peak_phase": fz_peak_phase,
                "Fz_tilt_peak_tau": fz_peak_tau,
                "Fz_tilt_peak_value": fz_peak_value,
                "tilt_cancellation_peak_phase": cancel_peak_phase,
                "tilt_cancellation_peak_tau": cancel_peak_tau,
                "tilt_cancellation_peak_value": cancel_peak_value,
                "JT_improvement_peak_phase": jt_peak_phase,
                "JT_improvement_peak_tau": jt_peak_tau,
                "JT_improvement_peak_value": jt_peak_value,
                "Fz_tilt_peak_tau_minus_axis_peak_tau": fz_peak_tau - axis_peak_tau if np.isfinite(fz_peak_tau) and np.isfinite(axis_peak_tau) else np.nan,
                "tilt_cancellation_peak_tau_minus_axis_peak_tau": cancel_peak_tau - axis_peak_tau if np.isfinite(cancel_peak_tau) and np.isfinite(axis_peak_tau) else np.nan,
                "JT_peak_tau_minus_axis_peak_tau": jt_peak_tau - axis_peak_tau if np.isfinite(jt_peak_tau) and np.isfinite(axis_peak_tau) else np.nan,
                "interpretation_label": interpret_tilt_evolution_row(corr_daxis_jt, corr_axis_abs, corr_daxis_fz, mean_qn),
            }
        )
    return pd.DataFrame(rows)


def build_tilt_evolution_region_sensitivity(output_dir: Path, phase: pd.DataFrame) -> pd.DataFrame:
    path = output_dir / "core_environment_sensitivity" / "FJ_tilt_region_sensitivity.csv"
    if not path.exists():
        return pd.DataFrame()
    region = pd.read_csv(path)
    keep = [
        "polarity",
        "phase_name",
        "tau_center",
        "partial_tau_axis_slope",
        "improvement_fraction",
        "Q_n_valid_fraction",
    ]
    merged = region.merge(phase[keep], on=["polarity", "phase_name", "tau_center"], how="left")
    if "median_tilt_fraction" in merged.columns:
        merged["abs_median_tilt_fraction"] = np.abs(merged["median_tilt_fraction"].to_numpy(dtype="f8"))
    return merged


def plot_tilt_axis_vs_fz_tilt(fig_dir: Path, phase: pd.DataFrame) -> None:
    polarities = list(phase["polarity"].drop_duplicates())
    fig, axes = plt.subplots(len(polarities), 1, figsize=(8.5, 4.2 * max(len(polarities), 1)), dpi=150, squeeze=False)
    for ax, polarity in zip(axes.ravel(), polarities):
        part = sort_lifecycle_table(phase[phase["polarity"] == polarity])
        x = part["tau_center"].to_numpy(dtype="f8")
        ax.plot(x, part["median_axis_slope_mag"], marker="o", color="black", label="axis slope |Rc,z|")
        ax.bar(x, part["partial_tau_axis_slope"], width=0.045, color="0.75", alpha=0.65, label="partial_tau |Rc,z|")
        ax.set_ylabel("axis slope / tendency")
        ax2 = ax.twinx()
        ax2.plot(x, part["rms_Fz_tilt_correction"], marker="s", color="#b2182b", label="rms Fz tilt correction")
        ax2.set_ylabel("rms Fz^(tilt)")
        ax.set_title(f"{polarity}: tilt axis vs Fz tilt correction")
        ax.set_xlabel("tau")
        ax.grid(True, color="0.9")
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(fig_dir / "tilt_axis_vs_Fz_tilt_by_phase.png", bbox_inches="tight")
    plt.close(fig)


def plot_tilt_fraction_vs_tilt_evolution(fig_dir: Path, phase: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), dpi=150, sharex=True)
    for polarity, part in phase.groupby("polarity", sort=False):
        part = sort_lifecycle_table(part)
        axes[0].plot(part["tau_center"], part["median_tilt_fraction"], marker="o", label=polarity)
        axes[1].plot(part["tau_center"], part["abs_median_tilt_fraction"], marker="o", label=f"{polarity} |fraction|")
        axes[1].plot(part["tau_center"], part["median_axis_slope_mag"] / np.nanmax(part["median_axis_slope_mag"]), marker="s", linestyle="--", label=f"{polarity} axis norm")
    axes[0].axhline(0, color="0.35", linewidth=0.8)
    axes[0].set_title("signed tilt fraction")
    axes[1].set_title("cancellation strength vs normalized tilt")
    for ax in axes:
        ax.set_xlabel("tau")
        ax.grid(True, color="0.9")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "tilt_fraction_vs_tilt_evolution.png", bbox_inches="tight")
    plt.close(fig)


def plot_jt_improvement_vs_tilt_evolution(fig_dir: Path, phase: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), dpi=150)
    for polarity, part in phase.groupby("polarity", sort=False):
        part = sort_lifecycle_table(part)
        axes[0].plot(part["tau_center"], part["improvement_fraction"], marker="o", label=polarity)
        axes[0].plot(part["tau_center"], part["median_axis_slope_mag"] / np.nanmax(part["median_axis_slope_mag"]), marker="s", linestyle="--", label=f"{polarity} axis norm")
        axes[1].scatter(part["partial_tau_axis_slope"], part["improvement_fraction"], label=polarity)
        for row in part.itertuples(index=False):
            axes[1].annotate(str(row.phase_name), (float(row.partial_tau_axis_slope), float(row.improvement_fraction)), fontsize=7)
    axes[0].set_title("JT improvement and tilt amplitude")
    axes[0].set_xlabel("tau")
    axes[0].set_ylabel("improvement fraction / normalized axis")
    axes[1].set_title("JT improvement vs tilt tendency")
    axes[1].set_xlabel("partial_tau |Rc,z|")
    axes[1].set_ylabel("JT improvement fraction")
    for ax in axes:
        ax.grid(True, color="0.9")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "JT_improvement_vs_tilt_evolution.png", bbox_inches="tight")
    plt.close(fig)


def plot_phase_lag_summary(fig_dir: Path, corr: pd.DataFrame) -> None:
    metrics = [
        ("axis_slope_peak_tau", "axis peak"),
        ("Fz_tilt_peak_tau", "Fz tilt peak"),
        ("tilt_cancellation_peak_tau", "max cancellation"),
        ("JT_improvement_peak_tau", "JT improvement peak"),
    ]
    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=150)
    y_positions = np.arange(len(metrics))
    markers = {"anticyclonic": "o", "cyclonic": "s"}
    colors = {"anticyclonic": "#2166ac", "cyclonic": "#d95f02"}
    for _, row in corr.iterrows():
        for y, (col, label) in zip(y_positions, metrics):
            ax.scatter(float(row[col]), y, marker=markers.get(str(row["polarity"]), "o"), s=80, color=colors.get(str(row["polarity"]), "0.3"), label=str(row["polarity"]) if y == 0 else None)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([label for _, label in metrics])
    ax.set_xticks([TAU_CENTERS[p] for p in PHASE_ORDER])
    ax.set_xticklabels(PHASE_ORDER, rotation=25)
    ax.set_xlabel("lifecycle phase tau")
    ax.set_title("Peak-phase lag summary")
    ax.grid(True, axis="x", color="0.9")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "phase_lag_summary.png", bbox_inches="tight")
    plt.close(fig)


def plot_region_tilt_evolution_sensitivity(fig_dir: Path, region: pd.DataFrame) -> None:
    if region.empty:
        return
    use = region[region["region"].isin(["core", "near_field", "outer_ring", "diagnostic_window"])].copy()
    fig, axes = plt.subplots(3, 2, figsize=(13, 10), dpi=150, sharex=True)
    for col, polarity in enumerate(use["polarity"].drop_duplicates()):
        part_pol = use[use["polarity"] == polarity]
        for region_name, part in part_pol.groupby("region", sort=False):
            part = sort_lifecycle_table(part)
            axes[0, col].plot(part["tau_center"], part["rms_Fz_tilt_correction"], marker="o", label=region_name)
            axes[1, col].plot(part["tau_center"], part["median_tilt_fraction"], marker="o", label=region_name)
            axes[2, col].plot(part["tau_center"], part["J_residual_rms"], marker="o", label=region_name)
        axes[0, col].set_title(f"{polarity}: rms Fz tilt")
        axes[1, col].set_title(f"{polarity}: median tilt fraction")
        axes[2, col].set_title(f"{polarity}: J residual rms")
    for ax in axes.ravel():
        ax.set_xlabel("tau")
        ax.grid(True, color="0.9")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(fig_dir / "region_tilt_evolution_sensitivity.png", bbox_inches="tight")
    plt.close(fig)


def write_tilt_evolution_summary(coupling_dir: Path, phase: pd.DataFrame, corr: pd.DataFrame, region: pd.DataFrame) -> None:
    lines = [
        "# Tilt-flux correction and lifecycle tilt-evolution coupling",
        "",
        "This is a nondimensional lifecycle-phase post-processing diagnostic. It uses five phase centers only, so it supports trend association, not real-time causality or a complete dynamical budget.",
        "",
        "## Main Phase Metrics",
        "```csv",
        phase.to_csv(index=False).strip(),
        "```",
        "",
        "## Coupling Metrics",
        "```csv",
        corr.to_csv(index=False).strip(),
        "```",
        "",
        "## Interpretation",
    ]
    for _, row in corr.iterrows():
        pol = str(row["polarity"])
        part = sort_lifecycle_table(phase[phase["polarity"] == pol])
        bg = part[part["phase_name"].isin(["birth", "growth"])]
        md = part[part["phase_name"] == "mature"]
        dd = part[part["phase_name"].isin(["decay", "death"])]
        lines.append(f"### {pol}")
        if len(bg) == 2:
            b, g = bg.iloc[0], bg.iloc[1]
            lines.append(
                f"- Birth to growth: axis slope changes from {b['median_axis_slope_mag']:.3g} to {g['median_axis_slope_mag']:.3g}; "
                f"rms Fz^(tilt) changes from {b['rms_Fz_tilt_correction']:.3g} to {g['rms_Fz_tilt_correction']:.3g}; "
                f"median tilt fraction changes from {b['median_tilt_fraction']:.3g} to {g['median_tilt_fraction']:.3g}."
            )
        if len(md):
            m = md.iloc[0]
            lines.append(
                f"- Mature phase: axis slope={m['median_axis_slope_mag']:.3g}, rms Fz^(tilt)={m['rms_Fz_tilt_correction']:.3g}, "
                f"median tilt fraction={m['median_tilt_fraction']:.3g}, JT improvement={m['improvement_fraction']:.3g}."
            )
        if len(dd) == 2:
            d, death = dd.iloc[0], dd.iloc[1]
            lines.append(
                f"- Decay to death: axis slope changes from {d['median_axis_slope_mag']:.3g} to {death['median_axis_slope_mag']:.3g}; "
                f"partial_tau axis slope at death={death['partial_tau_axis_slope']:.3g}; JT improvement changes from {d['improvement_fraction']:.3g} to {death['improvement_fraction']:.3g}."
            )
        lines.append(
            f"- Zero-lag correlations with n_phase=5: corr(axis,Fz_tilt)={row['corr_axis_slope_rms_Fz_tilt_correction']:.3g}, "
            f"corr(partial_tau axis,Fz_tilt)={row['corr_partial_tau_axis_slope_rms_Fz_tilt_correction']:.3g}, "
            f"corr(axis,abs(tilt_fraction))={row['corr_axis_slope_abs_tilt_fraction']:.3g}, "
            f"corr(partial_tau axis,JT_improvement)={row['corr_partial_tau_axis_slope_improvement_fraction']:.3g}."
        )
        lines.append(
            f"- Peak phases: axis={row['axis_slope_peak_phase']}, Fz_tilt={row['Fz_tilt_peak_phase']}, "
            f"maximum cancellation={row['tilt_cancellation_peak_phase']}, JT improvement={row['JT_improvement_peak_phase']}."
        )
        lines.append(f"- Interpretation label: `{row['interpretation_label']}`.")
        if np.isfinite(float(row["mean_Q_n_valid_fraction"])) and float(row["mean_Q_n_valid_fraction"]) < 0.15:
            lines.append("- Caution: Q_n valid fraction is low, so wave-action interpretations should be treated as phase-trend evidence rather than a robust local conservation proof.")
    if not region.empty:
        lines.extend(
            [
                "",
                "## Region Sensitivity",
                "The region table keeps core, near-field, outer-ring, diagnostic-window, and layer-body views. Use it to distinguish whether the coupling is core dominated or driven by the diagnostic environment ring.",
            ]
        )
    lines.extend(
        [
            "",
            "## Figures",
            "- `figures/tilt_axis_vs_Fz_tilt_by_phase.png`",
            "- `figures/tilt_fraction_vs_tilt_evolution.png`",
            "- `figures/JT_improvement_vs_tilt_evolution.png`",
            "- `figures/phase_lag_summary.png`",
            "- `figures/region_tilt_evolution_sensitivity.png`",
        ]
    )
    (coupling_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_tilt_evolution_coupling_only(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    coupling_dir = output_dir / "tilt_evolution_coupling"
    fig_dir = coupling_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    phase = build_tilt_evolution_phase_metrics(output_dir)
    corr = build_tilt_evolution_correlation_metrics(phase)
    region = build_tilt_evolution_region_sensitivity(output_dir, phase)
    phase.to_csv(coupling_dir / "tilt_evolution_phase_metrics.csv", index=False)
    corr.to_csv(coupling_dir / "tilt_evolution_correlation_metrics.csv", index=False)
    region.to_csv(coupling_dir / "tilt_evolution_region_sensitivity.csv", index=False)
    plot_tilt_axis_vs_fz_tilt(fig_dir, phase)
    plot_tilt_fraction_vs_tilt_evolution(fig_dir, phase)
    plot_jt_improvement_vs_tilt_evolution(fig_dir, phase)
    plot_phase_lag_summary(fig_dir, corr)
    plot_region_tilt_evolution_sensitivity(fig_dir, region)
    write_tilt_evolution_summary(coupling_dir, phase, corr, region)
    print(f"Tilt evolution coupling: {coupling_dir}")


def plot_outputs(
    output_dir: Path,
    flux: pd.DataFrame,
    feedback: pd.DataFrame,
    wave: pd.DataFrame,
    upright: pd.DataFrame,
    profiles: pd.DataFrame,
    axis_dir: Path,
    radii: dict[str, float],
    args: argparse.Namespace,
) -> None:
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    for table, y, title, filename in (
        (flux, "median_tilt_fraction", "Tilt fraction by lifecycle phase", "tilt_fraction_by_phase.png"),
        (feedback, "corr_divF_partial_tau_Ubar", "divF vs partial_tau Ubar", "divF_vs_partial_tau_Ubar.png"),
        (wave, "wave_activity_residual_nd_rms", "Wave activity nondimensional residual", "wave_activity_phase_budget.png"),
        (upright, "normalized_Fz_difference", "Tilted vs upright limit", "tilted_vs_upright_limit.png"),
    ):
        if table.empty:
            continue
        fig, ax = plt.subplots(figsize=(8, 5), dpi=160)
        for polarity, part in table.groupby("polarity"):
            part = part.sort_values("tau_center")
            ax.plot(part["tau_center"], part[y], marker="o", label=polarity)
        ax.set_xlabel("tau")
        ax.set_title(title)
        ax.grid(True, color="0.9")
        ax.legend()
        fig.tight_layout()
        fig.savefig(fig_dir / filename)
        plt.close(fig)

    tilted = profiles[profiles["axis_mode"] == "tilted"]
    for (polarity, phase_name), part in tilted.groupby(["polarity", "phase_name"], sort=True):
        pivots = [part.pivot(index="depth_m", columns="r_over_R", values=name).sort_index() for name, _ in FZ_COMPONENTS]
        fig, axes = plt.subplots(1, 3, figsize=(14, 5), dpi=150)
        for ax, data, (_, title) in zip(axes, pivots, FZ_COMPONENTS):
            values = data.to_numpy(dtype="f8")
            finite_abs = np.abs(values[np.isfinite(values)])
            if finite_abs.size:
                vmax = np.nanpercentile(finite_abs, 99)
                if not np.isfinite(vmax) or vmax <= 0:
                    vmax = float(np.nanmax(finite_abs))
            else:
                vmax = np.nan
            if not np.isfinite(vmax) or vmax <= 0:
                vmax = 1.0
            nonzero = finite_abs[finite_abs > 0]
            if nonzero.size:
                linthresh = np.nanpercentile(nonzero, 50)
            else:
                linthresh = vmax * 0.02
            if not np.isfinite(linthresh) or linthresh <= 0 or linthresh >= vmax:
                linthresh = vmax * 0.02
            norm = SymLogNorm(linthresh=linthresh, linscale=0.8, vmin=-vmax, vmax=vmax, base=10)
            mesh = ax.pcolormesh(data.columns.to_numpy(dtype="f8"), data.index.to_numpy(dtype="f8"), values, shading="auto", cmap="RdBu_r", norm=norm)
            ax.invert_yaxis()
            ax.set_xlabel("r/R")
            ax.set_title(title)
            rms = np.sqrt(np.nanmean(values * values)) if finite_abs.size else np.nan
            p98 = np.nanpercentile(finite_abs, 98) if finite_abs.size else np.nan
            ax.text(
                0.02,
                0.98,
                f"rms={rms:.2e}\np98={p98:.2e}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=8,
                bbox={"facecolor": "white", "edgecolor": "0.8", "alpha": 0.75, "pad": 2},
            )
            cbar = fig.colorbar(mesh, ax=ax, shrink=0.85)
            cbar.ax.tick_params(labelsize=8)
        axes[0].set_ylabel("depth m")
        fig.suptitle(f"{polarity} {phase_name}: Fz components")
        fig.tight_layout()
        fig.savefig(fig_dir / f"{polarity}_{phase_name}_Fz_components_rz_by_phase.png")
        plt.close(fig)
    plot_tilted_structure_outputs(
        fig_dir,
        profiles,
        axis_dir,
        radii,
        make_png=bool(args.tilted_structure_plots),
        make_html=bool(args.tilted_structure_html),
        stride=int(args.tilted_structure_stride),
    )


def write_summary(output_dir: Path, flux: pd.DataFrame, feedback: pd.DataFrame, wave: pd.DataFrame, upright: pd.DataFrame, checklist: pd.DataFrame, args: argparse.Namespace) -> None:
    lines = [
        "# Nondimensional lifecycle E-P theory validation",
        "",
        f"- Polarities: {args.polarities}",
        "- Phase coordinate: tau = birth 0.1, growth 0.3, mature 0.5, decay 0.7, death 0.9.",
        "- Main feedback diagnostic uses partial_tau_Ubar, not dUdt or partial_t.",
        "- Tilted-structure Fz figures place the azimuthal-mean Fz components on the pooled east-aligned tilted vortex axis; they are geometric curtains, not instantaneous 3D asymmetric fields.",
        "- Tilt fraction percentages are medians over the azimuthal-mean r-z section. `*_tilt_fraction_tilted_structure_3d.png` places that same r-z field on the tilted axis; color clipping only affects display, not CSV/parquet values.",
        "- Tilt-term mechanism diagnostics are written under `tilt_term_mechanism/` and decompose F_z^(tilt) into raw covariance and N2 weighting.",
        "- Total wave-action flux diagnostics are written under `wave_action_total_flux/`; the legacy wave_activity_phase_budget.png is E-P-only, not J_T.",
        "- Strong nonlinearity diagnostics are written under `strong_nonlinearity_diagnostics/` and separate weak PV-gradient singularity, finite-amplitude PV perturbation, and J_T budget nonclosure.",
        "- Core-vs-environment sensitivity diagnostics are written under `core_environment_sensitivity/` and separate core, near-field, outer-ring, diagnostic-window, and layer-body interpretations.",
        "",
        "## Checklist",
        "```csv",
        checklist.to_csv(index=False).strip(),
        "```",
        "",
        "## Vertical Flux Decomposition",
        "```csv",
        flux.to_csv(index=False).strip(),
        "```",
        "",
        "## Lifecycle Feedback",
        "```csv",
        feedback.to_csv(index=False).strip(),
        "```",
        "",
        "## Wave Activity",
        "```csv",
        wave.to_csv(index=False).strip(),
        "```",
        "",
        "## Upright Limit",
        "```csv",
        upright.to_csv(index=False).strip(),
        "```",
    ]
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    if bool(args.nonlinearity_diagnostics_only):
        run_nonlinearity_diagnostics_only(args)
        return
    if bool(args.core_environment_sensitivity_only):
        run_core_environment_sensitivity_only(args)
        return
    if bool(args.mlrw_applicability_only):
        run_mlrw_applicability_only(args)
        return
    if bool(args.tilt_evolution_coupling_only):
        run_tilt_evolution_coupling_only(args)
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    shapes = parse_csv_list(args.shapes, DEFAULT_SHAPES)
    polarities = parse_csv_list(args.polarities, DEFAULT_POLARITIES)
    objects = load_lifecycle_objects(Path(args.axis_dir), Path(args.catalog_dir), Path(args.shape_dir), shapes, polarities)
    objects = apply_lifecycle_limits(objects, int(args.max_days), int(args.max_objects_per_polarity), int(args.random_seed))
    if objects.empty:
        raise RuntimeError("No lifecycle objects selected.")
    centers = load_center_lines(Path(args.axis_dir), set(objects["eddy3d_object_id"].astype(int)))
    radial, theta, rr, tt, _ = make_polar_grid(float(args.rmax), int(args.radial_bins), int(args.azimuth_bins))
    lat_ref = float(objects["surface_lat"].median())
    f0 = 2.0 * OMEGA * np.sin(np.radians(lat_ref))
    radii = {str(p): float(part["mean_radius_m"].median()) for p, part in objects.groupby("polarity")}

    accum = {}
    depth_ref = None
    for date, day_objects in tqdm(list(objects.groupby("date")), desc="Nondim theory", unit="day"):
        path = Path(args.input_daily_dir) / f"uv_{pd.Timestamp(date):%Y%m%d}.nc"
        if not path.exists():
            continue
        lon, lat, depth, u, v = read_daily_uv(path)
        u_clim, v_clim = read_climatology_uv(Path(args.climatology_path), str(date))
        _, dy, dx = grid_spacing_m(lon, lat)
        psi = streamfunction_from_zeta(relative_vorticity(lon, lat, u, v), dx, dy)
        n2 = load_n2(Path(args.n2_profile_path), depth)
        depth_ref = depth if depth_ref is None else depth_ref
        for obj in day_objects.itertuples(index=False):
            center = centers.get(int(obj.eddy3d_object_id))
            if center is None:
                continue
            for mode, line, slope_mode in (
                ("tilted", center, "tilted"),
                ("tilted_sampling_no_slope", center, "upright"),
                ("upright", zero_center_line(center), "upright"),
            ):
                fields = sample_object_fields(obj, line, lon, lat, depth, psi, u, v, u_clim, v_clim, None, None, rr, tt)
                if fields is None:
                    continue
                terms = compute_nondim_terms(fields, line, depth, radial, theta, float(obj.mean_radius_m), n2, f0, axis_mode=slope_mode)
                add_terms(accum, (mode, str(obj.polarity), int(obj.phase_index), str(obj.phase_name)), terms, int(obj.eddy3d_object_id), str(obj.date))
    if depth_ref is None:
        raise RuntimeError("No daily uv files were processed.")
    profiles = rows_from_final(finalize(accum), radial, depth_ref, radii)
    profiles = add_wave_activity(profiles, radii)
    flux, feedback, wave, upright, checklist = build_metrics(profiles)

    profiles.to_parquet(output_dir / "nondim_vertical_flux_decomposition.parquet", index=False)
    profiles.to_csv(output_dir / "nondim_vertical_flux_decomposition.csv", index=False)
    feedback.to_csv(output_dir / "nondim_lifecycle_feedback_metrics.csv", index=False)
    profiles.to_parquet(output_dir / "nondim_wave_activity_budget.parquet", index=False)
    wave.to_csv(output_dir / "nondim_wave_activity_budget.csv", index=False)
    upright.to_csv(output_dir / "nondim_upright_limit_comparison.csv", index=False)
    checklist.to_csv(output_dir / "nondim_theory_validation_checklist.csv", index=False)
    flux.to_csv(output_dir / "nondim_vertical_flux_decomposition_metrics.csv", index=False)
    tilt_metrics = write_mechanism_outputs(output_dir, profiles)
    write_total_flux_outputs(output_dir, profiles, tilt_metrics, Path(args.axis_dir), radii, args)
    write_nonlinearity_outputs(output_dir, profiles, radii, Path(args.axis_dir), args)
    nonlinear_path = output_dir / "strong_nonlinearity_diagnostics" / "nonlinearity_profiles.parquet"
    if nonlinear_path.exists():
        write_core_environment_sensitivity_outputs(output_dir, profiles, pd.read_parquet(nonlinear_path), radii, args)
    plot_outputs(output_dir, flux, feedback, wave, upright, profiles, Path(args.axis_dir), radii, args)
    write_summary(output_dir, flux, feedback, wave, upright, checklist, args)
    print(f"Output: {output_dir}")
    print(f"Summary: {output_dir / 'summary.md'}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Nondimensional lifecycle validation of tilted E-P flux theory.")
    parser.add_argument("--axis-dir", default=str(DEFAULT_AXIS_DIR))
    parser.add_argument("--catalog-dir", default=str(DEFAULT_CATALOG))
    parser.add_argument("--shape-dir", default=str(DEFAULT_SHAPE_BY_SHAPE_DIR))
    parser.add_argument("--input-daily-dir", default=str(DEFAULT_INPUT_DAILY))
    parser.add_argument("--climatology-path", default=str(DEFAULT_CLIMATOLOGY_NC))
    parser.add_argument("--n2-profile-path", default=str(DEFAULT_CLIMATOLOGY))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--shapes", default=",".join(DEFAULT_SHAPES))
    parser.add_argument("--polarities", default=",".join(DEFAULT_POLARITIES))
    parser.add_argument("--max-days", type=int, default=0)
    parser.add_argument("--max-objects-per-polarity", type=int, default=0)
    parser.add_argument("--rmax", type=float, default=2.5)
    parser.add_argument("--radial-bins", type=int, default=40)
    parser.add_argument("--azimuth-bins", type=int, default=72)
    parser.add_argument("--random-seed", type=int, default=20260711)
    parser.add_argument("--tilted-structure-plots", dest="tilted_structure_plots", action="store_true", default=True)
    parser.add_argument("--no-tilted-structure-plots", dest="tilted_structure_plots", action="store_false")
    parser.add_argument("--tilted-structure-html", dest="tilted_structure_html", action="store_true", default=True)
    parser.add_argument("--no-tilted-structure-html", dest="tilted_structure_html", action="store_false")
    parser.add_argument("--tilted-structure-stride", type=int, default=2)
    parser.add_argument("--jt-3d-structure", dest="jt_3d_structure", action="store_true", default=True)
    parser.add_argument("--no-jt-3d-structure", dest="jt_3d_structure", action="store_false")
    parser.add_argument("--jt-3d-html", dest="jt_3d_html", action="store_true", default=True)
    parser.add_argument("--no-jt-3d-html", dest="jt_3d_html", action="store_false")
    parser.add_argument("--jt-3d-stride", type=int, default=2)
    parser.add_argument("--nonlinearity-stack-3d", dest="nonlinearity_stack_3d", action="store_true", default=True)
    parser.add_argument("--no-nonlinearity-stack-3d", dest="nonlinearity_stack_3d", action="store_false")
    parser.add_argument("--nonlinearity-stack-html", dest="nonlinearity_stack_html", action="store_true", default=False)
    parser.add_argument("--no-nonlinearity-stack-html", dest="nonlinearity_stack_html", action="store_false")
    parser.add_argument("--nonlinearity-stack-depth-levels", default="0,1,2,4,6,8,10,15,20,50,100,200,500,1000,1500")
    parser.add_argument("--nonlinearity-stack-grid-size", type=int, default=81)
    parser.add_argument("--nonlinearity-stack-xy-extent", type=float, default=2.5)
    parser.add_argument("--nonlinearity-diagnostics-only", action="store_true", help="Read existing nondim_vertical_flux_decomposition.parquet and only rebuild strong nonlinearity diagnostics.")
    parser.add_argument("--core-environment-sensitivity-only", action="store_true", help="Read existing profile outputs and only rebuild core-vs-environment sensitivity diagnostics.")
    parser.add_argument("--mlrw-applicability-only", action="store_true", help="Read existing nonlinearity outputs and build MLRW applicability diagnostics.")
    parser.add_argument("--tilt-evolution-coupling-only", action="store_true", help="Read existing full-result tables and only build tilt-flux/lifecycle-evolution coupling diagnostics.")
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
