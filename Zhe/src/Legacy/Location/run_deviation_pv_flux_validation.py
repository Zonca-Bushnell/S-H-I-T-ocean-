from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import netCDF4
import numpy as np
import pandas as pd

from src.Legacy.First_temp.axis_streamfunction_separation import grid_spacing_m, relative_vorticity, streamfunction_from_zeta
from src.Legacy.First_temp.lifecycle_ep_flux_nondim_validation import (
    azimuth_second_derivative,
    ddz,
    load_n2,
    make_polar_grid,
    radial_derivative,
    read_climatology_uv,
)
from src.Legacy.First_temp.tilted_ep_flux_validation import bilinear_sample, sanitize_ocean_field, xy_to_lonlat
from src.Legacy.Location.common import load_config, parse_ymd
from src.Legacy.Location.streaming_cmems import read_day_data


SECONDS_PER_DAY = 86400.0
EARTH_RADIUS_M = 6_371_000.0
OMEGA = 7.2921159e-5


def _wrap_pi(values: np.ndarray) -> np.ndarray:
    return (values + np.pi) % (2.0 * np.pi) - np.pi


def _corr(x: np.ndarray, y: np.ndarray) -> float:
    good = np.isfinite(x) & np.isfinite(y)
    return float(np.corrcoef(x[good], y[good])[0, 1]) if int(np.sum(good)) > 2 else np.nan


def _slope(x: np.ndarray, y: np.ndarray) -> float:
    good = np.isfinite(x) & np.isfinite(y)
    if int(np.sum(good)) <= 2:
        return np.nan
    xx = x[good].astype("float64", copy=False)
    yy = y[good].astype("float64", copy=False)
    xx = xx - float(np.nanmean(xx))
    yy = yy - float(np.nanmean(yy))
    var = float(np.nanmean(xx * xx))
    if not np.isfinite(var) or var <= 1e-300:
        return np.nan
    return float(np.nanmean(xx * yy) / var)


def _same_sign(x: np.ndarray, y: np.ndarray) -> float:
    good = np.isfinite(x) & np.isfinite(y) & (x != 0.0) & (y != 0.0)
    return float(np.mean(np.sign(x[good]) == np.sign(y[good]))) if np.any(good) else np.nan


def _rmse_ratio(pred: np.ndarray, obs: np.ndarray) -> float:
    good = np.isfinite(pred) & np.isfinite(obs)
    if not np.any(good):
        return np.nan
    rmse = float(np.sqrt(np.nanmean((pred[good] - obs[good]) ** 2)))
    denom = float(np.sqrt(np.nanmean(obs[good] ** 2)))
    return rmse / denom if denom > 0.0 and np.isfinite(denom) else np.nan


def _read_representative_radius(rv_root: Path) -> dict[str, float]:
    radii = pd.read_csv(rv_root / "representative_radii.csv")
    return {str(row.polarity): float(row.representative_radius_m) for row in radii.itertuples(index=False)}


def _read_tau_grid(rv_root: Path) -> np.ndarray:
    tau = pd.read_csv(rv_root / "continuous_tau_grid.csv")
    if "tau_center" in tau.columns:
        return tau["tau_center"].to_numpy(dtype="float64")
    if "tau" in tau.columns:
        return tau["tau"].to_numpy(dtype="float64")
    numeric = tau.select_dtypes(include=[np.number])
    if numeric.empty:
        raise ValueError(f"No numeric tau column found in {rv_root / 'continuous_tau_grid.csv'}")
    return numeric.iloc[:, 0].to_numpy(dtype="float64")


def _load_objects(rv_root: Path, start: str | None = None, end: str | None = None) -> pd.DataFrame:
    objects = pd.read_parquet(rv_root / "object_cache" / "selected_lifecycle_objects.parquet")
    objects["date"] = pd.to_datetime(objects["date"]).dt.strftime("%Y-%m-%d")
    objects = objects[objects["shape_class"].astype(str).eq("coherent")].copy()
    if start:
        objects = objects[objects["date"] >= start]
    if end:
        objects = objects[objects["date"] <= end]
    return objects


def _load_points(rv_root: Path, object_ids: set[int] | None = None) -> pd.DataFrame:
    cols = [
        "track3d_id",
        "eddy3d_object_id",
        "date",
        "polarity",
        "depth_index",
        "depth_m",
        "longitude",
        "latitude",
        "x_m",
        "y_m",
        "x_rot_m",
        "y_rot_m",
        "global_deviate_angle_rad",
        "global_theta0_rad",
    ]
    points = pd.read_parquet(rv_root / "axis" / "rotated_points.parquet", columns=cols)
    if object_ids is not None:
        points = points[points["eddy3d_object_id"].astype("int64").isin(object_ids)].copy()
    points["date"] = pd.to_datetime(points["date"])
    points["track3d_id"] = points["track3d_id"].astype("int64")
    points["eddy3d_object_id"] = points["eddy3d_object_id"].astype("int64")
    points["depth_index"] = points["depth_index"].astype("int16")
    return points


def _center_lines_from_points(points: pd.DataFrame) -> dict[int, pd.DataFrame]:
    out: dict[int, pd.DataFrame] = {}
    for object_id, part in points.groupby("eddy3d_object_id", sort=False):
        out[int(object_id)] = part.sort_values("depth_index").reset_index(drop=True)
    return out


def _center_kinematics(points: pd.DataFrame) -> pd.DataFrame:
    p = points.sort_values(["track3d_id", "depth_index", "date"]).copy()
    p["xi_y_m"] = p["y_m"].astype("float64")
    theta0 = p["global_theta0_rad"].to_numpy(dtype="float64")
    p["e_y"] = np.sin(theta0)
    p["n_y"] = np.cos(theta0)
    p["a_m"] = p["x_rot_m"].astype("float64")
    p["dt_day"] = p.groupby(["track3d_id", "depth_index"])["date"].diff().dt.total_seconds() / SECONDS_PER_DAY
    p["dxi_y_dt_m_day"] = p.groupby(["track3d_id", "depth_index"])["xi_y_m"].diff() / p["dt_day"]
    p["da_dt_m_day"] = p.groupby(["track3d_id", "depth_index"])["a_m"].diff() / p["dt_day"]
    day_alpha = (
        p[["track3d_id", "date", "global_deviate_angle_rad"]]
        .drop_duplicates(["track3d_id", "date"])
        .sort_values(["track3d_id", "date"])
        .copy()
    )
    day_alpha["prev_alpha"] = day_alpha.groupby("track3d_id")["global_deviate_angle_rad"].shift(1)
    day_alpha["alpha_dot_rad_day"] = _wrap_pi(
        day_alpha["global_deviate_angle_rad"].to_numpy(dtype="float64")
        - day_alpha["prev_alpha"].to_numpy(dtype="float64")
    )
    p = p.merge(day_alpha[["track3d_id", "date", "alpha_dot_rad_day"]], on=["track3d_id", "date"], how="left")
    p["vdev_y_decomp_m_day"] = p["da_dt_m_day"] * p["e_y"] + p["a_m"] * p["alpha_dot_rad_day"] * p["n_y"]
    p["vdev_y_actual_m_day"] = p["dxi_y_dt_m_day"]
    return p[
        [
            "eddy3d_object_id",
            "track3d_id",
            "date",
            "polarity",
            "depth_index",
            "depth_m",
            "xi_y_m",
            "e_y",
            "n_y",
            "a_m",
            "dxi_y_dt_m_day",
            "da_dt_m_day",
            "alpha_dot_rad_day",
            "vdev_y_decomp_m_day",
            "vdev_y_actual_m_day",
        ]
    ].copy()


def _proxy_validation(rv_root: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    points = _load_points(rv_root)
    kin = _center_kinematics(points)
    kin["date"] = pd.to_datetime(kin["date"]).dt.strftime("%Y-%m-%d")
    kin["tau_center"] = np.nan
    objects = _load_objects(rv_root)
    object_tau = objects[["eddy3d_object_id", "life_phase", "mean_radius_m"]].copy()
    kin = kin.merge(object_tau, on="eddy3d_object_id", how="left")
    tau_grid = _read_tau_grid(rv_root)
    kin["tau_center"] = tau_grid[np.nanargmin(np.abs(kin["life_phase"].to_numpy(dtype="float64")[:, None] - tau_grid[None, :]), axis=1)]

    profiles = pd.read_parquet(rv_root / "ep_flux_terms" / "continuous_ep_flux_profiles.parquet")
    profiles = profiles[(profiles["axis_mode"] == "tilted") & (profiles["r_over_R"] <= 1.5)].copy()
    w = profiles["count"].to_numpy(dtype="float64")
    grouped = []
    for keys, part in profiles.groupby(["polarity", "tau_center", "depth_index", "depth_m"], sort=True):
        weights = part["count"].to_numpy(dtype="float64")
        row = dict(zip(["polarity", "tau_center", "depth_index", "depth_m"], keys))
        for col in ["pv_flux", "divF", "q_mean", "Q_n", "count"]:
            if col not in part:
                continue
            if col == "count":
                row[col] = float(np.nansum(weights))
            else:
                vals = part[col].to_numpy(dtype="float64")
                ok = np.isfinite(vals) & np.isfinite(weights) & (weights > 0)
                row[col] = float(np.nansum(vals[ok] * weights[ok]) / np.nansum(weights[ok])) if np.any(ok) else np.nan
        grouped.append(row)
    prof = pd.DataFrame(grouped)

    # Proxy linear qbar_y uses Q_n as the available representative PV-gradient proxy.
    kin_agg = []
    for keys, part in kin.groupby(["polarity", "tau_center", "depth_index", "depth_m"], sort=True):
        row = dict(zip(["polarity", "tau_center", "depth_index", "depth_m"], keys))
        row["xi_y_dxi_dt_mean_m2_day"] = float(np.nanmean(part["xi_y_m"] * part["dxi_y_dt_m_day"]))
        row["xi_y_vdev_decomp_mean_m2_day"] = float(np.nanmean(part["xi_y_m"] * part["vdev_y_decomp_m_day"]))
        row["vdev_y_actual_mean_m_day"] = float(np.nanmean(part["vdev_y_actual_m_day"]))
        row["vdev_y_decomp_mean_m_day"] = float(np.nanmean(part["vdev_y_decomp_m_day"]))
        row["n_objects"] = int(part["eddy3d_object_id"].nunique())
        kin_agg.append(row)
    proxy = pd.DataFrame(kin_agg).merge(prof, on=["polarity", "tau_center", "depth_index", "depth_m"], how="left")
    proxy["F_dev_linear_proxy"] = -proxy["Q_n"] * proxy["xi_y_dxi_dt_mean_m2_day"] / SECONDS_PER_DAY
    proxy["F_dev_linear_proxy_decomp"] = -proxy["Q_n"] * proxy["xi_y_vdev_decomp_mean_m2_day"] / SECONDS_PER_DAY
    proxy.to_parquet(output_dir / "proxy_deviation_pv_flux_tau_depth.parquet", index=False)
    proxy.to_csv(output_dir / "proxy_deviation_pv_flux_tau_depth.csv", index=False)

    rows = []
    for pol, part in proxy.groupby("polarity", sort=True):
        rows.append(
            {
                "relation": "proxy_linear_vs_existing_axis_pv_flux",
                "polarity": pol,
                "corr": _corr(part["F_dev_linear_proxy"].to_numpy(float), part["pv_flux"].to_numpy(float)),
                "same_sign_fraction": _same_sign(part["F_dev_linear_proxy"].to_numpy(float), part["pv_flux"].to_numpy(float)),
                "slope": _slope(part["F_dev_linear_proxy"].to_numpy(float), part["pv_flux"].to_numpy(float)),
                "rmse_ratio": _rmse_ratio(part["F_dev_linear_proxy"].to_numpy(float), part["pv_flux"].to_numpy(float)),
                "n_bins": int(np.sum(np.isfinite(part["F_dev_linear_proxy"]) & np.isfinite(part["pv_flux"]))),
            }
        )
        rows.append(
            {
                "relation": "proxy_linear_vs_divF",
                "polarity": pol,
                "corr": _corr(part["F_dev_linear_proxy"].to_numpy(float), part["divF"].to_numpy(float)),
                "same_sign_fraction": _same_sign(part["F_dev_linear_proxy"].to_numpy(float), part["divF"].to_numpy(float)),
                "slope": _slope(part["F_dev_linear_proxy"].to_numpy(float), part["divF"].to_numpy(float)),
                "rmse_ratio": _rmse_ratio(part["F_dev_linear_proxy"].to_numpy(float), part["divF"].to_numpy(float)),
                "n_bins": int(np.sum(np.isfinite(part["F_dev_linear_proxy"]) & np.isfinite(part["divF"]))),
            }
        )
    pd.DataFrame(rows).to_csv(output_dir / "proxy_summary.csv", index=False)

    for pol, part in proxy.groupby("polarity", sort=True):
        for col in ["F_dev_linear_proxy", "pv_flux", "divF"]:
            piv = part.pivot_table(index="depth_m", columns="tau_center", values=col)
            if piv.empty:
                continue
            vmax = np.nanpercentile(np.abs(piv.to_numpy()), 95)
            if not np.isfinite(vmax) or vmax <= 0:
                vmax = 1.0
            fig, ax = plt.subplots(figsize=(7, 5), dpi=160)
            im = ax.imshow(
                piv.to_numpy(),
                origin="lower",
                aspect="auto",
                cmap="coolwarm",
                vmin=-vmax,
                vmax=vmax,
                extent=[float(piv.columns.min()), float(piv.columns.max()), float(piv.index.min()), float(piv.index.max())],
            )
            ax.set_xlabel("tau")
            ax.set_ylabel("depth m")
            ax.set_title(f"{pol}: {col}")
            fig.colorbar(im, ax=ax)
            fig.tight_layout()
            fig.savefig(fig_dir / f"{pol}_{col}_tau_depth.png")
            plt.close(fig)


def _q_prime_from_psi(fields: dict[str, np.ndarray], depth: np.ndarray, radial: np.ndarray, theta: np.ndarray, radius_m: float, n2: np.ndarray, f0: float) -> np.ndarray:
    psi = np.where(np.abs(fields["psi_prime"]) > 1e20, np.nan, fields["psi_prime"])
    psi_prime = psi - np.nanmean(psi, axis=2, keepdims=True)
    r_m = np.maximum(radial * radius_m, 1.0)
    dpsi_dr = radial_derivative(psi_prime, r_m)
    radial_lap = radial_derivative(dpsi_dr * r_m[None, :, None], r_m) / r_m[None, :, None]
    az_lap = azimuth_second_derivative(psi_prime, theta) / (r_m[None, :, None] ** 2)
    dpsi_dz = ddz(psi_prime, depth)
    strat = (f0 * f0 / n2)[:, None, None] * dpsi_dz
    q_total = radial_lap + az_lap + ddz(strat, depth)
    return q_total - np.nanmean(q_total, axis=2, keepdims=True)


def _sample_extended_fields(obj, center_line: pd.DataFrame, lon: np.ndarray, lat: np.ndarray, depth: np.ndarray, psi_prime: np.ndarray, u: np.ndarray, v: np.ndarray, u_clim: np.ndarray, v_clim: np.ndarray, rr: np.ndarray, tt: np.ndarray) -> dict[str, np.ndarray] | None:
    if len(center_line) != len(depth):
        return None
    radius = float(obj.mean_radius_m)
    theta_obj = float(obj.temp_direction_rad)
    cos_t = math.cos(theta_obj)
    sin_t = math.sin(theta_obj)
    local_x = rr * radius * np.cos(tt)
    local_y = rr * radius * np.sin(tt)
    axis_x = center_line["x_rot_m"].to_numpy(dtype="float64")
    axis_y = center_line["y_rot_m"].to_numpy(dtype="float64")
    psi_layers, un_layers, us_layers, un_clim_layers, us_clim_layers = [], [], [], [], []
    vprime_global_layers = []
    vraw_layers = []
    vclim_layers = []
    for k in range(len(depth)):
        x_rot = axis_x[k] + local_x
        y_rot = axis_y[k] + local_y
        x_orig = x_rot * cos_t - y_rot * sin_t
        y_orig = x_rot * sin_t + y_rot * cos_t
        target_lon, target_lat = xy_to_lonlat(x_orig, y_orig, float(obj.surface_lon), float(obj.surface_lat))
        psi_s = bilinear_sample(lon, lat, psi_prime[k], target_lon, target_lat)
        u_samp = bilinear_sample(lon, lat, u[k], target_lon, target_lat)
        v_samp = bilinear_sample(lon, lat, v[k], target_lon, target_lat)
        u_clim_samp = bilinear_sample(lon, lat, u_clim[k], target_lon, target_lat)
        v_clim_samp = bilinear_sample(lon, lat, v_clim[k], target_lon, target_lat)
        u_rot = u_samp * cos_t + v_samp * sin_t
        v_rot = -u_samp * sin_t + v_samp * cos_t
        u_clim_rot = u_clim_samp * cos_t + v_clim_samp * sin_t
        v_clim_rot = -u_clim_samp * sin_t + v_clim_samp * cos_t
        un = u_rot * np.cos(tt) + v_rot * np.sin(tt)
        us = -u_rot * np.sin(tt) + v_rot * np.cos(tt)
        un_clim = u_clim_rot * np.cos(tt) + v_clim_rot * np.sin(tt)
        us_clim = -u_clim_rot * np.sin(tt) + v_clim_rot * np.cos(tt)
        psi_layers.append(psi_s)
        un_layers.append(un)
        us_layers.append(us)
        un_clim_layers.append(un_clim)
        us_clim_layers.append(us_clim)
        vraw_layers.append(v_samp)
        vclim_layers.append(v_clim_samp)
        vprime_global_layers.append(v_samp - v_clim_samp)
    return {
        "psi_prime": np.asarray(psi_layers),
        "u_n_prime": np.asarray(un_layers),
        "u_s_prime": np.asarray(us_layers),
        "u_n_clim": np.asarray(un_clim_layers),
        "u_s_clim": np.asarray(us_clim_layers),
        "v_prime_global_clim": np.asarray(vprime_global_layers),
        "v_raw_global": np.asarray(vraw_layers),
        "v_clim_global": np.asarray(vclim_layers),
    }


def _climatological_qbar_y(lon: np.ndarray, lat: np.ndarray, depth: np.ndarray, u_clim: np.ndarray, v_clim: np.ndarray, n2: np.ndarray, f0: float) -> np.ndarray:
    _, dy, dx = grid_spacing_m(lon, lat)
    zeta = relative_vorticity(lon, lat, u_clim, v_clim)
    psi = streamfunction_from_zeta(zeta, dx, dy)
    strat = (f0 * f0 / n2)[:, None, None] * ddz(psi, depth)
    qbar = zeta + ddz(strat, depth)
    return np.gradient(qbar, dy, axis=1, edge_order=1)


def _sample_qbar_y_at_centers(qbar_y: np.ndarray, lon: np.ndarray, lat: np.ndarray, center_line: pd.DataFrame) -> np.ndarray:
    vals = []
    for row in center_line.itertuples(index=False):
        vals.append(float(bilinear_sample(lon, lat, qbar_y[int(row.depth_index)], np.asarray([[row.longitude]]), np.asarray([[row.latitude]])).ravel()[0]))
    return np.asarray(vals, dtype="float64")


def _strict_day(
    config: dict,
    rv_root: Path,
    output_dir: Path,
    day: str,
    day_objects: pd.DataFrame,
    points: pd.DataFrame,
    kinematics: pd.DataFrame,
    tau_grid: np.ndarray,
    bandwidth: float,
    rmax: float,
    radial_bins: int,
    azimuth_bins: int,
    n2_profile_path: Path,
) -> Path:
    date = pd.Timestamp(day).date()
    part_dir = output_dir / "strict_deviation_pv_flux_parts" / f"year={date.year}"
    part_dir.mkdir(parents=True, exist_ok=True)
    part_path = part_dir / f"deviation_pv_flux_{date:%Y%m%d}.parquet"
    if part_path.exists():
        return part_path
    center_lines = _center_lines_from_points(points[points["date"].eq(pd.Timestamp(day))])
    kin_day = kinematics[kinematics["date"].eq(pd.Timestamp(day))]
    kin_by_object = {int(k): v.sort_values("depth_index") for k, v in kin_day.groupby("eddy3d_object_id", sort=False)}
    day_data = read_day_data(config, date)
    lon = day_data["lon"]
    lat = day_data["lat"]
    depth = day_data["depth"]
    u = sanitize_ocean_field(day_data["u_all"].astype("float64", copy=False))
    v = sanitize_ocean_field(day_data["v_all"].astype("float64", copy=False))
    u_clim, v_clim = read_climatology_uv(Path(config["representative"]["climatology_path"]), day)
    n2 = load_n2(n2_profile_path, depth)
    lat_ref = float(day_objects["surface_lat"].median()) if "surface_lat" in day_objects else float(np.nanmedian(lat))
    f0 = 2.0 * OMEGA * np.sin(np.radians(lat_ref))
    _, dy, dx = grid_spacing_m(lon, lat)
    psi_prime = streamfunction_from_zeta(relative_vorticity(lon, lat, u, v), dx, dy)
    qbar_y = _climatological_qbar_y(lon, lat, depth, u_clim, v_clim, n2, f0)
    radial, theta, rr, tt, _ = make_polar_grid(rmax, radial_bins, azimuth_bins)
    rows = []
    for obj in day_objects.itertuples(index=False):
        object_id = int(obj.eddy3d_object_id)
        center_line = center_lines.get(object_id)
        kin_line = kin_by_object.get(object_id)
        if center_line is None or kin_line is None or len(center_line) != len(depth) or len(kin_line) != len(depth):
            continue
        fields = _sample_extended_fields(obj, center_line, lon, lat, depth, psi_prime, u, v, u_clim, v_clim, rr, tt)
        if fields is None:
            continue
        q_prime = _q_prime_from_psi(fields, depth, radial, theta, float(obj.mean_radius_m), n2, f0)
        qbar_y_center = _sample_qbar_y_at_centers(qbar_y, lon, lat, center_line)
        un_az = fields["u_n_prime"] - np.nanmean(fields["u_n_prime"], axis=2, keepdims=True)
        un_clim_anom = fields["u_n_prime"] - fields["u_n_clim"]
        v_clim_anom = fields["v_prime_global_clim"]
        v_az = fields["v_raw_global"] - np.nanmean(fields["v_raw_global"], axis=2, keepdims=True)
        q_ann = np.nanmean(q_prime, axis=2)
        measured_v_clim = np.nanmean(v_clim_anom * q_prime, axis=2)
        measured_v_az = np.nanmean(v_az * q_prime, axis=2)
        measured_un_az = np.nanmean(un_az * q_prime, axis=2)
        measured_un_clim = np.nanmean(un_clim_anom * q_prime, axis=2)
        kin_line = kin_line.sort_values("depth_index")
        vdev_y = kin_line["vdev_y_actual_m_day"].to_numpy(dtype="float64") / SECONDS_PER_DAY
        vdev_y_decomp = kin_line["vdev_y_decomp_m_day"].to_numpy(dtype="float64") / SECONDS_PER_DAY
        xi_y = kin_line["xi_y_m"].to_numpy(dtype="float64")
        dxi_y_dt = kin_line["dxi_y_dt_m_day"].to_numpy(dtype="float64") / SECONDS_PER_DAY
        f_dev_center = q_ann * vdev_y[:, None]
        f_dev_center_decomp = q_ann * vdev_y_decomp[:, None]
        f_dev_linear = np.ones_like(q_ann) * (-qbar_y_center[:, None] * xi_y[:, None] * dxi_y_dt[:, None])
        weights = np.exp(-0.5 * ((tau_grid - float(obj.life_phase)) / max(bandwidth, 1e-12)) ** 2)
        valid_tau = np.where(weights >= 1e-4)[0]
        for tau_index in valid_tau:
            weight = float(weights[tau_index])
            for k, depth_m in enumerate(depth):
                for j, r_value in enumerate(radial):
                    rows.append(
                        {
                            "date": day,
                            "polarity": str(obj.polarity),
                            "tau_index": int(tau_index),
                            "tau_center": float(tau_grid[tau_index]),
                            "depth_index": int(k),
                            "depth_m": float(depth_m),
                            "r_over_R": float(r_value),
                            "weight": weight,
                            "n_objects": 1,
                            "q_annulus_mean": float(q_ann[k, j]),
                            "qbar_y_center": float(qbar_y_center[k]),
                            "xi_y_m": float(xi_y[k]),
                            "dxi_y_dt_m_s": float(dxi_y_dt[k]),
                            "vdev_y_actual_m_s": float(vdev_y[k]),
                            "vdev_y_decomp_m_s": float(vdev_y_decomp[k]),
                            "F_dev_center": float(f_dev_center[k, j]),
                            "F_dev_center_decomp": float(f_dev_center_decomp[k, j]),
                            "F_dev_linear": float(f_dev_linear[k, j]),
                            "measured_vprime_clim_qprime": float(measured_v_clim[k, j]),
                            "measured_vprime_az_qprime": float(measured_v_az[k, j]),
                            "measured_unprime_az_qprime": float(measured_un_az[k, j]),
                            "measured_unprime_clim_qprime": float(measured_un_clim[k, j]),
                        }
                    )
    tmp_path = part_path.with_suffix(part_path.suffix + ".tmp")
    pd.DataFrame.from_records(rows).to_parquet(tmp_path, index=False)
    tmp_path.replace(part_path)
    return part_path


def _run_strict(args: argparse.Namespace, config: dict, rv_root: Path, output_dir: Path) -> None:
    start = args.start
    end = args.end
    objects = _load_objects(rv_root, start, end)
    points = _load_points(rv_root, set(objects["eddy3d_object_id"].astype("int64")))
    kin = _center_kinematics(points)
    tau_grid = _read_tau_grid(rv_root)
    dates = sorted(objects["date"].unique())
    log_rows = []
    for day in dates:
        day_objects = objects[objects["date"].eq(day)].copy()
        path = _strict_day(
            config,
            rv_root,
            output_dir,
            day,
            day_objects,
            points,
            kin,
            tau_grid,
            float(args.kernel_bandwidth),
            float(args.rmax),
            int(args.radial_bins),
            int(args.azimuth_bins),
            Path(config["representative"]["n2_profile_path"]),
        )
        log_rows.append({"date": day, "objects": int(len(day_objects)), "path": str(path)})
        if len(log_rows) % 10 == 0:
            pd.DataFrame(log_rows).to_csv(output_dir / f"strict_progress_{start or 'all'}_{end or 'all'}.csv", index=False)
            print(f"processed {len(log_rows)} days through {day}", flush=True)
    pd.DataFrame(log_rows).to_csv(output_dir / f"strict_progress_{start or 'all'}_{end or 'all'}.csv", index=False)


def _finalize(rv_root: Path, output_dir: Path) -> None:
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    part_paths = sorted((output_dir / "strict_deviation_pv_flux_parts").glob("year=*/*.parquet"))
    if not part_paths:
        raise SystemExit("No strict part parquet files found")
    weighted_cols = [
        "q_annulus_mean",
        "qbar_y_center",
        "xi_y_m",
        "dxi_y_dt_m_s",
        "vdev_y_actual_m_s",
        "vdev_y_decomp_m_s",
        "F_dev_center",
        "F_dev_center_decomp",
        "F_dev_linear",
        "measured_vprime_clim_qprime",
        "measured_vprime_az_qprime",
        "measured_unprime_az_qprime",
        "measured_unprime_clim_qprime",
    ]
    keys = ["polarity", "tau_center", "depth_index", "depth_m", "r_over_R"]
    accum: pd.DataFrame | None = None
    for i, p in enumerate(part_paths, start=1):
        df = pd.read_parquet(p)
        if df.empty:
            continue
        df = df.copy()
        df["weighted_count"] = df["weight"]
        for col in weighted_cols:
            values = df[col].to_numpy(dtype="float64")
            weights = df["weight"].to_numpy(dtype="float64")
            ok = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
            df[f"{col}__sum"] = np.where(ok, values * weights, 0.0)
            df[f"{col}__w"] = np.where(ok, weights, 0.0)
        sum_cols = ["weighted_count", "n_objects"] + [f"{col}__sum" for col in weighted_cols] + [f"{col}__w" for col in weighted_cols]
        grouped = df.groupby(keys, sort=True)[sum_cols].sum()
        accum = grouped if accum is None else accum.add(grouped, fill_value=0.0)
        if i % 100 == 0:
            print(f"finalize aggregated {i}/{len(part_paths)} parts", flush=True)
    if accum is None or accum.empty:
        raise SystemExit("Strict part files were empty")
    tau_depth = accum.reset_index()
    for col in weighted_cols:
        denom = tau_depth[f"{col}__w"].to_numpy(dtype="float64")
        numer = tau_depth[f"{col}__sum"].to_numpy(dtype="float64")
        tau_depth[col] = np.divide(numer, denom, out=np.full_like(numer, np.nan), where=denom > 0)
    tau_depth = tau_depth[keys + ["weighted_count", "n_objects"] + weighted_cols]
    tau_depth.to_parquet(output_dir / "strict_deviation_pv_flux_tau_depth.parquet", index=False)
    tau_depth.to_csv(output_dir / "strict_deviation_pv_flux_tau_depth.csv", index=False)

    existing = pd.read_parquet(rv_root / "ep_flux_terms" / "continuous_ep_flux_profiles.parquet")
    existing = existing[existing["axis_mode"].eq("tilted")].copy()
    comp = tau_depth.merge(
        existing[["polarity", "tau_center", "depth_index", "depth_m", "r_over_R", "pv_flux", "divF", "count"]],
        on=["polarity", "tau_center", "depth_index", "depth_m", "r_over_R"],
        how="left",
    )
    comp.to_parquet(output_dir / "comparison_with_existing_ep_flux.parquet", index=False)
    comp.to_csv(output_dir / "comparison_with_existing_ep_flux.csv", index=False)

    relations = [
        ("center_formula_vs_meridional_clim_vq", "F_dev_center", "measured_vprime_clim_qprime"),
        ("center_formula_decomp_vs_meridional_clim_vq", "F_dev_center_decomp", "measured_vprime_clim_qprime"),
        ("linear_displacement_vs_meridional_clim_vq", "F_dev_linear", "measured_vprime_clim_qprime"),
        ("axis_measured_reconstruction_vs_existing_pv_flux", "measured_unprime_az_qprime", "pv_flux"),
        ("center_formula_vs_existing_axis_pv_flux", "F_dev_center", "pv_flux"),
        ("linear_displacement_vs_existing_axis_pv_flux", "F_dev_linear", "pv_flux"),
        ("center_formula_vs_divF", "F_dev_center", "divF"),
        ("linear_displacement_vs_divF", "F_dev_linear", "divF"),
    ]
    summary_rows = []
    for pol, part in comp.groupby("polarity", sort=True):
        for relation, pred_col, obs_col in relations:
            pred = part[pred_col].to_numpy(dtype="float64")
            obs = part[obs_col].to_numpy(dtype="float64")
            summary_rows.append(
                {
                    "relation": relation,
                    "polarity": pol,
                    "predictor": pred_col,
                    "observed": obs_col,
                    "corr": _corr(pred, obs),
                    "same_sign_fraction": _same_sign(pred, obs),
                    "slope": _slope(pred, obs),
                    "rmse_ratio": _rmse_ratio(pred, obs),
                    "n_bins": int(np.sum(np.isfinite(pred) & np.isfinite(obs))),
                }
            )
    summary = pd.DataFrame(summary_rows)
    summary["status"] = np.where(
        (summary["corr"] >= 0.6) & (summary["same_sign_fraction"] >= 0.65) & (summary["rmse_ratio"] <= 1.0),
        "pass",
        np.where((summary["corr"] >= 0.25) | (summary["same_sign_fraction"] >= 0.58), "partial", "fail"),
    )
    summary.to_csv(output_dir / "deviation_pv_flux_summary.csv", index=False)
    with (output_dir / "deviation_pv_flux_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary.to_dict(orient="records"), handle, ensure_ascii=False, indent=2)

    core = comp[comp["r_over_R"] <= 1.5].copy()
    for pol, part in core.groupby("polarity", sort=True):
        for col in ["F_dev_center", "F_dev_linear", "measured_vprime_clim_qprime", "pv_flux", "divF"]:
            piv = part.pivot_table(index="depth_m", columns="tau_center", values=col)
            if piv.empty:
                continue
            vmax = np.nanpercentile(np.abs(piv.to_numpy()), 95)
            if not np.isfinite(vmax) or vmax <= 0:
                vmax = 1.0
            fig, ax = plt.subplots(figsize=(7, 5), dpi=160)
            im = ax.imshow(
                piv.to_numpy(),
                origin="lower",
                aspect="auto",
                cmap="coolwarm",
                vmin=-vmax,
                vmax=vmax,
                extent=[float(piv.columns.min()), float(piv.columns.max()), float(piv.index.min()), float(piv.index.max())],
            )
            ax.set_xlabel("tau")
            ax.set_ylabel("depth m")
            ax.set_title(f"{pol}: {col}, r/R<=1.5")
            fig.colorbar(im, ax=ax)
            fig.tight_layout()
            fig.savefig(fig_dir / f"{pol}_{col}_strict_tau_depth_core.png")
            plt.close(fig)
        for pred_col, obs_col, name in [
            ("F_dev_center", "measured_vprime_clim_qprime", "center_vs_meridional_vq"),
            ("F_dev_linear", "measured_vprime_clim_qprime", "linear_vs_meridional_vq"),
            ("measured_unprime_az_qprime", "pv_flux", "reconstructed_axis_vs_existing"),
        ]:
            good = np.isfinite(part[pred_col]) & np.isfinite(part[obs_col])
            fig, ax = plt.subplots(figsize=(5, 5), dpi=160)
            ax.scatter(part.loc[good, pred_col], part.loc[good, obs_col], s=8, alpha=0.4)
            ax.set_xlabel(pred_col)
            ax.set_ylabel(obs_col)
            ax.set_title(f"{pol}: {name}")
            ax.grid(True, color="0.9")
            fig.tight_layout()
            fig.savefig(fig_dir / f"{pol}_{name}_scatter.png")
            plt.close(fig)

    lines = ["# Deviation-induced PV flux 楠岃瘉缁撴灉\n\n"]
    lines.append("鏈姤鍛婂尯鍒嗕袱涓彛寰勶細鍏ㄧ悆缁忓悜 `v' q'` 鐢ㄦ潵妫€楠?deviation 鏄惁浜х敓缁忓悜 PV flux锛涘€炬枩杞存硶鍚?`u_n' q'` 鐢ㄦ潵鍜屾棦鏈?E-P `pv_flux/divF` 杈撳嚭瀵归綈銆俓n\n")
    for row in summary.itertuples(index=False):
        lines.append(f"## {row.relation} / {row.polarity}\n\n")
        lines.append(f"- 鍒ゅ畾锛歿row.status}\n")
        lines.append(f"- corr: {row.corr:.4g}\n")
        lines.append(f"- same_sign_fraction: {row.same_sign_fraction:.4g}\n")
        lines.append(f"- slope: {row.slope:.4g}\n")
        lines.append(f"- rmse_ratio: {row.rmse_ratio:.4g}\n")
        lines.append(f"- n_bins: {row.n_bins}\n\n")
    lines.append("## 瑙ｉ噴\n\n")
    lines.append("- 鑻?`center_formula_vs_meridional_clim_vq` 閫氳繃锛岃鏄?`M_q/A_z * V_dev,y` 鑳界洿鎺ヨВ閲婄粡鍚戞壈鍔?PV 閫氶噺銆俓n")
    lines.append("- 鑻?`linear_displacement_vs_meridional_clim_vq` 閫氳繃锛岃鏄庤儗鏅?PV 姊害绾挎€т綅绉昏繎浼?`-qbar_y xi_y Dxi_y/Dt` 鍙敤銆俓n")
    lines.append("- 鑻?`axis_measured_reconstruction_vs_existing_pv_flux` 閫氳繃锛岃鏄庝弗鏍奸噸鏋勭殑 `q_prime` 鍜屾棦鏈?representative E-P 璇婃柇鍙ｅ緞涓€鑷淬€俓n")
    lines.append("- 鑻?deviation 鍏紡鑳借В閲?`pv_flux/divF` 浣嗕笉鑳借В閲婄粡鍚?`v'q'`锛屽垯浠ｈ〃瀹冧富瑕佸搴斿€炬枩杞存硶鍚戦€氶噺鑰屼笉鏄叏鐞冪粡鍚戦€氶噺銆俓n")
    (output_dir / "deviation_pv_flux_summary_zh.md").write_text("".join(lines), encoding="utf-8")


def _inject_config_paths(config: dict, results_root: Path, rv_root: Path) -> dict:
    cfg = dict(config)
    rep = dict(cfg.get("representative", {}))
    rep.setdefault("climatology_path", str(results_root / "climatology" / "cmems_doy_climatology_2020_2022_31d.nc"))
    rep.setdefault("n2_profile_path", str(rv_root / "climatology" / "cmems_doy_climatology_2020_2022_31d_sigma0_dz_profile.npz"))
    cfg["representative"] = rep
    return cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate deviation-induced PV flux for ACC coherent representative vortices.")
    parser.add_argument("--config", default="/root/Verify/config/config_acc_2020_2022_cpu.yaml")
    parser.add_argument("--results-root", default="/root/autodl-fs/2020_2022_acc/result")
    parser.add_argument("--rv-root", default="/root/autodl-fs/2020_2022_acc/result_coherent_only/representative_vortex")
    parser.add_argument("--output-dir", default="/root/autodl-fs/2020_2022_acc/result_coherent_only/Diagonise_EP_Chen_one/deviation_pv_flux_validation")
    parser.add_argument("--mode", choices=("proxy", "strict", "finalize", "all"), default="all")
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--kernel-bandwidth", type=float, default=0.075)
    parser.add_argument("--rmax", type=float, default=2.5)
    parser.add_argument("--radial-bins", type=int, default=40)
    parser.add_argument("--azimuth-bins", type=int, default=72)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rv_root = Path(args.rv_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = _inject_config_paths(load_config(Path(args.config)), Path(args.results_root), rv_root)
    if args.start:
        parse_ymd(args.start)
    if args.end:
        parse_ymd(args.end)
    if args.mode in ("proxy", "all"):
        _proxy_validation(rv_root, output_dir)
    if args.mode in ("strict", "all"):
        _run_strict(args, config, rv_root, output_dir)
    if args.mode in ("finalize", "all"):
        _finalize(rv_root, output_dir)


if __name__ == "__main__":
    main()
