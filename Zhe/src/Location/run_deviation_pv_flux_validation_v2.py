from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.First_temp.axis_streamfunction_separation import grid_spacing_m, relative_vorticity, streamfunction_from_zeta
from src.First_temp.lifecycle_ep_flux_nondim_validation import (
    azimuth_second_derivative,
    ddz,
    load_n2,
    make_polar_grid,
    polar_gradients,
    radial_derivative,
    read_climatology_uv,
)
from src.First_temp.tilted_ep_flux_validation import bilinear_sample, sanitize_ocean_field, xy_to_lonlat
from src.Location.common import load_config, parse_ymd
from src.Location.run_deviation_pv_flux_validation import (
    OMEGA,
    _center_kinematics,
    _center_lines_from_points,
    _corr,
    _inject_config_paths,
    _load_objects,
    _load_points,
    _read_tau_grid,
    _rmse_ratio,
    _same_sign,
    _slope,
)
from src.Location.streaming_cmems import read_day_data


SECONDS_PER_DAY = 86400.0


def _q_total_components(fields: dict[str, np.ndarray], depth: np.ndarray, radial: np.ndarray, theta: np.ndarray, radius_m: float, n2: np.ndarray, f0: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return QG-like q_total, azimuthal mean Q(r,z), and non-axisymmetric q."""
    psi = np.where(np.abs(fields["psi_prime"]) > 1e20, np.nan, fields["psi_prime"])
    psi_base = psi - np.nanmean(psi, axis=2, keepdims=True)
    r_m = np.maximum(radial * radius_m, 1.0)
    dpsi_dr = radial_derivative(psi_base, r_m)
    radial_lap = radial_derivative(dpsi_dr * r_m[None, :, None], r_m) / r_m[None, :, None]
    az_lap = azimuth_second_derivative(psi_base, theta) / (r_m[None, :, None] ** 2)
    dpsi_dz = ddz(psi_base, depth)
    strat = (f0 * f0 / n2)[:, None, None] * dpsi_dz
    q_total = radial_lap + az_lap + ddz(strat, depth)
    q_mean = np.nanmean(q_total, axis=2)
    q_anom = q_total - q_mean[:, :, None]
    return q_total, q_mean, q_anom


def _radial_modes(radial: np.ndarray, radius_m: float, profile: np.ndarray, theta: np.ndarray, max_order: int = 3) -> dict[int, np.ndarray]:
    """Directional Taylor modes for a radial profile Q(r,z) along x_rot."""
    r = np.maximum(radial * radius_m, 1.0)
    cos_t = np.cos(theta)[None, None, :]
    sin_t = np.sin(theta)[None, None, :]
    f0 = profile
    f1 = radial_derivative(f0[:, :, None], r)[:, :, 0]
    f2 = radial_derivative(f1[:, :, None], r)[:, :, 0]
    f3 = radial_derivative(f2[:, :, None], r)[:, :, 0]
    rr = r[None, :, None]
    d1 = f1[:, :, None] * cos_t
    d2 = f2[:, :, None] * cos_t * cos_t + (f1[:, :, None] / rr) * sin_t * sin_t
    d3 = (
        f3[:, :, None] * cos_t**3
        + 3.0 * cos_t * sin_t**2 * (f2[:, :, None] / rr - f1[:, :, None] / (rr * rr))
    )
    return {0: f0[:, :, None] + np.zeros((profile.shape[0], profile.shape[1], len(theta))), 1: d1, 2: d2, 3: d3}


def _tilt_modes_from_radial(profile: np.ndarray, radial: np.ndarray, theta: np.ndarray, radius_m: float, a_m: np.ndarray, max_order: int = 3) -> dict[int, np.ndarray]:
    derivs = _radial_modes(radial, radius_m, profile, theta, max_order=max_order)
    modes = {0: derivs[0]}
    for n in range(1, max_order + 1):
        modes[n] = ((-1.0) ** n / math.factorial(n)) * (a_m[:, None, None] ** n) * derivs[n]
    return modes


def _psi_velocity_modes(psi_modes: dict[int, np.ndarray], radial: np.ndarray, theta: np.ndarray, radius_m: float, angle_rad: float) -> dict[int, np.ndarray]:
    out: dict[int, np.ndarray] = {}
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    for order, psi in psi_modes.items():
        grad_xrot, grad_yrot = polar_gradients(psi, radial, theta, radius_m)
        # Existing rotation convention: x_orig = x_rot cos - y_rot sin.
        # Hence d/dx_global = cos*d/dx_rot - sin*d/dy_rot, and v=psi_x.
        out[order] = grad_xrot * cos_a - grad_yrot * sin_a
    return out


def _qbar_derivatives(lon: np.ndarray, lat: np.ndarray, depth: np.ndarray, u_clim: np.ndarray, v_clim: np.ndarray, n2: np.ndarray, f0: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    _, dy, dx = grid_spacing_m(lon, lat)
    zeta = relative_vorticity(lon, lat, u_clim, v_clim)
    psi = streamfunction_from_zeta(zeta, dx, dy)
    strat = (f0 * f0 / n2)[:, None, None] * ddz(psi, depth)
    qbar = zeta + ddz(strat, depth)
    qy = np.gradient(qbar, dy, axis=1, edge_order=1)
    qyy = np.gradient(qy, dy, axis=1, edge_order=1)
    qyyy = np.gradient(qyy, dy, axis=1, edge_order=1)
    return qy, qyy, qyyy


def _frame_targets(obj, center_line: pd.DataFrame, radius_m: float, rr: np.ndarray, tt: np.ndarray, *, reference_column: bool) -> tuple[list[np.ndarray], list[np.ndarray]]:
    theta_obj = float(obj.temp_direction_rad)
    cos_t = math.cos(theta_obj)
    sin_t = math.sin(theta_obj)
    local_x = rr * radius_m * np.cos(tt)
    local_y = rr * radius_m * np.sin(tt)
    axis_x = np.zeros(len(center_line), dtype="float64") if reference_column else center_line["x_rot_m"].to_numpy(dtype="float64")
    axis_y = np.zeros(len(center_line), dtype="float64") if reference_column else center_line["y_rot_m"].to_numpy(dtype="float64")
    lons: list[np.ndarray] = []
    lats: list[np.ndarray] = []
    for k in range(len(center_line)):
        x_rot = axis_x[k] + local_x
        y_rot = axis_y[k] + local_y
        x_orig = x_rot * cos_t - y_rot * sin_t
        y_orig = x_rot * sin_t + y_rot * cos_t
        target_lon, target_lat = xy_to_lonlat(x_orig, y_orig, float(obj.surface_lon), float(obj.surface_lat))
        lons.append(target_lon)
        lats.append(target_lat)
    return lons, lats


def _sample_frame(obj, center_line: pd.DataFrame, lon: np.ndarray, lat: np.ndarray, depth: np.ndarray, psi_prime: np.ndarray, u: np.ndarray, v: np.ndarray, u_clim: np.ndarray, v_clim: np.ndarray, rr: np.ndarray, tt: np.ndarray, *, reference_column: bool) -> dict[str, np.ndarray] | None:
    if len(center_line) != len(depth):
        return None
    radius_m = float(obj.mean_radius_m)
    target_lons, target_lats = _frame_targets(obj, center_line, radius_m, rr, tt, reference_column=reference_column)
    theta_obj = float(obj.temp_direction_rad)
    cos_t = math.cos(theta_obj)
    sin_t = math.sin(theta_obj)
    psi_layers, v_global_layers, un_raw_layers, un_clim_anom_layers = [], [], [], []
    v_clim_layers = []
    for k in range(len(depth)):
        tlon = target_lons[k]
        tlat = target_lats[k]
        psi_s = bilinear_sample(lon, lat, psi_prime[k], tlon, tlat)
        u_samp = bilinear_sample(lon, lat, u[k], tlon, tlat)
        v_samp = bilinear_sample(lon, lat, v[k], tlon, tlat)
        u_clim_samp = bilinear_sample(lon, lat, u_clim[k], tlon, tlat)
        v_clim_samp = bilinear_sample(lon, lat, v_clim[k], tlon, tlat)
        u_rot = u_samp * cos_t + v_samp * sin_t
        v_rot = -u_samp * sin_t + v_samp * cos_t
        u_clim_rot = u_clim_samp * cos_t + v_clim_samp * sin_t
        v_clim_rot = -u_clim_samp * sin_t + v_clim_samp * cos_t
        un = u_rot * np.cos(tt) + v_rot * np.sin(tt)
        un_clim = u_clim_rot * np.cos(tt) + v_clim_rot * np.sin(tt)
        psi_layers.append(psi_s)
        v_global_layers.append(v_samp - v_clim_samp)
        v_clim_layers.append(v_clim_samp)
        un_raw_layers.append(un)
        un_clim_anom_layers.append(un - un_clim)
    return {
        "psi_prime": np.asarray(psi_layers),
        "v_prime_global_clim": np.asarray(v_global_layers),
        "v_clim_global": np.asarray(v_clim_layers),
        "u_n_raw": np.asarray(un_raw_layers),
        "u_n_prime_clim": np.asarray(un_clim_anom_layers),
        "target_lons": target_lons,
        "target_lats": target_lats,
    }


def _sample_stack(stack: np.ndarray, lon: np.ndarray, lat: np.ndarray, target_lons: list[np.ndarray], target_lats: list[np.ndarray]) -> np.ndarray:
    layers = []
    for k in range(stack.shape[0]):
        layers.append(bilinear_sample(lon, lat, stack[k], target_lons[k], target_lats[k]))
    return np.asarray(layers)


def _strict_day_v2(
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
    max_order: int,
) -> Path:
    date = pd.Timestamp(day).date()
    part_dir = output_dir / "deviation_pv_flux_decomposition_parts" / f"year={date.year}"
    part_dir.mkdir(parents=True, exist_ok=True)
    part_path = part_dir / f"deviation_pv_flux_v2_{date:%Y%m%d}.parquet"
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
    qbar_y, qbar_yy, qbar_yyy = _qbar_derivatives(lon, lat, depth, u_clim, v_clim, n2, f0)
    radial, theta, rr, tt, _ = make_polar_grid(rmax, radial_bins, azimuth_bins)

    rows = []
    for obj in day_objects.itertuples(index=False):
        object_id = int(obj.eddy3d_object_id)
        center_line = center_lines.get(object_id)
        kin_line = kin_by_object.get(object_id)
        if center_line is None or kin_line is None or len(center_line) != len(depth) or len(kin_line) != len(depth):
            continue
        ref = _sample_frame(obj, center_line, lon, lat, depth, psi_prime, u, v, u_clim, v_clim, rr, tt, reference_column=True)
        tilted = _sample_frame(obj, center_line, lon, lat, depth, psi_prime, u, v, u_clim, v_clim, rr, tt, reference_column=False)
        if ref is None or tilted is None:
            continue
        q_ref_total, _, q_ref_anom = _q_total_components(ref, depth, radial, theta, float(obj.mean_radius_m), n2, f0)
        q_tilted_total, q_core, q_tilted_anom = _q_total_components(tilted, depth, radial, theta, float(obj.mean_radius_m), n2, f0)
        psi_core = np.nanmean(tilted["psi_prime"], axis=2)

        kin_line = kin_line.sort_values("depth_index")
        a_m = kin_line["a_m"].to_numpy(dtype="float64")
        xi_y = kin_line["xi_y_m"].to_numpy(dtype="float64")
        vdev_y = kin_line["vdev_y_actual_m_day"].to_numpy(dtype="float64") / SECONDS_PER_DAY
        q_modes = _tilt_modes_from_radial(q_core, radial, theta, float(obj.mean_radius_m), a_m, max_order=max_order)
        psi_modes = _tilt_modes_from_radial(psi_core, radial, theta, float(obj.mean_radius_m), a_m, max_order=max_order)
        v_modes = _psi_velocity_modes(psi_modes, radial, theta, float(obj.mean_radius_m), float(obj.temp_direction_rad))
        q_tilt_sum = sum(q_modes[n] for n in range(0, max_order + 1))
        qy = _sample_stack(qbar_y, lon, lat, ref["target_lons"], ref["target_lats"])
        qyy = _sample_stack(qbar_yy, lon, lat, ref["target_lons"], ref["target_lats"])
        qyyy = _sample_stack(qbar_yyy, lon, lat, ref["target_lons"], ref["target_lats"])
        q_bg_1 = -xi_y[:, None, None] * qy
        q_bg_2 = 0.5 * xi_y[:, None, None] ** 2 * qyy
        q_bg_3 = -(1.0 / 6.0) * xi_y[:, None, None] ** 3 * qyyy
        q_bg = q_bg_1 + q_bg_2 + q_bg_3
        q_asym = q_ref_total - q_tilt_sum - q_bg
        v_obs = ref["v_prime_global_clim"]
        un_tilted_az = tilted["u_n_raw"] - np.nanmean(tilted["u_n_raw"], axis=2, keepdims=True)
        f_obs_meridional = np.nanmean(v_obs * q_ref_total, axis=2)
        f_obs_meridional_anom = np.nanmean(v_obs * q_ref_anom, axis=2)
        f_existing_axis_recon = np.nanmean(un_tilted_az * q_tilted_anom, axis=2)
        f_trap = vdev_y[:, None] * q_core
        order_terms: dict[int, np.ndarray] = {}
        for order in range(1, 2 * max_order + 1):
            total = np.zeros_like(q_core)
            for m in range(0, max_order + 1):
                n = order - m
                if 0 <= n <= max_order and order >= 1:
                    total += np.nanmean(v_modes[m] * q_modes[n], axis=2)
            order_terms[order] = total
        f_tilt_poly = sum(order_terms.values())
        f_tilt_order1 = order_terms.get(1, np.zeros_like(q_core))
        f_tilt_order2 = order_terms.get(2, np.zeros_like(q_core))
        f_tilt_order3 = order_terms.get(3, np.zeros_like(q_core))
        f_tilt_high_cross = sum(order_terms[o] for o in range(4, 2 * max_order + 1))
        f_bg_1 = np.nanmean(v_obs * q_bg_1, axis=2)
        f_bg_2 = np.nanmean(v_obs * q_bg_2, axis=2)
        f_bg_3 = np.nanmean(v_obs * q_bg_3, axis=2)
        f_bg = f_bg_1 + f_bg_2 + f_bg_3
        f_asym = np.nanmean(v_obs * q_asym, axis=2)
        f_residual = f_obs_meridional - f_trap - f_tilt_poly - f_bg - f_asym
        weights = np.exp(-0.5 * ((tau_grid - float(obj.life_phase)) / max(bandwidth, 1e-12)) ** 2)
        for tau_index in np.where(weights >= 1e-4)[0]:
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
                            "F_obs_meridional": float(f_obs_meridional[k, j]),
                            "F_obs_meridional_anom": float(f_obs_meridional_anom[k, j]),
                            "F_existing_axis_recon": float(f_existing_axis_recon[k, j]),
                            "F_trap": float(f_trap[k, j]),
                            "F_tilt_poly": float(f_tilt_poly[k, j]),
                            "F_tilt_order1": float(f_tilt_order1[k, j]),
                            "F_tilt_order2": float(f_tilt_order2[k, j]),
                            "F_tilt_order3": float(f_tilt_order3[k, j]),
                            "F_tilt_high_cross": float(f_tilt_high_cross[k, j]),
                            "F_bg": float(f_bg[k, j]),
                            "F_bg_order1": float(f_bg_1[k, j]),
                            "F_bg_order2": float(f_bg_2[k, j]),
                            "F_bg_order3": float(f_bg_3[k, j]),
                            "F_asym_direct": float(f_asym[k, j]),
                            "F_residual_nl": float(f_residual[k, j]),
                            "q_core": float(q_core[k, j]),
                            "q_obs_ref_mean": float(np.nanmean(q_ref_total[k, j, :])),
                            "q_asym_rms": float(np.sqrt(np.nanmean(q_asym[k, j, :] ** 2))),
                        }
                    )
    tmp_path = part_path.with_suffix(part_path.suffix + ".tmp")
    pd.DataFrame.from_records(rows).to_parquet(tmp_path, index=False)
    tmp_path.replace(part_path)
    return part_path


def _run_strict(args: argparse.Namespace, config: dict, rv_root: Path, output_dir: Path) -> None:
    objects = _load_objects(rv_root, args.start, args.end)
    points = _load_points(rv_root, set(objects["eddy3d_object_id"].astype("int64")))
    kin = _center_kinematics(points)
    tau_grid = _read_tau_grid(rv_root)
    log_rows = []
    for day in sorted(objects["date"].unique()):
        day_objects = objects[objects["date"].eq(day)].copy()
        path = _strict_day_v2(
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
            int(args.poly_order),
        )
        log_rows.append({"date": day, "objects": int(len(day_objects)), "path": str(path)})
        if len(log_rows) % 10 == 0:
            pd.DataFrame(log_rows).to_csv(output_dir / f"strict_v2_progress_{args.start or 'all'}_{args.end or 'all'}.csv", index=False)
            print(f"processed {len(log_rows)} days through {day}", flush=True)
    pd.DataFrame(log_rows).to_csv(output_dir / f"strict_v2_progress_{args.start or 'all'}_{args.end or 'all'}.csv", index=False)


def _weighted_streaming_aggregate(part_paths: list[Path], value_cols: list[str]) -> pd.DataFrame:
    keys = ["polarity", "tau_center", "depth_index", "depth_m", "r_over_R"]
    accum: pd.DataFrame | None = None
    for i, path in enumerate(part_paths, start=1):
        df = pd.read_parquet(path)
        if df.empty:
            continue
        df = df.copy()
        weights = df["weight"].to_numpy(dtype="float64")
        df["weighted_count"] = weights
        for col in value_cols:
            vals = df[col].to_numpy(dtype="float64")
            ok = np.isfinite(vals) & np.isfinite(weights) & (weights > 0)
            df[f"{col}__sum"] = np.where(ok, vals * weights, 0.0)
            df[f"{col}__w"] = np.where(ok, weights, 0.0)
        sum_cols = ["weighted_count", "n_objects"] + [f"{col}__sum" for col in value_cols] + [f"{col}__w" for col in value_cols]
        grouped = df.groupby(keys, sort=True)[sum_cols].sum()
        accum = grouped if accum is None else accum.add(grouped, fill_value=0.0)
        if i % 100 == 0:
            print(f"finalize v2 aggregated {i}/{len(part_paths)} parts", flush=True)
    if accum is None or accum.empty:
        raise SystemExit("No non-empty V2 strict part files found")
    out = accum.reset_index()
    for col in value_cols:
        numer = out[f"{col}__sum"].to_numpy(dtype="float64")
        denom = out[f"{col}__w"].to_numpy(dtype="float64")
        out[col] = np.divide(numer, denom, out=np.full_like(numer, np.nan), where=denom > 0)
    return out[keys + ["weighted_count", "n_objects"] + value_cols]


def _finalize(rv_root: Path, output_dir: Path) -> None:
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    part_paths = sorted((output_dir / "deviation_pv_flux_decomposition_parts").glob("year=*/*.parquet"))
    if not part_paths:
        raise SystemExit("No V2 strict part parquet files found")
    value_cols = [
        "F_obs_meridional",
        "F_obs_meridional_anom",
        "F_existing_axis_recon",
        "F_trap",
        "F_tilt_poly",
        "F_tilt_order1",
        "F_tilt_order2",
        "F_tilt_order3",
        "F_tilt_high_cross",
        "F_bg",
        "F_bg_order1",
        "F_bg_order2",
        "F_bg_order3",
        "F_asym_direct",
        "F_residual_nl",
        "q_core",
        "q_obs_ref_mean",
        "q_asym_rms",
    ]
    decomposed = _weighted_streaming_aggregate(part_paths, value_cols)
    decomposed.to_parquet(output_dir / "deviation_pv_flux_decomposition_tau_depth.parquet", index=False)
    decomposed.to_csv(output_dir / "deviation_pv_flux_decomposition_tau_depth.csv", index=False)

    existing = pd.read_parquet(rv_root / "ep_flux_terms" / "continuous_ep_flux_profiles.parquet")
    existing = existing[existing["axis_mode"].eq("tilted")].copy()
    comp = decomposed.merge(
        existing[["polarity", "tau_center", "depth_index", "depth_m", "r_over_R", "pv_flux", "divF", "count"]],
        on=["polarity", "tau_center", "depth_index", "depth_m", "r_over_R"],
        how="left",
    )
    comp.to_parquet(output_dir / "comparison_with_existing_ep_flux_v2.parquet", index=False)
    comp.to_csv(output_dir / "comparison_with_existing_ep_flux_v2.csv", index=False)

    relations = [
        ("sanity_axis_reconstruction_vs_existing_pv_flux", "F_existing_axis_recon", "pv_flux"),
        ("trap_vs_observed_meridional", "F_trap", "F_obs_meridional"),
        ("tilt_poly_vs_observed_meridional", "F_tilt_poly", "F_obs_meridional"),
        ("bg_vs_observed_meridional", "F_bg", "F_obs_meridional"),
        ("asym_vs_observed_meridional", "F_asym_direct", "F_obs_meridional"),
        ("trap_plus_tilt_plus_bg_plus_asym_vs_observed", "F_sum_no_residual", "F_obs_meridional"),
        ("residual_vs_observed_meridional", "F_residual_nl", "F_obs_meridional"),
        ("tilt_poly_vs_existing_axis_pv_flux", "F_tilt_poly", "pv_flux"),
        ("asym_vs_existing_axis_pv_flux", "F_asym_direct", "pv_flux"),
    ]
    comp["F_sum_no_residual"] = comp["F_trap"] + comp["F_tilt_poly"] + comp["F_bg"] + comp["F_asym_direct"]
    rows = []
    for pol, part in comp.groupby("polarity", sort=True):
        obs_rms = float(np.sqrt(np.nanmean(part["F_obs_meridional"].to_numpy(dtype="float64") ** 2)))
        for relation, pred_col, obs_col in relations:
            pred = part[pred_col].to_numpy(dtype="float64")
            obs = part[obs_col].to_numpy(dtype="float64")
            pred_rms = float(np.sqrt(np.nanmean(pred**2)))
            rows.append(
                {
                    "relation": relation,
                    "polarity": pol,
                    "predictor": pred_col,
                    "observed": obs_col,
                    "corr": _corr(pred, obs),
                    "same_sign_fraction": _same_sign(pred, obs),
                    "slope": _slope(pred, obs),
                    "rmse_ratio": _rmse_ratio(pred, obs),
                    "rms_ratio_to_meridional_obs": pred_rms / obs_rms if obs_rms > 0 and np.isfinite(obs_rms) else np.nan,
                    "n_bins": int(np.sum(np.isfinite(pred) & np.isfinite(obs))),
                }
            )
        for col in ["F_trap", "F_tilt_order1", "F_tilt_order2", "F_tilt_order3", "F_tilt_high_cross", "F_bg", "F_asym_direct", "F_residual_nl"]:
            vals = part[col].to_numpy(dtype="float64")
            rows.append(
                {
                    "relation": f"component_amplitude_{col}",
                    "polarity": pol,
                    "predictor": col,
                    "observed": "F_obs_meridional",
                    "corr": _corr(vals, part["F_obs_meridional"].to_numpy(dtype="float64")),
                    "same_sign_fraction": _same_sign(vals, part["F_obs_meridional"].to_numpy(dtype="float64")),
                    "slope": _slope(vals, part["F_obs_meridional"].to_numpy(dtype="float64")),
                    "rmse_ratio": _rmse_ratio(vals, part["F_obs_meridional"].to_numpy(dtype="float64")),
                    "rms_ratio_to_meridional_obs": float(np.sqrt(np.nanmean(vals**2))) / obs_rms if obs_rms > 0 and np.isfinite(obs_rms) else np.nan,
                    "n_bins": int(np.sum(np.isfinite(vals) & np.isfinite(part["F_obs_meridional"]))),
                }
            )
    summary = pd.DataFrame(rows)
    summary["status"] = np.where(
        (summary["relation"].str.startswith("sanity")) & (summary["corr"] > 0.95) & (summary["same_sign_fraction"] > 0.95),
        "pass",
        np.where((summary["corr"].abs() >= 0.35) | (summary["rms_ratio_to_meridional_obs"] >= 0.5), "partial", "fail"),
    )
    summary.to_csv(output_dir / "deviation_pv_flux_v2_summary.csv", index=False)
    with (output_dir / "deviation_pv_flux_v2_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary.to_dict(orient="records"), handle, ensure_ascii=False, indent=2)
    _write_report(output_dir, summary, decomposed, len(part_paths))
    _plot_outputs(comp, fig_dir)


def _plot_outputs(comp: pd.DataFrame, fig_dir: Path) -> None:
    core = comp[comp["r_over_R"] <= 1.5].copy()
    plot_cols = ["F_obs_meridional", "F_trap", "F_tilt_poly", "F_bg", "F_asym_direct", "F_residual_nl", "pv_flux", "divF"]
    for pol, part in core.groupby("polarity", sort=True):
        for col in plot_cols:
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
            fig.savefig(fig_dir / f"{pol}_{col}_tau_depth_core.png")
            plt.close(fig)
        for pred_col, obs_col, name in [
            ("F_tilt_poly", "F_obs_meridional", "tilt_poly_vs_obs_meridional"),
            ("F_asym_direct", "F_obs_meridional", "asym_vs_obs_meridional"),
            ("F_existing_axis_recon", "pv_flux", "axis_sanity"),
        ]:
            good = np.isfinite(part[pred_col]) & np.isfinite(part[obs_col])
            fig, ax = plt.subplots(figsize=(5, 5), dpi=160)
            ax.scatter(part.loc[good, pred_col], part.loc[good, obs_col], s=8, alpha=0.35)
            ax.set_xlabel(pred_col)
            ax.set_ylabel(obs_col)
            ax.set_title(f"{pol}: {name}")
            ax.grid(True, color="0.9")
            fig.tight_layout()
            fig.savefig(fig_dir / f"{pol}_{name}_scatter.png")
            plt.close(fig)


def _write_report(output_dir: Path, summary: pd.DataFrame, decomposed: pd.DataFrame, part_count: int) -> None:
    lines = ["# Deviation-induced PV flux V2 诊断报告\n\n"]
    lines.append(f"全量 strict decomposition 覆盖 `{part_count}` 个 daily part；最终 tau-depth-radius 表有 `{len(decomposed)}` 行。\n\n")
    lines.append("本轮不再把 deviation 简化为中心线漂移，而是分解为 `trap + tilt-poly + bg + asym + residual/nl`。其中 `Q(r,z)` 来自每个对象 `q'` 的方位平均，tilt polynomial 展开到三阶。\n\n")
    for rel in [
        "sanity_axis_reconstruction_vs_existing_pv_flux",
        "trap_vs_observed_meridional",
        "tilt_poly_vs_observed_meridional",
        "bg_vs_observed_meridional",
        "asym_vs_observed_meridional",
        "trap_plus_tilt_plus_bg_plus_asym_vs_observed",
        "residual_vs_observed_meridional",
    ]:
        lines.append(f"## {rel}\n\n")
        for row in summary[summary["relation"].eq(rel)].itertuples(index=False):
            lines.append(
                f"- {row.polarity}: status `{row.status}`, corr `{row.corr:.4g}`, 同号率 `{row.same_sign_fraction:.4g}`, "
                f"RMSE ratio `{row.rmse_ratio:.4g}`, RMS/obs `{row.rms_ratio_to_meridional_obs:.4g}`。\n"
            )
        lines.append("\n")
    lines.append("## 机制项幅度\n\n")
    for row in summary[summary["relation"].str.startswith("component_amplitude_")].itertuples(index=False):
        lines.append(f"- {row.polarity} / {row.predictor}: RMS/obs `{row.rms_ratio_to_meridional_obs:.4g}`, corr `{row.corr:.4g}`, status `{row.status}`。\n")
    lines.append("\n## 解释框架\n\n")
    lines.append("- `sanity_axis_reconstruction` 若 pass，说明 V2 的 `q'` 重构仍与既有 E-P `pv_flux` 对齐。\n")
    lines.append("- 若 `F_trap` 幅度远小于 `F_asym_direct` 或 `F_residual_nl`，说明 PV transport 不是不漏水 PV blob 的整体搬运。\n")
    lines.append("- 若 `F_tilt_order1/2/3` 中某阶幅度显著，说明倾斜造成的多项式非轴对称结构确实参与通量。\n")
    lines.append("- 若 `F_asym_direct` 主导，说明内部 stirring、filament、相位差或非轴对称 PV 结构是主要来源。\n")
    lines.append("- 若 `F_residual_nl` 仍大，说明三阶 tilt polynomial 与背景 PV 梯度仍不足以闭合，需考虑更高阶、边界剥离或平均流反馈。\n")
    (output_dir / "deviation_pv_flux_v2_summary_zh.md").write_text("".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="V2 decomposition of ACC coherent deviation-induced PV flux.")
    parser.add_argument("--config", default="/root/Verify/config/config_acc_2020_2022_cpu.yaml")
    parser.add_argument("--results-root", default="/root/autodl-fs/2020_2022_acc/result")
    parser.add_argument("--rv-root", default="/root/autodl-fs/2020_2022_acc/result_coherent_only/representative_vortex")
    parser.add_argument("--output-dir", default="/root/autodl-fs/2020_2022_acc/result_coherent_only/Diagonise_EP_Chen_one/deviation_pv_flux_validation_v2")
    parser.add_argument("--mode", choices=("strict", "finalize", "all"), default="all")
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--poly-order", type=int, default=3)
    parser.add_argument("--kernel-bandwidth", type=float, default=0.075)
    parser.add_argument("--rmax", type=float, default=2.5)
    parser.add_argument("--radial-bins", type=int, default=40)
    parser.add_argument("--azimuth-bins", type=int, default=72)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.poly_order != 3:
        raise ValueError("This V2 diagnostic is locked to poly_order=3 for report comparability.")
    if args.start:
        parse_ymd(args.start)
    if args.end:
        parse_ymd(args.end)
    rv_root = Path(args.rv_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = _inject_config_paths(load_config(Path(args.config)), Path(args.results_root), rv_root)
    if args.mode in ("strict", "all"):
        _run_strict(args, config, rv_root, output_dir)
    if args.mode in ("finalize", "all"):
        _finalize(rv_root, output_dir)


if __name__ == "__main__":
    main()
