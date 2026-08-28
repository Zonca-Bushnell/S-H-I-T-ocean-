from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from .axis_streamfunction_separation import (
    DEFAULT_AXIS_DIR,
    DEFAULT_CATALOG,
    DEFAULT_INPUT_DAILY,
    grid_spacing_m,
    parse_csv_list,
    read_daily_uv,
    relative_vorticity,
    streamfunction_from_zeta,
)


EARTH_RADIUS_M = 6_371_000.0
OMEGA = 7.2921159e-5
G = 9.81
RHO0 = 1025.0
SECONDS_PER_DAY = 86400.0
DEFAULT_CLIMATOLOGY = Path(r"E:\Verify\outputs\kuroshio_cmems_3d\climatology\cmems_doy_climatology_1993_2022_31d_sigma0_dz_profile.npz")
DEFAULT_CLIMATOLOGY_NC = Path(r"E:\Verify\outputs\kuroshio_cmems_3d\climatology\cmems_doy_climatology_1993_2022_31d.nc")
DEFAULT_OUTPUT = Path(r"E:\Verify\outputs\kuroshio_cmems_3d\tilted_ep_flux_validation_1993_2022")
DEFAULT_POLARITIES = ("cyclonic", "anticyclonic")


def local_xy_m(lon: np.ndarray, lat: np.ndarray, lon0: float, lat0: float) -> tuple[np.ndarray, np.ndarray]:
    dlon = (lon - lon0 + 180.0) % 360.0 - 180.0
    x = np.radians(dlon) * EARTH_RADIUS_M * np.cos(np.radians(lat0))
    y = np.radians(lat - lat0) * EARTH_RADIUS_M
    return x, y


def xy_to_lonlat(x: np.ndarray, y: np.ndarray, lon0: float, lat0: float) -> tuple[np.ndarray, np.ndarray]:
    lat = lat0 + np.degrees(y / EARTH_RADIUS_M)
    lon = lon0 + np.degrees(x / (EARTH_RADIUS_M * np.cos(np.radians(lat0))))
    return lon, lat


def bilinear_sample(lon: np.ndarray, lat: np.ndarray, field: np.ndarray, target_lon: np.ndarray, target_lat: np.ndarray) -> np.ndarray:
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
    return np.clip(values, 1e-7, np.inf)


def read_climatology_uv(path: Path, date: str) -> tuple[np.ndarray, np.ndarray]:
    import h5py

    timestamp = pd.Timestamp(date)
    doy_index = int(timestamp.dayofyear) - 1
    with h5py.File(path, "r") as handle:
        u_ds = handle["u_clim"]
        v_ds = handle["v_clim"]
        u = np.asarray(u_ds[doy_index], dtype="f8")
        v = np.asarray(v_ds[doy_index], dtype="f8")
        for values, ds in ((u, u_ds), (v, v_ds)):
            for attr_name in ("_FillValue", "missing_value"):
                if attr_name not in ds.attrs:
                    continue
                for fill_value in np.asarray(ds.attrs[attr_name]).ravel():
                    if np.isfinite(fill_value):
                        values[np.isclose(values, float(fill_value), rtol=0.0, atol=max(abs(float(fill_value)) * 1e-7, 1.0))] = np.nan
    u[np.abs(u) > 1e20] = np.nan
    v[np.abs(v) > 1e20] = np.nan
    return u, v


def sanitize_ocean_field(values: np.ndarray) -> np.ndarray:
    cleaned = np.asarray(values, dtype="f8").copy()
    cleaned[np.abs(cleaned) > 1e20] = np.nan
    return cleaned


def load_objects(axis_dir: Path, catalog_dir: Path, polarities: tuple[str, ...]) -> pd.DataFrame:
    obj = pd.read_parquet(axis_dir / "object_diagnostics.parquet")
    obj = obj[obj["is_usable"] & obj["polarity"].isin(polarities)].copy()
    vertical = pd.read_parquet(catalog_dir / "vertical_objects.parquet", columns=["eddy3d_object_id", "mean_radius_m"])
    obj = obj.merge(vertical, on="eddy3d_object_id", how="left")
    obj = obj[np.isfinite(obj["mean_radius_m"]) & (obj["mean_radius_m"] > 0)].copy()
    obj["date"] = pd.to_datetime(obj["date"]).dt.strftime("%Y-%m-%d")
    return obj


def limit_objects_per_polarity(objects: pd.DataFrame, max_objects_per_polarity: int, seed: int) -> pd.DataFrame:
    if max_objects_per_polarity <= 0 or objects.empty:
        return objects
    rng = np.random.default_rng(seed)
    keep: list[int] = []
    for _, part in objects.groupby("polarity"):
        ids = part["eddy3d_object_id"].to_numpy(dtype="i8")
        if len(ids) > max_objects_per_polarity:
            ids = rng.choice(ids, max_objects_per_polarity, replace=False)
        keep.extend(int(v) for v in ids)
    return objects[objects["eddy3d_object_id"].isin(keep)].copy()


def load_center_lines(axis_dir: Path, object_ids: set[int]) -> dict[int, pd.DataFrame]:
    points = pd.read_parquet(
        axis_dir / "rotated_points.parquet",
        columns=["eddy3d_object_id", "depth_index", "z_m", "x_rot_m", "y_rot_m"],
    )
    points = points[points["eddy3d_object_id"].isin(object_ids)].copy()
    return {int(object_id): part.sort_values("depth_index").copy() for object_id, part in points.groupby("eddy3d_object_id", sort=False)}


def make_polar_grid(rmax: float, radial_bins: int, azimuth_bins: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    radial_edges = np.linspace(0.0, rmax, radial_bins + 1)
    radial = 0.5 * (radial_edges[:-1] + radial_edges[1:])
    theta = np.linspace(0.0, 2.0 * np.pi, azimuth_bins, endpoint=False)
    rr, tt = np.meshgrid(radial, theta, indexing="ij")
    return radial, theta, rr, tt, radial_edges


def sample_object_fields(
    obj,
    center_line: pd.DataFrame,
    lon: np.ndarray,
    lat: np.ndarray,
    depth: np.ndarray,
    psi_prime: np.ndarray,
    u_prime: np.ndarray,
    v_prime: np.ndarray,
    u_clim: np.ndarray,
    v_clim: np.ndarray,
    u_clim_next: np.ndarray | None,
    v_clim_next: np.ndarray | None,
    rr: np.ndarray,
    tt: np.ndarray,
) -> dict[str, np.ndarray] | None:
    nz = len(depth)
    if len(center_line) != nz:
        return None
    radius = float(obj.mean_radius_m)
    theta_obj = float(obj.temp_direction_rad)
    cos_t = np.cos(theta_obj)
    sin_t = np.sin(theta_obj)
    local_x = rr * radius * np.cos(tt)
    local_y = rr * radius * np.sin(tt)
    psi_layers = []
    un_layers = []
    us_layers = []
    un_clim_layers = []
    us_clim_layers = []
    us_clim_next_layers = []
    beta_y_layers = []
    has_clim_next = u_clim_next is not None and v_clim_next is not None
    axis_x = center_line["x_rot_m"].to_numpy(dtype="f8")
    axis_y = center_line["y_rot_m"].to_numpy(dtype="f8")
    for k in range(nz):
        x_rot = axis_x[k] + local_x
        y_rot = axis_y[k] + local_y
        x_orig = x_rot * cos_t - y_rot * sin_t
        y_orig = x_rot * sin_t + y_rot * cos_t
        target_lon, target_lat = xy_to_lonlat(x_orig, y_orig, float(obj.surface_lon), float(obj.surface_lat))
        psi_s = bilinear_sample(lon, lat, psi_prime[k], target_lon, target_lat)
        u_samp = bilinear_sample(lon, lat, u_prime[k], target_lon, target_lat)
        v_samp = bilinear_sample(lon, lat, v_prime[k], target_lon, target_lat)
        u_clim_samp = bilinear_sample(lon, lat, u_clim[k], target_lon, target_lat)
        v_clim_samp = bilinear_sample(lon, lat, v_clim[k], target_lon, target_lat)
        if has_clim_next:
            u_clim_next_samp = bilinear_sample(lon, lat, u_clim_next[k], target_lon, target_lat)
            v_clim_next_samp = bilinear_sample(lon, lat, v_clim_next[k], target_lon, target_lat)
        u_rot = u_samp * cos_t + v_samp * sin_t
        v_rot = -u_samp * sin_t + v_samp * cos_t
        u_clim_rot = u_clim_samp * cos_t + v_clim_samp * sin_t
        v_clim_rot = -u_clim_samp * sin_t + v_clim_samp * cos_t
        if has_clim_next:
            u_clim_next_rot = u_clim_next_samp * cos_t + v_clim_next_samp * sin_t
            v_clim_next_rot = -u_clim_next_samp * sin_t + v_clim_next_samp * cos_t
        un = u_rot * np.cos(tt) + v_rot * np.sin(tt)
        us = -u_rot * np.sin(tt) + v_rot * np.cos(tt)
        un_clim = u_clim_rot * np.cos(tt) + v_clim_rot * np.sin(tt)
        us_clim = -u_clim_rot * np.sin(tt) + v_clim_rot * np.cos(tt)
        if has_clim_next:
            us_clim_next = -u_clim_next_rot * np.sin(tt) + v_clim_next_rot * np.cos(tt)
            us_clim_next_layers.append(us_clim_next)
        psi_layers.append(psi_s)
        un_layers.append(un)
        us_layers.append(us)
        beta_y_layers.append(y_orig)
        un_clim_layers.append(un_clim)
        us_clim_layers.append(us_clim)
    out = {
        "psi_prime": np.asarray(psi_layers),
        "u_n_prime": np.asarray(un_layers),
        "u_s_prime": np.asarray(us_layers),
        "u_n_clim": np.asarray(un_clim_layers),
        "u_s_clim": np.asarray(us_clim_layers),
        "beta_y": np.asarray(beta_y_layers),
    }
    if has_clim_next:
        out["u_s_clim_next"] = np.asarray(us_clim_next_layers)
    return out


def ddz(values: np.ndarray, depth: np.ndarray) -> np.ndarray:
    return np.gradient(values, depth, axis=0, edge_order=1)


def radial_derivative(values: np.ndarray, r_m: np.ndarray) -> np.ndarray:
    return np.gradient(values, r_m, axis=1, edge_order=1)


def azimuth_second_derivative(values: np.ndarray, theta: np.ndarray) -> np.ndarray:
    dtheta = float(theta[1] - theta[0]) if len(theta) > 1 else 2.0 * np.pi
    return (np.roll(values, -1, axis=2) - 2.0 * values + np.roll(values, 1, axis=2)) / (dtheta * dtheta)


def compute_q_and_flux_terms(fields: dict[str, np.ndarray], depth: np.ndarray, radial: np.ndarray, theta: np.ndarray, radius_m: float, n2: np.ndarray, f0: float) -> dict[str, np.ndarray]:
    psi = fields["psi_prime"]
    un = fields["u_n_prime"]
    us = fields["u_s_prime"]
    us_clim = fields["u_s_clim"]
    us_clim_next = fields.get("u_s_clim_next")
    psi = np.where(np.abs(psi) > 1e20, np.nan, psi)
    psi_az_prime = psi - np.nanmean(psi, axis=2, keepdims=True)
    r_m = np.maximum(radial * radius_m, 1.0)
    dpsi_dr = radial_derivative(psi_az_prime, r_m)
    radial_lap = radial_derivative(dpsi_dr * r_m[None, :, None], r_m) / r_m[None, :, None]
    az_lap = azimuth_second_derivative(psi_az_prime, theta) / (r_m[None, :, None] ** 2)
    dpsi_dz = ddz(psi_az_prime, depth)
    strat = (f0 * f0 / n2)[:, None, None] * dpsi_dz
    q_prime_total = radial_lap + az_lap + ddz(strat, depth)

    un_prime = un - np.nanmean(un, axis=2, keepdims=True)
    us_prime = us - np.nanmean(us, axis=2, keepdims=True)
    q_prime = q_prime_total - np.nanmean(q_prime_total, axis=2, keepdims=True)
    b_prime = f0 * ddz(psi_az_prime, depth)

    ubar_clim = np.nanmean(us_clim, axis=2)
    terms = {
        "F_n": -np.nanmean(us_prime * un_prime, axis=2),
        "F_z": (f0 / n2[:, None]) * np.nanmean(un_prime * b_prime, axis=2),
        "pv_flux": np.nanmean(un_prime * q_prime, axis=2),
        "Ubar_prime_axisym": np.nanmean(us, axis=2),
        "Ubar_clim": ubar_clim,
        "valid": np.mean(np.isfinite(psi), axis=2),
    }
    if us_clim_next is not None:
        ubar_clim_next = np.nanmean(us_clim_next, axis=2)
        terms["Ubar_clim_next"] = ubar_clim_next
        terms["dUdt_clim"] = (ubar_clim_next - ubar_clim) / SECONDS_PER_DAY
    return terms


def add_to_accum(accum: dict, polarity: str, terms: dict[str, np.ndarray]) -> None:
    if polarity not in accum:
        accum[polarity] = {key: np.zeros_like(value, dtype="f8") for key, value in terms.items() if key != "valid"}
        accum[polarity]["count"] = np.zeros_like(terms["valid"], dtype="f8")
    valid = np.isfinite(terms["valid"]) & (terms["valid"] > 0)
    for key, value in terms.items():
        if key == "valid":
            continue
        accum[polarity][key] += np.nan_to_num(value, nan=0.0) * valid
    accum[polarity]["count"] += valid.astype("f8")


def finalize_accum(accum: dict) -> dict[str, dict[str, np.ndarray]]:
    out: dict[str, dict[str, np.ndarray]] = {}
    for polarity, item in accum.items():
        count = item["count"]
        out[polarity] = {"count": count}
        for key, value in item.items():
            if key == "count":
                continue
            out[polarity][key] = np.divide(value, count, out=np.full_like(value, np.nan), where=count > 0)
    return out


def divergence(Fn: np.ndarray, Fz: np.ndarray, radial: np.ndarray, depth: np.ndarray, radius_m: float) -> np.ndarray:
    r_m = np.maximum(radial * radius_m, 1.0)
    div_r = radial_derivative(Fn * r_m[None, :], r_m) / r_m[None, :]
    div_z = ddz(Fz, depth)
    return div_r + div_z


def profile_rows(final: dict, radial: np.ndarray, depth: np.ndarray, radii: dict[str, float]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    metrics = []
    for polarity, item in final.items():
        divf = divergence(item["F_n"], item["F_z"], radial, depth, radii[polarity])
        residual = divf - item["pv_flux"]
        good = np.isfinite(divf) & np.isfinite(item["pv_flux"]) & (item["count"] > 0)
        corr = float(np.corrcoef(divf[good].ravel(), item["pv_flux"][good].ravel())[0, 1]) if np.sum(good) > 2 else np.nan
        rmse = float(np.sqrt(np.nanmean(residual[good] ** 2))) if np.any(good) else np.nan
        denom = float(np.sqrt(np.nanmean(item["pv_flux"][good] ** 2))) if np.any(good) else np.nan
        same_sign = float(np.mean(np.sign(divf[good]) == np.sign(item["pv_flux"][good]))) if np.any(good) else np.nan
        metrics.append({"polarity": polarity, "closure_corr": corr, "closure_rmse": rmse, "closure_relative_rmse": rmse / denom if denom and np.isfinite(denom) else np.nan, "closure_same_sign_fraction": same_sign, "valid_bins": int(np.sum(good))})
        for k, z in enumerate(depth):
            for j, r in enumerate(radial):
                rows.append(
                    {
                        "polarity": polarity,
                        "depth_index": k,
                        "depth_m": float(z),
                        "r_over_R": float(r),
                        "F_n": float(item["F_n"][k, j]),
                        "F_z": float(item["F_z"][k, j]),
                        "divF": float(divf[k, j]),
                        "pv_flux": float(item["pv_flux"][k, j]),
                        "closure_residual": float(residual[k, j]),
                        "Ubar_prime_axisym": float(item["Ubar_prime_axisym"][k, j]),
                        "Ubar_clim": float(item["Ubar_clim"][k, j]),
                        "Ubar_clim_next": float(item.get("Ubar_clim_next", np.full_like(item["Ubar_clim"], np.nan))[k, j]),
                        "dUdt_clim": float(item.get("dUdt_clim", np.full_like(item["Ubar_clim"], np.nan))[k, j]),
                        "count": float(item["count"][k, j]),
                    }
                )
    return pd.DataFrame.from_records(rows), pd.DataFrame.from_records(metrics)


def feedback_metrics(profiles: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for polarity, part in profiles.groupby("polarity"):
        divf = part["divF"].to_numpy(dtype="f8")
        dudt = part["dUdt_clim"].to_numpy(dtype="f8") if "dUdt_clim" in part else np.full_like(divf, np.nan)
        good = np.isfinite(divf) & np.isfinite(dudt)
        corr = float(np.corrcoef(divf[good], dudt[good])[0, 1]) if np.sum(good) > 2 else np.nan
        slope = float(np.polyfit(divf[good], dudt[good], deg=1)[0]) if np.sum(good) > 2 else np.nan
        same_sign = float(np.mean(np.sign(divf[good]) == np.sign(dudt[good]))) if np.any(good) else np.nan
        rows.append(
            {
                "polarity": polarity,
                "feedback_corr_divF_dUdt_clim": corr,
                "feedback_slope_dUdt_per_divF": slope,
                "feedback_same_sign_fraction": same_sign,
                "valid_bins": int(np.sum(good)),
                "feedback_note": "dUdt is the next-day climatological along-axis flow tendency sampled on the same east-aligned tilted coordinates; it is a trend check, not a complete momentum budget.",
            }
        )
    return pd.DataFrame.from_records(rows)


def write_alignment(objects: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    cols = ["eddy3d_object_id", "polarity", "track3d_id", "date", "temp_direction_rad", "deep_x_rot_m", "deep_y_rot_m", "deep_rotation_abs_y_m"]
    diag = objects[cols].copy()
    diag["deep_points_east"] = diag["deep_x_rot_m"] > 0
    diag.to_csv(output_dir / "east_alignment_diagnostics.csv", index=False)
    return diag


def plot_outputs(profiles: pd.DataFrame, closure: pd.DataFrame, feedback: pd.DataFrame, alignment: pd.DataFrame, output_dir: Path) -> None:
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    for polarity, part in profiles.groupby("polarity"):
        pivot = lambda name: part.pivot(index="depth_m", columns="r_over_R", values=name).sort_index()
        divf = pivot("divF")
        pv = pivot("pv_flux")
        res = pivot("closure_residual")
        vmax = np.nanpercentile(np.abs(pd.concat([divf.stack(), pv.stack()])), 98)
        fig, axes = plt.subplots(1, 3, figsize=(14, 5), dpi=160)
        for ax, data, title in zip(axes, (divf, pv, res), ("div F_T", "mean u_n' q_T'", "residual")):
            mesh = ax.pcolormesh(data.columns.to_numpy(dtype="f8"), data.index.to_numpy(dtype="f8"), data.to_numpy(), shading="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
            ax.invert_yaxis()
            ax.set_xlabel("r/R")
            ax.set_title(title)
        axes[0].set_ylabel("depth m")
        fig.colorbar(mesh, ax=axes, label="diagnostic units")
        fig.suptitle(polarity)
        fig.tight_layout()
        fig.savefig(fig_dir / f"{polarity}_divF_pvflux_residual.png")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6, 5), dpi=160)
        good = np.isfinite(part["divF"]) & np.isfinite(part["pv_flux"])
        ax.scatter(part.loc[good, "pv_flux"], part.loc[good, "divF"], s=8, alpha=0.5)
        ax.set_xlabel("mean u_n' q_T'")
        ax.set_ylabel("div F_T")
        ax.set_title(f"{polarity}: closure scatter")
        ax.grid(True, color="0.9")
        fig.tight_layout()
        fig.savefig(fig_dir / f"{polarity}_divF_vs_pv_flux.png")
        plt.close(fig)

        if "dUdt_clim" in part:
            fig, ax = plt.subplots(figsize=(6, 5), dpi=160)
            good = np.isfinite(part["divF"]) & np.isfinite(part["dUdt_clim"])
            ax.scatter(part.loc[good, "dUdt_clim"], part.loc[good, "divF"], s=8, alpha=0.5)
            ax.set_xlabel("d U_clim / dt")
            ax.set_ylabel("div F_T")
            ax.set_title(f"{polarity}: feedback trend scatter")
            ax.grid(True, color="0.9")
            fig.tight_layout()
            fig.savefig(fig_dir / f"{polarity}_divF_vs_dUdt_clim.png")
            plt.close(fig)

            ubar = pivot("Ubar_clim")
            dudt = pivot("dUdt_clim")
            fig, axes = plt.subplots(1, 2, figsize=(10, 5), dpi=160)
            mesh0 = axes[0].pcolormesh(ubar.columns.to_numpy(dtype="f8"), ubar.index.to_numpy(dtype="f8"), ubar.to_numpy(), shading="auto", cmap="RdBu_r")
            axes[0].invert_yaxis()
            axes[0].set_xlabel("r/R")
            axes[0].set_ylabel("depth m")
            axes[0].set_title("U_clim")
            fig.colorbar(mesh0, ax=axes[0])
            vmax_dudt = np.nanpercentile(np.abs(dudt.to_numpy()), 98)
            mesh1 = axes[1].pcolormesh(dudt.columns.to_numpy(dtype="f8"), dudt.index.to_numpy(dtype="f8"), dudt.to_numpy(), shading="auto", cmap="RdBu_r", vmin=-vmax_dudt, vmax=vmax_dudt)
            axes[1].invert_yaxis()
            axes[1].set_xlabel("r/R")
            axes[1].set_title("dU_clim/dt")
            fig.colorbar(mesh1, ax=axes[1])
            fig.suptitle(f"{polarity}: climatological mean flow")
            fig.tight_layout()
            fig.savefig(fig_dir / f"{polarity}_Ubar_dUdt_clim.png")
            plt.close(fig)

        fn = pivot("F_n")
        fz = pivot("F_z")
        rr, zz = np.meshgrid(fn.columns.to_numpy(dtype="f8"), fn.index.to_numpy(dtype="f8"))
        fig, ax = plt.subplots(figsize=(8, 6), dpi=160)
        ax.quiver(rr[::3, ::3], zz[::3, ::3], fn.to_numpy()[::3, ::3], -fz.to_numpy()[::3, ::3], scale=None)
        ax.invert_yaxis()
        ax.set_xlabel("r/R")
        ax.set_ylabel("depth m")
        ax.set_title(f"{polarity}: E-P flux section")
        fig.tight_layout()
        fig.savefig(fig_dir / f"{polarity}_ep_flux_vectors.png")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5), dpi=160)
    ax.scatter(alignment["deep_x_rot_m"], alignment["deep_y_rot_m"], s=4, alpha=0.25)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xlabel("deep x_rot m")
    ax.set_ylabel("deep y_rot m")
    ax.set_title("East-alignment check")
    fig.tight_layout()
    fig.savefig(fig_dir / "east_alignment_check.png")
    plt.close(fig)


def write_summary(output_dir: Path, objects: pd.DataFrame, closure: pd.DataFrame, feedback: pd.DataFrame, alignment: pd.DataFrame, args: argparse.Namespace) -> None:
    lines = [
        "# Tilted E-P flux validation summary",
        "",
        f"- Objects selected: {len(objects):,}",
        f"- Polarities: `{args.polarities}`",
        f"- Perturbation field: `{args.input_daily_dir}` u/v",
        f"- Mean field: `{args.climatology_path}` u_clim/v_clim",
        f"- East alignment median |deep_y_rot|: {alignment['deep_y_rot_m'].abs().median():.3e} m",
        f"- East alignment positive deep_x fraction: {(alignment['deep_x_rot_m'] > 0).mean():.3f}",
        "- Perturbation q_T excludes planetary beta; beta belongs to the climatological mean PV.",
        "",
        "## Closure Metrics",
        "```csv",
        closure.to_csv(index=False).strip(),
        "```",
        "",
        "## Feedback Metrics",
        "```csv",
        feedback.to_csv(index=False).strip(),
        "```",
        "",
        "Note: the feedback table compares E-P flux divergence with the next-day climatological along-axis flow tendency sampled on the same east-aligned tilted coordinates. It is a trend validation, not a full momentum budget because dissipation, external forcing, and transformed residual circulation are not closed here.",
    ]
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    polarities = parse_csv_list(args.polarities, DEFAULT_POLARITIES)
    objects = load_objects(Path(args.axis_dir), Path(args.catalog_dir), polarities)
    if int(args.max_days) > 0:
        keep_mask = pd.Series(False, index=objects.index)
        for _, part in objects.groupby("polarity"):
            keep_dates = sorted(part["date"].unique())[: int(args.max_days)]
            keep_mask |= objects["polarity"].eq(part["polarity"].iloc[0]) & objects["date"].isin(keep_dates)
        objects = objects[keep_mask].copy()
    objects = limit_objects_per_polarity(objects, int(args.max_objects_per_polarity), int(args.random_seed))
    center_lines = load_center_lines(Path(args.axis_dir), set(objects["eddy3d_object_id"].astype(int)))
    alignment = write_alignment(objects, output_dir)
    radial, theta, rr, tt, _ = make_polar_grid(float(args.rmax), int(args.radial_bins), int(args.azimuth_bins))
    lat_ref = float(objects["surface_lat"].median()) if not objects.empty else 27.5
    f0 = 2.0 * OMEGA * np.sin(np.radians(lat_ref))
    radius_by_polarity = objects.groupby("polarity")["mean_radius_m"].median().to_dict()

    accum: dict = {}
    ubar_by_object: dict[int, np.ndarray] = {}
    for date, day_objects in tqdm(list(objects.groupby("date")), desc="Tilted EP flux", unit="day"):
        path = Path(args.input_daily_dir) / f"uv_{pd.Timestamp(date):%Y%m%d}.nc"
        if not path.exists():
            continue
        lon, lat, depth, u, v = read_daily_uv(path)
        u = sanitize_ocean_field(u)
        v = sanitize_ocean_field(v)
        u_clim, v_clim = read_climatology_uv(Path(args.climatology_path), str(date))
        next_date = (pd.Timestamp(date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        u_clim_next, v_clim_next = read_climatology_uv(Path(args.climatology_path), next_date)
        n2 = load_n2(Path(args.n2_profile_path), depth)
        _, dy, dx = grid_spacing_m(lon, lat)
        zeta = relative_vorticity(lon, lat, u, v)
        psi_prime = streamfunction_from_zeta(zeta, dx, dy)
        for obj in day_objects.itertuples(index=False):
            center_line = center_lines.get(int(obj.eddy3d_object_id))
            if center_line is None:
                continue
            fields = sample_object_fields(obj, center_line, lon, lat, depth, psi_prime, u, v, u_clim, v_clim, u_clim_next, v_clim_next, rr, tt)
            if fields is None:
                continue
            terms = compute_q_and_flux_terms(fields, depth, radial, theta, float(obj.mean_radius_m), n2, f0)
            add_to_accum(accum, str(obj.polarity), terms)
            ubar_by_object[int(obj.eddy3d_object_id)] = terms["Ubar_clim"]

    final = finalize_accum(accum)
    profiles, closure = profile_rows(final, radial, depth, radius_by_polarity)
    feedback = feedback_metrics(profiles)
    profiles.to_parquet(output_dir / "ep_flux_profiles.parquet", index=False)
    closure.to_csv(output_dir / "ep_flux_closure_metrics.csv", index=False)
    feedback.to_csv(output_dir / "ep_flux_feedback_metrics.csv", index=False)
    counts = profiles[["polarity", "depth_index", "depth_m", "r_over_R", "count"]].copy()
    counts.to_csv(output_dir / "ep_flux_object_counts.csv", index=False)
    plot_outputs(profiles, closure, feedback, alignment, output_dir)
    write_summary(output_dir, objects, closure, feedback, alignment, args)
    print(f"Output: {output_dir}")
    print(f"Profiles: {output_dir / 'ep_flux_profiles.parquet'}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate tilted-vortex E-P flux diagnostics in east-aligned moving-axis coordinates.")
    parser.add_argument("--axis-dir", default=str(DEFAULT_AXIS_DIR))
    parser.add_argument("--catalog-dir", default=str(DEFAULT_CATALOG))
    parser.add_argument("--input-daily-dir", default=str(DEFAULT_INPUT_DAILY))
    parser.add_argument("--climatology-path", default=str(DEFAULT_CLIMATOLOGY_NC))
    parser.add_argument("--n2-profile-path", default=str(DEFAULT_CLIMATOLOGY))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--polarities", default=",".join(DEFAULT_POLARITIES))
    parser.add_argument("--rmax", type=float, default=2.5)
    parser.add_argument("--radial-bins", type=int, default=40)
    parser.add_argument("--azimuth-bins", type=int, default=72)
    parser.add_argument("--max-objects-per-polarity", type=int, default=0)
    parser.add_argument("--max-days", type=int, default=0)
    parser.add_argument("--random-seed", type=int, default=20260710)
    parser.add_argument("--include-beta", action="store_true", default=False, help="Reserved for future mean-PV diagnostics; perturbation q_T excludes beta by default.")
    parser.add_argument("--no-beta", dest="include_beta", action="store_false")
    parser.add_argument("--east-align-from-temp-direction", action="store_true", default=True)
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
