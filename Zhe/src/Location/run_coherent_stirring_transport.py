from __future__ import annotations

import argparse
import json
import math
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date
from functools import lru_cache
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import netCDF4
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.First_temp.axis_streamfunction_separation import grid_spacing_m, relative_vorticity, streamfunction_from_zeta
from src.First_temp.lifecycle_ep_flux_nondim_validation import azimuth_second_derivative, ddz, load_n2, radial_derivative
from src.First_temp.tilted_ep_flux_validation import bilinear_sample, sanitize_ocean_field, xy_to_lonlat


EARTH_OMEGA = 7.2921159e-5
RHO0 = 1025.0
CP = 3992.0
DEFAULT_RV_ROOT = Path("/root/autodl-fs/kuroshiou/result_strict_contiguous/result_coherent_only/representative_vortex")
DEFAULT_FILTER_ROOT = Path("/root/autodl-fs/kuroshiou/Filter")
DEFAULT_OUTPUT = Path("/root/autodl-fs/kuroshiou/result_strict_contiguous/result_coherent_only/stirring_transport")
METRICS = (
    "heat_stir_rot",
    "heat_stir_rot_rel",
    "pv_stir_rot",
    "pv_stir_rot_rel",
    "v_rot_mean",
    "theta_mean",
    "q_prime_mean",
)
POLARITIES = ("anticyclonic", "cyclonic")


@dataclass(frozen=True)
class ChunkTask:
    chunk_id: int
    objects: pd.DataFrame
    points: pd.DataFrame
    output_path: Path


def _read_tau_grid(rv_root: Path) -> np.ndarray:
    table = pd.read_csv(rv_root / "continuous_tau_grid.csv")
    col = "tau" if "tau" in table.columns else "tau_center"
    return table[col].to_numpy(dtype="float64")


def _read_profile_grid(rv_root: Path) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    profiles = pd.read_parquet(rv_root / "streamfunction_templates" / "continuous_radial_psi_profiles.parquet")
    depth = profiles[["depth_index", "depth_m"]].drop_duplicates().sort_values("depth_index")
    radial = np.sort(profiles["r_over_R"].dropna().unique().astype("float64"))
    return depth["depth_m"].to_numpy(dtype="float64"), radial, profiles


def _load_objects(rv_root: Path, start: str | None, end: str | None, max_days: int, max_objects: int) -> pd.DataFrame:
    objects = pd.read_parquet(rv_root / "object_cache" / "selected_lifecycle_objects.parquet")
    diag_cols = [
        "eddy3d_object_id",
        "shape_class",
        "polarity",
        "axis_alignment_method",
        "global_alpha_ok",
        "global_deviate_angle_rad",
        "global_theta0_rad",
    ]
    diag = pd.read_parquet(rv_root / "axis" / "object_diagnostics.parquet", columns=diag_cols)
    objects["eddy3d_object_id"] = objects["eddy3d_object_id"].astype("int64")
    diag["eddy3d_object_id"] = diag["eddy3d_object_id"].astype("int64")
    keep_cols = ["eddy3d_object_id"]
    for col in ("axis_alignment_method", "global_alpha_ok", "global_deviate_angle_rad", "global_theta0_rad"):
        if col not in objects.columns:
            keep_cols.append(col)
    if len(keep_cols) > 1:
        objects = objects.merge(diag[keep_cols], on="eddy3d_object_id", how="left")
    for col in ("shape_class", "polarity"):
        if col not in objects.columns:
            objects = objects.merge(diag[["eddy3d_object_id", col]], on="eddy3d_object_id", how="left")
    objects["date"] = pd.to_datetime(objects["date"]).dt.strftime("%Y-%m-%d")
    objects = objects[objects["shape_class"].astype(str).eq("coherent")].copy()
    objects = objects[objects["axis_alignment_method"].astype(str).eq("global_ls_alpha")].copy()
    objects = objects[objects["global_alpha_ok"].astype(bool)].copy()
    if start:
        objects = objects[objects["date"] >= pd.Timestamp(start).strftime("%Y-%m-%d")]
    if end:
        objects = objects[objects["date"] <= pd.Timestamp(end).strftime("%Y-%m-%d")]
    if max_days > 0 and not objects.empty:
        days = sorted(objects["date"].unique())[: int(max_days)]
        objects = objects[objects["date"].isin(days)].copy()
    if max_objects > 0 and len(objects) > max_objects:
        objects = objects.sort_values(["date", "polarity", "eddy3d_object_id"]).head(int(max_objects)).copy()
    objects["track3d_id"] = objects["track3d_id"].astype("int64")
    objects["eddy3d_object_id"] = objects["eddy3d_object_id"].astype("int64")
    return objects.sort_values(["date", "polarity", "eddy3d_object_id"]).reset_index(drop=True)


def _load_points(rv_root: Path, object_ids: set[int]) -> pd.DataFrame:
    cols = [
        "eddy3d_object_id",
        "track3d_id",
        "date",
        "depth_index",
        "depth_m",
        "longitude",
        "latitude",
        "x_rot_m",
        "y_rot_m",
    ]
    points = pd.read_parquet(rv_root / "axis" / "rotated_points.parquet", columns=cols)
    points["eddy3d_object_id"] = points["eddy3d_object_id"].astype("int64")
    points = points[points["eddy3d_object_id"].isin(object_ids)].copy()
    points["date"] = pd.to_datetime(points["date"]).dt.strftime("%Y-%m-%d")
    points["depth_index"] = points["depth_index"].astype("int16")
    return points.sort_values(["eddy3d_object_id", "depth_index"]).reset_index(drop=True)


def _make_polar_grid(radial: np.ndarray, azimuth_bins: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    theta = np.linspace(0.0, 2.0 * np.pi, int(azimuth_bins), endpoint=False)
    rr, tt = np.meshgrid(radial, theta, indexing="ij")
    return theta, rr, tt


@lru_cache(maxsize=64)
def _time_index(path_text: str) -> dict[date, int]:
    with netCDF4.Dataset(path_text) as ds:
        tvar = ds.variables["time"]
        values = netCDF4.num2date(
            tvar[:],
            tvar.units,
            getattr(tvar, "calendar", "standard"),
            only_use_cftime_datetimes=False,
            only_use_python_datetimes=True,
        )
    return {value.date(): int(i) for i, value in enumerate(values)}


def _clean(values) -> np.ndarray:
    arr = np.ma.filled(values, np.nan).astype("float64", copy=False)
    arr[np.abs(arr) > 1e20] = np.nan
    return arr


def _read_filter_day(filter_root: Path, template: str, day: date) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    path = filter_root / template.format(year=day.year)
    if not path.exists():
        raise FileNotFoundError(path)
    index = _time_index(str(path)).get(day)
    if index is None:
        raise KeyError(f"{day} not found in {path}")
    with netCDF4.Dataset(path) as ds:
        for variable in ("uo_glor", "vo_glor", "thetao_glor"):
            if variable not in ds.variables:
                raise KeyError(f"{variable} not found in {path}")
        lon = np.asarray(ds.variables["longitude"][:], dtype="float64")
        lat = np.asarray(ds.variables["latitude"][:], dtype="float64")
        depth = np.asarray(ds.variables["depth"][:], dtype="float64")
        u = _clean(ds.variables["uo_glor"][index])
        v = _clean(ds.variables["vo_glor"][index])
        theta = _clean(ds.variables["thetao_glor"][index])
    return lon, lat, depth, u, v, theta


def _center_lines(points: pd.DataFrame) -> dict[int, pd.DataFrame]:
    return {int(object_id): part.sort_values("depth_index").copy() for object_id, part in points.groupby("eddy3d_object_id", sort=False)}


def _sample_rotated_fields(
    obj,
    center_line: pd.DataFrame,
    lon: np.ndarray,
    lat: np.ndarray,
    psi: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    theta_field: np.ndarray,
    radial_mesh: np.ndarray,
    theta_mesh: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    radius = float(obj.mean_radius_m)
    alpha = float(obj.global_deviate_angle_rad)
    cos_a = math.cos(alpha)
    sin_a = math.sin(alpha)
    local_x = radial_mesh * radius * np.cos(theta_mesh)
    local_y = radial_mesh * radius * np.sin(theta_mesh)
    depth_indices = center_line["depth_index"].to_numpy(dtype="int64")
    psi_layers: list[np.ndarray] = []
    vrot_layers: list[np.ndarray] = []
    theta_layers: list[np.ndarray] = []
    for row in center_line.itertuples(index=False):
        k = int(row.depth_index)
        x_rot = float(row.x_rot_m) + local_x
        y_rot = float(row.y_rot_m) + local_y
        x_orig = x_rot * cos_a + y_rot * sin_a
        y_orig = -x_rot * sin_a + y_rot * cos_a
        target_lon, target_lat = xy_to_lonlat(x_orig, y_orig, float(obj.surface_lon), float(obj.surface_lat))
        u_s = bilinear_sample(lon, lat, u[k], target_lon, target_lat)
        v_s = bilinear_sample(lon, lat, v[k], target_lon, target_lat)
        psi_layers.append(bilinear_sample(lon, lat, psi[k], target_lon, target_lat))
        theta_layers.append(bilinear_sample(lon, lat, theta_field[k], target_lon, target_lat))
        vrot_layers.append(u_s * sin_a + v_s * cos_a)
    return (
        depth_indices,
        center_line["depth_m"].to_numpy(dtype="float64"),
        np.asarray(psi_layers, dtype="float64"),
        np.asarray(vrot_layers, dtype="float64"),
        np.asarray(theta_layers, dtype="float64"),
    )


def _q_prime_from_psi(psi_sample: np.ndarray, depth: np.ndarray, radial: np.ndarray, theta: np.ndarray, radius_m: float, n2: np.ndarray, f0: float) -> np.ndarray:
    psi = np.where(np.abs(psi_sample) > 1e20, np.nan, psi_sample)
    psi_prime = psi - np.nanmean(psi, axis=2, keepdims=True)
    r_m = np.maximum(radial * float(radius_m), 1.0)
    dpsi_dr = radial_derivative(psi_prime, r_m)
    radial_lap = radial_derivative(dpsi_dr * r_m[None, :, None], r_m) / r_m[None, :, None]
    az_lap = azimuth_second_derivative(psi_prime, theta) / (r_m[None, :, None] ** 2)
    dpsi_dz = ddz(psi_prime, depth)
    strat = (f0 * f0 / n2)[:, None, None] * dpsi_dz
    q_total = radial_lap + az_lap + ddz(strat, depth)
    return q_total - np.nanmean(q_total, axis=2, keepdims=True)


def _empty_accum(ntau: int, ndepth: int, nr: int) -> dict[str, dict[str, np.ndarray]]:
    return {
        polarity: {
            **{f"sum_{name}": np.zeros((ntau, ndepth, nr), dtype="float64") for name in METRICS},
            **{f"count_{name}": np.zeros((ntau, ndepth, nr), dtype="float64") for name in METRICS},
        }
        for polarity in POLARITIES
    }


def _add_metric(acc: dict[str, np.ndarray], name: str, tau_index: int, depth_indices: np.ndarray, values: np.ndarray, weight: float) -> None:
    sums = acc[f"sum_{name}"]
    counts = acc[f"count_{name}"]
    for local_k, depth_index in enumerate(depth_indices):
        vals = values[local_k]
        ok = np.isfinite(vals)
        if not np.any(ok):
            continue
        sums[tau_index, int(depth_index), ok] += vals[ok] * weight
        counts[tau_index, int(depth_index), ok] += weight


def _process_object(
    obj,
    center_line: pd.DataFrame,
    lon: np.ndarray,
    lat: np.ndarray,
    depth_all: np.ndarray,
    psi: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    theta_field: np.ndarray,
    radial: np.ndarray,
    theta_angles: np.ndarray,
    rr: np.ndarray,
    tt: np.ndarray,
    n2_full: np.ndarray,
) -> dict[str, np.ndarray] | None:
    if center_line is None or len(center_line) < 3:
        return None
    depth_indices, depth, psi_s, vrot, theta_s = _sample_rotated_fields(
        obj,
        center_line,
        lon,
        lat,
        psi,
        u,
        v,
        theta_field,
        rr,
        tt,
    )
    n2 = n2_full[depth_indices]
    f0 = 2.0 * EARTH_OMEGA * math.sin(math.radians(float(obj.surface_lat)))
    q_prime = _q_prime_from_psi(psi_s, depth, radial, theta_angles, float(obj.mean_radius_m), n2, f0)
    v_mean = np.nanmean(vrot, axis=2)
    theta_mean = np.nanmean(theta_s, axis=2)
    q_mean = np.nanmean(q_prime, axis=2)
    v_rel = vrot - v_mean[:, :, None]
    theta_rel = theta_s - theta_mean[:, :, None]
    q_rel = q_prime - q_mean[:, :, None]
    return {
        "depth_indices": depth_indices,
        "heat_stir_rot": RHO0 * CP * np.nanmean(vrot * theta_s, axis=2),
        "heat_stir_rot_rel": RHO0 * CP * np.nanmean(v_rel * theta_rel, axis=2),
        "pv_stir_rot": np.nanmean(vrot * q_prime, axis=2),
        "pv_stir_rot_rel": np.nanmean(v_rel * q_rel, axis=2),
        "v_rot_mean": v_mean,
        "theta_mean": theta_mean,
        "q_prime_mean": q_mean,
    }


def _worker(task: ChunkTask, args_dict: dict, tau_grid: np.ndarray, depth_template: np.ndarray, radial: np.ndarray) -> dict[str, object]:
    out_path = Path(task.output_path)
    if out_path.exists() and not args_dict.get("force", False):
        return {"chunk_id": task.chunk_id, "path": str(out_path), "cached": True}
    theta_angles, rr, tt = _make_polar_grid(radial, int(args_dict["azimuth_bins"]))
    n2_full = load_n2(Path(args_dict["n2_profile"]), depth_template)
    accum = _empty_accum(len(tau_grid), len(depth_template), len(radial))
    center_lines = _center_lines(task.points)
    processed = 0
    skipped = 0
    for date_text, day_objects in task.objects.groupby("date", sort=True):
        day = pd.Timestamp(date_text).date()
        lon, lat, depth_all, u_all, v_all, theta_all = _read_filter_day(Path(args_dict["filter_root"]), args_dict["filter_template"], day)
        if len(depth_all) != len(depth_template):
            raise ValueError(f"Depth mismatch on {date_text}: {len(depth_all)} != {len(depth_template)}")
        u = sanitize_ocean_field(u_all)
        v = sanitize_ocean_field(v_all)
        theta_field = sanitize_ocean_field(theta_all)
        _, dy, dx = grid_spacing_m(lon, lat)
        psi = streamfunction_from_zeta(relative_vorticity(lon, lat, u, v), dx, dy)
        for obj in day_objects.itertuples(index=False):
            terms = _process_object(
                obj,
                center_lines.get(int(obj.eddy3d_object_id)),
                lon,
                lat,
                depth_all,
                psi,
                u,
                v,
                theta_field,
                radial,
                theta_angles,
                rr,
                tt,
                n2_full,
            )
            if terms is None:
                skipped += 1
                continue
            weights = np.exp(-0.5 * ((tau_grid - float(obj.life_phase)) / max(float(args_dict["kernel_bandwidth"]), 1e-12)) ** 2)
            valid_tau = np.where(weights >= float(args_dict["kernel_weight_min"]))[0]
            pol = str(obj.polarity)
            if pol not in accum:
                skipped += 1
                continue
            depth_indices = terms["depth_indices"]
            for tau_index in valid_tau:
                weight = float(weights[tau_index])
                for name in METRICS:
                    _add_metric(accum[pol], name, int(tau_index), depth_indices, terms[name], weight)
            processed += 1
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    with tmp_path.open("wb") as handle:
        pickle.dump(accum, handle, protocol=pickle.HIGHEST_PROTOCOL)
    tmp_path.replace(out_path)
    return {"chunk_id": task.chunk_id, "path": str(out_path), "processed": processed, "skipped": skipped}


def _make_tasks(objects: pd.DataFrame, points: pd.DataFrame, partial_dir: Path, chunk_days: int, force: bool) -> list[ChunkTask]:
    dates = sorted(objects["date"].unique())
    tasks: list[ChunkTask] = []
    for chunk_id, start in enumerate(range(0, len(dates), int(chunk_days))):
        chunk_dates = dates[start : start + int(chunk_days)]
        chunk_objects = objects[objects["date"].isin(chunk_dates)].copy()
        object_ids = set(chunk_objects["eddy3d_object_id"].astype("int64"))
        chunk_points = points[points["eddy3d_object_id"].isin(object_ids)].copy()
        path = partial_dir / f"chunk_{chunk_id:04d}_{chunk_dates[0].replace('-', '')}_{chunk_dates[-1].replace('-', '')}.pkl"
        if path.exists() and not force:
            continue
        tasks.append(ChunkTask(chunk_id=chunk_id, objects=chunk_objects, points=chunk_points, output_path=path))
    return tasks


def _merge_partials(paths: list[Path], ntau: int, ndepth: int, nr: int) -> dict[str, dict[str, np.ndarray]]:
    merged = _empty_accum(ntau, ndepth, nr)
    for path in sorted(paths):
        with path.open("rb") as handle:
            part = pickle.load(handle)
        for pol in POLARITIES:
            for key, value in part.get(pol, {}).items():
                merged[pol][key] += value
    return merged


def _finalize(
    accum: dict[str, dict[str, np.ndarray]],
    tau_grid: np.ndarray,
    depth: np.ndarray,
    radial: np.ndarray,
    rv_root: Path,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    psi_counts = pd.read_parquet(rv_root / "streamfunction_templates" / "continuous_radial_psi_profiles.parquet")
    count_lookup = psi_counts.set_index(["polarity", "tau_center", "depth_index", "r_over_R"])[["count", "n_objects", "n_tracks", "n_dates"]]
    for pol in POLARITIES:
        item = accum[pol]
        means = {}
        for name in METRICS:
            sums = item[f"sum_{name}"]
            counts = item[f"count_{name}"]
            means[name] = np.divide(sums, counts, out=np.full_like(sums, np.nan), where=counts > 0)
        heat_mean_product = RHO0 * CP * means["v_rot_mean"] * means["theta_mean"]
        pv_mean_product = means["v_rot_mean"] * means["q_prime_mean"]
        for ti, tau in enumerate(tau_grid):
            for k, depth_m in enumerate(depth):
                for j, r_value in enumerate(radial):
                    row = {
                        "polarity": pol,
                        "tau_center": float(tau),
                        "depth_index": int(k),
                        "depth_m": float(depth_m),
                        "r_over_R": float(r_value),
                        "heat_stir_rot": float(means["heat_stir_rot"][ti, k, j]),
                        "heat_stir_rot_rel": float(means["heat_stir_rot_rel"][ti, k, j]),
                        "pv_stir_rot": float(means["pv_stir_rot"][ti, k, j]),
                        "pv_stir_rot_rel": float(means["pv_stir_rot_rel"][ti, k, j]),
                        "heat_mean_product": float(heat_mean_product[ti, k, j]),
                        "pv_mean_product": float(pv_mean_product[ti, k, j]),
                        "v_rot_mean": float(means["v_rot_mean"][ti, k, j]),
                        "theta_mean": float(means["theta_mean"][ti, k, j]),
                        "q_prime_mean": float(means["q_prime_mean"][ti, k, j]),
                        "metric_count": float(item["count_heat_stir_rot"][ti, k, j]),
                    }
                    try:
                        ref = count_lookup.loc[(pol, float(tau), int(k), float(r_value))]
                        row.update(
                            {
                                "count": float(ref["count"]),
                                "n_objects": int(ref["n_objects"]),
                                "n_tracks": int(ref["n_tracks"]),
                                "n_dates": int(ref["n_dates"]),
                            }
                        )
                    except KeyError:
                        row.update({"count": np.nan, "n_objects": 0, "n_tracks": 0, "n_dates": 0})
                    rows.append(row)
    return pd.DataFrame.from_records(rows)


def _core_summary(profiles: pd.DataFrame, rmax: float) -> pd.DataFrame:
    part = profiles[profiles["r_over_R"] <= float(rmax)].copy()
    records = []
    for keys, group in part.groupby(["polarity", "tau_center", "depth_index", "depth_m"], sort=True):
        weight = np.maximum(group["r_over_R"].to_numpy(dtype="float64"), 1e-6)
        row = dict(zip(["polarity", "tau_center", "depth_index", "depth_m"], keys))
        for col in ("heat_stir_rot", "heat_stir_rot_rel", "pv_stir_rot", "pv_stir_rot_rel", "heat_mean_product", "pv_mean_product"):
            vals = group[col].to_numpy(dtype="float64")
            ok = np.isfinite(vals)
            row[col] = float(np.nansum(vals[ok] * weight[ok]) / np.nansum(weight[ok])) if np.any(ok) else np.nan
        row["n_objects"] = int(np.nanmax(group["n_objects"].to_numpy(dtype="float64"))) if len(group) else 0
        row["n_tracks"] = int(np.nanmax(group["n_tracks"].to_numpy(dtype="float64"))) if len(group) else 0
        records.append(row)
    return pd.DataFrame.from_records(records)


def _plot_core(core: pd.DataFrame, out_dir: Path, suffix: str) -> None:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    metrics = (
        "heat_stir_rot",
        "heat_stir_rot_rel",
        "pv_stir_rot",
        "pv_stir_rot_rel",
        "heat_mean_product",
        "pv_mean_product",
    )
    for pol, part in core.groupby("polarity", sort=True):
        for metric in metrics:
            pivot = part.pivot_table(index="depth_m", columns="tau_center", values=metric)
            if pivot.empty:
                continue
            data = pivot.to_numpy(dtype="float64")
            vmax = np.nanpercentile(np.abs(data), 98)
            if not np.isfinite(vmax) or vmax <= 0:
                vmax = 1.0
            fig, ax = plt.subplots(figsize=(7.4, 5.2), dpi=180)
            im = ax.imshow(
                data,
                origin="lower",
                aspect="auto",
                cmap="coolwarm",
                vmin=-vmax,
                vmax=vmax,
                extent=[float(pivot.columns.min()), float(pivot.columns.max()), float(pivot.index.min()), float(pivot.index.max())],
            )
            ax.set_xlabel("tau")
            ax.set_ylabel("depth (m)")
            ax.set_title(f"{pol} {metric} ({suffix}, y_rot)")
            fig.colorbar(im, ax=ax)
            fig.tight_layout()
            fig.savefig(fig_dir / f"{pol}_{metric}_{suffix}_tau_depth.png")
            plt.close(fig)


def _write_summary(output_dir: Path, profiles: pd.DataFrame, core15: pd.DataFrame, objects: pd.DataFrame, args: argparse.Namespace) -> None:
    def safe_median(values: np.ndarray) -> float:
        good = values[np.isfinite(values)]
        return float(np.median(good)) if good.size else np.nan

    def safe_p95_abs(values: np.ndarray) -> float:
        good = np.abs(values[np.isfinite(values)])
        return float(np.percentile(good, 95)) if good.size else np.nan

    def markdown_table(table: pd.DataFrame) -> str:
        if table.empty:
            return "_无可用统计行_"
        cols = list(table.columns)
        lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
        for record in table.to_dict(orient="records"):
            vals = []
            for col in cols:
                value = record[col]
                if isinstance(value, float):
                    vals.append("nan" if not np.isfinite(value) else f"{value:.6g}")
                else:
                    vals.append(str(value))
            lines.append("| " + " | ".join(vals) + " |")
        return "\n".join(lines)

    rows = []
    for pol, part in core15.groupby("polarity", sort=True):
        row = {"polarity": pol}
        for col in ("heat_stir_rot_rel", "pv_stir_rot_rel", "heat_mean_product", "pv_mean_product"):
            vals = part[col].to_numpy(dtype="float64")
            row[f"{col}_median"] = safe_median(vals)
            row[f"{col}_p95_abs"] = safe_p95_abs(vals)
        row["n_tracks"] = int(objects.loc[objects["polarity"].eq(pol), "track3d_id"].nunique())
        row["n_object_days"] = int(objects.loc[objects["polarity"].eq(pol), "eddy3d_object_id"].nunique())
        rows.append(row)
    summary = pd.DataFrame.from_records(rows)
    summary.to_csv(output_dir / "stirring_transport_summary_stats.csv", index=False)
    lines = [
        "# Coherent-only 代表涡旋 stirring 热通量与 PV 通量\n\n",
        "本诊断只使用 strict-contiguous coherent-only 代表涡旋，并只在 `global_ls_alpha` 旋转后的 `y_rot` 横向方向上解释输送；不输出地理北向，不计算 trapping。\n\n",
        f"- Representative root: `{args.rv_root}`\n",
        f"- Filter root: `{args.filter_root}`\n",
        f"- Object-days: `{len(objects):,}`\n",
        f"- Tracks: `{objects['track3d_id'].nunique():,}`\n",
        f"- Tau nodes: `{profiles['tau_center'].nunique()}`\n",
        f"- Depth levels: `{profiles['depth_index'].nunique()}`\n",
        f"- Radial bins: `{profiles['r_over_R'].nunique()}`\n\n",
        "## 口径\n\n",
        "- 主口径 `heat_stir_rot_rel` / `pv_stir_rot_rel` 是环向异常协方差，表示内部 stirring。\n",
        "- `heat_stir_rot` / `pv_stir_rot` 是未去环向均值的协方差参考。\n",
        "- `heat_mean_product` / `pv_mean_product` 是代表均值相乘的模板近似，只作对照，不作为 stirring 主结论。\n\n",
        "## 核心区 r/R <= 1.5 统计\n\n",
        markdown_table(summary),
        "\n",
    ]
    (output_dir / "stirring_transport_summary_zh.md").write_text("".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    rv_root = Path(args.rv_root)
    output_dir = Path(args.output_dir)
    partial_dir = output_dir / "partial_accum_parts"
    output_dir.mkdir(parents=True, exist_ok=True)
    partial_dir.mkdir(parents=True, exist_ok=True)
    tau_grid = _read_tau_grid(rv_root)
    depth_template, radial, _ = _read_profile_grid(rv_root)
    objects = _load_objects(rv_root, args.start, args.end, args.max_days, args.max_objects)
    if objects.empty:
        raise ValueError("No coherent global_ls_alpha objects selected.")
    object_ids = set(objects["eddy3d_object_id"].astype("int64"))
    points = _load_points(rv_root, object_ids)
    n2_profile = Path(args.n2_profile) if args.n2_profile else rv_root / "climatology" / "cmems_doy_climatology_1993_2022_31d_sigma0_dz_profile.npz"
    if not n2_profile.exists():
        raise FileNotFoundError(n2_profile)
    args_dict = {
        "filter_root": str(Path(args.filter_root)),
        "filter_template": str(args.filter_template),
        "n2_profile": str(n2_profile),
        "azimuth_bins": int(args.azimuth_bins),
        "kernel_bandwidth": float(args.kernel_bandwidth),
        "kernel_weight_min": float(args.kernel_weight_min),
        "force": bool(args.force),
    }
    tasks = _make_tasks(objects, points, partial_dir, args.chunk_days, args.force)
    summaries: list[dict[str, object]] = []
    if tasks:
        with ProcessPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
            futures = [pool.submit(_worker, task, args_dict, tau_grid, depth_template, radial) for task in tasks]
            for fut in tqdm(as_completed(futures), total=len(futures), desc="coherent stirring chunks"):
                summaries.append(fut.result())
    part_paths = sorted(partial_dir.glob("chunk_*.pkl"))
    if not part_paths:
        raise FileNotFoundError(f"No partial accumulators found under {partial_dir}")
    accum = _merge_partials(part_paths, len(tau_grid), len(depth_template), len(radial))
    profiles = _finalize(accum, tau_grid, depth_template, radial, rv_root)
    profiles.to_parquet(output_dir / "stirring_transport_profiles.parquet", index=False)
    profiles.to_csv(output_dir / "stirring_transport_profiles.csv", index=False)
    core15 = _core_summary(profiles, 1.5)
    core25 = _core_summary(profiles, 2.5)
    core15.to_csv(output_dir / "stirring_transport_core_r15.csv", index=False)
    core25.to_csv(output_dir / "stirring_transport_core_r25.csv", index=False)
    core15.to_parquet(output_dir / "stirring_transport_core_r15.parquet", index=False)
    core25.to_parquet(output_dir / "stirring_transport_core_r25.parquet", index=False)
    pd.DataFrame.from_records(summaries).to_csv(output_dir / "chunk_summary.csv", index=False)
    _plot_core(core15, output_dir, "core_r15")
    _plot_core(core25, output_dir, "core_r25")
    _write_summary(output_dir, profiles, core15, objects, args)
    manifest = {
        "status": "ok",
        "rv_root": str(rv_root),
        "output_dir": str(output_dir),
        "filter_root": str(Path(args.filter_root)),
        "n2_profile": str(n2_profile),
        "objects": int(len(objects)),
        "tracks": int(objects["track3d_id"].nunique()),
        "polarities": sorted(objects["polarity"].unique().tolist()),
        "tau_nodes": int(len(tau_grid)),
        "depth_levels": int(len(depth_template)),
        "radial_bins": int(len(radial)),
        "azimuth_bins": int(args.azimuth_bins),
        "direction": "global_ls_alpha y_rot only; no v_north output",
        "metrics": list(METRICS),
    }
    (output_dir / "stirring_transport_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute coherent-only representative stirring heat and PV transports in global-alpha y_rot coordinates.")
    parser.add_argument("--rv-root", default=str(DEFAULT_RV_ROOT))
    parser.add_argument("--filter-root", default=str(DEFAULT_FILTER_ROOT))
    parser.add_argument("--filter-template", default="global_phy_{year}_bandpass_30_180d.nc")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--n2-profile", default="")
    parser.add_argument("--start", default="")
    parser.add_argument("--end", default="")
    parser.add_argument("--max-days", type=int, default=0)
    parser.add_argument("--max-objects", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--chunk-days", type=int, default=14)
    parser.add_argument("--azimuth-bins", type=int, default=72)
    parser.add_argument("--kernel-bandwidth", type=float, default=0.075)
    parser.add_argument("--kernel-weight-min", type=float, default=1.0e-4)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
