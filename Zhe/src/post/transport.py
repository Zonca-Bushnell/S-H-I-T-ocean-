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

from src.Legacy.First_temp.axis_streamfunction_separation import (
    grid_spacing_m,
    relative_vorticity,
    streamfunction_from_zeta,
)
from src.Legacy.First_temp.lifecycle_ep_flux_nondim_validation import (
    OMEGA,
    azimuth_second_derivative,
    ddz,
    radial_derivative,
)
from src.Legacy.First_temp.tilted_ep_flux_validation import (
    bilinear_sample,
    load_n2,
    sanitize_ocean_field,
    xy_to_lonlat,
)


RHO0 = 1025.0
CP = 3992.0
POLARITIES = ("anticyclonic", "cyclonic")
MOMENT_NAMES = (
    "sum_v",
    "sum_theta",
    "sum_q",
    "sum_vtheta",
    "sum_vq",
    "sum_v2",
    "sum_theta2",
    "sum_q2",
    "sum_vrel_thetarel",
    "sum_vrel_qrel",
    "count",
)


def _parse_shapes(value: str) -> tuple[str, ...]:
    shapes = tuple(part.strip() for part in str(value).split(",") if part.strip())
    if not shapes:
        raise ValueError("--shapes must contain at least one shape class")
    return shapes


def _shape_label(shapes: tuple[str, ...]) -> str:
    return "-".join(shapes) + "-only" if len(shapes) == 1 else "-".join(shapes)


@dataclass(frozen=True)
class ChunkTask:
    chunk_id: int
    objects: pd.DataFrame
    points: pd.DataFrame
    output_path: Path


@lru_cache(maxsize=96)
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
    arr[np.abs(arr) > 1.0e20] = np.nan
    return arr


def _read_tau_grid(rv_root: Path) -> np.ndarray:
    table = pd.read_csv(rv_root / "continuous_tau_grid.csv")
    col = "tau" if "tau" in table.columns else "tau_center"
    return table[col].to_numpy(dtype="float64")


def _read_profile_grid(rv_root: Path) -> tuple[np.ndarray, np.ndarray]:
    profiles = pd.read_parquet(
        rv_root / "streamfunction_templates" / "continuous_radial_psi_profiles.parquet",
        columns=["depth_index", "depth_m", "r_over_R"],
    )
    depth_table = profiles[["depth_index", "depth_m"]].drop_duplicates().sort_values("depth_index")
    depth_index = depth_table["depth_index"].to_numpy(dtype="int64")
    depth = np.full(int(depth_index.max()) + 1, np.nan, dtype="float64")
    depth[depth_index] = depth_table["depth_m"].to_numpy(dtype="float64")
    radial = np.sort(profiles["r_over_R"].dropna().unique().astype("float64"))
    return depth, radial


def _load_objects(
    rv_root: Path,
    start: str,
    end: str,
    max_days: int,
    max_objects: int,
    shapes: tuple[str, ...],
) -> pd.DataFrame:
    objects = pd.read_parquet(rv_root / "object_cache" / "selected_lifecycle_objects.parquet")
    diag = pd.read_parquet(rv_root / "axis" / "object_diagnostics.parquet")
    objects["eddy3d_object_id"] = objects["eddy3d_object_id"].astype("int64")
    diag["eddy3d_object_id"] = diag["eddy3d_object_id"].astype("int64")

    add_cols = [
        col
        for col in (
            "shape_class",
            "polarity",
            "axis_alignment_method",
            "global_alpha_ok",
            "global_deviate_angle_rad",
            "global_theta0_rad",
        )
        if col not in objects.columns and col in diag.columns
    ]
    if add_cols:
        objects = objects.merge(diag[["eddy3d_object_id", *add_cols]], on="eddy3d_object_id", how="left")
    if "global_deviate_angle_rad" not in objects and "global_deviate_angle_deg" in objects:
        objects["global_deviate_angle_rad"] = np.radians(objects["global_deviate_angle_deg"].astype("float64"))
    required = {
        "eddy3d_object_id",
        "track3d_id",
        "date",
        "life_phase",
        "polarity",
        "shape_class",
        "axis_alignment_method",
        "global_alpha_ok",
        "global_deviate_angle_rad",
        "surface_lon",
        "surface_lat",
        "mean_radius_m",
    }
    missing = sorted(required - set(objects.columns))
    if missing:
        raise KeyError(f"selected/axis object tables are missing required columns: {missing}")

    objects["date"] = pd.to_datetime(objects["date"]).dt.strftime("%Y-%m-%d")
    objects = objects[objects["shape_class"].astype(str).isin(shapes)].copy()
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
        "x_rot_m",
        "y_rot_m",
    ]
    points = pd.read_parquet(rv_root / "axis" / "rotated_points.parquet", columns=cols)
    points["eddy3d_object_id"] = points["eddy3d_object_id"].astype("int64")
    points = points[points["eddy3d_object_id"].isin(object_ids)].copy()
    points["date"] = pd.to_datetime(points["date"]).dt.strftime("%Y-%m-%d")
    points["depth_index"] = points["depth_index"].astype("int16")
    return points.sort_values(["eddy3d_object_id", "depth_index"]).reset_index(drop=True)


def _read_filter_day(
    filter_root: Path,
    template: str,
    day: date,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    path = filter_root / template.format(year=day.year)
    if not path.exists():
        raise FileNotFoundError(path)
    day_index = _time_index(str(path)).get(day)
    if day_index is None:
        raise KeyError(f"{day} not found in {path}")
    with netCDF4.Dataset(path) as ds:
        for variable in ("uo_glor", "vo_glor", "thetao_glor"):
            if variable not in ds.variables:
                raise KeyError(f"{variable} not found in {path}; aggregate-product stirring requires bandpass thetao")
        lon = np.asarray(ds.variables["longitude"][:], dtype="float64")
        lat = np.asarray(ds.variables["latitude"][:], dtype="float64")
        depth = np.asarray(ds.variables["depth"][:], dtype="float64")
        u = _clean(ds.variables["uo_glor"][day_index])
        v = _clean(ds.variables["vo_glor"][day_index])
        theta = _clean(ds.variables["thetao_glor"][day_index])
    return lon, lat, depth, u, v, theta


def _make_polar_grid(radial: np.ndarray, azimuth_bins: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    theta = np.linspace(0.0, 2.0 * np.pi, int(azimuth_bins), endpoint=False)
    rr, tt = np.meshgrid(radial, theta, indexing="ij")
    return theta, rr, tt


def _center_lines(points: pd.DataFrame) -> dict[int, pd.DataFrame]:
    return {
        int(object_id): part.sort_values("depth_index").copy()
        for object_id, part in points.groupby("eddy3d_object_id", sort=False)
    }


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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    if center_line is None or len(center_line) < 3:
        return None
    radius_m = float(obj.mean_radius_m)
    alpha = float(obj.global_deviate_angle_rad)
    cos_a = math.cos(alpha)
    sin_a = math.sin(alpha)
    local_x = radial_mesh * radius_m * np.cos(theta_mesh)
    local_y = radial_mesh * radius_m * np.sin(theta_mesh)
    depth_indices: list[int] = []
    depth_values: list[float] = []
    psi_layers: list[np.ndarray] = []
    vrot_layers: list[np.ndarray] = []
    theta_layers: list[np.ndarray] = []

    for row in center_line.itertuples(index=False):
        k = int(row.depth_index)
        if k < 0 or k >= u.shape[0]:
            continue
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
        depth_indices.append(k)
        depth_values.append(float(row.depth_m))

    if len(depth_indices) < 3:
        return None
    return (
        np.asarray(depth_indices, dtype="int64"),
        np.asarray(depth_values, dtype="float64"),
        np.asarray(psi_layers, dtype="float64"),
        np.asarray(vrot_layers, dtype="float64"),
        np.asarray(theta_layers, dtype="float64"),
    )


def _q_prime_from_psi(
    psi_sample: np.ndarray,
    depth: np.ndarray,
    radial: np.ndarray,
    theta: np.ndarray,
    radius_m: float,
    n2: np.ndarray,
    f0: float,
) -> np.ndarray:
    psi = np.where(np.abs(psi_sample) > 1.0e20, np.nan, psi_sample)
    psi_prime = psi - np.nanmean(psi, axis=2, keepdims=True)
    r_m = np.maximum(radial * float(radius_m), 1.0)
    dpsi_dr = radial_derivative(psi_prime, r_m)
    radial_lap = radial_derivative(dpsi_dr * r_m[None, :, None], r_m) / r_m[None, :, None]
    az_lap = azimuth_second_derivative(psi_prime, theta) / (r_m[None, :, None] ** 2)
    dpsi_dz = ddz(psi_prime, depth)
    strat = (f0 * f0 / n2)[:, None, None] * dpsi_dz
    q_total = radial_lap + az_lap + ddz(strat, depth)
    return q_total - np.nanmean(q_total, axis=2, keepdims=True)


def _empty_accumulator(tau_count: int, depth_count: int, radial_count: int) -> dict:
    shape = (tau_count, depth_count, radial_count)
    return {
        polarity: {
            **{name: np.zeros(shape, dtype="float64") for name in MOMENT_NAMES},
            "objects": [set() for _ in range(tau_count)],
            "tracks": [set() for _ in range(tau_count)],
            "dates": [set() for _ in range(tau_count)],
        }
        for polarity in POLARITIES
    }


def _object_terms(
    vrot: np.ndarray,
    theta_field: np.ndarray,
    q_prime: np.ndarray,
) -> dict[str, np.ndarray]:
    valid = np.isfinite(vrot) & np.isfinite(theta_field) & np.isfinite(q_prime)
    count = valid.sum(axis=2).astype("float64")
    with np.errstate(invalid="ignore", divide="ignore"):
        v_mean = np.where(count > 0, np.nansum(np.where(valid, vrot, 0.0), axis=2) / count, np.nan)
        theta_mean = np.where(count > 0, np.nansum(np.where(valid, theta_field, 0.0), axis=2) / count, np.nan)
        q_mean = np.where(count > 0, np.nansum(np.where(valid, q_prime, 0.0), axis=2) / count, np.nan)
        terms = {
            "sum_v": v_mean,
            "sum_theta": theta_mean,
            "sum_q": q_mean,
            "sum_vtheta": np.where(count > 0, np.nansum(np.where(valid, vrot * theta_field, 0.0), axis=2) / count, np.nan),
            "sum_vq": np.where(count > 0, np.nansum(np.where(valid, vrot * q_prime, 0.0), axis=2) / count, np.nan),
            "sum_v2": np.where(count > 0, np.nansum(np.where(valid, vrot * vrot, 0.0), axis=2) / count, np.nan),
            "sum_theta2": np.where(count > 0, np.nansum(np.where(valid, theta_field * theta_field, 0.0), axis=2) / count, np.nan),
            "sum_q2": np.where(count > 0, np.nansum(np.where(valid, q_prime * q_prime, 0.0), axis=2) / count, np.nan),
            "count": np.where(count > 0, 1.0, 0.0),
        }
        v_rel = vrot - v_mean[:, :, None]
        theta_rel = theta_field - theta_mean[:, :, None]
        q_rel = q_prime - q_mean[:, :, None]
        terms["sum_vrel_thetarel"] = np.where(
            count > 0,
            np.nansum(np.where(valid, v_rel * theta_rel, 0.0), axis=2) / count,
            np.nan,
        )
        terms["sum_vrel_qrel"] = np.where(
            count > 0,
            np.nansum(np.where(valid, v_rel * q_rel, 0.0), axis=2) / count,
            np.nan,
        )
    return terms


def _add_terms(
    accumulator: dict,
    polarity: str,
    tau_index: int,
    depth_indices: np.ndarray,
    terms: dict[str, np.ndarray],
    object_id: int,
    track_id: int,
    date_text: str,
    depth_count: int,
) -> None:
    if polarity not in accumulator:
        return
    target = accumulator[polarity]
    valid_depth = (depth_indices >= 0) & (depth_indices < depth_count)
    if not np.any(valid_depth):
        return
    rows = np.where(valid_depth)[0]
    depth_slots = depth_indices[valid_depth].astype("int64")
    for name in MOMENT_NAMES:
        values = terms[name][rows]
        finite = np.isfinite(values)
        target[name][tau_index, depth_slots, :] += np.where(finite, values, 0.0)
    target["objects"][tau_index].add(int(object_id))
    target["tracks"][tau_index].add(int(track_id))
    target["dates"][tau_index].add(str(date_text))


def _tau_weights(life_phase: float, tau_grid: np.ndarray, bandwidth: float, min_weight: float) -> list[tuple[int, float]]:
    if not np.isfinite(life_phase):
        return []
    if bandwidth <= 0:
        index = int(np.argmin(np.abs(tau_grid - life_phase)))
        return [(index, 1.0)]
    raw = np.exp(-0.5 * ((tau_grid - float(life_phase)) / float(bandwidth)) ** 2)
    raw[raw < float(min_weight)] = 0.0
    total = float(raw.sum())
    if total <= 0:
        index = int(np.argmin(np.abs(tau_grid - life_phase)))
        return [(index, 1.0)]
    return [(int(i), float(w / total)) for i, w in enumerate(raw) if w > 0]


def _process_one_object(
    obj,
    center_lines: dict[int, pd.DataFrame],
    lon: np.ndarray,
    lat: np.ndarray,
    depth: np.ndarray,
    psi: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    theta_field: np.ndarray,
    n2_full: np.ndarray,
    radial: np.ndarray,
    theta_angles: np.ndarray,
    radial_mesh: np.ndarray,
    theta_mesh: np.ndarray,
) -> tuple[np.ndarray, dict[str, np.ndarray]] | None:
    line = center_lines.get(int(obj.eddy3d_object_id))
    sampled = _sample_rotated_fields(obj, line, lon, lat, psi, u, v, theta_field, radial_mesh, theta_mesh)
    if sampled is None:
        return None
    depth_indices, depth_values, psi_s, vrot_s, theta_s = sampled
    valid = (depth_indices >= 0) & (depth_indices < len(n2_full))
    if valid.sum() < 3:
        return None
    depth_indices = depth_indices[valid]
    depth_values = depth_values[valid]
    psi_s = psi_s[valid]
    vrot_s = vrot_s[valid]
    theta_s = theta_s[valid]
    f0 = 2.0 * OMEGA * math.sin(math.radians(float(obj.surface_lat)))
    q_prime = _q_prime_from_psi(
        psi_s,
        depth_values,
        radial,
        theta_angles,
        float(obj.mean_radius_m),
        n2_full[depth_indices],
        f0,
    )
    return depth_indices, _object_terms(vrot_s, theta_s, q_prime)


def _worker(
    task: ChunkTask,
    filter_root: Path,
    filter_template: str,
    n2_profile: Path,
    tau_grid: np.ndarray,
    depth_count: int,
    radial: np.ndarray,
    azimuth_bins: int,
    kernel_bandwidth: float,
    kernel_weight_min: float,
    force: bool,
) -> Path:
    if task.output_path.exists() and not force:
        return task.output_path
    task.output_path.parent.mkdir(parents=True, exist_ok=True)
    theta_angles, radial_mesh, theta_mesh = _make_polar_grid(radial, azimuth_bins)
    accumulator = _empty_accumulator(len(tau_grid), depth_count, len(radial))
    center_lines = _center_lines(task.points)

    for date_text, day_objects in task.objects.groupby("date", sort=True):
        day = pd.Timestamp(date_text).date()
        lon, lat, depth, u, v, theta_field = _read_filter_day(filter_root, filter_template, day)
        u = sanitize_ocean_field(u)
        v = sanitize_ocean_field(v)
        theta_field = sanitize_ocean_field(theta_field)
        _, dy, dx = grid_spacing_m(lon, lat)
        zeta = relative_vorticity(lon, lat, u, v)
        psi = streamfunction_from_zeta(zeta, dx=dx, dy=dy)
        n2_full = load_n2(n2_profile, depth)

        for obj in day_objects.itertuples(index=False):
            result = _process_one_object(
                obj,
                center_lines,
                lon,
                lat,
                depth,
                psi,
                u,
                v,
                theta_field,
                n2_full,
                radial,
                theta_angles,
                radial_mesh,
                theta_mesh,
            )
            if result is None:
                continue
            depth_indices, terms = result
            for tau_index, weight in _tau_weights(
                float(obj.life_phase),
                tau_grid,
                kernel_bandwidth,
                kernel_weight_min,
            ):
                weighted = {name: values * weight for name, values in terms.items()}
                _add_terms(
                    accumulator,
                    str(obj.polarity),
                    tau_index,
                    depth_indices,
                    weighted,
                    int(obj.eddy3d_object_id),
                    int(obj.track3d_id),
                    str(obj.date),
                    depth_count,
                )

    tmp_path = task.output_path.with_suffix(task.output_path.suffix + ".tmp")
    with tmp_path.open("wb") as handle:
        pickle.dump(accumulator, handle, protocol=pickle.HIGHEST_PROTOCOL)
    tmp_path.replace(task.output_path)
    return task.output_path


def _make_tasks(objects: pd.DataFrame, points: pd.DataFrame, partial_dir: Path, chunk_days: int) -> list[ChunkTask]:
    tasks: list[ChunkTask] = []
    days = sorted(objects["date"].unique())
    for chunk_id, start in enumerate(range(0, len(days), max(1, int(chunk_days)))):
        chunk_days_list = days[start : start + max(1, int(chunk_days))]
        chunk_objects = objects[objects["date"].isin(chunk_days_list)].copy()
        object_ids = set(chunk_objects["eddy3d_object_id"].astype("int64"))
        chunk_points = points[points["eddy3d_object_id"].isin(object_ids)].copy()
        first_day = chunk_days_list[0].replace("-", "")
        last_day = chunk_days_list[-1].replace("-", "")
        tasks.append(
            ChunkTask(
                chunk_id=chunk_id,
                objects=chunk_objects,
                points=chunk_points,
                output_path=partial_dir / f"chunk_{chunk_id:04d}_{first_day}_{last_day}.pkl",
            )
        )
    return tasks


def _resolve_n2_profile(rv_root: Path, explicit: str) -> Path:
    if explicit:
        path = Path(explicit)
        if path.exists():
            return path
        raise FileNotFoundError(path)
    filename = "cmems_doy_climatology_1993_2022_31d_sigma0_dz_profile.npz"
    candidates = [
        rv_root / "climatology" / filename,
        rv_root.parent / "climatology" / filename,
        rv_root.parent.parent / "climatology" / filename,
        rv_root.parent.parent.parent / "climatology" / filename,
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("N2 profile not found in: " + ", ".join(str(path) for path in candidates))


def _merge_one(target: dict, source: dict) -> None:
    for polarity in POLARITIES:
        if polarity not in source:
            continue
        for name in MOMENT_NAMES:
            target[polarity][name] += source[polarity][name]
        for set_name in ("objects", "tracks", "dates"):
            for index, values in enumerate(source[polarity][set_name]):
                target[polarity][set_name][index].update(values)


def _merge_partials(paths: list[Path], tau_count: int, depth_count: int, radial_count: int) -> dict:
    accumulator = _empty_accumulator(tau_count, depth_count, radial_count)
    for path in tqdm(paths, desc="Merging aggregate-product partials"):
        with path.open("rb") as handle:
            _merge_one(accumulator, pickle.load(handle))
    return accumulator


def _safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    return np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan, dtype="float64"),
        where=np.isfinite(denominator) & (np.abs(denominator) > 0),
    )


def _finalize_profiles(
    accumulator: dict,
    tau_grid: np.ndarray,
    depth_m: np.ndarray,
    radial: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict] = []
    for polarity in POLARITIES:
        data = accumulator[polarity]
        count = data["count"]
        mean_v = _safe_divide(data["sum_v"], count)
        mean_theta = _safe_divide(data["sum_theta"], count)
        mean_q = _safe_divide(data["sum_q"], count)
        product_heat = RHO0 * CP * _safe_divide(data["sum_vtheta"], count)
        mean_product_heat = RHO0 * CP * mean_v * mean_theta
        covariance_heat = product_heat - mean_product_heat
        product_pv = _safe_divide(data["sum_vq"], count)
        mean_product_pv = mean_v * mean_q
        covariance_pv = product_pv - mean_product_pv
        heat_rel = RHO0 * CP * _safe_divide(data["sum_vrel_thetarel"], count)
        pv_rel = _safe_divide(data["sum_vrel_qrel"], count)
        heat_fraction = _safe_divide(covariance_heat, product_heat)
        pv_fraction = _safe_divide(covariance_pv, product_pv)

        for tau_index, tau_value in enumerate(tau_grid):
            for depth_index, z in enumerate(depth_m):
                for radial_index, r_value in enumerate(radial):
                    c = float(count[tau_index, depth_index, radial_index])
                    if c <= 0:
                        continue
                    rows.append(
                        {
                            "polarity": polarity,
                            "tau_center": float(tau_value),
                            "depth_index": int(depth_index),
                            "depth_m": float(z),
                            "r_over_R": float(r_value),
                            "mean_v_rot": float(mean_v[tau_index, depth_index, radial_index]),
                            "mean_theta": float(mean_theta[tau_index, depth_index, radial_index]),
                            "mean_q": float(mean_q[tau_index, depth_index, radial_index]),
                            "product_mean_heat": float(product_heat[tau_index, depth_index, radial_index]),
                            "mean_product_heat": float(mean_product_heat[tau_index, depth_index, radial_index]),
                            "covariance_heat": float(covariance_heat[tau_index, depth_index, radial_index]),
                            "relative_covariance_fraction_heat": float(heat_fraction[tau_index, depth_index, radial_index]),
                            "product_mean_heat_rel": float(heat_rel[tau_index, depth_index, radial_index]),
                            "product_mean_pv": float(product_pv[tau_index, depth_index, radial_index]),
                            "mean_product_pv": float(mean_product_pv[tau_index, depth_index, radial_index]),
                            "covariance_pv": float(covariance_pv[tau_index, depth_index, radial_index]),
                            "relative_covariance_fraction_pv": float(pv_fraction[tau_index, depth_index, radial_index]),
                            "product_mean_pv_rel": float(pv_rel[tau_index, depth_index, radial_index]),
                            "count": c,
                            "n_objects": int(len(data["objects"][tau_index])),
                            "n_tracks": int(len(data["tracks"][tau_index])),
                            "n_dates": int(len(data["dates"][tau_index])),
                        }
                    )
    return pd.DataFrame(rows)


def _radial_weighted_core(profiles: pd.DataFrame, r_limit: float) -> pd.DataFrame:
    part = profiles[profiles["r_over_R"].le(float(r_limit))].copy()
    if part.empty:
        return part
    value_cols = [
        "mean_v_rot",
        "mean_theta",
        "mean_q",
        "product_mean_heat",
        "mean_product_heat",
        "covariance_heat",
        "relative_covariance_fraction_heat",
        "product_mean_heat_rel",
        "product_mean_pv",
        "mean_product_pv",
        "covariance_pv",
        "relative_covariance_fraction_pv",
        "product_mean_pv_rel",
    ]
    rows: list[dict] = []
    keys = ["polarity", "tau_center", "depth_index", "depth_m"]
    for key, group in part.groupby(keys, observed=True, sort=True):
        weight = np.maximum(group["r_over_R"].to_numpy(dtype="float64"), 1.0e-6)
        row = dict(zip(keys, key))
        for col in value_cols:
            values = group[col].to_numpy(dtype="float64")
            good = np.isfinite(values) & np.isfinite(weight)
            row[col] = float(np.average(values[good], weights=weight[good])) if np.any(good) else np.nan
        row["count"] = float(group["count"].sum())
        row["n_objects"] = int(group["n_objects"].max())
        row["n_tracks"] = int(group["n_tracks"].max())
        row["n_dates"] = int(group["n_dates"].max())
        row["r_limit"] = float(r_limit)
        rows.append(row)
    return pd.DataFrame(rows)


def _plot_tau_depth(core: pd.DataFrame, output_dir: Path, r_label: str, shape_label: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_specs = [
        ("product_mean_heat", "heat product mean"),
        ("mean_product_heat", "heat mean product"),
        ("covariance_heat", "heat covariance"),
        ("product_mean_heat_rel", "heat azimuthal-relative product"),
        ("product_mean_pv", "PV product mean"),
        ("mean_product_pv", "PV mean product"),
        ("covariance_pv", "PV covariance"),
        ("product_mean_pv_rel", "PV azimuthal-relative product"),
    ]
    for col, title in plot_specs:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True, constrained_layout=True)
        values = core[col].to_numpy(dtype="float64")
        vmax = np.nanpercentile(np.abs(values), 98) if np.isfinite(values).any() else 1.0
        vmax = float(vmax) if np.isfinite(vmax) and vmax > 0 else 1.0
        for ax, polarity in zip(axes, POLARITIES):
            part = core[core["polarity"].eq(polarity)]
            pivot = part.pivot_table(index="depth_m", columns="tau_center", values=col)
            pivot = pivot.loc[np.isfinite(pivot.index.to_numpy(dtype="float64"))]
            if pivot.empty or len(pivot.columns) == 0:
                ax.set_title(f"{polarity} {title}: no finite data")
                continue
            image = ax.imshow(
                pivot.to_numpy(dtype="float64"),
                aspect="auto",
                origin="upper",
                cmap="RdBu_r",
                vmin=-vmax,
                vmax=vmax,
                extent=[
                    float(pivot.columns.min()),
                    float(pivot.columns.max()),
                    float(pivot.index.max()),
                    float(pivot.index.min()),
                ],
            )
            ax.set_title(f"{polarity} {title}")
            ax.set_xlabel("tau")
            ax.set_ylabel("depth (m)")
            fig.colorbar(image, ax=ax, shrink=0.85)
        fig.suptitle(f"{shape_label} strict-contiguous, global_ls_alpha y_rot, {r_label}")
        fig.savefig(output_dir / f"{col}_{r_label}_tau_depth.png", dpi=180)
        plt.close(fig)


def _pivot_tau_depth(part: pd.DataFrame, column: str) -> pd.DataFrame:
    pivot = part.pivot_table(index="depth_m", columns="tau_center", values=column)
    return pivot.loc[np.isfinite(pivot.index.to_numpy(dtype="float64"))]


def _plot_product_mean_triptychs(core: pd.DataFrame, output_dir: Path, r_label: str, shape_label: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    specs = [
        ("heat", "product_mean_heat", "mean_product_heat", "W m$^{-2}$ equivalent"),
        ("pv", "product_mean_pv", "mean_product_pv", "PV flux units"),
    ]
    for name, product_col, mean_col, unit in specs:
        for polarity in POLARITIES:
            part = core[core["polarity"].eq(polarity)].copy()
            if part.empty:
                continue
            product = part[product_col].to_numpy(dtype="float64")
            mean_product = part[mean_col].to_numpy(dtype="float64")
            vmax = np.nanpercentile(np.abs(np.r_[product, mean_product]), 98)
            vmax = float(vmax) if np.isfinite(vmax) and vmax > 0 else 1.0

            ratio = np.divide(
                np.abs(product - mean_product),
                np.maximum(np.abs(mean_product), 1.0e-30),
                out=np.full_like(product, np.nan, dtype="float64"),
                where=np.isfinite(product) & np.isfinite(mean_product),
            )
            part["abs_covariance_over_mean_product"] = np.log10(np.clip(ratio, 1.0e-3, 1.0e3))

            panels = [
                (product_col, "product then mean", "RdBu_r", -vmax, vmax, unit),
                (mean_col, "mean then product", "RdBu_r", -vmax, vmax, unit),
                (
                    "abs_covariance_over_mean_product",
                    "|product-mean| / |mean-product|",
                    "viridis",
                    -2.0,
                    2.0,
                    "log10 absolute ratio",
                ),
            ]
            fig, axes = plt.subplots(1, 3, figsize=(15, 4.7), sharey=True, constrained_layout=True)
            for ax, (column, title, cmap, vmin, vmax_panel, label) in zip(axes, panels):
                pivot = _pivot_tau_depth(part, column)
                if pivot.empty or len(pivot.columns) == 0:
                    ax.set_title(f"{polarity} {name}: no finite data")
                    continue
                image = ax.imshow(
                    pivot.to_numpy(dtype="float64"),
                    aspect="auto",
                    origin="upper",
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax_panel,
                    extent=[
                        float(pivot.columns.min()),
                        float(pivot.columns.max()),
                        float(pivot.index.max()),
                        float(pivot.index.min()),
                    ],
                )
                ax.set_title(f"{polarity} {name}: {title}")
                ax.set_xlabel("tau")
                ax.set_ylabel("depth (m)")
                fig.colorbar(image, ax=ax, shrink=0.86, label=label)
            fig.suptitle(f"{shape_label} strict-contiguous, global_ls_alpha y_rot, {r_label}")
            fig.savefig(output_dir / f"{polarity}_{name}_product_mean_triptych_{r_label}.png", dpi=180)
            plt.close(fig)


def _depth_from_points(points: pd.DataFrame) -> np.ndarray:
    table = points[["depth_index", "depth_m"]].drop_duplicates().sort_values("depth_index")
    depth_index = table["depth_index"].to_numpy(dtype="int64")
    depth = np.full(int(depth_index.max()) + 1, np.nan, dtype="float64")
    depth[depth_index] = table["depth_m"].to_numpy(dtype="float64")
    return depth


def _plot_product_comparison(core: pd.DataFrame, output_dir: Path, r_label: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    specs = [
        ("product_mean_heat", "mean_product_heat", "heat"),
        ("product_mean_pv", "mean_product_pv", "pv"),
    ]
    for xcol, ycol, name in specs:
        fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
        for polarity in POLARITIES:
            part = core[core["polarity"].eq(polarity)]
            ax.scatter(part[xcol], part[ycol], s=10, alpha=0.45, label=polarity)
        all_values = np.r_[core[xcol].to_numpy(dtype="float64"), core[ycol].to_numpy(dtype="float64")]
        finite = all_values[np.isfinite(all_values)]
        if finite.size:
            lim = np.nanpercentile(np.abs(finite), 98)
            if np.isfinite(lim) and lim > 0:
                ax.plot([-lim, lim], [-lim, lim], "k--", lw=1)
                ax.set_xlim(-lim, lim)
                ax.set_ylim(-lim, lim)
        ax.set_xlabel("product mean")
        ax.set_ylabel("mean product")
        ax.set_title(f"{name}: product mean vs mean product ({r_label})")
        ax.legend()
        fig.savefig(output_dir / f"{name}_product_vs_mean_product_{r_label}.png", dpi=180)
        plt.close(fig)


def _plot_lifecycle(core: pd.DataFrame, output_dir: Path, r_label: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    specs = [("product_mean_heat", "covariance_heat", "heat"), ("product_mean_pv", "covariance_pv", "pv")]
    for product_col, covariance_col, name in specs:
        fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
        for polarity in POLARITIES:
            part = core[core["polarity"].eq(polarity)]
            phase = part.groupby("tau_center", observed=True)[[product_col, covariance_col]].mean().reset_index()
            ax.plot(phase["tau_center"], phase[product_col], marker="o", ms=3, label=f"{polarity} product")
            ax.plot(phase["tau_center"], phase[covariance_col], marker="s", ms=3, linestyle="--", label=f"{polarity} covariance")
        ax.axhline(0, color="0.2", lw=0.8)
        ax.set_xlabel("tau")
        ax.set_title(f"{name} lifecycle ({r_label})")
        ax.legend(ncol=2, fontsize=8)
        fig.savefig(output_dir / f"{name}_lifecycle_{r_label}.png", dpi=180)
        plt.close(fig)


def _metric_summary(core: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for polarity in POLARITIES:
        part = core[core["polarity"].eq(polarity)].copy()
        for prefix in ("heat", "pv"):
            product = part[f"product_mean_{prefix}"].to_numpy(dtype="float64")
            mean_product = part[f"mean_product_{prefix}"].to_numpy(dtype="float64")
            covariance = part[f"covariance_{prefix}"].to_numpy(dtype="float64")
            good = np.isfinite(product) & np.isfinite(mean_product) & np.isfinite(covariance)
            if not np.any(good):
                continue
            ratio = np.divide(
                np.abs(mean_product[good]),
                np.abs(product[good]),
                out=np.full(good.sum(), np.nan),
                where=np.abs(product[good]) > 0,
            )
            cov_frac = np.divide(
                covariance[good],
                product[good],
                out=np.full(good.sum(), np.nan),
                where=np.abs(product[good]) > 0,
            )
            rows.append(
                {
                    "polarity": polarity,
                    "quantity": prefix,
                    "median_abs_product_mean": float(np.nanmedian(np.abs(product[good]))),
                    "median_abs_mean_product": float(np.nanmedian(np.abs(mean_product[good]))),
                    "median_abs_mean_product_over_product": float(np.nanmedian(ratio)),
                    "median_covariance_fraction": float(np.nanmedian(cov_frac)),
                    "same_sign_fraction": float(np.nanmean(np.sign(product[good]) == np.sign(mean_product[good]))),
                    "n_grid_points": int(good.sum()),
                    "median_n_objects": float(np.nanmedian(part["n_objects"])),
                    "median_n_tracks": float(np.nanmedian(part["n_tracks"])),
                }
            )
    return pd.DataFrame(rows)



def _write_summary(output_dir: Path, summary: pd.DataFrame, core15: pd.DataFrame, args: argparse.Namespace) -> None:
    shapes = _parse_shapes(args.shapes)
    shape_label = _shape_label(shapes)
    lines = [
        f"# Kuroshiou {shape_label} aggregate-product stirring \u8bca\u65ad",
        "",
        f"\u672c\u5b9e\u9a8c\u53ea\u4f7f\u7528 strict-contiguous {shape_label} \u4ee3\u8868\u6da1\u5bf9\u8c61\uff0c\u5e76\u4e14\u53ea\u5728 `global_ls_alpha` \u65cb\u8f6c\u540e\u7684 `y_rot` \u6a2a\u5411\u65b9\u5411\u4e0a\u8ba1\u7b97 stirring\u3002\u8fd9\u91cc\u4e0d\u8ba1\u7b97 trapping\uff0c\u4e5f\u4e0d\u8f93\u51fa\u5730\u7406\u5317\u5411\u53e3\u5f84\u3002",
        "",
        "\u6838\u5fc3\u533a\u522b\u662f\uff1a\u4e3b\u8bca\u65ad\u4f7f\u7528\u4e58\u79ef\u540e\u5e73\u5747 `mean(v_rot * theta')` \u548c `mean(v_rot * q')`\uff1b\u5e73\u5747\u540e\u4e58\u79ef `mean(v_rot) * mean(theta')` \u4e0e `mean(v_rot) * mean(q')` \u53ea\u4f5c\u4e3a\u5bf9\u7167\u3002\u82e5\u4e8c\u8005\u5dee\u5f02\u5f88\u5927\uff0c\u8bf4\u660e\u8f93\u9001\u4e3b\u8981\u6765\u81ea\u534f\u65b9\u5dee\u7ed3\u6784\uff0c\u800c\u4e0d\u662f\u4e00\u9636\u5e73\u5747\u4ee3\u8868\u6da1\u573a\u672c\u8eab\u3002",
        "",
        "## r/R <= 1.5 \u6838\u5fc3\u533a\u6570\u503c\u6458\u8981",
        "",
        "| polarity | quantity | median_abs_mean_product_over_product | median covariance/product | same-sign fraction | median n_objects | median n_tracks |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.polarity} | {row.quantity} | "
            f"{row.median_abs_mean_product_over_product:.4g} | "
            f"{row.median_covariance_fraction:.4g} | "
            f"{row.same_sign_fraction:.3f} | "
            f"{row.median_n_objects:.0f} | {row.median_n_tracks:.0f} |"
        )
    lines.extend(
        [
            "",
            "## \u89e3\u91ca",
            "",
            "- `product_mean` \u662f\u805a\u5408\u578b stirring \u901a\u91cf\uff0c\u7b49\u4ef7\u4e8e\u6bcf\u4e2a object-day \u5148\u8ba1\u7b97\u5c40\u5730\u4e58\u79ef\uff0c\u518d\u8fdb\u5165\u751f\u547d\u5468\u671f-\u6df1\u5ea6-\u534a\u5f84\u5408\u6210\u3002",
            "- `mean_product` \u662f\u5e73\u5747\u4ee3\u8868\u6da1\u573a\u76f8\u4e58\uff1b\u5b83\u53ea\u80fd\u8bf4\u660e\u4e00\u9636\u5e73\u5747\u7ed3\u6784\u662f\u5426\u81ea\u8eab\u80fd\u4ea7\u751f\u8f93\u9001\uff0c\u4e0d\u80fd\u66ff\u4ee3\u771f\u5b9e\u534f\u65b9\u5dee\u8f93\u9001\u3002",
            "- `covariance = product_mean - mean_product`\u3002\u5982\u679c covariance/product \u63a5\u8fd1 1\uff0c\u8bf4\u660e\u901a\u91cf\u51e0\u4e4e\u5168\u90e8\u7531\u534f\u65b9\u5dee\u8d21\u732e\u3002",
            "- `product_mean_*_rel` \u662f\u73af\u5411\u5e73\u5747\u53bb\u9664\u540e\u7684\u975e\u8f74\u5bf9\u79f0 stirring \u5bf9\u7167\uff0c\u7528\u6765\u89c2\u5bdf\u6da1\u65cb\u5185\u90e8\u975e\u8f74\u5bf9\u79f0\u7ed3\u6784\u7684\u8d21\u732e\u3002",
            "",
            "## \u8f93\u5165\u4e0e\u53c2\u6570",
            "",
            f"- representative root: `{args.rv_root}`",
            f"- filter root: `{args.filter_root}`",
            f"- output: `{args.output_dir}`",
            f"- shape filter: `{', '.join(shapes)}`",
            f"- workers: `{args.workers}`; chunk days: `{args.chunk_days}`; azimuth bins: `{args.azimuth_bins}`",
            f"- tau kernel bandwidth: `{args.kernel_bandwidth}`",
            f"- profiles rows in core r/R<=1.5: `{len(core15)}`",
        ]
    )
    (output_dir / "aggregate_product_stirring_summary_zh.md").write_text("\n".join(lines), encoding="utf-8")

def run(args: argparse.Namespace) -> None:
    rv_root = Path(args.rv_root)
    output_dir = Path(args.output_dir)
    partial_dir = output_dir / "partial_accumulators"
    figure_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    partial_dir.mkdir(parents=True, exist_ok=True)

    tau_grid = _read_tau_grid(rv_root)
    _, radial = _read_profile_grid(rv_root)
    if args.rmax > 0:
        radial = radial[radial <= float(args.rmax)]
    if radial.size == 0:
        raise ValueError("No radial bins remain after applying --rmax")

    shapes = _parse_shapes(args.shapes)
    shape_label = _shape_label(shapes)
    objects = _load_objects(rv_root, args.start, args.end, args.max_days, args.max_objects, shapes)
    if objects.empty:
        raise ValueError(f"No {shape_label} global_ls_alpha objects selected")
    points = _load_points(rv_root, set(objects["eddy3d_object_id"].astype("int64")))
    if points.empty:
        raise ValueError("No rotated points found for selected objects")
    depth_m = _depth_from_points(points)

    n2_profile = _resolve_n2_profile(rv_root, args.n2_profile)

    tasks = _make_tasks(objects, points, partial_dir, args.chunk_days)
    worker_kwargs = dict(
        filter_root=Path(args.filter_root),
        filter_template=args.filter_template,
        n2_profile=n2_profile,
        tau_grid=tau_grid,
        depth_count=len(depth_m),
        radial=radial,
        azimuth_bins=args.azimuth_bins,
        kernel_bandwidth=args.kernel_bandwidth,
        kernel_weight_min=args.kernel_weight_min,
        force=args.force,
    )
    paths: list[Path] = []
    if args.workers <= 1:
        for task in tqdm(tasks, desc="Aggregate-product chunks"):
            paths.append(_worker(task, **worker_kwargs))
    else:
        with ProcessPoolExecutor(max_workers=int(args.workers)) as pool:
            futures = {pool.submit(_worker, task, **worker_kwargs): task for task in tasks}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Aggregate-product chunks"):
                paths.append(future.result())

    accumulator = _merge_partials(sorted(paths), len(tau_grid), len(depth_m), len(radial))
    profiles = _finalize_profiles(accumulator, tau_grid, depth_m, radial)
    if profiles.empty:
        raise RuntimeError("Aggregate-product profiles are empty")
    profiles.to_parquet(output_dir / "aggregate_product_stirring_profiles.parquet", index=False)
    profiles.to_csv(output_dir / "aggregate_product_stirring_profiles.csv", index=False)

    core15 = _radial_weighted_core(profiles, 1.5)
    core25 = _radial_weighted_core(profiles, 2.5)
    core15.to_parquet(output_dir / "aggregate_product_stirring_core_r15.parquet", index=False)
    core15.to_csv(output_dir / "aggregate_product_stirring_core_r15.csv", index=False)
    core25.to_parquet(output_dir / "aggregate_product_stirring_core_r25.parquet", index=False)
    core25.to_csv(output_dir / "aggregate_product_stirring_core_r25.csv", index=False)

    _plot_tau_depth(core15, figure_dir, "core_r15", shape_label)
    _plot_product_mean_triptychs(core15, figure_dir, "core_r15", shape_label)
    _plot_product_comparison(core15, figure_dir, "core_r15")
    _plot_lifecycle(core15, figure_dir, "core_r15")
    summary = _metric_summary(core15)
    summary.to_csv(output_dir / "product_mean_vs_mean_product_summary.csv", index=False)
    _write_summary(output_dir, summary, core15, args)

    manifest = {
        "rv_root": str(rv_root),
        "filter_root": str(args.filter_root),
        "output_dir": str(output_dir),
        "shape_filter": list(shapes),
        "shape_label": shape_label,
        "n_selected_objects": int(objects["eddy3d_object_id"].nunique()),
        "n_selected_tracks": int(objects["track3d_id"].nunique()),
        "n_selected_days": int(objects["date"].nunique()),
        "tau_nodes": [float(v) for v in tau_grid],
        "depth_count": int(len(depth_m)),
        "radial_count": int(len(radial)),
        "azimuth_bins": int(args.azimuth_bins),
        "scientific_scope": f"{shape_label} strict-contiguous global_ls_alpha y_rot aggregate-product stirring",
    }
    (output_dir / "aggregate_product_stirring_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute aggregate-product heat/PV stirring in global_ls_alpha y_rot coordinates.",
    )
    parser.add_argument("--rv-root", required=True, help="Representative vortex root directory.")
    parser.add_argument("--filter-root", required=True, help="Directory containing 30-180d bandpass NetCDF files.")
    parser.add_argument("--output-dir", required=True, help="Output directory for profiles, plots, and summaries.")
    parser.add_argument("--shapes", default="coherent", help="Comma-separated shape classes to include.")
    parser.add_argument("--filter-template", default="global_phy_{year}_bandpass_30_180d.nc")
    parser.add_argument("--n2-profile", default="", help="Optional sigma0 dz profile npz. Defaults under rv-root/climatology.")
    parser.add_argument("--start", default="", help="Inclusive YYYY-MM-DD object-date filter.")
    parser.add_argument("--end", default="", help="Inclusive YYYY-MM-DD object-date filter.")
    parser.add_argument("--max-days", type=int, default=0, help="Debug cap on number of dates.")
    parser.add_argument("--max-objects", type=int, default=0, help="Debug cap on selected object-days.")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--chunk-days", type=int, default=7)
    parser.add_argument("--azimuth-bins", type=int, default=72)
    parser.add_argument("--rmax", type=float, default=2.5)
    parser.add_argument("--kernel-bandwidth", type=float, default=0.075)
    parser.add_argument("--kernel-weight-min", type=float, default=1.0e-4)
    parser.add_argument("--resume", action="store_true", help="Compatibility flag; partial reuse is the default.")
    parser.add_argument("--force", action="store_true", help="Recompute existing partial accumulators.")
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
