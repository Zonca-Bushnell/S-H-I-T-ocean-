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

from src.First_temp.axis_streamfunction_separation import grid_spacing_m, relative_vorticity, streamfunction_from_zeta
from src.First_temp.lifecycle_continuous_representative import parse_tau_grid
from src.First_temp.lifecycle_ep_flux_nondim_validation import (
    OMEGA,
    ddz,
    load_n2,
    make_polar_grid,
    radial_derivative,
    azimuth_second_derivative,
)
from src.First_temp.tilted_ep_flux_validation import bilinear_sample, sanitize_ocean_field, xy_to_lonlat


RHO0 = 1025.0
CP = 3990.0


@dataclass(frozen=True)
class ChunkTask:
    chunk_id: int
    objects: pd.DataFrame
    center_points: pd.DataFrame
    output_path: Path


@lru_cache(maxsize=128)
def _dates_from_nc(path_text: str) -> dict[date, int]:
    path = Path(path_text)
    with netCDF4.Dataset(path) as ds:
        t = ds.variables["time"]
        values = netCDF4.num2date(
            t[:],
            units=t.units,
            calendar=getattr(t, "calendar", "standard"),
            only_use_cftime_datetimes=False,
            only_use_python_datetimes=True,
        )
    return {value.date(): int(i) for i, value in enumerate(values)}


def _first_filter_path(filter_root: Path, start_year: int, template: str) -> Path:
    path = filter_root / template.format(year=start_year)
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _filter_path(root: Path, year: int, template: str) -> Path:
    path = root / template.format(year=year)
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _clean(values: np.ndarray) -> np.ndarray:
    out = np.ma.filled(values, np.nan).astype("f8", copy=False)
    out[np.abs(out) > 1.0e20] = np.nan
    return out


def _selected_depth_count(path: Path, max_depth_m: float | None) -> int:
    with netCDF4.Dataset(path) as ds:
        depth = np.asarray(ds.variables["depth"][:], dtype="f8")
    if max_depth_m is None:
        return int(depth.size)
    return int(np.sum(depth <= float(max_depth_m) + 1.0e-6))


def _read_filter_day(
    filter_root: Path,
    theta_root: Path | None,
    day: date,
    *,
    filter_template: str,
    theta_template: str,
    max_depth_m: float | None,
) -> dict[str, np.ndarray]:
    path = _filter_path(filter_root, day.year, filter_template)
    day_index = _dates_from_nc(str(path))[day]
    with netCDF4.Dataset(path) as ds:
        depth_all = np.asarray(ds.variables["depth"][:], dtype="f8")
        depth_indexer = np.where(depth_all <= float(max_depth_m) + 1.0e-6)[0] if max_depth_m is not None else np.arange(depth_all.size)
        lon = np.asarray(ds.variables["longitude"][:], dtype="f8")
        lat = np.asarray(ds.variables["latitude"][:], dtype="f8")
        depth = depth_all[depth_indexer]
        u = _clean(ds.variables["uo_glor"][day_index, depth_indexer, :, :])
        v = _clean(ds.variables["vo_glor"][day_index, depth_indexer, :, :])
        if "thetao_glor" in ds.variables:
            theta = _clean(ds.variables["thetao_glor"][day_index, depth_indexer, :, :])
        else:
            if theta_root is None:
                raise KeyError(f"thetao_glor not found in {path}; provide --theta-root")
            theta_path = _filter_path(theta_root, day.year, theta_template)
            theta_index = _dates_from_nc(str(theta_path))[day]
            with netCDF4.Dataset(theta_path) as tds:
                theta = _clean(tds.variables["thetao_glor"][theta_index, depth_indexer, :, :])
    return {"lon": lon, "lat": lat, "depth": depth, "u": u, "v": v, "thetao": theta}


def _center_lines(points: pd.DataFrame) -> dict[int, pd.DataFrame]:
    return {
        int(object_id): part.sort_values("depth_index").reset_index(drop=True)
        for object_id, part in points.groupby("eddy3d_object_id", sort=False)
    }


def _load_objects(rv_root: Path, start: str | None, end: str | None) -> pd.DataFrame:
    objects = pd.read_parquet(rv_root / "object_cache" / "selected_lifecycle_objects.parquet")
    objects = objects[objects["shape_class"].astype(str).eq("coherent")].copy()
    objects["date"] = pd.to_datetime(objects["date"]).dt.strftime("%Y-%m-%d")
    if start:
        objects = objects[objects["date"] >= pd.Timestamp(start).strftime("%Y-%m-%d")].copy()
    if end:
        objects = objects[objects["date"] <= pd.Timestamp(end).strftime("%Y-%m-%d")].copy()
    required = {"eddy3d_object_id", "track3d_id", "polarity", "date", "life_phase", "mean_radius_m", "surface_lon", "surface_lat", "temp_direction_rad"}
    missing = sorted(required - set(objects.columns))
    if missing:
        raise KeyError(f"selected_lifecycle_objects missing columns: {missing}")
    return objects.sort_values(["date", "polarity", "eddy3d_object_id"]).reset_index(drop=True)


def _load_points(rv_root: Path, object_ids: set[int]) -> pd.DataFrame:
    cols = [
        "eddy3d_object_id",
        "track3d_id",
        "date",
        "polarity",
        "depth_index",
        "depth_m",
        "x_rot_m",
        "y_rot_m",
        "temp_direction_rad",
        "axis_alignment_method",
    ]
    points = pd.read_parquet(rv_root / "axis" / "rotated_points.parquet", columns=cols)
    points = points[points["eddy3d_object_id"].astype("int64").isin(object_ids)].copy()
    points["date"] = pd.to_datetime(points["date"]).dt.strftime("%Y-%m-%d")
    points["eddy3d_object_id"] = points["eddy3d_object_id"].astype("int64")
    points["depth_index"] = points["depth_index"].astype("int16")
    return points


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
    r_m = np.maximum(radial * radius_m, 1.0)
    dpsi_dr = radial_derivative(psi_prime, r_m)
    radial_lap = radial_derivative(dpsi_dr * r_m[None, :, None], r_m) / r_m[None, :, None]
    az_lap = azimuth_second_derivative(psi_prime, theta) / (r_m[None, :, None] ** 2)
    dpsi_dz = ddz(psi_prime, depth)
    strat = (f0 * f0 / n2)[:, None, None] * dpsi_dz
    q_total = radial_lap + az_lap + ddz(strat, depth)
    return q_total - np.nanmean(q_total, axis=2, keepdims=True)


def _sample_rotated_fields(
    obj,
    center_line: pd.DataFrame,
    lon: np.ndarray,
    lat: np.ndarray,
    depth: np.ndarray,
    psi_prime: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    thetao: np.ndarray,
    rr: np.ndarray,
    tt: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    if center_line.empty:
        return None
    radius = float(obj.mean_radius_m)
    theta_obj = float(obj.temp_direction_rad)
    cos_t = math.cos(theta_obj)
    sin_t = math.sin(theta_obj)
    local_x = rr * radius * np.cos(tt)
    local_y = rr * radius * np.sin(tt)
    psi_layers: list[np.ndarray] = []
    vrot_layers: list[np.ndarray] = []
    theta_layers: list[np.ndarray] = []
    for row in center_line.itertuples(index=False):
        source_k = int(row.depth_index)
        if source_k < 0 or source_k >= len(depth):
            continue
        x_rot = float(row.x_rot_m) + local_x
        y_rot = float(row.y_rot_m) + local_y
        x_orig = x_rot * cos_t - y_rot * sin_t
        y_orig = x_rot * sin_t + y_rot * cos_t
        target_lon, target_lat = xy_to_lonlat(x_orig, y_orig, float(obj.surface_lon), float(obj.surface_lat))
        psi_s = bilinear_sample(lon, lat, psi_prime[source_k], target_lon, target_lat)
        u_s = bilinear_sample(lon, lat, u[source_k], target_lon, target_lat)
        v_s = bilinear_sample(lon, lat, v[source_k], target_lon, target_lat)
        theta_s = bilinear_sample(lon, lat, thetao[source_k], target_lon, target_lat)
        v_rot = -u_s * sin_t + v_s * cos_t
        psi_layers.append(psi_s)
        vrot_layers.append(v_rot)
        theta_layers.append(theta_s)
    if len(psi_layers) < 3:
        return None
    return np.asarray(psi_layers), np.asarray(vrot_layers), np.asarray(theta_layers)


def _empty_accum(nz: int, nr: int) -> dict[str, np.ndarray | set]:
    names = (
        "heat_stir_rot",
        "heat_stir_rot_rel",
        "pv_stir_rot",
        "pv_stir_rot_rel",
        "v_rot_mean",
        "theta_mean",
        "q_mean",
        "valid_fraction",
    )
    out: dict[str, np.ndarray | set] = {name: np.zeros((nz, nr), dtype="f8") for name in names}
    out["count"] = np.zeros((nz, nr), dtype="f8")
    out["objects"] = set()
    out["tracks"] = set()
    out["dates"] = set()
    return out


def _add_to_accum(
    accum: dict[tuple[str, int], dict[str, np.ndarray | set]],
    polarity: str,
    tau_index: int,
    depth_indices: np.ndarray,
    terms: dict[str, np.ndarray],
    valid: np.ndarray,
    weight: float,
    object_id: int,
    track_id: int,
    date_text: str,
    nz: int,
    nr: int,
) -> None:
    key = (str(polarity), int(tau_index))
    if key not in accum:
        accum[key] = _empty_accum(nz, nr)
    item = accum[key]
    for local_k, global_k in enumerate(depth_indices):
        valid_row = valid[local_k]
        if not np.any(valid_row):
            continue
        for name, values in terms.items():
            arr = item[name]
            assert isinstance(arr, np.ndarray)
            arr[int(global_k), :] += np.nan_to_num(values[local_k], nan=0.0) * valid_row * weight
        count = item["count"]
        assert isinstance(count, np.ndarray)
        count[int(global_k), :] += valid_row.astype("f8") * weight
    assert isinstance(item["objects"], set)
    assert isinstance(item["tracks"], set)
    assert isinstance(item["dates"], set)
    item["objects"].add(int(object_id))
    item["tracks"].add(int(track_id))
    item["dates"].add(str(date_text))


def _merge_accum(target: dict, source: dict) -> None:
    for key, item in source.items():
        if key not in target:
            target[key] = {
                name: value.copy() if isinstance(value, np.ndarray) else set(value)
                for name, value in item.items()
            }
            continue
        for name, value in item.items():
            if isinstance(value, np.ndarray):
                target[key][name] += value
            elif isinstance(value, set):
                target[key][name].update(value)


def _worker(task: ChunkTask, args_dict: dict) -> dict[str, object]:
    args = argparse.Namespace(**args_dict)
    objects = task.objects.copy()
    if objects.empty:
        task.output_path.parent.mkdir(parents=True, exist_ok=True)
        with task.output_path.open("wb") as handle:
            pickle.dump({}, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return {"chunk_id": task.chunk_id, "objects": 0, "skipped": 0, "path": str(task.output_path)}

    filter_root = Path(args.filter_root)
    theta_root = Path(args.theta_root) if args.theta_root else None
    n2_path = Path(args.n2_profile)
    tau_grid = parse_tau_grid(args.tau_grid, float(args.tau_grid_step))
    radial, theta_angles, rr, tt, _ = make_polar_grid(float(args.rmax), int(args.radial_bins), int(args.azimuth_bins))
    first_path = _first_filter_path(filter_root, int(pd.Timestamp(objects["date"].min()).year), args.filter_template)
    nz = _selected_depth_count(first_path, args.max_depth_m)
    nr = len(radial)
    center_lines_all = _center_lines(task.center_points)
    accum: dict = {}
    processed = 0
    skipped = 0

    for date_text, day_objects in objects.groupby("date", sort=True):
        day = pd.Timestamp(date_text).date()
        data = _read_filter_day(
            filter_root,
            theta_root,
            day,
            filter_template=args.filter_template,
            theta_template=args.theta_template,
            max_depth_m=args.max_depth_m,
        )
        lon = data["lon"]
        lat = data["lat"]
        depth = data["depth"]
        u = sanitize_ocean_field(data["u"])
        v = sanitize_ocean_field(data["v"])
        thetao = sanitize_ocean_field(data["thetao"])
        _, dy, dx = grid_spacing_m(lon, lat)
        psi_prime = streamfunction_from_zeta(relative_vorticity(lon, lat, u, v), dx, dy)
        n2_full = load_n2(n2_path, depth)
        lat_ref = float(day_objects["surface_lat"].median()) if "surface_lat" in day_objects else float(np.nanmedian(lat))
        f0 = 2.0 * OMEGA * math.sin(math.radians(lat_ref))
        for obj in day_objects.itertuples(index=False):
            object_id = int(obj.eddy3d_object_id)
            center_line = center_lines_all.get(object_id)
            if center_line is None or center_line.empty:
                skipped += 1
                continue
            center_line = center_line.sort_values("depth_index").copy()
            depth_indices = center_line["depth_index"].to_numpy(dtype="int64")
            in_range = (depth_indices >= 0) & (depth_indices < len(depth))
            center_line = center_line.loc[in_range].reset_index(drop=True)
            depth_indices = depth_indices[in_range]
            if len(depth_indices) < 3:
                skipped += 1
                continue
            sampled = _sample_rotated_fields(obj, center_line, lon, lat, depth, psi_prime, u, v, thetao, rr, tt)
            if sampled is None:
                skipped += 1
                continue
            psi_s, vrot_s, theta_s = sampled
            depth_subset = depth[depth_indices]
            n2 = n2_full[depth_indices]
            q_prime = _q_prime_from_psi(psi_s, depth_subset, radial, theta_angles, float(obj.mean_radius_m), n2, f0)
            vbar = np.nanmean(vrot_s, axis=2, keepdims=True)
            tbar = np.nanmean(theta_s, axis=2, keepdims=True)
            qbar = np.nanmean(q_prime, axis=2, keepdims=True)
            v_rel = vrot_s - vbar
            t_rel = theta_s - tbar
            q_rel = q_prime - qbar
            heat = RHO0 * CP * np.nanmean(vrot_s * theta_s, axis=2)
            heat_rel = RHO0 * CP * np.nanmean(v_rel * t_rel, axis=2)
            pv = np.nanmean(vrot_s * q_prime, axis=2)
            pv_rel = np.nanmean(v_rel * q_rel, axis=2)
            finite = np.isfinite(vrot_s) & np.isfinite(theta_s) & np.isfinite(q_prime)
            valid = np.mean(finite, axis=2) >= float(args.min_valid_azimuth_fraction)
            terms = {
                "heat_stir_rot": heat,
                "heat_stir_rot_rel": heat_rel,
                "pv_stir_rot": pv,
                "pv_stir_rot_rel": pv_rel,
                "v_rot_mean": np.nanmean(vrot_s, axis=2),
                "theta_mean": np.nanmean(theta_s, axis=2),
                "q_mean": np.nanmean(q_prime, axis=2),
                "valid_fraction": np.mean(finite, axis=2),
            }
            weights = np.exp(-0.5 * ((tau_grid - float(obj.life_phase)) / max(float(args.kernel_bandwidth), 1.0e-12)) ** 2)
            for tau_index, weight in enumerate(weights):
                if weight < float(args.kernel_weight_min):
                    continue
                _add_to_accum(
                    accum,
                    str(obj.polarity),
                    int(tau_index),
                    depth_indices,
                    terms,
                    valid,
                    float(weight),
                    object_id,
                    int(obj.track3d_id),
                    str(obj.date),
                    nz,
                    nr,
                )
            processed += 1

    task.output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = task.output_path.with_suffix(task.output_path.suffix + ".tmp")
    with tmp.open("wb") as handle:
        pickle.dump(accum, handle, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(task.output_path)
    return {"chunk_id": task.chunk_id, "objects": processed, "skipped": skipped, "path": str(task.output_path)}


def _make_tasks(objects: pd.DataFrame, points: pd.DataFrame, part_dir: Path, chunk_days: int, resume: bool) -> list[ChunkTask]:
    dates = sorted(objects["date"].unique())
    tasks: list[ChunkTask] = []
    for chunk_id, start in enumerate(range(0, len(dates), int(chunk_days))):
        chunk_dates = dates[start : start + int(chunk_days)]
        chunk_objects = objects[objects["date"].isin(chunk_dates)].copy()
        object_ids = set(chunk_objects["eddy3d_object_id"].astype("int64"))
        chunk_points = points[points["eddy3d_object_id"].astype("int64").isin(object_ids)].copy()
        out_path = part_dir / f"chunk_{chunk_id:04d}_{chunk_dates[0].replace('-', '')}_{chunk_dates[-1].replace('-', '')}.pkl"
        if resume and out_path.exists():
            continue
        tasks.append(ChunkTask(chunk_id=chunk_id, objects=chunk_objects, center_points=chunk_points, output_path=out_path))
    return tasks


def _finalize(accum: dict, depth: np.ndarray, radial: np.ndarray, tau_grid: np.ndarray) -> pd.DataFrame:
    rows: list[dict] = []
    for (polarity, tau_index), item in sorted(accum.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        count = item["count"]
        assert isinstance(count, np.ndarray)
        for k, depth_m in enumerate(depth):
            for j, r in enumerate(radial):
                c = float(count[k, j])
                row = {
                    "polarity": polarity,
                    "phase_index": int(tau_index),
                    "phase_name": f"tau_{int(tau_index):03d}",
                    "tau_center": float(tau_grid[int(tau_index)]),
                    "depth_index": int(k),
                    "depth_m": float(depth_m),
                    "r_over_R": float(r),
                    "count": c,
                    "n_objects": len(item["objects"]),
                    "n_tracks": len(item["tracks"]),
                    "n_dates": len(item["dates"]),
                    "direction": "y_rot_global_alpha",
                }
                for name, value in item.items():
                    if name in {"count", "objects", "tracks", "dates"}:
                        continue
                    assert isinstance(value, np.ndarray)
                    row[name] = float(value[k, j] / c) if c > 0 else np.nan
                rows.append(row)
    return pd.DataFrame.from_records(rows)


def _core_summary(profiles: pd.DataFrame, r_limit: float) -> pd.DataFrame:
    rows = []
    for keys, part in profiles[profiles["r_over_R"] <= float(r_limit)].groupby(["polarity", "tau_center", "depth_index", "depth_m"], sort=True):
        weights = part["count"].to_numpy(dtype="f8")
        ok_w = np.isfinite(weights) & (weights > 0)
        row = dict(zip(["polarity", "tau_center", "depth_index", "depth_m"], keys))
        row["r_limit"] = float(r_limit)
        row["count"] = float(np.nansum(weights[ok_w])) if np.any(ok_w) else 0.0
        for col in ["heat_stir_rot", "heat_stir_rot_rel", "pv_stir_rot", "pv_stir_rot_rel", "v_rot_mean", "theta_mean", "q_mean", "valid_fraction"]:
            values = part[col].to_numpy(dtype="f8")
            ok = ok_w & np.isfinite(values)
            row[col] = float(np.nansum(values[ok] * weights[ok]) / np.nansum(weights[ok])) if np.any(ok) else np.nan
        row["n_objects"] = int(part["n_objects"].max()) if "n_objects" in part else 0
        row["n_tracks"] = int(part["n_tracks"].max()) if "n_tracks" in part else 0
        rows.append(row)
    return pd.DataFrame.from_records(rows)


def _plot_core(core: pd.DataFrame, output_dir: Path, r_label: str) -> None:
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    for polarity, part in core.groupby("polarity", sort=True):
        for col, label in [
            ("heat_stir_rot_rel", "heat stirring rel"),
            ("pv_stir_rot_rel", "PV stirring rel"),
            ("heat_stir_rot", "heat stirring raw covariance"),
            ("pv_stir_rot", "PV stirring raw covariance"),
        ]:
            pivot = part.pivot_table(index="depth_m", columns="tau_center", values=col)
            if pivot.empty:
                continue
            values = pivot.to_numpy(dtype="f8")
            vmax = float(np.nanpercentile(np.abs(values), 98)) if np.any(np.isfinite(values)) else 1.0
            if not np.isfinite(vmax) or vmax <= 0:
                vmax = 1.0
            fig, ax = plt.subplots(figsize=(7, 5), dpi=160)
            im = ax.imshow(
                values,
                origin="lower",
                aspect="auto",
                cmap="coolwarm",
                vmin=-vmax,
                vmax=vmax,
                extent=[float(pivot.columns.min()), float(pivot.columns.max()), float(pivot.index.min()), float(pivot.index.max())],
            )
            ax.set_title(f"{polarity} {label} ({r_label}, y_rot)")
            ax.set_xlabel("tau")
            ax.set_ylabel("depth m")
            fig.colorbar(im, ax=ax)
            fig.tight_layout()
            fig.savefig(fig_dir / f"{polarity}_{col}_{r_label}_tau_depth.png")
            plt.close(fig)


def _write_summary(output_dir: Path, objects: pd.DataFrame, profiles: pd.DataFrame, core15: pd.DataFrame, args: argparse.Namespace, chunk_rows: list[dict]) -> None:
    lines = [
        "# 代表涡旋 stirring 热通量与 PV 通量诊断",
        "",
        "本诊断只使用 `global_ls_alpha` 旋转后的 `y_rot` 横向速度，不输出地理北向通量，也不计算 trapping。",
        "",
        "## 输入",
        "",
        f"- representative root: `{args.representative_root}`",
        f"- filter root: `{args.filter_root}`",
        f"- theta root: `{args.theta_root or args.filter_root}`",
        f"- N2 profile: `{args.n2_profile}`",
        "",
        "## 样本",
        "",
        f"- selected coherent object-days: {len(objects):,}",
        f"- tracks: {objects['track3d_id'].nunique():,}",
        f"- polarities: {', '.join(sorted(objects['polarity'].astype(str).unique()))}",
        f"- processed chunks: {len(chunk_rows):,}",
        "",
        "## 公式",
        "",
        r"- 热 stirring 主口径：`heat_stir_rot_rel = rho0 Cp <(v_rot-<v_rot>_phi)(theta-<theta>_phi)>_phi`",
        r"- PV stirring 主口径：`pv_stir_rot_rel = <(v_rot-<v_rot>_phi)(q'-<q'>_phi)>_phi`",
        "",
        "## 核心区 r/R <= 1.5 概览",
        "",
    ]
    for polarity, part in core15.groupby("polarity", sort=True):
        lines.append(f"### {polarity}")
        for col in ["heat_stir_rot_rel", "pv_stir_rot_rel", "heat_stir_rot", "pv_stir_rot"]:
            values = part[col].to_numpy(dtype="f8")
            finite = values[np.isfinite(values)]
            if finite.size == 0:
                lines.append(f"- `{col}`: no finite values")
            else:
                lines.append(
                    f"- `{col}`: median={np.nanmedian(finite):.6e}, p10={np.nanpercentile(finite, 10):.6e}, p90={np.nanpercentile(finite, 90):.6e}"
                )
        lines.append("")
    lines.extend(
        [
            "## 解释边界",
            "",
            "- 这些量是代表涡旋旋转坐标中的横向 stirring，不是黑潮区域真实地理北向净输送。",
            "- `heat_stir_rot_rel` 与 `pv_stir_rot_rel` 去除了同一环上的方位平均，优先解释为内部非轴对称 stirring。",
            "- 若要研究海盆净经向输送，需要另做对象级轨迹和平移速度诊断。",
        ]
    )
    (output_dir / "stirring_transport_summary_zh.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    rv_root = Path(args.representative_root)
    output_dir = Path(args.output_dir)
    part_dir = output_dir / "partial_accum_parts"
    output_dir.mkdir(parents=True, exist_ok=True)
    part_dir.mkdir(parents=True, exist_ok=True)

    objects = _load_objects(rv_root, args.start, args.end)
    object_ids = set(objects["eddy3d_object_id"].astype("int64"))
    points = _load_points(rv_root, object_ids)
    tasks = _make_tasks(objects, points, part_dir, int(args.chunk_days), bool(args.resume))
    args_dict = vars(args).copy()
    summaries: list[dict] = []
    if tasks:
        with ProcessPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
            futures = [pool.submit(_worker, task, args_dict) for task in tasks]
            for future in as_completed(futures):
                result = future.result()
                summaries.append(result)
                print(f"[stirring] chunk={result['chunk_id']} objects={result['objects']} skipped={result['skipped']} path={result['path']}", flush=True)
    else:
        print("[stirring] no new chunks; using existing partial accumulators", flush=True)

    accum: dict = {}
    for path in sorted(part_dir.glob("chunk_*.pkl")):
        with path.open("rb") as handle:
            part = pickle.load(handle)
        _merge_accum(accum, part)
    if not accum:
        raise RuntimeError("No stirring accumulators found.")

    first_path = _first_filter_path(Path(args.filter_root), int(pd.Timestamp(objects["date"].min()).year), args.filter_template)
    with netCDF4.Dataset(first_path) as ds:
        depth_all = np.asarray(ds.variables["depth"][:], dtype="f8")
    depth = depth_all[: _selected_depth_count(first_path, args.max_depth_m)]
    tau_grid = parse_tau_grid(args.tau_grid, float(args.tau_grid_step))
    radial, _, _, _, _ = make_polar_grid(float(args.rmax), int(args.radial_bins), int(args.azimuth_bins))
    profiles = _finalize(accum, depth, radial, tau_grid)
    profiles.to_parquet(output_dir / "stirring_transport_profiles.parquet", index=False)
    profiles.to_csv(output_dir / "stirring_transport_profiles.csv", index=False)
    core15 = _core_summary(profiles, 1.5)
    core25 = _core_summary(profiles, 2.5)
    core15.to_parquet(output_dir / "stirring_transport_core_r15.parquet", index=False)
    core15.to_csv(output_dir / "stirring_transport_core_r15.csv", index=False)
    core25.to_parquet(output_dir / "stirring_transport_core_r25.parquet", index=False)
    core25.to_csv(output_dir / "stirring_transport_core_r25.csv", index=False)
    pd.DataFrame(summaries).to_csv(part_dir / "chunk_summary.csv", index=False)
    _plot_core(core15, output_dir, "r15")
    _plot_core(core25, output_dir, "r25")
    manifest = {
        "representative_root": str(rv_root),
        "output_dir": str(output_dir),
        "direction": "y_rot_global_alpha",
        "formulas": {
            "heat_stir_rot_rel": "rho0*Cp*mean((v_rot-mean_phi(v_rot))*(theta-mean_phi(theta)))",
            "pv_stir_rot_rel": "mean((v_rot-mean_phi(v_rot))*(q_prime-mean_phi(q_prime)))",
        },
        "n_objects": int(len(objects)),
        "n_tracks": int(objects["track3d_id"].nunique()),
        "profiles_rows": int(len(profiles)),
    }
    (output_dir / "stirring_transport_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_summary(output_dir, objects, profiles, core15, args, summaries)
    print(f"[stirring] complete output={output_dir}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build rotated-frame representative stirring heat and PV transport diagnostics.")
    parser.add_argument("--representative-root", default="/root/autodl-fs/kuroshiou/result_strict_contiguous/result_coherent_only/representative_vortex")
    parser.add_argument("--filter-root", default="/root/autodl-fs/kuroshiou/Filter")
    parser.add_argument("--theta-root", default="")
    parser.add_argument("--output-dir", default="/root/autodl-fs/kuroshiou/result_strict_contiguous/result_coherent_only/stirring_transport")
    parser.add_argument("--n2-profile", default="/root/autodl-fs/kuroshiou/result_strict_contiguous/result_coherent_only/representative_vortex/climatology/cmems_doy_climatology_1993_2022_31d_sigma0_dz_profile.npz")
    parser.add_argument("--filter-template", default="global_phy_{year}_bandpass_30_180d.nc")
    parser.add_argument("--theta-template", default="global_phy_{year}_thetao_bandpass_30_180d.nc")
    parser.add_argument("--start", default="")
    parser.add_argument("--end", default="")
    parser.add_argument("--tau-grid", default="")
    parser.add_argument("--tau-grid-step", type=float, default=0.05)
    parser.add_argument("--kernel-bandwidth", type=float, default=0.075)
    parser.add_argument("--kernel-weight-min", type=float, default=1.0e-4)
    parser.add_argument("--rmax", type=float, default=2.5)
    parser.add_argument("--radial-bins", type=int, default=40)
    parser.add_argument("--azimuth-bins", type=int, default=72)
    parser.add_argument("--chunk-days", type=int, default=7)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-depth-m", type=float, default=2000.0)
    parser.add_argument("--min-valid-azimuth-fraction", type=float, default=0.5)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
