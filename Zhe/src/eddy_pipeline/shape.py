from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import shutil
from pathlib import Path

import netCDF4
import numpy as np
import pandas as pd
import yaml

from .completion import _complete_day_worker, complete_centers
from .utils.common import ensure_dirs, load_config
from .utils.streaming_cmems import build_source_index, is_streaming_source_configured, selected_depth_count
from .utils.table_io import discover_parquet_parts, read_partitioned_table, read_table, table_exists, write_parquet_parts_to_single, write_table, write_table_fast


EARTH_RADIUS_M = 6_371_000.0
SHAPE_CLASSES = ["upright_like", "coherent", "complex", "mixed", "transitional", "unknown"]
RULE_VERSION = "shape_only_v1"


def _paths(config: dict, output_name: str = "shape_classification") -> dict[str, Path]:
    out = Path(config["paths"]["output_dir"]) / output_name
    return {
        "root": out,
        "eligible": out / "eligible_tracks.parquet",
        "daily": out / "shape_daily_metrics.parquet",
        "tracks": out / "shape_tracks.parquet",
        "thresholds": out / "shape_thresholds.yaml",
        "by_shape": out / "by_shape",
    }


def _catalog_paths(config: dict) -> dict[str, Path]:
    root = Path(config["paths"]["catalog_dir"])
    return {
        "tracks": root / "tracks_3d.parquet",
        "objects": root / "vertical_objects.parquet",
        "centers": root / "layer_centers_completed.parquet",
        "centers_parts": root / "layer_centers_completed_parts",
    }


def _uv_path(config: dict, day: pd.Timestamp) -> Path:
    return Path(config["paths"]["input_daily_dir"]) / f"uv_{day:%Y%m%d}.nc"


def _load_depth_count(config: dict, objects: pd.DataFrame, centers: pd.DataFrame | None = None) -> int:
    if centers is not None and not centers.empty and "depth_index" in centers.columns:
        return int(centers["depth_index"].max()) + 1
    if objects.empty:
        raise ValueError("Cannot infer fixed depth count from an empty object table.")
    day = pd.Timestamp(objects.sort_values("date").iloc[0].date)
    if is_streaming_source_configured(config) and not bool(config.get("conversion", {}).get("use_input_daily", True)):
        source_day = build_source_index(config, day.date(), day.date())[day.date()]
        return selected_depth_count(config, source_day.path)
    uv_path = _uv_path(config, day)
    if not uv_path.exists():
        raise FileNotFoundError(f"Cannot infer depth count; missing UV file: {uv_path}")
    with netCDF4.Dataset(uv_path) as ds:
        return int(len(ds.variables["depth"][:]))


def _local_xy_m(lon: np.ndarray, lat: np.ndarray, lon0: float, lat0: float) -> tuple[np.ndarray, np.ndarray]:
    x = np.radians(lon - lon0) * EARTH_RADIUS_M * np.cos(np.radians(lat0))
    y = np.radians(lat - lat0) * EARTH_RADIUS_M
    return x, y


def _wrap180(values: np.ndarray) -> np.ndarray:
    return (values + 180.0) % 360.0 - 180.0


def _monotonic_ratio(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    if values.size < 3:
        return np.nan
    dv = np.diff(values)
    if dv.size == 0:
        return np.nan
    up = np.sum(dv >= 0.0) / dv.size
    down = np.sum(dv <= 0.0) / dv.size
    return float(max(up, down))


def _direction_turn_metrics(direction_deg: np.ndarray, valid: np.ndarray) -> tuple[float, float]:
    use = valid & np.isfinite(direction_deg)
    if np.sum(use) < 2:
        return np.nan, np.nan
    dtheta = np.abs(_wrap180(np.diff(direction_deg[use])))
    if dtheta.size == 0:
        return 0.0, 0.0
    return float(np.nanmean(dtheta)), float(np.nanmax(dtheta))


def _eligible_tracks(tracks: pd.DataFrame, *, lifetime_min_days: int, radius_min_m: float) -> pd.DataFrame:
    out = tracks[
        (tracks["lifetime_days"].astype(float) > float(lifetime_min_days))
        & (tracks["mean_radius_m"].astype(float) > float(radius_min_m))
    ].copy()
    return out.sort_values(["lifetime_days", "mean_radius_m", "track3d_id"], ascending=[False, False, True])


def _completed_counts(centers: pd.DataFrame, objects: pd.DataFrame, depth_count: int) -> pd.DataFrame:
    requested = objects[["track3d_id", "eddy3d_object_id"]].drop_duplicates().copy()
    expected = (
        objects.groupby("track3d_id", as_index=False)
        .size()
        .rename(columns={"size": "object_count"})
    )
    expected["expected_center_rows"] = expected["object_count"].astype(int) * int(depth_count)
    if centers.empty:
        expected["completed_center_rows"] = 0
    else:
        if "eddy3d_object_id" in centers.columns:
            use_centers = centers[
                centers["eddy3d_object_id"].astype(int).isin(set(requested["eddy3d_object_id"].astype(int)))
            ].copy()
        else:
            use_centers = centers.iloc[0:0].copy()
        have = use_centers.groupby("track3d_id", as_index=False).size().rename(columns={"size": "completed_center_rows"})
        expected = expected.merge(have, on="track3d_id", how="left")
        expected["completed_center_rows"] = expected["completed_center_rows"].fillna(0).astype(int)
    expected["completion_ready"] = expected["completed_center_rows"] >= expected["expected_center_rows"]
    return expected


def _ensure_completed_centers(
    config_path: str | Path,
    config: dict,
    eligible: pd.DataFrame,
    objects: pd.DataFrame,
    *,
    complete_missing: bool,
    force: bool,
    completion_output_mode: str,
    workers: int,
) -> pd.DataFrame:
    cat = _catalog_paths(config)
    if table_exists(cat["centers"]):
        centers = read_table(cat["centers"])
    else:
        centers = pd.DataFrame()
    depth_count = _load_depth_count(config, objects, centers if not centers.empty else None)
    status = _completed_counts(centers, objects, depth_count)
    eligible_ids = set(eligible["track3d_id"].astype(int))
    status = status[status["track3d_id"].astype(int).isin(eligible_ids)].copy()
    missing_ids = set(status.loc[~status["completion_ready"], "track3d_id"].astype(int))
    missing_object_ids = set(
        objects.loc[objects["track3d_id"].astype(int).isin(missing_ids), "eddy3d_object_id"].astype(int)
    )
    if force and complete_missing:
        missing_ids = eligible_ids
        missing_object_ids = set(objects["eddy3d_object_id"].astype(int))
    if missing_object_ids:
        if not complete_missing:
            raise RuntimeError(
                "Completed centers are missing for eligible tracks. "
                "Run with --complete-missing or run complete_3d_layer_centers.py first. "
                f"Missing/incomplete object count: {len(missing_object_ids)}"
            )
        print(f"Completing fixed-depth centers for eligible objects: {len(missing_object_ids)}")
        complete_centers(
            config_path,
            object_ids=missing_object_ids,
            force=force,
            output_mode=completion_output_mode,
            workers=workers,
        )
        centers = read_table(cat["centers"])
    if centers.empty:
        raise RuntimeError("No completed centers are available for shape classification.")
    requested_object_ids = set(objects["eddy3d_object_id"].astype(int))
    if "eddy3d_object_id" in centers.columns:
        return centers[centers["eddy3d_object_id"].astype(int).isin(requested_object_ids)].copy()
    return centers[centers["track3d_id"].astype(int).isin(eligible_ids)].copy()


def _radius_for_rows(rows: pd.DataFrame, obj_row: pd.Series, track_row: pd.Series) -> np.ndarray:
    radius = rows["radius_m"].astype(float).to_numpy(copy=True)
    fallback_obj = float(obj_row.mean_radius_m) if np.isfinite(float(obj_row.mean_radius_m)) else np.nan
    fallback_track = float(track_row.mean_radius_m) if np.isfinite(float(track_row.mean_radius_m)) else np.nan
    bad = ~np.isfinite(radius) | (radius <= 0)
    if np.isfinite(fallback_obj) and fallback_obj > 0:
        radius[bad] = fallback_obj
    bad = ~np.isfinite(radius) | (radius <= 0)
    if np.isfinite(fallback_track) and fallback_track > 0:
        radius[bad] = fallback_track
    return radius


def _daily_metrics_for_object(rows: pd.DataFrame, obj_row: pd.Series, track_row: pd.Series, *, min_valid_layers: int) -> dict:
    rows = rows.sort_values("depth_index").copy()
    origin = rows.iloc[0]
    lon = rows["longitude"].astype(float).to_numpy()
    lat = rows["latitude"].astype(float).to_numpy()
    x, y = _local_xy_m(lon, lat, float(origin.longitude), float(origin.latitude))
    radius = _radius_for_rows(rows, obj_row, track_row)
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(radius) & (radius > 0)
    kx = np.full(rows.shape[0], np.nan, dtype="f8")
    ky = np.full(rows.shape[0], np.nan, dtype="f8")
    k_mag = np.full(rows.shape[0], np.nan, dtype="f8")
    kx[valid] = x[valid] / radius[valid]
    ky[valid] = y[valid] / radius[valid]
    k_mag[valid] = np.hypot(kx[valid], ky[valid])
    direction = np.degrees(np.arctan2(ky, kx))
    direction[k_mag <= 1e-8] = np.nan
    finite_k = np.isfinite(k_mag)
    n_valid = int(np.sum(finite_k))
    if n_valid < min_valid_layers:
        s_rms = np.nan
        mono = np.nan
        dmean = np.nan
        dmax = np.nan
        note = "insufficient_valid_layers_for_classification"
    else:
        s_rms = float(np.sqrt(np.nanmean(k_mag[finite_k] ** 2)))
        mono = _monotonic_ratio(k_mag)
        dmean, dmax = _direction_turn_metrics(direction, finite_k & (k_mag > 1e-8))
        note = ""
    return {
        "track3d_id": int(track_row.track3d_id),
        "eddy3d_object_id": int(obj_row.eddy3d_object_id),
        "date": f"{pd.Timestamp(obj_row.date):%Y-%m-%d}",
        "polarity": str(track_row.polarity),
        "n_fixed_layers": int(rows.shape[0]),
        "n_valid_layers": n_valid,
        "n_detected_layers": int(rows["center_is_detected"].sum()) if "center_is_detected" in rows else np.nan,
        "n_exact_centers": int((rows["center_method"].astype(str) == "exact").sum()) if "center_method" in rows else np.nan,
        "n_fallback_centers": int((rows["center_method"].astype(str) == "fallback").sum()) if "center_method" in rows else np.nan,
        "n_carry_forward_centers": int((rows["center_method"].astype(str) == "carry_forward").sum()) if "center_method" in rows else np.nan,
        "S_rms": s_rms,
        "S_max": float(np.nanmax(k_mag)) if np.isfinite(k_mag).any() else np.nan,
        "S_p95": float(np.nanpercentile(k_mag[np.isfinite(k_mag)], 95)) if np.isfinite(k_mag).any() else np.nan,
        "mono_ratio": mono,
        "dir_change_mean_deg": dmean,
        "dir_change_max_deg": dmax,
        "mean_radius_m": float(np.nanmean(radius)) if np.isfinite(radius).any() else np.nan,
        "note": note,
    }


def _build_daily_metrics(eligible: pd.DataFrame, objects: pd.DataFrame, centers: pd.DataFrame, *, min_valid_layers: int) -> pd.DataFrame:
    eligible_by_track = {int(row.track3d_id): row for row in eligible.itertuples(index=False)}
    objects = objects[objects["track3d_id"].astype(int).isin(eligible_by_track)].copy()
    center_groups = {int(k): g for k, g in centers.groupby("eddy3d_object_id")}
    rows: list[dict] = []
    for obj in objects.sort_values(["track3d_id", "date", "eddy3d_object_id"]).itertuples(index=False):
        part = center_groups.get(int(obj.eddy3d_object_id))
        if part is None or part.empty:
            rows.append(
                {
                    "track3d_id": int(obj.track3d_id),
                    "eddy3d_object_id": int(obj.eddy3d_object_id),
                    "date": f"{pd.Timestamp(obj.date):%Y-%m-%d}",
                    "polarity": str(obj.polarity),
                    "n_fixed_layers": 0,
                    "n_valid_layers": 0,
                    "S_rms": np.nan,
                    "mono_ratio": np.nan,
                    "dir_change_mean_deg": np.nan,
                    "dir_change_max_deg": np.nan,
                    "note": "missing_completed_centers",
                }
            )
            continue
        rows.append(_daily_metrics_for_object(part, pd.Series(obj._asdict()), pd.Series(eligible_by_track[int(obj.track3d_id)]._asdict()), min_valid_layers=min_valid_layers))
    return pd.DataFrame.from_records(rows)


def _complete_day_metrics_worker(
    config_path: str,
    day_label: str,
    object_records: list[dict],
    layer_records: list[dict],
    track_records: list[dict],
    min_valid_layers: int,
    centers_part_path: str,
    write_completed_centers: bool,
) -> tuple[str, list[dict], int]:
    _, center_rows, _ = _complete_day_worker(config_path, day_label, object_records, layer_records, "centers-only")
    centers = pd.DataFrame.from_records(center_rows)
    if write_completed_centers and not centers.empty:
        write_table_fast(centers, centers_part_path, index=False)
    track_by_id = {int(row["track3d_id"]): pd.Series(row) for row in track_records}
    center_groups = {int(k): g.copy() for k, g in centers.groupby("eddy3d_object_id")} if not centers.empty else {}
    daily_rows: list[dict] = []
    for record in object_records:
        obj_row = pd.Series(record)
        part = center_groups.get(int(obj_row.eddy3d_object_id))
        if part is None or part.empty:
            daily_rows.append(
                {
                    "track3d_id": int(obj_row.track3d_id),
                    "eddy3d_object_id": int(obj_row.eddy3d_object_id),
                    "date": f"{pd.Timestamp(obj_row.date):%Y-%m-%d}",
                    "polarity": str(obj_row.polarity),
                    "n_fixed_layers": 0,
                    "n_valid_layers": 0,
                    "S_rms": np.nan,
                    "mono_ratio": np.nan,
                    "dir_change_mean_deg": np.nan,
                    "dir_change_max_deg": np.nan,
                    "note": "missing_completed_centers",
                }
            )
            continue
        daily_rows.append(_daily_metrics_for_object(part, obj_row, track_by_id[int(obj_row.track3d_id)], min_valid_layers=min_valid_layers))
    return day_label, daily_rows, int(len(center_rows))


def _build_daily_metrics_streaming_completion(
    config_path: str | Path,
    config: dict,
    eligible: pd.DataFrame,
    objects: pd.DataFrame,
    *,
    force: bool,
    workers: int,
    min_valid_layers: int,
    write_completed_centers: bool,
    merge_completed_centers: bool,
) -> pd.DataFrame:
    cat = _catalog_paths(config)
    if write_completed_centers and force:
        shutil.rmtree(cat["centers_parts"], ignore_errors=True)
        if cat["centers"].exists():
            cat["centers"].unlink()
    layers = read_table(Path(config["paths"]["catalog_dir"]) / "layer_observations.parquet")
    layers["date"] = pd.to_datetime(layers["date"])
    objects = objects.copy()
    objects["date"] = pd.to_datetime(objects["date"])
    selected_object_ids = set(objects["eddy3d_object_id"].astype(int))
    selected_layers = layers[layers["eddy3d_object_id"].astype(int).isin(selected_object_ids)].copy()
    layers_by_day = {pd.Timestamp(k): g.copy() for k, g in selected_layers.groupby("date")}
    track_records = eligible.to_dict("records")
    day_jobs = []
    for day, day_objects in objects.sort_values(["date", "track3d_id", "eddy3d_object_id"]).groupby("date", sort=True):
        day_ts = pd.Timestamp(day)
        part_path = cat["centers_parts"] / f"year={day_ts:%Y}" / f"layer_centers_completed_{day_ts:%Y%m%d}.parquet"
        day_jobs.append(
            (
                f"{day_ts:%Y-%m-%d}",
                day_objects.to_dict("records"),
                layers_by_day.get(day_ts, pd.DataFrame()).to_dict("records"),
                str(part_path),
            )
        )

    daily_rows: list[dict] = []
    completed_center_rows = 0
    if not day_jobs and write_completed_centers and cat["centers_parts"].exists():
        centers = read_partitioned_table(cat["centers_parts"])
        return _build_daily_metrics(eligible, objects, centers, min_valid_layers=min_valid_layers)

    if workers <= 1 or len(day_jobs) <= 1:
        iterator = day_jobs
        for day_label, object_records, layer_records, part_path in iterator:
            _day, rows, count = _complete_day_metrics_worker(
                str(config_path),
                day_label,
                object_records,
                layer_records,
                track_records,
                min_valid_layers,
                part_path,
                write_completed_centers,
            )
            daily_rows.extend(rows)
            completed_center_rows += count
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    _complete_day_metrics_worker,
                    str(config_path),
                    day_label,
                    object_records,
                    layer_records,
                    track_records,
                    min_valid_layers,
                    part_path,
                    write_completed_centers,
                ): day_label
                for day_label, object_records, layer_records, part_path in day_jobs
            }
            for future in as_completed(futures):
                day_label = futures[future]
                try:
                    _day, rows, count = future.result()
                except Exception as exc:
                    raise RuntimeError(f"Streaming shape completion failed for {day_label}") from exc
                daily_rows.extend(rows)
                completed_center_rows += count

    if write_completed_centers and merge_completed_centers:
        write_parquet_parts_to_single(discover_parquet_parts(cat["centers_parts"]), cat["centers"])
    print(f"Streaming completed center rows: {completed_center_rows}")
    return pd.DataFrame.from_records(daily_rows)


def _track_metrics(daily: pd.DataFrame, eligible: pd.DataFrame, upright_threshold: float, opts: dict) -> pd.DataFrame:
    eligible_idx = eligible.set_index("track3d_id")
    out_rows: list[dict] = []
    for track_id, part in daily.groupby("track3d_id"):
        track = eligible_idx.loc[int(track_id)]
        valid = part[np.isfinite(part["S_rms"].astype(float))].copy()
        if valid.empty:
            s_rms = mono = dmean = dmax = np.nan
        else:
            s_rms = float(np.nanmedian(valid["S_rms"].astype(float)))
            mono = float(np.nanmedian(valid["mono_ratio"].astype(float)))
            dmean = float(np.nanmedian(valid["dir_change_mean_deg"].astype(float)))
            dmax = float(np.nanpercentile(valid["dir_change_max_deg"].astype(float), 95))
        if not np.isfinite(s_rms):
            shape = "unknown"
        elif s_rms < upright_threshold:
            shape = "upright_like"
        else:
            coherent = np.isfinite(mono) and mono >= float(opts["mono_coherent_threshold"]) and (
                not np.isfinite(dmean) or dmean <= float(opts["dir_mean_coherent_max_deg"])
            )
            complex_ = (np.isfinite(dmax) and dmax >= float(opts["dir_max_complex_min_deg"])) or (
                np.isfinite(mono) and mono <= float(opts["mono_complex_max"])
            )
            if coherent and not complex_:
                shape = "coherent"
            elif complex_ and not coherent:
                shape = "complex"
            elif coherent and complex_:
                shape = "mixed"
            else:
                shape = "transitional"
        out_rows.append(
            {
                "track3d_id": int(track_id),
                "shape_class": shape,
                "polarity": str(track.polarity),
                "start_date": track.start_date,
                "end_date": track.end_date,
                "lifetime_days": int(track.lifetime_days),
                "observation_count": int(track.observation_count),
                "max_layer_count": int(track.max_layer_count),
                "mean_radius_m": float(track.mean_radius_m),
                "n_daily_observations": int(part.shape[0]),
                "n_valid_daily_observations": int(valid.shape[0]),
                "S_rms": s_rms,
                "mono_ratio": mono,
                "dir_change_mean_deg": dmean,
                "dir_change_max_deg_p95": dmax,
                "upright_s_threshold_used": float(upright_threshold),
                "rule_version": RULE_VERSION,
            }
        )
    return pd.DataFrame.from_records(out_rows).sort_values(["shape_class", "track3d_id"])


def classify_shapes(
    config_path: str | Path,
    *,
    complete_missing: bool = False,
    force: bool = False,
    lifetime_min_days: int = 56,
    radius_min_m: float = 50_000.0,
    min_valid_layers: int = 6,
    completion_output_mode: str = "centers-only",
    workers: int = 1,
    start: str | None = None,
    end: str | None = None,
    output_name: str = "shape_classification",
    write_completed_centers: bool = True,
    merge_completed_centers: bool = False,
) -> None:
    config = load_config(config_path)
    ensure_dirs(config)
    paths = _paths(config, output_name)
    paths["root"].mkdir(parents=True, exist_ok=True)
    completion_output_mode = str(completion_output_mode).lower().strip()
    if completion_output_mode not in {"centers-only", "centers-and-contours"}:
        raise ValueError("--completion-output-mode must be centers-only or centers-and-contours")
    workers = max(1, int(workers))
    cat = _catalog_paths(config)
    tracks = read_table(cat["tracks"])
    objects = read_table(cat["objects"])
    objects["date"] = pd.to_datetime(objects["date"])
    start_ts = pd.Timestamp(start) if start else None
    end_ts = pd.Timestamp(end) if end else None
    if start_ts is not None:
        objects = objects[objects["date"] >= start_ts].copy()
    if end_ts is not None:
        objects = objects[objects["date"] <= end_ts].copy()
    if objects.empty:
        raise RuntimeError("No vertical objects found in requested date range.")

    eligible = _eligible_tracks(tracks, lifetime_min_days=lifetime_min_days, radius_min_m=radius_min_m)
    eligible_objects = objects[objects["track3d_id"].astype(int).isin(set(eligible["track3d_id"].astype(int)))].copy()
    eligible_track_ids_in_window = set(eligible_objects["track3d_id"].astype(int))
    eligible = eligible[eligible["track3d_id"].astype(int).isin(eligible_track_ids_in_window)].copy()
    write_table(eligible, paths["eligible"], index=False)
    if eligible.empty:
        raise RuntimeError("No tracks passed the shape-classification eligibility filter.")

    use_streaming_fast_completion = (
        complete_missing
        and completion_output_mode == "centers-only"
        and is_streaming_source_configured(config)
        and not bool(config.get("conversion", {}).get("use_input_daily", True))
    )
    if use_streaming_fast_completion:
        daily = _build_daily_metrics_streaming_completion(
            config_path,
            config,
            eligible,
            eligible_objects,
            force=force,
            workers=workers,
            min_valid_layers=min_valid_layers,
            write_completed_centers=write_completed_centers,
            merge_completed_centers=merge_completed_centers,
        )
    else:
        centers = _ensure_completed_centers(
            config_path,
            config,
            eligible,
            eligible_objects,
            complete_missing=complete_missing,
            force=force,
            completion_output_mode=completion_output_mode,
            workers=workers,
        )
        daily = _build_daily_metrics(eligible, eligible_objects, centers, min_valid_layers=min_valid_layers)
    valid_s = daily["S_rms"].astype(float)
    valid_s = valid_s[np.isfinite(valid_s)]
    if valid_s.size >= 10:
        upright_threshold = float(np.nanquantile(valid_s, 0.2))
        upright_mode = "eligible_daily_quantile"
    else:
        upright_threshold = 0.12
        upright_mode = "fixed_fallback"

    rule_opts = {
        "mono_coherent_threshold": 0.72,
        "dir_mean_coherent_max_deg": 35.0,
        "dir_max_complex_min_deg": 100.0,
        "mono_complex_max": 0.55,
    }
    shape_tracks = _track_metrics(daily, eligible, upright_threshold, rule_opts)
    write_table(daily, paths["daily"], index=False)
    write_table(shape_tracks, paths["tracks"], index=False)

    by_shape = paths["by_shape"]
    by_shape.mkdir(parents=True, exist_ok=True)
    for shape in SHAPE_CLASSES:
        folder = by_shape / shape
        folder.mkdir(parents=True, exist_ok=True)
        shape_tracks[shape_tracks["shape_class"] == shape].to_csv(folder / "tracks.csv", index=False)

    thresholds = {
        "rule_version": RULE_VERSION,
        "date_window": {
            "start": str(start_ts.date()) if start_ts is not None else None,
            "end": str(end_ts.date()) if end_ts is not None else None,
            "output_name": output_name,
        },
        "eligibility": {
            "lifetime_days_cmp": ">",
            "lifetime_days_min": int(lifetime_min_days),
            "mean_radius_m_cmp": ">",
            "mean_radius_m_min": float(radius_min_m),
            "radius_threshold_source": "ME_LIUTEX build_track_all_cols_from_meta_nc effective_radius > 50000 m",
        },
        "shape_classes": SHAPE_CLASSES,
        "upright_mode": upright_mode,
        "upright_s_threshold_used": float(upright_threshold),
        "upright_s_quantile": 0.2,
        "min_valid_layers": int(min_valid_layers),
        "completion_output_mode": completion_output_mode,
        "completion_workers": int(workers),
        "write_completed_centers": bool(write_completed_centers),
        "merge_completed_centers": bool(merge_completed_centers),
        "mono_coherent_threshold": 0.72,
        "dir_mean_coherent_max_deg": 35.0,
        "dir_max_complex_min_deg": 100.0,
        "mono_complex_max": 0.55,
        "n_eligible_tracks": int(eligible.shape[0]),
        "n_daily_metric_rows": int(daily.shape[0]),
        "shape_counts": {str(k): int(v) for k, v in shape_tracks["shape_class"].value_counts().sort_index().items()},
    }
    with paths["thresholds"].open("w", encoding="utf-8") as handle:
        yaml.safe_dump(thresholds, handle, allow_unicode=True, sort_keys=False)

    print(f"Eligible tracks: {paths['eligible']}")
    print(f"Daily shape metrics: {paths['daily']}")
    print(f"Track shape classes: {paths['tracks']}")
    print(f"Thresholds: {paths['thresholds']}")
    print(shape_tracks["shape_class"].value_counts().sort_index().to_string())


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify long-lived 3D eddy tracks by tilt shape only.")
    parser.add_argument("--config", default="config/config_3d.yaml")
    parser.add_argument("--complete-missing", dest="complete_missing", action="store_true")
    parser.add_argument("--no-complete-missing", dest="complete_missing", action="store_false")
    parser.set_defaults(complete_missing=False)
    parser.add_argument("--force", action="store_true", help="Rewrite classification outputs; with --complete-missing, recompute eligible completed centers.")
    parser.add_argument("--lifetime-min-days", type=int, default=56)
    parser.add_argument("--radius-min-m", type=float, default=50_000.0)
    parser.add_argument("--min-valid-layers", type=int, default=6)
    parser.add_argument("--completion-output-mode", choices=["centers-only", "centers-and-contours"], default="centers-only")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--output-name", default="shape_classification")
    parser.add_argument("--no-write-completed-centers", dest="write_completed_centers", action="store_false")
    parser.set_defaults(write_completed_centers=True)
    parser.add_argument("--merge-completed-centers", action="store_true")
    args = parser.parse_args()
    classify_shapes(
        args.config,
        complete_missing=args.complete_missing,
        force=args.force,
        lifetime_min_days=args.lifetime_min_days,
        radius_min_m=args.radius_min_m,
        min_valid_layers=args.min_valid_layers,
        completion_output_mode=args.completion_output_mode,
        workers=args.workers,
        start=args.start,
        end=args.end,
        output_name=args.output_name,
        write_completed_centers=args.write_completed_centers,
        merge_completed_centers=args.merge_completed_centers,
    )


if __name__ == "__main__":
    main()
