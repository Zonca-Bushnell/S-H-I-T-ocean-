from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable

import netCDF4
import numpy as np
import pandas as pd
from scipy.ndimage import minimum_filter

from .table_io import write_table
from .velocity3d_core import (
    LayerDetection,
    _closed_contour,
    _window_pixels,
    detect_surface_sla_fallback,
    detect_velocity_layer,
    grid_spacing_m,
    haversine_km,
    make_contour_context,
    polygon_area_m2,
)


VARIABLES = {
    "lon": "longitude",
    "lat": "latitude",
    "depth": "depth",
    "time": "time",
    "u": "uo_glor",
    "v": "vo_glor",
    "adt": "zos_glor",
}

DEFAULT_ID_CONFIG = {
    "surface_depth_index": 0,
    "surface_pixel_limit": [5, 2000],
    "core_window_km": 80.0,
    "min_core_reversal_speed": 0.02,
    "max_core_speed_percentile": 15.0,
    "min_closed_contour_pixels": 12,
    "max_closed_contour_pixels": 20000,
    "min_core_distance_km": 40.0,
    "max_candidates_per_layer": 80,
    "contour_levels": 16,
}


@dataclass(frozen=True)
class SourceDay:
    day: date
    path: str
    time_index: int


def parse_day(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def iter_days(start: date, end: date) -> Iterable[date]:
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def decode_time_days(ds: netCDF4.Dataset) -> dict[date, int]:
    time_var = ds.variables[VARIABLES["time"]]
    values = netCDF4.num2date(
        time_var[:],
        time_var.units,
        getattr(time_var, "calendar", "standard"),
        only_use_cftime_datetimes=False,
        only_use_python_datetimes=True,
    )
    return {value.date(): i for i, value in enumerate(values)}


def build_source_index(input_dir: Path, days: list[date]) -> dict[date, SourceDay]:
    by_year = {day.year for day in days}
    out: dict[date, SourceDay] = {}
    for year in sorted(by_year):
        path = input_dir / f"global_phy_{year}.nc"
        if not path.exists():
            raise FileNotFoundError(path)
        with netCDF4.Dataset(path) as ds:
            indices = decode_time_days(ds)
        for day in days:
            if day.year == year:
                if day not in indices:
                    raise FileNotFoundError(f"{day.isoformat()} not present in {path}")
                out[day] = SourceDay(day=day, path=str(path), time_index=int(indices[day]))
    return out


def selected_depth_indices(path: str, max_depth_m: float) -> list[int]:
    with netCDF4.Dataset(path) as ds:
        depth = np.asarray(ds.variables[VARIABLES["depth"]][:], dtype="f8")
    indices = np.where(depth <= max_depth_m)[0]
    if indices.size == 0:
        raise ValueError(f"No depth <= {max_depth_m}")
    return [int(i) for i in indices]


def filled_f8(values) -> np.ndarray:
    return np.ma.filled(values, np.nan).astype("f8", copy=False)


def records_from_detection(det: LayerDetection) -> tuple[dict, list[dict]]:
    record = {
        "layer_detection_id": det.detection_id,
        "date": det.date,
        "depth_m": det.depth_m,
        "depth_index": det.depth_index,
        "polarity": det.polarity,
        "longitude": det.longitude,
        "latitude": det.latitude,
        "core_speed": det.core_speed,
        "vorticity": det.vorticity,
        "area_m2": det.area_m2,
        "radius_m": det.radius_m,
        "method": det.method,
        "reversal_passed": det.reversal_passed,
    }
    contours = [
        {
            "layer_detection_id": det.detection_id,
            "date": det.date,
            "depth_m": det.depth_m,
            "depth_index": det.depth_index,
            "point_index": i,
            "longitude": float(x),
            "latitude": float(y),
        }
        for i, (x, y) in enumerate(zip(det.contour_lon, det.contour_lat))
    ]
    return record, contours


def relative_vorticity_gpu(lon: np.ndarray, lat: np.ndarray, u, v, cp):
    dx_by_lat, dy = grid_spacing_m(lon, lat)
    dx = cp.asarray(dx_by_lat, dtype=cp.float64)[:, None]
    dvdx = cp.gradient(v, axis=1) / dx
    dudy = cp.gradient(u, axis=0) / float(dy)
    return dvdx - dudy


def pseudo_streamfunction_gpu(lon: np.ndarray, lat: np.ndarray, u, v, cp):
    dx_by_lat, dy = grid_spacing_m(lon, lat)
    dx = cp.asarray(dx_by_lat, dtype=cp.float64)[:, None]
    u0 = cp.nan_to_num(u, nan=0.0, posinf=0.0, neginf=0.0)
    v0 = cp.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
    psi_x = cp.cumsum(v0 * dx, axis=1)
    psi_y = cp.cumsum(-u0 * float(dy), axis=0)
    psi_x -= cp.nanmean(psi_x)
    psi_y -= cp.nanmean(psi_y)
    return 0.5 * (psi_x + psi_y)


def has_core_velocity_reversal_fast(u: np.ndarray, v: np.ndarray, j: int, i: int, rx: int, ry: int, min_speed: float) -> bool:
    west_v = np.nanmean(v[max(j - ry, 0) : min(j + ry + 1, v.shape[0]), max(i - rx, 0) : i])
    east_v = np.nanmean(v[max(j - ry, 0) : min(j + ry + 1, v.shape[0]), i + 1 : min(i + rx + 1, v.shape[1])])
    south_u = np.nanmean(u[max(j - ry, 0) : j, max(i - rx, 0) : min(i + rx + 1, u.shape[1])])
    north_u = np.nanmean(u[j + 1 : min(j + ry + 1, u.shape[0]), max(i - rx, 0) : min(i + rx + 1, u.shape[1])])
    vals = np.array([west_v, east_v, south_u, north_u], dtype="f8")
    if not np.all(np.isfinite(vals)):
        return False
    return (
        west_v * east_v < 0
        and south_u * north_u < 0
        and min(abs(west_v), abs(east_v), abs(south_u), abs(north_u)) >= min_speed
    )


def detect_velocity_layer_gpu(
    lon: np.ndarray,
    lat: np.ndarray,
    depth_m: float,
    depth_index: int,
    u: np.ndarray,
    v: np.ndarray,
    date_value: str,
    cfg: dict,
) -> list[LayerDetection]:
    import cupy as cp

    finite = np.isfinite(u) & np.isfinite(v)
    if finite.sum() < int(cfg["min_closed_contour_pixels"]):
        return []

    speed = np.empty(u.shape, dtype="f8")
    np.hypot(u, v, out=speed)
    speed[~finite] = np.nan
    threshold = float(np.nanpercentile(speed, float(cfg["max_core_speed_percentile"])))
    rx, ry = _window_pixels(lon, lat, float(cfg["core_window_km"]))
    min_speed = minimum_filter(np.nan_to_num(speed, nan=np.inf), size=(2 * ry + 1, 2 * rx + 1), mode="nearest")
    candidates = np.argwhere((speed <= threshold) & (speed == min_speed) & finite)
    if candidates.size == 0:
        return []

    u_gpu = cp.asarray(u, dtype=cp.float64)
    v_gpu = cp.asarray(v, dtype=cp.float64)
    zeta = cp.asnumpy(relative_vorticity_gpu(lon, lat, cp.nan_to_num(u_gpu, nan=0.0), cp.nan_to_num(v_gpu, nan=0.0), cp))
    psi = cp.asnumpy(pseudo_streamfunction_gpu(lon, lat, u_gpu, v_gpu, cp))
    contour_context = make_contour_context(psi, finite)
    order = np.argsort(speed[candidates[:, 0], candidates[:, 1]])

    detections: list[LayerDetection] = []
    used_centers: list[tuple[float, float]] = []
    for cand_index in order[: int(cfg["max_candidates_per_layer"]) * 4]:
        j, i = [int(v_) for v_ in candidates[cand_index]]
        if j < ry or i < rx or j >= u.shape[0] - ry or i >= u.shape[1] - rx:
            continue
        lon_i = float(lon[i])
        lat_j = float(lat[j])
        if any(haversine_km(lon_i, lat_j, old_lon, old_lat) < float(cfg["min_core_distance_km"]) for old_lon, old_lat in used_centers):
            continue
        if not has_core_velocity_reversal_fast(u, v, j, i, rx, ry, float(cfg["min_core_reversal_speed"])):
            continue
        z = float(zeta[j, i])
        if not np.isfinite(z) or z == 0:
            continue
        contour = _closed_contour(
            lon,
            lat,
            psi,
            finite,
            j,
            i,
            int(cfg["min_closed_contour_pixels"]),
            int(cfg["max_closed_contour_pixels"]),
            int(cfg["contour_levels"]),
            contour_context=contour_context,
        )
        if contour is None:
            continue
        contour_lon, contour_lat = contour
        area = polygon_area_m2(contour_lon, contour_lat)
        detections.append(
            LayerDetection(
                detection_id=len(detections),
                date=str(date_value),
                depth_m=float(depth_m),
                depth_index=int(depth_index),
                polarity="cyclonic" if z > 0 else "anticyclonic",
                longitude=lon_i,
                latitude=lat_j,
                core_speed=float(speed[j, i]),
                vorticity=z,
                contour_lon=contour_lon,
                contour_lat=contour_lat,
                area_m2=area,
                radius_m=float(np.sqrt(area / np.pi)) if area > 0 else 0.0,
                method="velocity_core_reversal_gpu_prefilter",
                reversal_passed=True,
            )
        )
        used_centers.append((lon_i, lat_j))
        if len(detections) >= int(cfg["max_candidates_per_layer"]):
            break
    return detections


def run_layer_task(task: tuple[str, int, int, str, dict]) -> tuple[str, int, list[dict], list[dict], float]:
    path, time_index, depth_index, day_iso, cfg = task
    started = time.perf_counter()
    with netCDF4.Dataset(path) as ds:
        lon = np.asarray(ds.variables[VARIABLES["lon"]][:], dtype="f8")
        lat = np.asarray(ds.variables[VARIABLES["lat"]][:], dtype="f8")
        depth = np.asarray(ds.variables[VARIABLES["depth"]][:], dtype="f8")
        if int(depth_index) == int(cfg["surface_depth_index"]):
            adt = filled_f8(ds.variables[VARIABLES["adt"]][time_index, :, :])
            detections = detect_surface_sla_fallback(lon, lat, adt, float(depth[depth_index]), int(depth_index), day_iso, **cfg)
        else:
            u = filled_f8(ds.variables[VARIABLES["u"]][time_index, depth_index, :, :])
            v = filled_f8(ds.variables[VARIABLES["v"]][time_index, depth_index, :, :])
            if cfg.get("mode") == "gpu":
                detections = detect_velocity_layer_gpu(lon, lat, float(depth[depth_index]), int(depth_index), u, v, day_iso, cfg)
            else:
                detections = detect_velocity_layer(lon, lat, float(depth[depth_index]), int(depth_index), u, v, day_iso, **cfg)

    records: list[dict] = []
    contours: list[dict] = []
    for local_id, det in enumerate(detections):
        det = det.__class__(**{**det.__dict__, "detection_id": int(depth_index) * 100000 + local_id})
        record, rows = records_from_detection(det)
        records.append(record)
        contours.extend(rows)
    return day_iso, int(depth_index), records, contours, time.perf_counter() - started


def write_day_outputs(output_root: Path, scenario: str, day_iso: str, records: list[dict], contours: list[dict]) -> None:
    day = datetime.strptime(day_iso, "%Y-%m-%d").date()
    layer_dir = output_root / scenario / "layers"
    write_table(pd.DataFrame.from_records(records), layer_dir / f"layer_observations_{day:%Y%m%d}.parquet", index=False)
    write_table(pd.DataFrame.from_records(contours), layer_dir / f"contours_{day:%Y%m%d}.parquet", index=False)


def run_scenario(
    scenario: str,
    mode: str,
    workers: int,
    source_index: dict[date, SourceDay],
    depth_indices: list[int],
    cfg: dict,
    output_root: Path,
) -> dict:
    scenario_root = output_root / scenario
    if scenario_root.exists():
        shutil.rmtree(scenario_root)
    (scenario_root / "layers").mkdir(parents=True, exist_ok=True)

    cfg = dict(cfg)
    cfg["mode"] = mode
    tasks = [
        (source_index[day].path, source_index[day].time_index, depth_index, day.isoformat(), cfg)
        for day in sorted(source_index)
        for depth_index in depth_indices
    ]
    started = time.perf_counter()
    by_day_records: dict[str, list[dict]] = {day.isoformat(): [] for day in sorted(source_index)}
    by_day_contours: dict[str, list[dict]] = {day.isoformat(): [] for day in sorted(source_index)}
    layer_seconds = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(run_layer_task, task) for task in tasks]
        for future in as_completed(futures):
            day_iso, _depth_index, records, contours, seconds = future.result()
            by_day_records[day_iso].extend(records)
            by_day_contours[day_iso].extend(contours)
            layer_seconds.append(float(seconds))

    for day_iso in sorted(by_day_records):
        write_day_outputs(output_root, scenario, day_iso, by_day_records[day_iso], by_day_contours[day_iso])

    elapsed = time.perf_counter() - started
    days = len(source_index)
    detections = sum(len(v) for v in by_day_records.values())
    contour_points = sum(len(v) for v in by_day_contours.values())
    return {
        "scenario": scenario,
        "mode": mode,
        "workers": workers,
        "days": days,
        "depth_layers": len(depth_indices),
        "tasks": len(tasks),
        "elapsed_seconds": elapsed,
        "days_per_hour": days / elapsed * 3600.0 if elapsed > 0 else 0.0,
        "tasks_per_hour": len(tasks) / elapsed * 3600.0 if elapsed > 0 else 0.0,
        "detections": detections,
        "contour_points": contour_points,
        "mean_layer_seconds": float(np.mean(layer_seconds)) if layer_seconds else 0.0,
        "p95_layer_seconds": float(np.percentile(layer_seconds, 95)) if layer_seconds else 0.0,
    }


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark streaming Kuroshio layer identification without input_daily files.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--max-depth-m", type=float, default=2000.0)
    parser.add_argument("--cpu-workers", default="32,56,96,112,160")
    parser.add_argument("--gpu-workers", default="1,4,8")
    parser.add_argument("--skip-gpu", action="store_true")
    args = parser.parse_args()

    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(name, "1")

    days = list(iter_days(parse_day(args.start), parse_day(args.end)))
    source_index = build_source_index(args.input_dir, days)
    first_path = next(iter(source_index.values())).path
    depth_indices = selected_depth_indices(first_path, args.max_depth_m)
    args.output_root.mkdir(parents=True, exist_ok=True)

    cfg = dict(DEFAULT_ID_CONFIG)
    results = []
    for workers in parse_int_list(args.cpu_workers):
        scenario = f"cpu_w{workers}"
        result = run_scenario(scenario, "cpu", workers, source_index, depth_indices, cfg, args.output_root)
        results.append(result)
        print(json.dumps(result, sort_keys=True), flush=True)

    if not args.skip_gpu:
        for workers in parse_int_list(args.gpu_workers):
            scenario = f"gpu_w{workers}"
            result = run_scenario(scenario, "gpu", workers, source_index, depth_indices, cfg, args.output_root)
            results.append(result)
            print(json.dumps(result, sort_keys=True), flush=True)

    csv_path = args.output_root / "benchmark_summary.csv"
    json_path = args.output_root / "benchmark_summary.json"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0].keys()) if results else [])
        writer.writeheader()
        writer.writerows(results)
    json_path.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
