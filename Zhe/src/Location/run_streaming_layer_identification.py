from __future__ import annotations

import argparse
import ctypes
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from multiprocessing import shared_memory
from pathlib import Path
import os
import sys

import netCDF4
import numpy as np
import pandas as pd
from tqdm import tqdm

from .common import ensure_dirs, iter_days, load_config, parse_ymd
from .run_3d_layer_identification import layer_output_paths
from .streaming_cmems import SourceDay, YearCache, build_source_index, load_year_cache, read_day_data, read_layer_arrays, selected_depth_indices, spatial_window, variable_names
from .table_io import table_exists, write_table
from .velocity3d_core import LayerDetection, detect_surface_sla_fallback, detect_velocity_layer


ROOT = Path(__file__).resolve().parents[2]
PYEDDY_CANDIDATES = (
    ROOT / "vendor" / "py-eddy-tracker" / "src",
    Path(__file__).resolve().parents[1] / "vendor" / "py-eddy-tracker" / "src",
)
for PYEDDY_SRC in PYEDDY_CANDIDATES:
    if PYEDDY_SRC.exists() and str(PYEDDY_SRC) not in sys.path:
        sys.path.insert(0, str(PYEDDY_SRC))
        break


_YEAR_CACHE: YearCache | None = None
_SHARED_YEAR_CACHE: dict | None = None


def _valid_outputs(paths: tuple[Path, Path]) -> bool:
    return all(table_exists(path) for path in paths)


def _records_from_detection(det: LayerDetection) -> tuple[dict, list[dict]]:
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
    contour_rows = [
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
    return record, contour_rows


def _detection_kwargs(config: dict) -> dict:
    kwargs = dict(config.get("identification", {}))
    kwargs.pop("workers", None)
    kwargs.pop("overwrite_existing", None)
    return kwargs


def _surface_pyeddy_kwargs(config: dict) -> dict:
    id_cfg = config["identification"]
    return dict(
        step=float(id_cfg.get("surface_isoline_step_m", 0.002)),
        shape_error=float(id_cfg.get("surface_fit_errmax_percent", 55)),
        pixel_limit=tuple(int(v) for v in id_cfg.get("surface_pixel_limit", [5, 2000])),
        force_height_unit="m",
        force_speed_unit=None,
        nb_step_to_be_mle=0,
        nb_step_min=int(id_cfg.get("surface_nb_step_min", 2)),
        sampling=int(id_cfg.get("surface_sampling", 50)),
        sampling_method=id_cfg.get("surface_sampling_method", "visvalingam"),
    )


def _ensure_conda_libstdcxx() -> None:
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if not conda_prefix:
        return
    for name in ("libstdc++.so.6", "libgcc_s.so.1"):
        path = Path(conda_prefix) / "lib" / name
        if path.exists():
            try:
                ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
            except OSError:
                pass


def _surface_from_pyeddy_file(
    config: dict,
    nc_path: str | Path,
    source_day: SourceDay,
    depth_m: float,
    depth_index: int,
    *,
    lon_name: str,
    lat_name: str,
    height_name: str,
    indexs: dict | None,
) -> list[LayerDetection]:
    _ensure_conda_libstdcxx()
    from py_eddy_tracker.appli.grid import identification

    id_cfg = config["identification"]
    pyeddy_date = datetime.combine(source_day.day, datetime.min.time())
    anticyclonic, cyclonic = identification(
        str(nc_path),
        lon_name,
        lat_name,
        pyeddy_date,
        height_name,
        "None",
        "None",
        unregular=False,
        cut_wavelength=float(id_cfg.get("surface_cut_wavelength_km", 500)),
        cut_highwavelength=float(id_cfg.get("surface_cut_highwavelength_km", 0)),
        lat_max=float(id_cfg.get("surface_lat_max", 80)),
        filter_order=int(id_cfg.get("surface_filter_order", 3)),
        indexs=indexs,
        **_surface_pyeddy_kwargs(config),
    )

    date_label = source_day.day.isoformat()
    detections: list[LayerDetection] = []
    for polarity, eddies in (("anticyclonic", anticyclonic), ("cyclonic", cyclonic)):
        for n in range(len(eddies)):
            detections.append(
                LayerDetection(
                    detection_id=len(detections),
                    date=date_label,
                    depth_m=float(depth_m),
                    depth_index=int(depth_index),
                    polarity=polarity,
                    longitude=float(eddies.lon[n]),
                    latitude=float(eddies.lat[n]),
                    core_speed=float(eddies.speed_average[n]) if "speed_average" in eddies.fields else np.nan,
                    vorticity=np.nan,
                    contour_lon=np.asarray(eddies.contour_lon_e[n], dtype="f4"),
                    contour_lat=np.asarray(eddies.contour_lat_e[n], dtype="f4"),
                    area_m2=float(eddies.effective_area[n]) if "effective_area" in eddies.fields else np.nan,
                    radius_m=float(eddies.radius_e[n]) if "radius_e" in eddies.fields else np.nan,
                    method="sla_surface",
                    reversal_passed=False,
                )
            )
    return detections


def _write_surface_tmp(config: dict, source_day: SourceDay, lon: np.ndarray, lat: np.ndarray, adt: np.ndarray) -> Path:
    temp_root = Path(config.get("paths", {}).get("temp_dir", "/tmp")) / "pyeddy_surface_tmp"
    temp_root.mkdir(parents=True, exist_ok=True)
    path = temp_root / f"surface_{source_day.day:%Y%m%d}_{os.getpid()}.nc"
    fill_value = np.float32(1.0e20)
    with netCDF4.Dataset(path, "w", format="NETCDF4") as ds:
        ds.createDimension("longitude", int(lon.size))
        ds.createDimension("latitude", int(lat.size))
        lon_var = ds.createVariable("longitude", "f8", ("longitude",))
        lat_var = ds.createVariable("latitude", "f8", ("latitude",))
        adt_var = ds.createVariable("adt", "f4", ("latitude", "longitude"), fill_value=fill_value)
        lon_var.units = "degrees_east"
        lat_var.units = "degrees_north"
        adt_var.units = "m"
        ds.date = source_day.day.isoformat()
        lon_var[:] = lon
        lat_var[:] = lat
        arr = np.asarray(adt, dtype="f4")
        adt_var[:, :] = np.where(np.isfinite(arr), arr, fill_value)
    return path


def _surface_from_pyeddy_streaming(
    config: dict,
    source_day: SourceDay,
    arrays: dict[str, np.ndarray | float],
    depth_index: int,
) -> list[LayerDetection]:
    names = variable_names(config)
    window = spatial_window(config, source_day.path)
    indexs = {
        names["time"]: int(source_day.time_index),
        names["lat"]: window.lat_slice,
        names["lon"]: window.lon_slice,
    }
    try:
        return _surface_from_pyeddy_file(
            config,
            source_day.path,
            source_day,
            float(arrays["depth_m"]),
            int(depth_index),
            lon_name=names["lon"],
            lat_name=names["lat"],
            height_name=names["adt"],
            indexs=indexs,
        )
    except Exception as annual_exc:
        tmp_path: Path | None = None
        try:
            tmp_path = _write_surface_tmp(
                config,
                source_day,
                np.asarray(arrays["lon"], dtype="f8"),
                np.asarray(arrays["lat"], dtype="f8"),
                np.asarray(arrays["adt"], dtype="f8"),
            )
            print(
                f"[surface pyeddy tmp] annual direct failed for {source_day.day}: "
                f"{type(annual_exc).__name__}: {annual_exc}",
                flush=True,
            )
            return _surface_from_pyeddy_file(
                config,
                tmp_path,
                source_day,
                float(arrays["depth_m"]),
                int(depth_index),
                lon_name="longitude",
                lat_name="latitude",
                height_name="adt",
                indexs=None,
            )
        finally:
            if tmp_path is not None:
                tmp_path.unlink(missing_ok=True)


def _run_layer_task(task: tuple[dict, SourceDay, int, bool]) -> tuple[str, int, list[dict], list[dict]]:
    config, source_day, depth_index, include_contours = task
    id_cfg = config.get("identification", {})
    arrays = read_layer_arrays(config, source_day, depth_index)
    kwargs = _detection_kwargs(config)
    day_iso = source_day.day.isoformat()
    if int(depth_index) == int(id_cfg.get("surface_depth_index", 0)):
        if str(id_cfg.get("surface_method", "sla_pyeddy")).lower() in {"sla_pyeddy", "pyeddy", "py-eddy-tracker"}:
            try:
                detections = _surface_from_pyeddy_streaming(config, source_day, arrays, depth_index)
            except Exception as exc:
                print(f"[surface fallback] py-eddy-tracker failed for {day_iso}: {type(exc).__name__}: {exc}", flush=True)
                detections = detect_surface_sla_fallback(
                    arrays["lon"],
                    arrays["lat"],
                    arrays["adt"],
                    float(arrays["depth_m"]),
                    int(depth_index),
                    day_iso,
                    **kwargs,
                )
        else:
            detections = detect_surface_sla_fallback(
                arrays["lon"],
                arrays["lat"],
                arrays["adt"],
                float(arrays["depth_m"]),
                int(depth_index),
                day_iso,
                **kwargs,
            )
    else:
        detections = detect_velocity_layer(
            arrays["lon"],
            arrays["lat"],
            float(arrays["depth_m"]),
            int(depth_index),
            arrays["u"],
            arrays["v"],
            day_iso,
            **kwargs,
        )

    records: list[dict] = []
    contours: list[dict] = []
    for local_id, det in enumerate(detections):
        det = det.__class__(**{**det.__dict__, "detection_id": int(depth_index) * 100000 + int(local_id)})
        record, rows = _records_from_detection(det)
        records.append(record)
        if include_contours:
            contours.extend(rows)
    return day_iso, int(depth_index), records, contours


def set_year_cache(cache: YearCache | None) -> None:
    global _YEAR_CACHE
    _YEAR_CACHE = cache


def cache_mode(config: dict) -> str:
    return str(config.get("identification", {}).get("cache_mode", "per_layer_netcdf")).lower()


def is_year_ram_cache_enabled(config: dict) -> bool:
    return cache_mode(config) in {"year_ram", "year-cache", "year_cache", "ram_year"}


def is_year_shared_cache_enabled(config: dict) -> bool:
    return cache_mode(config) in {"year_shared", "shared_year", "year-shared"}


def load_source_year_cache(config: dict, source_day: SourceDay) -> YearCache:
    return load_year_cache(config, source_day.path, year=source_day.day.year)


@dataclass
class SharedYearCacheOwner:
    payload: dict
    shms: list[shared_memory.SharedMemory]

    def close(self) -> None:
        for shm in self.shms:
            try:
                shm.close()
            except FileNotFoundError:
                pass

    def unlink(self) -> None:
        for shm in self.shms:
            try:
                shm.unlink()
            except FileNotFoundError:
                pass

    def cleanup(self) -> None:
        self.close()
        self.unlink()


def create_shared_year_cache(config: dict, source_day: SourceDay) -> SharedYearCacheOwner:
    cache = load_source_year_cache(config, source_day)
    payload = {
        "year": cache.year,
        "path": cache.path,
        "day_to_time_index": cache.day_to_time_index,
        "lon": cache.lon,
        "lat": cache.lat,
        "depth": cache.depth,
        "depth_indices": cache.depth_indices,
        "arrays": {},
    }
    shms: list[shared_memory.SharedMemory] = []
    for key, arr in (("u_all", cache.u_all), ("v_all", cache.v_all), ("adt_all", cache.adt_all)):
        arr = np.ascontiguousarray(arr)
        shm = shared_memory.SharedMemory(create=True, size=arr.nbytes)
        view = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)
        view[:] = arr
        payload["arrays"][key] = {
            "name": shm.name,
            "shape": arr.shape,
            "dtype": str(arr.dtype),
        }
        shms.append(shm)
    return SharedYearCacheOwner(payload=payload, shms=shms)


def init_shared_year_cache(payload: dict) -> None:
    global _SHARED_YEAR_CACHE
    attached_shms: list[shared_memory.SharedMemory] = []
    arrays = {}
    for key, meta in payload["arrays"].items():
        shm = shared_memory.SharedMemory(name=meta["name"])
        attached_shms.append(shm)
        arrays[key] = np.ndarray(tuple(meta["shape"]), dtype=np.dtype(meta["dtype"]), buffer=shm.buf)
    _SHARED_YEAR_CACHE = {
        "year": payload["year"],
        "path": payload["path"],
        "day_to_time_index": payload["day_to_time_index"],
        "lon": payload["lon"],
        "lat": payload["lat"],
        "depth": payload["depth"],
        "depth_indices": payload["depth_indices"],
        "arrays": arrays,
        "shms": attached_shms,
    }


def _cached_arrays(config: dict, source_day: SourceDay, depth_index: int) -> dict[str, np.ndarray | float]:
    if _YEAR_CACHE is None:
        raise RuntimeError("Year cache has not been initialized in this worker.")
    cache = _YEAR_CACHE
    time_index = cache.day_to_time_index[source_day.day]
    try:
        depth_pos = cache.depth_indices.index(int(depth_index))
    except ValueError as exc:
        raise IndexError(f"depth_index {depth_index} is not present in year cache.") from exc
    if int(depth_index) == int(config.get("identification", {}).get("surface_depth_index", 0)):
        return {
            "lon": cache.lon,
            "lat": cache.lat,
            "depth_m": float(cache.depth[depth_pos]),
            "adt": np.asarray(cache.adt_all[time_index], dtype="f8"),
        }
    return {
        "lon": cache.lon,
        "lat": cache.lat,
        "depth_m": float(cache.depth[depth_pos]),
        "u": np.asarray(cache.u_all[time_index, depth_pos], dtype="f8"),
        "v": np.asarray(cache.v_all[time_index, depth_pos], dtype="f8"),
    }


def _shared_arrays(config: dict, source_day: SourceDay, depth_index: int) -> dict[str, np.ndarray | float]:
    if _SHARED_YEAR_CACHE is None:
        raise RuntimeError("Shared year cache has not been initialized in this worker.")
    cache = _SHARED_YEAR_CACHE
    time_index = cache["day_to_time_index"][source_day.day]
    try:
        depth_pos = cache["depth_indices"].index(int(depth_index))
    except ValueError as exc:
        raise IndexError(f"depth_index {depth_index} is not present in shared year cache.") from exc
    if int(depth_index) == int(config.get("identification", {}).get("surface_depth_index", 0)):
        return {
            "lon": cache["lon"],
            "lat": cache["lat"],
            "depth_m": float(cache["depth"][depth_pos]),
            "adt": np.asarray(cache["arrays"]["adt_all"][time_index], dtype="f8"),
        }
    return {
        "lon": cache["lon"],
        "lat": cache["lat"],
        "depth_m": float(cache["depth"][depth_pos]),
        "u": np.asarray(cache["arrays"]["u_all"][time_index, depth_pos], dtype="f8"),
        "v": np.asarray(cache["arrays"]["v_all"][time_index, depth_pos], dtype="f8"),
    }


def _run_cached_layer_task(task: tuple[dict, SourceDay, int, bool]) -> tuple[str, int, list[dict], list[dict]]:
    config, source_day, depth_index, include_contours = task
    id_cfg = config.get("identification", {})
    arrays = _cached_arrays(config, source_day, depth_index)
    kwargs = _detection_kwargs(config)
    day_iso = source_day.day.isoformat()
    if int(depth_index) == int(id_cfg.get("surface_depth_index", 0)):
        if str(id_cfg.get("surface_method", "sla_pyeddy")).lower() in {"sla_pyeddy", "pyeddy", "py-eddy-tracker"}:
            try:
                tmp_path = _write_surface_tmp(
                    config,
                    source_day,
                    np.asarray(arrays["lon"], dtype="f8"),
                    np.asarray(arrays["lat"], dtype="f8"),
                    np.asarray(arrays["adt"], dtype="f8"),
                )
                try:
                    detections = _surface_from_pyeddy_file(
                        config,
                        tmp_path,
                        source_day,
                        float(arrays["depth_m"]),
                        int(depth_index),
                        lon_name="longitude",
                        lat_name="latitude",
                        height_name="adt",
                        indexs=None,
                    )
                finally:
                    tmp_path.unlink(missing_ok=True)
            except Exception as exc:
                print(f"[surface fallback] cached py-eddy-tracker failed for {day_iso}: {type(exc).__name__}: {exc}", flush=True)
                detections = detect_surface_sla_fallback(
                    arrays["lon"],
                    arrays["lat"],
                    arrays["adt"],
                    float(arrays["depth_m"]),
                    int(depth_index),
                    day_iso,
                    **kwargs,
                )
        else:
            detections = detect_surface_sla_fallback(
                arrays["lon"],
                arrays["lat"],
                arrays["adt"],
                float(arrays["depth_m"]),
                int(depth_index),
                day_iso,
                **kwargs,
            )
    else:
        detections = detect_velocity_layer(
            arrays["lon"],
            arrays["lat"],
            float(arrays["depth_m"]),
            int(depth_index),
            arrays["u"],
            arrays["v"],
            day_iso,
            **kwargs,
        )

    records: list[dict] = []
    contours: list[dict] = []
    for local_id, det in enumerate(detections):
        det = det.__class__(**{**det.__dict__, "detection_id": int(depth_index) * 100000 + int(local_id)})
        record, rows = _records_from_detection(det)
        records.append(record)
        if include_contours:
            contours.extend(rows)
    return day_iso, int(depth_index), records, contours


def _run_shared_layer_task(task: tuple[dict, SourceDay, int, bool]) -> tuple[str, int, list[dict], list[dict]]:
    config, source_day, depth_index, include_contours = task
    id_cfg = config.get("identification", {})
    arrays = _shared_arrays(config, source_day, depth_index)
    kwargs = _detection_kwargs(config)
    day_iso = source_day.day.isoformat()
    if int(depth_index) == int(id_cfg.get("surface_depth_index", 0)):
        if str(id_cfg.get("surface_method", "sla_pyeddy")).lower() in {"sla_pyeddy", "pyeddy", "py-eddy-tracker"}:
            try:
                tmp_path = _write_surface_tmp(
                    config,
                    source_day,
                    np.asarray(arrays["lon"], dtype="f8"),
                    np.asarray(arrays["lat"], dtype="f8"),
                    np.asarray(arrays["adt"], dtype="f8"),
                )
                try:
                    detections = _surface_from_pyeddy_file(
                        config,
                        tmp_path,
                        source_day,
                        float(arrays["depth_m"]),
                        int(depth_index),
                        lon_name="longitude",
                        lat_name="latitude",
                        height_name="adt",
                        indexs=None,
                    )
                finally:
                    tmp_path.unlink(missing_ok=True)
            except Exception as exc:
                print(f"[surface fallback] shared py-eddy-tracker failed for {day_iso}: {type(exc).__name__}: {exc}", flush=True)
                detections = detect_surface_sla_fallback(
                    arrays["lon"],
                    arrays["lat"],
                    arrays["adt"],
                    float(arrays["depth_m"]),
                    int(depth_index),
                    day_iso,
                    **kwargs,
                )
        else:
            detections = detect_surface_sla_fallback(
                arrays["lon"],
                arrays["lat"],
                arrays["adt"],
                float(arrays["depth_m"]),
                int(depth_index),
                day_iso,
                **kwargs,
            )
    else:
        detections = detect_velocity_layer(
            arrays["lon"],
            arrays["lat"],
            float(arrays["depth_m"]),
            int(depth_index),
            arrays["u"],
            arrays["v"],
            day_iso,
            **kwargs,
        )
    records, contours = _records_from_detections(depth_index, detections, include_contours)
    return day_iso, int(depth_index), records, contours


def _records_from_detections(depth_index: int, detections: list[LayerDetection], include_contours: bool) -> tuple[list[dict], list[dict]]:
    records: list[dict] = []
    contours: list[dict] = []
    for local_id, det in enumerate(detections):
        det = det.__class__(**{**det.__dict__, "detection_id": int(depth_index) * 100000 + int(local_id)})
        record, rows = _records_from_detection(det)
        records.append(record)
        if include_contours:
            contours.extend(rows)
    return records, contours


def _run_array_layer_task(task: tuple[dict, SourceDay, int, int, dict[str, np.ndarray], bool]) -> tuple[str, int, list[dict], list[dict]]:
    config, source_day, depth_index, depth_pos, day_data, include_contours = task
    id_cfg = config.get("identification", {})
    kwargs = _detection_kwargs(config)
    day_iso = source_day.day.isoformat()
    if int(depth_index) == int(id_cfg.get("surface_depth_index", 0)):
        adt = np.asarray(day_data["adt"], dtype="f8")
        if str(id_cfg.get("surface_method", "sla_pyeddy")).lower() in {"sla_pyeddy", "pyeddy", "py-eddy-tracker"}:
            try:
                tmp_path = _write_surface_tmp(config, source_day, day_data["lon"], day_data["lat"], adt)
                try:
                    detections = _surface_from_pyeddy_file(
                        config,
                        tmp_path,
                        source_day,
                        float(day_data["depth"][depth_pos]),
                        int(depth_index),
                        lon_name="longitude",
                        lat_name="latitude",
                        height_name="adt",
                        indexs=None,
                    )
                finally:
                    tmp_path.unlink(missing_ok=True)
            except Exception as exc:
                print(f"[surface fallback] preloaded py-eddy-tracker failed for {day_iso}: {type(exc).__name__}: {exc}", flush=True)
                detections = detect_surface_sla_fallback(day_data["lon"], day_data["lat"], adt, float(day_data["depth"][depth_pos]), int(depth_index), day_iso, **kwargs)
        else:
            detections = detect_surface_sla_fallback(day_data["lon"], day_data["lat"], adt, float(day_data["depth"][depth_pos]), int(depth_index), day_iso, **kwargs)
    else:
        detections = detect_velocity_layer(
            day_data["lon"],
            day_data["lat"],
            float(day_data["depth"][depth_pos]),
            int(depth_index),
            np.asarray(day_data["u_all"][depth_pos], dtype="f8"),
            np.asarray(day_data["v_all"][depth_pos], dtype="f8"),
            day_iso,
            **kwargs,
        )
    records, contours = _records_from_detections(depth_index, detections, include_contours)
    return day_iso, int(depth_index), records, contours


def _write_day_outputs(config: dict, day_iso: str, records: list[dict], contours: list[dict]) -> None:
    day = parse_ymd(day_iso)
    obs_path, contour_path = layer_output_paths(config, day)
    write_table(pd.DataFrame.from_records(records), obs_path, index=False)
    write_table(pd.DataFrame.from_records(contours), contour_path, index=False)


def identify_day_streaming(
    config: dict,
    source_day: SourceDay,
    depth_indices: list[int],
    *,
    workers: int = 1,
    include_contours: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    records: list[dict] = []
    contours: list[dict] = []
    tasks = [(config, source_day, int(k), include_contours) for k in depth_indices]
    if workers <= 1 or len(tasks) <= 1:
        for task in tasks:
            _day_iso, _k, layer_records, layer_contours = _run_layer_task(task)
            records.extend(layer_records)
            contours.extend(layer_contours)
    else:
        with ProcessPoolExecutor(max_workers=int(workers)) as executor:
            futures = {executor.submit(_run_layer_task, task): task for task in tasks}
            for future in as_completed(futures):
                try:
                    _day_iso, _k, layer_records, layer_contours = future.result()
                except Exception as exc:
                    raise RuntimeError(f"Streaming layer identification failed for {source_day.day}") from exc
                records.extend(layer_records)
                contours.extend(layer_contours)
    return pd.DataFrame.from_records(records), pd.DataFrame.from_records(contours)


def identify_day_preloaded(
    config: dict,
    source_day: SourceDay,
    depth_indices: list[int],
    *,
    workers: int = 1,
    include_contours: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    day_data = read_day_data(config, source_day.day)
    depth_pos_by_index = {int(source_index): int(pos) for pos, source_index in enumerate(depth_indices)}
    tasks = [
        (config, source_day, int(k), depth_pos_by_index[int(k)], day_data, include_contours)
        for k in depth_indices
    ]
    records: list[dict] = []
    contours: list[dict] = []
    if workers <= 1 or len(tasks) <= 1:
        for task in tasks:
            _day_iso, _k, layer_records, layer_contours = _run_array_layer_task(task)
            records.extend(layer_records)
            contours.extend(layer_contours)
    else:
        with ProcessPoolExecutor(max_workers=int(workers)) as executor:
            futures = {executor.submit(_run_array_layer_task, task): task for task in tasks}
            for future in as_completed(futures):
                try:
                    _day_iso, _k, layer_records, layer_contours = future.result()
                except Exception as exc:
                    raise RuntimeError(f"Preloaded layer identification failed for {source_day.day}") from exc
                records.extend(layer_records)
                contours.extend(layer_contours)
    return pd.DataFrame.from_records(records), pd.DataFrame.from_records(contours)


def identify_day_year_cached(
    config: dict,
    source_day: SourceDay,
    depth_indices: list[int],
    *,
    workers: int = 1,
    include_contours: bool = False,
    executor: ProcessPoolExecutor | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    records: list[dict] = []
    contours: list[dict] = []
    tasks = [(config, source_day, int(k), include_contours) for k in depth_indices]
    if workers <= 1 or len(tasks) <= 1:
        for task in tasks:
            _day_iso, _k, layer_records, layer_contours = _run_cached_layer_task(task)
            records.extend(layer_records)
            contours.extend(layer_contours)
    else:
        owns_executor = executor is None
        if owns_executor:
            executor = ProcessPoolExecutor(max_workers=int(workers))
        try:
            futures = {executor.submit(_run_cached_layer_task, task): task for task in tasks}
            for future in as_completed(futures):
                try:
                    _day_iso, _k, layer_records, layer_contours = future.result()
                except Exception as exc:
                    raise RuntimeError(f"Cached layer identification failed for {source_day.day}") from exc
                records.extend(layer_records)
                contours.extend(layer_contours)
        finally:
            if owns_executor and executor is not None:
                executor.shutdown()
    return pd.DataFrame.from_records(records), pd.DataFrame.from_records(contours)


def identify_day_year_shared(
    config: dict,
    source_day: SourceDay,
    depth_indices: list[int],
    *,
    workers: int = 1,
    include_contours: bool = False,
    executor: ProcessPoolExecutor | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    records: list[dict] = []
    contours: list[dict] = []
    tasks = [(config, source_day, int(k), include_contours) for k in depth_indices]
    if workers <= 1 or len(tasks) <= 1:
        for task in tasks:
            _day_iso, _k, layer_records, layer_contours = _run_shared_layer_task(task)
            records.extend(layer_records)
            contours.extend(layer_contours)
    else:
        owns_executor = executor is None
        if owns_executor:
            raise RuntimeError("identify_day_year_shared requires an executor initialized with shared cache metadata.")
        futures = {executor.submit(_run_shared_layer_task, task): task for task in tasks}
        for future in as_completed(futures):
            try:
                _day_iso, _k, layer_records, layer_contours = future.result()
            except Exception as exc:
                raise RuntimeError(f"Shared-cache layer identification failed for {source_day.day}") from exc
            records.extend(layer_records)
            contours.extend(layer_contours)
    return pd.DataFrame.from_records(records), pd.DataFrame.from_records(contours)


def identify_range_streaming(
    config_path: str | Path,
    start: str | None = None,
    end: str | None = None,
    force: bool = False,
    workers: int | None = None,
    depth_index: int | None = None,
) -> None:
    config = load_config(config_path)
    ensure_dirs(config)
    start_day = parse_ymd(start or config["date_range"]["start"])
    end_day = parse_ymd(end or config["date_range"]["end"])
    days = list(iter_days(start_day, end_day))
    source_index = build_source_index(config, start_day, end_day)
    first_path = source_index[days[0]].path
    depth_indices = [int(depth_index)] if depth_index is not None else selected_depth_indices(config, first_path)
    workers = int(workers or config.get("identification", {}).get("workers", 1) or 1)

    pending_days = []
    for day in days:
        paths = layer_output_paths(config, day)
        if force or config.get("identification", {}).get("overwrite_existing", False) or not _valid_outputs(paths):
            pending_days.append(day)
    if not pending_days:
        print("Streaming layer identification: all daily outputs already exist.")
        return

    with tqdm(total=len(pending_days), desc=f"Streaming layer identification ({workers} workers/day)", unit="day") as bar:
        for day in pending_days:
            obs, contours = identify_day_streaming(
                config,
                source_index[day],
                depth_indices,
                workers=workers,
                include_contours=True,
            )
            _write_day_outputs(config, day.isoformat(), obs.to_dict("records"), contours.to_dict("records"))
            bar.update(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="CPU-only streaming CMEMS layer identification without input_daily files.")
    parser.add_argument("--config", default="config/config_3d.yaml")
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--depth-index", type=int)
    args = parser.parse_args()
    identify_range_streaming(args.config, args.start, args.end, force=args.force, workers=args.workers, depth_index=args.depth_index)


if __name__ == "__main__":
    main()
