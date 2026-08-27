from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable

import netCDF4
import numpy as np

from .common import iter_days


DEFAULT_VARIABLES = {
    "lon": "longitude",
    "lat": "latitude",
    "depth": "depth",
    "time": "time",
    "u": "uo_glor",
    "v": "vo_glor",
    "adt": "zos_glor",
}


@dataclass(frozen=True)
class SourceDay:
    day: date
    path: str
    time_index: int


@dataclass(frozen=True)
class SpatialWindow:
    lon_slice: slice
    lat_slice: slice


@dataclass
class YearCache:
    year: int
    path: str
    day_to_time_index: dict[date, int]
    lon: np.ndarray
    lat: np.ndarray
    depth: np.ndarray
    depth_indices: list[int]
    u_all: np.ndarray
    v_all: np.ndarray
    adt_all: np.ndarray


def is_streaming_source_configured(config: dict) -> bool:
    source = config.get("data_source", {})
    if str(source.get("kind", "")).lower() in {
        "cmems_streaming",
        "cmems_netcdf_timeseries",
        "cmems_annual_netcdf",
    }:
        return True
    return any(key in source for key in ("input_nc_dir", "input_nc_files", "input_nc_file"))


def variable_names(config: dict) -> dict[str, str]:
    vars_cfg = config.get("variables", {})
    return {
        "lon": vars_cfg.get("source_lon", DEFAULT_VARIABLES["lon"]),
        "lat": vars_cfg.get("source_lat", DEFAULT_VARIABLES["lat"]),
        "depth": vars_cfg.get("source_depth", DEFAULT_VARIABLES["depth"]),
        "time": vars_cfg.get("source_time", DEFAULT_VARIABLES["time"]),
        "u": vars_cfg.get("source_u", DEFAULT_VARIABLES["u"]),
        "v": vars_cfg.get("source_v", DEFAULT_VARIABLES["v"]),
        "adt": vars_cfg.get("source_height", vars_cfg.get("output_height", DEFAULT_VARIABLES["adt"])),
    }


def _source_paths_for_years(config: dict, years: Iterable[int]) -> list[Path]:
    source = config.get("data_source", {})
    if source.get("input_nc_files"):
        return [Path(path) for path in source["input_nc_files"]]
    if source.get("input_nc_dir"):
        root = Path(source["input_nc_dir"])
        template = source.get("annual_file_template", "global_phy_{year}.nc")
        return [root / template.format(year=year) for year in sorted(set(years))]
    if source.get("input_nc_file"):
        return [Path(source["input_nc_file"])]
    raise ValueError("Streaming CMEMS source requires input_nc_dir, input_nc_files, or input_nc_file.")


def source_paths_for_years(config: dict, years: Iterable[int]) -> list[Path]:
    return _source_paths_for_years(config, years)


def _decode_time_days(ds: netCDF4.Dataset, time_name: str) -> dict[date, int]:
    time_var = ds.variables[time_name]
    values = netCDF4.num2date(
        time_var[:],
        time_var.units,
        getattr(time_var, "calendar", "standard"),
        only_use_cftime_datetimes=False,
        only_use_python_datetimes=True,
    )
    return {value.date(): int(i) for i, value in enumerate(values)}


def build_source_index(config: dict, start_day: date, end_day: date) -> dict[date, SourceDay]:
    names = variable_names(config)
    days = list(iter_days(start_day, end_day))
    paths = _source_paths_for_years(config, (day.year for day in days))
    out: dict[date, SourceDay] = {}
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        with netCDF4.Dataset(path) as ds:
            indices = _decode_time_days(ds, names["time"])
        needed = [day for day in days if day in indices]
        for day in needed:
            out[day] = SourceDay(day=day, path=str(path), time_index=indices[day])
    missing = [day.isoformat() for day in days if day not in out]
    if missing:
        raise FileNotFoundError("Missing CMEMS days: " + ", ".join(missing[:10]))
    return out


def _contiguous_slice(indices: np.ndarray, label: str) -> slice:
    if indices.size == 0:
        raise ValueError(f"No {label} values selected by region.")
    if not np.all(np.diff(indices) == 1):
        raise ValueError(f"Selected {label} values are not contiguous; split the source first.")
    return slice(int(indices[0]), int(indices[-1]) + 1)


def spatial_window(config: dict, path: str | Path) -> SpatialWindow:
    names = variable_names(config)
    with netCDF4.Dataset(path) as ds:
        lon = np.asarray(ds.variables[names["lon"]][:], dtype="f8")
        lat = np.asarray(ds.variables[names["lat"]][:], dtype="f8")
    bbox = config.get("region", {}).get("bbox")
    if not bbox:
        return SpatialWindow(slice(0, lon.size), slice(0, lat.size))
    lon_min, lon_max, lat_min, lat_max = [float(v) for v in bbox]
    lon_mask = (lon >= lon_min) & (lon <= lon_max) if lon_min <= lon_max else ((lon >= lon_min) | (lon <= lon_max))
    lat_mask = (lat >= lat_min) & (lat <= lat_max) if lat_min <= lat_max else ((lat >= lat_max) & (lat <= lat_min))
    return SpatialWindow(
        lon_slice=_contiguous_slice(np.where(lon_mask)[0], "longitude"),
        lat_slice=_contiguous_slice(np.where(lat_mask)[0], "latitude"),
    )


def selected_depth_indices(config: dict, path: str | Path) -> list[int]:
    names = variable_names(config)
    max_depth = float(config.get("region", {}).get("max_depth_m", np.inf))
    with netCDF4.Dataset(path) as ds:
        depth = np.asarray(ds.variables[names["depth"]][:], dtype="f8")
    indices = np.where(depth <= max_depth)[0]
    if indices.size == 0:
        raise ValueError(f"No source depth <= {max_depth}")
    return [int(i) for i in indices]


def selected_depth_count(config: dict, path: str | Path) -> int:
    return len(selected_depth_indices(config, path))


def grid_metadata(config: dict, path: str | Path) -> dict[str, np.ndarray | SpatialWindow | list[int]]:
    names = variable_names(config)
    window = spatial_window(config, path)
    depth_indices = selected_depth_indices(config, path)
    with netCDF4.Dataset(path) as ds:
        lon = np.asarray(ds.variables[names["lon"]][window.lon_slice], dtype="f8")
        lat = np.asarray(ds.variables[names["lat"]][window.lat_slice], dtype="f8")
        depth_all = np.asarray(ds.variables[names["depth"]][:], dtype="f8")
    return {
        "lon": lon,
        "lat": lat,
        "depth": depth_all[np.asarray(depth_indices, dtype=int)],
        "depth_indices": depth_indices,
        "window": window,
    }


def filled_f8(values) -> np.ndarray:
    return np.ma.filled(values, np.nan).astype("f8", copy=False)


def filled_f4(values) -> np.ndarray:
    return np.ma.filled(values, np.nan).astype("f4", copy=False)


def load_year_cache(config: dict, path: str | Path, *, year: int | None = None) -> YearCache:
    names = variable_names(config)
    path = Path(path)
    window = spatial_window(config, path)
    depth_indices = selected_depth_indices(config, path)
    depth_indexer = np.asarray(depth_indices, dtype=int)
    with netCDF4.Dataset(path) as ds:
        day_to_time_index = _decode_time_days(ds, names["time"])
        lon = np.asarray(ds.variables[names["lon"]][window.lon_slice], dtype="f8")
        lat = np.asarray(ds.variables[names["lat"]][window.lat_slice], dtype="f8")
        depth_all = np.asarray(ds.variables[names["depth"]][:], dtype="f8")
        depth = depth_all[depth_indexer]
        u_all = filled_f4(ds.variables[names["u"]][:, depth_indexer, window.lat_slice, window.lon_slice])
        v_all = filled_f4(ds.variables[names["v"]][:, depth_indexer, window.lat_slice, window.lon_slice])
        adt_all = filled_f4(ds.variables[names["adt"]][:, window.lat_slice, window.lon_slice])
    return YearCache(
        year=int(year if year is not None else min(day_to_time_index).year),
        path=str(path),
        day_to_time_index=day_to_time_index,
        lon=lon,
        lat=lat,
        depth=depth,
        depth_indices=[int(v) for v in depth_indices],
        u_all=u_all,
        v_all=v_all,
        adt_all=adt_all,
    )


def read_layer_arrays(config: dict, source_day: SourceDay, depth_index: int) -> dict[str, np.ndarray | float]:
    names = variable_names(config)
    window = spatial_window(config, source_day.path)
    with netCDF4.Dataset(source_day.path) as ds:
        lon = np.asarray(ds.variables[names["lon"]][window.lon_slice], dtype="f8")
        lat = np.asarray(ds.variables[names["lat"]][window.lat_slice], dtype="f8")
        depth = np.asarray(ds.variables[names["depth"]][:], dtype="f8")
        if int(depth_index) == int(config.get("identification", {}).get("surface_depth_index", 0)):
            adt = filled_f8(ds.variables[names["adt"]][source_day.time_index, window.lat_slice, window.lon_slice])
            return {"lon": lon, "lat": lat, "depth_m": float(depth[depth_index]), "adt": adt}
        u = filled_f8(ds.variables[names["u"]][source_day.time_index, depth_index, window.lat_slice, window.lon_slice])
        v = filled_f8(ds.variables[names["v"]][source_day.time_index, depth_index, window.lat_slice, window.lon_slice])
        return {"lon": lon, "lat": lat, "depth_m": float(depth[depth_index]), "u": u, "v": v}


def read_day_data(config: dict, day: date) -> dict[str, np.ndarray]:
    source_day = build_source_index(config, day, day)[day]
    names = variable_names(config)
    depth_indices = selected_depth_indices(config, source_day.path)
    window = spatial_window(config, source_day.path)
    with netCDF4.Dataset(source_day.path) as ds:
        lon = np.asarray(ds.variables[names["lon"]][window.lon_slice], dtype="f8")
        lat = np.asarray(ds.variables[names["lat"]][window.lat_slice], dtype="f8")
        depth_all = np.asarray(ds.variables[names["depth"]][:], dtype="f8")
        depth = depth_all[np.asarray(depth_indices, dtype=int)]
        u_all = filled_f8(ds.variables[names["u"]][source_day.time_index, depth_indices, window.lat_slice, window.lon_slice])
        v_all = filled_f8(ds.variables[names["v"]][source_day.time_index, depth_indices, window.lat_slice, window.lon_slice])
        adt = filled_f8(ds.variables[names["adt"]][source_day.time_index, window.lat_slice, window.lon_slice])
    return {"lon": lon, "lat": lat, "depth": depth, "u_all": u_all, "v_all": v_all, "adt": adt}
