from __future__ import annotations

import argparse
import shutil
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import netCDF4
import numpy as np
from tqdm import tqdm

from .common import ensure_dirs, load_config, parse_ymd
from .streaming_cmems import grid_metadata, source_paths_for_years, spatial_window, variable_names


VAR_MAP = {
    "uo_glor": "u_clim",
    "vo_glor": "v_clim",
    "thetao_glor": "thetao_clim",
    "so_glor": "so_clim",
    "zos_glor": "zos_clim",
    "mlotst_glor": "mlotst_clim",
}


@dataclass(frozen=True)
class ClimTask:
    source_name: str
    out_name: str
    is3d: bool
    out_depth_start: int
    out_depth_stop: int
    source_depth_start: int
    source_depth_stop: int
    part_path: str


def climatology_path(config: dict, start: date, end: date, smooth_days: int) -> Path:
    root = Path(config["paths"]["output_dir"]) / "climatology"
    return root / f"cmems_doy_climatology_{start:%Y}_{end:%Y}_{int(smooth_days)}d.nc"


def _is_leap(year: int) -> bool:
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)


def clim_doy_index(day: date) -> int:
    doy = day.timetuple().tm_yday
    if not _is_leap(day.year) and (day.month, day.day) >= (3, 1):
        doy += 1
    return int(doy - 1)


def _date_to_index(ds: netCDF4.Dataset, time_name: str) -> dict[date, int]:
    time_var = ds.variables[time_name]
    values = netCDF4.num2date(
        time_var[:],
        time_var.units,
        getattr(time_var, "calendar", "standard"),
        only_use_cftime_datetimes=False,
        only_use_python_datetimes=True,
    )
    return {value.date(): int(i) for i, value in enumerate(values)}


def _rolling_sum_doy(values: np.ndarray, smooth_days: int, dtype) -> np.ndarray:
    smooth_days = max(1, int(smooth_days))
    if smooth_days % 2 == 0:
        smooth_days += 1
    half = smooth_days // 2
    if half == 0:
        return values.astype(dtype, copy=False)
    padded = np.concatenate([values[-half:], values, values[:half]], axis=0)
    zero = np.zeros((1, *values.shape[1:]), dtype=dtype)
    csum = np.concatenate([zero, np.cumsum(padded, axis=0, dtype=dtype)], axis=0)
    return csum[smooth_days:] - csum[:-smooth_days]


def _smooth_mean(raw_sum: np.ndarray, raw_count: np.ndarray, smooth_days: int, fill_value: np.float32) -> np.ndarray:
    smooth_sum = _rolling_sum_doy(raw_sum, smooth_days, "f8")
    smooth_count = _rolling_sum_doy(raw_count, smooth_days, "u4")
    out = np.full(smooth_sum.shape, fill_value, dtype="f4")
    np.divide(smooth_sum, smooth_count, out=out, where=smooth_count > 0)
    return out


def _run_clim_task(
    config_path: str,
    start_iso: str,
    end_iso: str,
    smooth_days: int,
    task: ClimTask,
) -> dict:
    config = load_config(config_path)
    names = variable_names(config)
    start_day = parse_ymd(start_iso)
    end_day = parse_ymd(end_iso)
    source_paths = source_paths_for_years(config, range(start_day.year, end_day.year + 1))
    first_path = source_paths[0]
    window = spatial_window(config, first_path)
    with netCDF4.Dataset(first_path) as ds:
        lat_count = len(ds.variables[names["lat"]][window.lat_slice])
        lon_count = len(ds.variables[names["lon"]][window.lon_slice])

    if task.is3d:
        depth_count = task.source_depth_stop - task.source_depth_start
        raw_shape = (366, depth_count, lat_count, lon_count)
    else:
        raw_shape = (366, lat_count, lon_count)
    raw_sum = np.zeros(raw_shape, dtype="f8")
    raw_count = np.zeros(raw_shape, dtype="u2")

    for path in source_paths:
        if not path.exists():
            raise FileNotFoundError(path)
        with netCDF4.Dataset(path) as ds:
            date_index = _date_to_index(ds, names["time"])
            var = ds.variables[task.source_name]
            for day in sorted(d for d in date_index if start_day <= d <= end_day):
                t = date_index[day]
                if task.is3d:
                    values = np.ma.filled(
                        var[t, task.source_depth_start : task.source_depth_stop, window.lat_slice, window.lon_slice],
                        np.nan,
                    ).astype("f4", copy=False)
                else:
                    values = np.ma.filled(var[t, window.lat_slice, window.lon_slice], np.nan).astype("f4", copy=False)
                valid = np.isfinite(values)
                d = clim_doy_index(day)
                raw_sum[d] += np.where(valid, values, 0.0)
                raw_count[d] += valid.astype("u2")

    fill_value = np.float32(9.96921e36)
    smoothed = _smooth_mean(raw_sum, raw_count, smooth_days, fill_value)
    with Path(task.part_path).open("wb") as handle:
        np.save(handle, smoothed)
    return {
        "source_name": task.source_name,
        "out_name": task.out_name,
        "is3d": task.is3d,
        "out_depth_start": task.out_depth_start,
        "out_depth_stop": task.out_depth_stop,
        "part_path": task.part_path,
        "rows": int(smoothed.shape[0]),
    }


def _make_tasks(config: dict, first_path: Path, part_dir: Path, depth_block: int) -> list[ClimTask]:
    meta = grid_metadata(config, first_path)
    selected_depth = [int(v) for v in meta["depth_indices"]]
    tasks: list[ClimTask] = []
    with netCDF4.Dataset(first_path) as ds:
        for source_name, out_name in VAR_MAP.items():
            if source_name not in ds.variables:
                raise KeyError(f"Missing climatology source variable {source_name} in {first_path}")
            var = ds.variables[source_name]
            is3d = "depth" in var.dimensions
            if is3d:
                for out_start in range(0, len(selected_depth), int(depth_block)):
                    out_stop = min(len(selected_depth), out_start + int(depth_block))
                    source_start = selected_depth[out_start]
                    source_stop = selected_depth[out_stop - 1] + 1
                    tasks.append(
                        ClimTask(
                            source_name=source_name,
                            out_name=out_name,
                            is3d=True,
                            out_depth_start=out_start,
                            out_depth_stop=out_stop,
                            source_depth_start=source_start,
                            source_depth_stop=source_stop,
                            part_path=str(part_dir / f"{out_name}_z{out_start:03d}_{out_stop:03d}.npy"),
                        )
                    )
            else:
                tasks.append(
                    ClimTask(
                        source_name=source_name,
                        out_name=out_name,
                        is3d=False,
                        out_depth_start=0,
                        out_depth_stop=0,
                        source_depth_start=0,
                        source_depth_stop=0,
                        part_path=str(part_dir / f"{out_name}.npy"),
                    )
                )
    return tasks


def _create_output(config: dict, first_path: Path, tmp_path: Path, smooth_days: int, compression_level: int) -> dict[str, object]:
    meta = grid_metadata(config, first_path)
    lon = meta["lon"]
    lat = meta["lat"]
    depth = meta["depth"]
    fill_value = np.float32(9.96921e36)
    handles: dict[str, object] = {}
    dst = netCDF4.Dataset(tmp_path, "w", format="NETCDF4")
    handles["_dataset"] = dst
    dst.createDimension("doy", 366)
    dst.createDimension("depth", len(depth))
    dst.createDimension("latitude", len(lat))
    dst.createDimension("longitude", len(lon))
    dst.title = "CMEMS day-of-year climatology with circular smoothing"
    dst.source = str(config.get("data_source", {}).get("input_nc_dir", config.get("data_source", {}).get("input_nc_file", "")))
    dst.smooth_days = int(smooth_days)
    dst.createVariable("doy", "i2", ("doy",))[:] = np.arange(1, 367, dtype="i2")
    dst.createVariable("depth", "f8", ("depth",))[:] = depth
    dst.createVariable("latitude", "f8", ("latitude",))[:] = lat
    dst.createVariable("longitude", "f8", ("longitude",))[:] = lon
    with netCDF4.Dataset(first_path) as src:
        for source_name, out_name in VAR_MAP.items():
            var = src.variables[source_name]
            is3d = "depth" in var.dimensions
            dims = ("doy", "depth", "latitude", "longitude") if is3d else ("doy", "latitude", "longitude")
            chunks = (1, min(len(depth), 8), len(lat), len(lon)) if is3d else (1, len(lat), len(lon))
            out_var = dst.createVariable(
                out_name,
                "f4",
                dims,
                zlib=True,
                complevel=int(compression_level),
                fill_value=fill_value,
                chunksizes=chunks,
            )
            out_var.units = getattr(var, "units", "")
            out_var.long_name = f"{getattr(var, 'long_name', source_name)} day-of-year climatology"
            handles[out_name] = out_var
    return handles


def build_climatology(
    config_path: str | Path,
    start: str | date = "1993-01-01",
    end: str | date = "2022-12-31",
    smooth_days: int = 31,
    force: bool = False,
    workers: int = 8,
    depth_block: int | None = None,
    temp_dir: str | Path | None = None,
    compression_level: int | None = None,
) -> Path:
    config = load_config(config_path)
    ensure_dirs(config)
    start_day = parse_ymd(start)
    end_day = parse_ymd(end)
    smooth_days = max(1, int(smooth_days))
    if smooth_days % 2 == 0:
        smooth_days += 1
    out_path = climatology_path(config, start_day, end_day, smooth_days)
    if out_path.exists() and out_path.stat().st_size > 0 and not force:
        return out_path

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    source_paths = source_paths_for_years(config, range(start_day.year, end_day.year + 1))
    first_path = source_paths[0]
    depth_block = int(depth_block or config.get("conversion", {}).get("depth_block", 8) or 8)
    compression_level = int(compression_level if compression_level is not None else config.get("conversion", {}).get("compression_level", 1))
    work_root = Path(temp_dir or config.get("paths", {}).get("temp_dir", tempfile.gettempdir()))
    work_root.mkdir(parents=True, exist_ok=True)
    part_dir = Path(tempfile.mkdtemp(prefix="climatology_parts_", dir=str(work_root)))

    started = time.perf_counter()
    try:
        tasks = _make_tasks(config, first_path, part_dir, depth_block)
        completed: list[dict] = []
        if int(workers) <= 1 or len(tasks) <= 1:
            for task in tqdm(tasks, desc="Climatology chunks", unit="chunk"):
                completed.append(_run_clim_task(str(config_path), start_day.isoformat(), end_day.isoformat(), smooth_days, task))
        else:
            with ProcessPoolExecutor(max_workers=int(workers)) as executor:
                futures = {
                    executor.submit(_run_clim_task, str(config_path), start_day.isoformat(), end_day.isoformat(), smooth_days, task): task
                    for task in tasks
                }
                with tqdm(total=len(futures), desc=f"Climatology chunks ({workers} workers)", unit="chunk") as bar:
                    for future in as_completed(futures):
                        task = futures[future]
                        try:
                            completed.append(future.result())
                        except Exception as exc:
                            raise RuntimeError(f"Climatology chunk failed for {task.out_name}") from exc
                        bar.update(1)

        handles = _create_output(config, first_path, tmp_path, smooth_days, compression_level)
        try:
            for item in tqdm(sorted(completed, key=lambda row: (str(row["out_name"]), int(row["out_depth_start"]))), desc="Write climatology", unit="chunk"):
                data = np.load(str(item["part_path"]))
                out_var = handles[str(item["out_name"])]
                if bool(item["is3d"]):
                    out_var[:, int(item["out_depth_start"]) : int(item["out_depth_stop"]), :, :] = data
                else:
                    out_var[:, :, :] = data
        finally:
            handles["_dataset"].close()
        tmp_path.replace(out_path)
    finally:
        shutil.rmtree(part_dir, ignore_errors=True)

    print(f"Climatology elapsed seconds: {time.perf_counter() - started:.1f}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build streaming CMEMS day-of-year climatology for lifecycle composites.")
    parser.add_argument("--config", default="config/config_3d_cmems.yaml")
    parser.add_argument("--start", default="1993-01-01")
    parser.add_argument("--end", default="2022-12-31")
    parser.add_argument("--smooth-days", type=int, default=31)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--depth-block", type=int)
    parser.add_argument("--temp-dir")
    parser.add_argument("--compression-level", type=int)
    args = parser.parse_args()
    out = build_climatology(
        args.config,
        args.start,
        args.end,
        args.smooth_days,
        force=args.force,
        workers=args.workers,
        depth_block=args.depth_block,
        temp_dir=args.temp_dir,
        compression_level=args.compression_level,
    )
    print(f"Climatology: {out}")


if __name__ == "__main__":
    main()
