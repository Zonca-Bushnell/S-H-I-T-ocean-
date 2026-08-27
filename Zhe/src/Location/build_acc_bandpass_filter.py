from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Iterable

import numpy as np
from netCDF4 import Dataset, num2date
from scipy import signal


COORD_NAMES = ("time", "depth", "latitude", "longitude")
DEFAULT_VARIABLES = ("uo_glor", "vo_glor", "zos_glor")
FILL_LIMIT = 1.0e10


@dataclass(frozen=True)
class TimePart:
    year: int
    path: str
    input_indices: np.ndarray
    output_start: int


@dataclass(frozen=True)
class ChunkTask:
    variable: str
    dimensions: tuple[str, ...]
    input_root: str
    temp_dir: str
    years: tuple[int, ...]
    time_parts: tuple[TimePart, ...]
    depth_slice: tuple[int, int] | None
    lat_slice: tuple[int, int]
    lon_slice: tuple[int, int]
    period_low_days: float
    period_high_days: float
    task_id: int
    cache_path: str | None = None
    cache_shape: tuple[int, ...] | None = None


def parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def as_dates(ds: Dataset) -> list[date]:
    tvar = ds.variables["time"]
    values = num2date(tvar[:], units=tvar.units, calendar=getattr(tvar, "calendar", "standard"))
    return [date(int(v.year), int(v.month), int(v.day)) for v in values]


def selected_time_parts(input_root: Path, years: Iterable[int], start: date, end: date) -> list[TimePart]:
    parts: list[TimePart] = []
    for year in years:
        path = input_root / f"global_phy_{year}.nc"
        if not path.exists():
            raise FileNotFoundError(path)
        with Dataset(path) as ds:
            dates = as_dates(ds)
        keep = np.asarray([i for i, day in enumerate(dates) if start <= day <= end], dtype=np.int64)
        if keep.size:
            parts.append(TimePart(year=year, path=str(path), input_indices=keep, output_start=0))
    if not parts:
        raise ValueError(f"No time records selected between {start} and {end}")
    return parts


def output_splits(parts: list[TimePart]) -> tuple[TimePart, ...]:
    out: list[TimePart] = []
    for part in parts:
        out.append(
            TimePart(
                year=part.year,
                path=part.path,
                input_indices=part.input_indices,
                output_start=0,
            )
        )
    return tuple(out)


def time_selector(indices: np.ndarray) -> slice | np.ndarray:
    if indices.size == 0:
        return indices
    if indices.size == 1 or np.all(np.diff(indices) == 1):
        return slice(int(indices[0]), int(indices[-1]) + 1)
    return indices


def clean_array(value: np.ndarray) -> np.ndarray:
    arr = np.ma.filled(value, np.nan).astype("float32", copy=False)
    arr[np.abs(arr) > FILL_LIMIT] = np.nan
    return arr


def filter_flat_series(data: np.ndarray, period_low_days: float, period_high_days: float) -> np.ndarray:
    ntime = data.shape[0]
    flat = np.reshape(data, (ntime, -1)).astype("float64", copy=False)
    out = np.full_like(flat, np.nan, dtype="float64")
    fs = 1.0
    low_freq = 1.0 / float(period_high_days)
    high_freq = 1.0 / float(period_low_days)
    sos = signal.butter(4, (low_freq, high_freq), btype="bandpass", fs=fs, output="sos")
    x = np.arange(ntime, dtype="float64")
    padlen = min(3 * (2 * sos.shape[0] + 1), max(0, ntime - 1))
    finite_mask = np.isfinite(flat)
    all_finite = np.all(finite_mask, axis=0)
    if np.any(all_finite):
        out[:, all_finite] = signal.sosfiltfilt(sos, flat[:, all_finite], axis=0, padlen=padlen)
    partial = np.where((np.sum(finite_mask, axis=0) >= max(12, padlen + 2)) & ~all_finite)[0]
    for idx in partial:
        series = flat[:, idx]
        finite = finite_mask[:, idx]
        nfinite = int(finite.sum())
        filled = np.interp(x, x[finite], series[finite])
        filtered = signal.sosfiltfilt(sos, filled, padlen=padlen)
        filtered[~finite] = np.nan
        out[:, idx] = filtered
    return np.reshape(out.astype("float32"), data.shape)


def read_chunk(task: ChunkTask) -> np.ndarray:
    if task.cache_path and task.cache_shape:
        mmap = np.memmap(task.cache_path, dtype="float32", mode="r", shape=task.cache_shape)
        ys = slice(*task.lat_slice)
        xs = slice(*task.lon_slice)
        if task.dimensions == ("time", "depth", "latitude", "longitude"):
            if task.depth_slice is None:
                raise ValueError(f"{task.variable} requires depth slice")
            return np.asarray(mmap[:, slice(*task.depth_slice), ys, xs], dtype="float32")
        if task.dimensions == ("time", "latitude", "longitude"):
            return np.asarray(mmap[:, ys, xs], dtype="float32")
        raise ValueError(f"Unsupported dimensions for {task.variable}: {task.dimensions}")

    yearly_arrays: list[np.ndarray] = []
    ys = slice(*task.lat_slice)
    xs = slice(*task.lon_slice)
    dslice = slice(*task.depth_slice) if task.depth_slice is not None else None
    for part in task.time_parts:
        with Dataset(part.path) as ds:
            var = ds.variables[task.variable]
            tidx = time_selector(part.input_indices)
            if task.dimensions == ("time", "depth", "latitude", "longitude"):
                if dslice is None:
                    raise ValueError(f"{task.variable} requires depth slice")
                arr = clean_array(var[tidx, dslice, ys, xs])
            elif task.dimensions == ("time", "latitude", "longitude"):
                arr = clean_array(var[tidx, ys, xs])
            else:
                raise ValueError(f"Unsupported dimensions for {task.variable}: {task.dimensions}")
            yearly_arrays.append(arr)
    return np.concatenate(yearly_arrays, axis=0)


def run_chunk(task: ChunkTask) -> dict[str, object]:
    data = read_chunk(task)
    filtered = filter_flat_series(data, task.period_low_days, task.period_high_days)
    del data
    out_dir = Path(task.temp_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    final_path = out_dir / f"{task.variable}_chunk_{task.task_id:06d}.npy"
    fd, tmp_name = tempfile.mkstemp(prefix=final_path.name, suffix=".tmp", dir=out_dir)
    os.close(fd)
    try:
        with open(tmp_name, "wb") as handle:
            np.save(handle, filtered)
        os.replace(tmp_name, final_path)
    finally:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)
    return {
        "variable": task.variable,
        "dimensions": task.dimensions,
        "depth_slice": task.depth_slice,
        "lat_slice": task.lat_slice,
        "lon_slice": task.lon_slice,
        "path": str(final_path),
        "shape": tuple(int(v) for v in filtered.shape),
    }


def copy_variable_attrs(src, dst) -> None:
    for attr in src.ncattrs():
        if attr in {"_FillValue"}:
            continue
        try:
            setattr(dst, attr, getattr(src, attr))
        except Exception:
            pass


def output_chunks(
    src: Dataset,
    dimensions: tuple[str, ...],
    depth_block: int,
    lat_block: int,
    lon_block: int,
) -> tuple[int, ...] | None:
    if dimensions == ("time", "depth", "latitude", "longitude"):
        return (
            min(30, len(src.dimensions["time"])),
            min(depth_block, len(src.dimensions["depth"])),
            min(lat_block, len(src.dimensions["latitude"])),
            min(lon_block, len(src.dimensions["longitude"])),
        )
    if dimensions == ("time", "latitude", "longitude"):
        return (
            min(30, len(src.dimensions["time"])),
            min(lat_block, len(src.dimensions["latitude"])),
            min(lon_block, len(src.dimensions["longitude"])),
        )
    return None


def ensure_outputs(
    input_root: Path,
    output_dir: Path,
    years: tuple[int, ...],
    variables: tuple[str, ...],
    time_parts: tuple[TimePart, ...],
    compression_level: int,
    depth_block: int,
    lat_block: int,
    lon_block: int,
    force: bool,
) -> dict[int, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[int, Path] = {}
    for part in time_parts:
        year = part.year
        src_path = input_root / f"global_phy_{year}.nc"
        out_path = output_dir / f"global_phy_{year}_bandpass_30_180d.nc"
        if out_path.exists() and force:
            out_path.unlink()
        paths[year] = out_path
        with Dataset(src_path) as src, Dataset(out_path, "a", format="NETCDF4") as dst:
            try:
                dst.set_fill_off()
            except Exception:
                pass
            for dim_name in COORD_NAMES:
                if dim_name not in src.dimensions:
                    continue
                if dim_name == "time":
                    dim_len = int(len(part.input_indices))
                else:
                    dim_len = len(src.dimensions[dim_name])
                if dim_name not in dst.dimensions:
                    dst.createDimension(dim_name, dim_len)
            for coord_name in COORD_NAMES:
                if coord_name not in src.variables:
                    continue
                src_var = src.variables[coord_name]
                if coord_name not in dst.variables:
                    dtype = src_var.dtype
                    dst_var = dst.createVariable(coord_name, dtype, src_var.dimensions)
                    copy_variable_attrs(src_var, dst_var)
                dst_var = dst.variables[coord_name]
                if coord_name == "time":
                    dst_var[:] = src_var[time_selector(part.input_indices)]
                else:
                    dst_var[:] = src_var[:]
            for variable in variables:
                if variable not in src.variables:
                    raise KeyError(f"{variable} not found in {src_path}")
                src_var = src.variables[variable]
                if variable not in dst.variables:
                    fill_value = getattr(src_var, "_FillValue", np.nan)
                    chunksizes = output_chunks(src, tuple(src_var.dimensions), depth_block, lat_block, lon_block)
                    dst_var = dst.createVariable(
                        variable,
                        "f4",
                        src_var.dimensions,
                        zlib=True,
                        complevel=int(compression_level),
                        shuffle=True,
                        fill_value=fill_value,
                        chunksizes=chunksizes,
                    )
                    copy_variable_attrs(src_var, dst_var)
                    dst_var.long_name = f"30-180 day bandpass filtered {getattr(src_var, 'long_name', variable)}"
                    dst_var.filter_note = "30-180 day zero-phase 4th-order Butterworth bandpass; background = raw - bandpass"
            dst.title = "ACC 30-180 day bandpass filtered mesoscale anomaly fields"
            dst.filter_note = "Only bandpass filtered variables are stored; raw fields remain in source annual NetCDF files"
            dst.filter_period_low_days = 30.0
            dst.filter_period_high_days = 180.0
            dst.filter_backend = "scipy.signal.butter(order=4, output='sos') + sosfiltfilt"
            dst.source_files = ",".join(str(input_root / f"global_phy_{y}.nc") for y in years)
            dst.history = f"{datetime.utcnow().isoformat()}Z created by build_acc_bandpass_filter.py"
    return paths


def variable_dimensions(input_root: Path, first_year: int, variables: tuple[str, ...]) -> dict[str, tuple[str, ...]]:
    with Dataset(input_root / f"global_phy_{first_year}.nc") as ds:
        return {variable: tuple(ds.variables[variable].dimensions) for variable in variables}


def grid_sizes(input_root: Path, first_year: int) -> tuple[int, int, int]:
    with Dataset(input_root / f"global_phy_{first_year}.nc") as ds:
        ndepth = len(ds.dimensions["depth"])
        nlat = len(ds.dimensions["latitude"])
        nlon = len(ds.dimensions["longitude"])
    return ndepth, nlat, nlon


def variable_shape(input_root: Path, first_year: int, variable: str) -> tuple[int, ...]:
    with Dataset(input_root / f"global_phy_{first_year}.nc") as ds:
        shape = tuple(int(v) for v in ds.variables[variable].shape)
    return shape


def preload_variable_to_memmap(
    variable: str,
    dimensions: tuple[str, ...],
    input_root: Path,
    years: tuple[int, ...],
    time_parts: tuple[TimePart, ...],
    temp_dir: Path,
    time_block: int,
    max_depth_count: int | None = None,
    max_lat_count: int | None = None,
    max_lon_count: int | None = None,
) -> tuple[Path, tuple[int, ...]]:
    temp_dir.mkdir(parents=True, exist_ok=True)
    src_shape = variable_shape(input_root, years[0], variable)
    total_time = int(sum(len(part.input_indices) for part in time_parts))
    if dimensions == ("time", "depth", "latitude", "longitude"):
        depth_stop = src_shape[1] if max_depth_count is None else min(src_shape[1], int(max_depth_count))
        lat_stop = src_shape[2] if max_lat_count is None else min(src_shape[2], int(max_lat_count))
        lon_stop = src_shape[3] if max_lon_count is None else min(src_shape[3], int(max_lon_count))
        cache_shape = (total_time, depth_stop, lat_stop, lon_stop)
        spatial_selector = (slice(0, depth_stop), slice(0, lat_stop), slice(0, lon_stop))
    elif dimensions == ("time", "latitude", "longitude"):
        lat_stop = src_shape[1] if max_lat_count is None else min(src_shape[1], int(max_lat_count))
        lon_stop = src_shape[2] if max_lon_count is None else min(src_shape[2], int(max_lon_count))
        cache_shape = (total_time, lat_stop, lon_stop)
        spatial_selector = (slice(0, lat_stop), slice(0, lon_stop))
    else:
        raise ValueError(f"Unsupported dimensions for {variable}: {dimensions}")
    cache_path = temp_dir / f"{variable}_raw_30_180d_input_cache.dat"
    if cache_path.exists():
        cache_path.unlink()
    mmap = np.memmap(cache_path, dtype="float32", mode="w+", shape=cache_shape)
    out_pos = 0
    for part in time_parts:
        selector = time_selector(part.input_indices)
        with Dataset(part.path) as ds:
            var = ds.variables[variable]
            if isinstance(selector, slice):
                start = int(selector.start)
                stop = int(selector.stop)
                for t0 in range(start, stop, max(1, int(time_block))):
                    t1 = min(stop, t0 + max(1, int(time_block)))
                    arr = clean_array(var[(slice(t0, t1),) + spatial_selector])
                    n = int(arr.shape[0])
                    mmap[out_pos : out_pos + n] = arr
                    out_pos += n
                    print(f"[bandpass] preload {variable} {part.year} time={t0}:{t1}", flush=True)
            else:
                indices = np.asarray(selector, dtype=np.int64)
                for offset in range(0, len(indices), max(1, int(time_block))):
                    idx = indices[offset : offset + max(1, int(time_block))]
                    arr = clean_array(var[(idx,) + spatial_selector])
                    n = int(arr.shape[0])
                    mmap[out_pos : out_pos + n] = arr
                    out_pos += n
                    print(f"[bandpass] preload {variable} {part.year} records={offset}:{offset+n}", flush=True)
    if out_pos != total_time:
        raise RuntimeError(f"Preload wrote {out_pos} records for {variable}, expected {total_time}")
    mmap.flush()
    del mmap
    print(f"[bandpass] preload complete {variable} cache={cache_path} shape={cache_shape}", flush=True)
    return cache_path, cache_shape


def block_ranges(size: int, block: int, limit: int | None = None) -> list[tuple[int, int]]:
    stop = size if limit is None else min(size, int(limit))
    return [(start, min(stop, start + block)) for start in range(0, stop, block)]


def build_tasks(
    input_root: Path,
    temp_dir: Path,
    years: tuple[int, ...],
    time_parts: tuple[TimePart, ...],
    variables: tuple[str, ...],
    period_low_days: float,
    period_high_days: float,
    depth_block: int,
    lat_block: int,
    lon_block: int,
    max_depth_count: int | None,
    max_lat_count: int | None,
    max_lon_count: int | None,
    cache_path: Path | None = None,
    cache_shape: tuple[int, ...] | None = None,
) -> list[ChunkTask]:
    dims = variable_dimensions(input_root, years[0], variables)
    ndepth, nlat, nlon = grid_sizes(input_root, years[0])
    lat_ranges = block_ranges(nlat, lat_block, max_lat_count)
    lon_ranges = block_ranges(nlon, lon_block, max_lon_count)
    depth_ranges = block_ranges(ndepth, depth_block, max_depth_count)
    tasks: list[ChunkTask] = []
    task_id = 0
    for variable in variables:
        vdims = dims[variable]
        if vdims == ("time", "depth", "latitude", "longitude"):
            for dsl in depth_ranges:
                for ysl in lat_ranges:
                    for xsl in lon_ranges:
                        tasks.append(
                            ChunkTask(
                                variable=variable,
                                dimensions=vdims,
                                input_root=str(input_root),
                                temp_dir=str(temp_dir),
                                years=years,
                                time_parts=time_parts,
                                depth_slice=dsl,
                                lat_slice=ysl,
                                lon_slice=xsl,
                                period_low_days=period_low_days,
                                period_high_days=period_high_days,
                                task_id=task_id,
                                cache_path=str(cache_path) if cache_path else None,
                                cache_shape=cache_shape,
                            )
                        )
                        task_id += 1
        elif vdims == ("time", "latitude", "longitude"):
            for ysl in lat_ranges:
                for xsl in lon_ranges:
                    tasks.append(
                        ChunkTask(
                            variable=variable,
                            dimensions=vdims,
                            input_root=str(input_root),
                            temp_dir=str(temp_dir),
                            years=years,
                            time_parts=time_parts,
                            depth_slice=None,
                            lat_slice=ysl,
                            lon_slice=xsl,
                            period_low_days=period_low_days,
                            period_high_days=period_high_days,
                            task_id=task_id,
                            cache_path=str(cache_path) if cache_path else None,
                            cache_shape=cache_shape,
                        )
                    )
                    task_id += 1
        else:
            raise ValueError(f"Unsupported dimensions for {variable}: {vdims}")
    return tasks


def write_result(result: dict[str, object], output_paths: dict[int, Path], time_parts: tuple[TimePart, ...]) -> None:
    variable = str(result["variable"])
    dimensions = tuple(result["dimensions"])
    depth_slice = result["depth_slice"]
    lat_slice = tuple(result["lat_slice"])
    lon_slice = tuple(result["lon_slice"])
    path = Path(str(result["path"]))
    filtered = np.load(path)
    pos = 0
    for part in time_parts:
        ntime = int(len(part.input_indices))
        chunk = filtered[pos : pos + ntime]
        pos += ntime
        with Dataset(output_paths[part.year], "a") as ds:
            var = ds.variables[variable]
            ys = slice(*lat_slice)
            xs = slice(*lon_slice)
            ts = slice(part.output_start, part.output_start + ntime)
            if dimensions == ("time", "depth", "latitude", "longitude"):
                dslice = slice(*depth_slice)
                var[ts, dslice, ys, xs] = chunk
            else:
                var[ts, ys, xs] = chunk
    path.unlink(missing_ok=True)


def mark_complete(output_paths: dict[int, Path], variables: tuple[str, ...], args: argparse.Namespace) -> None:
    for path in output_paths.values():
        with Dataset(path, "a") as ds:
            for variable in variables:
                ds.variables[variable].bandpass_complete = 1
            ds.bandpass_command = json.dumps(vars(args), ensure_ascii=False, default=str)


def run_tasks(
    tasks: list[ChunkTask],
    output_paths: dict[int, Path],
    parts: tuple[TimePart, ...],
    workers: int,
    max_in_flight: int | None = None,
) -> None:
    completed = 0
    next_task = 0
    limit = max(1, int(max_in_flight or max(1, int(workers)) * 2))
    with ProcessPoolExecutor(max_workers=max(1, int(workers))) as pool:
        future_map = {}
        while next_task < len(tasks) or future_map:
            while next_task < len(tasks) and len(future_map) < limit:
                task = tasks[next_task]
                future_map[pool.submit(run_chunk, task)] = task
                next_task += 1
            done, _ = wait(future_map, return_when=FIRST_COMPLETED)
            for future in done:
                task = future_map.pop(future)
                result = future.result()
                write_result(result, output_paths, parts)
                completed += 1
                print(
                    f"[bandpass] wrote {completed}/{len(tasks)} {task.variable} "
                    f"depth={task.depth_slice} lat={task.lat_slice} lon={task.lon_slice}",
                    flush=True,
                )


def parse_variables(value: str) -> tuple[str, ...]:
    variables = tuple(v.strip() for v in value.split(",") if v.strip())
    if not variables:
        raise argparse.ArgumentTypeError("At least one variable is required")
    return variables


def main() -> None:
    parser = argparse.ArgumentParser(description="Build ACC 30-180 day bandpass filtered anomaly NetCDF files.")
    parser.add_argument("--input-root", default="/root/autodl-fs/2020_2022_acc")
    parser.add_argument("--output-dir", default="/root/autodl-fs/2020_2022_acc/Filter")
    parser.add_argument("--temp-dir", default="/root/autodl-tmp/acc_filter_work")
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2022-12-31")
    parser.add_argument("--variables", type=parse_variables, default=DEFAULT_VARIABLES)
    parser.add_argument("--period-low-days", type=float, default=30.0, help="Short-period cutoff in days.")
    parser.add_argument("--period-high-days", type=float, default=180.0, help="Long-period cutoff in days.")
    parser.add_argument("--depth-block", type=int, default=4)
    parser.add_argument("--lat-block", type=int, default=8)
    parser.add_argument("--lon-block", type=int, default=360)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--compression-level", type=int, default=1)
    parser.add_argument("--cache-mode", choices=("memmap", "direct"), default="memmap")
    parser.add_argument("--preload-time-block", type=int, default=4)
    parser.add_argument("--max-in-flight", type=int, default=None, help="Maximum submitted chunks waiting for write.")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max-depth-count", type=int, default=None, help="Smoke-test limiter.")
    parser.add_argument("--max-lat-count", type=int, default=None, help="Smoke-test limiter.")
    parser.add_argument("--max-lon-count", type=int, default=None, help="Smoke-test limiter.")
    args = parser.parse_args()

    input_root = Path(args.input_root)
    output_dir = Path(args.output_dir)
    temp_dir = Path(args.temp_dir)
    start = parse_date(args.start)
    end = parse_date(args.end)
    if end < start:
        raise ValueError("--end must be on or after --start")
    years = tuple(range(start.year, end.year + 1))
    variables = tuple(args.variables)
    if not (0.0 < args.period_low_days < args.period_high_days):
        raise ValueError("--period-low-days must be shorter than --period-high-days")

    parts = output_splits(selected_time_parts(input_root, years, start, end))
    output_paths = ensure_outputs(
        input_root=input_root,
        output_dir=output_dir,
        years=years,
        variables=variables,
        time_parts=parts,
        compression_level=args.compression_level,
        depth_block=args.depth_block,
        lat_block=args.lat_block,
        lon_block=args.lon_block,
        force=args.force,
    )
    print(
        f"[bandpass] variables={variables} years={years} selected_days={sum(len(p.input_indices) for p in parts)} "
        f"workers={args.workers} cache_mode={args.cache_mode}",
        flush=True,
    )
    dims = variable_dimensions(input_root, years[0], variables)
    if args.cache_mode == "memmap":
        for variable in variables:
            cache_path, cache_shape = preload_variable_to_memmap(
                variable=variable,
                dimensions=dims[variable],
                input_root=input_root,
                years=years,
                time_parts=parts,
                temp_dir=temp_dir,
                time_block=args.preload_time_block,
                max_depth_count=args.max_depth_count,
                max_lat_count=args.max_lat_count,
                max_lon_count=args.max_lon_count,
            )
            try:
                tasks = build_tasks(
                    input_root=input_root,
                    temp_dir=temp_dir,
                    years=years,
                    time_parts=parts,
                    variables=(variable,),
                    period_low_days=args.period_low_days,
                    period_high_days=args.period_high_days,
                    depth_block=args.depth_block,
                    lat_block=args.lat_block,
                    lon_block=args.lon_block,
                    max_depth_count=args.max_depth_count,
                    max_lat_count=args.max_lat_count,
                    max_lon_count=args.max_lon_count,
                    cache_path=cache_path,
                    cache_shape=cache_shape,
                )
                print(f"[bandpass] filtering {variable} tasks={len(tasks)}", flush=True)
                run_tasks(tasks, output_paths, parts, args.workers, args.max_in_flight)
                mark_complete(output_paths, (variable,), args)
            finally:
                cache_path.unlink(missing_ok=True)
                print(f"[bandpass] removed cache {cache_path}", flush=True)
    else:
        tasks = build_tasks(
            input_root=input_root,
            temp_dir=temp_dir,
            years=years,
            time_parts=parts,
            variables=variables,
            period_low_days=args.period_low_days,
            period_high_days=args.period_high_days,
            depth_block=args.depth_block,
            lat_block=args.lat_block,
            lon_block=args.lon_block,
            max_depth_count=args.max_depth_count,
            max_lat_count=args.max_lat_count,
            max_lon_count=args.max_lon_count,
        )
        print(f"[bandpass] direct filtering tasks={len(tasks)}", flush=True)
        run_tasks(tasks, output_paths, parts, args.workers, args.max_in_flight)
        mark_complete(output_paths, variables, args)
    print("[bandpass] complete", flush=True)


if __name__ == "__main__":
    main()
