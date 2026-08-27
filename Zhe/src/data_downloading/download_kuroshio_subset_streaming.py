from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date, datetime, time, timedelta
from itertools import product
from pathlib import Path
from typing import Iterable

import copernicusmarine
import numpy as np
from netCDF4 import Dataset, num2date


DATASET_ID = "cmems_mod_glo_phy-all_my_0.25deg_P1D-m"
DATASET_VERSION = "202311"

DEFAULT_OUTPUT_DIR = Path(r"E:\DATA\Copernicus_Data\Global Ocean Ensemble Physics Reanalysis Kuroshio Current")
DEFAULT_TEMP_DIR = Path(r"E:\DATA\Copernicus_Data\Kuroshio_tmp")

GLOBAL_START = date(1993, 1, 1)
GLOBAL_END = date(2023, 1, 1)
SAMPLE_START = date(1993, 1, 1)
SAMPLE_END = date(1993, 1, 3)

VARIABLES = [
    "thetao_glor",
    "uo_glor",
    "vo_glor",
    "mlotst_glor",
    "zos_glor",
    "so_glor",
]

MIN_LONGITUDE = 120.0
MAX_LONGITUDE = 145.0
MIN_LATITUDE = 20.0
MAX_LATITUDE = 35.0
MIN_DEPTH = 0.5057600140571594
MAX_DEPTH = 1516.3636474609375

TIME_NAMES = ("time", "time_counter")
DEPTH_NAMES = ("depth", "deptht", "lev", "level")
COORD_NAMES = ("time", "time_counter", "depth", "deptht", "lev", "level", "latitude", "lat", "longitude", "lon")

USERNAME_KEYS = [
    "COPERNICUSMARINE_SERVICE_USERNAME",
    "COPERNICUSMARINE_USERNAME",
    "CMEMS_USERNAME",
]
PASSWORD_KEYS = [
    "COPERNICUSMARINE_SERVICE_PASSWORD",
    "COPERNICUSMARINE_PASSWORD",
    "CMEMS_PASSWORD",
]


def _first_non_empty_env(keys: list[str]) -> str | None:
    for key in keys:
        value = os.getenv(key)
        if value:
            return value
    return None


def _read_registry_env(var_name: str) -> str | None:
    if os.name != "nt":
        return None
    try:
        import winreg

        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Environment") as key:
            value, _ = winreg.QueryValueEx(key, var_name)
            return value or None
    except Exception:
        return None


def resolve_credentials() -> tuple[str | None, str | None]:
    username = _first_non_empty_env(USERNAME_KEYS)
    password = _first_non_empty_env(PASSWORD_KEYS)

    if not username:
        for key in USERNAME_KEYS:
            username = _read_registry_env(key)
            if username:
                break
    if not password:
        for key in PASSWORD_KEYS:
            password = _read_registry_env(key)
            if password:
                break

    return username, password


def validate_credentials(username: str | None, password: str | None) -> None:
    if username and password:
        ok = copernicusmarine.login(
            username=username,
            password=password,
            check_credentials_valid=True,
        )
    else:
        ok = copernicusmarine.login(check_credentials_valid=True)
    if not ok:
        raise RuntimeError(
            "No valid Copernicus credentials found. Set "
            "COPERNICUSMARINE_SERVICE_USERNAME and COPERNICUSMARINE_SERVICE_PASSWORD "
            "or run `copernicusmarine login` once."
        )


def parse_day(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def iter_year_windows(start: date, end: date) -> Iterable[tuple[date, date]]:
    current = start
    while current < end:
        year_end = date(current.year + 1, 1, 1)
        window_end = min(end, year_end)
        yield current, window_end
        current = window_end


def iter_blocks(start: date, end: date, block_days: int) -> Iterable[tuple[date, date]]:
    if block_days >= 365 and start.month == 1 and start.day == 1 and end.month == 1 and end.day == 1:
        years_per_block = max(1, round(block_days / 365))
        current = start
        while current < end:
            block_end = min(date(current.year + years_per_block, 1, 1), end)
            yield current, block_end
            current = block_end
        return

    current = start
    while current < end:
        block_end = min(end, current + timedelta(days=block_days))
        yield current, block_end
        current = block_end


def iter_days(start: date, end: date) -> Iterable[date]:
    current = start
    while current < end:
        yield current
        current += timedelta(days=1)


def output_name(start: date, end: date, sample: bool) -> str:
    if sample:
        return f"global_phy_sample_{start:%Y%m%d}_{end:%Y%m%d}.nc"
    return f"global_phy_{start:%Y}.nc"


def part_name(final_name: str) -> str:
    return final_name.removesuffix(".nc") + ".part.nc"


def subset_temp_name(start: date, end: date) -> str:
    return f"kuroshio_subset_{start:%Y%m%d}_{end:%Y%m%d}.nc"


def time_coord_name(ds: Dataset) -> str:
    for name in TIME_NAMES:
        if name in ds.variables:
            return name
    raise KeyError(f"No time coordinate found; expected one of {TIME_NAMES}.")


def depth_coord_name(ds: Dataset) -> str | None:
    for name in DEPTH_NAMES:
        if name in ds.variables or name in ds.dimensions:
            return name
    return None


def copy_attrs(src_var, dst_var) -> None:
    for attr in src_var.ncattrs():
        if attr == "_FillValue":
            continue
        dst_var.setncattr(attr, src_var.getncattr(attr))


def create_variable_like(
    dst: Dataset,
    src_var,
    name: str,
    *,
    compression_level: int,
    depth_block: int,
    lat_block: int,
    lon_block: int,
):
    fill_value = getattr(src_var, "_FillValue", None)
    kwargs = {}
    if fill_value is not None:
        kwargs["fill_value"] = fill_value

    if name in VARIABLES:
        kwargs.update({"zlib": True, "complevel": compression_level})
        chunks = [
            chunk_size_for_dim(
                dim,
                len(dst.dimensions[dim]),
                depth_block=depth_block,
                lat_block=lat_block,
                lon_block=lon_block,
            )
            for dim in src_var.dimensions
        ]
        kwargs["chunksizes"] = tuple(chunks)

    dst_var = dst.createVariable(name, src_var.datatype, src_var.dimensions, **kwargs)
    copy_attrs(src_var, dst_var)
    return dst_var


def chunk_size_for_dim(dim: str, size: int, *, depth_block: int, lat_block: int, lon_block: int) -> int:
    if dim in TIME_NAMES:
        return 1
    if dim in DEPTH_NAMES:
        if depth_block <= 0:
            return size
        return min(depth_block, size)
    if dim in ("latitude", "lat"):
        if lat_block <= 0:
            return size
        return min(lat_block, size)
    if dim in ("longitude", "lon"):
        if lon_block <= 0:
            return size
        return min(lon_block, size)
    return size


def create_part_from_subset(
    src: Dataset,
    part_path: Path,
    *,
    compression_level: int,
    depth_block: int,
    lat_block: int,
    lon_block: int,
) -> None:
    tmp_path = part_path.with_suffix(part_path.suffix + ".creating")
    if tmp_path.exists():
        tmp_path.unlink()

    with Dataset(tmp_path, "w", format="NETCDF4") as dst:
        for attr in src.ncattrs():
            dst.setncattr(attr, src.getncattr(attr))
        dst.title = "Kuroshio Copernicus Marine server-side subset, streamed into annual file"
        dst.source_dataset_id = DATASET_ID
        dst.source_dataset_version = DATASET_VERSION
        dst.completed_iso_dates = ""

        for name, dim in src.dimensions.items():
            dst.createDimension(name, None if name in TIME_NAMES else len(dim))

        for name, src_var in src.variables.items():
            if name in VARIABLES or name in COORD_NAMES or name in src.dimensions:
                create_variable_like(
                    dst,
                    src_var,
                    name,
                    compression_level=compression_level,
                    depth_block=depth_block,
                    lat_block=lat_block,
                    lon_block=lon_block,
                )

        for name, src_var in src.variables.items():
            if name in TIME_NAMES:
                continue
            if name in dst.variables and name not in VARIABLES and src_var.dimensions:
                dst.variables[name][:] = src_var[:]

    tmp_path.replace(part_path)


def dates_from_time_var(time_var, indices: slice | np.ndarray | list[int] | None = None) -> list[date]:
    values = time_var[:] if indices is None else time_var[indices]
    units = getattr(time_var, "units", None)
    if not units:
        raise ValueError("Time variable has no units attribute.")
    calendar = getattr(time_var, "calendar", "standard")
    decoded = num2date(
        values,
        units,
        calendar=calendar,
        only_use_cftime_datetimes=False,
        only_use_python_datetimes=True,
    )
    if np.ndim(decoded) == 0:
        decoded = [decoded.item()]
    return [value.date() for value in decoded]


def completed_dates(dst: Dataset) -> list[str]:
    raw = getattr(dst, "completed_iso_dates", "")
    if raw:
        return [item for item in raw.splitlines() if item]

    time_name = time_coord_name(dst)
    if len(dst.dimensions[time_name]) == 0:
        return []

    derived = [day.isoformat() for day in dates_from_time_var(dst.variables[time_name])]
    dst.completed_iso_dates = "\n".join(derived)
    dst.sync()
    return derived


def validate_part_state(dst: Dataset) -> list[str]:
    done = completed_dates(dst)
    time_name = time_coord_name(dst)
    time_len = len(dst.dimensions[time_name])
    if time_len != len(done):
        raise RuntimeError(
            f"{dst.filepath()} has {time_len} time records but {len(done)} completed dates. "
            "It may have been interrupted during a write. Move it aside or rerun with --force."
        )
    return done


def ensure_coordinates_match(src: Dataset, dst: Dataset) -> None:
    for name in ("depth", "deptht", "lev", "level", "latitude", "lat", "longitude", "lon"):
        if name not in src.variables or name not in dst.variables:
            continue
        src_values = np.asarray(src.variables[name][:])
        dst_values = np.asarray(dst.variables[name][:])
        if src_values.shape != dst_values.shape or not np.allclose(src_values, dst_values, equal_nan=True):
            raise ValueError(f"Coordinate mismatch for {name}; refusing to append incompatible subset.")


def filled_array(values, fill_value):
    if np.ma.isMaskedArray(values):
        return np.ma.filled(values, fill_value)
    return values


def block_slices_for_variable(
    var,
    *,
    depth_block: int,
    lat_block: int,
    lon_block: int,
) -> Iterable[dict[int, slice]]:
    ranges: list[tuple[int, list[slice]]] = []
    for axis, dim in enumerate(var.dimensions):
        if dim in TIME_NAMES:
            continue
        size = var.shape[axis]
        block = chunk_size_for_dim(
            dim,
            size,
            depth_block=depth_block,
            lat_block=lat_block,
            lon_block=lon_block,
        )
        if block >= size:
            continue
        ranges.append(
            (
                axis,
                [slice(start, min(start + block, size)) for start in range(0, size, block)],
            )
        )

    if not ranges:
        yield {}
        return

    axes = [axis for axis, _ in ranges]
    slice_lists = [items for _, items in ranges]
    for combo in product(*slice_lists):
        yield dict(zip(axes, combo, strict=True))


def append_variable_time_slice(
    src_var,
    dst_var,
    src_time_index: int,
    dst_time_index: int,
    *,
    depth_block: int,
    lat_block: int,
    lon_block: int,
    buffers: dict[tuple[str, tuple[int, ...], str], np.ndarray],
) -> None:
    dims = src_var.dimensions
    if not any(dim in TIME_NAMES for dim in dims):
        return

    time_axis = next(i for i, dim in enumerate(dims) if dim in TIME_NAMES)
    fill_value = getattr(dst_var, "_FillValue", np.nan)

    for blocks in block_slices_for_variable(
        src_var,
        depth_block=depth_block,
        lat_block=lat_block,
        lon_block=lon_block,
    ):
        src_key = [slice(None)] * len(dims)
        dst_key = [slice(None)] * len(dims)
        src_key[time_axis] = src_time_index
        dst_key[time_axis] = dst_time_index
        for axis, block_slice in blocks.items():
            src_key[axis] = block_slice
            dst_key[axis] = block_slice

        values = filled_array(src_var[tuple(src_key)], fill_value)
        values = np.asarray(values)
        cache_key = (src_var.name, values.shape, str(values.dtype))
        buffer = buffers.get(cache_key)
        if buffer is None:
            buffer = np.empty(values.shape, dtype=values.dtype)
            buffers[cache_key] = buffer
        np.copyto(buffer, values, casting="unsafe")
        dst_var[tuple(dst_key)] = buffer


def append_subset_to_part(
    subset_path: Path,
    part_path: Path,
    start: date,
    end: date,
    *,
    compression_level: int,
    depth_block: int,
    lat_block: int,
    lon_block: int,
) -> int:
    with Dataset(subset_path) as src:
        missing = [name for name in VARIABLES if name not in src.variables]
        if missing:
            raise KeyError(f"{subset_path} is missing variables: {', '.join(missing)}")

        if not part_path.exists():
            create_part_from_subset(
                src,
                part_path,
                compression_level=compression_level,
                depth_block=depth_block,
                lat_block=lat_block,
                lon_block=lon_block,
            )

        with Dataset(part_path, "r+") as dst:
            ensure_coordinates_match(src, dst)
            done = validate_part_state(dst)
            done_set = set(done)

            src_time_name = time_coord_name(src)
            dst_time_name = time_coord_name(dst)
            src_days = dates_from_time_var(src.variables[src_time_name])
            append_indices = [
                i
                for i, day in enumerate(src_days)
                if start <= day < end and day.isoformat() not in done_set
            ]
            if not append_indices:
                return 0

            buffers: dict[tuple[str, tuple[int, ...], str], np.ndarray] = {}
            appended = 0
            for src_time_index in append_indices:
                day = src_days[src_time_index]
                dst_time_index = len(dst.dimensions[dst_time_name])
                for name in VARIABLES:
                    append_variable_time_slice(
                        src.variables[name],
                        dst.variables[name],
                        src_time_index,
                        dst_time_index,
                        depth_block=depth_block,
                        lat_block=lat_block,
                        lon_block=lon_block,
                        buffers=buffers,
                    )
                dst.variables[dst_time_name][dst_time_index] = src.variables[src_time_name][src_time_index]
                done.append(day.isoformat())
                dst.completed_iso_dates = "\n".join(done)
                dst.sync()
                appended += 1

            return appended


def iter_year_windows_for_block(start: date, end: date) -> Iterable[tuple[date, date]]:
    current = start
    while current < end:
        year_end = date(current.year + 1, 1, 1)
        window_end = min(end, year_end)
        yield current, window_end
        current = window_end


def annual_output_name(start: date) -> str:
    return f"global_phy_{start:%Y}.nc"


def contiguous_index_slice(indices: list[int]) -> slice | None:
    if not indices:
        return None
    expected = list(range(indices[0], indices[-1] + 1))
    if indices != expected:
        return None
    return slice(indices[0], indices[-1] + 1)


def valid_annual_netcdf(path: Path, expected_days: int) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        with Dataset(path) as ds:
            time_name = time_coord_name(ds)
            if len(ds.dimensions[time_name]) != expected_days:
                return False
            return all(name in ds.variables for name in VARIABLES)
    except Exception:
        return False


def create_annual_from_subset(
    subset_path: Path,
    output_path: Path,
    start: date,
    end: date,
    *,
    compression_level: int,
    depth_block: int,
    lat_block: int,
    lon_block: int,
    force: bool,
) -> tuple[str, int, str]:
    expected_days = list(iter_days(start, end))
    if not force and valid_annual_netcdf(output_path, len(expected_days)):
        return (output_path.name, 0, "skipped")

    tmp_path = output_path.with_suffix(output_path.suffix + ".creating")
    if tmp_path.exists():
        tmp_path.unlink()

    with Dataset(subset_path) as src:
        missing = [name for name in VARIABLES if name not in src.variables]
        if missing:
            raise KeyError(f"{subset_path} is missing variables: {', '.join(missing)}")

        src_time_name = time_coord_name(src)
        src_days = dates_from_time_var(src.variables[src_time_name])
        indices = [i for i, day in enumerate(src_days) if start <= day < end]
        if len(indices) != len(expected_days):
            raise RuntimeError(
                f"{subset_path} has {len(indices)} records for {start} to {end}, "
                f"expected {len(expected_days)}."
            )
        index_slice = contiguous_index_slice(indices)
        if index_slice is None:
            raise RuntimeError(f"Non-contiguous time indices for {start} to {end} in {subset_path}.")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with Dataset(tmp_path, "w", format="NETCDF4") as dst:
            for attr in src.ncattrs():
                dst.setncattr(attr, src.getncattr(attr))
            dst.title = "Kuroshio Copernicus Marine server-side annual subset"
            dst.source_dataset_id = DATASET_ID
            dst.source_dataset_version = DATASET_VERSION
            dst.completed_iso_dates = "\n".join(day.isoformat() for day in expected_days)
            dst.year_start = start.isoformat()
            dst.year_end_exclusive = end.isoformat()

            for name, dim in src.dimensions.items():
                if name in TIME_NAMES:
                    dst.createDimension(name, len(indices))
                else:
                    dst.createDimension(name, len(dim))

            for name, src_var in src.variables.items():
                if name in VARIABLES or name in COORD_NAMES or name in src.dimensions:
                    create_variable_like(
                        dst,
                        src_var,
                        name,
                        compression_level=compression_level,
                        depth_block=depth_block,
                        lat_block=lat_block,
                        lon_block=lon_block,
                    )

            for name, src_var in src.variables.items():
                if name not in dst.variables:
                    continue

                dst_var = dst.variables[name]
                dims = src_var.dimensions
                if any(dim in TIME_NAMES for dim in dims):
                    time_axis = next(i for i, dim in enumerate(dims) if dim in TIME_NAMES)
                    src_key = [slice(None)] * len(dims)
                    src_key[time_axis] = index_slice
                    values = filled_array(src_var[tuple(src_key)], getattr(dst_var, "_FillValue", np.nan))
                    dst_var[:] = np.asarray(values)
                elif name not in VARIABLES and dims:
                    dst_var[:] = src_var[:]

            dst.sync()

    tmp_path.replace(output_path)
    return (output_path.name, len(indices), "written")


def split_subset_to_annual_files(
    subset_path: Path,
    output_dir: Path,
    start: date,
    end: date,
    *,
    split_workers: int,
    compression_level: int,
    depth_block: int,
    lat_block: int,
    lon_block: int,
    force: bool,
) -> list[tuple[str, int, str]]:
    tasks = [
        (
            subset_path,
            output_dir / annual_output_name(year_start),
            year_start,
            year_end,
            compression_level,
            depth_block,
            lat_block,
            lon_block,
            force,
        )
        for year_start, year_end in iter_year_windows_for_block(start, end)
    ]

    if split_workers <= 1 or len(tasks) <= 1:
        return [
            create_annual_from_subset(
                subset_path,
                output_path,
                year_start,
                year_end,
                compression_level=compression_level,
                depth_block=depth_block,
                lat_block=lat_block,
                lon_block=lon_block,
                force=force,
            )
            for (
                subset_path,
                output_path,
                year_start,
                year_end,
                compression_level,
                depth_block,
                lat_block,
                lon_block,
                force,
            ) in tasks
        ]

    results: list[tuple[str, int, str]] = []
    with ProcessPoolExecutor(max_workers=min(split_workers, len(tasks))) as executor:
        futures = [
            executor.submit(
                create_annual_from_subset,
                subset_path,
                output_path,
                year_start,
                year_end,
                compression_level=compression_level,
                depth_block=depth_block,
                lat_block=lat_block,
                lon_block=lon_block,
                force=force,
            )
            for (
                subset_path,
                output_path,
                year_start,
                year_end,
                compression_level,
                depth_block,
                lat_block,
                lon_block,
                force,
            ) in tasks
        ]
        for future in as_completed(futures):
            results.append(future.result())
    return sorted(results)


def valid_netcdf(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        with Dataset(path) as ds:
            time_coord_name(ds)
            return all(name in ds.variables for name in VARIABLES)
    except Exception:
        return False


def subset_block(
    start: date,
    end: date,
    output_path: Path,
    *,
    username: str | None,
    password: str | None,
    compression_level: int,
    chunk_size_limit: int,
    disable_progress_bar: bool,
) -> None:
    subset_start = datetime.combine(start, time.min)
    subset_end = datetime.combine(end, time.min) - timedelta(seconds=1)

    kwargs = {
        "dataset_id": DATASET_ID,
        "dataset_version": DATASET_VERSION,
        "variables": VARIABLES,
        "minimum_longitude": MIN_LONGITUDE,
        "maximum_longitude": MAX_LONGITUDE,
        "minimum_latitude": MIN_LATITUDE,
        "maximum_latitude": MAX_LATITUDE,
        "minimum_depth": MIN_DEPTH,
        "maximum_depth": MAX_DEPTH,
        "start_datetime": subset_start.strftime("%Y-%m-%dT%H:%M:%S"),
        "end_datetime": subset_end.strftime("%Y-%m-%dT%H:%M:%S"),
        "coordinates_selection_method": "strict-inside",
        "netcdf_compression_level": compression_level,
        "chunk_size_limit": chunk_size_limit,
        "disable_progress_bar": disable_progress_bar,
        "overwrite": True,
        "output_directory": str(output_path.parent),
        "output_filename": output_path.name,
    }
    if username and password:
        kwargs["username"] = username
        kwargs["password"] = password

    copernicusmarine.subset(**kwargs)


def cleanup_temp_file(path: Path, *, keep_on_error: bool = False) -> None:
    if keep_on_error:
        return
    if path.exists():
        path.unlink()


def process_window(
    start: date,
    end: date,
    *,
    sample: bool,
    output_dir: Path,
    temp_dir: Path,
    block_days: int,
    depth_block: int,
    lat_block: int,
    lon_block: int,
    chunk_size_limit: int,
    compression_level: int,
    username: str | None,
    password: str | None,
    force: bool,
    keep_temp_on_error: bool,
    disable_progress_bar: bool,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)

    final_path = output_dir / output_name(start, end, sample)
    part_path = output_dir / part_name(final_path.name)
    expected_days = {day.isoformat() for day in iter_days(start, end)}

    if force:
        for path in (final_path, part_path):
            if path.exists():
                path.unlink()
    elif final_path.exists():
        raise FileExistsError(f"Final output already exists: {final_path}. Use --force to overwrite.")

    for block_start, block_end in iter_blocks(start, end, block_days):
        if part_path.exists():
            with Dataset(part_path, "r+") as dst:
                done = set(validate_part_state(dst))
            if all(day.isoformat() in done for day in iter_days(block_start, block_end)):
                print(f"Skip completed block {block_start} to {block_end}")
                continue

        temp_path = temp_dir / subset_temp_name(block_start, block_end)
        try:
            if temp_path.exists() and not valid_netcdf(temp_path):
                temp_path.unlink()
            if not temp_path.exists():
                print(f"Subset {block_start} to {block_end} -> {temp_path.name}")
                subset_block(
                    block_start,
                    block_end,
                    temp_path,
                    username=username,
                    password=password,
                    compression_level=compression_level,
                    chunk_size_limit=chunk_size_limit,
                    disable_progress_bar=disable_progress_bar,
                )
            appended = append_subset_to_part(
                temp_path,
                part_path,
                block_start,
                block_end,
                compression_level=compression_level,
                depth_block=depth_block,
                lat_block=lat_block,
                lon_block=lon_block,
            )
            print(f"Appended {appended} day(s) from {temp_path.name}")
            cleanup_temp_file(temp_path)
        except Exception:
            cleanup_temp_file(temp_path, keep_on_error=keep_temp_on_error)
            raise

    with Dataset(part_path, "r+") as dst:
        done = set(validate_part_state(dst))
        missing = sorted(expected_days - done)
        if missing:
            raise RuntimeError("Output is incomplete; missing dates: " + ", ".join(missing[:10]))
        dst.history = (
            f"Completed by download_kuroshio_subset_streaming.py on "
            f"{datetime.now().isoformat(timespec='seconds')}"
        )
        dst.sync()

    part_path.replace(final_path)
    print(f"Completed {final_path}")
    return final_path


def process_block(
    start: date,
    end: date,
    *,
    output_dir: Path,
    temp_dir: Path,
    split_workers: int,
    depth_block: int,
    lat_block: int,
    lon_block: int,
    chunk_size_limit: int,
    compression_level: int,
    username: str | None,
    password: str | None,
    force: bool,
    keep_temp_on_error: bool,
    disable_progress_bar: bool,
) -> list[tuple[str, int, str]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)

    temp_path = temp_dir / subset_temp_name(start, end)
    try:
        if temp_path.exists() and not valid_netcdf(temp_path):
            temp_path.unlink()
        if not temp_path.exists():
            print(f"Subset {start} to {end} -> {temp_path.name}")
            subset_block(
                start,
                end,
                temp_path,
                username=username,
                password=password,
                compression_level=compression_level,
                chunk_size_limit=chunk_size_limit,
                disable_progress_bar=disable_progress_bar,
            )

        results = split_subset_to_annual_files(
            temp_path,
            output_dir,
            start,
            end,
            split_workers=split_workers,
            compression_level=compression_level,
            depth_block=depth_block,
            lat_block=lat_block,
            lon_block=lon_block,
            force=force,
        )
        for name, days, status in results:
            print(f"{status}: {name} ({days} day(s))")
        cleanup_temp_file(temp_path)
        return results
    except Exception:
        cleanup_temp_file(temp_path, keep_on_error=keep_temp_on_error)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Copernicus Marine Kuroshio server-side subsets in low-memory time blocks."
    )
    parser.add_argument("--start", type=parse_day, help="Start date, inclusive, as YYYY-MM-DD.")
    parser.add_argument("--end", type=parse_day, help="End date, exclusive, as YYYY-MM-DD.")
    parser.add_argument("--sample", action="store_true", help="Run a one-day 1993-01-01 sample.")
    parser.add_argument("--block-days", type=int, default=1, help="Days per Copernicus subset request.")
    parser.add_argument("--split-workers", type=int, default=1, help="Annual NetCDF files written in parallel per subset block.")
    parser.add_argument("--depth-block", type=int, default=4, help="Depth chunk size; 0 uses the full depth dimension.")
    parser.add_argument("--lat-block", type=int, default=32, help="Latitude chunk size; 0 uses the full latitude dimension.")
    parser.add_argument("--lon-block", type=int, default=256, help="Longitude chunk size; 0 uses the full longitude dimension.")
    parser.add_argument("--chunk-size-limit", type=int, default=100, help="Copernicus toolbox dask chunk size limit.")
    parser.add_argument("--compression-level", type=int, default=1, choices=range(0, 10))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--temp-dir", type=Path, default=DEFAULT_TEMP_DIR)
    parser.add_argument("--force", action="store_true", help="Overwrite existing final/part output.")
    parser.add_argument("--keep-temp-on-error", action="store_true", help="Keep temporary subset files after errors.")
    parser.add_argument("--show-progress", action="store_true", help="Show Copernicus progress bars.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.block_days < 1:
        raise ValueError("--block-days must be >= 1")
    if args.split_workers < 1:
        raise ValueError("--split-workers must be >= 1")
    if args.depth_block < 0:
        raise ValueError("--depth-block must be >= 0")
    if args.lat_block < 0:
        raise ValueError("--lat-block must be >= 0")
    if args.lon_block < 0:
        raise ValueError("--lon-block must be >= 0")

    if args.sample:
        start_day, end_day = SAMPLE_START, SAMPLE_END
    else:
        start_day = args.start or GLOBAL_START
        end_day = args.end or GLOBAL_END

    if start_day >= end_day:
        raise ValueError("Start date must be earlier than end date.")

    username, password = resolve_credentials()
    validate_credentials(username, password)

    for block_start, block_end in iter_blocks(start_day, end_day, args.block_days):
        process_block(
            block_start,
            block_end,
            output_dir=args.output_dir,
            temp_dir=args.temp_dir,
            split_workers=args.split_workers,
            depth_block=args.depth_block,
            lat_block=args.lat_block,
            lon_block=args.lon_block,
            chunk_size_limit=args.chunk_size_limit,
            compression_level=args.compression_level,
            username=username,
            password=password,
            force=args.force,
            keep_temp_on_error=args.keep_temp_on_error,
            disable_progress_bar=not args.show_progress,
        )

    print("Download completed.")


if __name__ == "__main__":
    main()
