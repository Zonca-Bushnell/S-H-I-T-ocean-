from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import numpy as np
from netCDF4 import Dataset, num2date
from tqdm import tqdm

from .common import ensure_dirs, iter_days, load_config, parse_ymd


def output_path(config: dict, day: date) -> Path:
    return Path(config["paths"]["input_daily_dir"]) / f"uv_{day:%Y%m%d}.nc"


def valid_uv_file(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        with Dataset(path) as ds:
            required = ("longitude", "latitude", "depth", "u", "v", "adt")
            return all(name in ds.variables for name in required) and all(name in ds.dimensions for name in ("depth", "latitude", "longitude"))
    except OSError:
        return False


def _slice_indices(values: np.ndarray, low: float, high: float) -> np.ndarray:
    idx = np.where((values >= low) & (values <= high))[0]
    if idx.size == 0:
        raise ValueError(f"No coordinate values in range {low}..{high}")
    return idx


def _contiguous_selector(indices: np.ndarray) -> slice | np.ndarray:
    if indices.size and np.array_equal(indices, np.arange(indices[0], indices[-1] + 1)):
        return slice(int(indices[0]), int(indices[-1]) + 1)
    return indices


def _date_to_index(ds: Dataset, time_name: str) -> dict[date, int]:
    time_var = ds.variables[time_name]
    dates = num2date(
        time_var[:],
        time_var.units,
        getattr(time_var, "calendar", "standard"),
        only_use_cftime_datetimes=False,
        only_use_python_datetimes=True,
    )
    out: dict[date, int] = {}
    for i, value in enumerate(dates):
        out[value.date()] = i
    return out


def _input_nc_files(config: dict) -> list[Path]:
    source = config.get("data_source", {})
    files = source.get("input_nc_files")
    if files:
        paths = [Path(item) for item in files]
    else:
        value = source.get("input_nc_file")
        paths = [Path(value)] if value else []
    if not paths:
        raise ValueError("No CMEMS input NetCDF file configured. Use data_source.input_nc_file or data_source.input_nc_files.")
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing CMEMS input files: " + ", ".join(missing[:10]))
    return paths


def _build_date_file_index(paths: list[Path], time_name: str) -> dict[date, tuple[Path, int]]:
    out: dict[date, tuple[Path, int]] = {}
    conflicts: list[str] = []
    for path in paths:
        with Dataset(path) as ds:
            for day, time_index in _date_to_index(ds, time_name).items():
                if day in out:
                    conflicts.append(f"{day.isoformat()}: {out[day][0]} and {path}")
                else:
                    out[day] = (path, time_index)
    if conflicts:
        raise ValueError("Duplicate dates across CMEMS input files: " + "; ".join(conflicts[:8]))
    return out


def _filled_f4(values, fill_value: np.float32) -> np.ndarray:
    arr = np.ma.filled(values, np.nan).astype("f4", copy=False)
    np.nan_to_num(arr, copy=False, nan=fill_value, posinf=fill_value, neginf=fill_value)
    return arr


def convert_one(ds: Dataset, time_index: int, nc_path: Path, day: date, config: dict, force: bool = False) -> None:
    conv = config["conversion"]
    if not force and not conv.get("overwrite_existing", False) and valid_uv_file(nc_path):
        return

    names = config["variables"]
    lon0, lon1, lat0, lat1 = [float(v) for v in config["region"]["bbox"]]
    max_depth = float(config["region"]["max_depth_m"])
    fill_value = np.float32(9.96921e36)
    depth_block = int(conv.get("depth_block", 8))

    lon_all = np.asarray(ds.variables[names["source_lon"]][:], dtype="f8")
    lat_all = np.asarray(ds.variables[names["source_lat"]][:], dtype="f8")
    depth_all = np.asarray(ds.variables[names["source_depth"]][:], dtype="f8")
    ix = _slice_indices(lon_all, lon0, lon1)
    iy = _slice_indices(lat_all, lat0, lat1)
    iz = np.where(depth_all <= max_depth)[0]
    if iz.size == 0:
        raise ValueError(f"No depth <= {max_depth}")

    lon = lon_all[ix]
    lat = lat_all[iy]
    depth = depth_all[iz]
    x_sel = _contiguous_selector(ix)
    y_sel = _contiguous_selector(iy)

    nc_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = nc_path.with_suffix(nc_path.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    with Dataset(tmp_path, "w", format=conv.get("netcdf_format", "NETCDF4")) as dst:
        dst.createDimension("depth", depth.size)
        dst.createDimension("latitude", lat.size)
        dst.createDimension("longitude", lon.size)
        dst.title = "CMEMS hybrid 3D eddy input subset"
        dst.source_file = str(Path(ds.filepath()))
        dst.source_time_index = int(time_index)
        dst.date = day.isoformat()
        dst.bbox = ",".join(str(v) for v in config["region"]["bbox"])
        dst.max_depth_m = max_depth

        lon_var = dst.createVariable("longitude", "f8", ("longitude",))
        lat_var = dst.createVariable("latitude", "f8", ("latitude",))
        depth_var = dst.createVariable("depth", "f8", ("depth",))
        lon_var.units = "degrees_east"
        lat_var.units = "degrees_north"
        depth_var.units = "m"
        lon_var[:] = lon
        lat_var[:] = lat
        depth_var[:] = depth

        chunksizes = (min(depth_block, depth.size), lat.size, lon.size)
        compression = int(conv.get("compression_level", 4))
        u_var = dst.createVariable("u", "f4", ("depth", "latitude", "longitude"), zlib=True, complevel=compression, fill_value=fill_value, chunksizes=chunksizes)
        v_var = dst.createVariable("v", "f4", ("depth", "latitude", "longitude"), zlib=True, complevel=compression, fill_value=fill_value, chunksizes=chunksizes)
        adt_var = dst.createVariable("adt", "f4", ("latitude", "longitude"), zlib=True, complevel=compression, fill_value=fill_value, chunksizes=(lat.size, lon.size))
        u_var.units = "m/s"
        v_var.units = "m/s"
        adt_var.units = "m"
        adt_var.long_name = "sea surface height for surface SLA eddy identification"

        adt = _filled_f4(ds.variables[names["source_height"]][time_index, y_sel, x_sel], fill_value)
        adt_var[:, :] = adt

        for out0 in range(0, depth.size, depth_block):
            out1 = min(out0 + depth_block, depth.size)
            z_indices = iz[out0:out1]
            z_sel = _contiguous_selector(z_indices)
            u_block = _filled_f4(ds.variables[names["source_u"]][time_index, z_sel, y_sel, x_sel], fill_value)
            v_block = _filled_f4(ds.variables[names["source_v"]][time_index, z_sel, y_sel, x_sel], fill_value)
            u_var[out0:out1, :, :] = u_block
            v_var[out0:out1, :, :] = v_block

    tmp_path.replace(nc_path)


def convert_range(config_path: str | Path, start: str | None = None, end: str | None = None, force: bool = False) -> None:
    config = load_config(config_path)
    ensure_dirs(config)
    start_day = parse_ymd(start or config["date_range"]["start"])
    end_day = parse_ymd(end or config["date_range"]["end"])
    names = config["variables"]
    input_files = _input_nc_files(config)
    time_index = _build_date_file_index(input_files, names.get("source_time", "time"))
    days = list(iter_days(start_day, end_day))
    missing = [day.isoformat() for day in days if day not in time_index]
    if missing:
        raise FileNotFoundError("Missing CMEMS time records for dates: " + ", ".join(missing[:10]))

    by_file: dict[Path, list[tuple[date, int]]] = {}
    for day in days:
        path, index = time_index[day]
        by_file.setdefault(path, []).append((day, index))

    progress = tqdm(days, desc="CMEMS UV+ADT daily subset", unit="day")
    try:
        for path in input_files:
            items = by_file.get(path, [])
            if not items:
                continue
            with Dataset(path) as ds:
                for day, time_i in items:
                    convert_one(ds, time_i, output_path(config, day), day, config, force=force)
                    progress.update(1)
    finally:
        progress.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert a CMEMS time-series NetCDF to daily Kuroshio UV+ADT subsets.")
    parser.add_argument("--config", default="config/config_3d_cmems.yaml")
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    convert_range(args.config, args.start, args.end, force=args.force)


if __name__ == "__main__":
    main()
