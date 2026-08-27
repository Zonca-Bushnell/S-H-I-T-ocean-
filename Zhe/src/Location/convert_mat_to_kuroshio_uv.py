from __future__ import annotations

import argparse
import time
from pathlib import Path

import h5py
import numpy as np
from netCDF4 import Dataset
from tqdm import tqdm

from .common import discover_mat_files, ensure_dirs, load_config, parse_ymd


def output_path(config: dict, day) -> Path:
    return Path(config["paths"]["input_daily_dir"]) / f"uv_{day:%Y%m%d}.nc"


def valid_uv_file(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        with Dataset(path) as ds:
            return all(name in ds.variables for name in ("longitude", "latitude", "depth", "u", "v", "adt"))
    except OSError:
        return False


def _slice_indices(values: np.ndarray, low: float, high: float) -> np.ndarray:
    idx = np.where((values >= low) & (values <= high))[0]
    if idx.size == 0:
        raise ValueError(f"No coordinate values in range {low}..{high}")
    return idx


def _depth_selector(indices: np.ndarray) -> slice | np.ndarray:
    if indices.size and np.array_equal(indices, np.arange(indices[0], indices[-1] + 1)):
        return slice(int(indices[0]), int(indices[-1]) + 1)
    return indices


def _read_velocity_block(
    mat_path: Path,
    names: dict,
    z_indices: np.ndarray,
    y_slice: slice,
    x_slice: slice,
    retries: int,
    retry_sleep: float,
    label: str,
) -> tuple[np.ndarray, np.ndarray]:
    z_sel = _depth_selector(z_indices)
    last_error: Exception | None = None
    for attempt in range(1, retries + 2):
        try:
            with h5py.File(mat_path, "r") as src:
                u_block = np.asarray(src[names["source_u"]][z_sel, y_slice, x_slice], dtype="f4")
                v_block = np.asarray(src[names["source_v"]][z_sel, y_slice, x_slice], dtype="f4")
            return u_block, v_block
        except OSError as exc:
            last_error = exc
            if attempt > retries:
                break
            print(f"[retry {attempt}/{retries}] {label}: {exc}")
            time.sleep(retry_sleep * attempt)
    raise OSError(f"Failed reading {label} after {retries + 1} attempts") from last_error


def convert_one(mat_path: Path, nc_path: Path, day, config: dict, force: bool = False) -> None:
    conv = config["conversion"]
    if not force and not conv.get("overwrite_existing", False) and valid_uv_file(nc_path):
        return

    nc_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = nc_path.with_suffix(nc_path.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    names = config["variables"]
    lon0, lon1, lat0, lat1 = [float(v) for v in config["region"]["bbox"]]
    max_depth = float(config["region"]["max_depth_m"])
    fill_value = np.float32(9.96921e36)
    depth_block = int(conv.get("depth_block", 4))
    retries = int(conv.get("retries", 3))
    retry_sleep = float(conv.get("retry_sleep", 2.0))

    with h5py.File(mat_path, "r") as src, Dataset(tmp_path, "w", format=conv.get("netcdf_format", "NETCDF4")) as dst:
        lon_all = np.asarray(src[names["source_lon"]][0], dtype="f8")
        lat_all = np.asarray(src[names["source_lat"]][0], dtype="f8")
        depth_all = np.asarray(src[names["source_depth"]][0], dtype="f8")
        ix = _slice_indices(lon_all, lon0, lon1)
        iy = _slice_indices(lat_all, lat0, lat1)
        iz = np.where(depth_all <= max_depth)[0]
        if iz.size == 0:
            raise ValueError(f"No depth <= {max_depth}")

        lon = lon_all[ix]
        lat = lat_all[iy]
        depth = depth_all[iz]
        dst.createDimension("depth", depth.size)
        dst.createDimension("latitude", lat.size)
        dst.createDimension("longitude", lon.size)
        dst.title = "Kuroshio hybrid 3D eddy input subset"
        dst.source_file = str(mat_path)
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
        u_var = dst.createVariable("u", "f4", ("depth", "latitude", "longitude"), zlib=True, complevel=int(conv.get("compression_level", 4)), fill_value=fill_value, chunksizes=chunksizes)
        v_var = dst.createVariable("v", "f4", ("depth", "latitude", "longitude"), zlib=True, complevel=int(conv.get("compression_level", 4)), fill_value=fill_value, chunksizes=chunksizes)
        adt_var = dst.createVariable("adt", "f4", ("latitude", "longitude"), zlib=True, complevel=int(conv.get("compression_level", 4)), fill_value=fill_value, chunksizes=(lat.size, lon.size))
        u_var.units = "m/s"
        v_var.units = "m/s"
        adt_var.units = "m"
        adt_var.long_name = "sea surface height for surface SLA eddy identification"

        y_slice = slice(int(iy[0]), int(iy[-1]) + 1)
        x_slice = slice(int(ix[0]), int(ix[-1]) + 1)
        adt = np.asarray(src[names["source_height"]][y_slice, x_slice], dtype="f4")
        np.nan_to_num(adt, copy=False, nan=fill_value, posinf=fill_value, neginf=fill_value)
        adt_var[:, :] = adt

        for out0 in range(0, depth.size, depth_block):
            out1 = min(out0 + depth_block, depth.size)
            z_indices = iz[out0:out1]
            label = f"{day.isoformat()} {mat_path.name} depth_index={int(z_indices[0])}:{int(z_indices[-1])}"
            try:
                z_sel = _depth_selector(z_indices)
                u_block = np.asarray(src[names["source_u"]][z_sel, y_slice, x_slice], dtype="f4")
                v_block = np.asarray(src[names["source_v"]][z_sel, y_slice, x_slice], dtype="f4")
            except OSError as exc:
                print(f"[read failed] {label}: {exc}")
                u_block, v_block = _read_velocity_block(mat_path, names, z_indices, y_slice, x_slice, retries, retry_sleep, label)
            np.nan_to_num(u_block, copy=False, nan=fill_value, posinf=fill_value, neginf=fill_value)
            np.nan_to_num(v_block, copy=False, nan=fill_value, posinf=fill_value, neginf=fill_value)
            u_var[out0:out1, :, :] = u_block
            v_var[out0:out1, :, :] = v_block

    tmp_path.replace(nc_path)


def convert_range(config_path: str | Path, start: str | None = None, end: str | None = None, force: bool = False) -> None:
    config = load_config(config_path)
    ensure_dirs(config)
    start_day = parse_ymd(start or config["date_range"]["start"])
    end_day = parse_ymd(end or config["date_range"]["end"])
    files = discover_mat_files(config["paths"]["input_mat_dir"], start_day, end_day)
    for item in tqdm(files, desc="Kuroshio UV+ADT subset", unit="day"):
        try:
            convert_one(item.path, output_path(config, item.date), item.date, config, force=force)
        except Exception as exc:
            raise RuntimeError(f"Failed converting {item.date.isoformat()} from {item.path}") from exc


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert MAT fields to Kuroshio UV+ADT NetCDF subsets.")
    parser.add_argument("--config", default="config/config_3d.yaml")
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    convert_range(args.config, args.start, args.end, force=args.force)


if __name__ == "__main__":
    main()
