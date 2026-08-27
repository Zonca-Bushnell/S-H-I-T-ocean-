import argparse
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import xarray as xr

from acc_config import (
    MAX_DEPTH,
    MAX_LATITUDE,
    MAX_LONGITUDE,
    MIN_DEPTH,
    MIN_LATITUDE,
    MIN_LONGITUDE,
    OUTPUT_DIR,
    RAW_DIR,
    VARIABLES,
    build_time_windows,
)


DATE_RE = re.compile(r"(?<!\d)((?:19|20)\d{2})(\d{2})?(\d{2})?(?!\d)")


def coord_name(ds, candidates):
    for name in candidates:
        if name in ds.coords or name in ds.variables or name in ds.dims:
            return name
    return None


def path_may_overlap(path, start, end):
    matches = list(DATE_RE.finditer(path.name))
    if not matches:
        return True

    for match in matches:
        year = int(match.group(1))
        month = int(match.group(2) or 1)
        day = int(match.group(3) or 1)
        try:
            value = datetime(year, month, day)
        except ValueError:
            continue

        if match.group(3):
            if start <= value < end:
                return True
        elif match.group(2):
            next_month = datetime(year + (month == 12), 1 if month == 12 else month + 1, 1)
            if value < end and next_month > start:
                return True
        else:
            next_year = datetime(year + 1, 1, 1)
            if value < end and next_year > start:
                return True

    return False


def find_raw_files(raw_dir, start, end):
    files = sorted(raw_dir.rglob("*.nc"))
    files = [path for path in files if path_may_overlap(path, start, end)]
    if not files:
        raise FileNotFoundError(f"No NetCDF files found for {start:%Y-%m-%d} to {end:%Y-%m-%d}")
    return files


def open_raw_dataset(files):
    kwargs = {
        "combine": "by_coords",
        "parallel": False,
        "data_vars": "minimal",
        "coords": "minimal",
        "compat": "override",
        "chunks": {"time": 1},
    }
    try:
        return xr.open_mfdataset(files, **kwargs)
    except Exception:
        kwargs.pop("data_vars", None)
        kwargs.pop("coords", None)
        kwargs.pop("compat", None)
        return xr.open_mfdataset(files, combine="by_coords", parallel=False, chunks={"time": 1})


def strict_spatial_subset(ds):
    lon_name = coord_name(ds, ["longitude", "lon", "nav_lon"])
    lat_name = coord_name(ds, ["latitude", "lat", "nav_lat"])
    depth_name = coord_name(ds, ["depth", "deptht", "lev", "level"])

    if lon_name:
        lon = ds[lon_name]
        signed_lon = ((lon + 180) % 360) - 180
        ds = ds.where((signed_lon > MIN_LONGITUDE) & (signed_lon < MAX_LONGITUDE), drop=True)
    else:
        print("Warning: longitude coordinate not found; longitude crop skipped.")

    if lat_name:
        lat = ds[lat_name]
        ds = ds.where((lat > MIN_LATITUDE) & (lat < MAX_LATITUDE), drop=True)
    else:
        print("Warning: latitude coordinate not found; latitude crop skipped.")

    if depth_name:
        depth = ds[depth_name]
        ds = ds.where((depth > MIN_DEPTH) & (depth < MAX_DEPTH), drop=True)
    else:
        print("Warning: depth coordinate not found; depth crop skipped.")

    return ds


def crop_time(ds, start, end):
    time_name = coord_name(ds, ["time", "time_counter"])
    if not time_name:
        print("Warning: time coordinate not found; time crop skipped.")
        return ds

    start64 = np.datetime64(start)
    end64 = np.datetime64(end)
    return ds.where((ds[time_name] >= start64) & (ds[time_name] < end64), drop=True)


def select_variables(ds):
    available = [name for name in VARIABLES if name in ds.data_vars]
    missing = [name for name in VARIABLES if name not in ds.data_vars]
    if missing:
        print("Warning: missing variables in raw files: " + ", ".join(missing))
    if not available:
        raise KeyError("None of the requested variables were found in the raw files.")
    return ds[available]


def write_window(raw_dir, output_dir, start, end, filename):
    files = find_raw_files(raw_dir, start, end)
    print(f"Opening {len(files)} raw files for {start:%Y-%m-%d} to {end:%Y-%m-%d}")

    ds = open_raw_dataset([str(path) for path in files])
    try:
        ds = crop_time(ds, start, end)
        ds = strict_spatial_subset(ds)
        ds = select_variables(ds)

        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / filename
        encoding = {
            name: {"zlib": True, "complevel": 1}
            for name in ds.data_vars
        }
        print(f"Writing {output_path}")
        ds.to_netcdf(output_path, engine="netcdf4", format="NETCDF4", encoding=encoding)
    finally:
        ds.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Crop raw Copernicus Marine NetCDF files locally.")
    parser.add_argument("--sample", action="store_true", help="Use SAMPLE_START/SAMPLE_END.")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def main():
    args = parse_args()
    for start, end, filename in build_time_windows(args.sample):
        write_window(args.raw_dir, args.output_dir, start, end, filename)
    print("Local crop completed.")


if __name__ == "__main__":
    main()
