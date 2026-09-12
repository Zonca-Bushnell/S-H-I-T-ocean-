from __future__ import annotations

import argparse
import json
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np

from ..ofes_io import (
    ctl_path,
    expected_dta_bytes,
    open_dta_memmap,
    parse_ctl,
    read_ssh_latlon_daily_only,
    require_daily_file,
)


DEFAULT_DATA_ROOT = Path(r"F:\OFES\external_OFES2")
DEFAULT_OUTPUT_DIR = Path(r"E:\DATA\01_Eddy_correspond\02_OFES\origin_compatible_filter")
UNDEF_ABS_THRESHOLD = 1.0e30


def main() -> None:
    args = build_parser().parse_args()
    export_origin_netcdf(
        data_root=Path(args.data_root),
        output_dir=Path(args.output_dir),
        start=parse_date(args.start),
        end=parse_date(args.end),
        mean_start=parse_date(args.mean_start),
        mean_end=parse_date(args.mean_end),
        max_depth_layers=int(args.max_depth_layers),
        depth_chunk=int(args.depth_chunk),
        overwrite=bool(args.overwrite),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export OFES daily .dta fields to Origin_eddy_detection-compatible NetCDF."
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--start", default="1991-01-01")
    parser.add_argument("--end", default="1991-01-19")
    parser.add_argument("--mean-start", default="1991-01-01")
    parser.add_argument("--mean-end", default="1991-01-19")
    parser.add_argument("--max-depth-layers", type=int, default=105)
    parser.add_argument("--depth-chunk", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def export_origin_netcdf(
    *,
    data_root: Path,
    output_dir: Path,
    start: date,
    end: date,
    mean_start: date,
    mean_end: date,
    max_depth_layers: int,
    depth_chunk: int,
    overwrite: bool,
) -> Path:
    if start.year != end.year:
        raise ValueError("This exporter writes one yearly NetCDF at a time; start/end must be in the same year.")
    if mean_start.year != start.year or mean_end.year != start.year:
        raise ValueError("The first exporter version expects the mean window to stay inside the exported year.")

    writer_backend = resolve_writer_backend()

    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / "mean_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"global_phy_{start.year:04d}.nc"
    if out_path.exists() and not overwrite:
        raise FileExistsError(f"{out_path} already exists. Use --overwrite to replace it.")

    eta_meta = parse_ctl(ctl_path(data_root, "eta"))
    u_meta = parse_ctl(ctl_path(data_root, "u"))
    v_meta = parse_ctl(ctl_path(data_root, "v"))
    depth_count = min(max_depth_layers, u_meta.z.count, v_meta.z.count)
    lon = u_meta.x.values.astype("f8")
    lat = u_meta.y.values.astype("f8")
    depth = u_meta.z.values[:depth_count].astype("f8")

    export_days = date_range(start, end)
    mean_days = date_range(mean_start, mean_end)
    ssh_mean = load_or_build_ssh_mean(data_root, cache_dir, mean_days, eta_meta, lon, lat)
    u_mean = load_or_build_velocity_mean(data_root, cache_dir, "u", mean_days, u_meta, depth_count, depth_chunk)
    v_mean = load_or_build_velocity_mean(data_root, cache_dir, "v", mean_days, v_meta, depth_count, depth_chunk)

    if out_path.exists():
        out_path.unlink()
    units = f"days since {start.year:04d}-01-01 00:00:00"
    ds = open_output_dataset(out_path, writer_backend)
    try:
        ds.createDimension("time", len(export_days))
        ds.createDimension("depth", depth_count)
        ds.createDimension("latitude", len(lat))
        ds.createDimension("longitude", len(lon))

        time_var = create_output_variable(ds, writer_backend, "time", "f8", ("time",))
        depth_var = create_output_variable(ds, writer_backend, "depth", "f4", ("depth",))
        lat_var = create_output_variable(ds, writer_backend, "latitude", "f4", ("latitude",))
        lon_var = create_output_variable(ds, writer_backend, "longitude", "f4", ("longitude",))
        zos = create_output_variable(ds, writer_backend, "zos_glor", "f4", ("time", "latitude", "longitude"))
        uo = create_output_variable(ds, writer_backend, "uo_glor", "f4", ("time", "depth", "latitude", "longitude"))
        vo = create_output_variable(ds, writer_backend, "vo_glor", "f4", ("time", "depth", "latitude", "longitude"))

        time_var.units = units
        time_var.calendar = "standard"
        depth_var.units = "m"
        lat_var.units = "degrees_north"
        lon_var.units = "degrees_east"
        zos.units = "cm"
        zos.long_name = "OFES sea surface height anomaly on velocity grid"
        uo.units = "m s-1"
        vo.units = "m s-1"
        uo.long_name = "OFES zonal velocity anomaly"
        vo.long_name = "OFES meridional velocity anomaly"
        ds.source = "OFES2 daily .dta via Detection_for_OFES data adapter"
        ds.science_tag = "raw_minus_jan_mean_diagnostic"
        ds.mean_window_start = mean_start.isoformat()
        ds.mean_window_end = mean_end.isoformat()
        ds.grid_note = "All exported variables are on the OFES velocity grid for Origin_eddy_detection."

        time_var[:] = np.asarray([(day - date(start.year, 1, 1)).days for day in export_days], dtype="f8")
        depth_var[:] = depth
        lat_var[:] = lat
        lon_var[:] = lon

        for t_index, day in enumerate(export_days):
            ssh = read_ssh_latlon_daily_only(data_root, day)
            zos[t_index, :, :] = regrid_scalar_to_velocity(ssh, eta_meta.x.values, eta_meta.y.values, lon, lat) - ssh_mean
            write_velocity_day(uo, t_index, data_root, "u", day, u_meta, u_mean, depth_count, depth_chunk)
            write_velocity_day(vo, t_index, data_root, "v", day, v_meta, v_mean, depth_count, depth_chunk)
            print(f"[export-origin-netcdf] wrote {day.isoformat()}", flush=True)
    finally:
        ds.close()

    manifest = {
        "output_path": str(out_path),
        "data_root": str(data_root),
        "start": start.isoformat(),
        "end": end.isoformat(),
        "mean_start": mean_start.isoformat(),
        "mean_end": mean_end.isoformat(),
        "max_depth_layers": depth_count,
        "writer_backend": writer_backend,
        "dimensions": {
            "time": len(export_days),
            "depth": depth_count,
            "latitude": len(lat),
            "longitude": len(lon),
        },
        "variables": {
            "zos_glor": "SSH anomaly in cm, regridded from OFES scalar grid to velocity grid",
            "uo_glor": "u anomaly in m/s",
            "vo_glor": "v anomaly in m/s",
        },
    }
    (output_dir / "export_origin_netcdf_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return out_path


def resolve_writer_backend() -> str:
    try:
        import netCDF4  # noqa: F401

        return "netcdf4"
    except Exception as exc:
        print(f"[export-origin-netcdf] netCDF4 unavailable, using scipy NetCDF3 fallback: {exc}", flush=True)
        try:
            from scipy.io import netcdf_file  # noqa: F401
        except Exception as scipy_exc:
            raise RuntimeError(
                "NetCDF export needs either netCDF4 or scipy.io.netcdf_file in the active environment."
            ) from scipy_exc
        return "scipy_netcdf3_64bit"


def open_output_dataset(path: Path, writer_backend: str):
    if writer_backend == "netcdf4":
        from netCDF4 import Dataset

        return Dataset(path, "w", format="NETCDF4")
    from scipy.io import netcdf_file

    return netcdf_file(str(path), "w", version=2)


def create_output_variable(ds, writer_backend: str, name: str, dtype: str, dimensions: tuple[str, ...]):
    if writer_backend == "netcdf4":
        if len(dimensions) == 4:
            return ds.createVariable(
                name,
                dtype,
                dimensions,
                zlib=True,
                complevel=3,
                chunksizes=(1, min(4, len(ds.dimensions["depth"])), min(256, len(ds.dimensions["latitude"])), min(512, len(ds.dimensions["longitude"]))),
            )
        if name == "zos_glor":
            return ds.createVariable(name, dtype, dimensions, zlib=True, complevel=3)
        return ds.createVariable(name, dtype, dimensions)
    scipy_dtype = "d" if dtype == "f8" else "f"
    return ds.createVariable(name, scipy_dtype, dimensions)


def load_or_build_ssh_mean(
    data_root: Path,
    cache_dir: Path,
    days: list[date],
    eta_meta,
    lon_target: np.ndarray,
    lat_target: np.ndarray,
) -> np.ndarray:
    cache_path = cache_dir / f"ssh_mean_{days[0]:%Y%m%d}_{days[-1]:%Y%m%d}_velocity_grid_cm.npy"
    if cache_path.exists():
        return np.load(cache_path)
    total = np.zeros((len(lat_target), len(lon_target)), dtype="f8")
    count = np.zeros_like(total, dtype="f8")
    for day in days:
        ssh = read_ssh_latlon_daily_only(data_root, day)
        layer = regrid_scalar_to_velocity(ssh, eta_meta.x.values, eta_meta.y.values, lon_target, lat_target)
        finite = np.isfinite(layer)
        total[finite] += layer[finite]
        count[finite] += 1.0
    mean = np.divide(total, count, out=np.full_like(total, np.nan, dtype="f8"), where=count > 0).astype("f4")
    np.save(cache_path, mean)
    return mean


def load_or_build_velocity_mean(
    data_root: Path,
    cache_dir: Path,
    variable: str,
    days: list[date],
    meta,
    depth_count: int,
    depth_chunk: int,
) -> np.memmap:
    mean_path = cache_dir / f"{variable}_mean_{days[0]:%Y%m%d}_{days[-1]:%Y%m%d}_cms.dat"
    shape = (meta.x.count, meta.y.count, depth_count)
    if mean_path.exists() and mean_path.stat().st_size == int(np.prod(shape) * np.dtype("f4").itemsize):
        return np.memmap(mean_path, dtype="float32", mode="r", shape=shape, order="F")
    mean = np.memmap(mean_path, dtype="float32", mode="w+", shape=shape, order="F")
    expected = expected_dta_bytes(meta)
    for k0 in range(0, depth_count, max(1, depth_chunk)):
        k1 = min(depth_count, k0 + max(1, depth_chunk))
        total = np.zeros((meta.x.count, meta.y.count, k1 - k0), dtype="f8")
        count = np.zeros_like(total, dtype="f8")
        for day in days:
            raw = open_dta_memmap(require_daily_file(data_root, variable, day, expected), meta)
            block = np.array(raw[:, :, k0:k1], dtype="f4", copy=True)
            block[np.abs(block) > UNDEF_ABS_THRESHOLD] = np.nan
            finite = np.isfinite(block)
            total[finite] += block[finite]
            count[finite] += 1.0
        mean[:, :, k0:k1] = np.divide(total, count, out=np.full_like(total, np.nan), where=count > 0).astype("f4")
        mean.flush()
    return np.memmap(mean_path, dtype="float32", mode="r", shape=shape, order="F")


def write_velocity_day(
    out_var,
    t_index: int,
    data_root: Path,
    variable: str,
    day: date,
    meta,
    mean: np.memmap,
    depth_count: int,
    depth_chunk: int,
) -> None:
    raw = open_dta_memmap(require_daily_file(data_root, variable, day, expected_dta_bytes(meta)), meta)
    for k0 in range(0, depth_count, max(1, depth_chunk)):
        k1 = min(depth_count, k0 + max(1, depth_chunk))
        block = np.array(raw[:, :, k0:k1], dtype="f4", copy=True)
        block[np.abs(block) > UNDEF_ABS_THRESHOLD] = np.nan
        anomaly_m_s = (block - np.asarray(mean[:, :, k0:k1], dtype="f4")) / 100.0
        out_var[t_index, k0:k1, :, :] = np.transpose(anomaly_m_s, (2, 1, 0))


def regrid_scalar_to_velocity(
    field_latlon: np.ndarray,
    lon_source: np.ndarray,
    lat_source: np.ndarray,
    lon_target: np.ndarray,
    lat_target: np.ndarray,
) -> np.ndarray:
    source = np.asarray(field_latlon, dtype="f8")
    lon_source = np.asarray(lon_source, dtype="f8")
    lat_source = np.asarray(lat_source, dtype="f8")
    lon_target = np.asarray(lon_target, dtype="f8")
    lat_target = np.asarray(lat_target, dtype="f8")
    lon_ext = np.concatenate([lon_source, [lon_source[0] + 360.0]])
    source_ext = np.concatenate([source, source[:, :1]], axis=1)
    lon_interp = np.empty((source_ext.shape[0], len(lon_target)), dtype="f8")
    wrapped_target = np.where(lon_target < lon_source[0], lon_target + 360.0, lon_target)
    for row in range(source_ext.shape[0]):
        lon_interp[row, :] = np.interp(wrapped_target, lon_ext, source_ext[row, :], left=np.nan, right=np.nan)
    out = np.empty((len(lat_target), len(lon_target)), dtype="f8")
    for col in range(lon_interp.shape[1]):
        out[:, col] = np.interp(lat_target, lat_source, lon_interp[:, col], left=np.nan, right=np.nan)
    return out.astype("f4")


def date_range(start: date, end: date) -> list[date]:
    if end < start:
        raise ValueError("end must be >= start")
    days = []
    current = start
    while current <= end:
        days.append(current)
        current += timedelta(days=1)
    return days


def parse_date(value: str) -> date:
    return datetime.strptime(str(value)[:10], "%Y-%m-%d").date()


if __name__ == "__main__":
    main()
