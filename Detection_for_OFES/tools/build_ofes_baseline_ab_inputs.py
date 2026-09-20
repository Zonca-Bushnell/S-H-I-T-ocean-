"""Build five-day OFES surface inputs that differ only in the SSH baseline.

This diagnostic utility does not alter the production detection inputs.  It
constructs native-grid SSH first, subtracts either the historical Jan01-Jan19
mean or the 1993-2012 January climatology, then maps the anomaly to the
existing velocity grid.  Surface u/v are copied unchanged from the retained
Origin-compatible input.
"""

from __future__ import annotations

import argparse
import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

from Detection_for_OFES.ofes_io import ctl_path, parse_ctl, read_ssh_latlon_daily_only
from Detection_for_OFES.tools.export_origin_netcdf import regrid_scalar_to_velocity


DATA_ROOT = Path(r"F:\OFES\external_OFES2")
SOURCE_ROOT = Path(r"E:\DATA\01_Eddy_correspond\02_OFES\origin_compatible_filter")
CLIMATOLOGY_PATH = Path(
    r"E:\DATA\01_Eddy_correspond\02_OFES\ofes2_monthly_climatology_1993_2012"
    r"\climatology\ofes2_eta_pair_h_monthly_climatology_1993_2012.npz"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--source-root", type=Path, default=SOURCE_ROOT)
    parser.add_argument("--climatology-path", type=Path, default=CLIMATOLOGY_PATH)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--start", default="1991-01-01")
    parser.add_argument("--end", default="1991-01-05")
    parser.add_argument("--old-mean-start", default="1991-01-01")
    parser.add_argument("--old-mean-end", default="1991-01-19")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def parse_day(value: str) -> date:
    return date.fromisoformat(value)


def days_between(start: date, end: date) -> list[date]:
    return [start + timedelta(days=offset) for offset in range((end - start).days + 1)]


def native_mean(data_root: Path, days: list[date]) -> np.ndarray:
    total: np.ndarray | None = None
    count: np.ndarray | None = None
    for day in days:
        field = np.asarray(read_ssh_latlon_daily_only(data_root, day), dtype="f4")
        finite = np.isfinite(field)
        if total is None:
            total = np.zeros(field.shape, dtype="f8")
            count = np.zeros(field.shape, dtype="u1")
        total[finite] += field[finite]
        count[finite] += 1
        print(f"[baseline-ab-input] old baseline day {day.isoformat()}", flush=True)
    assert total is not None and count is not None
    return np.divide(total, count, out=np.full(total.shape, np.nan, dtype="f8"), where=count > 0).astype("f4")


def january_climatology(path: Path, eta_lon: np.ndarray, eta_lat: np.ndarray) -> np.ndarray:
    with np.load(path, allow_pickle=False) as data:
        months = np.asarray(data["months"])
        index = int(np.flatnonzero(months == 1)[0])
        lon = np.asarray(data["longitude"], dtype="f8")
        lat = np.asarray(data["latitude"], dtype="f8")
        h = np.asarray(data["h_monthly_mean_cm"][index], dtype="f4")
    if lon.shape != eta_lon.shape or lat.shape != eta_lat.shape or not np.allclose(lon, eta_lon) or not np.allclose(lat, eta_lat):
        raise RuntimeError("January climatology grid does not match the native OFES eta grid")
    h[~np.isfinite(h)] = np.nan
    return h


def source_metadata(source_root: Path, day: date) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, str]]:
    path = source_root / f"global_phy_{day:%Y%m%d}.nc"
    with Dataset(path) as ds:
        lon = np.asarray(ds.variables["longitude"][:], dtype="f4")
        lat = np.asarray(ds.variables["latitude"][:], dtype="f4")
        depth = np.asarray(ds.variables["depth"][:1], dtype="f4")
        u = np.asarray(ds.variables["uo_glor"][0, :1], dtype="f4")
        v = np.asarray(ds.variables["vo_glor"][0, :1], dtype="f4")
        time = np.asarray(ds.variables["time"][:], dtype="f8")
        attrs = {
            "time_units": str(getattr(ds.variables["time"], "units", "")),
            "time_calendar": str(getattr(ds.variables["time"], "calendar", "standard")),
            "u_units": str(getattr(ds.variables["uo_glor"], "units", "m s-1")),
            "v_units": str(getattr(ds.variables["vo_glor"], "units", "m s-1")),
        }
    for field in (u, v):
        field[np.abs(field) > 1.0e20] = np.nan
    return lon, lat, depth, time, np.stack((u, v)), attrs


def write_input(
    path: Path,
    lon: np.ndarray,
    lat: np.ndarray,
    depth: np.ndarray,
    time: np.ndarray,
    anomaly: np.ndarray,
    uv: np.ndarray,
    attrs: dict[str, str],
    definition: str,
    overwrite: bool,
) -> None:
    if path.exists() and not overwrite:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with Dataset(path, "w", format="NETCDF4") as ds:
        ds.createDimension("time", 1)
        ds.createDimension("depth", 1)
        ds.createDimension("latitude", len(lat))
        ds.createDimension("longitude", len(lon))
        t = ds.createVariable("time", "f8", ("time",))
        z = ds.createVariable("depth", "f4", ("depth",))
        y = ds.createVariable("latitude", "f4", ("latitude",))
        x = ds.createVariable("longitude", "f4", ("longitude",))
        ssh = ds.createVariable("zos_glor", "f4", ("time", "latitude", "longitude"), zlib=True, complevel=3)
        u = ds.createVariable("uo_glor", "f4", ("time", "depth", "latitude", "longitude"), zlib=True, complevel=3)
        v = ds.createVariable("vo_glor", "f4", ("time", "depth", "latitude", "longitude"), zlib=True, complevel=3)
        t.units, t.calendar = attrs["time_units"], attrs["time_calendar"]
        z.units, y.units, x.units = "m", "degrees_north", "degrees_east"
        ssh.units, u.units, v.units = "cm", attrs["u_units"], attrs["v_units"]
        ssh.long_name = "OFES SSH anomaly for baseline A/B diagnostic"
        u.long_name, v.long_name = "Copied surface zonal velocity anomaly", "Copied surface meridional velocity anomaly"
        ds.source = "build_ofes_baseline_ab_inputs"
        ds.baseline_definition = definition
        ds.velocity_policy = "Copied unchanged from origin_compatible_filter surface layer"
        t[:] = time
        z[:] = depth
        y[:] = lat
        x[:] = lon
        ssh[0] = anomaly
        u[0] = uv[0]
        v[0] = uv[1]


def main() -> None:
    args = parse_args()
    start, end = parse_day(args.start), parse_day(args.end)
    old_days = days_between(parse_day(args.old_mean_start), parse_day(args.old_mean_end))
    run_days = days_between(start, end)
    eta_meta = parse_ctl(ctl_path(args.data_root, "eta"))
    old_baseline = native_mean(args.data_root, old_days)
    long_baseline = january_climatology(args.climatology_path, eta_meta.x.values, eta_meta.y.values)
    definitions = {
        "old_jan01_jan19_mean": "H_OFES(day) - native-grid mean H_OFES(1991-01-01..1991-01-19)",
        "longterm_january_1993_2012": "H_OFES(day) - native-grid January H climatology(1993-2012)",
    }
    validation: list[dict[str, object]] = []
    for day in run_days:
        lon, lat, depth, time, uv, attrs = source_metadata(args.source_root, day)
        h = np.asarray(read_ssh_latlon_daily_only(args.data_root, day), dtype="f4")
        for label, baseline in (("old_jan01_jan19_mean", old_baseline), ("longterm_january_1993_2012", long_baseline)):
            native_anomaly = h - baseline
            native_anomaly[~np.isfinite(h) | ~np.isfinite(baseline)] = np.nan
            remapped = regrid_scalar_to_velocity(native_anomaly, eta_meta.x.values, eta_meta.y.values, lon, lat)
            output = args.output_root / label / "input" / f"global_phy_{day:%Y%m%d}.nc"
            write_input(output, lon, lat, depth, time, remapped, uv, attrs, definitions[label], args.overwrite)
            if label == "old_jan01_jan19_mean":
                with Dataset(args.source_root / f"global_phy_{day:%Y%m%d}.nc") as ds:
                    existing = np.asarray(ds.variables["zos_glor"][0], dtype="f4")
                common = np.isfinite(existing) & np.isfinite(remapped)
                delta = np.abs(existing[common] - remapped[common])
                validation.append({
                    "day": day.isoformat(), "finite_common_fraction": float(common.mean()),
                    "max_abs_delta_cm": float(delta.max()) if delta.size else float("nan"),
                    "rms_delta_cm": float(np.sqrt(np.mean(delta ** 2))) if delta.size else float("nan"),
                })
        print(f"[baseline-ab-input] wrote {day.isoformat()}", flush=True)
    manifest = {
        "status": "complete", "run_days": [day.isoformat() for day in run_days],
        "old_mean_days": [day.isoformat() for day in old_days], "definitions": definitions,
        "ssh_policy": "Construct H and subtract baseline on native eta grid before regridding.",
        "velocity_policy": "Copied unchanged from origin_compatible_filter surface u/v.",
        "old_baseline_validation_against_existing_origin_input": validation,
    }
    (args.output_root / "ab_input_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
