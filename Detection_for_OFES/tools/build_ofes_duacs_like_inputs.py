"""Build OFES surface inputs referenced to the 1993-2012 annual MSS.

Only SSH changes relative to the retained Origin-compatible source files.
Velocity fields are copied byte-for-value as float32 arrays so the E-path
experiment changes one scientific input: the SSH reference surface.
"""

from __future__ import annotations

import argparse
import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

from Detection_for_OFES.ofes_io import ctl_path, parse_ctl, read_variable_latlon_daily_only
from Detection_for_OFES.tools.export_origin_netcdf import regrid_scalar_to_velocity


DATA_ROOT = Path(r"F:\OFES\external_OFES2")
SOURCE_ROOT = Path(r"E:\DATA\01_Eddy_correspond\02_OFES\origin_compatible_filter")
MSS_PATH = Path(
    r"E:\DATA\01_Eddy_correspond\02_OFES\ofes2_monthly_climatology_1993_2012"
    r"\climatology\ofes2_eta_annual_mss_1993_2012.npz"
)
OUTPUT_ROOT = Path(
    r"E:\DATA\01_Eddy_correspond\02_OFES"
    r"\origin_compatible_filter_eta_mss_1993_2012"
)
BASELINE_DEFINITION = "ofes_eta_annual_mss_1993_2012"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--source-root", type=Path, default=SOURCE_ROOT)
    parser.add_argument("--annual-mss-path", type=Path, default=MSS_PATH)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--start", default="1991-01-01")
    parser.add_argument("--end", default="1991-01-19")
    parser.add_argument("--max-depth-layers", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def days_between(start: date, end: date) -> list[date]:
    if end < start:
        raise ValueError("--end must not precede --start")
    return [start + timedelta(days=offset) for offset in range((end - start).days + 1)]


def read_annual_mss(
    path: Path,
    expected_lon: np.ndarray,
    expected_lat: np.ndarray,
) -> tuple[np.ndarray, dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError(f"Annual MSS cache does not exist: {path}")
    with np.load(path, allow_pickle=False) as data:
        required = {"longitude", "latitude", "eta_annual_mss_cm", "valid_day_weight", "months_used"}
        missing = required.difference(data.files)
        if missing:
            raise RuntimeError(f"Annual MSS cache is missing arrays: {sorted(missing)}")
        lon = np.asarray(data["longitude"], dtype="f8")
        lat = np.asarray(data["latitude"], dtype="f8")
        mss = np.asarray(data["eta_annual_mss_cm"], dtype="f4")
        weights = np.asarray(data["valid_day_weight"])
        months = np.asarray(data["months_used"]).astype(str)
    if not (
        lon.shape == expected_lon.shape
        and lat.shape == expected_lat.shape
        and mss.shape == (expected_lat.size, expected_lon.size)
        and np.allclose(lon, expected_lon, rtol=0.0, atol=1.0e-9)
        and np.allclose(lat, expected_lat, rtol=0.0, atol=1.0e-9)
    ):
        raise RuntimeError("Annual MSS grid does not match the native OFES eta grid")
    if months.size != 240 or str(months[0]) != "199301" or str(months[-1]) != "201212":
        raise RuntimeError("Annual MSS must contain every monthly mean from 199301 through 201212")
    if int(np.nanmax(weights)) != 7305:
        raise RuntimeError(f"Annual MSS maximum valid-day weight is {int(np.nanmax(weights))}, expected 7305")
    mss[~np.isfinite(mss)] = np.nan
    return mss, {
        "path": str(path),
        "shape": list(mss.shape),
        "months_used": int(months.size),
        "first_month": str(months[0]),
        "last_month": str(months[-1]),
        "maximum_valid_day_weight": int(np.nanmax(weights)),
        "minimum_positive_valid_day_weight": int(np.nanmin(weights[weights > 0])),
        "weighting": "actual calendar days per cached monthly mean with per-cell finite support",
    }


def copy_attributes(source, destination) -> None:
    for name in source.ncattrs():
        if name == "_FillValue":
            continue
        destination.setncattr(name, source.getncattr(name))


def write_day(
    *,
    source_path: Path,
    output_path: Path,
    ssh_anomaly: np.ndarray,
    depth_count: int,
    mss_path: Path,
) -> dict[str, object]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".partial")
    temporary.unlink(missing_ok=True)
    with Dataset(source_path) as source:
        lon = np.asarray(source.variables["longitude"][:], dtype="f4")
        lat = np.asarray(source.variables["latitude"][:], dtype="f4")
        depth = np.asarray(source.variables["depth"][:depth_count], dtype="f4")
        time = np.asarray(source.variables["time"][:], dtype="f8")
        if ssh_anomaly.shape != (lat.size, lon.size):
            raise RuntimeError(
                f"Remapped SSH shape {ssh_anomaly.shape} does not match source grid {(lat.size, lon.size)}"
            )
        with Dataset(temporary, "w", format="NETCDF4") as output:
            output.createDimension("time", 1)
            output.createDimension("depth", depth_count)
            output.createDimension("latitude", lat.size)
            output.createDimension("longitude", lon.size)
            time_out = output.createVariable("time", "f8", ("time",))
            depth_out = output.createVariable("depth", "f4", ("depth",))
            lat_out = output.createVariable("latitude", "f4", ("latitude",))
            lon_out = output.createVariable("longitude", "f4", ("longitude",))
            ssh_out = output.createVariable(
                "zos_glor", "f4", ("time", "latitude", "longitude"), zlib=True, complevel=3
            )
            u_out = output.createVariable(
                "uo_glor", "f4", ("time", "depth", "latitude", "longitude"),
                zlib=True, complevel=3, chunksizes=(1, 1, min(256, lat.size), min(512, lon.size)),
            )
            v_out = output.createVariable(
                "vo_glor", "f4", ("time", "depth", "latitude", "longitude"),
                zlib=True, complevel=3, chunksizes=(1, 1, min(256, lat.size), min(512, lon.size)),
            )
            for name, variable in (
                ("time", time_out), ("depth", depth_out), ("latitude", lat_out),
                ("longitude", lon_out), ("zos_glor", ssh_out), ("uo_glor", u_out), ("vo_glor", v_out),
            ):
                copy_attributes(source.variables[name], variable)
            ssh_out.long_name = "OFES eta anomaly relative to day-weighted 1993-2012 annual eta MSS"
            output.source = "OFES2 daily eta and retained Origin-compatible velocity input"
            output.science_tag = "ofes_eta_annual_mss_1993_2012"
            output.baseline_definition = BASELINE_DEFINITION
            output.baseline_formula = "SLA_E(day)=eta(day)-MSS_eta_1993_2012"
            output.atmospheric_pressure_policy = "no algebraic pair correction; physical response retained in eta"
            output.annual_mss_path = str(mss_path)
            output.annual_mss_months = 240
            output.annual_mss_maximum_valid_day_weight = 7305
            output.seasonal_cycle_policy = "retained"
            output.daily_global_mean_removal = "none"
            output.velocity_policy = "copied unchanged from origin_compatible_filter"
            output.grid_note = "SSH baseline removed on native eta grid, then remapped once to velocity grid"
            time_out[:] = time[:1]
            depth_out[:] = depth
            lat_out[:] = lat
            lon_out[:] = lon
            ssh_out[0] = np.asarray(ssh_anomaly, dtype="f4")
            for depth_index in range(depth_count):
                u_out[0, depth_index] = source.variables["uo_glor"][0, depth_index]
                v_out[0, depth_index] = source.variables["vo_glor"][0, depth_index]
    temporary.replace(output_path)
    return {
        "path": str(output_path),
        "finite_ssh_fraction": float(np.isfinite(ssh_anomaly).mean()),
        "ssh_min_cm": float(np.nanmin(ssh_anomaly)),
        "ssh_max_cm": float(np.nanmax(ssh_anomaly)),
        "depth_layers": depth_count,
    }


def main() -> None:
    args = parse_args()
    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    days = days_between(start, end)
    eta_meta = parse_ctl(ctl_path(args.data_root, "eta"))
    eta_lon = np.asarray(eta_meta.x.values, dtype="f8")
    eta_lat = np.asarray(eta_meta.y.values, dtype="f8")
    annual_mss, mss_meta = read_annual_mss(args.annual_mss_path, eta_lon, eta_lat)
    rows: list[dict[str, object]] = []
    for current_day in days:
        source_path = args.source_root / f"global_phy_{current_day:%Y%m%d}.nc"
        if not source_path.exists():
            raise FileNotFoundError(source_path)
        output_path = args.output_root / f"global_phy_{current_day:%Y%m%d}.nc"
        if output_path.exists() and not args.overwrite:
            rows.append({"day": current_day.isoformat(), "status": "existing", "path": str(output_path)})
            print(f"[duacs-like-input] existing {current_day}: {output_path}", flush=True)
            continue
        with Dataset(source_path) as source:
            velocity_lon = np.asarray(source.variables["longitude"][:], dtype="f8")
            velocity_lat = np.asarray(source.variables["latitude"][:], dtype="f8")
            source_depth_count = len(source.dimensions["depth"])
        depth_count = min(max(1, int(args.max_depth_layers)), source_depth_count)
        raw_eta = np.asarray(
            read_variable_latlon_daily_only(args.data_root, "eta", current_day), dtype="f4"
        )
        native_anomaly = np.asarray(raw_eta - annual_mss, dtype="f4")
        native_anomaly[~np.isfinite(raw_eta) | ~np.isfinite(annual_mss)] = np.nan
        remapped = regrid_scalar_to_velocity(
            native_anomaly, eta_lon, eta_lat, velocity_lon, velocity_lat
        )
        row = write_day(
            source_path=source_path,
            output_path=output_path,
            ssh_anomaly=remapped,
            depth_count=depth_count,
            mss_path=args.annual_mss_path,
        )
        row.update({"day": current_day.isoformat(), "status": "written"})
        rows.append(row)
        print(f"[duacs-like-input] wrote {current_day}: {output_path}", flush=True)

    manifest = {
        "status": "complete",
        "baseline_definition": BASELINE_DEFINITION,
        "formula": "SLA_E(day)=eta(day)-MSS_eta_1993_2012",
        "atmospheric_pressure_policy": "No pair subtraction; eta already contains the modeled free-surface response.",
        "baseline_application_order": "subtract on native eta grid, then remap once to velocity grid",
        "seasonal_cycle_policy": "retained",
        "daily_global_mean_removal": "none",
        "velocity_policy": "copied unchanged from origin_compatible_filter",
        "data_root": str(args.data_root),
        "source_root": str(args.source_root),
        "output_root": str(args.output_root),
        "start": start.isoformat(),
        "end": end.isoformat(),
        "mss": mss_meta,
        "outputs": rows,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "build_ofes_duacs_like_inputs_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
