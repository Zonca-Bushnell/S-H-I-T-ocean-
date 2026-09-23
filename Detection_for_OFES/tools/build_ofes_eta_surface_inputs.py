"""Build surface detection inputs from OFES free-surface ``eta`` only."""
from __future__ import annotations

import argparse
import shutil
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

from Detection_for_OFES.ofes_io import ctl_path, parse_ctl, read_variable_latlon_daily_only
from Detection_for_OFES.tools.export_origin_netcdf import regrid_scalar_to_velocity


def parse_day(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def days(start: date, end: date) -> list[date]:
    return [start + timedelta(days=offset) for offset in range((end - start).days + 1)]


def build_eta_surface_inputs(data_root: Path, velocity_root: Path, output_root: Path, start: date, end: date) -> list[Path]:
    eta_meta = parse_ctl(ctl_path(data_root, "eta"))
    eta_lon = np.asarray(eta_meta.x.values, dtype="f8")
    eta_lat = np.asarray(eta_meta.y.values, dtype="f8")
    output_root.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for current in days(start, end):
        destination = output_root / f"global_phy_{current:%Y%m%d}.nc"
        if destination.exists():
            continue
        source = velocity_root / destination.name
        if not source.exists():
            raise FileNotFoundError(source)
        eta = np.asarray(read_variable_latlon_daily_only(data_root, "eta", current), dtype="f4")
        with Dataset(source) as source_ds:
            lon = np.asarray(source_ds.variables["longitude"][:], dtype="f8")
            lat = np.asarray(source_ds.variables["latitude"][:], dtype="f8")
        remapped = regrid_scalar_to_velocity(eta, eta_lon, eta_lat, lon, lat).astype("f4")
        # NetCDF is copied rather than rebuilt so u/v remain bitwise unchanged.
        shutil.copy2(source, destination)
        with Dataset(destination, "r+") as data:
            data.variables["zos_glor"][0] = remapped
            data.baseline_definition = "OFES eta free-surface height; no atmospheric-pressure subtraction"
            data.ssh_definition = "ofes_eta_free_surface"
            data.science_tag = "eta_gaussian_highpass_500km"
        written.append(destination)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--velocity-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    args = parser.parse_args()
    written = build_eta_surface_inputs(args.data_root, args.velocity_root, args.output_root, parse_day(args.start), parse_day(args.end))
    print("\n".join(str(path) for path in written), flush=True)


if __name__ == "__main__":
    main()
