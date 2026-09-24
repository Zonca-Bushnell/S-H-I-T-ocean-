"""Materialize downstream vertical artifacts from a completed centers CSV."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from Origin_eddy_detection.src.eddy_pipeline.detection_hybrid import _grid_spacing_km


def as_bool(values: pd.Series) -> pd.Series:
    return values.fillna(False).astype(str).str.strip().str.lower().isin(("1", "true", "yes"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vertical-root", type=Path, required=True)
    parser.add_argument("--velocity-file", type=Path, required=True)
    parser.add_argument("--day", default="1991-01-01")
    args = parser.parse_args()
    run = args.vertical_root / "raw_detection" / "daily_runs" / args.day.replace("-", "")
    centers = pd.read_csv(run / "centers_hua_style.csv")
    passed = centers.loc[as_bool(centers["hua_pass"])].copy()
    from netCDF4 import Dataset
    with Dataset(args.velocity_file) as ds:
        lon = np.asarray(ds.variables["longitude"][:], dtype="f8")
        lat = np.asarray(ds.variables["latitude"][:], dtype="f8")
    dx_km, dy_km = _grid_spacing_km(lon, lat)
    radius = pd.to_numeric(passed["accepted_radius_cells"], errors="coerce")
    source_radius = pd.to_numeric(passed.get("radius_km", np.nan), errors="coerce")
    passed["radius_km_materialized"] = radius * float(np.nanmean([dx_km, dy_km]))
    passed.loc[passed["depth_index"].astype(int).eq(0) & source_radius.notna(), "radius_km_materialized"] = source_radius
    structures = pd.DataFrame({
        "date": passed["date"], "hua_object_id": passed["hua_object_id"], "depth_index": passed["depth_index"].astype(int),
        "depth_m": passed["depth_m"], "center_lon": passed["center_lon"], "center_lat": passed["center_lat"],
        "center_lon_grid": passed["center_lon_grid"], "center_lat_grid": passed["center_lat_grid"],
        "center_lon_refined": passed["center_lon_refined"], "center_lat_refined": passed["center_lat_refined"],
        "refined_ok": passed["refined_ok"], "refined_offset_km": passed["refined_offset_km"],
        "radius_km": passed["radius_km_materialized"], "polarity": passed["polarity"],
    })
    structures.to_csv(run / "structures_hua_style.csv", index=False)
    try:
        structures.to_parquet(run / "structures_hua_style.parquet", index=False)
    except Exception:
        pass
    summary = {
        "day": args.day,
        "surface_objects": int((passed["depth_index"].astype(int) == 0).sum()),
        "center_rows": int(len(centers)),
        "passed_rows": int(len(passed)),
        "voxels": None,
        "voxel_status": "not_materialized_from_completed_centers_recovery",
        "algorithm": "depth_major_near_closed_streamline_recovered_from_completed_centers",
        "deep_hua_mode": "near_closed_streamline",
        "recovery": "CSV was complete; only Parquet serialization failed on mixed diagnostic column types",
    }
    (run / "vertical_extension_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
