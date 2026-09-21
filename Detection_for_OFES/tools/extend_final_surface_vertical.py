"""Extend the final OFES surface catalog downward without re-running surface detection.

The implementation deliberately keeps the existing Hua layer test untouched.  Its
only algorithmic change is the loop order: one depth layer is prepared once and
then checked for every still-active surface object.
"""
from __future__ import annotations

import argparse
import json
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from netCDF4 import Dataset

from Origin_eddy_detection.src.eddy_pipeline.detection_hybrid import (
    DetectionParams,
    _grid_spacing_km,
    _hua_verify_radius,
    _object_voxels_for_layer,
    _refine_speed_min_subgrid,
    _seeded_speed_min,
)


DEFAULT_SURFACE_ROOT = Path(
    r"E:\DATA\01_Eddy_correspond\02_OFES"
    r"\origin_unified_eta_mss_ssh_primary_target_all_open_ocean_surface_jan01_jan19"
)
DEFAULT_FILTER_ROOT = Path(
    r"E:\DATA\01_Eddy_correspond\02_OFES\origin_compatible_filter_rossby_lower_upper180_full105"
)
DEFAULT_OUTPUT_ROOT = Path(
    r"E:\DATA\01_Eddy_correspond\02_OFES"
    r"\origin_unified_eta_mss_ssh_primary_target_all_open_ocean_vertical_jan01_jan19"
)


def parse_day(value: str) -> date:
    return datetime.strptime(value[:10], "%Y-%m-%d").date()


def day_path(root: Path, day: date) -> Path:
    return root / "final_catalog" / "daily_runs" / f"{day:%Y%m%d}" / "centers_hua_style.csv"


def write_table(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path.with_suffix(".parquet"), index=False)
    frame.to_csv(path.with_suffix(".csv"), index=False)


def params_from_args(args: argparse.Namespace) -> DetectionParams:
    # These are the existing b3/start2 deep Hua values.  SSH-primary is a
    # surface definition only; with ssh=None `_hua_verify_radius` takes this
    # same strict circle branch used by the hybrid detector at depth > 0.
    return DetectionParams(
        ssh_window_cells=7,
        surface_search_cells=8,
        deep_search_cells=int(args.deep_search_cells),
        start_radius_cells=int(args.start_radius_cells),
        max_radius_cells=int(args.max_radius_cells),
        speed_ratio_max=3.0,
        angle_jump_max_deg=150.0,
        tangent_tolerance_deg=24.0,
        symmetry_tolerance_deg=120.0,
        min_tangent_fraction=0.70,
        min_reversal_fraction=0.70,
        min_finite_fraction=0.95,
        direction_exception_extra=0,
        boundary_mode="ssh_primary_open_ocean_no_streamline_gate",
    )


def surface_rows(root: Path, day: date) -> pd.DataFrame:
    path = day_path(root, day)
    if not path.exists():
        raise FileNotFoundError(path)
    rows = pd.read_csv(path)
    rows = rows[rows.get("hua_pass", pd.Series(False, index=rows.index)).fillna(False).astype(bool)].copy()
    if "persistence_class" in rows:
        rows = rows[~rows["persistence_class"].astype(str).eq("transient")].copy()
    return rows.reset_index(drop=True)


def row_center_i(row: pd.Series) -> int:
    for name in ("center_i_refined", "speed_min_i", "speed_min_i_grid", "seed_i"):
        value = pd.to_numeric(pd.Series([row.get(name)]), errors="coerce").iloc[0]
        if np.isfinite(value):
            return int(round(float(value)))
    raise ValueError(f"Missing center i for {row.get('hua_object_id')}")


def row_center_j(row: pd.Series) -> int:
    for name in ("center_j_refined", "speed_min_j", "speed_min_j_grid", "seed_j"):
        value = pd.to_numeric(pd.Series([row.get(name)]), errors="coerce").iloc[0]
        if np.isfinite(value):
            return int(round(float(value)))
    raise ValueError(f"Missing center j for {row.get('hua_object_id')}")


def make_surface_record(row: pd.Series, *, day: date) -> dict[str, object]:
    out = row.to_dict()
    out.update(
        {
            "date": day.isoformat(),
            "depth_index": 0,
            "depth_m": float(row.get("depth_m", 2.5)),
            "vertical_extension_source": "final_surface_catalog",
            "vertical_extension_algorithm": "existing_hua_depth_continuation",
            "vertical_extension_stop_reason": "",
            "hua_pass": True,
        }
    )
    return out


def extend_day(args: argparse.Namespace, day: date) -> dict[str, object]:
    source = surface_rows(args.surface_root, day)
    if int(args.max_surface_objects) > 0:
        source = source.sort_values(["hua_object_id"]).head(int(args.max_surface_objects)).copy()
    out_dir = args.output_root / "raw_detection" / "daily_runs" / f"{day:%Y%m%d}"
    if args.resume and (out_dir / "centers_hua_style.parquet").exists():
        return {"day": day.isoformat(), "status": "resume"}
    input_path = args.filter_root / f"global_phy_{day:%Y%m%d}.nc"
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    params = params_from_args(args)
    centers: list[dict[str, object]] = [make_surface_record(row, day=day) for _, row in source.iterrows()]
    structures: list[dict[str, object]] = []
    voxels: list[dict[str, object]] = []
    states: list[dict[str, object]] = []

    with Dataset(input_path) as ds:
        lon = np.asarray(ds.variables["longitude"][:], dtype="f8")
        lat = np.asarray(ds.variables["latitude"][:], dtype="f8")
        depth = np.asarray(ds.variables["depth"][:], dtype="f8")
        dx_km, dy_km = _grid_spacing_km(lon, lat)
        max_layers = min(int(args.max_depth_layers), len(depth))
        u0 = np.asarray(ds.variables["uo_glor"][0, 0], dtype="f4")
        v0 = np.asarray(ds.variables["vo_glor"][0, 0], dtype="f4")

        for _, row in source.iterrows():
            object_id = str(row["hua_object_id"])
            ci, cj = row_center_i(row), row_center_j(row)
            radius = float(row.get("accepted_radius_cells", row.get("radius_cells", 0.0)))
            radius_km = float(row.get("radius_km", np.nan))
            structures.append({
                "date": day.isoformat(), "hua_object_id": object_id, "depth_index": 0,
                "depth_m": float(depth[0]), "center_lon": float(row.get("center_lon", lon[ci])),
                "center_lat": float(row.get("center_lat", lat[cj])), "center_lon_grid": float(lon[ci]),
                "center_lat_grid": float(lat[cj]), "center_lon_refined": float(row.get("center_lon_refined", lon[ci])),
                "center_lat_refined": float(row.get("center_lat_refined", lat[cj])),
                "refined_ok": bool(row.get("refined_ok", True)), "refined_offset_km": float(row.get("refined_offset_km", 0.0)),
                "radius_km": radius_km if np.isfinite(radius_km) else radius * float(np.nanmean([dx_km, dy_km])),
                "polarity": str(row.get("polarity", "")),
            })
            if np.isfinite(radius) and radius > 0:
                voxels.extend(
                    _object_voxels_for_layer(
                        u0, v0, lon, lat, float(depth[0]), day=day, object_id=object_id,
                        depth_index=0, center_i=ci, center_j=cj, radius_cells=radius,
                        polarity=str(row.get("polarity", "")),
                    )
                )
            states.append({"row": row, "object_id": object_id, "prev_i": ci, "prev_j": cj})

        active = states
        for depth_index in range(1, max_layers):
            if not active:
                break
            u = np.asarray(ds.variables["uo_glor"][0, depth_index], dtype="f4")
            v = np.asarray(ds.variables["vo_glor"][0, depth_index], dtype="f4")
            speed = np.hypot(u, v)
            next_active: list[dict[str, object]] = []
            for state in active:
                source_row = state["row"]
                center_i, center_j, center_speed, min_steps = _seeded_speed_min(
                    speed, int(state["prev_i"]), int(state["prev_j"]), params.deep_search_cells
                )
                refined = _refine_speed_min_subgrid(
                    speed, u, v, lon, lat, center_i, center_j,
                    target_degree=1.0 / 24.0, window_radius_cells=2, min_finite_fraction=0.6,
                )
                check = _hua_verify_radius(
                    u, v, float(refined["center_i_refined"]), float(refined["center_j_refined"]), params,
                    ssh=None, speed=speed, extremum_type=str(source_row.get("ssh_extremum_type", "")), lon=lon, lat=lat,
                )
                passed = bool(check.get("hua_pass", False))
                record = source_row.to_dict()
                record.update(check)
                record.update(
                    {
                        "date": day.isoformat(), "hua_object_id": state["object_id"],
                        "depth_index": int(depth_index), "depth_m": float(depth[depth_index]),
                        "speed_min_i": int(round(float(refined["center_i_refined"]))),
                        "speed_min_j": int(round(float(refined["center_j_refined"]))),
                        "speed_min_i_grid": int(center_i), "speed_min_j_grid": int(center_j),
                        "center_lon_grid": float(lon[center_i]), "center_lat_grid": float(lat[center_j]),
                        "center_lon": float(refined["center_lon_refined"]), "center_lat": float(refined["center_lat_refined"]),
                        "center_i_refined": float(refined["center_i_refined"]), "center_j_refined": float(refined["center_j_refined"]),
                        "center_lon_refined": float(refined["center_lon_refined"]), "center_lat_refined": float(refined["center_lat_refined"]),
                        "center_speed_ms": float(refined["refined_speed_ms"] if refined["refined_ok"] else center_speed),
                        "center_speed_grid_ms": float(center_speed), "refined_ok": bool(refined["refined_ok"]),
                        "refined_offset_km": float(refined["refined_offset_km"]),
                        "subgrid_fit_quality": str(refined["subgrid_fit_quality"]), "local_min_steps": int(min_steps),
                        "vertical_extension_source": "final_surface_catalog",
                        "vertical_extension_algorithm": "existing_hua_depth_continuation",
                        "vertical_extension_stop_reason": "" if passed else str(check.get("first_hard_failure", "hua_failed")),
                    }
                )
                centers.append(record)
                if not passed:
                    continue
                radius = float(check.get("accepted_radius_cells", 0.0))
                structures.append({
                    "date": day.isoformat(), "hua_object_id": state["object_id"], "depth_index": int(depth_index),
                    "depth_m": float(depth[depth_index]), "center_lon": record["center_lon"], "center_lat": record["center_lat"],
                    "center_lon_grid": record["center_lon_grid"], "center_lat_grid": record["center_lat_grid"],
                    "center_lon_refined": record["center_lon_refined"], "center_lat_refined": record["center_lat_refined"],
                    "refined_ok": record["refined_ok"], "refined_offset_km": record["refined_offset_km"],
                    "radius_km": radius * float(np.nanmean([dx_km, dy_km])), "polarity": str(source_row.get("polarity", "")),
                })
                voxels.extend(_object_voxels_for_layer(
                    u, v, lon, lat, float(depth[depth_index]), day=day, object_id=state["object_id"],
                    depth_index=int(depth_index), center_i=center_i, center_j=center_j, radius_cells=radius,
                    polarity=str(source_row.get("polarity", "")),
                ))
                state["prev_i"] = int(np.clip(round(float(refined["center_i_refined"])), 0, len(lon) - 1))
                state["prev_j"] = int(np.clip(round(float(refined["center_j_refined"])), 0, len(lat) - 1))
                next_active.append(state)
            active = next_active
            print(f"[vertical] {day.isoformat()} depth={depth_index}/{max_layers - 1} active={len(active)}", flush=True)

    centers_df = pd.DataFrame(centers)
    structures_df = pd.DataFrame(structures)
    voxels_df = pd.DataFrame(voxels)
    write_table(centers_df, out_dir / "centers_hua_style")
    write_table(structures_df, out_dir / "structures_hua_style")
    voxel_path = args.output_root / "raw_detection" / "object_voxels_parts" / f"year={day.year}" / f"date={day:%Y%m%d}.parquet"
    voxel_path.parent.mkdir(parents=True, exist_ok=True)
    voxels_df.to_parquet(voxel_path, index=False)
    summary = {
        "day": day.isoformat(), "surface_objects": int(len(source)), "center_rows": int(len(centers_df)),
        "passed_rows": int(centers_df["hua_pass"].fillna(False).astype(bool).sum()), "voxels": int(len(voxels_df)),
        "algorithm": "depth_major_existing_hua", "max_depth_layers": int(args.max_depth_layers),
    }
    (out_dir / "vertical_extension_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Depth-major Hua continuation from the final OFES surface catalog.")
    parser.add_argument("--surface-root", type=Path, default=DEFAULT_SURFACE_ROOT)
    parser.add_argument("--filter-root", type=Path, default=DEFAULT_FILTER_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--day", required=True)
    parser.add_argument("--max-depth-layers", type=int, default=105)
    parser.add_argument("--deep-search-cells", type=int, default=6)
    parser.add_argument("--start-radius-cells", type=int, default=2)
    parser.add_argument("--max-radius-cells", type=int, default=12)
    parser.add_argument("--max-surface-objects", type=int, default=0, help="Smoke-only cap; <=0 keeps every final surface object.")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    print(json.dumps(extend_day(args, parse_day(args.day)), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
