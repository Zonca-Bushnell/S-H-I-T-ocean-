"""Extend the final OFES surface catalog downward without re-running surface detection.

The implementation deliberately keeps the existing Hua layer test untouched.  Its
only algorithmic change is the loop order: one depth layer is prepared once and
then checked for every still-active surface object.
"""
from __future__ import annotations

import argparse
import json
from contextlib import nullcontext
from datetime import date, datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy.ndimage import map_coordinates

from Origin_eddy_detection.src.eddy_pipeline.detection_hybrid import (
    DetectionParams,
    EARTH_RADIUS_M,
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
    frame.to_csv(path.with_suffix(".csv"), index=False)
    # CSV is the portable primary artifact.  Some Windows hosts block native
    # Parquet DLLs even though the numerical NetCDF stack is available.
    try:
        frame.to_parquet(path.with_suffix(".parquet"), index=False)
    except Exception:
        pass


def open_binary_velocity_dataset(root: Path, day: date) -> SimpleNamespace:
    daily = root / f"{day:%Y%m%d}"
    metadata_path = daily / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(metadata_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    nz, ny, nx = (int(value) for value in metadata["shape_depth_lat_lon"])
    expected = nz * ny * nx * np.dtype("f4").itemsize
    u_path, v_path = daily / "uo_glor.f32", daily / "vo_glor.f32"
    for path in (u_path, v_path):
        if path.stat().st_size != expected:
            raise ValueError(f"Incomplete binary velocity file: {path} ({path.stat().st_size} != {expected})")
    variables = {
        "longitude": np.fromfile(daily / "longitude.f64", dtype="<f8"),
        "latitude": np.fromfile(daily / "latitude.f64", dtype="<f8"),
        "depth": np.fromfile(daily / "depth.f64", dtype="<f8"),
        "uo_glor": np.memmap(u_path, mode="r", dtype="<f4", shape=(1, nz, ny, nx)),
        "vo_glor": np.memmap(v_path, mode="r", dtype="<f4", shape=(1, nz, ny, nx)),
    }
    if variables["longitude"].size != nx or variables["latitude"].size != ny or variables["depth"].size != nz:
        raise ValueError(f"Coordinate dimensions do not match velocity shape in {daily}")
    return SimpleNamespace(variables=variables)


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
        speed_ratio_max=float(args.deep_speed_ratio_max),
        angle_jump_max_deg=float(args.deep_angle_jump_max_deg),
        tangent_tolerance_deg=float(args.deep_tangent_tolerance_deg),
        symmetry_tolerance_deg=120.0,
        min_tangent_fraction=float(args.deep_min_tangent_fraction),
        min_reversal_fraction=float(args.deep_min_reversal_fraction),
        min_finite_fraction=float(args.near_streamline_min_finite_fraction),
        direction_exception_extra=0,
        enforce_velocity_ratio_hard_gate=bool(args.enforce_velocity_ratio_hard_gate),
        enforce_tangent_alignment_hard_gate=bool(args.enforce_tangent_alignment_hard_gate),
        enforce_angle_jump_hard_gate=not bool(args.disable_angle_jump_hard_gate),
        direction_exception_multiplier=float(args.direction_exception_multiplier),
        enforce_direction_exception_hard_gate=not bool(args.disable_direction_exception_hard_gate),
        enforce_opposite_reversal_hard_gate=not bool(args.disable_opposite_reversal_hard_gate),
        deep_hua_mode=str(getattr(args, "deep_hua_mode", "minimal_finite_only")),
        boundary_mode=(
            "near_closed_streamline"
            if str(getattr(args, "deep_hua_mode", "")) in {"near_closed_streamline", "tangent_then_near_closed_streamline"}
            else "ssh_primary_open_ocean_no_streamline_gate"
        ),
        streamline_closure_tolerance_cells=float(args.near_streamline_closure_tolerance_cells),
        streamline_min_winding_turns=float(args.near_streamline_min_winding_turns),
        streamline_min_points=int(args.near_streamline_min_points),
    )


def select_deep_center(
    speed: np.ndarray,
    anchor_i: int,
    anchor_j: int,
    args: argparse.Namespace,
    *,
    u: np.ndarray | None = None,
    v: np.ndarray | None = None,
    previous_i: int | None = None,
    previous_j: int | None = None,
    radius_cells: float = np.nan,
    dx_km: float = np.nan,
    dy_km: float = np.nan,
    lat: np.ndarray | None = None,
    lon_step_deg: float = np.nan,
) -> tuple[int, int, float, int, int, float, int]:
    """Choose a deep speed minimum without silently changing the Hua test.

    ``global_disk_min`` preserves the historical search: the lowest speed in
    the complete deep-search disk wins.  ``local_step_min`` instead follows a
    local minimum branch from the previous accepted center.  It is intended
    for a layer-to-layer continuation test, where a weak, nearly equal speed
    minimum at the far edge of the six-cell disk must not cause a one-layer
    branch hop.
    """
    if args.deep_center_selection == "global_disk_min":
        radius = int(args.deep_search_cells)
    else:
        radius = min(int(args.deep_search_cells), int(args.deep_center_step_cells))
    center_i, center_j, center_speed, steps = _seeded_speed_min(speed, anchor_i, anchor_j, radius)
    if args.deep_center_selection != "local_step_section_bipolar":
        return center_i, center_j, center_speed, steps, radius, np.nan, 0

    # The trajectory-axis section can only break a tie after the path has a
    # resolved direction.  It never expands the local speed-minimum search.
    if (
        u is None or v is None or previous_i is None or previous_j is None
        or not np.isfinite(radius_cells) or radius_cells <= 0.0
        or not np.isfinite(dx_km) or not np.isfinite(dy_km)
    ):
        return center_i, center_j, center_speed, steps, radius, np.nan, 0
    local_dx_km = float(dx_km)
    if lat is not None and np.isfinite(lon_step_deg):
        local_cosine = max(abs(float(np.cos(np.deg2rad(lat[anchor_j])))), 0.05)
        local_dx_km = abs(float(np.deg2rad(lon_step_deg)) * EARTH_RADIUS_M * local_cosine / 1000.0)
    tx = (anchor_i - previous_i) * local_dx_km
    ty = (anchor_j - previous_j) * dy_km
    axis_norm = float(np.hypot(tx, ty))
    if axis_norm < float(args.deep_center_min_axis_km):
        return center_i, center_j, center_speed, steps, radius, np.nan, 0
    tx /= axis_norm
    ty /= axis_norm
    nx, ny = -ty, tx

    i0 = max(1, anchor_i - radius)
    i1 = min(speed.shape[1] - 2, anchor_i + radius)
    j0 = max(1, anchor_j - radius)
    j1 = min(speed.shape[0] - 2, anchor_j + radius)
    candidates: list[tuple[int, int, float]] = []
    speed_limit = float(center_speed) * (1.0 + float(args.deep_center_speed_tolerance))
    for candidate_j in range(j0, j1 + 1):
        for candidate_i in range(i0, i1 + 1):
            if (candidate_i - anchor_i) ** 2 + (candidate_j - anchor_j) ** 2 > radius ** 2:
                continue
            candidate_speed = float(speed[candidate_j, candidate_i])
            if not np.isfinite(candidate_speed) or candidate_speed > speed_limit:
                continue
            neighborhood = speed[candidate_j - 1:candidate_j + 2, candidate_i - 1:candidate_i + 2]
            if candidate_speed <= float(np.nanmin(neighborhood)):
                candidates.append((candidate_i, candidate_j, candidate_speed))
    if not candidates:
        return center_i, center_j, center_speed, steps, radius, np.nan, 0

    def bipolar_score(candidate_i: int, candidate_j: int) -> int:
        score = 0
        for factor in (0.6, 1.0):
            distance_km = max(2.0, factor * float(radius_cells) * 0.5 * (local_dx_km + dy_km))
            plus_i = candidate_i + nx * distance_km / local_dx_km
            plus_j = candidate_j + ny * distance_km / dy_km
            minus_i = candidate_i - nx * distance_km / dx_km
            minus_j = candidate_j - ny * distance_km / dy_km
            up = float(map_coordinates(u, [[plus_j], [plus_i]], order=1, mode="nearest", prefilter=False)[0])
            vp = float(map_coordinates(v, [[plus_j], [plus_i]], order=1, mode="nearest", prefilter=False)[0])
            um = float(map_coordinates(u, [[minus_j], [minus_i]], order=1, mode="nearest", prefilter=False)[0])
            vm = float(map_coordinates(v, [[minus_j], [minus_i]], order=1, mode="nearest", prefilter=False)[0])
            plus_axis = up * tx + vp * ty
            minus_axis = um * tx + vm * ty
            if np.isfinite(plus_axis) and np.isfinite(minus_axis) and plus_axis * minus_axis < 0.0:
                score += 1
        return score

    ranked = [
        (bipolar_score(candidate_i, candidate_j), candidate_speed,
         float(np.hypot(candidate_i - anchor_i, candidate_j - anchor_j)), candidate_i, candidate_j)
        for candidate_i, candidate_j, candidate_speed in candidates
    ]
    # Highest physically coherent section score wins; speed and displacement
    # resolve only ties between locally valid speed minima.
    score, chosen_speed, _, center_i, center_j = min(ranked, key=lambda item: (-item[0], item[1], item[2]))
    return center_i, center_j, chosen_speed, steps, radius, float(score), len(candidates)


def _as_bool(series: pd.Series) -> pd.Series:
    """Read boolean CSV fields without treating the string 'False' as true."""
    if series.dtype == bool:
        return series.fillna(False)
    return series.fillna(False).astype(str).str.strip().str.lower().isin(("1", "true", "yes"))


def surface_rows(root: Path, day: date, surface_table: Path | None = None) -> pd.DataFrame:
    """Return source surface objects from a final catalog or an explicit QC table."""
    path = surface_table if surface_table is not None else day_path(root, day)
    if not path.exists():
        raise FileNotFoundError(path)
    rows = pd.read_csv(path)
    if surface_table is not None:
        if "qc_pass" not in rows:
            raise ValueError(f"Explicit surface table has no qc_pass column: {path}")
        rows = rows[_as_bool(rows["qc_pass"])].copy()
    else:
        rows = rows[_as_bool(rows.get("hua_pass", pd.Series(False, index=rows.index)))].copy()
    if surface_table is None and "persistence_class" in rows:
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


def make_surface_record(row: pd.Series, *, day: date, source_label: str) -> dict[str, object]:
    out = row.to_dict()
    out.update(
        {
            "date": day.isoformat(),
            "depth_index": 0,
            "depth_m": float(row.get("depth_m", 2.5)),
            "vertical_extension_source": source_label,
            "vertical_extension_algorithm": "existing_hua_depth_continuation",
            "vertical_extension_stop_reason": "",
            "hua_pass": True,
        }
    )
    return out


def extend_day(args: argparse.Namespace, day: date) -> dict[str, object]:
    source_table = getattr(args, "surface_table", None)
    source = surface_rows(args.surface_root, day, source_table)
    source_label = "qc_surface_table" if source_table is not None else "final_surface_catalog"
    if int(args.max_surface_objects) > 0:
        source = source.sort_values(["hua_object_id"]).head(int(args.max_surface_objects)).copy()
    out_dir = args.output_root / "raw_detection" / "daily_runs" / f"{day:%Y%m%d}"
    resume_outputs = (
        out_dir / "centers_hua_style.parquet",
        out_dir / "structures_hua_style.parquet",
        out_dir / "vertical_extension_summary.json",
    )
    if args.resume and all(path.exists() and path.stat().st_size > 0 for path in resume_outputs):
        return {"day": day.isoformat(), "status": "resume"}
    binary_root = getattr(args, "filter_binary_root", None)
    input_path = args.filter_root / f"global_phy_{day:%Y%m%d}.nc"
    if binary_root is None and not input_path.exists():
        raise FileNotFoundError(input_path)

    params = params_from_args(args)
    vertical_algorithm = (
        "tangent_then_near_closed_streamline_depth_continuation"
        if params.deep_hua_mode == "tangent_then_near_closed_streamline"
        else "near_closed_streamline_depth_continuation"
        if params.deep_hua_mode == "near_closed_streamline"
        else "existing_hua_depth_continuation"
    )
    centers: list[dict[str, object]] = [
        make_surface_record(row, day=day, source_label=source_label) for _, row in source.iterrows()
    ]
    structures: list[dict[str, object]] = []
    voxels: list[dict[str, object]] = []
    states: list[dict[str, object]] = []

    if binary_root is not None:
        dataset_context = nullcontext(open_binary_velocity_dataset(binary_root, day))
        velocity_input_mode = "matlab_netcdf_to_raw_binary_bridge"
    else:
        from netCDF4 import Dataset
        dataset_context = Dataset(input_path)
        velocity_input_mode = "netcdf4"

    with dataset_context as ds:
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
            if args.write_object_voxels and np.isfinite(radius) and radius > 0:
                voxels.extend(
                    _object_voxels_for_layer(
                        u0, v0, lon, lat, float(depth[0]), day=day, object_id=object_id,
                        depth_index=0, center_i=ci, center_j=cj, radius_cells=radius,
                        polarity=str(row.get("polarity", "")),
                    )
                )
            states.append({
                "row": row, "object_id": object_id, "prev_i": ci, "prev_j": cj,
                "prev_prev_i": None, "prev_prev_j": None, "prev_radius_cells": radius,
            })

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
                anchor_i, anchor_j = int(state["prev_i"]), int(state["prev_j"])
                center_i, center_j, center_speed, min_steps, search_radius, section_score, section_candidates = select_deep_center(
                    speed, anchor_i, anchor_j, args,
                    u=u, v=v, previous_i=state["prev_prev_i"], previous_j=state["prev_prev_j"],
                    radius_cells=float(state["prev_radius_cells"]), dx_km=float(dx_km), dy_km=float(dy_km),
                    lat=lat, lon_step_deg=float(np.nanmedian(np.abs(np.diff(lon)))),
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
                        "deep_center_selection": str(args.deep_center_selection),
                        "deep_center_search_radius_cells": int(search_radius),
                        "deep_center_anchor_i": int(anchor_i), "deep_center_anchor_j": int(anchor_j),
                        "deep_center_section_bipolar_score": float(section_score),
                        "deep_center_section_candidate_count": int(section_candidates),
                        "deep_center_grid_step_cells": float(np.hypot(center_i - anchor_i, center_j - anchor_j)),
                        "deep_center_refined_step_cells": float(np.hypot(
                            float(refined["center_i_refined"]) - anchor_i,
                            float(refined["center_j_refined"]) - anchor_j,
                        )),
                        "vertical_extension_source": source_label,
                        "vertical_extension_algorithm": vertical_algorithm,
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
                if args.write_object_voxels:
                    voxels.extend(_object_voxels_for_layer(
                        u, v, lon, lat, float(depth[depth_index]), day=day, object_id=state["object_id"],
                        depth_index=int(depth_index), center_i=center_i, center_j=center_j, radius_cells=radius,
                        polarity=str(source_row.get("polarity", "")),
                    ))
                state["prev_prev_i"], state["prev_prev_j"] = anchor_i, anchor_j
                state["prev_i"] = int(np.clip(round(float(refined["center_i_refined"])), 0, len(lon) - 1))
                state["prev_j"] = int(np.clip(round(float(refined["center_j_refined"])), 0, len(lat) - 1))
                state["prev_radius_cells"] = radius
                next_active.append(state)
            active = next_active
            print(f"[vertical] {day.isoformat()} depth={depth_index}/{max_layers - 1} active={len(active)}", flush=True)

    centers_df = pd.DataFrame(centers)
    structures_df = pd.DataFrame(structures)
    if args.vertical_profile_id:
        centers_df["vertical_profile_id"] = str(args.vertical_profile_id)
        structures_df["vertical_profile_id"] = str(args.vertical_profile_id)
    voxels_df = pd.DataFrame(voxels)
    write_table(centers_df, out_dir / "centers_hua_style")
    write_table(structures_df, out_dir / "structures_hua_style")
    if args.write_object_voxels:
        voxel_path = args.output_root / "raw_detection" / "object_voxels_parts" / f"year={day.year}" / f"date={day:%Y%m%d}.parquet"
        voxel_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            voxels_df.to_parquet(voxel_path, index=False)
        except (ImportError, OSError):
            voxels_df.to_csv(voxel_path.with_suffix(".csv.gz"), index=False, compression="gzip")
    summary = {
        "day": day.isoformat(), "surface_objects": int(len(source)), "center_rows": int(len(centers_df)),
        "passed_rows": int(centers_df["hua_pass"].fillna(False).astype(bool).sum()), "voxels": int(len(voxels_df)),
        "algorithm": f"depth_major_{params.deep_hua_mode}", "max_depth_layers": int(args.max_depth_layers),
        "enforce_velocity_ratio_hard_gate": bool(args.enforce_velocity_ratio_hard_gate),
        "enforce_tangent_alignment_hard_gate": bool(args.enforce_tangent_alignment_hard_gate),
        "enforce_angle_jump_hard_gate": not bool(args.disable_angle_jump_hard_gate),
        "direction_exception_multiplier": float(args.direction_exception_multiplier),
        "enforce_direction_exception_hard_gate": not bool(args.disable_direction_exception_hard_gate),
        "enforce_opposite_reversal_hard_gate": not bool(args.disable_opposite_reversal_hard_gate),
        "deep_hua_mode": str(getattr(args, "deep_hua_mode", "minimal_finite_only")),
        "vertical_profile_id": str(args.vertical_profile_id or "unspecified"),
        "deep_center_selection": str(args.deep_center_selection),
        "deep_center_step_cells": int(args.deep_center_step_cells),
        "deep_center_speed_tolerance": float(args.deep_center_speed_tolerance),
        "deep_center_min_axis_km": float(args.deep_center_min_axis_km),
        "deep_search_cells": int(args.deep_search_cells),
        "streamline_near_closed_parameters": {
            "start_angles": int(params.streamline_start_angles),
            "integration_directions": 2,
            "step_cells": float(params.streamline_step_cells),
            "max_steps": int(params.streamline_max_steps),
            "closure_tolerance_cells": float(params.streamline_closure_tolerance_cells),
            "min_winding_turns": float(params.streamline_min_winding_turns),
            "min_points": int(params.streamline_min_points),
            "min_finite_fraction": float(params.min_finite_fraction),
        },
        "velocity_input_mode": velocity_input_mode,
        "write_object_voxels": bool(args.write_object_voxels),
        "surface_input": str(source_table) if source_table is not None else str(day_path(args.surface_root, day)),
        "surface_selection": "qc_pass=True" if source_table is not None else "hua_pass=True and non-transient",
    }
    (out_dir / "vertical_extension_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Depth-major Hua continuation from a final catalog or explicit QC surface table.")
    parser.add_argument("--surface-root", type=Path, default=DEFAULT_SURFACE_ROOT)
    parser.add_argument(
        "--surface-table", type=Path,
        help="Explicit centers_hua_style.csv source. Selects qc_pass=True and does not apply persistence filtering.",
    )
    parser.add_argument("--filter-root", type=Path, default=DEFAULT_FILTER_ROOT)
    parser.add_argument(
        "--filter-binary-root", type=Path,
        help="Optional raw-binary velocity bridge root containing YYYYMMDD/metadata.json and u/v float32 files.",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--day", required=True)
    parser.add_argument("--vertical-profile-id", default="")
    parser.add_argument("--max-depth-layers", type=int, default=105)
    parser.add_argument("--deep-search-cells", type=int, default=6)
    parser.add_argument(
        "--deep-center-selection", choices=["global_disk_min", "local_step_min", "local_step_section_bipolar"], default="local_step_min",
        help="Deep center path: bounded local continuation step (default), local section-bipolar tie-break, or historical full-disk minimum.",
    )
    parser.add_argument(
        "--deep-center-step-cells", type=int, default=2,
        help="Maximum per-layer local minimum search radius for --deep-center-selection local_step_min.",
    )
    parser.add_argument(
        "--deep-center-speed-tolerance", type=float, default=0.20,
        help="For local_step_section_bipolar, compare only local minima within this relative speed of the local minimum.",
    )
    parser.add_argument(
        "--deep-center-min-axis-km", type=float, default=1.0,
        help="For local_step_section_bipolar, minimum previous-layer displacement required to define a section axis.",
    )
    parser.add_argument("--start-radius-cells", type=int, default=2)
    parser.add_argument("--max-radius-cells", type=int, default=12)
    parser.add_argument("--deep-speed-ratio-max", type=float, default=3.0)
    parser.add_argument("--deep-angle-jump-max-deg", type=float, default=150.0)
    parser.add_argument("--deep-tangent-tolerance-deg", type=float, default=24.0)
    parser.add_argument("--deep-min-tangent-fraction", type=float, default=0.70)
    parser.add_argument("--deep-min-reversal-fraction", type=float, default=0.70)
    parser.add_argument(
        "--enforce-velocity-ratio-hard-gate", action="store_true",
        help="Keep the velocity-ratio threshold as a hard deep Hua rejection. Disabled by default.",
    )
    parser.add_argument(
        "--enforce-tangent-alignment-hard-gate", action="store_true",
        help="Keep the tangent-alignment threshold as a hard deep Hua rejection. Disabled by default.",
    )
    parser.add_argument(
        "--direction-exception-multiplier", type=float, default=1.0,
        help="Multiplier applied to floor(radius_cells/5)+1 before the direction-exception hard gate.",
    )
    parser.add_argument(
        "--disable-angle-jump-hard-gate", action="store_true",
        help="Retain angle-jump diagnostics but never reject a deep layer for them.",
    )
    parser.add_argument(
        "--disable-direction-exception-hard-gate", action="store_true",
        help="Retain direction-exception diagnostics but never reject a deep layer for them.",
    )
    parser.add_argument(
        "--disable-opposite-reversal-hard-gate", action="store_true",
        help="Retain opposite-reversal diagnostics but never reject a deep layer for them.",
    )
    parser.add_argument(
        "--deep-hua-mode", choices=["full", "minimal_reversal_only", "minimal_finite_only", "near_closed_streamline", "tangent_then_near_closed_streamline"], default="minimal_finite_only",
        help="Deep-only Hua validation kernel; the default retains only finite-velocity coverage.",
    )
    parser.add_argument("--near-streamline-closure-tolerance-cells", type=float, default=1.75)
    parser.add_argument("--near-streamline-min-winding-turns", type=float, default=0.75)
    parser.add_argument("--near-streamline-min-points", type=int, default=16)
    parser.add_argument("--near-streamline-min-finite-fraction", type=float, default=0.95)
    parser.add_argument("--write-object-voxels", action="store_true",
                        help="Materialize tracking voxels. Disabled for continuation/section/W-only runs.")
    parser.add_argument("--max-surface-objects", type=int, default=0, help="Smoke-only cap; <=0 keeps every final surface object.")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    print(json.dumps(extend_day(args, parse_day(args.day)), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
