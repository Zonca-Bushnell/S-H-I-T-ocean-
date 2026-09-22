from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw


def main() -> None:
    args = parse_args()
    day = str(args.day)
    ymd = day.replace("-", "")
    source_root = table_root(args.source_root, ymd)
    centers = read_table(source_root / "centers_hua_style")
    structures = read_table(source_root / "structures_hua_style")
    normalize_dates(centers)
    normalize_dates(structures)
    centers = centers[centers["date"].astype(str).eq(day)].copy() if "date" in centers.columns else centers.copy()
    structures = structures[structures["date"].astype(str).eq(day)].copy() if "date" in structures.columns else structures.copy()
    lon, lat = read_lon_lat(args.filter_root / f"global_phy_{ymd}.nc")

    out_centers, out_structures, summary_rows = apply_qc(centers, structures, lon, lat, args)
    out_dir = args.output_root / "daily_runs" / ymd
    out_dir.mkdir(parents=True, exist_ok=True)
    out_centers.to_csv(out_dir / "centers_hua_style.csv", index=False)
    out_structures.to_csv(out_dir / "structures_hua_style.csv", index=False)
    write_catalog(out_centers, out_dir / "isolated_eddy_candidates.csv", "isolated_eddy_candidate")
    write_catalog(out_centers, out_dir / "jet_meander_candidates.csv", "jet_meander_candidate")

    summary = {
        "day": day,
        "source_root": str(args.source_root),
        "output_root": str(args.output_root),
        "thresholds": {
            "max_shape_error_percent": float(args.max_shape_error_percent),
            "acc_max_shape_error_percent": float(args.acc_max_shape_error_percent),
            "open_ocean_max_shape_error_percent": float(args.open_ocean_max_shape_error_percent),
            "acc_bbox": [float(v) for v in args.acc_bbox],
            "min_compactness": float(args.min_compactness),
            "open_ocean_min_compactness": float(args.open_ocean_min_compactness),
            "min_boundary_points": int(args.min_boundary_points),
            "open_ocean_min_boundary_points": int(args.open_ocean_min_boundary_points),
            "min_radius_km": float(args.min_radius_km),
            "open_ocean_min_radius_km": float(args.open_ocean_min_radius_km),
            "min_area_cells": float(args.min_area_cells),
            "open_ocean_min_area_cells": float(args.open_ocean_min_area_cells),
            "jet_core_overlap_max": float(args.jet_core_overlap_max),
            "enable_jet_split": bool(args.enable_jet_split),
            "overlap_center_factor": float(args.overlap_center_factor),
            "overlap_area_fraction": float(args.overlap_area_fraction),
            "persistence_lookahead_days": int(args.persistence_lookahead_days),
            "persistence_min_consecutive_days": int(args.persistence_min_consecutive_days),
            "persistence_distance_factor": float(args.persistence_distance_factor),
            "persistence_radius_ratio_max": float(args.persistence_radius_ratio_max),
            "persistence_open_ocean_distance_factor": float(args.persistence_open_ocean_distance_factor),
            "persistence_open_ocean_radius_ratio_max": float(args.persistence_open_ocean_radius_ratio_max),
            "persistence_open_ocean_distance_factor_max": float(args.persistence_open_ocean_distance_factor_max),
            "persistence_open_ocean_radius_ratio_max_max": float(args.persistence_open_ocean_radius_ratio_max_max),
            "persistence_open_ocean_lat_min": float(args.persistence_open_ocean_lat_min),
            "persistence_open_ocean_lat_max": float(args.persistence_open_ocean_lat_max),
            "persistence_open_ocean_drift_fraction_min": float(args.persistence_open_ocean_drift_fraction_min),
            "persistence_open_ocean_drift_fraction_max": float(args.persistence_open_ocean_drift_fraction_max),
            "persistence_open_ocean_max_gap_days": int(args.persistence_open_ocean_max_gap_days),
            "persistence_applied": not bool(args.skip_persistence),
            "accept_ssh_primary_without_streamline": bool(args.accept_ssh_primary_without_streamline),
        },
        "counts": summary_rows,
    }
    (out_dir / f"qc_summary_{ymd}.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(summary_rows).to_csv(out_dir / f"qc_summary_{ymd}.csv", index=False)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply META-like shape, overlap, and jet-meander QC to OFES SSH-primary surface candidates.")
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--filter-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--day", default="1991-01-01")
    parser.add_argument("--max-shape-error-percent", type=float, default=70.0)
    parser.add_argument("--acc-max-shape-error-percent", type=float, default=55.0)
    parser.add_argument("--open-ocean-max-shape-error-percent", type=float, default=80.0)
    parser.add_argument("--acc-bbox", type=float, nargs=4, metavar=("LON_MIN", "LON_MAX", "LAT_MIN", "LAT_MAX"), default=(0.0, 360.0, -62.0, -40.0))
    parser.add_argument("--min-compactness", type=float, default=0.20)
    parser.add_argument("--open-ocean-min-compactness", type=float, default=0.12)
    parser.add_argument("--min-boundary-points", type=int, default=9)
    parser.add_argument("--open-ocean-min-boundary-points", type=int, default=6)
    parser.add_argument("--min-radius-km", type=float, default=25.0)
    parser.add_argument("--open-ocean-min-radius-km", type=float, default=18.0)
    parser.add_argument("--min-area-cells", type=float, default=16.0)
    parser.add_argument("--open-ocean-min-area-cells", type=float, default=9.0)
    parser.add_argument("--jet-core-overlap-max", type=float, default=0.50)
    parser.add_argument("--enable-jet-split", action="store_true")
    parser.add_argument("--overlap-center-factor", type=float, default=0.75)
    parser.add_argument("--overlap-area-fraction", type=float, default=0.50)
    parser.add_argument("--persistence-source-root", type=Path, default=None)
    parser.add_argument("--persistence-lookahead-days", type=int, default=5)
    parser.add_argument("--persistence-min-consecutive-days", type=int, default=2)
    parser.add_argument("--persistence-distance-factor", type=float, default=1.5)
    parser.add_argument("--persistence-radius-ratio-max", type=float, default=2.0)
    parser.add_argument("--persistence-open-ocean-distance-factor", type=float, default=2.5)
    parser.add_argument("--persistence-open-ocean-radius-ratio-max", type=float, default=3.0)
    parser.add_argument("--persistence-open-ocean-distance-factor-max", type=float, default=3.0)
    parser.add_argument("--persistence-open-ocean-radius-ratio-max-max", type=float, default=3.5)
    parser.add_argument("--persistence-open-ocean-lat-min", type=float, default=20.0)
    parser.add_argument("--persistence-open-ocean-lat-max", type=float, default=60.0)
    parser.add_argument("--persistence-open-ocean-drift-fraction-min", type=float, default=0.10)
    parser.add_argument("--persistence-open-ocean-drift-fraction-max", type=float, default=0.35)
    parser.add_argument("--persistence-open-ocean-max-gap-days", type=int, default=1)
    parser.add_argument(
        "--skip-persistence",
        action="store_true",
        help="Apply contour geometry and overlap QC only; retain accepted rows without temporal persistence screening.",
    )
    parser.add_argument(
        "--accept-ssh-primary-without-streamline",
        action="store_true",
        help="Promote closed, single-extremum SSH-primary contours rejected only for no_closed_streamline_effective into geometry QC.",
    )
    return parser.parse_args()


def table_root(root: Path, ymd: str) -> Path:
    daily = root / "daily_runs" / ymd
    if table_exists(daily / "centers_hua_style") and table_exists(daily / "structures_hua_style"):
        return daily
    if table_exists(root / "centers_hua_style") and table_exists(root / "structures_hua_style"):
        return root
    raise FileNotFoundError(f"No centers/structures tables under {root}")


def table_exists(path_without_suffix: Path) -> bool:
    return path_without_suffix.with_suffix(".csv").exists() or path_without_suffix.with_suffix(".parquet").exists()


def read_table(path_without_suffix: Path) -> pd.DataFrame:
    csv_path = path_without_suffix.with_suffix(".csv")
    parquet_path = path_without_suffix.with_suffix(".parquet")
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    if csv_path.exists():
        return pd.read_csv(csv_path)
    raise FileNotFoundError(path_without_suffix)


def normalize_dates(df: pd.DataFrame) -> None:
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")


def read_lon_lat(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as ds:
        return ds["longitude"][:].astype("f8"), ds["latitude"][:].astype("f8")


def apply_qc(
    centers: pd.DataFrame,
    structures: pd.DataFrame,
    lon: np.ndarray,
    lat: np.ndarray,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, object]]]:
    out = centers.copy()
    if "depth_index" in out.columns:
        surface_idx = out.index[out["depth_index"].astype(int).eq(0)]
    else:
        surface_idx = out.index
    original_hua_pass = out.get("hua_pass", pd.Series(False, index=out.index)).fillna(False).astype(bool)
    raw_pass = original_hua_pass.copy()
    out["original_hua_pass"] = original_hua_pass
    out["raw_boundary_source"] = out.get("boundary_source", pd.Series("", index=out.index)).astype(str)
    out["streamline_gate_removed"] = False
    out["streamline_gate_original_reason"] = ""
    if bool(args.accept_ssh_primary_without_streamline):
        promoted = promotable_ssh_primary_without_streamline(out)
        raw_pass = raw_pass | promoted
        out.loc[promoted, "streamline_gate_removed"] = True
        out.loc[promoted, "streamline_gate_original_reason"] = out.loc[promoted, "catalog_acceptance_reason"].astype(str)
        out.loc[promoted, "raw_boundary_source"] = "ssh_effective_contour_streamline_gate_removed"
        out.loc[promoted, "boundary_source"] = "ssh_effective_contour_streamline_gate_removed"
        out.loc[promoted, "catalog_acceptance_reason"] = "ssh_effective_contour_streamline_gate_removed"
        out.loc[promoted, "dynamical_core_class"] = "weak_or_no_streamline_core"
    out["raw_hua_pass"] = raw_pass
    out["qc_pass"] = False
    out["qc_class"] = "not_surface_or_not_source_pass"
    out["qc_reject_reason"] = ""
    out["shape_error_percent"] = np.nan
    out["compactness"] = np.nan
    out["boundary_point_count"] = 0
    out["pixel_area_cells"] = np.nan
    out["radius_km"] = np.nan
    out["single_extremum_pass"] = False
    out["overlap_group_id"] = -1
    out["overlap_keep"] = False
    out["jet_meander_class"] = "not_evaluated"
    out["persistence_days_available"] = 0
    out["persistence_match_count"] = 0
    out["persistence_class"] = "not_evaluated"

    candidate_idx = [idx for idx in surface_idx if bool(raw_pass.loc[idx])]
    for idx in surface_idx:
        if not bool(raw_pass.loc[idx]):
            raw_reason = str(out.at[idx, "catalog_acceptance_reason"]) if "catalog_acceptance_reason" in out.columns else "boundary_rejected"
            out.at[idx, "qc_class"] = "boundary_rejected"
            out.at[idx, "qc_reject_reason"] = raw_reason or "boundary_rejected"
    for idx in candidate_idx:
        metrics = contour_metrics(out.loc[idx], lon, lat)
        if "ssh_contour_shape_error_percent" in out.columns:
            core_shape = numeric(out.at[idx, "ssh_contour_shape_error_percent"])
            if np.isfinite(core_shape):
                metrics["shape_error_percent"] = core_shape
        if "ssh_contour_compactness" in out.columns:
            core_compactness = numeric(out.at[idx, "ssh_contour_compactness"])
            if np.isfinite(core_compactness):
                metrics["compactness"] = core_compactness
        for key, value in metrics.items():
            out.at[idx, key] = value
        reasons = []
        open_ocean = is_open_ocean(out.loc[idx])
        min_boundary_points = int(args.open_ocean_min_boundary_points if open_ocean else args.min_boundary_points)
        min_area_cells = float(args.open_ocean_min_area_cells if open_ocean else args.min_area_cells)
        min_compactness = float(args.open_ocean_min_compactness if open_ocean else args.min_compactness)
        if int(metrics["boundary_point_count"]) < min_boundary_points:
            reasons.append("boundary_points_too_few")
        max_shape = (
            float(args.open_ocean_max_shape_error_percent)
            if open_ocean
            else float(args.acc_max_shape_error_percent)
            if in_bbox(out.loc[idx], tuple(float(v) for v in args.acc_bbox))
            else float(args.max_shape_error_percent)
        )
        if np.isfinite(metrics["shape_error_percent"]) and float(metrics["shape_error_percent"]) > max_shape:
            reasons.append("shape_error_high")
        if np.isfinite(metrics["compactness"]) and float(metrics["compactness"]) < min_compactness:
            reasons.append("compactness_low")
        if np.isfinite(metrics["pixel_area_cells"]) and float(metrics["pixel_area_cells"]) < min_area_cells:
            reasons.append("area_too_small")
        min_radius_km = float(args.open_ocean_min_radius_km if open_ocean else args.min_radius_km)
        if np.isfinite(metrics["radius_km"]) and float(metrics["radius_km"]) < min_radius_km:
            reasons.append("radius_too_small")
        same_extrema = pd.to_numeric(pd.Series([out.at[idx, "ssh_contour_same_extrema_count"]]) if "ssh_contour_same_extrema_count" in out.columns else pd.Series([1]), errors="coerce").iloc[0]
        out.at[idx, "single_extremum_pass"] = bool(np.isfinite(same_extrema) and float(same_extrema) <= 1.0)
        if np.isfinite(same_extrema) and float(same_extrema) > 1.0:
            reasons.append("multiple_same_sign_extrema")
        if reasons:
            out.at[idx, "qc_reject_reason"] = "|".join(reasons)
            out.at[idx, "qc_class"] = "shape_rejected"
        else:
            out.at[idx, "qc_class"] = "shape_pass"

    shape_pass_idx = [idx for idx in candidate_idx if out.at[idx, "qc_class"] == "shape_pass"]
    group_ids, keep_idx = overlap_groups(out, shape_pass_idx, args)
    for idx in shape_pass_idx:
        out.at[idx, "overlap_group_id"] = group_ids.get(idx, -1)
        out.at[idx, "overlap_keep"] = idx in keep_idx
        if idx not in keep_idx:
            out.at[idx, "qc_reject_reason"] = "overlap_duplicate"
            out.at[idx, "qc_class"] = "overlap_duplicate"

    for idx in keep_idx:
        jet_fraction = numeric(out.at[idx, "jet_core_overlap_fraction"]) if "jet_core_overlap_fraction" in out.columns else np.nan
        is_jet = bool(np.isfinite(jet_fraction) and jet_fraction > float(args.jet_core_overlap_max))
        if is_jet and bool(args.enable_jet_split):
            out.at[idx, "qc_pass"] = True
            out.at[idx, "qc_class"] = "jet_meander_candidate"
            out.at[idx, "jet_meander_class"] = "jet_core_overlap"
        else:
            out.at[idx, "qc_pass"] = True
            out.at[idx, "qc_class"] = "isolated_eddy_candidate"
            out.at[idx, "jet_meander_class"] = "jet_flag_recorded_not_split" if is_jet else "isolated"

    if bool(args.skip_persistence):
        accepted_idx = out.index[out["qc_pass"].astype(bool)]
        out.loc[accepted_idx, "persistence_class"] = "not_run_geometry_qc_only"
    else:
        apply_persistence_diagnostic(out, args)
        transient_idx = out.index[
            out["qc_pass"].astype(bool)
            & out["persistence_class"].astype(str).eq("transient")
        ]
        for idx in transient_idx:
            prior = str(out.at[idx, "qc_reject_reason"] or "")
            reason = "transient" if not prior else f"{prior}|transient"
            out.at[idx, "qc_pass"] = False
            out.at[idx, "qc_class"] = "transient"
            out.at[idx, "qc_reject_reason"] = reason

    out["hua_pass"] = out["qc_pass"].astype(bool)
    out["boundary_source"] = out.get("boundary_source", pd.Series("", index=out.index)).astype(str)
    out.loc[out["qc_pass"].astype(bool), "boundary_source"] = out.loc[out["qc_pass"].astype(bool), "raw_boundary_source"]
    out.loc[out["qc_reject_reason"].ne(""), "boundary_source"] = out.loc[out["qc_reject_reason"].ne(""), "qc_reject_reason"]

    accepted_ids = set(out.loc[out["qc_pass"].astype(bool), "hua_object_id"].astype(str)) if "hua_object_id" in out.columns else set()
    out_struct = structures[structures["hua_object_id"].astype(str).isin(accepted_ids)].copy() if "hua_object_id" in structures.columns else structures.iloc[0:0].copy()
    if not out_struct.empty and "hua_object_id" in out_struct.columns:
        cols = [
            "hua_object_id",
            "qc_class",
            "qc_pass",
            "qc_reject_reason",
            "raw_hua_pass",
            "original_hua_pass",
            "raw_boundary_source",
            "streamline_gate_removed",
            "streamline_gate_original_reason",
            "ssh_primary_discovery_pass",
            "ssh_primary_discovery_reason",
            "shape_error_percent",
            "compactness",
            "boundary_point_count",
            "pixel_area_cells",
            "radius_km",
            "single_extremum_pass",
            "overlap_group_id",
            "overlap_keep",
            "jet_meander_class",
            "persistence_days_available",
            "persistence_match_count",
            "persistence_class",
        ]
        out_struct = out_struct.merge(out[cols].drop_duplicates("hua_object_id"), on="hua_object_id", how="left")

    summary_rows = summarize(out, candidate_idx)
    return out, out_struct, summary_rows


def promotable_ssh_primary_without_streamline(out: pd.DataFrame) -> pd.Series:
    """Select SSH-primary successes rejected solely by the streamline hard gate."""
    reason = out.get("catalog_acceptance_reason", pd.Series("", index=out.index)).fillna("").astype(str)
    discovery = out.get("ssh_primary_discovery_pass", pd.Series(False, index=out.index)).fillna(False).astype(bool)
    closed = out.get("ssh_contour_closed", pd.Series(False, index=out.index)).fillna(False).astype(bool)
    extrema = pd.to_numeric(out.get("ssh_contour_same_extrema_count", pd.Series(np.nan, index=out.index)), errors="coerce")
    valid_boundary = out.apply(
        lambda row: min(
            parse_index_list(row.get("ssh_contour_boundary_i", "")).size,
            parse_index_list(row.get("ssh_contour_boundary_j", "")).size,
        ) >= 3,
        axis=1,
    )
    return reason.eq("no_closed_streamline_effective") & discovery & closed & extrema.le(1.0) & valid_boundary


def contour_metrics(row: pd.Series, lon: np.ndarray, lat: np.ndarray) -> dict[str, float | int]:
    # The geometry catalog is defined by the saved SSH effective contour.  A
    # streamline boundary is only a legacy fallback when that contour is absent.
    ii = parse_index_list(row.get("ssh_contour_boundary_i", ""))
    jj = parse_index_list(row.get("ssh_contour_boundary_j", ""))
    if ii.size < 3 or jj.size < 3:
        ii = parse_index_list(row.get("streamline_boundary_i", ""))
        jj = parse_index_list(row.get("streamline_boundary_j", ""))
    n = min(ii.size, jj.size)
    if n < 3:
        return {"shape_error_percent": np.nan, "compactness": np.nan, "boundary_point_count": int(n), "pixel_area_cells": np.nan, "radius_km": radius_km(row)}
    ii = ii[:n]
    jj = jj[:n]
    valid = (ii >= 0) & (ii < len(lon)) & (jj >= 0) & (jj < len(lat))
    ii = ii[valid]
    jj = jj[valid]
    if ii.size < 3:
        return {"shape_error_percent": np.nan, "compactness": np.nan, "boundary_point_count": int(ii.size), "pixel_area_cells": np.nan, "radius_km": radius_km(row)}
    lon0 = numeric(row.get("center_lon_refined", row.get("ssh_contour_center_lon", np.nan)))
    lat0 = numeric(row.get("center_lat_refined", row.get("ssh_contour_center_lat", np.nan)))
    if not np.isfinite(lon0):
        lon0 = float(np.nanmean(lon[ii]))
    if not np.isfinite(lat0):
        lat0 = float(np.nanmean(lat[jj]))
    x = (lon[ii].astype("f8") - lon0) * 111.2 * max(math.cos(math.radians(float(lat0))), 0.2)
    y = (lat[jj].astype("f8") - lat0) * 111.2
    order = np.argsort(np.arctan2(y, x))
    x = x[order]
    y = y[order]
    if x[0] != x[-1] or y[0] != y[-1]:
        x = np.r_[x, x[0]]
        y = np.r_[y, y[0]]
    area = 0.5 * abs(float(np.sum(x[:-1] * y[1:] - x[1:] * y[:-1])))
    dx = np.diff(x)
    dy = np.diff(y)
    perimeter = float(np.sum(np.hypot(dx, dy)))
    cx = float(np.nanmean(x[:-1]))
    cy = float(np.nanmean(y[:-1]))
    rr = np.hypot(x[:-1] - cx, y[:-1] - cy)
    mean_r = float(np.nanmean(rr)) if rr.size else np.nan
    shape_error = float(np.nanstd(rr) / mean_r * 100.0) if np.isfinite(mean_r) and mean_r > 0 else np.nan
    compactness = float(4.0 * math.pi * area / max(perimeter * perimeter, 1.0e-12)) if perimeter > 0 else np.nan
    pixel_area = numeric(row.get("ssh_contour_area_cells", np.nan))
    if not np.isfinite(pixel_area):
        mask = contour_mask(row)
        pixel_area = float(mask["area"]) if mask is not None else np.nan
    rad_km = radius_km(row)
    if not np.isfinite(rad_km) and np.isfinite(pixel_area) and pixel_area > 0:
        rad_km = math.sqrt(pixel_area / math.pi) * 11.12
    return {
        "shape_error_percent": shape_error,
        "compactness": compactness,
        "boundary_point_count": int(ii.size),
        "pixel_area_cells": pixel_area,
        "radius_km": rad_km,
    }


def overlap_groups(out: pd.DataFrame, indices: list[int], args: argparse.Namespace) -> tuple[dict[int, int], set[int]]:
    parent = {idx: idx for idx in indices}
    masks: dict[int, dict[str, object] | None] = {}
    n = len(indices)
    polarity = np.asarray([str(out.at[idx, "polarity"]) for idx in indices], dtype=object)
    lon_values = np.asarray([row_lon(out.loc[idx]) for idx in indices], dtype="f8")
    lat_values = np.asarray([row_lat(out.loc[idx]) for idx in indices], dtype="f8")
    radii = np.asarray([radius_km(out.loc[idx]) for idx in indices], dtype="f8")

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for pos, a in enumerate(indices):
        if not (np.isfinite(lon_values[pos]) and np.isfinite(lat_values[pos]) and np.isfinite(radii[pos]) and radii[pos] > 0):
            continue
        tail = np.arange(pos + 1, n)
        if tail.size == 0:
            continue
        same = polarity[tail] == polarity[pos]
        finite = np.isfinite(lon_values[tail]) & np.isfinite(lat_values[tail]) & np.isfinite(radii[tail]) & (radii[tail] > 0)
        candidates = tail[same & finite]
        if candidates.size == 0:
            continue
        dlon = ((lon_values[candidates] - lon_values[pos] + 180.0) % 360.0) - 180.0
        dlat = lat_values[candidates] - lat_values[pos]
        lat_mid = 0.5 * (lat_values[candidates] + lat_values[pos])
        dist = np.hypot(dlon * 111.2 * np.maximum(np.cos(np.deg2rad(lat_mid)), 0.2), dlat * 111.2)
        max_dist = np.maximum(radii[pos] + radii[candidates], float(args.overlap_center_factor) * np.minimum(radii[pos], radii[candidates]) * 2.0)
        candidates = candidates[dist <= max_dist]
        dist = dist[dist <= max_dist]
        for cand_pos, d in zip(candidates, dist):
            b = indices[int(cand_pos)]
            r1 = float(radii[pos])
            r2 = float(radii[int(cand_pos)])
            proxy_overlap = circle_overlap_fraction(d, r1, r2)
            contour_overlap = np.nan
            if proxy_overlap > 0.0 or d < 1.25 * (r1 + r2):
                if a not in masks:
                    masks[a] = contour_mask(out.loc[a])
                if b not in masks:
                    masks[b] = contour_mask(out.loc[b])
                contour_overlap = contour_overlap_fraction(masks.get(a), masks.get(b))
            if (
                d < float(args.overlap_center_factor) * min(r1, r2)
                or (np.isfinite(contour_overlap) and contour_overlap > float(args.overlap_area_fraction))
                or proxy_overlap > float(args.overlap_area_fraction)
            ):
                union(a, b)

    roots = {idx: find(idx) for idx in indices}
    root_to_group: dict[int, int] = {}
    group_ids: dict[int, int] = {}
    for idx, root in roots.items():
        if root not in root_to_group:
            root_to_group[root] = len(root_to_group)
        group_ids[idx] = root_to_group[root]

    keep: set[int] = set()
    for group in sorted(set(group_ids.values())):
        members = [idx for idx, gid in group_ids.items() if gid == group]
        keep.add(sorted(members, key=lambda idx: rank_tuple(out.loc[idx]))[0])
    return group_ids, keep


def center_distance_km(a: pd.Series, b: pd.Series) -> float:
    lon1 = row_lon(a)
    lat1 = row_lat(a)
    lon2 = row_lon(b)
    lat2 = row_lat(b)
    if not all(np.isfinite(v) for v in [lon1, lat1, lon2, lat2]):
        return np.nan
    dlon = ((lon2 - lon1 + 180.0) % 360.0) - 180.0
    dlat = lat2 - lat1
    lat_mid = 0.5 * (lat1 + lat2)
    return float(math.hypot(dlon * 111.2 * max(math.cos(math.radians(lat_mid)), 0.2), dlat * 111.2))


def row_lon(row: pd.Series) -> float:
    for name in ["center_lon_refined", "center_lon", "ssh_contour_center_lon", "seed_lon"]:
        value = numeric(row.get(name, np.nan))
        if np.isfinite(value):
            return value
    return np.nan


def row_lat(row: pd.Series) -> float:
    for name in ["center_lat_refined", "center_lat", "ssh_contour_center_lat", "seed_lat"]:
        value = numeric(row.get(name, np.nan))
        if np.isfinite(value):
            return value
    return np.nan


def radius_km(row: pd.Series) -> float:
    if "radius_km" in row and np.isfinite(numeric(row["radius_km"])):
        return numeric(row["radius_km"])
    if "ssh_contour_radius_cells" in row and np.isfinite(numeric(row["ssh_contour_radius_cells"])):
        return numeric(row["ssh_contour_radius_cells"]) * 11.12
    if "accepted_radius_cells" in row and np.isfinite(numeric(row["accepted_radius_cells"])):
        return numeric(row["accepted_radius_cells"]) * 11.12
    return np.nan


def circle_overlap_fraction(d: float, r1: float, r2: float) -> float:
    if d >= r1 + r2:
        return 0.0
    if d <= abs(r1 - r2):
        return 1.0
    part1 = r1 * r1 * math.acos((d * d + r1 * r1 - r2 * r2) / (2.0 * d * r1))
    part2 = r2 * r2 * math.acos((d * d + r2 * r2 - r1 * r1) / (2.0 * d * r2))
    part3 = 0.5 * math.sqrt(max((-d + r1 + r2) * (d + r1 - r2) * (d - r1 + r2) * (d + r1 + r2), 0.0))
    overlap = part1 + part2 - part3
    return float(overlap / max(math.pi * min(r1, r2) ** 2, 1.0e-12))


def contour_mask(row: pd.Series) -> dict[str, object] | None:
    ii = parse_index_list(row.get("ssh_contour_boundary_i", ""))
    jj = parse_index_list(row.get("ssh_contour_boundary_j", ""))
    n = min(ii.size, jj.size)
    if n < 3:
        return None
    ii = ii[:n]
    jj = jj[:n]
    valid = np.isfinite(ii) & np.isfinite(jj)
    ii = ii[valid].astype(int)
    jj = jj[valid].astype(int)
    if ii.size < 3:
        return None
    i0, i1 = int(np.nanmin(ii)), int(np.nanmax(ii))
    j0, j1 = int(np.nanmin(jj)), int(np.nanmax(jj))
    width = i1 - i0 + 3
    height = j1 - j0 + 3
    if width <= 2 or height <= 2 or width * height > 2_000_000:
        return None
    pts = [(int(i - i0 + 1), int(j - j0 + 1)) for i, j in zip(ii, jj)]
    image = Image.new("1", (width, height), 0)
    draw = ImageDraw.Draw(image)
    draw.polygon(pts, fill=1)
    mask = np.asarray(image, dtype=bool)
    area = int(mask.sum())
    if area <= 0:
        return None
    return {"i0": i0 - 1, "j0": j0 - 1, "mask": mask, "area": area}


def contour_overlap_fraction(a: dict[str, object] | None, b: dict[str, object] | None) -> float:
    if a is None or b is None:
        return np.nan
    ma = a["mask"]
    mb = b["mask"]
    ai0 = int(a["i0"])
    aj0 = int(a["j0"])
    bi0 = int(b["i0"])
    bj0 = int(b["j0"])
    ai1 = ai0 + ma.shape[1]
    aj1 = aj0 + ma.shape[0]
    bi1 = bi0 + mb.shape[1]
    bj1 = bj0 + mb.shape[0]
    i0 = max(ai0, bi0)
    i1 = min(ai1, bi1)
    j0 = max(aj0, bj0)
    j1 = min(aj1, bj1)
    if i0 >= i1 or j0 >= j1:
        return 0.0
    a_slice = ma[j0 - aj0 : j1 - aj0, i0 - ai0 : i1 - ai0]
    b_slice = mb[j0 - bj0 : j1 - bj0, i0 - bi0 : i1 - bi0]
    overlap = int(np.logical_and(a_slice, b_slice).sum())
    denom = max(min(int(a["area"]), int(b["area"])), 1)
    return float(overlap / denom)


def rank_tuple(row: pd.Series) -> tuple[float, float, float, float]:
    amp = numeric(row.get("ssh_contour_amplitude_cm", row.get("amplitude", np.nan)))
    shape = numeric(row.get("shape_error_percent", np.nan))
    jet = numeric(row.get("jet_core_overlap_fraction", np.nan))
    rad = radius_km(row)
    return (
        -amp if np.isfinite(amp) else 0.0,
        shape if np.isfinite(shape) else 999.0,
        jet if np.isfinite(jet) else 999.0,
        abs(rad - 80.0) if np.isfinite(rad) else 9999.0,
    )


def apply_persistence_diagnostic(out: pd.DataFrame, args: argparse.Namespace) -> None:
    root = args.persistence_source_root or args.source_root
    day0 = pd.Timestamp(str(args.day))
    future_by_day: dict[str, pd.DataFrame] = {}
    for offset in range(1, int(args.persistence_lookahead_days) + 1):
        day = (day0 + pd.Timedelta(days=offset)).strftime("%Y-%m-%d")
        ymd = day.replace("-", "")
        try:
            source_root = table_root(root, ymd)
            future = read_table(source_root / "centers_hua_style")
        except FileNotFoundError:
            continue
        normalize_dates(future)
        if "date" in future.columns:
            future = future[future["date"].astype(str).eq(day)].copy()
        if "depth_index" in future.columns:
            future = future[future["depth_index"].astype(int).eq(0)].copy()
        if "hua_pass" in future.columns:
            future = future[future["hua_pass"].fillna(False).astype(bool)].copy()
        # Keep empty existing days: an empty day is a real break in a
        # consecutive track, unlike a missing file outside the run window.
        future_by_day[day] = future

    accepted_idx = out.index[out["qc_pass"].astype(bool)]
    if not future_by_day:
        out.loc[accepted_idx, "persistence_class"] = "not_evaluated_no_future_days"
        return

    days_available = len(future_by_day)
    future_arrays: dict[str, dict[str, np.ndarray]] = {}
    for day, future in future_by_day.items():
        if future.empty:
            future_arrays[day] = {
                "lon": np.empty(0, dtype="f8"),
                "lat": np.empty(0, dtype="f8"),
                "radius": np.empty(0, dtype="f8"),
                "polarity": np.empty(0, dtype=object),
            }
            continue
        future = future.copy()
        future["_lon"] = future.apply(row_lon, axis=1)
        future["_lat"] = future.apply(row_lat, axis=1)
        future["_radius_km"] = future.apply(radius_km, axis=1)
        valid = (
            np.isfinite(future["_lon"])
            & np.isfinite(future["_lat"])
            & np.isfinite(future["_radius_km"])
            & (future["_radius_km"] > 0)
        )
        future = future.loc[valid].copy()
        future_arrays[day] = {
            "lon": future["_lon"].to_numpy(dtype="f8"),
            "lat": future["_lat"].to_numpy(dtype="f8"),
            "radius": future["_radius_km"].to_numpy(dtype="f8"),
            "polarity": future.get("polarity", pd.Series("", index=future.index)).astype(str).to_numpy(dtype=object),
        }

    min_days = max(2, int(args.persistence_min_consecutive_days))
    for idx in accepted_idx:
        row = out.loc[idx]
        r0 = radius_km(row)
        lon0 = row_lon(row)
        lat0 = row_lat(row)
        matches = 0
        if np.isfinite(r0) and r0 > 0 and np.isfinite(lon0) and np.isfinite(lat0):
            polarity = str(row.get("polarity", ""))
            open_ocean = is_open_ocean(row)
            lat_abs = abs(float(lat0))
            lat_min = float(args.persistence_open_ocean_lat_min)
            lat_max = max(lat_min + 1.0, float(args.persistence_open_ocean_lat_max))
            latitude_weight = np.clip((lat_abs - lat_min) / (lat_max - lat_min), 0.0, 1.0)
            if open_ocean:
                # Open-ocean features are weaker and drift farther between
                # daily snapshots; relax smoothly toward higher latitudes.
                distance_factor = float(args.persistence_open_ocean_distance_factor) + latitude_weight * (
                    float(args.persistence_open_ocean_distance_factor_max)
                    - float(args.persistence_open_ocean_distance_factor)
                )
                radius_ratio_max = float(args.persistence_open_ocean_radius_ratio_max) + latitude_weight * (
                    float(args.persistence_open_ocean_radius_ratio_max_max)
                    - float(args.persistence_open_ocean_radius_ratio_max)
                )
            else:
                distance_factor = float(args.persistence_distance_factor)
                radius_ratio_max = float(args.persistence_radius_ratio_max)
            for offset in range(1, int(args.persistence_lookahead_days) + 1):
                day = (day0 + pd.Timedelta(days=offset)).strftime("%Y-%m-%d")
                future = future_arrays.get(day)
                if future is None:
                    break
                same = future["polarity"] == polarity
                if open_ocean:
                    drift_fraction = float(args.persistence_open_ocean_drift_fraction_min) + latitude_weight * (
                        float(args.persistence_open_ocean_drift_fraction_max)
                        - float(args.persistence_open_ocean_drift_fraction_min)
                    )
                    max_gap_days = int(round(latitude_weight * max(0, int(args.persistence_open_ocean_max_gap_days))))
                else:
                    drift_fraction = 0.0
                    max_gap_days = 0
                ok = np.zeros(future["lon"].size, dtype=bool)
                if np.any(same):
                    lon1 = future["lon"][same]
                    lat1 = future["lat"][same]
                    r1 = future["radius"][same]
                    dlon = ((lon1 - lon0 + 180.0) % 360.0) - 180.0
                    dlat = lat1 - lat0
                    lat_mid = 0.5 * (lat1 + lat0)
                    dist = np.hypot(dlon * 111.2 * np.maximum(np.cos(np.deg2rad(lat_mid)), 0.2), dlat * 111.2)
                    ratio = np.maximum(r0, r1) / np.maximum(np.minimum(r0, r1), 1.0e-12)
                    drift_allowance = drift_fraction * np.maximum(np.maximum(r0, r1), 50.0)
                    ok[same] = (dist <= distance_factor * np.maximum(r0, r1) + drift_allowance) & (ratio <= radius_ratio_max)
                if not np.any(ok):
                    if open_ocean and max_gap_days > 0 and offset - matches <= max_gap_days + 1:
                        continue
                    break
                matches += 1
        out.at[idx, "persistence_days_available"] = days_available
        out.at[idx, "persistence_match_count"] = int(matches)
        total_track_days = matches + 1
        if total_track_days < min_days:
            out.at[idx, "persistence_class"] = "transient"
        elif total_track_days == min_days:
            out.at[idx, "persistence_class"] = "short_track"
        else:
            out.at[idx, "persistence_class"] = "persistent_like"


def is_open_ocean(row: pd.Series) -> bool:
    """Use relaxed persistence only away from major boundary-current belts."""
    lon = row_lon(row)
    lat = row_lat(row)
    if not (np.isfinite(lon) and np.isfinite(lat)):
        return False
    jet_flag = row.get("jet_meander_flag", False)
    if isinstance(jet_flag, str):
        jet_flag = jet_flag.strip().lower() in {"1", "true", "yes", "y"}
    if bool(jet_flag):
        return False
    if -62.0 <= lat <= -40.0:
        return False
    boundary_boxes = (
        (120.0, 160.0, 20.0, 45.0),   # Kuroshio/Oyashio transition
        (260.0, 320.0, 20.0, 50.0),   # Gulf Stream/North Atlantic drift
        (225.0, 260.0, 15.0, 40.0),   # California Current
        (330.0, 360.0, 15.0, 40.0),   # Canary Current
        (0.0, 50.0, -50.0, -15.0),    # Benguela/Agulhas
        (140.0, 185.0, -50.0, -15.0), # East Australia/Leeuwin sector
        (285.0, 335.0, -50.0, -15.0), # Brazil/Malvinas sector
    )
    return not any(lon_min <= lon <= lon_max and lat_min <= lat <= lat_max for lon_min, lon_max, lat_min, lat_max in boundary_boxes)


def summarize(out: pd.DataFrame, candidate_idx: list[int]) -> list[dict[str, object]]:
    surface = out.loc[candidate_idx].copy()
    regions = {
        "global": (0.0, 360.0, -76.0, 76.0),
        "northeast_pacific_subtropical": (210.0, 250.0, 20.0, 45.0),
        "northwest_pacific_subtropical": (145.0, 180.0, 20.0, 45.0),
        "kuroshio": (120.0, 145.0, 20.0, 35.0),
        "north_pacific_interior_dense": (170.0, 210.0, 20.0, 40.0),
        "acc_reference": (0.0, 360.0, -62.0, -40.0),
    }
    rows: list[dict[str, object]] = []
    for region, bbox in regions.items():
        part = surface_in_bbox(surface, bbox)
        rows.append({"region": region, "metric": "raw_ssh_primary", "count": int(len(part))})
        rows.append({"region": region, "metric": "streamline_gate_removed_promoted", "count": int(part.get("streamline_gate_removed", pd.Series(False, index=part.index)).fillna(False).astype(bool).sum())})
        for label in ["shape_rejected", "overlap_duplicate", "jet_meander_candidate", "isolated_eddy_candidate"]:
            rows.append({"region": region, "metric": label, "count": int(part["qc_class"].eq(label).sum())})
        rows.append({"region": region, "metric": "qc_pass_total", "count": int(part["qc_pass"].astype(bool).sum())})
        rows.append({"region": region, "metric": "qc_rejected_total", "count": int((~part["qc_pass"].astype(bool)).sum())})
        rows.append({"region": region, "metric": "transient", "count": int(part["persistence_class"].eq("transient").sum())})
    return rows


def in_bbox(row: pd.Series, bbox: tuple[float, float, float, float]) -> bool:
    lon_col_values = [row.get(name, np.nan) for name in ["center_lon_refined", "center_lon", "ssh_contour_center_lon", "seed_lon"]]
    lat_col_values = [row.get(name, np.nan) for name in ["center_lat_refined", "center_lat", "ssh_contour_center_lat", "seed_lat"]]
    lon_value = next((numeric(v) for v in lon_col_values if np.isfinite(numeric(v))), np.nan)
    lat_value = next((numeric(v) for v in lat_col_values if np.isfinite(numeric(v))), np.nan)
    if not (np.isfinite(lon_value) and np.isfinite(lat_value)):
        return False
    lon_min, lon_max, lat_min, lat_max = bbox
    return bool(lon_min <= lon_value <= lon_max and lat_min <= lat_value <= lat_max)


def surface_in_bbox(surface: pd.DataFrame, bbox: tuple[float, float, float, float]) -> pd.DataFrame:
    lon_col = first_existing(surface, ["center_lon_refined", "center_lon", "ssh_contour_center_lon", "seed_lon"])
    lat_col = first_existing(surface, ["center_lat_refined", "center_lat", "ssh_contour_center_lat", "seed_lat"])
    if lon_col is None or lat_col is None:
        return surface.iloc[0:0].copy()
    lon_min, lon_max, lat_min, lat_max = bbox
    lon_values = pd.to_numeric(surface[lon_col], errors="coerce")
    lat_values = pd.to_numeric(surface[lat_col], errors="coerce")
    return surface[lon_values.between(lon_min, lon_max) & lat_values.between(lat_min, lat_max)].copy()


def first_existing(df: pd.DataFrame, names: list[str]) -> str | None:
    for name in names:
        if name in df.columns:
            return name
    return None


def write_catalog(df: pd.DataFrame, path: Path, qc_class: str) -> None:
    subset = df[df["qc_class"].eq(qc_class)].copy()
    subset.to_csv(path, index=False)


def parse_index_list(value: object) -> np.ndarray:
    text = "" if pd.isna(value) else str(value)
    if not text:
        return np.asarray([], dtype=int)
    out: list[int] = []
    for part in text.replace(",", ";").split(";"):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(int(float(part)))
        except ValueError:
            continue
    return np.asarray(out, dtype=int)


def numeric(value: object) -> float:
    try:
        return float(value)
    except Exception:
        return np.nan


if __name__ == "__main__":
    main()
