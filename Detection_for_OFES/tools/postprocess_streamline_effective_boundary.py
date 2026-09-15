from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
ORIGIN_SRC = REPO_ROOT / "Origin_eddy_detection" / "src"
if str(ORIGIN_SRC) not in sys.path:
    sys.path.insert(0, str(ORIGIN_SRC))

from eddy_pipeline.detection_hybrid import DetectionParams, _jet_core_overlap_check, _streamline_contour_check  # noqa: E402


def main() -> None:
    args = parse_args()
    day = str(args.day)
    ymd = day.replace("-", "")
    source_root = args.source_root / "daily_runs" / ymd
    if not source_root.exists():
        source_root = args.source_root
    centers = read_table(source_root / "centers_hua_style")
    structures = read_table(source_root / "structures_hua_style")
    centers["date"] = pd.to_datetime(centers["date"]).dt.strftime("%Y-%m-%d")
    structures["date"] = pd.to_datetime(structures["date"]).dt.strftime("%Y-%m-%d")
    centers = centers[centers["date"].eq(day)].copy()
    structures = structures[structures["date"].eq(day)].copy()

    lon, lat, u, v = read_uv(args.filter_root / f"global_phy_{ymd}.nc")
    speed = np.hypot(u, v)
    params = DetectionParams(
        ssh_window_cells=3,
        start_radius_cells=int(args.start_radius_cells),
        max_radius_cells=int(args.max_radius_cells),
        streamline_start_angles=int(args.streamline_start_angles),
        streamline_step_cells=float(args.streamline_step_cells),
        streamline_max_steps=int(args.streamline_max_steps),
        streamline_closure_tolerance_cells=float(args.streamline_closure_tolerance_cells),
        streamline_min_winding_turns=float(args.streamline_min_winding_turns),
        streamline_min_points=int(args.streamline_min_points),
        speed_ratio_max=float(args.speed_ratio_max),
        angle_jump_max_deg=float(args.angle_jump_max_deg),
        tangent_tolerance_deg=float(args.tangent_tolerance_deg),
        symmetry_tolerance_deg=float(args.symmetry_tolerance_deg),
        min_tangent_fraction=float(args.min_tangent_fraction),
        min_reversal_fraction=float(args.min_reversal_fraction),
        min_finite_fraction=float(args.min_finite_fraction),
        direction_exception_extra=0,
        surface_search_cells=8,
        deep_search_cells=6,
        streamline_direction_exception_fraction=float(args.streamline_direction_exception_fraction),
        jet_core_speed_percentile=float(args.jet_core_speed_percentile),
        jet_core_overlap_max=float(args.jet_core_overlap_max),
    )

    surface_mask = centers.get("depth_index", pd.Series(0, index=centers.index)).astype(int).eq(0)
    accepted_mask = surface_mask & centers.get("hua_pass", pd.Series(False, index=centers.index)).fillna(False).astype(bool)
    new_rows: list[dict[str, object]] = []
    accepted_ids: set[str] = set()
    for _, row in centers.iterrows():
        out = row.to_dict()
        if bool(accepted_mask.loc[row.name]):
            center_i = float(row.get("ssh_contour_center_i", row.get("center_i_refined", row.get("seed_i", np.nan))))
            center_j = float(row.get("ssh_contour_center_j", row.get("center_j_refined", row.get("seed_j", np.nan))))
            ssh_radius = float(row.get("ssh_contour_radius_cells", row.get("accepted_radius_cells", params.start_radius_cells)))
            if not np.isfinite(center_i) or not np.isfinite(center_j):
                center_i = float(row.get("center_i_refined", row.get("seed_i", np.nan)))
                center_j = float(row.get("center_j_refined", row.get("seed_j", np.nan)))
            max_radius = int(min(int(args.max_effective_radius_cells), max(params.start_radius_cells, int(math.ceil(1.25 * ssh_radius)) + 2)))
            best = None
            first = None
            for radius in range(max_radius, params.start_radius_cells - 1, -1):
                stream = _streamline_contour_check(u, v, center_i, center_j, radius, params)
                if first is None:
                    first = stream
                if bool(stream.get("circle_passed", False)):
                    best = stream
                    break
            source = best if best is not None else (first or {})
            jet = _jet_core_overlap_check(speed, center_i, center_j, int(max(params.start_radius_cells, round(float(source.get("radius_cells", ssh_radius))))), params)
            passed = best is not None
            out.update(
                {
                    "boundary_mode": "ssh_primary_velocity_streamline_effective",
                    "surface_definition": "ssh_primary_velocity_streamline_effective",
                    "boundary_source": "velocity_streamline_effective_contour" if passed else "velocity_streamline_effective_rejected",
                    "catalog_acceptance_reason": "ssh_primary_velocity_streamline_effective" if passed else "no_closed_streamline_effective",
                    "hua_pass": bool(passed),
                    "circle_passed": bool(passed),
                    "accepted_radius_cells": float(source.get("radius_cells", np.nan)) if passed else 0.0,
                    "radius_cells": float(source.get("radius_cells", np.nan)),
                    "streamline_closed": bool(source.get("streamline_closed", False)),
                    "streamline_radius_cells": float(source.get("radius_cells", np.nan)),
                    "streamline_points": float(source.get("streamline_points", 0.0)),
                    "streamline_boundary_i": str(source.get("streamline_boundary_i", "")),
                    "streamline_boundary_j": str(source.get("streamline_boundary_j", "")),
                    "streamline_closure_error_cells": float(source.get("streamline_closure_error_cells", np.nan)),
                    "streamline_winding_turns": float(source.get("streamline_winding_turns", 0.0)),
                    "streamline_boundary_quality": "closed_streamline_effective" if passed else str(source.get("first_hard_failure", "no_closed_streamline")),
                    "dynamical_core_class": "closed_streamline_core" if passed else "no_streamline_core",
                    **jet,
                }
            )
            if passed:
                accepted_ids.add(str(row.get("hua_object_id", "")))
        new_rows.append(out)

    out_centers = pd.DataFrame(new_rows)
    out_structures = structures[structures["hua_object_id"].astype(str).isin(accepted_ids)].copy() if "hua_object_id" in structures.columns else structures.iloc[0:0].copy()
    out_dir = args.output_root / "daily_runs" / ymd
    out_dir.mkdir(parents=True, exist_ok=True)
    out_centers.to_csv(out_dir / "centers_hua_style.csv", index=False)
    out_structures.to_csv(out_dir / "structures_hua_style.csv", index=False)
    summary = {
        "day": day,
        "source_root": str(args.source_root),
        "output_root": str(args.output_root),
        "surface_source_pass": int(accepted_mask.sum()),
        "surface_streamline_effective_pass": int(out_centers[surface_mask]["hua_pass"].fillna(False).astype(bool).sum()),
        "streamline_start_angles": int(args.streamline_start_angles),
    }
    (out_dir / "streamline_effective_postprocess_summary.json").write_text(pd.Series(summary).to_json(force_ascii=False, indent=2), encoding="utf-8")
    print(summary)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Re-score SSH-primary OFES surface objects with velocity-streamline effective boundaries.")
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--filter-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--day", default="1991-01-01")
    parser.add_argument("--start-radius-cells", type=int, default=3)
    parser.add_argument("--max-radius-cells", type=int, default=8)
    parser.add_argument("--max-effective-radius-cells", type=int, default=16)
    parser.add_argument("--speed-ratio-max", type=float, default=3.0)
    parser.add_argument("--angle-jump-max-deg", type=float, default=150.0)
    parser.add_argument("--tangent-tolerance-deg", type=float, default=24.0)
    parser.add_argument("--symmetry-tolerance-deg", type=float, default=120.0)
    parser.add_argument("--min-tangent-fraction", type=float, default=0.70)
    parser.add_argument("--min-reversal-fraction", type=float, default=0.70)
    parser.add_argument("--min-finite-fraction", type=float, default=0.95)
    parser.add_argument("--streamline-direction-exception-fraction", type=float, default=0.10)
    parser.add_argument("--streamline-step-cells", type=float, default=0.5)
    parser.add_argument("--streamline-max-steps", type=int, default=120)
    parser.add_argument("--streamline-start-angles", type=int, default=2)
    parser.add_argument("--streamline-closure-tolerance-cells", type=float, default=1.75)
    parser.add_argument("--streamline-min-winding-turns", type=float, default=0.75)
    parser.add_argument("--streamline-min-points", type=int, default=16)
    parser.add_argument("--jet-core-speed-percentile", type=float, default=80.0)
    parser.add_argument("--jet-core-overlap-max", type=float, default=0.50)
    return parser.parse_args()


def read_table(path_without_suffix: Path) -> pd.DataFrame:
    for suffix in [".parquet", ".csv"]:
        path = path_without_suffix.with_suffix(suffix)
        if path.exists():
            return pd.read_parquet(path) if suffix == ".parquet" else pd.read_csv(path)
    raise FileNotFoundError(path_without_suffix)


def read_uv(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as ds:
        lon = ds["longitude"][:].astype("f8")
        lat = ds["latitude"][:].astype("f8")
        u = ds["uo_glor"][0, 0, :, :].astype("f4")
        v = ds["vo_glor"][0, 0, :, :].astype("f4")
    u[np.abs(u) > 1.0e20] = np.nan
    v[np.abs(v) > 1.0e20] = np.nan
    return lon, lat, u, v


if __name__ == "__main__":
    main()
