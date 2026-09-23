"""Render reproducible, depth-ranked signed-section panels for a Hua run.

This is a diagnostic renderer.  It deliberately ranks existing accepted tracks
instead of randomly sampling them, and it does not alter vertical acceptance.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

from Detection_for_OFES.tools.run_relaxed_hua_vertical_jan1 import (
    DAY,
    FILTER_ROOT,
    REGIONS,
    create_panel_bridge,
    region_mask,
)
from Origin_eddy_detection.src.post.original_eddy_panels import _candidate_objects


def read_bridge_table(path: Path) -> pd.DataFrame:
    csv = path.with_suffix(".csv")
    if path.exists():
        try:
            return pd.read_parquet(path)
        except (ImportError, OSError):
            if not csv.exists():
                raise
    return pd.read_csv(csv)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render signed-speed curve-section panels for the deepest tracks in each diagnostic region."
    )
    parser.add_argument("--vertical-root", type=Path, required=True)
    parser.add_argument("--per-region", type=int, default=4)
    parser.add_argument("--quality-summary", type=Path, default=None)
    parser.add_argument("--min-bipolar-fraction", type=float, default=None)
    parser.add_argument("--output-dir-name", default="curve_section_quality_ranked")
    parser.add_argument("--figure-dpi", type=int, default=150)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.vertical_root
    detection = root / "raw_detection" / "daily_runs" / "19910101"
    summary_path = root / "vertical_object_summary.csv"
    if not detection.exists() or not summary_path.exists():
        raise FileNotFoundError("Vertical output is incomplete; structures and vertical_object_summary.csv are required")

    output_root = root / args.output_dir_name
    bridge = output_root / "panel_bridge"
    create_panel_bridge(detection, bridge)
    centers = read_bridge_table(bridge / "catalog" / "layer_centers_completed.parquet")
    shapes = read_bridge_table(bridge / "shape_classification_detect_only" / "shape_tracks.parquet")
    _, candidates = _candidate_objects(centers, shapes, {"detect_only"}, min_layers=2, year_limit=1991)
    summary = pd.read_csv(summary_path).set_index("hua_object_id")
    source_by_track = shapes.set_index("track3d_id")["source_hua_object_id"].astype(str)
    quality = None
    if args.quality_summary is not None:
        quality = pd.read_csv(args.quality_summary).set_index("hua_object_id")
        needed = {"bipolar_fraction_0p6r", "bipolar_fraction_1p0r"}
        missing = needed.difference(quality.columns)
        if missing:
            raise ValueError(f"Quality summary lacks required columns: {sorted(missing)}")

    selected_rows: list[dict[str, object]] = []
    for region, box in REGIONS.items():
        region_rows: list[dict[str, object]] = []
        for _, candidate in candidates.iterrows():
            source_id = source_by_track.loc[int(candidate["track3d_id"])]
            if source_id not in summary.index:
                continue
            stats = summary.loc[source_id]
            if quality is not None:
                if source_id not in quality.index:
                    continue
                quality_row = quality.loc[source_id]
                score = float(min(quality_row["bipolar_fraction_0p6r"], quality_row["bipolar_fraction_1p0r"]))
                if args.min_bipolar_fraction is not None and score < args.min_bipolar_fraction:
                    continue
            else:
                quality_row = None
                score = np.nan
            if not bool(region_mask(pd.DataFrame([stats]), box).iloc[0]):
                continue
            row = candidate.to_dict()
            row["source_hua_object_id"] = source_id
            row["selection_region"] = region
            row["max_depth_m"] = float(stats["max_depth_m"])
            row["pass_layers"] = int(stats["pass_layers"])
            row["first_hard_failure"] = str(stats["first_hard_failure"])
            row["section_bipolarity_min_fraction"] = score
            if quality_row is not None:
                row["section_bipolarity_0p6r"] = float(quality_row["bipolar_fraction_0p6r"])
                row["section_bipolarity_1p0r"] = float(quality_row["bipolar_fraction_1p0r"])
            region_rows.append(row)
        region_rows.sort(key=lambda row: (row["pass_layers"], row["max_depth_m"]), reverse=True)
        for order, row in enumerate(region_rows[: max(1, args.per_region)], start=1):
            row["selection_rank"] = order
            row["output_dir"] = str(output_root / f"{region}_rank_{order:02d}_{row['source_hua_object_id']}")
            selected_rows.append(row)

    if not selected_rows:
        raise RuntimeError("No tracks with at least two accepted layers fall in the requested regions")
    output_root.mkdir(parents=True, exist_ok=True)
    # ``original_eddy_panels`` writes its own selected_objects_metadata.csv.
    # Keep our ranking input separate so that the selection rationale survives.
    metadata = output_root / "quality_ranked_selection_input.csv"
    pd.DataFrame(selected_rows).to_csv(metadata, index=False)
    (output_root / "selection_manifest.json").write_text(
        json.dumps(
            {
                "day": DAY,
                "selection": "per-region rank by accepted pass_layers then max_depth_m; no new QC gate",
                "minimum_bipolar_fraction": args.min_bipolar_fraction,
                "right_panel_mode": "signed_horizontal_speed",
                "w_section_mode": "axis_curved",
                "objects": selected_rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    input_root = output_root / "filter_input"
    input_root.mkdir(exist_ok=True)
    source = FILTER_ROOT / "global_phy_19910101.nc"
    alias = input_root / "global_phy_1991.nc"
    if alias.exists() or alias.is_symlink():
        alias.unlink()
    os.link(source, alias)
    command = [
        sys.executable, "-m", "Origin_eddy_detection.src.post.original_eddy_panels",
        "--results-root", str(bridge), "--shape-dir-name", "shape_classification_detect_only",
        "--raw-root", str(root / "unused_raw_for_velocity_sections"), "--filter-root", str(input_root),
        "--output-dir", str(output_root), "--selected-metadata", str(metadata), "--min-layers", "2",
        "--right-panel-mode", "signed_horizontal_speed", "--w-section-mode", "axis_curved",
        "--figure-dpi", str(args.figure_dpi),
        "--output-name-stem", "quality_ranked_signed_curve_section_family",
    ]
    subprocess.run(command, check=True)
    print(json.dumps({"output_root": str(output_root), "selected": len(selected_rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
