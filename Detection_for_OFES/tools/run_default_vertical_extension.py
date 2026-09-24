"""Legacy Rossby/catalog vertical workflow retained for historical reproduction.

The active OFES research default is ``run_default_geometry_vertical``.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from .extend_final_surface_vertical import DEFAULT_FILTER_ROOT, DEFAULT_OUTPUT_ROOT, DEFAULT_SURFACE_ROOT, write_table


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_FILTER_ROOT = Path(r"E:\DATA\01_Eddy_correspond\02_OFES\origin_compatible_filter")


def dates(start: date, end: date) -> list[date]:
    return [start + timedelta(days=offset) for offset in range((end - start).days + 1)]


def run_filter(args: argparse.Namespace, days: list[date]) -> None:
    missing = [args.filter_root / f"global_phy_{day:%Y%m%d}.nc" for day in days]
    if args.resume and all(path.exists() for path in missing):
        return
    cmd = [
        sys.executable, "-m", "Detection_for_OFES.tools.build_ofes_meso_filter",
        "--input-root", str(args.source_filter_root), "--output-root", str(args.filter_root),
        "--start", args.start, "--end", args.end, "--available-start", args.start, "--available-end", args.end,
        "--temporal-window-days", "1", "--filter-mode", "bandpass",
        "--large-cutoff-mode", "rossby_lower_latadaptive_upper",
        "--rossby-small-factor", "0.5", "--adaptive-large-cutoff-min-km", "180",
        "--adaptive-large-cutoff-max-km", "180", "--rossby-min-km", "10", "--rossby-max-km", "180",
        "--zonal-scale-mode", "km", "--max-depth-layers", str(args.max_depth_layers), "--overwrite",
    ]
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def run_days(args: argparse.Namespace, days: list[date]) -> None:
    logs = args.output_root / "logs"
    logs.mkdir(parents=True, exist_ok=True)

    def run_day(day: date) -> tuple[date, bool]:
        stem = f"{day:%Y%m%d}"
        out_dir = args.output_root / "raw_detection" / "daily_runs" / stem
        if args.resume and (out_dir / "centers_hua_style.parquet").exists():
            return day, True
        cmd = [
            sys.executable, "-m", "Detection_for_OFES.tools.extend_final_surface_vertical",
            "--surface-root", str(args.surface_root), "--filter-root", str(args.filter_root),
            "--output-root", str(args.output_root), "--day", day.isoformat(),
            "--max-depth-layers", str(args.max_depth_layers), "--resume",
            # Every layer starts with the calibrated tangent rule.  Only a
            # rejected layer is retried with a looser, non-circular near-closed
            # streamline; the next layer returns to tangent-first.
            "--deep-hua-mode", "tangent_then_near_closed_streamline",
            "--deep-center-selection", "local_step_min",
            "--deep-center-step-cells", "2",
            "--deep-tangent-tolerance-deg", "45",
            "--deep-min-tangent-fraction", "0.35",
            "--enforce-tangent-alignment-hard-gate",
            "--disable-angle-jump-hard-gate",
            "--disable-direction-exception-hard-gate",
            "--disable-opposite-reversal-hard-gate",
            "--near-streamline-min-points", "12",
            "--near-streamline-min-winding-turns", "0.50",
            "--near-streamline-closure-tolerance-cells", "2.50",
            "--near-streamline-min-finite-fraction", "0.90",
        ]
        env = os.environ.copy()
        env["HDF5_USE_FILE_LOCKING"] = "FALSE"
        with (logs / f"{stem}.stdout.log").open("w", encoding="utf-8") as out, (logs / f"{stem}.stderr.log").open("w", encoding="utf-8") as err:
            result = subprocess.run(cmd, cwd=REPO_ROOT, stdout=out, stderr=err, env=env)
        return day, result.returncode == 0

    with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
        futures = [pool.submit(run_day, day) for day in days]
        for future in as_completed(futures):
            day, ok = future.result()
            if not ok:
                raise RuntimeError(f"Vertical extension failed for {day}; see {args.output_root / 'logs' / f'{day:%Y%m%d}.stderr.log'}")


def finalize(args: argparse.Namespace, days: list[date]) -> None:
    detection = args.output_root / "raw_detection"
    parts = [detection / "daily_runs" / f"{day:%Y%m%d}" for day in days]
    centers = pd.concat([pd.read_parquet(path / "centers_hua_style.parquet") for path in parts], ignore_index=True)
    structures = pd.concat([pd.read_parquet(path / "structures_hua_style.parquet") for path in parts], ignore_index=True)
    write_table(centers, detection / "centers_hua_style")
    write_table(structures, detection / "structures_hua_style")
    passed = centers[centers["hua_pass"].fillna(False).astype(bool)].copy()
    summary = (
        passed.groupby("hua_object_id", as_index=False)
        .agg(date=("date", "first"), polarity=("polarity", "first"), pass_layers=("depth_index", "size"), max_depth_m=("depth_m", "max"))
    )
    write_table(summary, detection / "frame_object_summary")

    tracking = args.output_root / "feature_group_tracking"
    subprocess.run([
        sys.executable, "-m", "Origin_eddy_detection.src.eddy_pipeline.tracking",
        "--input-dir", str(detection), "--output-dir", str(tracking), "--skip-plots",
    ], cwd=REPO_ROOT, check=True)
    subprocess.run([
        sys.executable, "-m", "Origin_eddy_detection.src.eddy_pipeline.catalog",
        "--detection-dir", str(detection), "--tracking-dir", str(tracking), "--output-root", str(args.output_root),
        "--start", args.start, "--end", args.end, "--lifetime-min-days", "1", "--radius-min-m", "25000",
        "--min-valid-layers", "6", "--shape-output-name", "shape_classification_1991_1991_hua_b3_start2_life1",
    ], cwd=REPO_ROOT, check=True)
    payload = {
        "days": len(days), "center_rows": int(len(centers)), "passed_rows": int(len(passed)), "vertical_objects": int(len(summary)),
        "vertical_profile": "continuous_local_min_2cells_tangent45_fraction35_then_near_closed_streamline",
        "diagnostic_only_gates": ["velocity_ratio", "angle_jump", "direction_exception", "opposite_reversal"],
    }
    (args.output_root / "vertical_extension_run_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the default OFES final-catalog vertical extension.")
    parser.add_argument("--legacy-rossby-workflow", action="store_true", help="Required acknowledgement for this historical workflow.")
    parser.add_argument("--surface-root", type=Path, default=DEFAULT_SURFACE_ROOT)
    parser.add_argument("--source-filter-root", type=Path, default=SOURCE_FILTER_ROOT)
    parser.add_argument("--filter-root", type=Path, default=DEFAULT_FILTER_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--start", default="1991-01-01")
    parser.add_argument("--end", default="1991-01-19")
    parser.add_argument("--max-depth-layers", type=int, default=105)
    parser.add_argument("--workers", type=int, default=19)
    parser.add_argument("--skip-filter", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not args.legacy_rossby_workflow:
        raise SystemExit(
            "This is a legacy Rossby/catalog workflow. Use "
            "Detection_for_OFES.tools.run_default_geometry_vertical, or pass "
            "--legacy-rossby-workflow for historical reproduction."
        )
    selected = dates(date.fromisoformat(args.start), date.fromisoformat(args.end))
    args.output_root.mkdir(parents=True, exist_ok=True)
    if not args.skip_filter:
        run_filter(args, selected)
    run_days(args, selected)
    finalize(args, selected)


if __name__ == "__main__":
    main()
