from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_FILTER_INPUT_ROOT = Path(r"E:\DATA\01_Eddy_correspond\02_OFES\origin_compatible_filter")
DEFAULT_FILTER_OUTPUT_ROOT = Path(
    r"E:\DATA\01_Eddy_correspond\02_OFES\origin_compatible_filter_rossby_lower_upper180"
)
DEFAULT_OUTPUT_ROOT = Path(
    r"E:\DATA\01_Eddy_correspond\02_OFES\origin_unified_ssh_streamline_rossby"
)
DEFAULT_ROSSBY_RADIUS_PATH = Path(
    r"E:\DATA\01_Eddy_correspond\02_OFES\rossby_radius_chelton1998"
    r"\unzip\fecampos-campos2025-a5d24c0\rossrad.nc"
)


def main() -> None:
    args = parse_args()
    start = parse_date(args.start)
    end = parse_date(args.end)
    days = [start + timedelta(days=offset) for offset in range((end - start).days + 1)]

    if not args.skip_filter:
        ensure_filter(args, days)

    args.output_root.mkdir(parents=True, exist_ok=True)
    log_root = args.output_root / "logs"
    log_root.mkdir(parents=True, exist_ok=True)

    run_days(days, args, log_root)
    run_qc_days(days, args)
    write_summary(days, args)

    print(
        json.dumps(
            {
                "output_root": str(args.output_root),
                "days": len(days),
                "filter_output_root": str(args.filter_output_root),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a unified OFES surface eddy catalog: Rossby filter, SSH discovery, streamline boundary, and QC."
    )
    parser.add_argument("--filter-input-root", type=Path, default=DEFAULT_FILTER_INPUT_ROOT)
    parser.add_argument("--filter-output-root", type=Path, default=DEFAULT_FILTER_OUTPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--start", default="1991-01-01")
    parser.add_argument("--end", default="1991-01-01")
    parser.add_argument("--max-depth-m", type=float, default=3.0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--skip-filter", action="store_true")
    parser.add_argument("--force-filter", action="store_true")

    parser.add_argument("--candidate-selection", choices=["global_topn", "tile_topn"], default="tile_topn")
    parser.add_argument("--tile-top-n", type=int, default=10)
    parser.add_argument("--start-radius-cells", type=int, default=2)
    parser.add_argument("--max-radius-cells", type=int, default=12)
    parser.add_argument("--ssh-primary-min-amplitude-cm", type=float, default=1.0)
    parser.add_argument("--ssh-primary-max-shape-error-percent", type=float, default=70.0)
    parser.add_argument("--ssh-primary-acc-max-shape-error-percent", type=float, default=55.0)
    parser.add_argument("--hua-backend", choices=["python", "matlab"], default="python")

    parser.add_argument("--filter-max-depth-layers", type=int, default=1)
    parser.add_argument("--temporal-window-days", type=int, default=1)
    parser.add_argument("--rossby-radius-path", type=Path, default=DEFAULT_ROSSBY_RADIUS_PATH)
    parser.add_argument("--adaptive-large-cutoff-min-km", type=float, default=180.0)
    parser.add_argument("--adaptive-large-cutoff-max-km", type=float, default=180.0)

    parser.add_argument("--max-shape-error-percent", type=float, default=70.0)
    parser.add_argument("--acc-max-shape-error-percent", type=float, default=55.0)
    parser.add_argument("--min-compactness", type=float, default=0.20)
    parser.add_argument("--min-boundary-points", type=int, default=9)
    parser.add_argument("--min-radius-km", type=float, default=25.0)
    parser.add_argument("--min-area-cells", type=float, default=16.0)
    parser.add_argument("--overlap-center-factor", type=float, default=0.75)
    parser.add_argument("--overlap-area-fraction", type=float, default=0.50)
    parser.add_argument("--persistence-lookahead-days", type=int, default=5)
    parser.add_argument("--persistence-distance-factor", type=float, default=1.5)
    parser.add_argument("--persistence-radius-ratio-max", type=float, default=2.0)
    parser.add_argument("--jet-core-overlap-max", type=float, default=0.50)
    parser.add_argument("--enable-jet-split", action="store_true")
    return parser.parse_args()


def parse_date(value: str) -> date:
    return date.fromisoformat(value)


def day_str(day: date) -> str:
    return day.isoformat()


def ymd(day: date) -> str:
    return day.strftime("%Y%m%d")


def ensure_filter(args: argparse.Namespace, days: list[date]) -> None:
    missing = []
    for day in days:
        path = args.filter_output_root / f"global_phy_{ymd(day)}.nc"
        if not path.exists():
            missing.append(path)
    if not missing and not args.force_filter:
        return

    cmd = [
        sys.executable,
        "-m",
        "Detection_for_OFES.tools.build_ofes_meso_filter",
        "--input-root",
        str(args.filter_input_root),
        "--output-root",
        str(args.filter_output_root),
        "--start",
        day_str(days[0]),
        "--end",
        day_str(days[-1]),
        "--temporal-window-days",
        str(args.temporal_window_days),
        "--filter-mode",
        "bandpass",
        "--large-cutoff-mode",
        "rossby_lower_latadaptive_upper",
        "--rossby-radius-path",
        str(args.rossby_radius_path),
        "--rossby-small-factor",
        "0.5",
        "--adaptive-large-cutoff-min-km",
        str(args.adaptive_large_cutoff_min_km),
        "--adaptive-large-cutoff-max-km",
        str(args.adaptive_large_cutoff_max_km),
        "--rossby-min-km",
        "10",
        "--rossby-max-km",
        "180",
        "--zonal-scale-mode",
        "km",
        "--max-depth-layers",
        str(args.filter_max_depth_layers),
    ]
    if args.force_filter or missing:
        cmd.append("--overwrite")
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def run_days(days: list[date], args: argparse.Namespace, log_root: Path) -> None:
    def run_day(day: date) -> tuple[date, bool, str]:
        out_dir = args.output_root / "raw_detection" / "daily_runs" / ymd(day)
        stdout_path = log_root / f"{ymd(day)}.stdout.log"
        stderr_path = log_root / f"{ymd(day)}.stderr.log"
        cmd = detection_command(args, day, out_dir)
        with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
            proc = subprocess.run(cmd, cwd=REPO_ROOT, stdout=stdout, stderr=stderr)
        return day, proc.returncode == 0, str(stderr_path)

    workers = max(1, int(args.workers))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run_day, day) for day in days]
        for future in as_completed(futures):
            day, ok, stderr_path = future.result()
            if not ok:
                raise RuntimeError(f"Detection failed for {day}; see {stderr_path}")


def detection_command(args: argparse.Namespace, day: date, out_dir: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        "Origin_eddy_detection.src.eddy_pipeline.detection_hybrid",
        "--filter-root",
        str(args.filter_output_root),
        "--raw-root",
        str(args.filter_output_root),
        "--filter-template",
        "global_phy_{yyyymmdd}.nc",
        "--raw-template",
        "global_phy_{yyyymmdd}.nc",
        "--output-dir",
        str(out_dir),
        "--start",
        day_str(day),
        "--end",
        day_str(day),
        "--max-depth-m",
        str(args.max_depth_m),
        "--boundary-mode",
        "ssh_primary_velocity_streamline_effective",
        "--candidate-selection",
        str(args.candidate_selection),
        "--tile-top-n",
        str(args.tile_top_n),
        "--start-radius-cells",
        str(args.start_radius_cells),
        "--max-radius-cells",
        str(args.max_radius_cells),
        "--ssh-primary-min-amplitude-cm",
        str(args.ssh_primary_min_amplitude_cm),
        "--ssh-primary-max-shape-error-percent",
        str(args.ssh_primary_max_shape_error_percent),
        "--ssh-primary-acc-max-shape-error-percent",
        str(args.ssh_primary_acc_max_shape_error_percent),
        "--hua-backend",
        str(args.hua_backend),
        "--preload-day-uv",
        "--skip-axis-examples",
    ]


def run_qc_days(days: list[date], args: argparse.Namespace) -> None:
    raw_root = args.output_root / "raw_detection"
    for day in days:
        cmd = [
            sys.executable,
            "-m",
            "Detection_for_OFES.tools.postprocess_ofes_eddy_qc",
            "--source-root",
            str(raw_root),
            "--filter-root",
            str(args.filter_output_root),
            "--output-root",
            str(args.output_root),
            "--day",
            day_str(day),
            "--max-shape-error-percent",
            str(args.max_shape_error_percent),
            "--acc-max-shape-error-percent",
            str(args.acc_max_shape_error_percent),
            "--min-compactness",
            str(args.min_compactness),
            "--min-boundary-points",
            str(args.min_boundary_points),
            "--min-radius-km",
            str(args.min_radius_km),
            "--min-area-cells",
            str(args.min_area_cells),
            "--overlap-center-factor",
            str(args.overlap_center_factor),
            "--overlap-area-fraction",
            str(args.overlap_area_fraction),
            "--persistence-source-root",
            str(raw_root),
            "--persistence-lookahead-days",
            str(args.persistence_lookahead_days),
            "--persistence-distance-factor",
            str(args.persistence_distance_factor),
            "--persistence-radius-ratio-max",
            str(args.persistence_radius_ratio_max),
            "--jet-core-overlap-max",
            str(args.jet_core_overlap_max),
        ]
        if args.enable_jet_split:
            cmd.append("--enable-jet-split")
        subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def read_table(path: Path) -> pd.DataFrame:
    csv = path.with_suffix(".csv")
    parquet = path.with_suffix(".parquet")
    if csv.exists():
        return pd.read_csv(csv)
    if parquet.exists():
        return pd.read_parquet(parquet)
    return pd.DataFrame()


def write_summary(days: list[date], args: argparse.Namespace) -> None:
    rows: list[dict[str, object]] = []
    for day in days:
        final = read_table(args.output_root / "daily_runs" / ymd(day) / "centers_hua_style")
        if final.empty:
            rows.append({"day": day_str(day), "final_pass": 0, "raw_pass": 0})
            continue
        surface = final[final["depth_index"].astype(int).eq(0)] if "depth_index" in final.columns else final
        final_pass = int(surface["hua_pass"].fillna(False).astype(bool).sum())
        raw_pass = int(surface["raw_hua_pass"].fillna(False).astype(bool).sum()) if "raw_hua_pass" in surface.columns else final_pass
        discovery_pass = (
            int(surface["ssh_primary_discovery_pass"].fillna(False).astype(bool).sum())
            if "ssh_primary_discovery_pass" in surface.columns
            else 0
        )
        counts = {
            "day": day_str(day),
            "raw_discovery_pass": discovery_pass,
            "raw_streamline_pass": raw_pass,
            "final_pass": final_pass,
        }
        for label in ["boundary_rejected", "shape_rejected", "overlap_duplicate", "transient"]:
            counts[label] = int(surface["qc_class"].astype(str).eq(label).sum()) if "qc_class" in surface.columns else 0
        rows.append(counts)
    summary = pd.DataFrame(rows)
    summary.to_csv(args.output_root / "unified_catalog_summary.csv", index=False)
    (args.output_root / "unified_catalog_summary.json").write_text(
        summary.to_json(orient="records", indent=2, force_ascii=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
