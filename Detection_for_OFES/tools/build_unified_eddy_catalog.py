from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
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
    r"E:\DATA\01_Eddy_correspond\02_OFES\origin_unified_ssh_primary_target_all_open_ocean_recovery_surface_jan01_jan19"
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

    if not args.skip_detection:
        run_days(days, args, log_root)
    run_qc_days(days, args)
    write_final_catalog(days, args)
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
        description=(
            "Build the default OFES surface catalog: Rossby-adaptive filtering, "
            "multiscale SSH seeds, four open-ocean recovery regions, SSH-primary "
            "open-ocean acceptance, shape/overlap QC, and persistence."
        )
    )
    parser.add_argument("--filter-input-root", type=Path, default=DEFAULT_FILTER_INPUT_ROOT)
    parser.add_argument("--filter-output-root", type=Path, default=DEFAULT_FILTER_OUTPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--detection-root", type=Path, default=None)
    parser.add_argument("--start", default="1991-01-01")
    parser.add_argument("--end", default="1991-01-19")
    parser.add_argument("--max-depth-m", type=float, default=3.0)
    parser.add_argument("--workers", type=int, default=19)
    parser.add_argument("--intra-day-workers", type=int, default=1)
    parser.add_argument(
        "--detection-retries",
        type=int,
        default=2,
        help="Retries per day after a native NetCDF/HDF5 worker failure.",
    )
    parser.add_argument("--skip-filter", action="store_true")
    parser.add_argument("--skip-detection", action="store_true")
    parser.add_argument("--resume", dest="resume", action="store_true", default=True, help="Skip days whose raw centers and structures tables already exist (default).")
    parser.add_argument("--rerun-existing", dest="resume", action="store_false", help="Re-run daily detection even when raw tables exist.")
    parser.add_argument("--force-filter", action="store_true")

    parser.add_argument("--candidate-selection", choices=["global_topn", "tile_topn"], default="tile_topn")
    parser.add_argument("--tile-top-n", type=int, default=10)
    parser.add_argument("--open-ocean-tile-top-n", type=int, default=30)
    parser.add_argument("--open-ocean-low-lat-tile-top-n", type=int, default=15)
    parser.add_argument("--seed-windows-cells", default="3,5,7,11")
    parser.add_argument("--start-radius-cells", type=int, default=2)
    parser.add_argument("--max-radius-cells", type=int, default=12)
    parser.add_argument("--ssh-window-cells", type=int, default=7)
    parser.add_argument("--ssh-primary-level-count", type=int, default=8)
    parser.add_argument("--ssh-primary-window-factor", type=float, default=4.0)
    parser.add_argument("--ssh-primary-max-radius-factor", type=float, default=2.0)
    parser.add_argument("--ssh-primary-min-amplitude-cm", type=float, default=1.0)
    parser.add_argument("--ssh-primary-open-ocean-min-amplitude-cm", type=float, default=0.4)
    parser.add_argument("--ssh-primary-open-ocean-low-lat-amplitude-cm", type=float, default=0.8)
    parser.add_argument("--ssh-primary-open-ocean-high-lat-amplitude-cm", type=float, default=0.25)
    parser.add_argument("--ssh-primary-open-ocean-window-factor", type=float, default=6.0)
    parser.add_argument("--ssh-primary-open-ocean-max-radius-factor", type=float, default=5.0)
    parser.add_argument("--ssh-open-ocean-seed-window-cells", type=int, default=3)
    parser.add_argument(
        "--target-open-ocean-boxes",
        default=(
            "190,245,25,55,north_pacific;190,280,-50,-30,south_pacific;"
            "320,330,25,45,north_atlantic;335,355,-45,-20,south_atlantic"
        ),
    )
    parser.add_argument("--target-open-ocean-tile-top-n", type=int, default=60)
    parser.add_argument("--target-open-ocean-min-amplitude-cm", type=float, default=0.10)
    parser.add_argument("--target-open-ocean-window-factor", type=float, default=8.0)
    parser.add_argument("--target-open-ocean-max-radius-factor", type=float, default=6.0)
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
    parser.add_argument("--open-ocean-max-shape-error-percent", type=float, default=80.0)
    parser.add_argument("--min-compactness", type=float, default=0.20)
    parser.add_argument("--open-ocean-min-compactness", type=float, default=0.12)
    parser.add_argument("--min-boundary-points", type=int, default=9)
    parser.add_argument("--open-ocean-min-boundary-points", type=int, default=6)
    parser.add_argument("--min-radius-km", type=float, default=25.0)
    parser.add_argument("--open-ocean-min-radius-km", type=float, default=18.0)
    parser.add_argument("--min-area-cells", type=float, default=16.0)
    parser.add_argument("--open-ocean-min-area-cells", type=float, default=9.0)
    parser.add_argument("--overlap-center-factor", type=float, default=0.75)
    parser.add_argument("--overlap-area-fraction", type=float, default=0.50)
    parser.add_argument("--persistence-lookahead-days", type=int, default=5)
    parser.add_argument("--persistence-min-consecutive-days", type=int, default=2)
    parser.add_argument("--persistence-distance-factor", type=float, default=1.5)
    parser.add_argument("--persistence-radius-ratio-max", type=float, default=2.0)
    parser.add_argument("--persistence-open-ocean-distance-factor", type=float, default=2.5)
    parser.add_argument("--persistence-open-ocean-radius-ratio-max", type=float, default=3.0)
    parser.add_argument("--persistence-open-ocean-distance-factor-max", type=float, default=3.5)
    parser.add_argument("--persistence-open-ocean-radius-ratio-max-max", type=float, default=4.0)
    parser.add_argument("--persistence-open-ocean-lat-min", type=float, default=20.0)
    parser.add_argument("--persistence-open-ocean-lat-max", type=float, default=60.0)
    parser.add_argument("--persistence-open-ocean-drift-fraction-min", type=float, default=0.10)
    parser.add_argument("--persistence-open-ocean-drift-fraction-max", type=float, default=0.35)
    parser.add_argument("--persistence-open-ocean-max-gap-days", type=int, default=1)
    parser.add_argument("--jet-core-overlap-max", type=float, default=0.50)
    parser.add_argument("--open-ocean-streamline-diagnostic", action="store_true", help="Retain the expensive streamline trace for SSH-accepted open-ocean candidates.")
    parser.add_argument("--enable-jet-split", action="store_true")
    parser.add_argument("--open-ocean-ssh-fallback", action="store_true")
    parser.add_argument("--open-ocean-no-streamline-gate", dest="open_ocean_no_streamline_gate", action="store_true", default=True, help="Accept SSH-primary contours in open ocean; retain streamline diagnostics and the effective gate elsewhere (default).")
    parser.add_argument("--open-ocean-streamline-gate", dest="open_ocean_no_streamline_gate", action="store_false", help="Use the streamline gate in open ocean for a legacy diagnostic run.")
    parser.add_argument("--regional-amplitude-profile", type=Path, default=None)
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
        if args.resume and all((out_dir / name).exists() for name in ("centers_hua_style.csv", "structures_hua_style.csv")):
            stdout_path.write_text(f"[resume] skipped existing {day_str(day)}\n", encoding="utf-8")
            stderr_path.write_text("", encoding="utf-8")
            return day, True, str(stderr_path)
        cmd = detection_command(args, day, out_dir)
        run_env = os.environ.copy()
        # Avoid HDF5 file-lock/cache contention when many daily NetCDF files
        # are opened concurrently by separate detection workers.
        run_env.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
        attempts = max(0, int(args.detection_retries)) + 1
        for attempt in range(1, attempts + 1):
            mode = "w" if attempt == 1 else "a"
            with stdout_path.open(mode, encoding="utf-8") as stdout, stderr_path.open(mode, encoding="utf-8") as stderr:
                if attempt > 1:
                    stdout.write(f"\n[retry] day={day_str(day)} attempt={attempt}/{attempts}\n")
                    stderr.write(f"\n[retry] day={day_str(day)} attempt={attempt}/{attempts}\n")
                proc = subprocess.run(cmd, cwd=REPO_ROOT, stdout=stdout, stderr=stderr, env=run_env)
            if proc.returncode == 0:
                return day, True, str(stderr_path)
            if attempt < attempts:
                time.sleep(float(attempt))
        return day, False, str(stderr_path)

    workers = max(1, int(args.workers))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run_day, day) for day in days]
        for future in as_completed(futures):
            day, ok, stderr_path = future.result()
            if not ok:
                raise RuntimeError(f"Detection failed for {day}; see {stderr_path}")


def detection_command(args: argparse.Namespace, day: date, out_dir: Path) -> list[str]:
    if args.open_ocean_no_streamline_gate:
        boundary_mode = "ssh_primary_open_ocean_no_streamline_gate"
    elif args.open_ocean_ssh_fallback:
        boundary_mode = "ssh_primary_velocity_streamline_effective_open_ocean_fallback"
    else:
        boundary_mode = "ssh_primary_velocity_streamline_effective"
    profile_args = (
        ["--regional-amplitude-profile", str(args.regional_amplitude_profile)]
        if args.regional_amplitude_profile
        else []
    )
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
        boundary_mode,
        "--candidate-selection",
        str(args.candidate_selection),
        "--tile-top-n",
        str(args.tile_top_n),
        "--open-ocean-tile-top-n",
        str(args.open_ocean_tile_top_n),
        "--open-ocean-low-lat-tile-top-n",
        str(args.open_ocean_low_lat_tile_top_n),
        "--seed-windows-cells",
        str(args.seed_windows_cells),
        "--intra-day-workers",
        str(args.intra_day_workers),
        "--start-radius-cells",
        str(args.start_radius_cells),
        "--max-radius-cells",
        str(args.max_radius_cells),
        "--ssh-window-cells",
        str(args.ssh_window_cells),
        "--ssh-primary-level-count",
        str(args.ssh_primary_level_count),
        "--ssh-primary-window-factor",
        str(args.ssh_primary_window_factor),
        "--ssh-primary-max-radius-factor",
        str(args.ssh_primary_max_radius_factor),
        "--ssh-primary-min-amplitude-cm",
        str(args.ssh_primary_min_amplitude_cm),
        "--ssh-primary-open-ocean-min-amplitude-cm",
        str(args.ssh_primary_open_ocean_min_amplitude_cm),
        "--ssh-primary-open-ocean-low-lat-amplitude-cm",
        str(args.ssh_primary_open_ocean_low_lat_amplitude_cm),
        "--ssh-primary-open-ocean-high-lat-amplitude-cm",
        str(args.ssh_primary_open_ocean_high_lat_amplitude_cm),
        "--ssh-primary-open-ocean-window-factor",
        str(args.ssh_primary_open_ocean_window_factor),
        "--ssh-primary-open-ocean-max-radius-factor",
        str(args.ssh_primary_open_ocean_max_radius_factor),
        "--ssh-open-ocean-seed-window-cells",
        str(args.ssh_open_ocean_seed_window_cells),
        "--target-open-ocean-boxes",
        str(args.target_open_ocean_boxes),
        "--target-open-ocean-tile-top-n",
        str(args.target_open_ocean_tile_top_n),
        "--target-open-ocean-min-amplitude-cm",
        str(args.target_open_ocean_min_amplitude_cm),
        "--target-open-ocean-window-factor",
        str(args.target_open_ocean_window_factor),
        "--target-open-ocean-max-radius-factor",
        str(args.target_open_ocean_max_radius_factor),
        *profile_args,
        "--ssh-primary-max-shape-error-percent",
        str(args.ssh_primary_max_shape_error_percent),
        "--ssh-primary-acc-max-shape-error-percent",
        str(args.ssh_primary_acc_max_shape_error_percent),
        "--hua-backend",
        str(args.hua_backend),
        "--preload-day-uv",
        "--skip-axis-examples",
        *( [] if args.open_ocean_streamline_diagnostic else ["--skip-open-ocean-streamline-diagnostic"] ),
    ]


def run_qc_days(days: list[date], args: argparse.Namespace) -> None:
    raw_root = args.detection_root or (args.output_root / "raw_detection")
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
            "--open-ocean-max-shape-error-percent",
            str(args.open_ocean_max_shape_error_percent),
            "--min-compactness",
            str(args.min_compactness),
            "--open-ocean-min-compactness",
            str(args.open_ocean_min_compactness),
            "--min-boundary-points",
            str(args.min_boundary_points),
            "--open-ocean-min-boundary-points",
            str(args.open_ocean_min_boundary_points),
            "--min-radius-km",
            str(args.min_radius_km),
            "--open-ocean-min-radius-km",
            str(args.open_ocean_min_radius_km),
            "--min-area-cells",
            str(args.min_area_cells),
            "--open-ocean-min-area-cells",
            str(args.open_ocean_min_area_cells),
            "--overlap-center-factor",
            str(args.overlap_center_factor),
            "--overlap-area-fraction",
            str(args.overlap_area_fraction),
            "--persistence-source-root",
            str(raw_root),
            "--persistence-lookahead-days",
            str(args.persistence_lookahead_days),
            "--persistence-min-consecutive-days",
            str(args.persistence_min_consecutive_days),
            "--persistence-distance-factor",
            str(args.persistence_distance_factor),
            "--persistence-radius-ratio-max",
            str(args.persistence_radius_ratio_max),
            "--persistence-open-ocean-distance-factor",
            str(args.persistence_open_ocean_distance_factor),
            "--persistence-open-ocean-radius-ratio-max",
            str(args.persistence_open_ocean_radius_ratio_max),
            "--persistence-open-ocean-distance-factor-max",
            str(args.persistence_open_ocean_distance_factor_max),
            "--persistence-open-ocean-radius-ratio-max-max",
            str(args.persistence_open_ocean_radius_ratio_max_max),
            "--persistence-open-ocean-lat-min",
            str(args.persistence_open_ocean_lat_min),
            "--persistence-open-ocean-lat-max",
            str(args.persistence_open_ocean_lat_max),
            "--persistence-open-ocean-drift-fraction-min",
            str(args.persistence_open_ocean_drift_fraction_min),
            "--persistence-open-ocean-drift-fraction-max",
            str(args.persistence_open_ocean_drift_fraction_max),
            "--persistence-open-ocean-max-gap-days",
            str(args.persistence_open_ocean_max_gap_days),
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


def write_final_catalog(days: list[date], args: argparse.Namespace) -> None:
    """Materialize the catalog consumed by default plots and downstream users."""
    for day in days:
        source = args.output_root / "daily_runs" / ymd(day)
        destination = args.output_root / "final_catalog" / "daily_runs" / ymd(day)
        centers = read_table(source / "centers_hua_style")
        structures = read_table(source / "structures_hua_style")
        if centers.empty:
            continue

        accepted = centers[centers["hua_pass"].fillna(False).astype(bool)].copy()
        if "persistence_class" in accepted.columns:
            accepted = accepted[~accepted["persistence_class"].astype(str).eq("transient")].copy()
        destination.mkdir(parents=True, exist_ok=True)
        accepted.to_csv(destination / "centers_hua_style.csv", index=False)

        if not structures.empty and "hua_object_id" in accepted.columns and "hua_object_id" in structures.columns:
            structures = structures[structures["hua_object_id"].isin(accepted["hua_object_id"])].copy()
        structures.to_csv(destination / "structures_hua_style.csv", index=False)


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
        for label in ["boundary_rejected", "shape_rejected", "overlap_duplicate", "transient", "short_track", "persistent_like", "not_evaluated_no_future_days"]:
            counts[label] = int(surface["qc_class"].astype(str).eq(label).sum()) if "qc_class" in surface.columns else 0
        if "persistence_class" in surface.columns:
            for label in ["transient", "short_track", "persistent_like", "not_evaluated_no_future_days"]:
                counts[f"persistence_{label}"] = int(surface["persistence_class"].astype(str).eq(label).sum())
        rows.append(counts)
    summary = pd.DataFrame(rows)
    summary.to_csv(args.output_root / "unified_catalog_summary.csv", index=False)
    (args.output_root / "unified_catalog_summary.json").write_text(
        summary.to_json(orient="records", indent=2, force_ascii=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
