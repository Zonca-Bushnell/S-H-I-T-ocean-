"""Run the current OFES eta/500-km geometry-to-vertical research profile."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path

from Detection_for_OFES.profiles import OFES_DATA_ROOT, VELOCITY_SOURCE_ROOT, geometry_vertical_profile


REPO_ROOT = Path(__file__).resolve().parents[2]


def days(start: date, end: date) -> list[date]:
    return [start + timedelta(days=offset) for offset in range((end - start).days + 1)]


def run(command: list[str], *, stdout=None, stderr=None) -> None:
    subprocess.run(command, cwd=REPO_ROOT, check=True, stdout=stdout, stderr=stderr, env={**os.environ, "HDF5_USE_FILE_LOCKING": "FALSE"})


def daily_netcdf_complete(root: Path, selected: list[date]) -> bool:
    return all((root / f"global_phy_{current:%Y%m%d}.nc").exists() for current in selected)


def daily_table_complete(root: Path, selected: list[date]) -> bool:
    return all((root / "daily_runs" / f"{current:%Y%m%d}" / "centers_hua_style.csv").exists() for current in selected)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="eta_hp500_geometry_vertical")
    parser.add_argument("--start", default="1991-01-01")
    parser.add_argument("--end", default="1991-01-19")
    parser.add_argument("--workers", type=int, default=19)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    profile = geometry_vertical_profile(args.profile)
    start, end = date.fromisoformat(args.start), date.fromisoformat(args.end)
    selected = days(start, end)
    root = profile.output_root
    inputs = root / "surface_inputs_eta"
    filtered = root / "surface_inputs_highpass_500km"
    raw = root / "raw_detection"
    qc = root / "surface_geometry_qc"
    velocity = root / "full_depth_velocity_highpass_500km"
    vertical = root / "vertical_continuation_local_step_2cells_section_bipolar"
    logs = root / "logs"
    logs.mkdir(parents=True, exist_ok=True)

    filter_workers = min(3, max(1, args.workers))
    if not (args.resume and daily_netcdf_complete(inputs, selected)):
        run([sys.executable, "-m", "Detection_for_OFES.tools.build_ofes_eta_surface_inputs", "--data-root", str(OFES_DATA_ROOT), "--velocity-root", str(VELOCITY_SOURCE_ROOT), "--output-root", str(inputs), "--start", args.start, "--end", args.end], stdout=(logs / "eta_inputs.stdout.log").open("w", encoding="utf-8"), stderr=(logs / "eta_inputs.stderr.log").open("w", encoding="utf-8"))
    if not (args.resume and daily_netcdf_complete(filtered, selected)):
        run([sys.executable, "-m", "Detection_for_OFES.tools.build_ofes_meso_filter", "--input-root", str(inputs), "--output-root", str(filtered), "--start", args.start, "--end", args.end, "--available-start", args.start, "--available-end", args.end, "--temporal-window-days", str(profile.temporal_window_days), "--filter-mode", "highpass", "--large-cutoff-mode", "fixed", "--large-cutoff-km", str(profile.highpass_cutoff_km), "--spatial-kernel", profile.kernel, "--zonal-scale-mode", "km", "--meridional-scale-mode", "median", "--max-depth-layers", "1", "--science-tag", "eta_gaussian_highpass_500km", "--workers", str(filter_workers), "--convolution-engine", "fft", "--compression-level", "1", "--overwrite"], stdout=(logs / "surface_filter.stdout.log").open("w", encoding="utf-8"), stderr=(logs / "surface_filter.stderr.log").open("w", encoding="utf-8"))
    if not (args.resume and daily_table_complete(raw / "raw_detection", selected)):
        run([sys.executable, "-m", "Detection_for_OFES.tools.build_unified_eddy_catalog", "--filter-input-root", str(filtered), "--filter-output-root", str(filtered), "--output-root", str(raw), "--start", args.start, "--end", args.end, "--max-depth-m", "3", "--filter-max-depth-layers", "1", "--workers", str(args.workers), "--skip-filter", "--candidate-only", "--open-ocean-no-streamline-gate", "--candidate-selection", profile.candidate_selection], stdout=(logs / "surface_detection.stdout.log").open("w", encoding="utf-8"), stderr=(logs / "surface_detection.stderr.log").open("w", encoding="utf-8"))

    def qc_day(current: date) -> None:
        stem = f"{current:%Y%m%d}"
        if args.resume and (qc / "daily_runs" / stem / "centers_hua_style.csv").exists():
            return
        with (logs / f"{stem}.qc.stdout.log").open("w", encoding="utf-8") as out, (logs / f"{stem}.qc.stderr.log").open("w", encoding="utf-8") as err:
            run([sys.executable, "-m", "Detection_for_OFES.tools.postprocess_ofes_eddy_qc", "--source-root", str(raw / "raw_detection"), "--filter-root", str(filtered), "--output-root", str(qc), "--day", current.isoformat(), "--skip-persistence", "--accept-ssh-primary-without-streamline"], stdout=out, stderr=err)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(qc_day, selected))

    if not (args.resume and daily_netcdf_complete(velocity, selected)):
        missing_velocity_days = [
            current for current in selected
            if not (velocity / f"global_phy_{current:%Y%m%d}.nc").exists()
        ]
        velocity_start = missing_velocity_days[0] if args.resume else start
        velocity_end = missing_velocity_days[-1] if args.resume else end
        run([sys.executable, "-m", "Detection_for_OFES.tools.build_ofes_meso_filter", "--input-root", str(VELOCITY_SOURCE_ROOT), "--output-root", str(velocity), "--start", velocity_start.isoformat(), "--end", velocity_end.isoformat(), "--available-start", args.start, "--available-end", args.end, "--temporal-window-days", "1", "--filter-mode", "highpass", "--large-cutoff-mode", "fixed", "--large-cutoff-km", str(profile.highpass_cutoff_km), "--spatial-kernel", profile.kernel, "--zonal-scale-mode", "km", "--meridional-scale-mode", "median", "--max-depth-layers", "105", "--science-tag", "eta_hp500_velocity", "--workers", str(filter_workers), "--convolution-engine", "fft", "--compression-level", "1", "--overwrite"], stdout=(logs / "velocity_filter.stdout.log").open("w", encoding="utf-8"), stderr=(logs / "velocity_filter.stderr.log").open("w", encoding="utf-8"))

    def extend_day(current: date) -> None:
        stem = f"{current:%Y%m%d}"
        table = qc / "daily_runs" / stem / "centers_hua_style.csv"
        command = [sys.executable, "-m", "Detection_for_OFES.tools.extend_final_surface_vertical", "--surface-table", str(table), "--filter-root", str(velocity), "--output-root", str(vertical), "--day", current.isoformat(), "--max-depth-layers", "105", "--deep-search-cells", "6", "--start-radius-cells", "2", "--max-radius-cells", "12", "--deep-hua-mode", "full", "--deep-center-selection", profile.deep_center_selection, "--deep-center-step-cells", str(profile.deep_center_step_cells), "--deep-center-speed-tolerance", str(profile.deep_center_speed_tolerance), "--deep-tangent-tolerance-deg", str(profile.deep_tangent_tolerance_deg), "--deep-min-tangent-fraction", str(profile.deep_min_tangent_fraction), "--enforce-tangent-alignment-hard-gate", "--disable-angle-jump-hard-gate", "--disable-direction-exception-hard-gate", "--disable-opposite-reversal-hard-gate"]
        if args.resume:
            command.append("--resume")
        with (logs / f"{stem}.vertical.stdout.log").open("w", encoding="utf-8") as out, (logs / f"{stem}.vertical.stderr.log").open("w", encoding="utf-8") as err:
            run(command, stdout=out, stderr=err)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(extend_day, selected))
    (root / "profile_manifest.json").write_text(json.dumps({"profile": profile.name, "ssh_definition": profile.ssh_definition, "persistence": "not_enabled", "streamline_gate": "removed_for_ssh_primary_geometry_qc", "vertical": profile.__dict__}, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
