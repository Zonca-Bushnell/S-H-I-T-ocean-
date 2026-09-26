"""Canonical no-tile-cap SSH-primary surface detection stage."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path

from Detection_for_OFES.profiles import geometry_vertical_profile


def _dates(start: date, end: date):
    for offset in range((end - start).days + 1):
        yield start + timedelta(days=offset)


def _command(filter_root: Path, output: Path, day: date, profile_name: str) -> list[str]:
    profile = geometry_vertical_profile(profile_name)
    p = profile.detection
    return [
        sys.executable, "-m", "Detection_for_OFES.core.detection",
        "--filter-root", str(filter_root), "--raw-root", str(filter_root),
        "--filter-template", "global_phy_{yyyymmdd}.nc", "--raw-template", "global_phy_{yyyymmdd}.nc",
        "--output-dir", str(output), "--start", day.isoformat(), "--end", day.isoformat(),
        "--max-depth-m", "3", "--boundary-mode", "ssh_primary_open_ocean_no_streamline_gate",
        "--candidate-selection", profile.surface.candidate_selection,
        "--max-candidates-per-day", "0", "--seed-windows-cells", p.seed_windows_cells,
        "--start-radius-cells", str(p.start_radius_cells), "--max-radius-cells", str(p.max_radius_cells),
        "--ssh-window-cells", str(p.ssh_window_cells), "--ssh-primary-level-count", str(p.ssh_primary_level_count),
        "--ssh-primary-window-factor", str(p.ssh_primary_window_factor),
        "--ssh-primary-max-radius-factor", str(p.ssh_primary_max_radius_factor),
        "--ssh-primary-min-amplitude-cm", str(p.ssh_primary_min_amplitude_cm),
        "--ssh-primary-open-ocean-min-amplitude-cm", str(p.open_ocean_min_amplitude_cm),
        "--ssh-primary-open-ocean-low-lat-amplitude-cm", str(p.open_ocean_low_lat_amplitude_cm),
        "--ssh-primary-open-ocean-high-lat-amplitude-cm", str(p.open_ocean_high_lat_amplitude_cm),
        "--ssh-primary-open-ocean-window-factor", str(p.open_ocean_window_factor),
        "--ssh-primary-open-ocean-max-radius-factor", str(p.open_ocean_max_radius_factor),
        "--ssh-open-ocean-seed-window-cells", str(p.open_ocean_seed_window_cells),
        "--target-open-ocean-boxes", p.target_boxes, "--target-open-ocean-tile-top-n", "60",
        "--target-open-ocean-min-amplitude-cm", str(p.target_min_amplitude_cm),
        "--target-open-ocean-window-factor", str(p.target_window_factor),
        "--target-open-ocean-max-radius-factor", str(p.target_max_radius_factor),
        "--ssh-primary-max-shape-error-percent", str(p.contour_shape_error_percent),
        "--ssh-primary-acc-max-shape-error-percent", str(p.contour_acc_shape_error_percent),
        "--hua-backend", "python", "--intra-day-workers", "1", "--preload-day-uv",
        "--skip-axis-examples", "--skip-open-ocean-streamline-diagnostic",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--filter-output-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--profile", default="eta_hp500_geometry_vertical_v1")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    start, end = date.fromisoformat(args.start), date.fromisoformat(args.end)
    env = {**os.environ, "HDF5_USE_FILE_LOCKING": "FALSE", "PYTHONNOUSERSITE": "1"}

    def run_day(day: date) -> None:
        output = args.output_root / "raw_detection" / "daily_runs" / day.strftime("%Y%m%d")
        output.mkdir(parents=True, exist_ok=True)
        subprocess.run(_command(args.filter_output_root, output, day, args.profile), check=True, env=env)

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        list(pool.map(run_day, _dates(start, end)))


if __name__ == "__main__":
    main()
