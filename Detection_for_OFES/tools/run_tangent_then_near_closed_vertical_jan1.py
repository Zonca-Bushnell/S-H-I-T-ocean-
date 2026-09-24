"""Run Jan1 vertical continuation with tangent-first, relaxed-streamline fallback."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd


DAY = "1991-01-01"
ROOT = Path(
    r"E:\DATA\01_Eddy_correspond\02_OFES\Fromthebeginning\04_Vertical"
    r"\eta_highpass_500km_no_tilecap_ssh_geometry_section_bipolar_jan01_jan19"
)
QC_TABLE = ROOT / "surface_geometry_qc" / "daily_runs" / "19910101" / "centers_hua_style.csv"
VELOCITY_ROOT = ROOT / "full_depth_velocity_highpass_500km"
OUTPUT_ROOT = ROOT / "vertical_continuation_tangent_then_near_closed_relaxed_jan01"


def run_logged(command: list[str], stdout: Path, stderr: Path) -> None:
    stdout.parent.mkdir(parents=True, exist_ok=True)
    with stdout.open("w", encoding="utf-8") as out, stderr.open("w", encoding="utf-8") as err:
        result = subprocess.run(
            command, cwd=Path.cwd(), stdout=out, stderr=err, text=True,
            env={**os.environ, "HDF5_USE_FILE_LOCKING": "FALSE"},
        )
    if result.returncode:
        raise RuntimeError(f"Command failed ({result.returncode}); see {stderr}")


def main() -> None:
    if not QC_TABLE.exists() or not (VELOCITY_ROOT / "global_phy_19910101.nc").exists():
        raise FileNotFoundError("Required Jan1 QC table or 105-layer velocity input is missing")
    source = pd.read_csv(QC_TABLE)
    selected = source["qc_pass"].fillna(False).astype(str).str.lower().isin(("true", "1", "yes"))
    if int(selected.sum()) != 1163:
        raise ValueError(f"Expected 1163 qc_pass=True objects, found {int(selected.sum())}")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    logs = OUTPUT_ROOT / "logs"
    started = time.time()
    python = sys.executable
    run_logged(
        [python, "-m", "unittest", "Origin_eddy_detection.tests.test_near_closed_streamline"],
        logs / "kernel_tests.stdout.log", logs / "kernel_tests.stderr.log",
    )
    run_logged(
        [
            python, "-m", "Detection_for_OFES.tools.extend_final_surface_vertical",
            "--surface-table", str(QC_TABLE), "--filter-root", str(VELOCITY_ROOT), "--output-root", str(OUTPUT_ROOT),
            "--day", DAY, "--max-depth-layers", "105", "--deep-search-cells", "6",
            "--start-radius-cells", "2", "--max-radius-cells", "12",
            "--deep-hua-mode", "tangent_then_near_closed_streamline",
            "--deep-center-selection", "local_step_section_bipolar", "--deep-center-step-cells", "2",
            "--deep-center-speed-tolerance", "0.20", "--deep-center-min-axis-km", "1.0",
            "--deep-tangent-tolerance-deg", "45", "--deep-min-tangent-fraction", "0.35",
            "--enforce-tangent-alignment-hard-gate", "--disable-angle-jump-hard-gate",
            "--disable-direction-exception-hard-gate", "--disable-opposite-reversal-hard-gate",
            "--near-streamline-min-points", "12", "--near-streamline-min-winding-turns", "0.50",
            "--near-streamline-closure-tolerance-cells", "2.50", "--near-streamline-min-finite-fraction", "0.90",
        ],
        logs / "vertical_extension.stdout.log", logs / "vertical_extension.stderr.log",
    )
    run_logged(
        [python, "-m", "Detection_for_OFES.tools.summarize_qc_vertical_extension", "--vertical-root", str(OUTPUT_ROOT), "--day", DAY],
        logs / "summary.stdout.log", logs / "summary.stderr.log",
    )
    manifest = {
        "day": DAY,
        "surface_qc_pass_objects": 1163,
        "surface_table": str(QC_TABLE),
        "velocity_root": str(VELOCITY_ROOT),
        "mode": "tangent_then_near_closed_streamline",
        "primary_tangent_rule": {"tolerance_degrees": 45.0, "minimum_fraction": 0.35},
        "fallback_near_closed_rule": {
            "radii_cells": [2, 12], "start_angles": 4, "directions": 2,
            "min_points": 12, "min_winding_turns": 0.50,
            "closure_tolerance_cells": 2.50, "min_finite_fraction": 0.90,
        },
        "unchanged_center_continuation": {"selection": "local_step_section_bipolar", "step_cells": 2, "search_cap_cells": 6},
        "elapsed_seconds": time.time() - started,
    }
    (OUTPUT_ROOT / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "complete", "output_root": str(OUTPUT_ROOT), "elapsed_seconds": manifest["elapsed_seconds"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
