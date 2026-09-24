"""Run the isolated Jan1 non-circular near-closed-streamline validation."""
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
VELOCITY_FILE = VELOCITY_ROOT / "global_phy_19910101.nc"
BASELINE_ROOT = ROOT / "vertical_continuation_local_step_2cells_section_bipolar"
OUTPUT_ROOT = ROOT / "vertical_continuation_near_closed_streamline_jan01"


def run_logged(command: list[str], stdout: Path, stderr: Path) -> None:
    stdout.parent.mkdir(parents=True, exist_ok=True)
    with stdout.open("w", encoding="utf-8") as out, stderr.open("w", encoding="utf-8") as err:
        result = subprocess.run(command, cwd=Path.cwd(), stdout=out, stderr=err, text=True, env={**os.environ, "HDF5_USE_FILE_LOCKING": "FALSE"})
    if result.returncode:
        raise RuntimeError(f"Command failed ({result.returncode}); see {stderr}")


def qc_count() -> int:
    frame = pd.read_csv(QC_TABLE)
    values = frame["qc_pass"].fillna(False).astype(str).str.lower().isin(("true", "1", "yes"))
    return int(values.sum())


def main() -> None:
    if not QC_TABLE.exists() or not VELOCITY_FILE.exists() or not BASELINE_ROOT.exists():
        raise FileNotFoundError("Required Jan1 QC surface table, velocity file, or tangent baseline is missing")
    count = qc_count()
    if count != 1163:
        raise ValueError(f"Expected 1163 qc_pass=True Jan1 surface objects, found {count}")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    logs = OUTPUT_ROOT / "logs"
    started = time.time()
    python = sys.executable
    run_logged(
        [python, "-m", "unittest", "Origin_eddy_detection.tests.test_near_closed_streamline"],
        logs / "synthetic_kernel_tests.stdout.log", logs / "synthetic_kernel_tests.stderr.log",
    )
    run_logged(
        [
            python, "-m", "Detection_for_OFES.tools.extend_final_surface_vertical",
            "--surface-table", str(QC_TABLE), "--filter-root", str(VELOCITY_ROOT), "--output-root", str(OUTPUT_ROOT),
            "--day", DAY, "--max-depth-layers", "105", "--deep-search-cells", "6",
            "--start-radius-cells", "2", "--max-radius-cells", "12", "--deep-hua-mode", "near_closed_streamline",
            "--deep-center-selection", "local_step_section_bipolar", "--deep-center-step-cells", "2",
            "--deep-center-speed-tolerance", "0.20", "--deep-center-min-axis-km", "1.0",
        ],
        logs / "vertical_extension.stdout.log", logs / "vertical_extension.stderr.log",
    )
    run_logged(
        [python, "-m", "Detection_for_OFES.tools.summarize_qc_vertical_extension", "--vertical-root", str(OUTPUT_ROOT), "--day", DAY],
        logs / "near_summary.stdout.log", logs / "near_summary.stderr.log",
    )
    # The tangent branch predates the summary tool.  Materialize its summary
    # only when absent so the section-bipolar catalog can use the same object
    # metadata as the new branch.
    if not (BASELINE_ROOT / "vertical_object_summary.csv").exists():
        run_logged(
            [python, "-m", "Detection_for_OFES.tools.summarize_qc_vertical_extension", "--vertical-root", str(BASELINE_ROOT), "--day", DAY],
            logs / "baseline_summary.stdout.log", logs / "baseline_summary.stderr.log",
        )
    diagnostics = OUTPUT_ROOT / "section_bipolarity"
    run_logged(
        [
            python, "-m", "Detection_for_OFES.tools.diagnose_deep_hua_section_bipolarity", "--velocity-file", str(VELOCITY_FILE),
            "--vertical-root", str(BASELINE_ROOT), "--vertical-root", str(OUTPUT_ROOT), "--output-root", str(diagnostics), "--day", DAY,
        ],
        logs / "section_bipolarity.stdout.log", logs / "section_bipolarity.stderr.log",
    )
    for vertical_root, definition in ((BASELINE_ROOT, "kinematic_continuation_tangent45_fraction35"), (OUTPUT_ROOT, "near_closed_streamline_non_circular")):
        run_logged(
            [
                python, "-m", "Detection_for_OFES.tools.build_section_bipolar_vertical_catalog", "--vertical-root", str(vertical_root),
                "--bipolarity-root", str(diagnostics / vertical_root.name), "--output-root", str(OUTPUT_ROOT / "section_bipolar_catalogs" / vertical_root.name),
                "--vertical-definition", definition,
            ],
            logs / f"section_catalog_{vertical_root.name}.stdout.log", logs / f"section_catalog_{vertical_root.name}.stderr.log",
        )
    run_logged(
        [python, "-m", "Detection_for_OFES.tools.compare_near_closed_streamline_vertical_jan1", "--baseline-root", str(BASELINE_ROOT), "--near-root", str(OUTPUT_ROOT), "--velocity-file", str(VELOCITY_FILE), "--output-root", str(OUTPUT_ROOT / "comparison"), "--day", DAY],
        logs / "comparison.stdout.log", logs / "comparison.stderr.log",
    )
    manifest = {
        "day": DAY,
        "surface_source": str(QC_TABLE),
        "surface_qc_pass_objects": count,
        "velocity_file": str(VELOCITY_FILE),
        "baseline_root": str(BASELINE_ROOT),
        "mode": "near_closed_streamline",
        "center_continuation": {"selection": "local_step_section_bipolar", "step_cells": 2, "search_cap_cells": 6, "subgrid_refinement": True},
        "streamline_acceptance": {"radii_cells": [2, 12], "start_angles": 4, "directions": 2, "min_points": 16, "min_winding_turns": 0.75, "closure_tolerance_cells": 1.75, "min_finite_fraction": 0.95, "select": "largest radius, then smallest closure error"},
        "not_evaluated": ["tangent_alignment", "velocity_ratio", "angle_jump", "direction_exception", "opposite_reversal", "symmetry"],
        "elapsed_seconds": time.time() - started,
    }
    (OUTPUT_ROOT / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "complete", "output_root": str(OUTPUT_ROOT), "elapsed_seconds": manifest["elapsed_seconds"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
