"""Run the default deep Hua minimal-reversal-only Jan1 diagnostic."""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

from Detection_for_OFES.tools import run_relaxed_hua_vertical_jan1 as panel_helper


DAY = "1991-01-01"
ROOT = Path(r"E:\DATA\01_Eddy_correspond\02_OFES\Fromthebeginning\04_Vertical")
STRICT_ROOT = ROOT / "500km_highpass_no_tilecap_streamline_gate_removed_geometry_qc_19910101"
RELAXED_ROOT = ROOT / "500km_highpass_no_tilecap_streamline_gate_removed_geometry_qc_19910101_relaxed_ratio_tangent"
DIRECTION_ROOT = ROOT / "direction_exception_sweep_19910101" / "direction_no_hard_gate"
OUTPUT_ROOT = ROOT / "500km_highpass_no_tilecap_streamline_gate_removed_geometry_qc_19910101_minimal_reversal_only"
QC_TABLE = Path(
    r"E:\DATA\01_Eddy_correspond\02_OFES\Fromthebeginning\03_QC"
    r"\500km_highpass_no_tilecap_streamline_gate_removed_geometry_qc_19910101"
    r"\daily_runs\19910101\centers_hua_style.csv"
)
FILTER_ROOT = STRICT_ROOT / "full_depth_velocity_highpass_500km"


def run_logged(command: list[str], stdout: Path, stderr: Path) -> float:
    stdout.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with stdout.open("w", encoding="utf-8") as out, stderr.open("w", encoding="utf-8") as err:
        result = subprocess.run(command, stdout=out, stderr=err, text=True, check=False)
    elapsed = time.perf_counter() - started
    if result.returncode:
        raise RuntimeError(f"Command failed ({result.returncode}); see {stderr}")
    return elapsed


def branch_metrics(name: str, root: Path, elapsed_seconds: float | None = None) -> dict[str, object]:
    objects = pd.read_csv(root / "vertical_object_summary.csv")
    centers = pd.read_csv(root / "raw_detection" / "daily_runs" / "19910101" / "centers_hua_style.csv", low_memory=False)
    failed = centers.loc[centers["first_hard_failure"].astype(str).ne("none") & centers["depth_index"].gt(0), "first_hard_failure"].value_counts().to_dict()
    return {
        "branch": name, "objects": int(len(objects)), "extend_below_surface": int((objects["pass_layers"] > 1).sum()),
        "median_pass_layers": float(objects["pass_layers"].median()), "median_max_depth_m": float(objects["max_depth_m"].median()),
        "max_depth_m": float(objects["max_depth_m"].max()), "first_hard_failures": json.dumps(failed, ensure_ascii=False, sort_keys=True),
        "run_seconds": elapsed_seconds,
    }


def main() -> None:
    if not QC_TABLE.exists() or not (FILTER_ROOT / "global_phy_19910101.nc").exists():
        raise FileNotFoundError("Missing QC source table or existing full-depth 500 km velocity input")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    python = sys.executable
    vertical = [
        python, "-m", "Detection_for_OFES.tools.extend_final_surface_vertical",
        "--surface-table", str(QC_TABLE), "--filter-root", str(FILTER_ROOT), "--output-root", str(OUTPUT_ROOT),
        "--day", DAY, "--max-depth-layers", "105", "--deep-search-cells", "6",
        "--start-radius-cells", "2", "--max-radius-cells", "12", "--deep-hua-mode", "minimal_reversal_only",
    ]
    elapsed = run_logged(vertical, OUTPUT_ROOT / "logs" / "vertical_extension.stdout.log", OUTPUT_ROOT / "logs" / "vertical_extension.stderr.log")
    summary = [python, "-m", "Detection_for_OFES.tools.summarize_qc_vertical_extension", "--vertical-root", str(OUTPUT_ROOT), "--day", DAY]
    run_logged(summary, OUTPUT_ROOT / "logs" / "summary_plots.stdout.log", OUTPUT_ROOT / "logs" / "summary_plots.stderr.log")
    rows = [branch_metrics("strict", STRICT_ROOT), branch_metrics("ratio_tangent_relaxed", RELAXED_ROOT), branch_metrics("direction_no_hard_gate", DIRECTION_ROOT), branch_metrics("minimal_reversal_only", OUTPUT_ROOT, elapsed)]
    pd.DataFrame(rows).to_csv(OUTPUT_ROOT / "vertical_branch_comparison.csv", index=False)
    panel_helper.OUTPUT_ROOT = OUTPUT_ROOT
    panel_helper.STRICT_ROOT = STRICT_ROOT
    panel_helper.FILTER_ROOT = FILTER_ROOT
    panel_helper.PANEL_OUTPUT_STEM = "minimal_reversal_only_curve_section_family"
    panel_helper.CURRENT_BRANCH_LABEL = "minimal_reversal_only"
    panel_helper.make_family_panels(python)
    (OUTPUT_ROOT / "run_manifest.json").write_text(
        json.dumps({"day": DAY, "surface_objects": 1131, "deep_hua_mode": "minimal_reversal_only", "velocity_input": str(FILTER_ROOT), "vertical_seconds": elapsed, "comparisons": rows}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"status": "complete", "output_root": str(OUTPUT_ROOT)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
