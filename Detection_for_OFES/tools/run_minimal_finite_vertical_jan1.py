"""Run the Jan1 deep-Hua finite-velocity-only comparison."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

from Detection_for_OFES.tools import run_minimal_reversal_vertical_jan1 as base


OUTPUT_ROOT = base.ROOT / "500km_highpass_no_tilecap_streamline_gate_removed_geometry_qc_19910101_minimal_finite_only"


def main() -> None:
    if not base.QC_TABLE.exists() or not (base.FILTER_ROOT / "global_phy_19910101.nc").exists():
        raise FileNotFoundError("Missing QC source table or existing full-depth 500 km velocity input")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    python = sys.executable
    command = [
        python, "-m", "Detection_for_OFES.tools.extend_final_surface_vertical",
        "--surface-table", str(base.QC_TABLE), "--filter-root", str(base.FILTER_ROOT),
        "--output-root", str(OUTPUT_ROOT), "--day", base.DAY, "--max-depth-layers", "105",
        "--deep-search-cells", "6", "--start-radius-cells", "2", "--max-radius-cells", "12",
        "--deep-hua-mode", "minimal_finite_only",
    ]
    elapsed = base.run_logged(command, OUTPUT_ROOT / "logs" / "vertical_extension.stdout.log", OUTPUT_ROOT / "logs" / "vertical_extension.stderr.log")
    summary = [python, "-m", "Detection_for_OFES.tools.summarize_qc_vertical_extension", "--vertical-root", str(OUTPUT_ROOT), "--day", base.DAY]
    base.run_logged(summary, OUTPUT_ROOT / "logs" / "summary_plots.stdout.log", OUTPUT_ROOT / "logs" / "summary_plots.stderr.log")
    rows = [
        base.branch_metrics("strict", base.STRICT_ROOT),
        base.branch_metrics("ratio_tangent_relaxed", base.RELAXED_ROOT),
        base.branch_metrics("direction_no_hard_gate", base.DIRECTION_ROOT),
        base.branch_metrics("minimal_reversal_only", base.OUTPUT_ROOT),
        base.branch_metrics("minimal_finite_only", OUTPUT_ROOT, elapsed),
    ]
    pd.DataFrame(rows).to_csv(OUTPUT_ROOT / "vertical_branch_comparison.csv", index=False)
    base.panel_helper.OUTPUT_ROOT = OUTPUT_ROOT
    base.panel_helper.STRICT_ROOT = base.STRICT_ROOT
    base.panel_helper.FILTER_ROOT = base.FILTER_ROOT
    base.panel_helper.PANEL_OUTPUT_STEM = "minimal_finite_only_curve_section_family"
    base.panel_helper.CURRENT_BRANCH_LABEL = "minimal_finite_only"
    base.panel_helper.make_family_panels(python)
    (OUTPUT_ROOT / "run_manifest.json").write_text(
        json.dumps(
            {
                "day": base.DAY,
                "surface_objects": 1131,
                "deep_hua_mode": "minimal_finite_only",
                "hard_conditions": ["finite_fraction >= 0.95"],
                "not_evaluated": ["velocity_ratio", "tangent_alignment", "angle_jump", "direction_exception", "symmetry", "opposite_reversal"],
                "velocity_input": str(base.FILTER_ROOT),
                "vertical_seconds": elapsed,
                "comparisons": rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"status": "complete", "output_root": str(OUTPUT_ROOT)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
