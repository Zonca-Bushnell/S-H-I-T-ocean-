"""Test a section-bipolar tie-break between locally equivalent deep minima.

The surface catalog, velocity field, radius scan, and wide calibrated Hua
acceptance are fixed.  The sole change is how an already local (two-cell)
speed-minimum ambiguity is resolved.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from Detection_for_OFES.tools import run_deep_hua_constraint_ladder_jan1 as base
from Detection_for_OFES.tools.run_deep_hua_center_continuity_jan1 import trajectory_metrics


OUTPUT_ROOT = base.ROOT / "hua_center_section_selection_19910101"

OPTIONS: tuple[str, ...] = (
    "--deep-hua-mode", "full",
    "--disable-angle-jump-hard-gate",
    "--disable-direction-exception-hard-gate",
    "--disable-opposite-reversal-hard-gate",
    "--deep-tangent-tolerance-deg", "45",
    "--deep-min-tangent-fraction", "0.35",
    "--enforce-tangent-alignment-hard-gate",
    "--deep-center-selection", "local_step_section_bipolar",
    "--deep-center-step-cells", "2",
    "--deep-center-speed-tolerance", "0.20",
    "--deep-center-min-axis-km", "1.0",
)


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    base.OUTPUT_ROOT = OUTPUT_ROOT
    row = base.run_profile("local_step_2cells_section_bipolar_tiebreak", OPTIONS)
    root = OUTPUT_ROOT / str(row["profile"])
    summary = {**row, **trajectory_metrics(root)}
    pd.DataFrame([summary]).to_csv(OUTPUT_ROOT / "section_center_selection_summary.csv", index=False)
    (OUTPUT_ROOT / "section_center_selection_manifest.json").write_text(
        json.dumps(
            {
                "day": base.DAY,
                "surface_objects": 1131,
                "fixed_acceptance_profile": "tangent=45deg fraction=0.35; ratio/angle/direction/reversal hard gates disabled",
                "selection": "within a two-cell local speed search, score only distinct local minima within 20 percent of the local minimum by opposite signs on 0.6R and 1.0R trajectory-normal sections",
                "result": summary,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"status": "complete", "output_root": str(OUTPUT_ROOT)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
