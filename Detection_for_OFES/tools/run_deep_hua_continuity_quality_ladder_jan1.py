"""Calibrate tangent and velocity-ratio gates on a fixed continuous center path."""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from Detection_for_OFES.tools import run_deep_hua_constraint_ladder_jan1 as base
from Detection_for_OFES.tools.run_deep_hua_center_continuity_jan1 import trajectory_metrics


OUTPUT_ROOT = base.ROOT / "hua_center_continuity_quality_ladder_19910101"
CONTINUOUS: tuple[str, ...] = (
    "--deep-center-selection", "local_step_min", "--deep-center-step-cells", "2",
    "--deep-hua-mode", "full", "--disable-angle-jump-hard-gate",
    "--disable-direction-exception-hard-gate", "--disable-opposite-reversal-hard-gate",
)

PROFILES: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "01_tangent40_fraction045",
        (*CONTINUOUS, "--deep-tangent-tolerance-deg", "40", "--deep-min-tangent-fraction", "0.45", "--enforce-tangent-alignment-hard-gate"),
    ),
    (
        "02_tangent35_fraction050",
        (*CONTINUOUS, "--deep-tangent-tolerance-deg", "35", "--deep-min-tangent-fraction", "0.50", "--enforce-tangent-alignment-hard-gate"),
    ),
    (
        "03_ratio8_tangent45_fraction035",
        (*CONTINUOUS, "--deep-speed-ratio-max", "8", "--enforce-velocity-ratio-hard-gate", "--deep-tangent-tolerance-deg", "45", "--deep-min-tangent-fraction", "0.35", "--enforce-tangent-alignment-hard-gate"),
    ),
    (
        "04_ratio8_tangent40_fraction045",
        (*CONTINUOUS, "--deep-speed-ratio-max", "8", "--enforce-velocity-ratio-hard-gate", "--deep-tangent-tolerance-deg", "40", "--deep-min-tangent-fraction", "0.45", "--enforce-tangent-alignment-hard-gate"),
    ),
)


def main() -> None:
    if not base.QC_TABLE.exists() or not (base.FILTER_ROOT / "global_phy_19910101.nc").exists():
        raise FileNotFoundError("Missing fixed surface table or cached velocity file")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    base.OUTPUT_ROOT = OUTPUT_ROOT
    with ThreadPoolExecutor(max_workers=2) as executor:
        rows = list(executor.map(lambda profile: base.run_profile(*profile), PROFILES))
    rows = [{**row, **trajectory_metrics(OUTPUT_ROOT / str(row["profile"]))} for row in rows]
    pd.DataFrame(rows).to_csv(OUTPUT_ROOT / "continuity_quality_ladder_summary.csv", index=False)
    (OUTPUT_ROOT / "continuity_quality_ladder_manifest.json").write_text(
        json.dumps(
            {
                "day": base.DAY,
                "surface_objects": 1131,
                "fixed_center_path": "local_step_min, two cells per layer",
                "disabled_hard_gates": ["angle_jump", "direction_exception", "opposite_reversal"],
                "profiles": rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"status": "complete", "output_root": str(OUTPUT_ROOT)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
