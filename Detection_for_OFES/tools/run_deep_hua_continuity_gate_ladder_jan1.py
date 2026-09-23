"""Reintroduce deep Hua gates from loose to strict on the continuous center path."""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from Detection_for_OFES.tools import run_deep_hua_constraint_ladder_jan1 as base
from Detection_for_OFES.tools.run_deep_hua_center_continuity_jan1 import trajectory_metrics


OUTPUT_ROOT = base.ROOT / "hua_center_continuity_gate_ladder_19910101"
BASELINE_ROOT = base.ROOT / "hua_center_continuity_19910101" / "local_step_2cells"
CONTINUOUS_CENTER: tuple[str, ...] = (
    "--deep-center-selection", "local_step_min", "--deep-center-step-cells", "2",
)
LOOSE_TANGENT: tuple[str, ...] = (
    "--deep-hua-mode", "full", "--deep-tangent-tolerance-deg", "45",
    "--deep-min-tangent-fraction", "0.35", "--enforce-tangent-alignment-hard-gate",
)

PROFILES: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "01_reversal040",
        (*CONTINUOUS_CENTER, *LOOSE_TANGENT, "--deep-min-reversal-fraction", "0.40",
         "--disable-angle-jump-hard-gate", "--disable-direction-exception-hard-gate"),
    ),
    (
        "02_rotation_lax",
        (*CONTINUOUS_CENTER, *LOOSE_TANGENT, "--deep-min-reversal-fraction", "0.40",
         "--deep-angle-jump-max-deg", "175", "--direction-exception-multiplier", "4.0"),
    ),
    (
        "03_all_lax",
        (*CONTINUOUS_CENTER, "--deep-hua-mode", "full", "--deep-min-reversal-fraction", "0.45",
         "--deep-angle-jump-max-deg", "175", "--direction-exception-multiplier", "4.0",
         "--deep-speed-ratio-max", "8.0", "--enforce-velocity-ratio-hard-gate",
         "--deep-tangent-tolerance-deg", "40", "--deep-min-tangent-fraction", "0.45",
         "--enforce-tangent-alignment-hard-gate"),
    ),
    (
        "04_balanced",
        (*CONTINUOUS_CENTER, "--deep-hua-mode", "full", "--deep-min-reversal-fraction", "0.50",
         "--deep-angle-jump-max-deg", "170", "--direction-exception-multiplier", "2.0",
         "--deep-speed-ratio-max", "6.0", "--enforce-velocity-ratio-hard-gate",
         "--deep-tangent-tolerance-deg", "36", "--deep-min-tangent-fraction", "0.50",
         "--enforce-tangent-alignment-hard-gate"),
    ),
    (
        "05_strict_reference",
        (*CONTINUOUS_CENTER, "--deep-hua-mode", "full", "--deep-min-reversal-fraction", "0.70",
         "--deep-angle-jump-max-deg", "150", "--direction-exception-multiplier", "1.0",
         "--deep-speed-ratio-max", "3.0", "--enforce-velocity-ratio-hard-gate",
         "--deep-tangent-tolerance-deg", "24", "--deep-min-tangent-fraction", "0.70",
         "--enforce-tangent-alignment-hard-gate"),
    ),
)


def baseline_row() -> dict[str, object]:
    summary = pd.read_csv(BASELINE_ROOT / "vertical_object_summary.csv")
    return {
        "profile": "00_loose_continuous_center",
        "surface_objects": int(len(summary)),
        "extend_below_surface": int(summary["pass_layers"].gt(1).sum()),
        "median_pass_layers": float(summary["pass_layers"].median()),
        "median_max_depth_m": float(summary["max_depth_m"].median()),
        "p90_max_depth_m": float(summary["max_depth_m"].quantile(0.90)),
        "max_depth_m": float(summary["max_depth_m"].max()),
        "first_hard_failures": "{}",
        "run_seconds": 0.0,
        **trajectory_metrics(BASELINE_ROOT),
    }


def main() -> None:
    if not BASELINE_ROOT.exists() or not base.QC_TABLE.exists() or not (base.FILTER_ROOT / "global_phy_19910101.nc").exists():
        raise FileNotFoundError("Missing continuous-center baseline, fixed surface table, or cached velocity field")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    base.OUTPUT_ROOT = OUTPUT_ROOT
    with ThreadPoolExecutor(max_workers=2) as executor:
        rows = list(executor.map(lambda profile: base.run_profile(*profile), PROFILES))
    rows = [{**row, **trajectory_metrics(OUTPUT_ROOT / str(row["profile"]))} for row in rows]
    result = [baseline_row(), *rows]
    pd.DataFrame(result).to_csv(OUTPUT_ROOT / "continuity_gate_ladder_summary.csv", index=False)
    (OUTPUT_ROOT / "continuity_gate_ladder_manifest.json").write_text(
        json.dumps(
            {
                "day": base.DAY,
                "surface_objects": 1131,
                "center_selection": "local_step_min, two cells per depth interval",
                "profile_sequence": "loose continuous baseline, reversal, rotation, all lax, balanced, strict reference",
                "profiles": result,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"status": "complete", "output_root": str(OUTPUT_ROOT)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
