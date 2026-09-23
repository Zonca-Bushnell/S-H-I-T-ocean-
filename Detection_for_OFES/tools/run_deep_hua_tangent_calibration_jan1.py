"""Calibrate the only remaining hard deep-Hua quality gate: tangential flow."""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from Detection_for_OFES.tools import run_deep_hua_constraint_ladder_jan1 as base


OUTPUT_ROOT = base.ROOT / "hua_tangent_calibration_19910101"
PROFILES: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "14_tangent40_fraction045",
        (
            "--deep-hua-mode", "full", "--disable-angle-jump-hard-gate",
            "--disable-direction-exception-hard-gate", "--disable-opposite-reversal-hard-gate",
            "--deep-tangent-tolerance-deg", "40", "--deep-min-tangent-fraction", "0.45",
            "--enforce-tangent-alignment-hard-gate",
        ),
    ),
    (
        "15_tangent45_fraction035",
        (
            "--deep-hua-mode", "full", "--disable-angle-jump-hard-gate",
            "--disable-direction-exception-hard-gate", "--disable-opposite-reversal-hard-gate",
            "--deep-tangent-tolerance-deg", "45", "--deep-min-tangent-fraction", "0.35",
            "--enforce-tangent-alignment-hard-gate",
        ),
    ),
    (
        "16_tangent50_fraction030",
        (
            "--deep-hua-mode", "full", "--disable-angle-jump-hard-gate",
            "--disable-direction-exception-hard-gate", "--disable-opposite-reversal-hard-gate",
            "--deep-tangent-tolerance-deg", "50", "--deep-min-tangent-fraction", "0.30",
            "--enforce-tangent-alignment-hard-gate",
        ),
    ),
)


def main() -> None:
    if not base.QC_TABLE.exists() or not (base.FILTER_ROOT / "global_phy_19910101.nc").exists():
        raise FileNotFoundError("Missing fixed 1,131-object surface table or the cached 105-layer velocity file")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    base.OUTPUT_ROOT = OUTPUT_ROOT
    with ThreadPoolExecutor(max_workers=3) as executor:
        rows = list(executor.map(lambda profile: base.run_profile(*profile), PROFILES))
    pd.DataFrame(rows).to_csv(OUTPUT_ROOT / "tangent_calibration_summary.csv", index=False)
    (OUTPUT_ROOT / "tangent_calibration_manifest.json").write_text(
        json.dumps(
            {
                "day": base.DAY,
                "surface_objects": 1131,
                "constant_hard_conditions": "finite velocity only, plus the profile tangential-flow gate",
                "disabled_hard_conditions": ["velocity_ratio", "angle_jump", "direction_exception", "opposite_reversal"],
                "profiles": rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"status": "complete", "output_root": str(OUTPUT_ROOT)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
