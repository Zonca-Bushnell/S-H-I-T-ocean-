"""Test low, explicit opposite-side reversal gates on the viable tangent range."""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from Detection_for_OFES.tools import run_deep_hua_constraint_ladder_jan1 as base


OUTPUT_ROOT = base.ROOT / "hua_tangent_reversal_bridge_19910101"
PROFILES: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "17_tangent45_fraction035_reversal020",
        (
            "--deep-hua-mode", "full", "--disable-angle-jump-hard-gate",
            "--disable-direction-exception-hard-gate", "--deep-min-reversal-fraction", "0.20",
            "--deep-tangent-tolerance-deg", "45", "--deep-min-tangent-fraction", "0.35",
            "--enforce-tangent-alignment-hard-gate",
        ),
    ),
    (
        "18_tangent50_fraction030_reversal020",
        (
            "--deep-hua-mode", "full", "--disable-angle-jump-hard-gate",
            "--disable-direction-exception-hard-gate", "--deep-min-reversal-fraction", "0.20",
            "--deep-tangent-tolerance-deg", "50", "--deep-min-tangent-fraction", "0.30",
            "--enforce-tangent-alignment-hard-gate",
        ),
    ),
    (
        "19_tangent45_fraction035_reversal030",
        (
            "--deep-hua-mode", "full", "--disable-angle-jump-hard-gate",
            "--disable-direction-exception-hard-gate", "--deep-min-reversal-fraction", "0.30",
            "--deep-tangent-tolerance-deg", "45", "--deep-min-tangent-fraction", "0.35",
            "--enforce-tangent-alignment-hard-gate",
        ),
    ),
)


def main() -> None:
    if not base.QC_TABLE.exists() or not (base.FILTER_ROOT / "global_phy_19910101.nc").exists():
        raise FileNotFoundError("Missing fixed 1,131-object surface table or cached 105-layer velocity file")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    base.OUTPUT_ROOT = OUTPUT_ROOT
    with ThreadPoolExecutor(max_workers=3) as executor:
        rows = list(executor.map(lambda profile: base.run_profile(*profile), PROFILES))
    pd.DataFrame(rows).to_csv(OUTPUT_ROOT / "tangent_reversal_bridge_summary.csv", index=False)
    (OUTPUT_ROOT / "tangent_reversal_bridge_manifest.json").write_text(
        json.dumps(
            {
                "day": base.DAY,
                "surface_objects": 1131,
                "constant_hard_conditions": "finite velocity, tangent alignment, and listed opposite-side reversal fraction",
                "disabled_hard_conditions": ["velocity_ratio", "angle_jump", "direction_exception"],
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
