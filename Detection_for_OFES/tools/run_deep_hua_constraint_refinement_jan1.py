"""Refine the usable deep-Hua constraint ladder around the loose-reversal regime."""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from Detection_for_OFES.tools import run_deep_hua_constraint_ladder_jan1 as base


OUTPUT_ROOT = base.ROOT / "hua_constraint_refinement_19910101"
PROFILES: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "06_reversal040_tangent_lax",
        (
            "--deep-hua-mode", "full", "--deep-min-reversal-fraction", "0.40",
            "--disable-angle-jump-hard-gate", "--disable-direction-exception-hard-gate",
            "--deep-tangent-tolerance-deg", "45", "--deep-min-tangent-fraction", "0.35",
            "--enforce-tangent-alignment-hard-gate",
        ),
    ),
    (
        "07_tangent_ratio_lax",
        (
            "--deep-hua-mode", "full", "--deep-min-reversal-fraction", "0.40",
            "--disable-angle-jump-hard-gate", "--disable-direction-exception-hard-gate",
            "--deep-tangent-tolerance-deg", "45", "--deep-min-tangent-fraction", "0.35",
            "--enforce-tangent-alignment-hard-gate",
            "--deep-speed-ratio-max", "8.0", "--enforce-velocity-ratio-hard-gate",
        ),
    ),
    (
        "08_tangent_ratio_angle175",
        (
            "--deep-hua-mode", "full", "--deep-min-reversal-fraction", "0.40",
            "--disable-direction-exception-hard-gate",
            "--deep-tangent-tolerance-deg", "45", "--deep-min-tangent-fraction", "0.35",
            "--enforce-tangent-alignment-hard-gate",
            "--deep-speed-ratio-max", "8.0", "--enforce-velocity-ratio-hard-gate",
            "--deep-angle-jump-max-deg", "175",
        ),
    ),
    (
        "09_tangent_ratio_angle175_direction8",
        (
            "--deep-hua-mode", "full", "--deep-min-reversal-fraction", "0.40",
            "--deep-tangent-tolerance-deg", "45", "--deep-min-tangent-fraction", "0.35",
            "--enforce-tangent-alignment-hard-gate",
            "--deep-speed-ratio-max", "8.0", "--enforce-velocity-ratio-hard-gate",
            "--deep-angle-jump-max-deg", "175", "--direction-exception-multiplier", "8.0",
        ),
    ),
)


def main() -> None:
    if not base.QC_TABLE.exists() or not (base.FILTER_ROOT / "global_phy_19910101.nc").exists():
        raise FileNotFoundError("Missing fixed surface table or 105-layer 500 km velocity input")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    base.OUTPUT_ROOT = OUTPUT_ROOT
    with ThreadPoolExecutor(max_workers=2) as executor:
        rows = list(executor.map(lambda profile: base.run_profile(*profile), PROFILES))
    pd.DataFrame(rows).to_csv(OUTPUT_ROOT / "constraint_refinement_summary.csv", index=False)
    (OUTPUT_ROOT / "constraint_refinement_manifest.json").write_text(
        json.dumps({"day": base.DAY, "surface_objects": 1131, "profiles": rows}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"status": "complete", "output_root": str(OUTPUT_ROOT)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
