"""Test tangent-led deep-Hua profiles with reversal retained as a diagnostic."""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from Detection_for_OFES.tools import run_deep_hua_constraint_ladder_jan1 as base


OUTPUT_ROOT = base.ROOT / "hua_section_quality_19910101"
PROFILES: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "10_tangent_strong_no_reversal",
        (
            "--deep-hua-mode", "full", "--disable-angle-jump-hard-gate",
            "--disable-direction-exception-hard-gate", "--disable-opposite-reversal-hard-gate",
            "--deep-tangent-tolerance-deg", "36", "--deep-min-tangent-fraction", "0.55",
            "--enforce-tangent-alignment-hard-gate",
        ),
    ),
    (
        "11_tangent_ratio8_no_reversal",
        (
            "--deep-hua-mode", "full", "--disable-angle-jump-hard-gate",
            "--disable-direction-exception-hard-gate", "--disable-opposite-reversal-hard-gate",
            "--deep-tangent-tolerance-deg", "36", "--deep-min-tangent-fraction", "0.55",
            "--enforce-tangent-alignment-hard-gate",
            "--deep-speed-ratio-max", "8.0", "--enforce-velocity-ratio-hard-gate",
        ),
    ),
    (
        "12_tangent_ratio8_reversal020",
        (
            "--deep-hua-mode", "full", "--disable-angle-jump-hard-gate",
            "--disable-direction-exception-hard-gate",
            "--deep-min-reversal-fraction", "0.20",
            "--deep-tangent-tolerance-deg", "36", "--deep-min-tangent-fraction", "0.55",
            "--enforce-tangent-alignment-hard-gate",
            "--deep-speed-ratio-max", "8.0", "--enforce-velocity-ratio-hard-gate",
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
    pd.DataFrame(rows).to_csv(OUTPUT_ROOT / "section_quality_summary.csv", index=False)
    (OUTPUT_ROOT / "section_quality_manifest.json").write_text(
        json.dumps({"day": base.DAY, "surface_objects": 1131, "profiles": rows}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"status": "complete", "output_root": str(OUTPUT_ROOT)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
