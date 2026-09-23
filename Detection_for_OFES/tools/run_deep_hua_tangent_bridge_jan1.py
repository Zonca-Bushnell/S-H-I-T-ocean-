"""Run one deliberately loose but nonzero tangent-quality bridge profile."""
from __future__ import annotations

import json

import pandas as pd

from Detection_for_OFES.tools import run_deep_hua_constraint_ladder_jan1 as base


OUTPUT_ROOT = base.ROOT / "hua_tangent_bridge_19910101"
PROFILE = (
    "13_tangent60_fraction020_no_reversal",
    (
        "--deep-hua-mode", "full", "--disable-angle-jump-hard-gate",
        "--disable-direction-exception-hard-gate", "--disable-opposite-reversal-hard-gate",
        "--deep-tangent-tolerance-deg", "60", "--deep-min-tangent-fraction", "0.20",
        "--enforce-tangent-alignment-hard-gate",
    ),
)


def main() -> None:
    if not base.QC_TABLE.exists() or not (base.FILTER_ROOT / "global_phy_19910101.nc").exists():
        raise FileNotFoundError("Missing fixed surface table or 105-layer 500 km velocity input")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    base.OUTPUT_ROOT = OUTPUT_ROOT
    row = base.run_profile(*PROFILE)
    pd.DataFrame([row]).to_csv(OUTPUT_ROOT / "tangent_bridge_summary.csv", index=False)
    (OUTPUT_ROOT / "tangent_bridge_manifest.json").write_text(
        json.dumps({"day": base.DAY, "surface_objects": 1131, "profile": row}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"status": "complete", "output_root": str(OUTPUT_ROOT)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
