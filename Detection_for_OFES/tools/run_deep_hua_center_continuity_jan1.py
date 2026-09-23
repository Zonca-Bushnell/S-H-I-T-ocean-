"""Compare continuous deep-center paths against the established tangent profile.

This is deliberately a center-selection experiment.  The fixed 1,131 surface
objects, 105-layer velocity input, and every Hua acceptance gate are identical
to profile 15 (45 degrees / 0.35 tangential fraction).  Only the way a speed
minimum is chosen around the previous accepted center changes.
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from Detection_for_OFES.tools import run_deep_hua_constraint_ladder_jan1 as base


OUTPUT_ROOT = base.ROOT / "hua_center_continuity_19910101"
BASELINE_ROOT = base.ROOT / "hua_tangent_calibration_19910101" / "15_tangent45_fraction035"

COMMON: tuple[str, ...] = (
    "--deep-hua-mode", "full",
    "--disable-angle-jump-hard-gate",
    "--disable-direction-exception-hard-gate",
    "--disable-opposite-reversal-hard-gate",
    "--deep-tangent-tolerance-deg", "45",
    "--deep-min-tangent-fraction", "0.35",
    "--enforce-tangent-alignment-hard-gate",
    "--deep-center-selection", "local_step_min",
)

PROFILES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("local_step_2cells", (*COMMON, "--deep-center-step-cells", "2")),
    ("local_step_3cells", (*COMMON, "--deep-center-step-cells", "3")),
)


def trajectory_metrics(root: Path | str) -> dict[str, object]:
    run = Path(root) / "raw_detection" / "daily_runs" / "19910101"
    parquet = run / "structures_hua_style.parquet"
    structures = pd.read_parquet(parquet) if parquet.exists() else pd.read_csv(run / "structures_hua_style.csv")
    pieces: list[pd.DataFrame] = []
    for _, group in structures.groupby("hua_object_id", sort=False):
        part = group.sort_values("depth_index").copy()
        lon = np.unwrap(np.deg2rad(part["center_lon"].to_numpy(dtype="f8")))
        lat = part["center_lat"].to_numpy(dtype="f8")
        dx = np.diff(lon) * 6371.0 * np.cos(np.deg2rad(lat[:-1]))
        dy = np.diff(np.deg2rad(lat)) * 6371.0
        part_steps = np.hypot(dx, dy)
        if part_steps.size:
            pieces.append(pd.DataFrame({"hua_object_id": part["hua_object_id"].iloc[1:].to_numpy(), "step_km": part_steps}))
    steps = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame(columns=["hua_object_id", "step_km"])
    return {
        "layer_to_layer_step_km_median": float(steps["step_km"].median()) if len(steps) else np.nan,
        "layer_to_layer_step_km_p90": float(steps["step_km"].quantile(0.9)) if len(steps) else np.nan,
        "layer_to_layer_step_km_p99": float(steps["step_km"].quantile(0.99)) if len(steps) else np.nan,
        "layer_to_layer_step_over_30km": int((steps["step_km"] > 30.0).sum()),
        "layer_to_layer_step_over_50km": int((steps["step_km"] > 50.0).sum()),
    }


def main() -> None:
    if not base.QC_TABLE.exists() or not (base.FILTER_ROOT / "global_phy_19910101.nc").exists():
        raise FileNotFoundError("Missing fixed 1,131-object surface table or cached 105-layer velocity input")
    if not BASELINE_ROOT.exists():
        raise FileNotFoundError(f"Missing established profile-15 baseline: {BASELINE_ROOT}")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    base.OUTPUT_ROOT = OUTPUT_ROOT
    with ThreadPoolExecutor(max_workers=2) as executor:
        rows = list(executor.map(lambda profile: base.run_profile(*profile), PROFILES))

    baseline = {"profile": "global_disk_min_6cells_profile15", **trajectory_metrics(BASELINE_ROOT)}
    enriched: list[dict[str, object]] = []
    for row in rows:
        root = OUTPUT_ROOT / str(row["profile"])
        enriched.append({**row, **trajectory_metrics(root)})

    pd.DataFrame([baseline, *enriched]).to_csv(OUTPUT_ROOT / "center_continuity_summary.csv", index=False)
    (OUTPUT_ROOT / "center_continuity_manifest.json").write_text(
        json.dumps(
            {
                "day": base.DAY,
                "surface_objects": 1131,
                "fixed_acceptance_profile": "profile15 tangent=45deg fraction=0.35; ratio/angle/direction/reversal gates disabled",
                "baseline": str(BASELINE_ROOT),
                "selection_change": "global six-cell disk minimum versus local two- or three-cell minimum from previous accepted center",
                "profiles": enriched,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"status": "complete", "output_root": str(OUTPUT_ROOT)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
