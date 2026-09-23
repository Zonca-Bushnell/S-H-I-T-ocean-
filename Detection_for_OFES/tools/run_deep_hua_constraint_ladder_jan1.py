"""Run a staged deep-Hua constraint calibration on one fixed surface catalog."""
from __future__ import annotations

import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd


DAY = "1991-01-01"
ROOT = Path(r"E:\DATA\01_Eddy_correspond\02_OFES\Fromthebeginning\04_Vertical")
OUTPUT_ROOT = ROOT / "hua_constraint_ladder_19910101"
QC_TABLE = Path(
    r"E:\DATA\01_Eddy_correspond\02_OFES\Fromthebeginning\03_QC"
    r"\500km_highpass_no_tilecap_streamline_gate_removed_geometry_qc_19910101"
    r"\daily_runs\19910101\centers_hua_style.csv"
)
FILTER_ROOT = ROOT / "500km_highpass_no_tilecap_streamline_gate_removed_geometry_qc_19910101" / "full_depth_velocity_highpass_500km"


PROFILES: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "01_reversal_040",
        ("--deep-hua-mode", "minimal_reversal_only", "--deep-min-reversal-fraction", "0.40"),
    ),
    (
        "02_rotation_lax",
        (
            "--deep-hua-mode", "full", "--deep-min-reversal-fraction", "0.45",
            "--deep-angle-jump-max-deg", "175", "--direction-exception-multiplier", "4.0",
        ),
    ),
    (
        "03_section_lax",
        (
            "--deep-hua-mode", "full", "--deep-min-reversal-fraction", "0.50",
            "--deep-angle-jump-max-deg", "170", "--direction-exception-multiplier", "2.0",
            "--deep-tangent-tolerance-deg", "36", "--deep-min-tangent-fraction", "0.45",
            "--enforce-tangent-alignment-hard-gate",
        ),
    ),
    (
        "04_balanced",
        (
            "--deep-hua-mode", "full", "--deep-min-reversal-fraction", "0.60",
            "--deep-angle-jump-max-deg", "165", "--direction-exception-multiplier", "1.5",
            "--deep-speed-ratio-max", "5.0", "--enforce-velocity-ratio-hard-gate",
            "--deep-tangent-tolerance-deg", "30", "--deep-min-tangent-fraction", "0.55",
            "--enforce-tangent-alignment-hard-gate",
        ),
    ),
    (
        "05_strict_explicit",
        (
            "--deep-hua-mode", "full", "--deep-min-reversal-fraction", "0.70",
            "--deep-angle-jump-max-deg", "150", "--direction-exception-multiplier", "1.0",
            "--deep-speed-ratio-max", "3.0", "--enforce-velocity-ratio-hard-gate",
            "--deep-tangent-tolerance-deg", "24", "--deep-min-tangent-fraction", "0.70",
            "--enforce-tangent-alignment-hard-gate",
        ),
    ),
)


def run_profile(name: str, options: tuple[str, ...]) -> dict[str, object]:
    out = OUTPUT_ROOT / name
    logs = out / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, "-m", "Detection_for_OFES.tools.extend_final_surface_vertical",
        "--surface-table", str(QC_TABLE), "--filter-root", str(FILTER_ROOT), "--output-root", str(out),
        "--day", DAY, "--max-depth-layers", "105", "--deep-search-cells", "6",
        "--start-radius-cells", "2", "--max-radius-cells", "12", *options,
    ]
    started = time.perf_counter()
    with (logs / "vertical_extension.stdout.log").open("w", encoding="utf-8") as stdout, (logs / "vertical_extension.stderr.log").open("w", encoding="utf-8") as stderr:
        subprocess.run(command, stdout=stdout, stderr=stderr, check=True, text=True)
    with (logs / "summary_plots.stdout.log").open("w", encoding="utf-8") as stdout, (logs / "summary_plots.stderr.log").open("w", encoding="utf-8") as stderr:
        subprocess.run([sys.executable, "-m", "Detection_for_OFES.tools.summarize_qc_vertical_extension", "--vertical-root", str(out), "--day", DAY], stdout=stdout, stderr=stderr, check=True, text=True)
    objects = pd.read_csv(out / "vertical_object_summary.csv")
    centers = pd.read_csv(out / "raw_detection" / "daily_runs" / "19910101" / "centers_hua_style.csv", low_memory=False)
    failures = centers.loc[centers["depth_index"].gt(0) & centers["first_hard_failure"].astype(str).ne("none"), "first_hard_failure"].value_counts().to_dict()
    return {
        "profile": name,
        "options": " ".join(options),
        "surface_objects": int(len(objects)),
        "extend_below_surface": int(objects["pass_layers"].gt(1).sum()),
        "median_pass_layers": float(objects["pass_layers"].median()),
        "median_max_depth_m": float(objects["max_depth_m"].median()),
        "p90_max_depth_m": float(objects["max_depth_m"].quantile(0.90)),
        "max_depth_m": float(objects["max_depth_m"].max()),
        "first_hard_failures": json.dumps(failures, ensure_ascii=False, sort_keys=True),
        "run_seconds": time.perf_counter() - started,
    }


def main() -> None:
    if not QC_TABLE.exists() or not (FILTER_ROOT / "global_phy_19910101.nc").exists():
        raise FileNotFoundError("Missing fixed surface table or 105-layer 500 km velocity input")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    # Each profile reads the same immutable cached velocity file but writes to an
    # isolated directory. Two runs use the available memory without oversubscribing I/O.
    with ThreadPoolExecutor(max_workers=2) as executor:
        rows = list(executor.map(lambda profile: run_profile(*profile), PROFILES))
    pd.DataFrame(rows).to_csv(OUTPUT_ROOT / "constraint_ladder_summary.csv", index=False)
    (OUTPUT_ROOT / "constraint_ladder_manifest.json").write_text(
        json.dumps({"day": DAY, "surface_objects": 1131, "profiles": rows}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"status": "complete", "output_root": str(OUTPUT_ROOT)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
