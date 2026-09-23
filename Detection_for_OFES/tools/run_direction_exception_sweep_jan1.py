"""Quantify the isolated deep-Hua direction-exception gate sensitivity."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


DAY = "1991-01-01"
QC_TABLE = Path(
    r"E:\DATA\01_Eddy_correspond\02_OFES\Fromthebeginning\03_QC"
    r"\500km_highpass_no_tilecap_streamline_gate_removed_geometry_qc_19910101"
    r"\daily_runs\19910101\centers_hua_style.csv"
)
FILTER_ROOT = Path(
    r"E:\DATA\01_Eddy_correspond\02_OFES\Fromthebeginning\04_Vertical"
    r"\500km_highpass_no_tilecap_streamline_gate_removed_geometry_qc_19910101"
    r"\full_depth_velocity_highpass_500km"
)
OUTPUT_ROOT = Path(
    r"E:\DATA\01_Eddy_correspond\02_OFES\Fromthebeginning\04_Vertical"
    r"\direction_exception_sweep_19910101"
)
CASES = (
    ("direction_current_1x", 1.0, False),
    ("direction_2x", 2.0, False),
    ("direction_4x", 4.0, False),
    ("direction_no_hard_gate", 1.0, True),
)


def run_logged(command: list[str], stdout: Path, stderr: Path) -> None:
    stdout.parent.mkdir(parents=True, exist_ok=True)
    with stdout.open("w", encoding="utf-8") as out, stderr.open("w", encoding="utf-8") as err:
        result = subprocess.run(command, stdout=out, stderr=err, text=True, check=False)
    if result.returncode:
        raise RuntimeError(f"Command failed ({result.returncode}); see {stderr}")


def main() -> None:
    if not QC_TABLE.exists() or not (FILTER_ROOT / "global_phy_19910101.nc").exists():
        raise FileNotFoundError("QC source table or existing 105-layer 500 km velocity input is missing")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    python = sys.executable
    rows: list[dict[str, object]] = []
    for name, multiplier, disabled in CASES:
        root = OUTPUT_ROOT / name
        vertical = [
            python, "-m", "Detection_for_OFES.tools.extend_final_surface_vertical",
            "--surface-table", str(QC_TABLE), "--filter-root", str(FILTER_ROOT), "--output-root", str(root),
            "--day", DAY, "--max-depth-layers", "105", "--deep-search-cells", "6",
            "--start-radius-cells", "2", "--max-radius-cells", "12", "--deep-hua-mode", "full",
            "--direction-exception-multiplier", str(multiplier),
        ]
        if disabled:
            vertical.append("--disable-direction-exception-hard-gate")
        run_logged(vertical, root / "logs" / "vertical_extension.stdout.log", root / "logs" / "vertical_extension.stderr.log")
        summary = [python, "-m", "Detection_for_OFES.tools.summarize_qc_vertical_extension", "--vertical-root", str(root), "--day", DAY]
        run_logged(summary, root / "logs" / "summary_plots.stdout.log", root / "logs" / "summary_plots.stderr.log")
        objects = pd.read_csv(root / "vertical_object_summary.csv")
        centers = pd.read_csv(root / "raw_detection" / "daily_runs" / "19910101" / "centers_hua_style.csv", low_memory=False)
        failures = centers.loc[centers["first_hard_failure"].astype(str).eq("too_many_direction_exceptions")]
        rows.append({
            "case": name, "direction_exception_multiplier": multiplier,
            "direction_exception_hard_gate": not disabled, "surface_objects": int(len(objects)),
            "extend_below_surface": int((objects["pass_layers"] > 1).sum()),
            "median_pass_layers": float(objects["pass_layers"].median()),
            "median_max_depth_m": float(objects["max_depth_m"].median()),
            "max_depth_m": float(objects["max_depth_m"].max()),
            "direction_first_hard_failures": int(len(failures)),
        })
    table = pd.DataFrame(rows)
    table.to_csv(OUTPUT_ROOT / "direction_exception_sweep_summary.csv", index=False)
    (OUTPUT_ROOT / "direction_exception_sweep_manifest.json").write_text(
        json.dumps({"day": DAY, "surface_objects": 1131, "velocity_input": str(FILTER_ROOT), "ratio_and_tangent": "soft diagnostics in every case", "angle_jump": "hard in every case", "opposite_reversal": "hard in every case", "cases": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"status": "complete", "output_root": str(OUTPUT_ROOT)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
