"""Run the Jan1 QC-surface vertical diagnostic without touching production catalogs."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


DAY = "1991-01-01"
QC_TABLE = Path(
    r"E:\DATA\01_Eddy_correspond\02_OFES\Fromthebeginning\03_QC"
    r"\500km_highpass_no_tilecap_streamline_gate_removed_geometry_qc_19910101"
    r"\daily_runs\19910101\centers_hua_style.csv"
)
ORIGIN_ROOT = Path(r"E:\DATA\01_Eddy_correspond\02_OFES\origin_compatible_filter")
OUTPUT_ROOT = Path(
    r"E:\DATA\01_Eddy_correspond\02_OFES\Fromthebeginning\04_Vertical"
    r"\500km_highpass_no_tilecap_streamline_gate_removed_geometry_qc_19910101"
)


def run_logged(command: list[str], stdout_path: Path, stderr_path: Path) -> None:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        result = subprocess.run(command, stdout=stdout, stderr=stderr, text=True, check=False)
    if result.returncode:
        raise SystemExit(f"Command failed ({result.returncode}); see {stderr_path}")


def main() -> None:
    if not QC_TABLE.exists():
        raise FileNotFoundError(QC_TABLE)
    output = OUTPUT_ROOT
    velocity_root = output / "full_depth_velocity_highpass_500km"
    logs = output / "logs"
    output.mkdir(parents=True, exist_ok=True)
    run_manifest = {
        "day": DAY,
        "surface_table": str(QC_TABLE),
        "surface_selection": "qc_pass=True",
        "filter_input": str(ORIGIN_ROOT),
        "velocity_filter": "Gaussian field - LP_500km(field), 105 layers",
        "vertical_algorithm": "existing_hua_depth_continuation",
        "persistence": "not used",
    }
    (output / "input_manifest.json").write_text(json.dumps(run_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    python = sys.executable
    build = [
        python, "-m", "Detection_for_OFES.tools.build_ofes_meso_filter",
        "--input-root", str(ORIGIN_ROOT), "--output-root", str(velocity_root),
        "--start", DAY, "--end", DAY, "--available-start", DAY, "--available-end", DAY,
        "--temporal-window-days", "1", "--filter-mode", "highpass", "--large-cutoff-mode", "fixed",
        "--large-cutoff-km", "500", "--small-cutoff-km", "25", "--spatial-kernel", "gaussian",
        "--zonal-scale-mode", "km", "--meridional-scale-mode", "median", "--max-depth-layers", "105",
        "--science-tag", "qc_surface_vertical_gaussian_highpass_500km", "--overwrite",
    ]
    run_logged(build, logs / "build_full_depth_filter.stdout.log", logs / "build_full_depth_filter.stderr.log")
    extend = [
        python, "-m", "Detection_for_OFES.tools.extend_final_surface_vertical",
        "--surface-table", str(QC_TABLE), "--filter-root", str(velocity_root), "--output-root", str(output),
        "--day", DAY, "--max-depth-layers", "105", "--deep-search-cells", "6",
        "--start-radius-cells", "2", "--max-radius-cells", "12",
    ]
    run_logged(extend, logs / "vertical_extension.stdout.log", logs / "vertical_extension.stderr.log")
    summarize = [python, "-m", "Detection_for_OFES.tools.summarize_qc_vertical_extension", "--vertical-root", str(output), "--day", DAY]
    run_logged(summarize, logs / "summary_plots.stdout.log", logs / "summary_plots.stderr.log")
    print(json.dumps({"status": "complete", "output_root": str(output)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
