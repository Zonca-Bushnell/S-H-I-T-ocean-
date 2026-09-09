"""Run the 20N EP-flux material-volume validation backend.

This wrapper keeps the command line stable while the numerical work stays in
MATLAB, where the existing Argo/META products are already stored.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


DEFAULT_THERMAL_ROOT = (
    r"E:\DATA\01_Eddy_correspond\01_Vertical_asymmetric"
    r"\META4_CoreArgo_W_3D_20N_repeat_all_recommended_thermalwind"
)
DEFAULT_REFERENCE_ROOT = (
    r"E:\DATA\01_Eddy_correspond\01_Vertical_asymmetric"
    r"\META4_CoreArgo_W_3D_20N_reference_like_reversal_worktree"
)
DEFAULT_OUTPUT_ROOT = (
    r"E:\DATA\01_Eddy_correspond\01_Vertical_asymmetric"
    r"\EP_FLUX_20N_validation"
)


def matlab_string(value: str | Path) -> str:
    text = str(value).replace("'", "''")
    return f"'{text}'"


def build_matlab_batch(args: argparse.Namespace) -> str:
    repo_root = Path(__file__).resolve().parents[1]
    matlab_dir = repo_root / "EP-FLUX" / "matlab"
    parts = [
        f"addpath(genpath({matlab_string(matlab_dir)}))",
        "run_epflux_20n_validation_backend("
        f"'ThermalRoot',{matlab_string(args.thermal_root)},"
        f"'ReferenceRoot',{matlab_string(args.reference_root)},"
        f"'OutputRoot',{matlab_string(args.output_root)},"
        f"'DepthLevels',{matlab_string(args.depth_levels)},"
        f"'MaxDepthLevels',{int(args.max_depth_levels)},"
        f"'BootstrapN',{int(args.bootstrap_n)},"
        f"'Alpha',{float(args.alpha)},"
        f"'RhoRef',{float(args.rho_ref)})",
    ]
    return "; ".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate the 20N material-volume EP-flux diagnostics from existing 3D W grids."
    )
    parser.add_argument("--thermal-root", default=DEFAULT_THERMAL_ROOT)
    parser.add_argument("--reference-root", default=DEFAULT_REFERENCE_ROOT)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--depth-levels", default="10:10:2000")
    parser.add_argument(
        "--max-depth-levels",
        type=int,
        default=0,
        help="Optional cap after parsing depth levels; useful for smoke tests.",
    )
    parser.add_argument("--bootstrap-n", type=int, default=300)
    parser.add_argument("--alpha", type=float, default=2.0e-4)
    parser.add_argument("--rho-ref", type=float, default=1025.0)
    parser.add_argument("--matlab-bin", default="matlab")
    ns = parser.parse_args()

    cmd = [ns.matlab_bin, "-batch", build_matlab_batch(ns)]
    print("Running MATLAB EP-flux validation backend...")
    print(" ".join(cmd))
    completed = subprocess.run(cmd, check=False)
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
