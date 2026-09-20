"""Run the controlled five-day OFES SSH-baseline x filter-kernel experiment."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = Path(r"E:\DATA\01_Eddy_correspond\02_OFES\baseline_ab_longterm_january_climatology_filter6_19910101")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--start", default="1991-01-01")
    parser.add_argument("--end", default="1991-01-05")
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--kernels", default="gaussian,lanczos,bessel")
    parser.add_argument("--stage", choices=("inputs", "catalogs", "plot", "all"), default="all")
    parser.add_argument("--overwrite-inputs", action="store_true")
    parser.add_argument("--rerun-existing", action="store_true")
    return parser.parse_args()


def call(command: list[str]) -> None:
    print("[baseline-ab] " + " ".join(command), flush=True)
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def tag_and_materialize_jan1(root: Path, label: str, kernel: str) -> dict[str, object]:
    for table in root.rglob("centers_hua_style.csv"):
        frame = pd.read_csv(table)
        frame["baseline_definition"] = label
        frame["spatial_filter_kernel"] = kernel
        frame.to_csv(table, index=False)
    final_source = root / "final_catalog" / "daily_runs" / "19910101"
    final_dest = root / "jan1_final_catalog" / "daily_runs" / "19910101"
    final_dest.mkdir(parents=True, exist_ok=True)
    for name in ("centers_hua_style.csv", "structures_hua_style.csv"):
        source = final_source / name
        if source.exists():
            shutil.copy2(source, final_dest / name)
    centers = pd.read_csv(final_dest / "centers_hua_style.csv")
    if "persistence_class" in centers.columns and centers["persistence_class"].astype(str).eq("transient").any():
        raise RuntimeError("Jan1 final catalog still contains transient objects")
    return {
        "baseline": label,
        "spatial_filter_kernel": kernel,
        "jan1_final_count": int(len(centers)),
        "jan1_final_catalog": str(final_dest),
    }


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    labels = ("old_jan01_jan19_mean", "longterm_january_1993_2012")
    kernels = tuple(item.strip().lower() for item in args.kernels.split(",") if item.strip())
    if not kernels or any(item not in {"gaussian", "lanczos", "bessel"} for item in kernels):
        raise ValueError("--kernels must be a non-empty subset of gaussian,lanczos,bessel")
    if args.stage in ("inputs", "all"):
        command = [sys.executable, "-m", "Detection_for_OFES.tools.build_ofes_baseline_ab_inputs", "--output-root", str(args.output_root), "--start", args.start, "--end", args.end]
        if args.overwrite_inputs:
            command.append("--overwrite")
        call(command)
    summaries: list[dict[str, object]] = []
    if args.stage in ("catalogs", "all"):
        for label in labels:
            for kernel in kernels:
                branch_name = f"{label}__{kernel}"
                branch = args.output_root / branch_name
                input_branch = args.output_root / label
                command = [
                    sys.executable, "-m", "Detection_for_OFES.tools.build_unified_eddy_catalog",
                    "--filter-input-root", str(input_branch / "input"),
                    "--filter-output-root", str(branch / "rossby_filter"),
                    "--output-root", str(branch / "catalog"),
                    "--start", args.start, "--end", args.end,
                    "--workers", str(args.workers), "--filter-max-depth-layers", "1",
                    "--spatial-kernel", kernel,
                ]
                if args.rerun_existing:
                    command.append("--rerun-existing")
                call(command)
                summaries.append(tag_and_materialize_jan1(branch / "catalog", label, kernel))
    if args.stage in ("plot", "all"):
        call([sys.executable, "-m", "Detection_for_OFES.tools.plot_ofes_baseline_ab", "--root", str(args.output_root)])
    (args.output_root / "ab_run_manifest.json").write_text(json.dumps({
        "status": "complete" if args.stage == "all" else args.stage,
        "run_days": [args.start, args.end], "workers_per_branch": args.workers,
        "scientific_difference": "SSH baseline x spatial-filter kernel only; all unified-catalog CLI defaults are shared.",
        "kernels": list(kernels),
        "support_days": "Jan02-Jan05 are persistence support only; conclusions use Jan01 final_catalog only.",
        "branches": summaries,
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
