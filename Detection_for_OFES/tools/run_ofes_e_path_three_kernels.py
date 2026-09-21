"""Run the production OFES E-path catalog with three spatial filter kernels."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
INPUT_ROOT = Path(
    r"E:\DATA\01_Eddy_correspond\02_OFES"
    r"\origin_compatible_filter_eta_mss_1993_2012"
)
OUTPUT_ROOT = Path(
    r"E:\DATA\01_Eddy_correspond\02_OFES"
    r"\origin_unified_eta_mss_three_kernel_surface_jan01_jan19"
)
ALLOWED_KERNELS = ("gaussian", "lanczos", "bessel")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--start", default="1991-01-01")
    parser.add_argument("--end", default="1991-01-19")
    parser.add_argument("--workers", type=int, default=19)
    parser.add_argument("--kernels", default=",".join(ALLOWED_KERNELS))
    parser.add_argument("--rerun-existing", action="store_true")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def validate_inputs(input_root: Path, start: str, end: str) -> int:
    days = pd.date_range(start, end, freq="D")
    missing = [
        input_root / f"global_phy_{day:%Y%m%d}.nc"
        for day in days
        if not (input_root / f"global_phy_{day:%Y%m%d}.nc").exists()
    ]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} E-path inputs; first missing file: {missing[0]}")
    return len(days)


def summarize_branch(branch_root: Path, start: str, end: str) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for day in pd.date_range(start, end, freq="D"):
        day_text = day.strftime("%Y%m%d")
        path = branch_root / "catalog" / "final_catalog" / "daily_runs" / day_text / "centers_hua_style.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path)
        transient = int(
            frame.get("persistence_class", pd.Series(dtype=str)).astype(str).eq("transient").sum()
        )
        if transient:
            raise RuntimeError(f"Final catalog still contains {transient} transient rows: {path}")
        rows.append({
            "day": day.date().isoformat(),
            "final_count": int(len(frame)),
            "persistence_classes": sorted(
                frame.get("persistence_class", pd.Series(dtype=str)).astype(str).unique().tolist()
            ),
        })
    return {
        "days": len(rows),
        "final_total_rows": int(sum(int(row["final_count"]) for row in rows)),
        "jan1_final_count": int(rows[0]["final_count"]),
        "daily": rows,
    }


def main() -> None:
    args = parse_args()
    kernels = tuple(item.strip().lower() for item in args.kernels.split(",") if item.strip())
    if not kernels or len(set(kernels)) != len(kernels) or any(kernel not in ALLOWED_KERNELS for kernel in kernels):
        raise ValueError("--kernels must be a unique non-empty subset of gaussian,lanczos,bessel")
    day_count = validate_inputs(args.input_root, args.start, args.end)
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_root / "three_kernel_run_manifest.json"
    manifest: dict[str, object] = {
        "status": "running",
        "started_utc": utc_now(),
        "input_root": str(args.input_root),
        "output_root": str(args.output_root),
        "baseline_definition": "ofes_eta_annual_mss_1993_2012",
        "atmospheric_pressure_policy": "no pair subtraction; modeled response retained in eta",
        "scientific_difference": "spatial filter kernel only; all E-path catalog parameters are shared",
        "start": args.start,
        "end": args.end,
        "days": day_count,
        "workers_per_branch": int(args.workers),
        "execution_policy": "branches sequential; daily detection parallel within each branch",
        "kernels": list(kernels),
        "branches": {},
    }
    write_json(manifest_path, manifest)

    for kernel in kernels:
        branch_root = args.output_root / kernel
        filter_root = branch_root / "filtered_inputs"
        catalog_root = branch_root / "catalog"
        log_root = branch_root / "logs"
        log_root.mkdir(parents=True, exist_ok=True)
        stdout_path = log_root / "branch.stdout.log"
        stderr_path = log_root / "branch.stderr.log"
        branch_status: dict[str, object] = {
            "status": "running",
            "started_utc": utc_now(),
            "kernel": kernel,
            "filter_root": str(filter_root),
            "catalog_root": str(catalog_root),
            "stdout": str(stdout_path),
            "stderr": str(stderr_path),
        }
        manifest["active_kernel"] = kernel
        manifest["branches"][kernel] = branch_status  # type: ignore[index]
        write_json(manifest_path, manifest)
        write_json(branch_root / "branch_status.json", branch_status)

        command = [
            sys.executable,
            "-m",
            "Detection_for_OFES.tools.build_unified_eddy_catalog",
            "--filter-input-root", str(args.input_root),
            "--filter-output-root", str(filter_root),
            "--output-root", str(catalog_root),
            "--start", args.start,
            "--end", args.end,
            "--max-depth-m", "3",
            "--filter-max-depth-layers", "1",
            "--workers", str(args.workers),
            "--spatial-kernel", kernel,
            "--open-ocean-no-streamline-gate",
            "--resume",
        ]
        if args.rerun_existing:
            command[-1] = "--rerun-existing"
        with stdout_path.open("a", encoding="utf-8") as stdout, stderr_path.open("a", encoding="utf-8") as stderr:
            stdout.write(f"[three-kernel] command: {' '.join(command)}\n")
            stdout.flush()
            try:
                subprocess.run(command, cwd=REPO_ROOT, stdout=stdout, stderr=stderr, check=True)
                branch_status["summary"] = summarize_branch(branch_root, args.start, args.end)
                branch_status["status"] = "complete"
                branch_status["completed_utc"] = utc_now()
            except Exception as exc:
                branch_status["status"] = "failed"
                branch_status["failed_utc"] = utc_now()
                branch_status["error"] = repr(exc)
                manifest["status"] = "failed"
                write_json(branch_root / "branch_status.json", branch_status)
                write_json(manifest_path, manifest)
                raise
        write_json(branch_root / "branch_status.json", branch_status)
        write_json(manifest_path, manifest)

    manifest.pop("active_kernel", None)
    manifest["status"] = "complete"
    manifest["completed_utc"] = utc_now()
    write_json(manifest_path, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
