from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path


REQUIRED_MODULES = [
    "numpy",
    "scipy",
    "pandas",
    "netCDF4",
    "matplotlib",
    "yaml",
    "tqdm",
    "fastparquet",
    "pyarrow",
    "contourpy",
]


def has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def run_command(cmd: list[str], cwd: Path, log_path: Path, env: dict[str, str] | None = None) -> dict[str, object]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as handle:
        proc = subprocess.run(cmd, cwd=str(cwd), env=env, stdout=handle, stderr=subprocess.STDOUT)
    elapsed = time.perf_counter() - started
    return {
        "cmd": cmd,
        "cwd": str(cwd),
        "log_path": str(log_path),
        "returncode": int(proc.returncode),
        "elapsed_seconds": float(elapsed),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Origin_eddy_detection acceleration paths.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-root", type=Path, default=Path(r"E:\DATA\01_Eddy_correspond\03_Original_detection\benchmarks"))
    parser.add_argument("--filter-root", type=Path, default=Path(r"F:\Global Ocean Ensemble Physics Reanalysis Kuroshio Current\FILTER"))
    parser.add_argument("--start", default="1993-01-01")
    parser.add_argument("--end", default="1993-01-03")
    parser.add_argument("--max-depth-m", type=float, default=1000.0)
    parser.add_argument("--max-candidates-per-day", type=int, default=20)
    parser.add_argument("--lifetime-min-days", type=int, default=1)
    parser.add_argument("--radius-min-m", type=float, default=10_000.0)
    parser.add_argument("--min-valid-layers", type=int, default=1)
    parser.add_argument("--matlab-use-gpu", action="store_true")
    parser.add_argument("--matlab-exe", type=Path, default=Path(r"D:\Util\Ma\01_Matlab\bin\matlab.exe"))
    parser.add_argument("--skip-python-runs", action="store_true")
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    dependency_state = {name: has_module(name) for name in REQUIRED_MODULES}
    results: dict[str, object] = {
        "python": sys.executable,
        "dependency_state": dependency_state,
        "date_window": {"start": args.start, "end": args.end},
        "runs": [],
    }

    env = os.environ.copy()
    env["PYTHONNOUSERSITE"] = "1"
    base_cmd = [
        sys.executable,
        str(project_root / "run_origin_eddy_pipeline.py"),
        "run-detection-to-shape",
        "--start",
        args.start,
        "--end",
        args.end,
        "--max-depth-m",
        str(args.max_depth_m),
        "--max-candidates-per-day",
        str(args.max_candidates_per_day),
        "--detect-parallel",
        "1",
        "--detect-shard-mode",
        "year",
        "--lifetime-min-days",
        str(args.lifetime_min_days),
        "--radius-min-m",
        str(args.radius_min_m),
        "--min-valid-layers",
        str(args.min_valid_layers),
    ]

    matlab_cache_dir = output_root / ("matlab_candidate_cache_gpu" if args.matlab_use_gpu else "matlab_candidate_cache_cpu")
    matlab_code = (
        "addpath('" + str(project_root / "matlab").replace("'", "''") + "'); "
        "origin_precompute_candidates("
        "'" + str(args.filter_root).replace("'", "''") + "',"
        "'" + str(matlab_cache_dir).replace("'", "''") + "',"
        "'" + args.start + "',"
        "'" + args.end + "',"
        "7,"
        + str(int(args.max_candidates_per_day))
        + ","
        + str(bool(args.matlab_use_gpu)).lower()
        + ");"
    )
    matlab_run = run_command(
        [str(args.matlab_exe), "-batch", matlab_code],
        project_root,
        output_root / "logs" / ("matlab_candidate_gpu.log" if args.matlab_use_gpu else "matlab_candidate_cpu.log"),
    )
    results["runs"].append({"name": "matlab_candidate_precompute_gpu" if args.matlab_use_gpu else "matlab_candidate_precompute_cpu", **matlab_run})

    if not args.skip_python_runs and all(dependency_state[name] for name in ["numpy", "scipy", "pandas", "netCDF4", "matplotlib", "tqdm"]):
        no_cache_dir = output_root / "python_year_cache_smoke"
        run = run_command(
            [*base_cmd, "--output-root", str(no_cache_dir)],
            project_root,
            output_root / "logs" / "python_year_cache_smoke.log",
            env,
        )
        results["runs"].append({"name": "python_year_shard_no_candidate_cache", **run})

        with_cache_dir = output_root / "python_with_matlab_candidate_cache_smoke"
        cmd = [
            *base_cmd,
            "--output-root",
            str(with_cache_dir),
            "--candidate-cache-dir",
            str(matlab_cache_dir),
        ]
        run = run_command(cmd, project_root, output_root / "logs" / "python_with_matlab_candidate_cache_smoke.log", env)
        results["runs"].append({"name": "python_year_shard_with_matlab_candidate_cache", **run})
    else:
        results["skipped_python_runs"] = True
        results["skip_reason"] = "Missing required Python modules or --skip-python-runs was set."

    summary_path = output_root / "benchmark_summary.json"
    summary_path.write_text(json.dumps(results, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(summary_path)


if __name__ == "__main__":
    main()
