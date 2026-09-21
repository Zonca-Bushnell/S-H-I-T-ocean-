"""Wait for the three E-path catalogs, then extend Jan1 through all 105 layers."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from netCDF4 import Dataset


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ROOT = Path(
    r"E:\DATA\01_Eddy_correspond\02_OFES"
    r"\origin_unified_eta_mss_three_kernel_surface_jan01_jan19"
)
REUSABLE_VELOCITY_ROOT = Path(
    r"E:\DATA\01_Eddy_correspond\02_OFES"
    r"\origin_unified_duacs_like_mss_three_kernel_surface_jan01_jan19"
)
FULL_DEPTH_SOURCE_ROOT = Path(r"E:\DATA\01_Eddy_correspond\02_OFES\origin_compatible_filter")
ROSSBY_RADIUS_PATH = Path(
    r"E:\DATA\01_Eddy_correspond\02_OFES\rossby_radius_chelton1998"
    r"\unzip\fecampos-campos2025-a5d24c0\rossrad.nc"
)
KERNELS = ("gaussian", "lanczos", "bessel")
DAY = "1991-01-01"
DAY_COMPACT = "19910101"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, default=EXPERIMENT_ROOT)
    parser.add_argument("--full-depth-source-root", type=Path, default=FULL_DEPTH_SOURCE_ROOT)
    parser.add_argument("--rossby-radius-path", type=Path, default=ROSSBY_RADIUS_PATH)
    parser.add_argument("--reusable-velocity-root", type=Path, default=REUSABLE_VELOCITY_ROOT)
    parser.add_argument("--wait-seconds", type=int, default=30)
    parser.add_argument("--wait-timeout-hours", type=float, default=6.0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def wait_for_surface_catalogs(root: Path, wait_seconds: int, timeout_hours: float) -> dict[str, object]:
    manifest_path = root / "three_kernel_run_manifest.json"
    deadline = time.monotonic() + timeout_hours * 3600.0
    while time.monotonic() < deadline:
        if manifest_path.exists():
            manifest = read_json(manifest_path)
            status = str(manifest.get("status", ""))
            if status == "failed":
                raise RuntimeError(f"Three-kernel surface run failed: {manifest_path}")
            branches = manifest.get("branches", {})
            if status == "complete" and all(
                str(branches.get(kernel, {}).get("status", "")) == "complete"  # type: ignore[union-attr]
                for kernel in KERNELS
            ):
                for kernel in KERNELS:
                    surface = root / kernel / "catalog" / "final_catalog" / "daily_runs" / DAY_COMPACT / "centers_hua_style.csv"
                    if not surface.exists():
                        raise FileNotFoundError(surface)
                return manifest
        time.sleep(max(1, int(wait_seconds)))
    raise TimeoutError(f"Surface catalogs did not complete within {timeout_hours:g} hours")


def run_logged(command: list[str], stdout_path: Path, stderr_path: Path) -> None:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    with stdout_path.open("a", encoding="utf-8") as stdout, stderr_path.open("a", encoding="utf-8") as stderr:
        stdout.write(f"[{now_utc()}] command: {' '.join(command)}\n")
        stdout.flush()
        subprocess.run(command, cwd=REPO_ROOT, stdout=stdout, stderr=stderr, check=True)


def filtered_input_is_complete(path: Path, kernel: str) -> bool:
    if not path.exists():
        return False
    try:
        with Dataset(path) as dataset:
            return (
                len(dataset.dimensions["depth"]) == 105
                and str(getattr(dataset, "horizontal_filter_kernel", "")) == kernel
            )
    except Exception:
        return False


def build_full_depth_input(args: argparse.Namespace, kernel: str, branch: Path) -> Path:
    output_root = branch / "vertical_inputs_full105"
    output_path = output_root / f"global_phy_{DAY_COMPACT}.nc"
    if filtered_input_is_complete(output_path, kernel) and not args.force:
        return output_root
    reusable_root = args.reusable_velocity_root / kernel / "vertical_inputs_full105"
    reusable_path = reusable_root / f"global_phy_{DAY_COMPACT}.nc"
    if filtered_input_is_complete(reusable_path, kernel) and not args.force:
        return reusable_root
    command = [
        sys.executable,
        "-m",
        "Detection_for_OFES.tools.build_ofes_meso_filter",
        "--input-root", str(args.full_depth_source_root),
        "--output-root", str(output_root),
        "--start", DAY,
        "--end", DAY,
        "--available-start", DAY,
        "--available-end", DAY,
        "--temporal-window-days", "1",
        "--filter-mode", "bandpass",
        "--spatial-kernel", kernel,
        "--science-tag", f"e_path_vertical105_rossby_lower_upper180_{kernel}",
        "--large-cutoff-mode", "rossby_lower_latadaptive_upper",
        "--rossby-radius-path", str(args.rossby_radius_path),
        "--rossby-small-factor", "0.5",
        "--adaptive-large-cutoff-min-km", "180",
        "--adaptive-large-cutoff-max-km", "180",
        "--rossby-min-km", "10",
        "--rossby-max-km", "180",
        "--zonal-scale-mode", "km",
        "--meridional-scale-mode", "median",
        "--min-valid-weight-fraction", "0",
        "--max-depth-layers", "105",
        "--overwrite",
    ]
    run_logged(
        command,
        branch / "logs" / "vertical_filter_jan1.stdout.log",
        branch / "logs" / "vertical_filter_jan1.stderr.log",
    )
    if not filtered_input_is_complete(output_path, kernel):
        raise RuntimeError(f"Full-depth {kernel} input failed validation: {output_path}")
    return output_root


def read_table(path_without_suffix: Path) -> pd.DataFrame:
    parquet = path_without_suffix.with_suffix(".parquet")
    csv = path_without_suffix.with_suffix(".csv")
    if parquet.exists():
        return pd.read_parquet(parquet)
    if csv.exists():
        return pd.read_csv(csv)
    raise FileNotFoundError(path_without_suffix)


def validate_vertical(branch: Path, kernel: str, expected_surface_count: int) -> dict[str, object]:
    output_root = branch / "vertical_jan01"
    day_root = output_root / "raw_detection" / "daily_runs" / DAY_COMPACT
    summary = read_json(day_root / "vertical_extension_summary.json")
    centers = read_table(day_root / "centers_hua_style")
    structures = read_table(day_root / "structures_hua_style")
    if int(summary["surface_objects"]) != expected_surface_count:
        raise RuntimeError(
            f"{kernel} surface count mismatch: {summary['surface_objects']} != {expected_surface_count}"
        )
    if str(summary.get("algorithm")) != "depth_major_existing_hua":
        raise RuntimeError(f"Unexpected vertical algorithm for {kernel}: {summary.get('algorithm')}")
    surface_ids = structures.loc[pd.to_numeric(structures["depth_index"], errors="coerce").eq(0), "hua_object_id"]
    if int(surface_ids.nunique()) != expected_surface_count:
        raise RuntimeError(f"{kernel} surface object IDs were not preserved")
    passed_structures = structures[pd.to_numeric(structures["depth_index"], errors="coerce").ge(0)].copy()
    layers_per_object = passed_structures.groupby("hua_object_id")["depth_index"].nunique()
    stop_reason = centers.get("vertical_extension_stop_reason", pd.Series("", index=centers.index)).fillna("").astype(str)
    failure_counts = stop_reason[stop_reason.ne("")].value_counts().to_dict()
    result = {
        "kernel": kernel,
        "surface_objects": expected_surface_count,
        "center_rows": int(len(centers)),
        "passed_structure_rows": int(len(structures)),
        "maximum_accepted_depth_m": float(pd.to_numeric(structures["depth_m"], errors="coerce").max()),
        "maximum_accepted_depth_index": int(pd.to_numeric(structures["depth_index"], errors="coerce").max()),
        "mean_accepted_layers_per_object": float(layers_per_object.mean()),
        "median_accepted_layers_per_object": float(layers_per_object.median()),
        "failure_reasons": {str(key): int(value) for key, value in failure_counts.items()},
        "algorithm": str(summary["algorithm"]),
        "max_depth_layers_requested": int(summary["max_depth_layers"]),
        "output_root": str(output_root),
    }
    write_json(output_root / "vertical_jan01_summary_enriched.json", result)
    return result


def run_branch(
    args: argparse.Namespace,
    kernel: str,
    filter_root: Path,
) -> dict[str, object]:
    branch = args.experiment_root / kernel
    status_path = branch / "vertical_jan01_status.json"
    status: dict[str, object] = {
        "kernel": kernel,
        "status": "extending",
        "started_utc": now_utc(),
        "filter_root": str(filter_root),
    }
    write_json(status_path, status)
    try:
        surface_path = branch / "catalog" / "final_catalog" / "daily_runs" / DAY_COMPACT / "centers_hua_style.csv"
        expected_surface_count = int(len(pd.read_csv(surface_path)))
        status.update({
            "status": "extending",
            "filter_root": str(filter_root),
            "surface_objects": expected_surface_count,
            "filter_validated_utc": now_utc(),
        })
        write_json(status_path, status)
        output_root = branch / "vertical_jan01"
        command = [
            sys.executable,
            "-m",
            "Detection_for_OFES.tools.extend_final_surface_vertical",
            "--surface-root", str(branch / "catalog"),
            "--filter-root", str(filter_root),
            "--output-root", str(output_root),
            "--day", DAY,
            "--max-depth-layers", "105",
        ]
        run_logged(
            command,
            branch / "logs" / "vertical_extension_jan1.stdout.log",
            branch / "logs" / "vertical_extension_jan1.stderr.log",
        )
        result = validate_vertical(branch, kernel, expected_surface_count)
        status.update({"status": "complete", "completed_utc": now_utc(), "summary": result})
        write_json(status_path, status)
        return result
    except Exception as exc:
        status.update({"status": "failed", "failed_utc": now_utc(), "error": repr(exc)})
        write_json(status_path, status)
        raise


def main() -> None:
    args = parse_args()
    args.experiment_root.mkdir(parents=True, exist_ok=True)
    status_path = args.experiment_root / "three_kernel_vertical_jan01_manifest.json"
    manifest: dict[str, object] = {
        "status": "waiting_for_surface_catalogs",
        "started_utc": now_utc(),
        "day": DAY,
        "kernels": list(KERNELS),
        "max_depth_layers": 105,
        "algorithm": "depth_major_existing_hua",
        "full_depth_filter_policy": "kernel-specific Rossby LP(0.5R1)-LP(180km), temporal_window_days=1",
        "velocity_reuse_policy": (
            "Reuse validated kernel-specific 105-layer u/v from the prior baseline experiment; "
            "vertical extension does not read SSH."
        ),
        "branches": {},
    }
    write_json(status_path, manifest)
    wait_for_surface_catalogs(args.experiment_root, args.wait_seconds, args.wait_timeout_hours)
    manifest["status"] = "running"
    manifest["surface_catalogs_ready_utc"] = now_utc()
    write_json(status_path, manifest)

    # netCDF4/HDF5 is not thread-safe in the Windows environment used here.
    # Validate or build each kernel-specific velocity cache serially, then run
    # the independent vertical-extension subprocesses in parallel.
    filter_roots: dict[str, Path] = {}
    manifest["status"] = "preparing_velocity_inputs_serially"
    write_json(status_path, manifest)
    for kernel in KERNELS:
        branch = args.experiment_root / kernel
        branch_status_path = branch / "vertical_jan01_status.json"
        write_json(branch_status_path, {
            "kernel": kernel,
            "status": "validating_velocity_input",
            "started_utc": now_utc(),
        })
        try:
            filter_roots[kernel] = build_full_depth_input(args, kernel, branch)
            write_json(branch_status_path, {
                "kernel": kernel,
                "status": "velocity_input_ready",
                "filter_root": str(filter_roots[kernel]),
                "completed_utc": now_utc(),
            })
        except Exception as exc:
            write_json(branch_status_path, {
                "kernel": kernel,
                "status": "failed",
                "failed_utc": now_utc(),
                "error": repr(exc),
            })
            manifest["status"] = "failed"
            manifest["failed_kernel"] = kernel
            manifest["error"] = repr(exc)
            write_json(status_path, manifest)
            raise

    manifest["status"] = "running_vertical_subprocesses"
    manifest["velocity_inputs_ready_utc"] = now_utc()
    manifest["velocity_inputs"] = {key: str(value) for key, value in filter_roots.items()}
    write_json(status_path, manifest)

    results: dict[str, dict[str, object]] = {}
    failures: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            executor.submit(run_branch, args, kernel, filter_roots[kernel]): kernel
            for kernel in KERNELS
        }
        for future in as_completed(futures):
            kernel = futures[future]
            try:
                results[kernel] = future.result()
            except Exception as exc:
                failures[kernel] = repr(exc)
            manifest["branches"] = {**results, **{key: {"status": "failed", "error": value} for key, value in failures.items()}}
            write_json(status_path, manifest)

    summary_rows = [results[kernel] for kernel in KERNELS if kernel in results]
    pd.DataFrame(summary_rows).drop(columns=["failure_reasons"], errors="ignore").to_csv(
        args.experiment_root / "three_kernel_vertical_jan01_summary.csv", index=False
    )
    write_json(args.experiment_root / "three_kernel_vertical_jan01_summary.json", {
        "day": DAY,
        "results": summary_rows,
        "failures": failures,
    })
    manifest["status"] = "failed" if failures else "complete"
    manifest["completed_utc"] = now_utc()
    manifest["results"] = summary_rows
    manifest["failures"] = failures
    write_json(status_path, manifest)
    if failures:
        raise RuntimeError(f"Vertical branches failed: {failures}")
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
