"""Resume missing vertical days through a transient MATLAB NetCDF binary bridge."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from datetime import datetime
from pathlib import Path

from Detection_for_OFES.profiles import geometry_vertical_profile


REPO_ROOT = Path(__file__).resolve().parents[2]
MATLAB_TOOL_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="eta_hp500_geometry_vertical")
    parser.add_argument("--start", default="1991-01-01")
    parser.add_argument("--end", default="1991-01-19")
    parser.add_argument("--matlab", default="matlab")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--retry-delay-seconds", type=float, default=30.0)
    parser.add_argument("--keep-bridge", action="store_true")
    return parser.parse_args()


def selected_days(start: date, end: date) -> list[date]:
    return [start + timedelta(days=offset) for offset in range((end - start).days + 1)]


def complete(output_root: Path, current: date) -> bool:
    daily = output_root / "raw_detection" / "daily_runs" / f"{current:%Y%m%d}"
    paths = tuple(daily / name for name in (
        "centers_hua_style.csv", "structures_hua_style.csv", "vertical_extension_summary.json",
    ))
    if not all(path.exists() and path.stat().st_size > 0 for path in paths):
        return False
    try:
        json.loads(paths[-1].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return True


def matlab_quote(value: Path) -> str:
    return str(value).replace("'", "''")


def bridge_complete(daily: Path) -> bool:
    metadata_path = daily / "metadata.json"
    if not metadata_path.exists():
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        nz, ny, nx = (int(value) for value in metadata["shape_depth_lat_lon"])
        expected = nz * ny * nx * 4
        return all((daily / name).stat().st_size == expected for name in ("uo_glor.f32", "vo_glor.f32"))
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return False


def write_status(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    profile = geometry_vertical_profile(args.profile)
    root = profile.output_root
    velocity = root / "full_depth_velocity_highpass_500km"
    surface = root / "surface_geometry_qc" / "daily_runs"
    output = root / "vertical_continuation_local_step_2cells_section_bipolar"
    bridge = root / "_matlab_binary_bridge"
    logs = root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    selected = selected_days(date.fromisoformat(args.start), date.fromisoformat(args.end))
    pending = [current for current in selected if not complete(output, current)]
    workers = max(1, int(args.workers))
    status_path = root / "vertical_worker_status.json"
    status_lock = threading.Lock()
    status: dict[str, object] = {
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "workers": workers,
        "max_attempts": int(args.max_attempts),
        "selected_days": [str(day) for day in selected],
        "initially_pending_days": [str(day) for day in pending],
        "running_days": [],
        "completed_days": [str(day) for day in selected if complete(output, day)],
        "failed_days": {},
        "phase": "bridge_export",
    }
    write_status(status_path, status)
    print(json.dumps({"pending_days": [str(day) for day in pending], "mode": "serial_bridge_then_parallel_vertical", "workers": workers}), flush=True)

    # MATLAB performs the NetCDF reads serially. The expensive continuation then
    # runs in isolated Python processes, so HDF5 state is never shared by workers.
    for current in pending:
        stem = f"{current:%Y%m%d}"
        daily_bridge = bridge / stem
        source_nc = velocity / f"global_phy_{stem}.nc"
        if bridge_complete(daily_bridge):
            print(f"[bridge] {current} reuse complete export", flush=True)
        else:
            if daily_bridge.exists():
                shutil.rmtree(daily_bridge)
            daily_bridge.mkdir(parents=True)
            matlab_command = (
                f"addpath('{matlab_quote(MATLAB_TOOL_DIR)}');"
                f"export_filtered_velocity_binary('{matlab_quote(source_nc)}','{matlab_quote(daily_bridge)}');"
            )
            print(f"[bridge] {current} export start", flush=True)
            with (logs / f"{stem}.bridge.stdout.log").open("w", encoding="utf-8") as out, (logs / f"{stem}.bridge.stderr.log").open("w", encoding="utf-8") as err:
                subprocess.run([args.matlab, "-batch", matlab_command], cwd=REPO_ROOT, check=True, stdout=out, stderr=err)

    status["phase"] = "vertical_continuation"
    write_status(status_path, status)

    def extend_one(current: date) -> tuple[date, bool, str]:
        stem = f"{current:%Y%m%d}"
        daily_bridge = bridge / stem
        command = [
            sys.executable, "-m", "Detection_for_OFES.tools.extend_final_surface_vertical",
            "--surface-table", str(surface / stem / "centers_hua_style.csv"),
            "--filter-root", str(velocity), "--filter-binary-root", str(bridge),
            "--output-root", str(output), "--day", current.isoformat(),
            "--max-depth-layers", "105", "--deep-search-cells", "6",
            "--start-radius-cells", "2", "--max-radius-cells", "12",
            "--deep-hua-mode", "full", "--deep-center-selection", profile.deep_center_selection,
            "--deep-center-step-cells", str(profile.deep_center_step_cells),
            "--deep-center-speed-tolerance", str(profile.deep_center_speed_tolerance),
            "--deep-tangent-tolerance-deg", str(profile.deep_tangent_tolerance_deg),
            "--deep-min-tangent-fraction", str(profile.deep_min_tangent_fraction),
            "--enforce-tangent-alignment-hard-gate", "--disable-angle-jump-hard-gate",
            "--disable-direction-exception-hard-gate", "--disable-opposite-reversal-hard-gate", "--resume",
        ]
        with status_lock:
            running = set(status["running_days"])
            running.add(str(current))
            status["running_days"] = sorted(running)
            write_status(status_path, status)
        last_error = ""
        for attempt in range(1, max(1, int(args.max_attempts)) + 1):
            print(f"[vertical] {current} attempt={attempt} continuation start", flush=True)
            with (logs / f"{stem}.vertical.stdout.log").open("a", encoding="utf-8") as out, (logs / f"{stem}.vertical.stderr.log").open("a", encoding="utf-8") as err:
                out.write(f"\n===== attempt {attempt} {datetime.now().isoformat(timespec='seconds')} =====\n")
                err.write(f"\n===== attempt {attempt} {datetime.now().isoformat(timespec='seconds')} =====\n")
                result = subprocess.run(
                    command, cwd=REPO_ROOT, stdout=out, stderr=err,
                    env={**os.environ, "HDF5_USE_FILE_LOCKING": "FALSE"},
                )
            if result.returncode == 0 and complete(output, current):
                if not args.keep_bridge and daily_bridge.exists():
                    shutil.rmtree(daily_bridge)
                print(f"[vertical] {current} complete attempt={attempt}", flush=True)
                return current, True, ""
            last_error = f"returncode={result.returncode}; complete={complete(output, current)}"
            print(f"[vertical] {current} attempt={attempt} failed: {last_error}", flush=True)
            if attempt < max(1, int(args.max_attempts)):
                time.sleep(max(0.0, float(args.retry_delay_seconds)))
        return current, False, last_error

    failures: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(extend_one, current): current for current in pending}
        for future in as_completed(futures):
            current, succeeded, error = future.result()
            with status_lock:
                running = set(status["running_days"])
                running.discard(str(current))
                status["running_days"] = sorted(running)
                if succeeded:
                    completed = set(status["completed_days"])
                    completed.add(str(current))
                    status["completed_days"] = sorted(completed)
                else:
                    failures[str(current)] = error
                    status["failed_days"] = dict(failures)
                write_status(status_path, status)

    if bridge.exists() and not any(bridge.iterdir()):
        bridge.rmdir()
    status["phase"] = "complete" if not failures else "complete_with_failures"
    status["finished_at"] = datetime.now().isoformat(timespec="seconds")
    status["failed_days"] = failures
    write_status(status_path, status)
    print(json.dumps({"status": status["phase"], "processed_days": len(pending), "failed_days": failures}), flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
