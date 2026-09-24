"""Keep the two resumable OFES2 monthly-prho download workers alive."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from .download_ofes2_monthly_prho import OUTPUT_ROOT


@dataclass(frozen=True)
class WorkerSpec:
    name: str
    start_year: int
    end_year: int


SPECS = (WorkerSpec("prho_part_1993_2002", 1993, 2002), WorkerSpec("prho_part_2003_2012", 2003, 2012))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--stall-minutes", type=float, default=45.0)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    return parser.parse_args()


def log(path: Path, message: str) -> None:
    line = f"{time.strftime('%Y-%m-%dT%H:%M:%S%z')} {message}"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    print(line, flush=True)


def complete(root: Path, spec: WorkerSpec) -> bool:
    path = root / "workers" / spec.name / "monthly_prho_manifest.json"
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("status") == "complete"
    except (OSError, ValueError, json.JSONDecodeError):
        return False


def start(root: Path, spec: WorkerSpec, log_dir: Path, supervisor_log: Path) -> subprocess.Popen[bytes]:
    stdout = (log_dir / f"ofes2_monthly_prho_{spec.name}.stdout.log").open("ab")
    stderr = (log_dir / f"ofes2_monthly_prho_{spec.name}.stderr.log").open("ab")
    command = [
        sys.executable, "-u", "-m", "Detection_for_OFES.tools.download_ofes2_monthly_prho",
        "--output-root", str(root), "--start-year", str(spec.start_year), "--end-year", str(spec.end_year),
        "--worker-name", spec.name, "--lat-block-rows", "5",
    ]
    process = subprocess.Popen(command, stdout=stdout, stderr=stderr)
    stdout.close()
    stderr.close()
    log(supervisor_log, f"started {spec.name} pid={process.pid} period={spec.start_year}-{spec.end_year}")
    return process


def main() -> None:
    args = parse_args()
    log_dir = args.output_root / "logs"
    supervisor_log = log_dir / "ofes2_monthly_prho_supervisor.log"
    active: dict[str, subprocess.Popen[bytes]] = {}
    while True:
        done = 0
        for spec in SPECS:
            if complete(args.output_root, spec):
                done += 1
                continue
            process = active.get(spec.name)
            if process is None or process.poll() is not None:
                if process is not None:
                    log(supervisor_log, f"{spec.name} exited with code {process.returncode}; resuming complete-month cache")
                active[spec.name] = start(args.output_root, spec, log_dir, supervisor_log)
                continue
            stdout = log_dir / f"ofes2_monthly_prho_{spec.name}.stdout.log"
            age = time.time() - stdout.stat().st_mtime if stdout.exists() else 0.0
            if age >= args.stall_minutes * 60.0:
                log(supervisor_log, f"{spec.name} stalled for {age / 60.0:.1f} minutes; terminating for clean cache resume")
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                active[spec.name] = start(args.output_root, spec, log_dir, supervisor_log)
        if done == len(SPECS):
            break
        time.sleep(args.poll_seconds)
    log(supervisor_log, "all monthly prho caches complete; building day-weighted annual MSS")
    build_log = (log_dir / "ofes2_prho_annual_mss.log").open("ab")
    try:
        subprocess.run([sys.executable, "-u", "-m", "Detection_for_OFES.tools.build_ofes2_prho_annual_mss", "--output-root", str(args.output_root)], stdout=build_log, stderr=subprocess.STDOUT, check=True)
    finally:
        build_log.close()
    log(supervisor_log, "annual prho MSS complete")


if __name__ == "__main__":
    main()
