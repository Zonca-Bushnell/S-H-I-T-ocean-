"""Run the corrected eta-MSS three-kernel surface and Jan1 vertical pipeline."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
OFES_ROOT = Path(r"F:\OFES\external_OFES2")
SOURCE_ROOT = Path(r"E:\DATA\01_Eddy_correspond\02_OFES\origin_compatible_filter")
CLIMATOLOGY_ROOT = Path(
    r"E:\DATA\01_Eddy_correspond\02_OFES\ofes2_monthly_climatology_1993_2012"
)
MSS_PATH = CLIMATOLOGY_ROOT / "climatology" / "ofes2_eta_annual_mss_1993_2012.npz"
INPUT_ROOT = Path(
    r"E:\DATA\01_Eddy_correspond\02_OFES\origin_compatible_filter_eta_mss_1993_2012"
)
OUTPUT_ROOT = Path(
    r"E:\DATA\01_Eddy_correspond\02_OFES"
    r"\origin_unified_eta_mss_three_kernel_surface_jan01_jan19"
)
REUSABLE_VELOCITY_ROOT = Path(
    r"E:\DATA\01_Eddy_correspond\02_OFES"
    r"\origin_unified_duacs_like_mss_three_kernel_surface_jan01_jan19"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ofes-root", type=Path, default=OFES_ROOT)
    parser.add_argument("--source-root", type=Path, default=SOURCE_ROOT)
    parser.add_argument("--climatology-root", type=Path, default=CLIMATOLOGY_ROOT)
    parser.add_argument("--mss-path", type=Path, default=MSS_PATH)
    parser.add_argument("--input-root", type=Path, default=INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--reusable-velocity-root", type=Path, default=REUSABLE_VELOCITY_ROOT)
    parser.add_argument("--start", default="1991-01-01")
    parser.add_argument("--end", default="1991-01-19")
    parser.add_argument("--workers", type=int, default=19)
    parser.add_argument("--overwrite-inputs", action="store_true")
    parser.add_argument("--rerun-existing", action="store_true")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def run_stage(
    *,
    name: str,
    module: str,
    arguments: list[str],
    root: Path,
    manifest: dict[str, object],
    manifest_path: Path,
) -> None:
    log_root = root / "logs" / "corrected_eta_mss_pipeline"
    log_root.mkdir(parents=True, exist_ok=True)
    stdout_path = log_root / f"{name}.stdout.log"
    stderr_path = log_root / f"{name}.stderr.log"
    command = [sys.executable, "-m", module, *arguments]
    stages = manifest["stages"]
    if not isinstance(stages, dict):
        raise TypeError("manifest stages must be a dictionary")
    stages[name] = {
        "status": "running",
        "started_utc": utc_now(),
        "command": command,
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
    }
    manifest["active_stage"] = name
    write_json(manifest_path, manifest)
    try:
        with stdout_path.open("a", encoding="utf-8") as stdout, stderr_path.open(
            "a", encoding="utf-8"
        ) as stderr:
            stdout.write(f"[{utc_now()}] command: {' '.join(command)}\n")
            stdout.flush()
            subprocess.run(command, cwd=REPO_ROOT, stdout=stdout, stderr=stderr, check=True)
    except Exception as exc:
        stages[name]["status"] = "failed"
        stages[name]["failed_utc"] = utc_now()
        stages[name]["error"] = repr(exc)
        manifest["status"] = "failed"
        manifest["failed_stage"] = name
        write_json(manifest_path, manifest)
        raise
    stages[name]["status"] = "complete"
    stages[name]["completed_utc"] = utc_now()
    write_json(manifest_path, manifest)


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_root / "corrected_eta_mss_pipeline_manifest.json"
    manifest: dict[str, object] = {
        "status": "running",
        "started_utc": utc_now(),
        "scientific_definition": "SLA_E(day)=eta(day)-MSS_eta_1993_2012",
        "atmospheric_pressure_policy": (
            "No algebraic pair correction. The modeled free-surface response already resides in eta."
        ),
        "start": args.start,
        "end": args.end,
        "workers": int(args.workers),
        "input_root": str(args.input_root),
        "output_root": str(args.output_root),
        "vertical_policy": (
            "Jan1 only; existing Hua depth continuation; reuse validated kernel-specific u/v caches "
            "because vertical extension does not read SSH"
        ),
        "stages": {},
    }
    write_json(manifest_path, manifest)

    stages = [
        (
            "build_eta_mss",
            "Detection_for_OFES.tools.build_ofes_eta_annual_mss",
            [
                "--cache-root", str(args.climatology_root),
                "--ofes-root", str(args.ofes_root),
                "--output", str(args.mss_path),
            ],
        ),
        (
            "build_eta_mss_inputs",
            "Detection_for_OFES.tools.build_ofes_duacs_like_inputs",
            [
                "--data-root", str(args.ofes_root),
                "--source-root", str(args.source_root),
                "--annual-mss-path", str(args.mss_path),
                "--output-root", str(args.input_root),
                "--start", args.start,
                "--end", args.end,
                "--max-depth-layers", "1",
                *( ["--overwrite"] if args.overwrite_inputs else [] ),
            ],
        ),
        (
            "plot_raw_sla",
            "Detection_for_OFES.tools.plot_ofes_eta_mss_sla",
            [
                "--ofes-root", str(args.ofes_root),
                "--mss-path", str(args.mss_path),
                "--output-root", str(args.output_root / "comparison"),
                "--day", args.start,
            ],
        ),
        (
            "run_three_kernel_surface",
            "Detection_for_OFES.tools.run_ofes_e_path_three_kernels",
            [
                "--input-root", str(args.input_root),
                "--output-root", str(args.output_root),
                "--start", args.start,
                "--end", args.end,
                "--workers", str(args.workers),
                *( ["--rerun-existing"] if args.rerun_existing else [] ),
            ],
        ),
        (
            "plot_three_kernel_final",
            "Detection_for_OFES.tools.plot_ofes_eta_mss_three_kernel_final",
            ["--root", str(args.output_root), "--day", args.start],
        ),
        (
            "extend_three_kernel_vertical_jan1",
            "Detection_for_OFES.tools.run_three_kernel_vertical_jan1",
            [
                "--experiment-root", str(args.output_root),
                "--reusable-velocity-root", str(args.reusable_velocity_root),
                "--wait-seconds", "5",
                "--wait-timeout-hours", "1",
            ],
        ),
    ]
    for name, module, arguments in stages:
        run_stage(
            name=name,
            module=module,
            arguments=arguments,
            root=args.output_root,
            manifest=manifest,
            manifest_path=manifest_path,
        )

    manifest.pop("active_stage", None)
    manifest["status"] = "complete"
    manifest["completed_utc"] = utc_now()
    manifest["products"] = {
        "raw_sla": str(args.output_root / "comparison" / "ofes_native_eta_mss_sla_19910101.png"),
        "three_kernel_global": str(
            args.output_root / "comparison" / "eta_mss_three_kernel_final_global_19910101.png"
        ),
        "three_kernel_north_pacific": str(
            args.output_root
            / "comparison"
            / "eta_mss_three_kernel_final_north_pacific_open_ocean_19910101.png"
        ),
        "three_kernel_south_pacific": str(
            args.output_root
            / "comparison"
            / "eta_mss_three_kernel_final_south_pacific_open_ocean_19910101.png"
        ),
        "vertical_manifest": str(args.output_root / "three_kernel_vertical_jan01_manifest.json"),
    }
    write_json(manifest_path, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
