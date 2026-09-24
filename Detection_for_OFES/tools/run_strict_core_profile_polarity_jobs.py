"""Run strict-core native-field diagnostics for two vertical profiles in sequence."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


DATA_ROOT = Path(r"E:\DATA\01_Eddy_correspond\02_OFES")
VERTICAL_ROOT = DATA_ROOT / "Fromthebeginning" / "04_Vertical" / "eta_highpass_500km_no_tilecap_ssh_geometry_section_bipolar_jan01_jan19"
TEMP_ROOT = DATA_ROOT / "Fromthebeginning" / "05_TEMP"
COMPOSITE_ROOT = TEMP_ROOT / "strict_core_profile_polarity_native_fields_19910101_19910119"
ANTI_ROOT = TEMP_ROOT / "latest_profile_representative_strict_core_anticyclonic_raw_fields_19910101_19910119"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=COMPOSITE_ROOT)
    parser.add_argument("--anti-output-root", type=Path, default=ANTI_ROOT)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def run(command: list[str], status: dict[str, object], state_path: Path, label: str) -> None:
    status["current_stage"] = label
    status["updated_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(state_path, status)
    print("[strict-core-jobs] " + " ".join(command), flush=True)
    subprocess.run(command, check=True)
    status.setdefault("completed_stages", []).append(label)
    status["updated_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(state_path, status)


def complete(*paths: Path) -> bool:
    return all(path.exists() for path in paths)


def composite_paths(root: Path, polarity: str) -> tuple[Path, Path, Path]:
    group = f"NH_{polarity}_strict_core_19910101_19910119"
    group_root = root / "paper_pointwise_no_rotation" / group
    return group_root / "isopycnal_composite.npz", root / "selected_object_days.csv", group_root


def main() -> None:
    args = parse_args()
    python = sys.executable
    state_path = args.output_root / "job_status.json"
    status: dict[str, object] = {
        "started_utc": datetime.now(timezone.utc).isoformat(), "status": "running", "current_stage": None,
        "completed_stages": [], "profile_pair": ["old_full", "tangent45_fraction35_then_near_closed_relaxed"],
        "density_anomaly": "same-depth 1.5R-2R outer-ring median reference; not climatology",
        "prho_climatology_dependency": False,
    }
    write_json(state_path, status)

    catalog_root = VERTICAL_ROOT / "section_bipolar_catalogs_tangent_then_near_closed_relaxed"
    surface_qc = VERTICAL_ROOT / "surface_geometry_qc"
    new_run = VERTICAL_ROOT / "vertical_continuation_tangent_then_near_closed_relaxed_jan01_jan19"
    selection = args.anti_output_root / "selected_representative_object_days.csv"
    anti_panel = args.anti_output_root / "with_outer_ring_reference"
    if not args.resume or not complete(selection, args.anti_output_root / "selection_manifest.json"):
        run([
            python, "-m", "Detection_for_OFES.tools.build_latest_strict_core_representative_selection",
            "--catalog-root", str(catalog_root), "--surface-qc-root", str(surface_qc),
            "--output-root", str(args.anti_output_root), "--polarity", "anticyclonic",
        ], status, state_path, "select_latest_anticyclonic_representatives")
    expected_anti = [
        anti_panel / "19910119" / "19910119_00033" / "raw_prho_outer_ring_anomaly_native_w_family.png",
        anti_panel / "19910117" / "19910117_00035" / "raw_prho_outer_ring_anomaly_native_w_family.png",
        anti_panel / "19910103" / "19910103_02969" / "raw_prho_outer_ring_anomaly_native_w_family.png",
        anti_panel / "19910114" / "19910114_08374" / "raw_prho_outer_ring_anomaly_native_w_family.png",
        anti_panel / "19910107" / "19910107_02633" / "raw_prho_outer_ring_anomaly_native_w_family.png",
        anti_panel / "19910101" / "19910101_05347" / "raw_prho_outer_ring_anomaly_native_w_family.png",
    ]
    if not args.resume or not complete(*expected_anti):
        run([
            python, "-m", "Detection_for_OFES.tools.render_multiday_strict_core_raw_fields",
            "--vertical-root", str(VERTICAL_ROOT), "--vertical-run-root", str(new_run),
            "--selection-file", str(selection), "--output-root", str(args.anti_output_root),
            "--polarity", "anticyclonic", "--mode", "outer-ring-reference",
        ], status, state_path, "render_latest_anticyclonic_representatives")

    profiles = {
        "old_full": {
            "vertical_run": VERTICAL_ROOT / "vertical_continuation_local_step_2cells_section_bipolar",
            "diagnostics": VERTICAL_ROOT / "strict_core_diagnostics_local_lat_scaled",
        },
        "new_hybrid": {
            "vertical_run": new_run,
            "diagnostics": VERTICAL_ROOT / "section_bipolar_diagnostics_tangent_then_near_closed_relaxed",
        },
    }
    for profile, source in profiles.items():
        for polarity in ("cyclonic", "anticyclonic"):
            output = args.output_root / profile / f"NH_{polarity}"
            archive, selected, _ = composite_paths(output, polarity)
            if not args.resume or not complete(archive, selected, output / "manifest.json"):
                run([
                    python, "-m", "Detection_for_OFES.tools.run_multiday_nh_cyclonic_native_w_composite",
                    "--vertical-root", str(VERTICAL_ROOT), "--surface-qc-root", str(surface_qc),
                    "--vertical-run-root", str(source["vertical_run"]),
                    "--strict-core-diagnostics-root", str(source["diagnostics"]),
                    "--output-root", str(output), "--selection-mode", "strict_core_nh",
                    "--hemisphere", "NH", "--polarity", polarity, "--profile-label", profile,
                    "--start", "1991-01-01", "--end", "1991-01-19", "--workers", "8", "--min-objects", "8",
                ], status, state_path, f"composite_{profile}_{polarity}")
            figure = output / "strict_core_native_prho_outer_ring_w.png"
            if not args.resume or not complete(figure, figure.with_suffix(".pdf")):
                run([
                    python, "-m", "Detection_for_OFES.tools.plot_strict_core_native_fields_composite",
                    "--input", str(archive), "--selected-object-days", str(selected),
                    "--output-root", str(output), "--profile-label", profile, "--polarity", polarity,
                ], status, state_path, f"plot_{profile}_{polarity}")

    for polarity in ("cyclonic", "anticyclonic"):
        old_root = args.output_root / "old_full" / f"NH_{polarity}"
        new_root = args.output_root / "new_hybrid" / f"NH_{polarity}"
        comparison = args.output_root / "comparison" / f"NH_{polarity}"
        if not args.resume or not complete(comparison / "native_w_plan_comparison.png", comparison / "native_fields_difference_metrics.csv"):
            run([
                python, "-m", "Detection_for_OFES.tools.compare_native_w_composites",
                "--baseline-root", str(old_root), "--candidate-root", str(new_root),
                "--output-root", str(comparison), "--polarity", polarity,
            ], status, state_path, f"compare_old_full_new_hybrid_{polarity}")

    status["status"] = "complete"
    status["current_stage"] = None
    status["completed_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(state_path, status)
    print(json.dumps(status, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
