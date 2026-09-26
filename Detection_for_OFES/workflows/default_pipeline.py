"""Canonical OFES eta/high-pass/geometry/vertical/native-W workflow.

This module owns orchestration only.  Scientific algorithms remain in the
existing focused tools so a stage can be audited and rerun independently.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable

import pandas as pd

from Detection_for_OFES.io.ofes import ctl_path, expected_dta_bytes, parse_ctl, require_daily_file
from Detection_for_OFES.profiles import (
    HISTORICAL_FULL_TANGENT_W,
    OFES_DATA_ROOT,
    VELOCITY_SOURCE_ROOT,
    geometry_vertical_profile,
)
from Detection_for_OFES.workflows.context import RunContext
from Detection_for_OFES.workflows.contracts import (
    code_fingerprint,
    file_identity,
    fingerprint,
    is_current,
    status_path,
    write_status,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DAY_STAGES = (
    "surface-inputs",
    "surface-filter",
    "raw-detection",
    "geometry-qc",
    "velocity-filter",
    "vertical",
    "section-bipolar",
)
OPTIONAL_STAGES = ("tracking", "shape")
RANGE_STAGES = ("native-w", "report")
ALL_STAGES = DAY_STAGES + RANGE_STAGES


def dates(start: date, end: date) -> list[date]:
    return [start + timedelta(days=offset) for offset in range((end - start).days + 1)]


def ymd(current: date) -> str:
    return current.strftime("%Y%m%d")


def run(command: list[str], log_prefix: Path) -> None:
    log_prefix.parent.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "HDF5_USE_FILE_LOCKING": "FALSE", "PYTHONNOUSERSITE": "1"}
    with log_prefix.with_suffix(".stdout.log").open("w", encoding="utf-8") as stdout, \
         log_prefix.with_suffix(".stderr.log").open("w", encoding="utf-8") as stderr:
        subprocess.run(command, cwd=REPO_ROOT, check=True, stdout=stdout, stderr=stderr, env=env)


def required(paths: Iterable[Path]) -> None:
    absent = [str(path) for path in paths if not path.exists()]
    if absent:
        raise FileNotFoundError("Required stage input is missing:\n" + "\n".join(absent))


def contract(context: RunContext, stage: str, token: str, inputs: Iterable[Path], parameters: dict[str, object]) -> dict[str, object]:
    input_paths = list(inputs)
    required(input_paths)
    payload: dict[str, object] = {
        "run_id": context.run_id,
        "profile_id": context.profile.name,
        "vertical_profile_id": context.profile.vertical_profile_id,
        "code_fingerprint": code_fingerprint(),
        "stage": stage,
        "token": token,
        "inputs": [file_identity(path) for path in input_paths],
        "parameters": parameters,
    }
    payload["fingerprint"] = fingerprint(payload)
    return payload


def annotate_csv(path: Path, context: RunContext, current: date | None = None) -> None:
    """Attach canonical identity to portable table artifacts without changing science."""
    if not path.exists() or path.stat().st_size == 0:
        return
    frame = pd.read_csv(path, low_memory=False)
    values = {
        "run_id": context.run_id,
        "profile_id": context.profile.name,
        "code_fingerprint": code_fingerprint(),
        "vertical_profile_id": context.profile.vertical_profile_id,
    }
    if current is not None:
        values["date"] = current.isoformat()
    for column, value in values.items():
        if column not in frame.columns:
            frame[column] = value
    frame.to_csv(path, index=False)


def complete_or_run(
    context: RunContext,
    stage: str,
    token: str,
    inputs: Iterable[Path],
    outputs: list[Path],
    parameters: dict[str, object],
    command: list[str],
    *,
    resume: bool,
) -> bool:
    payload = contract(context, stage, token, inputs, parameters)
    status = status_path(context.manifest_root, stage, token)
    if resume and is_current(status, payload, outputs):
        print(f"[resume] {stage} {token}", flush=True)
        return False
    if resume and status.exists() and any(path.exists() for path in outputs):
        print(f"[invalidate] {stage} {token}: contract changed; rebuilding this stage", flush=True)
    print(f"[run] {stage} {token}", flush=True)
    run(command, context.logs / stage / token)
    required(outputs)
    write_status(status, payload, outputs)
    return True


def native_eta_file(current: date) -> Path:
    meta = parse_ctl(ctl_path(OFES_DATA_ROOT, "eta"))
    return require_daily_file(OFES_DATA_ROOT, "eta", current, expected_dta_bytes(meta))


def stage_surface_inputs(context: RunContext, current: date, resume: bool) -> None:
    output = context.eta_inputs / f"global_phy_{ymd(current)}.nc"
    source_velocity = VELOCITY_SOURCE_ROOT / output.name
    complete_or_run(
        context, "surface-inputs", ymd(current), [native_eta_file(current), source_velocity], [output],
        {"ssh_definition": context.profile.surface.ssh_definition, "eta_only": True},
        [sys.executable, "-m", "Detection_for_OFES.stages.surface_inputs",
         "--data-root", str(OFES_DATA_ROOT), "--velocity-root", str(VELOCITY_SOURCE_ROOT),
         "--output-root", str(context.eta_inputs), "--start", current.isoformat(), "--end", current.isoformat(),
         "--overwrite"],
        resume=resume,
    )


def filter_command(context: RunContext, current: date, output_root: Path, layers: int, workers: int) -> list[str]:
    profile = context.profile
    return [
        sys.executable, "-m", "Detection_for_OFES.filters.gaussian_highpass",
        "--input-root", str(context.eta_inputs), "--output-root", str(output_root),
        "--start", current.isoformat(), "--end", current.isoformat(),
        "--available-start", context.start, "--available-end", context.end,
        "--temporal-window-days", "1",
        "--filter-mode", "highpass", "--large-cutoff-km", str(profile.surface.highpass_cutoff_km),
        "--spatial-kernel", profile.surface.kernel, "--large-cutoff-mode", "fixed",
        "--max-depth-layers", str(layers), "--workers", str(workers),
        "--convolution-engine", profile.convolution_engine,
        "--compression-level", str(profile.compression_level),
        "--science-tag", profile.name,
        "--overwrite",
    ]


def stage_surface_filter(context: RunContext, current: date, resume: bool) -> None:
    source = context.eta_inputs / f"global_phy_{ymd(current)}.nc"
    output = context.surface_filter / source.name
    complete_or_run(
        context, "surface-filter", ymd(current), [source], [output],
        {"mode": "daily_gaussian_highpass", "cutoff_km": 500.0, "layers": 1,
         "engine": context.profile.convolution_engine, "compression": context.profile.compression_level},
        filter_command(context, current, context.surface_filter, 1, 1), resume=resume,
    )


def stage_raw_detection(context: RunContext, current: date, resume: bool) -> None:
    source = context.surface_filter / f"global_phy_{ymd(current)}.nc"
    directory = context.raw_detection / "daily_runs" / ymd(current)
    centers = directory / "centers_hua_style.csv"
    structures = directory / "structures_hua_style.csv"
    complete_or_run(
        context, "raw-detection", ymd(current), [source], [centers, structures],
        {"candidate_selection": "global_topn", "tile_cap": "disabled", "ssh_primary": True,
         "streamline_hard_gate": False, "persistence": False, "tracking": False},
        [sys.executable, "-m", "Detection_for_OFES.stages.surface_detection",
         "--filter-output-root", str(context.surface_filter),
         "--output-root", str(context.surface_root), "--start", current.isoformat(), "--end", current.isoformat(),
         "--profile", context.profile.name, "--workers", "1"],
        resume=resume,
    )
    annotate_csv(centers, context, current)
    annotate_csv(structures, context, current)


def stage_geometry_qc(context: RunContext, current: date, resume: bool) -> None:
    geometry = context.profile.geometry
    raw = context.raw_detection / "daily_runs" / ymd(current)
    filtered = context.surface_filter / f"global_phy_{ymd(current)}.nc"
    directory = context.geometry_qc / "daily_runs" / ymd(current)
    centers = directory / "centers_hua_style.csv"
    structures = directory / "structures_hua_style.csv"
    complete_or_run(
        context, "geometry-qc", ymd(current), [raw / "centers_hua_style.csv", raw / "structures_hua_style.csv", filtered],
        [centers, structures],
        {"streamline_hard_gate": "removed", "persistence": "disabled", "jet_split": "disabled",
         "shape_error_percent": {"normal": geometry.shape_error_normal_percent,
                                   "acc": geometry.shape_error_acc_percent,
                                   "open_ocean": geometry.shape_error_open_ocean_percent},
         "min_area_cells": {"normal": geometry.area_cells_normal,
                            "open_ocean": geometry.area_cells_open_ocean},
         "min_radius_km": {"normal": geometry.radius_km_normal,
                           "open_ocean": geometry.radius_km_open_ocean},
         "same_polarity_overlap": {"center_factor": geometry.overlap_center_factor,
                                   "area_fraction": geometry.overlap_fraction}},
        [sys.executable, "-m", "Detection_for_OFES.stages.geometry_qc",
         "--source-root", str(context.raw_detection), "--filter-root", str(context.surface_filter),
         "--output-root", str(context.geometry_qc), "--day", current.isoformat(),
         "--accept-ssh-primary-without-streamline", "--skip-persistence",
         "--max-shape-error-percent", str(geometry.shape_error_normal_percent),
         "--acc-max-shape-error-percent", str(geometry.shape_error_acc_percent),
         "--open-ocean-max-shape-error-percent", str(geometry.shape_error_open_ocean_percent),
         "--min-compactness", str(geometry.compactness_normal),
         "--open-ocean-min-compactness", str(geometry.compactness_open_ocean),
         "--min-boundary-points", str(geometry.boundary_points_normal),
         "--open-ocean-min-boundary-points", str(geometry.boundary_points_open_ocean),
         "--min-area-cells", str(geometry.area_cells_normal),
         "--open-ocean-min-area-cells", str(geometry.area_cells_open_ocean),
         "--min-radius-km", str(geometry.radius_km_normal),
         "--open-ocean-min-radius-km", str(geometry.radius_km_open_ocean),
         "--overlap-center-factor", str(geometry.overlap_center_factor),
         "--overlap-area-fraction", str(geometry.overlap_fraction)],
        resume=resume,
    )
    annotate_csv(centers, context, current)
    annotate_csv(structures, context, current)


def stage_velocity_filter(context: RunContext, current: date, resume: bool) -> None:
    source = context.eta_inputs / f"global_phy_{ymd(current)}.nc"
    output = context.velocity_filter / source.name
    complete_or_run(
        context, "velocity-filter", ymd(current), [source], [output],
        {"mode": "daily_gaussian_highpass", "cutoff_km": 500.0, "layers": 105,
         "engine": context.profile.convolution_engine, "compression": context.profile.compression_level},
        filter_command(context, current, context.velocity_filter, 105, 1), resume=resume,
    )


def stage_vertical(context: RunContext, current: date, resume: bool) -> None:
    profile = context.profile
    table = context.geometry_qc / "daily_runs" / ymd(current) / "centers_hua_style.csv"
    velocity = context.velocity_filter / f"global_phy_{ymd(current)}.nc"
    directory = context.vertical / "raw_detection" / "daily_runs" / ymd(current)
    centers, structures = directory / "centers_hua_style.csv", directory / "structures_hua_style.csv"
    summary = directory / "vertical_extension_summary.json"
    complete_or_run(
        context, "vertical", ymd(current), [table, velocity], [centers, structures, summary],
        {"depth_major": True, "max_depth_layers": 105, "center_search_cells": 6,
         "center_step_cells": profile.vertical.center_step_cells,
         "mode": profile.vertical.mode, "tangent_deg": profile.vertical.tangent_tolerance_deg,
         "tangent_fraction": profile.vertical.tangent_min_fraction,
         "near_closed": {"radius_cells": [2, 12], "starts": 4, "directions": 2,
                         "min_points": profile.vertical.near_min_points,
                         "min_winding_turns": profile.vertical.near_min_winding_turns,
                         "closure_tolerance_cells": profile.vertical.near_closure_tolerance_cells,
                         "min_finite_fraction": profile.vertical.near_min_finite_fraction},
         "disabled_hard_gates": ["angle_jump", "direction_exception", "opposite_reversal"],
         "write_object_voxels": profile.vertical.write_object_voxels},
        [sys.executable, "-m", "Detection_for_OFES.stages.vertical",
         "--surface-table", str(table), "--filter-root", str(context.velocity_filter),
         "--output-root", str(context.vertical), "--day", current.isoformat(),
         "--vertical-profile-id", profile.vertical_profile_id, "--max-depth-layers", "105",
         "--deep-search-cells", "6", "--start-radius-cells", "2", "--max-radius-cells", "12",
         "--deep-hua-mode", profile.vertical.mode, "--deep-center-selection", profile.vertical.center_selection,
         "--deep-center-step-cells", str(profile.vertical.center_step_cells),
         "--deep-center-speed-tolerance", str(profile.vertical.center_speed_tolerance),
         "--deep-tangent-tolerance-deg", str(profile.vertical.tangent_tolerance_deg),
         "--deep-min-tangent-fraction", str(profile.vertical.tangent_min_fraction),
         "--enforce-tangent-alignment-hard-gate", "--disable-angle-jump-hard-gate",
         "--disable-direction-exception-hard-gate", "--disable-opposite-reversal-hard-gate",
         "--near-streamline-min-points", str(profile.vertical.near_min_points),
         "--near-streamline-min-winding-turns", str(profile.vertical.near_min_winding_turns),
         "--near-streamline-closure-tolerance-cells", str(profile.vertical.near_closure_tolerance_cells),
         "--near-streamline-min-finite-fraction", str(profile.vertical.near_min_finite_fraction)],
        resume=resume,
    )
    annotate_csv(centers, context, current)
    annotate_csv(structures, context, current)


def section_root(context: RunContext, current: date) -> Path:
    return context.section_bipolar / ymd(current) / context.vertical.name


def stage_section_bipolar(context: RunContext, current: date, resume: bool) -> None:
    thresholds = context.profile.section_bipolar
    velocity = context.velocity_filter / f"global_phy_{ymd(current)}.nc"
    structures = context.vertical / "raw_detection" / "daily_runs" / ymd(current) / "structures_hua_style.csv"
    diagnostics = section_root(context, current) / "section_bipolarity_object_summary.csv"
    catalog_dir = context.section_bipolar / ymd(current) / "catalog"
    catalog = catalog_dir / "vertical_continuation_object_catalog.csv"
    complete_or_run(
        context, "section-bipolar", ymd(current), [velocity, structures], [diagnostics, catalog],
        {"classification_only": True, "radii": list(thresholds.radii_r),
         "supported": thresholds.supported_fraction, "core": thresholds.core_fraction,
         "strict_core": thresholds.strict_core_fraction},
        [sys.executable, "-m", "Detection_for_OFES.workflows.section_stage",
         "--velocity-file", str(velocity), "--vertical-root", str(context.vertical),
         "--diagnostic-root", str(context.section_bipolar / ymd(current)),
         "--catalog-root", str(catalog_dir), "--day", current.isoformat(),
         "--vertical-definition", context.profile.vertical_profile_id,
         "--supported-min-fraction", str(thresholds.supported_fraction),
         "--core-min-fraction", str(thresholds.core_fraction),
         "--strict-core-min-fraction", str(thresholds.strict_core_fraction)],
        resume=resume,
    )
    annotate_csv(catalog, context, current)


def vertical_center_inputs(context: RunContext) -> list[Path]:
    start, end = date.fromisoformat(context.start), date.fromisoformat(context.end)
    return [
        context.vertical / "raw_detection" / "daily_runs" / ymd(day) / "centers_hua_style.csv"
        for day in dates(start, end)
    ]


def stage_tracking(context: RunContext, resume: bool) -> None:
    profile = context.profile.tracking
    output = context.tracking / "tracked_object_days.csv"
    complete_or_run(
        context, "tracking", f"{context.start}_{context.end}", vertical_center_inputs(context), [output],
        {"filtering": False, "shift_cells": profile.shift_cells,
         "continuous_score": profile.continuous_score, "split_merge_score": profile.split_merge_score,
         "boundary_contract": "surface_ssh_contour; deep_tangent_or_near_closed"},
        [sys.executable, "-m", "Detection_for_OFES.stages.tracking",
         "--vertical-root", str(context.vertical), "--output-root", str(context.tracking),
         "--start", context.start, "--end", context.end,
         "--shift-cells", str(profile.shift_cells), "--continuous-score", str(profile.continuous_score),
         "--split-merge-score", str(profile.split_merge_score)],
        resume=resume,
    )
    annotate_csv(output, context)


def stage_shape(context: RunContext, resume: bool, with_tracking: bool) -> None:
    profile = context.profile.shape
    tracking_table = context.tracking / "tracked_object_days.csv"
    inputs = vertical_center_inputs(context) + ([tracking_table] if with_tracking else [])
    output = context.shape / "object_day_shape.csv"
    command = [
        sys.executable, "-m", "Detection_for_OFES.stages.shape",
        "--vertical-root", str(context.vertical), "--output-root", str(context.shape),
        "--start", context.start, "--end", context.end,
        "--min-layers", str(profile.min_layers),
        "--upright-quantile", str(profile.upright_quantile),
        "--upright-fallback", str(profile.upright_fallback),
        "--coherent-monotonic-ratio", str(profile.coherent_monotonic_ratio),
        "--coherent-mean-turn-deg", str(profile.coherent_mean_turn_deg),
        "--complex-max-turn-deg", str(profile.complex_max_turn_deg),
        "--complex-monotonic-ratio", str(profile.complex_monotonic_ratio),
    ]
    if with_tracking:
        command += ["--tracking-table", str(tracking_table)]
    complete_or_run(
        context, "shape", f"{context.start}_{context.end}", inputs, [output],
        {"filtering": False, "min_layers": profile.min_layers,
         "upright": {"quantile": profile.upright_quantile, "fallback": profile.upright_fallback},
         "coherent": {"monotonic": profile.coherent_monotonic_ratio, "mean_turn_deg": profile.coherent_mean_turn_deg},
         "complex": {"max_turn_deg": profile.complex_max_turn_deg, "monotonic": profile.complex_monotonic_ratio}},
        command, resume=resume,
    )
    annotate_csv(output, context)


def stage_native_w(context: RunContext, resume: bool, stage_raw: bool) -> None:
    output = context.native_w / "manifest.json"
    input_catalogs = [context.section_bipolar / ymd(day) / "catalog" / "vertical_continuation_object_catalog.csv" for day in dates(date.fromisoformat(context.start), date.fromisoformat(context.end))]
    complete_or_run(
        context, "native-w", f"{context.start}_{context.end}", input_catalogs, [output],
        {"selection": "NH cyclonic section-bipolar strict-core", "w": "native OFES", "orientation": "unrotated",
         "grid": "[-2R,2R] at 0.04R", "method": "pointwise_mean", "min_objects": 8,
         "workers": context.profile.composite_workers, "raw_mode": "staged" if stage_raw else "direct_memmap"},
        [sys.executable, "-m", "Detection_for_OFES.composite.native_w",
         "--surface-qc-root", str(context.geometry_qc), "--vertical-run-root", str(context.vertical),
         "--strict-core-catalog-root", str(context.section_bipolar), "--output-root", str(context.native_w),
         "--data-root", str(OFES_DATA_ROOT), "--start", context.start, "--end", context.end,
         "--workers", str(context.profile.composite_workers), "--selection-mode", "strict_core_nh"] +
        (["--stage-raw"] if stage_raw else []),
        resume=resume,
    )
    annotate_csv(context.native_w / "selected_object_days.csv", context)


def stage_report(context: RunContext, resume: bool) -> None:
    candidate = context.native_w / "manifest.json"
    output_dir = context.reports / "native_w_comparison_to_historical_full"
    output = output_dir / "native_w_plan_comparison.png"
    complete_or_run(
        context, "report", f"{context.start}_{context.end}", [candidate, HISTORICAL_FULL_TANGENT_W / "manifest.json"], [output],
        {"comparison_baseline": str(HISTORICAL_FULL_TANGENT_W), "color_map": "coolwarm"},
        [sys.executable, "-m", "Detection_for_OFES.composite.compare",
         "--baseline-root", str(HISTORICAL_FULL_TANGENT_W), "--candidate-root", str(context.native_w),
         "--output-root", str(output_dir)],
        resume=resume,
    )


def write_run_manifest(context: RunContext, selected_stages: tuple[str, ...]) -> None:
    context.ensure_layout()
    payload = {
        "run_id": context.run_id,
        "profile_id": context.profile.name,
        "code_fingerprint": code_fingerprint(),
        "profile": asdict(context.profile),
        "date_range": [context.start, context.end],
        "selected_stages": list(selected_stages),
        "scientific_contract": {
            "ssh": "OFES eta only; no pressure correction and no MSS anomaly",
            "surface": "daily Gaussian eta - LP500km(eta)",
            "surface_qc": "SSH-primary geometry + same-polarity overlap; no streamline gate or persistence",
            "vertical": "tangent45/fraction0.35 then relaxed near-closed fallback",
            "strict_core": "section-bipolar classification only",
            "native_w": "NH cyclonic strict-core, native unrotated pointwise mean",
            "optional_non_filtering": ["tracking", "shape"],
            "excluded": ["pressur", "annual_mss", "rossby_filter", "persistence", "rebuild_w"],
        },
    }
    (context.manifest_root / "run_manifest.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    historical = {
        "read_only_baseline": str(HISTORICAL_FULL_TANGENT_W),
        "note": "Historical paths are comparison-only and are never rewritten by this workflow.",
    }
    (context.manifest_root / "historical_registry.json").write_text(json.dumps(historical, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_stages(value: str) -> tuple[str, ...]:
    requested = ALL_STAGES if value.strip().lower() == "all" else tuple(item.strip() for item in value.split(",") if item.strip())
    invalid = sorted(set(requested).difference(ALL_STAGES + OPTIONAL_STAGES))
    if invalid:
        raise ValueError(f"Unknown stages: {', '.join(invalid)}. Valid stages: {', '.join(ALL_STAGES + OPTIONAL_STAGES)}")
    return requested


def run_days_parallel(work, values: list[date], workers: int) -> None:
    """Bound date-level parallelism; each child tool remains single-date deterministic."""
    if not values:
        return
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        list(pool.map(work, values))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="eta_hp500_geometry_vertical_v1")
    parser.add_argument("--start", default="1991-01-01")
    parser.add_argument("--end", default="1991-01-19")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stages", default="all", help="Comma-separated stage names, or all.")
    parser.add_argument("--stage-raw", action="store_true", help="Optional native-W raw-file staging for network storage.")
    parser.add_argument("--with-tracking", action="store_true", help="Run optional tracking without filtering catalogs.")
    parser.add_argument("--with-shape", action="store_true", help="Run optional object-day shape classification.")
    args = parser.parse_args()
    start, end = date.fromisoformat(args.start), date.fromisoformat(args.end)
    if end < start:
        raise ValueError("--end must be on or after --start")
    profile = geometry_vertical_profile(args.profile)
    context = RunContext(profile, args.start, args.end)
    selected = parse_stages(args.stages)
    if args.with_tracking and "tracking" not in selected:
        selected += ("tracking",)
    if args.with_shape and "shape" not in selected:
        selected += ("shape",)
    write_run_manifest(context, selected)
    selected_days = dates(start, end)
    if "surface-inputs" in selected:
        run_days_parallel(lambda current: stage_surface_inputs(context, current, args.resume), selected_days, profile.surface_workers)
    if "surface-filter" in selected:
        run_days_parallel(lambda current: stage_surface_filter(context, current, args.resume), selected_days, profile.surface_workers)
    if "raw-detection" in selected:
        run_days_parallel(lambda current: stage_raw_detection(context, current, args.resume), selected_days, profile.surface_workers)
    if "geometry-qc" in selected:
        run_days_parallel(lambda current: stage_geometry_qc(context, current, args.resume), selected_days, profile.surface_workers)
    if "velocity-filter" in selected:
        run_days_parallel(lambda current: stage_velocity_filter(context, current, args.resume), selected_days, profile.velocity_filter_workers)
    if "vertical" in selected:
        run_days_parallel(lambda current: stage_vertical(context, current, args.resume), selected_days, profile.vertical_workers)
    if "section-bipolar" in selected:
        run_days_parallel(lambda current: stage_section_bipolar(context, current, args.resume), selected_days, profile.vertical_workers)
    if "tracking" in selected:
        stage_tracking(context, args.resume)
    if "shape" in selected:
        stage_shape(context, args.resume, "tracking" in selected)
    if "native-w" in selected:
        stage_native_w(context, args.resume, args.stage_raw)
    if "report" in selected:
        stage_report(context, args.resume)


if __name__ == "__main__":
    main()
