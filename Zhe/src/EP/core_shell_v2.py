"""Core-shell V2 partition diagnostics.

This module keeps the V1 numerical kernel in :mod:`src.EP.core_shell_runner` and
changes only the partition contract and output bookkeeping.  V2 tightens the
inner material core and lets the PV-active shell extend farther outward so the
two-zone interpretation can be tested without overwriting V1 results.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from .core_shell_runner import CoreShellRequest, request_from_args as _v1_request_from_args
from .core_shell_runner import run_core_shell_ep_validation


DEFAULT_CORE_SHELL_V2_OUTPUT_ROOT = Path(
    "/root/autodl-fs/kuroshiou/EP-FLUX/core_shell_partition_v2"
)

V2_CONTRACT: dict[str, Any] = {
    "partition_version": "core_shell_v2",
    "inner_material_core": {
        "meaning": "Hua/LAVD-near velocity core used for trapping and material coherence.",
        "default_radius_over_R": 1.2,
        "selection": "low-speed connected core optimized with the configured material-boundary mode",
        "required_diagnostics": [
            "weak_core_retention",
            "pv_high_quantile_retention",
            "boundary_exchange",
            "particle_retention_when_available",
        ],
    },
    "pv_active_shell": {
        "meaning": "PV-anomaly and shear-active shell used for heat/PV stirring diagnostics.",
        "default_outer_radius_over_R": 2.5,
        "selection": "connected high-|q'| area outside inner core plus a narrow bridge for continuity",
        "default_pv_quantile": 0.80,
    },
    "exchange_layer": {
        "meaning": "Interface budget between material core and PV-active shell.",
        "selection": "inner-core boundary and the adjacent shell/contact cells",
        "budget_terms": [
            "heat_boundary_flux",
            "pv_boundary_flux",
            "buoyancy_boundary_flux",
            "momentum_boundary_flux",
        ],
    },
}


def request_from_args(args) -> CoreShellRequest:
    """Build a V2 request while preserving the V1 runner API."""

    request = _v1_request_from_args(args)
    return replace(
        request,
        output_root=Path(args.output_root),
        core_radius_over_R=float(args.core_radius_over_R),
        shell_outer_radius_over_R=float(args.shell_outer_radius_over_R),
        pv_shell_quantile=float(args.pv_shell_quantile),
        object_aggregate_transport=not bool(args.no_object_aggregate_transport),
    )


def _write_v2_contract(output_root: Path, request: CoreShellRequest) -> None:
    notes = output_root / "literature_notes"
    notes.mkdir(parents=True, exist_ok=True)
    tables = output_root / "tables"
    figures = output_root / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    manifest = {
        **V2_CONTRACT,
        "runtime_defaults": {
            "shapes": list(request.shapes),
            "axis_sources": list(request.axis_sources),
            "orientations": list(request.orientations),
            "buoyancy_sources": list(request.buoyancy_sources),
            "inner_boundary_mode": request.inner_boundary_mode,
            "boundary_budget": request.boundary_budget,
            "core_radius_over_R": request.core_radius_over_R,
            "shell_outer_radius_over_R": request.shell_outer_radius_over_R,
            "speed_core_quantile": request.speed_core_quantile,
            "pv_core_quantile": request.pv_core_quantile,
            "pv_shell_quantile": request.pv_shell_quantile,
            "shell_dilation_cells": request.shell_dilation_cells,
            "min_core_retention": request.min_core_retention,
            "object_aggregate_transport": request.object_aggregate_transport,
        },
    }
    (output_root / "core_shell_partition_v2_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    (notes / "partition_criteria_zh.md").write_text(
        "\n".join(
            [
                "# Core-Shell V2 分区判据",
                "",
                "V2 不再要求一个低泄漏边界同时圈住旋转材料核和 PV 动力核。",
                "",
                "## inner_material_core",
                "",
                "- 默认半径：`r/R <= 1.2`。",
                "- 判据：弱速核心连通、Hua/LAVD 近同位、低 boundary leakage、高 weak-core retention。",
                "- 物理用途：trapping、material coherence、低交换旋转核。",
                "",
                "## pv_active_shell",
                "",
                "- 默认外半径：`r/R <= 2.5`。",
                "- 判据：inner core 外侧高 `|q'|`、高剪切或月牙强速带；默认取 `|q'|` 前 20%。",
                "- 物理用途：heat/PV stirring、aggregate-product covariance、EP tilt correction。",
                "",
                "## exchange_layer",
                "",
                "- 判据：inner core 边界及其与 shell 接触区。",
                "- 物理用途：单独记录 heat/PV/momentum boundary exchange。",
                "",
                "## 判读",
                "",
                "若 PV retention 提高伴随 leakage/exchange 增大，应解释为 PV-active shell 与 material core 分离，",
                "而不是继续强迫 `PV core subset LAVD core`。",
                "",
            ]
        ),
        encoding="utf-8",
    )


def run_core_shell_v2_ep_validation(request: CoreShellRequest) -> dict[str, Path]:
    """Run V2 diagnostics and write a V2 contract next to the numeric outputs."""

    if request.dry_run:
        print("Core-shell V2 partition dry-run")
        print(f"output root: {request.output_root}")
        print(f"inner material core radius: r/R <= {request.core_radius_over_R}")
        print(f"PV-active shell outer radius: r/R <= {request.shell_outer_radius_over_R}")
        print(f"PV shell quantile: {request.pv_shell_quantile}")
        print("regions: inner_material_core, pv_active_shell, exchange_layer")
        return run_core_shell_ep_validation(request)

    request.output_root.mkdir(parents=True, exist_ok=True)
    _write_v2_contract(request.output_root, request)
    outputs = run_core_shell_ep_validation(request)
    _write_v2_contract(request.output_root, request)
    outputs["v2_manifest"] = request.output_root / "core_shell_partition_v2_manifest.json"
    outputs["v2_literature_notes"] = request.output_root / "literature_notes" / "partition_criteria_zh.md"
    return outputs
