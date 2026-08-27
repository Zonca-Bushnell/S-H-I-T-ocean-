from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd


HUA_B3_LABEL = "hua_b3_start2"
HUA_B3_METHOD_BASE = "hua_nencioli_b3_start2_bandpass_30_180d"
HUA_B3_METHOD_STRICT_CONTIGUOUS = f"{HUA_B3_METHOD_BASE}_strict_contiguous"
STRICT_CONTIGUOUS_COMPLETION_MODE = "hua_surface_contiguous_passed_layers_only"
NONCONTIGUOUS_COMPLETION_MODE = "hua_passed_layers_only"
PRODUCTION_SCIENCE_MOUTHFUL = (
    "hua_b3_start2 + 30-180d bandpass + strict_contiguous + life30 + "
    "coherent_only + global_ls_alpha"
)

HUA_B3_START2_DETECTION_PARAMS = {
    "surface_search_cells": 3,
    "deep_search_cells": 3,
    "start_radius_cells": 2,
    "max_radius_cells": 8,
    "speed_ratio_max": 3,
    "angle_jump_max_deg": 150,
    "tangent_tolerance_deg": 24,
    "symmetry_tolerance_deg": 120,
    "min_tangent_fraction": 0.55,
    "min_reversal_fraction": 0.55,
    "min_finite_fraction": 0.75,
    "direction_exception_extra": 2,
}


def strict_depth_policy_text(*, allow_noncontiguous_depth: bool) -> str:
    if allow_noncontiguous_depth:
        return "Legacy diagnostic mode: allow non-contiguous Hua-passed layers."
    return "Stop at first failed layer and keep only surface-connected contiguous layers."


def hua_method_name(*, strict_contiguous: bool) -> str:
    return HUA_B3_METHOD_STRICT_CONTIGUOUS if strict_contiguous else HUA_B3_METHOD_BASE


def completion_output_mode(*, strict_contiguous: bool) -> str:
    return STRICT_CONTIGUOUS_COMPLETION_MODE if strict_contiguous else NONCONTIGUOUS_COMPLETION_MODE


def strict_contiguous_passed_layers(
    centers: pd.DataFrame,
    *,
    group_columns: Sequence[str] = ("date", "hua_object_id"),
    depth_column: str = "depth_index",
    pass_column: str = "hua_pass",
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Keep only surface-connected Hua-passed layers 0..k for each object.

    This is the production vertical-extension rule: if layer 0 is not passed,
    the object is dropped; after the first failed layer, deeper isolated passes
    are treated as diagnostic leftovers rather than physical extension.
    """
    passed = centers[centers[pass_column].astype(bool)].copy() if pass_column in centers.columns else centers.iloc[0:0].copy()
    if passed.empty:
        return passed, {
            "strict_contiguous": 1,
            "passed_rows_before_strict": 0,
            "passed_rows_after_strict": 0,
            "isolated_pass_rows_removed": 0,
            "objects_before_strict": 0,
            "objects_after_strict": 0,
            "objects_dropped_no_surface": 0,
        }

    keep: list[pd.DataFrame] = []
    dropped_no_surface = 0
    removed_isolated = 0
    sort_columns = [col for col in (*group_columns, depth_column) if col in passed.columns]
    grouped = passed.sort_values(sort_columns).groupby(list(group_columns), sort=False)
    for _, group in grouped:
        depth_indices = set(group[depth_column].astype(int).tolist())
        if 0 not in depth_indices:
            dropped_no_surface += 1
            removed_isolated += len(group)
            continue
        last = -1
        while (last + 1) in depth_indices:
            last += 1
        kept = group[group[depth_column].astype(int).le(last)].copy()
        removed_isolated += int((group[depth_column].astype(int) > last).sum())
        if not kept.empty:
            keep.append(kept)

    strict = pd.concat(keep, ignore_index=True) if keep else passed.iloc[0:0].copy()
    strict = strict.sort_values(sort_columns).reset_index(drop=True) if sort_columns else strict.reset_index(drop=True)
    return strict, {
        "strict_contiguous": 1,
        "passed_rows_before_strict": int(len(passed)),
        "passed_rows_after_strict": int(len(strict)),
        "isolated_pass_rows_removed": int(removed_isolated),
        "objects_before_strict": int(passed[list(group_columns)].drop_duplicates().shape[0]),
        "objects_after_strict": int(strict[list(group_columns)].drop_duplicates().shape[0]) if not strict.empty else 0,
        "objects_dropped_no_surface": int(dropped_no_surface),
    }


def default_shape_output_name(start: str, end: str, lifetime_min_days: int) -> str:
    return f"shape_classification_{str(start)[:4]}_{str(end)[:4]}_{HUA_B3_LABEL}_life{int(lifetime_min_days)}"


def default_runtime_config_name() -> str:
    return f"config_{HUA_B3_LABEL}_runtime.yaml"


def default_detection_dir(output_root: Path) -> Path:
    return output_root / f"{HUA_B3_LABEL}_detection"
