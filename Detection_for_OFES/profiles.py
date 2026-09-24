"""Named OFES workflows and their single sources of truth.

Profiles describe scientific choices and output layout.  Runners consume these
values instead of repeating long, subtly different command lines.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


OFES_ROOT = Path(r"E:\DATA\01_Eddy_correspond\02_OFES")
OFES_DATA_ROOT = Path(r"F:\OFES\external_OFES2")
VELOCITY_SOURCE_ROOT = OFES_ROOT / "origin_compatible_filter"
RUNS_ROOT = OFES_ROOT / "Fromthebeginning" / "runs"
HISTORICAL_ROOT = OFES_ROOT / "Fromthebeginning"
HISTORICAL_FULL_TANGENT_W = (
    HISTORICAL_ROOT / "05_TEMP"
    / "nh_cyclonic_strict_core_native_w_pointwise_unrotated_eta_19910101_19910119"
)


@dataclass(frozen=True)
class GeometryVerticalProfile:
    name: str
    ssh_definition: str
    temporal_window_days: int
    highpass_cutoff_km: float
    kernel: str
    candidate_selection: str
    remove_streamline_gate: bool
    deep_center_selection: str
    deep_center_step_cells: int
    deep_center_speed_tolerance: float
    deep_tangent_tolerance_deg: float
    deep_min_tangent_fraction: float
    vertical_profile_id: str
    vertical_output_dir_name: str
    deep_hua_mode: str
    near_streamline_min_points: int
    near_streamline_min_winding_turns: float
    near_streamline_closure_tolerance_cells: float
    near_streamline_min_finite_fraction: float
    surface_workers: int
    velocity_filter_workers: int
    vertical_workers: int
    composite_workers: int
    convolution_engine: str
    compression_level: int
    write_object_voxels: bool
    native_w_stage_raw: bool


ETA_HP500_GEOMETRY_VERTICAL = GeometryVerticalProfile(
    name="eta_hp500_geometry_vertical_v1",
    ssh_definition="ofes_eta_free_surface",
    temporal_window_days=1,
    highpass_cutoff_km=500.0,
    kernel="gaussian",
    candidate_selection="global_topn",
    remove_streamline_gate=True,
    deep_center_selection="local_step_section_bipolar",
    deep_center_step_cells=2,
    deep_center_speed_tolerance=0.20,
    deep_tangent_tolerance_deg=45.0,
    deep_min_tangent_fraction=0.35,
    vertical_profile_id="tangent45_fraction35_then_near_closed_relaxed_v1",
    vertical_output_dir_name="tangent45_fraction35_then_near_closed_relaxed",
    deep_hua_mode="tangent_then_near_closed_streamline",
    near_streamline_min_points=12,
    near_streamline_min_winding_turns=0.50,
    near_streamline_closure_tolerance_cells=2.50,
    near_streamline_min_finite_fraction=0.90,
    surface_workers=8,
    velocity_filter_workers=2,
    vertical_workers=2,
    composite_workers=8,
    convolution_engine="fft",
    compression_level=1,
    write_object_voxels=False,
    native_w_stage_raw=False,
)


def geometry_vertical_profile(name: str = ETA_HP500_GEOMETRY_VERTICAL.name) -> GeometryVerticalProfile:
    if name == "eta_hp500_geometry_vertical":
        return ETA_HP500_GEOMETRY_VERTICAL
    if name != ETA_HP500_GEOMETRY_VERTICAL.name:
        raise ValueError(f"Unknown OFES profile: {name}")
    return ETA_HP500_GEOMETRY_VERTICAL


def run_root(profile: GeometryVerticalProfile, start: str, end: str) -> Path:
    """Return the only writable root for one profile/date-range execution."""
    return RUNS_ROOT / profile.name / f"{start.replace('-', '')}_{end.replace('-', '')}"
