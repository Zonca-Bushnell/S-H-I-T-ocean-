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


@dataclass(frozen=True)
class GeometryVerticalProfile:
    name: str
    output_root: Path
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


ETA_HP500_GEOMETRY_VERTICAL = GeometryVerticalProfile(
    name="eta_hp500_geometry_vertical",
    output_root=OFES_ROOT
    / "Fromthebeginning"
    / "04_Vertical"
    / "eta_highpass_500km_no_tilecap_ssh_geometry_section_bipolar_jan01_jan19",
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
)


def geometry_vertical_profile(name: str = ETA_HP500_GEOMETRY_VERTICAL.name) -> GeometryVerticalProfile:
    if name != ETA_HP500_GEOMETRY_VERTICAL.name:
        raise ValueError(f"Unknown OFES profile: {name}")
    return ETA_HP500_GEOMETRY_VERTICAL
