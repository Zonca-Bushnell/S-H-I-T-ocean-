"""Canonical scientific and execution profile for OFES eddy detection."""
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
LAYOUT_SCHEMA = "layout_v2"


@dataclass(frozen=True)
class SurfaceProfile:
    ssh_definition: str = "ofes_eta_free_surface"
    highpass_cutoff_km: float = 500.0
    kernel: str = "gaussian"
    convolution_engine: str = "fft"
    candidate_selection: str = "global_topn"
    tile_cap: bool = False
    ssh_primary: bool = True
    streamline_hard_gate: bool = False
    persistence: bool = False


@dataclass(frozen=True)
class DetectionProfile:
    seed_windows_cells: str = "3,5,7,11"
    start_radius_cells: int = 2
    max_radius_cells: int = 12
    ssh_window_cells: int = 7
    ssh_primary_level_count: int = 8
    ssh_primary_window_factor: float = 4.0
    ssh_primary_max_radius_factor: float = 2.0
    ssh_primary_min_amplitude_cm: float = 1.0
    open_ocean_min_amplitude_cm: float = 0.4
    open_ocean_low_lat_amplitude_cm: float = 0.8
    open_ocean_high_lat_amplitude_cm: float = 0.25
    open_ocean_window_factor: float = 6.0
    open_ocean_max_radius_factor: float = 5.0
    open_ocean_seed_window_cells: int = 3
    target_boxes: str = (
        "190,245,25,55,north_pacific;190,280,-50,-30,south_pacific;"
        "320,330,25,45,north_atlantic;335,355,-45,-20,south_atlantic"
    )
    target_min_amplitude_cm: float = 0.10
    target_window_factor: float = 8.0
    target_max_radius_factor: float = 6.0
    contour_shape_error_percent: float = 70.0
    contour_acc_shape_error_percent: float = 55.0


@dataclass(frozen=True)
class GeometryProfile:
    shape_error_normal_percent: float = 70.0
    shape_error_acc_percent: float = 55.0
    shape_error_open_ocean_percent: float = 80.0
    compactness_normal: float = 0.20
    compactness_open_ocean: float = 0.12
    boundary_points_normal: int = 9
    boundary_points_open_ocean: int = 6
    area_cells_normal: int = 16
    area_cells_open_ocean: int = 9
    radius_km_normal: float = 25.0
    radius_km_open_ocean: float = 18.0
    overlap_center_factor: float = 0.75
    overlap_fraction: float = 0.50


@dataclass(frozen=True)
class VerticalProfile:
    profile_id: str = "tangent45_fraction35_then_near_closed_relaxed_v1"
    mode: str = "tangent_then_near_closed_streamline"
    center_selection: str = "local_step_section_bipolar"
    center_step_cells: int = 2
    center_speed_tolerance: float = 0.20
    search_cells: int = 6
    radius_min_cells: int = 2
    radius_max_cells: int = 12
    tangent_tolerance_deg: float = 45.0
    tangent_min_fraction: float = 0.35
    near_min_points: int = 12
    near_min_winding_turns: float = 0.50
    near_closure_tolerance_cells: float = 2.50
    near_min_finite_fraction: float = 0.90
    disable_angle_jump: bool = True
    disable_direction_exception: bool = True
    disable_opposite_reversal: bool = True
    write_object_voxels: bool = False


@dataclass(frozen=True)
class SectionBipolarProfile:
    supported_fraction: float = 0.50
    core_fraction: float = 0.60
    strict_core_fraction: float = 0.70
    radii_r: tuple[float, float] = (0.6, 1.0)


@dataclass(frozen=True)
class TrackingProfile:
    shift_cells: int = 1
    continuous_score: float = 0.25
    split_merge_score: float = 0.75


@dataclass(frozen=True)
class ShapeProfile:
    min_layers: int = 6
    upright_quantile: float = 0.20
    upright_fallback: float = 0.12
    coherent_monotonic_ratio: float = 0.72
    coherent_mean_turn_deg: float = 35.0
    complex_max_turn_deg: float = 100.0
    complex_monotonic_ratio: float = 0.55


@dataclass(frozen=True)
class CompositeProfile:
    selection: str = "nh_cyclonic_section_bipolar_strict_core"
    coordinate_min_r: float = -2.0
    coordinate_max_r: float = 2.0
    spacing_r: float = 0.04
    min_objects: int = 8
    orientation: str = "unrotated"
    method: str = "pointwise_mean"
    stage_raw: bool = False


@dataclass(frozen=True)
class PerformanceProfile:
    surface_workers: int = 8
    velocity_workers: int = 2
    vertical_workers: int = 2
    composite_workers: int = 8
    compression_level: int = 1


@dataclass(frozen=True)
class OfesProfile:
    name: str
    surface: SurfaceProfile = SurfaceProfile()
    detection: DetectionProfile = DetectionProfile()
    geometry: GeometryProfile = GeometryProfile()
    vertical: VerticalProfile = VerticalProfile()
    section_bipolar: SectionBipolarProfile = SectionBipolarProfile()
    tracking: TrackingProfile = TrackingProfile()
    shape: ShapeProfile = ShapeProfile()
    composite: CompositeProfile = CompositeProfile()
    performance: PerformanceProfile = PerformanceProfile()

    @property
    def vertical_profile_id(self) -> str: return self.vertical.profile_id
    @property
    def vertical_output_dir_name(self) -> str: return "tangent45_fraction35_then_near_closed_relaxed"
    @property
    def convolution_engine(self) -> str: return self.surface.convolution_engine
    @property
    def compression_level(self) -> int: return self.performance.compression_level
    @property
    def surface_workers(self) -> int: return self.performance.surface_workers
    @property
    def velocity_filter_workers(self) -> int: return self.performance.velocity_workers
    @property
    def vertical_workers(self) -> int: return self.performance.vertical_workers
    @property
    def composite_workers(self) -> int: return self.performance.composite_workers


ETA_HP500_GEOMETRY_VERTICAL = OfesProfile(name="eta_hp500_geometry_vertical_v1")
GeometryVerticalProfile = OfesProfile


def geometry_vertical_profile(name: str = ETA_HP500_GEOMETRY_VERTICAL.name) -> OfesProfile:
    if name != ETA_HP500_GEOMETRY_VERTICAL.name:
        raise ValueError(f"Unknown OFES profile: {name}")
    return ETA_HP500_GEOMETRY_VERTICAL


def run_root(profile: OfesProfile, start: str, end: str) -> Path:
    token = f"{start.replace('-', '')}_{end.replace('-', '')}"
    return RUNS_ROOT / profile.name / LAYOUT_SCHEMA / token
