from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace


DEFAULT_DATA_ROOT = Path(r"F:\OFES\external_OFES2")
DEFAULT_RESULT_ROOT = Path(r"E:\DATA\01_Eddy_correspond\02_OFES\available_jan01_jan19_refined_ofes_grid")


@dataclass(frozen=True)
class IOConfig:
    data_root: Path = DEFAULT_DATA_ROOT
    result_root: Path = DEFAULT_RESULT_ROOT
    output_root: Path | None = None
    start: str = "1991-01-01"
    end: str = "1991-01-19"
    extract_workers: int = 1


@dataclass(frozen=True)
class RebuildConfig:
    max_depth_layers: int = 105
    grid_n: int = 81
    extent_r: float = 4.0
    farfield_inner_r: float = 2.0
    farfield_outer_r: float = 4.0
    smooth_sigma_cells: float = 1.0
    rho_bg_source: str = "regional_box"
    regional_bg_lon_half_width_deg: float = 5.0
    regional_bg_lat_half_width_deg: float = 2.0
    regional_bg_exclude_r: float = 1.5
    regional_bg_min_valid_fraction: float = 0.25
    disable_rho_stabilization: bool = False
    rho_bg_smooth_sigma_layers: float = 2.0
    rho_z_min: float = 2.0e-5
    eta_rho_cap_m: float = 500.0
    eta_rho_cap_mode: str = "fixed"
    bad_rho_z_mode: str = "mask"
    farfield_valid_min_fraction: float = 0.35
    rebuild_formula: str = "relative_advection"
    velocity_reference: str = "farfield_relative"
    translation_profile: str = "layer_tracking"
    layer_tracking_max_distance_r: float = 2.0
    layer_tracking_max_distance_km: float = 250.0
    translation_background_layers: int = 10


@dataclass(frozen=True)
class FilterConfig:
    native_w_temporal_filter: str = "lowpass_running_mean"
    native_w_filter_window_days: int = 10
    rebuild_density_filter: str = "joint_lowpass"
    rebuild_density_filter_window_days: int = 10
    eta_horizontal_lowpass_sigma_r: float = 0.5
    density_bg_detrend_order: int = 1
    density_bg_detrend_fit_ring: str = "1.5,4.0"
    rebuild_velocity_filter: str = "joint_lowpass"
    rebuild_velocity_filter_window_days: int = 10
    uv_horizontal_lowpass_sigma_r: float = 0.5
    translation_profile_smooth_sigma_layers: float = 2.0
    translation_profile_max_speed_m_s: float = 0.5


@dataclass(frozen=True)
class SelectionConfig:
    date: str = "1991-01-10"
    hua_object_id: str | None = None
    max_objects: int = 8
    crossing_lats: str = "20,40"
    intersect_radius_r: float = 1.0
    strict_crossing_only: bool = False
    band_lat_min: float = 30.0
    band_lat_max: float = 35.0
    band_max_objects: int = 8


@dataclass(frozen=True)
class PlotConfig:
    backend: str = "pillow"


@dataclass(frozen=True)
class WRebuildConfig:
    stages: str
    io: IOConfig
    rebuild: RebuildConfig
    filters: FilterConfig
    selection: SelectionConfig
    plot: PlotConfig

    @property
    def output_root(self) -> Path:
        return self.io.output_root if self.io.output_root else self.io.result_root / "w_rebuild_diagnostics"

    def to_runtime_args(self) -> SimpleNamespace:
        payload: dict[str, object] = {"stages": self.stages, "native_w_vertical_alignment": "layer_center"}
        for group in [self.io, self.rebuild, self.filters, self.selection, self.plot]:
            payload.update(group.__dict__)
        payload["output_root"] = self.output_root
        return SimpleNamespace(**payload)


def config_from_args(args: argparse.Namespace) -> WRebuildConfig:
    return WRebuildConfig(
        stages=str(args.stages),
        io=IOConfig(
            data_root=Path(args.data_root),
            result_root=Path(args.result_root),
            output_root=Path(args.output_root) if args.output_root else None,
            start=str(args.start),
            end=str(args.end),
            extract_workers=int(args.extract_workers),
        ),
        rebuild=RebuildConfig(
            max_depth_layers=int(args.max_depth_layers),
            grid_n=int(args.grid_n),
            extent_r=float(args.extent_r),
            farfield_inner_r=float(args.farfield_inner_r),
            farfield_outer_r=float(args.farfield_outer_r),
            smooth_sigma_cells=float(args.smooth_sigma_cells),
            rho_bg_source=str(args.rho_bg_source),
            regional_bg_lon_half_width_deg=float(args.regional_bg_lon_half_width_deg),
            regional_bg_lat_half_width_deg=float(args.regional_bg_lat_half_width_deg),
            regional_bg_exclude_r=float(args.regional_bg_exclude_r),
            regional_bg_min_valid_fraction=float(args.regional_bg_min_valid_fraction),
            disable_rho_stabilization=bool(args.disable_rho_stabilization),
            rho_bg_smooth_sigma_layers=float(args.rho_bg_smooth_sigma_layers),
            rho_z_min=float(args.rho_z_min),
            eta_rho_cap_m=float(args.eta_rho_cap_m),
            eta_rho_cap_mode=str(args.eta_rho_cap_mode),
            bad_rho_z_mode=str(args.bad_rho_z_mode),
            farfield_valid_min_fraction=float(args.farfield_valid_min_fraction),
            rebuild_formula=str(args.rebuild_formula),
            velocity_reference=str(args.velocity_reference),
            translation_profile=str(args.translation_profile),
            layer_tracking_max_distance_r=float(args.layer_tracking_max_distance_r),
            layer_tracking_max_distance_km=float(args.layer_tracking_max_distance_km),
            translation_background_layers=int(args.translation_background_layers),
        ),
        filters=FilterConfig(
            native_w_temporal_filter=str(args.native_w_temporal_filter),
            native_w_filter_window_days=int(args.native_w_filter_window_days),
            rebuild_density_filter=str(args.rebuild_density_filter),
            rebuild_density_filter_window_days=int(args.rebuild_density_filter_window_days),
            eta_horizontal_lowpass_sigma_r=float(args.eta_horizontal_lowpass_sigma_r),
            density_bg_detrend_order=int(args.density_bg_detrend_order),
            density_bg_detrend_fit_ring=str(args.density_bg_detrend_fit_ring),
            rebuild_velocity_filter=str(args.rebuild_velocity_filter),
            rebuild_velocity_filter_window_days=int(args.rebuild_velocity_filter_window_days),
            uv_horizontal_lowpass_sigma_r=float(args.uv_horizontal_lowpass_sigma_r),
            translation_profile_smooth_sigma_layers=float(args.translation_profile_smooth_sigma_layers),
            translation_profile_max_speed_m_s=float(args.translation_profile_max_speed_m_s),
        ),
        selection=SelectionConfig(
            date=str(args.date),
            hua_object_id=args.hua_object_id,
            max_objects=int(args.max_objects),
            crossing_lats=str(args.crossing_lats),
            intersect_radius_r=float(args.intersect_radius_r),
            strict_crossing_only=bool(args.strict_crossing_only),
            band_lat_min=float(args.band_lat_min),
            band_lat_max=float(args.band_lat_max),
            band_max_objects=int(args.band_max_objects),
        ),
        plot=PlotConfig(backend=str(args.backend)),
    )
