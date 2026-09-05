from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


RHO0 = 1025.0
EARTH_OMEGA = 7.2921159e-5
DEFAULT_REFERENCE_LAT = 30.0
DEFAULT_CONSTANT_N2 = 2.0e-5

DEFAULT_RESULT_ROOT = Path(
    "/root/autodl-fs/kuroshiou/result_boundary_monotonic_subgrid_1_24deg"
)
DEFAULT_SHAPE_OUTPUT_NAME = "result_coherent_only"
DEFAULT_OUTPUT_ROOT = Path("G:/EDDY_detection/S-H-I-T-ocean-/EP-FLUX/smoke_outputs")
DEFAULT_FULL_OUTPUT_ROOT = Path("G:/EDDY_detection/S-H-I-T-ocean-/EP-FLUX/full_lifecycle_validation")

AXIS_SOURCES = ("radial_seed", "composite_hua_refined")
ORIENTATIONS = ("turned", "unturned")
BUOYANCY_SOURCES = ("thermal_wind", "streamfunction_dz")
CURVED_TUBE_MODES = ("scale_audit", "jacobian_only", "jacobian_christoffel")
SHAPE_OUTPUT_NAMES = {
    "coherent": "result_coherent_only",
    "upright_like": "result_upright_like",
}


def tau_tag(tau: float) -> str:
    return f"{int(round(tau * 100)):03d}"


def orientation_dir_name(orientation: str) -> str:
    if orientation == "turned":
        return "representative_vortex_me_liutex"
    if orientation == "unturned":
        return "representative_vortex_me_liutex_unturned"
    raise ValueError(f"Unsupported orientation: {orientation}")


def default_me_liutex_root(
    result_root: Path = DEFAULT_RESULT_ROOT,
    shape_output_name: str = DEFAULT_SHAPE_OUTPUT_NAME,
    orientation: str = "turned",
) -> Path:
    return result_root / shape_output_name / orientation_dir_name(orientation)


def default_radial_seed_root(
    result_root: Path = DEFAULT_RESULT_ROOT,
    shape_output_name: str = DEFAULT_SHAPE_OUTPUT_NAME,
) -> Path:
    return result_root / shape_output_name / "representative_vortex_radial_seed"


def shape_output_name(shape: str) -> str:
    return SHAPE_OUTPUT_NAMES.get(shape, f"result_{shape}")


def axis_source_filename(axis_source: str, tau: float) -> str:
    if axis_source not in AXIS_SOURCES:
        raise ValueError(f"Unsupported axis source: {axis_source}")
    return f"{axis_source}_axis_tau{tau_tag(tau)}.csv"


@dataclass(frozen=True)
class EPCase:
    shape: str = "coherent"
    axis_source: str = "radial_seed"
    orientation: str = "turned"
    buoyancy_source: str = "thermal_wind"

    def validate_contract(self) -> None:
        if self.axis_source not in AXIS_SOURCES:
            raise ValueError(f"axis_source must be one of {AXIS_SOURCES}")
        if self.orientation not in ORIENTATIONS:
            raise ValueError(f"orientation must be one of {ORIENTATIONS}")
        if self.buoyancy_source not in BUOYANCY_SOURCES:
            raise ValueError(f"buoyancy_source must be one of {BUOYANCY_SOURCES}")

    @property
    def shape_output_name(self) -> str:
        return shape_output_name(self.shape)

    @property
    def shape_label(self) -> str:
        return f"{self.shape}-only"

    def output_dir(self, root: Path) -> Path:
        return Path(root) / self.shape / self.axis_source / self.orientation / self.buoyancy_source

    def me_liutex_root(self, result_root: Path = DEFAULT_RESULT_ROOT) -> Path:
        return default_me_liutex_root(Path(result_root), self.shape_output_name, self.orientation)

    def radial_seed_root(self, result_root: Path = DEFAULT_RESULT_ROOT) -> Path:
        return default_radial_seed_root(Path(result_root), self.shape_output_name)


@dataclass(frozen=True)
class EPFluxConfig:
    me_liutex_root: Path
    radial_seed_root: Path
    output_dir: Path
    orientation: str = "turned"
    axis_source: str = "radial_seed"
    tau: float = 0.5
    reference_lat: float = DEFAULT_REFERENCE_LAT
    constant_n2: float = DEFAULT_CONSTANT_N2
    buoyancy_source: str = "thermal_wind"
    curved_tube_mode: str = "scale_audit"
    large_curvature_threshold: float = 1.0
    shape_label: str = "coherent-only"
    run_label: str = "subgrid_1_24deg"

    def validate_contract(self) -> None:
        if self.orientation not in ORIENTATIONS:
            raise ValueError(f"orientation must be one of {ORIENTATIONS}")
        if self.axis_source not in AXIS_SOURCES:
            raise ValueError(f"axis_source must be one of {AXIS_SOURCES}")
        if self.constant_n2 <= 0:
            raise ValueError("constant_n2 must be positive")
        if self.buoyancy_source not in BUOYANCY_SOURCES:
            raise ValueError(f"buoyancy_source must be one of {BUOYANCY_SOURCES}")
        if self.curved_tube_mode not in CURVED_TUBE_MODES:
            raise ValueError(f"curved_tube_mode must be one of {CURVED_TUBE_MODES}")
        if self.large_curvature_threshold <= 0:
            raise ValueError("large_curvature_threshold must be positive")

    @property
    def vortex_npz(self) -> Path:
        return self.me_liutex_root / "azimuthal_representative_velocity.npz"

    @property
    def axis_source_path(self) -> Path:
        return (
            self.me_liutex_root
            / "axis_sources"
            / axis_source_filename(self.axis_source, self.tau)
        )

    @property
    def f0(self) -> float:
        import math

        return 2.0 * EARTH_OMEGA * math.sin(math.radians(self.reference_lat))

    def manifest(self) -> dict[str, object]:
        return {
            "me_liutex_root": str(self.me_liutex_root),
            "radial_seed_root": str(self.radial_seed_root),
            "vortex_npz": str(self.vortex_npz),
            "axis_source": self.axis_source,
            "axis_source_path": str(self.axis_source_path),
            "orientation": self.orientation,
            "tau": self.tau,
            "reference_lat": self.reference_lat,
            "f0": self.f0,
            "constant_n2": self.constant_n2,
            "buoyancy_source": self.buoyancy_source,
            "curved_tube_mode": self.curved_tube_mode,
            "large_curvature_threshold": self.large_curvature_threshold,
            "shape_label": self.shape_label,
            "run_label": self.run_label,
        }
