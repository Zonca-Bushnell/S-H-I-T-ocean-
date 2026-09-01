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

AXIS_SOURCES = ("radial_seed", "composite_hua_refined")
ORIENTATIONS = ("turned", "unturned")


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


def axis_source_filename(axis_source: str, tau: float) -> str:
    if axis_source not in AXIS_SOURCES:
        raise ValueError(f"Unsupported axis source: {axis_source}")
    return f"{axis_source}_axis_tau{tau_tag(tau)}.csv"


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
    shape_label: str = "coherent-only"
    run_label: str = "subgrid_1_24deg"

    def validate_contract(self) -> None:
        if self.orientation not in ORIENTATIONS:
            raise ValueError(f"orientation must be one of {ORIENTATIONS}")
        if self.axis_source not in AXIS_SOURCES:
            raise ValueError(f"axis_source must be one of {AXIS_SOURCES}")
        if self.constant_n2 <= 0:
            raise ValueError("constant_n2 must be positive")

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
            "shape_label": self.shape_label,
            "run_label": self.run_label,
        }
