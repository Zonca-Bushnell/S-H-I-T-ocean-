"""Unified material-boundary strategy interface for EP validation.

The EP package has several boundary search families.  This module gives them a
single vocabulary without forcing the validated numerical implementations into
one giant class hierarchy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol


@dataclass(frozen=True)
class BoundaryStrategy:
    """Description of one material-boundary selection strategy."""

    name: str
    family: str
    description: str
    supports_particle_advection: bool = False
    supports_cauchy_green: bool = False
    supports_lavd: bool = False
    prioritizes_pv_retention: bool = False

    def selection_contract(self) -> dict[str, object]:
        return {
            "boundary_mode": self.name,
            "boundary_family": self.family,
            "supports_particle_advection": self.supports_particle_advection,
            "supports_cauchy_green": self.supports_cauchy_green,
            "supports_lavd": self.supports_lavd,
            "prioritizes_pv_retention": self.prioritizes_pv_retention,
            "description": self.description,
        }


class BoundaryBuilder(Protocol):
    """Protocol for future strategy implementations that build masks."""

    strategy: BoundaryStrategy

    def build_mask(self, *args: object, **kwargs: object) -> tuple[object, dict[str, object]]:
        """Return a mask and audit metadata."""


class ThresholdBoundary:
    strategy = BoundaryStrategy(
        name="threshold",
        family="eulerian_threshold",
        description="Instantaneous speed/PV threshold connected to the core.",
    )


class LevelSetBoundary:
    strategy = BoundaryStrategy(
        name="levelset_v2",
        family="eulerian_optimized",
        description="Level-set/morphology refinement with leakage, smoothness, and retention penalties.",
    )


class LagrangianBoundary:
    strategy = BoundaryStrategy(
        name="lagrangian_v1",
        family="advected_eulerian",
        description="Track-wise mask advection plus local boundary optimization.",
        supports_particle_advection=True,
    )


class LAVDBoundary:
    strategy = BoundaryStrategy(
        name="lavd_material_v1",
        family="lagrangian_rotational",
        description="LAVD closed-contour candidate around a rotationally coherent core.",
        supports_particle_advection=True,
        supports_lavd=True,
    )


class GeodesicBoundary:
    strategy = BoundaryStrategy(
        name="cauchy_green_geodesic_v1",
        family="lagrangian_geodesic",
        description="Closed lambda-line candidate from Cauchy-Green strain diagnostics.",
        supports_particle_advection=True,
        supports_cauchy_green=True,
    )


class PVRetentionBoundary:
    strategy = BoundaryStrategy(
        name="pv_retention_hybrid_v1",
        family="pv_retention_lagrangian",
        description="Hybrid material boundary that prioritizes PV-core retention in the selection score.",
        supports_particle_advection=True,
        supports_cauchy_green=True,
        supports_lavd=True,
        prioritizes_pv_retention=True,
    )


BOUNDARY_STRATEGIES: Mapping[str, BoundaryStrategy] = {
    item.strategy.name: item.strategy
    for item in (
        ThresholdBoundary,
        LevelSetBoundary,
        LagrangianBoundary,
        LAVDBoundary,
        GeodesicBoundary,
        PVRetentionBoundary,
    )
}

BOUNDARY_ALIASES: Mapping[str, str] = {
    "active_contour": "levelset_v2",
    "particle_retention_v1": "lagrangian_v1",
    "lavd_hybrid_v1": "lavd_material_v1",
    "hybrid_geodesic_lavd_v1": "pv_retention_hybrid_v1",
    "pv_retention_geodesic_v1": "pv_retention_hybrid_v1",
    "pv_retention_lavd_v1": "pv_retention_hybrid_v1",
}


def resolve_boundary_strategy(boundary_mode: str) -> BoundaryStrategy:
    canonical = BOUNDARY_ALIASES.get(boundary_mode, boundary_mode)
    if canonical not in BOUNDARY_STRATEGIES:
        valid = sorted(set(BOUNDARY_STRATEGIES) | set(BOUNDARY_ALIASES))
        raise ValueError(f"unknown boundary mode {boundary_mode!r}; expected one of {valid}")
    base = BOUNDARY_STRATEGIES[canonical]
    if canonical == boundary_mode:
        return base
    return BoundaryStrategy(
        name=boundary_mode,
        family=base.family,
        description=f"Alias of {canonical}: {base.description}",
        supports_particle_advection=base.supports_particle_advection,
        supports_cauchy_green=base.supports_cauchy_green,
        supports_lavd=base.supports_lavd,
        prioritizes_pv_retention=base.prioritizes_pv_retention,
    )


def describe_boundary_modes(boundary_modes: tuple[str, ...] | list[str]) -> list[dict[str, object]]:
    return [resolve_boundary_strategy(str(mode)).selection_contract() for mode in boundary_modes]


__all__ = [
    "BOUNDARY_ALIASES",
    "BOUNDARY_STRATEGIES",
    "BoundaryBuilder",
    "BoundaryStrategy",
    "GeodesicBoundary",
    "LAVDBoundary",
    "LagrangianBoundary",
    "LevelSetBoundary",
    "PVRetentionBoundary",
    "ThresholdBoundary",
    "describe_boundary_modes",
    "resolve_boundary_strategy",
]
