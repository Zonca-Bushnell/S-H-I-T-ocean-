"""Paths and immutable identifiers for one canonical OFES run."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from Detection_for_OFES.profiles import LAYOUT_SCHEMA, OfesProfile, run_root


@dataclass(frozen=True)
class RunContext:
    profile: OfesProfile
    start: str
    end: str

    @property
    def root(self) -> Path: return run_root(self.profile, self.start, self.end)
    @property
    def run_id(self) -> str:
        token = f"{self.start.replace('-', '')}_{self.end.replace('-', '')}"
        return f"{self.profile.name}/{LAYOUT_SCHEMA}/{token}"
    @property
    def manifest_root(self) -> Path: return self.root / "00_manifest"
    @property
    def surface_root(self) -> Path: return self.root / "01_surface"
    @property
    def eta_inputs(self) -> Path: return self.surface_root / "eta_inputs"
    @property
    def surface_filter(self) -> Path: return self.surface_root / "highpass_500km"
    @property
    def raw_detection(self) -> Path: return self.surface_root / "raw_detection"
    @property
    def geometry_qc(self) -> Path: return self.surface_root / "geometry_qc"
    @property
    def velocity_filter(self) -> Path: return self.root / "02_velocity" / "highpass_500km_full105"
    @property
    def vertical(self) -> Path: return self.root / "03_vertical" / "tangent45_fraction35_then_near_closed_relaxed"
    @property
    def classification(self) -> Path: return self.root / "04_classification"
    @property
    def section_bipolar(self) -> Path: return self.classification / "section_bipolar"
    @property
    def tracking(self) -> Path: return self.root / "05_tracking"
    @property
    def shape(self) -> Path: return self.root / "06_shape"
    @property
    def composites(self) -> Path: return self.root / "07_composites"
    @property
    def native_w(self) -> Path: return self.composites / "native_w" / "nh_cyclonic_strict_core"
    @property
    def reports(self) -> Path: return self.root / "08_reports"
    @property
    def logs(self) -> Path: return self.root / "logs"

    def ensure_layout(self) -> None:
        for path in (
            self.manifest_root, self.surface_root, self.velocity_filter,
            self.vertical, self.section_bipolar, self.tracking, self.shape,
            self.native_w, self.reports, self.logs,
        ):
            path.mkdir(parents=True, exist_ok=True)
