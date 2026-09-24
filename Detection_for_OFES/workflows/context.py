"""Path and identifier contract for one canonical OFES run."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from Detection_for_OFES.profiles import GeometryVerticalProfile, run_root


@dataclass(frozen=True)
class RunContext:
    profile: GeometryVerticalProfile
    start: str
    end: str

    @property
    def root(self) -> Path:
        return run_root(self.profile, self.start, self.end)

    @property
    def run_id(self) -> str:
        return f"{self.profile.name}/{self.start.replace('-', '')}_{self.end.replace('-', '')}"

    @property
    def manifest_root(self) -> Path:
        return self.root / "00_manifest"

    @property
    def surface_root(self) -> Path:
        return self.root / "01_surface"

    @property
    def eta_inputs(self) -> Path:
        return self.surface_root / "eta_inputs"

    @property
    def surface_filter(self) -> Path:
        return self.surface_root / "highpass_500km"

    @property
    def raw_detection(self) -> Path:
        return self.surface_root / "raw_detection"

    @property
    def geometry_qc(self) -> Path:
        return self.surface_root / "geometry_qc"

    @property
    def velocity_filter(self) -> Path:
        return self.root / "02_velocity" / "highpass_500km_full105"

    @property
    def vertical(self) -> Path:
        return self.root / "03_vertical" / "tangent45_fraction35_then_near_closed_relaxed"

    @property
    def section_bipolar(self) -> Path:
        return self.root / "04_section_bipolar"

    @property
    def native_w(self) -> Path:
        return self.root / "05_native_w" / "nh_cyclonic_strict_core"

    @property
    def reports(self) -> Path:
        return self.root / "06_reports"

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    def ensure_layout(self) -> None:
        for path in (self.manifest_root, self.surface_root, self.velocity_filter, self.vertical,
                     self.section_bipolar, self.native_w, self.reports, self.logs):
            path.mkdir(parents=True, exist_ok=True)
