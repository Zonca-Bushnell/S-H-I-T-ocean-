from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PRODUCTION_POST_SCOPE = (
    "hua_b3_start2 + 30-180d bandpass + boundary_monotonic + "
    "strict_contiguous + optional local_1_24deg_refined_velocity_center + "
    "life30 + coherent_only + ME_LIUTEX azimuth_preserved + "
    "global_ls_alpha"
)

DEFAULT_RESULT_ROOT = Path("/root/autodl-fs/kuroshiou/result_boundary_monotonic")
DEFAULT_FILTER_ROOT = Path("/root/autodl-fs/kuroshiou/Filter")
DEFAULT_FILTER_TEMPLATE = "global_phy_{year}_bandpass_30_180d.nc"
DEFAULT_SHAPE = "coherent"
DEFAULT_ORIENTATION = "turned"


@dataclass(frozen=True)
class PostPaths:
    result_root: Path
    shape: str

    @property
    def shape_root(self) -> Path:
        return self.result_root / f"result_{self.shape}_only"

    @property
    def radial_seed_root(self) -> Path:
        return self.shape_root / "representative_vortex_radial_seed"

    @property
    def turned_root(self) -> Path:
        return self.shape_root / "representative_vortex_me_liutex"

    @property
    def unturned_root(self) -> Path:
        return self.shape_root / "representative_vortex_me_liutex_unturned"

    @property
    def transport_root(self) -> Path:
        return self.shape_root / "aggregate_product_stirring"

    @property
    def figures_root(self) -> Path:
        return self.shape_root / "post_figures"

    @property
    def double_core_root(self) -> Path:
        return self.shape_root / "double_core_analysis"


def orientation_roots(paths: PostPaths, orientation: str) -> list[tuple[str, Path]]:
    if orientation == "turned":
        return [("turned", paths.turned_root)]
    if orientation == "unturned":
        return [("unturned", paths.unturned_root)]
    if orientation == "both":
        return [("turned", paths.turned_root), ("unturned", paths.unturned_root)]
    raise ValueError(f"Unsupported orientation: {orientation}")

