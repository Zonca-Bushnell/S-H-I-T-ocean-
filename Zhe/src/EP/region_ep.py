"""Region-wise EP flux interfaces for core-shell diagnostics."""

from __future__ import annotations

from .core_shell_runner import (
    _budget_for_region as budget_for_region,
    _region_ep_tilt_stats as region_ep_tilt_stats,
)

__all__ = [
    "budget_for_region",
    "region_ep_tilt_stats",
]
