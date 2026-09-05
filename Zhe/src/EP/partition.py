"""Core-shell partition interfaces for EP diagnostics.

This module is the formal seam for region masks:

- ``inner_material_core``: low-leakage, weak-speed/LAVD-like material core.
- ``pv_active_shell``: PV-active stirring shell outside the material core.
- ``exchange_layer``: interface band used for boundary-exchange budgets.

The first refactor keeps the validated implementation in
``core_shell_runner`` and exposes it here under non-private names.  Follow-up
work can move the bodies here without changing callers.
"""

from __future__ import annotations

from .core_shell_runner import (
    _build_shell_mask as build_shell_mask,
    _dilate as dilate,
    _interface_metrics as interface_metrics,
    _pv_centroid_seed as pv_centroid_seed,
    _retention as retention,
)

__all__ = [
    "build_shell_mask",
    "dilate",
    "interface_metrics",
    "pv_centroid_seed",
    "retention",
]
