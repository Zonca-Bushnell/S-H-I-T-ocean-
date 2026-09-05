"""Compatibility facade for the core-shell EP validation.

The maintained implementation now lives in :mod:`src.EP.core_shell_runner`.
Keep this module as a stable import path for older commands and notebooks.
"""

from __future__ import annotations

from .core_shell_runner import (
    DEFAULT_CORE_SHELL_OUTPUT_ROOT,
    CoreShellRequest,
    request_from_args,
    run_core_shell_ep_validation,
)

__all__ = [
    "DEFAULT_CORE_SHELL_OUTPUT_ROOT",
    "CoreShellRequest",
    "request_from_args",
    "run_core_shell_ep_validation",
]
