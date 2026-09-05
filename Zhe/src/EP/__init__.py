"""Object-oriented EP-flux diagnostics for representative eddies.

This package is intentionally independent from the historical
``src.utils.ep_flux`` implementation.  It provides a small, explicit API for
classic, tilted, and curved-tube EP diagnostics.
"""

from .contracts import EPCase, EPFluxConfig

__all__ = ["EPCase", "EPFluxConfig"]
