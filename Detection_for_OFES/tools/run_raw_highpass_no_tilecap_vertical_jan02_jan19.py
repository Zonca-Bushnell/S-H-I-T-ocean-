"""Deprecated compatibility entry point for the invalid eta-pressure run.

Use :mod:`Detection_for_OFES.tools.run_default_geometry_vertical` instead.
"""
from __future__ import annotations

def main() -> None:
    raise SystemExit(
        "This eta-pressure workflow is invalid. Use "
        "Detection_for_OFES.tools.run_default_geometry_vertical."
    )


if __name__ == "__main__":
    main()
