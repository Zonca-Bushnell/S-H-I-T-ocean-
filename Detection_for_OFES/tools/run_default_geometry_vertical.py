"""Compatibility entry point for the canonical OFES default pipeline.

New callers should invoke ``Detection_for_OFES.workflows.default_pipeline``.
The historical module name remains only so old launch notes cannot recreate a
mixed-root run.
"""
from Detection_for_OFES.workflows.default_pipeline import main


if __name__ == "__main__":
    main()
