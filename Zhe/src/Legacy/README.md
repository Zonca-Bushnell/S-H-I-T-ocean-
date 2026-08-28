# Legacy Source Packages

This directory is intentionally not a peer of the production packages.

Active production code lives in:

- `src.eddy_pipeline`
- `src.post`
- `src.data_downloading`

The packages archived here are retained for reproducibility, old diagnostics,
paper replications, and numerical helpers that are still being phased out. New
production code should not import from this directory unless the dependency is
explicitly documented as legacy-but-active.

Previously legacy-but-active utilities from `src.Legacy.First_temp` have been
promoted into `src.utils`. New production code should depend on `src.utils`
instead of reaching back into this directory.
