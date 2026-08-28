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

Current legacy-but-active exception:

- `src.Legacy.First_temp`: selected QG, polar-grid, and streamfunction helpers
  are still reused by `src.eddy_pipeline` and `src.post` until those utilities
  are promoted into stable production modules.
