# Codebase Layout And Usage Boundaries

This repository is a local import of the server code from:

`root@connect.westc.seetacloud.com:32305:/root/Verify`

Imported on: see `SERVER_SYNC_MANIFEST.json`.

## Production Pipeline

- `src/eddy_pipeline/`
  - Production pipeline for Hua/Nencioli detection, feature/group tracking, strict-contiguous catalog construction, life30 shape classification, and final representative vortex generation.
  - Treat this as the current source of truth for identification through representative-vortex structure builds.

- `src/post/`
  - Formal post-processing after representative-vortex construction: aggregate-product stirring, structure figures, and double-core diagnostics.
  - This package should not rerun detection, tracking, or shape classification.

- `src/utils/`
  - Shared numerical, geospatial, field-sampling, streamfunction, E-P/PV, and representative-composite helpers.
  - `src/eddy_pipeline` and `src/post` should depend on this package instead of importing from `src/Legacy`.

- `src/Legacy/First_temp/`
  - Historical numerical helpers. Stable production dependencies have been promoted into `src/utils`.
  - Keep it for reproducibility and old scripts, but do not add new production imports from it.

- `src/data_downloading/`
  - Data download and subset scripts.
  - Keep credentials and local data paths out of commits.

## Experiments

- `src/Legacy/experiments/temp/`
  - Temporary research scripts and one-off diagnostics.
  - Code here is not automatically production-approved.
  - Promote stable scripts into `src/eddy_pipeline`, `src/post`, or `src/utils` only after naming, tests, and documentation are cleaned up.

- `src/Legacy/experiments/theory_validation/`
  - Theory and mechanism validation experiments.

- `src/Legacy/validation/`
  - Older validation code. Check current imports before moving or deleting.

## Vendor Code

- `vendor/`
  - Third-party or copied reference code, including Hua/Nencioli, MITgcm, and py-eddy-tracker material.
  - Do not edit vendor code in place unless the change is explicitly documented.
  - Prefer adapters/wrappers in `src/` for our own behavior.

## Root-Level Scripts

Root-level Python files are historical entrypoints and deployment helpers. Before using one, check whether an equivalent maintained entrypoint exists under `src/eddy_pipeline/` or `src/post/`.

## Current Scientific Default

For Kuroshiou representative vortex work, the current default scientific鍙ｅ緞 is:

`Hua b3_start2 + 30-180d bandpass + boundary-monotonic + strict-contiguous + life30 + coherent-only + ME_LIUTEX azimuth-preserved + global_ls_alpha`

Transport diagnostics should be kept separate from structural composites:

- structure: `representative_vortex_me_liutex/`
- covariance transport: `aggregate_product_stirring/`
