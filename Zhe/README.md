# Zhe Eddy Analysis Codebase

This directory is the current code root inside the `S-H-I-T-ocean-` Git
repository. Development is now centered on the `Zonca` branch.

The project has accumulated several historical eddy-detection and representative
vortex variants. The current README records the production path so old scripts
are not accidentally mixed into new scientific runs.

## Current Production Contract

The current default scientific workflow is:

```text
Hua b3_start2
+ 30-180 day bandpass velocity
+ boundary_monotonic circular-boundary rotation constraint
+ strict_contiguous vertical extension
+ life30 shape classification
+ coherent_only selection
+ ME_LIUTEX azimuth-preserved representative vortex
+ global_ls_alpha alignment
```

TURN and UNTURN are both allowed only as final representative-vortex structure
options:

- `turned`: rotate each object by `global_ls_alpha`; this is the main structure
  and transport frame.
- `unturned`: keep the original local east/north frame; this is a structure
  comparison, not the main transport frame.

## Main Packages

| Path | Role |
| --- | --- |
| `src/eddy_pipeline/` | Production detection, tracking, catalog, shape classification, and representative-vortex structure pipeline. |
| `src/post/` | Formal post-processing after representative-vortex construction: aggregate-product stirring, structure plots, and double-core diagnostics. |
| `src/data_downloading/` | Data download and raw preprocessing utilities. Kept separate from the detection refactor. |
| `src/Legacy/experiments/` | Temporary research experiments. Useful, but not production entry points. |
| `src/Legacy/First_temp/` | Legacy-but-active numerical helpers used by older representative and E-P/PV diagnostics. |
| `src/Legacy/Location/` | Compatibility wrappers and older entry points. Prefer `src.eddy_pipeline` and `src.post`. |
| `legacy/` | Archived diagnostics, paper replications, older variants, and historical scripts. |
| `vendor/` | Third-party/reference code such as Hua/Nencioli/MITgcm material. Do not edit as production code. |

## Production Entry Points

Use the new self-contained pipeline package first:

```powershell
python -m src.eddy_pipeline.cli --help
python -m src.eddy_pipeline.cli run-detection-to-shape --dry-run
python -m src.eddy_pipeline.cli build-representative --dry-run --shape coherent --orientation both
python -m src.eddy_pipeline.cli run-all --dry-run
```

Post-processing starts after representative vortex outputs exist:

```powershell
python -m src.post.cli --help
python -m src.post.cli run-default --dry-run --shape coherent --orientation both
python -m src.post.cli build-transport --shape coherent --orientation turned
python -m src.post.cli analyze-double-core --shape coherent --orientation both
```

The post package must not rerun detection, tracking, or shape classification.

## Output Layout

For the current Kuroshiou production result, keep structure and transport
diagnostics separate:

| Output | Meaning |
| --- | --- |
| `result_boundary_monotonic/result_coherent_only/representative_vortex_radial_seed/` | Coherent-only lifecycle objects and radial seed diagnostics. |
| `result_boundary_monotonic/result_coherent_only/representative_vortex_me_liutex/` | TURN ME_LIUTEX azimuth-preserved representative vortex. |
| `result_boundary_monotonic/result_coherent_only/representative_vortex_me_liutex_unturned/` | UNTURN structure comparison. |
| `result_boundary_monotonic/result_coherent_only/aggregate_product_stirring/` | Heat/PV stirring from product-then-mean statistics and covariance. |
| `double_core_analysis/` | Velocity-center axis vs rotation-core axis diagnostics. |

Do not use old radial-only, non-strict-contiguous, non-boundary-monotonic, or
all-shape representative outputs as the production baseline unless a study
explicitly says it is a legacy comparison.

## Transport Rule

Representative structure plots show the mean eddy, but transport must be
computed as an aggregate product:

```text
product_mean = mean(v_rot * X)
mean_product = mean(v_rot) * mean(X)
covariance = product_mean - mean_product
```

where:

- `X = theta_30_180d` for heat stirring,
- `X = q_prime` for QG-like PV stirring,
- `v_rot` is the `global_ls_alpha` rotated `y_rot` velocity component.

The mean-product field is a diagnostic baseline. The primary transport estimate
is `product_mean`, with `covariance` explaining why mean fields alone are not
enough.

## Quick Checks

```powershell
python -m py_compile src/eddy_pipeline/*.py src/post/*.py
python -m src.eddy_pipeline.cli --help
python -m src.post.cli run-default --dry-run --shape coherent --orientation both
```

See the engineering notes for the migration history:

- `docs/main_pipeline_refactor.md`
- `docs/post_pipeline_refactor.md`
- `docs/main_pipeline_contract.md`
- `docs/ambiguous_method_registry.md`
- `docs/redundancy_delete_candidates.md`
