# S-H-I-T-ocean

This repository currently stores the working code under `Zhe/`.

The active branch for day-to-day development is `Zonca`. The current production
scientific workflow is the Kuroshiou eddy pipeline:

```text
Hua b3_start2
+ 30-180 day bandpass velocity
+ boundary_monotonic
+ strict_contiguous vertical extension
+ life30 shape filtering
+ coherent_only
+ ME_LIUTEX azimuth-preserved representative vortex
+ global_ls_alpha alignment
```

## Where To Start

```powershell
cd Zhe
python -m src.eddy_pipeline.cli --help
python -m src.post.cli --help
python -m src.EP.cli --help
```

Formal EP diagnostics must be launched through `python -m src.EP.cli ...`.
Files under `EP-FLUX/archive/` are historical outputs, not active entry points.

For the detailed code map, see:

- `Zhe/README.md`
- `Zhe/docs/main_pipeline_refactor.md`
- `Zhe/docs/post_pipeline_refactor.md`
- `Zhe/docs/main_pipeline_contract.md`
- `EP-FLUX/engineering/ep_package_refactor_notes.md`

## Package Boundaries

- `Zhe/src/eddy_pipeline/`: production detection, tracking, catalog, shape
  classification, and final representative vortex construction.
- `Zhe/src/post/`: production post-processing after representative vortex
  construction, including aggregate-product stirring, structure figures, panel
  family plots, and double-core diagnostics.
- `Zhe/src/EP/`: self-contained object-oriented EP theory diagnostics. Formal
  EP calculations use internal EP IO, numerics, axis-source, and transport
  moment helpers instead of importing from `src.post` or historical `src.utils`.
- `Zhe/src/utils/`: shared numerical, geospatial, field-sampling, and
  representative-composite helpers used by `eddy_pipeline` and `post`.
- `Zhe/src/data_downloading/`: data acquisition utilities, intentionally kept
  outside the current refactor boundary.
- `Zhe/src/Legacy/`: compatibility and historical code kept out of the
  production package namespace.
- `Zhe/legacy/`: archived historical scripts, paper replications, diagnostics,
  and non-default representative variants.
- `Zhe/vendor/`: third-party or reference source code. Do not edit directly as
  production code.

## Dependency Direction

Production code should follow this direction:

```text
src.eddy_pipeline  ┐
                   ├──> src.utils
src.post           ┘

src.EP ──> src.EP.io / src.EP.numerics / src.EP.axis_sources / src.EP.transport_moments
```

`src.Legacy/*` is retained for reproducibility and old experiments, but it is
not a production dependency target. When an old numerical helper is still useful
for `eddy_pipeline` or `post`, promote it into `src.utils` first.

Formal EP diagnostics are stricter: keep them self-contained inside `src.EP`.
Do not make EP theory code import from `src.post` or historical utility modules.

## Output Convention

For the current Kuroshiou production result:

- Structure composite:
  `result_boundary_monotonic_subgrid_1_24deg/result_coherent_only/representative_vortex_me_liutex/`
- Unturned structure comparison:
  `result_boundary_monotonic_subgrid_1_24deg/result_coherent_only/representative_vortex_me_liutex_unturned/`
- Transport diagnostics:
  `result_boundary_monotonic_subgrid_1_24deg/result_coherent_only/aggregate_product_stirring/`

Transport diagnostics must use aggregate-product statistics:

```text
product_mean = mean(v_rot * X)
mean_product = mean(v_rot) * mean(X)
covariance = product_mean - mean_product
```

where `X` is `theta_30_180d` for heat stirring or QG-like `q_prime` for PV
stirring. The mean-product result is only a comparison baseline, not the primary
transport estimate.
