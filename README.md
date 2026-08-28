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
```

For the detailed code map, see:

- `Zhe/README.md`
- `Zhe/docs/main_pipeline_refactor.md`
- `Zhe/docs/post_pipeline_refactor.md`
- `Zhe/docs/main_pipeline_contract.md`

## Package Boundaries

- `Zhe/src/eddy_pipeline/`: production detection, tracking, catalog, shape
  classification, and final representative vortex construction.
- `Zhe/src/post/`: production post-processing after representative vortex
  construction, including aggregate-product stirring, structure figures, and
  double-core diagnostics.
- `Zhe/src/data_downloading/`: data acquisition utilities, intentionally kept
  outside the current refactor boundary.
- `Zhe/src/Legacy/experiments/`: temporary research experiments, kept out of
  the production package namespace.
- `Zhe/src/Legacy/First_temp/`: legacy-but-active numerical helpers that are
  still imported explicitly by selected production diagnostics.
- `Zhe/src/Legacy/Location/`: older entry points and compatibility code.
- `Zhe/src/Legacy/validation/`: historical validation scripts.
- `Zhe/legacy/`: archived historical scripts, paper replications, diagnostics,
  and non-default representative variants.
- `Zhe/vendor/`: third-party or reference source code. Do not edit directly as
  production code.

## Output Convention

For the current Kuroshiou production result:

- Structure composite:
  `result_boundary_monotonic/result_coherent_only/representative_vortex_me_liutex/`
- Unturned structure comparison:
  `result_boundary_monotonic/result_coherent_only/representative_vortex_me_liutex_unturned/`
- Transport diagnostics:
  `result_boundary_monotonic/result_coherent_only/aggregate_product_stirring/`
- Double-core diagnostics:
  `double_core_analysis/` or a dedicated validation directory.

Transport diagnostics must use aggregate-product statistics:

```text
product_mean = mean(v_rot * X)
mean_product = mean(v_rot) * mean(X)
covariance = product_mean - mean_product
```

where `X` is `theta_30_180d` for heat stirring or QG-like `q_prime` for PV
stirring. The mean-product result is only a comparison baseline, not the primary
transport estimate.
