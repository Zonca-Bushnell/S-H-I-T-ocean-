# Origin Eddy Detection

This directory is a focused extraction of the original `Zhe` eddy-identification
chain from source CMEMS fields through final 3-D shape classification.

It is intentionally narrower than `Zhe`: post-processing, representative-vortex
composites, EP diagnostics, vendor code, and legacy experiments are not part of
this extraction.

## Scope

The preserved chain is:

```text
raw / filtered CMEMS NetCDF
  -> Hua hybrid detection
  -> feature / group tracking
  -> strict-contiguous catalog adapter
  -> layer-center completion
  -> life30 shape classification
```

The scientific contract follows the original `Zhe` production mouthful through
shape:

```text
Hua b3_start2
+ 30-180 day bandpass velocity
+ boundary-monotonic circular-boundary rotation constraint
+ strict-contiguous vertical extension
+ life30 shape classification
```

`coherent_only`, `ME_LIUTEX`, representative-vortex composites, and transport
diagnostics start after shape and are deliberately excluded here.

## Recommended Entry

Run from this directory or use the launcher from the worktree root:

```powershell
python -m src.eddy_pipeline.cli --help
python run_origin_eddy_pipeline.py --help
python run_origin_eddy_pipeline.py run-detection-to-shape --dry-run
```

Local defaults in this extraction point to:

```text
raw root:    F:\Global Ocean Ensemble Physics Reanalysis Kuroshio Current
filter root: F:\Global Ocean Ensemble Physics Reanalysis Kuroshio Current\FILTER
output root: E:\DATA\01_Eddy_correspond\03_Original_detection
```

The local filter files are expected to be named:

```text
FILTER/global_phy_1993.nc
FILTER/global_phy_1994.nc
...
```

The supported command for this extraction is:

```powershell
python run_origin_eddy_pipeline.py run-detection-to-shape `
  --filter-root <FilterRoot> `
  --raw-root <RawRoot> `
  --climatology <ClimatologyNc> `
  --output-root <OutputRoot> `
  --start <YYYY-MM-DD> `
  --end <YYYY-MM-DD>
```

The launcher keeps the original internal module names (`src.eddy_pipeline.*`)
working without requiring files outside this folder.

Current production defaults are:

```text
candidate_selection = tile_topn
tile_lon_deg = 10
tile_lat_deg = 10
tile_top_n = 15
max_depth_m = 0        # use every depth level in the source NetCDF file
detect_parallel = 12   # year shards run concurrently when multiple years exist
```

## Acceleration Notes

This extraction keeps the Python scientific checks as the authority, but adds
local speed-oriented changes:

- Detection sharding defaults to `--detect-shard-mode year`. This keeps one
  process scanning one yearly NetCDF file sequentially, reducing repeated file
  opens and HDF5 chunk-cache churn compared with quarterly sharding.
- Surface seed selection defaults to `10 deg x 10 deg` spatial tiles with at
  most 15 candidates per tile. This avoids a global top-N bottleneck while
  preserving geographically distributed candidates.
- `--max-depth-m 0` means all source depth levels. For the current local
  `global_phy_1993.nc` files this is 51 levels down to about 1516 m.
- `--detect-parallel 12` is the default launcher concurrency. With year
  sharding, a 30-year run can execute multiple annual workers at once.
- `--netcdf-chunk-cache-mb 512` is passed to detection workers by default, so
  repeated daily reads can reuse NetCDF chunks inside each yearly worker.
- Detection and tracking skip nonessential QA plots in the default launcher
  path. This avoids spending smoke-test time in matplotlib while keeping table
  products unchanged.
- The subgrid quadratic velocity-center fit avoids `numpy.linalg`/BLAS for its
  tiny 6x6 least-squares system and uses an explicit small linear solve. This is
  a Windows stability fix for this local environment; it does not change the
  fitted model or Hua acceptance criteria.

There is also an optional MATLAB backend, isolated under `matlab/`:

```powershell
python run_origin_eddy_pipeline.py run-detection-to-shape `
  --candidate-selection global_topn `
  --max-candidates-per-day 80 `
  --matlab-candidate-precompute `
  --matlab-use-gpu `
  --output-root E:\DATA\01_Eddy_correspond\03_Original_detection\run_full
```

The MATLAB backend currently precomputes only global top-N surface SSH extrema
candidates from `zos_glor` and writes `candidates_YYYYMMDD.csv` files. It does
not yet write tile-topN caches. Python still owns the Hua velocity-center
search, circular checks, strict-contiguous extension, tracking, catalog, and
shape classification. This keeps the physical decision chain unchanged while
moving the easiest matrix stage toward MATLAB/GPU.

Benchmark helper:

```powershell
$env:PYTHONNOUSERSITE='1'
& 'D:\Util\lever\02_miniforge\envs\eddy_detection\python.exe' `
  tools\benchmark_origin_chain.py `
  --output-root E:\DATA\01_Eddy_correspond\03_Original_detection\benchmarks `
  --start 1993-01-01 `
  --end 1993-01-03 `
  --matlab-use-gpu
```

The benchmark writes `benchmark_summary.json` plus logs under the requested
output root. If dependencies are missing, it records the missing modules instead
of attempting to install anything.

The benchmark precomputes MATLAB candidates once, then times Python with and
without that existing candidate cache. On very short smoke windows, the
no-cache path can legitimately fail at shape eligibility if no feature track
survives long enough; the cached path is the intended MATLAB-accelerated
detection-to-shape smoke.

For robustness, catalog/shape now treats "no eligible shape tracks" as a valid
empty product state. It writes empty `eligible_tracks`, `shape_daily_metrics`,
and `shape_tracks` tables plus thresholds metadata instead of failing the whole
pipeline. This matters for short smoke windows and strict life/radius filters.

## Included Files

Core package:

```text
src/
  __init__.py
  eddy_pipeline/
    __init__.py
    cli.py
    contracts.py
    detection.py
    detection_hybrid.py
    tracking.py
    catalog.py
    completion.py
    shape.py
    utils/
      __init__.py
      common.py
      streaming_cmems.py
      table_io.py
      velocity3d_core.py
```

Configuration and provenance:

```text
config/config_kuroshio_results_cpu.yaml
docs/Zhe_README_original.md
docs/Zhe_CODEBASE_LAYOUT_original.md
docs/Zhe_main_pipeline_contract_original.md
docs/performance_validation_20260909.md
docs/source_manifest.json
```

## Chain Stages

| Stage | Module | Role |
| --- | --- | --- |
| Detection | `src.eddy_pipeline.detection_hybrid` | Hua SSH-extrema seed, speed-minimum center, circular velocity checks, optional boundary-monotonic check, strict vertical continuation. |
| Tracking | `src.eddy_pipeline.tracking` | Links detected feature/group objects through time. |
| Catalog + Shape | `src.eddy_pipeline.catalog` | Adapts detections/tracks into catalog tables and calls shape classification. |
| Completion | `src.eddy_pipeline.completion` | Completes speed-leading layer centers needed by shape diagnostics. |
| Shape | `src.eddy_pipeline.shape` | Builds life-thresholded 3-D shape classes. |

## Important Differences From Full `Zhe`

- No `src.post`, `src.EP`, `src.Legacy`, or `vendor` code is copied.
- No OFES-specific `.ctl/.dta` reader or native/rebuild-W diagnostics are copied.
- Representative-vortex and transport stages are excluded even though the
  original CLI can describe them.
- The extracted package is meant to preserve the original detection-to-shape
  behavior before any OFES-specific changes.

## Dependencies

The original chain expects a scientific Python environment with at least:

```text
numpy
pandas
scipy
netCDF4
matplotlib
pyyaml
tqdm
fastparquet
pyarrow
contourpy
```

Do not install anything into `base`. If dependencies need to be created or
updated, use `mamba` in a dedicated environment.
