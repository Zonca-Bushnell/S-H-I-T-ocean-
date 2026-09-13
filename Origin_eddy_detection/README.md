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
- Velocity weak-center search now builds the circular search mask only on the
  local `seed +/- radius` window instead of on the full horizontal grid for
  every candidate. This preserves the same search radius and iterative descent
  result while reducing that step from domain-sized work to radius-sized work.
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

The MATLAB backend currently has two layers:

- `origin_precompute_candidates.m` precomputes global top-N surface SSH extrema
  candidates from `zos_glor` and writes `candidates_YYYYMMDD.csv` files. It does
  not yet write tile-topN caches.
- `hua_circle_batch_kernel.m` and `velocity_streamline_boundary_kernel.m` are
  one-layer matrix kernels for the fixed-circle and closed-streamline Hua
  boundary checks. They are kept under `matlab/` for acceleration validation
  against the Python reference diagnostics.

Python still owns NetCDF IO, vertical continuation, strict-contiguous stopping,
parquet/csv/json outputs, tracking, catalog, and shape classification. This
keeps the physical decision chain unchanged while allowing the Hua boundary
check itself to move to MATLAB/GPU. The detection CLI accepts
`--hua-backend python|matlab`; `matlab` starts a MATLAB Engine session in each
detection worker, loads each day of `u/v` into MATLAB once, and calls the
matching kernel once per depth layer with all currently active centers. The
same layer batch also runs the pre-Hua subgrid speed-minimum refinement through
`matlab/refine_speed_min_batch_kernel.m`, then sends the refined float centers
into the Hua kernel. This reduces Python-to-MATLAB calls from per-center to
per-layer and gives both the refinement and streamline boundary stages enough
batch width for vectorized CPU/GPU execution.
For MATLAB-backed production runs, this is now lifted to a day-level kernel:
`matlab/hua_day_batch_kernel.m` receives the whole daily `u/v` cube plus surface
seed centers once, advances active centers through depth, applies local
weak-center search, runs pre-Hua refinement, performs the Hua boundary check,
and returns one row per attempted candidate-layer. Python still owns source IO,
SSH candidate selection, table assembly, tracking, catalog, and shape.
MATLAB-to-Python day rows now use a temporary `.mat` bridge written by
`matlab/hua_day_batch_to_mat.m`; Python reads it with `scipy.io.loadmat` and
deletes it by default. Use `--keep-matlab-bridge-files` only for debugging a
specific day.
Within each layer, exact duplicate integer speed-minimum anchors are refined
once and exact duplicate refined centers are checked once, then the cached
result is mapped back to every active object that shared the center. This is an
identity-preserving reduction in effective center count; it does not change the
Hua criteria or accepted radius. The output rows record
`matlab_batch_size`, `matlab_unique_refine_batch_size`,
`matlab_unique_hua_batch_size`, `matlab_refine_dedup_count`, and
`matlab_hua_dedup_count`.
When `--matlab-use-gpu` is enabled, each worker stages the whole daily `u/v`
cube on the GPU once, then reuses those arrays across depth-layer boundary
checks.

MATLAB Hua parity is defined against the Python implementation at the
per-radius, per-boundary-point, per-metric level. Both kernels must follow the
same radius semantics as Python: test radii in order, keep the last passing
radius, and stop at the first failing radius. Pass-rate statistics use the first
hard failed gate in decision order:

```text
finite -> velocity_ratio -> angle_jump -> boundary_monotonic -> tangent -> opposite_reversal
```

For `velocity_streamline_contour`, `no_closed_streamline` is recorded before the
finite gate when no acceptable closed streamline exists for the current radius.
`dominant_failure` remains a diagnostic largest-count label and can differ from
the first hard failure used in rejection summaries. Large MATLAB-backed TEST
runs should only be launched after same-day/same-layer parity confirms matching
`hua_pass` and accepted-radius behavior. The `eddy_detection` environment must
have MATLAB Engine for Python installed.

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

## ACC Velocity-Streamline Boundary Test

The original ACC Hua baseline is now fixed as:

```text
boundary_mode = circle_strict_original
SSH seed -> local velocity-minimum center -> fixed-radius circular Hua check
```

An experimental ACC-only diagnostic mode is available:

```text
boundary_mode = velocity_streamline_contour
SSH seed -> local velocity-minimum center -> closed velocity-streamline contour check
```

For OFES and other energetic western-boundary-current cases, use the consensus
catalog mode when a streamline-only catalog would confuse jet meanders with
isolated eddy interiors:

```text
boundary_mode = velocity_streamline_ssh_consensus
SSH seed -> local velocity-minimum center -> closed velocity-streamline diagnostic
         -> SSH/effective-contour consensus
         -> fast-current-core overlap diagnostic
         -> fallback fixed-circle + SSH consensus for no_closed_streamline seeds
```

In this mode, `velocity_streamline_contour` is treated as a strong dynamical-core
diagnostic rather than the only eddy definition. Surface rows add
`ssh_contour_closed`, `ssh_contour_radius_cells`, `ssh_contour_area_cells`,
`ssh_consensus_pass`, `jet_core_overlap_fraction`, `jet_meander_flag`,
`boundary_source`, and `catalog_acceptance_reason`. A row accepted by a closed
streamline plus SSH consensus is marked `boundary_source=pure_streamline_ssh`.
A weak-velocity but SSH-coherent fallback object is marked
`boundary_source=fallback_circle_ssh`. Jet-axis/meander-like candidates are
kept in diagnostics but rejected with `jet_core_overlap`.

For OFES/META-like surface catalogs, the preferred diagnostic mode is:

```text
boundary_mode = ssh_effective_contour_primary
SSH seed -> SSH anomaly closed effective-contour interior
         -> contour centroid, area, and equivalent radius define the surface object
         -> velocity streamline is sampled only as a dynamical-core diagnostic
         -> deeper layers continue with the existing Hua velocity checks
```

In this mode, a missing closed velocity streamline does not reject a surface SSH
eddy. It is written as `dynamical_core_class=no_streamline_core` or
`weak_dynamical_core`; a closed dynamical core is written as
`closed_streamline_core`. Jet-axis overlap is recorded as `jet_meander_flag`
rather than silently deleting the SSH object. Surface rows add
`surface_definition=ssh_effective_contour_primary`,
`ssh_contour_center_i/j`, `ssh_contour_center_lon/lat`,
`ssh_contour_level`, and sampled `ssh_contour_boundary_i/j` for overview maps.

The seed and vertical-extension chain is unchanged: surface seeds still come
from filtered `zos_glor`, surface centers still come from local
`sqrt(u^2+v^2)` minima near each SSH seed, and deeper centers are still searched
near the previous accepted layer. The change is only the boundary used for
`boundary_monotonic_rotation` and `tangent_alignment`: the experimental mode
tests them on the closed velocity streamline rather than on fixed circular
points.

Center refinement order is now explicit:

```text
center_refinement_stage = pre_hua
SSH seed -> integer speed minimum -> subgrid refined speed minimum -> Hua check
```

Earlier diagnostics used `post_hua`, where subgrid refinement was only written
after a layer had already passed Hua. That legacy branch has been removed from
the production TEST path. `pre_hua` is now the only supported order: the refined
continuous center is the actual center used by both `circle_strict_original`
and `velocity_streamline_contour`.

Recommended TEST smoke commands:

```powershell
$env:PYTHONNOUSERSITE='1'
& 'D:\Util\lever\02_miniforge\condabin\mamba.bat' run -n eddy_detection python `
  -m src.eddy_pipeline.detection_hybrid `
  --filter-root 'E:\DATA\01_Eddy_correspond\03_Original_detection\ACC\FILTER' `
  --raw-root 'F:\Global Ocean Ensemble Physics Reanalysis  ACC' `
  --filter-template 'global_phy_{year}_bandpass_30_180d.nc' `
  --raw-template 'global_phy_{year}.nc' `
  --output-dir 'E:\DATA\01_Eddy_correspond\03_Original_detection\TEST\baseline_circle_strict_original_smoke' `
  --start 2018-01-01 --end 2018-01-01 `
  --candidate-selection tile_topn --tile-lon-deg 10 --tile-lat-deg 10 --tile-top-n 15 `
  --surface-search-cells 3 --deep-search-cells 3 --start-radius-cells 2 --max-radius-cells 8 `
  --speed-ratio-max 3 --angle-jump-max-deg 150 `
  --tangent-tolerance-deg 24 --min-tangent-fraction 0.55 `
  --symmetry-tolerance-deg 120 --min-reversal-fraction 0.55 --min-finite-fraction 0.75 `
  --direction-exception-extra 2 `
  --boundary-mode circle_strict_original `
  --hua-backend matlab `
  --require-boundary-monotonic-rotation --boundary-monotonic-exception-limit 0 `
  --stop-at-first-failed-layer --preload-day-uv --resume --skip-axis-examples

& 'D:\Util\lever\02_miniforge\condabin\mamba.bat' run -n eddy_detection python `
  -m src.eddy_pipeline.detection_hybrid `
  --filter-root 'E:\DATA\01_Eddy_correspond\03_Original_detection\ACC\FILTER' `
  --raw-root 'F:\Global Ocean Ensemble Physics Reanalysis  ACC' `
  --filter-template 'global_phy_{year}_bandpass_30_180d.nc' `
  --raw-template 'global_phy_{year}.nc' `
  --output-dir 'E:\DATA\01_Eddy_correspond\03_Original_detection\TEST\velocity_streamline_contour_smoke' `
  --start 2018-01-01 --end 2018-01-01 `
  --candidate-selection tile_topn --tile-lon-deg 10 --tile-lat-deg 10 --tile-top-n 15 `
  --surface-search-cells 3 --deep-search-cells 3 --start-radius-cells 2 --max-radius-cells 8 `
  --speed-ratio-max 3 --angle-jump-max-deg 150 `
  --tangent-tolerance-deg 24 --min-tangent-fraction 0.55 `
  --symmetry-tolerance-deg 120 --min-reversal-fraction 0.55 --min-finite-fraction 0.75 `
  --direction-exception-extra 2 `
  --boundary-mode velocity_streamline_contour `
  --hua-backend matlab `
  --baseline-summary-path 'E:\DATA\01_Eddy_correspond\03_Original_detection\TEST\baseline_circle_strict_original_smoke\run_summary.json' `
  --streamline-direction-exception-fraction 0.10 `
  --stop-at-first-failed-layer --preload-day-uv --resume --skip-axis-examples
```

OFES one-day consensus smoke:

```powershell
$env:PYTHONNOUSERSITE='1'
Set-Location 'D:\01_Eddy\01_Vertical_asymmetric\S-H-I-T-ocean-\Origin_eddy_detection'
& 'D:\Util\lever\02_miniforge\condabin\mamba.bat' run -n OFES_detection python `
  -m src.eddy_pipeline.detection_hybrid `
  --filter-root 'E:\DATA\01_Eddy_correspond\02_OFES\origin_compatible_filter\daily_parts' `
  --raw-root 'E:\DATA\01_Eddy_correspond\02_OFES\origin_compatible_filter\daily_parts' `
  --filter-template 'global_phy_{yyyymmdd}.nc' `
  --raw-template 'global_phy_{yyyymmdd}.nc' `
  --output-dir 'E:\DATA\01_Eddy_correspond\02_OFES\origin_streamline_ssh_consensus_smoke_19910101' `
  --start 1991-01-01 --end 1991-01-01 --max-depth-m 1 `
  --candidate-selection tile_topn --tile-lon-deg 10 --tile-lat-deg 10 --tile-top-n 10 `
  --surface-search-cells 8 --deep-search-cells 6 --start-radius-cells 3 --max-radius-cells 8 `
  --speed-ratio-max 3 --angle-jump-max-deg 150 `
  --tangent-tolerance-deg 24 --min-tangent-fraction 0.70 `
  --symmetry-tolerance-deg 120 --min-reversal-fraction 0.70 --min-finite-fraction 0.95 `
  --boundary-mode velocity_streamline_ssh_consensus `
  --ssh-consensus-min-finite-fraction 0.70 `
  --jet-core-speed-percentile 80 --jet-core-overlap-max 0.50 `
  --hua-backend python --preload-day-uv --write-day-figures --resume --skip-axis-examples
```

The smoke writes `figures\kuroshio_seed_fate_ssh_consensus_19910101.png`,
which overlays pure streamline, fallback, jet-overlap rejected, and other
rejected seeds on the Kuroshio velocity-speed background. Keep this output in a
separate directory from `origin_streamline_cpu_jan01_jan19_life1`; it is a
catalog-definition smoke, not a replacement for the previous full run.

OFES SSH-primary one-day smoke:

```powershell
$env:PYTHONNOUSERSITE='1'
Set-Location 'D:\01_Eddy\01_Vertical_asymmetric\S-H-I-T-ocean-\Origin_eddy_detection'
& 'D:\Util\lever\02_miniforge\condabin\mamba.bat' run -n OFES_detection python `
  -m src.eddy_pipeline.detection_hybrid `
  --filter-root 'E:\DATA\01_Eddy_correspond\02_OFES\origin_compatible_filter' `
  --raw-root 'E:\DATA\01_Eddy_correspond\02_OFES\origin_compatible_filter' `
  --filter-template 'global_phy_{yyyymmdd}.nc' `
  --raw-template 'global_phy_{yyyymmdd}.nc' `
  --output-dir 'E:\DATA\01_Eddy_correspond\02_OFES\origin_ssh_effective_contour_primary_smoke_19910101' `
  --start 1991-01-01 --end 1991-01-01 --max-depth-m 3 `
  --candidate-selection tile_topn --tile-lon-deg 10 --tile-lat-deg 10 --tile-top-n 10 `
  --surface-search-cells 8 --deep-search-cells 6 --start-radius-cells 3 --max-radius-cells 8 `
  --boundary-mode ssh_effective_contour_primary `
  --ssh-primary-level-count 16 --ssh-primary-window-factor 4 --ssh-primary-max-radius-factor 2 `
  --ssh-consensus-min-finite-fraction 0.70 `
  --jet-core-speed-percentile 80 --jet-core-overlap-max 0.50 `
  --hua-backend python --preload-day-uv --write-day-figures --resume --skip-axis-examples
```

The SSH-primary smoke is expected to return more surface objects than the
strict `velocity_streamline_ssh_consensus` gate, because velocity-streamline
closure is diagnostic rather than required.

Both modes write `pass_rate_breakdown.csv/json` with surface/all-layer pass
rates and rejection fractions. The streamline mode also writes
`sensitivity_matrix.csv/parquet` for:

```text
min_tangent_fraction = 0.50,0.60,0.70
tangent_tolerance_deg = 24,30,36,45
streamline_direction_exception_fraction = 0.05,0.10,0.15
```

If `--baseline-summary-path` points to the fixed-circle baseline
`run_summary.json`, `sensitivity_matrix.csv` also includes
`pass_fraction_delta_vs_baseline` and `pass_rows_delta_vs_baseline`.

For the full ACC TEST experiment, use the same commands with
`--start 2018-01-01 --end 2019-12-31` and output roots under:

```text
E:\DATA\01_Eddy_correspond\03_Original_detection\TEST
```

This TEST path intentionally stops at layer-level Hua pass rows. It does not
run feature tracking, catalog adaptation, shape classification, or panel-family
diagnostics.

For the full ACC TEST chain through shape classification with the MATLAB Hua
kernel backend, use `run-detection-to-shape` instead of detect-only stages:

```powershell
$env:PYTHONNOUSERSITE='1'
& 'D:\Util\lever\02_miniforge\condabin\mamba.bat' run -n eddy_detection python `
  D:\01_Eddy\01_Vertical_asymmetric\Original_eddy_detection_worktree\Origin_eddy_detection\run_origin_eddy_pipeline.py `
  run-detection-to-shape `
  --region-name ACC `
  --project-name acc_hua_streamline_pre_hua_matlab `
  --output-root 'E:\DATA\01_Eddy_correspond\03_Original_detection\TEST\full_shape_pre_hua_matlab\velocity_streamline_contour' `
  --raw-root 'F:\Global Ocean Ensemble Physics Reanalysis  ACC' `
  --filter-root 'E:\DATA\01_Eddy_correspond\03_Original_detection\ACC\FILTER' `
  --filter-template 'global_phy_{year}_bandpass_30_180d.nc' `
  --raw-template 'global_phy_{year}.nc' `
  --start 2018-01-01 --end 2019-12-31 `
  --candidate-selection tile_topn --tile-lon-deg 10 --tile-lat-deg 10 --tile-top-n 15 `
  --boundary-mode velocity_streamline_contour `
  --hua-backend matlab `
  --detect-shard-mode year --detect-parallel 12 `
  --lifetime-min-days 30 --radius-min-m 50000 --min-valid-layers 6 `
  --netcdf-chunk-cache-mb 512
```

To force the GPU-oriented fixed-step batch streamline kernel, add:

```powershell
  --matlab-use-gpu
```

The current 256-center ACC surface benchmark confirms identical MATLAB
reference-vs-batch decisions:

```text
reference CPU: 1.38 s
batch CPU:     1.18 s
batch GPU:     5.37 s
```

For this same sample, `baseline_circle_strict_original` has 256/256
Python-vs-MATLAB parity across pass/radius/failure metrics. The
`velocity_streamline_contour` batch kernel has 256/256 parity against the
MATLAB reference, while Python-vs-MATLAB still has one edge-case streamline path
selection difference out of 256 centers. The mismatch is isolated to the
streamline contour path chosen for one center, not to the batch/GPU vectorized
stage. Treat GPU as an experiment until a larger short-window benchmark proves
wall-clock improvement; the CPU batch path is currently the safer accelerated
default, with the larger win coming from per-layer MATLAB Engine batching rather
than from the small standalone surface-layer kernel benchmark alone.

Current day-level smoke timings on `2018-01-01`,
`velocity_streamline_contour`, `max_candidates_per_day=256`,
`start_radius_cells=2`, `max_radius_cells=4`:

```text
surface layer only, previous layer-batch path: 8.79 s
surface layer only, day-batch path:           9.13 s
0-100 m, day-batch CPU:                      18.36 s
0-100 m, day-batch GPU:                      68.58 s
0-100 m, day-batch CPU + slim MAT bridge:    20.01 s
```

The 0-100 m CPU/GPU day-batch products both produced 2040 attempted
candidate-layer rows and 1838 Hua-pass rows with identical pass/radius/failure
and refined-center outputs. The surface-only timing confirms that day-batch is
not meant to help one-layer smoke tests; it mainly removes cross-layer Python
and MATLAB Engine round-trips. The GPU route is still slower for this
branch-heavy streamline workload, so the recommended accelerated path remains
MATLAB day-batch on CPU. The MAT bridge removes large Engine JSON transfers and
is now the default bridge, but on this small 2040-row smoke it is slightly
slower than the previous JSON bridge due to MATLAB save/load overhead; it is
kept for robustness on larger day outputs and for easier temporary-file
inspection.

The current `pre_hua` single-day redraws are under:

```text
E:\DATA\01_Eddy_correspond\03_Original_detection\TEST\refined_search_center_order
```

with surface overview and detect-only curve-section panel-family samples in:

```text
E:\DATA\01_Eddy_correspond\03_Original_detection\TEST\refined_search_center_order\panel_curve_and_backend_diff
```

## ACC Bandpass Stage

The original `Zhe` chain includes a legacy ACC filter builder. It is mirrored
here as:

```text
tools/build_acc_bandpass_filter.py
```

This stage should be used when the raw ACC annual NetCDF files exist but the
30-180 day `Filter` files do not. It preserves the Zhe filter definition:

```text
scipy.signal.butter(order=4, output="sos") + scipy.signal.sosfiltfilt
variables: uo_glor, vo_glor, zos_glor
output: global_phy_YYYY_bandpass_30_180d.nc
```

Local ACC defaults point to:

```text
raw ACC root: E/F local data under F:\Global Ocean Ensemble Physics Reanalysis  ACC
ACC result root: E:\DATA\01_Eddy_correspond\03_Original_detection\ACC
ACC filter root: E:\DATA\01_Eddy_correspond\03_Original_detection\ACC\Filter
```

Example filter command:

```powershell
$env:PYTHONNOUSERSITE='1'
& 'D:\Util\lever\02_miniforge\condabin\mamba.bat' run -n eddy_detection python `
  tools\build_acc_bandpass_filter.py `
  --input-root 'F:\Global Ocean Ensemble Physics Reanalysis  ACC' `
  --output-dir 'E:\DATA\01_Eddy_correspond\03_Original_detection\ACC\Filter' `
  --temp-dir 'E:\DATA\01_Eddy_correspond\03_Original_detection\ACC\_filter_work' `
  --start 2018-01-01 `
  --end 2019-12-31 `
  --variables uo_glor,vo_glor,zos_glor `
  --depth-block 4 `
  --lat-block 81 `
  --lon-block 96 `
  --workers 8 `
  --max-in-flight 8
```

Then run detection-to-shape with:

```powershell
& 'D:\Util\lever\02_miniforge\condabin\mamba.bat' run -n eddy_detection python `
  run_origin_eddy_pipeline.py run-detection-to-shape `
  --region-name ACC `
  --project-name acc_hua_b3_start2 `
  --bbox=-179,180,-65,-45 `
  --raw-root 'F:\Global Ocean Ensemble Physics Reanalysis  ACC' `
  --filter-root 'E:\DATA\01_Eddy_correspond\03_Original_detection\ACC\Filter' `
  --filter-template 'global_phy_{year}_bandpass_30_180d.nc' `
  --output-root 'E:\DATA\01_Eddy_correspond\03_Original_detection\ACC' `
  --start 2018-01-01 `
  --end 2019-12-31 `
  --candidate-selection tile_topn `
  --tile-lon-deg 10 `
  --tile-lat-deg 10 `
  --tile-top-n 15
```

## Original Eddy Panel-Family

The original object-day panel-family from `Zhe/src/post/original_eddy_panels.py`
is mirrored under `src/post/` and exposed through the main launcher:

```text
run_origin_eddy_pipeline.py plot-original-eddy-panels
```

It is a diagnostic for one original object-day, not a representative composite.
The figure checks vertical center offsets, first/second abrupt jump layers,
upper/lower velocity and pressure-proxy fields, section diagnostics, and the
selected track's surface-center lifecycle trajectory.

Use `mamba run` on Windows so the conda DLL paths for NumPy/Matplotlib are
activated correctly:

```powershell
$env:PYTHONNOUSERSITE='1'
& 'D:\Util\lever\02_miniforge\condabin\mamba.bat' run -n eddy_detection python `
  run_origin_eddy_pipeline.py plot-original-eddy-panels `
  --results-root E:\DATA\01_Eddy_correspond\03_Original_detection\ACC `
  --shape-dir-name shape_classification_2018_2019_hua_b3_start2_life30 `
  --raw-root 'F:\Global Ocean Ensemble Physics Reanalysis  ACC' `
  --filter-root E:\DATA\01_Eddy_correspond\03_Original_detection\ACC\Filter `
  --output-dir E:\DATA\01_Eddy_correspond\03_Original_detection\ACC\panel_family `
  --max-examples 1 `
  --right-panel-mode normal_horizontal_velocity
```

The right-side panels support the original modes:

```text
omega_w
normal_horizontal_velocity
horizontal_speed
signed_horizontal_speed
```

`omega_w` follows the original density-based omega diagnostic and requires
`gsw`. If `gsw` is not installed, use one of the horizontal-velocity modes or
install `gsw` in the dedicated `eddy_detection` environment before running that
mode.

When `--w-section-mode axis_curved` is used, the panel-family expands the
layout with two extra 3-D geometry panels on the far right:

```text
12  J1 axis-curved geometry
13  J2 axis-curved geometry
```

These panels show the interpolated vertical eddy-center axis, discrete layer
centers, the upper/from and lower/to jump layers, the jump segment, and the
local normal section lines used to sample the curved sections. They are visual
diagnostics only; they do not change detection, tracking, or shape results.

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
  post/
    __init__.py
    original_eddy_panels.py
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
| Original Panel-Family | `src.post.original_eddy_panels` | Recreates the Zhe original object-day discontinuity diagnostic panels. |

## Kuroshio Theory Rebuild-W Diagnostic

`tools/plot_kuroshio_theory_rebuild_w.py` is a standalone diagnostic for the
Kuroshio 30-year velocity-streamline result. It uses raw Kuroshio
temperature/salinity to estimate density with a linear equation of state, uses
the `FILTER` velocity fields for the 30-180 day dynamical velocity, derives
layerwise center propagation from the catalog, and plots:

```text
term1 = c_rel(z) · grad(eta_rho)
term2 = -u_rel(x,y,z) · grad(eta_rho)
term1 + term2
```

The diagnostic is for testing the tilted-eddy kinematic rebuild-W theory. It
does not compare against native W because the CMEMS Kuroshio source files do
not contain a native vertical velocity variable.

Example:

```powershell
python Origin_eddy_detection\tools\plot_kuroshio_theory_rebuild_w.py
```

## Important Differences From Full `Zhe`

- Only the original object-day panel-family from `src.post` is copied; other
  post-processing, `src.EP`, `src.Legacy`, and `vendor` code are excluded.
- No OFES-specific `.ctl/.dta` reader or native/rebuild-W diagnostics are copied.
- Representative-vortex and transport stages are excluded even though the
  original CLI can describe them.
- The extracted package is meant to preserve the original detection-to-shape
  behavior before any OFES-specific changes.

## Dependencies

### MATLAB day-batch voxel output fix

With `--write-object-voxels`, the MATLAB day-batch path now generates tracking
voxels for each Hua-passed layer using the same finite connected-component
helper, integer grid anchor, and accepted radius as the Python path. Previously
this branch returned before generating voxels, producing empty daily Parquet
files and preventing all cross-day overlap links.

Existing empty voxel parts are not repaired by this code change. They must be
regenerated before rerunning tracking, catalog, and shape; resume checks file
existence and therefore does not repair those files automatically.

To backfill existing empty daily files, run from the worktree root in the
`eddy_detection` environment:

```powershell
python -m Origin_eddy_detection.tools.backfill_matlab_day_voxels --detection-dir '<result-root>\hua_b3_start2_detection' --refresh-frame-summary
```

The tool reads saved passed centers and the original filtered daily velocity
fields, preserves the existing voxel geometry, writes each part atomically,
skips nonempty parts, and records daily counts in `voxel_backfill_*.json`.
It does not rerun Hua. After completion, run the main pipeline with
`--stages tracking,catalog_shape` and the original date and shape thresholds.

Regression check (in the `eddy_detection` environment, from the worktree root):

```powershell
python -m unittest discover -s Origin_eddy_detection/tests -p test_matlab_day_voxels.py
```

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
