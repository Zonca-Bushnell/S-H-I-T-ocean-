# Detection for OFES2

`Detection_for_OFES` is the self-contained OFES2 workspace used for the
January 1991 diagnostic eddy work. It should run without importing `Zhe/src`,
without calling the `Dipole_vertical _transport` MATLAB backend, and without
depending on one-off scripts outside this package.

Current priorities:

- OFES2 `.ctl/.dta` metadata and binary I/O.
- Hua hybrid eddy detection on OFES fields.
- Zhe-style panel-family diagnostics for selected OFES eddies.
- OFES native `w` versus rebuilt `W` diagnostics.
- Refactoring toward object-oriented components with explicit scientific
  assumptions.

All Python commands should be run in the dedicated `OFES_detection`
environment. Do not install packages into `base`.

```powershell
$env:PYTHONNOUSERSITE='1'
& 'D:\Util\lever\02_miniforge\envs\OFES_detection\python.exe' -m Detection_for_OFES.run_hua_ofes2 --help
```

Dependencies are managed with `mamba`, not `pip`. The current environment file
is `Detection_for_OFES/environment.yml`.

## Data Roots

The active OFES2 data root on this machine is:

```text
F:\OFES\external_OFES2
```

The analysis output roots currently used are:

```text
E:\DATA\01_Eddy_correspond\02_OFES
E:\DATA\01_Eddy_correspond\02_OFES\available_jan01_jan19_refined_ofes_grid
E:\DATA\01_Eddy_correspond\02_OFES\available_jan01_jan19_refined_ofes_grid_tile10_top10
```

The old path spelling `F:\OFES\external\_OFES2` is treated as a historical or
mistyped path unless it is explicitly reintroduced on another machine.

## Expected OFES2 Layout

The hot path is daily-only. Production diagnostics should read extracted
`.dta` files and should not silently fall back to `.tgz` archives.

```text
external_OFES2/
  metadata/
    ctl/
      eta.ctl
      pressur.ctl
      u.ctl
      v.ctl
      w.ctl
      temp.ctl
      salinity.ctl
      prho.ctl
    manifests/
      *.json
      *.csv
  compressed/
    eta/1991/*.tgz
    pressur/1991/*.tgz
    u/1991/*.tgz
    v/1991/*.tgz
    w/1991/*.tgz
    temp/1991/*.tgz
    salinity/1991/*.tgz
    prho/1991/*.tgz
  daily/
    y1991/
      eta/eta.MM.DD.YYYY.dta
      pressur/pressur.MM.DD.YYYY.dta
      u/u.MM.DD.YYYY.dta
      v/v.MM.DD.YYYY.dta
      w/w.MM.DD.YYYY.dta
      prho/prho.MM.DD.YYYY.dta
  docs/
    SSH说明.txt
    read_data.m
```

`organize_ofes2.py` has been retired. The current repository assumes the data
tree has already been organized. Future data migration should be handled by a
separate, explicitly named maintenance tool rather than by the analysis
package.

## Core Definitions

### Binary Format

OFES `.dta` files are read as big-endian `float32`. Values with absolute
magnitude larger than the configured missing-value threshold are treated as
missing and converted to `NaN`.

### Grids

OFES variables use staggered horizontal grids. The reader keeps each
variable's `.ctl` metadata instead of silently assuming a common grid.

Important assumption for refactoring: every object that mixes variables must
carry the source grid and the target grid explicitly. `u/v/w` alignment to the
scalar grid must be a named operation, not an invisible side effect.

### SSH

The SSH field used by the OFES Matlab example is:

```text
SSH = eta - (pressur - 1000)
```

The unit is centimeters.

### Anomaly Convention

Current OFES v1 detection uses:

```text
raw_minus_jan_mean_diagnostic
```

That means each daily field is compared with the January 1991 monthly mean.
This is not the same scientific product as the previous 30-180 day bandpass
workflow. Any output using this convention must keep the diagnostic label in
metadata.

### Hua Hybrid Detection

The OFES Hua runner keeps the main Hua-style checks:

- SSH anomaly extrema as surface seeds.
- Weak-speed center search.
- Circular velocity verification.
- Boundary-monotonic rotation.
- Strict contiguous vertical extension.
- Optional local subgrid center refinement on the OFES grid.

The old global top-N surface seed limit is retained for compatibility, but it
is too restrictive for global OFES:

```text
--candidate-selection global_topn --max-candidates-per-day 80
```

The preferred global diagnostic mode is now tile top-N:

```text
--candidate-selection tile_topn --tile-lon-deg 10 --tile-lat-deg 10 --tile-top-n 10
```

This keeps the strongest candidates inside each geographic tile and avoids
letting a few energetic regions consume the whole daily candidate budget.

### Native W

OFES provides native model `w`. In this package it is used as a reference
field, not as an input to Hua detection.

Current W assumptions:

- Native `w` is converted to `m/s` before comparison.
- Native W vertical alignment is fixed to layer-center comparison. The runner
  interpolates native W from its native vertical placement to the layer-center
  depth used by the rebuilt field.
- Native W can be temporally smoothed with
  `--native-w-temporal-filter lowpass_running_mean`.
- The default diagnostic lowpass window is 10 days. This is a running-mean
  scale-separation tool, not a strict frequency-domain filter.

### Rebuilt W

The current rebuild formula uses relative advection:

```text
rho_bg(z) = background density profile
rho'(x,y,z) = rho(x,y,z) - rho_bg(z)

eta_rho(x,y,z) = -rho'(x,y,z) / rho_z(z)
rho_z(z) = d rho_bg / dz

U_rel(x,y,z) = U(x,y,z) - U_bg(z)
c_rel(z) = c_abs(z) - U_bg(z)

W_rebuild =
  c_rel(z) · grad_h eta_rho
  - U_rel(x,y,z) · grad_h eta_rho
```

The old `legacy_double_c` formula is kept only for comparison and should not
be used as the default interpretation.

### Density Background

Two background density modes exist:

```text
--rho-bg-source farfield_ring
--rho-bg-source regional_box
```

The current preferred mode is `regional_box`.

`farfield_ring` uses the local `2R-4R` ring around each eddy. It is useful as a
legacy comparison but can be unstable near shelves, fronts, or sparse valid
water points.

`regional_box` uses a box centered on the surface refined eddy center:

```text
lon0 +/- 5 deg
lat0 +/- 2 deg
excluding r < 1.5R
```

Each layer uses the regional ocean-point median as `rho_bg_raw(z)`, then a
NaN-aware vertical Gaussian smoothing step builds `rho_bg(z)`.

### Rho-Z Stabilization

Rebuilt W is very sensitive to weak or sign-changing stratification. The
current default stabilizes isopycnal displacement by:

```text
rho_bg = smooth_z(rho_bg_raw)
rho_z = d rho_bg / dz

if |rho_z| < 2e-5:
    eta_rho = NaN

eta_rho = clip(eta_rho, -500 m, +500 m)
```

Compatibility flag:

```text
--disable-rho-stabilization
```

This should be used only to reproduce older diagnostic failures.

### Scale Separation

Simple band-stop filtering is not the preferred method here because the
current diagnostic window is only 19 days. The package instead supports
running-mean and spatial lowpass scale separation.

Density path:

```text
--rebuild-density-filter none
--rebuild-density-filter joint_lowpass
--rebuild-density-filter bg_perturb_decomp
```

`joint_lowpass` applies a 10-day running mean to `prho` and then applies an
R-scale horizontal Gaussian lowpass to `eta_rho` before computing gradients.

`bg_perturb_decomp` additionally removes a per-layer robust planar density
trend fitted outside the eddy core. This is intended to reduce contamination
from large-scale fronts being misread as eddy isopycnal displacement.

Velocity path:

```text
--rebuild-velocity-filter none
--rebuild-velocity-filter joint_lowpass
```

`joint_lowpass` applies a 10-day running mean to `u/v`, then an R-scale
horizontal Gaussian lowpass before computing `U_bg` and `U_rel`.

### Translation Speed c(z)

The current default is true layerwise center tracking:

```text
--translation-profile layer_tracking
```

For each depth layer, the selected eddy center is matched to the nearest
same-depth center on neighboring days. The displacement gives `c_abs(z)`. Bad
or unrealistically fast layer matches are masked, and the profile can be
vertically smoothed:

```text
--translation-profile-smooth-sigma-layers 2.0
--translation-profile-max-speed-m-s 0.5
```

Older options remain for comparison:

```text
--translation-profile barotropic
--translation-profile layerwise
```

These are not the preferred W-rebuild interpretation.

## Main Entrypoints

### `ofes_io.py`

Core package infrastructure. Keep and refactor carefully.

Current responsibilities:

- Parse `.ctl` files.
- Resolve expected daily file paths.
- Read 2-D and 3-D `.dta` fields.
- Read SSH from `eta/pressur`.
- Open memmaps for large 3-D variables.
- Provide limited extraction/cache helpers.

Refactor suggestion: convert this into object-oriented data access classes:

- `CtlMetadata`
- `OfesVariable`
- `OfesDataset`
- `OfesGrid`
- `DailyFieldReader`

### `validate_ofes2_sample.py`

Smoke-test utility. It validates basic sample reading and SSH construction.

Status: keep, but move later to `tests/smoke/` or `tools/`.

### `run_hua_ofes2.py`

Main Hua detection runner.

Current stages:

```text
metadata
extract
means
detect
```

Current status:

- Works as the primary OFES Hua detection CLI.
- Supports old global top-N and new tile top-N candidate selection.
- Writes detection tables, daily part files, previews, and summary metadata.
- Still combines CLI parsing, anomaly building, candidate generation, Hua
  checks, summaries, and plots in one file.

Recommended object-oriented split:

- `HuaRunConfig`
- `MonthlyMeanBuilder`
- `SurfaceCandidateSelector`
- `HuaHybridDetector`
- `DetectionTableWriter`
- `DetectionSummary`

Compatibility items to isolate:

- `global_topn` and `--max-candidates-per-day`.
- `depth_chunk` mean strategy.
- Resume/force day-part handling.
- Subgrid refinement toggles.

### `plot_ofes_panel_family.py`

Zhe-style OFES panel-family renderer.

Current status:

- Self-contained.
- Does not import `Zhe/src`.
- Supports matplotlib and Pillow backends.
- Recreates the major panel-family visual language: 5x6 layout, compact panel
  labels, colorbars, contours, quivers, center trajectory, and right-panel
  modes.

Recommended object-oriented split:

- `PanelObjectSelector`
- `PanelDataBuilder`
- `PanelFamilyFigure`
- `MatplotlibPanelRenderer`
- `PillowPanelRenderer`

Compatibility items:

- Pillow backend is an emergency fallback for unstable matplotlib stacks.
- Panel 7 is currently object-day vertical center trajectory, not full
  lifecycle trajectory.

### `run_ofes_rebuild_w.py`

Main native-W versus rebuild-W diagnostic runner.

Current stages:

```text
extract
diagnose
crossing
band
all
```

Current status:

- Scientifically active and changing.
- Uses immutable config dataclasses from `w_rebuild_config.py` for I/O,
  rebuild, filter, selection, and plotting settings.
- Dispatches W/prho extraction through `tools/extract_ofes_daily.py`.
- Uses one diagnostic dispatcher for single-day object diagnostics, strict
  latitude crossing diagnostics, and latitude-band `match_all` diagnostics.
- The old global structured eddy `map` stage has been removed from this runner.
- Produces 2x2 W four-panel figures, horizontal sections, slice sections,
  grids, JSON metadata, and CSV/JSON summaries.
- Includes current best rebuild-W pathway: regional density background,
  rho-z stabilization, layerwise center tracking, fixed native W layer-center
  alignment, density scale separation, velocity scale separation, and smoothed
  `c(z)`.
- `all` now means extraction plus the default latitude-band diagnostic, so it
  does not create multiple overlapping horizontal/slice figure families.

Recommended current scientific default:

```text
--rho-bg-source regional_box
--native-w-temporal-filter lowpass_running_mean
--rebuild-formula relative_advection
--velocity-reference farfield_relative
--translation-profile layer_tracking
--rebuild-density-filter joint_lowpass
--rebuild-velocity-filter joint_lowpass
--translation-profile-smooth-sigma-layers 2.0
```

The stronger density decomposition mode is available:

```text
--rebuild-density-filter bg_perturb_decomp
```

but should be treated as an experiment until its effect is compared across
more eddies.

Compatibility and legacy items to isolate:

- `--disable-rho-stabilization`
- `--rho-bg-source farfield_ring`
- `--rebuild-formula legacy_double_c`
- `--velocity-reference absolute`
- `--translation-profile barotropic`
- `--translation-profile layerwise`
- `--native-w-temporal-filter none`
- `--rebuild-density-filter none`
- `--rebuild-velocity-filter none`

Current refactor split:

- `IOConfig`, `RebuildConfig`, `FilterConfig`, `SelectionConfig`, and
  `PlotConfig` in `w_rebuild_config.py`.
- W/prho extraction in `tools/extract_ofes_daily.py`.

Further object-oriented split still recommended:

- `EddyObjectSelector`
- `CrossingSelector`
- `BandSelector`
- `DensityBackgroundEstimator`
- `IsopycnalDisplacementEstimator`
- `VelocityScaleSeparator`
- `LayerwiseTranslationTracker`
- `NativeWAligner`
- `RebuildWCalculator`
- `WDiagnosticGrid`
- `WFourPanelRenderer`
- `WRunSummaryWriter`

The core `rebuild_object_w()` calculation is intentionally left in place for
now. It should be split only after the scientific behavior is frozen.

## Output Layout

Detection output:

```text
<result-root>/
  metadata/
  anomaly_jan1991/
  detection_hua_global_jan1991/
  figures/
  logs/
```

W diagnostic output:

```text
<result-root>/
  w_rebuild_diagnostics/
    metadata/
    grids/
    figures/
      band_30N_35N_sections/
      band_30N_35N_horizontal/
    *.csv
    *.json
```

Panel-family output:

```text
<result-root>/
  figures/
    panel_family/
```

## Current Refactor Rules

The next refactor should be object-oriented.

Recommended principles:

- Keep CLIs thin. A CLI should parse arguments, build a config object, call a
  runner, and write high-level status.
- Put physical assumptions into named classes or config fields.
- Do not let plotting functions read raw OFES data directly unless the class
  name makes that responsibility explicit.
- Do not hide grid interpolation or vertical alignment.
- Keep legacy modes runnable but isolate them behind compatibility classes or
  explicitly named methods.
- Prefer immutable config dataclasses for reproducibility.
- Keep output metadata rich enough to reconstruct the exact scientific path.

## Known Caveats

- The current Jan 1-19 window is short. Ten-day running means are diagnostic
  lowpass filters, not formal spectral filters.
- The W-rebuild relationship is sensitive to `rho_z`, density background
  choice, velocity scale separation, and layerwise center tracking quality.
- OFES native `w` grid placement and unit interpretation should stay visible
  in metadata for every W comparison.
- Hua detection output is still diagnostic and should not yet be treated as a
  final global eddy census.
- `plot_ofes_panel_family.py` and `run_ofes_rebuild_w.py` are intentionally
  self-contained today, but both need splitting before long-term use.
