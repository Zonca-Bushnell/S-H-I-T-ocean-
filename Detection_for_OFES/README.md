# Detection for OFES2

`Detection_for_OFES` is the OFES2 data-interface package for this repository.
It stays deliberately narrow: read OFES `.ctl/.dta` data, prepare daily files,
export Origin-compatible NetCDF, and support OFES-only native `w` versus rebuilt
`W` diagnostics.

General eddy detection, tracking, shape classification, and panel rendering are
owned by `Origin_eddy_detection`. Do not add another detection algorithm copy
here.

## Environment

Run commands in the dedicated OFES environment. Dependencies are managed with
`mamba`, not `pip`.

```powershell
$env:PYTHONNOUSERSITE='1'
& 'D:\Util\lever\02_miniforge\Library\bin\mamba.exe' run -n OFES_detection python -m Detection_for_OFES.validate_ofes2_sample --help
```

The NetCDF exporter defaults to SciPy's NetCDF3 64-bit writer. This avoids the
local Windows application-control policy that blocks `netCDF4`'s `_netCDF4`
DLL. `Origin_eddy_detection` can read the exported file through its own
`netCDF4` environment.

## Data Layout

The active OFES2 root on this machine is:

```text
F:\OFES\external_OFES2
```

The hot path is daily-only. Production diagnostics should read extracted
`.dta` files and should not silently fall back to `.tgz` archives.

```text
external_OFES2/
  metadata/ctl/
    eta.ctl
    pressur.ctl
    u.ctl
    v.ctl
    w.ctl
    temp.ctl
    salinity.ctl
    prho.ctl
  compressed/<variable>/1991/*.tgz
  daily/y1991/<variable>/<variable>.MM.DD.YYYY.dta
  docs/
```

The old path spelling `F:\OFES\external\_OFES2` is treated as historical unless
it is explicitly reintroduced on another machine.

## Interface Components

`ofes_io.py` is the core reader. It parses GrADS-style `.ctl` files, resolves
daily `.dta` paths, checks expected binary sizes, opens OFES big-endian
Fortran-order `float32` memmaps, and constructs SSH:

```text
SSH = eta - (pressur - 1000)
```

`validate_ofes2_sample.py` is the I/O smoke check for one daily sample.

`tools/extract_ofes_daily.py` contains the daily extraction helper used by OFES
diagnostics, especially `w/prho` preparation.

`tools/export_origin_netcdf.py` adapts OFES `.dta` files to the NetCDF layout
expected by `Origin_eddy_detection`.

## Export to Origin

The recommended OFES workflow starts by exporting a yearly Origin-compatible
NetCDF:

```powershell
$env:PYTHONNOUSERSITE='1'
& 'D:\Util\lever\02_miniforge\Library\bin\mamba.exe' run -n OFES_detection python -m Detection_for_OFES.tools.export_origin_netcdf `
  --data-root F:\OFES\external_OFES2 `
  --output-dir E:\DATA\01_Eddy_correspond\02_OFES\origin_compatible_filter `
  --start 1991-01-01 `
  --end 1991-01-19 `
  --mean-start 1991-01-01 `
  --mean-end 1991-01-19 `
  --max-depth-layers 105 `
  --depth-chunk 4 `
  --output-layout daily_parts `
  --writer-backend scipy_netcdf3_64bit `
  --overwrite
```

The default exporter writes daily part files instead of one huge yearly file:

```text
E:\DATA\01_Eddy_correspond\02_OFES\origin_compatible_filter\global_phy_19910101.nc
E:\DATA\01_Eddy_correspond\02_OFES\origin_compatible_filter\global_phy_19910102.nc
...
```

The NetCDF variables match the current Origin input contract:

```text
time
longitude
latitude
depth
zos_glor
uo_glor
vo_glor
```

`zos_glor` is OFES SSH anomaly in centimeters, regridded from the scalar grid to
the OFES velocity grid. `uo_glor` and `vo_glor` are OFES velocity anomalies in
`m s-1`. All three use the same `raw_minus_jan_mean_diagnostic` anomaly
convention, with the exact mean window written into NetCDF attributes and the
export manifest.

After export, run the Origin pipeline with the export directory as the filter
root. Keep Origin outputs under a separate result directory so the data adapter
and algorithm results do not blur together.

When using daily part files in PowerShell, quote the template:
`'global_phy_{yyyymmdd}.nc'`. Without quotes, PowerShell treats `{yyyymmdd}` as
a script block and passes the wrong string.

## Native W Diagnostics

`run_ofes_rebuild_w.py` remains in this package because it compares OFES native
model `w` with rebuilt `W`; this is OFES-specific and is not part of the Origin
detection chain.

The runner expects an existing Origin result table produced from exported OFES
NetCDF, or an older compatible result directory with:

```text
detection_hua_global_jan1991/centers_hua_style.csv
detection_hua_global_jan1991/structures_hua_style.csv
```

Example:

```powershell
$env:PYTHONNOUSERSITE='1'
& 'D:\Util\lever\02_miniforge\Library\bin\mamba.exe' run -n OFES_detection python -m Detection_for_OFES.run_ofes_rebuild_w `
  --data-root F:\OFES\external_OFES2 `
  --result-root E:\DATA\01_Eddy_correspond\02_OFES\<origin_ofes_detection_result> `
  --stages band `
  --band-lat-min 30 `
  --band-lat-max 35
```

The W runner keeps the current OFES diagnostic choices: regional density
background, stabilized `rho_z`, layer-center native W alignment, layerwise
translation tracking, and density/velocity scale separation. Its physical
formula and `rebuild_object_w()` calculation are intentionally not changed by
the interface cleanup.

### 20N crossing Cressman composite

For the OFES `velocity_streamline` Origin run, the W runner can also make a
strict-crossing composite instead of plotting single objects. This is a
diagnostic of the rebuilt-W chain, not a new eddy detection step.

```powershell
$env:PYTHONNOUSERSITE='1'
& 'D:\Util\lever\02_miniforge\Library\bin\mamba.exe' run -n OFES_detection python -m Detection_for_OFES.run_ofes_rebuild_w `
  --data-root F:\OFES\external_OFES2 `
  --result-root E:\DATA\01_Eddy_correspond\02_OFES\origin_streamline_cpu_jan01_jan19_life1 `
  --stages crossing_composite `
  --composite-lat 20 `
  --composite-polarities cyclonic,anticyclonic `
  --intersect-radius-r 1 `
  --cressman-radius-r 1 `
  --cressman-min-objects 8 `
  --composite-workers 4
```

The stage reads `centers_hua_style` and `structures_hua_style` as Parquet when
available, with CSV as fallback. It selects strict `1R` crossings only, splits
objects by polarity, calls the existing `rebuild_object_w()` for each source
object, and then maps the object-centered fields to a shared `x/R, y/R, depth`
grid with Cressman weights. The output is written under:

```text
<result-root>\w_rebuild_diagnostics\composite_20N_crossing\
```

Each polarity receives native-W crossing-section and focus figures, compressed
grid file, JSON metadata, and a summary table. Composite support is counted by
contributing object; cells below `--cressman-min-objects` remain missing.

The same stage also classifies each source object by the native-W multipole
structure before writing structure-aware composites. The current default
classifier uses the mesoscale native-W diagnostic:

```text
W_t      = available-day 10 day running mean of OFES native W
W_lp50   = horizontal Gaussian lowpass with approximate 50 km FWHM
W_lp500  = horizontal Gaussian lowpass with approximate 500 km FWHM
W_meso   = W_lp50 - W_lp500
```

This is a diagnostic approximation to the mesoscale `50-500 km` band: it removes
much of the day-scale native-W noise, small/submesoscale patches, and basin-scale
background without claiming a sharp spectral cutoff. The raw layer-center native
`w` remains in the grid files as `ofes_w_native_m_s`, while the classifier and
native-only composite figures prefer `ofes_w_native_meso_aligned_m_s`.

The multipole classifier averages `W_meso` over `300-500 m`, samples 60 azimuth
directions, and uses a true ray-based `0-1R` radial integral to judge the
azimuthal structure. This is not a sector-area average: for each azimuth the
field is sampled along the radial line from the center to `1R` and integrated
with trapezoidal weights. It applies QC before assigning a physical class:
sparse azimuth coverage, excessive NaNs near the outer radius, weak amplitude,
low SNR, or mixed harmonics are written as explicit `qc_*` classes instead of
being folded into `other`.

```text
0-1 crossings -> monopole
2 crossings   -> dipole
4 crossings   -> quadrupole
other counts  -> other
```

Each multipole class receives native-W-only crossing-section and focus figures.
The focus figure contains `0-100 m`, `300-500 m`, and `450 m` horizontal views
so the shallow filament layer is separated from the mesoscale depth range. The
older five-panel rebuild/native comparison is intentionally not emitted in this
native-W classification workflow.

For dipole objects, the code computes the mode-1 complex phase from the same
60-azimuth `300-500 m, 0-1R` radial-integral curve and defines
`positive_lobe_angle = -arg(C1)`. The local native-W mesoscale field is rotated
by this angle so the positive mode-1 lobe points toward `+x/R`, then Cressman
composited. The crossing section for the aligned field is therefore taken along
the aligned `y/R=0` axis; unaligned fields still use the strict geographic
crossing latitude.

The legacy raw-native temporal filter remains off by default:

```text
--native-w-temporal-filter none
```

Use `--native-w-temporal-filter lowpass_running_mean` only for a deliberate
legacy comparison. The new mesoscale native-W path is controlled separately by:

```text
--native-w-meso-filter temporal10d_spatial50_500km
--native-w-meso-time-window-days 10
--native-w-meso-small-cutoff-km 50
--native-w-meso-large-cutoff-km 500
--native-w-phase-align
```

To check whether the paper-style native-W composite is recovered only in a
specific dynamical setting, the same stage can be narrowed to dipole objects
and grouped into three diagnostic longitude sectors. This keeps cyclonic and
anticyclonic objects separate; signed native-W composites should not combine
opposite polarities.

```powershell
$env:PYTHONNOUSERSITE='1'
& 'D:\Util\lever\02_miniforge\Library\bin\mamba.exe' run -n OFES_detection python -m Detection_for_OFES.run_ofes_rebuild_w `
  --data-root F:\OFES\external_OFES2 `
  --result-root E:\DATA\01_Eddy_correspond\02_OFES\origin_streamline_cpu_jan01_jan19_life1 `
  --output-root E:\DATA\01_Eddy_correspond\02_OFES\origin_streamline_cpu_jan01_jan19_life1\w_rebuild_diagnostics `
  --stages crossing_composite `
  --composite-lat 20 `
  --composite-polarities cyclonic,anticyclonic `
  --composite-selection-class dipole `
  --composite-region-mode longitude_bins `
  --composite-region-boxes 'western_boundary:60,140;interior:140,240;eastern_basin:240,360' `
  --native-w-meso-filter temporal10d_spatial50_500km `
  --native-w-phase-align `
  --intersect-radius-r 1 `
  --cressman-radius-r 1 `
  --cressman-min-objects 8 `
  --composite-workers 4
```

The regional mode still uses strict `1R` crossing and the native-W multipole
classifier above, but writes only the native-W cross-section and focus figures
for each region and polarity. Do not combine cyclonic and anticyclonic native-W
composites; the signed W pattern has polarity-dependent phase. The focus figure
is intentionally limited to `0-100 m`, `300-500 m`, and `450 m` views; the
`300-500 m` and `450 m` rows are the primary reproduction checks. The default
longitude sectors are diagnostic bins:
`western_boundary=60-140E`, `interior=140-240E`, and
`eastern_basin=240-360E`. They can be changed with
`--composite-region-boxes` without changing the rebuild-W formula.

The new native-W mesoscale/aligned output is written under:

```text
<output-root>\w_native_meso50_500km_mode1_aligned_by_polarity\
```

This keeps it separate from earlier raw-native or unaligned composite products.

The supporting literature manifest for the mesoscale filtering choice is saved
at:

```text
E:\Report\01_Eddy_correspond\02_OFES\2026_09_09\documents\references\mesoscale_filter_literature_manifest.json
```

## Current Boundary

Keep in `Detection_for_OFES`:

- OFES `.ctl/.dta` readers and path helpers.
- OFES daily extraction and validation utilities.
- OFES-to-Origin NetCDF export.
- OFES native `w` versus rebuilt `W` diagnostics.

Keep out of `Detection_for_OFES`:

- eddy detection algorithm implementations;
- tracking and shape classification;
- panel rendering that duplicates Origin output;
- MATLAB or Origin backend logic.

Old OFES detection and panel code remains recoverable from git history, but new
runs should use the Origin pipeline after NetCDF export.
