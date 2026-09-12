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

The NetCDF exporter prefers `netCDF4`. On this Windows host, `netCDF4` can be
blocked by application-control policy; the exporter then falls back to SciPy's
NetCDF3 64-bit writer, which `Origin_eddy_detection` can read. If you want the
compressed NetCDF4 path and the environment does not have a working `netCDF4`,
install or repair it with mamba:

```powershell
& 'D:\Util\lever\02_miniforge\Library\bin\mamba.exe' install -n OFES_detection -c conda-forge netcdf4
```

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
  --overwrite
```

The exporter writes:

```text
E:\DATA\01_Eddy_correspond\02_OFES\origin_compatible_filter\global_phy_1991.nc
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
