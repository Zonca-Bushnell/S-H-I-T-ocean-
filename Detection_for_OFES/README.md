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

`tools/download_fes2014.py` and `tools/start_fes2014_download.ps1` prepare an
authenticated AVISO FES2014 download into `F:\Tide\FES2014`. OFES2 itself is
treated here as having tidal mixing parameterization but no explicit
time-varying barotropic tide forcing in the daily prognostic fields, so FES2014
is kept as an external tide reference rather than an OFES input dependency.
Set `AVISO_USERNAME` and `AVISO_PASSWORD` before launching the background
download:

```powershell
$env:AVISO_USERNAME='<your-aviso-user>'
$env:AVISO_PASSWORD='<your-aviso-password>'
& .\Detection_for_OFES\tools\start_fes2014_download.ps1 -DestRoot F:\Tide\FES2014
Get-Content F:\Tide\FES2014\logs\download_fes2014.stdout.log -Wait -Tail 40
```

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

## Velocity-Streamline Result Views

The current OFES detection result used for reporting is the Origin
`velocity_streamline_contour` run:

```text
E:\DATA\01_Eddy_correspond\02_OFES\origin_streamline_cpu_jan01_jan19_life1
```

For a quick surface overview and two representative family-panel views with
eddy boundaries, run:

```powershell
$env:PYTHONNOUSERSITE='1'
& 'D:\Util\lever\02_miniforge\Library\bin\mamba.exe' run -n OFES_detection python -m Detection_for_OFES.tools.plot_velocity_streamline_edges_pillow
```

For a LAVD-style regional map focused on the Kuroshio basin, with surface
velocity-anomaly speed as the background and `velocity_streamline_contour`
boundaries overlaid, run:

```powershell
$env:PYTHONNOUSERSITE='1'
& 'D:\Util\lever\02_miniforge\Library\bin\mamba.exe' run -n OFES_detection python -m Detection_for_OFES.tools.plot_kuroshio_velocity_streamline_map
```

The tool intentionally filters to:

```text
boundary_mode == velocity_streamline_contour
```

and writes figures under:

```text
<result-root>\figures\latest_velocity_streamline_surface_and_family_with_edges\
```

The plotted eddy edges are accepted-boundary radius proxies from the
velocity-streamline metadata. They are more informative than center points, but
they are not exact streamline polygons because the current
`centers_hua_style.parquet` table does not persist the full streamline vertex
coordinates.

To compare the OFES Kuroshio overview with META4.0, overlay the filtered META4
track table on the same map:

```powershell
& 'C:\Users\admin\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m Detection_for_OFES.tools.overlay_meta4_on_kuroshio_map `
  --ofes-day 1991-01-10 `
  --meta-day 2019-07-01
```

The local META4.0 source starts at 1993-01-01, so it cannot provide a strict
same-year overlay for the OFES 1991 sample. The overlay uses META4 filtered
track-table centers and radii, while OFES uses the velocity-streamline result
and its accepted-boundary radius proxy.

To audit why many SSH seeds fail the `velocity_streamline_contour` filter,
generate the rejection diagnostics:

```powershell
& 'C:\Users\admin\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m Detection_for_OFES.tools.diagnose_streamline_rejections `
  --result-root E:\DATA\01_Eddy_correspond\02_OFES\origin_streamline_cpu_jan01_jan19_life1 `
  --day 1991-01-10
```

This writes seed-fate tables, a Kuroshio seed-fate map, rejected-seed sample
figures, and the literature note under:

```text
<result-root>\diagnostics\streamline_rejection_audit\
```

The diagnostic does not rerun or modify detection. It reads existing
`centers_hua_style.parquet` / `circle_check_diagnostics.parquet` and highlights
whether failures are dominated by `no_closed_streamline`, velocity-ratio,
boundary-monotonic, or other Hua checks.

## SSH-Primary Overview Views

The current OFES META-like surface catalog experiment is the Origin
`ssh_effective_contour_primary` run:

```text
E:\DATA\01_Eddy_correspond\02_OFES\origin_ssh_effective_contour_primary_rawnc4_jan01_jan19_parallel
```

To show the latest SSH-primary eddies on an SSH-anomaly background with surface
velocity vectors, run:

```powershell
$env:PYTHONNOUSERSITE='1'
& 'D:\Util\lever\02_miniforge\Library\bin\mamba.exe' run -n OFES_detection python -m Detection_for_OFES.tools.plot_latest_ssh_vector_overview `
  --day 1991-01-10 `
  --result-root E:\DATA\01_Eddy_correspond\02_OFES\origin_ssh_effective_contour_primary_rawnc4_jan01_jan19_parallel `
  --filter-root E:\DATA\01_Eddy_correspond\02_OFES\origin_compatible_filter
```

The tool reads `zos_glor`, `uo_glor`, and `vo_glor` from the daily OFES NetCDF
part and overlays accepted surface objects from `centers_hua_style` /
`structures_hua_style`. If `ssh_contour_boundary_i/j` is present, the saved SSH
contour boundary pixels are drawn; otherwise the figure uses an equivalent-radius
boundary proxy.

Outputs are written under:

```text
<result-root>\figures\ssh_vector_overview\
  ofes_ssh_primary_global_ssh_uv_vectors_YYYYMMDD.png
  ofes_ssh_primary_kuroshio_ssh_uv_vectors_YYYYMMDD.png
  ofes_ssh_primary_north_pacific_interior_dense_ssh_uv_vectors_YYYYMMDD.png
  ssh_vector_overview_manifest_YYYYMMDD.json
```

The Kuroshio and open-ocean zoom products draw vectors every three OFES surface
grid cells by default, about `0.3 deg` (`--kuroshio-vector-step 3` and
`--ocean-vector-step 3`). The third product is a dense open-ocean zoom. Its
default box is `170E-210E, 20N-40N`, and can be changed with
`--ocean-zoom-bbox`.

## Jan1 Mesoscale-Filtered SSH-Primary Smoke

For the first filtering sensitivity test, build a diagnostic mesoscale input
from the existing OFES daily NetCDF parts. This does not reread `.dta` files and
does not change the SSH-primary detection code. The filter is:

```text
field_slow = available-day 10 day running mean(field)
field_meso = LP_50km(field_slow) - LP_500km(field_slow)
```

For Jan 1, the available running window is `1991-01-01` to `1991-01-06`.
Generate a surface-only Origin-compatible input with:

```powershell
$env:PYTHONNOUSERSITE='1'
& 'D:\Util\lever\02_miniforge\Library\bin\mamba.exe' run -n OFES_detection python -m Detection_for_OFES.tools.build_ofes_meso_filter `
  --input-root E:\DATA\01_Eddy_correspond\02_OFES\origin_compatible_filter `
  --output-root E:\DATA\01_Eddy_correspond\02_OFES\origin_compatible_filter_meso10d_50_500km `
  --start 1991-01-01 `
  --end 1991-01-01 `
  --temporal-window-days 10 `
  --small-cutoff-km 50 `
  --large-cutoff-km 500 `
  --max-depth-layers 1 `
  --overwrite
```

Then run a Jan1 surface-only SSH-primary smoke against that filtered input:

```powershell
$env:PYTHONNOUSERSITE='1'
Set-Location 'D:\01_Eddy\01_Vertical_asymmetric\S-H-I-T-ocean-\Origin_eddy_detection'
& 'D:\Util\lever\02_miniforge\Library\bin\mamba.exe' run -n OFES_detection python -m src.eddy_pipeline.detection_hybrid `
  --filter-root E:\DATA\01_Eddy_correspond\02_OFES\origin_compatible_filter_meso10d_50_500km `
  --raw-root E:\DATA\01_Eddy_correspond\02_OFES\origin_compatible_filter_meso10d_50_500km `
  --filter-template 'global_phy_{yyyymmdd}.nc' `
  --raw-template 'global_phy_{yyyymmdd}.nc' `
  --output-dir E:\DATA\01_Eddy_correspond\02_OFES\origin_ssh_effective_contour_primary_meso10d_50_500km_smoke_19910101 `
  --start 1991-01-01 --end 1991-01-01 --max-depth-m 3 `
  --candidate-selection tile_topn --tile-lon-deg 10 --tile-lat-deg 10 --tile-top-n 10 `
  --surface-search-cells 8 --deep-search-cells 6 --start-radius-cells 3 --max-radius-cells 8 `
  --boundary-mode ssh_effective_contour_primary `
  --ssh-primary-level-count 16 --ssh-primary-window-factor 4 --ssh-primary-max-radius-factor 2 `
  --ssh-consensus-min-finite-fraction 0.70 `
  --jet-core-speed-percentile 80 --jet-core-overlap-max 0.50 `
  --hua-backend python --preload-day-uv --write-day-figures --resume --skip-axis-examples
```

Overview figures can be generated with:

```powershell
$env:PYTHONNOUSERSITE='1'
& 'D:\Util\lever\02_miniforge\Library\bin\mamba.exe' run -n OFES_detection python -m Detection_for_OFES.tools.plot_latest_ssh_vector_overview `
  --day 1991-01-01 `
  --result-root E:\DATA\01_Eddy_correspond\02_OFES\origin_ssh_effective_contour_primary_meso10d_50_500km_smoke_19910101 `
  --filter-root E:\DATA\01_Eddy_correspond\02_OFES\origin_compatible_filter_meso10d_50_500km
```

The `science_tag` is `meso10d_50_500km_diagnostic`. This scale separation is a
diagnostic way to suppress daily high-frequency signals, submesoscale/small
features, and large-scale background. It is not strict barotropic detiding and
not a 30-180 day frequency-domain bandpass.

### Literature Spatial-Filter Smoke

For the OFES/META literature-aligned preprocessing check, treat broad
barotropic/tide-like and basin-scale background signals primarily as a spatial
scale-separation problem. Two Jan1 surface-only inputs are currently used:

- `spatial25_200km_ofes_literature_diagnostic`: OFES KOE literature-style
  spatial band-pass, `LP_25km - LP_200km`.
- `pet500km_highpass_diagnostic`: PET-style high-pass, `field - LP_500km`.

Both products keep `temporal_window_days=1` for this smoke so the comparison is
about spatial filtering, not time filtering:

```powershell
$env:PYTHONNOUSERSITE='1'
& 'D:\Util\lever\02_miniforge\Library\bin\mamba.exe' run -n OFES_detection python -m Detection_for_OFES.tools.build_ofes_meso_filter `
  --input-root E:\DATA\01_Eddy_correspond\02_OFES\origin_compatible_filter `
  --output-root E:\DATA\01_Eddy_correspond\02_OFES\origin_compatible_filter_spatial25_200km_ofes_literature `
  --start 1991-01-01 --end 1991-01-01 `
  --temporal-window-days 1 `
  --filter-mode bandpass `
  --small-cutoff-km 25 `
  --large-cutoff-km 200 `
  --science-tag spatial25_200km_ofes_literature_diagnostic `
  --max-depth-layers 1 `
  --overwrite

& 'D:\Util\lever\02_miniforge\Library\bin\mamba.exe' run -n OFES_detection python -m Detection_for_OFES.tools.build_ofes_meso_filter `
  --input-root E:\DATA\01_Eddy_correspond\02_OFES\origin_compatible_filter `
  --output-root E:\DATA\01_Eddy_correspond\02_OFES\origin_compatible_filter_pet500km_highpass `
  --start 1991-01-01 --end 1991-01-01 `
  --temporal-window-days 1 `
  --filter-mode highpass `
  --large-cutoff-km 500 `
  --science-tag pet500km_highpass_diagnostic `
  --max-depth-layers 1 `
  --overwrite
```

### Rossby-radius adaptive band-pass smoke

For latitude-sensitive mesoscale separation, the preferred Jan1 surface
diagnostic uses a Chelton et al. (1998) first-baroclinic Rossby-radius profile
for the lower cutoff and a fixed 180 km upper cutoff:

```text
L_small(lat) = clip(0.5 * R1(lat), 10, 180) km
L_large(lat) = 180 km
meso = LP(L_small) - LP(L_large)
```

This keeps a narrow near-equatorial pass-band, while retaining the smaller
high-latitude eddy scales. The Rossby-radius ASCII/NetCDF input is expected at:

```text
E:\DATA\01_Eddy_correspond\02_OFES\rossby_radius_chelton1998\...\rossrad.nc
```

```powershell
& 'D:\Util\lever\02_miniforge\Library\bin\mamba.exe' run -n OFES_detection python -m Detection_for_OFES.tools.build_ofes_meso_filter `
  --input-root E:\DATA\01_Eddy_correspond\02_OFES\origin_compatible_filter `
  --output-root E:\DATA\01_Eddy_correspond\02_OFES\origin_compatible_filter_rossby_lower_upper180 `
  --start 1991-01-01 --end 1991-01-01 `
  --temporal-window-days 1 `
  --filter-mode bandpass `
  --large-cutoff-mode rossby_lower_latadaptive_upper `
  --rossby-radius-path E:\DATA\01_Eddy_correspond\02_OFES\rossby_radius_chelton1998\unzip\fecampos-campos2025-a5d24c0\rossrad.nc `
  --rossby-small-factor 0.5 `
  --adaptive-large-cutoff-min-km 180 `
  --adaptive-large-cutoff-max-km 180 `
  --rossby-min-km 10 --rossby-max-km 180 `
  --max-depth-layers 1 --overwrite
```

For the Jan1 smoke, also relax the SSH-contour minimum equivalent radius from
`3` cells to `2` cells. High-latitude coherent eddies can be smaller than the
3-cell gate, while the R1-scaled lower cutoff prevents the same relaxation from
reintroducing near-equatorial speckle:

```text
--start-radius-cells 2 --max-radius-cells 12
```

With `tile_topn=10`, this combined scheme reduces `abs(lat)<20` accepted objects
from `371` to `326`, keeps non-WBC subtropics at `515` versus `519`, and raises
non-ACC mid/high-lat objects from `261` to `265`; south non-ACC mid/high rises
from `80` to `88`. North mid/high remains `177` versus `181`, so the remaining
gaps are subpolar-specific and should be handled in a later region-aware radius
or persistence step rather than by further global filter retuning.

Run the corresponding Jan1 SSH-primary smoke with the same detection settings
and `--ssh-primary-min-amplitude-cm 1`. The SSH-primary scan can be slow on
some filtered fields, so prefer logging/background runs for `tile-top-n=10`:

```powershell
$env:PYTHONNOUSERSITE='1'
Set-Location 'D:\01_Eddy\01_Vertical_asymmetric\S-H-I-T-ocean-\Origin_eddy_detection'
$filterRoot = 'E:\DATA\01_Eddy_correspond\02_OFES\origin_compatible_filter_spatial25_200km_ofes_literature'
$outRoot = 'E:\DATA\01_Eddy_correspond\02_OFES\origin_ssh_primary_spatial25_200km_amp1cm_smoke_19910101'
New-Item -ItemType Directory -Path "$outRoot\logs" -Force | Out-Null
Start-Process -FilePath 'D:\Util\lever\02_miniforge\envs\OFES_detection\python.exe' `
  -ArgumentList @(
    '-m','src.eddy_pipeline.detection_hybrid',
    '--filter-root',$filterRoot,'--raw-root',$filterRoot,
    '--filter-template','global_phy_{yyyymmdd}.nc','--raw-template','global_phy_{yyyymmdd}.nc',
    '--output-dir',$outRoot,'--start','1991-01-01','--end','1991-01-01',
    '--boundary-mode','ssh_effective_contour_primary',
    '--ssh-primary-level-count','16','--ssh-primary-window-factor','4','--ssh-primary-max-radius-factor','2',
    '--ssh-primary-min-amplitude-cm','1',
    '--candidate-selection','tile_topn','--tile-lon-deg','10','--tile-lat-deg','10','--tile-top-n','10',
    '--max-depth-m','3','--hua-backend','python','--preload-day-uv','--resume','--skip-axis-examples'
  ) `
  -WorkingDirectory 'D:\01_Eddy\01_Vertical_asymmetric\S-H-I-T-ocean-\Origin_eddy_detection' `
  -RedirectStandardOutput "$outRoot\logs\stdout.log" `
  -RedirectStandardError "$outRoot\logs\stderr.log" `
  -WindowStyle Hidden
```

Repeat the same command with:

```text
filterRoot = E:\DATA\01_Eddy_correspond\02_OFES\origin_compatible_filter_pet500km_highpass
outRoot    = E:\DATA\01_Eddy_correspond\02_OFES\origin_ssh_primary_pet500km_highpass_amp1cm_smoke_19910101
```

OFES KOE literature used `25-200 km` spatial band-pass filtering and then kept
tracked eddies with lifetime at least 30 days, radius above 25 km, and vertical
extent deeper than 100 m. PET/META-style detection commonly removes broad
background with an approximately `500 km` high-pass before closed-contour
identification and records shape error, amplitude, masked-pixel, and
multi-extremum rejection diagnostics. These literature settings are smoke-test
preprocessing references, not final OFES catalog policy.

For a qualitative META4.0 comparison over the same three boxes (`global`,
`Kuroshio`, and `North Pacific interior dense`), draw centers and effective
contours with:

```powershell
$env:PYTHONNOUSERSITE='1'
& 'D:\Util\lever\02_miniforge\Library\bin\mamba.exe' run -n OFES_detection python -m Detection_for_OFES.tools.plot_meta4_three_region_overview `
  --meta-root F:\Eddy\Eddy\META4.0_DT_allsat `
  --day 1993-01-01
```

The META figure is catalog-only; the public META NetCDF files used here contain
center/effective-contour samples, not the SSH background field.

To check whether the filtered OFES Jan1 smoke is over-dense or over-fragmented
relative to META4.0, compare radius and amplitude distributions with:

```powershell
$env:PYTHONNOUSERSITE='1'
& 'D:\Util\lever\02_miniforge\Library\bin\mamba.exe' run -n OFES_detection python -m Detection_for_OFES.tools.diagnose_ofes_meta_filter_density `
  --ofes-root E:\DATA\01_Eddy_correspond\02_OFES\origin_ssh_effective_contour_primary_meso10d_50_500km_smoke_19910101 `
  --meta-root F:\Eddy\Eddy\META4.0_DT_allsat `
  --ofes-day 1991-01-01 `
  --meta-day 1993-01-01
```

This diagnostic is only for preprocessing/filter tuning. It does not rerun
detection.

### SSH-Primary Boundary Shape Constraint

For the final Rossby-radius `rossby_lower_upper180` surface smoke, the
`ssh_effective_contour_primary` catalog now applies a PET/META-style fitted-circle
shape check inside the core contour acceptance stage. The metric is:

```text
fit a best circle to the saved SSH contour boundary
shape_error = (A_polygon + A_circle - 2 * A_intersection) / A_circle * 100
```

The default gates are:

```text
global shape_error <= 70%
ACC shape_error <= 55%  (0-360E, 62S-40S)
```

This is not the older local `std(radius)/mean(radius)` heuristic. A contour
that passes closure, boundary, radius, amplitude, and single-extremum checks
but exceeds the shape gate is rejected with:

```text
catalog_acceptance_reason = ssh_primary_shape_error_high
```

Run the Jan1 smoke from the worktree root:

```powershell
$env:PYTHONNOUSERSITE='1'
& 'D:\Util\lever\02_miniforge\Library\bin\mamba.exe' run -n OFES_detection python -m Origin_eddy_detection.src.eddy_pipeline.detection_hybrid `
  --filter-root 'E:\DATA\01_Eddy_correspond\02_OFES\origin_compatible_filter_rossby_lower_upper180' `
  --raw-root 'E:\DATA\01_Eddy_correspond\02_OFES\origin_compatible_filter_rossby_lower_upper180' `
  --filter-template 'global_phy_{yyyymmdd}.nc' `
  --raw-template 'global_phy_{yyyymmdd}.nc' `
  --output-dir 'E:\DATA\01_Eddy_correspond\02_OFES\origin_ssh_primary_rossby_lower_upper180_r2_shape70_acc55_exact_amp1cm_smoke_19910101' `
  --start 1991-01-01 --end 1991-01-01 `
  --max-depth-m 3 `
  --boundary-mode ssh_effective_contour_primary `
  --ssh-primary-min-amplitude-cm 1 `
  --ssh-primary-max-shape-error-percent 70 `
  --ssh-primary-acc-max-shape-error-percent 55 `
  --start-radius-cells 2 --max-radius-cells 12 `
  --candidate-selection tile_topn --tile-top-n 10 `
  --preload-day-uv --skip-axis-examples
```

The new core fields are `ssh_contour_shape_error_percent`,
`ssh_contour_compactness`, and `ssh_contour_boundary_point_count`. In the Jan1
smoke, this reduces the radius-only `1920` accepted surface objects to `1766`;
the two direct shape failures are retained as
`ssh_primary_shape_error_high`. Overview figures can be produced with the same
North Pacific open-ocean and ACC boxes as:

```powershell
$env:PYTHONNOUSERSITE='1'
& 'D:\Util\lever\02_miniforge\Library\bin\mamba.exe' run -n OFES_detection python -m Detection_for_OFES.tools.plot_latest_ssh_vector_overview `
  --day 1991-01-01 `
  --result-root 'E:\DATA\01_Eddy_correspond\02_OFES\origin_ssh_primary_rossby_lower_upper180_r2_shape70_acc55_exact_amp1cm_smoke_19910101' `
  --filter-root 'E:\DATA\01_Eddy_correspond\02_OFES\origin_compatible_filter_rossby_lower_upper180' `
  --run-tag shape70_acc55 `
  --regional-zoom-name 'North Pacific open ocean' `
  --regional-zoom-bbox 145 250 20 45 `
  --regional-vector-step 3 `
  --ocean-zoom-name 'ACC Southern Ocean' `
  --ocean-zoom-bbox 0 360 -62 -40 `
  --ocean-vector-step 3
```

Add `--exclude-jet-flagged` for the no-jet comparison views.

### SSH-Primary Velocity-Streamline Effective Without SSH Gate

`ssh_primary_velocity_streamline_effective` now uses SSH extrema only as seeds.
The surface acceptance gate is the velocity streamline circle check around the
speed weak center; the SSH anomaly closed contour is not used as an
accept/reject threshold.

```text
SSH extremum seed
  -> seeded speed-minimum center + subgrid refinement
  -> radius scan start_radius_cells..max_radius_cells
  -> closed velocity streamline circle check
  -> pass = first valid streamline boundary
  -> fail = no_closed_streamline_effective
```

Run the Jan1 surface smoke on the final Rossby-radius filter:

```powershell
$env:PYTHONNOUSERSITE='1'
& 'D:\Util\lever\02_miniforge\Library\bin\mamba.exe' run -n OFES_detection python -m Origin_eddy_detection.src.eddy_pipeline.detection_hybrid `
  --filter-root 'E:\DATA\01_Eddy_correspond\02_OFES\origin_compatible_filter_rossby_lower_upper180' `
  --raw-root 'E:\DATA\01_Eddy_correspond\02_OFES\origin_compatible_filter_rossby_lower_upper180' `
  --filter-template 'global_phy_{yyyymmdd}.nc' `
  --raw-template 'global_phy_{yyyymmdd}.nc' `
  --output-dir 'E:\DATA\01_Eddy_correspond\02_OFES\origin_ssh_primary_velocity_streamline_effective_weakcenter_rossby_19910101' `
  --start 1991-01-01 --end 1991-01-01 --max-depth-m 3 `
  --boundary-mode ssh_primary_velocity_streamline_effective `
  --start-radius-cells 2 --max-radius-cells 12 `
  --candidate-selection tile_topn --tile-top-n 10 `
  --preload-day-uv --skip-axis-examples
```

In this Jan1 smoke, `1422` of `5025` surface candidates pass the velocity
streamline circle check. Rejection reasons are now exclusively streamline/Hua
failures such as `no_closed_streamline`, `velocity_ratio`, `angle_jump`,
`boundary_monotonic_rotation`, `opposite_reversal`, and `tangent_alignment`;
there are no `ssh_primary_*` contour rejections.

### Unified OFES Eddy Catalog

For production-style runs, use the unified orchestrator. It first ensures the
latest Rossby-radius filter exists, then runs the default
`ssh_primary_velocity_streamline_effective` detection by day, applies shape,
overlap, and persistence QC, and writes raw and final catalogs.

```powershell
$env:PYTHONNOUSERSITE='1'
& 'D:\Util\lever\02_miniforge\Library\bin\mamba.exe' run -n OFES_detection python -m Detection_for_OFES.tools.build_unified_eddy_catalog `
  --filter-input-root 'E:\DATA\01_Eddy_correspond\02_OFES\origin_compatible_filter' `
  --filter-output-root 'E:\DATA\01_Eddy_correspond\02_OFES\origin_compatible_filter_rossby_lower_upper180' `
  --output-root 'E:\DATA\01_Eddy_correspond\02_OFES\origin_unified_ssh_streamline_rossby_19910101_smoke' `
  --start 1991-01-01 --end 1991-01-01 `
  --max-depth-m 3 `
  --workers 1
```

The default filter is `rossby_lower_latadaptive_upper`:

```text
L_small(lat) = clip(0.5 * R1, 10, 180) km
L_large(lat) = lat_adaptive(180, 180) = 180 km
```

The orchestrator writes:

```text
<output-root>/raw_detection/daily_runs/YYYYMMDD/
<output-root>/daily_runs/YYYYMMDD/
<output-root>/logs/
<output-root>/unified_catalog_summary.csv/json
```

### Latitude-Adaptive 50-180 km Smoke

The fixed `LP50-LP500` experiment can over-retain broad wave packets in the
open ocean and high latitudes. A narrower latitude-adaptive diagnostic keeps the
50 km small-scale cutoff and sets the large-scale cutoff to:

```text
L_hi(lat) = 50 km + 130 km * cos(lat)^2
field_meso = LP_50km(field_slow) - LP_L_hi(lat)(field_slow)
```

Run the Jan1 surface-only export with:

```powershell
$env:PYTHONNOUSERSITE='1'
& 'D:\Util\lever\02_miniforge\Library\bin\mamba.exe' run -n OFES_detection python -m Detection_for_OFES.tools.build_ofes_meso_filter `
  --input-root E:\DATA\01_Eddy_correspond\02_OFES\origin_compatible_filter `
  --output-root E:\DATA\01_Eddy_correspond\02_OFES\origin_compatible_filter_meso10d_50_180km_latadaptive `
  --start 1991-01-01 `
  --end 1991-01-01 `
  --temporal-window-days 10 `
  --small-cutoff-km 50 `
  --large-cutoff-mode latitude_adaptive `
  --adaptive-large-cutoff-min-km 50 `
  --adaptive-large-cutoff-max-km 180 `
  --max-depth-layers 1 `
  --overwrite
```

Run the matching Jan1 SSH-primary smoke by using the adaptive filter root and a
separate output directory:

```powershell
$env:PYTHONNOUSERSITE='1'
Set-Location 'D:\01_Eddy\01_Vertical_asymmetric\S-H-I-T-ocean-\Origin_eddy_detection'
& 'D:\Util\lever\02_miniforge\Library\bin\mamba.exe' run -n OFES_detection python -m src.eddy_pipeline.detection_hybrid `
  --filter-root E:\DATA\01_Eddy_correspond\02_OFES\origin_compatible_filter_meso10d_50_180km_latadaptive `
  --raw-root E:\DATA\01_Eddy_correspond\02_OFES\origin_compatible_filter_meso10d_50_180km_latadaptive `
  --filter-template 'global_phy_{yyyymmdd}.nc' `
  --raw-template 'global_phy_{yyyymmdd}.nc' `
  --output-dir E:\DATA\01_Eddy_correspond\02_OFES\origin_ssh_effective_contour_primary_meso10d_50_180km_latadaptive_amp2cm_smoke_19910101 `
  --start 1991-01-01 --end 1991-01-01 --max-depth-m 3 `
  --candidate-selection tile_topn --tile-lon-deg 10 --tile-lat-deg 10 --tile-top-n 10 `
  --surface-search-cells 8 --deep-search-cells 6 --start-radius-cells 3 --max-radius-cells 8 `
  --boundary-mode ssh_effective_contour_primary `
  --ssh-primary-level-count 16 --ssh-primary-window-factor 4 --ssh-primary-max-radius-factor 2 `
  --ssh-primary-min-amplitude-cm 2 `
  --ssh-consensus-min-finite-fraction 0.70 `
  --jet-core-speed-percentile 80 --jet-core-overlap-max 0.50 `
  --hua-backend python --preload-day-uv --resume --skip-axis-examples
```

Then redraw the three overview products by pointing the existing overview tool
at the adaptive filter and adaptive result root. The `--ssh-primary-min-amplitude-cm 2`
gate removes weak-amplitude closed contours before they enter the surface
catalog.

To test the stricter question "keep SSH object discovery, but replace the saved
boundary with a velocity-streamline effective contour", post-process the
accepted Jan1 SSH-primary catalog and redraw the same three overviews:

```powershell
$env:PYTHONNOUSERSITE='1'
& 'D:\Util\lever\02_miniforge\Library\bin\mamba.exe' run -n OFES_detection python -m Detection_for_OFES.tools.postprocess_streamline_effective_boundary `
  --source-root E:\DATA\01_Eddy_correspond\02_OFES\origin_ssh_effective_contour_primary_meso10d_50_180km_latadaptive_amp2cm_effective_multi_smoke_19910101 `
  --filter-root E:\DATA\01_Eddy_correspond\02_OFES\origin_compatible_filter_meso10d_50_180km_latadaptive `
  --output-root E:\DATA\01_Eddy_correspond\02_OFES\origin_ssh_primary_velocity_streamline_effective_meso10d_50_180km_latadaptive_amp2cm_smoke_19910101 `
  --day 1991-01-01

& 'D:\Util\lever\02_miniforge\Library\bin\mamba.exe' run -n OFES_detection python -m Detection_for_OFES.tools.plot_latest_ssh_vector_overview `
  --filter-root E:\DATA\01_Eddy_correspond\02_OFES\origin_compatible_filter_meso10d_50_180km_latadaptive `
  --result-root E:\DATA\01_Eddy_correspond\02_OFES\origin_ssh_primary_velocity_streamline_effective_meso10d_50_180km_latadaptive_amp2cm_smoke_19910101 `
  --day 1991-01-01 `
  --kuroshio-vector-step 3 `
  --ocean-vector-step 3
```

In the Jan1 smoke, this reduced the adaptive amp2 multi-contour SSH-primary
surface objects from 675 to 611, with Kuroshio 17 to 16 and North Pacific
interior dense unchanged at 2. The output maps draw saved streamline boundaries
first, then SSH boundaries, and only fall back to equivalent-radius proxies when
no saved boundary is available.

To convert the SSH-primary closed-contour candidates into a more conservative
OFES eddy-candidate catalog, run the META-like QC post-process. This does not
change the original detection tables; it writes a new Jan1 smoke directory with
shape QC, overlap de-duplication, and jet-meander split labels:

```powershell
$env:PYTHONNOUSERSITE='1'
& 'D:\Util\lever\02_miniforge\Library\bin\mamba.exe' run -n OFES_detection python -m Detection_for_OFES.tools.postprocess_ofes_eddy_qc `
  --source-root E:\DATA\01_Eddy_correspond\02_OFES\origin_ssh_effective_contour_primary_meso10d_50_180km_latadaptive_amp2cm_effective_multi_smoke_19910101 `
  --filter-root E:\DATA\01_Eddy_correspond\02_OFES\origin_compatible_filter_meso10d_50_180km_latadaptive `
  --output-root E:\DATA\01_Eddy_correspond\02_OFES\origin_ssh_primary_qc_meso10d_50_180km_latadaptive_amp2cm_smoke_19910101 `
  --day 1991-01-01

& 'D:\Util\lever\02_miniforge\Library\bin\mamba.exe' run -n OFES_detection python -m Detection_for_OFES.tools.plot_latest_ssh_vector_overview `
  --filter-root E:\DATA\01_Eddy_correspond\02_OFES\origin_compatible_filter_meso10d_50_180km_latadaptive `
  --result-root E:\DATA\01_Eddy_correspond\02_OFES\origin_ssh_primary_qc_meso10d_50_180km_latadaptive_amp2cm_smoke_19910101 `
  --day 1991-01-01 `
  --catalog-layer all
```

The QC output includes `isolated_eddy_candidates.csv`,
`jet_meander_candidates.csv`, and `qc_summary_YYYYMMDD.csv/json`. For the Jan1
smoke, the 675 raw SSH-primary objects became 208 isolated eddy candidates, 450
jet-meander candidates, and 17 shape-rejected candidates. Kuroshio changed from
17 raw objects to 1 isolated, 15 jet-meander, and 1 shape-rejected object. The
North Pacific interior dense box has 2 objects, both split into jet-meander.
Use `--catalog-layer isolated` or `--catalog-layer jet_meander` to redraw only
one layer without overwriting the all-candidate maps.

### Open-Ocean Recovery Amplitude Smoke

For open-ocean subtropical recovery, keep the latitude-adaptive
`meso10d_50_180km` input fixed and compare only the SSH-primary amplitude
threshold. This tests whether the `2 cm` threshold is too strict for weak
open-ocean eddies. Run Jan1 surface-only smokes with:

```powershell
$env:PYTHONNOUSERSITE='1'
Set-Location 'D:\01_Eddy\01_Vertical_asymmetric\S-H-I-T-ocean-\Origin_eddy_detection'

foreach ($amp in @('0.4','1.0','2.0')) {
  $tag = if ($amp -eq '0.4') {'amp0p4cm_openocean'} elseif ($amp -eq '1.0') {'amp1cm_openocean'} else {'amp2cm_existing_baseline'}
  & 'D:\Util\lever\02_miniforge\Library\bin\mamba.exe' run -n OFES_detection python -m src.eddy_pipeline.detection_hybrid `
    --filter-root E:\DATA\01_Eddy_correspond\02_OFES\origin_compatible_filter_meso10d_50_180km_latadaptive `
    --raw-root E:\DATA\01_Eddy_correspond\02_OFES\origin_compatible_filter_meso10d_50_180km_latadaptive `
    --filter-template 'global_phy_{yyyymmdd}.nc' `
    --raw-template 'global_phy_{yyyymmdd}.nc' `
    --output-dir "E:\DATA\01_Eddy_correspond\02_OFES\origin_ssh_effective_contour_primary_meso10d_50_180km_latadaptive_${tag}_smoke_19910101" `
    --start 1991-01-01 --end 1991-01-01 --max-depth-m 3 `
    --candidate-selection tile_topn --tile-lon-deg 10 --tile-lat-deg 10 --tile-top-n 10 `
    --surface-search-cells 8 --deep-search-cells 6 --start-radius-cells 3 --max-radius-cells 8 `
    --boundary-mode ssh_effective_contour_primary `
    --ssh-primary-level-count 16 --ssh-primary-window-factor 4 --ssh-primary-max-radius-factor 2 `
    --ssh-primary-min-amplitude-cm $amp `
    --ssh-consensus-min-finite-fraction 0.70 `
    --jet-core-speed-percentile 80 --jet-core-overlap-max 0.50 `
    --hua-backend python --preload-day-uv --resume --skip-axis-examples
}
```

For these smokes, redraw global plus two North Pacific open-ocean zooms:

```powershell
& 'D:\Util\lever\02_miniforge\Library\bin\mamba.exe' run -n OFES_detection python -m Detection_for_OFES.tools.plot_latest_ssh_vector_overview `
  --filter-root E:\DATA\01_Eddy_correspond\02_OFES\origin_compatible_filter_meso10d_50_180km_latadaptive `
  --result-root <one-of-the-three-smoke-roots> `
  --day 1991-01-01 `
  --regional-zoom-name 'North Pacific open ocean' `
  --regional-zoom-bbox 145 250 20 45 `
  --ocean-zoom-name 'North Pacific interior dense' `
  --ocean-zoom-bbox 170 210 20 40
```

To make a display-only comparison with the fast-current-overlap diagnostic
hidden, add `--exclude-jet-flagged`. This does not modify the saved detection
tables; it only removes rows with `jet_meander_flag=True` from the overview:

```powershell
& 'D:\Util\lever\02_miniforge\Library\bin\mamba.exe' run -n OFES_detection python -m Detection_for_OFES.tools.plot_latest_ssh_vector_overview `
  --filter-root E:\DATA\01_Eddy_correspond\02_OFES\origin_compatible_filter_meso10d_50_180km_latadaptive `
  --result-root E:\DATA\01_Eddy_correspond\02_OFES\origin_ssh_effective_contour_primary_meso10d_50_180km_latadaptive_amp0p4cm_openocean_smoke_19910101 `
  --day 1991-01-01 `
  --regional-zoom-name 'North Pacific open ocean' `
  --regional-zoom-bbox 145 250 20 45 `
  --ocean-zoom-name 'North Pacific interior dense' `
  --ocean-zoom-bbox 170 210 20 40 `
  --exclude-jet-flagged
```

Summarize the open-ocean recovery with:

```powershell
& 'D:\Util\lever\02_miniforge\Library\bin\mamba.exe' run -n OFES_detection python -m Detection_for_OFES.tools.summarize_open_ocean_recovery `
  --filter-root E:\DATA\01_Eddy_correspond\02_OFES\origin_compatible_filter_meso10d_50_180km_latadaptive `
  --output-dir E:\DATA\01_Eddy_correspond\02_OFES\open_ocean_recovery_meso10d_50_180km_latadaptive_19910101 `
  --day 1991-01-01 `
  --case 'amp0p4=E:\DATA\01_Eddy_correspond\02_OFES\origin_ssh_effective_contour_primary_meso10d_50_180km_latadaptive_amp0p4cm_openocean_smoke_19910101' `
  --case 'amp1=E:\DATA\01_Eddy_correspond\02_OFES\origin_ssh_effective_contour_primary_meso10d_50_180km_latadaptive_amp1cm_openocean_smoke_19910101' `
  --case 'amp2=E:\DATA\01_Eddy_correspond\02_OFES\origin_ssh_effective_contour_primary_meso10d_50_180km_latadaptive_amp2cm_existing_baseline_smoke_19910101'
```

Jan1 results show that `2 cm` is too strict for the North Pacific open-ocean
target: global counts are `2252 / 1338 / 675` for `0.4 / 1.0 / 2.0 cm`;
North Pacific open-ocean counts are `175 / 84 / 27`; the dense interior box is
`52 / 27 / 2`. The `1 cm` threshold is the current practical compromise for
OFES open-ocean recovery: it greatly reduces the `2 cm` undercount without
returning to the much denser `0.4 cm` global catalog.

### Spatial 25-200 km ACC QC Smoke

The `spatial25_200km_ofes_literature_diagnostic` input improves open-ocean
recovery but can leave many instantaneous ACC closed-contour fragments. Run this
Jan1 post-process to keep the `1 cm` open-ocean threshold while adding contour
QC, ACC-specific shape control, overlap de-duplication, and optional
persistence metadata:

```powershell
$env:PYTHONNOUSERSITE='1'
& 'D:\Util\lever\02_miniforge\Library\bin\mamba.exe' run -n OFES_detection python -m Detection_for_OFES.tools.postprocess_ofes_eddy_qc `
  --source-root E:\DATA\01_Eddy_correspond\02_OFES\origin_ssh_primary_spatial25_200km_amp1cm_smoke_19910101 `
  --filter-root E:\DATA\01_Eddy_correspond\02_OFES\origin_compatible_filter_spatial25_200km_ofes_literature `
  --output-root E:\DATA\01_Eddy_correspond\02_OFES\origin_ssh_primary_spatial25_200km_amp1cm_acc_qc_smoke_19910101 `
  --day 1991-01-01
```

The default QC keeps global `shape_error_percent <= 70`, applies stricter ACC
`shape_error_percent <= 55` for `40S-62S`, rejects contours with multiple
same-sign extrema, radius below `25 km`, too few boundary points, or too small
area, and performs same-polarity contour/radius overlap de-duplication. Jet-core
overlap is recorded but not split out unless `--enable-jet-split` is explicitly
set.

Redraw the three overview maps with:

```powershell
& 'D:\Util\lever\02_miniforge\Library\bin\mamba.exe' run -n OFES_detection python -m Detection_for_OFES.tools.plot_latest_ssh_vector_overview `
  --result-root E:\DATA\01_Eddy_correspond\02_OFES\origin_ssh_primary_spatial25_200km_amp1cm_acc_qc_smoke_19910101 `
  --filter-root E:\DATA\01_Eddy_correspond\02_OFES\origin_compatible_filter_spatial25_200km_ofes_literature `
  --day 1991-01-01 `
  --global-vector-step 120 `
  --regional-zoom-name 'North Pacific open ocean' `
  --regional-zoom-bbox 145 250 20 45 `
  --regional-vector-step 5 `
  --ocean-zoom-name 'ACC Southern Ocean' `
  --ocean-zoom-bbox 0 360 -62 -40 `
  --ocean-vector-step 40
```

To inspect the same catalog after hiding objects tagged by the 19-day surface
persistence diagnostic as `transient`, add `--exclude-transient`. This only
changes the overview display and output file suffix (`_no_transient`); it does
not delete or rewrite the QC catalog:

```powershell
& 'D:\Util\lever\02_miniforge\Library\bin\mamba.exe' run -n OFES_detection python -m Detection_for_OFES.tools.plot_latest_ssh_vector_overview `
  --result-root E:\DATA\01_Eddy_correspond\02_OFES\origin_ssh_primary_spatial25_200km_amp1cm_acc_qc_persistence_jan01_jan19 `
  --filter-root E:\DATA\01_Eddy_correspond\02_OFES\origin_compatible_filter_spatial25_200km_ofes_literature_jan01_jan19 `
  --day 1991-01-01 `
  --global-vector-step 120 `
  --regional-zoom-name 'North Pacific open ocean' `
  --regional-zoom-bbox 145 250 20 45 `
  --regional-vector-step 5 `
  --ocean-zoom-name 'ACC Southern Ocean' `
  --ocean-zoom-bbox 0 360 -62 -40 `
  --ocean-vector-step 40 `
  --exclude-transient
```

For the Jan1 smoke, `spatial25_200_amp1` changes from `1934` raw objects to
`1897` QC-pass objects. The North Pacific subtropical boxes remain recovered
(`NE 44 -> 43`, `NW 77 -> 76`), while ACC changes only from `619 -> 608`.
Because this source directory currently only contains Jan1, persistence is
written as `not_evaluated_no_future_days`; actual ACC fragment suppression
should use Jan1-Jan6 or longer same-filter detections so transient closed
contours can be separated from tracked eddies.

### Velocity-Streamline Spatial25-200 Effective Comparison

To compare how the current OFES preprocessing/QC chain changes the
`velocity_streamline_contour` catalog, use the four-panel global overview tool:

```powershell
$env:PYTHONNOUSERSITE='1'
& 'D:\Util\lever\02_miniforge\Library\bin\mamba.exe' run -n OFES_detection python -m Detection_for_OFES.tools.plot_velocity_streamline_four_overview `
  --day 1991-01-01 `
  --filter-root E:\DATA\01_Eddy_correspond\02_OFES\origin_compatible_filter_spatial25_200km_ofes_literature_jan01_jan19 `
  --old-streamline-root E:\DATA\01_Eddy_correspond\02_OFES\origin_streamline_cpu_jan01_jan19_life1 `
  --spatial-streamline-root E:\DATA\01_Eddy_correspond\02_OFES\origin_velocity_streamline_spatial25_200km_surface_smoke_19910101_ready `
  --effective-root E:\DATA\01_Eddy_correspond\02_OFES\origin_velocity_streamline_spatial25_200km_effective_smoke_19910101 `
  --persistence-root E:\DATA\01_Eddy_correspond\02_OFES\origin_velocity_streamline_spatial25_200km_effective_persistence_smoke_19910101 `
  --output-root E:\DATA\01_Eddy_correspond\02_OFES\origin_velocity_streamline_spatial25_200_effective_persistence_compare_19910101
```

For the Jan1 smoke, the old `velocity_streamline_contour` has `1229` global
surface objects and `67` in the North Pacific open-ocean box. Re-running with
the `spatial25_200km_ofes_literature_diagnostic` input gives `1702` global and
`114` North Pacific open-ocean objects. The streamline effective-boundary
post-process reduces this to `1543` global and `108` North Pacific open-ocean
objects. In this smoke, the no-transient panel is identical to the effective
panel because only Jan1 has been produced for this exact velocity-streamline
effective catalog; multi-day persistence requires Jan1-Jan6 or longer under the
same catalog definition.

Add `--exclude-jet-flagged` to hide objects marked by `jet_meander_flag` in the
four global overview figures. On the Jan1 smoke this leaves the old and raw
spatial velocity-streamline panels unchanged (`1229` and `1702` global objects,
respectively) because those tables do not carry the same jet flag. For the
effective-boundary panels it reduces the Jan1 global count from `1543` to
`1098`, the North Pacific open-ocean box from `108` to `70`, and the ACC box
from `379` to `261`.

The long-running same-definition Jan01-Jan19 surface pipeline is written outside
the repo under:

```text
E:\DATA\01_Eddy_correspond\02_OFES\origin_velocity_streamline_spatial25_200_effective_persistence_jan01_jan19_bg
```

It runs detection, streamline effective-boundary post-processing, and then
surface persistence labeling, with logs in the `logs` subdirectory.

Using the completed Jan01-Jan06 same-definition window, the Jan1
`--exclude-jet-flagged --exclude-transient` comparison drops the effective
catalog from `1098` to `557` global objects. North Pacific open-ocean falls
from `70` to `39`, the ACC box from `261` to `153`, NE Pacific from `25` to
`8`, and NW Pacific from `41` to `29`.

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

## Alpha-Aligned Theory Rebuild-W Coherent Composites

To diagnose the theoretical rebuild-W terms on the current OFES
`velocity_streamline_contour` result, use the coherent track composite helper.
The current recommended口径 follows Zhe's `global_ls_alpha` convention: each
coherent object-day is rotated by its vertical centerline tilt so that the
surface-to-deep center displacement points toward `+x_rot` before compositing.

```powershell
$env:PYTHONNOUSERSITE='1'
& 'D:\Util\lever\02_miniforge\Library\bin\mamba.exe' run -n OFES_detection python -m Detection_for_OFES.tools.plot_ofes_theory_rebuild_w_composite `
  --data-root F:\OFES\external_OFES2 `
  --result-root E:\DATA\01_Eddy_correspond\02_OFES\origin_streamline_cpu_jan01_jan19_life1 `
  --workers 6
```

It reads:

```text
<result-root>\shape_classification_1991_1991_hua_b3_start2_life1\shape_tracks.parquet
<result-root>\catalog\vertical_objects.parquet
<result-root>\catalog\layer_observations.parquet
<result-root>\hua_b3_start2_detection\centers_hua_style.parquet
<result-root>\hua_b3_start2_detection\structures_hua_style.parquet
```

Only `shape_class == coherent` object-days are used. They are grouped into
`NH_cyclonic`, `NH_anticyclonic`, `SH_cyclonic`, and `SH_anticyclonic`, then
rebuilt through the existing `rebuild_object_w()` formula:

```text
eta_rho = -rho'_eddy / rho_z
term1 = c_rel(z) dot grad_h(eta_rho)
term2 = -u_rel(x,y,z) dot grad_h(eta_rho)
W_rebuild = term1 + term2
```

For each object-day, `alpha_deg = -atan2(y_deep - y_surface, x_deep - x_surface)`.
If fewer than two center layers are valid, or the total displacement is below
`0.02R`, `alpha_deg=0`. The scalar theory fields are sampled in the rotated
frame using the same convention as `Zhe/composite_3d_lifecycle.py`.

The density part of this composite is intentionally derivative-safe. It
composites only `prho_for_rebuild`; it does not composite `rho_z`,
`rho_prime`, `eta_rho`, or `grad_eta`. Equivalent isopycnal depths are then
diagnosed by vertical interpolation from the composite density field. Any W
terms in this helper are legacy per-object diagnostics from `rebuild_object_w()`
and are not recomputed from a composite density gradient.

For full runs, coherent object-days are scheduled by date and daily `.dta`
memmaps are cached across workers. This reduces repeated file opens and keeps
the 10-day `prho/u/v/w` windows warm in the OS file cache. Start with
`--workers 4` to `--workers 6`; increase only if disk queue and memory pressure
remain acceptable.

The helper supports group-level resume by default, but resume only accepts
outputs carrying the derivative-safe density-composite metadata. Older outputs
that still contain `z_rho_anom`, `rho_z`, or `rho_prime` composite arrays are
rebuilt. This does not checkpoint individual object-days inside a still-running
group.

The theory composite helper also precomputes each object-day's local 10-day
sampled blocks for `prho/u/v` and the regional density-background profile.
Those cached blocks are then reused by the rebuild-W density and velocity
filters, so the same local window is not sampled repeatedly inside one object
calculation. Native-W mesoscale filtering is disabled for this theory-only
composite because native W is not plotted in the three-column output.

Outputs are written to:

```text
<result-root>\w_rebuild_diagnostics\theory_rebuild_w_coherent_alpha_aligned_by_polarity_hemisphere\
  figures\theory_rebuild_w_section_alpha_<hemisphere>_<polarity>.png
  figures\theory_rebuild_w_slices_alpha_<hemisphere>_<polarity>.png
  grids\theory_rebuild_w_composite_alpha_<hemisphere>_<polarity>.npz
  grids\theory_rebuild_w_composite_alpha_<hemisphere>_<polarity>.json
  theory_rebuild_w_alpha_group_summary.csv
  theory_rebuild_w_alpha_group_summary.json
```

The older unrotated output directory
`theory_rebuild_w_coherent_by_polarity_hemisphere` is retained only as a
comparison product.

### Single coherent tilted-density check

To inspect one original coherent OFES eddy instead of a composite, use the
single-object tilted-density diagnostic:

```powershell
$env:PYTHONNOUSERSITE='1'
& 'D:\Util\lever\02_miniforge\Library\bin\mamba.exe' run -n OFES_detection python -m Detection_for_OFES.tools.plot_single_coherent_tilt_theory `
  --data-root F:\OFES\external_OFES2 `
  --result-root E:\DATA\01_Eddy_correspond\02_OFES\origin_streamline_cpu_jan01_jan19_life1 `
  --scale-mode raw_fast
```

If `--hua-object-id` is omitted, the tool selects a coherent object-day with a
large layer count and clear vertical-center tilt. It calls the existing
`rebuild_object_w()` once, rotates the original object into the same
`global_ls_alpha` frame, and writes:

```text
<result-root>\w_rebuild_diagnostics\single_coherent_tilt_theory\
  figures\single_coherent_density_tilt_theory_<date>_<object>.png
  figures\single_coherent_w_terms_<date>_<object>.png
  figures\single_coherent_density_w_450m_slice_<date>_<object>.png
  grids\single_coherent_tilt_theory_<date>_<object>.npz
  grids\single_coherent_tilt_theory_<date>_<object>.json
  single_coherent_tilt_theory_summary.csv
  single_coherent_tilt_theory_summary.json
```

The density figure checks the first-order tilted-vortex approximation:

```text
xi = x - Xc_rot(z)
rho_hat(xi,z) = rho_prime(xi + Xc_rot(z), z)
rho_prime_odd_surface(x,z) ~= -Xc_rot(z) * d rho_even_intrinsic / d(x/R)
```

where `Xc_rot(z)` is the layer-center displacement after rotating the
surface-to-deep centerline toward `+x_rot`. The tool first recenters each layer
on its own eddy center to estimate the intrinsic even and odd density
structures, then compares the surface-centered actual odd component with the
first-order tilt term. The W figure compares `term1`, `term2`, `term1+term2`,
and layer-center-aligned OFES native `w` for the same single object. This is a
validation view, not a new detection or composite workflow.

`--scale-mode recommended` keeps the full current rebuild-W scale separation
and is slower. `--scale-mode raw_fast` uses same-day local `prho/u/v/w` fields
and is intended for quick checks of the original object geometry and
density-tilt mechanism.

For a long full run, redirect stdout/stderr to a log file:

```powershell
$env:PYTHONNOUSERSITE='1'
$log = 'E:\DATA\01_Eddy_correspond\02_OFES\origin_streamline_cpu_jan01_jan19_life1\w_rebuild_diagnostics\theory_rebuild_w_coherent_alpha_aligned_by_polarity_hemisphere\theory_rebuild_w_alpha_run.log'
& 'D:\Util\lever\02_miniforge\Library\bin\mamba.exe' run -n OFES_detection python -m Detection_for_OFES.tools.plot_ofes_theory_rebuild_w_composite `
  --data-root F:\OFES\external_OFES2 `
  --result-root E:\DATA\01_Eddy_correspond\02_OFES\origin_streamline_cpu_jan01_jan19_life1 `
  --workers 6 *> $log
```

Monitor progress from another PowerShell:

```powershell
Get-Content 'E:\DATA\01_Eddy_correspond\02_OFES\origin_streamline_cpu_jan01_jan19_life1\w_rebuild_diagnostics\theory_rebuild_w_coherent_alpha_aligned_by_polarity_hemisphere\theory_rebuild_w_alpha_run.log' -Wait -Tail 40
```

To rebuild only the density-safe `NH_cyclonic` composite, use `--groups` and a
separate log. This writes
`theory_rebuild_w_alpha_group_summary_NH_cyclonic.csv/json` and does not
overwrite the four-group summary:

```powershell
$env:PYTHONNOUSERSITE='1'
$root='E:\DATA\01_Eddy_correspond\02_OFES\origin_streamline_cpu_jan01_jan19_life1'
$out="$root\w_rebuild_diagnostics\theory_rebuild_w_coherent_alpha_aligned_by_polarity_hemisphere"
$log="$out\theory_rebuild_w_alpha_NH_cyclonic_density_safe_worker6.log"
& 'D:\Util\lever\02_miniforge\Library\bin\mamba.exe' run -n OFES_detection python -m Detection_for_OFES.tools.plot_ofes_theory_rebuild_w_composite `
  --data-root F:\OFES\external_OFES2 `
  --result-root $root `
  --output-root $out `
  --groups NH_cyclonic `
  --workers 6 `
  --no-resume *> $log
```

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
