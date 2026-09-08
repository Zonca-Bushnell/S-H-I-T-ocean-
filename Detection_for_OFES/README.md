# Detection for OFES2

This folder is a self-contained OFES2 eddy-detection workspace. The current
scope is:

- organize the OFES2 external data tree into a stable layout;
- parse GrADS-style `.ctl` metadata;
- read OFES `.dta` binary files as big-endian float32 arrays;
- build the sea-surface-height field used by the Matlab example:
  `SSH = eta - (pressur - 1000)`, in centimeters;
- validate one daily sample;
- build Jan 1991 monthly anomaly fields;
- run the first diagnostic Hua hybrid detection pass.

## Expected Data Root

The detected local OFES2 data root is:

```text
F:\OFES\external_OFES2
```

The similarly named path `F:\OFES\external\_OFES2` was checked and does not
exist on this machine.

## Target Data Layout

After running the organizer with `--apply`, the data root should look like:

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
      manifest_before.json
      manifest_before.csv
      manifest_after.json
      manifest_after.csv
      organize_plan.json
      organize_plan.csv
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
  docs/
    SSH说明.txt
    read_data.m
```

The formal hot path is daily-only: after the explicit `extract` stage, monthly
means and detection require extracted `.dta` files and do not silently fall back
to `.tgz`.

## Commands

Run from the repository root in the OFES worktree:

```powershell
python -m Detection_for_OFES.organize_ofes2 --root F:\OFES\external_OFES2 --dry-run
python -m Detection_for_OFES.organize_ofes2 --root F:\OFES\external_OFES2 --apply
python -m Detection_for_OFES.validate_ofes2_sample --root F:\OFES\external_OFES2 --date 1991-01-01
python -m Detection_for_OFES.run_hua_ofes2 --data-root F:\OFES\external_OFES2 --output-root E:\DATA\01_Eddy_correspond\02_OFES --start 1991-01-01 --end 1991-01-31 --stages metadata,extract
python -m Detection_for_OFES.run_hua_ofes2 --data-root F:\OFES\external_OFES2 --output-root E:\DATA\01_Eddy_correspond\02_OFES --start 1991-01-01 --end 1991-01-31 --stages means --max-depth-layers 105
python -m Detection_for_OFES.run_hua_ofes2 --data-root F:\OFES\external_OFES2 --output-root E:\DATA\01_Eddy_correspond\02_OFES --start 1991-01-01 --end 1991-01-31 --stages detect --max-depth-layers 105
```

Use the dedicated `OFES_detection` environment for all commands. Install
dependencies with mamba, not pip:

```powershell
$env:CONDA_PKGS_DIRS="E:\DATA\01_Eddy_correspond\02_OFES\mamba_pkgs_cache"
D:\Util\lever\02_miniforge\Library\bin\mamba.exe env update -n OFES_detection -f Detection_for_OFES\environment.yml
```

The scripts require `numpy`; preview PNG writing also requires `Pillow`. The
Hua runner also uses `pandas` and writes parquet when `pyarrow` is available.

The validator writes summary files under:

```text
Detection_for_OFES/outputs/ofes2_sample_validation/
```

The OFES Hua runner writes its stage outputs under:

```text
E:\DATA\01_Eddy_correspond\02_OFES
  metadata/
  anomaly_jan1991/
  detection_hua_global_jan1991/
  figures/
  logs/
```

For the full 105-layer January run, first materialize the daily `u/v` files:

```powershell
python -m Detection_for_OFES.run_hua_ofes2 --data-root F:\OFES\external_OFES2 --output-root E:\DATA\01_Eddy_correspond\02_OFES --start 1991-01-01 --end 1991-01-31 --stages extract --max-depth-layers 105 --extract-workers 2
```

Then run means. The optimized mean reads each daily 3-D file once with memmap
and Fortran-order sequential chunks:

```powershell
python -m Detection_for_OFES.run_hua_ofes2 --data-root F:\OFES\external_OFES2 --output-root E:\DATA\01_Eddy_correspond\02_OFES --start 1991-01-01 --end 1991-01-31 --stages means --max-depth-layers 105 --mean-strategy volume
```

Use `--mean-strategy depth_chunk --mean-depth-chunk 8` only if the machine does
not have enough memory for the volume accumulator.

Current local data status: `u` Jan 1991 is extracted for 31/31 days. `v` Jan
1991 is extracted for 19/31 days; `v.01.20.1991.dta` through
`v.01.31.1991.dta` were not found in the visible data tree. The archives named
`v_y1991_jan2.tgz` and `v_y1991_jan3.tgz` contain February/March member names,
not January 20-31.

## Grid Notes

OFES2 variables are stored on slightly different horizontal grids:

- `eta`, `pressur`, `temp`, `salinity`, and `prho`: longitude starts at `0.05`,
  latitude starts at `-75.95`.
- `u`, `v`, and `w`: longitude starts at `0.10`, latitude starts at `-75.90`.

This staggered-grid difference is preserved in the parsed metadata. Future
detection code must regrid or explicitly choose the target grid before mixing
velocity and scalar fields.

## OFES Hua Hybrid V1

The v1 detection runner is diagnostic rather than a strict Kuroshio production
clone. It uses `raw_minus_jan_mean_diagnostic`: each daily field is compared to
the January 1991 monthly mean because the available first target period is too
short for the production 30-180 day bandpass.

The runner keeps the Hua b3 logic: SSH extrema seed, velocity speed minimum,
circular velocity verification, boundary-monotonic rotation, and strict
contiguous vertical extension. Velocity is read from OFES `u/v` on their native
staggered grid and the scalar SSH seed is mapped onto the nearest velocity grid
point before checks.

The OFES runner records both the original grid-cell weak-speed center and an
OFES-aware local refined center. The default refinement spacing is `0.025`
degrees, one quarter of the OFES `0.1` degree grid, with a two-cell local
window. Refined fields are written as `center_lon_refined`,
`center_lat_refined`, `center_i_refined`, `center_j_refined`,
`refined_offset_km`, `refined_ok`, and `subgrid_fit_quality`.

## OFES Panel Family Diagnostics

The OFES panel-family diagnostic is an object-day version of the original eddy
panel family. The default renderer is a local `matplotlib` implementation that
matches the reference `Zhe/src/post/original_eddy_panels.py` layout and visual
style: 5x6 gridspec, `coolwarm`/`RdBu_r` palettes, original-style colorbars,
contours, white quiver arrows, jump boxes, and the right-side section modes. It
does not import or run the original `Zhe/src` programs.

```powershell
python -m Detection_for_OFES.plot_ofes_panel_family --data-root F:\OFES\external_OFES2 --result-root E:\DATA\01_Eddy_correspond\02_OFES\available_jan01_jan19_refined_ofes_grid --max-examples 3 --right-panel-mode normal_horizontal_velocity
```

Outputs are written to:

```text
E:\DATA\01_Eddy_correspond\02_OFES\available_jan01_jan19_refined_ofes_grid\figures\panel_family
```

The default selection ranks object-days by vertical center jump strength and
plots the top examples. Use `--hua-object-id 19910101_00001` to render a
specific identified object. OFES v1 does not yet include lifecycle tracking, so
panel 7 shows the refined vertical center trajectory within the selected
object-day rather than a multi-day track.

The right-side panels support the same diagnostic family as the reference code:
`normal_horizontal_velocity`, `horizontal_speed`, and
`signed_horizontal_speed`. If the local matplotlib stack is broken, use
`--backend pillow` for an emergency static fallback.
