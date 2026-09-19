# Detection_for_OFES

`Detection_for_OFES` is the OFES data-interface and OFES-specific diagnostic layer. Eddy detection is implemented in `Origin_eddy_detection`; this package exports OFES data in the required form, invokes the production surface catalog, and retains the OFES native-W/rebuild-W workflow.

## Default Surface Catalog

The only default OFES surface catalog is:

```text
Rossby-adaptive SSH/u/v filter
+ multiscale SSH seeds (3, 5, 7, 11 cells)
+ SSH/effective-contour primary discovery
+ no velocity-streamline hard gate in open ocean
+ shape and overlap QC
+ two-day minimum consecutive persistence
```

The default recovery regions are always enabled:

| Region | Longitude | Latitude |
| --- | --- | --- |
| North Pacific | 190-245E | 25-55N |
| South Pacific | 190-280E | 50-30S |
| North Atlantic | 320-330E | 25-45N |
| South Atlantic | 335-355E | 45-20S |

Production root:

```text
E:\DATA\01_Eddy_correspond\02_OFES\origin_unified_ssh_primary_target_all_open_ocean_recovery_surface_jan01_jan19
```

`final_catalog\daily_runs\YYYYMMDD` is the consumer-facing catalog. It only contains `hua_pass=True` objects after shape/overlap QC and excludes `persistence_class=transient`. Raw candidates remain under `raw_detection` for traceability.

Resume the default catalog:

```powershell
& 'D:\01_Eddy\01_Vertical_asymmetric\S-H-I-T-ocean-\Detection_for_OFES\run_default_surface_catalog.ps1'
```

Create the default Jan1 overview from the final catalog:

```powershell
$env:PYTHONNOUSERSITE='1'
& 'D:\Util\lever\02_miniforge\Library\bin\mamba.exe' run -n OFES_detection python -m Detection_for_OFES.tools.plot_latest_ssh_vector_overview
```

## OFES Data Interface

- `ofes_io.py`: `.ctl/.dta` reader.
- `validate_ofes2_sample.py`: OFES I/O smoke check.
- `tools/extract_ofes_daily.py`: daily extraction and integrity validation.
- `tools/export_origin_netcdf.py`: Origin-compatible NetCDF export.
- `tools/build_ofes_meso_filter.py`: Rossby-adaptive detection input filter.

Retained base inputs:

```text
E:\DATA\01_Eddy_correspond\02_OFES\origin_compatible_filter
E:\DATA\01_Eddy_correspond\02_OFES\origin_compatible_filter_rossby_lower_upper180
```

## OFES W Diagnostics

`run_ofes_rebuild_w.py`, `w_rebuild_config.py`, and the coherent-theory plotting tools remain OFES-specific diagnostics. They currently use `origin_streamline_cpu_jan01_jan19_life1` as their coherent-object input; that legacy catalog is retained until the vertical-extension definition for the new surface catalog is decided.

## Latest Catalog Vertical Extension

`tools.run_default_vertical_extension` extends only the final surface catalog through the existing layerwise Hua continuation. It does not re-run seed discovery, contour QC, or persistence. The optimized implementation computes each depth-layer velocity magnitude once for all active objects and writes the full-depth Rossby-filter input through disk-backed layers.

```powershell
$env:PYTHONNOUSERSITE='1'
& 'D:\Util\lever\02_miniforge\Library\bin\mamba.exe' run -n OFES_detection python -m Detection_for_OFES.tools.run_default_vertical_extension --workers 19 --resume
```

Outputs are written to:

```text
E:\DATA\01_Eddy_correspond\02_OFES\origin_unified_ssh_primary_target_all_open_ocean_vertical_jan01_jan19
```

The resulting `catalog/` and `shape_classification_1991_1991_hua_b3_start2_life1/` tables are the future input for the latest coherent rebuild-W composite. Jan19 is retained but carries its existing no-future-day persistence status and is excluded from default coherent rebuild-W selection.

## Legacy Archive

Superseded OFES smoke runs, old filter comparisons, and streamline-only surface catalogs are removed after their names, sizes, and purpose are recorded in:

```text
E:\DATA\01_Eddy_correspond\02_OFES\legacy_manifest.json
```

Origin's historical non-OFES boundary modes remain in `Origin_eddy_detection` for reproducibility, but they are not OFES entry points.

## AVISO Single-Day SLA Preview

`tools/plot_aviso_sla_day.py` archives an official AVISO Sea Views global SLA map for a selected day. The current preview uses `1993-01-01`, matching the beginning of the local META4.0 record:

```powershell
$env:PYTHONNOUSERSITE='1'
& 'D:\Util\lever\02_miniforge\Library\bin\mamba.exe' run -n OFES_detection python -m Detection_for_OFES.tools.plot_aviso_sla_day --day 1993-01-01
```

The output is written to `E:\DATA\01_Eddy_correspond\03_AVISO_altimetry_preview\19930101`. When the legacy daily DT NetCDF directory is not exposed by the AVISO account, the tool uses the official AVISO quicklook PNG and records that fallback explicitly in the manifest; it never relabels the quicklook as a decoded local NetCDF field.
