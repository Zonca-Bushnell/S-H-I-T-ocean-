# Detection_for_OFES

## Official Workflow

There is one production scientific profile:

```text
eta_hp500_geometry_vertical_v1
```

It is intentionally narrow and traceable:

```text
SSH              OFES native eta only; never pressur or an MSS anomaly
surface          daily Gaussian eta - LP500 km(eta)
detection        no tile cap, SSH-primary contour
surface QC       geometry + same-day, same-polarity overlap
excluded         streamline hard gate, persistence, tracking, rebuild-W
vertical         tangent 45 degrees / fraction >= 0.35, then relaxed
                 near-closed-streamline fallback
strict core      section-bipolar classification only; never changes acceptance
native W         NH cyclonic strict-core; native W; unrotated pointwise mean
```

Run or resume through the sole launcher:

```powershell
& 'D:\01_Eddy\01_Vertical_asymmetric\S-H-I-T-ocean-\Detection_for_OFES\start_ofes_default_run.ps1' -Resume
```

For a partial, auditable run:

```powershell
D:\Util\lever\02_miniforge\envs\OFES_detection\python.exe -m Detection_for_OFES.workflows.default_pipeline `
  --profile eta_hp500_geometry_vertical_v1 `
  --start 1991-01-01 --end 1991-01-19 --resume `
  --stages surface-inputs,surface-filter,raw-detection,geometry-qc
```

All official results live only below:

```text
E:\DATA\01_Eddy_correspond\02_OFES\Fromthebeginning\runs\
  eta_hp500_geometry_vertical_v1\YYYYMMDD_YYYYMMDD\
    00_manifest\
    01_surface\{eta_inputs,highpass_500km,raw_detection,geometry_qc}\
    02_velocity\highpass_500km_full105\
    03_vertical\tangent45_fraction35_then_near_closed_relaxed\
    04_section_bipolar\
    05_native_w\nh_cyclonic_strict_core\
    06_reports\
    logs\
```

Every stage has a date-level input/profile fingerprint under `00_manifest/stages`.
`--resume` skips only a matching complete day; it refuses to reuse an output
whose profile or input identity changed. `hua_object_id` remains the object
primary key. Official outputs add `run_id`, `profile_id`,
`vertical_profile_id`, date, and depth index where applicable.

Default execution is FFT horizontal convolution with compression level 1,
up to 8 date workers for surface stages, 2 for full-depth velocity and vertical
stages, depth-major continuation, no default object voxels, and direct readonly
DTA memmaps for Native W. `--stage-raw` remains an explicit network-storage
fallback for W sampling.

## Historical And Experiments

Existing results under `Fromthebeginning/04_Vertical`, `05_TEMP`, threshold
sweeps, three-kernel comparisons, MSS comparisons, Rossby paths, MATLAB bridge,
and rebuild-W are historical or experimental. They are not moved, deleted, or
rewritten by the official workflow. Their registry is in
[`legacy/HISTORICAL_RUNS.json`](legacy/HISTORICAL_RUNS.json).

- `tools.run_default_geometry_vertical`, `start_default_geometry_vertical.ps1`,
  and `run_default_surface_catalog.ps1` are compatibility forwarders.
- `tools.run_default_vertical_extension --legacy-rossby-workflow` is explicit
  legacy reproduction only.
- `run_ofes_rebuild_w.py` remains a separate legacy workflow and cannot read
  official-profile objects implicitly.
- Research commands are documented under `experiments/`; they never appear in
  the official launch instructions.
