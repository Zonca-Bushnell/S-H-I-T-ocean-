# Detection_for_OFES

`Detection_for_OFES` is the sole maintained OFES eddy package. It has one
scientific default: `eta_hp500_geometry_vertical_v1`.

```text
SSH          native OFES eta
filter       daily Gaussian eta - LP500 km(eta), FFT implementation
surface      no tile cap; SSH-primary contour
QC           geometry plus same-day/same-polarity overlap
excluded     streamline hard gate, persistence and rebuilt W
vertical     tangent 45 deg / fraction 0.35, then relaxed near-closed fallback
strict-core  section-bipolar classification only
W            native OFES W, layer-center aligned, unrotated pointwise mean
```

## Run

There is one user entry point:

```powershell
& 'D:\01_Eddy\01_Vertical_asymmetric\S-H-I-T-ocean-\Detection_for_OFES\start_ofes_default_run.ps1' -Resume
```

The equivalent Python command is:

```powershell
D:\Util\lever\02_miniforge\envs\OFES_detection\python.exe -m Detection_for_OFES run `
  --profile eta_hp500_geometry_vertical_v1 `
  --start 1991-01-01 --end 1991-01-19 --resume
```

Tracking and shape are optional, non-filtering post-processors:

```powershell
... -m Detection_for_OFES run --resume --with-tracking --with-shape
```

`--stages` exists only for recovery and debugging. Scientific thresholds come
from the nested dataclasses in `profiles.py`; stage CLIs receive explicit
values and do not define a competing workflow profile.

## Layout

```text
Fromthebeginning/runs/eta_hp500_geometry_vertical_v1/layout_v2/YYYYMMDD_YYYYMMDD/
  00_manifest/
  01_surface/{eta_inputs,highpass_500km,raw_detection,geometry_qc}/
  02_velocity/highpass_500km_full105/
  03_vertical/tangent45_fraction35_then_near_closed_relaxed/
  04_classification/section_bipolar/
  05_tracking/
  06_shape/
  07_composites/native_w/
  08_reports/
  logs/
```

Every stage records its input identity, code/profile fingerprint, parameters,
outputs and completion state. Resume invalidates only the changed date and its
downstream work. `hua_object_id` remains the object primary key throughout.

## Package Map

- `core`: seed/contour/Hua geometry, vertical numerical API, tracking, shape
- `filters`: cached FFT Gaussian 500 km filtering
- `io`: OFES DTA/CTL, NetCDF, tables and native-W layer alignment
- `stages`: formal surface, QC, vertical, classification and optional stages
- `composite`: the maintained native-field accumulator and Native W composite
- `datasets`: OFES/prho download and climatology builders
- `workflows`: run context, contracts, fingerprints and orchestration

Historical output directories remain read-only and are listed in
`legacy/HISTORICAL_RUNS.json`. Historical runnable source is intentionally not
kept in this repository; use the recorded Git commit to reproduce it.
