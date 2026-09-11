# MATLAB Hua Kernels

This folder contains MATLAB acceleration kernels for the Origin eddy detection
experiments. Python remains responsible for IO, vertical continuation, output
tables, tracking, catalog, and shape classification; the MATLAB backend can
execute the Hua boundary checks through MATLAB Engine for Python.

## Kernels

- `origin_precompute_candidates.m`
  - Existing matrix-friendly surface SSH candidate precompute.
  - Writes `candidates_YYYYMMDD.csv` caches consumed by Python.

- `hua_circle_batch_kernel.m`
  - Fixed-radius circular Hua boundary checks for one velocity layer.
  - Intended to reproduce `circle_strict_original` diagnostics.

- `velocity_streamline_boundary_kernel.m`
  - Closed velocity-streamline boundary diagnostics for one velocity layer.
  - Returns continuous tangent, closure, winding, symmetry, and failure fields
    so Python can sweep threshold matrices without rereading NetCDF.

- `refine_speed_min_batch_kernel.m`
  - Batch pre-Hua subgrid speed-minimum refinement for one velocity layer.
  - Mirrors Python's quadratic fit to `speed^2` around each integer velocity
    minimum and returns refined float centers in MATLAB 1-based coordinates.
  - The Python backend subtracts one before writing its normal zero-based
    diagnostic indices.

- `hua_day_batch_kernel.m`
  - Day-level vertical continuation kernel.
  - Receives the whole daily `u/v` cube plus surface seed centers, advances
    active centers through depth, performs local weak-center search, pre-Hua
    refinement, boundary checks, strict stop-after-failure, and returns one row
    per attempted candidate-layer.
  - This is the production MATLAB backend path because it removes cross-layer
    MATLAB Engine calls from the Python loop.

- `hua_day_batch_to_mat.m`
  - Thin bridge wrapper around `hua_day_batch_kernel.m`.
  - Writes column arrays to a temporary `.mat` file, avoiding large
    `jsonencode -> MATLAB Engine -> json.loads` transfers.
  - Python deletes the temporary bridge file by default after reading it.

- `streamline_gpu_batch_kernel.m`
  - Batch/GPU-oriented closed-streamline kernel. It traces all active centers,
    all start angles, and both directions with a fixed maximum step count and
    active/done masks, so the integration stage runs as batched array
    operations.
  - It is the production MATLAB streamline backend. Without
    `--matlab-use-gpu`, it runs the same batch algorithm on CPU arrays; with
    `--matlab-use-gpu`, it runs the integration and contour-scoring arrays on
    `gpuArray` where MATLAB supports those operations.

- `benchmark_streamline_batch_kernel.m`
  - Compares `velocity_streamline_boundary_kernel.m`,
    `streamline_gpu_batch_kernel.m` on CPU, and
    `streamline_gpu_batch_kernel.m` on GPU for the same layer-center input.

## Smoke Example

Run from the repository root:

```powershell
& 'D:\Util\Ma\01_Matlab\bin\matlab.exe' -batch "addpath('D:\01_Eddy\01_Vertical_asymmetric\Original_eddy_detection_worktree\Origin_eddy_detection\matlab'); [yy,xx]=ndgrid(-40:40,-40:40); r=hypot(xx,yy); u=-yy./max(r,1); v=xx./max(r,1); rows=hua_circle_batch_kernel(u,v,[41,41],struct('start_radius_cells',3,'max_radius_cells',8)); disp(rows.circle_passed); rows2=velocity_streamline_boundary_kernel(u,v,[41,41],3:8,struct()); disp(rows2.streamline_closed);"
```

## Current Status

The kernels have same-day/same-layer parity checks against the Python reference
for `hua_pass`, accepted radius, and continuous diagnostics. The main Python CLI
exposes `--hua-backend python|matlab`; the `matlab` backend starts one MATLAB
Engine session per detection worker, loads each daily `u/v` cube once, and calls
these kernels for the layer-center boundary checks.
With `--matlab-use-gpu`, the daily `u/v` cube is staged on the GPU once and
reused by all depth-layer kernel calls for that day.

The production MATLAB backend now uses `hua_day_batch_kernel.m`: Python sends
one day of velocity fields and the surface seed centers once, and MATLAB runs
the vertical continuation internally before returning layer rows.

Current benchmark on the 2018-01-01 ACC surface sample with 256 centers:

```text
reference CPU: 1.38 s
batch CPU:     1.18 s
batch GPU:     5.37 s
```

All three variants matched `hua_pass`, accepted radius, and first hard failure.
The batch CPU variant is modestly faster in the standalone surface-layer
benchmark, while the larger practical gain comes from calling MATLAB once per
depth layer instead of once per center. The GPU double-precision variant is also
numerically aligned, but this 256-center layer sample is still too small to
overcome GPU transfer and launch overhead. Use `--matlab-use-gpu` for controlled
GPU experiments; leave it off when the CPU batch path is faster.

Day-level smoke timings on `2018-01-01`, 256 candidates, 0-100 m:

```text
day-batch CPU: 18.36 s, 2040 rows, 1838 pass
day-batch GPU: 68.58 s, 2040 rows, 1838 pass
day-batch CPU + slim MAT bridge: 20.01 s, 2040 rows, 1838 pass
```

CPU and GPU produced identical pass/radius/failure/refined-center outputs, but
GPU remains slower for this branch-heavy streamline workload. Use CPU day-batch
as the current default acceleration path. The MAT bridge is the default
MATLAB-to-Python transfer path; it is more inspectable and avoids Engine JSON
string transfer, although this small smoke remains slightly faster with the old
JSON bridge.

Before launching a MATLAB-backed run, verify the active Python environment can
import MATLAB Engine:

```powershell
& 'D:\Util\lever\02_miniforge\condabin\mamba.bat' run -n eddy_detection python -c "import matlab.engine; print('matlab engine ok')"
```
