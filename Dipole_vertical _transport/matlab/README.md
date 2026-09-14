# MATLAB backend

`../meta4_core_argo_vertical_transport.py` handles CLI parsing, MATLAB script
generation, and launching MATLAB. This directory contains the real MATLAB
backend. `run_meta4_core_argo_backend.m` is now a dispatcher; the numerical work
is split into modules:

- `core/`: Argo, META, and BOA loading; history velocity matching; crossing
  collocation; QC; caches.
- `physics/`: strict-bracket isopycnal depth, BOA density anomaly,
  thermal-wind velocity, and W formulas.
- `mapping/`: Cressman objective mapping, GPU/CPU backends, support counts, and
  smoothing.
- `output/`: MAT, NetCDF, PNG, and Markdown writers.
- `diagnostics/`: retained reversal-factor helpers only.

Formal production is fixed to crossing + BOA anomaly geometry + Cressman +
thermal-wind W. Composite analysis now stops at isopycnal geometry; W terms are
computed by separate `physics/rebuild_w_from_geometry.m` and
`physics/rebuild_w_3d_from_geometry.m` modules. Early `lat_band`,
farfield/absolute background, non-Cressman mapping, `profile_diff` velocity,
fast sensitivity, and z-geometry comparison branches have been removed from the
formal entry point.

Current acceleration support:

- `--compute-device auto|cpu|gpu` selects the Cressman mapping device.
- GPU mode uses MATLAB `gpuArray` for chunked Cressman weight calculations and
  falls back to CPU if GPU execution fails.
- `--workers` is retained for QC/cache parallel work where available.

Default output is MAT/NetCDF plus PNG. Large CSV and JSON outputs are opt-in.
