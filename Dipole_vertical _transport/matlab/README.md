# MATLAB backend

`../meta4_core_argo_vertical_transport.py` handles CLI parsing, MATLAB script
generation, and launching MATLAB. This directory contains the real MATLAB
backend. `run_meta4_core_argo_backend.m` is now a dispatcher; the numerical work
is split into modules:

- `core/`: Argo, META, and BOA loading; history velocity matching; crossing
  collocation; QC; caches.
- `physics/`: strict-bracket `z_rho`, BOA density anomaly, thermal-wind velocity,
  and W formulas.
- `mapping/`: Cressman objective mapping, GPU/CPU backends, support counts, and
  smoothing.
- `output/`: MAT, NetCDF, PNG, and Markdown writers.
- `diagnostics/`: 2D sensitivity, reversal-factor tests, z-geometry comparison,
  and reference-like absolute-density isopycnal-slope experiments.

Formal production is fixed to crossing + BOA anomaly geometry + Cressman +
thermal-wind W. Early `lat_band`, farfield/absolute background, non-Cressman
mapping, and `profile_diff` velocity branches have been removed from the formal
entry point.

Current acceleration support:

- `--compute-device auto|cpu|gpu` selects the Cressman mapping device.
- GPU mode uses MATLAB `gpuArray` for chunked Cressman weight calculations and
  falls back to CPU if GPU execution fails.
- Fast 2D sensitivity builds the crossing matching/QC cache once per polarity,
  then reuses it across parameter combinations.
- CPU mode can parallelize sensitivity parameter combinations with MATLAB
  `parfor`; GPU mode avoids spawning competing workers for one GPU.

Default output is MAT/NetCDF plus PNG. Large CSV and JSON outputs are opt-in.
