# MATLAB backend

This folder contains the MATLAB execution backend used by
`../meta4_core_argo_vertical_transport.py`.

The Python entry point keeps command-line parsing, temporary script generation,
and JSON-to-NPZ post-processing. The heavy data workflow is implemented in:

- `run_meta4_core_argo_backend.m`

That backend reads the Argo, META4.0, BOA, and historical velocity `.mat` files,
performs the matching/QC/composite calculations, and writes CSV/JSON/PNG outputs.
It is stored as a real MATLAB source file so the numerical workflow can be
inspected and edited directly instead of being hidden inside a long Python string.

Current acceleration support:

- `--compute-device auto|cpu|gpu` selects the Cressman mapping device.
- GPU mode uses MATLAB `gpuArray` for chunked Cressman weight calculations and
  falls back to CPU if GPU execution fails.
- Fast 2D sensitivity mode builds the 20N matching/QC cache once per polarity,
  then reuses it across parameter combinations.
- CPU mode can parallelize sensitivity parameter combinations with MATLAB
  `parfor`; GPU mode avoids spawning competing workers for one GPU.
