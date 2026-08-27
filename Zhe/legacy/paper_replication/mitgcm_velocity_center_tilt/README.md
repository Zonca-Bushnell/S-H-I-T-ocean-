# Kuroshiou MITgcm Velocity-Center Tilt Validation

This temporary experiment builds an observation-constrained MITgcm idealized
case from the Kuroshiou representative vortex.

The scientific distinction is important:

- Yang/Xu/Li 2026 diagnose tilt mainly with temperature-anomaly or
  streamfunction-like centers after idealized MITgcm/theory analysis.
- This experiment keeps the dynamical mechanism test, but diagnoses the eddy
  center with our velocity-anomaly center definition.
- The target question is therefore not "can we reproduce the paper exactly?"
  but "does the mode-1/mode-2 propagation mechanism still explain tilt when
  eddy centers are defined by velocity geometry?"

## Workflow

1. `prepare_velocity_center_experiment.py`
   reads the ME_LIUTEX azimuthal representative vortex and radial-seed center
   diagnostics, constructs QG vertical modes from the Kuroshiou `N2` profile,
   and writes four MITgcm initial states:

   - `real`
   - `mode1`
   - `mode2`
   - `mode1_plus_mode2`

2. The generated case directories contain MITgcm namelist templates and,
   with `--write-binary`, big-endian `float64` initial fields for `U`, `V`,
   `Theta`, `Salt`, `Eta`, and bathymetry.

3. `check_mitgcm_environment.py` records whether the server has MITgcm source
   and a usable Fortran compiler.

The package stays under `src/experiments/temp` until a real MITgcm smoke run is
complete and the binary field layout has been verified.

