# Detection_for_OFES

OFES-specific I/O, SSH-contour detection orchestration, vertical continuation,
and native-W composites. Generic detection remains in `Origin_eddy_detection`.

## Current Default

`eta_hp500_geometry_vertical` is the only active OFES research profile.

```text
SSH              OFES eta free-surface height only
surface filter   Gaussian field - LP_500 km(field), daily field
candidates       multiscale SSH seeds, no tile cap, SSH-primary contours
surface QC       geometry + same-day overlap; no streamline hard gate
persistence      disabled
vertical         local 2-cell continuation, section-bipolar tie-break,
                 tangent tolerance 45 degrees and fraction >= 0.35
```

`eta` already is the OFES free surface. Production code must not subtract
`pressur`; `eta - (pressur - 1000)` is historical diagnostic-only.

Start or resume Jan01-Jan19 with 19 workers:

```powershell
& 'D:\01_Eddy\01_Vertical_asymmetric\S-H-I-T-ocean-\Detection_for_OFES\start_default_geometry_vertical.ps1'
```

Outputs:

```text
E:\DATA\01_Eddy_correspond\02_OFES\Fromthebeginning\04_Vertical\eta_highpass_500km_no_tilecap_ssh_geometry_section_bipolar_jan01_jan19
```

`hua_object_id` is the immutable surface source ID. All derived tables retain
it; any numeric tracking ID must also retain `source_hua_object_id`.

## Interfaces

- `profiles.py`: named workflow definitions and all default scientific values.
- `tools.build_ofes_eta_surface_inputs`: one-time native `eta` to velocity-grid
  input construction; surface `u/v` are copied unchanged.
- `tools.run_default_geometry_vertical`: the only default end-to-end runner.
- `tools.run_ofes_isopycnal_composite`: shared native `prho`/`w` composite
  engine; specialty composites call this engine rather than rebuilding it.

## Historical And Experiments

- `tools.run_default_vertical_extension --legacy-rossby-workflow` is retained
  only for the former Rossby/persistence catalog.
- eta-MSS, three-kernel, pressure-adjusted and persistence comparisons are
  historical experiments, not production commands.
- Hua center/gate sensitivity scripts live under `tools.experiments.vertical`;
  compatibility entry points are not default launchers.
- Legacy rebuild-W still consumes `origin_streamline_cpu_jan01_jan19_life1`
  until its object definition is migrated independently.
