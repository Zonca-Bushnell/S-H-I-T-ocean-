# OFES Experiments

This namespace documents non-default research paths. They are intentionally
separate from `workflows.default_pipeline` and require an explicit experiment
choice in their own launcher or command.

- `surface/`: MSS baselines, Rossby and kernel comparisons, seed/QC sweeps.
- `vertical/`: tangent, near-closed streamline and direction-gate sensitivity.
- `composite/`: alternative W/density composites and kernel comparisons.

Existing tools remain in `Detection_for_OFES/tools` during the compatibility
window because their historical command paths are recorded in published logs.
They are not official defaults and may not be called from the canonical profile.
