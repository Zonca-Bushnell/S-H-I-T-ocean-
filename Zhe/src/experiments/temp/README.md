# Temporary Representative Experiments

This package holds research scripts that are not part of the production
`src.eddy_pipeline` or `src.post` interfaces.

## Aggregate-Product Stirring

`run_aggregate_product_stirring.py` is now a compatibility wrapper around
`src.post.transport`. The formal post-processing interface is:

```bash
python -m src.post.cli build-transport --shape coherent --orientation turned
```

The production diagnostic aggregates object-day products rather than
multiplying first-order mean fields:

```text
product_mean = mean(v_rot * tracer)
mean_product = mean(v_rot) * mean(tracer)
covariance = product_mean - mean_product
```

It does not calculate trapping and does not report geographic northward
transport.

## Azimuth-Preserved Representative Vortex

`run_azimuthal_representative_vortex.py` compares the current radial
representative vortex with a new azimuth-preserved composite. It keeps
`tau x depth x r x phi` velocity fields after `global_ls_alpha` alignment, so
systematic crescent/open-ring speed structures can survive the composite.
Use `--shapes` to run the same ME_LIUTEX-style azimuth-preserved composite for
`upright_like` or another accepted shape class.

The current production representative vortex stores radial profiles such as
`continuous_radial_psi_profiles.parquet`; those outputs are useful for radial
structure and E-P diagnostics, but they cannot display azimuthal crescent
features because the azimuth dimension has already been averaged out.

## Shape Representative Bundle

`run_shape_representative_bundle.py` is a small orchestration entrypoint for the
current default experiment contract: a representative-vortex command should
produce both the structure composite and the aggregate-product stirring
diagnostics, but in separate directories.

For a shape set such as `upright_like`, it writes:

```text
result_upright_like/
  representative_vortex_radial_seed/
  representative_vortex_me_liutex/
  aggregate_product_stirring/
```

The bundle does not change the scientific computations; it calls the radial
representative vortex entrypoint, the azimuth-preserved ME_LIUTEX structure
composite, and the formal `src.post.transport` aggregate-product diagnostic.

## Hua Boundary-Monotonic Rotation

`run_hua_boundary_monotonic_rotation_compare.py` compares the current
Kuroshiou Hua/Nencioli `b3_start2` detector with the same detector plus one
extra hard constraint: velocity vectors sampled on the accepted circular
boundary must rotate monotonically around the candidate center.

The production detector keeps this option disabled. This experiment writes a
side-by-side 1993-01-01 to 1993-01-07 comparison of layer detections, 3D axes,
and feature tracking using the 30-180 day bandpass Filter fields.
