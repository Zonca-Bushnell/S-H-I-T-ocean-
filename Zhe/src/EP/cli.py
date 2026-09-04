from __future__ import annotations

import argparse
from pathlib import Path

from .contracts import (
    AXIS_SOURCES,
    BUOYANCY_SOURCES,
    CURVED_TUBE_MODES,
    DEFAULT_FULL_OUTPUT_ROOT,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_RESULT_ROOT,
    DEFAULT_SHAPE_OUTPUT_NAME,
    EPFluxConfig,
    ORIENTATIONS,
    default_me_liutex_root,
    default_radial_seed_root,
)


def _config_from_args(args: argparse.Namespace) -> EPFluxConfig:
    result_root = Path(args.result_root)
    me_root = Path(args.me_liutex_root) if args.me_liutex_root else default_me_liutex_root(
        result_root=result_root,
        shape_output_name=args.shape_output_name,
        orientation=args.orientation,
    )
    radial_root = Path(args.radial_seed_root) if args.radial_seed_root else default_radial_seed_root(
        result_root=result_root,
        shape_output_name=args.shape_output_name,
    )
    return EPFluxConfig(
        me_liutex_root=me_root,
        radial_seed_root=radial_root,
        output_dir=Path(args.output_dir),
        orientation=args.orientation,
        axis_source=args.axis_source,
        tau=args.tau,
        reference_lat=args.reference_lat,
        constant_n2=args.constant_n2,
        buoyancy_source=args.buoyancy_source,
        curved_tube_mode=args.curved_tube_mode,
        large_curvature_threshold=args.large_curvature_threshold,
        shape_label=args.shape_label,
        run_label=args.run_label,
    )


def cmd_build_smoke(args: argparse.Namespace) -> int:
    config = _config_from_args(args)
    config.validate_contract()
    if args.dry_run:
        print("EP smoke dry-run")
        for key, value in config.manifest().items():
            print(f"{key}: {value}")
        print("mode: classic EP + tilted EP + curved-tube EP QG approximation")
        print(f"buoyancy_source: {config.buoyancy_source}")
        print(f"curved_tube_mode: {config.curved_tube_mode}")
        print(f"large_curvature_threshold: {config.large_curvature_threshold}")
        return 0
    from .diagnostics import build_smoke

    outputs = build_smoke(config, n2_profile=args.n2_profile)
    print("EP smoke outputs:")
    for key, path in outputs.items():
        print(f"{key}: {path}")
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    from .diagnostics import compare_classic_and_curved

    outputs = compare_classic_and_curved(Path(args.output_dir))
    for key, path in outputs.items():
        print(f"{key}: {path}")
    return 0


def cmd_explain_contract(_: argparse.Namespace) -> int:
    print(
        "\n".join(
            [
                "# EP / Curved-Tube EP Contract",
                "",
                "Default scientific target:",
                "  Hua b3_start2 + 30-180d bandpass + boundary-monotonic + strict-contiguous",
                "  + life30 + coherent-only + ME_LIUTEX azimuth-preserved + global_ls_alpha.",
                "",
                "Axis sources:",
                "  radial_seed: object-center composite axis; this is the default.",
                "  composite_hua_refined: Hua-like refined center re-detected on the composite velocity field.",
                "",
                "Classic EP:",
                "  F_n = -rho0 <u_s' u_n'>",
                "  F_z = rho0 f0 <u_n' b'> / N2",
                "  Default b' comes from thermal-wind inversion of geostrophic velocity shear.",
                "",
                "Tilted EP:",
                "  uses ordinary vertical derivative plus an explicit axis-tilt correction term.",
                "",
                "Curved-tube EP first version:",
                "  exposes metric/Jacobian/Christoffel audit terms on a local-axis framework.",
                "  Christoffel remains a QG first-order approximation, not the final full tensor theory.",
                "",
                "Material-volume EP validation:",
                "  uses a Cartesian coherent-volume mask on the representative vortex field.",
                "  It reports R_ij, B_i, P_i, centroid drift, and boundary-leakage proxies.",
                "  It is the large-curvature validation route; it does not require thin-tube metric validity.",
                "",
                "Boundaries:",
                "  src.EP does not import src.utils.ep_flux as an implementation dependency.",
                "  src.EP does not overwrite representative vortex npz/catalog files.",
            ]
        )
    )
    return 0


def cmd_run_lifecycle_validation(args: argparse.Namespace) -> int:
    from .lifecycle import request_from_args, run_lifecycle_validation

    request = request_from_args(args)
    outputs = run_lifecycle_validation(request)
    if args.dry_run:
        return 0
    print("EP lifecycle validation outputs:")
    for key, path in outputs.items():
        print(f"{key}: {path}")
    return 0


def cmd_run_material_volume_validation(args: argparse.Namespace) -> int:
    from .material_volume import request_from_args, run_material_volume_validation

    request = request_from_args(args)
    outputs = run_material_volume_validation(request)
    if args.dry_run:
        return 0
    print("Material-volume EP validation outputs:")
    for key, path in outputs.items():
        print(f"{key}: {path}")
    return 0


def cmd_run_material_boundary_validation(args: argparse.Namespace) -> int:
    from .material_volume import request_from_args, run_material_volume_validation

    request = request_from_args(args)
    outputs = run_material_volume_validation(request)
    if args.dry_run:
        return 0
    print("Material-volume dynamic-boundary validation outputs:")
    for key, path in outputs.items():
        print(f"{key}: {path}")
    return 0


def cmd_run_object_material_boundary_validation(args: argparse.Namespace) -> int:
    if args.dry_run:
        print("Object-level material-boundary validation dry-run")
        print(f"result_root: {args.result_root}")
        print(f"filter_root: {args.filter_root}")
        print(f"output_root: {args.output_root}")
        print(f"shapes: {args.shapes}")
        print(f"orientations: {args.orientations}")
        print(f"boundary_mode: {args.boundary_mode}")
        print(f"boundary_budget: {args.boundary_budget}")
        print(f"max_tracks_per_shape: {args.max_tracks_per_shape}")
        print(f"max_objectdays: {args.max_objectdays}")
        return 0
    from .object_material_boundary import request_from_args, run_object_material_boundary_validation

    request = request_from_args(args)
    outputs = run_object_material_boundary_validation(request)
    if args.dry_run:
        return 0
    print("Object-level material-boundary validation outputs:")
    for key, path in outputs.items():
        print(f"{key}: {path}")
    return 0


def cmd_run_object_material_coherence_validation(args: argparse.Namespace) -> int:
    if args.dry_run:
        from .material_coherence import request_from_args

        request = request_from_args(args)
        from .material_coherence import run_object_material_coherence_validation

        run_object_material_coherence_validation(request)
        return 0
    from .material_coherence import request_from_args, run_object_material_coherence_validation

    request = request_from_args(args)
    outputs = run_object_material_coherence_validation(request)
    print("Object material-coherence EP validation outputs:")
    for key, path in outputs.items():
        print(f"{key}: {path}")
    return 0


def cmd_run_object_material_geodesic_validation(args: argparse.Namespace) -> int:
    if args.dry_run:
        from .material_geodesic import request_from_args, run_object_material_geodesic_validation

        request = request_from_args(args)
        run_object_material_geodesic_validation(request)
        return 0
    from .material_geodesic import request_from_args, run_object_material_geodesic_validation

    request = request_from_args(args)
    outputs = run_object_material_geodesic_validation(request)
    print("Object material-geodesic EP validation outputs:")
    for key, path in outputs.items():
        print(f"{key}: {path}")
    return 0


def cmd_run_core_shell_ep_validation(args: argparse.Namespace) -> int:
    from .core_shell import request_from_args, run_core_shell_ep_validation

    request = request_from_args(args)
    outputs = run_core_shell_ep_validation(request)
    if args.dry_run:
        return 0
    print("Core-shell EP validation outputs:")
    for key, path in outputs.items():
        print(f"{key}: {path}")
    return 0


def _add_material_arguments(parser: argparse.ArgumentParser, *, output_root: str, boundary_mode: str) -> None:
    from .dynamic_boundary import BOUNDARY_MODES
    from .material_volume import BOUNDARY_BUDGETS

    parser.add_argument("--result-root", default=str(DEFAULT_RESULT_ROOT))
    parser.add_argument("--output-root", default=output_root)
    parser.add_argument("--shapes", default="coherent,upright_like")
    parser.add_argument("--axis-sources", default="radial_seed")
    parser.add_argument("--orientations", default="turned")
    parser.add_argument("--buoyancy-sources", default="thermal_wind")
    parser.add_argument(
        "--tau-values",
        default="",
        help="Comma-separated tau values. Empty means all tau nodes from the representative npz.",
    )
    parser.add_argument("--reference-lat", type=float, default=30.0)
    parser.add_argument("--constant-n2", type=float, default=2.0e-5)
    parser.add_argument("--n2-profile", default="auto")
    parser.add_argument("--core-radius-over-R", type=float, default=1.5)
    parser.add_argument("--speed-core-quantile", type=float, default=0.45)
    parser.add_argument("--pv-core-quantile", type=float, default=0.70)
    parser.add_argument("--min-mask-fraction", type=float, default=0.01)
    parser.add_argument("--boundary-mode", choices=BOUNDARY_MODES, default=boundary_mode)
    parser.add_argument("--boundary-budget", choices=BOUNDARY_BUDGETS, default="edge_proxy")
    parser.add_argument("--active-contour-iterations", type=int, default=12)
    parser.add_argument("--leakage-weight", type=float, default=1.0)
    parser.add_argument("--smoothness-weight", type=float, default=0.08)
    parser.add_argument("--containment-weight", type=float, default=0.35)
    parser.add_argument("--area-weight", type=float, default=0.12)
    parser.add_argument("--vertical-continuity-weight", type=float, default=0.18)
    parser.add_argument("--time-continuity-weight", type=float, default=0.08)
    parser.add_argument("--levelset-sigma-cells", type=float, default=1.0)
    parser.add_argument("--min-core-retention", type=float, default=0.75)
    parser.add_argument("--min-area-fraction", type=float, default=0.15)
    parser.add_argument("--max-area-fraction", type=float, default=0.65)
    parser.add_argument("--skip-missing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")


def _add_object_boundary_arguments(parser: argparse.ArgumentParser) -> None:
    from .dynamic_boundary import BOUNDARY_MODES
    from .material_volume import BOUNDARY_BUDGETS

    parser.add_argument("--result-root", default=str(DEFAULT_RESULT_ROOT))
    parser.add_argument("--filter-root", default="/root/autodl-fs/kuroshiou/Filter")
    parser.add_argument("--output-root", default="/root/autodl-fs/kuroshiou/EP-FLUX/object_material_boundary_validation")
    parser.add_argument("--shapes", default="coherent,upright_like")
    parser.add_argument("--orientations", default="turned")
    parser.add_argument("--buoyancy-sources", default="thermal_wind")
    parser.add_argument("--filter-template", default="global_phy_{year}_bandpass_30_180d.nc")
    parser.add_argument("--radial-bins", type=int, default=24)
    parser.add_argument("--azimuth-bins", type=int, default=48)
    parser.add_argument("--rmax", type=float, default=1.5)
    parser.add_argument("--reference-lat", type=float, default=30.0)
    parser.add_argument("--constant-n2", type=float, default=2.0e-5)
    parser.add_argument("--core-radius-over-R", type=float, default=1.5)
    parser.add_argument("--speed-core-quantile", type=float, default=0.45)
    parser.add_argument("--pv-core-quantile", type=float, default=0.70)
    parser.add_argument("--min-mask-fraction", type=float, default=0.01)
    parser.add_argument("--boundary-mode", choices=BOUNDARY_MODES, default="levelset_v2")
    parser.add_argument("--boundary-budget", choices=BOUNDARY_BUDGETS, default="edge_proxy")
    parser.add_argument("--active-contour-iterations", type=int, default=14)
    parser.add_argument("--leakage-weight", type=float, default=1.0)
    parser.add_argument("--smoothness-weight", type=float, default=0.08)
    parser.add_argument("--containment-weight", type=float, default=0.35)
    parser.add_argument("--area-weight", type=float, default=0.12)
    parser.add_argument("--vertical-continuity-weight", type=float, default=0.18)
    parser.add_argument("--time-continuity-weight", type=float, default=0.08)
    parser.add_argument("--levelset-sigma-cells", type=float, default=1.0)
    parser.add_argument("--min-core-retention", type=float, default=0.75)
    parser.add_argument("--min-area-fraction", type=float, default=0.15)
    parser.add_argument("--max-area-fraction", type=float, default=0.65)
    parser.add_argument("--max-tracks-per-shape", type=int, default=0)
    parser.add_argument("--max-objectdays", type=int, default=0)
    parser.add_argument("--skip-missing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")


def _add_material_coherence_arguments(parser: argparse.ArgumentParser) -> None:
    from .material_coherence import BOUNDARY_BUDGETS, MATERIAL_COHERENCE_BOUNDARY_MODES

    parser.add_argument("--result-root", default=str(DEFAULT_RESULT_ROOT))
    parser.add_argument("--filter-root", default="/root/autodl-fs/kuroshiou/Filter")
    parser.add_argument("--output-root", default="/root/autodl-fs/kuroshiou/EP-FLUX/object_material_coherence_ep_validation")
    parser.add_argument("--shapes", default="coherent,upright_like")
    parser.add_argument("--orientations", default="turned")
    parser.add_argument("--buoyancy-sources", default="thermal_wind")
    parser.add_argument(
        "--boundary-mode",
        default=",".join(MATERIAL_COHERENCE_BOUNDARY_MODES),
        help="Comma-separated material-coherence modes: particle_retention_v1,lavd_hybrid_v1",
    )
    parser.add_argument("--boundary-budget", choices=BOUNDARY_BUDGETS, default="full_3d")
    parser.add_argument("--filter-template", default="global_phy_{year}_bandpass_30_180d.nc")
    parser.add_argument("--radial-bins", type=int, default=24)
    parser.add_argument("--azimuth-bins", type=int, default=48)
    parser.add_argument("--rmax", type=float, default=1.5)
    parser.add_argument("--reference-lat", type=float, default=30.0)
    parser.add_argument("--constant-n2", type=float, default=2.0e-5)
    parser.add_argument("--core-radius-over-R", type=float, default=1.5)
    parser.add_argument("--speed-core-quantile", type=float, default=0.45)
    parser.add_argument("--pv-core-quantile", type=float, default=0.70)
    parser.add_argument("--min-mask-fraction", type=float, default=0.01)
    parser.add_argument("--active-contour-iterations", type=int, default=14)
    parser.add_argument("--leakage-weight", type=float, default=1.0)
    parser.add_argument("--smoothness-weight", type=float, default=0.08)
    parser.add_argument("--containment-weight", type=float, default=0.35)
    parser.add_argument("--area-weight", type=float, default=0.12)
    parser.add_argument("--vertical-continuity-weight", type=float, default=0.18)
    parser.add_argument("--time-continuity-weight", type=float, default=0.08)
    parser.add_argument("--levelset-sigma-cells", type=float, default=1.0)
    parser.add_argument("--min-core-retention", type=float, default=0.75)
    parser.add_argument("--min-area-fraction", type=float, default=0.15)
    parser.add_argument("--max-area-fraction", type=float, default=0.65)
    parser.add_argument("--trajectory-window-days", type=int, default=7)
    parser.add_argument("--particle-spacing-km", type=float, default=5.0)
    parser.add_argument("--advection-step-hours", type=float, default=6.0)
    parser.add_argument("--max-tracks-per-shape", type=int, default=0)
    parser.add_argument("--max-objectdays", type=int, default=0)
    parser.add_argument("--skip-missing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")


def _add_material_geodesic_arguments(parser: argparse.ArgumentParser) -> None:
    from .material_geodesic import BOUNDARY_BUDGETS, GEODESIC_BOUNDARY_MODES

    parser.add_argument("--result-root", default=str(DEFAULT_RESULT_ROOT))
    parser.add_argument("--filter-root", default="/root/autodl-fs/kuroshiou/Filter")
    parser.add_argument("--output-root", default="/root/autodl-fs/kuroshiou/EP-FLUX/object_material_geodesic_ep_validation")
    parser.add_argument("--shapes", default="coherent,upright_like")
    parser.add_argument("--orientations", default="turned")
    parser.add_argument("--buoyancy-sources", default="thermal_wind")
    parser.add_argument(
        "--boundary-mode",
        default=",".join(GEODESIC_BOUNDARY_MODES),
        help="Comma-separated modes: cauchy_green_geodesic_v1,lavd_material_v1,hybrid_geodesic_lavd_v1",
    )
    parser.add_argument("--boundary-budget", choices=BOUNDARY_BUDGETS, default="full_3d")
    parser.add_argument("--filter-template", default="global_phy_{year}_bandpass_30_180d.nc")
    parser.add_argument("--radial-bins", type=int, default=18)
    parser.add_argument("--azimuth-bins", type=int, default=36)
    parser.add_argument("--rmax", type=float, default=1.5)
    parser.add_argument("--reference-lat", type=float, default=30.0)
    parser.add_argument("--constant-n2", type=float, default=2.0e-5)
    parser.add_argument("--core-radius-over-R", type=float, default=1.5)
    parser.add_argument("--speed-core-quantile", type=float, default=0.45)
    parser.add_argument("--pv-core-quantile", type=float, default=0.70)
    parser.add_argument("--min-mask-fraction", type=float, default=0.01)
    parser.add_argument("--min-core-retention", type=float, default=0.75)
    parser.add_argument("--min-pv-retention", type=float, default=0.75)
    parser.add_argument("--pv-retention-weight", type=float, default=0.80)
    parser.add_argument("--weak-retention-weight", type=float, default=0.40)
    parser.add_argument("--particle-retention-weight", type=float, default=0.20)
    parser.add_argument("--require-pv-retention", action="store_true")
    parser.add_argument("--min-area-fraction", type=float, default=0.10)
    parser.add_argument("--max-area-fraction", type=float, default=0.75)
    parser.add_argument("--trajectory-window-days", type=int, default=7)
    parser.add_argument("--particle-spacing-km", type=float, default=5.0)
    parser.add_argument("--advection-step-hours", type=float, default=6.0)
    parser.add_argument("--max-tracks-per-shape", type=int, default=0)
    parser.add_argument("--max-objectdays", type=int, default=0)
    parser.add_argument("--max-depth-layers", type=int, default=0)
    parser.add_argument("--skip-missing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")


def _add_core_shell_arguments(parser: argparse.ArgumentParser) -> None:
    from .core_shell import DEFAULT_CORE_SHELL_OUTPUT_ROOT
    from .dynamic_boundary import BOUNDARY_MODES
    from .material_volume import BOUNDARY_BUDGETS

    parser.add_argument("--result-root", default=str(DEFAULT_RESULT_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_CORE_SHELL_OUTPUT_ROOT))
    parser.add_argument("--shapes", default="coherent,upright_like")
    parser.add_argument("--axis-sources", default="radial_seed")
    parser.add_argument("--orientations", default="turned")
    parser.add_argument("--buoyancy-sources", default="thermal_wind")
    parser.add_argument(
        "--tau-values",
        default="",
        help="Comma-separated tau values. Empty means all tau nodes from the representative npz.",
    )
    parser.add_argument("--reference-lat", type=float, default=30.0)
    parser.add_argument("--constant-n2", type=float, default=2.0e-5)
    parser.add_argument("--n2-profile", default="auto")
    parser.add_argument("--inner-boundary-mode", choices=BOUNDARY_MODES, default="levelset_v2")
    parser.add_argument("--boundary-budget", choices=BOUNDARY_BUDGETS, default="full_3d")
    parser.add_argument("--core-radius-over-R", type=float, default=1.5)
    parser.add_argument("--shell-outer-radius-over-R", type=float, default=1.5)
    parser.add_argument("--speed-core-quantile", type=float, default=0.45)
    parser.add_argument("--pv-core-quantile", type=float, default=0.70)
    parser.add_argument("--pv-shell-quantile", type=float, default=0.80)
    parser.add_argument("--shell-dilation-cells", type=int, default=2)
    parser.add_argument("--min-mask-fraction", type=float, default=0.01)
    parser.add_argument("--min-core-retention", type=float, default=0.75)
    parser.add_argument("--skip-missing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Object-oriented EP flux diagnostics")
    sub = parser.add_subparsers(dest="command", required=True)

    smoke = sub.add_parser("build-smoke", help="Run a coherent-only representative-eddy EP smoke diagnostic")
    smoke.add_argument("--result-root", default=str(DEFAULT_RESULT_ROOT))
    smoke.add_argument("--shape-output-name", default=DEFAULT_SHAPE_OUTPUT_NAME)
    smoke.add_argument("--me-liutex-root", default=None)
    smoke.add_argument("--radial-seed-root", default=None)
    smoke.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_ROOT))
    smoke.add_argument("--orientation", choices=ORIENTATIONS, default="turned")
    smoke.add_argument("--axis-source", choices=AXIS_SOURCES, default="radial_seed")
    smoke.add_argument("--tau", type=float, default=0.5)
    smoke.add_argument("--reference-lat", type=float, default=30.0)
    smoke.add_argument("--constant-n2", type=float, default=2.0e-5)
    smoke.add_argument("--buoyancy-source", choices=BUOYANCY_SOURCES, default="thermal_wind")
    smoke.add_argument("--curved-tube-mode", choices=CURVED_TUBE_MODES, default="scale_audit")
    smoke.add_argument(
        "--large-curvature-threshold",
        type=float,
        default=1.0,
        help="Flag metric cells as large-curvature when kappa*radius exceeds this value.",
    )
    smoke.add_argument(
        "--n2-profile",
        default="auto",
        help="Path to an N2 npz profile, 'auto' to search common result paths, or 'none' for constant N2.",
    )
    smoke.add_argument("--shape-label", default="coherent-only")
    smoke.add_argument("--run-label", default="subgrid_1_24deg")
    smoke.add_argument("--dry-run", action="store_true")
    smoke.set_defaults(func=cmd_build_smoke)

    compare = sub.add_parser("compare-classic-and-curved", help="Compare classic, tilted, and curved smoke outputs")
    compare.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_ROOT))
    compare.set_defaults(func=cmd_compare)

    lifecycle = sub.add_parser("run-lifecycle-validation", help="Run full lifecycle EP diagnostics across tau and comparison dimensions")
    lifecycle.add_argument("--result-root", default=str(DEFAULT_RESULT_ROOT))
    lifecycle.add_argument("--output-root", default=str(DEFAULT_FULL_OUTPUT_ROOT))
    lifecycle.add_argument("--shapes", default="coherent")
    lifecycle.add_argument("--axis-sources", default="radial_seed")
    lifecycle.add_argument("--orientations", default="turned")
    lifecycle.add_argument("--buoyancy-sources", default="thermal_wind")
    lifecycle.add_argument(
        "--tau-values",
        default="",
        help="Comma-separated tau values. Empty means all tau nodes from the representative npz.",
    )
    lifecycle.add_argument("--reference-lat", type=float, default=30.0)
    lifecycle.add_argument("--constant-n2", type=float, default=2.0e-5)
    lifecycle.add_argument("--n2-profile", default="auto")
    lifecycle.add_argument("--curved-tube-mode", choices=CURVED_TUBE_MODES, default="scale_audit")
    lifecycle.add_argument("--large-curvature-threshold", type=float, default=1.0)
    lifecycle.add_argument("--bootstrap-samples", type=int, default=0)
    lifecycle.add_argument("--bootstrap-unit", choices=["track"], default="track")
    lifecycle.add_argument(
        "--ensure-axis-sources",
        action="store_true",
        help="Create missing persisted representative axis-source files before computing EP diagnostics.",
    )
    lifecycle.add_argument("--skip-missing", action="store_true")
    lifecycle.add_argument("--dry-run", action="store_true")
    lifecycle.set_defaults(func=cmd_run_lifecycle_validation)

    material = sub.add_parser(
        "run-material-volume-validation",
        help="Run Cartesian material-volume EP diagnostics on representative vortices",
    )
    _add_material_arguments(
        material,
        output_root=str(DEFAULT_FULL_OUTPUT_ROOT.parent / "material_volume_validation"),
        boundary_mode="threshold",
    )
    material.set_defaults(func=cmd_run_material_volume_validation)

    dynamic = sub.add_parser(
        "run-material-boundary-validation",
        help="Run material-volume EP diagnostics with low-leakage dynamic boundary optimization",
    )
    _add_material_arguments(
        dynamic,
        output_root=str(DEFAULT_FULL_OUTPUT_ROOT.parent / "material_volume_dynamic_boundary_validation"),
        boundary_mode="active_contour",
    )
    dynamic.set_defaults(func=cmd_run_material_boundary_validation)

    object_boundary = sub.add_parser(
        "run-object-material-boundary-validation",
        help="Run object-day/track material-boundary diagnostics on original eddy fields",
    )
    _add_object_boundary_arguments(object_boundary)
    object_boundary.set_defaults(func=cmd_run_object_material_boundary_validation)

    object_coherence = sub.add_parser(
        "run-object-material-coherence-validation",
        help="Run object-level particle-retention/LAVD material-coherence EP diagnostics",
    )
    _add_material_coherence_arguments(object_coherence)
    object_coherence.set_defaults(func=cmd_run_object_material_coherence_validation)

    object_geodesic = sub.add_parser(
        "run-object-material-geodesic-validation",
        help="Run object-level finite-time Cauchy-Green/LAVD material-boundary EP diagnostics",
    )
    _add_material_geodesic_arguments(object_geodesic)
    object_geodesic.set_defaults(func=cmd_run_object_material_geodesic_validation)

    core_shell = sub.add_parser(
        "run-core-shell-ep-validation",
        help="Run core-shell EP diagnostics separating material core, PV-active shell, and boundary exchange",
    )
    _add_core_shell_arguments(core_shell)
    core_shell.set_defaults(func=cmd_run_core_shell_ep_validation)

    explain = sub.add_parser("explain-contract", help="Print the EP diagnostic contract")
    explain.set_defaults(func=cmd_explain_contract)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
