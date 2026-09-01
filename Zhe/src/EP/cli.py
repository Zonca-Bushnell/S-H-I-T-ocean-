from __future__ import annotations

import argparse
from pathlib import Path

from .contracts import (
    AXIS_SOURCES,
    BUOYANCY_SOURCES,
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
                "  exposes a local-axis tensor/divergence framework and currently uses a QG first-order",
                "  curvature proxy. It is a smoke implementation, not the final full tensor theory.",
                "",
                "Boundaries:",
                "  src.EP does not import src.utils.ep_flux as an implementation dependency.",
                "  src.EP does not overwrite representative vortex npz/catalog files.",
            ]
        )
    )
    return 0


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

    explain = sub.add_parser("explain-contract", help="Print the EP diagnostic contract")
    explain.set_defaults(func=cmd_explain_contract)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
