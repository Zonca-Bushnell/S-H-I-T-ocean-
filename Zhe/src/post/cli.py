from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from .contracts import (
    DEFAULT_FILTER_ROOT,
    DEFAULT_FILTER_TEMPLATE,
    DEFAULT_ORIENTATION,
    DEFAULT_RESULT_ROOT,
    DEFAULT_SHAPE,
    PRODUCTION_POST_SCOPE,
    PostPaths,
)


def _print_scope(args: argparse.Namespace) -> None:
    paths = PostPaths(result_root=Path(args.result_root), shape=args.shape)
    print(f"[post] science: {PRODUCTION_POST_SCOPE}")
    print(f"[post] shape root: {paths.shape_root}")
    print(f"[post] radial seed: {paths.radial_seed_root}")
    print(f"[post] turned structure: {paths.turned_root}")
    print(f"[post] unturned structure: {paths.unturned_root}")
    print(f"[post] transport output: {paths.transport_root}")
    print(f"[post] structure figures: {paths.figures_root}")
    print(f"[post] double-core output: {paths.double_core_root}")


def build_transport(args: argparse.Namespace) -> None:
    paths = PostPaths(result_root=Path(args.result_root), shape=args.shape)
    cmd = [
        sys.executable,
        "-m",
        "src.post.transport",
        "--rv-root",
        str(paths.radial_seed_root),
        "--filter-root",
        str(args.filter_root),
        "--output-dir",
        str(paths.transport_root),
        "--shapes",
        args.shape,
        "--filter-template",
        args.filter_template,
        "--workers",
        str(args.workers),
        "--chunk-days",
        str(args.chunk_days),
        "--azimuth-bins",
        str(args.azimuth_bins),
        "--rmax",
        str(args.rmax),
        "--resume",
    ]
    if args.dry_run:
        _print_scope(args)
        print("[dry-run] " + " ".join(cmd))
        return
    subprocess.run(cmd, check=True)


def plot_structure(args: argparse.Namespace) -> None:
    if args.dry_run:
        _print_scope(args)
        print(f"[dry-run] plot structure for shape={args.shape}, orientation={args.orientation}")
        return
    from .structure import plot_structure as run

    run(result_root=Path(args.result_root), shape=args.shape, orientation=args.orientation)


def analyze_double_core(args: argparse.Namespace) -> None:
    if args.dry_run:
        _print_scope(args)
        print(f"[dry-run] analyze double core for shape={args.shape}, orientation={args.orientation}")
        return
    from .double_core import analyze_double_core as run

    run(result_root=Path(args.result_root), shape=args.shape, orientation=args.orientation)


def analyze_jump_wshear_relation(args: argparse.Namespace) -> None:
    if args.dry_run:
        print(f"[post] science: {PRODUCTION_POST_SCOPE}")
        print(f"[post] results root: {args.results_root}")
        print(f"[post] filter root: {args.filter_root}")
        print(
            "[dry-run] analyze jump-wshear relation for "
            f"shapes={args.shapes}, section_modes={args.section_modes}, output={args.output_dir}"
        )
        return
    from .discontinuity_relation import analyze_jump_wshear_relation as run

    run(
        results_root=Path(args.results_root),
        shape_dir_name=args.shape_dir_name,
        filter_root=Path(args.filter_root),
        raw_root=Path(args.raw_root) if args.raw_root else None,
        output_dir=Path(args.output_dir),
        shapes=args.shapes,
        jump_ranks=args.jump_ranks,
        half_width_deg=args.half_width_deg,
        depth_padding_layers=args.w_shear_depth_padding_layers,
        half_width_r=args.w_shear_half_width_r,
        min_half_width_km=args.w_shear_min_half_width_km,
        section_modes=args.section_modes,
        vertical_velocity_method=args.vertical_velocity_method,
        year_limit=args.year_limit,
        resume=args.resume,
    )


def run_default(args: argparse.Namespace) -> None:
    _print_scope(args)
    plot_structure(args)
    analyze_double_core(args)
    build_transport(args)


def _add_common_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--filter-root", type=Path, default=DEFAULT_FILTER_ROOT)
    parser.add_argument("--filter-template", default=DEFAULT_FILTER_TEMPLATE)
    parser.add_argument("--shape", default=DEFAULT_SHAPE)
    parser.add_argument("--orientation", choices=["turned", "unturned", "both"], default=DEFAULT_ORIENTATION)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--chunk-days", type=int, default=7)
    parser.add_argument("--azimuth-bins", type=int, default=72)
    parser.add_argument("--rmax", type=float, default=2.5)
    parser.add_argument("--dry-run", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    _add_common_options(common)
    parser = argparse.ArgumentParser(description="Production post-processing for representative eddy outputs.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name, func in (
        ("build-transport", build_transport),
        ("plot-structure", plot_structure),
        ("analyze-double-core", analyze_double_core),
        ("run-default", run_default),
    ):
        child = subparsers.add_parser(name, parents=[common])
        child.set_defaults(func=func)
    relation = subparsers.add_parser("analyze-jump-wshear-relation")
    relation.add_argument("--results-root", type=Path, default=DEFAULT_RESULT_ROOT)
    relation.add_argument("--shape-dir-name", default="shape_classification_1993_2022_hua_b3_start2_life30")
    relation.add_argument("--filter-root", type=Path, default=DEFAULT_FILTER_ROOT)
    relation.add_argument("--raw-root", type=Path, default=None)
    relation.add_argument("--output-dir", type=Path, required=True)
    relation.add_argument("--shapes", default=DEFAULT_SHAPE)
    relation.add_argument("--jump-ranks", type=int, default=2)
    relation.add_argument("--half-width-deg", type=float, default=2.0)
    relation.add_argument("--w-shear-depth-padding-layers", type=int, default=6)
    relation.add_argument("--w-shear-half-width-r", type=float, default=1.2)
    relation.add_argument("--w-shear-min-half-width-km", type=float, default=75.0)
    relation.add_argument("--section-modes", default="parallel")
    relation.add_argument("--vertical-velocity-method", choices=["proxy", "omega"], default="proxy")
    relation.add_argument("--year-limit", type=int, default=None)
    relation.add_argument("--chunk-days", type=int, default=14)
    relation.add_argument("--resume", action="store_true")
    relation.add_argument("--dry-run", action="store_true")
    relation.set_defaults(func=analyze_jump_wshear_relation)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
