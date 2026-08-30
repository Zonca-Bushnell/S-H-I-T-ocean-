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


def plot_original_eddy_panels(args: argparse.Namespace) -> None:
    cmd = [
        sys.executable,
        "-m",
        "src.post.original_eddy_panels",
        "--results-root",
        str(args.results_root),
        "--shape-dir-name",
        args.shape_dir_name,
        "--raw-root",
        str(args.raw_root),
        "--filter-root",
        str(args.filter_root),
        "--output-dir",
        str(args.output_dir),
        "--preferred-shapes",
        args.preferred_shapes,
        "--min-layers",
        str(args.min_layers),
        "--abrupt-threshold-over-r",
        str(args.abrupt_threshold_over_r),
        "--half-width-deg",
        str(args.half_width_deg),
        "--max-examples",
        str(args.max_examples),
        "--w-shear-depth-padding-layers",
        str(args.w_shear_depth_padding_layers),
        "--w-shear-half-width-r",
        str(args.w_shear_half_width_r),
        "--w-shear-min-half-width-km",
        str(args.w_shear_min_half_width_km),
        "--w-section-mode",
        args.w_section_mode,
        "--right-panel-mode",
        args.right_panel_mode,
        "--horizontal-smooth-sigma-cells",
        str(args.horizontal_smooth_sigma_cells),
        "--output-name-stem",
        args.output_name_stem,
    ]
    if args.year_limit is not None:
        cmd.extend(["--year-limit", str(args.year_limit)])
    if args.selected_metadata is not None:
        cmd.extend(["--selected-metadata", str(args.selected_metadata)])
    if args.no_horizontal_smoothing:
        cmd.append("--no-horizontal-smoothing")
    if args.show_grid_centers:
        cmd.append("--show-grid-centers")
    if args.dry_run:
        print(f"[post] science: {PRODUCTION_POST_SCOPE}")
        print(f"[post] results root: {args.results_root}")
        print(f"[post] output: {args.output_dir}")
        print("[dry-run] " + " ".join(cmd))
        return
    subprocess.run(cmd, check=True)


def plot_jump_section_geometry(args: argparse.Namespace) -> None:
    if args.dry_run:
        print(f"[post] output: {args.output}")
        print("[dry-run] python -m src.post.jump_section_geometry --output " + str(args.output))
        return
    from .jump_section_geometry import plot_jump_section_geometry as run

    run(Path(args.output))


def plot_representative_eddy_panels(args: argparse.Namespace) -> None:
    if args.dry_run:
        print(f"[post] radial seed: {args.radial_seed_root}")
        print(f"[post] output: {args.output_dir}")
        print(f"[post] orientation: {args.orientation}")
        print(f"[post] latest panel family: axis top-2 steps + upper/lower fields + {args.section_mode} sections")
        print(f"[post] right panel mode: {args.right_panel_mode}")
        return
    from .representative_eddy_panels import plot_representative_eddy_panels as run

    roots = []
    if args.orientation in ("turned", "both"):
        roots.append(("turned", Path(args.me_liutex_root)))
    if args.orientation in ("unturned", "both"):
        roots.append(("unturned", Path(args.me_liutex_unturned_root)))
    for orientation, root in roots:
        run(
            me_liutex_root=root,
            radial_seed_root=Path(args.radial_seed_root),
            output_dir=Path(args.output_dir) / orientation,
            orientation=orientation,
            tau=args.tau,
            axis_bandwidth=args.axis_bandwidth,
            grid_size=args.grid_size,
            reference_lat=args.reference_lat,
            section_mode=args.section_mode,
            horizontal_smooth_sigma_cells=args.horizontal_smooth_sigma_cells,
            section_depth_padding_layers=args.section_depth_padding_layers,
            section_half_width_r=args.section_half_width_r,
            section_min_half_width_km=args.section_min_half_width_km,
            right_panel_mode=args.right_panel_mode,
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
    panels = subparsers.add_parser("plot-original-eddy-panels")
    panels.add_argument("--results-root", type=Path, default=DEFAULT_RESULT_ROOT)
    panels.add_argument("--shape-dir-name", default="shape_classification_1993_2022_hua_b3_start2_life30")
    panels.add_argument("--raw-root", type=Path, default=Path("/root/autodl-fs/kuroshiou/raw"))
    panels.add_argument("--filter-root", type=Path, default=DEFAULT_FILTER_ROOT)
    panels.add_argument("--output-dir", type=Path, required=True)
    panels.add_argument("--preferred-shapes", default="coherent,complex,mixed,upright_like,transitional")
    panels.add_argument("--min-layers", type=int, default=10)
    panels.add_argument("--abrupt-threshold-over-r", type=float, default=0.15)
    panels.add_argument("--half-width-deg", type=float, default=2.0)
    panels.add_argument("--year-limit", type=int, default=None)
    panels.add_argument("--max-examples", type=int, default=1)
    panels.add_argument("--w-shear-depth-padding-layers", type=int, default=6)
    panels.add_argument("--w-shear-half-width-r", type=float, default=1.2)
    panels.add_argument("--w-shear-min-half-width-km", type=float, default=75.0)
    panels.add_argument("--w-section-mode", choices=["parallel", "normal"], default="parallel")
    panels.add_argument(
        "--right-panel-mode",
        choices=["omega_w", "normal_horizontal_velocity", "horizontal_speed", "signed_horizontal_speed"],
        default="omega_w",
    )
    panels.add_argument("--horizontal-smooth-sigma-cells", type=float, default=0.8)
    panels.add_argument("--no-horizontal-smoothing", action="store_true")
    panels.add_argument("--show-grid-centers", action="store_true")
    panels.add_argument("--selected-metadata", type=Path, default=None)
    panels.add_argument("--output-name-stem", default="original_eddy_discontinuity_9panel")
    panels.add_argument("--dry-run", action="store_true")
    panels.set_defaults(func=plot_original_eddy_panels)
    geometry = subparsers.add_parser("plot-jump-section-geometry")
    geometry.add_argument("--output", type=Path, required=True)
    geometry.add_argument("--dry-run", action="store_true")
    geometry.set_defaults(func=plot_jump_section_geometry)
    rep_panels = subparsers.add_parser("plot-representative-eddy-panels")
    rep_panels.add_argument("--me-liutex-root", type=Path, required=True)
    rep_panels.add_argument("--me-liutex-unturned-root", type=Path, required=True)
    rep_panels.add_argument("--radial-seed-root", type=Path, required=True)
    rep_panels.add_argument("--output-dir", type=Path, required=True)
    rep_panels.add_argument("--orientation", choices=["turned", "unturned", "both"], default="both")
    rep_panels.add_argument("--tau", type=float, default=0.5)
    rep_panels.add_argument("--axis-bandwidth", type=float, default=0.075)
    rep_panels.add_argument("--grid-size", type=int, default=121)
    rep_panels.add_argument("--reference-lat", type=float, default=28.0)
    rep_panels.add_argument("--section-mode", choices=["parallel", "normal"], default="normal")
    rep_panels.add_argument(
        "--right-panel-mode",
        choices=["normal_horizontal_velocity", "horizontal_speed", "signed_horizontal_speed"],
        default="normal_horizontal_velocity",
    )
    rep_panels.add_argument("--horizontal-smooth-sigma-cells", type=float, default=0.8)
    rep_panels.add_argument("--section-depth-padding-layers", type=int, default=6)
    rep_panels.add_argument("--section-half-width-r", type=float, default=1.2)
    rep_panels.add_argument("--section-min-half-width-km", type=float, default=75.0)
    rep_panels.add_argument("--dry-run", action="store_true")
    rep_panels.set_defaults(func=plot_representative_eddy_panels)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
