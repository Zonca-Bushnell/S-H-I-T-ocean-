from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .axis_streamfunction_separation import DEFAULT_AXIS_DIR, DEFAULT_CATALOG, fit_rank1, parse_csv_list
from .lifecycle_common import DEFAULT_LIFECYCLE_ROOT, DEFAULT_POLARITIES, PHASE_NAMES
from .representative_velocity_stack import (
    DEFAULT_DEPTH_LEVELS,
    load_representative_radii,
    make_xy_grid,
    nearest_depth_indices,
    parse_float_list,
    plot_vtheta_rz,
    velocity_components_on_grid,
    velocity_from_psi,
)
from .representative_velocity_stack_tilted import (
    axis_xy_m,
    fit_pooled_axis,
    offset_layers,
    plot_html_tilted_stack,
    plot_png_tilted_stack,
    plot_tilted_axis_sections,
)


DEFAULT_TEMPLATE_DIR = DEFAULT_LIFECYCLE_ROOT / "streamfunction_templates"
DEFAULT_OUTPUT = DEFAULT_LIFECYCLE_ROOT / "velocity_stack_tilted"


def load_lifecycle_profile_matrix(template_dir: Path, polarity: str, phase_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    profiles = pd.read_parquet(template_dir / "lifecycle_radial_psi_profiles.parquet")
    part = profiles[(profiles["polarity"] == polarity) & (profiles["phase_name"] == phase_name)].copy()
    if part.empty:
        raise ValueError(f"No lifecycle psi profiles found for polarity={polarity!r}, phase={phase_name!r}.")
    depth = np.sort(part["depth_m"].unique().astype("f8"))
    r = np.sort(part["r_over_R"].unique().astype("f8"))
    depth_index = {float(value): i for i, value in enumerate(depth)}
    r_index = {float(value): i for i, value in enumerate(r)}
    sums = np.zeros((len(depth), len(r)), dtype="f8")
    counts = np.zeros_like(sums)
    for row in part.itertuples(index=False):
        i = depth_index[float(row.depth_m)]
        j = r_index[float(row.r_over_R)]
        count = float(row.count)
        sums[i, j] += float(row.psi_mean) * count
        counts[i, j] += count
    matrix = np.divide(sums, counts, out=np.full_like(sums, np.nan), where=counts > 0)
    recon, _, _, metrics = fit_rank1(matrix, counts)
    return depth, r, matrix, recon, metrics


def process_phase(polarity: str, phase_name: str, args: argparse.Namespace, radii: dict[str, float], figure_dir: Path) -> dict:
    radius_m = radii.get(polarity)
    if radius_m is None:
        raise ValueError(f"No representative radius found for polarity {polarity!r}.")
    depth, r, _, psi, metrics = load_lifecycle_profile_matrix(Path(args.template_dir), polarity, phase_name)
    vtheta = velocity_from_psi(psi, r, radius_m)
    depth_indices = nearest_depth_indices(depth, parse_float_list(args.depth_levels))
    x_local, y_local, r_norm = make_xy_grid(float(args.xy_extent), int(args.grid_size))
    u_layers, v_layers, speed_layers = velocity_components_on_grid(vtheta, r, depth_indices, x_local, y_local, r_norm)

    axis = fit_pooled_axis(Path(args.axis_dir), polarity)
    axis_x_m, axis_y_m = axis_xy_m(axis, depth)
    axis_x_over_r = axis_x_m / radius_m
    axis_y_over_r = axis_y_m / radius_m
    x_layers, y_layers = offset_layers(x_local, y_local, axis_x_over_r, axis_y_over_r, depth_indices)

    label = f"all_shapes_{polarity}_{phase_name}"
    if args.png:
        plot_png_tilted_stack(
            label,
            depth,
            depth_indices,
            x_layers,
            y_layers,
            u_layers,
            v_layers,
            speed_layers,
            axis_x_over_r,
            axis_y_over_r,
            arrow_step=int(args.arrow_step),
            figure_dir=figure_dir,
        )
        plot_vtheta_rz(label, depth, r, vtheta, figure_dir)
        plot_tilted_axis_sections(label, depth, axis_x_over_r, axis_y_over_r, depth_indices, figure_dir)
    if args.html:
        plot_html_tilted_stack(
            label,
            depth,
            depth_indices,
            x_layers,
            y_layers,
            u_layers,
            v_layers,
            speed_layers,
            axis_x_over_r,
            axis_y_over_r,
            arrow_step=int(args.arrow_step),
            figure_dir=figure_dir,
        )

    deep_offset_m = float(np.hypot(axis_x_m[-1], axis_y_m[-1]))
    depth_span_m = float(depth[-1] - depth[0])
    return {
        "shape_class": "all_shapes",
        "polarity": polarity,
        "phase_name": phase_name,
        "label": label,
        "median_radius_m": radius_m,
        "depth_levels_m": ",".join(f"{float(depth[index]):.3f}" for index in depth_indices),
        "deep_axis_offset_m": deep_offset_m,
        "deep_axis_offset_over_R": deep_offset_m / radius_m,
        "tilt_angle_deg": float(np.degrees(np.arctan2(deep_offset_m, depth_span_m))) if depth_span_m > 0 else np.nan,
        "rank1_energy_fraction": metrics.get("rank1_energy_fraction", np.nan),
        "relative_rmse": metrics.get("relative_rmse", np.nan),
        "max_abs_vtheta_m_s": float(np.nanmax(np.abs(vtheta))),
        "p98_abs_vtheta_m_s": float(np.nanpercentile(np.abs(vtheta), 98)),
        "max_speed_stack_m_s": float(np.nanmax(speed_layers)),
        "p98_speed_stack_m_s": float(np.nanpercentile(speed_layers, 98)),
    }


def write_summary(output_dir: Path, rows: list[dict], args: argparse.Namespace) -> None:
    summary = pd.DataFrame.from_records(rows)
    summary.to_csv(output_dir / "lifecycle_velocity_stack_summary.csv", index=False)
    lines = [
        "# Lifecycle tilted velocity stack summary",
        "",
        f"- Template dir: `{args.template_dir}`",
        f"- Axis dir: `{args.axis_dir}`",
        f"- Catalog dir: `{args.catalog_dir}`",
        f"- Polarities: {args.polarities}",
        f"- Phases: {args.phases}",
        f"- Requested depth levels: {args.depth_levels}",
        f"- x/y extent: +/- {args.xy_extent} R",
        "",
        "Each phase uses its own lifecycle representative streamfunction and the pooled polarity quadratic tilted axis.",
        "",
        "```csv",
        summary.to_csv(index=False).strip(),
        "```",
    ]
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    figure_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    radii = load_representative_radii(Path(args.axis_dir), Path(args.catalog_dir))
    rows = []
    for polarity in parse_csv_list(args.polarities, DEFAULT_POLARITIES):
        for phase_name in parse_csv_list(args.phases, PHASE_NAMES):
            rows.append(process_phase(polarity, phase_name, args, radii, figure_dir))
    write_summary(output_dir, rows, args)
    print(f"Output: {output_dir}")
    print(f"Summary: {output_dir / 'lifecycle_velocity_stack_summary.csv'}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot lifecycle-normalized tilted 3D representative velocity stacks.")
    parser.add_argument("--template-dir", default=str(DEFAULT_TEMPLATE_DIR))
    parser.add_argument("--axis-dir", default=str(DEFAULT_AXIS_DIR))
    parser.add_argument("--catalog-dir", default=str(DEFAULT_CATALOG))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--polarities", default=",".join(DEFAULT_POLARITIES))
    parser.add_argument("--phases", default=",".join(PHASE_NAMES))
    parser.add_argument("--depth-levels", default=",".join(str(value) for value in DEFAULT_DEPTH_LEVELS))
    parser.add_argument("--xy-extent", type=float, default=2.5)
    parser.add_argument("--grid-size", type=int, default=41)
    parser.add_argument("--arrow-step", type=int, default=4)
    parser.add_argument("--html", dest="html", action="store_true", default=True)
    parser.add_argument("--no-html", dest="html", action="store_false")
    parser.add_argument("--png", dest="png", action="store_true", default=True)
    parser.add_argument("--no-png", dest="png", action="store_false")
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
