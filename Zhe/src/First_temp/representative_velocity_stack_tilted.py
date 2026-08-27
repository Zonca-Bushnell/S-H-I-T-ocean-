from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .axis_streamfunction_separation import DEFAULT_AXIS_DIR, DEFAULT_CATALOG, DEFAULT_OUTPUT as DEFAULT_TEMPLATE_DIR, fit_rank1, parse_csv_list
from .representative_velocity_stack import (
    DEFAULT_DEPTH_LEVELS,
    DEFAULT_POLARITIES,
    load_profile_matrix,
    load_representative_radii,
    make_xy_grid,
    nearest_depth_indices,
    parse_float_list,
    plot_vtheta_rz,
    velocity_components_on_grid,
    velocity_from_psi,
)


DEFAULT_OUTPUT = Path(r"E:\Verify\outputs\kuroshio_cmems_3d\representative_velocity_stack_tilted_1993_2022")


def fit_pooled_axis(axis_dir: Path, polarity: str) -> dict[str, float]:
    points = pd.read_parquet(axis_dir / "rotated_points.parquet", columns=["polarity", "z_m", "x_rot_m", "y_rot_m"])
    part = points[points["polarity"] == polarity].copy()
    if part.empty:
        raise ValueError(f"No rotated axis points found for polarity {polarity!r}.")
    z = part["z_m"].to_numpy(dtype="f8")
    design = np.column_stack([np.ones_like(z), z, z * z])
    xcoef, *_ = np.linalg.lstsq(design, part["x_rot_m"].to_numpy(dtype="f8"), rcond=None)
    ycoef, *_ = np.linalg.lstsq(design, part["y_rot_m"].to_numpy(dtype="f8"), rcond=None)
    xfit = design @ xcoef
    yfit = design @ ycoef
    rmse_x = float(np.sqrt(np.mean((part["x_rot_m"].to_numpy(dtype="f8") - xfit) ** 2)))
    rmse_y = float(np.sqrt(np.mean((part["y_rot_m"].to_numpy(dtype="f8") - yfit) ** 2)))
    return {
        "c1": float(xcoef[0]),
        "c2": float(xcoef[1]),
        "c3": float(xcoef[2]),
        "c4": float(ycoef[0]),
        "c5": float(ycoef[1]),
        "c6": float(ycoef[2]),
        "n_points": int(len(part)),
        "rmse_x_m": rmse_x,
        "rmse_y_m": rmse_y,
        "rmse_2d_m": float(np.hypot(rmse_x, rmse_y)),
    }


def axis_xy_m(axis: dict[str, float], depth: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    z = np.asarray(depth, dtype="f8") - float(depth[0])
    x = axis["c1"] + axis["c2"] * z + axis["c3"] * z * z
    y = axis["c4"] + axis["c5"] * z + axis["c6"] * z * z
    x0 = axis["c1"]
    y0 = axis["c4"]
    return x - x0, y - y0


def offset_layers(
    x: np.ndarray,
    y: np.ndarray,
    axis_x_over_r: np.ndarray,
    axis_y_over_r: np.ndarray,
    depth_indices: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    x_layers = []
    y_layers = []
    for depth_index in depth_indices:
        x_layers.append(x + axis_x_over_r[depth_index])
        y_layers.append(y + axis_y_over_r[depth_index])
    return np.asarray(x_layers), np.asarray(y_layers)


def plot_tilted_axis_sections(
    label: str,
    depth: np.ndarray,
    axis_x_over_r: np.ndarray,
    axis_y_over_r: np.ndarray,
    depth_indices: list[int],
    figure_dir: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), dpi=160)
    axes[0].plot(axis_x_over_r, depth, color="black")
    axes[0].scatter(axis_x_over_r[depth_indices], depth[depth_indices], color="#d62728", s=18)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("x_axis/R")
    axes[0].set_ylabel("depth m")
    axes[0].set_title("x-z axis")
    axes[0].grid(True, color="0.9")
    axes[1].plot(axis_y_over_r, depth, color="black")
    axes[1].scatter(axis_y_over_r[depth_indices], depth[depth_indices], color="#d62728", s=18)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("y_axis/R")
    axes[1].set_title("y-z axis")
    axes[1].grid(True, color="0.9")
    fig.suptitle(f"{label}: pooled tilted axis")
    fig.tight_layout()
    fig.savefig(figure_dir / f"{label}_tilted_axis_xz_yz.png")
    plt.close(fig)


def plot_png_tilted_stack(
    label: str,
    depth: np.ndarray,
    depth_indices: list[int],
    x_layers: np.ndarray,
    y_layers: np.ndarray,
    u_layers: np.ndarray,
    v_layers: np.ndarray,
    speed_layers: np.ndarray,
    axis_x_over_r: np.ndarray,
    axis_y_over_r: np.ndarray,
    *,
    arrow_step: int,
    figure_dir: Path,
) -> None:
    vmax = float(np.nanpercentile(speed_layers, 98))
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0
    fig = plt.figure(figsize=(10.5, 8.5), dpi=170)
    ax = fig.add_subplot(111, projection="3d")
    norm = plt.Normalize(vmin=0.0, vmax=vmax)
    cmap = plt.get_cmap("magma")
    for layer_idx, depth_index in enumerate(depth_indices):
        x_plot = x_layers[layer_idx]
        y_plot = y_layers[layer_idx]
        z_plot = np.full_like(x_plot, -float(depth[depth_index]))
        colors = cmap(norm(speed_layers[layer_idx]))
        colors[..., 3] = np.where(np.isfinite(speed_layers[layer_idx]), 0.36, 0.0)
        ax.plot_surface(x_plot, y_plot, z_plot, facecolors=colors, linewidth=0, antialiased=False, shade=False)

        xs = x_plot[::arrow_step, ::arrow_step]
        ys = y_plot[::arrow_step, ::arrow_step]
        zs = z_plot[::arrow_step, ::arrow_step]
        us = u_layers[layer_idx, ::arrow_step, ::arrow_step]
        vs = v_layers[layer_idx, ::arrow_step, ::arrow_step]
        good = np.isfinite(us) & np.isfinite(vs)
        ax.quiver(
            xs[good],
            ys[good],
            zs[good],
            us[good],
            vs[good],
            np.zeros(np.sum(good)),
            length=1.25,
            normalize=True,
            color="black",
            linewidth=0.55,
            alpha=0.85,
        )
        ax.text(
            float(np.nanmax(x_plot)) + 0.12,
            float(np.nanmax(y_plot)) + 0.12,
            -float(depth[depth_index]),
            f"{depth[depth_index]:.0f} m",
            fontsize=8,
        )

    ax.plot(axis_x_over_r, axis_y_over_r, -depth, color="black", linewidth=2.4, label="pooled tilted axis")
    ax.scatter([axis_x_over_r[0]], [axis_y_over_r[0]], [-depth[0]], color="black", s=24)
    ax.scatter([axis_x_over_r[-1]], [axis_y_over_r[-1]], [-depth[-1]], color="#d62728", s=42)
    scalar = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    scalar.set_array([])
    fig.colorbar(scalar, ax=ax, shrink=0.65, pad=0.08, label="speed (m/s)")
    ax.set_xlabel("x/R")
    ax.set_ylabel("y/R")
    ax.set_zlabel("-depth m")
    ax.set_title(f"{label}: tilted 3D stacked representative velocity")
    ax.view_init(elev=23, azim=-52)
    ax.set_box_aspect((1.25, 1, 0.82))
    fig.tight_layout()
    fig.savefig(figure_dir / f"{label}_velocity_stack_tilted_3d.png")
    plt.close(fig)


def plot_html_tilted_stack(
    label: str,
    depth: np.ndarray,
    depth_indices: list[int],
    x_layers: np.ndarray,
    y_layers: np.ndarray,
    u_layers: np.ndarray,
    v_layers: np.ndarray,
    speed_layers: np.ndarray,
    axis_x_over_r: np.ndarray,
    axis_y_over_r: np.ndarray,
    *,
    arrow_step: int,
    figure_dir: Path,
) -> None:
    import plotly.graph_objects as go

    vmax = float(np.nanpercentile(speed_layers, 98))
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0
    fig = go.Figure()
    for layer_idx, depth_index in enumerate(depth_indices):
        x_plot = x_layers[layer_idx]
        y_plot = y_layers[layer_idx]
        z_plot = np.full_like(x_plot, -float(depth[depth_index]))
        fig.add_trace(
            go.Surface(
                x=x_plot,
                y=y_plot,
                z=z_plot,
                surfacecolor=speed_layers[layer_idx],
                cmin=0,
                cmax=vmax,
                colorscale="Magma",
                opacity=0.52,
                showscale=layer_idx == 0,
                colorbar={"title": "speed (m/s)"},
                name=f"{depth[depth_index]:.0f} m speed",
            )
        )
        xs = x_plot[::arrow_step, ::arrow_step]
        ys = y_plot[::arrow_step, ::arrow_step]
        zs = z_plot[::arrow_step, ::arrow_step]
        us = u_layers[layer_idx, ::arrow_step, ::arrow_step]
        vs = v_layers[layer_idx, ::arrow_step, ::arrow_step]
        good = np.isfinite(us) & np.isfinite(vs)
        fig.add_trace(
            go.Cone(
                x=xs[good].ravel(),
                y=ys[good].ravel(),
                z=zs[good].ravel(),
                u=us[good].ravel(),
                v=vs[good].ravel(),
                w=np.zeros(np.sum(good)),
                sizemode="absolute",
                sizeref=0.16,
                anchor="tail",
                colorscale="Greys",
                showscale=False,
                name=f"{depth[depth_index]:.0f} m vectors",
            )
        )
    fig.add_trace(
        go.Scatter3d(
            x=axis_x_over_r,
            y=axis_y_over_r,
            z=-depth,
            mode="lines",
            line={"color": "black", "width": 8},
            name="pooled tilted axis",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=[axis_x_over_r[-1]],
            y=[axis_y_over_r[-1]],
            z=[-depth[-1]],
            mode="markers",
            marker={"color": "red", "size": 5},
            name="deep axis end",
        )
    )
    fig.update_layout(
        title=f"{label}: tilted 3D stacked representative velocity",
        scene={
            "xaxis_title": "x/R",
            "yaxis_title": "y/R",
            "zaxis_title": "-depth m",
            "aspectmode": "manual",
            "aspectratio": {"x": 1.25, "y": 1, "z": 0.82},
            "camera": {"eye": {"x": 1.55, "y": -1.75, "z": 0.9}},
        },
        margin={"l": 0, "r": 0, "t": 48, "b": 0},
    )
    fig.write_html(figure_dir / f"{label}_velocity_stack_tilted_3d.html", include_plotlyjs="cdn")


def process_polarity(polarity: str, args: argparse.Namespace, radii: dict[str, float], figure_dir: Path) -> dict:
    radius_m = radii.get(polarity)
    if radius_m is None:
        raise ValueError(f"No representative radius found for polarity {polarity!r}.")

    depth, r, _, psi, metrics = load_profile_matrix(Path(args.template_dir), polarity)
    vtheta = velocity_from_psi(psi, r, radius_m)
    depth_indices = nearest_depth_indices(depth, parse_float_list(args.depth_levels))
    x_local, y_local, r_norm = make_xy_grid(float(args.xy_extent), int(args.grid_size))
    u_layers, v_layers, speed_layers = velocity_components_on_grid(vtheta, r, depth_indices, x_local, y_local, r_norm)

    axis = fit_pooled_axis(Path(args.axis_dir), polarity)
    axis_x_m, axis_y_m = axis_xy_m(axis, depth)
    axis_x_over_r = axis_x_m / radius_m
    axis_y_over_r = axis_y_m / radius_m
    x_layers, y_layers = offset_layers(x_local, y_local, axis_x_over_r, axis_y_over_r, depth_indices)

    label = f"all_shapes_{polarity}"
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
        "polarity": polarity,
        "label": label,
        "median_radius_m": radius_m,
        "depth_levels_m": ",".join(f"{float(depth[index]):.3f}" for index in depth_indices),
        "axis_c1": axis["c1"],
        "axis_c2": axis["c2"],
        "axis_c3": axis["c3"],
        "axis_c4": axis["c4"],
        "axis_c5": axis["c5"],
        "axis_c6": axis["c6"],
        "axis_fit_n_points": axis["n_points"],
        "axis_fit_rmse_2d_m": axis["rmse_2d_m"],
        "deep_axis_offset_m": deep_offset_m,
        "deep_axis_offset_over_R": deep_offset_m / radius_m,
        "deep_axis_x_over_R": float(axis_x_over_r[-1]),
        "deep_axis_y_over_R": float(axis_y_over_r[-1]),
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
    summary.to_csv(output_dir / "representative_velocity_stack_tilted_summary.csv", index=False)
    lines = [
        "# Representative tilted velocity stack summary",
        "",
        f"- Template dir: `{args.template_dir}`",
        f"- Axis dir: `{args.axis_dir}`",
        f"- Catalog dir: `{args.catalog_dir}`",
        f"- Polarities: {args.polarities}",
        f"- Requested depth levels: {args.depth_levels}",
        f"- x/y extent: +/- {args.xy_extent} R",
        f"- Grid size: {args.grid_size}",
        "- Axis scale: radius-normalized x/R and y/R",
        "",
        "Each layer is shifted by the pooled quadratic axis fitted from rotated_points.parquet and anchored at the surface.",
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
        rows.append(process_polarity(polarity, args, radii, figure_dir))
    write_summary(output_dir, rows, args)
    print(f"Output: {output_dir}")
    print(f"Summary: {output_dir / 'representative_velocity_stack_tilted_summary.csv'}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot tilted 3D representative horizontal velocity stacks using pooled quadratic axes.")
    parser.add_argument("--template-dir", default=str(DEFAULT_TEMPLATE_DIR))
    parser.add_argument("--axis-dir", default=str(DEFAULT_AXIS_DIR))
    parser.add_argument("--catalog-dir", default=str(DEFAULT_CATALOG))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--polarities", default=",".join(DEFAULT_POLARITIES))
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
