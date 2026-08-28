from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .axis_streamfunction import DEFAULT_AXIS_DIR, DEFAULT_CATALOG, DEFAULT_OUTPUT as DEFAULT_TEMPLATE_DIR, fit_rank1, parse_csv_list


DEFAULT_OUTPUT = Path(r"E:\Verify\outputs\kuroshio_cmems_3d\representative_velocity_stack_1993_2022")
DEFAULT_POLARITIES = ("cyclonic", "anticyclonic")
DEFAULT_DEPTH_LEVELS = (0.0, 50.0, 100.0, 200.0, 300.0, 500.0, 800.0, 1200.0)


def load_representative_radii(axis_dir: Path, catalog_dir: Path) -> dict[str, float]:
    objects = pd.read_parquet(axis_dir / "object_diagnostics.parquet", columns=["eddy3d_object_id", "polarity", "is_usable"])
    radii = pd.read_parquet(catalog_dir / "vertical_objects.parquet", columns=["eddy3d_object_id", "mean_radius_m"])
    data = objects[objects["is_usable"]].merge(radii, on="eddy3d_object_id", how="inner")
    data = data[np.isfinite(data["mean_radius_m"]) & (data["mean_radius_m"] > 0)].copy()
    return {str(polarity): float(part["mean_radius_m"].median()) for polarity, part in data.groupby("polarity")}


def load_profile_matrix(template_dir: Path, polarity: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    profiles = pd.read_parquet(template_dir / "radial_psi_profiles.parquet")
    part = profiles[profiles["polarity"] == polarity].copy()
    if part.empty:
        raise ValueError(f"No radial psi profiles found for polarity {polarity!r}.")
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


def velocity_from_psi(psi: np.ndarray, r: np.ndarray, radius_m: float) -> np.ndarray:
    dpsi_dr_norm = np.gradient(psi, r, axis=1)
    return dpsi_dr_norm / radius_m


def nearest_depth_indices(depth: np.ndarray, requested_depths: tuple[float, ...]) -> list[int]:
    indices: list[int] = []
    for value in requested_depths:
        index = int(np.nanargmin(np.abs(depth - value)))
        if index not in indices:
            indices.append(index)
    return indices


def make_xy_grid(extent: float, grid_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xy = np.linspace(-extent, extent, grid_size)
    x, y = np.meshgrid(xy, xy)
    r_norm = np.hypot(x, y)
    return x, y, r_norm


def velocity_components_on_grid(
    vtheta_by_depth: np.ndarray,
    r: np.ndarray,
    depth_indices: list[int],
    x: np.ndarray,
    y: np.ndarray,
    r_norm: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    u_layers = []
    v_layers = []
    speed_layers = []
    angle_good = r_norm > 1e-9
    for depth_index in depth_indices:
        vtheta = np.interp(np.clip(r_norm.ravel(), r.min(), r.max()), r, vtheta_by_depth[depth_index]).reshape(r_norm.shape)
        vtheta = np.where(r_norm <= r.max(), vtheta, np.nan)
        u = np.zeros_like(vtheta)
        v = np.zeros_like(vtheta)
        u[angle_good] = -vtheta[angle_good] * y[angle_good] / r_norm[angle_good]
        v[angle_good] = vtheta[angle_good] * x[angle_good] / r_norm[angle_good]
        u = np.where(np.isfinite(vtheta), u, np.nan)
        v = np.where(np.isfinite(vtheta), v, np.nan)
        u_layers.append(u)
        v_layers.append(v)
        speed_layers.append(np.hypot(u, v))
    return np.asarray(u_layers), np.asarray(v_layers), np.asarray(speed_layers)


def plot_vtheta_rz(label: str, depth: np.ndarray, r: np.ndarray, vtheta: np.ndarray, output_dir: Path) -> None:
    vmax = float(np.nanpercentile(np.abs(vtheta), 98))
    fig, ax = plt.subplots(figsize=(8, 6), dpi=160)
    mesh = ax.pcolormesh(r, depth, vtheta, shading="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.invert_yaxis()
    ax.axvline(1.0, color="k", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_xlabel("r/R")
    ax.set_ylabel("depth m")
    ax.set_title(f"{label}: tangential velocity v_theta(r,z)")
    fig.colorbar(mesh, ax=ax, label="v_theta (m/s)")
    fig.tight_layout()
    fig.savefig(output_dir / f"{label}_vtheta_rz.png")
    plt.close(fig)


def plot_png_stack(
    label: str,
    depth: np.ndarray,
    depth_indices: list[int],
    x: np.ndarray,
    y: np.ndarray,
    u_layers: np.ndarray,
    v_layers: np.ndarray,
    speed_layers: np.ndarray,
    *,
    arrow_step: int,
    output_dir: Path,
) -> None:
    vmax = float(np.nanpercentile(speed_layers, 98))
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0
    fig = plt.figure(figsize=(10, 8), dpi=170)
    ax = fig.add_subplot(111, projection="3d")
    norm = plt.Normalize(vmin=0.0, vmax=vmax)
    cmap = plt.get_cmap("magma")
    for layer_idx, depth_index in enumerate(depth_indices):
        z = np.full_like(x, -float(depth[depth_index]))
        colors = cmap(norm(speed_layers[layer_idx]))
        colors[..., 3] = np.where(np.isfinite(speed_layers[layer_idx]), 0.36, 0.0)
        ax.plot_surface(x, y, z, facecolors=colors, linewidth=0, antialiased=False, shade=False)
        xs = x[::arrow_step, ::arrow_step]
        ys = y[::arrow_step, ::arrow_step]
        zs = z[::arrow_step, ::arrow_step]
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
        ax.text(float(np.nanmax(x)) + 0.12, float(np.nanmax(y)) + 0.12, -float(depth[depth_index]), f"{depth[depth_index]:.0f} m", fontsize=8)
    scalar = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    scalar.set_array([])
    fig.colorbar(scalar, ax=ax, shrink=0.65, pad=0.08, label="speed (m/s)")
    ax.set_xlabel("x/R")
    ax.set_ylabel("y/R")
    ax.set_zlabel("-depth m")
    ax.set_title(f"{label}: 3D stacked representative horizontal velocity")
    ax.view_init(elev=23, azim=-52)
    ax.set_box_aspect((1, 1, 0.72))
    fig.tight_layout()
    fig.savefig(output_dir / f"{label}_velocity_stack_3d.png")
    plt.close(fig)


def plot_html_stack(
    label: str,
    depth: np.ndarray,
    depth_indices: list[int],
    x: np.ndarray,
    y: np.ndarray,
    u_layers: np.ndarray,
    v_layers: np.ndarray,
    speed_layers: np.ndarray,
    *,
    arrow_step: int,
    output_dir: Path,
) -> None:
    import plotly.graph_objects as go

    vmax = float(np.nanpercentile(speed_layers, 98))
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0
    fig = go.Figure()
    for layer_idx, depth_index in enumerate(depth_indices):
        z = np.full_like(x, -float(depth[depth_index]))
        fig.add_trace(
            go.Surface(
                x=x,
                y=y,
                z=z,
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
        xs = x[::arrow_step, ::arrow_step]
        ys = y[::arrow_step, ::arrow_step]
        zs = z[::arrow_step, ::arrow_step]
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
    fig.update_layout(
        title=f"{label}: 3D stacked representative horizontal velocity",
        scene={
            "xaxis_title": "x/R",
            "yaxis_title": "y/R",
            "zaxis_title": "-depth m",
            "aspectmode": "manual",
            "aspectratio": {"x": 1, "y": 1, "z": 0.72},
            "camera": {"eye": {"x": 1.55, "y": -1.75, "z": 0.9}},
        },
        margin={"l": 0, "r": 0, "t": 48, "b": 0},
    )
    fig.write_html(output_dir / f"{label}_velocity_stack_3d.html", include_plotlyjs="cdn")


def process_polarity(
    polarity: str,
    args: argparse.Namespace,
    radii: dict[str, float],
    figure_dir: Path,
) -> dict:
    radius_m = radii.get(polarity)
    if radius_m is None:
        raise ValueError(f"No representative radius found for polarity {polarity!r}.")
    depth, r, _, psi, metrics = load_profile_matrix(Path(args.template_dir), polarity)
    vtheta = velocity_from_psi(psi, r, radius_m)
    depth_indices = nearest_depth_indices(depth, parse_float_list(args.depth_levels))
    x, y, r_norm = make_xy_grid(float(args.xy_extent), int(args.grid_size))
    u_layers, v_layers, speed_layers = velocity_components_on_grid(vtheta, r, depth_indices, x, y, r_norm)
    label = f"all_shapes_{polarity}"

    if args.png:
        plot_png_stack(label, depth, depth_indices, x, y, u_layers, v_layers, speed_layers, arrow_step=int(args.arrow_step), output_dir=figure_dir)
        plot_vtheta_rz(label, depth, r, vtheta, figure_dir)
    if args.html:
        plot_html_stack(label, depth, depth_indices, x, y, u_layers, v_layers, speed_layers, arrow_step=int(args.arrow_step), output_dir=figure_dir)

    used_depths = [float(depth[index]) for index in depth_indices]
    return {
        "polarity": polarity,
        "label": label,
        "median_radius_m": radius_m,
        "depth_levels_m": ",".join(f"{value:.3f}" for value in used_depths),
        "rank1_energy_fraction": metrics.get("rank1_energy_fraction", np.nan),
        "relative_rmse": metrics.get("relative_rmse", np.nan),
        "max_abs_vtheta_m_s": float(np.nanmax(np.abs(vtheta))),
        "p98_abs_vtheta_m_s": float(np.nanpercentile(np.abs(vtheta), 98)),
        "surface_peak_abs_vtheta_m_s": float(np.nanmax(np.abs(vtheta[0]))),
        "max_speed_stack_m_s": float(np.nanmax(speed_layers)),
        "p98_speed_stack_m_s": float(np.nanpercentile(speed_layers, 98)),
    }


def parse_float_list(value: str | tuple[float, ...]) -> tuple[float, ...]:
    if isinstance(value, tuple):
        return value
    return tuple(float(part.strip()) for part in value.split(",") if part.strip())


def write_summary(output_dir: Path, rows: list[dict], args: argparse.Namespace) -> None:
    summary = pd.DataFrame.from_records(rows)
    summary.to_csv(output_dir / "representative_velocity_stack_summary.csv", index=False)
    lines = [
        "# Representative velocity stack summary",
        "",
        f"- Template dir: `{args.template_dir}`",
        f"- Axis dir: `{args.axis_dir}`",
        f"- Catalog dir: `{args.catalog_dir}`",
        f"- Polarities: {args.polarities}",
        f"- Requested depth levels: {args.depth_levels}",
        f"- x/y extent: +/- {args.xy_extent} R",
        f"- Grid size: {args.grid_size}",
        "",
        "Velocity is derived from the rank-1 representative streamfunction using `v_theta=dpsi/dr_m`, then projected as horizontal vectors.",
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
    print(f"Summary: {output_dir / 'representative_velocity_stack_summary.csv'}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot stacked 3D representative horizontal velocity fields from psi(r,z).")
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

