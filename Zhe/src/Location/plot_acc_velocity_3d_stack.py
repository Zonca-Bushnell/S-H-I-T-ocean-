from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm, colors
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


POLARITY_COLORS = {"cyclonic": "#2563eb", "anticyclonic": "#dc2626"}


def _load_profiles(root: Path, dataset: str, polarity: str) -> pd.DataFrame:
    if dataset == "all":
        path = root / polarity / "streamfunction_templates" / "continuous_radial_psi_profiles.parquet"
    else:
        path = root / "streamfunction_templates" / "continuous_radial_psi_profiles.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    prof = pd.read_parquet(path)
    return prof[prof["polarity"].eq(polarity)].copy()


def _load_axis(root: Path, dataset: str, polarity: str, all_shape_mode: str) -> pd.Series:
    fits = pd.read_csv(root / "axis" / "fit_coefficients.csv")
    if dataset == "coherent":
        part = fits[(fits["shape_class"].eq("coherent")) & (fits["polarity"].eq(polarity))]
    else:
        if all_shape_mode == "weighted_mean":
            part = fits[fits["polarity"].eq(polarity)].copy()
            if part.empty:
                raise ValueError(f"No axis fit rows for {dataset}/{polarity}")
            weights = part["n_objects"].to_numpy(dtype="float64")
            row = {}
            for col in part.columns:
                if col in {"shape_class", "polarity"}:
                    row[col] = "all_shapes" if col == "shape_class" else polarity
                elif np.issubdtype(part[col].dtype, np.number):
                    row[col] = float(np.average(part[col].to_numpy(dtype="float64"), weights=weights))
            return pd.Series(row)
        part = fits[(fits["shape_class"].eq(all_shape_mode)) & (fits["polarity"].eq(polarity))]
    if part.empty:
        raise ValueError(f"No axis fit row for {dataset}/{polarity}")
    return part.iloc[0]


def _axis_xy(row: pd.Series, depth_m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    z = depth_m - float(np.nanmin(depth_m))
    x = float(row["c1"]) + float(row["c2"]) * z + float(row["c3"]) * z * z
    y = float(row["c4"]) + float(row["c5"]) * z + float(row["c6"]) * z * z
    return x, y


def _representative_radius_m(root: Path, dataset: str, polarity: str, fallback_km: float) -> float:
    candidates = []
    if dataset == "all":
        candidates.append(root / polarity / "representative_radii.csv")
    candidates.append(root / "representative_radii.csv")
    for path in candidates:
        if not path.exists():
            continue
        radii = pd.read_csv(path)
        part = radii[radii["polarity"].eq(polarity)]
        if not part.empty and "representative_radius_m" in part:
            value = float(part.iloc[0]["representative_radius_m"])
            if np.isfinite(value) and value > 0:
                return value
    return fallback_km * 1000.0


def _select_depths(depths: np.ndarray, n_layers: int) -> np.ndarray:
    if len(depths) <= n_layers:
        return depths
    idx = np.unique(np.linspace(0, len(depths) - 1, n_layers).round().astype(int))
    return depths[idx]


def _velocity_slice(prof: pd.DataFrame, tau: float, depth_m: float, radius_scale: float, grid_n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    tau_values = np.sort(prof["tau_center"].dropna().unique())
    tau_sel = float(tau_values[np.argmin(np.abs(tau_values - tau))])
    depth_values = np.sort(prof["depth_m"].dropna().unique())
    depth_sel = float(depth_values[np.argmin(np.abs(depth_values - depth_m))])
    part = prof[(prof["tau_center"].eq(tau_sel)) & (prof["depth_m"].eq(depth_sel))].sort_values("r_over_R")
    r = part["r_over_R"].to_numpy(dtype="float64")
    psi = part["psi_mean"].to_numpy(dtype="float64")
    ok = np.isfinite(r) & np.isfinite(psi)
    r = r[ok]
    psi = psi[ok]
    if len(r) < 3:
        raise ValueError(f"Not enough radial points for tau={tau_sel}, depth={depth_sel}")
    radius_m = radius_scale
    r_m = np.maximum(r * radius_m, 1.0)
    dpsi_dr = np.gradient(psi, r_m, edge_order=1)
    tangential = dpsi_dr

    limit = float(np.nanmax(r))
    xy = np.linspace(-limit, limit, grid_n)
    xx, yy = np.meshgrid(xy, xy)
    rr = np.sqrt(xx * xx + yy * yy)
    mask = rr <= limit
    speed_t = np.interp(rr, r, tangential, left=np.nan, right=np.nan)
    speed_t[~mask] = np.nan
    # axisymmetric nondivergent velocity from streamfunction: u=-psi_y, v=psi_x
    with np.errstate(invalid="ignore", divide="ignore"):
        ux = -speed_t * yy / np.maximum(rr, 1e-6)
        vy = speed_t * xx / np.maximum(rr, 1e-6)
    speed = np.sqrt(ux * ux + vy * vy)
    return xx * radius_m, yy * radius_m, ux, vy, speed


def _plot_dataset(root: Path, dataset: str, output_dir: Path, tau: float, depth_count: int, grid_n: int, radius_scale_km: float, all_shape_mode: str) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    fig = plt.figure(figsize=(15, 8))
    for panel, polarity in enumerate(["cyclonic", "anticyclonic"], start=1):
        ax = fig.add_subplot(1, 2, panel, projection="3d")
        prof = _load_profiles(root, dataset, polarity)
        axis_row = _load_axis(root, dataset, polarity, all_shape_mode)
        depths = np.sort(prof["depth_m"].dropna().unique().astype("float64"))
        selected_depths = _select_depths(depths, depth_count)
        x_axis, y_axis = _axis_xy(axis_row, selected_depths)
        x_axis = x_axis - x_axis[0]
        y_axis = y_axis - y_axis[0]
        radius_m = _representative_radius_m(root, dataset, polarity, radius_scale_km)
        max_speed = 0.0
        slices = []
        for d, cx, cy in zip(selected_depths, x_axis, y_axis):
            xx, yy, ux, vy, speed = _velocity_slice(prof, tau, float(d), radius_m, grid_n)
            max_speed = max(max_speed, float(np.nanpercentile(speed, 98)))
            slices.append((d, cx, cy, xx, yy, ux, vy, speed))
        norm = colors.Normalize(vmin=0.0, vmax=max(max_speed, 1e-12))
        cmap = cm.viridis
        for d, cx, cy, xx, yy, ux, vy, speed in slices:
            z = -np.ones_like(xx) * float(d)
            x = (xx + cx) / 1000.0
            y = (yy + cy) / 1000.0
            face = cmap(norm(speed))
            face[..., -1] = np.where(np.isfinite(speed), 0.62, 0.0)
            ax.plot_surface(x, y, z, facecolors=face, linewidth=0, antialiased=False, shade=False)
            step = max(2, grid_n // 9)
            qmask = np.isfinite(speed[::step, ::step])
            ax.quiver(
                x[::step, ::step][qmask],
                y[::step, ::step][qmask],
                z[::step, ::step][qmask],
                ux[::step, ::step][qmask],
                vy[::step, ::step][qmask],
                np.zeros_like(ux[::step, ::step][qmask]),
                length=0.018,
                normalize=True,
                color="black",
                alpha=0.22,
                linewidth=0.35,
            )
        ax.plot(x_axis / 1000.0, y_axis / 1000.0, -selected_depths, color=POLARITY_COLORS[polarity], linewidth=2.8, label="representative tilted axis")
        ax.scatter(x_axis[0] / 1000.0, y_axis[0] / 1000.0, -selected_depths[0], color="black", s=24, label="surface reference")
        ax.set_title(f"{dataset} / {polarity} / tau={tau:g}")
        ax.set_xlabel("x_rot relative to surface center (km)")
        ax.set_ylabel("y_rot relative to surface center (km)")
        ax.set_zlabel("depth (m)")
        ax.view_init(elev=24, azim=-58)
        ax.set_box_aspect((1.15, 1.0, 0.75))
        ax.legend(loc="upper left", fontsize=7)
        rows.append(
            {
                "dataset": dataset,
                "polarity": polarity,
                "tau": tau,
                "depth_layers_plotted": len(selected_depths),
                "min_depth_m": float(np.nanmin(selected_depths)),
                "max_depth_m": float(np.nanmax(selected_depths)),
                "axis_surface_x_km": float(x_axis[0] / 1000.0),
                "axis_surface_y_km": float(y_axis[0] / 1000.0),
                "axis_deep_x_km": float(x_axis[-1] / 1000.0),
                "axis_deep_y_km": float(y_axis[-1] / 1000.0),
                "axis_deep_offset_km": float(np.hypot(x_axis[-1] - x_axis[0], y_axis[-1] - y_axis[0]) / 1000.0),
                "speed_p98_m_s": max_speed,
                "axis_mode": "coherent_fit" if dataset == "coherent" else all_shape_mode,
                "representative_radius_km": float(radius_m / 1000.0),
            }
        )
    mappable = cm.ScalarMappable(norm=colors.Normalize(vmin=0, vmax=max(r["speed_p98_m_s"] for r in rows)), cmap=cm.viridis)
    cbar = fig.colorbar(mappable, ax=fig.axes, shrink=0.62, pad=0.04)
    cbar.set_label("horizontal speed from radial streamfunction template (m/s)")
    fig.suptitle(f"ACC representative vortex 3D velocity stack, {dataset}, tilted axis preserved", fontsize=14)
    fig.text(
        0.02,
        0.02,
        "Only the surface center is set to zero; deeper slices keep the fitted axis offset and are not re-centered.",
        fontsize=9,
    )
    fig.savefig(output_dir / f"acc_{dataset}_velocity_3d_stack_tau{tau:.2f}.png", dpi=220, bbox_inches="tight")
    fig.savefig(output_dir / f"acc_{dataset}_velocity_3d_stack_tau{tau:.2f}.pdf", bbox_inches="tight")
    plt.close(fig)
    return {"dataset": dataset, "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot ACC representative vortex 3D stacked velocity slices with tilted offsets preserved.")
    parser.add_argument("--all-root", default="/root/autodl-fs/2020_2022_acc/result/representative_vortex")
    parser.add_argument("--coherent-root", default="/root/autodl-fs/2020_2022_acc/result_coherent_only/representative_vortex")
    parser.add_argument("--output-dir", default="/root/autodl-fs/2020_2022_acc/result/Diagonise_EP_Chen_one/velocity_3d_stack")
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument("--depth-count", type=int, default=8)
    parser.add_argument("--grid-n", type=int, default=41)
    parser.add_argument("--radius-scale-km", type=float, default=100.0, help="Fallback radius if representative_radii.csv is missing.")
    parser.add_argument("--all-shape-axis-mode", choices=["weighted_mean", "coherent", "complex", "mixed", "transitional"], default="weighted_mean")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = [
        _plot_dataset(Path(args.all_root), "all", output_dir, args.tau, args.depth_count, args.grid_n, args.radius_scale_km, args.all_shape_axis_mode),
        _plot_dataset(Path(args.coherent_root), "coherent", output_dir, args.tau, args.depth_count, args.grid_n, args.radius_scale_km, args.all_shape_axis_mode),
    ]
    flat = [row for item in summaries for row in item["rows"]]
    pd.DataFrame(flat).to_csv(output_dir / "velocity_3d_stack_summary.csv", index=False)
    (output_dir / "velocity_3d_stack_summary.json").write_text(json.dumps(flat, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "README_zh.md").write_text(
        "# ACC 代表涡旋 3D 速度堆叠图\n\n"
        f"- tau: `{args.tau}`。\n"
        f"- 每个深度切片用 `psi_mean(r,z,tau)` 的径向导数恢复轴对称水平速度；径向尺度优先采用 `representative_radii.csv`。\n"
        "- 图中只把表层中心设为原点；每层切片按代表性轴线相对表层的 `x_rot(z), y_rot(z)` 平移，因此保留倾斜偏移信息，不做逐层中心对齐。\n"
        "- all-shape 的轴线使用各 shape fit 按对象数加权平均；coherent-only 使用 coherent fit。\n"
        "- 图中的速度是代表性径向模板速度，不是原始三维日场瞬时速度。\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
