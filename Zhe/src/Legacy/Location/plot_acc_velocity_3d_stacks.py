from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm, colors


def _read_profiles(root: Path, dataset: str) -> pd.DataFrame:
    if dataset == "all_shapes":
        frames = []
        for polarity in ("anticyclonic", "cyclonic"):
            path = root / polarity / "streamfunction_templates" / "continuous_radial_psi_profiles.parquet"
            if not path.exists():
                raise FileNotFoundError(path)
            frames.append(pd.read_parquet(path))
        return pd.concat(frames, ignore_index=True)
    path = root / "streamfunction_templates" / "continuous_radial_psi_profiles.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_parquet(path)


def _read_radii(root: Path, dataset: str) -> dict[str, float]:
    if dataset == "all_shapes":
        out: dict[str, float] = {}
        for polarity in ("anticyclonic", "cyclonic"):
            path = root / polarity / "representative_radii.csv"
            if not path.exists():
                raise FileNotFoundError(path)
            row = pd.read_csv(path).iloc[0]
            out[str(row["polarity"])] = float(row["representative_radius_m"])
        return out
    path = root / "representative_radii.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    return {str(row.polarity): float(row.representative_radius_m) for row in df.itertuples(index=False)}


def _nearest_tau(df: pd.DataFrame, tau: float) -> float:
    vals = np.sort(df["tau_center"].dropna().unique().astype("float64"))
    if len(vals) == 0:
        raise ValueError("No tau_center values")
    return float(vals[np.argmin(np.abs(vals - tau))])


def _velocity_grid(profile: pd.DataFrame, radius_m: float, tau: float, polarity: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tau_value = _nearest_tau(profile[profile["polarity"].eq(polarity)], tau)
    part = profile[(profile["polarity"].eq(polarity)) & (profile["tau_center"].eq(tau_value))].copy()
    if part.empty:
        raise ValueError(f"No profile rows for {polarity} tau={tau_value}")
    pivot = part.pivot_table(index="depth_m", columns="r_over_R", values="psi_mean", aggfunc="mean")
    pivot = pivot.sort_index().sort_index(axis=1)
    depth = pivot.index.to_numpy(dtype="float64")
    radial = pivot.columns.to_numpy(dtype="float64")
    psi = pivot.to_numpy(dtype="float64")
    r_m = np.maximum(radial * radius_m, 1.0)
    dpsi_dr = np.gradient(psi, r_m, axis=1, edge_order=1)
    v_theta = -dpsi_dr
    return depth, radial, v_theta


def _panel(ax, depth: np.ndarray, radial: np.ndarray, vtheta: np.ndarray, title: str, norm: colors.Normalize, cmap) -> None:
    theta = np.linspace(0, 2 * np.pi, 121)
    radial_idx = np.linspace(0, len(radial) - 1, min(len(radial), 24)).astype(int)
    depth_idx = np.linspace(0, len(depth) - 1, min(len(depth), 22)).astype(int)
    r = radial[radial_idx]
    th, rr = np.meshgrid(theta, r, indexing="xy")
    x = rr * np.cos(th)
    y = rr * np.sin(th)
    for k in depth_idx:
        vals = vtheta[k, radial_idx]
        val_grid = np.repeat(vals[:, None], len(theta), axis=1)
        z = np.full_like(x, -depth[k] / 1000.0)
        facecolors = cmap(norm(val_grid))
        ax.plot_surface(x, y, z, rstride=1, cstride=1, facecolors=facecolors, linewidth=0, antialiased=False, shade=False, alpha=0.82)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("x/R")
    ax.set_ylabel("y/R")
    ax.set_zlabel("depth km")
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)
    ax.set_zlim(-3.0, 0.0)
    ax.view_init(elev=24, azim=-58)
    ax.set_box_aspect((1, 1, 0.9))


def _summary_stats(profile: pd.DataFrame, radii: dict[str, float], tau: float, dataset: str) -> pd.DataFrame:
    rows = []
    for polarity in ("anticyclonic", "cyclonic"):
        depth, radial, vtheta = _velocity_grid(profile, radii[polarity], tau, polarity)
        rows.append(
            {
                "dataset": dataset,
                "polarity": polarity,
                "tau_requested": tau,
                "vtheta_min_m_s": float(np.nanmin(vtheta)),
                "vtheta_max_m_s": float(np.nanmax(vtheta)),
                "vtheta_abs_p95_m_s": float(np.nanpercentile(np.abs(vtheta), 95)),
                "depth_min_m": float(np.nanmin(depth)),
                "depth_max_m": float(np.nanmax(depth)),
                "radius_m": float(radii[polarity]),
            }
        )
    return pd.DataFrame(rows)


def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    inputs = [
        ("all_shapes", Path(args.all_root), "ACC all-shape"),
        ("coherent_only", Path(args.coherent_root), "ACC coherent-only"),
    ]
    loaded = []
    stats = []
    vmax_values = []
    for dataset, root, label in inputs:
        profile = _read_profiles(root, dataset)
        radii = _read_radii(root, dataset)
        loaded.append((dataset, label, profile, radii))
        stats.append(_summary_stats(profile, radii, args.tau, dataset))
        for polarity in ("anticyclonic", "cyclonic"):
            _, _, vtheta = _velocity_grid(profile, radii[polarity], args.tau, polarity)
            vmax_values.append(float(np.nanpercentile(np.abs(vtheta), 98)))
    vmax = max(vmax_values) if vmax_values else 1.0
    vmax = max(vmax, 1e-6)
    norm = colors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    cmap = cm.get_cmap("RdBu_r")
    fig = plt.figure(figsize=(15, 12))
    axes = [fig.add_subplot(2, 2, i + 1, projection="3d") for i in range(4)]
    idx = 0
    for dataset, label, profile, radii in loaded:
        for polarity in ("anticyclonic", "cyclonic"):
            depth, radial, vtheta = _velocity_grid(profile, radii[polarity], args.tau, polarity)
            _panel(axes[idx], depth, radial, vtheta, f"{label} / {polarity} / tau≈{_nearest_tau(profile[profile['polarity'].eq(polarity)], args.tau):.2f}", norm, cmap)
            idx += 1
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, shrink=0.62, pad=0.03)
    cbar.set_label(r"axisymmetric tangential speed $v_\theta \approx -\partial\psi/\partial r$ (m/s)")
    fig.suptitle("ACC Representative Vortex Velocity 3D Stacks", fontsize=16)
    fig.text(
        0.02,
        0.02,
        "Disks are depth layers; color is tangential speed inferred from radial streamfunction templates. "
        "This is an axisymmetric representative view, not raw instantaneous 3D velocity.",
        fontsize=9,
    )
    fig.savefig(output_dir / "acc_velocity_3d_stack_all_vs_coherent_tau050.png", dpi=220, bbox_inches="tight")
    fig.savefig(output_dir / "acc_velocity_3d_stack_all_vs_coherent_tau050.pdf", bbox_inches="tight")
    plt.close(fig)
    pd.concat(stats, ignore_index=True).to_csv(output_dir / "acc_velocity_3d_stack_stats.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot ACC representative vortex 3D stacked tangential velocity.")
    parser.add_argument("--all-root", default="/root/autodl-fs/2020_2022_acc/result/representative_vortex")
    parser.add_argument("--coherent-root", default="/root/autodl-fs/2020_2022_acc/result_coherent_only/representative_vortex")
    parser.add_argument("--output-dir", default="/root/autodl-fs/2020_2022_acc/result/Diagonise_EP_Chen_one/velocity_3d_stacks")
    parser.add_argument("--tau", type=float, default=0.5)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
