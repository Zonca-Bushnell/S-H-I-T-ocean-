from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _signed_lon_delta(lon: pd.Series | np.ndarray, lon0: float) -> np.ndarray:
    return (np.asarray(lon, dtype="float64") - lon0 + 180.0) % 360.0 - 180.0


def _xy_km(lon: pd.Series | np.ndarray, lat: pd.Series | np.ndarray, lon0: float, lat0: float) -> tuple[np.ndarray, np.ndarray]:
    x = _signed_lon_delta(lon, lon0) * 111.2 * math.cos(math.radians(lat0))
    y = (np.asarray(lat, dtype="float64") - lat0) * 111.2
    return x, y


def _set_equal_xy(ax, x: np.ndarray, y: np.ndarray) -> None:
    finite = np.isfinite(x) & np.isfinite(y)
    if not finite.any():
        return
    xmin, xmax = float(np.nanmin(x[finite])), float(np.nanmax(x[finite]))
    ymin, ymax = float(np.nanmin(y[finite])), float(np.nanmax(y[finite]))
    cx = 0.5 * (xmin + xmax)
    cy = 0.5 * (ymin + ymax)
    half = max(20.0, 0.55 * max(xmax - xmin, ymax - ymin))
    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)


def _plot_long_track_map(points: pd.DataFrame, tracks: pd.DataFrame, fig_dir: Path, *, min_snapshots: int) -> None:
    colors = {"cyclonic": "#2563eb", "anticyclonic": "#dc2626"}
    keep_ids = tracks.loc[tracks["n_snapshots"].ge(min_snapshots), "hua_track_id"].tolist()
    if not keep_ids:
        keep_ids = tracks.sort_values(["n_snapshots", "total_pass_layers"], ascending=False).head(30)["hua_track_id"].tolist()
    use = points[points["hua_track_id"].isin(keep_ids)].copy()
    fig, ax = plt.subplots(figsize=(14, 5.4))
    for track_id, group in use.sort_values("date_ts").groupby("hua_track_id"):
        pol = str(group["polarity"].iloc[0])
        color = colors.get(pol, "#525252")
        lw = 1.0 + 0.28 * min(len(group), 10)
        ax.plot(group["center_lon"], group["center_lat"], color=color, alpha=0.72, linewidth=lw)
        ax.scatter(
            group["center_lon"],
            group["center_lat"],
            s=18 + 1.5 * group["pass_layers"],
            color=color,
            edgecolor="white",
            linewidth=0.3,
            alpha=0.85,
        )
    top_labels = tracks[tracks["hua_track_id"].isin(keep_ids)].sort_values(["n_snapshots", "total_pass_layers"], ascending=False).head(16)
    for _, track in top_labels.iterrows():
        group = points[points["hua_track_id"].eq(track["hua_track_id"])].sort_values("date_ts")
        if group.empty:
            continue
        ax.text(float(group["center_lon"].iloc[-1]), float(group["center_lat"].iloc[-1]), str(int(track["hua_track_id"])), fontsize=8)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-66, -44)
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_title(f"ACC Hua-calibrated trajectories, tracks with >= {min_snapshots} snapshots")
    ax.grid(alpha=0.25)
    ax.legend(
        handles=[
            plt.Line2D([0], [0], color=colors["cyclonic"], lw=2.5, label="cyclonic"),
            plt.Line2D([0], [0], color=colors["anticyclonic"], lw=2.5, label="anticyclonic"),
        ],
        loc="lower left",
    )
    fig.savefig(fig_dir / "hua_paper_like_long_tracks_map.png", dpi=220, bbox_inches="tight")
    fig.savefig(fig_dir / "hua_paper_like_long_tracks_map.pdf", bbox_inches="tight")
    plt.close(fig)


def _plot_depth_time(points: pd.DataFrame, tracks: pd.DataFrame, fig_dir: Path) -> None:
    top = tracks.sort_values(["n_snapshots", "total_pass_layers"], ascending=False).head(8)["hua_track_id"].tolist()
    fig, axes = plt.subplots(4, 2, figsize=(12, 12), sharex=False, sharey=True)
    axes = axes.ravel()
    for ax, track_id in zip(axes, top):
        group = points[points["hua_track_id"].eq(track_id)].sort_values("date_ts")
        days = (group["date_ts"] - group["date_ts"].min()).dt.days.to_numpy(dtype="float64")
        depth_km = group["max_depth_m"].to_numpy(dtype="float64") / 1000.0
        color = "#2563eb" if str(group["polarity"].iloc[0]) == "cyclonic" else "#dc2626"
        ax.plot(days, depth_km, color=color, linewidth=2.0)
        ax.scatter(days, depth_km, s=28 + group["pass_layers"], color=color, edgecolor="white", linewidth=0.35)
        ax.invert_yaxis()
        ax.grid(alpha=0.25)
        ax.set_title(f"track {track_id}, {group['polarity'].iloc[0]}, n={len(group)}")
        ax.set_xlabel("days from first detection")
        ax.set_ylabel("max passed depth (km)")
    for ax in axes[len(top) :]:
        ax.axis("off")
    fig.suptitle("Hua 3D extension depth along linked trajectories", fontsize=14)
    fig.tight_layout()
    fig.savefig(fig_dir / "hua_paper_like_depth_time_examples.png", dpi=220, bbox_inches="tight")
    fig.savefig(fig_dir / "hua_paper_like_depth_time_examples.pdf", bbox_inches="tight")
    plt.close(fig)


def _plot_layer_clouds(centers: pd.DataFrame, points: pd.DataFrame, tracks: pd.DataFrame, fig_dir: Path) -> None:
    top = tracks.sort_values(["n_snapshots", "total_pass_layers"], ascending=False).head(6)["hua_track_id"].tolist()
    fig = plt.figure(figsize=(14, 9))
    for panel, track_id in enumerate(top, start=1):
        group = points[points["hua_track_id"].eq(track_id)].sort_values("date_ts")
        rows = centers[centers["hua_object_id"].isin(group["hua_object_id"]) & centers["hua_pass"].astype(bool)].copy()
        ax = fig.add_subplot(2, 3, panel, projection="3d")
        if rows.empty:
            ax.axis("off")
            continue
        lon0 = float(group["center_lon"].iloc[0])
        lat0 = float(group["center_lat"].iloc[0])
        x, y = _xy_km(rows["center_lon"], rows["center_lat"], lon0, lat0)
        z = -rows["depth_m"].to_numpy(dtype="float64") / 1000.0
        date_lookup = group.set_index("hua_object_id")["date_ts"].to_dict()
        t = pd.to_datetime(rows["hua_object_id"].map(date_lookup))
        day = (t - group["date_ts"].min()).dt.days.to_numpy(dtype="float64")
        for _, day_group in rows.assign(day=day, x=x, y=y, z=z).groupby("hua_object_id", sort=False):
            ax.plot(day_group["x"], day_group["y"], day_group["z"], color="#64748b", alpha=0.45, linewidth=0.9)
        sc = ax.scatter(x, y, z, c=day, cmap="viridis", s=22, depthshade=True)
        _set_equal_xy(ax, x, y)
        ax.set_zlim(float(np.nanmin(z)) - 0.05, 0.05)
        ax.set_title(f"track {track_id}, {group['polarity'].iloc[0]}\n{group['date'].iloc[0]} to {group['date'].iloc[-1]}")
        ax.set_xlabel("east km")
        ax.set_ylabel("north km")
        ax.set_zlabel("depth km")
        ax.view_init(elev=23, azim=-55)
    fig.colorbar(sc, ax=fig.axes, shrink=0.68, pad=0.03, label="days from first detection")
    fig.suptitle("Hua paper-like 3D centers: linked objects and passed depth layers", fontsize=14)
    fig.savefig(fig_dir / "hua_paper_like_3d_layer_centers.png", dpi=220, bbox_inches="tight")
    fig.savefig(fig_dir / "hua_paper_like_3d_layer_centers.pdf", bbox_inches="tight")
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    root = Path(args.input_dir)
    tracking = root / "tracking"
    fig_dir = root / "paper_like_figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    centers = pd.read_parquet(root / "centers_hua_style.parquet")
    points = pd.read_parquet(tracking / "hua_track_points.parquet")
    tracks = pd.read_parquet(tracking / "hua_tracks.parquet")
    points["date_ts"] = pd.to_datetime(points["date"])
    _plot_long_track_map(points, tracks, fig_dir, min_snapshots=args.min_snapshots)
    _plot_depth_time(points, tracks, fig_dir)
    _plot_layer_clouds(centers, points, tracks, fig_dir)
    lines = [
        "# Hua 论文式图像复刻说明",
        "",
        "这些图只使用 Hua SSH+velocity hybrid 检测与原型 tracking 的输出，不读取我们原有 catalog，也不使用 completed centers。",
        "",
        "- `hua_paper_like_long_tracks_map.png`：只显示较长 track，减少编号遮挡，用来检查 ACC 域内轨迹连续性。",
        "- `hua_paper_like_depth_time_examples.png`：显示每条轨迹随时间能通过 Hua 圆周检验的最大深度。",
        "- `hua_paper_like_3d_layer_centers.png`：把同一轨迹中每个 snapshot 的通过层中心画成三维点云，颜色为生命周期内日期。",
        "",
        "注意：这里仍是 ACC-calibrated Hua。严格 Red Sea 参数在 ACC 上太稀疏，难以形成论文式连续轨迹。",
    ]
    (fig_dir / "paper_like_figures_readme_zh.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {fig_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw paper-like Hua trajectory and 3D extension figures for ACC replication.")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--min-snapshots", type=int, default=3)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
