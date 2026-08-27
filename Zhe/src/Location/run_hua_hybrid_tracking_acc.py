from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


EARTH_RADIUS_KM = 6371.0


def _haversine_km(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = phi2 - phi1
    dlambda = math.radians(((lon2 - lon1 + 180.0) % 360.0) - 180.0)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_KM * math.asin(min(1.0, math.sqrt(a)))


def _signed_lon_delta(lon: pd.Series, lon0: float) -> pd.Series:
    return (lon - lon0 + 180.0) % 360.0 - 180.0


def _deduplicate_snapshots(centers: pd.DataFrame, *, dedup_km: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    passed = centers[centers["hua_pass"].astype(bool)].copy()
    if passed.empty:
        return pd.DataFrame(), pd.DataFrame()

    surface = (
        passed.sort_values(["hua_object_id", "depth_index"])
        .groupby("hua_object_id", as_index=False)
        .first()
    )
    layer_stats = (
        passed.groupby("hua_object_id")
        .agg(
            pass_layers=("depth_index", "size"),
            max_depth_m=("depth_m", "max"),
            mean_radius_cells=("accepted_radius_cells", "mean"),
            mean_center_speed_ms=("center_speed_ms", "mean"),
            min_center_speed_ms=("center_speed_ms", "min"),
        )
        .reset_index()
    )
    snapshots = surface.merge(layer_stats, on="hua_object_id", how="left", suffixes=("", "_stat"))
    snapshots["snapshot_rank_score"] = snapshots["pass_layers"] * 10.0 + snapshots["ssh_value_m"].abs()
    snapshots = snapshots.sort_values(["date", "polarity", "snapshot_rank_score"], ascending=[True, True, False]).reset_index(drop=True)

    kept: list[dict[str, object]] = []
    duplicates: list[dict[str, object]] = []
    for (day, polarity), group in snapshots.groupby(["date", "polarity"], sort=True):
        day_kept: list[pd.Series] = []
        for _, row in group.iterrows():
            duplicate_of = None
            for kept_row in day_kept:
                dist = _haversine_km(float(row["center_lon"]), float(row["center_lat"]), float(kept_row["center_lon"]), float(kept_row["center_lat"]))
                if dist <= dedup_km:
                    duplicate_of = kept_row
                    break
            if duplicate_of is None:
                day_kept.append(row)
                out = row.to_dict()
                out["dedup_snapshot_id"] = f"{str(day).replace('-', '')}_{polarity}_{len(day_kept):04d}"
                kept.append(out)
            else:
                duplicates.append(
                    {
                        "date": day,
                        "polarity": polarity,
                        "hua_object_id": row["hua_object_id"],
                        "duplicate_of": duplicate_of["hua_object_id"],
                        "distance_km": _haversine_km(
                            float(row["center_lon"]),
                            float(row["center_lat"]),
                            float(duplicate_of["center_lon"]),
                            float(duplicate_of["center_lat"]),
                        ),
                    }
                )
    return pd.DataFrame(kept), pd.DataFrame(duplicates)


def _track_snapshots(snapshots: pd.DataFrame, *, max_gap_days: int, max_speed_km_day: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    if snapshots.empty:
        return pd.DataFrame(), pd.DataFrame()
    snaps = snapshots.copy()
    snaps["date_ts"] = pd.to_datetime(snaps["date"])
    active: list[dict[str, object]] = []
    point_rows: list[dict[str, object]] = []
    next_track = 1

    for _, row in snaps.sort_values(["date_ts", "polarity", "snapshot_rank_score"], ascending=[True, True, False]).iterrows():
        best_idx = None
        best_score = np.inf
        best_dist = np.nan
        best_gap = 0
        for idx, track in enumerate(active):
            gap = int((row["date_ts"] - track["last_date"]).days)
            if gap < 1 or gap > max_gap_days:
                continue
            if row["polarity"] != track["polarity"]:
                continue
            dist = _haversine_km(float(track["last_lon"]), float(track["last_lat"]), float(row["center_lon"]), float(row["center_lat"]))
            allowed = max_speed_km_day * gap
            if dist > allowed:
                continue
            score = dist / max(gap, 1)
            if score < best_score:
                best_idx = idx
                best_score = score
                best_dist = dist
                best_gap = gap
        if best_idx is None:
            track_id = next_track
            next_track += 1
            active.append(
                {
                    "hua_track_id": track_id,
                    "polarity": row["polarity"],
                    "last_date": row["date_ts"],
                    "last_lon": float(row["center_lon"]),
                    "last_lat": float(row["center_lat"]),
                    "n": 1,
                }
            )
            link_distance = np.nan
            gap_days = 0
        else:
            track = active[best_idx]
            track_id = int(track["hua_track_id"])
            link_distance = float(best_dist)
            gap_days = int(best_gap)
            track["last_date"] = row["date_ts"]
            track["last_lon"] = float(row["center_lon"])
            track["last_lat"] = float(row["center_lat"])
            track["n"] = int(track["n"]) + 1
        out = row.to_dict()
        out["hua_track_id"] = track_id
        out["link_distance_km"] = link_distance
        out["link_gap_days"] = gap_days
        point_rows.append(out)

    points = pd.DataFrame(point_rows)
    tracks = (
        points.groupby("hua_track_id")
        .agg(
            polarity=("polarity", "first"),
            n_snapshots=("dedup_snapshot_id", "size"),
            start_date=("date", "min"),
            end_date=("date", "max"),
            mean_lon=("center_lon", "mean"),
            mean_lat=("center_lat", "mean"),
            max_depth_m=("max_depth_m", "max"),
            total_pass_layers=("pass_layers", "sum"),
            median_radius_cells=("mean_radius_cells", "median"),
            median_speed_ms=("mean_center_speed_ms", "median"),
        )
        .reset_index()
    )
    tracks["duration_days"] = (pd.to_datetime(tracks["end_date"]) - pd.to_datetime(tracks["start_date"])).dt.days + 1
    return tracks, points


def _plot_tracks(points: pd.DataFrame, tracks: pd.DataFrame, out_dir: Path) -> None:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    if points.empty:
        return

    fig, ax = plt.subplots(figsize=(13.5, 5.0))
    colors = {"cyclonic": "#2563eb", "anticyclonic": "#dc2626"}
    for track_id, group in points.sort_values("date_ts").groupby("hua_track_id"):
        pol = str(group["polarity"].iloc[0])
        lw = 1.2 + 0.25 * min(len(group), 8)
        ax.plot(group["center_lon"], group["center_lat"], color=colors.get(pol, "#525252"), alpha=0.75, linewidth=lw)
        ax.scatter(group["center_lon"], group["center_lat"], color=colors.get(pol, "#525252"), s=20 + 2 * group["pass_layers"], alpha=0.85)
        ax.text(float(group["center_lon"].iloc[-1]), float(group["center_lat"].iloc[-1]), str(track_id), fontsize=7)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-66, -44)
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_title("ACC Hua-replicated strict detections linked into prototype trajectories")
    ax.grid(alpha=0.25)
    handles = [
        plt.Line2D([0], [0], color=colors["cyclonic"], lw=2, label="cyclonic"),
        plt.Line2D([0], [0], color=colors["anticyclonic"], lw=2, label="anticyclonic"),
    ]
    ax.legend(handles=handles, loc="lower left")
    fig.savefig(fig_dir / "hua_tracks_map.png", dpi=200, bbox_inches="tight")
    fig.savefig(fig_dir / "hua_tracks_map.pdf", bbox_inches="tight")
    plt.close(fig)

    top_tracks = tracks.sort_values(["n_snapshots", "total_pass_layers"], ascending=False).head(6)["hua_track_id"].tolist()
    fig = plt.figure(figsize=(14, 8))
    for panel, track_id in enumerate(top_tracks, start=1):
        group = points[points["hua_track_id"].eq(track_id)].sort_values("date_ts")
        ax = fig.add_subplot(2, 3, panel, projection="3d")
        lon0 = float(group["center_lon"].iloc[0])
        lat0 = float(group["center_lat"].iloc[0])
        x = _signed_lon_delta(group["center_lon"], lon0).to_numpy(dtype="float64") * 111.2 * math.cos(math.radians(lat0))
        y = (group["center_lat"] - lat0).to_numpy(dtype="float64") * 111.2
        t = (group["date_ts"] - group["date_ts"].min()).dt.days.to_numpy(dtype="float64")
        ax.plot(x, y, t, color="#7c3aed", linewidth=2.2)
        ax.scatter(x, y, t, c=group["pass_layers"], cmap="viridis", s=55)
        ax.set_title(f"track {track_id}, {group['polarity'].iloc[0]}\n{group['date'].iloc[0]} to {group['date'].iloc[-1]}")
        ax.set_xlabel("east km")
        ax.set_ylabel("north km")
        ax.set_zlabel("days")
        ax.view_init(elev=24, azim=-55)
    fig.suptitle("Prototype Hua trajectories: surface centers colored by 3D pass-layer count", fontsize=14)
    fig.savefig(fig_dir / "hua_tracks_3d_examples.png", dpi=200, bbox_inches="tight")
    fig.savefig(fig_dir / "hua_tracks_3d_examples.pdf", bbox_inches="tight")
    plt.close(fig)


def _write_summary(out_dir: Path, snapshots: pd.DataFrame, duplicates: pd.DataFrame, tracks: pd.DataFrame, points: pd.DataFrame) -> None:
    summary = {
        "deduplicated_snapshots": int(len(snapshots)),
        "duplicate_surface_candidates_removed": int(len(duplicates)),
        "tracks": int(len(tracks)),
        "tracks_len_ge_2": int((tracks["n_snapshots"] >= 2).sum()) if not tracks.empty else 0,
        "tracks_len_ge_3": int((tracks["n_snapshots"] >= 3).sum()) if not tracks.empty else 0,
        "max_track_snapshots": int(tracks["n_snapshots"].max()) if not tracks.empty else 0,
        "total_pass_layers_in_tracked_points": int(points["pass_layers"].sum()) if not points.empty else 0,
    }
    (out_dir / "hua_tracking_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# Hua 论文式 trajectory 复刻结果",
        "",
        "这一步是在 Hua strict detection 输出之后追加的 tracking/visualization 层。它不重新识别涡旋，也不使用我们现有 ACC catalog/tracks。",
        "",
        "## 关键结论",
        "",
        f"- 去重后的通过 snapshot：`{summary['deduplicated_snapshots']}`。",
        f"- 同日重复候选删除：`{summary['duplicate_surface_candidates_removed']}`。",
        f"- prototype tracks：`{summary['tracks']}`。",
        f"- 长度 >=2 的 tracks：`{summary['tracks_len_ge_2']}`。",
        f"- 长度 >=3 的 tracks：`{summary['tracks_len_ge_3']}`。",
        f"- 最长 track snapshot 数：`{summary['max_track_snapshots']}`。",
        "",
        "## 为什么还不像论文图那样完美",
        "",
        "Hua 论文中的漂亮轨迹图依赖两个前提：第一，每帧能检测到足够多连续出现的三维结构；第二，Feature Tracking 框架把每帧结构输出为 `.uocd/.trak/.group` 后再跨帧关联。我们这里的 strict Hua detection 在 ACC 60-frame top-80 候选中只留下少量通过 snapshot，因此可连接轨迹很短。这个结果不是画图失败，而是 detection input 太稀疏，导致 tracking 没有足够连续点。",
        "",
        "## 物理解释",
        "",
        "ACC 的 30-180 天带通信号里，SSH 极值附近常有低速点，但只有少数点同时满足近圆周切向、方向连续、对称和两侧速度反转。Hua strict 方法筛出的更像是高置信度闭合旋转瞬时结构，而不是覆盖所有中尺度涡的全量轨迹产品。",
        "",
        "## 输出",
        "",
        "- `hua_snapshots_dedup.parquet/csv`：去重后的通过 snapshot。",
        "- `hua_tracks.parquet/csv`：prototype tracks。",
        "- `hua_track_points.parquet/csv`：track 点表。",
        "- `figures/hua_tracks_map.png`：论文式 ACC 平面轨迹图。",
        "- `figures/hua_tracks_3d_examples.png`：轨迹 3D 示例图。",
    ]
    (out_dir / "hua_tracking_summary_zh.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    root = Path(args.input_dir)
    out_dir = Path(args.output_dir) if args.output_dir else root / "tracking"
    out_dir.mkdir(parents=True, exist_ok=True)
    centers = pd.read_parquet(root / "centers_hua_style.parquet")
    snapshots, duplicates = _deduplicate_snapshots(centers, dedup_km=args.dedup_km)
    tracks, points = _track_snapshots(snapshots, max_gap_days=args.max_gap_days, max_speed_km_day=args.max_speed_km_day)
    snapshots.to_parquet(out_dir / "hua_snapshots_dedup.parquet", index=False)
    snapshots.to_csv(out_dir / "hua_snapshots_dedup.csv", index=False)
    duplicates.to_parquet(out_dir / "hua_duplicate_snapshots.parquet", index=False)
    duplicates.to_csv(out_dir / "hua_duplicate_snapshots.csv", index=False)
    tracks.to_parquet(out_dir / "hua_tracks.parquet", index=False)
    tracks.to_csv(out_dir / "hua_tracks.csv", index=False)
    points.to_parquet(out_dir / "hua_track_points.parquet", index=False)
    points.to_csv(out_dir / "hua_track_points.csv", index=False)
    _plot_tracks(points, tracks, out_dir)
    _write_summary(out_dir, snapshots, duplicates, tracks, points)
    print(
        json.dumps(
            {
                "snapshots": int(len(snapshots)),
                "duplicates": int(len(duplicates)),
                "tracks": int(len(tracks)),
                "tracks_len_ge_2": int((tracks["n_snapshots"] >= 2).sum()) if not tracks.empty else 0,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Track Hua-replicated ACC strict detections and plot paper-like trajectories.")
    parser.add_argument("--input-dir", default="/root/autodl-fs/2020_2022_acc/hua_paper_replication/window_20200101_20200301")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--dedup-km", type=float, default=30.0)
    parser.add_argument("--max-gap-days", type=int, default=3)
    parser.add_argument("--max-speed-km-day", type=float, default=75.0)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
