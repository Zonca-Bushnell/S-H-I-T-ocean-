from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _read_table(path: Path) -> pd.DataFrame:
    if path.with_suffix(".parquet").exists():
        return pd.read_parquet(path.with_suffix(".parquet"), engine="fastparquet")
    return pd.read_csv(path.with_suffix(".csv"))


def _read_voxels_for_date(root: Path, date: str) -> pd.DataFrame:
    if (root / "object_voxels.parquet").exists() or (root / "object_voxels.csv").exists():
        voxels = _read_table(root / "object_voxels")
        voxels["date"] = voxels["date"].astype(str)
        return voxels[voxels["date"].eq(date)].copy()
    day = pd.to_datetime(date).date()
    path = root / "object_voxels_parts" / f"year={day.year}" / f"date={day:%Y%m%d}.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path, engine="fastparquet")


def _expand_nodes(voxels: pd.DataFrame, shift_cells: int, *, surface_only: bool) -> pd.DataFrame:
    if voxels.empty:
        return pd.DataFrame(columns=["hua_object_id", "match_key"])
    use = voxels[voxels["depth_index"].eq(0)].copy() if surface_only else voxels.copy()
    rows = []
    for di in range(-shift_cells, shift_cells + 1):
        for dj in range(-shift_cells, shift_cells + 1):
            item = use[["hua_object_id", "depth_index", "i", "j"]].copy()
            item["i2"] = item["i"].astype("int64") + di
            item["j2"] = item["j"].astype("int64") + dj
            if surface_only:
                item["match_key"] = item["j2"].astype("int64") * 10_000_000 + item["i2"].astype("int64")
            else:
                item["match_key"] = (
                    item["depth_index"].astype("int64") * 100_000_000_000
                    + item["j2"].astype("int64") * 10_000_000
                    + item["i2"].astype("int64")
                )
            rows.append(item[["hua_object_id", "match_key"]].drop_duplicates())
    return pd.concat(rows, ignore_index=True).drop_duplicates()


def _overlap_edges(
    voxels1: pd.DataFrame,
    voxels2: pd.DataFrame,
    volumes1: pd.Series,
    volumes2: pd.Series,
    *,
    shift_cells: int,
    surface_only: bool,
) -> pd.DataFrame:
    if voxels1.empty or voxels2.empty:
        return pd.DataFrame(columns=["object_id_t0", "object_id_t1", "overlap", "score", "mode"])
    left = _expand_nodes(voxels1, shift_cells, surface_only=surface_only)
    right = _expand_nodes(voxels2, 0, surface_only=surface_only).rename(columns={"hua_object_id": "object_id_t1"})
    joined = left.merge(right, on="match_key", how="inner")
    if joined.empty:
        return pd.DataFrame(columns=["object_id_t0", "object_id_t1", "overlap", "score", "mode"])
    joined = joined.rename(columns={"hua_object_id": "object_id_t0"})
    edges = joined.groupby(["object_id_t0", "object_id_t1"], as_index=False).size().rename(columns={"size": "overlap"})
    edges["volume_t0"] = edges["object_id_t0"].map(volumes1).astype("float64")
    edges["volume_t1"] = edges["object_id_t1"].map(volumes2).astype("float64")
    edges = edges[(edges["volume_t0"] > 0) & (edges["volume_t1"] > 0)].copy()
    edges["score"] = edges["overlap"] / np.sqrt(edges["volume_t0"] * edges["volume_t1"])
    edges["mode"] = "surface_2d" if surface_only else "volume_3d"
    return edges.sort_values("score", ascending=False).reset_index(drop=True)


def _best_by(edges: pd.DataFrame, key: str) -> pd.DataFrame:
    if edges.empty:
        return edges
    idx = edges.groupby(key)["score"].idxmax()
    return edges.loc[idx].copy()


def _classify_pair(
    date0: str,
    date1: str,
    objs0: pd.DataFrame,
    objs1: pd.DataFrame,
    edges3d: pd.DataFrame,
    edges2d: pd.DataFrame,
    *,
    continuous_score_min: float,
    split_merge_score_min: float,
) -> tuple[list[dict[str, object]], pd.DataFrame]:
    events: list[dict[str, object]] = []
    tagged0: set[str] = set()
    tagged1: set[str] = set()
    edges = edges3d[edges3d["score"].gt(0)].copy()
    by0 = {k: g for k, g in edges.groupby("object_id_t0")}
    by1 = {k: g for k, g in edges.groupby("object_id_t1")}

    for object_id, group in by0.items():
        cand = group[group["score"].ge(split_merge_score_min)]
        if len(cand) > 1:
            targets = cand["object_id_t1"].astype(str).tolist()
            tagged0.add(str(object_id))
            tagged1.update(targets)
            events.append(
                {
                    "date0": date0,
                    "date1": date1,
                    "event_type": "split",
                    "object_id_t0": str(object_id),
                    "object_id_t1": "|".join(targets),
                    "n_targets": int(len(targets)),
                    "score_max": float(cand["score"].max()),
                    "score_sum": float(cand["score"].sum()),
                }
            )

    for object_id, group in by1.items():
        cand = group[group["score"].ge(split_merge_score_min)]
        if len(cand) > 1 and str(object_id) not in tagged1:
            sources = [x for x in cand["object_id_t0"].astype(str).tolist() if x not in tagged0]
            if len(sources) > 1:
                tagged0.update(sources)
                tagged1.add(str(object_id))
                events.append(
                    {
                        "date0": date0,
                        "date1": date1,
                        "event_type": "merge",
                        "object_id_t0": "|".join(sources),
                        "object_id_t1": str(object_id),
                        "n_targets": int(len(sources)),
                        "score_max": float(cand["score"].max()),
                        "score_sum": float(cand["score"].sum()),
                    }
                )

    row_best = _best_by(edges, "object_id_t0")
    col_best = _best_by(edges, "object_id_t1")
    mutual = row_best.merge(col_best[["object_id_t0", "object_id_t1"]], on=["object_id_t0", "object_id_t1"], how="inner")
    for _, edge in mutual.sort_values("score", ascending=False).iterrows():
        obj0 = str(edge["object_id_t0"])
        obj1 = str(edge["object_id_t1"])
        if obj0 in tagged0 or obj1 in tagged1 or float(edge["score"]) < continuous_score_min:
            continue
        tagged0.add(obj0)
        tagged1.add(obj1)
        surface_edge = edges2d[(edges2d["object_id_t0"].eq(obj0)) & (edges2d["object_id_t1"].eq(obj1))]
        events.append(
            {
                "date0": date0,
                "date1": date1,
                "event_type": "continuous",
                "object_id_t0": obj0,
                "object_id_t1": obj1,
                "n_targets": 1,
                "score_max": float(edge["score"]),
                "score_sum": float(edge["score"]),
                "surface_score": float(surface_edge["score"].iloc[0]) if not surface_edge.empty else 0.0,
            }
        )

    for obj0 in objs0["hua_object_id"].astype(str):
        if obj0 not in tagged0:
            events.append({"date0": date0, "date1": date1, "event_type": "dissipate", "object_id_t0": obj0, "object_id_t1": "-1", "n_targets": 0, "score_max": 0.0, "score_sum": 0.0})
    for obj1 in objs1["hua_object_id"].astype(str):
        if obj1 not in tagged1:
            events.append({"date0": date0, "date1": date1, "event_type": "new", "object_id_t0": "-1", "object_id_t1": obj1, "n_targets": 0, "score_max": 0.0, "score_sum": 0.0})

    return events, edges


class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, item: str) -> str:
        self.parent.setdefault(item, item)
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, a: str, b: str) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def _build_tracks(events: pd.DataFrame, objects: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    uf = _UnionFind()
    for oid in objects["hua_object_id"].astype(str):
        uf.find(oid)
    for _, event in events.iterrows():
        if event["event_type"] in {"continuous", "split", "merge"}:
            sources = [x for x in str(event["object_id_t0"]).split("|") if x and x != "-1"]
            targets = [x for x in str(event["object_id_t1"]).split("|") if x and x != "-1"]
            for source in sources:
                for target in targets:
                    uf.union(source, target)
    track_key = {oid: uf.find(str(oid)) for oid in objects["hua_object_id"].astype(str)}
    ordered_roots = {root: idx + 1 for idx, root in enumerate(sorted(set(track_key.values())))}
    points = objects.copy()
    points["feature_track_id"] = points["hua_object_id"].astype(str).map(lambda oid: ordered_roots[track_key[oid]])
    tracks = (
        points.groupby("feature_track_id")
        .agg(
            polarity=("polarity", "first"),
            n_objects=("hua_object_id", "nunique"),
            start_date=("date", "min"),
            end_date=("date", "max"),
            mean_lon=("surface_center_lon", "mean"),
            mean_lat=("surface_center_lat", "mean"),
            max_depth_m=("max_depth_m", "max"),
            total_voxels=("voxel_count_3d", "sum"),
            total_pass_layers=("pass_layers", "sum"),
        )
        .reset_index()
    )
    tracks["duration_days"] = (pd.to_datetime(tracks["end_date"]) - pd.to_datetime(tracks["start_date"])).dt.days + 1
    return tracks, points


def _assign_groups(objects: pd.DataFrame, *, group_radius_km: float) -> pd.DataFrame:
    rows = []
    next_group = 1
    for (day, polarity), group in objects.groupby(["date", "polarity"], sort=True):
        ids = group["hua_object_id"].astype(str).tolist()
        uf = _UnionFind()
        for oid in ids:
            uf.find(oid)
        lon = group["surface_center_lon"].to_numpy(dtype="float64")
        lat = group["surface_center_lat"].to_numpy(dtype="float64")
        for a in range(len(group)):
            for b in range(a + 1, len(group)):
                dx = ((lon[b] - lon[a] + 180.0) % 360.0 - 180.0) * 111.2 * math.cos(math.radians(float(0.5 * (lat[a] + lat[b]))))
                dy = (lat[b] - lat[a]) * 111.2
                if math.hypot(dx, dy) <= group_radius_km:
                    uf.union(ids[a], ids[b])
        roots = sorted({uf.find(oid) for oid in ids})
        root_to_group = {root: next_group + idx for idx, root in enumerate(roots)}
        next_group += len(roots)
        for oid in ids:
            rows.append({"date": day, "polarity": polarity, "hua_object_id": oid, "hua_group_id": root_to_group[uf.find(oid)]})
    return pd.DataFrame(rows)


def _plot_feature_tracks(points: pd.DataFrame, tracks: pd.DataFrame, out_dir: Path) -> None:
    fig_dir = out_dir / "paper_like_full_tracking_figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    if points.empty:
        return
    points = points.copy()
    points["date_ts"] = pd.to_datetime(points["date"])
    colors = {"cyclonic": "#2563eb", "anticyclonic": "#dc2626"}
    long_ids = tracks.sort_values(["n_objects", "total_pass_layers"], ascending=False).head(40)["feature_track_id"].tolist()
    use = points[points["feature_track_id"].isin(long_ids)]

    fig, ax = plt.subplots(figsize=(14, 5.4))
    for track_id, group in use.sort_values("date_ts").groupby("feature_track_id"):
        pol = str(group["polarity"].iloc[0])
        color = colors.get(pol, "#525252")
        ax.plot(group["surface_center_lon"], group["surface_center_lat"], color=color, alpha=0.70, linewidth=1.4)
        ax.scatter(group["surface_center_lon"], group["surface_center_lat"], color=color, s=12 + group["pass_layers"], alpha=0.86, edgecolor="white", linewidth=0.25)
        if len(group) >= 2:
            ax.text(float(group["surface_center_lon"].iloc[-1]), float(group["surface_center_lat"].iloc[-1]), str(int(track_id)), fontsize=8)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-66, -44)
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_title("ACC Hua feature tracks from object-voxel overlap")
    ax.grid(alpha=0.25)
    ax.legend(handles=[plt.Line2D([0], [0], color=colors[k], lw=2, label=k) for k in ["cyclonic", "anticyclonic"]], loc="lower left")
    fig.savefig(fig_dir / "hua_feature_tracks_map.png", dpi=220, bbox_inches="tight")
    fig.savefig(fig_dir / "hua_feature_tracks_map.pdf", bbox_inches="tight")
    plt.close(fig)

    top = tracks.sort_values(["n_objects", "total_pass_layers"], ascending=False).head(8)["feature_track_id"].tolist()
    fig, axes = plt.subplots(4, 2, figsize=(12, 12), sharey=True)
    for ax, track_id in zip(axes.ravel(), top):
        group = points[points["feature_track_id"].eq(track_id)].sort_values("date_ts")
        t = (group["date_ts"] - group["date_ts"].min()).dt.days.to_numpy(dtype="float64")
        depth = group["max_depth_m"].to_numpy(dtype="float64") / 1000.0
        color = colors.get(str(group["polarity"].iloc[0]), "#525252")
        ax.plot(t, depth, color=color, linewidth=2.0)
        ax.scatter(t, depth, s=18 + group["pass_layers"], color=color, edgecolor="white", linewidth=0.3)
        ax.invert_yaxis()
        ax.grid(alpha=0.25)
        ax.set_title(f"feature track {track_id}, n={len(group)}")
        ax.set_xlabel("days")
        ax.set_ylabel("max depth km")
    for ax in axes.ravel()[len(top) :]:
        ax.axis("off")
    fig.suptitle("Hua feature-track depth evolution")
    fig.tight_layout()
    fig.savefig(fig_dir / "hua_feature_tracks_depth_time.png", dpi=220, bbox_inches="tight")
    fig.savefig(fig_dir / "hua_feature_tracks_depth_time.pdf", bbox_inches="tight")
    plt.close(fig)


def _write_docs(out_dir: Path, summary: dict[str, object], event_counts: pd.DataFrame) -> None:
    lines = [
        "# Hua/Rutgers feature-group tracking 等价复刻",
        "",
        "本结果使用 Hua 检测输出中的真实对象体素做相邻帧 overlap，而不是 prototype nearest-neighbor 连点。",
        "",
        "## 总数",
        "",
        f"- frame objects: `{summary['n_frame_objects']}`",
        f"- feature tracks: `{summary['n_feature_tracks']}`",
        f"- group objects: `{summary['n_group_assignments']}`",
        f"- longest track objects: `{summary['max_track_objects']}`",
        f"- tracks with >=2 objects: `{summary['tracks_len_ge_2']}`",
        f"- tracks with >=3 objects: `{summary['tracks_len_ge_3']}`",
        "",
        "## 事件计数",
        "",
    ]
    for row in event_counts.to_dict("records"):
        lines.append(f"- `{row['event_type']}`: `{row['count']}`")
    lines.extend(
        [
            "",
            "## 口径说明",
            "",
            "3D overlap score = overlap_voxels / sqrt(volume_t0 * volume_t1)。",
            "continuous 默认要求 mutual best 且 score >= 0.25；split/merge 默认要求同一行或同一列有多个 score >= 0.75 的候选。",
            "为了对应源码中坐标距离阈值，这里默认允许 1 个水平格点的 overlap shift；这仍然是体素 overlap，不是中心距离 tracking。",
        ]
    )
    (out_dir / "feature_group_tracking_summary_zh.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    root = Path(args.input_dir)
    out_dir = Path(args.output_dir) if args.output_dir else root / "feature_group_tracking"
    out_dir.mkdir(parents=True, exist_ok=True)
    objects = _read_table(root / "frame_object_summary")
    has_voxel_file = (root / "object_voxels.parquet").exists() or (root / "object_voxels.csv").exists()
    has_voxel_parts = (root / "object_voxels_parts").exists()
    if objects.empty or (not has_voxel_file and not has_voxel_parts):
        raise SystemExit("frame_object_summary/object_voxels are required; rerun detection with --write-object-voxels")
    objects["date"] = objects["date"].astype(str)

    events: list[dict[str, object]] = []
    overlap_parts = []
    n_voxels_total = 0
    dates = sorted(objects["date"].unique().tolist())
    for date0, date1 in zip(dates[:-1], dates[1:]):
        objs0 = objects[objects["date"].eq(date0)].copy()
        objs1 = objects[objects["date"].eq(date1)].copy()
        vox0 = _read_voxels_for_date(root, date0)
        vox1 = _read_voxels_for_date(root, date1)
        n_voxels_total += int(len(vox0))
        vol3_0 = vox0.groupby("hua_object_id")["node_key_3d"].nunique()
        vol3_1 = vox1.groupby("hua_object_id")["node_key_3d"].nunique()
        surf0 = vox0[vox0["depth_index"].eq(0)].groupby("hua_object_id")["node_key_2d"].nunique()
        surf1 = vox1[vox1["depth_index"].eq(0)].groupby("hua_object_id")["node_key_2d"].nunique()
        edges3 = _overlap_edges(vox0, vox1, vol3_0, vol3_1, shift_cells=args.overlap_shift_cells, surface_only=False)
        edges2 = _overlap_edges(vox0, vox1, surf0, surf1, shift_cells=args.overlap_shift_cells, surface_only=True)
        pair_events, pair_edges = _classify_pair(
            date0,
            date1,
            objs0,
            objs1,
            edges3,
            edges2,
            continuous_score_min=args.continuous_score_min,
            split_merge_score_min=args.split_merge_score_min,
        )
        events.extend(pair_events)
        if not pair_edges.empty:
            pair_edges["date0"] = date0
            pair_edges["date1"] = date1
            overlap_parts.append(pair_edges)
        print(f"[feature-track] {date0}->{date1} events={len(pair_events)} overlaps={len(pair_edges)}", flush=True)

    event_df = pd.DataFrame(events)
    overlap_df = pd.concat(overlap_parts, ignore_index=True) if overlap_parts else pd.DataFrame()
    groups = _assign_groups(objects, group_radius_km=args.group_radius_km)
    tracks, points = _build_tracks(event_df, objects)
    points = points.merge(groups[["hua_object_id", "hua_group_id"]], on="hua_object_id", how="left")

    event_df.to_parquet(out_dir / "feature_track_events.parquet", index=False, engine="fastparquet")
    event_df.to_csv(out_dir / "feature_track_events.csv", index=False)
    overlap_df.to_parquet(out_dir / "feature_overlap_table.parquet", index=False, engine="fastparquet")
    overlap_df.to_csv(out_dir / "feature_overlap_table.csv", index=False)
    groups.to_parquet(out_dir / "group_tracks.parquet", index=False, engine="fastparquet")
    groups.to_csv(out_dir / "group_tracks.csv", index=False)
    tracks.to_parquet(out_dir / "feature_tracks.parquet", index=False, engine="fastparquet")
    tracks.to_csv(out_dir / "feature_tracks.csv", index=False)
    points.to_parquet(out_dir / "feature_track_points.parquet", index=False, engine="fastparquet")
    points.to_csv(out_dir / "feature_track_points.csv", index=False)
    trak_lines = []
    for _, event in event_df.iterrows():
        trak_lines.append(f"Frame #{event['date1']}\t{event['event_type']}\t{event['object_id_t0']}\t-1\t{event['object_id_t1']}")
    (out_dir / "feature_tracking.trakTable").write_text("\n".join(trak_lines) + "\n", encoding="utf-8")
    _plot_feature_tracks(points, tracks, out_dir)

    event_counts = event_df["event_type"].value_counts().rename_axis("event_type").reset_index(name="count") if not event_df.empty else pd.DataFrame(columns=["event_type", "count"])
    summary = {
        "n_frame_objects": int(len(objects)),
        "n_voxels": int(n_voxels_total),
        "n_overlap_edges": int(len(overlap_df)),
        "n_feature_events": int(len(event_df)),
        "n_feature_tracks": int(len(tracks)),
        "n_group_assignments": int(len(groups)),
        "max_track_objects": int(tracks["n_objects"].max()) if not tracks.empty else 0,
        "tracks_len_ge_2": int((tracks["n_objects"] >= 2).sum()) if not tracks.empty else 0,
        "tracks_len_ge_3": int((tracks["n_objects"] >= 3).sum()) if not tracks.empty else 0,
        "parameters": vars(args),
    }
    (out_dir / "feature_group_tracking_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    event_counts.to_csv(out_dir / "feature_track_event_counts.csv", index=False)
    _write_docs(out_dir, summary, event_counts)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Hua/Rutgers-style object-overlap feature and group tracking for ACC Hua detections.")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--overlap-shift-cells", type=int, default=1)
    parser.add_argument("--continuous-score-min", type=float, default=0.25)
    parser.add_argument("--split-merge-score-min", type=float, default=0.75)
    parser.add_argument("--group-radius-km", type=float, default=75.0)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
