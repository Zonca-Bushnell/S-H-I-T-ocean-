"""Optional contour/layer based tracking for canonical OFES object-days."""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.path import Path as PolygonPath
from scipy.spatial import cKDTree


def _numbers(value: object) -> np.ndarray:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.empty(0, dtype=float)
    text = str(value).strip()
    if not text:
        return np.empty(0, dtype=float)
    return np.asarray([float(part) for part in text.replace(",", ";").split(";") if part.strip()], dtype=float)


def _disk(ci: float, cj: float, radius: float) -> set[tuple[int, int]]:
    r = max(float(radius), 1.0)
    i0, i1 = math.floor(ci - r), math.ceil(ci + r)
    j0, j1 = math.floor(cj - r), math.ceil(cj + r)
    return {
        (i, j) for i in range(i0, i1 + 1) for j in range(j0, j1 + 1)
        if (i - ci) ** 2 + (j - cj) ** 2 <= r ** 2
    }


def _polygon(ii: np.ndarray, jj: np.ndarray) -> set[tuple[int, int]]:
    if len(ii) < 3 or len(ii) != len(jj):
        return set()
    i0, i1 = math.floor(float(np.nanmin(ii))), math.ceil(float(np.nanmax(ii)))
    j0, j1 = math.floor(float(np.nanmin(jj))), math.ceil(float(np.nanmax(jj)))
    gi, gj = np.meshgrid(np.arange(i0, i1 + 1), np.arange(j0, j1 + 1), indexing="ij")
    points = np.column_stack([gi.ravel(), gj.ravel()])
    inside = PolygonPath(np.column_stack([ii, jj]), closed=True).contains_points(points, radius=1e-9)
    return {tuple(map(int, point)) for point in points[inside]}


def layer_cells(row: pd.Series) -> tuple[set[tuple[int, int]], str]:
    depth = int(row.get("depth_index", 0))
    if depth == 0:
        ii, jj = _numbers(row.get("ssh_contour_boundary_i")), _numbers(row.get("ssh_contour_boundary_j"))
        cells = _polygon(ii, jj)
        if cells:
            return cells, "ssh_contour"
    ii, jj = _numbers(row.get("streamline_boundary_i")), _numbers(row.get("streamline_boundary_j"))
    cells = _polygon(ii, jj)
    if cells:
        return cells, "near_closed_streamline"
    ci = float(row.get("center_i_refined", row.get("hua_center_i", row.get("speed_min_i", np.nan))))
    cj = float(row.get("center_j_refined", row.get("hua_center_j", row.get("speed_min_j", np.nan))))
    radius = float(row.get("accepted_radius_cells", row.get("radius_cells", np.nan)))
    if np.isfinite(ci) and np.isfinite(cj) and np.isfinite(radius) and radius > 0:
        return _disk(ci, cj, radius), "tangent_radius_circle"
    return set(), "missing_boundary"


def object_volume(rows: pd.DataFrame) -> tuple[set[tuple[int, int, int]], dict[int, str]]:
    volume: set[tuple[int, int, int]] = set()
    sources: dict[int, str] = {}
    for _, row in rows.sort_values("depth_index").iterrows():
        if not bool(row.get("hua_pass", True)):
            break
        depth = int(row["depth_index"])
        cells, source = layer_cells(row)
        sources[depth] = source
        volume.update((depth, i, j) for i, j in cells)
    return volume, sources


def overlap_score(left: set[tuple[int, int, int]], right: set[tuple[int, int, int]], shift_cells: int = 1) -> float:
    if not left or not right:
        return 0.0
    best = 0
    for di in range(-shift_cells, shift_cells + 1):
        for dj in range(-shift_cells, shift_cells + 1):
            shifted = {(k, i + di, j + dj) for k, i, j in left}
            best = max(best, len(shifted.intersection(right)))
    return float(best / math.sqrt(len(left) * len(right)))


@dataclass
class _UnionFind:
    parent: dict[str, str]

    def find(self, item: str) -> str:
        self.parent.setdefault(item, item)
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, left: str, right: str) -> None:
        a, b = self.find(left), self.find(right)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


def track_object_days(
    centers: pd.DataFrame,
    *,
    shift_cells: int = 1,
    continuous_score: float = 0.25,
    split_merge_score: float = 0.75,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    required = {"date", "hua_object_id", "depth_index", "polarity"}
    missing = required.difference(centers.columns)
    if missing:
        raise ValueError(f"Tracking input misses columns: {sorted(missing)}")
    accepted = centers.loc[centers.get("hua_pass", True).fillna(False).astype(bool)].copy()
    accepted["date"] = pd.to_datetime(accepted["date"]).dt.normalize()
    volumes: dict[str, set[tuple[int, int, int]]] = {}
    rows_by_id: dict[str, pd.DataFrame] = {}
    objects: list[dict[str, object]] = []
    for object_id, rows in accepted.groupby("hua_object_id", sort=True):
        first = rows.sort_values("depth_index").iloc[0]
        key = str(object_id)
        rows_by_id[key] = rows
        sources = {
            int(row.depth_index): (
                "ssh_contour" if int(row.depth_index) == 0
                else "near_closed_streamline" if str(getattr(row, "deep_boundary_branch", "")).startswith("near_closed")
                else "tangent_radius_circle"
            )
            for row in rows.sort_values("depth_index").itertuples()
        }
        ci = float(first.get("center_i_refined", first.get("hua_center_i", first.get("speed_min_i", np.nan))))
        cj = float(first.get("center_j_refined", first.get("hua_center_j", first.get("speed_min_j", np.nan))))
        radius = float(first.get("ssh_contour_radius_cells", first.get("accepted_radius_cells", first.get("radius_cells", 1.0))))
        objects.append({
            "date": first["date"], "hua_object_id": key,
            "polarity": str(first["polarity"]), "volume_cells": np.nan,
            "accepted_layers": int(rows["depth_index"].nunique()),
            "boundary_sources": json.dumps(sources, sort_keys=True),
            "surface_center_i": ci, "surface_center_j": cj,
            "surface_radius_cells": radius if np.isfinite(radius) and radius > 0 else 1.0,
        })

    def volume_for(object_id: str) -> set[tuple[int, int, int]]:
        if object_id not in volumes:
            volumes[object_id], _ = object_volume(rows_by_id[object_id])
        return volumes[object_id]

    object_df = pd.DataFrame(objects).sort_values(["date", "hua_object_id"]).reset_index(drop=True)
    edge_rows: list[dict[str, object]] = []
    days = sorted(object_df["date"].unique())
    for day0, day1 in zip(days, days[1:]):
        if (pd.Timestamp(day1) - pd.Timestamp(day0)).days != 1:
            continue
        left = object_df[object_df["date"] == day0]
        right = object_df[object_df["date"] == day1]
        for polarity in sorted(set(left["polarity"]).intersection(right["polarity"])):
            aa, bb = left[left["polarity"] == polarity], right[right["polarity"] == polarity]
            bb = bb.reset_index(drop=True)
            tree = cKDTree(bb[["surface_center_i", "surface_center_j"]].to_numpy(dtype=float))
            max_target_radius = float(bb["surface_radius_cells"].max())
            for source_row in aa.itertuples():
                search_radius = float(source_row.surface_radius_cells) + max_target_radius + 2 * shift_cells
                nearby = tree.query_ball_point([source_row.surface_center_i, source_row.surface_center_j], search_radius)
                for target_index in nearby:
                    target_row = bb.iloc[int(target_index)]
                    distance = math.hypot(
                        float(source_row.surface_center_i) - float(target_row.surface_center_i),
                        float(source_row.surface_center_j) - float(target_row.surface_center_j),
                    )
                    if distance > float(source_row.surface_radius_cells) + float(target_row.surface_radius_cells) + 2 * shift_cells:
                        continue
                    source, target = str(source_row.hua_object_id), str(target_row.hua_object_id)
                    score = overlap_score(volume_for(source), volume_for(target), shift_cells)
                    if score >= continuous_score:
                        edge_rows.append({"date_t0": day0, "date_t1": day1, "source_id": source,
                                          "target_id": target, "polarity": polarity, "score": score})
    if volumes:
        object_df["volume_cells"] = object_df["hua_object_id"].map({key: len(value) for key, value in volumes.items()})
    edges = pd.DataFrame(edge_rows)
    if edges.empty:
        edges = pd.DataFrame(columns=["date_t0", "date_t1", "source_id", "target_id", "polarity", "score"])
    edges["source_rank"] = edges.groupby(["date_t0", "source_id"])["score"].rank(method="first", ascending=False)
    edges["target_rank"] = edges.groupby(["date_t1", "target_id"])["score"].rank(method="first", ascending=False)
    edges["mutual_best"] = (edges["source_rank"] == 1) & (edges["target_rank"] == 1)
    uf = _UnionFind({})
    for object_id in object_df["hua_object_id"]:
        uf.find(object_id)
    for row in edges.loc[edges["mutual_best"]].itertuples():
        uf.union(row.source_id, row.target_id)
    roots = sorted({uf.find(item) for item in object_df["hua_object_id"]})
    track_ids = {root: f"track_{index:06d}" for index, root in enumerate(roots, 1)}
    object_df["track_id"] = [track_ids[uf.find(item)] for item in object_df["hua_object_id"]]
    strong = edges[edges["score"] >= split_merge_score]
    source_counts = strong.groupby("source_id")["target_id"].nunique()
    target_counts = strong.groupby("target_id")["source_id"].nunique()
    events: list[dict[str, object]] = []
    incoming = set(edges["target_id"])
    outgoing = set(edges["source_id"])
    for row in object_df.itertuples():
        if row.hua_object_id not in incoming:
            events.append({"date": row.date, "hua_object_id": row.hua_object_id, "event": "new", "track_id": row.track_id})
        if row.hua_object_id not in outgoing:
            events.append({"date": row.date, "hua_object_id": row.hua_object_id, "event": "dissipate", "track_id": row.track_id})
        if source_counts.get(row.hua_object_id, 0) > 1:
            events.append({"date": row.date, "hua_object_id": row.hua_object_id, "event": "split", "track_id": row.track_id})
        if target_counts.get(row.hua_object_id, 0) > 1:
            events.append({"date": row.date, "hua_object_id": row.hua_object_id, "event": "merge", "track_id": row.track_id})
    for row in edges.loc[edges["mutual_best"]].itertuples():
        track_id = object_df.loc[object_df["hua_object_id"] == row.target_id, "track_id"].iloc[0]
        events.append({"date": row.date_t1, "hua_object_id": row.target_id, "event": "continuous", "track_id": track_id})
    return object_df, edges, pd.DataFrame(events)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--centers", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--shift-cells", type=int, default=1)
    parser.add_argument("--continuous-score", type=float, default=0.25)
    parser.add_argument("--split-merge-score", type=float, default=0.75)
    args = parser.parse_args()
    centers = pd.read_csv(args.centers, low_memory=False)
    objects, edges, events = track_object_days(
        centers, shift_cells=args.shift_cells, continuous_score=args.continuous_score,
        split_merge_score=args.split_merge_score,
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    objects.to_csv(args.output_root / "tracked_object_days.csv", index=False)
    edges.to_csv(args.output_root / "tracking_edges.csv", index=False)
    events.to_csv(args.output_root / "tracking_events.csv", index=False)


if __name__ == "__main__":
    main()
