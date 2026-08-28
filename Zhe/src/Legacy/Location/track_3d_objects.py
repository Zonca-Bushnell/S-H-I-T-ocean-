from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from tqdm import tqdm

from .common import ensure_dirs, load_config, parse_ymd
from .link_3d_vertical_objects import output_paths as link_output_paths
from .table_io import read_table, table_exists, write_table_fast
from .velocity3d_core import haversine_km

EARTH_RADIUS_KM = 6371.0


def tracks_output_path(config: dict) -> Path:
    return Path(config["paths"]["catalog_dir"]) / "tracks_3d.parquet"


def _depth_overlap(a: pd.Series, b: pd.Series) -> int:
    lo = max(float(a.min_depth_m), float(b.min_depth_m))
    hi = min(float(a.max_depth_m), float(b.max_depth_m))
    return int(hi >= lo)


def _xy_km(lon: np.ndarray, lat: np.ndarray, ref_lat: float) -> tuple[np.ndarray, np.ndarray]:
    x = EARTH_RADIUS_KM * np.cos(np.radians(ref_lat)) * np.radians(lon)
    y = EARTH_RADIUS_KM * np.radians(lat)
    return x, y


def _active_trees(active: dict[int, pd.Series], ref_lat: float) -> dict[str, tuple[cKDTree, list[int]]]:
    grouped: dict[str, list[tuple[int, pd.Series]]] = {}
    for tid, row in active.items():
        grouped.setdefault(str(row.polarity), []).append((tid, row))
    trees: dict[str, tuple[cKDTree, list[int]]] = {}
    for polarity, items in grouped.items():
        tids = [tid for tid, _ in items]
        lon = np.array([float(row.longitude) for _, row in items], dtype="f8")
        lat = np.array([float(row.latitude) for _, row in items], dtype="f8")
        x, y = _xy_km(lon, lat, ref_lat)
        trees[polarity] = (cKDTree(np.column_stack([x, y])), tids)
    return trees


def track_range(config_path: str | Path, start: str | None = None, end: str | None = None, force: bool = False) -> None:
    config = load_config(config_path)
    ensure_dirs(config)
    _, obj_path = link_output_paths(config)
    out_path = tracks_output_path(config)
    if not force and table_exists(out_path):
        return
    if not table_exists(obj_path):
        raise FileNotFoundError(f"Missing vertical objects: {obj_path}")
    objects = read_table(obj_path)
    if objects.empty:
        write_table_fast(pd.DataFrame(), out_path, index=False)
        return
    objects["date"] = pd.to_datetime(objects["date"]).dt.date
    start_day = parse_ymd(start or str(objects["date"].min()))
    end_day = parse_ymd(end or str(objects["date"].max()))
    objects = objects[(objects["date"] >= start_day) & (objects["date"] <= end_day)].copy()

    max_dist = float(config["tracking"].get("max_daily_displacement_km", 120.0))
    min_overlap = int(config["tracking"].get("min_depth_overlap_layers", 1))
    max_gap = int(config["tracking"].get("max_gap_days", 0))
    active: dict[int, pd.Series] = {}
    next_track_id = 0
    track_rows = []
    assignments = {}
    new_count = 0
    day_groups = list(objects.groupby("date"))
    bar = tqdm(day_groups, desc="Track 3D objects", unit="day")
    for day, group in bar:
        stale = [tid for tid, row in active.items() if (day - row.date).days > max_gap + 1]
        for tid in stale:
            active.pop(tid, None)
        ref_lat = float(group["latitude"].mean()) if len(group) else 0.0
        trees = _active_trees(active, ref_lat) if active else {}
        same_day_tids: list[int] = []
        for row in group.itertuples(index=False):
            best_tid = None
            best_dist = np.inf
            candidate_tids: list[int]
            tree_entry = trees.get(str(row.polarity))
            if tree_entry is None:
                candidate_tids = []
            else:
                tree, tids = tree_entry
                qx, qy = _xy_km(np.array([float(row.longitude)]), np.array([float(row.latitude)]), ref_lat)
                idxs = tree.query_ball_point([qx[0], qy[0]], r=max_dist * 1.5)
                candidate_tids = [tids[int(i)] for i in idxs]
            candidate_tids.extend(
                tid for tid in same_day_tids if tid in active and str(active[tid].polarity) == str(row.polarity)
            )
            if candidate_tids:
                candidate_tids = list(dict.fromkeys(candidate_tids))
            for tid in candidate_tids:
                prev = active[tid]
                if _depth_overlap(prev, row) < min_overlap:
                    continue
                d = haversine_km(prev.longitude, prev.latitude, row.longitude, row.latitude)
                if d < best_dist:
                    best_dist = d
                    best_tid = tid
            if best_tid is None or best_dist > max_dist:
                best_tid = next_track_id
                next_track_id += 1
                new_count += 1
            assignments[int(row.eddy3d_object_id)] = best_tid
            active[best_tid] = row
            same_day_tids.append(best_tid)
        bar.set_postfix(date=str(day), active=len(active), new_tracks=new_count)

    objects["track3d_id"] = objects["eddy3d_object_id"].map(assignments).fillna(-1).astype(int)
    for tid, group in objects[objects.track3d_id >= 0].groupby("track3d_id"):
        g = group.sort_values("date")
        track_rows.append(
            {
                "track3d_id": int(tid),
                "polarity": g.iloc[0].polarity,
                "start_date": g.date.min().isoformat(),
                "end_date": g.date.max().isoformat(),
                "lifetime_days": int(g.date.nunique()),
                "observation_count": int(len(g)),
                "max_layer_count": int(g.layer_count.max()),
                "mean_radius_m": float(g.mean_radius_m.mean()),
            }
        )
    _, obj_path = link_output_paths(config)
    write_table_fast(objects, obj_path, index=False)
    tracks = pd.DataFrame.from_records(track_rows)
    write_table_fast(tracks, out_path, index=False)
    tracks.to_csv(out_path.with_suffix(".csv"), index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Track same-day 3D eddy objects across dates.")
    parser.add_argument("--config", default="config/config_3d.yaml")
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    track_range(args.config, args.start, args.end, force=args.force)


if __name__ == "__main__":
    main()
