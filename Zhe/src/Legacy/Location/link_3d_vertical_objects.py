from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from .common import ensure_dirs, iter_days, load_config, parse_ymd
from .run_3d_layer_identification import layer_output_paths
from .table_io import read_table, table_exists, write_table
from .velocity3d_core import haversine_km


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def output_paths(config: dict) -> tuple[Path, Path]:
    root = Path(config["paths"]["catalog_dir"])
    return root / "layer_observations.parquet", root / "vertical_objects.parquet"


def _link_day(obs: pd.DataFrame, config: dict, next_object_id: int) -> tuple[pd.DataFrame, list[dict], int]:
    if obs.empty:
        return obs, [], next_object_id
    obs = obs.copy().reset_index(drop=True)
    uf = UnionFind(len(obs))
    max_dist = float(config["vertical_linking"].get("max_center_distance_km", 70.0))
    for polarity, group in obs.groupby("polarity"):
        ordered = group.sort_values("depth_index")
        by_depth = {int(k): v for k, v in ordered.groupby("depth_index").groups.items()}
        for depth_idx, indices in by_depth.items():
            for offset in (1, 2):
                other = by_depth.get(depth_idx + offset)
                if other is None:
                    continue
                for i in indices:
                    row_i = obs.loc[i]
                    best_j = None
                    best_d = np.inf
                    for j in other:
                        row_j = obs.loc[j]
                        d = haversine_km(row_i.longitude, row_i.latitude, row_j.longitude, row_j.latitude)
                        if d < best_d:
                            best_d = d
                            best_j = j
                    if best_j is not None and best_d <= max_dist:
                        uf.union(int(i), int(best_j))

    groups: dict[int, list[int]] = {}
    for i in range(len(obs)):
        groups.setdefault(uf.find(i), []).append(i)

    objects = []
    obs["eddy3d_object_id"] = -1
    min_layers = int(config["vertical_linking"].get("min_layers", 1))
    for indices in groups.values():
        if len(set(int(obs.loc[i, "depth_index"]) for i in indices)) < min_layers:
            continue
        object_id = next_object_id
        next_object_id += 1
        obs.loc[indices, "eddy3d_object_id"] = object_id
        sub = obs.loc[indices]
        shallow = sub.sort_values("depth_m").iloc[0]
        objects.append(
            {
                "date": shallow.date,
                "eddy3d_object_id": object_id,
                "polarity": shallow.polarity,
                "longitude": float(shallow.longitude),
                "latitude": float(shallow.latitude),
                "layer_count": int(sub.depth_index.nunique()),
                "min_depth_m": float(sub.depth_m.min()),
                "max_depth_m": float(sub.depth_m.max()),
                "mean_radius_m": float(sub.radius_m.mean()),
                "track3d_id": -1,
            }
        )
    return obs[obs["eddy3d_object_id"] >= 0], objects, next_object_id


def _link_one_day(config_path: str, day_iso: str) -> tuple[str, pd.DataFrame, list[dict]]:
    config = load_config(config_path)
    day = parse_ymd(day_iso)
    obs_path, _ = layer_output_paths(config, day)
    if not table_exists(obs_path):
        return day_iso, pd.DataFrame(), []
    obs = read_table(obs_path)
    linked, objects, _ = _link_day(obs, config, 0)
    return day_iso, linked, objects


def _apply_object_offset(linked: pd.DataFrame, objects: list[dict], offset: int) -> tuple[pd.DataFrame, list[dict], int]:
    if not objects:
        return linked, objects, offset
    local_ids = sorted({int(row["eddy3d_object_id"]) for row in objects})
    id_map = {old: offset + i for i, old in enumerate(local_ids)}
    out_linked = linked.copy()
    out_linked["eddy3d_object_id"] = out_linked["eddy3d_object_id"].astype(int).map(id_map).astype(int)
    out_objects = []
    for row in objects:
        new_row = dict(row)
        new_row["eddy3d_object_id"] = id_map[int(row["eddy3d_object_id"])]
        out_objects.append(new_row)
    return out_linked, out_objects, offset + len(local_ids)


def link_range(
    config_path: str | Path,
    start: str | None = None,
    end: str | None = None,
    force: bool = False,
    workers: int | None = None,
) -> None:
    config = load_config(config_path)
    ensure_dirs(config)
    layer_out, obj_out = output_paths(config)
    if not force and table_exists(layer_out) and table_exists(obj_out):
        return
    start_day = parse_ymd(start or config["date_range"]["start"])
    end_day = parse_ymd(end or config["date_range"]["end"])
    days = list(iter_days(start_day, end_day))
    workers = int(workers or config.get("identification", {}).get("workers", 1) or 1)

    day_results: list[tuple[str, pd.DataFrame, list[dict]]] = []
    if workers <= 1 or len(days) <= 1:
        for day in tqdm(days, desc="Link vertical objects", unit="day"):
            day_results.append(_link_one_day(str(config_path), day.isoformat()))
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_link_one_day, str(config_path), day.isoformat()): day for day in days}
            with tqdm(total=len(futures), desc=f"Link vertical objects ({workers} workers)", unit="day") as bar:
                for future in as_completed(futures):
                    day = futures[future]
                    try:
                        day_results.append(future.result())
                    except Exception as exc:
                        raise RuntimeError(f"Vertical linking failed for {day}") from exc
                    bar.update(1)

    all_layers = []
    all_objects = []
    next_object_id = 0
    for _, linked, objects in sorted(day_results, key=lambda item: item[0]):
        if linked.empty and not objects:
            continue
        linked, objects, next_object_id = _apply_object_offset(linked, objects, next_object_id)
        all_layers.append(linked)
        all_objects.extend(objects)
    layer_df = pd.concat(all_layers, ignore_index=True) if all_layers else pd.DataFrame()
    obj_df = pd.DataFrame.from_records(all_objects)
    write_table(layer_df, layer_out, index=False)
    write_table(obj_df, obj_out, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Link layer detections into same-day 3D eddy objects.")
    parser.add_argument("--config", default="config/config_3d.yaml")
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--workers", type=int)
    args = parser.parse_args()
    link_range(args.config, args.start, args.end, force=args.force, workers=args.workers)


if __name__ == "__main__":
    main()
