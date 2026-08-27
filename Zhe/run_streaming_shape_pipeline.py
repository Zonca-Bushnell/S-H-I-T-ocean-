from __future__ import annotations

import argparse
import shutil
import tempfile
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from .classify_3d_eddy_shape import classify_shapes
from .common import ensure_dirs, iter_days, load_config, parse_ymd
from .link_3d_vertical_objects import _link_day
from .run_3d_layer_identification import layer_output_paths
from .run_streaming_layer_identification import identify_day_streaming
from .streaming_cmems import build_source_index, selected_depth_indices
from .table_io import table_exists, write_parquet_parts_to_single, write_table, write_table_fast
from .track_3d_objects import track_range


def _catalog_paths(config: dict) -> tuple[Path, Path]:
    root = Path(config["paths"]["catalog_dir"])
    return root / "layer_observations.parquet", root / "vertical_objects.parquet"


def _write_layer_checkpoint(config: dict, day, obs: pd.DataFrame, contours: pd.DataFrame) -> None:
    obs_path, contour_path = layer_output_paths(config, day)
    write_table(obs, obs_path, index=False)
    write_table(contours, contour_path, index=False)


def _write_catalog_part(df: pd.DataFrame, root: Path, day, name: str) -> Path | None:
    if df.empty:
        return None
    part_dir = root / f"{name}_parts" / f"year={day:%Y}"
    part_path = part_dir / f"{name}_{day:%Y%m%d}.parquet"
    write_table_fast(df, part_path, index=False)
    return part_path


def _build_streaming_catalog(
    config_path: str | Path,
    *,
    start: str | None,
    end: str | None,
    workers: int,
    force: bool,
    write_layer_checkpoints: bool,
) -> None:
    config = load_config(config_path)
    ensure_dirs(config)
    layer_out, obj_out = _catalog_paths(config)
    if not force and table_exists(layer_out) and table_exists(obj_out):
        print("Streaming catalog: existing catalog outputs found; skipping rebuild.")
        return

    start_day = parse_ymd(start or config["date_range"]["start"])
    end_day = parse_ymd(end or config["date_range"]["end"])
    days = list(iter_days(start_day, end_day))
    source_index = build_source_index(config, start_day, end_day)
    depth_indices = selected_depth_indices(config, source_index[days[0]].path)

    catalog_dir = Path(config["paths"]["catalog_dir"])
    layer_parts_root = catalog_dir / "layer_observations_parts"
    object_parts_root = catalog_dir / "vertical_objects_parts"
    if force:
        shutil.rmtree(layer_parts_root, ignore_errors=True)
        shutil.rmtree(object_parts_root, ignore_errors=True)
    layer_parts: list[Path] = []
    object_parts: list[Path] = []
    next_object_id = 0
    layer_rows = 0
    object_rows = 0
    with tqdm(total=len(days), desc=f"Streaming detect + vertical link ({workers} workers/day)", unit="day") as bar:
        for day in days:
            obs, contours = identify_day_streaming(
                config,
                source_index[day],
                depth_indices,
                workers=workers,
                include_contours=write_layer_checkpoints,
            )
            linked, objects, next_object_id = _link_day(obs, config, next_object_id)
            objects_df = pd.DataFrame.from_records(objects)
            linked_part = _write_catalog_part(linked, catalog_dir, day, "layer_observations")
            object_part = _write_catalog_part(objects_df, catalog_dir, day, "vertical_objects")
            if linked_part is not None:
                layer_parts.append(linked_part)
            if object_part is not None:
                object_parts.append(object_part)
            layer_rows += len(linked)
            object_rows += len(objects_df)
            if write_layer_checkpoints:
                _write_layer_checkpoint(config, day, obs, contours)
            bar.update(1)
            bar.set_postfix(date=day.isoformat(), detections=len(obs), objects=len(objects_df))

    write_parquet_parts_to_single(layer_parts, layer_out)
    write_parquet_parts_to_single(object_parts, obj_out)
    print(f"Catalog layer observations: {layer_out} rows={layer_rows}")
    print(f"Catalog vertical objects: {obj_out} rows={object_rows}")


def run_pipeline(
    config_path: str | Path,
    *,
    start: str | None,
    end: str | None,
    workers: int,
    force: bool,
    output_name: str,
    write_layer_checkpoints: bool,
    lifetime_min_days: int,
    radius_min_m: float,
    min_valid_layers: int,
    merge_completed_centers: bool = False,
) -> None:
    started = time.perf_counter()
    config = load_config(config_path)
    ensure_dirs(config)

    stage_start = time.perf_counter()
    _build_streaming_catalog(
        config_path,
        start=start,
        end=end,
        workers=workers,
        force=force,
        write_layer_checkpoints=write_layer_checkpoints,
    )
    print(f"[stage] streaming catalog done in {time.perf_counter() - stage_start:.1f}s", flush=True)

    stage_start = time.perf_counter()
    track_range(config_path, start=start, end=end, force=force)
    print(f"[stage] tracking done in {time.perf_counter() - stage_start:.1f}s", flush=True)

    stage_start = time.perf_counter()
    classify_shapes(
        config_path,
        complete_missing=True,
        force=force,
        lifetime_min_days=lifetime_min_days,
        radius_min_m=radius_min_m,
        min_valid_layers=min_valid_layers,
        completion_output_mode="centers-only",
        workers=workers,
        start=start,
        end=end,
        output_name=output_name,
        write_completed_centers=True,
        merge_completed_centers=merge_completed_centers,
    )
    print(f"[stage] shape classification done in {time.perf_counter() - stage_start:.1f}s", flush=True)
    print(f"[done] streaming shape pipeline finished in {time.perf_counter() - started:.1f}s", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="CPU-only streaming Kuroshio pipeline from CMEMS NetCDF to shape classification.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--output-name", default="shape_classification")
    parser.add_argument("--write-layer-checkpoints", action="store_true", help="Also write layers/layer_observations_YYYYMMDD and contours_YYYYMMDD.")
    parser.add_argument("--lifetime-min-days", type=int, default=56)
    parser.add_argument("--radius-min-m", type=float, default=50_000.0)
    parser.add_argument("--min-valid-layers", type=int, default=6)
    parser.add_argument("--merge-completed-centers", action="store_true")
    args = parser.parse_args()
    run_pipeline(
        args.config,
        start=args.start,
        end=args.end,
        workers=args.workers,
        force=args.force,
        output_name=args.output_name,
        write_layer_checkpoints=args.write_layer_checkpoints,
        lifetime_min_days=args.lifetime_min_days,
        radius_min_m=args.radius_min_m,
        min_valid_layers=args.min_valid_layers,
        merge_completed_centers=args.merge_completed_centers,
    )


if __name__ == "__main__":
    main()
