from __future__ import annotations

import argparse
import csv
import shutil
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import yaml

from .build_cmems_climatology import _make_tasks, _run_clim_task, build_climatology
from .classify_3d_eddy_shape import _complete_day_metrics_worker, _eligible_tracks, classify_shapes
from .common import ensure_dirs, iter_days, load_config, parse_ymd
from .run_streaming_layer_identification import (
    identify_day_preloaded,
    identify_day_streaming,
    identify_day_year_cached,
    identify_day_year_shared,
    create_shared_year_cache,
    init_shared_year_cache,
    load_source_year_cache,
    set_year_cache,
)
from .run_streaming_shape_pipeline import _build_streaming_catalog
from .streaming_cmems import build_source_index, selected_depth_indices
from .table_io import read_table
from .track_3d_objects import track_range


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _runtime_config(config_path: str | Path, output_root: Path, temp_root: Path) -> Path:
    config = load_config(config_path)
    paths = config.setdefault("paths", {})
    paths["output_dir"] = str(output_root)
    paths["catalog_dir"] = str(output_root / "catalog")
    paths["layer_dir"] = str(output_root / "layers")
    paths["input_daily_dir"] = str(output_root / "input_daily")
    paths["logs_dir"] = str(output_root / "logs")
    paths["temp_dir"] = str(temp_root)
    config.setdefault("conversion", {})["use_input_daily"] = False
    config["conversion"]["compression_level"] = 1
    config["conversion"].setdefault("depth_block", 8)
    output_root.mkdir(parents=True, exist_ok=True)
    temp_root.mkdir(parents=True, exist_ok=True)
    runtime_path = output_root / "runtime_config.yaml"
    with runtime_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    return runtime_path


def benchmark_identification(config_path: str | Path, out_csv: Path, workers_list: list[int]) -> int:
    config = load_config(config_path)
    start_day = parse_ymd("2020-01-01")
    end_day = parse_ymd("2020-01-07")
    days = list(iter_days(start_day, end_day))
    source_index = build_source_index(config, start_day, end_day)
    depth_indices = selected_depth_indices(config, source_index[days[0]].path)
    rows = []
    best_workers = int(workers_list[0])
    best_elapsed = float("inf")
    for workers in workers_list:
        started = time.perf_counter()
        detections = 0
        for day in days:
            obs, _contours = identify_day_streaming(config, source_index[day], depth_indices, workers=workers, include_contours=False)
            detections += int(len(obs))
        elapsed = time.perf_counter() - started
        rows.append({"workers": int(workers), "days": len(days), "depth_layers": len(depth_indices), "detections": detections, "elapsed_seconds": elapsed})
        if elapsed < best_elapsed:
            best_elapsed = elapsed
            best_workers = int(workers)
    _write_csv(out_csv, rows)
    return best_workers


def benchmark_identification_cache_modes(config_path: str | Path, out_csv: Path, workers_list: list[int]) -> tuple[str, int]:
    config = load_config(config_path)
    start_day = parse_ymd("2020-01-01")
    end_day = parse_ymd("2020-01-30")
    days = list(iter_days(start_day, end_day))
    source_index = build_source_index(config, start_day, end_day)
    depth_indices = selected_depth_indices(config, source_index[days[0]].path)
    modes = ["per_layer_netcdf", "day_preload", "year_ram", "year_shared"]
    rows = []
    best_mode = modes[0]
    best_workers = int(workers_list[0])
    best_elapsed = float("inf")
    for mode in modes:
        for workers in workers_list:
            started = time.perf_counter()
            detections = 0
            if mode == "per_layer_netcdf":
                for day in days:
                    obs, _contours = identify_day_streaming(config, source_index[day], depth_indices, workers=workers, include_contours=False)
                    detections += int(len(obs))
            elif mode == "day_preload":
                for day in days:
                    obs, _contours = identify_day_preloaded(config, source_index[day], depth_indices, workers=workers, include_contours=False)
                    detections += int(len(obs))
            elif mode == "year_ram":
                cache = load_source_year_cache(config, source_index[days[0]])
                set_year_cache(cache)
                try:
                    with ProcessPoolExecutor(max_workers=int(workers)) as executor:
                        for day in days:
                            obs, _contours = identify_day_year_cached(
                                config,
                                source_index[day],
                                depth_indices,
                                workers=workers,
                                include_contours=False,
                                executor=executor,
                            )
                            detections += int(len(obs))
                finally:
                    set_year_cache(None)
            elif mode == "year_shared":
                owner = create_shared_year_cache(config, source_index[days[0]])
                try:
                    with ProcessPoolExecutor(
                        max_workers=int(workers),
                        initializer=init_shared_year_cache,
                        initargs=(owner.payload,),
                    ) as executor:
                        for day in days:
                            obs, _contours = identify_day_year_shared(
                                config,
                                source_index[day],
                                depth_indices,
                                workers=workers,
                                include_contours=False,
                                executor=executor,
                            )
                            detections += int(len(obs))
                finally:
                    owner.cleanup()
            else:
                raise ValueError(mode)
            elapsed = time.perf_counter() - started
            rows.append({
                "cache_mode": mode,
                "workers": int(workers),
                "days": len(days),
                "depth_layers": len(depth_indices),
                "detections": detections,
                "elapsed_seconds": elapsed,
            })
            if elapsed < best_elapsed:
                best_elapsed = elapsed
                best_mode = mode
                best_workers = int(workers)
    _write_csv(out_csv, rows)
    return best_mode, best_workers


def benchmark_climatology(config_path: str | Path, out_csv: Path, workers_list: list[int], temp_root: Path) -> int:
    config = load_config(config_path)
    first_path = Path(config["data_source"]["input_nc_dir"]) / "global_phy_1993.nc"
    rows = []
    best_workers = int(workers_list[0])
    best_elapsed = float("inf")
    for workers in workers_list:
        part_dir = Path(tempfile.mkdtemp(prefix=f"bench_clim_w{workers}_", dir=str(temp_root)))
        try:
            tasks = _make_tasks(config, first_path, part_dir, int(config.get("conversion", {}).get("depth_block", 8)))[: min(4, int(workers))]
            started = time.perf_counter()
            with ProcessPoolExecutor(max_workers=int(workers)) as executor:
                futures = [executor.submit(_run_clim_task, str(config_path), "1993-01-01", "1993-01-31", 31, task) for task in tasks]
                for future in as_completed(futures):
                    future.result()
            elapsed = time.perf_counter() - started
            rows.append({"workers": int(workers), "tasks": len(tasks), "elapsed_seconds": elapsed})
            if elapsed < best_elapsed:
                best_elapsed = elapsed
                best_workers = int(workers)
        finally:
            shutil.rmtree(part_dir, ignore_errors=True)
    _write_csv(out_csv, rows)
    return best_workers


def benchmark_shape(config_path: str | Path, out_csv: Path, workers_list: list[int], temp_root: Path) -> int:
    config = load_config(config_path)
    catalog = Path(config["paths"]["catalog_dir"])
    tracks = read_table(catalog / "tracks_3d.parquet")
    objects = read_table(catalog / "vertical_objects.parquet")
    layers = read_table(catalog / "layer_observations.parquet")
    objects["date"] = pd.to_datetime(objects["date"])
    layers["date"] = pd.to_datetime(layers["date"])
    eligible = _eligible_tracks(tracks, lifetime_min_days=56, radius_min_m=50_000.0)
    eligible_objects = objects[objects["track3d_id"].astype(int).isin(set(eligible["track3d_id"].astype(int)))].copy()
    if eligible_objects.empty:
        _write_csv(out_csv, [{"workers": int(w), "days": 0, "objects": 0, "elapsed_seconds": 0.0} for w in workers_list])
        return int(workers_list[0])
    sample_days = list(eligible_objects["date"].drop_duplicates().sort_values().head(7))
    sample_objects = eligible_objects[eligible_objects["date"].isin(sample_days)].copy()
    sample_ids = set(sample_objects["eddy3d_object_id"].astype(int))
    sample_layers = layers[layers["eddy3d_object_id"].astype(int).isin(sample_ids)].copy()
    layers_by_day = {pd.Timestamp(k): g.copy() for k, g in sample_layers.groupby("date")}
    track_records = eligible.to_dict("records")
    jobs = []
    for day, day_objects in sample_objects.groupby("date", sort=True):
        day_ts = pd.Timestamp(day)
        jobs.append((f"{day_ts:%Y-%m-%d}", day_objects.to_dict("records"), layers_by_day.get(day_ts, pd.DataFrame()).to_dict("records")))
    rows = []
    best_workers = int(workers_list[0])
    best_elapsed = float("inf")
    for workers in workers_list:
        part_root = Path(tempfile.mkdtemp(prefix=f"bench_shape_w{workers}_", dir=str(temp_root)))
        started = time.perf_counter()
        try:
            if int(workers) <= 1 or len(jobs) <= 1:
                for i, (day_label, object_records, layer_records) in enumerate(jobs):
                    _complete_day_metrics_worker(str(config_path), day_label, object_records, layer_records, track_records, 6, str(part_root / f"part_{i}.parquet"), True)
            else:
                with ProcessPoolExecutor(max_workers=int(workers)) as executor:
                    futures = {
                        executor.submit(
                            _complete_day_metrics_worker,
                            str(config_path),
                            day_label,
                            object_records,
                            layer_records,
                            track_records,
                            6,
                            str(part_root / f"part_{i}.parquet"),
                            True,
                        ): day_label
                        for i, (day_label, object_records, layer_records) in enumerate(jobs)
                    }
                    for future in as_completed(futures):
                        future.result()
            elapsed = time.perf_counter() - started
            rows.append({"workers": int(workers), "days": len(jobs), "objects": int(len(sample_objects)), "elapsed_seconds": elapsed})
            if elapsed < best_elapsed:
                best_elapsed = elapsed
                best_workers = int(workers)
        finally:
            shutil.rmtree(part_root, ignore_errors=True)
    _write_csv(out_csv, rows)
    return best_workers


def run_full_pipeline(
    config_path: str | Path,
    *,
    start: str,
    end: str,
    output_root: str | Path,
    temp_root: str | Path,
    identify_workers: int,
    climatology_workers: int | None,
    shape_workers: int | None,
    run_benchmarks: bool,
    force: bool,
    shape_output_name: str | None = None,
) -> None:
    output_root = Path(output_root)
    temp_root = Path(temp_root)
    runtime_config = _runtime_config(config_path, output_root, temp_root)
    config = load_config(runtime_config)
    ensure_dirs(config)
    benchmarks = output_root / "benchmarks"

    if run_benchmarks:
        best_cache_mode, identify_workers = benchmark_identification_cache_modes(
            runtime_config,
            benchmarks / "identification_cache_modes_v2.csv",
            [16, 20, 24, 25, 32],
        )
        config.setdefault("identification", {})["cache_mode"] = best_cache_mode
        config["identification"]["workers"] = int(identify_workers)
        with Path(runtime_config).open("w", encoding="utf-8") as handle:
            yaml.safe_dump(config, handle, sort_keys=False)
        climatology_workers = benchmark_climatology(runtime_config, benchmarks / "climatology_workers.csv", [8, 16, 24, 32], temp_root)

    _build_streaming_catalog(
        runtime_config,
        start=start,
        end=end,
        workers=identify_workers,
        force=force,
        write_layer_checkpoints=False,
    )
    track_range(runtime_config, start=start, end=end, force=force)

    if run_benchmarks:
        shape_workers = benchmark_shape(runtime_config, benchmarks / "shape_workers.csv", [32, 48, 64, 96], temp_root)

    build_climatology(
        runtime_config,
        start=start,
        end=end,
        smooth_days=31,
        force=force,
        workers=int(climatology_workers or 8),
        depth_block=int(config.get("conversion", {}).get("depth_block", 8)),
        temp_dir=temp_root,
        compression_level=1,
    )
    classify_shapes(
        runtime_config,
        complete_missing=True,
        force=force,
        lifetime_min_days=56,
        radius_min_m=50_000.0,
        min_valid_layers=6,
        completion_output_mode="centers-only",
        workers=int(shape_workers or identify_workers),
        start=start,
        end=end,
        output_name=shape_output_name or f"shape_classification_{start[:4]}_{end[:4]}",
        write_completed_centers=True,
        merge_completed_centers=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CPU-only full Kuroshio slim pipeline to results.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--start", default="1993-01-01")
    parser.add_argument("--end", default="2022-12-31")
    parser.add_argument("--identify-workers", type=int, default=32)
    parser.add_argument("--climatology-workers", type=int)
    parser.add_argument("--shape-workers", type=int)
    parser.add_argument("--output-root", default="/root/autodl-fs/1993_2022_kurushio/results")
    parser.add_argument("--temp-root", default="/root/autodl-tmp/kuroshio_streaming_work")
    parser.add_argument("--shape-output-name")
    parser.add_argument("--skip-benchmarks", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    run_full_pipeline(
        args.config,
        start=args.start,
        end=args.end,
        output_root=args.output_root,
        temp_root=args.temp_root,
        identify_workers=args.identify_workers,
        climatology_workers=args.climatology_workers,
        shape_workers=args.shape_workers,
        run_benchmarks=not args.skip_benchmarks,
        force=args.force,
        shape_output_name=args.shape_output_name,
    )


if __name__ == "__main__":
    main()
