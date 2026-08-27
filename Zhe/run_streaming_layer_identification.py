from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from .common import ensure_dirs, iter_days, load_config, parse_ymd
from .run_3d_layer_identification import layer_output_paths
from .streaming_cmems import SourceDay, build_source_index, read_layer_arrays, selected_depth_indices
from .table_io import table_exists, write_table
from .velocity3d_core import LayerDetection, detect_surface_sla_fallback, detect_velocity_layer


def _valid_outputs(paths: tuple[Path, Path]) -> bool:
    return all(table_exists(path) for path in paths)


def _records_from_detection(det: LayerDetection) -> tuple[dict, list[dict]]:
    record = {
        "layer_detection_id": det.detection_id,
        "date": det.date,
        "depth_m": det.depth_m,
        "depth_index": det.depth_index,
        "polarity": det.polarity,
        "longitude": det.longitude,
        "latitude": det.latitude,
        "core_speed": det.core_speed,
        "vorticity": det.vorticity,
        "area_m2": det.area_m2,
        "radius_m": det.radius_m,
        "method": det.method,
        "reversal_passed": det.reversal_passed,
    }
    contour_rows = [
        {
            "layer_detection_id": det.detection_id,
            "date": det.date,
            "depth_m": det.depth_m,
            "depth_index": det.depth_index,
            "point_index": i,
            "longitude": float(x),
            "latitude": float(y),
        }
        for i, (x, y) in enumerate(zip(det.contour_lon, det.contour_lat))
    ]
    return record, contour_rows


def _detection_kwargs(config: dict) -> dict:
    kwargs = dict(config.get("identification", {}))
    kwargs.pop("workers", None)
    kwargs.pop("overwrite_existing", None)
    return kwargs


def _run_layer_task(task: tuple[dict, SourceDay, int, bool]) -> tuple[str, int, list[dict], list[dict]]:
    config, source_day, depth_index, include_contours = task
    id_cfg = config.get("identification", {})
    arrays = read_layer_arrays(config, source_day, depth_index)
    kwargs = _detection_kwargs(config)
    day_iso = source_day.day.isoformat()
    if int(depth_index) == int(id_cfg.get("surface_depth_index", 0)):
        detections = detect_surface_sla_fallback(
            arrays["lon"],
            arrays["lat"],
            arrays["adt"],
            float(arrays["depth_m"]),
            int(depth_index),
            day_iso,
            **kwargs,
        )
    else:
        detections = detect_velocity_layer(
            arrays["lon"],
            arrays["lat"],
            float(arrays["depth_m"]),
            int(depth_index),
            arrays["u"],
            arrays["v"],
            day_iso,
            **kwargs,
        )

    records: list[dict] = []
    contours: list[dict] = []
    for local_id, det in enumerate(detections):
        det = det.__class__(**{**det.__dict__, "detection_id": int(depth_index) * 100000 + int(local_id)})
        record, rows = _records_from_detection(det)
        records.append(record)
        if include_contours:
            contours.extend(rows)
    return day_iso, int(depth_index), records, contours


def _write_day_outputs(config: dict, day_iso: str, records: list[dict], contours: list[dict]) -> None:
    day = parse_ymd(day_iso)
    obs_path, contour_path = layer_output_paths(config, day)
    write_table(pd.DataFrame.from_records(records), obs_path, index=False)
    write_table(pd.DataFrame.from_records(contours), contour_path, index=False)


def identify_day_streaming(
    config: dict,
    source_day: SourceDay,
    depth_indices: list[int],
    *,
    workers: int = 1,
    include_contours: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    records: list[dict] = []
    contours: list[dict] = []
    tasks = [(config, source_day, int(k), include_contours) for k in depth_indices]
    if workers <= 1 or len(tasks) <= 1:
        for task in tasks:
            _day_iso, _k, layer_records, layer_contours = _run_layer_task(task)
            records.extend(layer_records)
            contours.extend(layer_contours)
    else:
        with ProcessPoolExecutor(max_workers=int(workers)) as executor:
            futures = {executor.submit(_run_layer_task, task): task for task in tasks}
            for future in as_completed(futures):
                try:
                    _day_iso, _k, layer_records, layer_contours = future.result()
                except Exception as exc:
                    raise RuntimeError(f"Streaming layer identification failed for {source_day.day}") from exc
                records.extend(layer_records)
                contours.extend(layer_contours)
    return pd.DataFrame.from_records(records), pd.DataFrame.from_records(contours)


def identify_range_streaming(
    config_path: str | Path,
    start: str | None = None,
    end: str | None = None,
    force: bool = False,
    workers: int | None = None,
    depth_index: int | None = None,
) -> None:
    config = load_config(config_path)
    ensure_dirs(config)
    start_day = parse_ymd(start or config["date_range"]["start"])
    end_day = parse_ymd(end or config["date_range"]["end"])
    days = list(iter_days(start_day, end_day))
    source_index = build_source_index(config, start_day, end_day)
    first_path = source_index[days[0]].path
    depth_indices = [int(depth_index)] if depth_index is not None else selected_depth_indices(config, first_path)
    workers = int(workers or config.get("identification", {}).get("workers", 1) or 1)

    pending_days = []
    for day in days:
        paths = layer_output_paths(config, day)
        if force or config.get("identification", {}).get("overwrite_existing", False) or not _valid_outputs(paths):
            pending_days.append(day)
    if not pending_days:
        print("Streaming layer identification: all daily outputs already exist.")
        return

    with tqdm(total=len(pending_days), desc=f"Streaming layer identification ({workers} workers/day)", unit="day") as bar:
        for day in pending_days:
            obs, contours = identify_day_streaming(
                config,
                source_index[day],
                depth_indices,
                workers=workers,
                include_contours=True,
            )
            _write_day_outputs(config, day.isoformat(), obs.to_dict("records"), contours.to_dict("records"))
            bar.update(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="CPU-only streaming CMEMS layer identification without input_daily files.")
    parser.add_argument("--config", default="config/config_3d.yaml")
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--depth-index", type=int)
    args = parser.parse_args()
    identify_range_streaming(args.config, args.start, args.end, force=args.force, workers=args.workers, depth_index=args.depth_index)


if __name__ == "__main__":
    main()
