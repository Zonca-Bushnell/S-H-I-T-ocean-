"""Rebuild missing tracking voxels from saved detections, without rerunning Hua."""

from __future__ import annotations

import argparse
import json
import time
from datetime import date
from pathlib import Path

from ..src.eddy_pipeline import detection_hybrid as detection

import numpy as np
import pandas as pd
from fastparquet import ParquetFile
from netCDF4 import Dataset


def run(args):
    root = Path(args.detection_dir)
    saved = json.loads((root / 'run_summary.json').read_text(encoding='utf-8'))
    config = saved['parameters']
    start = date.fromisoformat(args.start or config['start'])
    end = date.fromisoformat(args.end or config['end'])
    dates = sorted(
        (date.fromisoformat(path.stem.removeprefix('date=')), path)
        for path in (root / 'parts' / 'centers').glob('date=*.parquet')
    )
    dates = [(day, path) for day, path in dates if start <= day <= end]
    if len(dates) != (end - start).days + 1:
        raise ValueError('Missing daily center parts in requested date interval')
    records = []
    dataset = None
    year = None
    started = time.perf_counter()
    try:
        for index, (day, path) in enumerate(dates, 1):
            tick = time.perf_counter()
            target = root / 'object_voxels_parts' / f'year={day.year}' / f'date={day:%Y%m%d}.parquet'
            existing_rows = ParquetFile(target).count() if target.exists() else 0
            if existing_rows:
                records.append({'date': day.isoformat(), 'voxel_rows': existing_rows, 'status': 'existing'})
                print(f'[backfill] {index}/{len(dates)} {day} existing rows={existing_rows}', flush=True)
                continue
            centers = detection._read_parquet(path)
            passed = centers.loc[centers.hua_pass.astype(bool)]
            if dataset is None or year != day.year:
                if dataset is not None:
                    dataset.close()
                year = day.year
                source = Path(config['filter_root']) / detection._format_year_template(config['filter_template'], year)
                dataset = Dataset(source)
                detection._configure_var_chunk_cache(dataset, ('uo_glor', 'vo_glor'), 256)
                times = detection._time_lookup(dataset)
                lon = np.asarray(dataset.variables['longitude'][:], dtype='float64')
                lat = np.asarray(dataset.variables['latitude'][:], dtype='float64')
                depth = np.asarray(dataset.variables['depth'][:], dtype='float64')
            tidx = times[day]
            max_layer = int(passed.depth_index.max()) + 1 if len(passed) else 0
            u = np.asarray(dataset.variables['uo_glor'][tidx, :max_layer], dtype='float32')
            v = np.asarray(dataset.variables['vo_glor'][tidx, :max_layer], dtype='float32')
            rows = []
            populated_layers = 0
            for row in passed.itertuples(index=False):
                layer = int(row.depth_index)
                voxels = detection._object_voxels_for_layer(
                    u[layer], v[layer], lon, lat, float(depth[layer]),
                    day=day, object_id=str(row.hua_object_id), depth_index=layer,
                    center_i=int(row.speed_min_i_grid), center_j=int(row.speed_min_j_grid),
                    radius_cells=float(row.accepted_radius_cells), polarity=row.polarity,
                )
                populated_layers += bool(voxels)
                rows.extend(voxels)
            voxels = pd.DataFrame(rows, columns=detection.OBJECT_VOXEL_COLUMNS)
            if len(passed) and voxels.empty:
                raise ValueError(f'{day}: passed layers exist but rebuilt voxels are empty')
            detection._write_parts(voxels, target)
            records.append({
                'date': day.isoformat(), 'status': 'rebuilt',
                'pass_layers': len(passed), 'populated_layers': populated_layers,
                'voxel_rows': len(voxels), 'seconds': time.perf_counter() - tick,
            })
            print(f'[backfill] {index}/{len(dates)} {day} pass={len(passed)} '
                  f'populated={populated_layers} voxels={len(voxels)} '
                  f'seconds={time.perf_counter() - tick:.2f}', flush=True)
            del rows, voxels, u, v
    finally:
        if dataset is not None:
            dataset.close()
    report = root / f'voxel_backfill_{start:%Y%m%d}_{end:%Y%m%d}.json'
    report.write_text(json.dumps({
        'start': start.isoformat(), 'end': end.isoformat(),
        'seconds': time.perf_counter() - started, 'days': records,
        'total_voxel_rows': sum(row['voxel_rows'] for row in records),
    }, indent=2), encoding='utf-8')
    if args.refresh_frame_summary:
        print('[backfill] refreshing frame_object_summary', flush=True)
        centers = detection._read_parquet(root / 'centers_hua_style.parquet')
        detection._write_frame_object_summary(
            centers, pd.DataFrame(columns=detection.OBJECT_VOXEL_COLUMNS), root,
        )
    print(f'[backfill] complete seconds={time.perf_counter() - started:.2f}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--detection-dir', required=True)
    parser.add_argument('--start')
    parser.add_argument('--end')
    parser.add_argument('--refresh-frame-summary', action='store_true')
    run(parser.parse_args())


if __name__ == '__main__':
    main()
