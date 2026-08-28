from __future__ import annotations

import argparse
import time
from pathlib import Path

from .common import ensure_dirs, load_config
from .table_io import read_table, table_exists


STAGES = ("convert", "identify", "link", "track", "export")


def _run_stage(name: str, fn, *args, **kwargs) -> None:
    started = time.time()
    print(f"[stage start] {name}")
    fn(*args, **kwargs)
    elapsed = time.time() - started
    print(f"[stage done] {name}: {elapsed / 60.0:.1f} min")


def _convert_fn(config: dict):
    kind = str(config.get("data_source", {}).get("kind", "mat_daily")).lower()
    if kind in {"cmems_netcdf_timeseries", "netcdf_timeseries", "cmems"}:
        from .convert_cmems_nc_to_kuroshio_uv import convert_range as convert_cmems_range

        return convert_cmems_range
    if kind in {"mat_daily", "mat"}:
        from .convert_mat_to_kuroshio_uv import convert_range as convert_mat_range

        return convert_mat_range
    raise ValueError(f"Unsupported data_source.kind: {kind}")


def _validate_catalog_range(config: dict, start: str, end: str) -> None:
    catalog = Path(config["paths"]["catalog_dir"])
    tracks_path = catalog / "tracks_3d.parquet"
    objects_path = catalog / "vertical_objects.parquet"
    if not table_exists(tracks_path) or not table_exists(objects_path):
        print("[catalog check] tracks_3d or vertical_objects missing; skip range check.")
        return
    objects = read_table(objects_path)
    if objects.empty or "date" not in objects.columns:
        print("[catalog check] vertical_objects is empty; skip range check.")
        return
    got_start = str(objects["date"].min())[:10]
    got_end = str(objects["date"].max())[:10]
    if got_start != str(start)[:10] or got_end != str(end)[:10]:
        print(f"[catalog warning] vertical_objects covers {got_start} -> {got_end}, requested {start} -> {end}.")
    else:
        print(f"[catalog check] vertical_objects covers requested range: {got_start} -> {got_end}.")


def run_pipeline(
    config_path: str | Path,
    start: str,
    end: str,
    force: bool = False,
    skip: set[str] | None = None,
    workers: int | None = None,
    depth_index: int | None = None,
) -> None:
    config = load_config(config_path)
    ensure_dirs(config)
    skip = skip or set()
    started = time.time()
    if "convert" not in skip:
        _run_stage("convert", _convert_fn(config), config_path, start, end, force)
    if "identify" not in skip:
        from .run_3d_layer_identification import identify_range

        _run_stage("identify", identify_range, config_path, start, end, force, workers, depth_index)
    ran_link_or_track = False
    if "link" not in skip and depth_index is None:
        from .link_3d_vertical_objects import link_range

        _run_stage("link", link_range, config_path, start, end, force, workers)
        ran_link_or_track = True
    if "track" not in skip and depth_index is None:
        from .track_3d_objects import track_range

        _run_stage("track", track_range, config_path, start, end, force)
        ran_link_or_track = True
    if ran_link_or_track and depth_index is None:
        _validate_catalog_range(config, start, end)
    elapsed = time.time() - started
    print(f"Hybrid 3D pipeline finished in {elapsed / 60.0:.1f} min.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Kuroshio hybrid 3D eddy pipeline.")
    parser.add_argument("--config", default="config/config_3d.yaml")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--depth-index", type=int)
    parser.add_argument("--skip", nargs="*", default=[], choices=STAGES)
    args = parser.parse_args()
    run_pipeline(
        args.config,
        args.start,
        args.end,
        force=args.force,
        skip=set(args.skip),
        workers=args.workers,
        depth_index=args.depth_index,
    )


if __name__ == "__main__":
    main()
