"""Download resumable OFES2 monthly potential-density fields via JAMSTEC DAP2."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from pathlib import Path

import numpy as np
from netCDF4 import num2date
from pydap.client import open_url


BASE_URL = "https://www.jamstec.go.jp/esc/fes/dods/OFES2/Monthly/prho"
OUTPUT_ROOT = Path(r"E:\DATA\01_Eddy_correspond\02_OFES\ofes2_monthly_climatology_1993_2012")
SHAPE = (105, 1520, 3600)
FILL_ABS_LIMIT = 1.0e10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--start-year", type=int, required=True)
    parser.add_argument("--end-year", type=int, required=True)
    parser.add_argument("--lat-block-rows", type=int, default=5)
    parser.add_argument(
        "--block-retries",
        type=int,
        default=0,
        help="Maximum retries for one latitude block; 0 keeps retrying the same resumable block.",
    )
    parser.add_argument("--retry-seconds", type=float, default=15.0)
    parser.add_argument("--retry-max-seconds", type=float, default=300.0)
    parser.add_argument("--retry-jitter-seconds", type=float, default=7.0)
    parser.add_argument("--worker-name", required=True)
    return parser.parse_args()


def cache_is_complete(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        data = np.load(path, mmap_mode="r", allow_pickle=False)
        return data.shape == SHAPE and data.dtype == np.dtype("f4")
    except (OSError, ValueError):
        return False


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def write_status(path: Path, rows: list[dict[str, object]]) -> None:
    fields = [
        "year", "month", "remote_index", "status", "bytes", "finite_fraction",
        "invalid_fill_fraction", "updated_at",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_time_index(dataset: object, start_year: int, end_year: int) -> tuple[dict[tuple[int, int], int], np.ndarray, np.ndarray, np.ndarray, str]:
    time_var = dataset["time"]  # type: ignore[index]
    units = str(time_var.attributes["units"])  # type: ignore[union-attr]
    values = np.asarray(time_var[:].data)  # type: ignore[index]
    dates = num2date(values, units, only_use_cftime_datetimes=True)
    index = {(item.year, item.month): position for position, item in enumerate(dates)}
    expected = {(year, month) for year in range(start_year, end_year + 1) for month in range(1, 13)}
    missing = sorted(expected - set(index))
    if missing:
        raise RuntimeError(f"Remote prho source is missing requested months: {missing}")
    return (
        index,
        np.asarray(dataset["lon"][:].data, dtype="f8"),  # type: ignore[index]
        np.asarray(dataset["lat"][:].data, dtype="f8"),  # type: ignore[index]
        np.asarray(dataset["lev"][:].data, dtype="f8"),  # type: ignore[index]
        units,
    )


def partial_paths(path: Path) -> tuple[Path, Path]:
    return path.with_suffix(".npy.part"), path.with_suffix(".npy.progress.json")


def load_progress(path: Path, block_rows: int, block_count: int) -> tuple[set[int], dict[int, int]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("latitude_block_rows") != block_rows or payload.get("block_count") != block_count:
            return set(), {}
        completed = {int(item) for item in payload.get("completed_blocks", [])}
        invalid = {int(key): int(value) for key, value in payload.get("invalid_counts", {}).items()}
        return completed, invalid
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return set(), {}


def save_progress(path: Path, block_rows: int, block_count: int, completed: set[int], invalid: dict[int, int]) -> None:
    write_json(path, {
        "latitude_block_rows": block_rows,
        "block_count": block_count,
        "completed_blocks": sorted(completed),
        "invalid_counts": {str(key): value for key, value in invalid.items()},
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    })


def open_partial(path: Path) -> np.memmap:
    try:
        values = np.load(path, mmap_mode="r+", allow_pickle=False)
        if values.shape == SHAPE and values.dtype == np.dtype("f4"):
            return values
    except (OSError, ValueError):
        pass
    path.unlink(missing_ok=True)
    return np.lib.format.open_memmap(path, mode="w+", dtype="f4", shape=SHAPE)


def retry_delay(args: argparse.Namespace, retry_number: int) -> float:
    """Back off a failed DAP response without re-synchronizing parallel workers."""
    base = args.retry_seconds * (2 ** min(retry_number - 1, 5))
    return min(args.retry_max_seconds, base) + random.uniform(0.0, args.retry_jitter_seconds)


def read_month(remote_index: int, label: str, target_path: Path, args: argparse.Namespace) -> float:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path, progress_path = partial_paths(target_path)
    block_count = (SHAPE[1] + args.lat_block_rows - 1) // args.lat_block_rows
    completed, invalid_counts = load_progress(progress_path, args.lat_block_rows, block_count)
    values = open_partial(partial_path)
    for block_number, start in enumerate(range(0, SHAPE[1], args.lat_block_rows), start=1):
        if block_number in completed:
            print(f"[prho] {label} block {block_number}/{block_count} cached", flush=True)
            continue
        stop = min(SHAPE[1], start + args.lat_block_rows)
        retry_number = 0
        while True:
            try:
                # A fresh DAP handle prevents a broken chunked response from being
                # reused for the next attempt or the next latitude block.
                source = open_url(BASE_URL, protocol="dap2")
                raw = np.ma.asarray(source["prho"][(slice(remote_index, remote_index + 1), slice(None), slice(start, stop), slice(None))].data).filled(np.nan)  # type: ignore[index]
                block = np.asarray(raw, dtype="f4")[0]
                invalid = ~np.isfinite(block) | (np.abs(block) > FILL_ABS_LIMIT)
                invalid_counts[block_number] = int(invalid.sum())
                if invalid.any():
                    block[invalid] = np.nan
                values[:, start:stop, :] = block
                values.flush()
                completed.add(block_number)
                save_progress(progress_path, args.lat_block_rows, block_count, completed, invalid_counts)
                break
            except Exception as exc:
                retry_number += 1
                if args.block_retries and retry_number > args.block_retries:
                    raise RuntimeError(f"{label} prho block {block_number}/{block_count} failed") from exc
                delay = retry_delay(args, retry_number)
                limit = "unbounded" if args.block_retries == 0 else str(args.block_retries)
                print(
                    f"[prho] retry {label} block {block_number}/{block_count} "
                    f"attempt={retry_number}/{limit} in {delay:.0f}s: {exc}",
                    flush=True,
                )
                time.sleep(delay)
        print(f"[prho] {label} block {block_number}/{block_count}", flush=True)
    values.flush()
    del values
    os.replace(partial_path, target_path)
    progress_path.unlink(missing_ok=True)
    return sum(invalid_counts.values()) / np.prod(SHAPE)


def main() -> None:
    args = parse_args()
    if args.start_year > args.end_year or args.lat_block_rows < 1:
        raise ValueError("Invalid year range or latitude block size")
    if args.block_retries < 0 or args.retry_seconds <= 0 or args.retry_max_seconds < args.retry_seconds or args.retry_jitter_seconds < 0:
        raise ValueError("Invalid retry configuration")
    state_root = args.output_root / "workers" / args.worker_name
    state_root.mkdir(parents=True, exist_ok=True)
    status_path = state_root / "monthly_prho_status.csv"
    manifest_path = state_root / "monthly_prho_manifest.json"

    dataset = open_url(BASE_URL, protocol="dap2")
    index, lon, lat, depth, time_units = read_time_index(dataset, args.start_year, args.end_year)
    if (depth.size, lat.size, lon.size) != SHAPE:
        raise RuntimeError(f"Unexpected remote prho shape: {(depth.size, lat.size, lon.size)}")
    manifest = {
            "status": "running",
            "source_url": BASE_URL,
            "protocol": "DAP2 via pydap",
            "period": {"start_year": args.start_year, "end_year": args.end_year},
            "native_shape": list(SHAPE),
            "coordinates": {"longitude_count": int(lon.size), "latitude_count": int(lat.size), "depth_count": int(depth.size), "time_units": time_units},
            "cache_policy": "raw_monthly/YYYYMM/prho.npy; complete native float32 files are resumed",
            "worker_name": args.worker_name,
            "latitude_block_rows": args.lat_block_rows,
            "retry_policy": {
                "maximum_retries_per_block": "unbounded" if args.block_retries == 0 else args.block_retries,
                "initial_backoff_seconds": args.retry_seconds,
                "maximum_backoff_seconds": args.retry_max_seconds,
                "jitter_seconds": args.retry_jitter_seconds,
                "fresh_dap_handle_per_attempt": True,
            },
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
    write_json(manifest_path, manifest)
    rows: list[dict[str, object]] = []
    for year in range(args.start_year, args.end_year + 1):
        for month in range(1, 13):
            path = args.output_root / "raw_monthly" / f"{year:04d}{month:02d}" / "prho.npy"
            remote_index = index[(year, month)]
            if cache_is_complete(path):
                data = np.load(path, mmap_mode="r", allow_pickle=False)
                status = "cached"
                invalid_fraction = float((~np.isfinite(data)).mean())
            else:
                invalid_fraction = read_month(remote_index, f"{year:04d}-{month:02d}", path, args)
                data = np.load(path, mmap_mode="r", allow_pickle=False)
                status = "downloaded"
            rows.append({
                "year": year, "month": month, "remote_index": remote_index, "status": status,
                "bytes": path.stat().st_size, "finite_fraction": float(np.isfinite(data).mean()),
                "invalid_fill_fraction": invalid_fraction, "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            })
            write_status(status_path, rows)
            print(f"[prho] {year:04d}-{month:02d} {status}", flush=True)
    manifest["status"] = "complete"
    manifest["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    write_json(manifest_path, manifest)
    print(f"[prho] worker complete: {args.worker_name}", flush=True)


if __name__ == "__main__":
    main()
