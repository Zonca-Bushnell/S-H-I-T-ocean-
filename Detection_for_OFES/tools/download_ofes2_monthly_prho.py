"""Download resumable OFES2 monthly potential-density fields via JAMSTEC DAP2."""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from contextlib import nullcontext
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
    parser.add_argument("--lat-block-rows", type=int, default=20)
    parser.add_argument("--block-retries", type=int, default=5)
    parser.add_argument("--retry-seconds", type=float, default=10.0)
    parser.add_argument("--worker-name", required=True)
    return parser.parse_args()


def atomic_save(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".npy.part")
    with temporary.open("wb") as handle:
        np.save(handle, values.astype("f4", copy=False), allow_pickle=False)
    os.replace(temporary, path)


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


def read_month(dataset: object, remote_index: int, label: str, args: argparse.Namespace) -> tuple[np.ndarray, float]:
    values = np.full(SHAPE, np.nan, dtype="f4")
    block_count = (SHAPE[1] + args.lat_block_rows - 1) // args.lat_block_rows
    invalid_count = 0
    for block_number, start in enumerate(range(0, SHAPE[1], args.lat_block_rows), start=1):
        stop = min(SHAPE[1], start + args.lat_block_rows)
        for attempt in range(args.block_retries + 1):
            try:
                source = dataset if attempt == 0 else open_url(BASE_URL, protocol="dap2")
                raw = np.ma.asarray(source["prho"][(slice(remote_index, remote_index + 1), slice(None), slice(start, stop), slice(None))].data).filled(np.nan)  # type: ignore[index]
                block = np.asarray(raw, dtype="f4")[0]
                invalid = ~np.isfinite(block) | (np.abs(block) > FILL_ABS_LIMIT)
                invalid_count += int(invalid.sum())
                if invalid.any():
                    block[invalid] = np.nan
                values[:, start:stop, :] = block
                break
            except Exception as exc:
                if attempt >= args.block_retries:
                    raise RuntimeError(f"{label} prho block {block_number}/{block_count} failed") from exc
                delay = args.retry_seconds * (attempt + 1)
                print(f"[prho] retry {label} block {block_number}/{block_count} in {delay:.0f}s: {exc}", flush=True)
                time.sleep(delay)
        print(f"[prho] {label} block {block_number}/{block_count}", flush=True)
    return values, invalid_count / values.size


def main() -> None:
    args = parse_args()
    if args.start_year > args.end_year or args.lat_block_rows < 1:
        raise ValueError("Invalid year range or latitude block size")
    state_root = args.output_root / "workers" / args.worker_name
    state_root.mkdir(parents=True, exist_ok=True)
    status_path = state_root / "monthly_prho_status.csv"
    manifest_path = state_root / "monthly_prho_manifest.json"

    with nullcontext(open_url(BASE_URL, protocol="dap2")) as dataset:
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
                    data, invalid_fraction = read_month(dataset, remote_index, f"{year:04d}-{month:02d}", args)
                    atomic_save(path, data)
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
