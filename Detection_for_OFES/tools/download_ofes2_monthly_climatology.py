"""Download a resumable OFES2 monthly SSH climatology from JAMSTEC OPeNDAP.

The source fields remain on their native OFES2 0.1-degree grid.  This tool is
deliberately independent from the detection inputs and catalogues.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from netCDF4 import Dataset, num2date


BASE_URL = "https://www.jamstec.go.jp/esc/fes/dods/OFES2/Monthly"
DEFAULT_OUTPUT_ROOT = Path(
    r"E:\DATA\01_Eddy_correspond\02_OFES\ofes2_monthly_climatology_1993_2012"
)
FIELDS = ("eta", "pair")
SHAPE = (1520, 3600)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--start-year", type=int, default=1993)
    parser.add_argument("--end-year", type=int, default=2012)
    parser.add_argument("--base-url", default=BASE_URL)
    parser.add_argument("--lat-block-rows", type=int, default=80)
    parser.add_argument("--block-retries", type=int, default=5)
    parser.add_argument("--retry-seconds", type=float, default=10.0)
    parser.add_argument("--preview", action="store_true", help="Write a January climatology preview after completion.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--download-only", action="store_true", help="Cache the requested monthly source fields without building products.")
    mode.add_argument("--build-only", action="store_true", help="Build products from an already complete raw_monthly cache.")
    parser.add_argument("--worker-name", default="single", help="Unique state/logical name when using --download-only.")
    return parser.parse_args()


def atomic_save_array(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("wb") as handle:
        np.save(handle, values.astype("f4", copy=False), allow_pickle=False)
    os.replace(temporary, path)


def load_complete_array(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    try:
        values = np.load(path, allow_pickle=False, mmap_mode="r")
        if values.shape != SHAPE or values.dtype != np.dtype("f4"):
            return None
        return np.asarray(values)
    except (OSError, ValueError):
        return None


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def build_time_index(dataset: Dataset, start_year: int, end_year: int) -> tuple[dict[tuple[int, int], int], np.ndarray, np.ndarray, str]:
    time_variable = dataset.variables["time"]
    dates = num2date(time_variable[:], time_variable.units, only_use_cftime_datetimes=True)
    index = {(item.year, item.month): position for position, item in enumerate(dates)}
    expected = {(year, month) for year in range(start_year, end_year + 1) for month in range(1, 13)}
    missing = sorted(expected - set(index))
    if missing:
        raise RuntimeError(f"Remote source is missing requested months: {missing}")
    return index, np.asarray(dataset.variables["lon"][:], dtype="f8"), np.asarray(dataset.variables["lat"][:], dtype="f8"), str(time_variable.units)


def read_remote_month(
    dataset: Dataset,
    url: str,
    field: str,
    index: int,
    label: str,
    lat_block_rows: int,
    block_retries: int,
    retry_seconds: float,
) -> np.ndarray:
    values = np.full(SHAPE, np.nan, dtype="f4")
    block_count = (SHAPE[0] + lat_block_rows - 1) // lat_block_rows
    for block_index, start in enumerate(range(0, SHAPE[0], lat_block_rows), start=1):
        stop = min(SHAPE[0], start + lat_block_rows)
        for attempt in range(block_retries + 1):
            retry_dataset: Dataset | None = None
            try:
                source = dataset if attempt == 0 else Dataset(url)
                retry_dataset = source if attempt > 0 else None
                block = np.ma.asarray(source.variables[field][index, 0, start:stop, :]).filled(np.nan)
                values[start:stop] = np.asarray(block, dtype="f4")
                break
            except Exception as exc:
                if attempt >= block_retries:
                    raise RuntimeError(f"{label} {field} block {block_index}/{block_count} failed after {block_retries + 1} attempts") from exc
                print(
                    f"[climatology] retry {label} {field} block {block_index}/{block_count} "
                    f"attempt {attempt + 1}/{block_retries}: {exc}",
                    flush=True,
                )
                time.sleep(retry_seconds * (attempt + 1))
            finally:
                if retry_dataset is not None:
                    retry_dataset.close()
        print(f"[climatology] {label} {field} block {block_index}/{block_count}", flush=True)
    values[~np.isfinite(values)] = np.nan
    return values


def write_status(path: Path, rows: list[dict[str, object]]) -> None:
    fields = [
        "year", "month", "remote_index", "eta_status", "pair_status", "eta_bytes", "pair_bytes",
        "eta_finite_fraction", "pair_finite_fraction", "h_finite_fraction", "updated_at",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build_products(output_root: Path, lon: np.ndarray, lat: np.ndarray, rows: list[dict[str, object]], start_year: int, end_year: int) -> None:
    sums = {field: np.zeros((12, *SHAPE), dtype="f8") for field in ("eta", "pair", "h")}
    counts = {field: np.zeros((12, *SHAPE), dtype="u1") for field in ("eta", "pair", "h")}
    for row in rows:
        month_index = int(row["month"]) - 1
        monthly = {}
        for field in FIELDS:
            path = output_root / "raw_monthly" / f"{int(row['year']):04d}{int(row['month']):02d}" / f"{field}.npy"
            values = load_complete_array(path)
            if values is None:
                raise RuntimeError(f"Cannot build climatology; missing cached field: {path}")
            monthly[field] = values.astype("f8")
            valid = np.isfinite(monthly[field])
            sums[field][month_index][valid] += monthly[field][valid]
            counts[field][month_index][valid] += 1
        h = monthly["eta"] - (monthly["pair"] - 1000.0)
        valid_h = np.isfinite(h)
        sums["h"][month_index][valid_h] += h[valid_h]
        counts["h"][month_index][valid_h] += 1

    means = {
        field: np.divide(sums[field], counts[field], out=np.full_like(sums[field], np.nan), where=counts[field] > 0).astype("f4")
        for field in sums
    }
    product_root = output_root / "climatology"
    product_root.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        product_root / f"ofes2_eta_pair_h_monthly_climatology_{start_year}_{end_year}.npz",
        longitude=lon,
        latitude=lat,
        months=np.arange(1, 13, dtype="i1"),
        eta_monthly_mean_cm=means["eta"],
        pair_monthly_mean_hpa=means["pair"],
        h_monthly_mean_cm=means["h"],
        eta_valid_sample_count=counts["eta"],
        pair_valid_sample_count=counts["pair"],
        h_valid_sample_count=counts["h"],
    )
    season_months = {"DJF": (12, 1, 2), "MAM": (3, 4, 5), "JJA": (6, 7, 8), "SON": (9, 10, 11)}
    seasonal_h = np.stack([np.nanmean(means["h"][np.asarray(months) - 1], axis=0) for months in season_months.values()]).astype("f4")
    np.savez_compressed(
        product_root / f"ofes2_eta_pair_h_seasonal_climatology_{start_year}_{end_year}.npz",
        longitude=lon,
        latitude=lat,
        seasons=np.asarray(tuple(season_months), dtype="U3"),
        h_seasonal_mean_cm=seasonal_h,
    )


def write_preview(output_root: Path, lon: np.ndarray, lat: np.ndarray, start_year: int, end_year: int) -> None:
    import matplotlib.pyplot as plt

    path = output_root / "climatology" / f"ofes2_eta_pair_h_monthly_climatology_{start_year}_{end_year}.npz"
    with np.load(path, allow_pickle=False) as data:
        january = data["h_monthly_mean_cm"][0]
    limit = float(np.nanpercentile(np.abs(january), 98.0))
    figure, axis = plt.subplots(figsize=(16, 8), constrained_layout=True)
    mesh = axis.pcolormesh(lon, lat, january, shading="auto", cmap="RdBu_r", vmin=-limit, vmax=limit)
    axis.set(title=f"OFES2 January climatological H | {start_year}-{end_year}", xlabel="longitude", ylabel="latitude")
    figure.colorbar(mesh, ax=axis, label="cm")
    figure.savefig(output_root / "climatology" / f"ofes2_h_january_climatology_preview_{start_year}_{end_year}.png", dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if args.start_year > args.end_year:
        raise ValueError("--start-year must not exceed --end-year")
    if args.lat_block_rows < 1:
        raise ValueError("--lat-block-rows must be positive")
    if args.block_retries < 0 or args.retry_seconds < 0:
        raise ValueError("--block-retries and --retry-seconds must not be negative")
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    state_root = output_root if not args.download_only else output_root / "workers" / args.worker_name
    manifest_path = state_root / "manifest.json"
    status_path = state_root / "monthly_status.csv"
    state_root.mkdir(parents=True, exist_ok=True)
    eta_url = f"{args.base_url}/eta"
    pair_url = f"{args.base_url}/pair"
    with Dataset(eta_url) as eta_dataset, Dataset(pair_url) as pair_dataset:
        index, lon, lat, time_units = build_time_index(eta_dataset, args.start_year, args.end_year)
        pair_index, pair_lon, pair_lat, pair_time_units = build_time_index(pair_dataset, args.start_year, args.end_year)
        if not (np.array_equal(lon, pair_lon) and np.array_equal(lat, pair_lat) and index == pair_index):
            raise RuntimeError("Remote eta/pair grids or monthly time coordinates do not match")
        manifest = {
            "status": "running",
            "source": {"eta_url": eta_url, "pair_url": pair_url, "time_units": time_units, "pair_time_units": pair_time_units},
            "period": {"start_year": args.start_year, "end_year": args.end_year, "samples_per_calendar_month": args.end_year - args.start_year + 1},
            "native_grid": {"shape": list(SHAPE), "longitude_count": int(lon.size), "latitude_count": int(lat.size), "longitude_step_degree": float(np.median(np.diff(lon))), "latitude_step_degree": float(np.median(np.diff(lat)))},
            "formula": "H_cm = eta_cm - (pair_hPa - 1000); hPa and mb are numerically equivalent",
            "cache_policy": "one native float32 eta.npy and pair.npy per month; valid caches are resumed",
            "mode": "build_only" if args.build_only else "download_only" if args.download_only else "download_and_build",
            "worker_name": args.worker_name,
            "remote_read_policy": {
                "latitude_block_rows": args.lat_block_rows,
                "full_longitude_per_request": True,
                "block_retries": args.block_retries,
                "retry_seconds": args.retry_seconds,
            },
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        write_json(manifest_path, manifest)
        rows: list[dict[str, object]] = []
        for year in range(args.start_year, args.end_year + 1):
            for month in range(1, 13):
                remote_index = index[(year, month)]
                cache_dir = output_root / "raw_monthly" / f"{year:04d}{month:02d}"
                eta_path, pair_path = cache_dir / "eta.npy", cache_dir / "pair.npy"
                eta = load_complete_array(eta_path)
                eta_status = "cached" if eta is not None else "missing"
                if eta is None and not args.build_only:
                    eta = read_remote_month(
                        eta_dataset, eta_url, "eta", remote_index, f"{year:04d}-{month:02d}",
                        args.lat_block_rows, args.block_retries, args.retry_seconds,
                    )
                    atomic_save_array(eta_path, eta)
                    eta_status = "downloaded"
                pair = load_complete_array(pair_path)
                pair_status = "cached" if pair is not None else "missing"
                if pair is None and not args.build_only:
                    pair = read_remote_month(
                        pair_dataset, pair_url, "pair", remote_index, f"{year:04d}-{month:02d}",
                        args.lat_block_rows, args.block_retries, args.retry_seconds,
                    )
                    atomic_save_array(pair_path, pair)
                    pair_status = "downloaded"
                if eta is None or pair is None:
                    raise RuntimeError(f"Required cache is incomplete for {year:04d}-{month:02d}")
                h = eta.astype("f8") - (pair.astype("f8") - 1000.0)
                row = {
                    "year": year, "month": month, "remote_index": remote_index,
                    "eta_status": eta_status, "pair_status": pair_status,
                    "eta_bytes": eta_path.stat().st_size, "pair_bytes": pair_path.stat().st_size,
                    "eta_finite_fraction": float(np.isfinite(eta).mean()), "pair_finite_fraction": float(np.isfinite(pair).mean()),
                    "h_finite_fraction": float(np.isfinite(h).mean()), "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                }
                rows.append(row)
                write_status(status_path, rows)
                print(f"[climatology] {year:04d}-{month:02d} eta={eta_status} pair={pair_status}", flush=True)
    if args.download_only:
        manifest["status"] = "complete"
        manifest["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        write_json(manifest_path, manifest)
        print(f"[climatology] download worker complete: {args.worker_name}", flush=True)
        return
    build_products(output_root, lon, lat, rows, args.start_year, args.end_year)
    if args.preview:
        write_preview(output_root, lon, lat, args.start_year, args.end_year)
    manifest["status"] = "complete"
    manifest["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    manifest["outputs"] = {"monthly": f"climatology/ofes2_eta_pair_h_monthly_climatology_{args.start_year}_{args.end_year}.npz", "seasonal": f"climatology/ofes2_eta_pair_h_seasonal_climatology_{args.start_year}_{args.end_year}.npz"}
    write_json(manifest_path, manifest)
    print(f"[climatology] complete: {output_root}", flush=True)


if __name__ == "__main__":
    main()
