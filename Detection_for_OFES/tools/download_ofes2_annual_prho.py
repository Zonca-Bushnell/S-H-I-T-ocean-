"""Download server-side, day-weighted OFES2 annual prho fields and their climatology."""

from __future__ import annotations

import argparse
import calendar
import csv
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import requests
from pydap.client import open_url


DATASET = "OFES2/Monthly/prho"
GDS_ROOT = "https://www.jamstec.go.jp/esc/fes/dods"
OUTPUT_ROOT = Path(r"E:\DATA\01_Eddy_correspond\02_OFES\ofes2_monthly_climatology_1993_2012")
SHAPE = (105, 1520, 3600)
LON = 0.05 + 0.1 * np.arange(SHAPE[2], dtype="f8")
LAT = -75.95 + 0.1 * np.arange(SHAPE[1], dtype="f8")
MONTH_NAMES = ("JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC")
FILL_ABS_LIMIT = 1.0e10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--start-year", type=int, default=1993)
    parser.add_argument("--end-year", type=int, default=2012)
    parser.add_argument("--worker-name", default="annual_prho")
    parser.add_argument("--lat-block-rows", type=int, default=10)
    parser.add_argument("--depth-block-layers", type=int, default=20)
    parser.add_argument("--block-retries", type=int, default=5)
    parser.add_argument("--retry-seconds", type=float, default=15.0)
    parser.add_argument("--finalize-only", action="store_true")
    return parser.parse_args()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def cache_is_complete(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        values = np.load(path, mmap_mode="r", allow_pickle=False)
        return values.shape == SHAPE and values.dtype == np.dtype("f4")
    except (OSError, ValueError):
        return False


def annual_expression(year: int) -> tuple[str, int]:
    terms: list[str] = []
    total_days = 0
    for month, name in enumerate(MONTH_NAMES, start=1):
        days = calendar.monthrange(year, month)[1]
        total_days += days
        terms.append(f"{days}*prho(time=00Z01{name}{year})")

    # GDS rejects a long left-associative sum but accepts the equivalent
    # balanced expression tree.
    def balanced_sum(values: list[str]) -> str:
        if len(values) == 1:
            return values[0]
        midpoint = len(values) // 2
        return f"({balanced_sum(values[:midpoint])}+{balanced_sum(values[midpoint:])})"

    return f"{balanced_sum(terms)}/{total_days}", total_days


def read_depth_coordinate() -> np.ndarray:
    response = requests.get(f"{GDS_ROOT}/{DATASET}.ascii?lev", timeout=120)
    response.raise_for_status()
    lines = response.text.splitlines()
    if len(lines) < 2:
        raise RuntimeError("OFES2 prho depth coordinate response is incomplete")
    depth = np.fromstring(" ".join(lines[1:]).replace(",", " "), sep=" ", dtype="f8")
    if depth.shape != (SHAPE[0],):
        raise RuntimeError(f"Unexpected OFES2 prho depth coordinate shape: {depth.shape}")
    return depth


def expression_url(
    year: int,
    k0: int,
    k1: int,
    j0: int,
    j1: int,
    depth: np.ndarray,
) -> tuple[str, str, int]:
    expression, total_days = annual_expression(year)
    domain = (
        f"{LON[0]:.2f}:{LON[-1]:.2f},"
        f"{LAT[j0]:.2f}:{LAT[j1 - 1]:.2f},"
        f"{depth[k0]:.6f}:{depth[k1 - 1]:.6f},"
        f"00Z01JAN{year}:00Z01JAN{year}"
    )
    url = f"{GDS_ROOT}/_expr_{{{DATASET}}}{{{expression}}}{{{domain}}}"
    digest = hashlib.sha256(expression.encode("ascii")).hexdigest()
    return url, digest, total_days


def load_resume_state(path: Path, digest: str) -> set[str]:
    if not path.exists():
        return set()
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return set()
    if state.get("expression_sha256") != digest:
        return set()
    return {str(value) for value in state.get("completed_blocks", [])}


def read_block(
    url: str,
    expected_layers: int,
    expected_rows: int,
    args: argparse.Namespace,
    label: str,
) -> np.ndarray:
    expected = (1, expected_layers, expected_rows, SHAPE[2])
    for attempt in range(args.block_retries + 1):
        try:
            dataset = open_url(url, protocol="dap2")
            raw = np.ma.asarray(dataset["result"][:].data).filled(np.nan)
            values = np.asarray(raw, dtype="f4")
            if values.shape != expected:
                raise RuntimeError(f"unexpected result shape {values.shape}, expected {expected}")
            values = values[0]
            invalid = ~np.isfinite(values) | (np.abs(values) > FILL_ABS_LIMIT)
            if invalid.any():
                values[invalid] = np.nan
            return values
        except Exception as exc:
            if attempt >= args.block_retries:
                raise RuntimeError(f"{label} failed after {attempt + 1} attempts") from exc
            delay = args.retry_seconds * (attempt + 1)
            print(f"[annual-prho] retry {label} in {delay:.0f}s: {exc}", flush=True)
            time.sleep(delay)
    raise AssertionError("unreachable")


def download_year(year: int, args: argparse.Namespace, depth: np.ndarray) -> dict[str, object]:
    annual_dir = args.output_root / "raw_annual" / f"{year:04d}"
    annual_dir.mkdir(parents=True, exist_ok=True)
    final_path = annual_dir / "prho_annual_mean.npy"
    state_path = annual_dir / "download_state.json"
    _, digest, total_days = expression_url(
        year,
        0,
        min(args.depth_block_layers, SHAPE[0]),
        0,
        min(args.lat_block_rows, SHAPE[1]),
        depth,
    )
    if cache_is_complete(final_path):
        data = np.load(final_path, mmap_mode="r", allow_pickle=False)
        return {
            "year": year,
            "status": "cached",
            "days": total_days,
            "bytes": final_path.stat().st_size,
            "finite_fraction": float(np.isfinite(data).mean()),
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }

    part_path = final_path.with_suffix(".npy.part")
    if part_path.exists():
        try:
            output = np.lib.format.open_memmap(part_path, mode="r+", dtype="f4", shape=SHAPE)
        except (OSError, ValueError):
            part_path.unlink(missing_ok=True)
            output = np.lib.format.open_memmap(part_path, mode="w+", dtype="f4", shape=SHAPE)
            output[:] = np.nan
    else:
        output = np.lib.format.open_memmap(part_path, mode="w+", dtype="f4", shape=SHAPE)
        output[:] = np.nan

    completed = load_resume_state(state_path, digest)
    latitude_blocks = list(range(0, SHAPE[1], args.lat_block_rows))
    depth_blocks = list(range(0, SHAPE[0], args.depth_block_layers))
    block_count = len(latitude_blocks) * len(depth_blocks)
    block_number = 0
    for k0 in depth_blocks:
        k1 = min(SHAPE[0], k0 + args.depth_block_layers)
        for j0 in latitude_blocks:
            block_number += 1
            j1 = min(SHAPE[1], j0 + args.lat_block_rows)
            block_key = f"{k0}:{k1},{j0}:{j1}"
            if block_key in completed:
                print(f"[annual-prho] {year} block {block_number}/{block_count} cached", flush=True)
                continue
            url, block_digest, _ = expression_url(year, k0, k1, j0, j1, depth)
            if block_digest != digest:
                raise RuntimeError("Annual expression changed between blocks")
            output[k0:k1, j0:j1, :] = read_block(
                url,
                k1 - k0,
                j1 - j0,
                args,
                f"{year} block {block_number}/{block_count}",
            )
            output.flush()
            completed.add(block_key)
            write_json(state_path, {
                "year": year,
                "days": total_days,
                "expression_sha256": digest,
                "completed_blocks": sorted(completed),
                "latitude_block_rows": args.lat_block_rows,
                "depth_block_layers": args.depth_block_layers,
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            })
            print(f"[annual-prho] {year} block {block_number}/{block_count}", flush=True)

    del output
    os.replace(part_path, final_path)
    state_path.unlink(missing_ok=True)
    data = np.load(final_path, mmap_mode="r", allow_pickle=False)
    return {
        "year": year,
        "status": "downloaded",
        "days": total_days,
        "bytes": final_path.stat().st_size,
        "finite_fraction": float(np.isfinite(data).mean()),
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }


def write_status(path: Path, rows: list[dict[str, object]]) -> None:
    fields = ["year", "status", "days", "bytes", "finite_fraction", "updated_at"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def download(args: argparse.Namespace) -> None:
    depth = read_depth_coordinate()
    state_root = args.output_root / "workers" / args.worker_name
    status_path = state_root / "annual_prho_status.csv"
    manifest_path = state_root / "annual_prho_manifest.json"
    manifest: dict[str, object] = {
        "status": "running",
        "source_dataset": f"{GDS_ROOT}/{DATASET}",
        "server_side_operation": "calendar-day-weighted sum of 12 monthly prho means per year",
        "period": {"start_year": args.start_year, "end_year": args.end_year},
        "native_shape": list(SHAPE),
        "worker_name": args.worker_name,
        "latitude_block_rows": args.lat_block_rows,
        "depth_block_layers": args.depth_block_layers,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    write_json(manifest_path, manifest)
    rows: list[dict[str, object]] = []
    for year in range(args.start_year, args.end_year + 1):
        row = download_year(year, args, depth)
        rows.append(row)
        write_status(status_path, rows)
        print(f"[annual-prho] {year} {row['status']}", flush=True)
    manifest["status"] = "complete"
    manifest["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    write_json(manifest_path, manifest)


def finalize(args: argparse.Namespace) -> None:
    depth = read_depth_coordinate()
    years = list(range(args.start_year, args.end_year + 1))
    paths = [args.output_root / "raw_annual" / f"{year:04d}" / "prho_annual_mean.npy" for year in years]
    missing = [str(path) for path in paths if not cache_is_complete(path)]
    if missing:
        raise FileNotFoundError("Annual prho inputs are incomplete: " + ", ".join(missing))

    climatology_dir = args.output_root / "climatology"
    climatology_dir.mkdir(parents=True, exist_ok=True)
    final_path = climatology_dir / "ofes2_prho_annual_mss_1993_2012.npy"
    part_path = final_path.with_suffix(".npy.part")
    output = np.lib.format.open_memmap(part_path, mode="w+", dtype="f4", shape=SHAPE)
    sources = [np.load(path, mmap_mode="r", allow_pickle=False) for path in paths]
    year_days = np.asarray([366 if calendar.isleap(year) else 365 for year in years], dtype="f8")
    for j0 in range(0, SHAPE[1], args.lat_block_rows):
        j1 = min(SHAPE[1], j0 + args.lat_block_rows)
        shape = (SHAPE[0], j1 - j0, SHAPE[2])
        weighted_sum = np.zeros(shape, dtype="f8")
        support_days = np.zeros(shape, dtype="f8")
        for source, days in zip(sources, year_days, strict=True):
            block = np.asarray(source[:, j0:j1, :], dtype="f4")
            valid = np.isfinite(block)
            weighted_sum[valid] += block[valid] * days
            support_days[valid] += days
        with np.errstate(invalid="ignore", divide="ignore"):
            output[:, j0:j1, :] = np.where(support_days > 0, weighted_sum / support_days, np.nan)
        output.flush()
        print(f"[annual-prho] climatology latitude rows {j0}:{j1}", flush=True)
    del output
    os.replace(part_path, final_path)
    result = np.load(final_path, mmap_mode="r", allow_pickle=False)
    np.savez_compressed(
        climatology_dir / "ofes2_prho_annual_mss_1993_2012_coordinates.npz",
        longitude=LON,
        latitude=LAT,
        depth=depth,
        years=np.asarray(years, dtype="i2"),
        year_days=year_days.astype("i2"),
    )
    write_json(climatology_dir / "ofes2_prho_annual_mss_1993_2012_manifest.json", {
        "status": "complete",
        "definition": "1993-2012 day-weighted mean of server-side day-weighted OFES2 annual prho fields",
        "source_dataset": f"{GDS_ROOT}/{DATASET}",
        "shape": list(SHAPE),
        "dtype": "float32",
        "years": years,
        "total_days": int(year_days.sum()),
        "finite_fraction": float(np.isfinite(result).mean()),
        "output": str(final_path),
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    })
    print(f"[annual-prho] climatology complete: {final_path}", flush=True)


def main() -> None:
    args = parse_args()
    if args.start_year > args.end_year or args.lat_block_rows < 1 or args.depth_block_layers < 1:
        raise ValueError("Invalid year range or block size")
    if args.finalize_only:
        finalize(args)
    else:
        download(args)


if __name__ == "__main__":
    main()
