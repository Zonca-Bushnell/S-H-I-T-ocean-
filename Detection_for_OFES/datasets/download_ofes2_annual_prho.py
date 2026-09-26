"""Download server-side, day-weighted OFES2 annual prho fields and their climatology."""

from __future__ import annotations

import argparse
import calendar
import csv
import hashlib
import json
import os
import random
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
    parser.add_argument(
        "--block-retries",
        type=int,
        default=5,
        help="Maximum retries for one request leaf; zero is forbidden because retries are bounded.",
    )
    parser.add_argument("--retry-seconds", type=float, default=15.0)
    parser.add_argument("--retry-max-seconds", type=float, default=300.0)
    parser.add_argument("--retry-jitter-seconds", type=float, default=7.0)
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
    i0: int = 0,
    i1: int = SHAPE[2],
) -> tuple[str, str, int]:
    expression, total_days = annual_expression(year)
    domain = (
        f"{LON[i0]:.2f}:{LON[i1 - 1]:.2f},"
        f"{LAT[j0]:.2f}:{LAT[j1 - 1]:.2f},"
        f"{depth[k0]:.6f}:{depth[k1 - 1]:.6f},"
        f"00Z01JAN{year}:00Z01JAN{year}"
    )
    url = f"{GDS_ROOT}/_expr_{{{DATASET}}}{{{expression}}}{{{domain}}}"
    digest = hashlib.sha256(expression.encode("ascii")).hexdigest()
    return url, digest, total_days


def load_resume_state(
    path: Path,
    digest: str,
    latitude_block_rows: int,
    depth_block_layers: int,
    expected_block_keys: set[str],
) -> set[str]:
    if not path.exists():
        return set()
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return set()
    if (
        state.get("expression_sha256") != digest
        or state.get("latitude_block_rows") != latitude_block_rows
        or state.get("depth_block_layers") != depth_block_layers
    ):
        return set()
    # Older interrupted runs could mix block geometries in one state file.
    # Only retain entries that exactly match this run's block lattice.
    return {str(value) for value in state.get("completed_blocks", [])} & expected_block_keys


def retry_delay(args: argparse.Namespace, retry_number: int) -> float:
    base = args.retry_seconds * (2 ** min(retry_number - 1, 5))
    return min(args.retry_max_seconds, base) + random.uniform(0.0, args.retry_jitter_seconds)


class ResponseShapeMismatch(RuntimeError):
    """The server returned a different array shape for the requested domain."""


def split_ranges(k0: int, k1: int, j0: int, j1: int, i0: int, i1: int) -> list[tuple[int, int, int, int, int, int]]:
    sizes = [(k1 - k0, "k"), (j1 - j0, "j"), (i1 - i0, "i")]
    axis = max(sizes)[1]
    if max(value for value, _ in sizes) <= 1:
        return []
    if axis == "k":
        mid = (k0 + k1) // 2
        return [(k0, mid, j0, j1, i0, i1), (mid, k1, j0, j1, i0, i1)]
    if axis == "j":
        mid = (j0 + j1) // 2
        return [(k0, k1, j0, mid, i0, i1), (k0, k1, mid, j1, i0, i1)]
    mid = (i0 + i1) // 2
    return [(k0, k1, j0, j1, i0, mid), (k0, k1, j0, j1, mid, i1)]


def read_block(
    url: str,
    expected_layers: int,
    expected_rows: int,
    args: argparse.Namespace,
    label: str,
    expected_columns: int = SHAPE[2],
) -> np.ndarray:
    expected = (1, expected_layers, expected_rows, expected_columns)
    retry_number = 0
    while True:
        try:
            dataset = open_url(url, protocol="dap2")
            raw = np.ma.asarray(dataset["result"][:].data).filled(np.nan)
            values = np.asarray(raw, dtype="f4")
            if values.ndim == 4 and values.shape[0] == 1 and all(
                actual >= required
                for actual, required in zip(values.shape[1:], expected[1:], strict=True)
            ):
                if values.shape != expected:
                    print(f"[annual-prho] trim {label}: returned {values.shape}, using {expected}", flush=True)
                values = values[:, :expected_layers, :expected_rows, :expected_columns]
            else:
                raise ResponseShapeMismatch(f"unexpected result shape {values.shape}, expected {expected}")
            values = values[0]
            invalid = ~np.isfinite(values) | (np.abs(values) > FILL_ABS_LIMIT)
            if invalid.any():
                values[invalid] = np.nan
            return values
        except Exception as exc:
            if isinstance(exc, ResponseShapeMismatch):
                raise
            retry_number += 1
            if retry_number > args.block_retries:
                raise RuntimeError(f"{label} failed after {retry_number} retries") from exc
            delay = retry_delay(args, retry_number)
            limit = str(args.block_retries)
            print(
                f"[annual-prho] retry {label} attempt={retry_number}/{limit} "
                f"in {delay:.0f}s: {exc}",
                flush=True,
            )
            time.sleep(delay)


def read_adaptive_block(
    year: int,
    k0: int,
    k1: int,
    j0: int,
    j1: int,
    i0: int,
    i1: int,
    depth: np.ndarray,
    args: argparse.Namespace,
    label: str,
) -> np.ndarray:
    url, _, _ = expression_url(year, k0, k1, j0, j1, depth, i0, i1)
    try:
        return read_block(
            url, k1 - k0, j1 - j0, args,
            label + f" [{k0}:{k1},{j0}:{j1},{i0}:{i1}]",
            expected_columns=i1 - i0,
        )
    except Exception as exc:
        pieces = split_ranges(k0, k1, j0, j1, i0, i1)
        if not pieces:
            raise
        print(f"[annual-prho] split {label}: {exc}", flush=True)
        chunks = [
            read_adaptive_block(year, *piece, depth, args, label + "/split")
            for piece in pieces
        ]
        axis = 0 if k1 - k0 == max(k1 - k0, j1 - j0, i1 - i0) else 1
        if i1 - i0 == max(k1 - k0, j1 - j0, i1 - i0):
            axis = 2
        elif j1 - j0 == max(k1 - k0, j1 - j0, i1 - i0):
            axis = 2
        return np.concatenate(chunks, axis=axis)


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

    latitude_blocks = list(range(0, SHAPE[1], args.lat_block_rows))
    depth_blocks = list(range(0, SHAPE[0], args.depth_block_layers))
    expected_block_keys = {
        f"{k0}:{min(k0 + args.depth_block_layers, SHAPE[0])},{j0}:{min(j0 + args.lat_block_rows, SHAPE[1])}"
        for k0 in depth_blocks for j0 in latitude_blocks
    }
    completed = load_resume_state(
        state_path,
        digest,
        args.lat_block_rows,
        args.depth_block_layers,
        expected_block_keys,
    )
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
            output[k0:k1, j0:j1, :] = read_adaptive_block(
                year, k0, k1, j0, j1, 0, SHAPE[2], depth, args,
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
    if args.block_retries < 1 or args.retry_seconds <= 0 or args.retry_max_seconds < args.retry_seconds or args.retry_jitter_seconds < 0:
        raise ValueError("Invalid retry configuration")
    if args.finalize_only:
        finalize(args)
    else:
        download(args)


if __name__ == "__main__":
    main()
