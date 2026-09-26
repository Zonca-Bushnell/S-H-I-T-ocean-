"""Build a day-weighted 1993--2012 annual OFES2 ``prho`` climatology.

The input is the resumable monthly cache produced by
``download_ofes2_monthly_prho``.  The calculation deliberately streams one
latitude stripe at a time: it never needs more than a small fraction of the
global three-dimensional monthly field in RAM.
"""
from __future__ import annotations

import argparse
import calendar
import json
import os
from pathlib import Path

import numpy as np

from .download_ofes2_monthly_prho import OUTPUT_ROOT, SHAPE, cache_is_complete


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--lat-block-rows", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def months() -> list[tuple[int, int]]:
    return [(year, month) for year in range(1993, 2013) for month in range(1, 13)]


def monthly_path(root: Path, year: int, month: int) -> Path:
    return root / "raw_monthly" / f"{year:04d}{month:02d}" / "prho.npy"


def atomic_save(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".part")
    with partial.open("wb") as handle:
        np.save(handle, values, allow_pickle=False)
    os.replace(partial, path)


def write_json(path: Path, payload: dict[str, object]) -> None:
    partial = path.with_suffix(path.suffix + ".part")
    partial.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(partial, path)


def main() -> None:
    args = parse_args()
    if args.lat_block_rows < 1:
        raise ValueError("--lat-block-rows must be positive")
    labels = months()
    missing = [f"{year:04d}{month:02d}" for year, month in labels if not cache_is_complete(monthly_path(args.output_root, year, month))]
    if missing:
        raise RuntimeError(f"Monthly prho cache is incomplete ({len(missing)} missing); first missing: {missing[:8]}")

    climatology_dir = args.output_root / "climatology"
    mean_path = climatology_dir / "ofes2_prho_annual_mss_1993_2012.npy"
    weight_path = climatology_dir / "ofes2_prho_annual_mss_valid_day_weight_1993_2012.npy"
    manifest_path = climatology_dir / "ofes2_prho_annual_mss_1993_2012_manifest.json"
    if mean_path.exists() and weight_path.exists() and not args.force:
        print(json.dumps({"status": "already_complete", "mean": str(mean_path), "weight": str(weight_path)}))
        return

    climatology_dir.mkdir(parents=True, exist_ok=True)
    mean_partial = mean_path.with_suffix(".npy.part")
    weight_partial = weight_path.with_suffix(".npy.part")
    if mean_partial.exists():
        mean_partial.unlink()
    if weight_partial.exists():
        weight_partial.unlink()
    mean = np.lib.format.open_memmap(mean_partial, mode="w+", dtype="f4", shape=SHAPE)
    weight = np.lib.format.open_memmap(weight_partial, mode="w+", dtype="u2", shape=SHAPE)
    sources = [(monthly_path(args.output_root, year, month), calendar.monthrange(year, month)[1]) for year, month in labels]
    total_days = int(sum(days for _, days in sources))

    for start in range(0, SHAPE[1], args.lat_block_rows):
        stop = min(SHAPE[1], start + args.lat_block_rows)
        numerator = np.zeros((SHAPE[0], stop - start, SHAPE[2]), dtype="f8")
        denominator = np.zeros((SHAPE[0], stop - start, SHAPE[2]), dtype="u2")
        for path, days in sources:
            field = np.load(path, mmap_mode="r", allow_pickle=False)[:, start:stop, :]
            valid = np.isfinite(field)
            numerator[valid] += field[valid] * days
            denominator[valid] += days
        block_mean = np.divide(numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator > 0)
        mean[:, start:stop, :] = block_mean.astype("f4")
        weight[:, start:stop, :] = denominator
        mean.flush()
        weight.flush()
        print(f"[prho-mss] latitude rows {start}:{stop}/{SHAPE[1]}", flush=True)

    del mean, weight
    os.replace(mean_partial, mean_path)
    os.replace(weight_partial, weight_path)
    data = np.load(mean_path, mmap_mode="r", allow_pickle=False)
    support = np.load(weight_path, mmap_mode="r", allow_pickle=False)
    manifest = {
        "status": "complete",
        "definition": "day-weighted annual mean of raw monthly OFES2 prho",
        "period": "1993-01 through 2012-12",
        "monthly_sample_count": len(labels),
        "calendar_day_weight_sum": total_days,
        "shape": list(SHAPE),
        "mean_path": str(mean_path),
        "valid_day_weight_path": str(weight_path),
        "finite_fraction": float(np.isfinite(data).mean()),
        "valid_day_weight_min": int(np.min(support)),
        "valid_day_weight_max": int(np.max(support)),
        "latitude_block_rows": args.lat_block_rows,
        "formula": "sum(days_in_month * prho_month where finite) / sum(days_in_month where finite)",
    }
    write_json(manifest_path, manifest)
    print(json.dumps({"status": "complete", "mean": str(mean_path), "manifest": str(manifest_path)}))


if __name__ == "__main__":
    main()
