"""Build the day-weighted 1993-2012 annual MSS directly from OFES eta."""

from __future__ import annotations

import argparse
import calendar
import json
from pathlib import Path

import numpy as np

from Detection_for_OFES.ofes_io import ctl_path, parse_ctl


DEFAULT_CACHE_ROOT = Path(
    r"E:\DATA\01_Eddy_correspond\02_OFES\ofes2_monthly_climatology_1993_2012"
)
DEFAULT_OUTPUT = DEFAULT_CACHE_ROOT / "climatology" / "ofes2_eta_annual_mss_1993_2012.npz"
OFES_ROOT = Path(r"F:\OFES\external_OFES2")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--ofes-root", type=Path, default=OFES_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--start-year", type=int, default=1993)
    parser.add_argument("--end-year", type=int, default=2012)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.overwrite:
        print(json.dumps({"status": "existing", "output": str(args.output)}, indent=2), flush=True)
        return

    meta = parse_ctl(ctl_path(args.ofes_root, "eta"))
    shape = (meta.y.values.size, meta.x.values.size)
    weighted_sum = np.zeros(shape, dtype="f8")
    valid_day_weight = np.zeros(shape, dtype="u2")
    months: list[str] = []

    for year in range(args.start_year, args.end_year + 1):
        for month in range(1, 13):
            label = f"{year:04d}{month:02d}"
            path = args.cache_root / "raw_monthly" / label / "eta.npy"
            if not path.exists():
                raise FileNotFoundError(path)
            eta = np.load(path, mmap_mode="r")
            if eta.shape != shape:
                raise RuntimeError(f"Unexpected eta shape for {label}: {eta.shape} != {shape}")
            days = calendar.monthrange(year, month)[1]
            field = np.asarray(eta, dtype="f8")
            valid = np.isfinite(field) & (np.abs(field) < 1.0e4)
            weighted_sum[valid] += field[valid] * days
            valid_day_weight[valid] += days
            months.append(label)
            print(f"[eta-mss] {label} days={days}", flush=True)

    mss = np.full(shape, np.nan, dtype="f4")
    valid = valid_day_weight > 0
    mss[valid] = (weighted_sum[valid] / valid_day_weight[valid]).astype("f4")
    expected_days = sum(
        calendar.monthrange(year, month)[1]
        for year in range(args.start_year, args.end_year + 1)
        for month in range(1, 13)
    )
    if len(months) != 240 or expected_days != 7305:
        raise RuntimeError(f"Expected 240 months and 7305 days, got {len(months)} and {expected_days}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".partial.npz")
    np.savez_compressed(
        temporary,
        longitude=np.asarray(meta.x.values, dtype="f8"),
        latitude=np.asarray(meta.y.values, dtype="f8"),
        eta_annual_mss_cm=mss,
        valid_day_weight=valid_day_weight,
        months_used=np.asarray(months),
        definition=np.asarray("day-weighted 1993-2012 annual mean of native OFES eta"),
    )
    temporary.replace(args.output)
    manifest = {
        "status": "complete",
        "output": str(args.output),
        "definition": "MSS_eta = day-weighted mean of monthly native OFES eta; no pair correction",
        "months": len(months),
        "first_month": months[0],
        "last_month": months[-1],
        "maximum_valid_day_weight": int(valid_day_weight.max()),
        "finite_fraction": float(np.isfinite(mss).mean()),
    }
    args.output.with_suffix(".json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
