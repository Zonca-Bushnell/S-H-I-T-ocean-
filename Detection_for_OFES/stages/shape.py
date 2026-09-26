"""Materialize optional shape classes from canonical accepted layers."""
from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from Detection_for_OFES.core.shape import classify_object_days


def _dates(start: date, end: date):
    for offset in range((end - start).days + 1):
        yield start + timedelta(days=offset)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vertical-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--tracking-table", type=Path)
    parser.add_argument("--min-layers", type=int, required=True)
    parser.add_argument("--upright-quantile", type=float, required=True)
    parser.add_argument("--upright-fallback", type=float, required=True)
    parser.add_argument("--coherent-monotonic-ratio", type=float, required=True)
    parser.add_argument("--coherent-mean-turn-deg", type=float, required=True)
    parser.add_argument("--complex-max-turn-deg", type=float, required=True)
    parser.add_argument("--complex-monotonic-ratio", type=float, required=True)
    args = parser.parse_args()
    frames = []
    for day in _dates(date.fromisoformat(args.start), date.fromisoformat(args.end)):
        path = args.vertical_root / "raw_detection" / "daily_runs" / day.strftime("%Y%m%d") / "centers_hua_style.csv"
        frames.append(pd.read_csv(path, low_memory=False))
    tracks = pd.read_csv(args.tracking_table, low_memory=False) if args.tracking_table and args.tracking_table.exists() else None
    metrics, summary = classify_object_days(
        pd.concat(frames, ignore_index=True), tracks=tracks,
        min_layers=args.min_layers, upright_quantile=args.upright_quantile,
        upright_fallback=args.upright_fallback,
        coherent_monotonic_ratio=args.coherent_monotonic_ratio,
        coherent_mean_turn_deg=args.coherent_mean_turn_deg,
        complex_max_turn_deg=args.complex_max_turn_deg,
        complex_monotonic_ratio=args.complex_monotonic_ratio,
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.output_root / "object_day_shape.csv", index=False)
    summary.to_csv(args.output_root / "track_shape_summary.csv", index=False)


if __name__ == "__main__":
    main()
