"""Materialize optional tracking from canonical daily vertical centers."""
from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from Detection_for_OFES.core.tracking import track_object_days


def _dates(start: date, end: date):
    for offset in range((end - start).days + 1):
        yield start + timedelta(days=offset)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vertical-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--shift-cells", type=int, default=1)
    parser.add_argument("--continuous-score", type=float, default=0.25)
    parser.add_argument("--split-merge-score", type=float, default=0.75)
    args = parser.parse_args()
    frames = []
    for day in _dates(date.fromisoformat(args.start), date.fromisoformat(args.end)):
        path = args.vertical_root / "raw_detection" / "daily_runs" / day.strftime("%Y%m%d") / "centers_hua_style.csv"
        frames.append(pd.read_csv(path, low_memory=False))
    centers = pd.concat(frames, ignore_index=True)
    objects, edges, events = track_object_days(
        centers, shift_cells=args.shift_cells, continuous_score=args.continuous_score,
        split_merge_score=args.split_merge_score,
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    objects.to_csv(args.output_root / "tracked_object_days.csv", index=False)
    edges.to_csv(args.output_root / "tracking_edges.csv", index=False)
    events.to_csv(args.output_root / "tracking_events.csv", index=False)


if __name__ == "__main__":
    main()
