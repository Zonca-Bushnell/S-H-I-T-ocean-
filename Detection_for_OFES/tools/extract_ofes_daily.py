from __future__ import annotations

import csv
import json
from datetime import date
from pathlib import Path

from ..ofes_io import (
    archive_candidates,
    ctl_path,
    data_path,
    expected_dta_bytes,
    parse_ctl,
    prefetch_daily_files,
)


def extract_w_prho(root: Path, metadata_dir: Path, days: list[date], workers: int) -> None:
    rows = []
    for variable in ["w", "prho"]:
        meta = parse_ctl(ctl_path(root, variable))
        expected_bytes = expected_dta_bytes(meta)
        try:
            prefetch_daily_files(root, variable, days, workers=max(1, workers))
        except FileNotFoundError as exc:
            print(f"[extract-warning] {variable}: {exc}", flush=True)
        for day in days:
            path = data_path(root, variable, day)
            archives = archive_candidates(root, variable, day)
            complete = path.exists() and path.stat().st_size == expected_bytes
            row = {
                "variable": variable,
                "date": day.isoformat(),
                "source_archive": str(archives[0]) if archives else "",
                "target_path": str(path),
                "expected_bytes": expected_bytes,
                "actual_bytes": path.stat().st_size if path.exists() else 0,
                "complete": bool(complete),
            }
            rows.append(row)
            print(f"[extract-ready] {variable} {day.isoformat()} complete={complete}", flush=True)
    _write_csv(metadata_dir / "w_prho_extract_manifest_jan01_jan19.csv", rows)
    _write_json(metadata_dir / "w_prho_extract_manifest_jan01_jan19.json", rows)
    missing = [row for row in rows if not row["complete"]]
    if missing:
        raise FileNotFoundError("w/prho extraction incomplete: " + ", ".join(f"{r['variable']}:{r['date']}" for r in missing))


def _write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=True), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
