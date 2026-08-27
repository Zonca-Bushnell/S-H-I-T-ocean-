from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable

import yaml


MAT_DATE_RE = re.compile(r"_(\d{8})T\d{4}Z\.mat$", re.IGNORECASE)


@dataclass(frozen=True)
class MatFile:
    path: Path
    date: date


def load_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def ensure_dirs(config: dict) -> None:
    for key, value in config.get("paths", {}).items():
        if key.endswith("_dir") or key in {"output_dir", "logs_dir"}:
            Path(value).mkdir(parents=True, exist_ok=True)


def parse_ymd(value: str | date | datetime) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return datetime.strptime(value, "%Y-%m-%d").date()


def iter_days(start: date, end: date) -> Iterable[date]:
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def parse_mat_date(path: Path) -> date:
    match = MAT_DATE_RE.search(path.name)
    if not match:
        raise ValueError(f"Cannot parse date from MAT filename: {path}")
    return datetime.strptime(match.group(1), "%Y%m%d").date()


def discover_mat_files(input_dir: str | Path, start: date, end: date) -> list[MatFile]:
    root = Path(input_dir)
    files: list[MatFile] = []
    for path in sorted(root.glob("*.mat")):
        try:
            day = parse_mat_date(path)
        except ValueError:
            continue
        if start <= day <= end:
            files.append(MatFile(path=path, date=day))
    found = {item.date for item in files}
    missing = [day.isoformat() for day in iter_days(start, end) if day not in found]
    if missing:
        raise FileNotFoundError("Missing MAT files for dates: " + ", ".join(missing[:10]))
    return files
