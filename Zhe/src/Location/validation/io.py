from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "config_3d_cmems.yaml"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "kuroshio_cmems_3d"
VALIDATION_ROOT = OUTPUT_ROOT / "theory_validation"


def resolve_path(value: str | Path, base: Path | None = None) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (base or PROJECT_ROOT) / path


def split_csv(value: str | None, default: Iterable[str] | None = None) -> list[str]:
    if value is None or str(value).strip() == "":
        return list(default or [])
    return [item.strip() for item in str(value).split(",") if item.strip()]


def write_run_metadata(output_dir: Path, command: str, args: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "project_root": str(PROJECT_ROOT),
        "command": command,
        "args": args,
    }
    (output_dir / "run_metadata.yaml").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
