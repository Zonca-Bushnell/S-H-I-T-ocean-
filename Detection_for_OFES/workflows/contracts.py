"""Small, explicit stage contracts used for date-level resume decisions."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def file_identity(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {"path": str(path), "size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}


def fingerprint(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def status_path(manifest_root: Path, stage: str, token: str) -> Path:
    return manifest_root / "stages" / stage / f"{token}.json"


def read_status(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def is_current(path: Path, contract: dict[str, Any], outputs: list[Path]) -> bool:
    saved = read_status(path)
    return bool(
        saved and saved.get("fingerprint") == contract["fingerprint"]
        and all(output.exists() and output.stat().st_size > 0 for output in outputs)
    )


def write_status(path: Path, contract: dict[str, Any], outputs: list[Path]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {**contract, "outputs": [str(output) for output in outputs]}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
