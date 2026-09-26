"""Small, explicit stage contracts used for date-level resume decisions."""
from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any


def file_identity(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {"path": str(path), "size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}


def fingerprint(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@lru_cache(maxsize=1)
def code_fingerprint() -> str:
    """Hash maintained Python and launcher source, excluding caches and history."""
    package_root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    paths = sorted(package_root.rglob("*.py")) + [package_root / "start_ofes_default_run.ps1"]
    for path in paths:
        if not path.exists() or "legacy" in path.parts or "tools" in path.parts or "tests" in path.parts:
            continue
        digest.update(path.relative_to(package_root).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


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
