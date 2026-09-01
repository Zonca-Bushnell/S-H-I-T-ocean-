from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrackCacheStatus:
    path: Path
    exists: bool
    usable: bool
    reason: str


def expected_track_cache_path(me_liutex_root: Path) -> Path:
    return me_liutex_root / "ep_track_cache" / "track_velocity_accumulators.npz"


def inspect_track_cache(me_liutex_root: Path) -> TrackCacheStatus:
    path = expected_track_cache_path(me_liutex_root)
    if not path.exists():
        return TrackCacheStatus(
            path=path,
            exists=False,
            usable=False,
            reason=(
                "missing track-level velocity accumulator; strict track bootstrap cannot be "
                "computed from the final representative mean field alone"
            ),
        )
    try:
        import numpy as np

        data = np.load(path, allow_pickle=True)
        required = {"track_ids", "polarities", "tau_grid", "depth", "radial", "theta", "sum_u", "sum_v", "count"}
        missing = sorted(required - set(data.files))
        if missing:
            return TrackCacheStatus(path=path, exists=True, usable=False, reason=f"missing arrays: {missing}")
        return TrackCacheStatus(path=path, exists=True, usable=True, reason="ready")
    except Exception as exc:
        return TrackCacheStatus(path=path, exists=True, usable=False, reason=f"cannot read cache: {exc}")


def write_track_cache_manifest(output_dir: Path, status: TrackCacheStatus) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "track_bootstrap_cache_status.json"
    path.write_text(
        json.dumps(
            {
                "track_cache_path": str(status.path),
                "exists": bool(status.exists),
                "usable": bool(status.usable),
                "reason": status.reason,
                "strict_bootstrap_policy": (
                    "Bootstrap confidence intervals are only physically valid when they are "
                    "computed by resampling track-level velocity accumulators and recomputing EP fields."
                ),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return path

