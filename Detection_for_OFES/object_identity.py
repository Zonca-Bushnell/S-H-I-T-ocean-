"""Stable OFES object identifiers across surface, vertical, and composite tables."""
from __future__ import annotations

import hashlib

import pandas as pd


def stable_numeric_object_id(hua_object_id: str) -> int:
    """Derive a deterministic positive 63-bit ID without row-order dependence."""
    digest = hashlib.blake2b(str(hua_object_id).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") & ((1 << 63) - 1)


def attach_source_identity(frame: pd.DataFrame) -> pd.DataFrame:
    """Preserve the immutable surface ID beside any derived numeric IDs."""
    if "hua_object_id" not in frame.columns:
        raise ValueError("Expected hua_object_id")
    out = frame.copy()
    out["hua_object_id"] = out["hua_object_id"].astype(str)
    out["source_hua_object_id"] = out["hua_object_id"]
    out["stable_object_id"] = out["hua_object_id"].map(stable_numeric_object_id).astype("int64")
    return out
