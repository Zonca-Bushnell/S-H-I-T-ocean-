from __future__ import annotations

import numpy as np


def depth_weights(depth: np.ndarray) -> np.ndarray:
    z = np.asarray(depth, dtype="f8")
    if z.size == 1:
        return np.ones(1, dtype="f8")
    edges = np.empty(z.size + 1, dtype="f8")
    edges[1:-1] = 0.5 * (z[:-1] + z[1:])
    edges[0] = z[0] - 0.5 * (z[1] - z[0])
    edges[-1] = z[-1] + 0.5 * (z[-1] - z[-2])
    w = np.diff(edges)
    w = np.where(np.isfinite(w) & (w > 0), w, np.nanmedian(w[w > 0]))
    return w / np.nansum(w)


def barotropic_baroclinic(u: np.ndarray, v: np.ndarray, depth: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    w = depth_weights(depth)[:, None, None]
    u_bt = np.nansum(np.asarray(u, dtype="f8") * w, axis=0, keepdims=True)
    v_bt = np.nansum(np.asarray(v, dtype="f8") * w, axis=0, keepdims=True)
    u_bt3 = np.broadcast_to(u_bt, u.shape).copy()
    v_bt3 = np.broadcast_to(v_bt, v.shape).copy()
    return u_bt3, v_bt3, np.asarray(u, dtype="f8") - u_bt3, np.asarray(v, dtype="f8") - v_bt3


def deep_top_ratio(field: np.ndarray, top_n: int = 8, deep_n: int = 12) -> float:
    arr = np.asarray(field, dtype="f8")
    if arr.ndim == 4:
        arr = np.sqrt(arr[0] * arr[0] + arr[1] * arr[1])
    prof = np.nanpercentile(np.abs(arr), 95, axis=tuple(range(1, arr.ndim)))
    top = float(np.nanmean(prof[:top_n]))
    deep = float(np.nanmean(prof[-deep_n:]))
    return deep / top if np.isfinite(top) and top > 0 else np.nan

