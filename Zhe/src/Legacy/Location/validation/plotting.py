from __future__ import annotations

import numpy as np


def p2_p98_limits(array: np.ndarray) -> tuple[float, float]:
    vals = np.asarray(array, dtype="f8").ravel()
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return -1.0, 1.0
    vmin, vmax = np.nanpercentile(vals, [2, 98])
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        center = float(np.nanmedian(vals)) if vals.size else 0.0
        spread = float(np.nanstd(vals))
        if not np.isfinite(spread) or spread <= 0:
            spread = max(abs(center), 1.0) * 0.05
        vmin, vmax = center - spread, center + spread
    return float(vmin), float(vmax)


def contour_levels(array: np.ndarray, vmin: float, vmax: float, count: int = 7) -> np.ndarray:
    vals = np.asarray(array, dtype="f8")
    if not np.any(np.isfinite(vals)) or not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        return np.asarray([], dtype="f8")
    amin = float(np.nanmin(vals))
    amax = float(np.nanmax(vals))
    levels = np.linspace(vmin, vmax, count)
    return np.unique(levels[(levels > amin) & (levels < amax)])

