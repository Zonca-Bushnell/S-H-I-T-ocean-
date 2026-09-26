"""Shared geographic classifications used by QC and composites."""
from __future__ import annotations

import numpy as np


def _number(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _coordinate(row, names: tuple[str, ...]) -> float:
    for name in names:
        value = _number(row.get(name, np.nan))
        if np.isfinite(value):
            return value
    return float("nan")


def is_open_ocean(row) -> bool:
    """Return whether an object uses relaxed open-ocean geometry thresholds."""
    lon = _coordinate(row, ("center_lon_refined", "center_lon", "ssh_contour_center_lon", "seed_lon"))
    lat = _coordinate(row, ("center_lat_refined", "center_lat", "ssh_contour_center_lat", "seed_lat"))
    if not (np.isfinite(lon) and np.isfinite(lat)):
        return False
    jet_flag = row.get("jet_meander_flag", False)
    if isinstance(jet_flag, str):
        jet_flag = jet_flag.strip().lower() in {"1", "true", "yes", "y"}
    if bool(jet_flag) or -62.0 <= lat <= -40.0:
        return False
    boundary_boxes = (
        (120.0, 160.0, 20.0, 45.0),
        (260.0, 320.0, 20.0, 50.0),
        (225.0, 260.0, 15.0, 40.0),
        (330.0, 360.0, 15.0, 40.0),
        (0.0, 50.0, -50.0, -15.0),
        (140.0, 185.0, -50.0, -15.0),
        (285.0, 335.0, -50.0, -15.0),
    )
    return not any(
        lon_min <= lon <= lon_max and lat_min <= lat <= lat_max
        for lon_min, lon_max, lat_min, lat_max in boundary_boxes
    )
