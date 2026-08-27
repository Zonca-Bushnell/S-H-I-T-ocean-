from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..table_io import read_table
from .models import PHASE_ORDER


def load_event_index(composite_root: Path, shapes: list[str]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for shape in shapes:
        path = composite_root / f"lifecycle_composites_1993_2022_{shape}" / "lifecycle_composite_index.parquet"
        df = read_table(path).copy()
        df["shape_class"] = shape
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No lifecycle composite index found under {composite_root}")
    out = pd.concat(frames, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"]).dt.date.astype(str)
    return out[out["phase_name"].isin(PHASE_ORDER)].copy()


def load_completed_centers(path: Path, event_index: pd.DataFrame) -> pd.DataFrame:
    centers = read_table(path).copy()
    centers["date"] = pd.to_datetime(centers["date"]).dt.date.astype(str)
    keys = event_index[["track3d_id", "eddy3d_object_id", "date"]].drop_duplicates()
    merged = centers.merge(keys, on=["track3d_id", "eddy3d_object_id", "date"], how="inner")
    if merged.empty:
        raise ValueError("No completed centers matched lifecycle composite events.")
    return merged


def compute_completed_center_offsets(index: pd.DataFrame, centers: pd.DataFrame) -> pd.DataFrame:
    meta_cols = [
        "track3d_id",
        "eddy3d_object_id",
        "date",
        "shape_class",
        "polarity",
        "phase_index",
        "phase_name",
        "life_phase",
        "radius_m",
        "alpha_deg",
    ]
    data = centers.merge(index[meta_cols], on=["track3d_id", "eddy3d_object_id", "date"], how="inner", suffixes=("_center", "_event"))
    surf = (
        data.sort_values(["track3d_id", "date", "depth_index"])
        .groupby(["track3d_id", "eddy3d_object_id", "date"], as_index=False)
        .first()[["track3d_id", "eddy3d_object_id", "date", "longitude", "latitude"]]
        .rename(columns={"longitude": "lon0", "latitude": "lat0"})
    )
    data = data.merge(surf, on=["track3d_id", "eddy3d_object_id", "date"], how="left")
    lat0_rad = np.deg2rad(data["lat0"].astype(float).to_numpy())
    dx_km = (data["longitude"].astype(float).to_numpy() - data["lon0"].astype(float).to_numpy()) * 111.32 * np.cos(lat0_rad)
    dy_km = (data["latitude"].astype(float).to_numpy() - data["lat0"].astype(float).to_numpy()) * 110.574
    alpha = np.deg2rad(data["alpha_deg"].fillna(0.0).astype(float).to_numpy())
    x_aligned_km = dx_km * np.cos(alpha) + dy_km * np.sin(alpha)
    y_aligned_km = -dx_km * np.sin(alpha) + dy_km * np.cos(alpha)
    radius_col = "radius_m_event" if "radius_m_event" in data.columns else "radius_m"
    radius_km = np.maximum(data[radius_col].astype(float).to_numpy() / 1000.0, 1e-6)
    data["center_x_R"] = x_aligned_km / radius_km
    data["center_y_R"] = y_aligned_km / radius_km
    data["TD_star"] = np.hypot(data["center_x_R"], data["center_y_R"])
    data["center_method_is_detected"] = data["center_is_detected"].astype(bool)
    return data[
        [
            "shape_class",
            "polarity",
            "phase_index",
            "phase_name",
            "depth_index",
            "depth_m",
            "center_x_R",
            "center_y_R",
            "TD_star",
            "center_method",
            "center_method_is_detected",
        ]
    ].copy()


def estimate_observed_growth(center_events: pd.DataFrame) -> pd.DataFrame:
    grouped = center_events.groupby(["shape_class", "polarity", "phase_name", "phase_index", "depth_index", "depth_m"], dropna=False)
    phase_summary = grouped.agg(
        center_x_R_median=("center_x_R", "median"),
        center_y_R_median=("center_y_R", "median"),
        TD_star_median=("TD_star", "median"),
    ).reset_index()
    rows: list[dict] = []
    for keys, grp in phase_summary.groupby(["shape_class", "polarity", "depth_index", "depth_m"], dropna=False):
        shape, polarity, depth_index, depth_m = keys
        g = grp.sort_values("phase_index")
        if g["phase_index"].nunique() < 2:
            continue
        slope_td = _linear_slope(g["phase_index"].to_numpy(dtype="f8"), g["TD_star_median"].to_numpy(dtype="f8"))
        slope_x = _linear_slope(g["phase_index"].to_numpy(dtype="f8"), g["center_x_R_median"].to_numpy(dtype="f8"))
        slope_y = _linear_slope(g["phase_index"].to_numpy(dtype="f8"), g["center_y_R_median"].to_numpy(dtype="f8"))
        rows.append(
            {
                "shape": shape,
                "polarity": polarity,
                "depth_index": int(depth_index),
                "depth_m": float(depth_m),
                "obs_TD_star_growth_per_phase": slope_td,
                "obs_center_x_R_speed_per_phase": slope_x,
                "obs_center_y_R_speed_per_phase": slope_y,
                "obs_center_speed_R_per_phase": float(np.hypot(slope_x, slope_y)),
            }
        )
    return pd.DataFrame(rows)


def _linear_slope(x: np.ndarray, y: np.ndarray) -> float:
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 2:
        return np.nan
    xx = x[valid]
    yy = y[valid]
    if np.nanstd(xx) <= 0:
        return np.nan
    return float(np.polyfit(xx, yy, 1)[0])
