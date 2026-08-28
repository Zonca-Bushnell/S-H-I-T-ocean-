from __future__ import annotations

import pandas as pd
import numpy as np

from ..validation.isopycnal_projection import build_isopycnal_surfaces
from ..validation.isopycnal_pv import compute_isopycnal_control_volume_pv
from .common import thermal_wind_velocity

from .models import ForecastState


def diagnose_forecast_pv(
    state: ForecastState,
    sigma_pred: np.ndarray,
    adt_pred: np.ndarray,
    shape: str,
    polarity: str,
    phase: str,
    phase_index: int,
    model: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    sigma_total = state.sigma_clim[:, None, None] + sigma_pred
    surfaces = build_isopycnal_surfaces(sigma_total, state.depth, rho_levels=None, smooth_sigma_grid=1.0)
    u_pred, v_pred = thermal_wind_velocity(sigma_pred, adt_pred, state.depth, state.x, state.y, state.radius_m, state.f0)
    pv = compute_isopycnal_control_volume_pv(
        u_pred,
        v_pred,
        sigma_total,
        surfaces.rho_levels,
        state.depth,
        state.x,
        state.y,
        state.radius_m,
        state.f0,
        {"shape": shape, "polarity": polarity, "phase": phase, "phase_index": phase_index, "model": model},
    )
    return u_pred, v_pred, surfaces.z_anom_m, pv.table


def pv_growth(pv_df: pd.DataFrame) -> pd.DataFrame:
    columns = ["shape", "polarity", "model", "rho_bin", "TD_PV_growth_per_phase"]
    if pv_df.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict] = []
    for keys, group in pv_df.groupby(["shape", "polarity", "model", "rho_bin"], dropna=False):
        shape, polarity, model, rho_bin = keys
        g = group.sort_values("phase_index")
        good = np.isfinite(g["phase_index"]) & np.isfinite(g["TD_PV_star"])
        if np.count_nonzero(good) < 2:
            continue
        slope = float(np.polyfit(g.loc[good, "phase_index"].to_numpy(dtype="f8"), g.loc[good, "TD_PV_star"].to_numpy(dtype="f8"), 1)[0])
        rows.append({"shape": shape, "polarity": polarity, "model": model, "rho_bin": int(rho_bin), "TD_PV_growth_per_phase": slope})
    return pd.DataFrame(rows, columns=columns)


def velocity_centroid_growth(centroid_df: pd.DataFrame) -> pd.DataFrame:
    columns = ["shape", "polarity", "model", "depth_index", "TD_velocity_growth_per_phase"]
    rows: list[dict] = []
    if centroid_df.empty:
        return pd.DataFrame(columns=columns)
    for (shape, polarity, model, depth_index), g in centroid_df.groupby(["shape", "polarity", "model", "depth_index"], dropna=False):
        good = np.isfinite(g["phase_index"]) & np.isfinite(g["TD_velocity_star"])
        if np.count_nonzero(good) < 2:
            continue
        slope = float(np.polyfit(g.loc[good, "phase_index"].to_numpy(dtype="f8"), g.loc[good, "TD_velocity_star"].to_numpy(dtype="f8"), 1)[0])
        rows.append({"shape": shape, "polarity": polarity, "model": model, "depth_index": int(depth_index), "TD_velocity_growth_per_phase": slope})
    return pd.DataFrame(rows, columns=columns)
