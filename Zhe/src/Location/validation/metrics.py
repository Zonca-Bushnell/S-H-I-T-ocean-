from __future__ import annotations

import numpy as np


def corr(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if int(mask.sum()) < 3:
        return np.nan
    aa = np.asarray(a[mask], dtype="f8")
    bb = np.asarray(b[mask], dtype="f8")
    if np.nanstd(aa) == 0 or np.nanstd(bb) == 0:
        return np.nan
    return float(np.corrcoef(aa, bb)[0, 1])


def vector_skill(u: np.ndarray, v: np.ndarray, u_obs: np.ndarray, v_obs: np.ndarray) -> dict:
    good = np.isfinite(u) & np.isfinite(v) & np.isfinite(u_obs) & np.isfinite(v_obs)
    n = int(good.sum())
    if n < 20:
        return {"n_grid": n, "corr_u": np.nan, "corr_v": np.nan, "r2_vector": np.nan, "rmse_vector": np.nan}
    up = np.asarray(u[good], dtype="f8")
    vp = np.asarray(v[good], dtype="f8")
    uo = np.asarray(u_obs[good], dtype="f8")
    vo = np.asarray(v_obs[good], dtype="f8")
    err = np.nanmean((up - uo) ** 2 + (vp - vo) ** 2)
    denom = np.nanmean(uo * uo + vo * vo)
    return {
        "n_grid": n,
        "corr_u": corr(up, uo),
        "corr_v": corr(vp, vo),
        "r2_vector": float(1.0 - err / denom) if denom > 0 else np.nan,
        "rmse_vector": float(np.sqrt(err)),
    }

