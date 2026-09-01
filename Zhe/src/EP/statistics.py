from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


EP_FIELDS = (
    "F_n_classic",
    "F_z_ordinary",
    "F_z_tilted",
    "F_z_tilt_correction",
    "divF_classic",
    "divF_tilted",
    "divF_curved_tube_qg_approx",
    "pv_flux_proxy",
)


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if int(mask.sum()) < 3:
        return float("nan")
    if np.nanstd(a[mask]) == 0 or np.nanstd(b[mask]) == 0:
        return float("nan")
    return float(np.corrcoef(a[mask], b[mask])[0, 1])


def lifecycle_summary(profiles: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    group_cols = ["shape", "axis_source", "orientation", "buoyancy_source", "polarity", "tau"]
    for keys, part in profiles.groupby(group_cols, dropna=False):
        core = part[part["radius_over_R"].astype(float) <= 1.5]
        row = dict(zip(group_cols, keys))
        row["rows"] = int(len(part))
        row["core_rows"] = int(len(core))
        for col in ("n_objects", "n_tracks"):
            row[col] = int(np.nanmax(part[col])) if col in part and len(part) else 0
        row["finite_divF_tilted_fraction"] = float(np.isfinite(part["divF_tilted"]).mean())
        row["finite_pv_flux_fraction"] = float(np.isfinite(part["pv_flux_proxy"]).mean())
        row["median_axis_tilt_km"] = float(np.nanmedian(part["axis_tilt_km"]))
        ordinary = core["F_z_ordinary"].to_numpy(float)
        correction = core["F_z_tilt_correction"].to_numpy(float)
        row["median_abs_tilt_correction_over_ordinary"] = float(
            np.nanmedian(np.abs(correction)) / (np.nanmedian(np.abs(ordinary)) + 1e-30)
        )
        div = core["divF_tilted"].to_numpy(float)
        pv = core["pv_flux_proxy"].to_numpy(float)
        row["divF_pv_flux_corr_core"] = _safe_corr(div, pv)
        row["divF_pv_flux_rmse_core"] = float(np.sqrt(np.nanmean((div - pv) ** 2)))
        for col in (
            "metric_valid_fraction",
            "epsilon_curvature",
            "jacobian_min",
            "jacobian_max",
        ):
            if col in core:
                row[f"{col}_median"] = float(np.nanmedian(core[col].to_numpy(float)))
        rows.append(row)
    return pd.DataFrame(rows)


def metric_audit(profiles: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["shape", "axis_source", "orientation", "buoyancy_source", "polarity", "tau", "depth_m"]
    rows: list[dict[str, object]] = []
    for keys, part in profiles.groupby(group_cols, dropna=False):
        core = part[part["radius_over_R"].astype(float) <= 1.5]
        if core.empty:
            continue
        row = dict(zip(group_cols, keys))
        row["epsilon_curvature_median"] = float(np.nanmedian(np.abs(core["epsilon_curvature"])))
        row["epsilon_curvature_p90"] = float(np.nanpercentile(np.abs(core["epsilon_curvature"]), 90))
        row["jacobian_min"] = float(np.nanmin(core["jacobian_min"]))
        row["jacobian_max"] = float(np.nanmax(core["jacobian_max"]))
        row["metric_valid_fraction"] = float(np.nanmean(core["metric_valid_fraction"]))
        rows.append(row)
    return pd.DataFrame(rows)


def write_placeholder_strict_ci(
    path: Path,
    *,
    bootstrap_samples: int,
    bootstrap_unit: str,
    reason: str,
) -> None:
    rows = [
        {
            "field": field,
            "bootstrap_samples_requested": int(bootstrap_samples),
            "bootstrap_unit": bootstrap_unit,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "status": "not_computed",
            "reason": reason,
        }
        for field in EP_FIELDS
    ]
    pd.DataFrame(rows).to_csv(path, index=False)


def write_summary_json(path: Path, summary: pd.DataFrame, manifest: dict[str, object]) -> None:
    payload = {
        "manifest": manifest,
        "rows": json.loads(summary.to_json(orient="records")),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

