from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import minimum_filter

from .original_eddy_panels import (
    _interp_section,
    _load_catalog,
    _object_offsets_km,
    _read_column_window,
    _relative_xy,
    _window_indices,
    _year_filter_path,
)


EARTH_RADIUS_M = 6_371_000.0


@dataclass(frozen=True)
class JumpRow:
    eddy3d_object_id: int
    track3d_id: int
    date: str
    polarity: str
    shape_class: str
    radius_m: float
    n_layers: int
    jump_rank: int
    from_depth_index: int
    to_depth_index: int
    from_depth_m: float
    to_depth_m: float
    jump_distance_km: float
    jump_distance_over_R: float
    surface_lon: float
    surface_lat: float
    from_x_km: float
    from_y_km: float
    to_x_km: float
    to_y_km: float


def _shape_filtered_layers(
    results_root: Path,
    shape_dir_name: str,
    shapes: set[str],
    year_limit: int | None,
) -> pd.DataFrame:
    centers, shape = _load_catalog(results_root, shape_dir_name)
    shape = shape[shape["shape_class"].astype(str).isin(shapes)].copy()
    if shape.empty:
        raise ValueError(f"No shape tracks found for {sorted(shapes)}")
    layers = centers.merge(shape[["track3d_id", "shape_class"]], on="track3d_id", how="inner")
    layers["date"] = layers["date"].astype(str)
    if year_limit is not None:
        layers = layers[layers["date"].str.slice(0, 4).astype(int).le(year_limit)].copy()
    if layers.empty:
        raise ValueError("No layer centers after shape/year filtering")
    return layers


def _jump_rows_for_object(object_layers: pd.DataFrame, jump_ranks: int) -> list[JumpRow]:
    obj = _object_offsets_km(object_layers)
    if len(obj) < 2:
        return []
    surface = obj.iloc[0]
    radius_m = float(np.nanmedian(obj["radius_m"].to_numpy(dtype="f8")))
    x = obj["delta_x_km"].to_numpy(dtype="f8")
    y = obj["delta_y_km"].to_numpy(dtype="f8")
    jumps_km = np.hypot(np.diff(x), np.diff(y))
    order = np.argsort(jumps_km)[::-1]
    out: list[JumpRow] = []
    for rank, idx in enumerate(order[: max(0, jump_ranks)], start=1):
        upper = obj.iloc[int(idx)]
        lower = obj.iloc[int(idx) + 1]
        jump_km = float(jumps_km[int(idx)])
        out.append(
            JumpRow(
                eddy3d_object_id=int(surface["eddy3d_object_id"]),
                track3d_id=int(surface["track3d_id"]),
                date=str(surface["date"]),
                polarity=str(surface["polarity"]),
                shape_class=str(surface["shape_class"]),
                radius_m=radius_m,
                n_layers=int(len(obj)),
                jump_rank=rank,
                from_depth_index=int(upper["depth_index"]),
                to_depth_index=int(lower["depth_index"]),
                from_depth_m=float(upper["depth_m"]),
                to_depth_m=float(lower["depth_m"]),
                jump_distance_km=jump_km,
                jump_distance_over_R=float(jump_km * 1000.0 / radius_m) if radius_m > 0 else np.nan,
                surface_lon=float(surface["longitude"]),
                surface_lat=float(surface["latitude"]),
                from_x_km=float(upper["delta_x_km"]),
                from_y_km=float(upper["delta_y_km"]),
                to_x_km=float(lower["delta_x_km"]),
                to_y_km=float(lower["delta_y_km"]),
            )
        )
    return out


def _safe_corr(a: pd.Series, b: pd.Series) -> float:
    av = a.to_numpy(dtype="f8")
    bv = b.to_numpy(dtype="f8")
    ok = np.isfinite(av) & np.isfinite(bv)
    if int(ok.sum()) < 3:
        return float("nan")
    return float(np.corrcoef(av[ok], bv[ok])[0, 1])


def _finite_quantile(values: np.ndarray, q: float, fallback: float = np.nan) -> float:
    finite = np.asarray(values, dtype="f8")
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float(fallback)
    return float(np.nanquantile(finite, q))


def _finite_mean(values: list[float] | np.ndarray) -> float:
    arr = np.asarray(values, dtype="f8")
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.nan
    return float(np.nanmean(finite))


def _finite_max(values: list[float] | np.ndarray) -> float:
    arr = np.asarray(values, dtype="f8")
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.nan
    return float(np.nanmax(finite))


def _finite_min(values: list[float] | np.ndarray) -> float:
    arr = np.asarray(values, dtype="f8")
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.nan
    return float(np.nanmin(finite))


def _polar_samples(
    field: np.ndarray,
    x_m: np.ndarray,
    y_m: np.ndarray,
    center_x_km: float,
    center_y_km: float,
    radius_m: float,
    *,
    radius_fracs: np.ndarray,
    n_phi: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    phi = np.linspace(0.0, 2.0 * np.pi, int(n_phi), endpoint=False, dtype="f8")
    rr = np.asarray(radius_fracs, dtype="f8")[:, None] * float(radius_m) / 1000.0
    x_km = float(center_x_km) + rr * np.cos(phi)[None, :]
    y_km = float(center_y_km) + rr * np.sin(phi)[None, :]
    interp = RegularGridInterpolator((y_m, x_m), field, bounds_error=False, fill_value=np.nan)
    sampled = interp(np.column_stack([y_km.ravel() * 1000.0, x_km.ravel() * 1000.0])).reshape(x_km.shape)
    return sampled, x_km, y_km, phi


def _weighted_axis_ratio(x_km: np.ndarray, y_km: np.ndarray, weights: np.ndarray) -> float:
    finite = np.isfinite(x_km) & np.isfinite(y_km) & np.isfinite(weights) & (weights > 0)
    if int(finite.sum()) < 6:
        return np.nan
    x = x_km[finite]
    y = y_km[finite]
    w = weights[finite]
    wsum = float(np.sum(w))
    if wsum <= 0:
        return np.nan
    xm = float(np.sum(w * x) / wsum)
    ym = float(np.sum(w * y) / wsum)
    dx = x - xm
    dy = y - ym
    cov = np.array(
        [
            [np.sum(w * dx * dx) / wsum, np.sum(w * dx * dy) / wsum],
            [np.sum(w * dx * dy) / wsum, np.sum(w * dy * dy) / wsum],
        ],
        dtype="f8",
    )
    eig = np.sort(np.linalg.eigvalsh(cov))
    if eig[1] <= 0:
        return np.nan
    return float(np.sqrt(max(eig[0], 0.0) / eig[1]))


def _weak_core_metrics(speed: np.ndarray, xx_km: np.ndarray, yy_km: np.ndarray, radius_m: float) -> dict[str, float]:
    radius_km = max(float(radius_m) / 1000.0, 1.0)
    rr = np.hypot(xx_km, yy_km)
    finite = np.isfinite(speed) & (rr <= 1.5 * radius_km)
    if int(finite.sum()) < 9:
        return {
            "weak_core_count": 0.0,
            "weak_core_first_speed_ms": np.nan,
            "weak_core_second_speed_ms": np.nan,
            "weak_core_first_to_second_ratio": np.nan,
            "weak_core_pair_distance_km": np.nan,
        }
    threshold = _finite_quantile(speed[finite], 0.25)
    local_min = speed == minimum_filter(np.where(np.isfinite(speed), speed, np.inf), size=3, mode="nearest")
    candidates = finite & local_min & (speed <= threshold)
    ys, xs = np.where(candidates)
    if len(xs) == 0:
        return {
            "weak_core_count": 0.0,
            "weak_core_first_speed_ms": np.nan,
            "weak_core_second_speed_ms": np.nan,
            "weak_core_first_to_second_ratio": np.nan,
            "weak_core_pair_distance_km": np.nan,
        }
    order = np.argsort(speed[ys, xs])
    ys = ys[order]
    xs = xs[order]
    first = float(speed[ys[0], xs[0]])
    if len(xs) > 1:
        second = float(speed[ys[1], xs[1]])
        dist = float(np.hypot(xx_km[ys[1], xs[1]] - xx_km[ys[0], xs[0]], yy_km[ys[1], xs[1]] - yy_km[ys[0], xs[0]]))
        ratio = float(first / second) if second > 0 else np.nan
    else:
        second = np.nan
        dist = np.nan
        ratio = np.nan
    return {
        "weak_core_count": float(len(xs)),
        "weak_core_first_speed_ms": first,
        "weak_core_second_speed_ms": second,
        "weak_core_first_to_second_ratio": ratio,
        "weak_core_pair_distance_km": dist,
    }


def _layer_roundness_metrics(
    column: dict[str, np.ndarray],
    layer_index: int,
    surface_lon: float,
    surface_lat: float,
    center_x_km: float,
    center_y_km: float,
    radius_m: float,
) -> dict[str, float]:
    depth = np.asarray(column["depth"], dtype="f8")
    if layer_index < 0 or layer_index >= len(depth):
        return {}
    x_m, y_m, xx_km, yy_km = _relative_xy(column["longitude"], column["latitude"], surface_lon, surface_lat)

    u = np.asarray(column["uo_glor"], dtype="f8")[layer_index]
    v = np.asarray(column["vo_glor"], dtype="f8")[layer_index]
    speed = np.hypot(u, v)
    radius_km = max(float(radius_m) / 1000.0, 1.0)
    radius_fracs = np.linspace(0.7, 1.3, 5, dtype="f8")
    speed_samples, x_s, y_s, phi = _polar_samples(
        speed,
        x_m,
        y_m,
        center_x_km,
        center_y_km,
        radius_m,
        radius_fracs=radius_fracs,
        n_phi=72,
    )
    u_samples, _, _, _ = _polar_samples(
        u,
        x_m,
        y_m,
        center_x_km,
        center_y_km,
        radius_m,
        radius_fracs=np.array([1.0], dtype="f8"),
        n_phi=72,
    )
    v_samples, _, _, _ = _polar_samples(
        v,
        x_m,
        y_m,
        center_x_km,
        center_y_km,
        radius_m,
        radius_fracs=np.array([1.0], dtype="f8"),
        n_phi=72,
    )

    ring_by_phi = np.array([_finite_mean(speed_samples[:, i]) for i in range(speed_samples.shape[1])], dtype="f8")
    finite_phi = np.isfinite(ring_by_phi)
    ring_mean = float(np.nanmean(ring_by_phi[finite_phi])) if np.any(finite_phi) else np.nan
    ring_std = float(np.nanstd(ring_by_phi[finite_phi])) if np.any(finite_phi) else np.nan
    ring_cv = float(ring_std / ring_mean) if np.isfinite(ring_mean) and ring_mean > 0 else np.nan
    valid_fraction = float(np.mean(np.isfinite(speed_samples)))
    if np.any(finite_phi) and np.nansum(ring_by_phi[finite_phi]) > 0:
        weights = np.where(finite_phi, ring_by_phi, 0.0)
        crescent_m1 = float(abs(np.nansum(weights * np.exp(1j * phi))) / np.nansum(weights))
        crescent_m2 = float(abs(np.nansum(weights * np.exp(2j * phi))) / np.nansum(weights))
        weak_sector_fraction = float(np.mean(ring_by_phi[finite_phi] < 0.6 * np.nanmedian(ring_by_phi[finite_phi])))
    else:
        crescent_m1 = np.nan
        crescent_m2 = np.nan
        weak_sector_fraction = np.nan

    tangent_x = -np.sin(phi)
    tangent_y = np.cos(phi)
    ring_u = u_samples[0]
    ring_v = v_samples[0]
    ring_speed = np.hypot(ring_u, ring_v)
    tangent_component = ring_u * tangent_x + ring_v * tangent_y
    tangent_cos = np.abs(tangent_component) / np.maximum(ring_speed, 1.0e-12)
    finite_tangent = np.isfinite(tangent_cos) & np.isfinite(ring_speed) & (ring_speed > 0)
    tangent_fraction = float(np.mean(tangent_cos[finite_tangent] >= np.cos(np.deg2rad(30.0)))) if np.any(finite_tangent) else np.nan
    signed = np.sign(tangent_component[finite_tangent])
    signed_consistency = float(max(np.mean(signed >= 0), np.mean(signed <= 0))) if signed.size else np.nan

    high_weight = np.where(np.isfinite(speed_samples), np.maximum(speed_samples - np.nanmedian(speed_samples), 0.0), 0.0)
    axis_ratio = _weighted_axis_ratio(x_s, y_s, high_weight)
    eccentricity = float(np.sqrt(max(0.0, 1.0 - axis_ratio * axis_ratio))) if np.isfinite(axis_ratio) else np.nan
    rr = np.hypot(xx_km - center_x_km, yy_km - center_y_km)
    core_mask = rr <= 1.5 * radius_km
    local_finite = speed[core_mask & np.isfinite(speed)]
    center_speed = float(np.nanmin(local_finite)) if local_finite.size else np.nan
    weak = _weak_core_metrics(speed, xx_km - center_x_km, yy_km - center_y_km, radius_m)
    roundness_score = _finite_mean([axis_ratio, 1.0 / (1.0 + max(ring_cv, 0.0)) if np.isfinite(ring_cv) else np.nan])
    disorder_score = _finite_mean(
        [
            1.0 - roundness_score if np.isfinite(roundness_score) else np.nan,
            crescent_m1,
            min(float(weak["weak_core_count"]) / 4.0, 1.0),
        ]
    )
    return {
        "layer_index": float(layer_index),
        "layer_depth_m": float(depth[layer_index]),
        "ring_valid_fraction": valid_fraction,
        "ring_speed_mean_ms": ring_mean,
        "ring_speed_cv": ring_cv,
        "axis_ratio_roundness": axis_ratio,
        "eccentricity": eccentricity,
        "crescent_m1": crescent_m1,
        "crescent_m2": crescent_m2,
        "weak_sector_fraction": weak_sector_fraction,
        "tangent_fraction": tangent_fraction,
        "tangent_signed_consistency": signed_consistency,
        "center_core_min_speed_ms": center_speed,
        "roundness_score": roundness_score,
        "disorder_score": disorder_score,
        **weak,
    }


def _prefix_metrics(prefix: str, metrics: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def _classify_jump_relation(row: dict[str, object]) -> str:
    jump_over_r = float(row.get("jump_distance_over_R", np.nan))
    zero_shift = float(row.get("parallel_zero_line_shift_over_R", np.nan))
    roundness = float(row.get("layer_roundness_mean", np.nan))
    disorder = float(row.get("layer_disorder_mean", np.nan))
    weak_count = float(row.get("layer_weak_core_count_max", np.nan))
    if not np.isfinite(jump_over_r) or not np.isfinite(zero_shift):
        return "unclear_or_missing"
    if zero_shift >= 0.35:
        return "zero_line_jump_related"
    if jump_over_r >= 0.15 and zero_shift <= 0.25 and (
        (np.isfinite(roundness) and roundness <= 0.55)
        or (np.isfinite(disorder) and disorder >= 0.45)
        or (np.isfinite(weak_count) and weak_count >= 2)
    ):
        return "zero_line_consistent_center_jump"
    return "unclear_or_missing"


def _metrics_for_jump(column: dict[str, np.ndarray], jump: JumpRow) -> dict[str, object]:
    upper = _layer_roundness_metrics(
        column,
        jump.from_depth_index,
        jump.surface_lon,
        jump.surface_lat,
        jump.from_x_km,
        jump.from_y_km,
        jump.radius_m,
    )
    lower = _layer_roundness_metrics(
        column,
        jump.to_depth_index,
        jump.surface_lon,
        jump.surface_lat,
        jump.to_x_km,
        jump.to_y_km,
        jump.radius_m,
    )
    row: dict[str, object] = dict(jump.__dict__)
    row.update(_prefix_metrics("upper", upper))
    row.update(_prefix_metrics("lower", lower))
    for mode in ("parallel", "normal"):
        row.update(_zero_line_metrics(column, jump, mode))

    for key in (
        "roundness_score",
        "disorder_score",
        "axis_ratio_roundness",
        "ring_speed_cv",
        "crescent_m1",
        "crescent_m2",
        "weak_sector_fraction",
        "tangent_fraction",
    ):
        values = [float(row.get(f"upper_{key}", np.nan)), float(row.get(f"lower_{key}", np.nan))]
        row[f"layer_{key}_mean"] = _finite_mean(values)
        row[f"layer_{key}_max"] = _finite_max(values)
        row[f"layer_{key}_min"] = _finite_min(values)
    counts = [float(row.get("upper_weak_core_count", np.nan)), float(row.get("lower_weak_core_count", np.nan))]
    row["layer_weak_core_count_max"] = _finite_max(counts)
    row["layer_weak_core_ambiguity_mean"] = _finite_mean(
        [
            float(row.get("upper_weak_core_first_to_second_ratio", np.nan)),
            float(row.get("lower_weak_core_first_to_second_ratio", np.nan)),
        ]
    )
    row["jump_relation_class"] = _classify_jump_relation(row)
    return row


def _write_tables(metrics: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_parquet(output_dir / "jump_roundness_metrics.parquet", index=False)
    metrics.to_csv(output_dir / "jump_roundness_metrics.csv", index=False)
    zero_cols = [
        col
        for col in metrics.columns
        if col.startswith("parallel_") or col.startswith("normal_") or col in {"eddy3d_object_id", "track3d_id", "date", "shape_class", "polarity", "jump_rank", "jump_distance_over_R"}
    ]
    metrics[zero_cols].to_csv(output_dir / "zero_line_mismatch_metrics.csv", index=False)
    summary_rows: list[dict[str, object]] = []
    for shape, part in metrics.groupby("shape_class", dropna=False):
        summary_rows.append(
            {
                "shape_class": str(shape),
                "n_jumps": int(len(part)),
                "n_objectdays": int(part["eddy3d_object_id"].nunique()),
                "n_tracks": int(part["track3d_id"].nunique()),
                "median_jump_over_R": float(np.nanmedian(part["jump_distance_over_R"])),
                "median_roundness_score": float(np.nanmedian(part["layer_roundness_score_mean"])),
                "median_disorder_score": float(np.nanmedian(part["layer_disorder_score_mean"])),
                "median_crescent_m1": float(np.nanmedian(part["layer_crescent_m1_mean"])),
                "median_weak_core_count": float(np.nanmedian(part["layer_weak_core_count_max"])),
                "median_parallel_zero_shift_over_R": float(np.nanmedian(part["parallel_zero_line_shift_over_R"])),
                "center_jump_with_continuous_zero_fraction": float((part["jump_relation_class"] == "zero_line_consistent_center_jump").mean()),
                "zero_line_jump_related_fraction": float((part["jump_relation_class"] == "zero_line_jump_related").mean()),
                "corr_jump_vs_roundness": _safe_corr(part["jump_distance_over_R"], part["layer_roundness_score_mean"]),
                "corr_jump_vs_disorder": _safe_corr(part["jump_distance_over_R"], part["layer_disorder_score_mean"]),
                "corr_zero_shift_vs_roundness": _safe_corr(part["parallel_zero_line_shift_over_R"], part["layer_roundness_score_mean"]),
                "relation_counts_json": json.dumps(part["jump_relation_class"].value_counts().to_dict(), ensure_ascii=False),
            }
        )
    all_part = metrics
    summary_rows.append(
        {
            "shape_class": "all",
            "n_jumps": int(len(all_part)),
            "n_objectdays": int(all_part["eddy3d_object_id"].nunique()),
            "n_tracks": int(all_part["track3d_id"].nunique()),
            "median_jump_over_R": float(np.nanmedian(all_part["jump_distance_over_R"])),
            "median_roundness_score": float(np.nanmedian(all_part["layer_roundness_score_mean"])),
            "median_disorder_score": float(np.nanmedian(all_part["layer_disorder_score_mean"])),
            "median_crescent_m1": float(np.nanmedian(all_part["layer_crescent_m1_mean"])),
            "median_weak_core_count": float(np.nanmedian(all_part["layer_weak_core_count_max"])),
            "median_parallel_zero_shift_over_R": float(np.nanmedian(all_part["parallel_zero_line_shift_over_R"])),
            "center_jump_with_continuous_zero_fraction": float((all_part["jump_relation_class"] == "zero_line_consistent_center_jump").mean()),
            "zero_line_jump_related_fraction": float((all_part["jump_relation_class"] == "zero_line_jump_related").mean()),
            "corr_jump_vs_roundness": _safe_corr(all_part["jump_distance_over_R"], all_part["layer_roundness_score_mean"]),
            "corr_jump_vs_disorder": _safe_corr(all_part["jump_distance_over_R"], all_part["layer_disorder_score_mean"]),
            "corr_zero_shift_vs_roundness": _safe_corr(all_part["parallel_zero_line_shift_over_R"], all_part["layer_roundness_score_mean"]),
            "relation_counts_json": json.dumps(all_part["jump_relation_class"].value_counts().to_dict(), ensure_ascii=False),
        }
    )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "roundness_relation_summary.csv", index=False)
    (output_dir / "roundness_relation_summary.json").write_text(
        json.dumps(summary_rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def _scatter_by_shape(ax: plt.Axes, df: pd.DataFrame, x: str, y: str, title: str, xlabel: str, ylabel: str) -> None:
    palette = {
        "coherent": "#1b9e77",
        "mixed": "#d95f02",
        "complex": "#7570b3",
        "upright_like": "#66a61e",
        "transitional": "#e7298a",
    }
    for shape, part in df.groupby("shape_class", dropna=False):
        ax.scatter(part[x], part[y], s=10, alpha=0.35, color=palette.get(str(shape), "0.35"), label=str(shape))
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)


def _plot_outputs(metrics: pd.DataFrame, output_dir: Path) -> None:
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 5))
    _scatter_by_shape(
        ax,
        metrics,
        "layer_roundness_score_mean",
        "jump_distance_over_R",
        "Jump magnitude vs layer roundness",
        "mean upper/lower roundness score",
        "jump distance / R",
    )
    fig.tight_layout()
    fig.savefig(fig_dir / "jump_over_R_vs_roundness.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    _scatter_by_shape(
        ax,
        metrics,
        "layer_roundness_score_mean",
        "parallel_zero_line_shift_over_R",
        "Zero-line shift vs layer roundness",
        "mean upper/lower roundness score",
        "parallel zero-line shift / R",
    )
    fig.tight_layout()
    fig.savefig(fig_dir / "zero_line_mismatch_vs_roundness.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    _scatter_by_shape(
        ax,
        metrics,
        "layer_weak_core_count_max",
        "jump_distance_over_R",
        "Jump magnitude vs weak-core multiplicity",
        "max weak-core count in upper/lower layer",
        "jump distance / R",
    )
    fig.tight_layout()
    fig.savefig(fig_dir / "multi_core_ambiguity_vs_jump.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    _scatter_by_shape(
        ax,
        metrics,
        "layer_crescent_m1_mean",
        "jump_distance_over_R",
        "Jump magnitude vs crescent index",
        "mean m=1 crescent index",
        "jump distance / R",
    )
    fig.tight_layout()
    fig.savefig(fig_dir / "crescent_index_vs_jump.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    metrics.boxplot(column="layer_roundness_score_mean", by="shape_class", ax=ax)
    ax.set_title("Roundness by shape class")
    fig.suptitle("")
    ax.set_xlabel("shape class")
    ax.set_ylabel("mean upper/lower roundness score")
    fig.tight_layout()
    fig.savefig(fig_dir / "roundness_by_shape_class.png", dpi=220)
    plt.close(fig)

    relation = pd.crosstab(metrics["shape_class"], metrics["jump_relation_class"])
    fig, ax = plt.subplots(figsize=(8, 5))
    relation.plot(kind="bar", stacked=True, ax=ax)
    ax.set_title("Jump-zero-line relation by shape class")
    ax.set_xlabel("shape class")
    ax.set_ylabel("count")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "jump_relation_by_shape_class.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    _scatter_by_shape(
        ax,
        metrics,
        "layer_disorder_score_mean",
        "jump_distance_over_R",
        "Jump magnitude vs combined disorder",
        "mean upper/lower disorder score",
        "jump distance / R",
    )
    fig.tight_layout()
    fig.savefig(fig_dir / "disorder_score_vs_jump.png", dpi=220)
    plt.close(fig)


def _write_report(metrics: pd.DataFrame, summary: pd.DataFrame, output_dir: Path) -> None:
    relation_counts = metrics["jump_relation_class"].value_counts().to_dict()
    coherent = metrics[metrics["shape_class"].astype(str).eq("coherent")]
    mixed = metrics[metrics["shape_class"].astype(str).eq("mixed")]
    lines = [
        "# 间断点跳跃与速度场圆度关系诊断",
        "",
        "本诊断回答一个具体问题：为什么有些样本里剖面上的 u_perp=0 零线看起来连续，但 Hua 中心线仍然发生跳跃。这里把二者分开处理：零线是某个剖面的速度投影零线，Hua 中心是二维速度弱核和旋转几何约束共同选出的中心。",
        "",
        "## 诊断量",
        "- 圆度：由速度环的轴比和环向均匀性给出，越接近 1 越圆。",
        "- 月牙指数：速度环的 m=1 角向集中度，越大表示强速带越偏向一侧。",
        "- 弱核多值性：核心区内局地速度极小值数量和第一、第二弱核相近程度。",
        "- 零线关系：计算 jump-parallel 与 jump-normal 剖面中中心到 u_perp=0 的距离，以及零线随上下层的移动量。",
        "",
        "## 样本量",
        f"- jump 数量：{len(metrics)}",
        f"- object-day 数量：{int(metrics['eddy3d_object_id'].nunique())}",
        f"- track 数量：{int(metrics['track3d_id'].nunique())}",
        "- 分类计数：",
    ]
    for name, count in metrics["shape_class"].value_counts().items():
        lines.append(f"  - {name}: {int(count)}")
    lines.extend(["", "## 主要关系"])
    for _, row in summary.iterrows():
        lines.append(
            f"- {row['shape_class']}: median J/R={row['median_jump_over_R']:.3f}, "
            f"roundness={row['median_roundness_score']:.3f}, disorder={row['median_disorder_score']:.3f}, "
            f"center-jump-with-continuous-zero={row['center_jump_with_continuous_zero_fraction']:.3f}"
        )
    lines.extend(
        [
            "",
            "## 关系分类",
        ]
    )
    for name, count in relation_counts.items():
        lines.append(f"- {name}: {int(count)}")
    if len(coherent) and len(mixed):
        lines.extend(
            [
                "",
                "## Coherent 与 Mixed 的对照",
                f"- coherent median roundness: {float(np.nanmedian(coherent['layer_roundness_score_mean'])):.3f}",
                f"- mixed median roundness: {float(np.nanmedian(mixed['layer_roundness_score_mean'])):.3f}",
                f"- coherent median weak-core count: {float(np.nanmedian(coherent['layer_weak_core_count_max'])):.2f}",
                f"- mixed median weak-core count: {float(np.nanmedian(mixed['layer_weak_core_count_max'])):.2f}",
                f"- coherent median crescent_m1: {float(np.nanmedian(coherent['layer_crescent_m1_mean'])):.3f}",
                f"- mixed median crescent_m1: {float(np.nanmedian(mixed['layer_crescent_m1_mean'])):.3f}",
            ]
        )
    lines.extend(
        [
            "",
            "## 解释边界",
            "如果 `zero_line_consistent_center_jump` 比例高，说明中心跳跃常常不是因为剖面零线突然跳走，而是因为二维速度弱核不够单一或速度环不够圆，导致相邻层选择了不同候选中心。这个结论是机制支持，不是因果证明；要证明因果还需要控制背景剪切、地形、缺测和检测阈值。",
        ]
    )
    (output_dir / "jump_roundness_relation_summary_zh.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_metrics(
    *,
    layers: pd.DataFrame,
    filter_root: Path,
    output_dir: Path,
    jump_ranks: int,
    half_width_deg: float,
    resume: bool,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    parts_dir = output_dir / "parts"
    parts_dir.mkdir(exist_ok=True)
    if resume and (output_dir / "jump_roundness_metrics.parquet").exists():
        return pd.read_parquet(output_dir / "jump_roundness_metrics.parquet")

    done_ids: set[int] = set()
    rows: list[dict[str, object]] = []
    part_index = len(list(parts_dir.glob("roundness_part_*.parquet"))) + 1
    if resume:
        for path in parts_dir.glob("roundness_part_*.parquet"):
            try:
                part = pd.read_parquet(path)
                rows.extend(part.to_dict("records"))
                done_ids.update(part["eddy3d_object_id"].astype(int).unique())
            except Exception:
                continue

    for n, (object_id, obj) in enumerate(layers.groupby("eddy3d_object_id", sort=False), start=1):
        if int(object_id) in done_ids:
            continue
        jumps = _jump_rows_for_object(obj.sort_values("depth_index"), jump_ranks)
        if not jumps:
            continue
        year = jumps[0].date[:4]
        filter_path = _year_filter_path(filter_root, year)
        if not filter_path.exists():
            continue
        try:
            column = _read_column_window(
                path=filter_path,
                date=jumps[0].date,
                center_lon=jumps[0].surface_lon,
                center_lat=jumps[0].surface_lat,
                half_width_deg=half_width_deg,
                variables=("uo_glor", "vo_glor"),
            )
            for jump in jumps:
                rows.append(_metrics_for_jump(column, jump))
        except Exception as exc:
            for jump in jumps:
                row = dict(jump.__dict__)
                row["jump_relation_class"] = "unclear_or_missing"
                row["error"] = repr(exc)
                rows.append(row)
        if n % 200 == 0 and rows:
            part_path = parts_dir / f"roundness_part_{part_index:05d}.parquet"
            pd.DataFrame.from_records(rows).to_parquet(part_path, index=False)
            print(json.dumps({"processed_objectdays": n, "part": str(part_path), "rows": len(rows)}, ensure_ascii=False), flush=True)
            part_index += 1
            rows = []

    if rows:
        part_path = parts_dir / f"roundness_part_{part_index:05d}.parquet"
        pd.DataFrame.from_records(rows).to_parquet(part_path, index=False)
    parts = [pd.read_parquet(path) for path in sorted(parts_dir.glob("roundness_part_*.parquet"))]
    if not parts:
        raise RuntimeError("No roundness metrics were produced")
    return pd.concat(parts, ignore_index=True)


def analyze_jump_roundness_relation(
    *,
    results_root: Path,
    shape_dir_name: str,
    filter_root: Path,
    output_dir: Path,
    shapes: str,
    jump_ranks: int,
    half_width_deg: float,
    year_limit: int | None,
    selected_metadata: Path | None,
    max_objectdays: int | None,
    resume: bool,
) -> None:
    shape_set = {part.strip() for part in shapes.split(",") if part.strip()}
    layers = _shape_filtered_layers(results_root, shape_dir_name, shape_set, year_limit)
    if selected_metadata is not None:
        selected = pd.read_csv(selected_metadata)
        if "eddy3d_object_id" not in selected.columns:
            raise ValueError(f"selected metadata lacks eddy3d_object_id: {selected_metadata}")
        keep_ids = set(selected["eddy3d_object_id"].astype(int).unique())
        layers = layers[layers["eddy3d_object_id"].astype(int).isin(keep_ids)].copy()
        if layers.empty:
            raise ValueError("No layers matched --selected-metadata object ids")
    if max_objectdays is not None:
        keep_ids = list(layers["eddy3d_object_id"].astype(int).drop_duplicates().head(max_objectdays))
        layers = layers[layers["eddy3d_object_id"].astype(int).isin(keep_ids)].copy()
    metrics = _run_metrics(
        layers=layers,
        filter_root=filter_root,
        output_dir=output_dir,
        jump_ranks=jump_ranks,
        half_width_deg=half_width_deg,
        resume=resume,
    )
    summary = _write_tables(metrics, output_dir)
    _plot_outputs(metrics, output_dir)
    _write_report(metrics, summary, output_dir)
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "rows": int(len(metrics)),
                "objectdays": int(metrics["eddy3d_object_id"].nunique()),
                "tracks": int(metrics["track3d_id"].nunique()),
                "shape_counts": {str(k): int(v) for k, v in metrics["shape_class"].value_counts().to_dict().items()},
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Analyze whether layer jumps are related to non-round or multi-core velocity fields.")
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--shape-dir-name", default="shape_classification_1993_2022_hua_b3_start2_life30")
    parser.add_argument("--filter-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shapes", default="coherent,mixed")
    parser.add_argument("--jump-ranks", type=int, default=2)
    parser.add_argument("--half-width-deg", type=float, default=2.0)
    parser.add_argument("--year-limit", type=int, default=None)
    parser.add_argument("--selected-metadata", type=Path, default=None)
    parser.add_argument("--max-objectdays", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    analyze_jump_roundness_relation(
        results_root=args.results_root,
        shape_dir_name=args.shape_dir_name,
        filter_root=args.filter_root,
        output_dir=args.output_dir,
        shapes=args.shapes,
        jump_ranks=args.jump_ranks,
        half_width_deg=args.half_width_deg,
        year_limit=args.year_limit,
        selected_metadata=args.selected_metadata,
        max_objectdays=args.max_objectdays,
        resume=args.resume,
    )


def _section_vectors(jump: JumpRow, mode: str) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float], str]:
    dx = float(jump.to_x_km - jump.from_x_km)
    dy = float(jump.to_y_km - jump.from_y_km)
    norm = float(np.hypot(dx, dy))
    if not np.isfinite(norm) or norm <= 1.0e-9:
        ex_jump, ey_jump = 1.0, 0.0
    else:
        ex_jump, ey_jump = dx / norm, dy / norm
    nx_jump, ny_jump = -ey_jump, ex_jump
    mid = (0.5 * (jump.from_x_km + jump.to_x_km), 0.5 * (jump.from_y_km + jump.to_y_km))
    if mode == "normal":
        return (nx_jump, ny_jump), (ex_jump, ey_jump), mid, "jump-normal section through center-pair midpoint"
    return (ex_jump, ey_jump), (nx_jump, ny_jump), mid, "jump-parallel section through center-pair midpoint"


def _nearest_zero(coord: np.ndarray, values: np.ndarray, target: float) -> float:
    coord = np.asarray(coord, dtype="f8")
    values = np.asarray(values, dtype="f8")
    ok = np.isfinite(coord) & np.isfinite(values)
    if int(ok.sum()) < 2:
        return np.nan
    c = coord[ok]
    v = values[ok]
    roots: list[float] = []
    for i in range(len(c) - 1):
        if v[i] == 0:
            roots.append(float(c[i]))
        if v[i] * v[i + 1] < 0 and c[i + 1] != c[i]:
            frac = -v[i] / (v[i + 1] - v[i])
            roots.append(float(c[i] + frac * (c[i + 1] - c[i])))
    if not roots:
        return np.nan
    roots_arr = np.asarray(roots, dtype="f8")
    return float(roots_arr[np.nanargmin(np.abs(roots_arr - float(target)))])


def _zero_line_metrics(column: dict[str, np.ndarray], jump: JumpRow, mode: str) -> dict[str, float | str]:
    section_vec, normal_vec, anchor, label = _section_vectors(jump, mode)
    x_m, y_m, _, _ = _relative_xy(column["longitude"], column["latitude"], jump.surface_lon, jump.surface_lat)
    u = np.asarray(column["uo_glor"], dtype="f8")
    v = np.asarray(column["vo_glor"], dtype="f8")
    velocity_normal = u * normal_vec[0] + v * normal_vec[1]
    half_width = max(float(jump.radius_m) / 1000.0 * 2.5, 150.0)
    coord = np.linspace(-half_width, half_width, 241, dtype="f8")
    x_line = anchor[0] + coord * section_vec[0]
    y_line = anchor[1] + coord * section_vec[1]
    sec = _interp_section(velocity_normal, x_m, y_m, x_line, y_line)
    from_coord = (jump.from_x_km - anchor[0]) * section_vec[0] + (jump.from_y_km - anchor[1]) * section_vec[1]
    to_coord = (jump.to_x_km - anchor[0]) * section_vec[0] + (jump.to_y_km - anchor[1]) * section_vec[1]
    ku = int(np.clip(jump.from_depth_index, 0, sec.shape[0] - 1))
    kl = int(np.clip(jump.to_depth_index, 0, sec.shape[0] - 1))
    z0 = _nearest_zero(coord, sec[ku], from_coord)
    z1 = _nearest_zero(coord, sec[kl], to_coord)
    radius_km = max(float(jump.radius_m) / 1000.0, 1.0)
    center_distance_mean = float(np.nanmean([abs(from_coord - z0), abs(to_coord - z1)]))
    zero_shift = float(abs(z1 - z0)) if np.isfinite(z0) and np.isfinite(z1) else np.nan
    center_shift = float(abs(to_coord - from_coord))
    return {
        f"{mode}_section_label": label,
        f"{mode}_from_center_section_coord_km": float(from_coord),
        f"{mode}_to_center_section_coord_km": float(to_coord),
        f"{mode}_from_nearest_zero_km": z0,
        f"{mode}_to_nearest_zero_km": z1,
        f"{mode}_center_to_zero_mean_km": center_distance_mean,
        f"{mode}_center_to_zero_mean_over_R": center_distance_mean / radius_km if np.isfinite(center_distance_mean) else np.nan,
        f"{mode}_zero_line_shift_km": zero_shift,
        f"{mode}_zero_line_shift_over_R": zero_shift / radius_km if np.isfinite(zero_shift) else np.nan,
        f"{mode}_center_projection_shift_km": center_shift,
        f"{mode}_center_projection_shift_over_R": center_shift / radius_km,
    }


if __name__ == "__main__":
    main()


def _projection_rows_for_object(
    object_layers: pd.DataFrame,
    jump_ranks: int,
    depth_padding_layers: int,
) -> list[dict[str, object]]:
    obj = _object_offsets_km(object_layers).sort_values("depth_index").reset_index(drop=True)
    jumps = _jump_rows_for_object(obj, jump_ranks)
    rows: list[dict[str, object]] = []
    if not jumps:
        return rows

    depth_index = obj["depth_index"].to_numpy(dtype="i4")
    x = obj["delta_x_km"].to_numpy(dtype="f8")
    y = obj["delta_y_km"].to_numpy(dtype="f8")
    radius_km = max(float(np.nanmedian(obj["radius_m"].to_numpy(dtype="f8"))) / 1000.0, 1.0)

    for jump in jumps:
        for mode in ("parallel", "normal"):
            section_vec, plane_normal_vec, anchor, label = _section_vectors(jump, mode)
            section_coord = (x - anchor[0]) * section_vec[0] + (y - anchor[1]) * section_vec[1]
            offplane_coord = (x - anchor[0]) * plane_normal_vec[0] + (y - anchor[1]) * plane_normal_vec[1]

            in_window = (
                depth_index >= min(jump.from_depth_index, jump.to_depth_index) - int(depth_padding_layers)
            ) & (
                depth_index <= max(jump.from_depth_index, jump.to_depth_index) + int(depth_padding_layers)
            )
            window_off = np.abs(offplane_coord[in_window])
            from_mask = depth_index == int(jump.from_depth_index)
            to_mask = depth_index == int(jump.to_depth_index)
            from_section = float(section_coord[from_mask][0]) if np.any(from_mask) else np.nan
            to_section = float(section_coord[to_mask][0]) if np.any(to_mask) else np.nan
            from_off = float(offplane_coord[from_mask][0]) if np.any(from_mask) else np.nan
            to_off = float(offplane_coord[to_mask][0]) if np.any(to_mask) else np.nan
            projected_jump = abs(to_section - from_section) if np.isfinite(from_section) and np.isfinite(to_section) else np.nan
            hidden_jump = abs(to_off - from_off) if np.isfinite(from_off) and np.isfinite(to_off) else np.nan
            compression = (
                1.0 - projected_jump / float(jump.jump_distance_km)
                if np.isfinite(projected_jump) and jump.jump_distance_km > 0
                else np.nan
            )
            rows.append(
                {
                    "eddy3d_object_id": jump.eddy3d_object_id,
                    "track3d_id": jump.track3d_id,
                    "date": jump.date,
                    "polarity": jump.polarity,
                    "shape_class": jump.shape_class,
                    "jump_rank": jump.jump_rank,
                    "from_depth_index": jump.from_depth_index,
                    "to_depth_index": jump.to_depth_index,
                    "from_depth_m": jump.from_depth_m,
                    "to_depth_m": jump.to_depth_m,
                    "jump_distance_km": jump.jump_distance_km,
                    "jump_distance_over_R": jump.jump_distance_over_R,
                    "section_mode": mode,
                    "section_label": label,
                    "projected_jump_km": float(projected_jump),
                    "projected_jump_over_R": projected_jump / radius_km if np.isfinite(projected_jump) else np.nan,
                    "hidden_jump_km": float(hidden_jump),
                    "hidden_jump_over_R": hidden_jump / radius_km if np.isfinite(hidden_jump) else np.nan,
                    "projection_compression_fraction": float(compression),
                    "from_offplane_km": from_off,
                    "to_offplane_km": to_off,
                    "window_offplane_mean_km": _finite_mean(window_off),
                    "window_offplane_p90_km": _finite_quantile(window_off, 0.9),
                    "window_offplane_max_km": _finite_max(window_off),
                    "window_offplane_mean_over_R": _finite_mean(window_off) / radius_km,
                    "window_offplane_p90_over_R": _finite_quantile(window_off, 0.9) / radius_km,
                    "window_offplane_max_over_R": _finite_max(window_off) / radius_km,
                    "n_window_layers": int(np.isfinite(window_off).sum()),
                }
            )
    return rows


def _roundness_join_keys() -> list[str]:
    return ["eddy3d_object_id", "jump_rank", "from_depth_index", "to_depth_index"]


def _join_roundness_metrics(projection: pd.DataFrame, roundness_metrics: Path | None) -> pd.DataFrame:
    if roundness_metrics is None or not roundness_metrics.exists():
        return projection
    suffix_cols = [
        "jump_relation_class",
        "layer_roundness_score_mean",
        "layer_disorder_score_mean",
        "layer_crescent_m1_mean",
        "layer_weak_core_count_max",
        "parallel_zero_line_shift_over_R",
        "normal_zero_line_shift_over_R",
    ]
    source = pd.read_csv(roundness_metrics, usecols=lambda col: col in set(_roundness_join_keys() + suffix_cols))
    keep = [col for col in _roundness_join_keys() + suffix_cols if col in source.columns]
    return projection.merge(source[keep], on=_roundness_join_keys(), how="left")


def _projection_dominance(row: pd.Series) -> str:
    relation = str(row.get("jump_relation_class", ""))
    compression = float(row.get("projection_compression_fraction", np.nan))
    hidden = float(row.get("window_offplane_p90_over_R", np.nan))
    if row.get("section_mode") == "normal" and np.isfinite(compression) and compression > 0.75:
        return "normal_projection_expected"
    if relation == "zero_line_consistent_center_jump" and np.isfinite(hidden) and hidden >= 0.15:
        return "offplane_projection_supported"
    if relation == "zero_line_consistent_center_jump":
        return "inplane_center_switch_likely"
    if relation == "zero_line_jump_related":
        return "zero_line_jump_related"
    return "unclear"


def _write_projection_outputs(projection: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    projection["projection_explanation_class"] = projection.apply(_projection_dominance, axis=1)
    projection.to_csv(output_dir / "jump_projection_geometry_metrics.csv", index=False)
    try:
        projection.to_parquet(output_dir / "jump_projection_geometry_metrics.parquet", index=False)
    except Exception:
        pass

    group_cols = ["shape_class", "section_mode"]
    rows: list[dict[str, object]] = []
    for keys, part in projection.groupby(group_cols, dropna=False):
        shape, mode = keys
        rows.append(
            {
                "shape_class": shape,
                "section_mode": mode,
                "n_rows": int(len(part)),
                "median_jump_over_R": float(np.nanmedian(part["jump_distance_over_R"])),
                "median_projected_jump_over_R": float(np.nanmedian(part["projected_jump_over_R"])),
                "median_hidden_jump_over_R": float(np.nanmedian(part["hidden_jump_over_R"])),
                "median_window_offplane_p90_over_R": float(np.nanmedian(part["window_offplane_p90_over_R"])),
                "median_projection_compression_fraction": float(np.nanmedian(part["projection_compression_fraction"])),
                "class_counts_json": json.dumps(
                    {str(k): int(v) for k, v in part["projection_explanation_class"].value_counts().to_dict().items()},
                    ensure_ascii=False,
                ),
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir / "jump_projection_geometry_summary.csv", index=False)
    (output_dir / "jump_projection_geometry_summary.json").write_text(
        json.dumps(summary.to_dict(orient="records"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _plot_projection_outputs(projection, output_dir)
    _write_projection_report(projection, summary, output_dir)
    return summary


def _plot_projection_outputs(projection: pd.DataFrame, output_dir: Path) -> None:
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    finite = projection.replace([np.inf, -np.inf], np.nan)

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    for (shape, mode), part in finite.groupby(["shape_class", "section_mode"], dropna=False):
        ax.scatter(
            part["jump_distance_over_R"],
            part["window_offplane_p90_over_R"],
            s=10,
            alpha=0.25,
            label=f"{shape} / {mode}",
        )
    ax.set_xlabel("center jump distance / R")
    ax.set_ylabel("local centerline off-plane p90 / R")
    ax.set_title("Does the section hide part of the 3D centerline motion?")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(fig_dir / "offplane_distance_vs_jump.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    finite.boxplot(
        column="projection_compression_fraction",
        by=["shape_class", "section_mode"],
        ax=ax,
        grid=False,
        rot=25,
    )
    ax.set_ylabel("1 - projected jump / true jump")
    ax.set_title("Projection compression by section mode")
    fig.suptitle("")
    fig.tight_layout()
    fig.savefig(fig_dir / "projection_compression_by_section_mode.png", dpi=220)
    plt.close(fig)

    if "jump_relation_class" in finite.columns:
        fig, ax = plt.subplots(figsize=(8.0, 5.2))
        finite.boxplot(
            column="window_offplane_p90_over_R",
            by=["jump_relation_class", "section_mode"],
            ax=ax,
            grid=False,
            rot=30,
        )
        ax.set_ylabel("local centerline off-plane p90 / R")
        ax.set_title("Off-plane distance by zero-line relation class")
        fig.suptitle("")
        fig.tight_layout()
        fig.savefig(fig_dir / "offplane_by_zero_line_relation.png", dpi=220)
        plt.close(fig)

    matrix = pd.crosstab(
        finite["section_mode"],
        finite["projection_explanation_class"],
        normalize="index",
    )
    fig, ax = plt.subplots(figsize=(8.0, 3.8))
    im = ax.imshow(matrix.to_numpy(dtype="f8"), cmap="viridis", aspect="auto", vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(matrix.columns)), labels=matrix.columns, rotation=25, ha="right", fontsize=7)
    ax.set_yticks(np.arange(len(matrix.index)), labels=matrix.index)
    ax.set_title("Projection explanation classes")
    fig.colorbar(im, ax=ax, label="row fraction")
    fig.tight_layout()
    fig.savefig(fig_dir / "projection_explanation_matrix.png", dpi=220)
    plt.close(fig)


def _write_projection_report(projection: pd.DataFrame, summary: pd.DataFrame, output_dir: Path) -> None:
    lines = [
        "# 间断点剖面投影几何验证",
        "",
        "## 目的",
        "",
        "验证“零线连续但中心跳跃”是否可能来自三维中心线没有落在当前二维剖面内。这里不重新计算速度场，只用已有中心线几何与 roundness 诊断结果。",
        "",
        "## 定义",
        "",
        "- `jump-parallel`：剖面方向沿上下两层中心跳变向量，理论上包含这两个端点的连线。",
        "- `jump-normal`：剖面方向垂直于跳变向量，并穿过两端点中点；它用于检查横切边界，天然会把真正的跳变方向投影到剖面外。",
        "- `projected_jump`：真实 jump 在该剖面横轴上能看到的分量。",
        "- `hidden_jump`：真实 jump 落在剖面外、图上被隐藏的分量。",
        "- `window_offplane_p90`：间断点附近若干层中心线到该剖面的出平面距离，用于判断整段中心线是否离开剖面。",
        "",
        "## 核心统计",
        "",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"- {row['shape_class']} / {row['section_mode']}: "
            f"median projected J/R={row['median_projected_jump_over_R']:.3f}, "
            f"median hidden J/R={row['median_hidden_jump_over_R']:.3f}, "
            f"median compression={row['median_projection_compression_fraction']:.3f}, "
            f"off-plane p90/R={row['median_window_offplane_p90_over_R']:.3f}"
        )

    normal = summary[summary["section_mode"].astype(str).eq("normal")]
    parallel = summary[summary["section_mode"].astype(str).eq("parallel")]
    normal_comp = float(np.nanmedian(normal["median_projection_compression_fraction"])) if not normal.empty else np.nan
    parallel_comp = float(np.nanmedian(parallel["median_projection_compression_fraction"])) if not parallel.empty else np.nan
    lines.extend(
        [
            "",
            "## 初步判定",
            "",
            f"- `jump-normal` 的典型投影压缩为 {normal_comp:.3f}，这意味着 normal 剖面主要用于看横切边界，不能直接用来判断中心是否沿跳变方向连续。",
            f"- `jump-parallel` 的典型投影压缩为 {parallel_comp:.3f}，它更接近真实跳变路径；如果这里仍然出现零线连续但中心跳，才更支持“二维弱核候选切换/不圆结构”。",
            "- 因此，你的判断是对的：一部分视觉不一致确实来自三维点没有落在同一个二维剖面上，尤其是 `jump-normal` 图。最终解释必须同时看 projection geometry、速度场圆度和弱核多值性。",
        ]
    )
    (output_dir / "jump_projection_geometry_summary_zh.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze_jump_projection_geometry(
    *,
    results_root: Path,
    shape_dir_name: str,
    output_dir: Path,
    shapes: str,
    jump_ranks: int,
    depth_padding_layers: int,
    year_limit: int | None,
    selected_metadata: Path | None,
    max_objectdays: int | None,
    roundness_metrics: Path | None,
) -> None:
    shape_set = {part.strip() for part in shapes.split(",") if part.strip()}
    layers = _shape_filtered_layers(results_root, shape_dir_name, shape_set, year_limit)
    if selected_metadata is not None:
        selected = pd.read_csv(selected_metadata)
        keep_ids = set(selected["eddy3d_object_id"].astype(int).unique())
        layers = layers[layers["eddy3d_object_id"].astype(int).isin(keep_ids)].copy()
        if layers.empty:
            raise ValueError("No layers matched --selected-metadata object ids")
    if max_objectdays is not None:
        keep_ids = list(layers["eddy3d_object_id"].astype(int).drop_duplicates().head(max_objectdays))
        layers = layers[layers["eddy3d_object_id"].astype(int).isin(keep_ids)].copy()

    rows: list[dict[str, object]] = []
    for _, obj in layers.groupby("eddy3d_object_id", sort=False):
        rows.extend(_projection_rows_for_object(obj, jump_ranks, depth_padding_layers))
    if not rows:
        raise RuntimeError("No projection metrics were produced")
    projection = pd.DataFrame.from_records(rows)
    projection = _join_roundness_metrics(projection, roundness_metrics)
    summary = _write_projection_outputs(projection, output_dir)
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "rows": int(len(projection)),
                "objectdays": int(projection["eddy3d_object_id"].nunique()),
                "tracks": int(projection["track3d_id"].nunique()),
                "summary_rows": int(len(summary)),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
