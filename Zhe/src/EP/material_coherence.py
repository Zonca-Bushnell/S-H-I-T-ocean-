from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np

try:
    import pandas as pd
except ModuleNotFoundError:  # pragma: no cover - dependency-light dry-run support.
    pd = None

from .contracts import DEFAULT_RESULT_ROOT
from .boundary_strategy import resolve_boundary_strategy
from .dynamic_boundary import boundary_flux_metrics, connected_component, edge_mask, neighbors4


MATERIAL_COHERENCE_BOUNDARY_MODES = ("particle_retention_v1", "lavd_hybrid_v1")
BOUNDARY_BUDGETS = ("edge_proxy", "full_3d")
DEFAULT_MATERIAL_COHERENCE_OUTPUT_ROOT = Path(
    "/root/autodl-fs/kuroshiou/EP-FLUX/object_material_coherence_ep_validation"
)


@dataclass(frozen=True)
class MaterialCoherenceRequest:
    result_root: Path = DEFAULT_RESULT_ROOT
    filter_root: Path = Path("/root/autodl-fs/kuroshiou/Filter")
    output_root: Path = DEFAULT_MATERIAL_COHERENCE_OUTPUT_ROOT
    shapes: tuple[str, ...] = ("coherent", "upright_like")
    orientations: tuple[str, ...] = ("turned",)
    buoyancy_sources: tuple[str, ...] = ("thermal_wind",)
    boundary_modes: tuple[str, ...] = MATERIAL_COHERENCE_BOUNDARY_MODES
    boundary_budget: str = "full_3d"
    filter_template: str = "global_phy_{year}_bandpass_30_180d.nc"
    radial_bins: int = 24
    azimuth_bins: int = 48
    rmax: float = 1.5
    reference_lat: float = 30.0
    constant_n2: float = 2.0e-5
    core_radius_over_R: float = 1.5
    speed_core_quantile: float = 0.45
    pv_core_quantile: float = 0.70
    min_mask_fraction: float = 0.01
    active_contour_iterations: int = 14
    leakage_weight: float = 1.0
    smoothness_weight: float = 0.08
    containment_weight: float = 0.35
    area_weight: float = 0.12
    vertical_continuity_weight: float = 0.18
    time_continuity_weight: float = 0.08
    levelset_sigma_cells: float = 1.0
    min_core_retention: float = 0.75
    min_area_fraction: float = 0.15
    max_area_fraction: float = 0.65
    trajectory_window_days: int = 7
    particle_spacing_km: float = 5.0
    advection_step_hours: float = 6.0
    max_tracks_per_shape: int = 0
    max_objectdays: int = 0
    skip_missing: bool = False
    dry_run: bool = False


def _validate_request(request: MaterialCoherenceRequest) -> None:
    unknown = [mode for mode in request.boundary_modes if mode not in MATERIAL_COHERENCE_BOUNDARY_MODES]
    if unknown:
        raise ValueError(f"boundary modes must be in {MATERIAL_COHERENCE_BOUNDARY_MODES}: {unknown}")
    for mode in request.boundary_modes:
        resolve_boundary_strategy(mode)
    if request.boundary_budget not in BOUNDARY_BUDGETS:
        raise ValueError(f"boundary budget must be one of {BOUNDARY_BUDGETS}")
    if request.trajectory_window_days < 1:
        raise ValueError("trajectory_window_days must be positive")
    if request.particle_spacing_km <= 0:
        raise ValueError("particle_spacing_km must be positive")
    if request.advection_step_hours <= 0:
        raise ValueError("advection_step_hours must be positive")


def _split_csv(value: str | tuple[str, ...] | list[str]) -> tuple[str, ...]:
    if isinstance(value, (tuple, list)):
        return tuple(str(v).strip() for v in value if str(v).strip())
    return tuple(part.strip() for part in str(value).split(",") if part.strip())


def _load_runtime_helpers() -> None:
    global MaterialVolumeRequest
    global _advect_mask_stack_center_following
    global _align_reference_masks_by_depth
    global _compute_one_slice
    global _full_boundary_flux_budget
    global _json_ready
    global _load_shape_objects
    global _polar_grid
    global _read_filter_day
    global _require_runtime
    global _sample_object_slice
    global _time_advection_audit
    global _write_table

    from .material_volume import (
        MaterialVolumeRequest,
        _compute_one_slice,
        _full_boundary_flux_budget,
        _json_ready,
        _write_table,
    )
    from .object_material_boundary import (
        _advect_mask_stack_center_following,
        _align_reference_masks_by_depth,
        _load_shape_objects,
        _polar_grid,
        _read_filter_day,
        _require_runtime,
        _sample_object_slice,
        _time_advection_audit,
    )


def _finite_mean(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    return float(np.nanmean(arr)) if np.any(np.isfinite(arr)) else np.nan


def _mask_iou(mask: np.ndarray, reference: np.ndarray | None) -> float:
    if reference is None or reference.shape != mask.shape:
        return np.nan
    union = np.count_nonzero(mask.astype(bool) | reference.astype(bool))
    if union == 0:
        return np.nan
    return float(np.count_nonzero(mask.astype(bool) & reference.astype(bool)) / union)


def _base_material_request(request: MaterialCoherenceRequest, shape: str, orientation: str, buoyancy_source: str) -> MaterialVolumeRequest:
    return MaterialVolumeRequest(
        result_root=request.result_root,
        output_root=request.output_root,
        shapes=(shape,),
        orientations=(orientation,),
        buoyancy_sources=(buoyancy_source,),
        reference_lat=request.reference_lat,
        constant_n2=request.constant_n2,
        core_radius_over_R=request.core_radius_over_R,
        speed_core_quantile=request.speed_core_quantile,
        pv_core_quantile=request.pv_core_quantile,
        min_mask_fraction=request.min_mask_fraction,
        boundary_mode="levelset_v2",
        boundary_budget=request.boundary_budget,
        active_contour_iterations=request.active_contour_iterations,
        leakage_weight=request.leakage_weight,
        smoothness_weight=request.smoothness_weight,
        containment_weight=request.containment_weight,
        area_weight=request.area_weight,
        vertical_continuity_weight=request.vertical_continuity_weight,
        time_continuity_weight=request.time_continuity_weight,
        levelset_sigma_cells=request.levelset_sigma_cells,
        min_core_retention=request.min_core_retention,
        min_area_fraction=request.min_area_fraction,
        max_area_fraction=request.max_area_fraction,
    )


def _vorticity_proxy(rep) -> np.ndarray:
    ur, ut = rep.polar_velocity()
    radial = np.asarray(rep.radial_m, dtype=float)
    theta = np.asarray(rep.theta_rad, dtype=float)
    radial_pos = radial.copy()
    if radial_pos.size > 1 and radial_pos[0] <= 0:
        radial_pos[0] = radial_pos[1] * 0.5
    d_r_ut_dr = np.gradient(radial_pos[None, :, None] * ut, radial_pos, axis=1, edge_order=1)
    d_ur_dtheta = (np.roll(ur, -1, axis=2) - np.roll(ur, 1, axis=2)) / (
        2.0 * (2.0 * np.pi / max(1, theta.size))
    )
    return (d_r_ut_dr - d_ur_dtheta) / radial_pos[None, :, None]


def _lavd_proxy(rep, mask_domain: np.ndarray, window_days: int) -> np.ndarray:
    zeta = _vorticity_proxy(rep)
    out = np.zeros_like(zeta, dtype=float)
    seconds = float(window_days) * 86400.0
    for iz in range(zeta.shape[0]):
        domain = mask_domain[iz] if mask_domain.ndim == 3 else np.isfinite(zeta[iz])
        bg = np.nanmedian(zeta[iz][domain]) if np.any(domain & np.isfinite(zeta[iz])) else np.nanmedian(zeta[iz])
        out[iz] = np.abs(zeta[iz] - bg) * seconds
    return out


def _candidate_masks(mask: np.ndarray, core_domain: np.ndarray, lavd: np.ndarray | None) -> list[np.ndarray]:
    candidates = [mask.astype(bool)]
    candidates.append((mask | neighbors4(mask)) & core_domain)
    candidates.append((mask & ~edge_mask(mask)) & core_domain)
    if lavd is not None and np.any(core_domain & np.isfinite(lavd)):
        values = lavd[core_domain & np.isfinite(lavd)]
        for quantile in (0.55, 0.65, 0.75):
            threshold = float(np.nanquantile(values, quantile))
            candidates.append((mask | (core_domain & (lavd >= threshold))) & core_domain)
            candidates.append((mask & (neighbors4(lavd >= threshold) | (lavd >= threshold))) & core_domain)
    out: list[np.ndarray] = []
    seed = tuple(int(i) for i in np.unravel_index(int(np.nanargmin(np.where(mask, 0.0, 1.0))), mask.shape))
    for candidate in candidates:
        if not np.any(candidate):
            continue
        component = connected_component(candidate.astype(bool), seed if candidate[seed] else None)
        if np.any(component) and not any(np.array_equal(component, old) for old in out):
            out.append(component)
    return out


def _core_domain_from_row(row, shape: tuple[int, int]) -> np.ndarray:
    radial_fraction = np.linspace(0.0, 1.0, shape[0], dtype=float)[:, None]
    limit = float(row.get("core_radius_over_R", 1.5)) if hasattr(row, "get") else 1.5
    return np.repeat(radial_fraction <= max(0.2, limit), shape[1], axis=1)


def _particle_metrics(mask: np.ndarray, next_reference: np.ndarray | None, previous_reference: np.ndarray | None) -> dict[str, float | str]:
    forward = _mask_iou(mask, next_reference)
    backward = _mask_iou(mask, previous_reference)
    values = [value for value in (forward, backward) if np.isfinite(value)]
    retention = float(np.nanmean(values)) if values else np.nan
    return {
        "particle_retention_forward": forward,
        "particle_retention_backward": backward,
        "particle_retention_mean": retention,
        "particle_escape_fraction": float(1.0 - retention) if np.isfinite(retention) else np.nan,
        "boundary_crossing_rate": float(1.0 - retention) if np.isfinite(retention) else np.nan,
        "particle_advection_model": "grid_cell_mask_advection_center_following_bandpass_uv",
    }


def _score_candidate(
    *,
    mask: np.ndarray,
    base_mask: np.ndarray,
    next_reference: np.ndarray | None,
    previous_reference: np.ndarray | None,
    u: np.ndarray,
    v: np.ndarray,
    buoyancy: np.ndarray,
    q_proxy: np.ndarray,
    theta_prime: np.ndarray,
    x_km: np.ndarray,
    y_km: np.ndarray,
    mean_u: float,
    mean_v: float,
    internal_flux_scale: float,
    request: MaterialCoherenceRequest,
) -> tuple[float, dict[str, float | str]]:
    flux = boundary_flux_metrics(
        mask=mask,
        u=u,
        v=v,
        buoyancy=buoyancy,
        q_proxy=q_proxy,
        theta_prime=theta_prime,
        x_km=x_km,
        y_km=y_km,
        mean_u=mean_u,
        mean_v=mean_v,
        internal_flux_scale=internal_flux_scale,
    )
    particle = _particle_metrics(mask, next_reference, previous_reference)
    area_drift = abs((np.count_nonzero(mask) - np.count_nonzero(base_mask)) / max(1, np.count_nonzero(base_mask)))
    retention_loss = 1.0 - particle["particle_retention_mean"] if np.isfinite(particle["particle_retention_mean"]) else 0.4
    leakage = float(flux.get("leakage_mean_abs_ms", np.nan))
    leakage_term = leakage if np.isfinite(leakage) else 1.0
    score = leakage_term + 0.025 * retention_loss + request.area_weight * area_drift
    return float(score), {**flux, **particle, "mask_area_drift_from_levelset_v2": float(area_drift)}


def _select_coherent_masks(
    *,
    rep,
    base_table,
    debug: dict[str, np.ndarray],
    boundary_mode: str,
    request: MaterialCoherenceRequest,
    previous_reference: np.ndarray | None,
    next_reference: np.ndarray | None,
) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    masks = debug["mask"].astype(bool)
    q_proxy = debug["q_proxy"]
    buoyancy = debug["buoyancy"]
    theta_prime = rep.theta_prime if rep.theta_prime is not None else np.full_like(buoyancy, np.nan)
    x_km, y_km = rep.mesh_xy_km
    radial_over_R = rep.radial_m[:, None] / max(float(rep.radius_m), 1.0)
    core_domains = np.isfinite(rep.speed) & np.isfinite(q_proxy) & (radial_over_R[None, :, :] <= request.core_radius_over_R)
    lavd = _lavd_proxy(rep, core_domains, request.trajectory_window_days)
    selected = np.zeros_like(masks, dtype=bool)
    metric_rows: list[dict[str, object]] = []

    for iz, base_mask in enumerate(masks):
        row = base_table.iloc[iz]
        candidates = _candidate_masks(
            base_mask,
            core_domains[iz],
            lavd[iz] if boundary_mode == "lavd_hybrid_v1" else None,
        )
        prev_ref = previous_reference[iz] if previous_reference is not None and iz < previous_reference.shape[0] else None
        next_ref = next_reference[iz] if next_reference is not None and iz < next_reference.shape[0] else None
        mean_u = float(row.get("mean_u_ms", np.nan))
        mean_v = float(row.get("mean_v_ms", np.nan))
        if not np.isfinite(mean_u):
            mean_u = _finite_mean(rep.u[iz][base_mask])
        if not np.isfinite(mean_v):
            mean_v = _finite_mean(rep.v[iz][base_mask])
        internal_flux_scale = float(row.get("pv_flux_magnitude", np.nan))
        scored = [
            _score_candidate(
                mask=candidate,
                base_mask=base_mask,
                next_reference=next_ref,
                previous_reference=prev_ref,
                u=rep.u[iz],
                v=rep.v[iz],
                buoyancy=buoyancy[iz],
                q_proxy=q_proxy[iz],
                theta_prime=theta_prime[iz],
                x_km=x_km,
                y_km=y_km,
                mean_u=mean_u,
                mean_v=mean_v,
                internal_flux_scale=internal_flux_scale,
                request=request,
            )
            + (candidate,)
            for candidate in candidates
        ]
        score, meta, best = min(scored, key=lambda item: item[0]) if scored else (np.nan, {}, base_mask)
        selected[iz] = best
        lavd_values = lavd[iz][best & np.isfinite(lavd[iz])]
        metric_rows.append(
            {
                "depth_index": int(iz),
                "depth_m": float(rep.depth_m[iz]),
                "boundary_mode": boundary_mode,
                "material_coherence_score": float(score) if np.isfinite(score) else np.nan,
                "levelset_v2_mask_cells": int(np.count_nonzero(base_mask)),
                "selected_mask_cells": int(np.count_nonzero(best)),
                "lavd_proxy_mean": _finite_mean(lavd_values),
                "lavd_proxy_p90": float(np.nanpercentile(lavd_values, 90)) if lavd_values.size else np.nan,
                "lavd_proxy_model": "frozen_local_relative_vorticity_deviation"
                if boundary_mode == "lavd_hybrid_v1"
                else "not_used",
                "trajectory_window_days_used": int(request.trajectory_window_days),
                **meta,
            }
        )
    metrics = pd.DataFrame(metric_rows)
    full_budget = _full_boundary_flux_budget(
        rep=rep,
        masks=selected,
        q_proxy=q_proxy,
        buoyancy=buoyancy,
        x_km=x_km,
        y_km=y_km,
    )
    return selected, metrics, full_budget


def _closure_residual_table(profiles: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in profiles.iterrows():
        pv_internal = abs(float(row.get("pv_flux_magnitude", np.nan)))
        boundary_area = abs(float(row.get("lateral_area_m2", 0.0) or 0.0))
        boundary_area += abs(float(row.get("top_area_m2", 0.0) or 0.0))
        boundary_area += abs(float(row.get("bottom_area_m2", 0.0) or 0.0))
        boundary_area = boundary_area if boundary_area > 0 else np.nan
        pv_boundary_integral = float(row.get("total_pv_flux_proxy", np.nan))
        heat_boundary_integral = float(row.get("total_heat_flux_watt_proxy", np.nan))
        momentum_x_integral = float(row.get("total_momentum_x_flux_proxy", 0.0) or 0.0)
        momentum_y_integral = float(row.get("total_momentum_y_flux_proxy", 0.0) or 0.0)
        pv_boundary_density = abs(pv_boundary_integral / boundary_area) if np.isfinite(boundary_area) else np.nan
        heat_boundary_density = abs(heat_boundary_integral / boundary_area) if np.isfinite(boundary_area) else np.nan
        momentum_boundary_density = math.hypot(
            momentum_x_integral / boundary_area if np.isfinite(boundary_area) else np.nan,
            momentum_y_integral / boundary_area if np.isfinite(boundary_area) else np.nan,
        )
        edge_ratio = float(row.get("boundary_flux_over_internal_flux", np.nan))
        residual = (
            pv_boundary_density / (pv_internal + 1e-18)
            if np.isfinite(pv_boundary_density)
            else edge_ratio
        )
        rows.append(
            {
                "shape": row.get("shape"),
                "orientation": row.get("orientation"),
                "buoyancy_source": row.get("buoyancy_source"),
                "boundary_mode": row.get("boundary_mode"),
                "polarity": row.get("polarity"),
                "track3d_id": row.get("track3d_id"),
                "eddy3d_object_id": row.get("eddy3d_object_id"),
                "date": row.get("date"),
                "depth_index": row.get("depth_index"),
                "depth_m": row.get("depth_m"),
                "boundary_area_m2_for_density_proxy": boundary_area,
                "pv_boundary_density_proxy": pv_boundary_density,
                "pv_boundary_over_internal_proxy": residual,
                "heat_boundary_density_abs_proxy": heat_boundary_density,
                "momentum_boundary_density_abs_proxy": momentum_boundary_density,
                "closure_residual_proxy": residual,
            }
        )
    return pd.DataFrame(rows)


def _track_summary(profiles: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["shape", "boundary_mode", "polarity", "track3d_id"]
    for keys, sub in profiles.groupby(group_cols, sort=True):
        shape, mode, polarity, track_id = keys
        leakage = sub["leakage_mean_abs_ms"].to_numpy(float)
        retention = sub["particle_retention_mean"].to_numpy(float)
        closure = sub["closure_residual_proxy"].to_numpy(float)
        rows.append(
            {
                "shape": shape,
                "boundary_mode": mode,
                "polarity": polarity,
                "track3d_id": int(track_id),
                "n_objectdays": int(sub["eddy3d_object_id"].nunique()),
                "n_rows": int(sub.shape[0]),
                "leakage_median_ms": float(np.nanmedian(leakage)),
                "particle_retention_median": float(np.nanmedian(retention)),
                "particle_escape_fraction_median": float(np.nanmedian(sub["particle_escape_fraction"].to_numpy(float))),
                "closure_residual_proxy_median": float(np.nanmedian(closure)),
                "pv_boundary_over_internal_proxy_median": float(
                    np.nanmedian(sub["pv_boundary_over_internal_proxy"].to_numpy(float))
                ),
                "materiality_score": float(
                    np.nanmedian(leakage) + 0.02 * (1.0 - np.nanmedian(retention)) + 0.01 * np.nanmedian(closure)
                ),
            }
        )
    return pd.DataFrame(rows)


def _plot_outputs(output_root: Path, profiles: pd.DataFrame, summary: pd.DataFrame) -> list[Path]:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return []

    figures = output_root / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    clean = profiles.replace([np.inf, -np.inf], np.nan)

    if {"shape", "boundary_mode", "leakage_mean_abs_ms"}.issubset(clean.columns):
        labels = []
        groups = []
        for (shape, mode), sub in clean.groupby(["shape", "boundary_mode"], sort=True):
            values = sub["leakage_mean_abs_ms"].dropna().to_numpy(float)
            if values.size:
                labels.append(f"{shape}\n{mode}")
                groups.append(values)
        if groups:
            fig, ax = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
            ax.boxplot(groups, labels=labels, showfliers=False)
            ax.set_ylabel("boundary leakage |u_n| (m/s)")
            ax.set_title("Material-coherence boundary leakage")
            ax.grid(True, axis="y", alpha=0.25)
            path = figures / "coherence_leakage_by_mode.png"
            fig.savefig(path, dpi=180)
            plt.close(fig)
            written.append(path)

    if {"particle_retention_mean", "boundary_flux_over_internal_flux", "shape"}.issubset(clean.columns):
        fig, ax = plt.subplots(figsize=(6.5, 5), constrained_layout=True)
        for (shape, mode), sub in clean.groupby(["shape", "boundary_mode"], sort=True):
            ax.scatter(
                sub["particle_retention_mean"],
                sub["boundary_flux_over_internal_flux"],
                s=12,
                alpha=0.35,
                label=f"{shape}/{mode}",
            )
        ax.set_xlabel("particle retention proxy")
        ax.set_ylabel("boundary flux / internal PV flux proxy")
        ax.set_title("Retention vs boundary exchange")
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False, fontsize=8)
        path = figures / "particle_retention_vs_boundary_flux.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        written.append(path)

    if not summary.empty and {"shape", "boundary_mode", "materiality_score"}.issubset(summary.columns):
        grouped = summary.groupby(["shape", "boundary_mode"], sort=True)["materiality_score"].median().reset_index()
        fig, ax = plt.subplots(figsize=(7.5, 4), constrained_layout=True)
        labels = grouped["shape"].astype(str) + "\n" + grouped["boundary_mode"].astype(str)
        ax.bar(np.arange(grouped.shape[0]), grouped["materiality_score"], color="#4c78a8")
        ax.set_xticks(np.arange(grouped.shape[0]))
        ax.set_xticklabels(labels)
        ax.set_ylabel("lower is more material")
        ax.set_title("Materiality score by shape and boundary")
        ax.grid(True, axis="y", alpha=0.25)
        path = figures / "coherent_vs_upright_like_materiality.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        written.append(path)

    return written


def _write_summary_md(path: Path, summary: pd.DataFrame, request: MaterialCoherenceRequest) -> None:
    lines = [
        "# Object Material-Coherence EP Validation",
        "",
        "## 方法定位",
        "本轮把严格材料体 EP 闭合从瞬时低 leakage 边界推进到有限时间材料相干性边界。",
        "它仍是可落地验证版：没有实现完整 geodesic Cauchy-Green 闭合曲线搜索，",
        "而是用 particle retention 与 LAVD hybrid 两个代理来判断边界是否更接近材料体。",
        "",
        "## 文献依据",
        "- Haller/Beron-Vera: 材料涡边界应是有限时间内低拉伸、少泄漏的闭合材料曲线。",
        "- LAVD/RCLV: 旋转相干涡可用粒子轨迹上的相对涡度偏差积分来约束旋转核心。",
        "- Abernathey/Haller 2018: Eulerian eddy 与真正 Lagrangian coherent vortex 不等价。",
        "- Froyland coherent sets: 材料相干性可定义为有限时间内区域与外界交换最小。",
        "",
        "## 本次口径",
        f"- shapes: `{','.join(request.shapes)}`",
        f"- boundary modes: `{','.join(request.boundary_modes)}`",
        f"- orientations: `{','.join(request.orientations)}`",
        f"- buoyancy sources: `{','.join(request.buoyancy_sources)}`",
        f"- boundary budget: `{request.boundary_budget}`",
        f"- trajectory window: `{request.trajectory_window_days}` days",
        "- 粒子保持第一版使用 30-180d bandpass 水平速度的中心随动 mask 平流代理。",
        "- 顶/底通量仍使用 continuity `w_proxy`，不是观测垂直速度。",
        "",
        "## 结果摘要",
        "```text",
        summary.to_string(index=False) if not summary.empty else "empty",
        "```",
        "",
        "## 判读边界",
        "- 若 particle retention 提高且 boundary exchange/closure residual 下降，说明动力边界更接近材料体。",
        "- 若 leakage 降低但 closure residual 不降，说明只优化边界法向速度还不足以闭合 heat/PV/momentum 预算。",
        "- 若 coherent 的 materiality score 低于 upright_like，才支持 coherent 更接近材料体；反之必须如实报告。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _print_dry_run(request: MaterialCoherenceRequest) -> None:
    print("Object material-coherence EP validation dry-run")
    print(f"result_root: {request.result_root}")
    print(f"filter_root: {request.filter_root}")
    print(f"output_root: {request.output_root}")
    print(f"shapes: {','.join(request.shapes)}")
    print(f"orientations: {','.join(request.orientations)}")
    print(f"buoyancy_sources: {','.join(request.buoyancy_sources)}")
    print(f"boundary_modes: {','.join(request.boundary_modes)}")
    print(f"boundary_budget: {request.boundary_budget}")
    print(f"trajectory_window_days: {request.trajectory_window_days}")
    print(f"particle_spacing_km: {request.particle_spacing_km}")
    print(f"advection_step_hours: {request.advection_step_hours}")
    print("advection_velocity: 30-180d bandpass uo_glor/vo_glor")


def _compute_base_record(
    *,
    obj,
    points,
    day_cache: dict[date, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    request: MaterialCoherenceRequest,
    radial: np.ndarray,
    theta: np.ndarray,
    radial_mesh: np.ndarray,
    theta_mesh: np.ndarray,
    orientation: str,
    buoyancy_source: str,
    f0: float,
):
    day = pd.Timestamp(obj.date).date()
    if day not in day_cache:
        day_cache[day] = _read_filter_day(request.filter_root, request.filter_template, day)
    lon, lat, u, v, theta_prime = day_cache[day]
    sampled = _sample_object_slice(
        obj,
        points,
        lon,
        lat,
        u,
        v,
        theta_prime,
        radial,
        theta,
        radial_mesh,
        theta_mesh,
        orientation,
    )
    if sampled is None:
        return None
    rep, axis = sampled
    n2 = np.full(rep.depth_m.shape, float(request.constant_n2), dtype="float64")
    base_request = _base_material_request(request, str(obj.shape_class), orientation, buoyancy_source)
    table, debug = _compute_one_slice(
        rep=rep,
        axis=axis,
        f0=f0,
        n2=n2,
        buoyancy_source=buoyancy_source,
        request=base_request,
        time_reference_masks=None,
    )
    return {
        "obj": obj,
        "day": day,
        "rep": rep,
        "axis": axis,
        "base_table": table,
        "debug": debug,
    }


def _advected_reference(record, target_record) -> np.ndarray | None:
    if record is None or target_record is None:
        return None
    day_gap = (target_record["day"] - record["day"]).days
    if day_gap <= 0:
        return None
    advected = _advect_mask_stack_center_following(record["rep"], record["debug"]["mask"], 86400.0 * day_gap)
    return _align_reference_masks_by_depth(record["rep"].depth_m, advected, target_record["rep"].depth_m)


def _enrich_table(
    table: pd.DataFrame,
    metrics: pd.DataFrame,
    full_budget: pd.DataFrame,
    *,
    record,
    shape: str,
    orientation: str,
    buoyancy_source: str,
    boundary_mode: str,
    request: MaterialCoherenceRequest,
) -> pd.DataFrame:
    out = table.copy()
    if "leakage_mean_abs_ms" in out.columns:
        out["levelset_v2_leakage_mean_abs_ms"] = out["leakage_mean_abs_ms"].to_numpy(float)
    for source in (metrics, full_budget):
        if source.empty:
            continue
        for col in source.columns:
            if col in {"depth_index", "depth_m"}:
                continue
            out[col] = source[col].to_numpy()[: out.shape[0]]
    if "initial_leakage_mean_abs_ms" in out.columns and "leakage_mean_abs_ms" in out.columns:
        initial = out["initial_leakage_mean_abs_ms"].to_numpy(float)
        selected = out["leakage_mean_abs_ms"].to_numpy(float)
        out["leakage_reduction_fraction"] = (initial - selected) / (np.abs(initial) + 1e-12)
    if "levelset_v2_leakage_mean_abs_ms" in out.columns and "leakage_mean_abs_ms" in out.columns:
        baseline = out["levelset_v2_leakage_mean_abs_ms"].to_numpy(float)
        selected = out["leakage_mean_abs_ms"].to_numpy(float)
        out["coherence_leakage_change_vs_levelset_v2"] = (baseline - selected) / (np.abs(baseline) + 1e-12)
    obj = record["obj"]
    out["shape"] = shape
    out["orientation"] = orientation
    out["buoyancy_source"] = buoyancy_source
    out["boundary_mode"] = boundary_mode
    out["boundary_budget"] = request.boundary_budget
    out["polarity"] = str(obj.polarity)
    out["tau"] = float(record["rep"].tau) if np.isfinite(record["rep"].tau) else np.nan
    out["date"] = str(obj.date)
    out["track3d_id"] = int(obj.track3d_id)
    out["eddy3d_object_id"] = int(obj.eddy3d_object_id)
    out["mean_radius_m"] = float(obj.mean_radius_m)
    out["material_coherence_family"] = "particle_lagrangian_lavd_proxy"
    out["trajectory_window_days"] = int(request.trajectory_window_days)
    out["particle_spacing_km"] = float(request.particle_spacing_km)
    out["advection_step_hours"] = float(request.advection_step_hours)
    closure = _closure_residual_table(out)
    for col in (
        "pv_boundary_over_internal_proxy",
        "heat_boundary_abs_proxy",
        "momentum_boundary_abs_proxy",
        "closure_residual_proxy",
    ):
        out[col] = closure[col].to_numpy() if col in closure else np.nan
    return out


def _write_subset_tables(root: Path, profiles: pd.DataFrame) -> None:
    particle_cols = [
        col
        for col in profiles.columns
        if col
        in {
            "shape",
            "orientation",
            "buoyancy_source",
            "boundary_mode",
            "polarity",
            "tau",
            "date",
            "track3d_id",
            "eddy3d_object_id",
            "depth_index",
            "depth_m",
            "particle_retention_forward",
            "particle_retention_backward",
            "particle_retention_mean",
            "particle_escape_fraction",
            "boundary_crossing_rate",
            "trajectory_window_days_used",
            "particle_advection_model",
        }
    ]
    lavd_cols = [
        col
        for col in profiles.columns
        if col
        in {
            "shape",
            "orientation",
            "buoyancy_source",
            "boundary_mode",
            "polarity",
            "tau",
            "date",
            "track3d_id",
            "eddy3d_object_id",
            "depth_index",
            "depth_m",
            "lavd_proxy_mean",
            "lavd_proxy_p90",
            "lavd_proxy_model",
        }
    ]
    budget_cols = [
        col
        for col in profiles.columns
        if col
        in {
            "shape",
            "orientation",
            "buoyancy_source",
            "boundary_mode",
            "polarity",
            "tau",
            "date",
            "track3d_id",
            "eddy3d_object_id",
            "depth_index",
            "depth_m",
            "boundary_budget",
            "vertical_velocity_source",
        }
        or col.startswith(("lateral_", "top_", "bottom_", "total_", "w_proxy_"))
    ]
    closure_cols = [
        col
        for col in profiles.columns
        if col
        in {
            "shape",
            "orientation",
            "buoyancy_source",
            "boundary_mode",
            "polarity",
            "tau",
            "date",
            "track3d_id",
            "eddy3d_object_id",
            "depth_index",
            "depth_m",
            "boundary_area_m2_for_density_proxy",
            "pv_boundary_density_proxy",
            "pv_boundary_over_internal_proxy",
            "heat_boundary_density_abs_proxy",
            "momentum_boundary_density_abs_proxy",
            "closure_residual_proxy",
        }
    ]
    _write_table(profiles[particle_cols], root / "particle_retention_budget.csv")
    _write_table(profiles[lavd_cols], root / "lavd_boundary_metrics.csv")
    _write_table(profiles[budget_cols], root / "full_boundary_flux_budget.csv")
    _write_table(profiles[closure_cols], root / "ep_boundary_closure_residual.csv")


def run_object_material_coherence_validation(request: MaterialCoherenceRequest) -> dict[str, Path]:
    _validate_request(request)
    if request.dry_run:
        _print_dry_run(request)
        return {}
    _load_runtime_helpers()
    _require_runtime()
    f0 = 2.0 * 7.2921159e-5 * math.sin(math.radians(request.reference_lat))
    radial, theta, radial_mesh, theta_mesh = _polar_grid(request.rmax, request.radial_bins, request.azimuth_bins)
    outputs: dict[str, Path] = {}
    all_profiles: list[pd.DataFrame] = []
    all_summaries: list[pd.DataFrame] = []

    for shape in request.shapes:
        try:
            rv_root, objects, points = _load_shape_objects(
                request.result_root,
                shape,
                request.max_tracks_per_shape,
                request.max_objectdays,
            )
        except FileNotFoundError:
            if request.skip_missing:
                continue
            raise
        if objects.empty:
            continue
        objects = objects.sort_values(["track3d_id", "date", "eddy3d_object_id"]).copy()
        for orientation in request.orientations:
            for buoyancy_source in request.buoyancy_sources:
                day_cache: dict[date, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
                records_by_track: dict[int, list[dict[str, object]]] = {}
                for obj in objects.itertuples(index=False):
                    record = _compute_base_record(
                        obj=obj,
                        points=points,
                        day_cache=day_cache,
                        request=request,
                        radial=radial,
                        theta=theta,
                        radial_mesh=radial_mesh,
                        theta_mesh=theta_mesh,
                        orientation=orientation,
                        buoyancy_source=buoyancy_source,
                        f0=f0,
                    )
                    if record is None:
                        continue
                    records_by_track.setdefault(int(obj.track3d_id), []).append(record)

                for boundary_mode in request.boundary_modes:
                    combo_dir = request.output_root / shape / orientation / buoyancy_source / boundary_mode
                    combo_dir.mkdir(parents=True, exist_ok=True)
                    rows: list[pd.DataFrame] = []
                    for records in records_by_track.values():
                        records.sort(key=lambda item: (item["day"], int(item["obj"].eddy3d_object_id)))
                        for idx, record in enumerate(records):
                            previous_ref = _advected_reference(records[idx - 1], record) if idx > 0 else None
                            next_record = records[idx + 1] if idx + 1 < len(records) else None
                            next_ref = (
                                _align_reference_masks_by_depth(
                                    next_record["rep"].depth_m,
                                    next_record["debug"]["mask"],
                                    record["rep"].depth_m,
                                )
                                if next_record is not None
                                else None
                            )
                            selected, metrics, full_budget = _select_coherent_masks(
                                rep=record["rep"],
                                base_table=record["base_table"],
                                debug=record["debug"],
                                boundary_mode=boundary_mode,
                                request=request,
                                previous_reference=previous_ref,
                                next_reference=next_ref,
                            )
                            time_audit = _time_advection_audit(selected, previous_ref)
                            metrics = metrics.merge(time_audit, on="depth_index", how="left", suffixes=("", "_time"))
                            rows.append(
                                _enrich_table(
                                    record["base_table"],
                                    metrics,
                                    full_budget,
                                    record=record,
                                    shape=shape,
                                    orientation=orientation,
                                    buoyancy_source=buoyancy_source,
                                    boundary_mode=boundary_mode,
                                    request=request,
                                )
                            )
                    if not rows:
                        continue
                    profiles = pd.concat(rows, ignore_index=True)
                    summary = _track_summary(profiles)
                    _write_table(profiles, combo_dir / "material_coherence_profiles.csv")
                    _write_table(summary, combo_dir / "track_material_coherence_summary.csv")
                    _write_subset_tables(combo_dir, profiles)
                    (combo_dir / "material_coherence_manifest.json").write_text(
                        json.dumps(
                            _json_ready(
                                {
                                    "shape": shape,
                                    "orientation": orientation,
                                    "buoyancy_source": buoyancy_source,
                                    "boundary_mode": boundary_mode,
                                    "rv_root": rv_root,
                                    "request": request.__dict__,
                                    "method_status": "object-level material-coherence proxy; no geodesic LCS claim",
                                }
                            ),
                            ensure_ascii=False,
                            indent=2,
                        ),
                        encoding="utf-8",
                    )
                    all_profiles.append(profiles)
                    all_summaries.append(summary)
                    outputs[str(combo_dir)] = combo_dir

    if all_profiles:
        request.output_root.mkdir(parents=True, exist_ok=True)
        root_profiles = pd.concat(all_profiles, ignore_index=True)
        root_summary = pd.concat(all_summaries, ignore_index=True)
        _write_table(root_profiles, request.output_root / "material_coherence_profiles.csv")
        _write_table(root_summary, request.output_root / "track_material_coherence_summary.csv")
        _write_subset_tables(request.output_root, root_profiles)
        for plot_path in _plot_outputs(request.output_root, root_profiles, root_summary):
            outputs[f"figure:{plot_path.name}"] = plot_path
        _write_summary_md(request.output_root / "material_coherence_ep_validation_summary_zh.md", root_summary, request)
        (request.output_root / "material_coherence_manifest.json").write_text(
            json.dumps(_json_ready({"request": request.__dict__}), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        outputs["root_summary"] = request.output_root / "track_material_coherence_summary.csv"
    return outputs


def request_from_args(args) -> MaterialCoherenceRequest:
    return MaterialCoherenceRequest(
        result_root=Path(args.result_root),
        filter_root=Path(args.filter_root),
        output_root=Path(args.output_root),
        shapes=_split_csv(args.shapes),
        orientations=_split_csv(args.orientations),
        buoyancy_sources=_split_csv(args.buoyancy_sources),
        boundary_modes=_split_csv(args.boundary_mode),
        boundary_budget=str(args.boundary_budget),
        filter_template=str(args.filter_template),
        radial_bins=int(args.radial_bins),
        azimuth_bins=int(args.azimuth_bins),
        rmax=float(args.rmax),
        reference_lat=float(args.reference_lat),
        constant_n2=float(args.constant_n2),
        core_radius_over_R=float(args.core_radius_over_R),
        speed_core_quantile=float(args.speed_core_quantile),
        pv_core_quantile=float(args.pv_core_quantile),
        min_mask_fraction=float(args.min_mask_fraction),
        active_contour_iterations=int(args.active_contour_iterations),
        leakage_weight=float(args.leakage_weight),
        smoothness_weight=float(args.smoothness_weight),
        containment_weight=float(args.containment_weight),
        area_weight=float(args.area_weight),
        vertical_continuity_weight=float(args.vertical_continuity_weight),
        time_continuity_weight=float(args.time_continuity_weight),
        levelset_sigma_cells=float(args.levelset_sigma_cells),
        min_core_retention=float(args.min_core_retention),
        min_area_fraction=float(args.min_area_fraction),
        max_area_fraction=float(args.max_area_fraction),
        trajectory_window_days=int(args.trajectory_window_days),
        particle_spacing_km=float(args.particle_spacing_km),
        advection_step_hours=float(args.advection_step_hours),
        max_tracks_per_shape=int(args.max_tracks_per_shape),
        max_objectdays=int(args.max_objectdays),
        skip_missing=bool(args.skip_missing),
        dry_run=bool(args.dry_run),
    )
