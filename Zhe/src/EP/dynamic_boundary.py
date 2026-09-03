from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    from scipy import ndimage
except ModuleNotFoundError:  # pragma: no cover - dependency-light dry-run support.
    ndimage = None


BOUNDARY_MODES = ("threshold", "active_contour", "levelset_v2", "lagrangian_v1")


@dataclass(frozen=True)
class DynamicBoundaryConfig:
    mode: str = "threshold"
    iterations: int = 12
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
    tolerance: float = 1e-4

    def validate(self) -> None:
        if self.mode not in BOUNDARY_MODES:
            raise ValueError(f"boundary mode must be one of {BOUNDARY_MODES}")
        if self.iterations < 0:
            raise ValueError("active-contour iterations must be non-negative")
        if self.levelset_sigma_cells < 0:
            raise ValueError("levelset sigma must be non-negative")
        if not 0.0 < self.min_core_retention <= 1.0:
            raise ValueError("min core retention must be in (0, 1]")
        if not 0.0 < self.min_area_fraction < self.max_area_fraction <= 1.0:
            raise ValueError("area fraction bounds must satisfy 0 < min < max <= 1")


def neighbors4(mask: np.ndarray) -> np.ndarray:
    return (
        np.roll(mask, 1, axis=1)
        | np.roll(mask, -1, axis=1)
        | np.vstack([np.zeros((1, mask.shape[1]), dtype=bool), mask[:-1]])
        | np.vstack([mask[1:], np.zeros((1, mask.shape[1]), dtype=bool)])
    )


def edge_mask(mask: np.ndarray) -> np.ndarray:
    return mask & ~(
        np.roll(mask, 1, axis=1)
        & np.roll(mask, -1, axis=1)
        & np.vstack([np.zeros((1, mask.shape[1]), dtype=bool), mask[:-1]])
        & np.vstack([mask[1:], np.zeros((1, mask.shape[1]), dtype=bool)])
    )


def connected_component(candidate: np.ndarray, seed: tuple[int, int] | None) -> np.ndarray:
    if ndimage is None:
        raise ModuleNotFoundError("scipy is required for active material-boundary optimization")
    if seed is None or not candidate[seed]:
        return np.zeros_like(candidate, dtype=bool)
    labels, _ = ndimage.label(candidate, structure=np.ones((3, 3), dtype=bool))
    label = labels[seed]
    if label == 0:
        return np.zeros_like(candidate, dtype=bool)
    component = labels == label

    # ndimage.label is not periodic in azimuth. Stitch first/last theta columns by
    # keeping the seed component plus directly connected wrapped components.
    changed = True
    while changed:
        changed = False
        left_labels = set(np.unique(labels[component[:, 0], -1]))
        right_labels = set(np.unique(labels[component[:, -1], 0]))
        linked = (left_labels | right_labels) - {0}
        if linked:
            expanded = component | np.isin(labels, list(linked))
            changed = bool(np.any(expanded != component))
            component = expanded
    return component


def vertical_mask_roughness(mask: np.ndarray, previous_mask: np.ndarray | None = None, next_mask: np.ndarray | None = None) -> float:
    refs = [item for item in (previous_mask, next_mask) if item is not None and item.shape == mask.shape]
    if not refs:
        return 0.0
    diffs = [np.mean(np.logical_xor(mask, ref)) for ref in refs]
    return float(np.nanmean(diffs))


def mask_mismatch(mask: np.ndarray, reference_mask: np.ndarray | None) -> float:
    if reference_mask is None or reference_mask.shape != mask.shape:
        return 0.0
    return float(np.mean(np.logical_xor(mask.astype(bool), reference_mask.astype(bool))))


def levelset_candidate_masks(
    mask: np.ndarray,
    core_domain: np.ndarray,
    seed: tuple[int, int] | None,
    *,
    sigma_cells: float,
) -> list[np.ndarray]:
    if ndimage is None:
        raise ModuleNotFoundError("scipy is required for level-set material-boundary optimization")
    if seed is None or not np.any(mask):
        return []
    phi = ndimage.distance_transform_edt(mask) - ndimage.distance_transform_edt(~mask)
    if sigma_cells > 0:
        phi = ndimage.gaussian_filter(phi.astype(float), sigma=float(sigma_cells), mode=("nearest", "wrap"))
    candidates: list[np.ndarray] = []
    for threshold in (-1.0, -0.5, 0.0, 0.5, 1.0):
        candidate = connected_component((phi >= threshold) & core_domain, seed)
        if np.any(candidate):
            candidates.append(candidate)
    closed = ndimage.binary_closing(mask, structure=np.ones((3, 3), dtype=bool))
    opened = ndimage.binary_opening(mask, structure=np.ones((3, 3), dtype=bool))
    for candidate in (closed, opened):
        component = connected_component(candidate & core_domain, seed)
        if np.any(component):
            candidates.append(component)
    return candidates


def boundary_normal_from_level_set(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if ndimage is None:
        raise ModuleNotFoundError("scipy is required for active material-boundary optimization")
    phi = ndimage.distance_transform_edt(mask) - ndimage.distance_transform_edt(~mask)
    dphi_dr, dphi_dt = np.gradient(phi)
    norm = np.hypot(dphi_dr, dphi_dt)
    nr = np.divide(dphi_dr, norm, out=np.zeros_like(dphi_dr, dtype=float), where=norm > 0)
    nt = np.divide(dphi_dt, norm, out=np.zeros_like(dphi_dt, dtype=float), where=norm > 0)
    return nr, nt


def normal_velocity(
    *,
    u: np.ndarray,
    v: np.ndarray,
    mean_u: float,
    mean_v: float,
    x_km: np.ndarray,
    y_km: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    nr, nt = boundary_normal_from_level_set(mask)
    radius = np.hypot(x_km, y_km)
    er = np.divide(x_km, radius, out=np.zeros_like(x_km, dtype=float), where=radius > 0)
    et = np.divide(y_km, radius, out=np.zeros_like(y_km, dtype=float), where=radius > 0)
    theta_x = -et
    theta_y = er
    nx = nr * er + nt * theta_x
    ny = nr * et + nt * theta_y
    return (u - mean_u) * nx + (v - mean_v) * ny


def boundary_flux_metrics(
    *,
    mask: np.ndarray,
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
) -> dict[str, float]:
    edge = edge_mask(mask)
    if not np.any(edge):
        return {
            "edge_cell_count": 0.0,
            "leakage_mean_abs_ms": np.nan,
            "leakage_rms_ms": np.nan,
            "signed_leakage_ms": np.nan,
            "leakage_over_tangential_speed": np.nan,
            "heat_boundary_flux_proxy": np.nan,
            "heat_boundary_flux_integral_proxy": np.nan,
            "pv_boundary_flux_proxy": np.nan,
            "pv_boundary_flux_integral_proxy": np.nan,
            "buoyancy_boundary_flux_proxy": np.nan,
            "buoyancy_boundary_flux_integral_proxy": np.nan,
            "momentum_x_boundary_flux_proxy": np.nan,
            "momentum_y_boundary_flux_proxy": np.nan,
            "momentum_x_boundary_flux_integral_proxy": np.nan,
            "momentum_y_boundary_flux_integral_proxy": np.nan,
            "abs_leakage_integral_proxy": np.nan,
            "signed_leakage_integral_proxy": np.nan,
            "boundary_flux_over_internal_flux": np.nan,
        }
    un = normal_velocity(u=u, v=v, mean_u=mean_u, mean_v=mean_v, x_km=x_km, y_km=y_km, mask=mask)
    speed = np.hypot(u - mean_u, v - mean_v)
    valid = edge & np.isfinite(un)
    if not np.any(valid):
        return {
            "edge_cell_count": float(np.count_nonzero(edge)),
            "leakage_mean_abs_ms": np.nan,
            "leakage_rms_ms": np.nan,
            "signed_leakage_ms": np.nan,
            "leakage_over_tangential_speed": np.nan,
            "heat_boundary_flux_proxy": np.nan,
            "heat_boundary_flux_integral_proxy": np.nan,
            "pv_boundary_flux_proxy": np.nan,
            "pv_boundary_flux_integral_proxy": np.nan,
            "buoyancy_boundary_flux_proxy": np.nan,
            "buoyancy_boundary_flux_integral_proxy": np.nan,
            "momentum_x_boundary_flux_proxy": np.nan,
            "momentum_y_boundary_flux_proxy": np.nan,
            "momentum_x_boundary_flux_integral_proxy": np.nan,
            "momentum_y_boundary_flux_integral_proxy": np.nan,
            "abs_leakage_integral_proxy": np.nan,
            "signed_leakage_integral_proxy": np.nan,
            "boundary_flux_over_internal_flux": np.nan,
        }
    abs_un = np.abs(un[valid])
    tangential = np.sqrt(np.maximum(speed[valid] ** 2 - un[valid] ** 2, 0.0))
    heat_flux = np.nanmean(theta_prime[valid] * un[valid]) if np.any(np.isfinite(theta_prime[valid])) else np.nan
    pv_flux = np.nanmean(q_proxy[valid] * un[valid]) if np.any(np.isfinite(q_proxy[valid])) else np.nan
    b_flux = np.nanmean(buoyancy[valid] * un[valid]) if np.any(np.isfinite(buoyancy[valid])) else np.nan
    mom_x_flux = np.nanmean((u[valid] - mean_u) * un[valid]) if np.any(np.isfinite(u[valid])) else np.nan
    mom_y_flux = np.nanmean((v[valid] - mean_v) * un[valid]) if np.any(np.isfinite(v[valid])) else np.nan
    boundary_scale = np.nanmean(np.abs(pv_flux)) if np.isfinite(pv_flux) else np.nan
    return {
        "edge_cell_count": float(np.count_nonzero(edge)),
        "leakage_mean_abs_ms": float(np.nanmean(abs_un)),
        "leakage_rms_ms": float(np.sqrt(np.nanmean(un[valid] ** 2))),
        "signed_leakage_ms": float(np.nanmean(un[valid])),
        "leakage_over_tangential_speed": float(np.nanmean(abs_un) / (np.nanmean(tangential) + 1e-12)),
        "heat_boundary_flux_proxy": float(heat_flux) if np.isfinite(heat_flux) else np.nan,
        "heat_boundary_flux_integral_proxy": float(np.nansum(theta_prime[valid] * un[valid])),
        "pv_boundary_flux_proxy": float(pv_flux) if np.isfinite(pv_flux) else np.nan,
        "pv_boundary_flux_integral_proxy": float(np.nansum(q_proxy[valid] * un[valid])),
        "buoyancy_boundary_flux_proxy": float(b_flux) if np.isfinite(b_flux) else np.nan,
        "buoyancy_boundary_flux_integral_proxy": float(np.nansum(buoyancy[valid] * un[valid])),
        "momentum_x_boundary_flux_proxy": float(mom_x_flux) if np.isfinite(mom_x_flux) else np.nan,
        "momentum_y_boundary_flux_proxy": float(mom_y_flux) if np.isfinite(mom_y_flux) else np.nan,
        "momentum_x_boundary_flux_integral_proxy": float(np.nansum((u[valid] - mean_u) * un[valid])),
        "momentum_y_boundary_flux_integral_proxy": float(np.nansum((v[valid] - mean_v) * un[valid])),
        "abs_leakage_integral_proxy": float(np.nansum(abs_un)),
        "signed_leakage_integral_proxy": float(np.nansum(un[valid])),
        "boundary_flux_over_internal_flux": float(boundary_scale / (abs(internal_flux_scale) + 1e-18))
        if np.isfinite(boundary_scale)
        else np.nan,
    }


def optimize_boundary(
    *,
    initial_mask: np.ndarray,
    core_domain: np.ndarray,
    seed: tuple[int, int] | None,
    u: np.ndarray,
    v: np.ndarray,
    q_proxy: np.ndarray,
    speed_core: np.ndarray,
    pv_core: np.ndarray,
    x_km: np.ndarray,
    y_km: np.ndarray,
    mean_u: float,
    mean_v: float,
    config: DynamicBoundaryConfig,
    previous_mask: np.ndarray | None = None,
    next_mask: np.ndarray | None = None,
    time_reference_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, float | str]]:
    config.validate()
    if config.mode == "threshold":
        return initial_mask.astype(bool), {
            "boundary_mode": "threshold",
            "boundary_status": "baseline",
            "active_contour_iterations_used": 0.0,
            "boundary_energy_initial": np.nan,
            "boundary_energy_final": np.nan,
            "boundary_energy_reduction_fraction": np.nan,
            "pv_core_retention": np.nan,
            "weak_core_retention": np.nan,
            "area_fraction_of_core_domain": np.count_nonzero(initial_mask) / max(1, np.count_nonzero(core_domain)),
        }

    if seed is None or not np.any(initial_mask):
        return initial_mask.astype(bool), {"boundary_mode": config.mode, "boundary_status": "empty_initial_mask"}

    speed_core_count = max(1, int(np.count_nonzero(speed_core)))
    pv_core_count = max(1, int(np.count_nonzero(pv_core)))
    core_count = max(1, int(np.count_nonzero(core_domain)))

    def score(mask: np.ndarray) -> tuple[float, dict[str, float]]:
        mask = mask.astype(bool)
        edge = edge_mask(mask)
        candidate_mean_u = float(np.nanmean(u[mask])) if np.any(mask & np.isfinite(u)) else mean_u
        candidate_mean_v = float(np.nanmean(v[mask])) if np.any(mask & np.isfinite(v)) else mean_v
        un = normal_velocity(
            u=u,
            v=v,
            mean_u=candidate_mean_u,
            mean_v=candidate_mean_v,
            x_km=x_km,
            y_km=y_km,
            mask=mask,
        )
        valid = edge & np.isfinite(un)
        leakage = float(np.nanmean(un[valid] ** 2)) if np.any(valid) else np.inf
        area_fraction = float(np.count_nonzero(mask) / core_count)
        perimeter_fraction = float(np.count_nonzero(edge) / core_count)
        speed_retention = float(np.count_nonzero(mask & speed_core) / speed_core_count)
        pv_retention = float(np.count_nonzero(mask & pv_core) / pv_core_count)
        containment_loss = max(0.0, config.min_core_retention - speed_retention) + max(
            0.0, config.min_core_retention - pv_retention
        )
        area_penalty = max(0.0, config.min_area_fraction - area_fraction) + max(
            0.0, area_fraction - config.max_area_fraction
        )
        vertical_penalty = vertical_mask_roughness(mask, previous_mask, next_mask)
        time_penalty = mask_mismatch(mask, time_reference_mask)
        energy = (
            config.leakage_weight * leakage
            + config.smoothness_weight * perimeter_fraction
            + config.containment_weight * containment_loss
            + config.area_weight * area_penalty
            + config.vertical_continuity_weight * vertical_penalty
            + config.time_continuity_weight * time_penalty
        )
        return energy, {
            "leakage_energy": leakage,
            "area_fraction_of_core_domain": area_fraction,
            "perimeter_fraction_of_core_domain": perimeter_fraction,
            "weak_core_retention": speed_retention,
            "pv_core_retention": pv_retention,
            "vertical_mask_roughness": vertical_penalty,
            "time_mask_roughness": time_penalty,
        }

    current = connected_component(initial_mask & core_domain, seed)
    best_energy, best_meta = score(current)
    initial_energy = best_energy
    iterations_used = 0

    for iteration in range(config.iterations):
        candidates = [current]
        candidates.append(connected_component((current | neighbors4(current)) & core_domain, seed))
        candidates.append(connected_component((current & ~edge_mask(current)) & core_domain, seed))
        if config.mode in {"levelset_v2", "lagrangian_v1"}:
            candidates.extend(
                levelset_candidate_masks(
                    current,
                    core_domain,
                    seed,
                    sigma_cells=config.levelset_sigma_cells,
                )
            )
            refs = [item for item in (previous_mask, next_mask) if item is not None and item.shape == current.shape]
            if refs:
                vote = np.zeros_like(current, dtype=int)
                for ref in refs:
                    vote += ref.astype(int)
                consensus = vote >= max(1, len(refs))
                candidates.append(connected_component((current | consensus) & core_domain, seed))
                candidates.append(connected_component((current & (consensus | speed_core | pv_core)) & core_domain, seed))
            if config.mode == "lagrangian_v1" and time_reference_mask is not None and time_reference_mask.shape == current.shape:
                time_ref = time_reference_mask.astype(bool)
                time_band = time_ref | neighbors4(time_ref)
                candidates.append(connected_component((current | time_ref) & core_domain, seed))
                candidates.append(connected_component((current & (time_band | speed_core | pv_core)) & core_domain, seed))
                candidates.append(connected_component(((current | time_band) & (core_domain | speed_core | pv_core)), seed))
        edge = edge_mask(current)
        un = normal_velocity(u=u, v=v, mean_u=mean_u, mean_v=mean_v, x_km=x_km, y_km=y_km, mask=current)
        edge_values = np.abs(un[edge & np.isfinite(un)])
        if edge_values.size:
            for cutoff_q in (0.60, 0.75, 0.90):
                cutoff = float(np.nanquantile(edge_values, cutoff_q))
                high_leak = edge & (np.abs(un) >= cutoff)
                candidates.append(connected_component((current & ~high_leak) & core_domain, seed))
        frontier = neighbors4(current) & core_domain & ~current
        frontier_values = np.abs(un[frontier & np.isfinite(un)])
        if frontier_values.size:
            for cutoff_q in (0.25, 0.40, 0.55):
                cutoff = float(np.nanquantile(frontier_values, cutoff_q))
                low_leak_frontier = frontier & (np.abs(un) <= cutoff)
                candidates.append(connected_component((current | low_leak_frontier) & core_domain, seed))

        scored = []
        for candidate in candidates:
            if np.count_nonzero(candidate) == 0 or not candidate[seed]:
                continue
            energy, meta = score(candidate)
            scored.append((energy, candidate, meta))
        if not scored:
            break
        energy, candidate, meta = min(scored, key=lambda item: item[0])
        improvement = (best_energy - energy) / (abs(best_energy) + 1e-12)
        if energy < best_energy and improvement > config.tolerance:
            current = candidate
            best_energy = energy
            best_meta = meta
            iterations_used = iteration + 1
        else:
            break

    status = "ok"
    if best_meta.get("pv_core_retention", 0.0) < config.min_core_retention:
        status = "pv_retention_below_target"
    if best_meta.get("weak_core_retention", 0.0) < config.min_core_retention:
        status = "weak_core_retention_below_target"
    return current.astype(bool), {
        "boundary_mode": config.mode,
        "boundary_status": status,
        "active_contour_iterations_used": float(iterations_used),
        "levelset_sigma_cells": float(config.levelset_sigma_cells) if config.mode in {"levelset_v2", "lagrangian_v1"} else np.nan,
        "vertical_continuity_weight": float(config.vertical_continuity_weight),
        "time_continuity_weight": float(config.time_continuity_weight),
        "time_constraint_skipped": "false" if time_reference_mask is not None else "true",
        "boundary_energy_initial": float(initial_energy),
        "boundary_energy_final": float(best_energy),
        "boundary_energy_reduction_fraction": float((initial_energy - best_energy) / (abs(initial_energy) + 1e-12)),
        **best_meta,
    }
