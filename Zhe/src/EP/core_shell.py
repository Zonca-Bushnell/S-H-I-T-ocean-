from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import pandas as pd
except ModuleNotFoundError:  # pragma: no cover - dependency-light dry-run support.
    pd = None

try:
    from scipy import ndimage
except ModuleNotFoundError:  # pragma: no cover - checked at runtime.
    ndimage = None

from .contracts import (
    AXIS_SOURCES,
    BUOYANCY_SOURCES,
    DEFAULT_RESULT_ROOT,
    ORIENTATIONS,
    EPFluxConfig,
    RHO0,
    axis_source_filename,
    default_me_liutex_root,
    default_radial_seed_root,
    shape_output_name,
)
from .dynamic_boundary import BOUNDARY_MODES, boundary_flux_metrics, connected_component, edge_mask, normal_velocity
from .material_volume import (
    BOUNDARY_BUDGETS,
    CP0,
    MaterialVolumeRequest,
    _compute_one_slice,
    _full_boundary_flux_budget,
    _json_ready,
    _weighted_mean,
    _write_table,
)


DEFAULT_CORE_SHELL_OUTPUT_ROOT = Path(
    "/root/autodl-fs/kuroshiou/EP-FLUX/core_shell_transport_tilt_partition"
)


@dataclass(frozen=True)
class CoreShellRequest:
    result_root: Path = DEFAULT_RESULT_ROOT
    output_root: Path = DEFAULT_CORE_SHELL_OUTPUT_ROOT
    filter_root: Path = Path("/root/autodl-fs/kuroshiou/Filter")
    filter_template: str = "global_phy_{year}_bandpass_30_180d.nc"
    shapes: tuple[str, ...] = ("coherent", "upright_like")
    axis_sources: tuple[str, ...] = ("radial_seed",)
    orientations: tuple[str, ...] = ("turned",)
    buoyancy_sources: tuple[str, ...] = ("thermal_wind",)
    tau_values: tuple[float, ...] | None = None
    reference_lat: float = 30.0
    constant_n2: float = 2.0e-5
    n2_profile: str | None = "auto"
    inner_boundary_mode: str = "levelset_v2"
    boundary_budget: str = "full_3d"
    core_radius_over_R: float = 1.5
    shell_outer_radius_over_R: float = 1.5
    speed_core_quantile: float = 0.45
    pv_core_quantile: float = 0.70
    pv_shell_quantile: float = 0.80
    shell_dilation_cells: int = 2
    min_mask_fraction: float = 0.01
    min_core_retention: float = 0.75
    object_aggregate_transport: bool = True
    object_aggregate_max_days: int = 0
    object_aggregate_max_objects: int = 0
    skip_missing: bool = False
    dry_run: bool = False


def _split_csv(value: str | tuple[str, ...] | list[str]) -> tuple[str, ...]:
    if isinstance(value, (tuple, list)):
        return tuple(str(v).strip() for v in value if str(v).strip())
    return tuple(part.strip() for part in str(value).split(",") if part.strip())


def _parse_tau_values(value: str | None) -> tuple[float, ...] | None:
    if value in (None, ""):
        return None
    return tuple(float(part.strip()) for part in str(value).split(",") if part.strip())


def _require_runtime() -> None:
    if pd is None:
        raise ModuleNotFoundError("pandas is required for core-shell EP validation")
    if ndimage is None:
        raise ModuleNotFoundError("scipy is required for core-shell EP validation")


def _validate_request(request: CoreShellRequest) -> None:
    bad_axis = sorted(set(request.axis_sources) - set(AXIS_SOURCES))
    bad_orient = sorted(set(request.orientations) - set(ORIENTATIONS))
    bad_buoy = sorted(set(request.buoyancy_sources) - set(BUOYANCY_SOURCES))
    if bad_axis or bad_orient or bad_buoy:
        raise ValueError(f"Bad options: axis={bad_axis}, orientation={bad_orient}, buoyancy={bad_buoy}")
    if request.inner_boundary_mode not in BOUNDARY_MODES:
        raise ValueError(f"inner_boundary_mode must be one of {BOUNDARY_MODES}")
    if request.boundary_budget not in BOUNDARY_BUDGETS:
        raise ValueError(f"boundary_budget must be one of {BOUNDARY_BUDGETS}")
    if request.shell_outer_radius_over_R < request.core_radius_over_R:
        raise ValueError("shell_outer_radius_over_R must be >= core_radius_over_R")


def _tau_grid_for_combo(result_root: Path, shape: str, orientation: str, tau_values: tuple[float, ...] | None) -> np.ndarray:
    if tau_values is not None:
        return np.asarray(tau_values, dtype=float)
    from .fields import RepresentativeVortexDataset

    output_name = shape_output_name(shape)
    dataset = RepresentativeVortexDataset.load(
        default_me_liutex_root(result_root, output_name, orientation) / "azimuthal_representative_velocity.npz",
        default_radial_seed_root(result_root, output_name),
    )
    return np.asarray(dataset.tau_grid, dtype=float)


def _combo_output(root: Path, shape: str, axis_source: str, orientation: str, buoyancy_source: str) -> Path:
    return root / shape / axis_source / orientation / buoyancy_source


def _dilate(mask: np.ndarray, cells: int) -> np.ndarray:
    out = mask.astype(bool)
    for _ in range(max(0, int(cells))):
        out = ndimage.binary_dilation(out, structure=np.ones((3, 3), dtype=bool))
    return out


def _pv_centroid_seed(pv_core: np.ndarray, q_abs: np.ndarray) -> tuple[int, int] | None:
    valid = pv_core & np.isfinite(q_abs)
    if not np.any(valid):
        return None
    flat = int(np.nanargmax(np.where(valid, q_abs, -np.inf)))
    return np.unravel_index(flat, q_abs.shape)


def _build_shell_mask(
    *,
    inner_mask: np.ndarray,
    core_domain: np.ndarray,
    pv_core: np.ndarray,
    q_abs: np.ndarray,
    dilation_cells: int,
) -> tuple[np.ndarray, str]:
    outside_inner = core_domain & ~inner_mask
    if not np.any(outside_inner):
        return np.zeros_like(inner_mask, dtype=bool), "empty_outside_inner"
    bridge = _dilate(inner_mask, dilation_cells) & core_domain
    candidate = outside_inner & (pv_core | bridge)
    if not np.any(candidate):
        return np.zeros_like(inner_mask, dtype=bool), "empty_shell_candidate"
    seed = _pv_centroid_seed(pv_core & outside_inner, q_abs)
    if seed is None or not candidate[seed]:
        seed = _pv_centroid_seed(candidate, q_abs)
    shell = connected_component(candidate, seed) if seed is not None else np.zeros_like(candidate, dtype=bool)
    if not np.any(shell):
        return shell, "empty_connected_shell"
    return shell & outside_inner, "ok"


def _retention(mask: np.ndarray, target: np.ndarray, weights: np.ndarray | None = None) -> float:
    valid_target = target.astype(bool)
    if not np.any(valid_target):
        return np.nan
    if weights is None:
        return float(np.count_nonzero(mask & valid_target) / np.count_nonzero(valid_target))
    weight = np.where(valid_target & np.isfinite(weights), weights, 0.0)
    total = float(np.nansum(weight))
    if total <= 0:
        return np.nan
    return float(np.nansum(weight[mask]) / total)


def _interface_metrics(inner_mask: np.ndarray, shell_mask: np.ndarray, u: np.ndarray, v: np.ndarray, x_km: np.ndarray, y_km: np.ndarray) -> dict[str, float]:
    if not np.any(inner_mask) or not np.any(shell_mask):
        return {"core_shell_interface_cell_count": 0.0, "core_shell_interface_abs_un_ms": np.nan}
    interface = edge_mask(inner_mask) & _dilate(shell_mask, 1)
    if not np.any(interface):
        return {"core_shell_interface_cell_count": 0.0, "core_shell_interface_abs_un_ms": np.nan}
    mean_u = _weighted_mean(u, inner_mask)
    mean_v = _weighted_mean(v, inner_mask)
    un = normal_velocity(u=u, v=v, mean_u=mean_u, mean_v=mean_v, x_km=x_km, y_km=y_km, mask=inner_mask)
    return {
        "core_shell_interface_cell_count": float(np.count_nonzero(interface)),
        "core_shell_interface_abs_un_ms": _weighted_mean(np.abs(un), interface),
    }


def _region_internal_stats(
    *,
    region: str,
    mask: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    speed: np.ndarray,
    q_proxy: np.ndarray,
    buoyancy: np.ndarray,
) -> dict[str, float | str]:
    if not np.any(mask):
        return {
            "region": region,
            "area_cell_count": 0.0,
            "mean_speed_ms": np.nan,
            "mean_abs_q_proxy": np.nan,
            "mean_abs_buoyancy": np.nan,
            "momentum_flux_xy_proxy": np.nan,
            "pv_flux_magnitude_proxy": np.nan,
            "buoyancy_flux_magnitude_proxy": np.nan,
        }
    mean_u = _weighted_mean(u, mask)
    mean_v = _weighted_mean(v, mask)
    mean_q = _weighted_mean(q_proxy, mask)
    mean_b = _weighted_mean(buoyancy, mask)
    up = u - mean_u
    vp = v - mean_v
    qp = q_proxy - mean_q
    bp = buoyancy - mean_b
    px = _weighted_mean(up * qp, mask)
    py = _weighted_mean(vp * qp, mask)
    bx = _weighted_mean(up * bp, mask)
    by = _weighted_mean(vp * bp, mask)
    return {
        "region": region,
        "area_cell_count": float(np.count_nonzero(mask)),
        "mean_speed_ms": _weighted_mean(speed, mask),
        "mean_abs_q_proxy": _weighted_mean(np.abs(q_proxy), mask),
        "mean_abs_buoyancy": _weighted_mean(np.abs(buoyancy), mask),
        "momentum_flux_xy_proxy": _weighted_mean(up * vp, mask),
        "pv_flux_magnitude_proxy": float(np.hypot(px, py)),
        "buoyancy_flux_magnitude_proxy": float(np.hypot(bx, by)),
    }


def _region_flux_partition_stats(
    *,
    region: str,
    mask: np.ndarray,
    v_rot: np.ndarray,
    theta_prime: np.ndarray | None,
    q_prime: np.ndarray,
) -> dict[str, float | str]:
    row: dict[str, float | str] = {"region": region}
    if not np.any(mask):
        row.update(
            {
                "transport_valid_cell_count": 0.0,
                "theta_prime_status": "missing" if theta_prime is None else "empty_region",
                "product_mean_heat": np.nan,
                "mean_product_heat": np.nan,
                "covariance_heat": np.nan,
                "product_mean_pv": np.nan,
                "mean_product_pv": np.nan,
                "covariance_pv": np.nan,
            }
        )
        return row

    valid_pv = mask & np.isfinite(v_rot) & np.isfinite(q_prime)
    mean_v_pv = _weighted_mean(v_rot, valid_pv)
    mean_q = _weighted_mean(q_prime, valid_pv)
    product_pv = _weighted_mean(v_rot * q_prime, valid_pv)
    row.update(
        {
            "transport_valid_cell_count": float(np.count_nonzero(valid_pv)),
            "product_mean_pv": product_pv,
            "mean_product_pv": mean_v_pv * mean_q,
            "covariance_pv": product_pv - mean_v_pv * mean_q,
        }
    )

    if theta_prime is None:
        row.update(
            {
                "theta_prime_status": "missing",
                "product_mean_heat": np.nan,
                "mean_product_heat": np.nan,
                "covariance_heat": np.nan,
            }
        )
        return row

    valid_heat = mask & np.isfinite(v_rot) & np.isfinite(theta_prime)
    mean_v_heat = _weighted_mean(v_rot, valid_heat)
    mean_theta = _weighted_mean(theta_prime, valid_heat)
    product_heat = RHO0 * CP0 * _weighted_mean(v_rot * theta_prime, valid_heat)
    mean_product_heat = RHO0 * CP0 * mean_v_heat * mean_theta
    row.update(
        {
            "theta_prime_status": "available",
            "heat_valid_cell_count": float(np.count_nonzero(valid_heat)),
            "product_mean_heat": product_heat,
            "mean_product_heat": mean_product_heat,
            "covariance_heat": product_heat - mean_product_heat,
        }
    )
    return row


def _region_ep_tilt_stats(
    *,
    region: str,
    mask: np.ndarray,
    fz_ordinary: np.ndarray,
    fz_tilted: np.ndarray,
    fz_tilt_correction: np.ndarray,
) -> dict[str, float | str]:
    if not np.any(mask):
        return {
            "region": region,
            "ep_valid_cell_count": 0.0,
            "F_z_ordinary_region": np.nan,
            "F_z_tilted_region": np.nan,
            "F_z_tilt_correction_region": np.nan,
            "abs_F_z_ordinary_region": np.nan,
            "abs_F_z_tilt_correction_region": np.nan,
            "tilt_correction_over_ordinary": np.nan,
        }
    valid = mask & np.isfinite(fz_ordinary) & np.isfinite(fz_tilted) & np.isfinite(fz_tilt_correction)
    ordinary = _weighted_mean(fz_ordinary, valid)
    tilted = _weighted_mean(fz_tilted, valid)
    correction = _weighted_mean(fz_tilt_correction, valid)
    abs_ordinary = _weighted_mean(np.abs(fz_ordinary), valid)
    abs_correction = _weighted_mean(np.abs(fz_tilt_correction), valid)
    return {
        "region": region,
        "ep_valid_cell_count": float(np.count_nonzero(valid)),
        "F_z_ordinary_region": ordinary,
        "F_z_tilted_region": tilted,
        "F_z_tilt_correction_region": correction,
        "abs_F_z_ordinary_region": abs_ordinary,
        "abs_F_z_tilt_correction_region": abs_correction,
        "tilt_correction_over_ordinary": abs_correction / (abs_ordinary + 1e-30),
    }


def _add_fraction_of_total_abs(
    rows: list[dict[str, float | str]],
    *,
    value_column: str,
    fraction_column: str,
) -> None:
    total = 0.0
    for row in rows:
        if row.get("region") in {"inner_core", "pv_shell"}:
            value = float(row.get(value_column, np.nan))
            if np.isfinite(value):
                total += abs(value)
    for row in rows:
        value = float(row.get(value_column, np.nan))
        row[fraction_column] = abs(value) / (total + 1e-30) if np.isfinite(value) else np.nan


def _nanmedian_no_warning(values: np.ndarray) -> float:
    data = np.asarray(values, dtype=float)
    finite = data[np.isfinite(data)]
    return float(np.median(finite)) if finite.size else np.nan


def _budget_for_region(
    *,
    rep,
    mask_stack: np.ndarray,
    q_proxy: np.ndarray,
    buoyancy: np.ndarray,
    x_km: np.ndarray,
    y_km: np.ndarray,
    region: str,
) -> "pd.DataFrame":
    budget = _full_boundary_flux_budget(
        rep=rep,
        masks=mask_stack.astype(bool),
        q_proxy=q_proxy,
        buoyancy=buoyancy,
        x_km=x_km,
        y_km=y_km,
    )
    budget["region"] = region
    return budget


def _compute_core_shell_for_slice(
    *,
    rep,
    axis,
    q_proxy: np.ndarray,
    buoyancy: np.ndarray,
    inner_masks: np.ndarray,
    ep_terms,
    request: CoreShellRequest,
) -> tuple["pd.DataFrame", dict[str, np.ndarray]]:
    x_km, y_km = rep.mesh_xy_km
    radial_over_R = rep.radial_m[:, None] / max(float(rep.radius_m), 1.0)
    outer_domain_by_depth = []
    shell_masks = []
    combined_masks = []
    rows: list[dict[str, float | str]] = []

    for iz in range(rep.depth_m.size):
        finite = np.isfinite(rep.speed[iz]) & np.isfinite(q_proxy[iz])
        core_domain = finite & (radial_over_R <= request.shell_outer_radius_over_R)
        q_abs = np.abs(q_proxy[iz])
        if np.any(core_domain):
            pv_shell_threshold = float(np.nanquantile(q_abs[core_domain], request.pv_shell_quantile))
        else:
            pv_shell_threshold = np.nan
        pv_core = core_domain & np.isfinite(q_abs) & (q_abs >= pv_shell_threshold)
        inner = inner_masks[iz].astype(bool) & core_domain
        shell, shell_status = _build_shell_mask(
            inner_mask=inner,
            core_domain=core_domain,
            pv_core=pv_core,
            q_abs=q_abs,
            dilation_cells=request.shell_dilation_cells,
        )
        combined = (inner | shell) & core_domain
        outer_domain_by_depth.append(core_domain)
        shell_masks.append(shell)
        combined_masks.append(combined)

        q_weight = np.where(np.isfinite(q_abs), q_abs, 0.0)
        weak_threshold = (
            float(np.nanquantile(rep.speed[iz][core_domain], request.speed_core_quantile))
            if np.any(core_domain)
            else np.nan
        )
        weak_core = core_domain & np.isfinite(rep.speed[iz]) & (rep.speed[iz] <= weak_threshold)
        region_rows = []
        for region_name, mask in (
            ("inner_core", inner),
            ("pv_shell", shell),
            ("combined_volume", combined),
        ):
            stats = _region_internal_stats(
                region=region_name,
                mask=mask,
                u=rep.u[iz],
                v=rep.v[iz],
                speed=rep.speed[iz],
                q_proxy=q_proxy[iz],
                buoyancy=buoyancy[iz],
            )
            stats.update(
                _region_flux_partition_stats(
                    region=region_name,
                    mask=mask,
                    v_rot=rep.v[iz],
                    theta_prime=None if rep.theta_prime is None else rep.theta_prime[iz],
                    q_prime=ep_terms.pv_prime[iz],
                )
            )
            stats.update(
                _region_ep_tilt_stats(
                    region=region_name,
                    mask=mask,
                    fz_ordinary=ep_terms.F_z_ordinary[iz],
                    fz_tilted=ep_terms.F_z_tilted[iz],
                    fz_tilt_correction=ep_terms.F_z_tilt_correction[iz],
                )
            )
            stats.update(
                {
                    "depth_index": int(iz),
                    "depth_m": float(rep.depth_m[iz]),
                    "axis_x_km": float(axis.x_km[iz]),
                    "axis_y_km": float(axis.y_km[iz]),
                    "radius_m": float(rep.radius_m),
                    "shell_status": shell_status,
                    "pv_shell_threshold": pv_shell_threshold,
                    "weak_speed_threshold_ms": weak_threshold,
                    "pv_high_quantile_retention": _retention(mask, pv_core, q_weight),
                    "pv_abs_retention": _retention(mask, core_domain, q_weight),
                    "weak_core_retention": _retention(mask, weak_core),
                    "mask_fraction_of_domain": float(np.count_nonzero(mask) / max(1, np.count_nonzero(core_domain))),
                }
            )
            region_rows.append(stats)
        _add_fraction_of_total_abs(
            region_rows,
            value_column="F_z_tilt_correction_region",
            fraction_column="region_fraction_of_total_abs_tilt_correction",
        )
        _add_fraction_of_total_abs(
            region_rows,
            value_column="covariance_pv",
            fraction_column="region_fraction_of_total_abs_pv_covariance",
        )
        _add_fraction_of_total_abs(
            region_rows,
            value_column="covariance_heat",
            fraction_column="region_fraction_of_total_abs_heat_covariance",
        )
        interface = _interface_metrics(inner, shell, rep.u[iz], rep.v[iz], x_km, y_km)
        for stats in region_rows:
            stats.update(interface)
            rows.append(stats)

    masks = {
        "inner_core": inner_masks.astype(bool),
        "pv_shell": np.asarray(shell_masks, dtype=bool),
        "combined_volume": np.asarray(combined_masks, dtype=bool),
        "outer_domain": np.asarray(outer_domain_by_depth, dtype=bool),
    }
    table = pd.DataFrame(rows)
    budgets = []
    for region_name in ("inner_core", "pv_shell", "combined_volume"):
        budget = _budget_for_region(
            rep=rep,
            mask_stack=masks[region_name],
            q_proxy=q_proxy,
            buoyancy=buoyancy,
            x_km=x_km,
            y_km=y_km,
            region=region_name,
        )
        budgets.append(budget)
    budget_all = pd.concat(budgets, ignore_index=True)
    table = table.merge(budget_all, on=["depth_index", "depth_m", "region"], how="left")
    return table, masks


def _empty_object_region_accumulator(depth_count: int, regions: tuple[str, ...]) -> dict[str, dict[str, np.ndarray]]:
    fields = ("sum_v", "sum_theta", "sum_q", "sum_vtheta", "sum_vq", "sum_v2", "sum_theta2", "sum_q2", "count")
    return {
        region: {name: np.zeros(depth_count, dtype="float64") for name in fields}
        for region in regions
    }


def _add_object_region_terms(
    accumulator: dict[str, dict[str, np.ndarray]],
    *,
    masks: dict[str, np.ndarray],
    depth_indices: np.ndarray,
    vrot: np.ndarray,
    theta_prime: np.ndarray,
    q_prime: np.ndarray,
    weight: float,
) -> None:
    for local_iz, depth_index in enumerate(depth_indices.astype(int)):
        for region, region_masks in masks.items():
            if region == "outer_domain" or depth_index < 0 or depth_index >= region_masks.shape[0]:
                continue
            mask = region_masks[depth_index].astype(bool)
            valid = mask & np.isfinite(vrot[local_iz]) & np.isfinite(theta_prime[local_iz]) & np.isfinite(q_prime[local_iz])
            if not np.any(valid):
                continue
            v_values = vrot[local_iz][valid]
            theta_values = theta_prime[local_iz][valid]
            q_values = q_prime[local_iz][valid]
            target = accumulator[region]
            target["sum_v"][depth_index] += weight * float(np.sum(v_values))
            target["sum_theta"][depth_index] += weight * float(np.sum(theta_values))
            target["sum_q"][depth_index] += weight * float(np.sum(q_values))
            target["sum_vtheta"][depth_index] += weight * float(np.sum(v_values * theta_values))
            target["sum_vq"][depth_index] += weight * float(np.sum(v_values * q_values))
            target["sum_v2"][depth_index] += weight * float(np.sum(v_values * v_values))
            target["sum_theta2"][depth_index] += weight * float(np.sum(theta_values * theta_values))
            target["sum_q2"][depth_index] += weight * float(np.sum(q_values * q_values))
            target["count"][depth_index] += weight * float(v_values.size)


def _finalize_object_region_accumulator(
    accumulator: dict[str, dict[str, np.ndarray]],
    *,
    shape: str,
    axis_source: str,
    orientation: str,
    buoyancy_source: str,
    tau: float,
    polarity: str,
    depth_m: np.ndarray,
) -> "pd.DataFrame":
    rows = []
    for region, data in accumulator.items():
        count = data["count"]
        with np.errstate(invalid="ignore", divide="ignore"):
            mean_v = np.divide(data["sum_v"], count, out=np.full_like(count, np.nan), where=count > 0)
            mean_theta = np.divide(data["sum_theta"], count, out=np.full_like(count, np.nan), where=count > 0)
            mean_q = np.divide(data["sum_q"], count, out=np.full_like(count, np.nan), where=count > 0)
            product_heat = RHO0 * CP0 * np.divide(data["sum_vtheta"], count, out=np.full_like(count, np.nan), where=count > 0)
            mean_product_heat = RHO0 * CP0 * mean_v * mean_theta
            product_pv = np.divide(data["sum_vq"], count, out=np.full_like(count, np.nan), where=count > 0)
            mean_product_pv = mean_v * mean_q
        for depth_index, depth_value in enumerate(depth_m):
            if count[depth_index] <= 0:
                continue
            rows.append(
                {
                    "shape": shape,
                    "axis_source": axis_source,
                    "orientation": orientation,
                    "buoyancy_source": buoyancy_source,
                    "tau": float(tau),
                    "polarity": polarity,
                    "depth_index": int(depth_index),
                    "depth_m": float(depth_value),
                    "region": region,
                    "object_aggregate_valid_cell_count": float(count[depth_index]),
                    "object_aggregate_mean_v_rot": float(mean_v[depth_index]),
                    "object_aggregate_mean_theta": float(mean_theta[depth_index]),
                    "object_aggregate_mean_q": float(mean_q[depth_index]),
                    "object_aggregate_product_mean_heat": float(product_heat[depth_index]),
                    "object_aggregate_mean_product_heat": float(mean_product_heat[depth_index]),
                    "object_aggregate_covariance_heat": float(product_heat[depth_index] - mean_product_heat[depth_index]),
                    "object_aggregate_product_mean_pv": float(product_pv[depth_index]),
                    "object_aggregate_mean_product_pv": float(mean_product_pv[depth_index]),
                    "object_aggregate_covariance_pv": float(product_pv[depth_index] - mean_product_pv[depth_index]),
                }
            )
    return pd.DataFrame(rows)


def _add_grouped_abs_fraction(
    table: "pd.DataFrame",
    *,
    value_column: str,
    fraction_column: str,
) -> "pd.DataFrame":
    if table.empty or value_column not in table.columns:
        return table
    keys = ["shape", "axis_source", "orientation", "buoyancy_source", "tau", "polarity", "depth_index"]
    out = table.copy()
    out[fraction_column] = np.nan
    for _, index in out.groupby(keys, sort=False).groups.items():
        idx = list(index)
        sub = out.loc[idx]
        core_shell = sub["region"].isin(["inner_core", "pv_shell"])
        total = np.nansum(np.abs(sub.loc[core_shell, value_column].to_numpy(float)))
        if np.isfinite(total) and total > 0:
            values = np.abs(sub[value_column].to_numpy(float)) / total
            out.loc[idx, fraction_column] = values
    return out


def _compute_object_aggregate_transport_partition(
    *,
    request: CoreShellRequest,
    shape: str,
    axis_source: str,
    orientation: str,
    buoyancy_source: str,
    radial_root: Path,
    dataset,
    tau_grid: np.ndarray,
    masks_by_tau_polarity: dict[tuple[int, str], dict[str, np.ndarray]],
) -> "pd.DataFrame":
    from .transport_moments import (
        _center_lines,
        _load_objects,
        _load_points,
        _q_prime_from_psi,
        _read_filter_day,
        _resolve_n2_profile,
        _sample_rotated_fields,
        _tau_weights,
    )
    from .numerics import grid_spacing_m, load_n2, relative_vorticity, sanitize_ocean_field, streamfunction_from_zeta

    objects = _load_objects(
        radial_root,
        start="",
        end="",
        max_days=request.object_aggregate_max_days,
        max_objects=request.object_aggregate_max_objects,
        shapes=(shape,),
    )
    if objects.empty:
        return pd.DataFrame()
    points = _load_points(radial_root, set(objects["eddy3d_object_id"].astype("int64")))
    center_lines = _center_lines(points)
    radial = np.asarray(dataset.radius_coord, dtype="float64")
    theta_angles = np.asarray(dataset.theta_rad, dtype="float64")
    radial_mesh, theta_mesh = np.meshgrid(radial, theta_angles, indexing="ij")
    explicit_n2 = "" if request.n2_profile in (None, "", "auto") else str(request.n2_profile)
    n2_profile = _resolve_n2_profile(radial_root, explicit_n2)

    accumulators: dict[tuple[int, str], dict[str, dict[str, np.ndarray]]] = {}
    for key, masks in masks_by_tau_polarity.items():
        regions = tuple(region for region in masks if region != "outer_domain")
        accumulators[key] = _empty_object_region_accumulator(len(dataset.depth_m), regions)

    for date_text, day_objects in objects.groupby("date", sort=True):
        day = pd.Timestamp(date_text).date()
        lon, lat, depth, u, v, theta_field = _read_filter_day(request.filter_root, request.filter_template, day)
        u = sanitize_ocean_field(u)
        v = sanitize_ocean_field(v)
        theta_field = sanitize_ocean_field(theta_field)
        _, dy, dx = grid_spacing_m(lon, lat)
        zeta = relative_vorticity(lon, lat, u, v)
        psi = streamfunction_from_zeta(zeta, dx=dx, dy=dy)
        n2_full = load_n2(n2_profile, depth)

        for obj in day_objects.itertuples(index=False):
            sampled = _sample_rotated_fields(
                obj,
                center_lines.get(int(obj.eddy3d_object_id)),
                lon,
                lat,
                psi,
                u,
                v,
                theta_field,
                radial_mesh,
                theta_mesh,
            )
            if sampled is None:
                continue
            depth_indices, depth_values, psi_s, vrot_s, theta_s = sampled
            valid_depth = (depth_indices >= 0) & (depth_indices < len(n2_full))
            if np.count_nonzero(valid_depth) < 3:
                continue
            depth_indices = depth_indices[valid_depth]
            depth_values = depth_values[valid_depth]
            psi_s = psi_s[valid_depth]
            vrot_s = vrot_s[valid_depth]
            theta_s = theta_s[valid_depth]
            f0 = 2.0 * 7.2921159e-5 * np.sin(np.deg2rad(float(obj.surface_lat)))
            q_prime = _q_prime_from_psi(
                psi_s,
                depth_values,
                radial,
                theta_angles,
                float(obj.mean_radius_m),
                n2_full[depth_indices],
                f0,
            )
            polarity = str(obj.polarity)
            for tau_index, weight in _tau_weights(float(obj.life_phase), tau_grid, 0.075, 1.0e-3):
                key = (int(tau_index), polarity)
                if key not in accumulators:
                    continue
                _add_object_region_terms(
                    accumulators[key],
                    masks=masks_by_tau_polarity[key],
                    depth_indices=depth_indices,
                    vrot=vrot_s,
                    theta_prime=theta_s,
                    q_prime=q_prime,
                    weight=float(weight),
                )

    tables = []
    for (tau_index, polarity), accumulator in accumulators.items():
        tables.append(
            _finalize_object_region_accumulator(
                accumulator,
                shape=shape,
                axis_source=axis_source,
                orientation=orientation,
                buoyancy_source=buoyancy_source,
                tau=float(tau_grid[tau_index]),
                polarity=polarity,
                depth_m=np.asarray(dataset.depth_m, dtype="float64"),
            )
        )
    nonempty = [table for table in tables if not table.empty]
    return pd.concat(nonempty, ignore_index=True) if nonempty else pd.DataFrame()


def _summary_table(profiles: "pd.DataFrame") -> "pd.DataFrame":
    rows = []
    keys = ["shape", "axis_source", "orientation", "buoyancy_source", "polarity", "region"]
    for group_key, sub in profiles.groupby(keys, sort=True):
        row = dict(zip(keys, group_key))
        for col in (
            "pv_high_quantile_retention",
            "pv_abs_retention",
            "weak_core_retention",
            "lateral_abs_volume_flux_m3s_proxy",
            "total_abs_volume_flux_m3s_proxy",
            "boundary_flux_over_internal_flux",
            "core_shell_interface_abs_un_ms",
            "pv_flux_magnitude_proxy",
            "buoyancy_flux_magnitude_proxy",
            "product_mean_heat",
            "mean_product_heat",
            "covariance_heat",
            "product_mean_pv",
            "mean_product_pv",
            "covariance_pv",
            "F_z_ordinary_region",
            "F_z_tilted_region",
            "F_z_tilt_correction_region",
            "abs_F_z_ordinary_region",
            "abs_F_z_tilt_correction_region",
            "tilt_correction_over_ordinary",
            "region_fraction_of_total_abs_tilt_correction",
            "region_fraction_of_total_abs_pv_covariance",
            "region_fraction_of_total_abs_heat_covariance",
            "object_aggregate_product_mean_heat",
            "object_aggregate_mean_product_heat",
            "object_aggregate_covariance_heat",
            "object_aggregate_product_mean_pv",
            "object_aggregate_mean_product_pv",
            "object_aggregate_covariance_pv",
            "object_aggregate_region_fraction_of_total_abs_heat_covariance",
            "object_aggregate_region_fraction_of_total_abs_pv_covariance",
        ):
            if col in sub.columns:
                row[f"{col}_median"] = _nanmedian_no_warning(sub[col].to_numpy(float))
        if "theta_prime_status" in sub.columns:
            row["theta_prime_available_fraction"] = float(
                (sub["theta_prime_status"].astype(str) == "available").mean()
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _write_partition_tables(profiles: "pd.DataFrame", output_dir: Path) -> dict[str, Path]:
    written: dict[str, Path] = {}
    base_cols = [
        "shape",
        "axis_source",
        "orientation",
        "buoyancy_source",
        "tau",
        "polarity",
        "depth_index",
        "depth_m",
        "region",
    ]
    transport_cols = base_cols + [
        "theta_prime_status",
        "product_mean_heat",
        "mean_product_heat",
        "covariance_heat",
        "product_mean_pv",
        "mean_product_pv",
        "covariance_pv",
        "region_fraction_of_total_abs_heat_covariance",
        "region_fraction_of_total_abs_pv_covariance",
        "object_aggregate_valid_cell_count",
        "object_aggregate_mean_v_rot",
        "object_aggregate_mean_theta",
        "object_aggregate_mean_q",
        "object_aggregate_product_mean_heat",
        "object_aggregate_mean_product_heat",
        "object_aggregate_covariance_heat",
        "object_aggregate_product_mean_pv",
        "object_aggregate_mean_product_pv",
        "object_aggregate_covariance_pv",
        "object_aggregate_region_fraction_of_total_abs_heat_covariance",
        "object_aggregate_region_fraction_of_total_abs_pv_covariance",
    ]
    ep_cols = base_cols + [
        "F_z_ordinary_region",
        "F_z_tilted_region",
        "F_z_tilt_correction_region",
        "abs_F_z_ordinary_region",
        "abs_F_z_tilt_correction_region",
        "tilt_correction_over_ordinary",
        "region_fraction_of_total_abs_tilt_correction",
    ]
    available_transport = [col for col in transport_cols if col in profiles.columns]
    available_ep = [col for col in ep_cols if col in profiles.columns]
    if available_transport:
        path = output_dir / "core_shell_transport_partition.csv"
        _write_table(profiles[available_transport], path)
        written["transport_partition"] = path
    if available_ep:
        path = output_dir / "core_shell_ep_tilt_partition.csv"
        _write_table(profiles[available_ep], path)
        written["ep_tilt_partition"] = path
    return written


def _plot_core_shell_figures(profiles: "pd.DataFrame", output_dir: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    if profiles.empty:
        return paths

    clean = profiles.copy()

    def grouped_median_bar(column: str, ylabel: str, title: str, filename: str, *, absolute: bool = True) -> None:
        if not {"region", "shape", column}.issubset(clean.columns):
            return
        labels = []
        values = []
        for (shape, region), sub in clean.groupby(["shape", "region"], sort=True):
            data = sub[column].to_numpy(float)
            if absolute:
                data = np.abs(data)
            value = _nanmedian_no_warning(data)
            if not np.isfinite(value):
                continue
            labels.append(f"{shape}\n{region}")
            values.append(value)
        if not values:
            return
        fig, ax = plt.subplots(figsize=(8.0, 4.5))
        ax.bar(np.arange(len(values)), values, color="#4c78a8")
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        fig.tight_layout()
        path = fig_dir / filename
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)

    if {"region", "pv_high_quantile_retention", "shape"}.issubset(clean.columns):
        fig, ax = plt.subplots(figsize=(8.0, 4.5))
        labels = []
        values = []
        for (shape, region), sub in clean.groupby(["shape", "region"], sort=True):
            labels.append(f"{shape}\n{region}")
            values.append(_nanmedian_no_warning(sub["pv_high_quantile_retention"].to_numpy(float)))
        ax.bar(np.arange(len(values)), values, color="#4c78a8")
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.set_ylim(0, 1)
        ax.set_ylabel("median high-|PV| retention")
        ax.set_title("PV retention split by inner material core and PV shell")
        fig.tight_layout()
        path = fig_dir / "pv_retention_by_region.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)

    grouped_median_bar(
        "covariance_pv",
        "median |covariance PV stirring proxy|",
        "PV stirring covariance partition by core and shell",
        "heat_pv_core_vs_shell_partition.png",
    )
    grouped_median_bar(
        "object_aggregate_covariance_heat",
        "median |rho0 Cp <v' theta'> covariance|",
        "Object aggregate-product heat stirring partition",
        "object_aggregate_heat_core_vs_shell_partition.png",
    )
    grouped_median_bar(
        "object_aggregate_covariance_pv",
        "median |<v' q'> covariance|",
        "Object aggregate-product PV stirring partition",
        "object_aggregate_pv_core_vs_shell_partition.png",
    )
    grouped_median_bar(
        "F_z_tilt_correction_region",
        "median |Fz tilt correction|",
        "EP tilt correction partition by core and shell",
        "ep_tilt_correction_core_vs_shell.png",
    )
    grouped_median_bar(
        "region_fraction_of_total_abs_tilt_correction",
        "median fraction of |core+shell tilt correction|",
        "Coherent vs upright-like tilt-correction partition",
        "coherent_vs_upright_like_partition.png",
        absolute=False,
    )
    grouped_median_bar(
        "total_abs_volume_flux_m3s_proxy",
        "median total |volume flux| proxy (m3/s)",
        "Core, shell, and exchange budget proxy",
        "core_shell_exchange_budget.png",
    )

    if {"region", "total_abs_volume_flux_m3s_proxy", "shape"}.issubset(clean.columns):
        fig, ax = plt.subplots(figsize=(8.0, 4.5))
        labels = []
        values = []
        for (shape, region), sub in clean.groupby(["shape", "region"], sort=True):
            labels.append(f"{shape}\n{region}")
            values.append(_nanmedian_no_warning(sub["total_abs_volume_flux_m3s_proxy"].to_numpy(float)))
        ax.bar(np.arange(len(values)), values, color="#f58518")
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.set_ylabel("median total |volume flux| proxy (m3/s)")
        ax.set_title("Boundary exchange split by core, shell, and combined volume")
        fig.tight_layout()
        path = fig_dir / "boundary_exchange_by_region.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)

    if {"pv_high_quantile_retention", "total_abs_volume_flux_m3s_proxy", "region"}.issubset(clean.columns):
        fig, ax = plt.subplots(figsize=(6.5, 5.0))
        for region, sub in clean.groupby("region", sort=True):
            ax.scatter(
                sub["pv_high_quantile_retention"],
                sub["total_abs_volume_flux_m3s_proxy"],
                s=14,
                alpha=0.4,
                label=str(region),
            )
        ax.set_xlabel("high-|PV| retention")
        ax.set_ylabel("total |volume flux| proxy (m3/s)")
        ax.set_title("PV retention versus boundary exchange")
        ax.legend(fontsize=8)
        fig.tight_layout()
        path = fig_dir / "pv_retention_vs_boundary_exchange.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.axis("off")
    boxes = [
        (0.05, 0.55, "inner core\nHua/LAVD\nlow leakage\ntrapping"),
        (0.38, 0.55, "PV-active shell\nPV anomaly\ncrescent speed band\nstirring"),
        (0.70, 0.55, "exchange layer\nheat/PV/momentum\nboundary exchange"),
        (0.28, 0.15, "T_total = T_core^trap + T_shell^stir + T_exchange"),
    ]
    for x0, y0, text in boxes:
        ax.text(
            x0,
            y0,
            text,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=10,
            bbox={"boxstyle": "round,pad=0.45", "facecolor": "#f6f8fa", "edgecolor": "#4c78a8"},
        )
    for start, end in [((0.18, 0.55), (0.30, 0.55)), ((0.51, 0.55), (0.62, 0.55))]:
        ax.annotate("", xy=end, xytext=start, xycoords="axes fraction", arrowprops={"arrowstyle": "->", "lw": 1.5})
    ax.set_title("Dual-zone eddy interpretation used by core-shell EP diagnostics")
    path = fig_dir / "dual_zone_framework_summary.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)
    return paths


def _summary_markdown_legacy_mojibake(request: CoreShellRequest, summary: "pd.DataFrame") -> str:
    lines = [
        "# Core-Shell EP Validation Summary",
        "",
        "## 核心口径",
        "- 这一步不再强迫一个边界同时承担 trapping 与 PV stirring。",
        "- `inner_core` 表示低泄漏、旋转/弱速材料核。",
        "- `pv_shell` 表示围绕内核的 PV-active 外壳，用来承载 PV、热和动量 stirring。",
        "- `combined_volume` 表示内核与 PV 外壳合并后的总控制体。",
        "",
        "## 参数",
        f"- result root: `{request.result_root}`",
        f"- filter root: `{request.filter_root}`",
        f"- shapes: `{','.join(request.shapes)}`",
        f"- orientations: `{','.join(request.orientations)}`",
        f"- axis sources: `{','.join(request.axis_sources)}`",
        f"- buoyancy sources: `{','.join(request.buoyancy_sources)}`",
        f"- inner boundary mode: `{request.inner_boundary_mode}`",
        f"- PV shell quantile: `{request.pv_shell_quantile}`",
        f"- object aggregate transport: `{request.object_aggregate_transport}`",
        "",
        "## 汇总表",
        "```text",
        summary.to_string(index=False) if not summary.empty else "(empty)",
        "```",
        "",
        "## 判读",
        "- 如果 `inner_core` leakage 低但 PV retention 低，而 `pv_shell/combined_volume` PV retention 高但 boundary exchange 上升，说明涡旋更像“材料俘获核 + PV 搅拌外壳”。",
        "- 如果 `combined_volume` 同时保持高 PV retention 和低 exchange，才支持单个材料体闭合。",
        "- 若外壳 exchange 很大，不能把 PV shell 解释成严格 trapping；它更像 stirring/exchange 区域。",
    ]
    return "\n".join(lines) + "\n"


def _summary_markdown(request: CoreShellRequest, summary: "pd.DataFrame") -> str:
    lines = [
        "# Core-Shell EP Validation Summary",
        "",
        "## 核心口径",
        "- 本诊断不再强迫一个边界同时承担 trapping 与 PV stirring。",
        "- `inner_core` 表示低泄漏、弱速/LAVD 近同位的材料核。",
        "- `pv_shell` 表示围绕内核的 PV-active 外壳，用来承载 PV、热和动量 stirring。",
        "- `combined_volume` 表示内核与 PV 外壳合并后的总体控制体。",
        "- 解释框架为 `T_total = T_core^trap + T_shell^stir + T_exchange`。",
        "",
        "## 参数",
        f"- result root: `{request.result_root}`",
        f"- shapes: `{','.join(request.shapes)}`",
        f"- orientations: `{','.join(request.orientations)}`",
        f"- axis sources: `{','.join(request.axis_sources)}`",
        f"- buoyancy sources: `{','.join(request.buoyancy_sources)}`",
        f"- inner boundary mode: `{request.inner_boundary_mode}`",
        f"- PV shell quantile: `{request.pv_shell_quantile}`",
        "",
        "## 汇总表",
        "```text",
        summary.to_string(index=False) if not summary.empty else "(empty)",
        "```",
        "",
        "## 判读",
        "- 若 `inner_core` 低 leakage、高 weak/LAVD retention，但 object aggregate heat/PV stirring 较弱，则它更像 trapping/material core。",
        "- 若 `pv_shell` 的 object aggregate heat/PV covariance 或 EP tilt correction 贡献更强，则它更像 PV-active stirring shell。",
        "- 若 shell 或 combined volume 的 boundary exchange 很大，不能把它解释成严格闭合材料体。",
        "- 热通量分区优先读取 object-day `thetao_glor` 并累计 `v_rot * theta`，而不是用代表涡平均温度场相乘。",
    ]
    return "\n".join(lines) + "\n"


def run_core_shell_ep_validation(request: CoreShellRequest) -> dict[str, Path]:
    _validate_request(request)
    if request.dry_run:
        for shape in request.shapes:
            for orientation in request.orientations:
                for axis_source in request.axis_sources:
                    for buoyancy_source in request.buoyancy_sources:
                        print(
                            "[dry-run]",
                            _combo_output(request.output_root, shape, axis_source, orientation, buoyancy_source),
                        )
        return {}

    _require_runtime()
    from .diagnostics import load_n2_profile, resolve_n2_profile_path
    from .fields import RepresentativeVortexDataset
    from .flux import EPFluxCalculator
    from .geometry import AxisLine

    written: dict[str, Path] = {}
    all_profiles: list[pd.DataFrame] = []
    all_summaries: list[pd.DataFrame] = []
    for shape in request.shapes:
        output_name = shape_output_name(shape)
        radial_root = default_radial_seed_root(request.result_root, output_name)
        for orientation in request.orientations:
            me_root = default_me_liutex_root(request.result_root, output_name, orientation)
            if not me_root.exists():
                if request.skip_missing:
                    continue
                raise FileNotFoundError(me_root)
            dataset = RepresentativeVortexDataset.load(me_root / "azimuthal_representative_velocity.npz", radial_root)
            tau_grid = _tau_grid_for_combo(request.result_root, shape, orientation, request.tau_values)
            for axis_source in request.axis_sources:
                for buoyancy_source in request.buoyancy_sources:
                    combo_dir = _combo_output(request.output_root, shape, axis_source, orientation, buoyancy_source)
                    combo_dir.mkdir(parents=True, exist_ok=True)
                    combo_profiles = []
                    combo_masks_by_tau_polarity: dict[tuple[int, str], dict[str, np.ndarray]] = {}
                    for tau in tau_grid:
                        tau_index = int(np.nanargmin(np.abs(np.asarray(dataset.tau_grid, dtype=float) - float(tau))))
                        axis_path = me_root / "axis_sources" / axis_source_filename(axis_source, float(tau))
                        if not axis_path.exists():
                            if request.skip_missing:
                                continue
                            raise FileNotFoundError(
                                f"Missing axis source {axis_path}. Run src.EP.cli build-representative-axis-sources first."
                            )
                        config = EPFluxConfig(
                            me_liutex_root=me_root,
                            radial_seed_root=radial_root,
                            output_dir=combo_dir,
                            orientation=orientation,
                            axis_source=axis_source,
                            tau=float(tau),
                            reference_lat=request.reference_lat,
                            constant_n2=request.constant_n2,
                            buoyancy_source=buoyancy_source,
                        )
                        n2_path = resolve_n2_profile_path(config, request.n2_profile)
                        for polarity in dataset.polarities:
                            rep = dataset.slice(polarity, float(tau))
                            axis = AxisLine.from_csv(axis_path, polarity=polarity).interpolate_to(rep.depth_m)
                            n2 = load_n2_profile(n2_path, rep.depth_m, request.constant_n2)
                            base_request = MaterialVolumeRequest(
                                result_root=request.result_root,
                                output_root=combo_dir,
                                shapes=(shape,),
                                axis_sources=(axis_source,),
                                orientations=(orientation,),
                                buoyancy_sources=(buoyancy_source,),
                                tau_values=(float(tau),),
                                reference_lat=request.reference_lat,
                                constant_n2=request.constant_n2,
                                core_radius_over_R=request.core_radius_over_R,
                                speed_core_quantile=request.speed_core_quantile,
                                pv_core_quantile=request.pv_core_quantile,
                                min_mask_fraction=request.min_mask_fraction,
                                boundary_mode=request.inner_boundary_mode,
                                boundary_budget=request.boundary_budget,
                                min_core_retention=request.min_core_retention,
                            )
                            _, debug = _compute_one_slice(
                                rep=rep,
                                axis=axis,
                                f0=config.f0,
                                n2=n2,
                                buoyancy_source=buoyancy_source,
                                request=base_request,
                            )
                            ep_terms = EPFluxCalculator(
                                rep,
                                axis,
                                f0=config.f0,
                                n2=n2,
                                buoyancy_source=buoyancy_source,
                            ).compute_field_terms()
                            profiles, masks = _compute_core_shell_for_slice(
                                rep=rep,
                                axis=axis,
                                q_proxy=debug["q_proxy"],
                                buoyancy=debug["buoyancy"],
                                inner_masks=debug["mask"],
                                ep_terms=ep_terms,
                                request=request,
                            )
                            profiles["shape"] = shape
                            profiles["axis_source"] = axis_source
                            profiles["orientation"] = orientation
                            profiles["buoyancy_source"] = buoyancy_source
                            profiles["tau"] = float(rep.tau)
                            profiles["polarity"] = polarity
                            combo_masks_by_tau_polarity[(tau_index, polarity)] = masks
                            combo_profiles.append(profiles)
                    if not combo_profiles:
                        continue
                    combo_all = pd.concat(combo_profiles, ignore_index=True)
                    if request.object_aggregate_transport:
                        object_transport = _compute_object_aggregate_transport_partition(
                            request=request,
                            shape=shape,
                            axis_source=axis_source,
                            orientation=orientation,
                            buoyancy_source=buoyancy_source,
                            radial_root=radial_root,
                            dataset=dataset,
                            tau_grid=np.asarray(dataset.tau_grid, dtype=float),
                            masks_by_tau_polarity=combo_masks_by_tau_polarity,
                        )
                        if not object_transport.empty:
                            object_transport = _add_grouped_abs_fraction(
                                object_transport,
                                value_column="object_aggregate_covariance_heat",
                                fraction_column="object_aggregate_region_fraction_of_total_abs_heat_covariance",
                            )
                            object_transport = _add_grouped_abs_fraction(
                                object_transport,
                                value_column="object_aggregate_covariance_pv",
                                fraction_column="object_aggregate_region_fraction_of_total_abs_pv_covariance",
                            )
                            merge_keys = [
                                "shape",
                                "axis_source",
                                "orientation",
                                "buoyancy_source",
                                "tau",
                                "polarity",
                                "depth_index",
                                "depth_m",
                                "region",
                            ]
                            combo_all = combo_all.merge(object_transport, on=merge_keys, how="left")
                    summary = _summary_table(combo_all)
                    _write_table(combo_all, combo_dir / "core_shell_profiles.csv")
                    _write_table(summary, combo_dir / "core_shell_summary.csv")
                    for key, path in _write_partition_tables(combo_all, combo_dir).items():
                        written[f"{combo_dir}:{key}"] = path
                    (combo_dir / "core_shell_summary.json").write_text(
                        json.dumps(_json_ready(summary.to_dict(orient="records")), ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                    (combo_dir / "core_shell_ep_validation_summary_zh.md").write_text(
                        _summary_markdown(request, summary),
                        encoding="utf-8",
                    )
                    for fig in _plot_core_shell_figures(combo_all, combo_dir):
                        written[str(fig)] = fig
                    written[str(combo_dir)] = combo_dir
                    all_profiles.append(combo_all)
                    all_summaries.append(summary)

    if all_profiles:
        root_profiles = pd.concat(all_profiles, ignore_index=True)
        root_summary = pd.concat(all_summaries, ignore_index=True)
        _write_table(root_profiles, request.output_root / "core_shell_all_profiles.csv")
        _write_table(root_summary, request.output_root / "core_shell_all_summary.csv")
        for key, path in _write_partition_tables(root_profiles, request.output_root).items():
            written[f"root:{key}"] = path
        (request.output_root / "core_shell_all_summary.json").write_text(
            json.dumps(_json_ready(root_summary.to_dict(orient="records")), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (request.output_root / "core_shell_ep_validation_summary_zh.md").write_text(
            _summary_markdown(request, root_summary),
            encoding="utf-8",
        )
        for fig in _plot_core_shell_figures(root_profiles, request.output_root):
            written[str(fig)] = fig
    return written


def request_from_args(args) -> CoreShellRequest:
    return CoreShellRequest(
        result_root=Path(args.result_root),
        output_root=Path(args.output_root),
        filter_root=Path(args.filter_root),
        filter_template=args.filter_template,
        shapes=_split_csv(args.shapes),
        axis_sources=_split_csv(args.axis_sources),
        orientations=_split_csv(args.orientations),
        buoyancy_sources=_split_csv(args.buoyancy_sources),
        tau_values=_parse_tau_values(args.tau_values),
        reference_lat=float(args.reference_lat),
        constant_n2=float(args.constant_n2),
        n2_profile=args.n2_profile,
        inner_boundary_mode=args.inner_boundary_mode,
        boundary_budget=args.boundary_budget,
        core_radius_over_R=float(args.core_radius_over_R),
        shell_outer_radius_over_R=float(args.shell_outer_radius_over_R),
        speed_core_quantile=float(args.speed_core_quantile),
        pv_core_quantile=float(args.pv_core_quantile),
        pv_shell_quantile=float(args.pv_shell_quantile),
        shell_dilation_cells=int(args.shell_dilation_cells),
        min_mask_fraction=float(args.min_mask_fraction),
        min_core_retention=float(args.min_core_retention),
        object_aggregate_transport=not bool(args.no_object_aggregate_transport),
        object_aggregate_max_days=int(args.object_aggregate_max_days),
        object_aggregate_max_objects=int(args.object_aggregate_max_objects),
        skip_missing=bool(args.skip_missing),
        dry_run=bool(args.dry_run),
    )
