from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
try:
    import pandas as pd
except ModuleNotFoundError:  # pragma: no cover - dependency-light dry-run support.
    pd = None

from .contracts import (
    AXIS_SOURCES,
    BUOYANCY_SOURCES,
    DEFAULT_FULL_OUTPUT_ROOT,
    DEFAULT_RESULT_ROOT,
    RHO0,
    EPFluxConfig,
    default_me_liutex_root,
    default_radial_seed_root,
    shape_output_name,
)

DEFAULT_MATERIAL_OUTPUT_ROOT = DEFAULT_FULL_OUTPUT_ROOT.parent / "material_volume_validation"


def _require_pandas() -> None:
    if pd is None:
        raise ModuleNotFoundError("pandas is required for material-volume EP validation runs")


def _safe_gradient(values: np.ndarray, coord: np.ndarray, axis: int) -> np.ndarray:
    coord = np.asarray(coord, dtype=float)
    if coord.size < 2:
        return np.zeros_like(values)
    return np.gradient(values, coord, axis=axis, edge_order=1)


def _periodic_gradient(values: np.ndarray, theta: np.ndarray) -> np.ndarray:
    dtheta = float(np.nanmedian(np.diff(np.unwrap(theta))))
    if not np.isfinite(dtheta) or dtheta == 0:
        dtheta = 2.0 * np.pi / values.shape[-1]
    return (np.roll(values, -1, axis=-1) - np.roll(values, 1, axis=-1)) / (2.0 * dtheta)


def _positive_radial_coord(radial: np.ndarray) -> np.ndarray:
    coord = np.asarray(radial, dtype=float).copy()
    if coord.size > 1 and coord[0] <= 0:
        coord[0] = coord[1] * 0.5
    return coord


@dataclass(frozen=True)
class MaterialVolumeRequest:
    result_root: Path = DEFAULT_RESULT_ROOT
    output_root: Path = DEFAULT_MATERIAL_OUTPUT_ROOT
    shapes: tuple[str, ...] = ("coherent", "upright_like")
    axis_sources: tuple[str, ...] = ("radial_seed",)
    orientations: tuple[str, ...] = ("turned",)
    buoyancy_sources: tuple[str, ...] = ("thermal_wind",)
    tau_values: tuple[float, ...] | None = None
    reference_lat: float = 30.0
    constant_n2: float = 2.0e-5
    n2_profile: str | None = "auto"
    core_radius_over_R: float = 1.5
    speed_core_quantile: float = 0.45
    pv_core_quantile: float = 0.70
    min_mask_fraction: float = 0.01
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


def _json_ready(value: object) -> object:
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, Path):
        return str(value)
    return value


def _tau_grid(dataset: RepresentativeVortexDataset, tau_values: tuple[float, ...] | None) -> np.ndarray:
    if tau_values is not None:
        return np.asarray(tau_values, dtype=float)
    return np.asarray(dataset.tau_grid, dtype=float)


def _streamfunction_from_tangential(ut: np.ndarray, radial: np.ndarray) -> np.ndarray:
    psi = np.zeros_like(ut, dtype=float)
    dr = np.diff(radial)
    for idx in range(1, radial.size):
        psi[:, idx, :] = psi[:, idx - 1, :] + 0.5 * (ut[:, idx, :] + ut[:, idx - 1, :]) * dr[idx - 1]
    return psi - np.nanmean(psi, axis=(1, 2), keepdims=True)


def _cartesian_gradient(values: np.ndarray, radial: np.ndarray, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    radial_positive = _positive_radial_coord(radial)
    d_dr = _safe_gradient(values, radial, axis=-2)
    d_dtheta = _periodic_gradient(values, theta)
    if values.ndim == 2:
        rr = radial_positive[:, None]
        cos_t = np.cos(theta)[None, :]
        sin_t = np.sin(theta)[None, :]
    else:
        rr = radial_positive[None, :, None]
        cos_t = np.cos(theta)[None, None, :]
        sin_t = np.sin(theta)[None, None, :]
    grad_x = d_dr * cos_t - d_dtheta * sin_t / rr
    grad_y = d_dr * sin_t + d_dtheta * cos_t / rr
    return grad_x, grad_y


def _qg_pv_proxy(psi: np.ndarray, radial: np.ndarray, theta: np.ndarray, depth: np.ndarray, f0: float, n2: np.ndarray) -> np.ndarray:
    dpsi_dr = _safe_gradient(psi, radial, axis=1)
    radial_positive = _positive_radial_coord(radial)
    lap_r = _safe_gradient(radial_positive[None, :, None] * dpsi_dr, radial_positive, axis=1) / radial_positive[None, :, None]
    lap_t = _periodic_gradient(_periodic_gradient(psi, theta), theta) / (radial_positive[None, :, None] ** 2)
    dpsi_dz = _safe_gradient(psi, depth, axis=0)
    stratified = (f0**2 / n2[:, None, None]) * dpsi_dz
    vertical = _safe_gradient(stratified, depth, axis=0)
    return lap_r + lap_t + vertical


def _thermal_wind_buoyancy(u: np.ndarray, v: np.ndarray, radial: np.ndarray, theta: np.ndarray, depth: np.ndarray, f0: float) -> np.ndarray:
    du_dz = _safe_gradient(u, depth, axis=0)
    dv_dz = _safe_gradient(v, depth, axis=0)
    db_dx = f0 * dv_dz
    db_dy = -f0 * du_dz
    cos_t = np.cos(theta)[None, None, :]
    sin_t = np.sin(theta)[None, None, :]
    db_dr = db_dx * cos_t + db_dy * sin_t
    b = np.zeros_like(db_dr, dtype=float)
    dr = np.diff(radial)
    for idx in range(1, radial.size):
        b[:, idx, :] = b[:, idx - 1, :] + 0.5 * (db_dr[:, idx, :] + db_dr[:, idx - 1, :]) * dr[idx - 1]
    return b - np.nanmean(b, axis=(1, 2), keepdims=True)


def _streamfunction_buoyancy(psi: np.ndarray, depth: np.ndarray, f0: float) -> np.ndarray:
    dpsi_dz = _safe_gradient(psi, depth, axis=0)
    return f0 * (dpsi_dz - np.nanmean(dpsi_dz, axis=(1, 2), keepdims=True))


def _neighbors(index: tuple[int, int], nr: int, nt: int):
    ir, it = index
    if ir > 0:
        yield ir - 1, it
    if ir < nr - 1:
        yield ir + 1, it
    yield ir, (it - 1) % nt
    yield ir, (it + 1) % nt


def _connected_component(candidate: np.ndarray, seed: tuple[int, int]) -> np.ndarray:
    nr, nt = candidate.shape
    out = np.zeros_like(candidate, dtype=bool)
    if not candidate[seed]:
        return out
    stack = [seed]
    out[seed] = True
    while stack:
        idx = stack.pop()
        for nxt in _neighbors(idx, nr, nt):
            if candidate[nxt] and not out[nxt]:
                out[nxt] = True
                stack.append(nxt)
    return out


def _edge_mask(mask: np.ndarray) -> np.ndarray:
    edge = np.zeros_like(mask, dtype=bool)
    nr, nt = mask.shape
    for ir in range(nr):
        for it in range(nt):
            if not mask[ir, it]:
                continue
            if any(not mask[nbr] for nbr in _neighbors((ir, it), nr, nt)):
                edge[ir, it] = True
    return edge


def _weighted_mean(values: np.ndarray, mask: np.ndarray, weights: np.ndarray | None = None) -> float:
    valid = mask & np.isfinite(values)
    if weights is not None:
        valid = valid & np.isfinite(weights) & (weights > 0)
    if not np.any(valid):
        return np.nan
    if weights is None:
        return float(np.nanmean(values[valid]))
    return float(np.nansum(values[valid] * weights[valid]) / np.nansum(weights[valid]))


def _divergence_xy(fx: np.ndarray, fy: np.ndarray, radial: np.ndarray, theta: np.ndarray) -> np.ndarray:
    dfx_dx, dfx_dy = _cartesian_gradient(fx, radial, theta)
    dfy_dx, dfy_dy = _cartesian_gradient(fy, radial, theta)
    return dfx_dx + dfy_dy


def _seed_index(axis: AxisLine, depth_index: int, x_km: np.ndarray, y_km: np.ndarray, candidate: np.ndarray) -> tuple[int, int] | None:
    ax = float(axis.x_km[depth_index])
    ay = float(axis.y_km[depth_index])
    distance = np.hypot(x_km - ax, y_km - ay)
    distance = np.where(candidate, distance, np.inf)
    if not np.any(np.isfinite(distance)):
        return None
    flat = int(np.nanargmin(distance))
    return np.unravel_index(flat, distance.shape)


def _layer_material_mask(
    *,
    rep: RepresentativeSlice,
    axis: AxisLine,
    q_proxy: np.ndarray,
    depth_index: int,
    x_km: np.ndarray,
    y_km: np.ndarray,
    core_radius_over_R: float,
    speed_core_quantile: float,
    pv_core_quantile: float,
    min_mask_fraction: float,
) -> tuple[np.ndarray, dict[str, float | str]]:
    radial_over_R = rep.radial_m[:, None] / rep.radius_m
    finite = np.isfinite(rep.speed[depth_index]) & np.isfinite(q_proxy[depth_index])
    core_domain = finite & (radial_over_R <= core_radius_over_R)
    if not np.any(core_domain):
        return np.zeros_like(finite, dtype=bool), {"mask_status": "empty_core_domain"}
    speed_values = rep.speed[depth_index][core_domain]
    q_values = np.abs(q_proxy[depth_index][core_domain])
    speed_threshold = float(np.nanquantile(speed_values, speed_core_quantile))
    pv_threshold = float(np.nanquantile(q_values, pv_core_quantile))
    candidate = core_domain & (
        (rep.speed[depth_index] <= speed_threshold)
        | (np.abs(q_proxy[depth_index]) >= pv_threshold)
    )
    seed = _seed_index(axis, depth_index, x_km, y_km, candidate)
    if seed is None:
        return np.zeros_like(candidate, dtype=bool), {"mask_status": "empty_candidate"}
    mask = _connected_component(candidate, seed)
    min_cells = max(1, int(np.ceil(min_mask_fraction * np.count_nonzero(core_domain))))
    if int(np.count_nonzero(mask)) < min_cells:
        fallback = core_domain & (radial_over_R <= min(0.75, core_radius_over_R))
        seed = _seed_index(axis, depth_index, x_km, y_km, fallback)
        mask = _connected_component(fallback, seed) if seed is not None else mask
        status = "fallback_axis_core"
    else:
        status = "ok"
    return mask, {
        "mask_status": status,
        "speed_threshold_ms": speed_threshold,
        "pv_abs_threshold": pv_threshold,
        "mask_cell_count": float(np.count_nonzero(mask)),
        "core_domain_cell_count": float(np.count_nonzero(core_domain)),
    }


def _layer_diagnostics(
    *,
    rep: RepresentativeSlice,
    axis: AxisLine,
    q_proxy: np.ndarray,
    buoyancy: np.ndarray,
    psi: np.ndarray,
    depth_index: int,
    x_km: np.ndarray,
    y_km: np.ndarray,
    core_radius_over_R: float,
    speed_core_quantile: float,
    pv_core_quantile: float,
    min_mask_fraction: float,
) -> dict[str, object]:
    mask, meta = _layer_material_mask(
        rep=rep,
        axis=axis,
        q_proxy=q_proxy,
        depth_index=depth_index,
        x_km=x_km,
        y_km=y_km,
        core_radius_over_R=core_radius_over_R,
        speed_core_quantile=speed_core_quantile,
        pv_core_quantile=pv_core_quantile,
        min_mask_fraction=min_mask_fraction,
    )
    edge = _edge_mask(mask)
    u = rep.u[depth_index]
    v = rep.v[depth_index]
    b = buoyancy[depth_index]
    q = q_proxy[depth_index]
    speed = rep.speed[depth_index]

    mean_u = _weighted_mean(u, mask)
    mean_v = _weighted_mean(v, mask)
    mean_b = _weighted_mean(b, mask)
    mean_q = _weighted_mean(q, mask)
    up = u - mean_u
    vp = v - mean_v
    bp = b - mean_b
    qp = q - mean_q

    rxx_field = up * up
    rxy_field = up * vp
    ryy_field = vp * vp
    bx_field = up * bp
    by_field = vp * bp
    px_field = up * qp
    py_field = vp * qp
    gx_field = -_divergence_xy(rxx_field, rxy_field, rep.radial_m, rep.theta_rad)
    gy_field = -_divergence_xy(rxy_field, ryy_field, rep.radial_m, rep.theta_rad)

    pv_weight = np.abs(q)
    weak_weight = 1.0 / (speed + np.nanmedian(speed[mask]) + 1e-12)
    pv_centroid_x = _weighted_mean(x_km, mask, pv_weight)
    pv_centroid_y = _weighted_mean(y_km, mask, pv_weight)
    weak_centroid_x = _weighted_mean(x_km, mask, weak_weight)
    weak_centroid_y = _weighted_mean(y_km, mask, weak_weight)

    edge_x = x_km - pv_centroid_x
    edge_y = y_km - pv_centroid_y
    edge_norm = np.hypot(edge_x, edge_y)
    nx = np.divide(edge_x, edge_norm, out=np.zeros_like(edge_x), where=edge_norm > 0)
    ny = np.divide(edge_y, edge_norm, out=np.zeros_like(edge_y), where=edge_norm > 0)
    boundary_normal_velocity = (u - mean_u) * nx + (v - mean_v) * ny

    row = {
        "depth_m": float(rep.depth_m[depth_index]),
        "axis_x_km": float(axis.x_km[depth_index]),
        "axis_y_km": float(axis.y_km[depth_index]),
        "axis_tilt_km": float(axis.tilt_km[depth_index]),
        "mask_fraction": float(np.count_nonzero(mask) / max(1, mask.size)),
        "mask_core_fraction": float(np.count_nonzero(mask) / max(1, int(meta.get("core_domain_cell_count", 1)))),
        "edge_cell_count": int(np.count_nonzero(edge)),
        "mean_u_ms": mean_u,
        "mean_v_ms": mean_v,
        "mean_speed_ms": _weighted_mean(speed, mask),
        "mean_b": mean_b,
        "mean_q_proxy": mean_q,
        "R_xx": _weighted_mean(rxx_field, mask),
        "R_xy": _weighted_mean(rxy_field, mask),
        "R_yy": _weighted_mean(ryy_field, mask),
        "B_x": _weighted_mean(bx_field, mask),
        "B_y": _weighted_mean(by_field, mask),
        "P_x": _weighted_mean(px_field, mask),
        "P_y": _weighted_mean(py_field, mask),
        "G_x_proxy": _weighted_mean(gx_field, mask),
        "G_y_proxy": _weighted_mean(gy_field, mask),
        "G_magnitude_proxy": float(np.hypot(_weighted_mean(gx_field, mask), _weighted_mean(gy_field, mask))),
        "pv_flux_magnitude": float(np.hypot(_weighted_mean(px_field, mask), _weighted_mean(py_field, mask))),
        "pv_centroid_x_km": pv_centroid_x,
        "pv_centroid_y_km": pv_centroid_y,
        "pv_centroid_distance_from_axis_km": float(np.hypot(pv_centroid_x - axis.x_km[depth_index], pv_centroid_y - axis.y_km[depth_index])),
        "weak_speed_centroid_x_km": weak_centroid_x,
        "weak_speed_centroid_y_km": weak_centroid_y,
        "weak_speed_centroid_distance_from_axis_km": float(np.hypot(weak_centroid_x - axis.x_km[depth_index], weak_centroid_y - axis.y_km[depth_index])),
        "boundary_leakage_proxy_mean_abs_ms": _weighted_mean(np.abs(boundary_normal_velocity), edge),
        "boundary_leakage_proxy_signed_ms": _weighted_mean(boundary_normal_velocity, edge),
        "psi_volume_mean": _weighted_mean(psi[depth_index], mask),
        **meta,
    }
    return row


def _compute_one_slice(
    *,
    rep: RepresentativeSlice,
    axis: AxisLine,
    f0: float,
    n2: np.ndarray,
    buoyancy_source: str,
    request: MaterialVolumeRequest,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    ur, ut = rep.polar_velocity()
    psi = _streamfunction_from_tangential(ut, rep.radial_m)
    q_proxy = _qg_pv_proxy(psi, rep.radial_m, rep.theta_rad, rep.depth_m, f0, n2)
    if buoyancy_source == "thermal_wind":
        buoyancy = _thermal_wind_buoyancy(rep.u, rep.v, rep.radial_m, rep.theta_rad, rep.depth_m, f0)
    elif buoyancy_source == "streamfunction_dz":
        buoyancy = _streamfunction_buoyancy(psi, rep.depth_m, f0)
    else:
        raise ValueError(f"Unsupported buoyancy source: {buoyancy_source}")
    x_km, y_km = rep.mesh_xy_km
    rows = []
    masks = []
    for iz in range(rep.depth_m.size):
        row = _layer_diagnostics(
            rep=rep,
            axis=axis,
            q_proxy=q_proxy,
            buoyancy=buoyancy,
            psi=psi,
            depth_index=iz,
            x_km=x_km,
            y_km=y_km,
            core_radius_over_R=request.core_radius_over_R,
            speed_core_quantile=request.speed_core_quantile,
            pv_core_quantile=request.pv_core_quantile,
            min_mask_fraction=request.min_mask_fraction,
        )
        rows.append(row)
        mask, _ = _layer_material_mask(
            rep=rep,
            axis=axis,
            q_proxy=q_proxy,
            depth_index=iz,
            x_km=x_km,
            y_km=y_km,
            core_radius_over_R=request.core_radius_over_R,
            speed_core_quantile=request.speed_core_quantile,
            pv_core_quantile=request.pv_core_quantile,
            min_mask_fraction=request.min_mask_fraction,
        )
        masks.append(mask)
    return pd.DataFrame(rows), {"psi": psi, "q_proxy": q_proxy, "buoyancy": buoyancy, "mask": np.asarray(masks)}


def _combo_output(root: Path, shape: str, axis_source: str, orientation: str, buoyancy_source: str) -> Path:
    return root / shape / axis_source / orientation / buoyancy_source


def _summary_table(profiles: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for polarity, sub in profiles.groupby("polarity", sort=True):
        rows.append(
            {
                "polarity": polarity,
                "n_tau": int(sub["tau"].nunique()),
                "n_depth": int(sub["depth_m"].nunique()),
                "finite_G_fraction": float(np.nanmean(np.isfinite(sub["G_magnitude_proxy"].to_numpy(float)))),
                "mask_fraction_median": float(np.nanmedian(sub["mask_fraction"].to_numpy(float))),
                "boundary_leakage_median_ms": float(np.nanmedian(sub["boundary_leakage_proxy_mean_abs_ms"].to_numpy(float))),
                "pv_centroid_offset_median_km": float(np.nanmedian(sub["pv_centroid_distance_from_axis_km"].to_numpy(float))),
                "weak_centroid_offset_median_km": float(np.nanmedian(sub["weak_speed_centroid_distance_from_axis_km"].to_numpy(float))),
                "G_magnitude_median": float(np.nanmedian(np.abs(sub["G_magnitude_proxy"].to_numpy(float)))),
                "pv_flux_magnitude_median": float(np.nanmedian(np.abs(sub["pv_flux_magnitude"].to_numpy(float)))),
            }
        )
    return pd.DataFrame(rows)


def _write_table(table: pd.DataFrame, csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(csv_path, index=False)
    try:
        table.to_parquet(csv_path.with_suffix(".parquet"), index=False)
    except Exception:
        pass


def _matrix(table: pd.DataFrame, field: str, polarity: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sub = table[table["polarity"].astype(str).eq(str(polarity))]
    tau = np.sort(sub["tau"].dropna().unique().astype(float))
    depth = np.sort(sub["depth_m"].dropna().unique().astype(float))
    image = np.full((depth.size, tau.size), np.nan)
    for iz, z in enumerate(depth):
        zsub = sub[np.isclose(sub["depth_m"].astype(float), z)]
        for it, t in enumerate(tau):
            part = zsub[np.isclose(zsub["tau"].astype(float), t)]
            if not part.empty:
                image[iz, it] = np.nanmedian(part[field].to_numpy(float))
    return tau, depth, image


def _plot_tau_depth(table: pd.DataFrame, field: str, path: Path, *, title: str, cmap: str = "viridis") -> None:
    import matplotlib.pyplot as plt

    polarities = list(table["polarity"].drop_duplicates())
    fig, axes = plt.subplots(1, len(polarities), figsize=(5.5 * len(polarities), 4.7), squeeze=False)
    for ax, polarity in zip(axes[0], polarities):
        tau, depth, image = _matrix(table, field, polarity)
        if cmap == "coolwarm":
            vmax = np.nanpercentile(np.abs(image), 95) if np.any(np.isfinite(image)) else 1.0
            im = ax.pcolormesh(tau, depth, image, shading="auto", cmap=cmap, vmin=-vmax, vmax=vmax)
        else:
            im = ax.pcolormesh(tau, depth, image, shading="auto", cmap=cmap)
        ax.invert_yaxis()
        ax.set_xlabel("life phase tau")
        ax.set_ylabel("depth (m)")
        ax.set_title(str(polarity))
        fig.colorbar(im, ax=ax, shrink=0.82)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_mask_example(debug: dict[str, np.ndarray], rep: RepresentativeSlice, path: Path, *, title: str) -> None:
    import matplotlib.pyplot as plt

    x_km, y_km = rep.mesh_xy_km
    depth_idx = int(np.nanargmin(np.abs(rep.depth_m - 500.0)))
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.0), squeeze=False)
    speed = rep.speed[depth_idx]
    mask = debug["mask"][depth_idx]
    q_proxy = debug["q_proxy"][depth_idx]
    for ax, values, panel_title, cmap in [
        (axes[0, 0], speed, "|u_h| with material mask", "coolwarm"),
        (axes[0, 1], q_proxy, "PV proxy with material mask", "RdBu_r"),
    ]:
        im = ax.pcolormesh(x_km, y_km, values, shading="auto", cmap=cmap)
        ax.contour(x_km, y_km, mask.astype(float), levels=[0.5], colors="k", linewidths=1.2)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x (km)")
        ax.set_ylabel("y (km)")
        ax.set_title(f"{panel_title}, z={rep.depth_m[depth_idx]:.0f} m")
        fig.colorbar(im, ax=ax, shrink=0.82)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_relation(table: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.0, 4.8))
    for polarity, sub in table.groupby("polarity", sort=True):
        ax.scatter(
            sub["boundary_leakage_proxy_mean_abs_ms"],
            sub["G_magnitude_proxy"],
            s=12,
            alpha=0.5,
            label=str(polarity),
        )
    ax.set_xlabel("boundary leakage proxy mean |u_n| (m/s)")
    ax.set_ylabel("|G| momentum-divergence proxy")
    ax.set_title("Material-Volume EP: boundary leakage vs volume forcing proxy")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _write_summary_md(path: Path, combo: dict[str, str], summary: pd.DataFrame, request: MaterialVolumeRequest) -> None:
    lines = [
        "# Material-Volume EP Validation Summary",
        "",
        "## 口径",
        f"- shape: `{combo['shape']}`",
        f"- axis source: `{combo['axis_source']}`",
        f"- orientation: `{combo['orientation']}`",
        f"- buoyancy source: `{combo['buoyancy_source']}`",
        "- object: representative coherent material volume in Cartesian coordinates",
        "",
        "## 验证版定义",
        f"- core radius: `r/R <= {request.core_radius_over_R}`",
        f"- speed core quantile: `{request.speed_core_quantile}`",
        f"- abs(PV proxy) core quantile: `{request.pv_core_quantile}`",
        "- mask: low-speed core OR high-|PV proxy| core, keeping the component connected to the axis.",
        "",
        "## 结果摘要",
        "```text",
        summary.to_string(index=False),
        "```",
        "",
        "## 判读",
        "- 这是大曲率 Material-Volume EP 的代表涡验证版，不是最终 object-level material-boundary 闭合。",
        "- `R_ij/B_i/P_i` 在 Cartesian 坐标中定义，避免依赖已经失效的小曲率曲管 Jacobian。",
        "- `G_x_proxy/G_y_proxy` 目前是动量通量水平散度代理，尚未包含完整 QG/PV 反演算子 `T_ij[B_j]`。",
        "- `boundary_leakage_proxy` 越大，越说明当前 mask 不是严格材料边界，不能把体内 forcing 单独解释为全部动力。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_material_volume_validation(request: MaterialVolumeRequest) -> dict[str, Path]:
    bad_axis = sorted(set(request.axis_sources) - set(AXIS_SOURCES))
    bad_buoy = sorted(set(request.buoyancy_sources) - set(BUOYANCY_SOURCES))
    if bad_axis or bad_buoy:
        raise ValueError(f"Bad options: axis={bad_axis}, buoyancy={bad_buoy}")
    if request.core_radius_over_R <= 0:
        raise ValueError("core_radius_over_R must be positive")
    if not 0.0 < request.speed_core_quantile < 1.0:
        raise ValueError("speed_core_quantile must be between 0 and 1")
    if not 0.0 < request.pv_core_quantile < 1.0:
        raise ValueError("pv_core_quantile must be between 0 and 1")

    written: dict[str, Path] = {}
    root_summaries: list[pd.DataFrame] = []
    manifest_rows: list[dict[str, object]] = []
    for shape in request.shapes:
        output_name = shape_output_name(shape)
        radial_root = default_radial_seed_root(request.result_root, output_name)
        for orientation in request.orientations:
            me_root = default_me_liutex_root(request.result_root, output_name, orientation)
            npz_path = me_root / "azimuthal_representative_velocity.npz"
            if request.dry_run:
                print(f"[dry-run] shape={shape} orientation={orientation} me_root={me_root}")
                continue
            _require_pandas()
            if not npz_path.exists():
                if request.skip_missing:
                    continue
                raise FileNotFoundError(npz_path)
            from .diagnostics import load_n2_profile, resolve_n2_profile_path
            from .fields import RepresentativeVortexDataset
            from .geometry import AxisLine

            dataset = RepresentativeVortexDataset.load(npz_path, radial_root)
            tau_grid = _tau_grid(dataset, request.tau_values)
            for axis_source in request.axis_sources:
                for buoyancy_source in request.buoyancy_sources:
                    combo_dir = _combo_output(request.output_root, shape, axis_source, orientation, buoyancy_source)
                    combo_dir.mkdir(parents=True, exist_ok=True)
                    combo = {
                        "shape": shape,
                        "axis_source": axis_source,
                        "orientation": orientation,
                        "buoyancy_source": buoyancy_source,
                    }
                    profiles_parts: list[pd.DataFrame] = []
                    first_debug: tuple[dict[str, np.ndarray], RepresentativeSlice] | None = None
                    for tau in tau_grid:
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
                            shape_label=f"{shape}-only",
                            run_label="material_volume_validation",
                        )
                        axis_path = config.axis_source_path
                        if not axis_path.exists():
                            raise FileNotFoundError(
                                f"Axis source does not exist: {axis_path}. "
                                "Run src.post.cli build-representative-axis-sources first."
                            )
                        n2_path = resolve_n2_profile_path(config, request.n2_profile)
                        for polarity in dataset.polarities:
                            rep = dataset.slice(polarity, float(tau))
                            axis = AxisLine.from_csv(axis_path, polarity=polarity).interpolate_to(rep.depth_m)
                            n2 = load_n2_profile(n2_path, rep.depth_m, request.constant_n2)
                            table, debug = _compute_one_slice(
                                rep=rep,
                                axis=axis,
                                f0=config.f0,
                                n2=n2,
                                buoyancy_source=buoyancy_source,
                                request=request,
                            )
                            table["shape"] = shape
                            table["axis_source"] = axis_source
                            table["orientation"] = orientation
                            table["buoyancy_source"] = buoyancy_source
                            table["polarity"] = polarity
                            table["tau"] = float(rep.tau)
                            profiles_parts.append(table)
                            if first_debug is None and np.isclose(float(rep.tau), 0.5, atol=0.026):
                                first_debug = (debug, rep)
                    profiles = pd.concat(profiles_parts, ignore_index=True)
                    summary = _summary_table(profiles)
                    _write_table(profiles, combo_dir / "material_volume_profiles.csv")
                    _write_table(summary, combo_dir / "material_volume_summary.csv")
                    manifest = {
                        "combo": combo,
                        "result_root": request.result_root,
                        "me_liutex_root": me_root,
                        "radial_seed_root": radial_root,
                        "tau_values": [float(v) for v in tau_grid],
                        "core_radius_over_R": request.core_radius_over_R,
                        "speed_core_quantile": request.speed_core_quantile,
                        "pv_core_quantile": request.pv_core_quantile,
                        "min_mask_fraction": request.min_mask_fraction,
                        "n2_profile": request.n2_profile,
                        "diagnostic_status": "representative-volume validation; not object-level material-boundary closure",
                        "G_proxy_definition": "horizontal divergence of momentum stress only; buoyancy mapping T_ij[B_j] not closed in v1",
                    }
                    (combo_dir / "material_volume_manifest.json").write_text(
                        json.dumps(_json_ready(manifest), ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                    _write_summary_md(combo_dir / "material_volume_ep_validation_summary_zh.md", combo, summary, request)
                    figures = combo_dir / "figures"
                    figures.mkdir(exist_ok=True)
                    _plot_tau_depth(
                        profiles,
                        "G_magnitude_proxy",
                        figures / "cartesian_tensor_flux_tau_depth.png",
                        title="Material-Volume EP / Cartesian |G| proxy",
                    )
                    _plot_tau_depth(
                        profiles,
                        "pv_centroid_distance_from_axis_km",
                        figures / "pv_flux_centroid_drift.png",
                        title="PV-weighted centroid offset from axis",
                    )
                    _plot_tau_depth(
                        profiles,
                        "boundary_leakage_proxy_mean_abs_ms",
                        figures / "boundary_leakage_tau_depth.png",
                        title="Boundary leakage proxy: edge mean |u_n|",
                    )
                    _plot_tau_depth(
                        profiles,
                        "mask_core_fraction",
                        figures / "curved_tube_failure_vs_material_volume.png",
                        title="Material-volume support despite curved-tube metric failure",
                    )
                    _plot_relation(profiles, figures / "tilted_ep_vs_material_volume_forcing.png")
                    if first_debug is not None:
                        debug, rep = first_debug
                        _plot_mask_example(
                            debug,
                            rep,
                            figures / "material_volume_mask_examples.png",
                            title=f"Material-Volume mask example: {shape}, {orientation}, {axis_source}, tau={rep.tau:.2f}",
                        )
                    root_summaries.append(summary.assign(**combo))
                    manifest_rows.append(manifest)
                    written[str(combo_dir)] = combo_dir
    if not request.dry_run and root_summaries:
        root_summary = pd.concat(root_summaries, ignore_index=True)
        _write_table(root_summary, request.output_root / "material_volume_all_combo_summary.csv")
        (request.output_root / "material_volume_manifest.json").write_text(
            json.dumps(_json_ready(manifest_rows), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        written["root_summary"] = request.output_root / "material_volume_all_combo_summary.csv"
    return written


def request_from_args(args) -> MaterialVolumeRequest:
    return MaterialVolumeRequest(
        result_root=Path(args.result_root),
        output_root=Path(args.output_root),
        shapes=_split_csv(args.shapes),
        axis_sources=_split_csv(args.axis_sources),
        orientations=_split_csv(args.orientations),
        buoyancy_sources=_split_csv(args.buoyancy_sources),
        tau_values=_parse_tau_values(args.tau_values),
        reference_lat=float(args.reference_lat),
        constant_n2=float(args.constant_n2),
        n2_profile=args.n2_profile,
        core_radius_over_R=float(args.core_radius_over_R),
        speed_core_quantile=float(args.speed_core_quantile),
        pv_core_quantile=float(args.pv_core_quantile),
        min_mask_fraction=float(args.min_mask_fraction),
        skip_missing=bool(args.skip_missing),
        dry_run=bool(args.dry_run),
    )
