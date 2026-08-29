from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from netCDF4 import Dataset, num2date


EARTH_RADIUS_M = 6_371_000.0
G = 9.81
RHO0 = 1025.0
OMEGA_EARTH = 7.2921159e-5


@dataclass(frozen=True)
class JumpCandidate:
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


def _meters_per_degree(lat_deg: float) -> tuple[float, float]:
    dy = np.pi * EARTH_RADIUS_M / 180.0
    dx = dy * np.cos(np.deg2rad(lat_deg))
    return float(dx), float(dy)


def _read_time_index(path: Path, date: str) -> int:
    with Dataset(path) as ds:
        time = ds.variables["time"]
        dates = num2date(time[:], units=time.units, calendar=getattr(time, "calendar", "standard"))
        labels = [getattr(d, "strftime")("%Y-%m-%d") for d in dates]
    try:
        return labels.index(date)
    except ValueError as exc:
        raise ValueError(f"{date} not found in {path}") from exc


def _window_indices(values: np.ndarray, center: float, half_width: float) -> np.ndarray:
    idx = np.where((values >= center - half_width) & (values <= center + half_width))[0]
    if idx.size >= 5:
        return idx
    nearest = int(np.nanargmin(np.abs(values - center)))
    return np.arange(max(0, nearest - 4), min(values.size, nearest + 5))


def _relative_xy(lon: np.ndarray, lat: np.ndarray, lon0: float, lat0: float) -> tuple[np.ndarray, np.ndarray]:
    dx_m, dy_m = _meters_per_degree(lat0)
    return (lon - lon0) * dx_m, (lat - lat0) * dy_m


def _clean_field(values: np.ndarray) -> np.ndarray:
    arr = np.ma.filled(values, np.nan).astype("f8", copy=False)
    arr = np.asarray(arr, dtype="f8")
    arr[np.abs(arr) > 1e10] = np.nan
    return arr


def _vertical_velocity_proxy(u3: np.ndarray, v3: np.ndarray, depth: np.ndarray, x_m: np.ndarray, y_m: np.ndarray) -> np.ndarray:
    dudx = np.gradient(u3, x_m, axis=2, edge_order=1)
    dvdy = np.gradient(v3, y_m, axis=1, edge_order=1)
    div = dudx + dvdy
    w = np.zeros_like(div)
    for k in range(1, len(depth)):
        dz = float(depth[k] - depth[k - 1])
        w[k] = w[k - 1] - 0.5 * (div[k] + div[k - 1]) * dz
    return w


def _load_inputs(results_root: Path, shape_dir_name: str, shapes: set[str], year_limit: int | None) -> pd.DataFrame:
    centers = pd.read_parquet(results_root / "catalog" / "layer_centers_completed.parquet")
    shape = pd.read_parquet(results_root / shape_dir_name / "shape_tracks.parquet")
    shape = shape[shape["shape_class"].astype(str).isin(shapes)]
    keep = centers.merge(shape[["track3d_id", "shape_class"]], on="track3d_id", how="inner")
    keep["date"] = keep["date"].astype(str)
    if year_limit is not None:
        keep = keep[keep["date"].str.slice(0, 4).astype(int).le(year_limit)].copy()
    if keep.empty:
        raise ValueError(f"No object layers found for shapes={sorted(shapes)}")
    return keep


def _object_offsets(object_layers: pd.DataFrame) -> pd.DataFrame:
    obj = object_layers.sort_values("depth_index").copy()
    surface = obj.iloc[0]
    dx_m, dy_m = _meters_per_degree(float(surface["latitude"]))
    obj["delta_x_km"] = (obj["longitude"] - float(surface["longitude"])) * dx_m / 1000.0
    obj["delta_y_km"] = (obj["latitude"] - float(surface["latitude"])) * dy_m / 1000.0
    return obj


def _jump_candidates(object_layers: pd.DataFrame, jump_ranks: int) -> list[JumpCandidate]:
    obj = _object_offsets(object_layers)
    if len(obj) < 2:
        return []
    surface = obj.iloc[0]
    radius_m = float(np.nanmedian(obj["radius_m"].to_numpy(dtype="f8")))
    x = obj["delta_x_km"].to_numpy(dtype="f8")
    y = obj["delta_y_km"].to_numpy(dtype="f8")
    jumps_km = np.hypot(np.diff(x), np.diff(y))
    order = np.argsort(jumps_km)[::-1]
    out: list[JumpCandidate] = []
    for rank, idx in enumerate(order[: max(0, jump_ranks)], start=1):
        upper = obj.iloc[int(idx)]
        lower = obj.iloc[int(idx) + 1]
        jump_km = float(jumps_km[int(idx)])
        out.append(
            JumpCandidate(
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


def _read_velocity_column(path: Path, date: str, lon0: float, lat0: float, half_width_deg: float) -> dict[str, np.ndarray]:
    with Dataset(path) as ds:
        t = _read_time_index(path, date)
        lon = np.asarray(ds.variables["longitude"][:], dtype="f8")
        lat = np.asarray(ds.variables["latitude"][:], dtype="f8")
        depth = np.asarray(ds.variables["depth"][:], dtype="f8")
        ix = _window_indices(lon, lon0, half_width_deg)
        iy = _window_indices(lat, lat0, half_width_deg)
        u = _clean_field(ds.variables["uo_glor"][t, :, iy.min() : iy.max() + 1, ix.min() : ix.max() + 1])
        v = _clean_field(ds.variables["vo_glor"][t, :, iy.min() : iy.max() + 1, ix.min() : ix.max() + 1])
        zos = _clean_field(ds.variables["zos_glor"][t, iy.min() : iy.max() + 1, ix.min() : ix.max() + 1])
    return {"lon": lon[ix], "lat": lat[iy], "depth": depth, "u": u, "v": v, "zos": zos}


def _read_raw_column(path: Path, date: str, lon0: float, lat0: float, half_width_deg: float) -> dict[str, np.ndarray]:
    with Dataset(path) as ds:
        t = _read_time_index(path, date)
        lon = np.asarray(ds.variables["longitude"][:], dtype="f8")
        lat = np.asarray(ds.variables["latitude"][:], dtype="f8")
        depth = np.asarray(ds.variables["depth"][:], dtype="f8")
        ix = _window_indices(lon, lon0, half_width_deg)
        iy = _window_indices(lat, lat0, half_width_deg)
        theta = _clean_field(ds.variables["thetao_glor"][t, :, iy.min() : iy.max() + 1, ix.min() : ix.max() + 1])
        salt = _clean_field(ds.variables["so_glor"][t, :, iy.min() : iy.max() + 1, ix.min() : ix.max() + 1])
    return {"lon": lon[ix], "lat": lat[iy], "depth": depth, "theta": theta, "salt": salt}


def _fill_nan_nearest(arr: np.ndarray) -> np.ndarray:
    from scipy import ndimage

    out = np.asarray(arr, dtype="f8").copy()
    mask = ~np.isfinite(out)
    if not mask.any():
        return out
    if mask.all():
        return np.zeros_like(out)
    idx = ndimage.distance_transform_edt(mask, return_distances=False, return_indices=True)
    out[mask] = out[tuple(ind[mask] for ind in idx)]
    return out


def _density_sigma0(raw_column: dict[str, np.ndarray], lon2: np.ndarray, lat2: np.ndarray) -> np.ndarray:
    try:
        import gsw
    except ImportError as exc:
        raise RuntimeError("gsw is required for omega vertical velocity diagnostics") from exc

    theta = _fill_nan_nearest(np.asarray(raw_column["theta"], dtype="f8"))
    salt = _fill_nan_nearest(np.asarray(raw_column["salt"], dtype="f8"))
    depth = np.asarray(raw_column["depth"], dtype="f8")
    pressure = gsw.p_from_z(-depth[:, None, None], lat2[None, :, :])
    sa = gsw.SA_from_SP(salt, pressure, lon2[None, :, :], lat2[None, :, :])
    ct = gsw.CT_from_pt(sa, theta)
    return np.asarray(gsw.sigma0(sa, ct), dtype="f8")


def _thermal_wind_geostrophic_velocity(
    column: dict[str, np.ndarray],
    raw_column: dict[str, np.ndarray],
    x_m: np.ndarray,
    y_m: np.ndarray,
    f0: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lon2, lat2 = np.meshgrid(np.asarray(column["lon"], dtype="f8"), np.asarray(column["lat"], dtype="f8"))
    sigma0 = _density_sigma0(raw_column, lon2, lat2)
    eta = _fill_nan_nearest(np.asarray(column["zos"], dtype="f8"))
    eta_y, eta_x = np.gradient(eta, y_m, x_m, edge_order=1)
    ug = np.zeros_like(sigma0)
    vg = np.zeros_like(sigma0)
    ug[0] = -(G / f0) * eta_y
    vg[0] = (G / f0) * eta_x
    rho_y = np.gradient(sigma0, y_m, axis=1, edge_order=1)
    rho_x = np.gradient(sigma0, x_m, axis=2, edge_order=1)
    depth = np.asarray(raw_column["depth"], dtype="f8")
    for k in range(1, len(depth)):
        dz = float(depth[k] - depth[k - 1])
        du_dz = -(G / (RHO0 * f0)) * 0.5 * (rho_y[k] + rho_y[k - 1])
        dv_dz = (G / (RHO0 * f0)) * 0.5 * (rho_x[k] + rho_x[k - 1])
        ug[k] = ug[k - 1] + du_dz * dz
        vg[k] = vg[k - 1] + dv_dz * dz
    return ug, vg, sigma0


def _omega_forcing_qtw_qdag(
    ug: np.ndarray,
    vg: np.ndarray,
    ua: np.ndarray,
    va: np.ndarray,
    sigma0: np.ndarray,
    depth: np.ndarray,
    x_m: np.ndarray,
    y_m: np.ndarray,
) -> np.ndarray:
    buoyancy = -G * sigma0 / RHO0
    bx = np.gradient(buoyancy, x_m, axis=2, edge_order=1)
    by = np.gradient(buoyancy, y_m, axis=1, edge_order=1)
    ugx = np.gradient(ug, x_m, axis=2, edge_order=1)
    ugy = np.gradient(ug, y_m, axis=1, edge_order=1)
    vgx = np.gradient(vg, x_m, axis=2, edge_order=1)
    vgy = np.gradient(vg, y_m, axis=1, edge_order=1)
    uax = np.gradient(ua, x_m, axis=2, edge_order=1)
    uay = np.gradient(ua, y_m, axis=1, edge_order=1)
    vax = np.gradient(va, x_m, axis=2, edge_order=1)
    vay = np.gradient(va, y_m, axis=1, edge_order=1)
    qtw_x = ugx * bx + vgx * by
    qtw_y = ugy * bx + vgy * by
    qdag_x = uax * bx + vax * by
    qdag_y = uay * bx + vay * by
    qx = -2.0 * qtw_x + qdag_x
    qy = -2.0 * qtw_y + qdag_y
    return np.gradient(qx, x_m, axis=2, edge_order=1) + np.gradient(qy, y_m, axis=1, edge_order=1)


def _solve_omega_dirichlet(div_q: np.ndarray, n2: np.ndarray, depth: np.ndarray, x_m: np.ndarray, y_m: np.ndarray, f0: float) -> np.ndarray:
    from scipy import sparse
    from scipy.sparse.linalg import spsolve

    nz, ny, nx = div_q.shape
    dx = float(np.nanmedian(np.diff(x_m)))
    dy = float(np.nanmedian(np.diff(y_m)))
    dz = float(np.nanmedian(np.diff(depth)))
    n = nz * ny * nx
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    rhs = np.asarray(div_q, dtype="f8").reshape(-1)

    def idx(k: int, j: int, i: int) -> int:
        return (k * ny + j) * nx + i

    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                row = idx(k, j, i)
                if k in (0, nz - 1) or j in (0, ny - 1) or i in (0, nx - 1):
                    rows.append(row); cols.append(row); data.append(1.0); rhs[row] = 0.0
                    continue
                c = -2.0 * n2[k] / dx**2 - 2.0 * n2[k] / dy**2 - 2.0 * f0**2 / dz**2
                rows.append(row); cols.append(row); data.append(c)
                for kk, jj, ii, val in (
                    (k, j, i - 1, n2[k] / dx**2),
                    (k, j, i + 1, n2[k] / dx**2),
                    (k, j - 1, i, n2[k] / dy**2),
                    (k, j + 1, i, n2[k] / dy**2),
                    (k - 1, j, i, f0**2 / dz**2),
                    (k + 1, j, i, f0**2 / dz**2),
                ):
                    rows.append(row); cols.append(idx(kk, jj, ii)); data.append(float(val))
    mat = sparse.csr_matrix((data, (rows, cols)), shape=(n, n))
    return np.asarray(spsolve(mat, rhs), dtype="f8").reshape((nz, ny, nx))


def _omega_vertical_velocity(column: dict[str, np.ndarray], raw_column: dict[str, np.ndarray], x_m: np.ndarray, y_m: np.ndarray) -> np.ndarray:
    lat0 = float(np.nanmean(column["lat"]))
    f0 = 2.0 * OMEGA_EARTH * np.sin(np.deg2rad(lat0))
    if abs(f0) < 1e-8:
        raise ValueError("Coriolis parameter is too small for omega diagnostics")
    ug, vg, sigma0 = _thermal_wind_geostrophic_velocity(column, raw_column, x_m, y_m, f0)
    u = _fill_nan_nearest(np.asarray(column["u"], dtype="f8"))
    v = _fill_nan_nearest(np.asarray(column["v"], dtype="f8"))
    div_q = _omega_forcing_qtw_qdag(ug, vg, u - ug, v - vg, sigma0, column["depth"], x_m, y_m)
    n2 = np.nanmedian(np.gradient(-G * sigma0 / RHO0, column["depth"], axis=0, edge_order=1), axis=(1, 2))
    n2 = np.clip(_fill_nan_nearest(n2), 1e-7, 1e-3)
    return _solve_omega_dirichlet(div_q, n2, column["depth"], x_m, y_m, f0)


def _vertical_velocity_fields(
    column: dict[str, np.ndarray],
    raw_column: dict[str, np.ndarray] | None,
    method: str,
    lon0: float,
    lat0: float,
) -> tuple[np.ndarray, np.ndarray]:
    x_m, y_m = _relative_xy(column["lon"], column["lat"], lon0, lat0)
    if method == "proxy":
        w = _vertical_velocity_proxy(column["u"], column["v"], column["depth"], x_m, y_m)
    elif method == "omega":
        if raw_column is None:
            raise ValueError("raw_column is required when vertical_velocity_method='omega'")
        w = _omega_vertical_velocity(column, raw_column, x_m, y_m)
    else:
        raise ValueError(f"Unsupported vertical velocity method: {method}")
    dwdz = np.gradient(w, column["depth"], axis=0, edge_order=1) if len(column["depth"]) > 1 else np.zeros_like(w)
    return w, dwdz


def _parallel_axis_aligned_section(jump: JumpCandidate, column: dict[str, np.ndarray], w: np.ndarray, dwdz: np.ndarray) -> dict[str, np.ndarray]:
    x_m, y_m = _relative_xy(column["lon"], column["lat"], jump.surface_lon, jump.surface_lat)
    dx = jump.to_x_km - jump.from_x_km
    dy = jump.to_y_km - jump.from_y_km
    if abs(dx) >= abs(dy):
        iy = int(np.nanargmin(np.abs(y_m / 1000.0 - jump.to_y_km)))
        coord = x_m / 1000.0
        w_sec = w[:, iy, :]
        dwdz_sec = dwdz[:, iy, :]
        target = jump.to_x_km
    else:
        ix = int(np.nanargmin(np.abs(x_m / 1000.0 - jump.to_x_km)))
        coord = y_m / 1000.0
        w_sec = w[:, :, ix]
        dwdz_sec = dwdz[:, :, ix]
        target = jump.to_y_km
    return {"coord": coord, "depth": column["depth"], "w": w_sec, "dwdz": dwdz_sec, "target": np.array(target)}


def _interpolate_vertical_section(
    field: np.ndarray,
    x_m: np.ndarray,
    y_m: np.ndarray,
    x_line_km: np.ndarray,
    y_line_km: np.ndarray,
) -> np.ndarray:
    from scipy.interpolate import RegularGridInterpolator

    points = np.column_stack([y_line_km * 1000.0, x_line_km * 1000.0])
    section = np.full((field.shape[0], len(x_line_km)), np.nan, dtype="f8")
    for k in range(field.shape[0]):
        interp = RegularGridInterpolator((y_m, x_m), field[k], bounds_error=False, fill_value=np.nan)
        section[k] = interp(points)
    return section


def _normal_interpolated_section(jump: JumpCandidate, column: dict[str, np.ndarray], w: np.ndarray, dwdz: np.ndarray) -> dict[str, np.ndarray]:
    x_m, y_m = _relative_xy(column["lon"], column["lat"], jump.surface_lon, jump.surface_lat)
    dx = jump.to_x_km - jump.from_x_km
    dy = jump.to_y_km - jump.from_y_km
    norm = float(np.hypot(dx, dy))
    if not np.isfinite(norm) or norm <= 1e-9:
        ex, ey = 1.0, 0.0
    else:
        ex, ey = -dy / norm, dx / norm
    mid_x = 0.5 * (jump.from_x_km + jump.to_x_km)
    mid_y = 0.5 * (jump.from_y_km + jump.to_y_km)
    max_half = max(float(jump.radius_m) / 1000.0 * 2.5, 150.0)
    coord = np.linspace(-max_half, max_half, 161, dtype="f8")
    x_line = mid_x + coord * ex
    y_line = mid_y + coord * ey
    return {
        "coord": coord,
        "depth": column["depth"],
        "w": _interpolate_vertical_section(w, x_m, y_m, x_line, y_line),
        "dwdz": _interpolate_vertical_section(dwdz, x_m, y_m, x_line, y_line),
        "target": np.array(0.0),
    }


def _section_for_jump(jump: JumpCandidate, column: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    w, dwdz = _vertical_velocity_fields(column, None, "proxy", jump.surface_lon, jump.surface_lat)
    return _parallel_axis_aligned_section(jump, column, w, dwdz)


METRIC_FIELDS = (
    "local_abs_dwdz_peak",
    "local_abs_dwdz_p90",
    "local_abs_dwdz_mean",
    "background_abs_dwdz_median",
    "shear_enrichment_ratio",
    "sign_change_fraction",
    "edge_alignment_score",
    "qualitative_relation",
)


def _parse_section_modes(section_modes: str) -> tuple[str, ...]:
    aliases = {"parallel_axis_aligned": "parallel", "normal_interpolated": "normal"}
    modes: list[str] = []
    for raw in section_modes.split(","):
        mode = aliases.get(raw.strip(), raw.strip())
        if not mode:
            continue
        if mode not in {"parallel", "normal"}:
            raise ValueError(f"Unsupported section mode: {raw}")
        if mode not in modes:
            modes.append(mode)
    return tuple(modes or ["parallel"])


def _jump_metrics(jump: JumpCandidate, section: dict[str, np.ndarray], depth_padding_layers: int, half_width_r: float, min_half_width_km: float) -> dict[str, object]:
    coord = section["coord"]
    depth = section["depth"]
    w = section["w"]
    dwdz = section["dwdz"]
    target = float(section["target"])
    x_half = max(float(jump.radius_m) / 1000.0 * half_width_r, min_half_width_km)
    xmask = (coord >= target - x_half) & (coord <= target + x_half)
    k0 = max(0, min(jump.from_depth_index, jump.to_depth_index) - max(0, depth_padding_layers))
    k1 = min(len(depth) - 1, max(jump.from_depth_index, jump.to_depth_index) + max(0, depth_padding_layers))
    zmask = np.zeros(len(depth), dtype=bool)
    zmask[k0 : k1 + 1] = True
    local = np.abs(dwdz[np.ix_(zmask, xmask)])
    background = np.abs(dwdz[np.ix_(~zmask, xmask)]) if np.any(~zmask) and np.any(xmask) else np.array([])
    local_f = local[np.isfinite(local)]
    back_f = background[np.isfinite(background)]
    if local_f.size < 10:
        return {"qualitative_relation": "insufficient_data"}
    back_med = float(np.nanmedian(back_f)) if back_f.size else np.nan
    p90 = float(np.nanpercentile(local_f, 90))
    peak = float(np.nanmax(local_f))
    mean = float(np.nanmean(local_f))
    ratio = float(p90 / back_med) if np.isfinite(back_med) and back_med > 0 else np.nan
    ku = int(np.clip(jump.from_depth_index, 0, len(depth) - 1))
    kl = int(np.clip(jump.to_depth_index, 0, len(depth) - 1))
    w_upper = w[ku, xmask]
    w_lower = w[kl, xmask]
    valid_pair = np.isfinite(w_upper) & np.isfinite(w_lower)
    sign_change = float(np.mean(np.sign(w_upper[valid_pair]) != np.sign(w_lower[valid_pair]))) if np.any(valid_pair) else np.nan
    upper_shear = np.abs(dwdz[ku, xmask])
    lower_shear = np.abs(dwdz[kl, xmask])
    edge_value = float(np.nanmean(np.r_[upper_shear[np.isfinite(upper_shear)], lower_shear[np.isfinite(lower_shear)]]))
    edge_score = float(edge_value / p90) if p90 > 0 and np.isfinite(edge_value) else np.nan
    if np.isfinite(ratio) and ratio >= 2.0 and ((np.isfinite(sign_change) and sign_change >= 0.2) or (np.isfinite(edge_score) and edge_score >= 0.6)):
        relation = "strong_edge_related"
    elif np.isfinite(ratio) and ratio >= 1.25:
        relation = "weak_or_broad_shear_related"
    else:
        relation = "not_obvious_from_wshear"
    return {
        "local_abs_dwdz_peak": peak,
        "local_abs_dwdz_p90": p90,
        "local_abs_dwdz_mean": mean,
        "background_abs_dwdz_median": back_med,
        "shear_enrichment_ratio": ratio,
        "sign_change_fraction": sign_change,
        "edge_alignment_score": edge_score,
        "qualitative_relation": relation,
    }


def _prefixed_metrics(prefix: str, metrics: dict[str, object]) -> dict[str, object]:
    return {f"{prefix}_{field}": metrics.get(field, np.nan) for field in METRIC_FIELDS}


def _relation_agreement(row: dict[str, object]) -> str:
    parallel = row.get("parallel_qualitative_relation")
    normal = row.get("normal_qualitative_relation")
    if parallel in (None, "not_computed") or normal in (None, "not_computed"):
        return "not_computed"
    if parallel == "insufficient_data" or normal == "insufficient_data":
        return "insufficient_data"
    if parallel == normal:
        return "same"
    if normal == "strong_edge_related" and parallel != "strong_edge_related":
        return "normal_stronger"
    if parallel == "strong_edge_related" and normal != "strong_edge_related":
        return "parallel_stronger"
    return "different_nonstrong"


def _metrics_for_jump(
    jump: JumpCandidate,
    column: dict[str, np.ndarray],
    w: np.ndarray,
    dwdz: np.ndarray,
    section_modes: tuple[str, ...],
    depth_padding_layers: int,
    half_width_r: float,
    min_half_width_km: float,
    vertical_velocity_method: str,
) -> dict[str, object]:
    row: dict[str, object] = dict(jump.__dict__)
    row["vertical_velocity_method"] = vertical_velocity_method
    if "parallel" in section_modes:
        metrics = _jump_metrics(jump, _parallel_axis_aligned_section(jump, column, w, dwdz), depth_padding_layers, half_width_r, min_half_width_km)
        row.update(_prefixed_metrics("parallel", metrics))
    if "normal" in section_modes:
        metrics = _jump_metrics(jump, _normal_interpolated_section(jump, column, w, dwdz), depth_padding_layers, half_width_r, min_half_width_km)
        row.update(_prefixed_metrics("normal", metrics))
    for prefix in ("parallel", "normal"):
        if prefix not in section_modes:
            row.update(_prefixed_metrics(prefix, {"qualitative_relation": "not_computed"}))

    p90_parallel = row.get("parallel_local_abs_dwdz_p90", np.nan)
    p90_normal = row.get("normal_local_abs_dwdz_p90", np.nan)
    enrich_parallel = row.get("parallel_shear_enrichment_ratio", np.nan)
    enrich_normal = row.get("normal_shear_enrichment_ratio", np.nan)
    row["normal_minus_parallel_enrichment"] = (
        float(enrich_normal) - float(enrich_parallel)
        if np.isfinite(enrich_normal) and np.isfinite(enrich_parallel)
        else np.nan
    )
    row["normal_to_parallel_p90_ratio"] = (
        float(p90_normal) / float(p90_parallel)
        if np.isfinite(p90_normal) and np.isfinite(p90_parallel) and float(p90_parallel) > 0
        else np.nan
    )
    row["relation_agreement_class"] = _relation_agreement(row)

    # Backward-compatible aliases keep older plotting scripts usable.
    source = "parallel" if "parallel" in section_modes else "normal"
    for field in METRIC_FIELDS:
        row[field] = row.get(f"{source}_{field}", np.nan)
    return row


def _safe_corr(a: pd.Series, b: pd.Series) -> float:
    ok = np.isfinite(a.to_numpy(dtype="f8")) & np.isfinite(b.to_numpy(dtype="f8"))
    if int(ok.sum()) < 3:
        return float("nan")
    return float(np.corrcoef(a.to_numpy(dtype="f8")[ok], b.to_numpy(dtype="f8")[ok])[0, 1])


def _write_tables(top: pd.DataFrame, all_jumps: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    top.to_parquet(output_dir / "coherent_objectday_jump_wshear_top2.parquet", index=False)
    top.to_csv(output_dir / "coherent_objectday_jump_wshear_top2.csv", index=False)
    all_jumps.to_parquet(output_dir / "coherent_all_layer_jump_wshear.parquet", index=False)
    summary_rows = []
    for key, part in [("top1", top[top["jump_rank"].eq(1)]), ("top2", top)]:
        for mode in ("parallel", "normal"):
            relation_col = f"{mode}_qualitative_relation"
            enrichment_col = f"{mode}_shear_enrichment_ratio"
            p90_col = f"{mode}_local_abs_dwdz_p90"
            if relation_col not in part.columns:
                continue
            counts = part[relation_col].value_counts(dropna=False).to_dict()
            summary_rows.append(
                {
                    "subset": key,
                    "section_mode": mode,
                    "n_jumps": int(len(part)),
                    "n_objectdays": int(part["eddy3d_object_id"].nunique()),
                    "n_tracks": int(part["track3d_id"].nunique()),
                    "corr_jump_over_R_vs_shear_enrichment": _safe_corr(part["jump_distance_over_R"], part[enrichment_col]),
                    "corr_jump_over_R_vs_abs_dwdz_p90": _safe_corr(part["jump_distance_over_R"], part[p90_col]),
                    "strong_edge_related_fraction": float((part[relation_col] == "strong_edge_related").mean()) if len(part) else np.nan,
                    "weak_or_broad_shear_related_fraction": float((part[relation_col] == "weak_or_broad_shear_related").mean()) if len(part) else np.nan,
                    "not_obvious_fraction": float((part[relation_col] == "not_obvious_from_wshear").mean()) if len(part) else np.nan,
                    "insufficient_data_fraction": float((part[relation_col] == "insufficient_data").mean()) if len(part) else np.nan,
                    "relation_counts_json": json.dumps(counts, ensure_ascii=False),
                }
            )
        if {"normal_shear_enrichment_ratio", "parallel_shear_enrichment_ratio"}.issubset(part.columns):
            agreement = part["relation_agreement_class"].value_counts(dropna=False).to_dict()
            summary_rows.append(
                {
                    "subset": key,
                    "section_mode": "parallel_vs_normal",
                    "n_jumps": int(len(part)),
                    "n_objectdays": int(part["eddy3d_object_id"].nunique()),
                    "n_tracks": int(part["track3d_id"].nunique()),
                    "median_normal_minus_parallel_enrichment": float(np.nanmedian(part["normal_minus_parallel_enrichment"])),
                    "median_normal_to_parallel_p90_ratio": float(np.nanmedian(part["normal_to_parallel_p90_ratio"])),
                    "relation_agreement_counts_json": json.dumps(agreement, ensure_ascii=False),
                    "normal_stronger_fraction": float((part["relation_agreement_class"] == "normal_stronger").mean()),
                    "parallel_stronger_fraction": float((part["relation_agreement_class"] == "parallel_stronger").mean()),
                    "same_fraction": float((part["relation_agreement_class"] == "same").mean()),
                }
            )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "coherent_jump_wshear_relation_summary.csv", index=False)
    (output_dir / "coherent_jump_wshear_relation_summary.json").write_text(
        json.dumps(summary_rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def _scatter(ax, df: pd.DataFrame, x: str, y: str, title: str, ylabel: str) -> None:
    colors = {"anticyclonic": "#d95f02", "cyclonic": "#1b9e77"}
    for polarity, part in df.groupby("polarity"):
        ax.scatter(part[x], part[y], s=9, alpha=0.35, label=str(polarity), color=colors.get(str(polarity), "0.4"))
    ax.set_xlabel("jump distance / R")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)


def _plot_outputs(top: pd.DataFrame, summary: pd.DataFrame, output_dir: Path) -> None:
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    rel_order = ["strong_edge_related", "weak_or_broad_shear_related", "not_obvious_from_wshear", "insufficient_data"]

    fig, ax = plt.subplots(figsize=(7, 5))
    _scatter(ax, top, "jump_distance_over_R", "shear_enrichment_ratio", "Jump vs local |dW/dz| enrichment", "Omega-w shear enrichment ratio")
    fig.tight_layout()
    fig.savefig(fig_dir / "jump_over_R_vs_shear_enrichment.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    _scatter(ax, top, "jump_distance_over_R", "local_abs_dwdz_p90", "Jump vs local |dW/dz| p90", "|dW_omega/dz| p90 (s^-1 diagnostic)")
    fig.tight_layout()
    fig.savefig(fig_dir / "jump_over_R_vs_abs_dwdz_p90.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    counts = top["qualitative_relation"].value_counts()
    ax.pie(counts.to_numpy(), labels=counts.index.to_list(), autopct="%1.1f%%", textprops={"fontsize": 8})
    ax.set_title("Qualitative jump-wshear relation")
    fig.tight_layout()
    fig.savefig(fig_dir / "qualitative_relation_pie.png", dpi=200)
    plt.close(fig)

    depth_bins = np.arange(0, max(2100.0, float(np.nanmax(top["to_depth_m"])) + 100.0), 100.0)
    top["depth_bin_m"] = pd.cut(top["to_depth_m"], depth_bins, include_lowest=True)
    depth_rel = top.groupby(["depth_bin_m", "qualitative_relation"], observed=True).size().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(9, 5))
    depth_rel.reindex(columns=rel_order, fill_value=0).plot(kind="bar", stacked=True, ax=ax, width=0.9)
    ax.set_xlabel("jump lower depth bin (m)")
    ax.set_ylabel("count")
    ax.set_title("Qualitative relation by depth")
    ax.tick_params(axis="x", labelsize=7, rotation=70)
    fig.tight_layout()
    fig.savefig(fig_dir / "qualitative_relation_by_depth.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    top.boxplot(column="shear_enrichment_ratio", by="jump_rank", ax=ax)
    ax.set_title("Shear enrichment by jump rank")
    fig.suptitle("")
    ax.set_xlabel("jump rank")
    ax.set_ylabel("shear enrichment ratio")
    fig.tight_layout()
    fig.savefig(fig_dir / "shear_enrichment_by_jump_rank.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    good = top[top["qualitative_relation"].eq("strong_edge_related")]
    hb = ax.hexbin(good["jump_distance_over_R"], good["to_depth_m"], gridsize=35, cmap="viridis", mincnt=1)
    ax.invert_yaxis()
    ax.set_xlabel("jump distance / R")
    ax.set_ylabel("jump lower depth (m)")
    ax.set_title("Depth density of strong jump-wshear relation")
    fig.colorbar(hb, ax=ax, label="count")
    fig.tight_layout()
    fig.savefig(fig_dir / "depth_tau_density_of_jump_related_shear.png", dpi=200)
    plt.close(fig)

    pol = top.groupby(["polarity", "qualitative_relation"]).size().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(7, 5))
    pol.reindex(columns=rel_order, fill_value=0).plot(kind="bar", stacked=True, ax=ax)
    ax.set_ylabel("count")
    ax.set_title("Coherent polarity comparison")
    fig.tight_layout()
    fig.savefig(fig_dir / "coherent_polarity_comparison.png", dpi=200)
    plt.close(fig)

    if {"parallel_shear_enrichment_ratio", "normal_shear_enrichment_ratio"}.issubset(top.columns):
        fig, ax = plt.subplots(figsize=(6, 6))
        for polarity, part in top.groupby("polarity"):
            ax.scatter(
                part["parallel_shear_enrichment_ratio"],
                part["normal_shear_enrichment_ratio"],
                s=9,
                alpha=0.35,
                label=str(polarity),
            )
        finite = top[["parallel_shear_enrichment_ratio", "normal_shear_enrichment_ratio"]].replace([np.inf, -np.inf], np.nan)
        limit = float(np.nanpercentile(finite.to_numpy(dtype="f8"), 98)) if np.isfinite(finite.to_numpy(dtype="f8")).any() else 1.0
        limit = max(1.0, min(limit, 20.0))
        ax.plot([0, limit], [0, limit], color="0.25", lw=1, ls="--")
        ax.set_xlim(0, limit)
        ax.set_ylim(0, limit)
        ax.set_xlabel("parallel shear enrichment")
        ax.set_ylabel("normal shear enrichment")
        ax.set_title("Parallel vs normal Omega-w |dW/dz| enrichment")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(fig_dir / "parallel_vs_normal_shear_enrichment.png", dpi=200)
        plt.close(fig)

        matrix = pd.crosstab(top["parallel_qualitative_relation"], top["normal_qualitative_relation"]).reindex(index=rel_order, columns=rel_order, fill_value=0)
        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(matrix.to_numpy(), cmap="YlGnBu")
        ax.set_xticks(np.arange(len(matrix.columns)), matrix.columns, rotation=35, ha="right", fontsize=8)
        ax.set_yticks(np.arange(len(matrix.index)), matrix.index, fontsize=8)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(j, i, str(int(matrix.iat[i, j])), ha="center", va="center", fontsize=8)
        ax.set_xlabel("normal relation")
        ax.set_ylabel("parallel relation")
        ax.set_title("Parallel vs normal qualitative relation matrix")
        fig.colorbar(im, ax=ax, label="count")
        fig.tight_layout()
        fig.savefig(fig_dir / "parallel_vs_normal_relation_matrix.png", dpi=200)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7, 5))
        _scatter(ax, top, "jump_distance_over_R", "normal_shear_enrichment_ratio", "Normal-section jump vs Omega-w |dW/dz| enrichment", "normal Omega-w shear enrichment ratio")
        fig.tight_layout()
        fig.savefig(fig_dir / "normal_jump_over_R_vs_shear_enrichment.png", dpi=200)
        plt.close(fig)

        normal_depth_rel = top.groupby(["depth_bin_m", "normal_qualitative_relation"], observed=True).size().unstack(fill_value=0)
        fig, ax = plt.subplots(figsize=(9, 5))
        normal_depth_rel.reindex(columns=rel_order, fill_value=0).plot(kind="bar", stacked=True, ax=ax, width=0.9)
        ax.set_xlabel("jump lower depth bin (m)")
        ax.set_ylabel("count")
        ax.set_title("Normal-section qualitative relation by depth")
        ax.tick_params(axis="x", labelsize=7, rotation=70)
        fig.tight_layout()
        fig.savefig(fig_dir / "normal_qualitative_relation_by_depth.png", dpi=200)
        plt.close(fig)

        normal_pol = top.groupby(["polarity", "normal_qualitative_relation"]).size().unstack(fill_value=0)
        fig, ax = plt.subplots(figsize=(7, 5))
        normal_pol.reindex(columns=rel_order, fill_value=0).plot(kind="bar", stacked=True, ax=ax)
        ax.set_ylabel("count")
        ax.set_title("Normal edge-related classification by polarity")
        fig.tight_layout()
        fig.savefig(fig_dir / "normal_edge_related_polarity_comparison.png", dpi=200)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(9, 5))
        top.boxplot(column="normal_to_parallel_p90_ratio", by="depth_bin_m", ax=ax, rot=70)
        ax.set_title("Normal / parallel Omega-w |dW/dz| p90 ratio by depth")
        fig.suptitle("")
        ax.set_xlabel("jump lower depth bin (m)")
        ax.set_ylabel("normal_to_parallel_p90_ratio")
        ax.tick_params(axis="x", labelsize=7)
        fig.tight_layout()
        fig.savefig(fig_dir / "normal_to_parallel_ratio_by_depth.png", dpi=200)
        plt.close(fig)


def _write_report(top: pd.DataFrame, summary: pd.DataFrame, output_dir: Path) -> None:
    top1 = top[top["jump_rank"].eq(1)]
    normal_available = "normal_qualitative_relation" in top.columns and not top["normal_qualitative_relation"].eq("not_computed").all()
    main_prefix = "normal" if normal_available else "parallel"
    relation_col = f"{main_prefix}_qualitative_relation"
    enrichment_col = f"{main_prefix}_shear_enrichment_ratio"
    p90_col = f"{main_prefix}_local_abs_dwdz_p90"
    relation_counts = top[relation_col].value_counts()
    strong_frac = float((top[relation_col] == "strong_edge_related").mean()) if len(top) else np.nan
    depth_strong = top[top[relation_col].eq("strong_edge_related")]["to_depth_m"]
    depth_text = "没有足够的 strong_edge_related 样本"
    if len(depth_strong):
        depth_text = f"{float(depth_strong.quantile(0.25)):.0f}-{float(depth_strong.quantile(0.75)):.0f} m 四分位区间"

    lines = [
        "# Coherent 全体间断点与垂直速度剪切关系诊断",
        "",
        "本诊断只使用 boundary-monotonic、strict-contiguous、life30 的 coherent object-day。它统计的是原始识别对象的间断点关系，不是代表涡旋。",
        "",
        "本版同时输出两种剖面：parallel 剖面近似沿中心跳变路径，normal 剖面沿 (-Delta y, Delta x) 方向并穿过上下两层中心中点。normal 剖面是判断“垂直速度剪切边界是否横切中心跳变”的主口径。",
        "",
        "## 样本量",
        f"- object-day 数：{int(top['eddy3d_object_id'].nunique())}",
        f"- track 数：{int(top['track3d_id'].nunique())}",
        f"- top-1 jump 数：{len(top1)}",
        f"- top-1/top-2 jump 总数：{len(top)}",
        "",
        "## 主口径定量关系",
        f"- 主口径：{main_prefix}",
        f"- top-1/top-2 的 `J/R` 与 `{enrichment_col}` 相关系数：{_safe_corr(top['jump_distance_over_R'], top[enrichment_col]):.3f}",
        f"- top-1/top-2 的 `J/R` 与 `{p90_col}` 相关系数：{_safe_corr(top['jump_distance_over_R'], top[p90_col]):.3f}",
        "",
        "## 定性关系",
        f"- strong_edge_related 比例：{strong_frac:.3f}",
        f"- strong_edge_related 主要深度段：{depth_text}",
        "- 分类计数：",
    ]
    for name, count in relation_counts.items():
        lines.append(f"  - {name}: {int(count)}")
    if {"normal_shear_enrichment_ratio", "parallel_shear_enrichment_ratio"}.issubset(top.columns):
        agreement_counts = top["relation_agreement_class"].value_counts()
        lines.extend(
            [
                "",
                "## Parallel 与 Normal 对比",
                f"- normal - parallel 的剪切增强中位差：{float(np.nanmedian(top['normal_minus_parallel_enrichment'])):.3f}",
                f"- normal / parallel 的 |dw/dz| p90 中位比：{float(np.nanmedian(top['normal_to_parallel_p90_ratio'])):.3f}",
                "- 分类一致性计数：",
            ]
        )
        for name, count in agreement_counts.items():
            lines.append(f"  - {name}: {int(count)}")
    lines.extend(
        [
            "",
            "## 解释边界",
            "`w_diag` 是由 30-180 天带通水平速度散度积分得到的连续方程诊断代理，`partial_z w_diag` 也是代理诊断量，不是直接观测垂直速度。",
            "如果 normal 剖面被标记为 `strong_edge_related`，可以说间断点附近存在横切中心跳变的垂直速度剪切边界证据；但这仍不是因果证明，因为中心跳变还可能受弱速区多核结构、缺测边界、圆周判据切换和速度中心定义影响。",
        ]
    )
    text = "\n".join(lines) + "\n"
    (output_dir / "coherent_jump_wshear_parallel_vs_normal_summary_zh.md").write_text(text, encoding="utf-8")
    (output_dir / "coherent_jump_wshear_relation_summary_zh.md").write_text(text, encoding="utf-8")


def _write_report_clean(top: pd.DataFrame, summary: pd.DataFrame, output_dir: Path) -> None:
    top1 = top[top["jump_rank"].eq(1)]
    normal_available = "normal_qualitative_relation" in top.columns and not top["normal_qualitative_relation"].eq("not_computed").all()
    main_prefix = "normal" if normal_available else "parallel"
    relation_col = f"{main_prefix}_qualitative_relation"
    enrichment_col = f"{main_prefix}_shear_enrichment_ratio"
    p90_col = f"{main_prefix}_local_abs_dwdz_p90"
    relation_counts = top[relation_col].value_counts()
    strong_frac = float((top[relation_col] == "strong_edge_related").mean()) if len(top) else np.nan
    depth_strong = top[top[relation_col].eq("strong_edge_related")]["to_depth_m"]
    depth_text = "没有足够的 strong_edge_related 样本"
    if len(depth_strong):
        depth_text = f"{float(depth_strong.quantile(0.25)):.0f}-{float(depth_strong.quantile(0.75)):.0f} m 四分位区间"
    method = str(top["vertical_velocity_method"].dropna().iloc[0]) if "vertical_velocity_method" in top.columns and top["vertical_velocity_method"].notna().any() else "unknown"
    if method == "omega":
        method_text = "Omega 方程诊断 W，保留 Q=-2Qtw+Qdag；地转速度由带通 SSH 与 raw 温盐热成风估计，非地转速度为 30-180 天带通速度减去地转速度。"
    elif method == "proxy":
        method_text = "连续方程散度积分代理 w_diag，由 30-180 天带通水平速度散度沿深度积分得到。"
    else:
        method_text = f"未知垂直速度口径：{method}"

    lines = [
        "# Coherent 全体间断点与垂直速度剪切关系诊断",
        "",
        "本诊断只使用 boundary-monotonic、strict-contiguous、life30 的 coherent object-day。它统计的是原始识别对象的间断点关系，不是代表涡旋。",
        "",
        f"垂直速度口径：{method_text}",
        "",
        "parallel 剖面近似沿中心跳变路径；normal 剖面沿 (-Delta y, Delta x) 方向并穿过上下两层中心中点。normal 剖面是判断“垂直速度剪切边界是否横切中心跳变”的主口径，parallel 剖面作为路径对照。",
        "",
        "## 样本量",
        f"- object-day 数：{int(top['eddy3d_object_id'].nunique())}",
        f"- track 数：{int(top['track3d_id'].nunique())}",
        f"- top-1 jump 数：{len(top1)}",
        f"- top-1/top-2 jump 总数：{len(top)}",
        "",
        "## 主口径定量关系",
        f"- 主口径：{main_prefix}",
        f"- `J/R` 与 `{enrichment_col}` 相关系数：{_safe_corr(top['jump_distance_over_R'], top[enrichment_col]):.3f}",
        f"- `J/R` 与 `{p90_col}` 相关系数：{_safe_corr(top['jump_distance_over_R'], top[p90_col]):.3f}",
        "",
        "## 定性关系",
        f"- strong_edge_related 比例：{strong_frac:.3f}",
        f"- strong_edge_related 主要深度段：{depth_text}",
        "- 分类计数：",
    ]
    for name, count in relation_counts.items():
        lines.append(f"  - {name}: {int(count)}")

    if {"normal_shear_enrichment_ratio", "parallel_shear_enrichment_ratio"}.issubset(top.columns):
        agreement_counts = top["relation_agreement_class"].value_counts()
        lines.extend(
            [
                "",
                "## Parallel 与 Normal 对比",
                f"- normal - parallel 的剪切增强中位差：{float(np.nanmedian(top['normal_minus_parallel_enrichment'])):.3f}",
                f"- normal / parallel 的 |dW/dz| p90 中位比：{float(np.nanmedian(top['normal_to_parallel_p90_ratio'])):.3f}",
                "- 分类一致性计数：",
            ]
        )
        for name, count in agreement_counts.items():
            lines.append(f"  - {name}: {int(count)}")

    lines.extend(
        [
            "",
            "## 解释边界",
            "`W` 与 `partial_z W` 是诊断量，不是直接观测垂直速度。若 normal 剖面被标记为 `strong_edge_related`，可以说间断点附近存在横切中心跳变的垂直速度剪切边界证据；但这仍不是因果证明，因为中心跳变还可能受弱速区多核结构、缺测边界、圆周判据切换和速度中心定义影响。",
        ]
    )
    text = "\n".join(lines) + "\n"
    (output_dir / "coherent_jump_wshear_parallel_vs_normal_summary_zh.md").write_text(text, encoding="utf-8")
    (output_dir / "coherent_jump_wshear_relation_summary_zh.md").write_text(text, encoding="utf-8")


def analyze_jump_wshear_relation(
    *,
    results_root: Path,
    shape_dir_name: str,
    filter_root: Path,
    raw_root: Path | None,
    output_dir: Path,
    shapes: str,
    jump_ranks: int,
    half_width_deg: float,
    depth_padding_layers: int,
    half_width_r: float,
    min_half_width_km: float,
    section_modes: str,
    vertical_velocity_method: str,
    year_limit: int | None,
    resume: bool,
) -> None:
    shape_set = {part.strip() for part in shapes.split(",") if part.strip()}
    parsed_section_modes = _parse_section_modes(section_modes)
    vertical_velocity_method = vertical_velocity_method.strip().lower()
    if vertical_velocity_method not in {"proxy", "omega"}:
        raise ValueError("--vertical-velocity-method must be proxy or omega")
    if vertical_velocity_method == "omega" and raw_root is None:
        raise ValueError("--raw-root is required for omega diagnostics")
    layers = _load_inputs(results_root, shape_dir_name, shape_set, year_limit)
    output_dir.mkdir(parents=True, exist_ok=True)
    parts_dir = output_dir / "parts"
    parts_dir.mkdir(exist_ok=True)
    if resume and (output_dir / "coherent_objectday_jump_wshear_top2.parquet").exists():
        print(json.dumps({"resume": "final output exists", "output_dir": str(output_dir)}, ensure_ascii=False), flush=True)
        return

    done_ids: set[int] = set()
    if resume:
        for part_path in parts_dir.glob("top2_part_*.parquet"):
            try:
                done_ids.update(pd.read_parquet(part_path, columns=["eddy3d_object_id"])["eddy3d_object_id"].astype(int).unique())
            except Exception:
                continue

    metric_rows: list[dict[str, object]] = []
    all_jump_rows: list[dict[str, object]] = []
    part_index = len(list(parts_dir.glob("top2_part_*.parquet"))) + 1
    flush_every_objectdays = 250
    for n, (object_id, obj) in enumerate(layers.groupby("eddy3d_object_id", sort=False), start=1):
        if int(object_id) in done_ids:
            continue
        jumps = _jump_candidates(obj, jump_ranks=max(jump_ranks, len(obj) - 1))
        if not jumps:
            continue
        all_jump_rows.extend([jump.__dict__ for jump in jumps])
        year = jumps[0].date[:4]
        filter_path = filter_root / f"global_phy_{year}_bandpass_30_180d.nc"
        if not filter_path.exists():
            continue
        try:
            column = _read_velocity_column(filter_path, jumps[0].date, jumps[0].surface_lon, jumps[0].surface_lat, half_width_deg)
            raw_column = None
            if vertical_velocity_method == "omega":
                raw_path = Path(raw_root) / f"global_phy_{year}.nc"
                if not raw_path.exists():
                    raise FileNotFoundError(f"Raw file required for omega diagnostics: {raw_path}")
                raw_column = _read_raw_column(raw_path, jumps[0].date, jumps[0].surface_lon, jumps[0].surface_lat, half_width_deg)
            w, dwdz = _vertical_velocity_fields(
                column,
                raw_column,
                vertical_velocity_method,
                jumps[0].surface_lon,
                jumps[0].surface_lat,
            )
            for jump in jumps[:jump_ranks]:
                metrics = _metrics_for_jump(
                    jump,
                    column,
                    w,
                    dwdz,
                    parsed_section_modes,
                    depth_padding_layers,
                    half_width_r,
                    min_half_width_km,
                    vertical_velocity_method,
                )
                metric_rows.append(metrics)
        except Exception as exc:
            for jump in jumps[:jump_ranks]:
                row = dict(jump.__dict__)
                row["vertical_velocity_method"] = vertical_velocity_method
                for prefix in ("parallel", "normal"):
                    row.update(_prefixed_metrics(prefix, {"qualitative_relation": "insufficient_data"}))
                row["qualitative_relation"] = "insufficient_data"
                row["error"] = str(exc)
                metric_rows.append(row)
        if len(metric_rows) >= flush_every_objectdays * max(1, jump_ranks):
            pd.DataFrame(metric_rows).to_parquet(parts_dir / f"top2_part_{part_index:04d}.parquet", index=False)
            pd.DataFrame(all_jump_rows).to_parquet(parts_dir / f"all_jumps_part_{part_index:04d}.parquet", index=False)
            print(
                json.dumps({"processed_objectdays": n, "written_part": part_index, "top_jump_rows": len(metric_rows)}, ensure_ascii=False),
                flush=True,
            )
            metric_rows.clear()
            all_jump_rows.clear()
            part_index += 1
        elif n % 100 == 0:
            print(json.dumps({"processed_objectdays": n, "pending_top_jump_rows": len(metric_rows)}, ensure_ascii=False), flush=True)

    if metric_rows:
        pd.DataFrame(metric_rows).to_parquet(parts_dir / f"top2_part_{part_index:04d}.parquet", index=False)
        pd.DataFrame(all_jump_rows).to_parquet(parts_dir / f"all_jumps_part_{part_index:04d}.parquet", index=False)

    top_parts = sorted(parts_dir.glob("top2_part_*.parquet"))
    all_parts = sorted(parts_dir.glob("all_jumps_part_*.parquet"))
    top = pd.concat([pd.read_parquet(path) for path in top_parts], ignore_index=True) if top_parts else pd.DataFrame()
    all_jumps = pd.concat([pd.read_parquet(path) for path in all_parts], ignore_index=True) if all_parts else pd.DataFrame()
    if top.empty:
        raise ValueError("No jump metrics were produced")
    summary = _write_tables(top, all_jumps, output_dir)
    _plot_outputs(top, summary, output_dir)
    _write_report_clean(top, summary, output_dir)
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "rows": int(len(top)),
                "objectdays": int(top["eddy3d_object_id"].nunique()),
                "section_modes": list(parsed_section_modes),
                "vertical_velocity_method": vertical_velocity_method,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
