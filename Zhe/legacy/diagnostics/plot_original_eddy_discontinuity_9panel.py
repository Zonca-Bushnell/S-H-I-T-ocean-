from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from netCDF4 import Dataset, num2date


RHO0 = 1025.0
OMEGA = 7.2921159e-5
EARTH_RADIUS_M = 6_371_000.0


@dataclass(frozen=True)
class SelectedObject:
    track3d_id: int
    eddy3d_object_id: int
    date: str
    polarity: str
    shape_class: str
    radius_m: float
    n_layers: int
    jump_from_depth_index: int | None
    jump_to_depth_index: int | None
    jump_from_depth_m: float | None
    jump_to_depth_m: float | None
    jump_distance_km: float
    jump_distance_over_R: float
    has_abrupt_jump: bool
    second_jump_from_depth_index: int | None
    second_jump_to_depth_index: int | None
    second_jump_from_depth_m: float | None
    second_jump_to_depth_m: float | None
    second_jump_distance_km: float
    second_jump_distance_over_R: float
    has_second_abrupt_jump: bool


def _coriolis(lat_deg: float) -> float:
    return float(2.0 * OMEGA * np.sin(np.deg2rad(lat_deg)))


def _meters_per_degree(lat_deg: float) -> tuple[float, float]:
    lat = np.deg2rad(lat_deg)
    dy = np.pi * EARTH_RADIUS_M / 180.0
    dx = dy * np.cos(lat)
    return float(dx), float(dy)


def _read_time_index(path: Path, date: str) -> int:
    with Dataset(path) as ds:
        time = ds.variables["time"]
        dates = num2date(time[:], units=time.units, calendar=getattr(time, "calendar", "standard"))
        targets = [getattr(d, "strftime")("%Y-%m-%d") for d in dates]
    try:
        return targets.index(date)
    except ValueError as exc:
        raise ValueError(f"{date} not found in {path}") from exc


def _load_catalog(results_root: Path, shape_dir_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    centers = pd.read_parquet(results_root / "catalog" / "layer_centers_completed.parquet")
    shape = pd.read_parquet(results_root / shape_dir_name / "shape_tracks.parquet")
    return centers, shape


def _candidate_objects(
    centers: pd.DataFrame,
    shape: pd.DataFrame,
    preferred_shapes: set[str],
    min_layers: int,
    year_limit: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    shape = shape[shape["shape_class"].astype(str).isin(preferred_shapes)].copy()
    if shape.empty:
        raise ValueError(f"No shape tracks found for {sorted(preferred_shapes)}")
    allowed = centers.merge(shape[["track3d_id", "shape_class"]], on="track3d_id", how="inner")
    allowed["date"] = allowed["date"].astype(str)
    if year_limit is not None:
        allowed = allowed[allowed["date"].str.slice(0, 4).astype(int).le(year_limit)].copy()
    if allowed.empty:
        raise ValueError("No layer centers after shape/year filtering")

    rows: list[dict] = []
    for object_id, part in allowed.groupby("eddy3d_object_id", sort=False):
        part = part.sort_values("depth_index")
        if len(part) < min_layers:
            continue
        lon0 = float(part.iloc[0]["longitude"])
        lat0 = float(part.iloc[0]["latitude"])
        radius_m = float(np.nanmedian(part["radius_m"].to_numpy(dtype="f8")))
        dx_m, dy_m = _meters_per_degree(lat0)
        x = (part["longitude"].to_numpy(dtype="f8") - lon0) * dx_m
        y = (part["latitude"].to_numpy(dtype="f8") - lat0) * dy_m
        jumps = np.hypot(np.diff(x), np.diff(y))
        ranked_jumps = np.argsort(jumps)[::-1] if jumps.size else np.array([], dtype=int)

        def jump_record(rank: int) -> tuple[float, float, int, int, float, float]:
            if ranked_jumps.size <= rank:
                return np.nan, np.nan, -1, -1, np.nan, np.nan
            idx = int(ranked_jumps[rank])
            jump_km_value = float(jumps[idx] / 1000.0)
            jump_r_value = float(jumps[idx] / radius_m) if radius_m > 0 else np.nan
            from_k = int(part.iloc[idx]["depth_index"])
            to_k = int(part.iloc[idx + 1]["depth_index"])
            from_z = float(part.iloc[idx]["depth_m"])
            to_z = float(part.iloc[idx + 1]["depth_m"])
            return jump_km_value, jump_r_value, from_k, to_k, from_z, to_z

        jump_km, jump_r, k0, k1, z0, z1 = jump_record(0)
        jump2_km, jump2_r, k20, k21, z20, z21 = jump_record(1)
        rows.append(
            {
                "eddy3d_object_id": int(object_id),
                "track3d_id": int(part.iloc[0]["track3d_id"]),
                "date": str(part.iloc[0]["date"]),
                "polarity": str(part.iloc[0]["polarity"]),
                "shape_class": str(part.iloc[0]["shape_class"]),
                "radius_m": radius_m,
                "n_layers": int(len(part)),
                "jump_from_depth_index": k0,
                "jump_to_depth_index": k1,
                "jump_from_depth_m": z0,
                "jump_to_depth_m": z1,
                "jump_distance_km": jump_km,
                "jump_distance_over_R": jump_r,
                "second_jump_from_depth_index": k20,
                "second_jump_to_depth_index": k21,
                "second_jump_from_depth_m": z20,
                "second_jump_to_depth_m": z21,
                "second_jump_distance_km": jump2_km,
                "second_jump_distance_over_R": jump2_r,
            }
        )
    candidates = pd.DataFrame(rows)
    if candidates.empty:
        raise ValueError("No candidate object-days with enough layers")
    candidates = candidates.sort_values(["jump_distance_over_R", "n_layers"], ascending=[False, False])
    return allowed, candidates


def _selected_from_row(row: pd.Series, abrupt_threshold_over_R: float) -> SelectedObject:
    return SelectedObject(
        track3d_id=int(row["track3d_id"]),
        eddy3d_object_id=int(row["eddy3d_object_id"]),
        date=str(row["date"]),
        polarity=str(row["polarity"]),
        shape_class=str(row["shape_class"]),
        radius_m=float(row["radius_m"]),
        n_layers=int(row["n_layers"]),
        jump_from_depth_index=int(row["jump_from_depth_index"]) if int(row["jump_from_depth_index"]) >= 0 else None,
        jump_to_depth_index=int(row["jump_to_depth_index"]) if int(row["jump_to_depth_index"]) >= 0 else None,
        jump_from_depth_m=float(row["jump_from_depth_m"]) if np.isfinite(float(row["jump_from_depth_m"])) else None,
        jump_to_depth_m=float(row["jump_to_depth_m"]) if np.isfinite(float(row["jump_to_depth_m"])) else None,
        jump_distance_km=float(row["jump_distance_km"]),
        jump_distance_over_R=float(row["jump_distance_over_R"]),
        has_abrupt_jump=bool(float(row["jump_distance_over_R"]) >= abrupt_threshold_over_R),
        second_jump_from_depth_index=int(row["second_jump_from_depth_index"]) if int(row["second_jump_from_depth_index"]) >= 0 else None,
        second_jump_to_depth_index=int(row["second_jump_to_depth_index"]) if int(row["second_jump_to_depth_index"]) >= 0 else None,
        second_jump_from_depth_m=float(row["second_jump_from_depth_m"]) if np.isfinite(float(row["second_jump_from_depth_m"])) else None,
        second_jump_to_depth_m=float(row["second_jump_to_depth_m"]) if np.isfinite(float(row["second_jump_to_depth_m"])) else None,
        second_jump_distance_km=float(row["second_jump_distance_km"]) if np.isfinite(float(row["second_jump_distance_km"])) else np.nan,
        second_jump_distance_over_R=float(row["second_jump_distance_over_R"]) if np.isfinite(float(row["second_jump_distance_over_R"])) else np.nan,
        has_second_abrupt_jump=bool(np.isfinite(float(row["second_jump_distance_over_R"])) and float(row["second_jump_distance_over_R"]) >= abrupt_threshold_over_R),
    )


def _choose_object(
    centers: pd.DataFrame,
    shape: pd.DataFrame,
    preferred_shapes: set[str],
    min_layers: int,
    abrupt_threshold_over_R: float,
    year_limit: int | None,
) -> tuple[SelectedObject, pd.DataFrame, pd.DataFrame]:
    allowed, candidates = _candidate_objects(centers, shape, preferred_shapes, min_layers, year_limit)
    selected = _selected_from_row(candidates.iloc[0], abrupt_threshold_over_R)
    obj = allowed[allowed["eddy3d_object_id"].astype(int).eq(selected.eddy3d_object_id)].sort_values("depth_index").copy()
    track = centers[centers["track3d_id"].astype(int).eq(selected.track3d_id)].copy()
    return selected, obj, track


def _choose_objects(
    centers: pd.DataFrame,
    shape: pd.DataFrame,
    preferred_shapes: set[str],
    min_layers: int,
    abrupt_threshold_over_R: float,
    year_limit: int | None,
    max_examples: int,
) -> list[tuple[SelectedObject, pd.DataFrame, pd.DataFrame]]:
    allowed, candidates = _candidate_objects(centers, shape, preferred_shapes, min_layers, year_limit)
    chosen: list[tuple[SelectedObject, pd.DataFrame, pd.DataFrame]] = []
    for _, row in candidates.head(max_examples).iterrows():
        selected = _selected_from_row(row, abrupt_threshold_over_R)
        obj = allowed[allowed["eddy3d_object_id"].astype(int).eq(selected.eddy3d_object_id)].sort_values("depth_index").copy()
        track = centers[centers["track3d_id"].astype(int).eq(selected.track3d_id)].copy()
        chosen.append((selected, obj, track))
    return chosen


def _legacy_selected_from_row(best: pd.Series, abrupt_threshold_over_R: float) -> SelectedObject:
    return SelectedObject(
        track3d_id=int(best["track3d_id"]),
        eddy3d_object_id=int(best["eddy3d_object_id"]),
        date=str(best["date"]),
        polarity=str(best["polarity"]),
        shape_class=str(best["shape_class"]),
        radius_m=float(best["radius_m"]),
        n_layers=int(best["n_layers"]),
        jump_from_depth_index=int(best["jump_from_depth_index"]) if int(best["jump_from_depth_index"]) >= 0 else None,
        jump_to_depth_index=int(best["jump_to_depth_index"]) if int(best["jump_to_depth_index"]) >= 0 else None,
        jump_from_depth_m=float(best["jump_from_depth_m"]) if np.isfinite(float(best["jump_from_depth_m"])) else None,
        jump_to_depth_m=float(best["jump_to_depth_m"]) if np.isfinite(float(best["jump_to_depth_m"])) else None,
        jump_distance_km=float(best["jump_distance_km"]),
        jump_distance_over_R=float(best["jump_distance_over_R"]),
        has_abrupt_jump=bool(float(best["jump_distance_over_R"]) >= abrupt_threshold_over_R),
        second_jump_from_depth_index=int(best["second_jump_from_depth_index"]) if int(best["second_jump_from_depth_index"]) >= 0 else None,
        second_jump_to_depth_index=int(best["second_jump_to_depth_index"]) if int(best["second_jump_to_depth_index"]) >= 0 else None,
        second_jump_from_depth_m=float(best["second_jump_from_depth_m"]) if np.isfinite(float(best["second_jump_from_depth_m"])) else None,
        second_jump_to_depth_m=float(best["second_jump_to_depth_m"]) if np.isfinite(float(best["second_jump_to_depth_m"])) else None,
        second_jump_distance_km=float(best["second_jump_distance_km"]) if np.isfinite(float(best["second_jump_distance_km"])) else np.nan,
        second_jump_distance_over_R=float(best["second_jump_distance_over_R"]) if np.isfinite(float(best["second_jump_distance_over_R"])) else np.nan,
        has_second_abrupt_jump=bool(np.isfinite(float(best["second_jump_distance_over_R"])) and float(best["second_jump_distance_over_R"]) >= abrupt_threshold_over_R),
    )


def _window_indices(values: np.ndarray, center: float, half_width: float) -> np.ndarray:
    idx = np.where((values >= center - half_width) & (values <= center + half_width))[0]
    if idx.size < 5:
        nearest = int(np.nanargmin(np.abs(values - center)))
        lo = max(0, nearest - 4)
        hi = min(values.size, nearest + 5)
        idx = np.arange(lo, hi)
    return idx


def _read_field_window(
    path: Path,
    date: str,
    center_lon: float,
    center_lat: float,
    depth_index: int,
    half_width_deg: float,
    variables: tuple[str, ...],
) -> dict[str, np.ndarray]:
    with Dataset(path) as ds:
        t = _read_time_index(path, date)
        lon = np.asarray(ds.variables["longitude"][:], dtype="f8")
        lat = np.asarray(ds.variables["latitude"][:], dtype="f8")
        depth = np.asarray(ds.variables["depth"][:], dtype="f8") if "depth" in ds.variables else None
        ix = _window_indices(lon, center_lon, half_width_deg)
        iy = _window_indices(lat, center_lat, half_width_deg)
        out = {"longitude": lon[ix], "latitude": lat[iy]}
        if depth is not None:
            out["depth"] = depth
        for name in variables:
            if name not in ds.variables:
                continue
            var = ds.variables[name]
            if var.ndim == 4:
                out[name] = np.asarray(var[t, depth_index, iy.min() : iy.max() + 1, ix.min() : ix.max() + 1], dtype="f8")
            elif var.ndim == 3:
                out[name] = np.asarray(var[t, iy.min() : iy.max() + 1, ix.min() : ix.max() + 1], dtype="f8")
    return out


def _read_velocity_column(
    path: Path,
    date: str,
    center_lon: float,
    center_lat: float,
    half_width_deg: float,
) -> dict[str, np.ndarray]:
    with Dataset(path) as ds:
        t = _read_time_index(path, date)
        lon = np.asarray(ds.variables["longitude"][:], dtype="f8")
        lat = np.asarray(ds.variables["latitude"][:], dtype="f8")
        depth = np.asarray(ds.variables["depth"][:], dtype="f8")
        ix = _window_indices(lon, center_lon, half_width_deg)
        iy = _window_indices(lat, center_lat, half_width_deg)
        u = np.asarray(ds.variables["uo_glor"][t, :, iy.min() : iy.max() + 1, ix.min() : ix.max() + 1], dtype="f8")
        v = np.asarray(ds.variables["vo_glor"][t, :, iy.min() : iy.max() + 1, ix.min() : ix.max() + 1], dtype="f8")
    return {"longitude": lon[ix], "latitude": lat[iy], "depth": depth, "uo_glor": u, "vo_glor": v}


def _relative_xy(lon: np.ndarray, lat: np.ndarray, lon0: float, lat0: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dx_m, dy_m = _meters_per_degree(lat0)
    x = (lon - lon0) * dx_m
    y = (lat - lat0) * dy_m
    xx, yy = np.meshgrid(x / 1000.0, y / 1000.0)
    return x, y, xx, yy


def _streamfunction_from_zeta_2d(zeta: np.ndarray, x_m: np.ndarray, y_m: np.ndarray) -> np.ndarray:
    from scipy.fft import dstn, idstn

    rhs = np.where(np.isfinite(zeta), zeta, 0.0)
    ny, nx = rhs.shape
    if ny < 3 or nx < 3:
        return np.zeros_like(rhs)
    dx = float(np.nanmedian(np.diff(x_m)))
    dy = float(np.nanmedian(np.diff(y_m)))
    inner = rhs[1:-1, 1:-1]
    hat = dstn(inner, type=1, axes=(0, 1), norm="ortho")
    mx = np.arange(1, inner.shape[1] + 1, dtype="f8")
    my = np.arange(1, inner.shape[0] + 1, dtype="f8")
    lam_x = -4.0 * np.sin(np.pi * mx / (2.0 * (inner.shape[1] + 1))) ** 2 / max(dx * dx, 1e-12)
    lam_y = -4.0 * np.sin(np.pi * my / (2.0 * (inner.shape[0] + 1))) ** 2 / max(dy * dy, 1e-12)
    denom = lam_y[:, None] + lam_x[None, :]
    psi_inner = idstn(np.divide(hat, denom, out=np.zeros_like(hat), where=np.abs(denom) > 1e-12), type=1, axes=(0, 1), norm="ortho")
    psi = np.zeros_like(rhs)
    psi[1:-1, 1:-1] = psi_inner
    return psi


def _pressure_proxy(u: np.ndarray, v: np.ndarray, x_m: np.ndarray, y_m: np.ndarray, f0: float) -> np.ndarray:
    dvdx = np.gradient(v, x_m, axis=1, edge_order=1)
    dudy = np.gradient(u, y_m, axis=0, edge_order=1)
    psi = _streamfunction_from_zeta_2d(dvdx - dudy, x_m, y_m)
    return RHO0 * f0 * psi


def _vertical_velocity_proxy(u3: np.ndarray, v3: np.ndarray, depth: np.ndarray, x_m: np.ndarray, y_m: np.ndarray) -> np.ndarray:
    dudx = np.gradient(u3, x_m, axis=2, edge_order=1)
    dvdy = np.gradient(v3, y_m, axis=1, edge_order=1)
    div = dudx + dvdy
    w = np.zeros_like(div)
    for k in range(1, len(depth)):
        dz = float(depth[k] - depth[k - 1])
        w[k] = w[k - 1] - 0.5 * (div[k] + div[k - 1]) * dz
    return w


def _vertical_gradient_w(w: np.ndarray, depth: np.ndarray) -> np.ndarray:
    if len(depth) < 2:
        return np.zeros_like(w)
    return np.gradient(w, depth, axis=0, edge_order=1)


def _interp_section(field: np.ndarray, x_m: np.ndarray, y_m: np.ndarray, x_line_km: np.ndarray, y_line_km: np.ndarray) -> np.ndarray:
    from scipy.interpolate import RegularGridInterpolator

    points = np.column_stack([y_line_km * 1000.0, x_line_km * 1000.0])
    section = np.full((field.shape[0], len(x_line_km)), np.nan, dtype="f8")
    for k in range(field.shape[0]):
        interp = RegularGridInterpolator((y_m, x_m), field[k], bounds_error=False, fill_value=np.nan)
        section[k] = interp(points)
    return section


def _sigma0(theta: np.ndarray, salinity: np.ndarray) -> tuple[np.ndarray, str]:
    try:
        import gsw

        sigma = gsw.sigma0(salinity, theta)
        return np.asarray(sigma, dtype="f8"), "sigma0 from gsw(thetao, so)"
    except Exception:
        sigma = 1027.0 - 0.2 * (theta - np.nanmean(theta)) + 0.78 * (salinity - np.nanmean(salinity))
        return np.asarray(sigma, dtype="f8"), "linear density proxy; gsw unavailable"


def _format_shape_list(text: str) -> list[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


def _nearest_layer_index(depth: np.ndarray, depth_index: int) -> int:
    if depth_index < 0:
        return 0
    if depth_index >= len(depth):
        return len(depth) - 1
    return int(depth_index)


def _finite_limits(values: np.ndarray, quantile: float = 0.98) -> tuple[float, float]:
    finite = np.asarray(values[np.isfinite(values)], dtype="f8")
    if finite.size == 0:
        return 0.0, 1.0
    vmax = float(np.nanquantile(np.abs(finite), quantile))
    if vmax <= 0 or not np.isfinite(vmax):
        vmax = float(np.nanmax(np.abs(finite))) if finite.size else 1.0
    if vmax <= 0 or not np.isfinite(vmax):
        vmax = 1.0
    return -vmax, vmax


def _mark_center(ax, x_km: float, y_km: float, label: str, color: str, marker: str = "x") -> None:
    ax.scatter([x_km], [y_km], s=90, marker=marker, c=color, linewidths=2.0, label=label, zorder=8)


def _hatch_unavailable(ax, text: str) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(0.5, 0.5, text, ha="center", va="center", transform=ax.transAxes, fontsize=10)
    for offset in np.linspace(-0.8, 1.4, 12):
        ax.plot([offset, offset + 0.8], [0.0, 1.0], color="0.55", lw=1.2, transform=ax.transAxes, clip_on=True)


def _plot_field(
    ax,
    xx: np.ndarray,
    yy: np.ndarray,
    field: np.ndarray,
    title: str,
    cmap: str,
    *,
    symmetric: bool = False,
    quiver: tuple[np.ndarray, np.ndarray] | None = None,
    center_marks: list[tuple[float, float, str, str, str]] | None = None,
):
    if symmetric:
        vmin, vmax = _finite_limits(field)
    else:
        finite = field[np.isfinite(field)]
        vmin = float(np.nanquantile(finite, 0.02)) if finite.size else 0.0
        vmax = float(np.nanquantile(finite, 0.98)) if finite.size else 1.0
    mesh = ax.pcolormesh(xx, yy, field, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    if quiver is not None:
        u, v = quiver
        step = max(1, int(max(u.shape) / 18))
        ax.quiver(xx[::step, ::step], yy[::step, ::step], u[::step, ::step], v[::step, ::step], color="white", alpha=0.65, scale=2.5)
    if center_marks:
        for x_km, y_km, label, color, marker in center_marks:
            _mark_center(ax, x_km, y_km, label, color, marker)
    ax.axhline(0, color="0.85", lw=0.8)
    ax.axvline(0, color="0.85", lw=0.8)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("east from surface center (km)")
    ax.set_ylabel("north from surface center (km)")
    return mesh


def _object_offsets_km(object_layers: pd.DataFrame) -> pd.DataFrame:
    obj = object_layers.sort_values("depth_index").copy()
    surface = obj.iloc[0]
    dx_m, dy_m = _meters_per_degree(float(surface["latitude"]))
    obj["delta_x_km"] = (obj["longitude"] - float(surface["longitude"])) * dx_m / 1000.0
    obj["delta_y_km"] = (obj["latitude"] - float(surface["latitude"])) * dy_m / 1000.0
    return obj


def _build_center_marks(
    object_layers: pd.DataFrame,
    surface_lon: float,
    surface_lat: float,
    jump_from: int | None,
    jump_to: int | None,
) -> list[tuple[float, float, str, str, str]]:
    dx_m, dy_m = _meters_per_degree(surface_lat)

    def xy(row: pd.Series) -> tuple[float, float]:
        return (
            (float(row["longitude"]) - surface_lon) * dx_m / 1000.0,
            (float(row["latitude"]) - surface_lat) * dy_m / 1000.0,
        )

    marks: list[tuple[float, float, str, str, str]] = [(0.0, 0.0, "surface", "red", "+")]
    if jump_from is not None:
        prev = object_layers[object_layers["depth_index"] == jump_from]
        if not prev.empty:
            x, y = xy(prev.iloc[0])
            marks.append((x, y, "upper side", "cyan", "x"))
    if jump_to is not None:
        cur = object_layers[object_layers["depth_index"] == jump_to]
        if not cur.empty:
            x, y = xy(cur.iloc[0])
            marks.append((x, y, "lower side", "yellow", "x"))
    return marks


def _plot_vertical_w_section(
    ax,
    section: dict[str, np.ndarray],
    title: str,
    selected: SelectedObject,
    *,
    second: bool = False,
):
    s = section["section_coord_km"]
    depth = section["depth"]
    w = section["w_section"]
    dwdz = section["dwdz_section"]
    xlim = section["xlim_km"]
    zlim = section["zlim_m"]
    xmask = (s >= xlim[0]) & (s <= xlim[1])
    zmask = (depth >= zlim[0]) & (depth <= zlim[1])
    local = dwdz[np.ix_(zmask, xmask)] if np.any(xmask) and np.any(zmask) else dwdz
    vmin, vmax = _finite_limits(local)
    mesh = ax.pcolormesh(s, depth, dwdz, shading="auto", cmap="RdBu_r", vmin=vmin, vmax=vmax)
    w_local = w[np.ix_(zmask, xmask)] if np.any(xmask) and np.any(zmask) else w
    finite_w = w_local[np.isfinite(w_local)]
    if finite_w.size:
        levels = np.linspace(float(np.nanquantile(finite_w, 0.1)), float(np.nanquantile(finite_w, 0.9)), 7)
        levels = levels[np.isfinite(levels)]
        if np.unique(levels).size >= 3:
            ax.contour(s, depth, w, levels=np.unique(levels), colors="0.2", linewidths=0.55, alpha=0.65)
    ax.invert_yaxis()
    ax.axvline(0, color="0.75", lw=0.8)
    from_z = selected.second_jump_from_depth_m if second else selected.jump_from_depth_m
    to_z = selected.second_jump_to_depth_m if second else selected.jump_to_depth_m
    if from_z is not None:
        ax.axhline(from_z, color="tab:red", ls="--", lw=1.0, alpha=0.8)
    if to_z is not None:
        ax.axhline(to_z, color="tab:red", ls=":", lw=1.0, alpha=0.8)
    center_s = section.get("center_section_coord_km")
    center_z = section.get("center_depth_m")
    if center_s is not None and center_z is not None:
        ax.plot(center_s, center_z, "k.-", ms=4, lw=1.0, alpha=0.75, label="layer centers")
        ax.legend(loc="best", fontsize=7)
    axis_name = str(section.get("section_axis", "section"))
    ax.set_xlim(float(xlim[0]), float(xlim[1]))
    ax.set_ylim(float(zlim[1]), float(zlim[0]))
    ax.set_title(f"{title}: vertical shear of w proxy ({axis_name})", fontsize=10)
    ax.set_xlabel("section distance from surface center (km)")
    ax.set_ylabel("depth (m)")
    ax.grid(alpha=0.2)
    return mesh


def _plot_9panel(
    selected: SelectedObject,
    object_layers: pd.DataFrame,
    track_layers: pd.DataFrame,
    fields: dict[str, dict[str, np.ndarray] | None],
    output_dir: Path,
    output_name_stem: str = "original_eddy_discontinuity_9panel",
) -> None:
    offsets = _object_offsets_km(object_layers)
    surface = offsets.iloc[0]
    first_fields = fields.get("first")
    second_fields = fields.get("second")
    has_first = bool(selected.has_abrupt_jump and first_fields is not None)
    has_second = bool(selected.has_second_abrupt_jump and second_fields is not None)

    fig = plt.figure(figsize=(20, 11), constrained_layout=True)
    gs = fig.add_gridspec(3, 5, height_ratios=[1.0, 1.0, 0.9], width_ratios=[0.9, 0.9, 1.05, 1.05, 1.05])
    ax1 = fig.add_subplot(gs[0:2, 0])
    ax2 = fig.add_subplot(gs[0:2, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[0, 3])
    ax5 = fig.add_subplot(gs[1, 2])
    ax6 = fig.add_subplot(gs[1, 3])
    ax8 = fig.add_subplot(gs[0, 4])
    ax9 = fig.add_subplot(gs[1, 4])
    ax7 = fig.add_subplot(gs[2, :])

    for ax, col, title in [(ax1, "delta_x_km", "1  delta x from surface center"), (ax2, "delta_y_km", "2  delta y from surface center")]:
        ax.plot(offsets[col], offsets["depth_m"], "-o", color="#244a9b", lw=1.8, ms=4)
        ax.axvline(0, color="0.75", lw=0.8)
        if selected.jump_from_depth_m is not None:
            ax.axhline(selected.jump_from_depth_m, color="tab:red", ls="--", lw=1.0, alpha=0.8)
        if selected.jump_to_depth_m is not None:
            ax.axhline(selected.jump_to_depth_m, color="tab:red", ls=":", lw=1.0, alpha=0.8)
        ax.invert_yaxis()
        ax.set_xlabel("offset (km)")
        ax.set_ylabel("depth (m)")
        ax.set_title(title)
        ax.grid(alpha=0.25)

    if has_first and first_fields is not None:
        xx = first_fields["xx"]
        yy = first_fields["yy"]
        marks = first_fields["marks"]
        first_title = f"first jump {selected.jump_from_depth_index}->{selected.jump_to_depth_index}"
        m3 = _plot_field(ax3, xx, yy, first_fields["speed"], f"3  {first_title}: speed |u',v'|", "magma", quiver=(first_fields["u"], first_fields["v"]), center_marks=marks)
        fig.colorbar(m3, ax=ax3, shrink=0.82, label="m/s")
        m4 = _plot_field(ax4, xx, yy, first_fields["pressure"], f"4  {first_title}: geostrophic p' proxy", "RdBu_r", symmetric=True, center_marks=marks)
        fig.colorbar(m4, ax=ax4, shrink=0.82, label="Pa proxy")
        if "w_section" in first_fields:
            m8 = _plot_vertical_w_section(ax8, first_fields["w_section"], f"8  {first_title}", selected, second=False)
            fig.colorbar(m8, ax=ax8, shrink=0.82, label="m/s proxy")
        else:
            _hatch_unavailable(ax8, "first jump vertical w section unavailable")
        for ax in [ax3, ax4]:
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(loc="upper right", fontsize=8)
    else:
        for ax in [ax3, ax4, ax8]:
            _hatch_unavailable(ax, "no first abrupt layer discontinuity detected")

    if has_second and second_fields is not None:
        xx = second_fields["xx"]
        yy = second_fields["yy"]
        marks = second_fields["marks"]
        second_title = f"second jump {selected.second_jump_from_depth_index}->{selected.second_jump_to_depth_index}"
        m5 = _plot_field(ax5, xx, yy, second_fields["speed"], f"5  {second_title}: speed |u',v'|", "magma", quiver=(second_fields["u"], second_fields["v"]), center_marks=marks)
        fig.colorbar(m5, ax=ax5, shrink=0.82, label="m/s")
        m6 = _plot_field(ax6, xx, yy, second_fields["pressure"], f"6  {second_title}: geostrophic p' proxy", "RdBu_r", symmetric=True, center_marks=marks)
        fig.colorbar(m6, ax=ax6, shrink=0.82, label="Pa proxy")
        if "w_section" in second_fields:
            m9 = _plot_vertical_w_section(ax9, second_fields["w_section"], f"9  {second_title}", selected, second=True)
            fig.colorbar(m9, ax=ax9, shrink=0.82, label="m/s proxy")
        else:
            _hatch_unavailable(ax9, "second jump vertical w section unavailable")
        for ax in [ax5, ax6]:
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(loc="upper right", fontsize=8)
    else:
        for ax in [ax5, ax6, ax9]:
            _hatch_unavailable(ax, "no second abrupt layer discontinuity detected")

    surface_track = track_layers[track_layers["depth_index"].astype(int).eq(0)].sort_values("date")
    if surface_track.empty:
        surface_track = track_layers.sort_values(["date", "depth_index"]).groupby("date", as_index=False).first()
    ax7.plot(surface_track["longitude"], surface_track["latitude"], "-", color="0.65", lw=1.0, label="surface-center track")
    ax7.scatter(surface_track["longitude"], surface_track["latitude"], c=np.arange(len(surface_track)), s=22, cmap="viridis", label="lifecycle")
    ax7.scatter([surface["longitude"]], [surface["latitude"]], s=130, c="red", marker="*", label="selected object-day", zorder=9)
    ax7.set_title("7  full lifecycle trajectory of selected original eddy")
    ax7.set_xlabel("longitude")
    ax7.set_ylabel("latitude")
    ax7.grid(alpha=0.25)
    ax7.legend(loc="best", fontsize=9)

    jump_text = "no abrupt jump"
    if selected.has_abrupt_jump:
        jump_text = (
            f"jump {selected.jump_from_depth_index}->{selected.jump_to_depth_index}, "
            f"{selected.jump_distance_km:.1f} km = {selected.jump_distance_over_R:.2f} R"
        )
    if selected.has_second_abrupt_jump:
        jump_text += (
            f"; second {selected.second_jump_from_depth_index}->{selected.second_jump_to_depth_index}, "
            f"{selected.second_jump_distance_km:.1f} km = {selected.second_jump_distance_over_R:.2f} R"
        )
    fig.suptitle(
        "Original eddy discontinuity diagnostic, not representative vortex\n"
        f"object {selected.eddy3d_object_id}, track {selected.track3d_id}, {selected.date}, "
        f"{selected.shape_class}/{selected.polarity}; {jump_text}",
        fontsize=15,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{output_name_stem}.png", dpi=220)
    fig.savefig(output_dir / f"{output_name_stem}.pdf")
    plt.close(fig)


def _plot_7panel(
    selected: SelectedObject,
    object_layers: pd.DataFrame,
    track_layers: pd.DataFrame,
    fields: dict[str, dict[str, np.ndarray] | None],
    output_dir: Path,
) -> None:
    _plot_9panel(selected, object_layers, track_layers, fields, output_dir)


def _make_jump_cross_section_fields(
    *,
    selected: SelectedObject,
    object_layers: pd.DataFrame,
    filter_root: Path,
    half_width_deg: float,
    jump_from_depth_index: int | None,
    jump_to_depth_index: int | None,
    w_shear_depth_padding_layers: int,
    w_shear_half_width_r: float,
    w_shear_min_half_width_km: float,
    w_section_mode: str,
) -> dict[str, np.ndarray] | None:
    if jump_to_depth_index is None:
        return None
    year = str(selected.date)[:4]
    filter_path = filter_root / f"global_phy_{year}_bandpass_30_180d.nc"
    if not filter_path.exists():
        return None

    offsets = _object_offsets_km(object_layers)
    surface = offsets.iloc[0]
    surface_lon = float(surface["longitude"])
    surface_lat = float(surface["latitude"])
    layer_index = int(jump_to_depth_index)

    vel2 = _read_field_window(
        path=filter_path,
        date=selected.date,
        center_lon=surface_lon,
        center_lat=surface_lat,
        depth_index=layer_index,
        half_width_deg=half_width_deg,
        variables=("uo_glor", "vo_glor"),
    )

    lon = vel2["longitude"]
    lat = vel2["latitude"]
    x_m, y_m, xx, yy = _relative_xy(lon, lat, surface_lon, surface_lat)
    u = vel2["uo_glor"]
    v = vel2["vo_glor"]
    speed = np.hypot(u, v)
    pressure = _pressure_proxy(u, v, x_m, y_m, _coriolis(surface_lat))
    marks = _build_center_marks(
        offsets,
        surface_lon,
        surface_lat,
        jump_from_depth_index,
        jump_to_depth_index,
    )
    column = _read_velocity_column(
        path=filter_path,
        date=selected.date,
        center_lon=surface_lon,
        center_lat=surface_lat,
        half_width_deg=half_width_deg,
    )
    w_section = _make_vertical_w_section(
        object_layers=offsets,
        column=column,
        surface_lon=surface_lon,
        surface_lat=surface_lat,
        jump_from_depth_index=jump_from_depth_index,
        jump_to_depth_index=jump_to_depth_index,
        radius_m=selected.radius_m,
        depth_padding_layers=w_shear_depth_padding_layers,
        half_width_r=w_shear_half_width_r,
        min_half_width_km=w_shear_min_half_width_km,
        section_mode=w_section_mode,
    )

    return {
        "xx": xx,
        "yy": yy,
        "u": u,
        "v": v,
        "speed": speed,
        "pressure": pressure,
        "marks": marks,
        "w_section": w_section,
    }


def _make_vertical_w_section(
    *,
    object_layers: pd.DataFrame,
    column: dict[str, np.ndarray],
    surface_lon: float,
    surface_lat: float,
    jump_from_depth_index: int | None,
    jump_to_depth_index: int | None,
    radius_m: float,
    depth_padding_layers: int,
    half_width_r: float,
    min_half_width_km: float,
    section_mode: str,
) -> dict[str, np.ndarray]:
    lon = column["longitude"]
    lat = column["latitude"]
    depth = column["depth"]
    x_m, y_m, _, _ = _relative_xy(lon, lat, surface_lon, surface_lat)
    w = _vertical_velocity_proxy(column["uo_glor"], column["vo_glor"], depth, x_m, y_m)
    dwdz = _vertical_gradient_w(w, depth)

    centers = object_layers.sort_values("depth_index").copy()
    center_x = centers["delta_x_km"].to_numpy(dtype="f8")
    center_y = centers["delta_y_km"].to_numpy(dtype="f8")
    center_z = centers["depth_m"].to_numpy(dtype="f8")

    def layer_xy(depth_index: int | None) -> tuple[float, float] | None:
        if depth_index is None:
            return None
        row = centers[centers["depth_index"].astype(int).eq(int(depth_index))]
        if row.empty:
            return None
        return float(row.iloc[0]["delta_x_km"]), float(row.iloc[0]["delta_y_km"])

    p0 = layer_xy(jump_from_depth_index)
    p1 = layer_xy(jump_to_depth_index)
    if p0 is not None and p1 is not None:
        dx = p1[0] - p0[0]
        dy = p1[1] - p0[1]
        target_x, target_y = p1
    elif p1 is not None:
        dx = p1[0]
        dy = p1[1]
        target_x, target_y = p1
    else:
        dx = float(np.nanmax(np.abs(center_x))) if center_x.size else 1.0
        dy = float(np.nanmax(np.abs(center_y))) if center_y.size else 0.0
        target_x = float(np.nanmedian(center_x)) if center_x.size else 0.0
        target_y = float(np.nanmedian(center_y)) if center_y.size else 0.0

    if section_mode == "normal":
        norm = float(np.hypot(dx, dy))
        if not np.isfinite(norm) or norm <= 1e-9:
            ex, ey = 1.0, 0.0
        else:
            ex, ey = -dy / norm, dx / norm
        mid_x = 0.5 * (p0[0] + p1[0]) if p0 is not None and p1 is not None else target_x
        mid_y = 0.5 * (p0[1] + p1[1]) if p0 is not None and p1 is not None else target_y
        max_half = max(float(radius_m) / 1000.0 * 2.5, 150.0)
        section_coord = np.linspace(-max_half, max_half, 161, dtype="f8")
        x_line = mid_x + section_coord * ex
        y_line = mid_y + section_coord * ey
        section = _interp_section(w, x_m, y_m, x_line, y_line)
        dwdz_section = _interp_section(dwdz, x_m, y_m, x_line, y_line)
        center_coord = (center_x - mid_x) * ex + (center_y - mid_y) * ey
        center_target = 0.0
        axis = "jump-normal through center-pair midpoint"
    elif abs(dx) >= abs(dy):
        iy = int(np.nanargmin(np.abs(y_m / 1000.0 - target_y)))
        section = w[:, iy, :]
        dwdz_section = dwdz[:, iy, :]
        section_coord = x_m / 1000.0
        center_coord = center_x
        center_target = target_x
        axis = f"x-z at y={y_m[iy] / 1000.0:.1f} km"
    else:
        ix = int(np.nanargmin(np.abs(x_m / 1000.0 - target_x)))
        section = w[:, :, ix]
        dwdz_section = dwdz[:, :, ix]
        section_coord = y_m / 1000.0
        center_coord = center_y
        axis = f"y-z at x={x_m[ix] / 1000.0:.1f} km"
        center_target = target_y

    if jump_from_depth_index is not None and jump_to_depth_index is not None:
        k_min = max(0, min(jump_from_depth_index, jump_to_depth_index) - max(0, depth_padding_layers))
        k_max = min(len(depth) - 1, max(jump_from_depth_index, jump_to_depth_index) + max(0, depth_padding_layers))
    elif jump_to_depth_index is not None:
        k_min = max(0, jump_to_depth_index - max(0, depth_padding_layers))
        k_max = min(len(depth) - 1, jump_to_depth_index + max(0, depth_padding_layers))
    else:
        k_min = 0
        k_max = min(len(depth) - 1, max(1, 2 * max(0, depth_padding_layers)))
    x_half = max(float(radius_m) / 1000.0 * float(half_width_r), float(min_half_width_km))
    xlim = np.array([center_target - x_half, center_target + x_half], dtype="f8")
    coord_min = float(np.nanmin(section_coord))
    coord_max = float(np.nanmax(section_coord))
    xlim[0] = max(xlim[0], coord_min)
    xlim[1] = min(xlim[1], coord_max)
    zlim = np.array([float(depth[k_min]), float(depth[k_max])], dtype="f8")

    return {
        "section_coord_km": np.asarray(section_coord, dtype="f8"),
        "depth": np.asarray(depth, dtype="f8"),
        "w_section": np.asarray(section, dtype="f8"),
        "dwdz_section": np.asarray(dwdz_section, dtype="f8"),
        "center_section_coord_km": np.asarray(center_coord, dtype="f8"),
        "center_depth_m": np.asarray(center_z, dtype="f8"),
        "section_axis": axis,
        "xlim_km": xlim,
        "zlim_m": zlim,
    }


def _make_cross_section_fields(
    selected: SelectedObject,
    object_layers: pd.DataFrame,
    raw_root: Path,
    filter_root: Path,
    half_width_deg: float,
    w_shear_depth_padding_layers: int,
    w_shear_half_width_r: float,
    w_shear_min_half_width_km: float,
    w_section_mode: str,
) -> dict[str, dict[str, np.ndarray] | None]:
    del raw_root
    return {
        "first": _make_jump_cross_section_fields(
            selected=selected,
            object_layers=object_layers,
            filter_root=filter_root,
            half_width_deg=half_width_deg,
            jump_from_depth_index=selected.jump_from_depth_index if selected.has_abrupt_jump else None,
            jump_to_depth_index=selected.jump_to_depth_index if selected.has_abrupt_jump else None,
            w_shear_depth_padding_layers=w_shear_depth_padding_layers,
            w_shear_half_width_r=w_shear_half_width_r,
            w_shear_min_half_width_km=w_shear_min_half_width_km,
            w_section_mode=w_section_mode,
        ),
        "second": _make_jump_cross_section_fields(
            selected=selected,
            object_layers=object_layers,
            filter_root=filter_root,
            half_width_deg=half_width_deg,
            jump_from_depth_index=selected.second_jump_from_depth_index if selected.has_second_abrupt_jump else None,
            jump_to_depth_index=selected.second_jump_to_depth_index if selected.has_second_abrupt_jump else None,
            w_shear_depth_padding_layers=w_shear_depth_padding_layers,
            w_shear_half_width_r=w_shear_half_width_r,
            w_shear_min_half_width_km=w_shear_min_half_width_km,
            w_section_mode=w_section_mode,
        ),
    }


def _write_metadata(selected: SelectedObject, output_dir: Path) -> None:
    payload = {
        "eddy3d_object_id": selected.eddy3d_object_id,
        "track3d_id": selected.track3d_id,
        "date": selected.date,
        "polarity": selected.polarity,
        "shape_class": selected.shape_class,
        "n_layers": selected.n_layers,
        "radius_m": selected.radius_m,
        "max_jump_distance_km": selected.jump_distance_km,
        "max_jump_distance_over_R": selected.jump_distance_over_R,
        "jump_from_depth_index": selected.jump_from_depth_index,
        "jump_to_depth_index": selected.jump_to_depth_index,
        "jump_from_depth_m": selected.jump_from_depth_m,
        "jump_to_depth_m": selected.jump_to_depth_m,
        "has_abrupt_jump": selected.has_abrupt_jump,
        "second_jump_distance_km": selected.second_jump_distance_km,
        "second_jump_distance_over_R": selected.second_jump_distance_over_R,
        "second_jump_from_depth_index": selected.second_jump_from_depth_index,
        "second_jump_to_depth_index": selected.second_jump_to_depth_index,
        "second_jump_from_depth_m": selected.second_jump_from_depth_m,
        "second_jump_to_depth_m": selected.second_jump_to_depth_m,
        "has_second_abrupt_jump": selected.has_second_abrupt_jump,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "selected_object_metadata.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame([payload]).to_csv(output_dir / "selected_object_metadata.csv", index=False)


def _metadata_payload(selected: SelectedObject, output_dir: Path) -> dict[str, object]:
    return {
        "eddy3d_object_id": selected.eddy3d_object_id,
        "track3d_id": selected.track3d_id,
        "date": selected.date,
        "polarity": selected.polarity,
        "shape_class": selected.shape_class,
        "n_layers": selected.n_layers,
        "radius_m": selected.radius_m,
        "max_jump_distance_km": selected.jump_distance_km,
        "max_jump_distance_over_R": selected.jump_distance_over_R,
        "jump_from_depth_index": selected.jump_from_depth_index,
        "jump_to_depth_index": selected.jump_to_depth_index,
        "jump_from_depth_m": selected.jump_from_depth_m,
        "jump_to_depth_m": selected.jump_to_depth_m,
        "has_abrupt_jump": selected.has_abrupt_jump,
        "second_jump_distance_km": selected.second_jump_distance_km,
        "second_jump_distance_over_R": selected.second_jump_distance_over_R,
        "second_jump_from_depth_index": selected.second_jump_from_depth_index,
        "second_jump_to_depth_index": selected.second_jump_to_depth_index,
        "second_jump_from_depth_m": selected.second_jump_from_depth_m,
        "second_jump_to_depth_m": selected.second_jump_to_depth_m,
        "has_second_abrupt_jump": selected.has_second_abrupt_jump,
        "output_dir": str(output_dir),
    }


def _selected_from_metadata_row(row: pd.Series, abrupt_threshold_over_R: float) -> SelectedObject:
    def maybe_int(name: str) -> int | None:
        value = row.get(name, np.nan)
        return int(value) if pd.notna(value) and int(value) >= 0 else None

    def maybe_float(name: str) -> float | None:
        value = row.get(name, np.nan)
        return float(value) if pd.notna(value) and np.isfinite(float(value)) else None

    jump_r = float(row.get("max_jump_distance_over_R", row.get("jump_distance_over_R", np.nan)))
    jump2_r = float(row.get("second_jump_distance_over_R", np.nan))
    return SelectedObject(
        track3d_id=int(row["track3d_id"]),
        eddy3d_object_id=int(row["eddy3d_object_id"]),
        date=str(row["date"]),
        polarity=str(row["polarity"]),
        shape_class=str(row["shape_class"]),
        radius_m=float(row["radius_m"]),
        n_layers=int(row["n_layers"]),
        jump_from_depth_index=maybe_int("jump_from_depth_index"),
        jump_to_depth_index=maybe_int("jump_to_depth_index"),
        jump_from_depth_m=maybe_float("jump_from_depth_m"),
        jump_to_depth_m=maybe_float("jump_to_depth_m"),
        jump_distance_km=float(row.get("max_jump_distance_km", row.get("jump_distance_km", np.nan))),
        jump_distance_over_R=jump_r,
        has_abrupt_jump=bool(pd.notna(jump_r) and np.isfinite(jump_r) and jump_r >= abrupt_threshold_over_R),
        second_jump_from_depth_index=maybe_int("second_jump_from_depth_index"),
        second_jump_to_depth_index=maybe_int("second_jump_to_depth_index"),
        second_jump_from_depth_m=maybe_float("second_jump_from_depth_m"),
        second_jump_to_depth_m=maybe_float("second_jump_to_depth_m"),
        second_jump_distance_km=float(row.get("second_jump_distance_km", np.nan)),
        second_jump_distance_over_R=jump2_r,
        has_second_abrupt_jump=bool(pd.notna(jump2_r) and np.isfinite(jump2_r) and jump2_r >= abrupt_threshold_over_R),
    )


def _selections_from_metadata(
    metadata_path: Path,
    centers: pd.DataFrame,
    abrupt_threshold_over_R: float,
) -> list[tuple[SelectedObject, pd.DataFrame, pd.DataFrame, str]]:
    metadata = pd.read_csv(metadata_path)
    selections: list[tuple[SelectedObject, pd.DataFrame, pd.DataFrame, str]] = []
    for idx, row in metadata.iterrows():
        selected = _selected_from_metadata_row(row, abrupt_threshold_over_R)
        object_layers = centers[centers["eddy3d_object_id"].astype(int).eq(selected.eddy3d_object_id)].sort_values("depth_index").copy()
        track_layers = centers[centers["track3d_id"].astype(int).eq(selected.track3d_id)].copy()
        basename = Path(str(row.get("output_dir", ""))).name
        if not basename:
            basename = f"example_{idx + 1:03d}_object_{selected.eddy3d_object_id}_track_{selected.track3d_id}_{selected.date}"
        selections.append((selected, object_layers, track_layers, basename))
    return selections


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot one original eddy 9-panel vertical discontinuity diagnostic.")
    parser.add_argument("--results-root", type=Path, default=Path("/root/autodl-fs/kuroshiou/result_boundary_monotonic"))
    parser.add_argument("--shape-dir-name", default="shape_classification_1993_2022_hua_b3_start2_life30")
    parser.add_argument("--raw-root", type=Path, default=Path("/root/autodl-fs/kuroshiou/raw"))
    parser.add_argument("--filter-root", type=Path, default=Path("/root/autodl-fs/kuroshiou/Filter"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preferred-shapes", default="coherent,complex,mixed,upright_like,transitional")
    parser.add_argument("--min-layers", type=int, default=10)
    parser.add_argument("--abrupt-threshold-over-r", type=float, default=0.15)
    parser.add_argument("--half-width-deg", type=float, default=2.0)
    parser.add_argument("--year-limit", type=int, default=None)
    parser.add_argument("--max-examples", type=int, default=1, help="Number of ranked original eddy object-days to plot.")
    parser.add_argument("--w-shear-depth-padding-layers", type=int, default=6)
    parser.add_argument("--w-shear-half-width-r", type=float, default=1.2)
    parser.add_argument("--w-shear-min-half-width-km", type=float, default=75.0)
    parser.add_argument("--w-section-mode", choices=["parallel", "normal"], default="parallel")
    parser.add_argument("--selected-metadata", type=Path, default=None, help="Reuse a selected_objects_metadata.csv object list instead of re-ranking candidates.")
    parser.add_argument("--output-name-stem", default="original_eddy_discontinuity_9panel")
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    centers, shape_tracks = _load_catalog(args.results_root, args.shape_dir_name)
    if args.selected_metadata is None and args.max_examples <= 1:
        selected, object_layers, track_layers = _choose_object(
            centers,
            shape_tracks,
            preferred_shapes=_format_shape_list(args.preferred_shapes),
            min_layers=args.min_layers,
            abrupt_threshold_over_R=args.abrupt_threshold_over_r,
            year_limit=args.year_limit,
        )
        fields = _make_cross_section_fields(
            selected,
            object_layers,
            args.raw_root,
            args.filter_root,
            args.half_width_deg,
            args.w_shear_depth_padding_layers,
            args.w_shear_half_width_r,
            args.w_shear_min_half_width_km,
            args.w_section_mode,
        )
        _write_metadata(selected, args.output_dir)
        _plot_9panel(selected, object_layers, track_layers, fields, args.output_dir, args.output_name_stem)
        print(json.dumps({"selected_object": selected.__dict__, "output_dir": str(args.output_dir)}, ensure_ascii=False))
        return

    if args.selected_metadata is not None:
        selections = _selections_from_metadata(args.selected_metadata, centers, args.abrupt_threshold_over_r)
    else:
        selections = [
            (*item, "")
            for item in _choose_objects(
                centers,
                shape_tracks,
                preferred_shapes=_format_shape_list(args.preferred_shapes),
                min_layers=args.min_layers,
                abrupt_threshold_over_R=args.abrupt_threshold_over_r,
                year_limit=args.year_limit,
                max_examples=args.max_examples,
            )
        ]
    rows: list[dict[str, object]] = []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for idx, (selected, object_layers, track_layers, basename) in enumerate(selections, start=1):
        child_name = basename or f"example_{idx:03d}_object_{selected.eddy3d_object_id}_track_{selected.track3d_id}_{selected.date}"
        child = args.output_dir / child_name
        fields = _make_cross_section_fields(
            selected,
            object_layers,
            args.raw_root,
            args.filter_root,
            args.half_width_deg,
            args.w_shear_depth_padding_layers,
            args.w_shear_half_width_r,
            args.w_shear_min_half_width_km,
            args.w_section_mode,
        )
        _write_metadata(selected, child)
        _plot_9panel(selected, object_layers, track_layers, fields, child, args.output_name_stem)
        rows.append(_metadata_payload(selected, child))
        print(json.dumps({"example": idx, "selected_object": selected.__dict__, "output_dir": str(child)}, ensure_ascii=False))

    summary = pd.DataFrame(rows)
    summary.to_csv(args.output_dir / "selected_objects_metadata.csv", index=False)
    (args.output_dir / "selected_objects_metadata.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
