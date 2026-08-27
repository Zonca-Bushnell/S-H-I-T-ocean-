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


def _choose_object(
    centers: pd.DataFrame,
    shape: pd.DataFrame,
    preferred_shapes: set[str],
    min_layers: int,
    abrupt_threshold_over_R: float,
    year_limit: int | None,
) -> tuple[SelectedObject, pd.DataFrame, pd.DataFrame]:
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
        if jumps.size:
            i = int(np.nanargmax(jumps))
            jump_km = float(jumps[i] / 1000.0)
            jump_r = float(jumps[i] / radius_m) if radius_m > 0 else np.nan
            k0 = int(part.iloc[i]["depth_index"])
            k1 = int(part.iloc[i + 1]["depth_index"])
            z0 = float(part.iloc[i]["depth_m"])
            z1 = float(part.iloc[i + 1]["depth_m"])
        else:
            jump_km = np.nan
            jump_r = np.nan
            k0 = k1 = -1
            z0 = z1 = np.nan
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
            }
        )
    candidates = pd.DataFrame(rows)
    if candidates.empty:
        raise ValueError("No candidate object-days with enough layers")
    candidates = candidates.sort_values(["jump_distance_over_R", "n_layers"], ascending=[False, False])
    best = candidates.iloc[0]
    selected = SelectedObject(
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
    )
    obj = allowed[allowed["eddy3d_object_id"].astype(int).eq(selected.eddy3d_object_id)].sort_values("depth_index").copy()
    track = centers[centers["track3d_id"].astype(int).eq(selected.track3d_id)].copy()
    return selected, obj, track


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


def _plot_7panel(
    selected: SelectedObject,
    object_layers: pd.DataFrame,
    track_layers: pd.DataFrame,
    fields: dict[str, np.ndarray] | None,
    output_dir: Path,
    density_note: str,
) -> None:
    offsets = _object_offsets_km(object_layers)
    surface = offsets.iloc[0]
    has_abrupt = bool(selected.has_abrupt_jump and fields is not None)

    fig = plt.figure(figsize=(16, 11), constrained_layout=True)
    gs = fig.add_gridspec(3, 4, height_ratios=[1.0, 1.0, 0.9], width_ratios=[0.95, 0.95, 1.1, 1.1])
    ax1 = fig.add_subplot(gs[0:2, 0])
    ax2 = fig.add_subplot(gs[0:2, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[0, 3])
    ax5 = fig.add_subplot(gs[1, 2])
    ax6 = fig.add_subplot(gs[1, 3])
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

    if has_abrupt and fields is not None:
        xx = fields["xx"]
        yy = fields["yy"]
        marks = fields["marks"]
        m3 = _plot_field(ax3, xx, yy, fields["speed"], "3  horizontal speed |u',v'|", "magma", quiver=(fields["u"], fields["v"]), center_marks=marks)
        fig.colorbar(m3, ax=ax3, shrink=0.82, label="m/s")
        m4 = _plot_field(ax4, xx, yy, fields["pressure"], "4  geostrophic pressure proxy p'", "RdBu_r", symmetric=True, center_marks=marks)
        fig.colorbar(m4, ax=ax4, shrink=0.82, label="Pa proxy")
        m5 = _plot_field(ax5, xx, yy, fields["wdiag"], "5  continuity vertical velocity proxy", "RdBu_r", symmetric=True, center_marks=marks)
        fig.colorbar(m5, ax=ax5, shrink=0.82, label="m/s proxy")
        m6 = _plot_field(ax6, xx, yy, fields["sigma0"], f"6  density: {density_note}", "viridis", center_marks=marks)
        fig.colorbar(m6, ax=ax6, shrink=0.82, label="kg m$^{-3}$ proxy")
        for ax in [ax3, ax4, ax5, ax6]:
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(loc="upper right", fontsize=8)
    else:
        for ax in [ax3, ax4, ax5, ax6]:
            _hatch_unavailable(ax, "no abrupt layer discontinuity detected")

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
    fig.suptitle(
        "Original eddy discontinuity diagnostic, not representative vortex\n"
        f"object {selected.eddy3d_object_id}, track {selected.track3d_id}, {selected.date}, "
        f"{selected.shape_class}/{selected.polarity}; {jump_text}",
        fontsize=15,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "original_eddy_discontinuity_7panel.png", dpi=220)
    fig.savefig(output_dir / "original_eddy_discontinuity_7panel.pdf")
    plt.close(fig)


def _make_cross_section_fields(
    selected: SelectedObject,
    object_layers: pd.DataFrame,
    raw_root: Path,
    filter_root: Path,
    half_width_deg: float,
) -> tuple[dict[str, np.ndarray] | None, str]:
    if not selected.has_abrupt_jump or selected.jump_to_depth_index is None:
        return None, ""

    year = str(selected.date)[:4]
    filter_path = filter_root / f"global_phy_{year}_bandpass_30_180d.nc"
    raw_path = raw_root / f"global_phy_{year}.nc"
    if not filter_path.exists() or not raw_path.exists():
        return None, "missing raw/filter files"

    offsets = _object_offsets_km(object_layers)
    surface = offsets.iloc[0]
    surface_lon = float(surface["longitude"])
    surface_lat = float(surface["latitude"])
    layer_index = int(selected.jump_to_depth_index)

    vel2 = _read_field_window(
        path=filter_path,
        date=selected.date,
        center_lon=surface_lon,
        center_lat=surface_lat,
        depth_index=layer_index,
        half_width_deg=half_width_deg,
        variables=("uo_glor", "vo_glor"),
    )
    raw2 = _read_field_window(
        path=raw_path,
        date=selected.date,
        center_lon=surface_lon,
        center_lat=surface_lat,
        depth_index=layer_index,
        half_width_deg=half_width_deg,
        variables=("thetao_glor", "so_glor"),
    )
    vel3 = _read_velocity_column(filter_path, selected.date, surface_lon, surface_lat, half_width_deg)

    lon = vel2["longitude"]
    lat = vel2["latitude"]
    x_m, y_m, xx, yy = _relative_xy(lon, lat, surface_lon, surface_lat)
    k = _nearest_layer_index(vel3["depth"], layer_index)
    u = vel2["uo_glor"]
    v = vel2["vo_glor"]
    speed = np.hypot(u, v)
    pressure = _pressure_proxy(u, v, x_m, y_m, _coriolis(surface_lat))
    wdiag = _vertical_velocity_proxy(vel3["uo_glor"], vel3["vo_glor"], vel3["depth"], x_m, y_m)[k]
    sigma0, density_note = _sigma0(raw2["thetao_glor"], raw2["so_glor"])
    marks = _build_center_marks(
        offsets,
        surface_lon,
        surface_lat,
        selected.jump_from_depth_index,
        selected.jump_to_depth_index,
    )

    fields = {
        "xx": xx,
        "yy": yy,
        "u": u,
        "v": v,
        "speed": speed,
        "pressure": pressure,
        "wdiag": wdiag,
        "sigma0": sigma0,
        "marks": marks,
    }
    return fields, density_note


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
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "selected_object_metadata.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame([payload]).to_csv(output_dir / "selected_object_metadata.csv", index=False)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot one original eddy 7-panel vertical discontinuity diagnostic.")
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
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    centers, shape_tracks = _load_catalog(args.results_root, args.shape_dir_name)
    selected, object_layers, track_layers = _choose_object(
        centers,
        shape_tracks,
        preferred_shapes=_format_shape_list(args.preferred_shapes),
        min_layers=args.min_layers,
        abrupt_threshold_over_R=args.abrupt_threshold_over_r,
        year_limit=args.year_limit,
    )
    fields, density_note = _make_cross_section_fields(
        selected,
        object_layers,
        args.raw_root,
        args.filter_root,
        args.half_width_deg,
    )
    if not density_note:
        density_note = "not used"
    _write_metadata(selected, args.output_dir)
    _plot_7panel(selected, object_layers, track_layers, fields, args.output_dir, density_note)
    print(json.dumps({"selected_object": selected.__dict__, "output_dir": str(args.output_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
