from __future__ import annotations

import argparse
import math
import shutil
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm, colors as mcolors, patheffects
from matplotlib.path import Path as MplPath
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import netCDF4
import numpy as np
import pandas as pd
from tqdm import tqdm

from .common import ensure_dirs, load_config
from .table_io import read_table, table_exists


EARTH_RADIUS_KM = 6371.0


def _catalog_paths(config: dict) -> tuple[Path, Path, Path]:
    root = Path(config["paths"]["catalog_dir"])
    return root / "tracks_3d.parquet", root / "vertical_objects.parquet", root / "layer_observations.parquet"


def _completed_paths(config: dict) -> tuple[Path, Path]:
    root = Path(config["paths"]["catalog_dir"])
    return root / "layer_centers_completed.parquet", root / "layer_contours_completed.parquet"


def _contour_path(config: dict, day: pd.Timestamp) -> Path:
    return Path(config["paths"]["layer_dir"]) / f"contours_{day:%Y%m%d}.parquet"


def _uv_path(config: dict, day: pd.Timestamp) -> Path:
    return Path(config["paths"]["input_daily_dir"]) / f"uv_{day:%Y%m%d}.nc"


def _xy_km(lon, lat, lon0: float, lat0: float) -> tuple[np.ndarray, np.ndarray]:
    lon_arr = np.asarray(lon, dtype="f8")
    lat_arr = np.asarray(lat, dtype="f8")
    x = EARTH_RADIUS_KM * np.cos(np.radians(lat0)) * np.radians(lon_arr - lon0)
    y = EARTH_RADIUS_KM * np.radians(lat_arr - lat0)
    return x, y


def _z_plot(
    depth_m,
    depth0_m: float,
    vertical_exaggeration: float,
    depth_index=None,
    z_mode: str = "layer",
    layer_gap_km: float = 18.0,
) -> np.ndarray:
    if z_mode == "layer" and depth_index is not None:
        return -np.asarray(depth_index, dtype="f8") * float(layer_gap_km)
    return -(np.asarray(depth_m, dtype="f8") - depth0_m) * vertical_exaggeration / 1000.0


def _circle_xy(cx: float, cy: float, radius_km: float, n: int = 160) -> tuple[np.ndarray, np.ndarray]:
    theta = np.linspace(0, 2 * np.pi, n)
    return cx + radius_km * np.cos(theta), cy + radius_km * np.sin(theta)


def _load_depth_axis(uv_path: Path) -> np.ndarray:
    with netCDF4.Dataset(uv_path) as ds:
        return np.asarray(ds.variables["depth"][:], dtype="f8")


def _load_speed_slice(uv_path: Path, depth_index: int, lon0: float, lat0: float, extent_km: float):
    with netCDF4.Dataset(uv_path) as ds:
        lon = np.asarray(ds.variables["longitude"][:], dtype="f8")
        lat = np.asarray(ds.variables["latitude"][:], dtype="f8")
        x, y = _xy_km(lon, lat, lon0, lat0)
        ix = np.where((x >= -extent_km) & (x <= extent_km))[0]
        iy = np.where((y >= -extent_km) & (y <= extent_km))[0]
        if ix.size == 0 or iy.size == 0:
            return None
        u = np.ma.filled(ds.variables["u"][depth_index, iy.min() : iy.max() + 1, ix.min() : ix.max() + 1], np.nan)
        v = np.ma.filled(ds.variables["v"][depth_index, iy.min() : iy.max() + 1, ix.min() : ix.max() + 1], np.nan)
        xs = x[ix.min() : ix.max() + 1]
        ys = y[iy.min() : iy.max() + 1]
        speed = np.hypot(u, v)
        return xs, ys, speed, u, v


def _interp_full_core_path(layers: pd.DataFrame, fixed_depths: np.ndarray, lon0: float, lat0: float) -> pd.DataFrame:
    centers = layers.sort_values("depth_m").copy()
    cx, cy = _xy_km(centers["longitude"].to_numpy(), centers["latitude"].to_numpy(), lon0, lat0)
    centers["x_km"] = cx
    centers["y_km"] = cy
    real_depths = centers["depth_m"].to_numpy(dtype="f8")
    real_x = centers["x_km"].to_numpy(dtype="f8")
    real_y = centers["y_km"].to_numpy(dtype="f8")
    real_indices = set(centers["depth_index"].astype(int).tolist())
    if len(real_depths) == 1:
        interp_x = np.full(fixed_depths.shape, real_x[0], dtype="f8")
        interp_y = np.full(fixed_depths.shape, real_y[0], dtype="f8")
    else:
        order = np.argsort(real_depths)
        interp_x = np.interp(fixed_depths, real_depths[order], real_x[order], left=real_x[order][0], right=real_x[order][-1])
        interp_y = np.interp(fixed_depths, real_depths[order], real_y[order], left=real_y[order][0], right=real_y[order][-1])
    return pd.DataFrame(
        {
            "depth_index": np.arange(fixed_depths.size, dtype=int),
            "depth_m": fixed_depths,
            "x_km": interp_x,
            "y_km": interp_y,
            "is_detected": [i in real_indices for i in range(fixed_depths.size)],
        }
    )


def _axis_extent_km(layers: pd.DataFrame, contour_parts: dict[int, pd.DataFrame], lon0: float, lat0: float, padding_km: float) -> float:
    values = []
    for _, row in layers.iterrows():
        x, y = _xy_km([row.longitude], [row.latitude], lon0, lat0)
        r = max(float(row.radius_m) / 1000.0, 0.0)
        values.extend([abs(float(x[0])) + r, abs(float(y[0])) + r])
    for part in contour_parts.values():
        if part.empty:
            continue
        x, y = _xy_km(part["longitude"].to_numpy(), part["latitude"].to_numpy(), lon0, lat0)
        values.extend(np.abs(x).tolist())
        values.extend(np.abs(y).tolist())
    base = max(values) if values else 60.0
    return max(120.0, math.ceil(base + padding_km))


def _axis_extent_completed_km(centers: pd.DataFrame, contour_parts: dict[int, pd.DataFrame], lon0: float, lat0: float, padding_km: float) -> float:
    values = []
    for _, row in centers.iterrows():
        x, y = _xy_km([row.longitude], [row.latitude], lon0, lat0)
        r = max(float(row.radius_m) / 1000.0, 0.0) if "radius_m" in row and np.isfinite(row.radius_m) else 0.0
        values.extend([abs(float(x[0])) + r, abs(float(y[0])) + r])
    for part in contour_parts.values():
        if part.empty:
            continue
        x, y = _xy_km(part["longitude"].to_numpy(), part["latitude"].to_numpy(), lon0, lat0)
        values.extend(np.abs(x).tolist())
        values.extend(np.abs(y).tolist())
    base = max(values) if values else 60.0
    return max(120.0, math.ceil(base + padding_km))


def _inside_layer_shape_mask(xx: np.ndarray, yy: np.ndarray, row: pd.Series, contour_part: pd.DataFrame | None, lon0: float, lat0: float) -> np.ndarray:
    points = np.column_stack([xx.ravel(), yy.ravel()])
    if contour_part is not None and not contour_part.empty:
        cx, cy = _xy_km(contour_part["longitude"].to_numpy(), contour_part["latitude"].to_numpy(), lon0, lat0)
        if cx.size >= 4 and np.isfinite(cx).all() and np.isfinite(cy).all():
            return MplPath(np.column_stack([cx, cy])).contains_points(points).reshape(xx.shape)
    radius_m = float(row.radius_m) if "radius_m" in row and np.isfinite(row.radius_m) else np.nan
    if np.isfinite(radius_m) and radius_m > 0:
        center_x = float(row.x_km) if "x_km" in row and np.isfinite(row.x_km) else 0.0
        center_y = float(row.y_km) if "y_km" in row and np.isfinite(row.y_km) else 0.0
        return np.hypot(xx - center_x, yy - center_y) <= radius_m / 1000.0
    return np.ones(xx.shape, dtype=bool)


def _draw_speed_fill(
    ax,
    uv_path: Path,
    row: pd.Series,
    lon0: float,
    lat0: float,
    z: float,
    extent_km: float,
    show_arrows: bool,
    contour_part: pd.DataFrame | None = None,
    speed_vmax: float | None = None,
    speed_alpha: float = 0.24,
):
    loaded = _load_speed_slice(uv_path, int(row.depth_index), lon0, lat0, extent_km)
    if loaded is None:
        return None
    xs, ys, speed, u, v = loaded
    if speed.size == 0 or not np.isfinite(speed).any():
        return None
    stride = max(1, int(max(speed.shape) / 55))
    xx, yy = np.meshgrid(xs[::stride], ys[::stride])
    zz = np.full_like(xx, z, dtype="f8")
    sp = speed[::stride, ::stride]
    mask = _inside_layer_shape_mask(xx, yy, row, contour_part, lon0, lat0)
    vmax = float(speed_vmax) if speed_vmax is not None and np.isfinite(speed_vmax) else np.nanpercentile(sp, 95)
    if not np.isfinite(vmax) or vmax <= 0:
        return None
    colors = cm.viridis(np.clip(sp / vmax, 0, 1))
    colors[..., -1] = np.where(mask & np.isfinite(sp), speed_alpha, 0.0)
    surf = ax.plot_surface(xx, yy, zz, facecolors=colors, linewidth=0, antialiased=False, shade=False)
    if show_arrows:
        qstride = max(1, int(max(speed.shape) / 18))
        qx, qy = np.meshgrid(xs[::qstride], ys[::qstride])
        qu = u[::qstride, ::qstride]
        qv = v[::qstride, ::qstride]
        qmask = _inside_layer_shape_mask(qx, qy, row, contour_part, lon0, lat0) & np.isfinite(qu) & np.isfinite(qv)
        qspeed = np.hypot(qu, qv)
        scale = np.nanpercentile(qspeed[qmask], 90) if qmask.any() else np.nan
        if np.isfinite(scale) and scale > 0:
            ax.quiver(
                qx[qmask],
                qy[qmask],
                np.full(qmask.sum(), z, dtype="f8"),
                qu[qmask] / scale * 18,
                qv[qmask] / scale * 18,
                np.zeros(qmask.sum(), dtype="f8"),
                color="0.08",
                linewidth=0.55,
                arrow_length_ratio=0.25,
                alpha=0.78,
            )
    return surf


def _frame_speed_vmax(
    uv_path: Path,
    rows: pd.DataFrame,
    contour_parts: dict[int, pd.DataFrame],
    contour_key: str,
    lon0: float,
    lat0: float,
    extent_km: float,
) -> float:
    values: list[float] = []
    for _, row in rows.iterrows():
        loaded = _load_speed_slice(uv_path, int(row.depth_index), lon0, lat0, extent_km)
        if loaded is None:
            continue
        xs, ys, speed, _u, _v = loaded
        if speed.size == 0 or not np.isfinite(speed).any():
            continue
        stride = max(1, int(max(speed.shape) / 55))
        xx, yy = np.meshgrid(xs[::stride], ys[::stride])
        sp = speed[::stride, ::stride]
        key = int(row[contour_key])
        mask = _inside_layer_shape_mask(xx, yy, row, contour_parts.get(key), lon0, lat0)
        inside = sp[mask & np.isfinite(sp)]
        if inside.size:
            values.append(float(np.nanpercentile(inside, 95)))
    if not values:
        return float("nan")
    return float(np.nanpercentile(values, 95))


def _add_speed_colorbar(fig, ax, speed_vmax: float) -> None:
    if not np.isfinite(speed_vmax) or speed_vmax <= 0:
        return
    sm = cm.ScalarMappable(norm=mcolors.Normalize(vmin=0.0, vmax=speed_vmax), cmap=cm.viridis)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.025, shrink=0.58)
    cbar.set_label("speed magnitude (m/s)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)


def _render_completed_frame(
    config: dict,
    track_id: int,
    frame_no: int,
    frame_total: int,
    obj_row: pd.Series,
    completed_centers: pd.DataFrame,
    completed_contours: pd.DataFrame,
    output_png: Path,
    *,
    elev: float,
    azim: float,
    vertical_exaggeration: float,
    padding_km: float,
    show_speed_fill: bool,
    show_arrows: bool,
    z_mode: str,
    layer_gap_km: float,
    z_aspect: float,
) -> dict:
    day = pd.Timestamp(obj_row.date)
    uv_path = _uv_path(config, day)
    fixed_depths = _load_depth_axis(uv_path)
    centers = completed_centers.sort_values("depth_index").copy()
    origin = centers.iloc[0]
    lon0 = float(origin.longitude)
    lat0 = float(origin.latitude)
    depth0 = float(origin.depth_m)
    centers["x_km"], centers["y_km"] = _xy_km(centers["longitude"].to_numpy(), centers["latitude"].to_numpy(), lon0, lat0)
    centers["z_plot"] = _z_plot(
        centers["depth_m"].to_numpy(),
        depth0,
        vertical_exaggeration,
        depth_index=centers["depth_index"].to_numpy(),
        z_mode=z_mode,
        layer_gap_km=layer_gap_km,
    )
    contour_parts = {
        int(depth_index): part.sort_values("point_index")
        for depth_index, part in completed_contours.groupby("depth_index")
    }
    extent = _axis_extent_completed_km(centers, contour_parts, lon0, lat0, padding_km)

    fig = plt.figure(figsize=(12, 9), dpi=150)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_proj_type("ortho")
    ax.view_init(elev=elev, azim=azim)

    speed_vmax = float("nan")
    if show_speed_fill:
        speed_vmax = _frame_speed_vmax(uv_path, centers, contour_parts, "depth_index", lon0, lat0, min(extent, 220))
        for _, row in centers.iterrows():
            depth_index = int(row.depth_index)
            _draw_speed_fill(
                ax,
                uv_path,
                row,
                lon0,
                lat0,
                float(row.z_plot),
                min(extent, 220),
                show_arrows,
                contour_part=contour_parts.get(depth_index),
                speed_vmax=speed_vmax,
            )
        _add_speed_colorbar(fig, ax, speed_vmax)

    polarity = str(obj_row.polarity)
    contour_color = "#d95f02" if polarity == "anticyclonic" else "#1f78b4"
    for _, row in centers.iterrows():
        depth_index = int(row.depth_index)
        z = float(row.z_plot)
        part = contour_parts.get(depth_index)
        found = bool(row.contour_found) if "contour_found" in row else False
        if part is not None and not part.empty:
            x, y = _xy_km(part["longitude"].to_numpy(), part["latitude"].to_numpy(), lon0, lat0)
            color = contour_color if found else "0.55"
            style = "-" if found else "--"
            alpha = 0.86 if found else 0.42
            line, = ax.plot(x, y, np.full_like(x, z), color=color, linestyle=style, linewidth=1.15 if found else 0.75, alpha=alpha)
            if found:
                line.set_path_effects([patheffects.Stroke(linewidth=2.0, foreground="white"), patheffects.Normal()])
        radius = float(row.radius_m) / 1000.0 if np.isfinite(row.radius_m) else 0.0
        if radius > 0:
            rx, ry = _circle_xy(float(row.x_km), float(row.y_km), radius)
            circle_color = "black" if found else "0.55"
            circle_style = ":" if found else "--"
            circle, = ax.plot(rx, ry, np.full_like(rx, z), color=circle_color, linestyle=circle_style, linewidth=0.75, alpha=0.55)
            circle.set_path_effects([patheffects.Stroke(linewidth=1.45, foreground="white"), patheffects.Normal()])
        detected = bool(row.center_is_detected) if "center_is_detected" in row else False
        ax.scatter(
            [float(row.x_km)],
            [float(row.y_km)],
            [z],
            s=28 if detected else 16,
            color="black" if detected else "0.40",
            edgecolor="white",
            linewidth=0.6,
            alpha=1.0 if detected else 0.65,
            depthshade=False,
        )

    centers = centers.sort_values("depth_index")
    for a, b in zip(centers.iloc[:-1].itertuples(index=False), centers.iloc[1:].itertuples(index=False)):
        both_detected = bool(a.center_is_detected) and bool(b.center_is_detected)
        style = "-" if both_detected else "--"
        color = "black" if both_detected else "0.45"
        width = 1.9 if both_detected else 1.1
        line, = ax.plot([a.x_km, b.x_km], [a.y_km, b.y_km], [a.z_plot, b.z_plot], linestyle=style, color=color, linewidth=width, alpha=0.92)
        if both_detected:
            line.set_path_effects([patheffects.Stroke(linewidth=3.0, foreground="white"), patheffects.Normal()])

    z_values = centers["z_plot"].to_numpy()
    zmin = float(np.nanmin(z_values))
    zmax = float(np.nanmax(z_values))
    ax.set_xlim(-extent, extent)
    ax.set_ylim(-extent, extent)
    ax.set_zlim(zmin - layer_gap_km * 0.6, zmax + layer_gap_km * 0.6)
    ax.set_box_aspect((1, 1, float(z_aspect)))
    ax.set_xlabel("x from completed surface core (km)", fontsize=8)
    ax.set_ylabel("y from completed surface core (km)", fontsize=8)
    z_label = f"fixed layer display ({layer_gap_km:g} km/layer)" if z_mode == "layer" else f"depth display (km), vertical x{vertical_exaggeration:g}"
    ax.set_zlabel(z_label, fontsize=8)
    ax.tick_params(labelsize=7, pad=0)
    ax.grid(True, alpha=0.16)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_alpha(0.02)

    detected_count = int(centers["center_is_detected"].sum())
    completed_count = int(len(centers))
    contour_count = int(centers["contour_found"].sum()) if "contour_found" in centers else 0
    title = (
        f"track3d_id={track_id} | {day:%Y-%m-%d} | day {frame_no}/{frame_total} | object={int(obj_row.eddy3d_object_id)}\n"
        f"{polarity}, detected layers={detected_count}/{len(fixed_depths)}, completed centers={completed_count}/{len(fixed_depths)}, closed contours={contour_count}/{len(fixed_depths)}"
    )
    ax.set_title(title, fontsize=10, pad=12)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, bbox_inches="tight")
    plt.close(fig)
    return {
        "frame": frame_no,
        "date": f"{day:%Y-%m-%d}",
        "eddy3d_object_id": int(obj_row.eddy3d_object_id),
        "png_path": str(output_png),
        "origin_lon": lon0,
        "origin_lat": lat0,
        "origin_depth_m": depth0,
        "detected_layer_count": detected_count,
        "fixed_depth_count": int(len(fixed_depths)),
        "completed_center_count": completed_count,
        "closed_contour_count": contour_count,
        "min_depth_m": float(centers.depth_m.min()),
        "max_depth_m": float(centers.depth_m.max()),
        "polarity": polarity,
        "z_mode": z_mode,
        "layer_gap_km": float(layer_gap_km),
    }


def _render_frame(
    config: dict,
    track_id: int,
    frame_no: int,
    frame_total: int,
    obj_row: pd.Series,
    layers: pd.DataFrame,
    contours: pd.DataFrame,
    output_png: Path,
    *,
    elev: float,
    azim: float,
    vertical_exaggeration: float,
    padding_km: float,
    show_all_depths: bool,
    show_speed_fill: bool,
    show_arrows: bool,
    completed_centers: pd.DataFrame | None = None,
    completed_contours: pd.DataFrame | None = None,
    z_mode: str = "layer",
    layer_gap_km: float = 18.0,
    z_aspect: float = 0.85,
) -> dict:
    if completed_centers is not None and not completed_centers.empty:
        return _render_completed_frame(
            config,
            track_id,
            frame_no,
            frame_total,
            obj_row,
            completed_centers,
            completed_contours if completed_contours is not None else pd.DataFrame(),
            output_png,
            elev=elev,
            azim=azim,
            vertical_exaggeration=vertical_exaggeration,
            padding_km=padding_km,
            show_speed_fill=show_speed_fill,
            show_arrows=show_arrows,
            z_mode=z_mode,
            layer_gap_km=layer_gap_km,
            z_aspect=z_aspect,
        )
    day = pd.Timestamp(obj_row.date)
    uv_path = _uv_path(config, day)
    if not uv_path.exists():
        raise FileNotFoundError(f"Missing UV input: {uv_path}")
    fixed_depths = _load_depth_axis(uv_path)
    layers = layers.sort_values("depth_m").copy()
    origin = layers.iloc[0]
    lon0 = float(origin.longitude)
    lat0 = float(origin.latitude)
    depth0 = float(origin.depth_m)
    layers["x_km"], layers["y_km"] = _xy_km(layers["longitude"].to_numpy(), layers["latitude"].to_numpy(), lon0, lat0)
    layers["z_plot"] = _z_plot(layers["depth_m"].to_numpy(), depth0, vertical_exaggeration)

    contour_parts = {
        int(det_id): part.sort_values("point_index")
        for det_id, part in contours.groupby("layer_detection_id")
    }
    extent = _axis_extent_km(layers, contour_parts, lon0, lat0, padding_km)
    full_path = _interp_full_core_path(layers, fixed_depths, lon0, lat0)
    full_path["z_plot"] = _z_plot(full_path["depth_m"].to_numpy(), depth0, vertical_exaggeration)

    fig = plt.figure(figsize=(12, 9), dpi=150)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_proj_type("ortho")
    ax.view_init(elev=elev, azim=azim)

    speed_vmax = float("nan")
    if show_speed_fill:
        speed_vmax = _frame_speed_vmax(uv_path, layers, contour_parts, "layer_detection_id", lon0, lat0, min(extent, 220))
        for _, row in layers.iterrows():
            det_id = int(row.layer_detection_id)
            _draw_speed_fill(
                ax,
                uv_path,
                row,
                lon0,
                lat0,
                float(row.z_plot),
                min(extent, 220),
                show_arrows,
                contour_part=contour_parts.get(det_id),
                speed_vmax=speed_vmax,
            )
        _add_speed_colorbar(fig, ax, speed_vmax)

    polarity = str(obj_row.polarity)
    contour_color = "#d95f02" if polarity == "anticyclonic" else "#1f78b4"
    for _, row in layers.iterrows():
        det_id = int(row.layer_detection_id)
        z = float(row.z_plot)
        part = contour_parts.get(det_id)
        if part is not None and not part.empty:
            x, y = _xy_km(part["longitude"].to_numpy(), part["latitude"].to_numpy(), lon0, lat0)
            line, = ax.plot(x, y, np.full_like(x, z), color=contour_color, linewidth=1.25, alpha=0.95)
            line.set_path_effects([patheffects.Stroke(linewidth=2.2, foreground="white"), patheffects.Normal()])
        radius = max(float(row.radius_m) / 1000.0, 0.0)
        cx = float(row.x_km)
        cy = float(row.y_km)
        if np.isfinite(radius) and radius > 0:
            rx, ry = _circle_xy(cx, cy, radius)
            circle, = ax.plot(rx, ry, np.full_like(rx, z), color="black", linestyle=":", linewidth=0.9, alpha=0.9)
            circle.set_path_effects([patheffects.Stroke(linewidth=1.8, foreground="white"), patheffects.Normal()])
        ax.scatter([cx], [cy], [z], s=28, color="black", edgecolor="white", linewidth=0.7, depthshade=False)

    real_by_depth = set(layers["depth_index"].astype(int).tolist())
    if show_all_depths:
        missing = full_path[~full_path["is_detected"]]
        if not missing.empty:
            ax.scatter(missing["x_km"], missing["y_km"], missing["z_plot"], s=10, color="0.68", alpha=0.55, depthshade=False)
            marker_r = max(4.0, min(14.0, extent * 0.025))
            for _, row in missing.iterrows():
                mx, my = _circle_xy(float(row.x_km), float(row.y_km), marker_r, n=48)
                ax.plot(mx, my, np.full_like(mx, float(row.z_plot)), color="0.75", linewidth=0.45, alpha=0.35)

    path_df = full_path if show_all_depths else layers[["depth_index", "depth_m", "x_km", "y_km", "z_plot"]].assign(is_detected=True)
    path_df = path_df.sort_values("depth_m")
    for a, b in zip(path_df.iloc[:-1].itertuples(index=False), path_df.iloc[1:].itertuples(index=False)):
        both_real = int(a.depth_index) in real_by_depth and int(b.depth_index) in real_by_depth
        style = "-" if both_real else "--"
        color = "black" if both_real else "0.55"
        width = 2.0 if both_real else 1.0
        line, = ax.plot([a.x_km, b.x_km], [a.y_km, b.y_km], [a.z_plot, b.z_plot], linestyle=style, color=color, linewidth=width, alpha=0.95)
        if both_real:
            line.set_path_effects([patheffects.Stroke(linewidth=3.3, foreground="white"), patheffects.Normal()])

    z_values = _z_plot(fixed_depths, depth0, vertical_exaggeration) if show_all_depths else layers["z_plot"].to_numpy()
    zmin = min(float(np.nanmin(z_values)), -1.0)
    zmax = max(float(np.nanmax(z_values)), 1.0)
    ax.set_xlim(-extent, extent)
    ax.set_ylim(-extent, extent)
    ax.set_zlim(zmin - 2, zmax + 2)
    ax.set_box_aspect((1, 1, max(0.18, min(0.65, (zmax - zmin + 4) / (2 * extent)))))
    ax.set_xlabel("x from daily shallowest core (km)", fontsize=8)
    ax.set_ylabel("y from daily shallowest core (km)", fontsize=8)
    ax.set_zlabel(f"depth display (km), vertical x{vertical_exaggeration:g}", fontsize=8)
    ax.tick_params(labelsize=7, pad=0)
    ax.grid(True, alpha=0.18)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_alpha(0.02)

    fixed_count = int(fixed_depths.size)
    detected_count = int(layers.depth_index.nunique())
    title = (
        f"track3d_id={track_id} | {day:%Y-%m-%d} | day {frame_no}/{frame_total} | "
        f"object={int(obj_row.eddy3d_object_id)}\n"
        f"{polarity}, detected layers={detected_count}/{fixed_count}, origin=({lon0:.3f}E, {lat0:.3f}N, {depth0:.1f} m)"
    )
    ax.set_title(title, fontsize=10, pad=12)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, bbox_inches="tight")
    plt.close(fig)
    return {
        "frame": frame_no,
        "date": f"{day:%Y-%m-%d}",
        "eddy3d_object_id": int(obj_row.eddy3d_object_id),
        "png_path": str(output_png),
        "origin_lon": lon0,
        "origin_lat": lat0,
        "origin_depth_m": depth0,
        "detected_layer_count": detected_count,
        "fixed_depth_count": fixed_count,
        "min_depth_m": float(layers.depth_m.min()),
        "max_depth_m": float(layers.depth_m.max()),
        "polarity": polarity,
    }


def _existing_frame_record(config: dict, track_id: int, frame_no: int, frame_total: int, obj_row: pd.Series, layers: pd.DataFrame, png_path: Path) -> dict:
    day = pd.Timestamp(obj_row.date)
    uv_path = _uv_path(config, day)
    fixed_depths = _load_depth_axis(uv_path) if uv_path.exists() else np.array([])
    layers = layers.sort_values("depth_m").copy()
    origin = layers.iloc[0]
    return {
        "frame": frame_no,
        "date": f"{day:%Y-%m-%d}",
        "eddy3d_object_id": int(obj_row.eddy3d_object_id),
        "png_path": str(png_path),
        "origin_lon": float(origin.longitude),
        "origin_lat": float(origin.latitude),
        "origin_depth_m": float(origin.depth_m),
        "detected_layer_count": int(layers.depth_index.nunique()),
        "fixed_depth_count": int(fixed_depths.size),
        "min_depth_m": float(layers.depth_m.min()),
        "max_depth_m": float(layers.depth_m.max()),
        "polarity": str(obj_row.polarity),
        "status": "exists",
    }


def _make_mp4(frame_paths: list[Path], output_mp4: Path, fps: int) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        print("[mp4 skipped] ffmpeg was not found on PATH.")
        return
    if not frame_paths:
        print("[mp4 skipped] no rendered frames were found.")
        return
    list_path = output_mp4.with_suffix(".ffmpeg_frames.txt")
    duration = 1.0 / max(float(fps), 0.1)
    lines = []
    for path in frame_paths:
        safe_path = str(path.resolve()).replace("\\", "/").replace("'", "'\\''")
        lines.append(f"file '{safe_path}'")
        lines.append(f"duration {duration:.6f}")
    last_path = str(frame_paths[-1].resolve()).replace("\\", "/").replace("'", "'\\''")
    lines.append(f"file '{last_path}'")
    list_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    cmd = [
        ffmpeg,
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(list_path),
        "-vf",
        f"fps={fps},scale=trunc(iw/2)*2:trunc(ih/2)*2,format=yuv420p",
        "-c:v",
        "libx264",
        str(output_mp4),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or str(exc)).strip().splitlines()
        tail = "\n".join(detail[-8:])
        print(f"[mp4 failed] ffmpeg returned {exc.returncode}:\n{tail}")
    except Exception as exc:
        print(f"[mp4 failed] {type(exc).__name__}: {exc}")


def plot_track_daily(
    config_path: str | Path,
    track3d_id: int,
    output_dir: str | Path | None = None,
    fps: int = 2,
    elev: float = 32,
    azim: float = -62,
    vertical_exaggeration: float = 20,
    padding_km: float = 60,
    show_all_depths: bool = False,
    show_speed_fill: bool = False,
    show_arrows: bool = False,
    z_mode: str | None = None,
    layer_gap_km: float | None = None,
    z_aspect: float | None = None,
    force: bool = False,
) -> None:
    config = load_config(config_path)
    ensure_dirs(config)
    pcfg = config.get("plotting", {})
    z_mode = (z_mode or pcfg.get("z_mode", "layer")).lower()
    layer_gap_km = float(layer_gap_km if layer_gap_km is not None else pcfg.get("layer_gap_km", 18.0))
    z_aspect = float(z_aspect if z_aspect is not None else pcfg.get("z_aspect", 0.85))
    tracks_path, objects_path, layers_path = _catalog_paths(config)
    tracks = read_table(tracks_path)
    objects = read_table(objects_path)
    layers_all = read_table(layers_path)
    completed_centers_path, completed_contours_path = _completed_paths(config)
    completed_centers_all = read_table(completed_centers_path) if table_exists(completed_centers_path) else pd.DataFrame()
    completed_contours_all = read_table(completed_contours_path) if table_exists(completed_contours_path) else pd.DataFrame()
    if track3d_id not in set(tracks["track3d_id"].astype(int)):
        raise ValueError(f"track3d_id={track3d_id} was not found in {tracks_path}")
    objects = objects[objects["track3d_id"].astype(int) == int(track3d_id)].copy()
    if objects.empty:
        raise ValueError(f"No vertical objects found for track3d_id={track3d_id}")
    objects["date"] = pd.to_datetime(objects["date"])
    objects = objects.sort_values("date")
    out_dir = Path(output_dir) if output_dir else Path(config["paths"]["output_dir"]) / "figures" / f"track3d_{track3d_id}"
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    records = []
    total = len(objects)
    for frame_no, (_, obj_row) in enumerate(tqdm(list(objects.iterrows()), desc=f"Render track3d {track3d_id}", unit="frame"), start=1):
        day = pd.Timestamp(obj_row.date)
        png_path = frames_dir / f"frame_{frame_no:04d}_{day:%Y%m%d}_object_{int(obj_row.eddy3d_object_id)}.png"
        layers = layers_all[layers_all["eddy3d_object_id"].astype(int) == int(obj_row.eddy3d_object_id)].copy()
        completed_centers = pd.DataFrame()
        completed_contours = pd.DataFrame()
        if not completed_centers_all.empty:
            completed_centers = completed_centers_all[
                completed_centers_all["eddy3d_object_id"].astype(int) == int(obj_row.eddy3d_object_id)
            ].copy()
        if not completed_contours_all.empty:
            completed_contours = completed_contours_all[
                completed_contours_all["eddy3d_object_id"].astype(int) == int(obj_row.eddy3d_object_id)
            ].copy()
        if layers.empty:
            if completed_centers.empty:
                records.append({"frame": frame_no, "date": f"{day:%Y-%m-%d}", "eddy3d_object_id": int(obj_row.eddy3d_object_id), "png_path": str(png_path), "status": "missing_layers"})
                continue
        if png_path.exists() and not force:
            records.append(_existing_frame_record(config, track3d_id, frame_no, total, obj_row, layers, png_path))
            continue
        contours = pd.DataFrame()
        if not layers.empty:
            contours = read_table(_contour_path(config, day))
            ids = set(layers["layer_detection_id"].astype(int))
            contours = contours[contours["layer_detection_id"].astype(int).isin(ids)].copy()
        try:
            record = _render_frame(
                config,
                track3d_id,
                frame_no,
                total,
                obj_row,
                layers,
                contours,
                png_path,
                elev=elev,
                azim=azim,
                vertical_exaggeration=vertical_exaggeration,
                padding_km=padding_km,
                show_all_depths=show_all_depths,
                show_speed_fill=show_speed_fill,
                show_arrows=show_arrows,
                completed_centers=completed_centers,
                completed_contours=completed_contours,
                z_mode=z_mode,
                layer_gap_km=layer_gap_km,
                z_aspect=z_aspect,
            )
            record["status"] = "rendered"
            records.append(record)
        except Exception as exc:
            records.append({"frame": frame_no, "date": f"{day:%Y-%m-%d}", "eddy3d_object_id": int(obj_row.eddy3d_object_id), "png_path": str(png_path), "status": f"failed: {type(exc).__name__}: {exc}"})
            print(f"[frame failed] {day:%Y-%m-%d} object={int(obj_row.eddy3d_object_id)}: {exc}")

    meta = pd.DataFrame.from_records(records)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / f"track3d_{track3d_id}_frames.csv"
    meta.to_csv(meta_path, index=False)
    mp4_path = out_dir / f"track3d_{track3d_id}_daily3d.mp4"
    rendered = meta[meta["status"].isin(["rendered", "exists"])]
    if not rendered.empty:
        frame_paths = [Path(p) for p in rendered.sort_values("frame")["png_path"].tolist()]
        _make_mp4(frame_paths, mp4_path, fps)
    print(f"Frames: {frames_dir}")
    print(f"Metadata: {meta_path}")
    print(f"MP4 target: {mp4_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render daily 3D frames for a single tracked 3D eddy.")
    parser.add_argument("--config", default="config/config_3d.yaml")
    parser.add_argument("--track3d-id", type=int, required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--fps", type=int, default=2)
    parser.add_argument("--elev", type=float, default=32)
    parser.add_argument("--azim", type=float, default=-62)
    parser.add_argument("--vertical-exaggeration", type=float, default=20)
    parser.add_argument("--padding-km", type=float, default=60)
    parser.add_argument("--z-mode", choices=("layer", "depth"), default=None)
    parser.add_argument("--layer-gap-km", type=float, default=None)
    parser.add_argument("--z-aspect", type=float, default=None)
    parser.add_argument("--show-all-depths", action="store_true")
    parser.add_argument("--show-speed-fill", action="store_true")
    parser.add_argument("--show-arrows", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    plot_track_daily(
        args.config,
        args.track3d_id,
        output_dir=args.output_dir,
        fps=args.fps,
        elev=args.elev,
        azim=args.azim,
        vertical_exaggeration=args.vertical_exaggeration,
        padding_km=args.padding_km,
        show_all_depths=args.show_all_depths,
        show_speed_fill=args.show_speed_fill,
        show_arrows=args.show_arrows,
        z_mode=args.z_mode,
        layer_gap_km=args.layer_gap_km,
        z_aspect=args.z_aspect,
        force=args.force,
    )


if __name__ == "__main__":
    main()
