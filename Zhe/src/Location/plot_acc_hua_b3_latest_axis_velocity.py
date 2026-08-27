from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm, colors
from matplotlib.patches import Circle
from netCDF4 import Dataset, num2date
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


EARTH_RADIUS_M = 6_371_000.0


@dataclass
class SliceData:
    depth_index: int
    depth_m: float
    x_km: np.ndarray
    y_km: np.ndarray
    u: np.ndarray
    v: np.ndarray
    u_raw: np.ndarray
    v_raw: np.ndarray


def _as_date(value: str) -> date:
    return datetime.strptime(str(value)[:10], "%Y-%m-%d").date()


def _wrap_lon_delta_deg(lon: np.ndarray, lon0: float) -> np.ndarray:
    return (lon - lon0 + 180.0) % 360.0 - 180.0


def _nearest_time_index(ds: Dataset, wanted: date) -> int:
    time_var = ds.variables["time"]
    times = num2date(time_var[:], units=time_var.units, calendar=getattr(time_var, "calendar", "standard"))
    keys = [date(int(t.year), int(t.month), int(t.day)) for t in times]
    return keys.index(wanted)


def _clean(value: np.ndarray) -> np.ndarray:
    arr = np.ma.filled(value, np.nan).astype("float64", copy=False)
    arr[np.abs(arr) > 1.0e10] = np.nan
    return arr


def _subset(lon: np.ndarray, lat: np.ndarray, lon0: float, lat0: float, half_width_km: float) -> tuple[np.ndarray, np.ndarray]:
    dx_m = np.deg2rad(_wrap_lon_delta_deg(lon, lon0)) * EARTH_RADIUS_M * math.cos(math.radians(lat0))
    dy_m = np.deg2rad(lat - lat0) * EARTH_RADIUS_M
    lon_idx = np.where(np.abs(dx_m) <= half_width_km * 1000.0)[0]
    lat_idx = np.where(np.abs(dy_m) <= half_width_km * 1000.0)[0]
    if len(lon_idx) < 8 or len(lat_idx) < 8:
        raise ValueError("Velocity window is too small; check center position and window size.")
    return lon_idx, lat_idx


def _read_table(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_parquet(path, columns=columns)


def _choose_object(root: Path, shape_dir: str, shape_class: str) -> dict[str, object]:
    shape = _read_table(root / shape_dir / "shape_tracks.parquet")
    use_tracks = shape[shape["shape_class"].eq(shape_class)].copy()
    if use_tracks.empty:
        raise ValueError(f"No {shape_class!r} tracks in {root / shape_dir}")
    centers = _read_table(
        root / "catalog" / "layer_centers_completed.parquet",
        columns=["date", "track3d_id", "eddy3d_object_id", "depth_index", "depth_m", "longitude", "latitude", "radius_m", "polarity"],
    )
    cand = centers[centers["track3d_id"].isin(use_tracks["track3d_id"])].copy()
    day = (
        cand.groupby(["track3d_id", "eddy3d_object_id", "date", "polarity"], as_index=False)
        .agg(
            n_layers=("depth_index", "nunique"),
            max_depth_m=("depth_m", "max"),
            mean_radius_m=("radius_m", "mean"),
            surface_lon=("longitude", "first"),
            surface_lat=("latitude", "first"),
        )
        .merge(use_tracks[["track3d_id", "shape_class", "lifetime_days"]], on="track3d_id", how="left")
    )
    # Prefer well-developed layer stacks, then longer tracks, then a middle-ish date.
    day["date_ts"] = pd.to_datetime(day["date"])
    chosen = day.sort_values(["n_layers", "lifetime_days", "date_ts"], ascending=[False, False, True]).iloc[0]
    return {
        "shape_dir": shape_dir,
        "shape_class": shape_class,
        "track3d_id": int(chosen["track3d_id"]),
        "eddy3d_object_id": int(chosen["eddy3d_object_id"]),
        "date": str(chosen["date"])[:10],
        "polarity": str(chosen["polarity"]),
        "n_layers": int(chosen["n_layers"]),
        "lifetime_days": int(chosen["lifetime_days"]),
    }


def _axis_for_object(root: Path, meta: dict[str, object]) -> pd.DataFrame:
    centers = _read_table(root / "catalog" / "layer_centers_completed.parquet")
    part = centers[
        centers["eddy3d_object_id"].eq(int(meta["eddy3d_object_id"]))
        & centers["track3d_id"].eq(int(meta["track3d_id"]))
        & centers["date"].astype(str).str[:10].eq(str(meta["date"]))
    ].copy()
    if part.empty:
        raise ValueError(f"No layer centers for {meta}")
    part = part.sort_values("depth_index").reset_index(drop=True)
    surface = part.iloc[0]
    lon0 = float(surface["longitude"])
    lat0 = float(surface["latitude"])
    part["x_km"] = np.deg2rad(_wrap_lon_delta_deg(part["longitude"].to_numpy("float64"), lon0)) * EARTH_RADIUS_M * math.cos(math.radians(lat0)) / 1000.0
    part["y_km"] = np.deg2rad(part["latitude"].to_numpy("float64") - lat0) * EARTH_RADIUS_M / 1000.0
    part["surface_lon"] = lon0
    part["surface_lat"] = lat0
    part["shape_class"] = str(meta["shape_class"])
    return part


def _select_depths(axis: pd.DataFrame, depth_count: int) -> np.ndarray:
    idx = axis["depth_index"].to_numpy(dtype=int)
    if len(idx) <= depth_count:
        return idx
    pick = np.unique(np.linspace(0, len(idx) - 1, depth_count).round().astype(int))
    return idx[pick]


def _read_slices(axis: pd.DataFrame, annual_root: Path, filter_root: Path, half_width_km: float, depth_indices: np.ndarray) -> list[SliceData]:
    wanted = _as_date(axis["date"].iloc[0])
    raw_path = annual_root / f"global_phy_{wanted.year}.nc"
    filt_path = filter_root / f"global_phy_{wanted.year}_bandpass_30_180d.nc"
    if not filt_path.exists():
        raise FileNotFoundError(filt_path)
    with Dataset(raw_path) as raw, Dataset(filt_path) as filt:
        ti = _nearest_time_index(raw, wanted)
        fti = _nearest_time_index(filt, wanted)
        lon = np.asarray(raw.variables["longitude"][:], dtype="float64")
        lat = np.asarray(raw.variables["latitude"][:], dtype="float64")
        depths = np.asarray(raw.variables["depth"][:], dtype="float64")
        lon0 = float(axis["surface_lon"].iloc[0])
        lat0 = float(axis["surface_lat"].iloc[0])
        lon_idx, lat_idx = _subset(lon, lat, lon0, lat0, half_width_km)
        ys = slice(int(lat_idx.min()), int(lat_idx.max()) + 1)
        xs = slice(int(lon_idx.min()), int(lon_idx.max()) + 1)
        lon_sub = lon[xs]
        lat_sub = lat[ys]
        x_km = np.deg2rad(_wrap_lon_delta_deg(lon_sub, lon0)) * EARTH_RADIUS_M * math.cos(math.radians(lat0)) / 1000.0
        y_km = np.deg2rad(lat_sub - lat0) * EARTH_RADIUS_M / 1000.0
        out: list[SliceData] = []
        for k in depth_indices:
            k = int(k)
            out.append(
                SliceData(
                    depth_index=k,
                    depth_m=float(depths[k]),
                    x_km=x_km,
                    y_km=y_km,
                    u=_clean(filt.variables["uo_glor"][fti, k, ys, xs]),
                    v=_clean(filt.variables["vo_glor"][fti, k, ys, xs]),
                    u_raw=_clean(raw.variables["uo_glor"][ti, k, ys, xs]),
                    v_raw=_clean(raw.variables["vo_glor"][ti, k, ys, xs]),
                )
            )
    return out


def _nearest_speed_min(sl: SliceData, center_x: float, center_y: float, radius_km: float) -> tuple[float, float, float]:
    xx, yy = np.meshgrid(sl.x_km, sl.y_km)
    speed = np.hypot(sl.u, sl.v)
    dist = np.hypot(xx - center_x, yy - center_y)
    mask = np.isfinite(speed) & (dist <= radius_km)
    if not mask.any():
        return np.nan, np.nan, np.nan
    score = np.where(mask, speed, np.nan)
    flat = int(np.nanargmin(score))
    iy, ix = np.unravel_index(flat, speed.shape)
    return float(sl.x_km[ix]), float(sl.y_km[iy]), float(speed[iy, ix])


def _plot_3d(axis: pd.DataFrame, slices: list[SliceData], output_dir: Path, stem: str, mode: str) -> dict[str, object]:
    use_raw = mode == "raw"
    p98 = [np.nanpercentile(np.hypot(sl.u_raw if use_raw else sl.u, sl.v_raw if use_raw else sl.v), 98) for sl in slices]
    vmax = max(float(np.nanmax(p98)), 1e-6)
    norm = colors.Normalize(0.0, vmax)
    cmap = cm.magma
    fig = plt.figure(figsize=(12.5, 9.5))
    ax = fig.add_subplot(111, projection="3d")
    for sl in slices:
        u = sl.u_raw if use_raw else sl.u
        v = sl.v_raw if use_raw else sl.v
        speed = np.hypot(u, v)
        xx, yy = np.meshgrid(sl.x_km, sl.y_km)
        z = -np.ones_like(xx) * sl.depth_m / 1000.0
        face = cmap(norm(speed))
        face[..., -1] = np.where(np.isfinite(speed), 0.50, 0.0)
        ax.plot_surface(xx, yy, z, facecolors=face, linewidth=0, antialiased=False, shade=False)
    ax.plot(axis["x_km"], axis["y_km"], -axis["depth_m"] / 1000.0, color="#00bcd4", linewidth=3.0, label="Hua b3 layer centers")
    ax.scatter(axis["x_km"], axis["y_km"], -axis["depth_m"] / 1000.0, c=axis["depth_m"], cmap="viridis", s=30, edgecolor="black", linewidth=0.25)
    ax.scatter([0], [0], [-float(axis["depth_m"].iloc[0]) / 1000.0], color="red", marker="+", s=220, linewidth=3.0, label="surface SLA/Hua center")
    title = f"{axis['shape_class'].iloc[0]} {axis['polarity'].iloc[0]} object {int(axis['eddy3d_object_id'].iloc[0])} track {int(axis['track3d_id'].iloc[0])}, {str(axis['date'].iloc[0])[:10]}"
    ax.set_title(f"ACC Hua b3 unaligned 3D axis + {'raw' if use_raw else '30-180d bandpass'} velocity\n{title}")
    ax.set_xlabel("east from surface center (km)")
    ax.set_ylabel("north from surface center (km)")
    ax.set_zlabel("depth (km, down)")
    ax.set_xlim(-220, 220)
    ax.set_ylim(-220, 220)
    ax.set_zlim(-3.05, 0.05)
    ax.view_init(elev=22, azim=-55)
    ax.set_box_aspect((1.15, 1.0, 0.85))
    ax.legend(loc="upper left")
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.62, pad=0.06)
    cbar.set_label("|u_raw,v_raw| (m/s)" if use_raw else "|u_30-180d,v_30-180d| (m/s)")
    path = output_dir / f"{stem}_3d_{mode}.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return {"mode": mode, "path": str(path), "vmax": vmax}


def _plot_layer_panels(axis: pd.DataFrame, slices: list[SliceData], output_dir: Path, stem: str, mode: str) -> tuple[dict[str, object], pd.DataFrame]:
    use_raw = mode == "raw"
    n = len(slices)
    cols = 4
    rows = int(math.ceil(n / cols))
    p98 = [np.nanpercentile(np.hypot(sl.u_raw if use_raw else sl.u, sl.v_raw if use_raw else sl.v), 98) for sl in slices]
    vmax = max(float(np.nanmax(p98)), 1e-6)
    norm = colors.Normalize(0.0, vmax)
    fig, axes = plt.subplots(rows, cols, figsize=(4.4 * cols, 4.25 * rows), squeeze=False)
    records = []
    axis_by_depth = axis.set_index("depth_index")
    for ax, sl in zip(axes.ravel(), slices):
        row = axis_by_depth.loc[int(sl.depth_index)]
        u = sl.u_raw if use_raw else sl.u
        v = sl.v_raw if use_raw else sl.v
        speed = np.hypot(u, v)
        xx, yy = np.meshgrid(sl.x_km, sl.y_km)
        im = ax.pcolormesh(xx, yy, speed, shading="auto", cmap="magma", norm=norm)
        step_y = max(1, speed.shape[0] // 18)
        step_x = max(1, speed.shape[1] // 18)
        ax.quiver(xx[::step_y, ::step_x], yy[::step_y, ::step_x], u[::step_y, ::step_x], v[::step_y, ::step_x], color="white", alpha=0.65, scale=4.2, width=0.003)
        cx = float(row["x_km"])
        cy = float(row["y_km"])
        rad_km = float(row["radius_m"]) / 1000.0
        min_x, min_y, min_speed = _nearest_speed_min(sl, cx, cy, radius_km=min(45.0, max(12.0, 0.7 * rad_km)))
        offset_km = float(np.hypot(min_x - cx, min_y - cy)) if np.isfinite(min_x) else np.nan
        ax.add_patch(Circle((cx, cy), rad_km, fill=False, edgecolor="#38bdf8", linewidth=1.25, alpha=0.75))
        ax.scatter([cx], [cy], marker="*", s=150, color="#00e5ff", edgecolor="black", linewidth=0.7, label="layer center")
        ax.scatter([min_x], [min_y], marker="x", s=90, color="#fde047", linewidth=2.0, label="local speed min")
        ax.scatter([0], [0], marker="+", s=90, color="red", linewidth=2.0, label="surface")
        ax.axhline(0, color="0.75", lw=0.7)
        ax.axvline(0, color="0.75", lw=0.7)
        ax.set_aspect("equal")
        ax.set_xlim(-170, 170)
        ax.set_ylim(-170, 170)
        ax.set_title(f"k={sl.depth_index}, z={sl.depth_m:.0f} m, min offset={offset_km:.1f} km", fontsize=10)
        records.append(
            {
                "eddy3d_object_id": int(row["eddy3d_object_id"]),
                "track3d_id": int(row["track3d_id"]),
                "shape_class": str(row["shape_class"]),
                "polarity": str(row["polarity"]),
                "date": str(row["date"])[:10],
                "mode": mode,
                "depth_index": int(sl.depth_index),
                "depth_m": float(sl.depth_m),
                "center_x_km": cx,
                "center_y_km": cy,
                "radius_km": rad_km,
                "nearest_speed_min_x_km": min_x,
                "nearest_speed_min_y_km": min_y,
                "nearest_speed_min_offset_km": offset_km,
                "nearest_speed_min_m_s": min_speed,
            }
        )
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles[:3], labels[:3], loc="lower center", ncol=3)
    title = f"{axis['shape_class'].iloc[0]} {axis['polarity'].iloc[0]} object {int(axis['eddy3d_object_id'].iloc[0])}, {str(axis['date'].iloc[0])[:10]}"
    fig.suptitle(f"ACC Hua b3 per-layer velocity check ({mode}): center, speed minimum, radius\n{title}", fontsize=14)
    fig.subplots_adjust(bottom=0.08, top=0.91, wspace=0.17, hspace=0.27)
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.78, pad=0.012)
    cbar.set_label("|u_raw,v_raw| (m/s)" if use_raw else "|u_30-180d,v_30-180d| (m/s)")
    path = output_dir / f"{stem}_layer_panels_{mode}.png"
    fig.savefig(path, dpi=210, bbox_inches="tight")
    plt.close(fig)
    return {"mode": mode, "path": str(path), "vmax": vmax}, pd.DataFrame(records)


def run(args: argparse.Namespace) -> None:
    root = Path(args.result_root)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    selections = [
        _choose_object(root, args.coherent_shape_dir, "coherent"),
        _choose_object(root, args.complex_shape_dir, "complex"),
    ]
    summary = []
    metrics = []
    for meta in selections:
        axis = _axis_for_object(root, meta)
        depth_indices = _select_depths(axis, args.depth_count)
        slices = _read_slices(axis, Path(args.annual_root), Path(args.filter_root), args.window_km, depth_indices)
        stem = f"{meta['shape_class']}_track{meta['track3d_id']}_object{meta['eddy3d_object_id']}_{meta['date'].replace('-', '')}"
        for mode in args.modes.split(","):
            mode = mode.strip()
            summary.append({"shape_class": meta["shape_class"], **meta, **_plot_3d(axis, slices, out, stem, mode)})
            panel_summary, df = _plot_layer_panels(axis, slices, out, stem, mode)
            summary.append({"shape_class": meta["shape_class"], **meta, **panel_summary})
            metrics.append(df)
    metrics_df = pd.concat(metrics, ignore_index=True)
    metrics_df.to_parquet(out / "layer_center_speed_min_metrics.parquet", index=False)
    metrics_df.to_csv(out / "layer_center_speed_min_metrics.csv", index=False)
    pd.DataFrame(summary).to_csv(out / "axis_velocity_plot_summary.csv", index=False)
    (out / "axis_velocity_plot_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "README_zh.md").write_text(
        "# ACC Hua b3 最新识别轴线与逐层速度场诊断\n\n"
        "- coherent 样本来自 `life30` 权威 shape 口径。\n"
        "- complex 样本来自 `life7`，因为 `life30` 中没有 complex track；它用于说明较短寿命 complex 的轴线/速度场特征。\n"
        "- 图中坐标是未对齐原始局地坐标：表层中心为原点，x 向东，y 向北；没有旋转、没有逐层重新定心。\n"
        "- `anom` 使用 30-180 天带通 `Filter` 速度场，是当前 Hua b3 识别口径；`raw` 只作为对照。\n"
        "- 每层面板中的星号是 Hua b3 层中心，黄色 x 是该中心附近的局地速度极小点，蓝色圆为该层半径。\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot latest ACC Hua b3 coherent/complex axes and layer velocity fields.")
    parser.add_argument("--result-root", default="/root/autodl-tmp/second_reslut")
    parser.add_argument("--annual-root", default="/root/autodl-fs/2020_2022_acc")
    parser.add_argument("--filter-root", default="/root/autodl-fs/2020_2022_acc/Filter")
    parser.add_argument("--output-dir", default="/root/autodl-tmp/second_reslut/axis_velocity_diagnostics_latest")
    parser.add_argument("--coherent-shape-dir", default="shape_classification_2020_2022_hua_b3_start2_life30")
    parser.add_argument("--complex-shape-dir", default="shape_classification_2020_2022_hua_b3_start2_life7")
    parser.add_argument("--window-km", type=float, default=220.0)
    parser.add_argument("--depth-count", type=int, default=12)
    parser.add_argument("--modes", default="anom,raw")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
