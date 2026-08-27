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
from netCDF4 import Dataset, num2date
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


EARTH_RADIUS_M = 6_371_000.0


@dataclass
class VelocitySlice:
    depth_index: int
    depth_m: float
    x_km: np.ndarray
    y_km: np.ndarray
    u_raw: np.ndarray
    v_raw: np.ndarray
    u_anom: np.ndarray
    v_anom: np.ndarray


def _date(value: str) -> date:
    return datetime.strptime(str(value)[:10], "%Y-%m-%d").date()


def _wrap_lon_delta_deg(lon: np.ndarray, lon0: float) -> np.ndarray:
    return (lon - lon0 + 180.0) % 360.0 - 180.0


def _nearest_time_index(ds: Dataset, wanted: date) -> int:
    tvar = ds.variables["time"]
    times = num2date(tvar[:], units=tvar.units, calendar=getattr(tvar, "calendar", "standard"))
    keys = [date(int(t.year), int(t.month), int(t.day)) for t in times]
    return keys.index(wanted)


def _doy_index(clim: Dataset, wanted: date) -> int:
    doy = wanted.timetuple().tm_yday
    vals = np.asarray(clim.variables["doy"][:], dtype=int)
    where = np.where(vals == doy)[0]
    return int(where[0]) if len(where) else doy - 1


def _subset_indices(lon: np.ndarray, lat: np.ndarray, lon0: float, lat0: float, half_width_km: float) -> tuple[np.ndarray, np.ndarray]:
    dlon_m = np.deg2rad(_wrap_lon_delta_deg(lon, lon0)) * EARTH_RADIUS_M * math.cos(math.radians(lat0))
    dlat_m = np.deg2rad(lat - lat0) * EARTH_RADIUS_M
    lon_idx = np.where(np.abs(dlon_m) <= half_width_km * 1000.0)[0]
    lat_idx = np.where(np.abs(dlat_m) <= half_width_km * 1000.0)[0]
    if len(lon_idx) < 5 or len(lat_idx) < 5:
        raise ValueError("Velocity subset too small")
    return lon_idx, lat_idx


def _load_slices(row: pd.Series, annual_root: Path, climatology_path: Path, half_width_km: float, filter_root: Path | None = None) -> list[VelocitySlice]:
    wanted = _date(str(row["date"]))
    nc_path = annual_root / f"global_phy_{wanted.year}.nc"
    filter_path = filter_root / f"global_phy_{wanted.year}_bandpass_30_180d.nc" if filter_root else None
    filter_ds = Dataset(filter_path) if filter_path and filter_path.exists() else None
    with Dataset(nc_path) as ds, Dataset(climatology_path) as clim:
        ti = _nearest_time_index(ds, wanted)
        fti = _nearest_time_index(filter_ds, wanted) if filter_ds is not None else None
        di = _doy_index(clim, wanted)
        lon = np.asarray(ds.variables["longitude"][:], dtype="float64")
        lat = np.asarray(ds.variables["latitude"][:], dtype="float64")
        depths = np.asarray(ds.variables["depth"][:], dtype="float64")
        lon0 = float(row["surface_longitude"])
        lat0 = float(row["surface_latitude"])
        lon_idx, lat_idx = _subset_indices(lon, lat, lon0, lat0, half_width_km)
        lon_sub = lon[lon_idx]
        lat_sub = lat[lat_idx]
        x_km = np.deg2rad(_wrap_lon_delta_deg(lon_sub, lon0)) * EARTH_RADIUS_M * math.cos(math.radians(lat0)) / 1000.0
        y_km = np.deg2rad(lat_sub - lat0) * EARTH_RADIUS_M / 1000.0
        slices: list[VelocitySlice] = []
        try:
            for depth_index in sorted(row["_depth_indices"]):
                ys = slice(int(lat_idx.min()), int(lat_idx.max()) + 1)
                xs = slice(int(lon_idx.min()), int(lon_idx.max()) + 1)
                u_raw = _clean_field(ds.variables["uo_glor"][ti, depth_index, ys, xs])
                v_raw = _clean_field(ds.variables["vo_glor"][ti, depth_index, ys, xs])
                if filter_ds is not None:
                    u_anom = _clean_field(filter_ds.variables["uo_glor"][fti, depth_index, ys, xs])
                    v_anom = _clean_field(filter_ds.variables["vo_glor"][fti, depth_index, ys, xs])
                else:
                    u_clim = _clean_field(clim.variables["u_clim"][di, depth_index, ys, xs])
                    v_clim = _clean_field(clim.variables["v_clim"][di, depth_index, ys, xs])
                    u_anom = u_raw - u_clim
                    v_anom = v_raw - v_clim
                slices.append(
                    VelocitySlice(
                        depth_index=int(depth_index),
                        depth_m=float(depths[depth_index]),
                        x_km=x_km,
                        y_km=y_km,
                        u_raw=u_raw,
                        v_raw=v_raw,
                        u_anom=u_anom,
                        v_anom=v_anom,
                    )
                )
        finally:
            if filter_ds is not None:
                filter_ds.close()
    return slices


def _clean_field(value: np.ndarray) -> np.ndarray:
    arr = np.ma.filled(value, np.nan).astype("float64", copy=False)
    arr[np.abs(arr) > 1.0e10] = np.nan
    return arr


def _select_depth_indices(indices: np.ndarray, count: int) -> np.ndarray:
    if len(indices) <= count:
        return indices
    pick = np.unique(np.linspace(0, len(indices) - 1, count).round().astype(int))
    return indices[pick]


def _plot_stack(
    audit: pd.DataFrame,
    slices: list[VelocitySlice],
    output_dir: Path,
    *,
    mode: str,
    depth_count: int,
    object_id: int,
    view_suffix: str,
    elev: float,
    azim: float,
) -> dict[str, float | str | int]:
    use_cols = ("u_raw", "v_raw") if mode == "raw" else ("u_anom", "v_anom")
    selected = set(_select_depth_indices(audit["depth_index"].to_numpy(dtype=int), depth_count))
    slices_sel = [sl for sl in slices if sl.depth_index in selected]
    if not slices_sel:
        raise ValueError("No selected slices")

    p98 = []
    for sl in slices_sel:
        u = getattr(sl, use_cols[0])
        v = getattr(sl, use_cols[1])
        p98.append(float(np.nanpercentile(np.hypot(u, v), 98)))
    vmax = max(max(p98), 1e-6)
    norm = colors.Normalize(vmin=0.0, vmax=vmax)
    cmap = cm.magma

    fig = plt.figure(figsize=(13.5, 10))
    ax = fig.add_subplot(111, projection="3d")
    for sl in slices_sel:
        u = getattr(sl, use_cols[0])
        v = getattr(sl, use_cols[1])
        speed = np.hypot(u, v)
        xx, yy = np.meshgrid(sl.x_km, sl.y_km)
        z = -np.ones_like(xx) * sl.depth_m / 1000.0
        face = cmap(norm(speed))
        face[..., -1] = np.where(np.isfinite(speed), 0.58, 0.0)
        ax.plot_surface(xx, yy, z, facecolors=face, linewidth=0, antialiased=False, shade=False)
        step_y = max(1, speed.shape[0] // 12)
        step_x = max(1, speed.shape[1] // 22)
        xs = xx[::step_y, ::step_x]
        ys = yy[::step_y, ::step_x]
        zs = z[::step_y, ::step_x]
        uq = u[::step_y, ::step_x]
        vq = v[::step_y, ::step_x]
        ok = np.isfinite(uq) & np.isfinite(vq)
        ax.quiver(xs[ok], ys[ok], zs[ok], uq[ok], vq[ok], np.zeros_like(uq[ok]), length=10.0, normalize=True, color="white", alpha=0.38, linewidth=0.3)

    axis = audit.sort_values("depth_index")
    z_axis = -axis["depth_m"].to_numpy(dtype="float64") / 1000.0
    ax.plot(
        axis["hua_x_km"].to_numpy(dtype="float64"),
        axis["hua_y_km"].to_numpy(dtype="float64"),
        z_axis,
        color="#facc15",
        linewidth=3.0,
        label="Hua candidate axis",
    )
    passed = axis["hua_passed"].astype(bool).to_numpy()
    ax.scatter(axis.loc[passed, "hua_x_km"], axis.loc[passed, "hua_y_km"], z_axis[passed], color="#22c55e", s=34, marker="o", label="Hua passed")
    ax.scatter(axis.loc[~passed, "hua_x_km"], axis.loc[~passed, "hua_y_km"], z_axis[~passed], color="#ef4444", s=28, marker="x", label="Hua failed")
    ax.scatter([0], [0], [z_axis[0]], color="red", marker="+", s=180, linewidth=3.2, label="SLA seed")

    first = axis.iloc[0]
    ax.set_title(f"Hua-only unaligned {mode} velocity stack\n{first['shape_class']} id {object_id}, {first['polarity']}, {first['date']}")
    ax.set_xlabel("east x from SLA seed (km)")
    ax.set_ylabel("north y from SLA seed (km)")
    ax.set_zlabel("depth (km, down)")
    ax.set_xlim(-240, 240)
    ax.set_ylim(-240, 240)
    ax.set_zlim(-3.1, 0.05)
    ax.view_init(elev=elev, azim=azim)
    ax.set_box_aspect((1.05, 1.0, 0.9))
    ax.legend(loc="upper left", fontsize=8)
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.64, pad=0.05)
    cbar.set_label("|u,v| raw (m/s)" if mode == "raw" else "|u',v'| 30-180d bandpass (m/s)")
    fig.text(0.5, 0.025, "Fixed original frame: origin is SLA surface seed; no rotation, no layer recentering. Only Hua candidate centers are shown.", ha="center", fontsize=10)
    out_base = output_dir / f"hua_only_{mode}_velocity_stack_id_{object_id}{view_suffix}"
    fig.savefig(out_base.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return {
        "eddy3d_object_id": int(object_id),
        "mode": mode,
        "view_suffix": view_suffix,
        "depth_layers_plotted": int(len(slices_sel)),
        "speed_p98_m_s": float(vmax),
        "hua_pass_layers": int(axis["hua_passed"].sum()),
        "n_layers": int(len(axis)),
        "output_png": str(out_base.with_suffix(".png")),
    }


def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    audit = pd.read_csv(args.audit_csv)
    axis = pd.read_parquet(args.axis_path, columns=["eddy3d_object_id", "depth_index", "longitude", "latitude"])
    object_ids = [int(x) for x in args.object_ids.split(",")]
    summaries = []
    for object_id in object_ids:
        part = audit[audit["eddy3d_object_id"].eq(object_id)].copy()
        if part.empty:
            raise ValueError(f"No audit rows for object {object_id}")
        surface = axis[(axis["eddy3d_object_id"].eq(object_id)) & (axis["depth_index"].eq(0))]
        if surface.empty:
            raise ValueError(f"No surface longitude/latitude for object {object_id}")
        part["_depth_indices"] = [part["depth_index"].astype(int).tolist()] * len(part)
        load_row = part.iloc[0].copy()
        load_row["surface_longitude"] = float(surface.iloc[0]["longitude"])
        load_row["surface_latitude"] = float(surface.iloc[0]["latitude"])
        filter_root = Path(args.filter_root) if args.filter_root else None
        slices = _load_slices(load_row, Path(args.annual_root), Path(args.climatology), args.window_km, filter_root=filter_root)
        for mode in ("raw", "anom"):
            summaries.append(_plot_stack(part, slices, output_dir, mode=mode, depth_count=args.depth_count, object_id=object_id, view_suffix="", elev=24, azim=-58))
            summaries.append(_plot_stack(part, slices, output_dir, mode=mode, depth_count=args.depth_count, object_id=object_id, view_suffix="_altview", elev=18, azim=35))
    pd.DataFrame(summaries).to_csv(output_dir / "hua_only_velocity_stack_summary.csv", index=False)
    (output_dir / "hua_only_velocity_stack_summary.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "README_zh.md").write_text(
        "# Hua-only 原始坐标速度场堆叠\n\n"
        "这些图只使用 Hua candidate centers/axis，不显示 current center、speed-min center 或 production recommended center。\n\n"
        "- `raw`: 年度 NetCDF 原始 `uo_glor/vo_glor`。\n"
        "- `anom`: `u'=u_{30-180d}`, `v'=v_{30-180d}`，来自 `/root/autodl-fs/2020_2022_acc/Filter`。\n"
        "- 坐标固定为表层 SLA seed 的 east/north 原始框架；没有旋转，没有逐层重定心，因此保留了 Hua 轴线的真实倾斜/跳变。\n"
        "- 绿色圆点表示 Hua 圆周检验通过层，红色叉表示失败层。\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Hua-only unaligned raw/anomaly velocity stacks for ACC objects.")
    parser.add_argument("--audit-csv", default="/root/autodl-fs/2020_2022_acc/result/hua_hybrid_center_experiment_source_aligned/hua_hybrid_center_audit_all.csv")
    parser.add_argument("--axis-path", default="/root/autodl-fs/2020_2022_acc/result/representative_vortex/axis/rotated_points.parquet")
    parser.add_argument("--annual-root", default="/root/autodl-fs/2020_2022_acc")
    parser.add_argument("--climatology", default="/root/autodl-fs/2020_2022_acc/result/climatology/cmems_doy_climatology_2020_2022_31d.nc")
    parser.add_argument("--filter-root", default="/root/autodl-fs/2020_2022_acc/Filter")
    parser.add_argument("--output-dir", default="/root/autodl-fs/2020_2022_acc/result/hua_hybrid_center_experiment_source_aligned/velocity_stacks_hua_only")
    parser.add_argument("--object-ids", default="1120461,1076249")
    parser.add_argument("--window-km", type=float, default=230.0)
    parser.add_argument("--depth-count", type=int, default=10)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
