from __future__ import annotations

import argparse
import math
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


def _surface_meta(axis_path: Path, object_id: int) -> dict[str, float | str]:
    df = pd.read_parquet(axis_path, filters=[("eddy3d_object_id", "=", object_id), ("depth_index", "=", 0)])
    if df.empty:
        all_df = pd.read_parquet(axis_path)
        df = all_df[(all_df["eddy3d_object_id"].eq(object_id)) & (all_df["depth_index"].eq(0))]
    row = df.iloc[0]
    return {
        "date": str(row["date"])[:10],
        "surface_lon": float(row["longitude"]),
        "surface_lat": float(row["latitude"]),
        "shape_class": str(row["shape_class"]),
        "polarity": str(row["polarity"]),
        "track3d_id": int(row["track3d_id"]),
    }


def _read_velocity_slices(
    annual_root: Path,
    clim_path: Path,
    meta: dict[str, float | str],
    depth_indices: np.ndarray,
    half_width_km: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[int, float, np.ndarray, np.ndarray, np.ndarray]]]:
    wanted = _date(str(meta["date"]))
    nc_path = annual_root / f"global_phy_{wanted.year}.nc"
    with Dataset(nc_path) as ds, Dataset(clim_path) as clim:
        ti = _nearest_time_index(ds, wanted)
        di = _doy_index(clim, wanted)
        lon = np.asarray(ds.variables["longitude"][:], dtype="float64")
        lat = np.asarray(ds.variables["latitude"][:], dtype="float64")
        depths = np.asarray(ds.variables["depth"][:], dtype="float64")
        lon0 = float(meta["surface_lon"])
        lat0 = float(meta["surface_lat"])
        lon_idx, lat_idx = _subset_indices(lon, lat, lon0, lat0, half_width_km)
        lon_sub = lon[lon_idx.min() : lon_idx.max() + 1]
        lat_sub = lat[lat_idx.min() : lat_idx.max() + 1]
        x_km = np.deg2rad(_wrap_lon_delta_deg(lon_sub, lon0)) * EARTH_RADIUS_M * math.cos(math.radians(lat0)) / 1000.0
        y_km = np.deg2rad(lat_sub - lat0) * EARTH_RADIUS_M / 1000.0
        out = []
        for k in depth_indices:
            u = np.asarray(ds.variables["uo_glor"][ti, k, lat_idx.min() : lat_idx.max() + 1, lon_idx.min() : lon_idx.max() + 1], dtype="float64")
            v = np.asarray(ds.variables["vo_glor"][ti, k, lat_idx.min() : lat_idx.max() + 1, lon_idx.min() : lon_idx.max() + 1], dtype="float64")
            uc = np.asarray(clim.variables["u_clim"][di, k, lat_idx.min() : lat_idx.max() + 1, lon_idx.min() : lon_idx.max() + 1], dtype="float64")
            vc = np.asarray(clim.variables["v_clim"][di, k, lat_idx.min() : lat_idx.max() + 1, lon_idx.min() : lon_idx.max() + 1], dtype="float64")
            ua = u - uc
            va = v - vc
            speed = np.hypot(ua, va)
            out.append((int(k), float(depths[k]), ua, va, speed))
    return x_km, y_km, depths, out


def _select_depth_indices(audit: pd.DataFrame, count: int) -> np.ndarray:
    idx = np.sort(audit["depth_index"].dropna().unique().astype(int))
    if len(idx) <= count:
        return idx
    return np.unique(np.linspace(0, len(idx) - 1, count).round().astype(int))


def _plot_object(
    audit: pd.DataFrame,
    meta: dict[str, float | str],
    annual_root: Path,
    clim_path: Path,
    output_dir: Path,
    half_width_km: float,
    depth_count: int,
    view: tuple[float, float],
) -> dict[str, float | str]:
    object_id = int(audit["eddy3d_object_id"].iloc[0])
    depth_indices = _select_depth_indices(audit, depth_count)
    x_km, y_km, depths, slices = _read_velocity_slices(annual_root, clim_path, meta, depth_indices, half_width_km)
    xx, yy = np.meshgrid(x_km, y_km)
    vmax = float(np.nanpercentile([np.nanpercentile(s[-1], 97) for s in slices], 95))
    vmax = max(vmax, 1e-6)
    norm = colors.Normalize(vmin=0.0, vmax=vmax)
    cmap = cm.magma

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")
    for depth_index, depth_m, ua, va, speed in slices:
        z = -np.ones_like(xx) * depth_m / 1000.0
        face = cmap(norm(speed))
        face[..., -1] = np.where(np.isfinite(speed), 0.52, 0.0)
        ax.plot_surface(xx, yy, z, facecolors=face, linewidth=0, antialiased=False, shade=False)
        step_y = max(1, len(y_km) // 18)
        step_x = max(1, len(x_km) // 22)
        qx = xx[::step_y, ::step_x]
        qy = yy[::step_y, ::step_x]
        qz = z[::step_y, ::step_x]
        qu = ua[::step_y, ::step_x]
        qv = va[::step_y, ::step_x]
        mask = np.isfinite(qu) & np.isfinite(qv)
        ax.quiver(
            qx[mask],
            qy[mask],
            qz[mask],
            qu[mask],
            qv[mask],
            np.zeros_like(qu[mask]),
            length=0.018,
            normalize=True,
            color="white",
            alpha=0.20,
            linewidth=0.35,
        )

    audit = audit.sort_values("depth_index")
    z_axis = -audit["depth_m"].to_numpy(dtype="float64") / 1000.0
    ax.plot(
        audit["hua_x_km"].to_numpy(dtype="float64"),
        audit["hua_y_km"].to_numpy(dtype="float64"),
        z_axis,
        color="#facc15",
        linewidth=3.0,
        label="Hua hybrid centers only",
    )
    passed = audit["hua_passed"].astype(bool).to_numpy()
    ax.scatter(
        audit.loc[passed, "hua_x_km"],
        audit.loc[passed, "hua_y_km"],
        -audit.loc[passed, "depth_m"] / 1000.0,
        marker="*",
        s=55,
        color="#22c55e",
        edgecolor="black",
        linewidth=0.4,
        label="Hua pass",
    )
    ax.scatter(
        audit.loc[~passed, "hua_x_km"],
        audit.loc[~passed, "hua_y_km"],
        -audit.loc[~passed, "depth_m"] / 1000.0,
        marker="*",
        s=42,
        color="#ef4444",
        edgecolor="black",
        linewidth=0.35,
        label="Hua failed",
    )
    ax.scatter([0], [0], [0], marker="+", s=160, linewidth=3.0, color="#06b6d4", label="SLA surface seed")

    ax.set_xlim(-half_width_km, half_width_km)
    ax.set_ylim(-half_width_km, half_width_km)
    ax.set_zlim(-3.05, 0.08)
    ax.set_xlabel("east x from SLA seed (km)")
    ax.set_ylabel("north y from SLA seed (km)")
    ax.set_zlabel("depth (km, down)")
    ax.view_init(elev=view[0], azim=view[1])
    ax.set_box_aspect((1.2, 1.0, 0.85))
    ax.legend(loc="upper left", fontsize=8)
    ax.set_title(
        f"Hua-only unaligned velocity stack\n"
        f"{meta['shape_class']} id {object_id}, {meta['polarity']}, {meta['date']}",
        fontsize=14,
    )
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, ax=ax, shrink=0.62, pad=0.08)
    cbar.set_label("|u',v'| (m/s)")
    fig.text(
        0.5,
        0.02,
        "Fixed original frame: origin is the surface SLA center; no rotation, no layer recentering. "
        "Only Hua hybrid centers are overlaid.",
        ha="center",
        fontsize=9,
    )
    stem = output_dir / f"hua_only_unaligned_velocity_stack_id_{object_id}"
    fig.savefig(stem.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    return {
        "eddy3d_object_id": object_id,
        "shape_class": str(meta["shape_class"]),
        "polarity": str(meta["polarity"]),
        "date": str(meta["date"]),
        "n_layers": int(len(audit)),
        "n_hua_pass": int(passed.sum()),
        "median_abs_hua_x_km": float(np.nanmedian(np.abs(audit["hua_x_km"]))),
        "median_abs_hua_y_km": float(np.nanmedian(np.abs(audit["hua_y_km"]))),
        "max_depth_m": float(np.nanmax(audit["depth_m"])),
        "speed_vmax_m_s": vmax,
        "png": str(stem.with_suffix(".png")),
    }


def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    audit = pd.read_csv(args.audit_csv)
    object_ids = [int(x) for x in args.object_ids.split(",")]
    rows = []
    for object_id in object_ids:
        part = audit[audit["eddy3d_object_id"].eq(object_id)].copy()
        if part.empty:
            raise ValueError(f"No audit rows for object {object_id}")
        meta = _surface_meta(Path(args.axis_path), object_id)
        rows.append(
            _plot_object(
                part,
                meta,
                Path(args.annual_root),
                Path(args.climatology),
                output_dir,
                args.window_km,
                args.depth_count,
                (args.elev, args.azim),
            )
        )
    pd.DataFrame(rows).to_csv(output_dir / "hua_only_unaligned_velocity_stack_summary.csv", index=False)
    (output_dir / "README_zh.md").write_text(
        "# Hua-only 未对齐速度场堆叠图\n\n"
        "这些图只叠加 Hua hybrid centers，不叠加 current centers、speed-min centers 或 production recommended centers。\n\n"
        "坐标系固定为原始空间：表层 SLA 中心为原点，x 为向东距离，y 为向北距离；没有旋转，没有逐层重新定心。因此图中保留了 Hua 轴线相对于真实速度场的倾斜和跳变。\n\n"
        "背景颜色是去气候态后的扰动速度大小 `|u',v'|`，箭头是同一扰动速度方向。绿色星号表示 Hua 圆周检验通过，红色星号表示 Hua 候选未通过源码式圆周检验。\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Hua-only unaligned 3D velocity stacks.")
    parser.add_argument("--audit-csv", default="/root/autodl-fs/2020_2022_acc/result/hua_hybrid_center_experiment_source_aligned/hua_hybrid_center_audit_all.csv")
    parser.add_argument("--axis-path", default="/root/autodl-fs/2020_2022_acc/result/representative_vortex/axis/rotated_points.parquet")
    parser.add_argument("--annual-root", default="/root/autodl-fs/2020_2022_acc")
    parser.add_argument("--climatology", default="/root/autodl-fs/2020_2022_acc/result/climatology/cmems_doy_climatology_2020_2022_31d.nc")
    parser.add_argument("--output-dir", default="/root/autodl-fs/2020_2022_acc/result/hua_hybrid_center_experiment_source_aligned/hua_only_velocity_stack")
    parser.add_argument("--object-ids", default="1120461,1076249")
    parser.add_argument("--window-km", type=float, default=230.0)
    parser.add_argument("--depth-count", type=int, default=10)
    parser.add_argument("--elev", type=float, default=22.0)
    parser.add_argument("--azim", type=float, default=-55.0)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
