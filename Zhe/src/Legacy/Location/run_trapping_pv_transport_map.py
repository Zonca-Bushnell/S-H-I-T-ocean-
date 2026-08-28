from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SECONDS_PER_DAY = 86400.0


def _wrap_lon_360(lon: pd.Series | np.ndarray) -> np.ndarray:
    arr = np.asarray(lon, dtype="float64")
    out = arr % 360.0
    out[np.isclose(out, 360.0)] = 0.0
    return out


def _shape_tracks_path(results_root: Path) -> Path:
    candidates = sorted(results_root.glob("shape_classification*/shape_tracks.parquet"))
    if not candidates:
        raise FileNotFoundError(f"No shape_classification*/shape_tracks.parquet under {results_root}")
    return candidates[0]


def _load_all_axis_objects(rv_root: Path) -> pd.DataFrame:
    results_root = rv_root.parent
    axis_path = rv_root / "axis" / "object_diagnostics.parquet"
    radii_path = results_root / "catalog" / "vertical_objects.parquet"
    if not axis_path.exists():
        raise FileNotFoundError(axis_path)
    if not radii_path.exists():
        raise FileNotFoundError(radii_path)
    objects = pd.read_parquet(axis_path)
    tracks = pd.read_parquet(_shape_tracks_path(results_root))
    tracks["track3d_id"] = tracks["track3d_id"].astype("int64")
    tracks["start_date"] = pd.to_datetime(tracks["start_date"])
    tracks["end_date"] = pd.to_datetime(tracks["end_date"])
    objects["track3d_id"] = objects["track3d_id"].astype("int64")
    objects["eddy3d_object_id"] = objects["eddy3d_object_id"].astype("int64")
    objects = objects.merge(
        tracks[["track3d_id", "shape_class", "polarity", "start_date", "end_date", "lifetime_days"]],
        on=["track3d_id", "shape_class", "polarity"],
        how="inner",
    )
    radii = pd.read_parquet(radii_path, columns=["eddy3d_object_id", "mean_radius_m"])
    radii["eddy3d_object_id"] = radii["eddy3d_object_id"].astype("int64")
    objects = objects.merge(radii, on="eddy3d_object_id", how="left")
    objects = objects[np.isfinite(objects["mean_radius_m"]) & (objects["mean_radius_m"] > 0)].copy()
    objects["date"] = pd.to_datetime(objects["date"])
    objects["life_day"] = (objects["date"] - objects["start_date"]).dt.days.astype("float64")
    denom = np.maximum(objects["lifetime_days"].to_numpy(dtype="float64") - 1.0, 1.0)
    objects["life_phase"] = np.clip(objects["life_day"].to_numpy(dtype="float64") / denom, 0.0, 1.0)
    objects["surface_lon_plot"] = _wrap_lon_360(objects["surface_lon"])
    return objects


def _load_points_for_objects(rv_root: Path, object_ids: set[int]) -> pd.DataFrame:
    path = rv_root / "axis" / "rotated_points.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    cols = [
        "eddy3d_object_id",
        "track3d_id",
        "date",
        "polarity",
        "depth_index",
        "depth_m",
        "y_m",
    ]
    points = pd.read_parquet(path, columns=cols)
    points["eddy3d_object_id"] = points["eddy3d_object_id"].astype("int64")
    points = points[points["eddy3d_object_id"].isin(object_ids)].copy()
    points["track3d_id"] = points["track3d_id"].astype("int64")
    points["date"] = pd.to_datetime(points["date"])
    points["depth_index"] = points["depth_index"].astype("int16")
    points["depth_m"] = points["depth_m"].astype("float64")
    points["y_m"] = points["y_m"].astype("float64")
    return points


def _centerline_vdev_y(points: pd.DataFrame) -> pd.DataFrame:
    p = points.sort_values(["track3d_id", "depth_index", "date"]).copy()
    group_cols = ["track3d_id", "depth_index"]
    p["dt_day"] = p.groupby(group_cols, sort=False)["date"].diff().dt.total_seconds() / SECONDS_PER_DAY
    p["dy_dt_m_day"] = p.groupby(group_cols, sort=False)["y_m"].diff() / p["dt_day"]
    p["vdev_y_m_s"] = p["dy_dt_m_day"] / SECONDS_PER_DAY
    return p[["eddy3d_object_id", "track3d_id", "date", "polarity", "depth_index", "depth_m", "vdev_y_m_s"]].copy()


def _depth_thickness(depth_by_index: pd.DataFrame) -> pd.DataFrame:
    d = depth_by_index.sort_values("depth_index").copy()
    depth = d["depth_m"].to_numpy(dtype="float64")
    if len(depth) == 1:
        dz = np.asarray([1.0], dtype="float64")
    else:
        edges = np.empty(len(depth) + 1, dtype="float64")
        edges[1:-1] = 0.5 * (depth[:-1] + depth[1:])
        edges[0] = max(0.0, depth[0] - 0.5 * (depth[1] - depth[0]))
        edges[-1] = depth[-1] + 0.5 * (depth[-1] - depth[-2])
        dz = np.diff(edges)
    d["dz_m"] = dz
    return d[["depth_index", "depth_m", "dz_m"]]


def _nearest_tau(values: pd.Series, tau_grid: np.ndarray) -> np.ndarray:
    arr = values.to_numpy(dtype="float64")
    idx = np.nanargmin(np.abs(arr[:, None] - tau_grid[None, :]), axis=1)
    return tau_grid[idx]


def _load_q_area_integrals(rv_root: Path, r_core: float) -> tuple[pd.DataFrame, np.ndarray]:
    frames: list[pd.DataFrame] = []
    tau_values: list[float] = []
    for polarity in ("cyclonic", "anticyclonic"):
        path = rv_root / polarity / "ep_flux_terms" / "continuous_ep_flux_profiles.parquet"
        if not path.exists():
            raise FileNotFoundError(path)
        prof = pd.read_parquet(
            path,
            columns=["polarity", "tau_center", "depth_index", "depth_m", "r_over_R", "q_mean", "count"],
        )
        prof = prof[(prof["polarity"].eq(polarity)) & (prof["r_over_R"].le(r_core))].copy()
        tau_values.extend(prof["tau_center"].dropna().unique().tolist())
        radial = np.sort(prof["r_over_R"].dropna().unique().astype("float64"))
        if len(radial) < 2:
            raise ValueError(f"Need at least two radial bins in {path}")
        dr = float(np.nanmedian(np.diff(radial)))
        prof["area_weight_unit_R2"] = 2.0 * math.pi * prof["r_over_R"].astype("float64") * dr
        prof["q_area_unit_R2_num"] = prof["q_mean"].astype("float64") * prof["area_weight_unit_R2"]
        grouped = (
            prof.groupby(["polarity", "tau_center", "depth_index", "depth_m"], observed=True, sort=True)
            .agg(q_area_unit_R2=("q_area_unit_R2_num", "sum"), q_weight_area=("area_weight_unit_R2", "sum"))
            .reset_index()
        )
        frames.append(grouped)
    if not frames:
        raise RuntimeError("No q profiles loaded")
    out = pd.concat(frames, ignore_index=True)
    tau_grid = np.sort(np.unique(np.asarray(tau_values, dtype="float64")))
    return out, tau_grid


def _object_transport(
    objects: pd.DataFrame,
    kin: pd.DataFrame,
    q_area: pd.DataFrame,
    tau_grid: np.ndarray,
    upper_depth_m: float,
) -> pd.DataFrame:
    meta_cols = [
        "eddy3d_object_id",
        "date",
        "track3d_id",
        "shape_class",
        "polarity",
        "life_phase",
        "mean_radius_m",
        "surface_lon",
        "surface_lon_plot",
        "surface_lat",
    ]
    meta = objects[meta_cols].copy()
    meta["tau_center"] = _nearest_tau(meta["life_phase"], tau_grid)
    layer = kin.merge(meta, on=["eddy3d_object_id", "track3d_id", "date", "polarity"], how="inner")
    layer = layer.merge(q_area, on=["polarity", "tau_center", "depth_index", "depth_m"], how="left")
    dz = _depth_thickness(layer[["depth_index", "depth_m"]].drop_duplicates())
    layer = layer.merge(dz, on=["depth_index", "depth_m"], how="left")
    layer["finite_layer"] = np.isfinite(layer["vdev_y_m_s"]) & np.isfinite(layer["q_area_unit_R2"])
    radius2 = layer["mean_radius_m"].astype("float64") ** 2
    layer["T_layer"] = layer["vdev_y_m_s"].astype("float64") * layer["q_area_unit_R2"].astype("float64") * radius2 * layer["dz_m"].astype("float64")
    layer.loc[~layer["finite_layer"], "T_layer"] = np.nan
    layer["T_upper_layer"] = np.where(layer["depth_m"].le(upper_depth_m), layer["T_layer"], np.nan)
    layer["T_deep_layer"] = np.where(layer["depth_m"].gt(upper_depth_m), layer["T_layer"], np.nan)
    grouped = (
        layer.groupby(
            [
                "eddy3d_object_id",
                "date",
                "track3d_id",
                "shape_class",
                "polarity",
                "life_phase",
                "tau_center",
                "mean_radius_m",
                "surface_lon",
                "surface_lon_plot",
                "surface_lat",
            ],
            observed=True,
            sort=False,
        )
        .agg(
            T_trap_core=("T_layer", "sum"),
            T_trap_upper=("T_upper_layer", "sum"),
            T_trap_deep=("T_deep_layer", "sum"),
            n_layers=("finite_layer", "sum"),
        )
        .reset_index()
    )
    grouped = meta.merge(grouped, on=[
        "eddy3d_object_id",
        "date",
        "track3d_id",
        "shape_class",
        "polarity",
        "life_phase",
        "tau_center",
        "mean_radius_m",
        "surface_lon",
        "surface_lon_plot",
        "surface_lat",
    ], how="left")
    for col in ["T_trap_core", "T_trap_upper", "T_trap_deep"]:
        grouped[col] = grouped[col].fillna(0.0)
    grouped["n_layers"] = grouped["n_layers"].fillna(0).astype("int64")
    return grouped


def _grid_transport(objects: pd.DataFrame, lon_bin_deg: float, lat_bin_deg: float) -> pd.DataFrame:
    out = objects.copy()
    out["lon_bin"] = np.floor(out["surface_lon_plot"] / lon_bin_deg).astype("int64")
    out.loc[out["lon_bin"].ge(int(round(360.0 / lon_bin_deg))), "lon_bin"] = int(round(360.0 / lon_bin_deg)) - 1
    out["lat_bin"] = np.floor((out["surface_lat"] + 90.0) / lat_bin_deg).astype("int64")
    out["lon_center"] = (out["lon_bin"] + 0.5) * lon_bin_deg
    out["lat_center"] = -90.0 + (out["lat_bin"] + 0.5) * lat_bin_deg
    out["T_abs"] = out["T_trap_core"].abs()
    grouped = (
        out.groupby(["lon_bin", "lat_bin", "lon_center", "lat_center"], observed=True, sort=True)
        .agg(
            T_net=("T_trap_core", "sum"),
            T_abs=("T_abs", "sum"),
            T_cyclonic=("T_trap_core", lambda s: s[out.loc[s.index, "polarity"].eq("cyclonic")].sum()),
            T_anticyclonic=("T_trap_core", lambda s: s[out.loc[s.index, "polarity"].eq("anticyclonic")].sum()),
            n_objects=("eddy3d_object_id", "nunique"),
            n_tracks=("track3d_id", "nunique"),
        )
        .reset_index()
    )
    return grouped


def _grid_to_matrix(grid: pd.DataFrame, value_col: str, lon_bin_deg: float, lat_bin_deg: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lon_edges = np.arange(0.0, 360.0 + lon_bin_deg, lon_bin_deg)
    lat_edges = np.arange(-65.0, -45.0 + lat_bin_deg, lat_bin_deg)
    matrix = np.full((len(lat_edges) - 1, len(lon_edges) - 1), np.nan, dtype="float64")
    for row in grid.itertuples(index=False):
        lon_idx = int(np.floor(row.lon_center / lon_bin_deg))
        lat_idx = int(np.floor((row.lat_center + 65.0) / lat_bin_deg))
        if 0 <= lon_idx < matrix.shape[1] and 0 <= lat_idx < matrix.shape[0]:
            matrix[lat_idx, lon_idx] = getattr(row, value_col)
    return lon_edges, lat_edges, matrix


def _plot_map(objects: pd.DataFrame, grid: pd.DataFrame, output_dir: Path, lon_bin_deg: float, lat_bin_deg: float) -> None:
    fig, ax = plt.subplots(figsize=(16, 7.4))
    fig.subplots_adjust(left=0.06, right=0.89, bottom=0.17, top=0.92)
    lon_edges, lat_edges, mat = _grid_to_matrix(grid, "T_net", lon_bin_deg, lat_bin_deg)
    finite = np.isfinite(mat)
    vmax = float(np.nanpercentile(np.abs(mat[finite]), 98)) if np.any(finite) else 1.0
    vmax = max(vmax, 1e-30)
    mesh = ax.pcolormesh(lon_edges, lat_edges, mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="auto")
    cb = fig.colorbar(mesh, ax=ax, pad=0.01)
    cb.set_label("Net trapping PV transport (area-depth integrated, signed)")
    colors = {"cyclonic": "#2b6cb0", "anticyclonic": "#c2410c"}
    for polarity, part in objects.groupby("polarity", sort=True):
        sample = part
        if len(sample) > 120000:
            sample = sample.sample(120000, random_state=13)
        ax.scatter(
            sample["surface_lon_plot"],
            sample["surface_lat"],
            s=1.0,
            alpha=0.12,
            c=colors.get(polarity, "0.3"),
            linewidths=0,
            label=f"{polarity} object positions",
        )
    if not grid.empty:
        threshold = float(np.nanpercentile(grid["T_abs"].to_numpy(dtype="float64"), 90))
        hot = grid[grid["T_abs"].ge(threshold)]
        ax.scatter(hot["lon_center"], hot["lat_center"], s=24, facecolors="none", edgecolors="black", linewidths=0.8, label="top 10% |transport| cells")
    ax.set_xlim(0, 360)
    ax.set_ylim(-65, -45)
    ax.set_xlabel("Longitude (0-360)")
    ax.set_ylabel("Latitude")
    ax.set_title("ACC all-shape object-level trapping PV redistribution")
    ax.grid(True, color="0.85", linewidth=0.5)
    ax.legend(loc="lower left", frameon=True, fontsize=8)

    inset = ax.inset_axes([0.73, 0.07, 0.25, 0.26])
    stats = []
    for polarity, part in objects.groupby("polarity", sort=True):
        stats.append((polarity, float(part["T_trap_core"].sum()), float(part["T_trap_core"].abs().sum())))
    labels = [s[0].replace("anticyclonic", "anti").replace("cyclonic", "cycl") for s in stats]
    x = np.arange(len(stats))
    signed = [s[1] for s in stats]
    absolute = [s[2] for s in stats]
    scale = max(max(abs(v) for v in signed), max(absolute), 1.0)
    inset.bar(x - 0.18, np.asarray(signed) / scale, width=0.36, color="#64748b", label="signed")
    inset.bar(x + 0.18, np.asarray(absolute) / scale, width=0.36, color="#f59e0b", label="absolute")
    inset.axhline(0, color="black", linewidth=0.6)
    inset.set_xticks(x, labels, rotation=0)
    inset.set_title("polarity totals / max scale", fontsize=8)
    inset.tick_params(labelsize=7)
    inset.legend(fontsize=6, frameon=False)

    caption = (
        "Sign shows redistribution by trapping transport, not net PV creation. "
        "A small signed/absolute ratio means strong internal redistribution with weak domain-integrated imbalance."
    )
    fig.text(0.06, 0.045, caption, fontsize=9)
    fig.savefig(output_dir / "acc_trapping_pv_redistribution_map.png", dpi=220)
    fig.savefig(output_dir / "acc_trapping_pv_redistribution_map.pdf")
    plt.close(fig)


def _write_report(objects: pd.DataFrame, grid: pd.DataFrame, output_dir: Path, r_core: float, upper_depth_m: float) -> None:
    signed_sum = float(objects["T_trap_core"].sum())
    abs_sum = float(objects["T_trap_core"].abs().sum())
    ratio = signed_sum / abs_sum if abs_sum > 0 else float("nan")
    rows = []
    for polarity, part in objects.groupby("polarity", sort=True):
        rows.append(
            {
                "polarity": polarity,
                "n_objects": int(part["eddy3d_object_id"].nunique()),
                "n_tracks": int(part["track3d_id"].nunique()),
                "signed_sum": float(part["T_trap_core"].sum()),
                "absolute_sum": float(part["T_trap_core"].abs().sum()),
                "signed_to_absolute_ratio": float(part["T_trap_core"].sum() / part["T_trap_core"].abs().sum()),
                "median": float(part["T_trap_core"].median()),
                "p10": float(part["T_trap_core"].quantile(0.10)),
                "p90": float(part["T_trap_core"].quantile(0.90)),
            }
        )
    stats = pd.DataFrame(rows)
    stats.to_csv(output_dir / "polarity_transport_totals.csv", index=False)
    hot = grid.sort_values("T_abs", ascending=False).head(10).copy()
    hot.to_csv(output_dir / "top_trapping_transport_hotspots.csv", index=False)

    lines = [
        "# ACC trapping PV 重分布总图解释\n\n",
        f"- 输入对象：all-shape representative axis object-days，共 `{objects['eddy3d_object_id'].nunique():,}` 个 object-day，`{objects['track3d_id'].nunique():,}` 条 track。\n",
        f"- 核心积分半径：`r/R <= {r_core:g}`；upper/deep 分界：`{upper_depth_m:g} m`。\n",
        f"- 全域 signed sum：`{signed_sum:.6e}`；absolute sum：`{abs_sum:.6e}`；signed/absolute：`{ratio:.4f}`。\n\n",
        "## 物理解释\n\n",
        "这张图表示 trapping transport 对 ACC 采样域内 PV 的**空间重分布**，不是大洋总 PV 的凭空生成或消失。颜色为对象级 `T_trap_core = ∫∫ V_dev,y Q_core dA dz` 映射到 surface-center 经纬度后的网格累积值：红/蓝表示符号相反的 trapping PV transport 贡献。\n\n",
    ]
    if math.isfinite(ratio) and abs(ratio) < 0.15:
        lines.append("全域 signed/absolute 比值较小，说明 trapping 项更像是在 ACC 内部搬运和重排 PV，而不是造成强净库存变化。\n\n")
    else:
        lines.append("全域 signed/absolute 比值不小，说明 trapping 项在当前采样域存在明显非对称累积；需要结合边界通量判断是否代表控制体净变化。\n\n")
    lines.append("## 极性贡献\n\n")
    for row in rows:
        lines.append(
            f"- `{row['polarity']}`：objects `{row['n_objects']:,}`，tracks `{row['n_tracks']:,}`，"
            f"signed `{row['signed_sum']:.6e}`，absolute `{row['absolute_sum']:.6e}`，"
            f"signed/absolute `{row['signed_to_absolute_ratio']:.4f}`。\n"
        )
    lines.append("\n## 热点区域\n\n")
    for row in hot.itertuples(index=False):
        lines.append(
            f"- lon `{row.lon_center:.1f}`, lat `{row.lat_center:.1f}`：net `{row.T_net:.6e}`，abs `{row.T_abs:.6e}`，objects `{int(row.n_objects)}`。\n"
        )
    lines.append("\n## 口径备注\n\n")
    lines.append("`Q_core` 使用 all-shape representative E-P profiles 中的 `q_mean(tau, depth, r/R)`，对象级轨迹贡献来自 centerline 的 `dy_m/dt`。因此该图是面向空间证据的快速对象级归约，而不是逐对象重新采样年度 NetCDF 的严格 q' 重构。\n")
    (output_dir / "trapping_pv_redistribution_interpretation_zh.md").write_text("".join(lines), encoding="utf-8")

    summary = {
        "n_objects": int(objects["eddy3d_object_id"].nunique()),
        "n_tracks": int(objects["track3d_id"].nunique()),
        "signed_sum": signed_sum,
        "absolute_sum": abs_sum,
        "signed_to_absolute_ratio": ratio,
        "r_core": r_core,
        "upper_depth_m": upper_depth_m,
        "polarity": rows,
        "top_hotspots": hot.to_dict(orient="records"),
    }
    (output_dir / "trapping_pv_redistribution_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    rv_root = Path(args.rv_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    objects = _load_all_axis_objects(rv_root)
    points = _load_points_for_objects(rv_root, set(objects["eddy3d_object_id"].astype("int64")))
    kin = _centerline_vdev_y(points)
    q_area, tau_grid = _load_q_area_integrals(rv_root, args.r_core)
    object_transport = _object_transport(objects, kin, q_area, tau_grid, args.upper_depth_m)
    grid = _grid_transport(object_transport, args.lon_bin_deg, args.lat_bin_deg)

    object_transport.to_parquet(output_dir / "object_level_trapping_transport_summary.parquet", index=False)
    object_transport.to_csv(output_dir / "object_level_trapping_transport_summary.csv", index=False)
    grid.to_parquet(output_dir / "gridded_trapping_pv_redistribution.parquet", index=False)
    grid.to_csv(output_dir / "gridded_trapping_pv_redistribution.csv", index=False)
    _plot_map(object_transport, grid, output_dir, args.lon_bin_deg, args.lat_bin_deg)
    _write_report(object_transport, grid, output_dir, args.r_core, args.upper_depth_m)


def main() -> None:
    parser = argparse.ArgumentParser(description="Map all-shape ACC object-level trapping PV redistribution.")
    parser.add_argument("--rv-root", default="/root/autodl-fs/2020_2022_acc/result/representative_vortex")
    parser.add_argument("--output-dir", default="/root/autodl-fs/2020_2022_acc/result/Diagonise_EP_Chen_one/trapping_pv_transport_map")
    parser.add_argument("--r-core", type=float, default=1.5)
    parser.add_argument("--upper-depth-m", type=float, default=1000.0)
    parser.add_argument("--lon-bin-deg", type=float, default=4.0)
    parser.add_argument("--lat-bin-deg", type=float, default=1.0)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
