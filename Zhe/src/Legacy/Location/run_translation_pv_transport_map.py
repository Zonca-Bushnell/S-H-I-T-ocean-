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


def _load_axis_objects(rv_root: Path) -> pd.DataFrame:
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


def _center_translation_velocity(objects: pd.DataFrame) -> pd.DataFrame:
    p = objects.sort_values(["track3d_id", "date"]).copy()
    p["dt_day"] = p.groupby("track3d_id", sort=False)["date"].diff().dt.total_seconds() / SECONDS_PER_DAY
    p["dlat_dt_deg_day"] = p.groupby("track3d_id", sort=False)["surface_lat"].diff() / p["dt_day"]
    # Positive is northward. This is a local metric approximation adequate for meridional velocity.
    p["V_center_y_m_s"] = p["dlat_dt_deg_day"] * 111_000.0 / SECONDS_PER_DAY
    return p


def _nearest_tau(values: pd.Series, tau_grid: np.ndarray) -> np.ndarray:
    arr = values.to_numpy(dtype="float64")
    idx = np.nanargmin(np.abs(arr[:, None] - tau_grid[None, :]), axis=1)
    return tau_grid[idx]


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


def _load_q_volume_unit_r2(rv_root: Path, r_core: float, upper_depth_m: float) -> tuple[pd.DataFrame, np.ndarray]:
    frames: list[pd.DataFrame] = []
    tau_values: list[float] = []
    for polarity in ("cyclonic", "anticyclonic"):
        path = rv_root / polarity / "ep_flux_terms" / "continuous_ep_flux_profiles.parquet"
        if not path.exists():
            raise FileNotFoundError(path)
        prof = pd.read_parquet(
            path,
            columns=["polarity", "tau_center", "depth_index", "depth_m", "r_over_R", "q_mean"],
        )
        prof = prof[(prof["polarity"].eq(polarity)) & (prof["r_over_R"].le(r_core))].copy()
        tau_values.extend(prof["tau_center"].dropna().unique().tolist())
        radial = np.sort(prof["r_over_R"].dropna().unique().astype("float64"))
        if len(radial) < 2:
            raise ValueError(f"Need at least two radial bins in {path}")
        dr = float(np.nanmedian(np.diff(radial)))
        prof["area_weight_unit_R2"] = 2.0 * math.pi * prof["r_over_R"].astype("float64") * dr
        prof["q_area_unit_R2"] = prof["q_mean"].astype("float64") * prof["area_weight_unit_R2"]
        dz = _depth_thickness(prof[["depth_index", "depth_m"]].drop_duplicates())
        prof = prof.merge(dz, on=["depth_index", "depth_m"], how="left")
        prof["q_volume_unit_R2_layer"] = prof["q_area_unit_R2"] * prof["dz_m"].astype("float64")
        prof["q_volume_upper_unit_R2_layer"] = np.where(
            prof["depth_m"].le(upper_depth_m), prof["q_volume_unit_R2_layer"], np.nan
        )
        prof["q_volume_deep_unit_R2_layer"] = np.where(
            prof["depth_m"].gt(upper_depth_m), prof["q_volume_unit_R2_layer"], np.nan
        )
        grouped = (
            prof.groupby(["polarity", "tau_center"], observed=True, sort=True)
            .agg(
                q_volume_unit_R2=("q_volume_unit_R2_layer", "sum"),
                q_volume_upper_unit_R2=("q_volume_upper_unit_R2_layer", "sum"),
                q_volume_deep_unit_R2=("q_volume_deep_unit_R2_layer", "sum"),
                depth_layers=("depth_index", "nunique"),
            )
            .reset_index()
        )
        frames.append(grouped)
    out = pd.concat(frames, ignore_index=True)
    tau_grid = np.sort(np.unique(np.asarray(tau_values, dtype="float64")))
    return out, tau_grid


def _object_transport(objects: pd.DataFrame, q_volume: pd.DataFrame, tau_grid: np.ndarray) -> pd.DataFrame:
    out = _center_translation_velocity(objects)
    out["tau_center"] = _nearest_tau(out["life_phase"], tau_grid)
    out = out.merge(q_volume, on=["polarity", "tau_center"], how="left")
    radius2 = out["mean_radius_m"].astype("float64") ** 2
    finite = np.isfinite(out["V_center_y_m_s"]) & np.isfinite(out["q_volume_unit_R2"])
    out["T_translation_core"] = out["V_center_y_m_s"] * out["q_volume_unit_R2"] * radius2
    out["T_translation_upper"] = out["V_center_y_m_s"] * out["q_volume_upper_unit_R2"] * radius2
    out["T_translation_deep"] = out["V_center_y_m_s"] * out["q_volume_deep_unit_R2"] * radius2
    for col in ["T_translation_core", "T_translation_upper", "T_translation_deep"]:
        out.loc[~finite, col] = 0.0
    out["has_translation_velocity"] = finite
    keep = [
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
        "V_center_y_m_s",
        "q_volume_unit_R2",
        "T_translation_core",
        "T_translation_upper",
        "T_translation_deep",
        "has_translation_velocity",
    ]
    return out[keep].copy()


def _grid_transport(objects: pd.DataFrame, lon_bin_deg: float, lat_bin_deg: float) -> pd.DataFrame:
    out = objects.copy()
    out["lon_bin"] = np.floor(out["surface_lon_plot"] / lon_bin_deg).astype("int64")
    max_lon_bin = int(round(360.0 / lon_bin_deg))
    out.loc[out["lon_bin"].ge(max_lon_bin), "lon_bin"] = max_lon_bin - 1
    out["lat_bin"] = np.floor((out["surface_lat"] + 90.0) / lat_bin_deg).astype("int64")
    out["lon_center"] = (out["lon_bin"] + 0.5) * lon_bin_deg
    out["lat_center"] = -90.0 + (out["lat_bin"] + 0.5) * lat_bin_deg
    out["T_abs"] = out["T_translation_core"].abs()
    grouped = (
        out.groupby(["lon_bin", "lat_bin", "lon_center", "lat_center"], observed=True, sort=True)
        .agg(
            T_net=("T_translation_core", "sum"),
            T_abs=("T_abs", "sum"),
            T_cyclonic=("T_translation_core", lambda s: s[out.loc[s.index, "polarity"].eq("cyclonic")].sum()),
            T_anticyclonic=("T_translation_core", lambda s: s[out.loc[s.index, "polarity"].eq("anticyclonic")].sum()),
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
    cb.set_label("Net translation PV transport (area-depth integrated, signed)")
    colors = {"cyclonic": "#2b6cb0", "anticyclonic": "#c2410c"}
    for polarity, part in objects.groupby("polarity", sort=True):
        sample = part[part["has_translation_velocity"]]
        if len(sample) > 120000:
            sample = sample.sample(120000, random_state=17)
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
    ax.set_title("ACC all-shape whole-eddy translation PV transport")
    ax.grid(True, color="0.85", linewidth=0.5)
    ax.legend(loc="lower left", frameon=True, fontsize=8)

    inset = ax.inset_axes([0.73, 0.07, 0.25, 0.26])
    stats = []
    for polarity, part in objects.groupby("polarity", sort=True):
        valid = part[part["has_translation_velocity"]]
        stats.append((polarity, float(valid["T_translation_core"].sum()), float(valid["T_translation_core"].abs().sum())))
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

    fig.text(
        0.06,
        0.045,
        "This map uses whole-eddy surface-center meridional translation, not vertical-deviation trapping speed.",
        fontsize=9,
    )
    fig.savefig(output_dir / "acc_translation_pv_transport_map.png", dpi=220)
    fig.savefig(output_dir / "acc_translation_pv_transport_map.pdf")
    plt.close(fig)


def _write_report(objects: pd.DataFrame, grid: pd.DataFrame, output_dir: Path, r_core: float, upper_depth_m: float) -> None:
    valid = objects[objects["has_translation_velocity"]].copy()
    signed_sum = float(valid["T_translation_core"].sum())
    abs_sum = float(valid["T_translation_core"].abs().sum())
    ratio = signed_sum / abs_sum if abs_sum > 0 else float("nan")
    rows = []
    for polarity, part in valid.groupby("polarity", sort=True):
        rows.append(
            {
                "polarity": polarity,
                "n_objects": int(part["eddy3d_object_id"].nunique()),
                "n_tracks": int(part["track3d_id"].nunique()),
                "mean_v_center_y_m_s": float(part["V_center_y_m_s"].mean()),
                "median_v_center_y_m_s": float(part["V_center_y_m_s"].median()),
                "northward_fraction": float((part["V_center_y_m_s"] > 0).mean()),
                "signed_sum": float(part["T_translation_core"].sum()),
                "absolute_sum": float(part["T_translation_core"].abs().sum()),
                "signed_to_absolute_ratio": float(part["T_translation_core"].sum() / part["T_translation_core"].abs().sum()),
                "positive_transport_fraction": float((part["T_translation_core"] > 0).mean()),
            }
        )
    stats = pd.DataFrame(rows)
    stats.to_csv(output_dir / "polarity_translation_transport_totals.csv", index=False)
    hot = grid.sort_values("T_abs", ascending=False).head(10).copy()
    hot.to_csv(output_dir / "top_translation_transport_hotspots.csv", index=False)

    lines = [
        "# ACC 整体涡旋平移 PV 通量图解释\n\n",
        f"- 输入对象：all-shape shape-classified axis objects，共 `{objects['eddy3d_object_id'].nunique():,}` 个 object-day；其中有有限整体平移速度的 object-day 为 `{len(valid):,}`。\n",
        f"- 核心积分半径：`r/R <= {r_core:g}`；upper/deep 分界：`{upper_depth_m:g} m`。\n",
        f"- 全域 signed sum：`{signed_sum:.6e}`；absolute sum：`{abs_sum:.6e}`；signed/absolute：`{ratio:.4f}`。\n\n",
        "## 物理口径\n\n",
        "本图使用 `T_translation = V_center,y ∫Q_core dA dz`，其中 `V_center,y` 是 surface center 沿纬向的整体南北平移速度。它不同于上一张 trapping/deviation 图中的 `V_dev,y = d(y_centerline)/dt`。\n\n",
        "因此这张图直接检验“整体涡旋平移 + PV 异常符号”是否形成净经向 PV 通量。\n\n",
        "## 极性贡献\n\n",
    ]
    for row in rows:
        lines.append(
            f"- `{row['polarity']}`：objects `{row['n_objects']:,}`，tracks `{row['n_tracks']:,}`，"
            f"mean V_center,y `{row['mean_v_center_y_m_s']:.6e} m/s`，northward fraction `{row['northward_fraction']:.4f}`，"
            f"signed `{row['signed_sum']:.6e}`，absolute `{row['absolute_sum']:.6e}`，"
            f"signed/absolute `{row['signed_to_absolute_ratio']:.4f}`，positive transport fraction `{row['positive_transport_fraction']:.4f}`。\n"
        )
    lines.append("\n## 热点区域\n\n")
    for row in hot.itertuples(index=False):
        lines.append(
            f"- lon `{row.lon_center:.1f}`, lat `{row.lat_center:.1f}`：net `{row.T_net:.6e}`，abs `{row.T_abs:.6e}`，objects `{int(row.n_objects)}`。\n"
        )
    lines.append("\n## 解释提醒\n\n")
    if math.isfinite(ratio) and abs(ratio) > 0.1:
        lines.append("整体平移口径下 signed/absolute 不小，说明与 trapping/deviation 图相比，它更可能代表一个域尺度有方向性的经向 PV 通量。\n")
    else:
        lines.append("整体平移口径下 signed/absolute 仍较小，说明当前 ACC 样本中的气旋/反气旋整体传播并没有形成强净经向 PV 通量。\n")
    (output_dir / "translation_pv_transport_interpretation_zh.md").write_text("".join(lines), encoding="utf-8")
    summary = {
        "n_objects_total": int(objects["eddy3d_object_id"].nunique()),
        "n_objects_with_velocity": int(valid["eddy3d_object_id"].nunique()),
        "signed_sum": signed_sum,
        "absolute_sum": abs_sum,
        "signed_to_absolute_ratio": ratio,
        "polarity": rows,
        "top_hotspots": hot.to_dict(orient="records"),
    }
    (output_dir / "translation_pv_transport_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    rv_root = Path(args.rv_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    objects = _load_axis_objects(rv_root)
    q_volume, tau_grid = _load_q_volume_unit_r2(rv_root, args.r_core, args.upper_depth_m)
    object_transport = _object_transport(objects, q_volume, tau_grid)
    grid = _grid_transport(object_transport[object_transport["has_translation_velocity"]].copy(), args.lon_bin_deg, args.lat_bin_deg)
    object_transport.to_parquet(output_dir / "object_level_translation_pv_transport_summary.parquet", index=False)
    object_transport.to_csv(output_dir / "object_level_translation_pv_transport_summary.csv", index=False)
    grid.to_parquet(output_dir / "gridded_translation_pv_transport.parquet", index=False)
    grid.to_csv(output_dir / "gridded_translation_pv_transport.csv", index=False)
    _plot_map(object_transport, grid, output_dir, args.lon_bin_deg, args.lat_bin_deg)
    _write_report(object_transport, grid, output_dir, args.r_core, args.upper_depth_m)


def main() -> None:
    parser = argparse.ArgumentParser(description="Map ACC whole-eddy meridional translation PV transport.")
    parser.add_argument("--rv-root", default="/root/autodl-fs/2020_2022_acc/result/representative_vortex")
    parser.add_argument("--output-dir", default="/root/autodl-fs/2020_2022_acc/result/Diagonise_EP_Chen_one/translation_pv_transport_map")
    parser.add_argument("--r-core", type=float, default=1.5)
    parser.add_argument("--upper-depth-m", type=float, default=1000.0)
    parser.add_argument("--lon-bin-deg", type=float, default=4.0)
    parser.add_argument("--lat-bin-deg", type=float, default=1.0)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
