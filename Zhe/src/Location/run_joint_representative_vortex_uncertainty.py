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

from src.First_temp.axis_streamfunction_separation import grid_spacing_m, relative_vorticity, streamfunction_from_zeta
from src.First_temp.lifecycle_ep_flux_nondim_validation import make_polar_grid
from src.First_temp.tilted_ep_flux_validation import bilinear_sample, sanitize_ocean_field, xy_to_lonlat
from src.Location.common import load_config
from src.Location.streaming_cmems import read_day_data


EPS = 1e-12


def _default_paths(dataset: str) -> tuple[Path, Path]:
    if dataset == "all_shape":
        return (
            Path("/root/autodl-fs/2020_2022_acc/result/representative_vortex"),
            Path("/root/autodl-fs/2020_2022_acc/result/joint_representativeness"),
        )
    if dataset == "coherent_only":
        return (
            Path("/root/autodl-fs/2020_2022_acc/result_coherent_only/representative_vortex"),
            Path("/root/autodl-fs/2020_2022_acc/result_coherent_only/joint_representativeness"),
        )
    raise ValueError(dataset)


def _load_objects(rv_root: Path, dataset: str, start: str | None, end: str | None) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    top = rv_root / "object_cache" / "selected_lifecycle_objects.parquet"
    if top.exists():
        frames.append(pd.read_parquet(top))
    else:
        for polarity in ("anticyclonic", "cyclonic"):
            p = rv_root / polarity / "object_cache" / "selected_lifecycle_objects.parquet"
            if p.exists():
                frames.append(pd.read_parquet(p))
    if not frames:
        raise FileNotFoundError(f"No selected_lifecycle_objects under {rv_root}")
    objects = pd.concat(frames, ignore_index=True).drop_duplicates("eddy3d_object_id").copy()
    objects["date"] = pd.to_datetime(objects["date"])
    if start:
        objects = objects[objects["date"].ge(pd.Timestamp(start))]
    if end:
        objects = objects[objects["date"].le(pd.Timestamp(end))]
    objects["eddy3d_object_id"] = objects["eddy3d_object_id"].astype("int64")
    objects["track3d_id"] = objects["track3d_id"].astype("int64")
    objects["life_phase"] = objects["life_phase"].astype("float64").clip(0.0, 1.0)
    if "mean_radius_m" not in objects:
        raise ValueError("selected_lifecycle_objects must include mean_radius_m")
    objects["mean_radius_m"] = objects["mean_radius_m"].astype("float64")
    objects["dataset"] = dataset
    return objects


def _load_points(rv_root: Path, object_ids: set[int]) -> pd.DataFrame:
    p = rv_root / "axis" / "rotated_points.parquet"
    if not p.exists():
        raise FileNotFoundError(p)
    cols = [
        "shape_class",
        "polarity",
        "track3d_id",
        "eddy3d_object_id",
        "date",
        "depth_index",
        "depth_m",
        "z_m",
        "longitude",
        "latitude",
        "x_m",
        "y_m",
        "x_rot_m",
        "y_rot_m",
        "temp_direction_rad",
        "temp_direction_deg",
    ]
    points = pd.read_parquet(p, columns=cols)
    points["eddy3d_object_id"] = points["eddy3d_object_id"].astype("int64")
    points = points[points["eddy3d_object_id"].isin(object_ids)].copy()
    points["date"] = pd.to_datetime(points["date"])
    points["track3d_id"] = points["track3d_id"].astype("int64")
    points["depth_index"] = points["depth_index"].astype("int16")
    return points


def _tau_grid_from_profiles(rv_root: Path) -> np.ndarray:
    for p in [
        rv_root / "streamfunction_templates" / "continuous_radial_psi_profiles.parquet",
        rv_root / "anticyclonic" / "streamfunction_templates" / "continuous_radial_psi_profiles.parquet",
        rv_root / "cyclonic" / "streamfunction_templates" / "continuous_radial_psi_profiles.parquet",
    ]:
        if p.exists():
            df = pd.read_parquet(p, columns=["tau_center"])
            return np.sort(df["tau_center"].dropna().unique().astype("float64"))
    raise FileNotFoundError(f"No streamfunction profile under {rv_root}")


def _nearest_tau(values: pd.Series, tau_grid: np.ndarray) -> np.ndarray:
    arr = values.to_numpy(dtype="float64")
    idx = np.nanargmin(np.abs(arr[:, None] - tau_grid[None, :]), axis=1)
    return tau_grid[idx]


def _load_psi_profiles(rv_root: Path, dataset: str, r_core: float) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    if dataset == "all_shape":
        paths = [
            rv_root / "anticyclonic" / "streamfunction_templates" / "continuous_radial_psi_profiles.parquet",
            rv_root / "cyclonic" / "streamfunction_templates" / "continuous_radial_psi_profiles.parquet",
        ]
    else:
        paths = [rv_root / "streamfunction_templates" / "continuous_radial_psi_profiles.parquet"]
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(p)
        part = pd.read_parquet(p)
        frames.append(part)
    psi = pd.concat(frames, ignore_index=True)
    psi = psi[psi["r_over_R"].le(r_core)].copy()
    return psi


def _axis_reference(rv_root: Path, output_dir: Path, objects: pd.DataFrame, points: pd.DataFrame, tau_grid: np.ndarray) -> pd.DataFrame:
    out = output_dir / "axis_reference_tau_depth.parquet"
    if out.exists():
        return pd.read_parquet(out)
    meta = objects[["eddy3d_object_id", "life_phase", "mean_radius_m", "polarity"]].copy()
    meta["tau_center"] = _nearest_tau(meta["life_phase"], tau_grid)
    p = points.merge(meta, on=["eddy3d_object_id", "polarity"], how="inner")
    p["x_nd"] = p["x_rot_m"].astype("float64") / p["mean_radius_m"].astype("float64")
    p["y_nd"] = p["y_rot_m"].astype("float64") / p["mean_radius_m"].astype("float64")
    grouped = (
        p.groupby(["polarity", "tau_center", "depth_index", "depth_m"], observed=True, sort=True)
        .agg(
            x_bar_nd=("x_nd", "mean"),
            y_bar_nd=("y_nd", "mean"),
            x_std_nd=("x_nd", "std"),
            y_std_nd=("y_nd", "std"),
            n_axis_objects=("eddy3d_object_id", "nunique"),
            n_axis_tracks=("track3d_id", "nunique"),
        )
        .reset_index()
    )
    tmp = out.with_suffix(out.suffix + ".tmp")
    grouped.to_parquet(tmp, index=False)
    tmp.replace(out)
    return grouped


def _psi_bar_lookup(psi: pd.DataFrame, polarity: str, tau_center: float) -> pd.DataFrame:
    part = psi[(psi["polarity"].eq(polarity)) & np.isclose(psi["tau_center"], tau_center)].copy()
    if part.empty:
        raise KeyError((polarity, tau_center))
    return part


def _matrix_from_profile(profile: pd.DataFrame, value_col: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    depths = np.sort(profile["depth_m"].unique().astype("float64"))
    radial = np.sort(profile["r_over_R"].unique().astype("float64"))
    pivot = profile.pivot_table(index="depth_m", columns="r_over_R", values=value_col, aggfunc="mean")
    pivot = pivot.reindex(index=depths, columns=radial)
    return depths, radial, pivot.to_numpy(dtype="float64")


def _depth_weights(depth: np.ndarray) -> np.ndarray:
    if len(depth) < 2:
        return np.ones_like(depth, dtype="float64")
    edges = np.empty(len(depth) + 1, dtype="float64")
    edges[1:-1] = 0.5 * (depth[:-1] + depth[1:])
    edges[0] = max(0.0, depth[0] - 0.5 * (depth[1] - depth[0]))
    edges[-1] = depth[-1] + 0.5 * (depth[-1] - depth[-2])
    return np.diff(edges)


def _sample_object_psi_radial_mean(obj, center_line: pd.DataFrame, lon: np.ndarray, lat: np.ndarray, depth: np.ndarray, psi_prime: np.ndarray, radial: np.ndarray, theta: np.ndarray, rr: np.ndarray, tt: np.ndarray) -> np.ndarray | None:
    if len(center_line) != len(depth):
        return None
    radius_m = float(obj.mean_radius_m)
    theta_obj = float(obj.temp_direction_rad)
    cos_t = math.cos(theta_obj)
    sin_t = math.sin(theta_obj)
    local_x = rr * radius_m * np.cos(tt)
    local_y = rr * radius_m * np.sin(tt)
    out_layers = []
    for k, row in enumerate(center_line.itertuples(index=False)):
        x_rot = float(row.x_rot_m) + local_x
        y_rot = float(row.y_rot_m) + local_y
        x_orig = x_rot * cos_t - y_rot * sin_t
        y_orig = x_rot * sin_t + y_rot * cos_t
        target_lon, target_lat = xy_to_lonlat(x_orig, y_orig, float(obj.surface_lon), float(obj.surface_lat))
        sampled = bilinear_sample(lon, lat, psi_prime[k], target_lon, target_lat)
        out_layers.append(np.nanmean(sampled, axis=1))
    return np.asarray(out_layers, dtype="float64")


def _axis_error(obj, center_line: pd.DataFrame, axis_ref: pd.DataFrame) -> tuple[float, float, float, int]:
    ref = axis_ref[(axis_ref["polarity"].eq(obj.polarity)) & np.isclose(axis_ref["tau_center"], float(obj.tau_center))]
    merged = center_line.merge(ref[["depth_index", "x_bar_nd", "y_bar_nd"]], on="depth_index", how="inner")
    if merged.empty:
        return np.nan, np.nan, np.nan, 0
    r = max(float(obj.mean_radius_m), EPS)
    x = merged["x_rot_m"].to_numpy(dtype="float64") / r
    y = merged["y_rot_m"].to_numpy(dtype="float64") / r
    xb = merged["x_bar_nd"].to_numpy(dtype="float64")
    yb = merged["y_bar_nd"].to_numpy(dtype="float64")
    diff = (x - xb) ** 2 + (y - yb) ** 2
    denom = np.nanmean(xb**2 + yb**2) + 1.0
    e_axis = float(np.nanmean(diff) / max(denom, EPS))
    return e_axis, float(np.nanmean(np.sqrt(diff))), float(np.nanmax(np.sqrt(diff))), int(len(merged))


def _strict_day(
    config: dict,
    rv_root: Path,
    output_dir: Path,
    day: pd.Timestamp,
    day_objects: pd.DataFrame,
    points: pd.DataFrame,
    psi_profiles: pd.DataFrame,
    axis_ref: pd.DataFrame,
    r_core: float,
    radial_bins: int,
    azimuth_bins: int,
) -> Path:
    part_dir = output_dir / "joint_object_errors_parts" / f"year={day.year}"
    part_dir.mkdir(parents=True, exist_ok=True)
    part_path = part_dir / f"joint_object_errors_{day:%Y%m%d}.parquet"
    if part_path.exists():
        return part_path
    day_data = read_day_data(config, day.date())
    lon = day_data["lon"]
    lat = day_data["lat"]
    depth = day_data["depth"]
    u = sanitize_ocean_field(day_data["u_all"].astype("float64", copy=False))
    v = sanitize_ocean_field(day_data["v_all"].astype("float64", copy=False))
    _, dy, dx = grid_spacing_m(lon, lat)
    psi_prime = streamfunction_from_zeta(relative_vorticity(lon, lat, u, v), dx, dy)
    radial, theta, rr, tt, _ = make_polar_grid(r_core, radial_bins, azimuth_bins)
    dweights = _depth_weights(depth)
    rweights = np.maximum(radial, 1e-6)
    weights = dweights[:, None] * rweights[None, :]
    points_by_object = {
        int(k): part.sort_values("depth_index").copy()
        for k, part in points[points["date"].eq(day)].groupby("eddy3d_object_id", sort=False)
    }
    rows: list[dict] = []
    profile_cache: dict[tuple[str, float], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for obj in day_objects.itertuples(index=False):
        center_line = points_by_object.get(int(obj.eddy3d_object_id))
        if center_line is None:
            continue
        key = (str(obj.polarity), float(obj.tau_center))
        if key not in profile_cache:
            prof = _psi_bar_lookup(psi_profiles, key[0], key[1])
            profile_cache[key] = _matrix_from_profile(prof, "psi_mean")
        prof_depth, prof_radial, psi_bar = profile_cache[key]
        if len(prof_depth) != len(depth) or len(prof_radial) != len(radial):
            continue
        psi_obj = _sample_object_psi_radial_mean(obj, center_line, lon, lat, depth, psi_prime, radial, theta, rr, tt)
        if psi_obj is None:
            continue
        ok = np.isfinite(psi_obj) & np.isfinite(psi_bar)
        if np.any(ok):
            w_ok = weights * ok
            psi_obj_center = np.nansum(psi_obj * w_ok) / max(np.nansum(w_ok), EPS)
            psi_bar_center = np.nansum(psi_bar * w_ok) / max(np.nansum(w_ok), EPS)
            psi_obj_anom = psi_obj - psi_obj_center
            psi_bar_anom = psi_bar - psi_bar_center
        else:
            psi_obj_anom = psi_obj
            psi_bar_anom = psi_bar
        numerator = np.nansum(((psi_obj_anom - psi_bar_anom) ** 2) * weights * ok)
        denom = np.nansum((psi_bar_anom**2) * weights * ok)
        e_psi = float(numerator / max(denom, EPS)) if np.any(ok) else np.nan
        e_axis, axis_mean_abs_nd, axis_max_abs_nd, n_axis_layers = _axis_error(obj, center_line, axis_ref)
        rows.append(
            {
                "date": day.strftime("%Y-%m-%d"),
                "dataset": str(obj.dataset),
                "eddy3d_object_id": int(obj.eddy3d_object_id),
                "track3d_id": int(obj.track3d_id),
                "shape_class": str(obj.shape_class),
                "polarity": str(obj.polarity),
                "life_phase": float(obj.life_phase),
                "tau_center": float(obj.tau_center),
                "mean_radius_m": float(obj.mean_radius_m),
                "E_psi_core": e_psi,
                "E_axis": e_axis,
                "E_joint": e_psi + e_axis if np.isfinite(e_psi) and np.isfinite(e_axis) else np.nan,
                "sqrt_E_psi_core": math.sqrt(e_psi) if np.isfinite(e_psi) and e_psi >= 0 else np.nan,
                "sqrt_E_axis": math.sqrt(e_axis) if np.isfinite(e_axis) and e_axis >= 0 else np.nan,
                "sqrt_E_joint": math.sqrt(e_psi + e_axis) if np.isfinite(e_psi) and np.isfinite(e_axis) and e_psi + e_axis >= 0 else np.nan,
                "axis_mean_abs_nd": axis_mean_abs_nd,
                "axis_max_abs_nd": axis_max_abs_nd,
                "n_axis_layers": n_axis_layers,
            }
        )
    tmp = part_path.with_suffix(part_path.suffix + ".tmp")
    pd.DataFrame.from_records(rows).to_parquet(tmp, index=False)
    tmp.replace(part_path)
    return part_path


def _load_all_parts(output_dir: Path) -> pd.DataFrame:
    paths = sorted((output_dir / "joint_object_errors_parts").glob("year=*/*.parquet"))
    if not paths:
        raise FileNotFoundError(output_dir / "joint_object_errors_parts")
    return pd.concat((pd.read_parquet(p) for p in paths), ignore_index=True)


def _summaries(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    finite = df[np.isfinite(df["E_joint"])].copy()
    track = (
        finite.groupby(["dataset", "track3d_id", "shape_class", "polarity", "tau_center"], observed=True, sort=True)
        .agg(
            n_object_days=("eddy3d_object_id", "nunique"),
            E_psi_core=("E_psi_core", "mean"),
            E_axis=("E_axis", "mean"),
            E_joint=("E_joint", "mean"),
            sqrt_E_joint=("sqrt_E_joint", "mean"),
        )
        .reset_index()
    )
    tau = (
        track.groupby(["dataset", "shape_class", "polarity", "tau_center"], observed=True, sort=True)
        .agg(
            n_tracks=("track3d_id", "nunique"),
            E_joint_mean=("E_joint", "mean"),
            E_joint_std=("E_joint", "std"),
            E_joint_sem=("E_joint", lambda s: float(s.std(ddof=1) / math.sqrt(max(s.count(), 1))) if s.count() > 1 else np.nan),
            sqrt_E_joint_median=("sqrt_E_joint", "median"),
            sqrt_E_joint_p90=("sqrt_E_joint", lambda s: float(s.quantile(0.90))),
            E_psi_core_mean=("E_psi_core", "mean"),
            E_axis_mean=("E_axis", "mean"),
        )
        .reset_index()
    )
    return track, tau


def _coverage(track: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, part in track.groupby(["dataset", "shape_class", "polarity"], observed=True, sort=True):
        row = dict(zip(["dataset", "shape_class", "polarity"], keys))
        row["n_tracks"] = int(part["track3d_id"].nunique())
        row["median_sqrt_E_joint"] = float(part["sqrt_E_joint"].median())
        row["p90_sqrt_E_joint"] = float(part["sqrt_E_joint"].quantile(0.90))
        for threshold in (0.1, 0.2, 0.3, 0.5):
            row[f"coverage_sqrt_E_joint_lt_{threshold:g}"] = float((part["sqrt_E_joint"] < threshold).mean())
        rows.append(row)
    return pd.DataFrame(rows)


def _plot_outputs(track: pd.DataFrame, tau: pd.DataFrame, output_dir: Path) -> None:
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    plt.figure(figsize=(10, 5))
    for (shape, polarity), part in tau.groupby(["shape_class", "polarity"], observed=True):
        label = f"{shape}/{polarity}"
        plt.plot(part["tau_center"], part["sqrt_E_joint_median"], marker="o", ms=3, label=label)
    plt.xlabel("tau")
    plt.ylabel("median sqrt(E_joint)")
    plt.title("Joint representativeness error by lifecycle")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(fig_dir / "joint_error_by_tau.png", dpi=180)
    plt.close()

    sample = track.sample(min(len(track), 50000), random_state=17) if len(track) > 50000 else track
    plt.figure(figsize=(6, 6))
    plt.scatter(np.sqrt(sample["E_psi_core"]), np.sqrt(sample["E_axis"]), s=5, alpha=0.25)
    plt.xlabel("sqrt(E_psi_core)")
    plt.ylabel("sqrt(E_axis)")
    plt.title("Track-block psi vs axis error")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_dir / "psi_vs_axis_error_scatter.png", dpi=180)
    plt.close()

    cov = _coverage(track)
    labels = (cov["shape_class"] + "/" + cov["polarity"]).tolist()
    x = np.arange(len(cov))
    plt.figure(figsize=(max(8, len(cov) * 0.8), 5))
    width = 0.2
    for i, threshold in enumerate((0.1, 0.2, 0.3, 0.5)):
        plt.bar(x + (i - 1.5) * width, cov[f"coverage_sqrt_E_joint_lt_{threshold:g}"], width=width, label=f"<{threshold:g}")
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel("track fraction")
    plt.ylim(0, 1)
    plt.title("Track coverage under joint-error thresholds")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "track_coverage_thresholds.png", dpi=180)
    plt.close()


def _write_report(df: pd.DataFrame, track: pd.DataFrame, tau: pd.DataFrame, output_dir: Path, dataset: str) -> None:
    cov = _coverage(track)
    cov.to_csv(output_dir / "joint_coverage_by_shape_polarity.csv", index=False)
    summary = {
        "dataset": dataset,
        "n_object_days": int(df["eddy3d_object_id"].nunique()),
        "n_tracks": int(df["track3d_id"].nunique()),
        "coverage": cov.to_dict(orient="records"),
        "median_sqrt_E_joint": float(track["sqrt_E_joint"].median()),
        "p90_sqrt_E_joint": float(track["sqrt_E_joint"].quantile(0.90)),
    }
    (output_dir / "joint_representativeness_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# ACC joint representativeness 诊断\n\n",
        f"- Dataset: `{dataset}`\n",
        f"- Object-days with finite joint error: `{df['eddy3d_object_id'].nunique():,}`\n",
        f"- Tracks with finite joint error: `{df['track3d_id'].nunique():,}`\n",
        f"- Track-block median sqrt(E_joint): `{summary['median_sqrt_E_joint']:.4f}`\n",
        f"- Track-block p90 sqrt(E_joint): `{summary['p90_sqrt_E_joint']:.4f}`\n\n",
        "## 口径\n\n",
        "本诊断把代表涡视作联合对象 `[psi(z,r), x_rot(z)/R, y_rot(z)/R]`，而不是把流函数和倾斜偏移拆成两个互不相干的误差条。`E_joint = E_psi_core + E_axis`，两项已无量纲化并默认等权。\n\n",
        "aligned-frame 用于判断结构代表性；若解释原始地理方向，需要按合成时角度逆旋转：`x_orig=x_rot cosθ-y_rot sinθ`, `y_orig=x_rot sinθ+y_rot cosθ`。all-shape 使用 `temp_direction`，coherent-only 使用 `global_ls_alpha` 对齐结果，因此两者方向相位不可直接比较。\n\n",
        "## 覆盖率\n\n",
    ]
    for _, row in cov.iterrows():
        lines.append(
            f"- `{row['shape_class']}/{row['polarity']}`: tracks `{int(row['n_tracks'])}`, "
            f"median sqrt(E_joint) `{row['median_sqrt_E_joint']:.4f}`, p90 `{row['p90_sqrt_E_joint']:.4f}`, "
            f"coverage <0.3 `{row['coverage_sqrt_E_joint_lt_0.3']:.3f}`, "
            f"coverage <0.5 `{row['coverage_sqrt_E_joint_lt_0.5']:.3f}`。\n"
        )
    lines.append("\n## 限制\n\n")
    lines.append("旧的 `psi_variance/n_tracks` 只可作为辅助，因为现有 `psi_variance` 主要来自对象内方位角方差；本报告的主结论来自逐对象重采样后的 joint residual 与 track-block 聚合。\n")
    (output_dir / "joint_representativeness_summary_zh.md").write_text("".join(lines), encoding="utf-8")


def run_strict(args: argparse.Namespace) -> None:
    rv_root, default_out = _default_paths(args.dataset)
    rv_root = Path(args.rv_root) if args.rv_root else rv_root
    output_dir = Path(args.output_dir) if args.output_dir else default_out
    output_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(args.config)
    objects = _load_objects(rv_root, args.dataset, args.start, args.end)
    tau_grid = _tau_grid_from_profiles(rv_root)
    objects["tau_center"] = _nearest_tau(objects["life_phase"], tau_grid)
    target_ids = set(objects["eddy3d_object_id"].astype("int64"))
    axis_path = output_dir / "axis_reference_tau_depth.parquet"
    if axis_path.exists():
        axis_ref = pd.read_parquet(axis_path)
        points = _load_points(rv_root, target_ids)
    else:
        all_objects = _load_objects(rv_root, args.dataset, None, None)
        all_objects["tau_center"] = _nearest_tau(all_objects["life_phase"], tau_grid)
        all_points = _load_points(rv_root, set(all_objects["eddy3d_object_id"].astype("int64")))
        points = all_points[all_points["eddy3d_object_id"].astype("int64").isin(target_ids)].copy()
        axis_ref = _axis_reference(rv_root, output_dir, all_objects, all_points, tau_grid)
    psi_profiles = _load_psi_profiles(rv_root, args.dataset, args.r_core)
    log_rows = []
    for day, day_objects in objects.groupby("date", sort=True):
        path = _strict_day(
            config,
            rv_root,
            output_dir,
            pd.Timestamp(day),
            day_objects.copy(),
            points,
            psi_profiles,
            axis_ref,
            args.r_core,
            args.radial_bins,
            args.azimuth_bins,
        )
        log_rows.append({"date": pd.Timestamp(day).strftime("%Y-%m-%d"), "objects": len(day_objects), "path": str(path)})
    pd.DataFrame(log_rows).to_csv(output_dir / f"strict_progress_{args.start or 'all'}_{args.end or 'all'}.csv", index=False)


def run_finalize(args: argparse.Namespace) -> None:
    _, default_out = _default_paths(args.dataset)
    output_dir = Path(args.output_dir) if args.output_dir else default_out
    output_dir.mkdir(parents=True, exist_ok=True)
    df = _load_all_parts(output_dir)
    df.to_parquet(output_dir / "joint_object_error_summary.parquet", index=False)
    df.to_csv(output_dir / "joint_object_error_summary.csv", index=False)
    track, tau = _summaries(df)
    track.to_parquet(output_dir / "joint_track_error_summary.parquet", index=False)
    track.to_csv(output_dir / "joint_track_error_summary.csv", index=False)
    tau.to_parquet(output_dir / "joint_tau_shape_polarity_summary.parquet", index=False)
    tau.to_csv(output_dir / "joint_tau_shape_polarity_summary.csv", index=False)
    _plot_outputs(track, tau, output_dir)
    _write_report(df, track, tau, output_dir, args.dataset)


def main() -> None:
    parser = argparse.ArgumentParser(description="Joint representativeness of ACC representative vortices.")
    parser.add_argument("--dataset", choices=["all_shape", "coherent_only"], required=True)
    parser.add_argument("--mode", choices=["strict", "finalize", "all"], default="all")
    parser.add_argument("--config", default="/root/Verify/config/config_acc_2020_2022_cpu.yaml")
    parser.add_argument("--rv-root", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--r-core", type=float, default=1.5)
    parser.add_argument("--radial-bins", type=int, default=24)
    parser.add_argument("--azimuth-bins", type=int, default=24)
    args = parser.parse_args()
    if args.mode in {"strict", "all"}:
        run_strict(args)
    if args.mode in {"finalize", "all"}:
        run_finalize(args)


if __name__ == "__main__":
    main()
