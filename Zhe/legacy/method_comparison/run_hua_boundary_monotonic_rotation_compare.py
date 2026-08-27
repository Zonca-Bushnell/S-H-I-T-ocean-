from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from netCDF4 import Dataset, num2date


EARTH_RADIUS_M = 6_371_000.0


def _parse_date(value: str) -> date:
    return datetime.strptime(str(value)[:10], "%Y-%m-%d").date()


def _date_range(start: date, end: date) -> list[date]:
    out: list[date] = []
    current = start
    while current <= end:
        out.append(current)
        current += timedelta(days=1)
    return out


def _time_lookup(ds: Dataset) -> dict[date, int]:
    tvar = ds.variables["time"]
    times = num2date(tvar[:], units=tvar.units, calendar=getattr(tvar, "calendar", "standard"))
    return {date(int(t.year), int(t.month), int(t.day)): i for i, t in enumerate(times)}


def _read_parquet_or_csv(stem: Path) -> pd.DataFrame:
    parquet = stem.with_suffix(".parquet")
    csv = stem.with_suffix(".csv")
    if parquet.exists():
        try:
            return pd.read_parquet(parquet, engine="fastparquet")
        except Exception:
            return pd.read_parquet(parquet)
    if csv.exists():
        return pd.read_csv(csv)
    return pd.DataFrame()


def _load_surface_velocity(filter_root: Path, template: str, day: date) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    path = filter_root / template.format(year=day.year)
    if not path.exists():
        raise FileNotFoundError(path)
    with Dataset(path) as ds:
        tidx = _time_lookup(ds)[day]
        lon = np.asarray(ds.variables["longitude"][:], dtype="float64")
        lat = np.asarray(ds.variables["latitude"][:], dtype="float64")
        u = np.asarray(ds.variables["uo_glor"][tidx, 0, :, :], dtype="float64")
        v = np.asarray(ds.variables["vo_glor"][tidx, 0, :, :], dtype="float64")
    u[np.abs(u) > 1e20] = np.nan
    v[np.abs(v) > 1e20] = np.nan
    speed = np.hypot(u, v)
    lon2, lat2 = np.meshgrid(lon, lat)
    return lon2, lat2, u, v, speed


def _run_command(cmd: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(cmd) + "\n\n")
        log.flush()
        subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)


def _detect_command(args: argparse.Namespace, output_dir: Path, *, monotonic: bool) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "src.Location.run_hua_hybrid_detection_acc",
        "--filter-root",
        str(args.filter_root),
        "--raw-root",
        str(args.raw_root),
        "--filter-template",
        args.filter_template,
        "--raw-template",
        args.raw_template,
        "--output-dir",
        str(output_dir),
        "--start",
        args.start,
        "--end",
        args.end,
        "--max-depth-m",
        str(args.max_depth_m),
        "--ssh-window-cells",
        str(args.ssh_window_cells),
        "--max-candidates-per-day",
        str(args.max_candidates_per_day),
        "--surface-search-cells",
        str(args.surface_search_cells),
        "--deep-search-cells",
        str(args.deep_search_cells),
        "--start-radius-cells",
        str(args.start_radius_cells),
        "--max-radius-cells",
        str(args.max_radius_cells),
        "--speed-ratio-max",
        str(args.speed_ratio_max),
        "--angle-jump-max-deg",
        str(args.angle_jump_max_deg),
        "--tangent-tolerance-deg",
        str(args.tangent_tolerance_deg),
        "--symmetry-tolerance-deg",
        str(args.symmetry_tolerance_deg),
        "--min-tangent-fraction",
        str(args.min_tangent_fraction),
        "--min-reversal-fraction",
        str(args.min_reversal_fraction),
        "--min-finite-fraction",
        str(args.min_finite_fraction),
        "--direction-exception-extra",
        str(args.direction_exception_extra),
        "--preload-day-uv",
        "--write-object-voxels",
        "--resume",
    ]
    if args.stop_at_first_failed_layer:
        cmd.append("--stop-at-first-failed-layer")
    if monotonic:
        cmd.extend(
            [
                "--require-boundary-monotonic-rotation",
                "--boundary-monotonic-exception-limit",
                str(args.boundary_monotonic_exception_limit),
            ]
        )
    return cmd


def _tracking_command(input_dir: Path, output_dir: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        "src.Location.run_hua_feature_group_tracking_acc",
        "--input-dir",
        str(input_dir),
        "--output-dir",
        str(output_dir),
    ]


def _summarize_detection(root: Path) -> dict[str, float | int]:
    centers = _read_parquet_or_csv(root / "centers_hua_style")
    objects = _read_parquet_or_csv(root / "frame_object_summary")
    tracks = _read_parquet_or_csv(root / "feature_group_tracking" / "feature_tracks")
    events = _read_parquet_or_csv(root / "feature_group_tracking" / "feature_track_events")
    if centers.empty:
        return {
            "center_rows": 0,
            "pass_layers": 0,
            "surface_candidates": 0,
            "frame_objects": 0,
            "feature_tracks": 0,
            "tracks_ge2": 0,
            "tracks_ge3": 0,
            "max_track_objects": 0,
            "continuous_events": 0,
        }
    pass_layers = centers[centers["hua_pass"].astype(bool)]
    return {
        "center_rows": int(len(centers)),
        "pass_layers": int(len(pass_layers)),
        "surface_candidates": int(centers[centers["depth_index"].eq(0)]["hua_object_id"].nunique()),
        "surface_pass": int(pass_layers[pass_layers["depth_index"].eq(0)]["hua_object_id"].nunique()),
        "frame_objects": int(len(objects)),
        "feature_tracks": int(len(tracks)),
        "tracks_ge2": int((tracks["n_objects"] >= 2).sum()) if not tracks.empty and "n_objects" in tracks else 0,
        "tracks_ge3": int((tracks["n_objects"] >= 3).sum()) if not tracks.empty and "n_objects" in tracks else 0,
        "max_track_objects": int(tracks["n_objects"].max()) if not tracks.empty and "n_objects" in tracks else 0,
        "continuous_events": int(events["event_type"].eq("continuous").sum()) if not events.empty and "event_type" in events else 0,
        "max_depth_median": float(objects["max_depth_m"].median()) if not objects.empty and "max_depth_m" in objects else float("nan"),
    }


def _plot_surface_overlay(args: argparse.Namespace, baseline: Path, strict: Path, fig_dir: Path) -> None:
    day = _parse_date(args.start)
    lon2, lat2, u, v, speed = _load_surface_velocity(Path(args.filter_root), args.filter_template, day)
    base = _read_parquet_or_csv(baseline / "centers_hua_style")
    strict_df = _read_parquet_or_csv(strict / "centers_hua_style")
    base = base[(base["date"].astype(str).eq(day.isoformat())) & base["depth_index"].eq(0) & base["hua_pass"].astype(bool)].copy()
    strict_df = strict_df[(strict_df["date"].astype(str).eq(day.isoformat())) & strict_df["depth_index"].eq(0) & strict_df["hua_pass"].astype(bool)].copy()

    fig, ax = plt.subplots(figsize=(14, 9))
    im = ax.pcolormesh(lon2, lat2, speed, cmap="magma", shading="nearest")
    step = max(1, int(round(max(lon2.shape) / 55)))
    ax.quiver(lon2[::step, ::step], lat2[::step, ::step], u[::step, ::step], v[::step, ::step], color="white", alpha=0.48, scale=8.0, width=0.0012)
    ax.scatter(base["center_lon"], base["center_lat"], marker="o", s=70, facecolors="none", edgecolors="#22c55e", linewidths=2.0, label="baseline Hua b3")
    ax.scatter(strict_df["center_lon"], strict_df["center_lat"], marker="x", s=80, color="#38bdf8", linewidths=2.4, label="with monotonic boundary")
    ax.set_title(f"Kuroshiou surface centers {day.isoformat()}: boundary monotonic rotation constraint")
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.legend(loc="upper right")
    fig.colorbar(im, ax=ax, label="|u', v'| 30-180d (m/s)")
    fig.savefig(fig_dir / "surface_centers_baseline_vs_monotonic.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_depth_impact(baseline: Path, strict: Path, fig_dir: Path) -> None:
    base = _read_parquet_or_csv(baseline / "centers_hua_style")
    strict_df = _read_parquet_or_csv(strict / "centers_hua_style")
    rows = []
    for label, df in [("baseline", base), ("monotonic", strict_df)]:
        if df.empty:
            continue
        passed = df[df["hua_pass"].astype(bool)].copy()
        agg = passed.groupby(["date", "depth_m"], as_index=False).size().rename(columns={"size": "pass_layers"})
        agg["mode"] = label
        rows.append(agg)
    data = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if data.empty:
        return
    pivots = {}
    for label in ["baseline", "monotonic"]:
        sub = data[data["mode"].eq(label)]
        pivots[label] = sub.pivot(index="depth_m", columns="date", values="pass_layers").fillna(0.0).sort_index()
    common_depth = pivots["baseline"].index.union(pivots["monotonic"].index)
    common_date = pivots["baseline"].columns.union(pivots["monotonic"].columns)
    base_mat = pivots["baseline"].reindex(index=common_depth, columns=common_date).fillna(0.0)
    strict_mat = pivots["monotonic"].reindex(index=common_depth, columns=common_date).fillna(0.0)
    diff = strict_mat - base_mat

    fig, axes = plt.subplots(1, 3, figsize=(15, 7), sharey=True)
    mats = [(base_mat, "baseline pass layers"), (strict_mat, "monotonic pass layers"), (diff, "monotonic - baseline")]
    vmax = max(float(base_mat.to_numpy().max()), float(strict_mat.to_numpy().max()), 1.0)
    for ax, (mat, title) in zip(axes, mats):
        cmap = "viridis" if "baseline" in title or "monotonic pass" in title else "coolwarm"
        vmin = 0 if cmap == "viridis" else -vmax
        vmax_use = vmax if cmap == "viridis" else vmax
        im = ax.imshow(mat.to_numpy(), aspect="auto", origin="upper", cmap=cmap, vmin=vmin, vmax=vmax_use)
        ax.set_title(title)
        ax.set_xticks(range(len(mat.columns)))
        ax.set_xticklabels([str(c)[5:] for c in mat.columns], rotation=45, ha="right")
        yticks = np.linspace(0, len(mat.index) - 1, min(8, len(mat.index))).astype(int)
        ax.set_yticks(yticks)
        ax.set_yticklabels([f"{mat.index[i]:.0f}" for i in yticks])
        ax.set_xlabel("date")
        fig.colorbar(im, ax=ax, fraction=0.046)
    axes[0].set_ylabel("depth (m)")
    fig.suptitle("3D detection impact by date and depth")
    fig.savefig(fig_dir / "date_depth_pass_layer_impact.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_axis_compare(baseline: Path, strict: Path, fig_dir: Path) -> None:
    base = _read_parquet_or_csv(baseline / "centers_hua_style")
    strict_df = _read_parquet_or_csv(strict / "centers_hua_style")
    if base.empty or strict_df.empty:
        return
    ranked = (
        base[base["hua_pass"].astype(bool)]
        .groupby("hua_object_id")
        .agg(n_pass=("depth_index", "size"), date=("date", "first"), polarity=("polarity", "first"))
        .sort_values("n_pass", ascending=False)
        .head(4)
        .reset_index()
    )
    if ranked.empty:
        return
    fig = plt.figure(figsize=(13, 10))
    for panel, row in enumerate(ranked.to_dict("records"), start=1):
        oid = row["hua_object_id"]
        ax = fig.add_subplot(2, 2, panel, projection="3d")
        for label, df, color in [("baseline", base, "#22c55e"), ("monotonic", strict_df, "#38bdf8")]:
            obj = df[df["hua_object_id"].astype(str).eq(str(oid)) & df["hua_pass"].astype(bool)].sort_values("depth_index")
            if obj.empty:
                continue
            z = -obj["depth_m"].to_numpy(dtype="float64") / 1000.0
            x = obj["center_x_from_seed_km"].to_numpy(dtype="float64")
            y = obj["center_y_from_seed_km"].to_numpy(dtype="float64")
            ax.plot(x, y, z, color=color, linewidth=2.2, label=f"{label}, n={len(obj)}")
            ax.scatter(x, y, z, color=color, s=28)
        ax.scatter([0], [0], [0], marker="+", c="red", s=140, linewidth=3, label="SSH seed")
        ax.set_title(f"{oid}\n{row['polarity']}, {row['date']}")
        ax.set_xlabel("east km")
        ax.set_ylabel("north km")
        ax.set_zlabel("depth km")
        ax.view_init(elev=24, azim=-55)
        ax.legend(fontsize=8)
    fig.suptitle("3D Hua axes: baseline vs boundary-monotonic constraint")
    fig.savefig(fig_dir / "3d_axis_baseline_vs_monotonic.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_tracking_compare(base_root: Path, strict_root: Path, fig_dir: Path) -> None:
    base_tracks = _read_parquet_or_csv(base_root / "feature_group_tracking" / "feature_tracks")
    strict_tracks = _read_parquet_or_csv(strict_root / "feature_group_tracking" / "feature_tracks")
    labels = ["baseline", "monotonic"]
    values = []
    for tracks in [base_tracks, strict_tracks]:
        if tracks.empty:
            values.append([0, 0, 0, 0])
        else:
            values.append(
                [
                    int(len(tracks)),
                    int((tracks["n_objects"] >= 2).sum()),
                    int((tracks["n_objects"] >= 3).sum()),
                    int(tracks["n_objects"].max()),
                ]
            )
    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(4)
    w = 0.35
    for i, (label, color) in enumerate(zip(labels, ["#22c55e", "#38bdf8"])):
        ax.bar(x + (i - 0.5) * w, values[i], width=w, label=label, color=color)
    ax.set_xticks(x)
    ax.set_xticklabels(["tracks", "tracks >=2", "tracks >=3", "max objects"])
    ax.set_ylabel("count")
    ax.set_title("Tracking impact of boundary monotonic rotation constraint")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(fig_dir / "tracking_metric_impact.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.3), sharex=True, sharey=True)
    for ax, tracks, root, title in zip(axes, [base_tracks, strict_tracks], [base_root, strict_root], labels):
        points = _read_parquet_or_csv(root / "feature_group_tracking" / "feature_track_points")
        if not points.empty and "feature_track_id" in points:
            for _, group in points.groupby("feature_track_id"):
                if len(group) < 2:
                    continue
                ax.plot(group["surface_center_lon"], group["surface_center_lat"], color="#334155", alpha=0.42, linewidth=1.1)
                ax.scatter(group["surface_center_lon"], group["surface_center_lat"], s=10 + group["pass_layers"], color="#7c3aed", alpha=0.72)
        ax.set_title(title)
        ax.set_xlabel("longitude")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("latitude")
    fig.suptitle("Feature-track maps from object-voxel overlap")
    fig.savefig(fig_dir / "feature_track_map_baseline_vs_monotonic.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def _write_summary(output_dir: Path, baseline: dict[str, float | int], strict: dict[str, float | int], args: argparse.Namespace) -> None:
    ratio = {}
    for key, base_value in baseline.items():
        strict_value = strict.get(key, 0)
        if isinstance(base_value, (int, float)) and isinstance(strict_value, (int, float)) and base_value:
            ratio[key] = float(strict_value) / float(base_value)
    payload = {"baseline": baseline, "monotonic": strict, "monotonic_over_baseline": ratio, "parameters": vars(args)}
    (output_dir / "boundary_monotonic_comparison_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    lines = [
        "# Kuroshiou Hua b3 边界速度向量单调旋转约束对比",
        "",
        "本实验只改一个条件：在当前 Hua b3 圆周速度判据之外，额外要求边界速度向量沿圆周单调旋转。baseline 保持当前生产参数；monotonic 只增加该硬约束。",
        "",
        "## 核心结果",
        "",
        "| 指标 | baseline | monotonic | monotonic / baseline |",
        "|---|---:|---:|---:|",
    ]
    keys = ["pass_layers", "surface_pass", "frame_objects", "feature_tracks", "tracks_ge2", "tracks_ge3", "max_track_objects", "continuous_events", "max_depth_median"]
    for key in keys:
        b = baseline.get(key, 0)
        s = strict.get(key, 0)
        r = ratio.get(key, float("nan"))
        lines.append(f"| `{key}` | {b} | {s} | {r:.3f} |")
    lines.extend(
        [
            "",
            "## 解释",
            "",
            "- 如果 `pass_layers` 明显下降，说明很多原本通过的层依赖“允许少量方向异常”的 Hua/Nencioli 容错。",
            "- 如果 `feature_tracks` 或 `tracks_ge2` 明显下降，说明该约束不仅筛掉单层，还会破坏相邻帧对象体素 overlap 的连续性。",
            "- 如果 `max_depth_median` 下降，说明它主要切掉深层弱旋转或开口/月牙状速度结构；这正是你关心的“强区呈月牙、速度弱区开口”的情形。",
        ]
    )
    (output_dir / "boundary_monotonic_comparison_summary_zh.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_dir = output_dir / "baseline"
    strict_dir = output_dir / "boundary_monotonic"
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    if not args.analysis_only:
        for label, target, monotonic in [("baseline", baseline_dir, False), ("boundary_monotonic", strict_dir, True)]:
            if args.force and target.exists():
                raise SystemExit(f"Refusing to delete existing output automatically: {target}")
            _run_command(_detect_command(args, target, monotonic=monotonic), output_dir / "logs" / f"{label}_detection.log")
            _run_command(_tracking_command(target, target / "feature_group_tracking"), output_dir / "logs" / f"{label}_tracking.log")

    baseline_summary = _summarize_detection(baseline_dir)
    strict_summary = _summarize_detection(strict_dir)
    _plot_surface_overlay(args, baseline_dir, strict_dir, figures)
    _plot_depth_impact(baseline_dir, strict_dir, figures)
    _plot_axis_compare(baseline_dir, strict_dir, figures)
    _plot_tracking_compare(baseline_dir, strict_dir, figures)
    _write_summary(output_dir, baseline_summary, strict_summary, args)
    print(json.dumps({"baseline": baseline_summary, "monotonic": strict_summary}, ensure_ascii=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Hua b3 with/without boundary velocity-vector monotonic rotation.")
    parser.add_argument("--filter-root", default="/root/autodl-fs/kuroshiou/Filter")
    parser.add_argument("--raw-root", default="/root/autodl-fs/kuroshiou/raw")
    parser.add_argument("--filter-template", default="global_phy_{year}_bandpass_30_180d.nc")
    parser.add_argument("--raw-template", default="global_phy_{year}.nc")
    parser.add_argument("--output-dir", default="/root/autodl-fs/kuroshiou/experiments/hua_boundary_monotonic_19930101_19930107")
    parser.add_argument("--start", default="1993-01-01")
    parser.add_argument("--end", default="1993-01-07")
    parser.add_argument("--max-depth-m", type=float, default=2000.0)
    parser.add_argument("--ssh-window-cells", type=int, default=7)
    parser.add_argument("--max-candidates-per-day", type=int, default=0)
    parser.add_argument("--surface-search-cells", type=int, default=3)
    parser.add_argument("--deep-search-cells", type=int, default=3)
    parser.add_argument("--start-radius-cells", type=int, default=2)
    parser.add_argument("--max-radius-cells", type=int, default=8)
    parser.add_argument("--speed-ratio-max", type=float, default=3.0)
    parser.add_argument("--angle-jump-max-deg", type=float, default=150.0)
    parser.add_argument("--tangent-tolerance-deg", type=float, default=24.0)
    parser.add_argument("--symmetry-tolerance-deg", type=float, default=120.0)
    parser.add_argument("--min-tangent-fraction", type=float, default=0.55)
    parser.add_argument("--min-reversal-fraction", type=float, default=0.55)
    parser.add_argument("--min-finite-fraction", type=float, default=0.75)
    parser.add_argument("--direction-exception-extra", type=int, default=2)
    parser.add_argument("--boundary-monotonic-exception-limit", type=int, default=0)
    parser.add_argument("--stop-at-first-failed-layer", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--analysis-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
