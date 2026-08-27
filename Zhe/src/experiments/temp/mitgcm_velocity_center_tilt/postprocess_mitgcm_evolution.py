"""Postprocess MITgcm velocity-center tilt validation runs.

This reads MITgcm MDS binary state files from the four idealized cases and
extracts velocity-speed-minimum centerlines in the aligned representative frame.
It is intentionally local to this temporary experiment package.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CASES = ("real", "mode1", "mode2", "mode1_plus_mode2", "mode1_to_5")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="/root/autodl-fs/kuroshiou_mitgcm_velocity_center_tilt_validation")
    parser.add_argument("--core-rmax", type=float, default=1.75)
    return parser.parse_args()


def _read_big_endian(path: Path, shape: tuple[int, ...]) -> np.ndarray:
    data = np.fromfile(path, dtype=">f8")
    expected = int(np.prod(shape))
    if data.size != expected:
        raise ValueError(f"{path} has {data.size} values, expected {expected}")
    return data.reshape(shape).astype("f8", copy=False)


def _read_vector(path: Path) -> np.ndarray:
    return np.fromfile(path, dtype=">f8").astype("f8", copy=False)


def _iteration_from_name(path: Path) -> int:
    match = re.search(r"\.(\d{10})\.data$", path.name)
    if not match:
        raise ValueError(path.name)
    return int(match.group(1))


def _grid_from_run(run_dir: Path, manifest: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    nx = int(manifest["grid"]["nx"])
    ny = int(manifest["grid"]["ny"])
    nz = int(manifest["grid"]["nz"])
    radius_m = float(manifest["radius_m"])
    depth = -_read_vector(run_dir / "RC.data")
    if depth.size != nz:
        raise ValueError(f"RC depth count {depth.size} != nz {nz}")
    dxc = _read_big_endian(run_dir / "DXC.data", (ny, nx))
    dyc = _read_big_endian(run_dir / "DYC.data", (ny, nx))
    dx = float(np.nanmedian(dxc))
    dy = float(np.nanmedian(dyc))
    x_m = (np.arange(nx, dtype="f8") - 0.5 * (nx - 1)) * dx
    y_m = (np.arange(ny, dtype="f8") - 0.5 * (ny - 1)) * dy
    return x_m / radius_m, y_m / radius_m, depth, radius_m


def _centerline_from_uv(u: np.ndarray, v: np.ndarray, x_over_r: np.ndarray, y_over_r: np.ndarray, rmax: float) -> pd.DataFrame:
    speed = np.hypot(u, v)
    xx, yy = np.meshgrid(x_over_r, y_over_r)
    mask = np.hypot(xx, yy) <= float(rmax)
    rows = []
    for k in range(speed.shape[0]):
        layer = np.where(mask, speed[k], np.nan)
        if not np.any(np.isfinite(layer)):
            rows.append((k, np.nan, np.nan, np.nan))
            continue
        idx = int(np.nanargmin(layer))
        j, i = np.unravel_index(idx, layer.shape)
        rows.append((k, float(x_over_r[i]), float(y_over_r[j]), float(layer[j, i])))
    return pd.DataFrame(rows, columns=["depth_index", "x_over_R", "y_over_R", "speed_min_ms"])


def _tilt_distance(x: np.ndarray, y: np.ndarray) -> float:
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 2:
        return np.nan
    x0 = float(x[valid][0])
    y0 = float(y[valid][0])
    return float(np.nanmax(np.hypot(x[valid] - x0, y[valid] - y0)))


def _rmse_xy(a: pd.DataFrame, b: pd.DataFrame) -> float:
    joined = a.merge(b, on="depth_index", suffixes=("_a", "_b"))
    valid = (
        np.isfinite(joined["x_over_R_a"])
        & np.isfinite(joined["y_over_R_a"])
        & np.isfinite(joined["x_over_R_b"])
        & np.isfinite(joined["y_over_R_b"])
    )
    if int(valid.sum()) < 2:
        return np.nan
    dx = joined.loc[valid, "x_over_R_a"].to_numpy() - joined.loc[valid, "x_over_R_b"].to_numpy()
    dy = joined.loc[valid, "y_over_R_a"].to_numpy() - joined.loc[valid, "y_over_R_b"].to_numpy()
    return float(np.sqrt(np.nanmean(dx * dx + dy * dy)))


def _corr_xy(a: pd.DataFrame, b: pd.DataFrame) -> float:
    joined = a.merge(b, on="depth_index", suffixes=("_a", "_b"))
    valid = (
        np.isfinite(joined["x_over_R_a"])
        & np.isfinite(joined["y_over_R_a"])
        & np.isfinite(joined["x_over_R_b"])
        & np.isfinite(joined["y_over_R_b"])
    )
    if int(valid.sum()) < 3:
        return np.nan
    avec = joined.loc[valid, ["x_over_R_a", "y_over_R_a"]].to_numpy().ravel()
    bvec = joined.loc[valid, ["x_over_R_b", "y_over_R_b"]].to_numpy().ravel()
    if np.nanstd(avec) <= 0 or np.nanstd(bvec) <= 0:
        return np.nan
    return float(np.corrcoef(avec, bvec)[0, 1])


def _plot_tilt(summary: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    for case, part in summary.groupby("case", sort=False):
        ax.plot(part["day"], part["tilt_distance_over_R"], marker="o", markersize=2, linewidth=1.5, label=str(case))
    ax.set_xlabel("model day")
    ax.set_ylabel("velocity-center tilt distance / R")
    ax.set_title("MITgcm velocity-center tilt evolution")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.savefig(out_dir / "tilt_distance_evolution.png", dpi=180)
    plt.close(fig)


def _plot_rmse(metrics: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    for case, part in metrics.groupby("case", sort=False):
        if case == "real":
            continue
        ax.plot(part["day"], part["rmse_vs_real_over_R"], marker="o", markersize=2, linewidth=1.5, label=str(case))
    ax.set_xlabel("model day")
    ax.set_ylabel("centerline RMSE vs real / R")
    ax.set_title("Modal reconstruction error relative to real velocity-center evolution")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.savefig(out_dir / "modal_centerline_rmse_vs_real.png", dpi=180)
    plt.close(fig)


def _write_summary(metrics: pd.DataFrame, out_dir: Path) -> None:
    final = metrics.sort_values("day").groupby("case", as_index=False).tail(1)
    mean = metrics.groupby("case", as_index=False).agg(
        mean_rmse_vs_real_over_R=("rmse_vs_real_over_R", "mean"),
        mean_corr_vs_real=("corr_vs_real", "mean"),
        mean_tilt_distance_over_R=("tilt_distance_over_R", "mean"),
    )
    lines = [
        "# MITgcm 模态倾斜演化验证结果",
        "",
        "口径：所有中心均为速度异常场的速度弱中心，坐标为 global_ls_alpha 对齐后的代表涡坐标。",
        "",
        "## 最后一天指标",
        "",
        "```csv",
        final.to_csv(index=False).strip(),
        "```",
        "",
        "## 60 天平均指标",
        "",
        "```csv",
        mean.to_csv(index=False).strip(),
        "```",
        "",
        "解释：如果 `mode1_plus_mode2` 的 `rmse_vs_real` 显著低于单独 `mode1`/`mode2`，且 `corr_vs_real` 更高，才支持“mode1+mode2 传播差异解释速度中心倾斜”。如果仍然偏大，则需要加入更多模态或非模态/非线性结构。",
    ]
    (out_dir / "mitgcm_mode_tilt_evolution_summary_zh.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = _parse_args()
    root = Path(args.output_root)
    diag_dir = root / "diagnostics" / "mitgcm_evolution"
    fig_dir = root / "figures" / "mitgcm_evolution"
    diag_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((root / "experiment_manifest.json").read_text(encoding="utf-8"))
    x_over_r, y_over_r, depth, radius_m = _grid_from_run(root / "runs" / "real" / "run", manifest)
    center_parts = []
    for case in CASES:
        run = root / "runs" / case / "run"
        files = sorted(run.glob("U.*.data"), key=_iteration_from_name)
        for u_path in files:
            it = _iteration_from_name(u_path)
            v_path = run / f"V.{it:010d}.data"
            if not v_path.exists():
                continue
            u = _read_big_endian(u_path, (depth.size, y_over_r.size, x_over_r.size))
            v = _read_big_endian(v_path, (depth.size, y_over_r.size, x_over_r.size))
            centers = _centerline_from_uv(u, v, x_over_r, y_over_r, args.core_rmax)
            centers.insert(0, "depth_m", depth[centers["depth_index"].to_numpy(dtype=int)])
            centers.insert(0, "day", it / 1440.0)
            centers.insert(0, "iteration", it)
            centers.insert(0, "case", case)
            center_parts.append(centers)
    centers = pd.concat(center_parts, ignore_index=True)
    centers.to_parquet(diag_dir / "velocity_centerlines_evolution.parquet", index=False)
    centers.to_csv(diag_dir / "velocity_centerlines_evolution.csv", index=False)
    summary = (
        centers.groupby(["case", "iteration", "day"], as_index=False)
        .apply(lambda g: pd.Series({"tilt_distance_over_R": _tilt_distance(g["x_over_R"].to_numpy(), g["y_over_R"].to_numpy())}))
        .reset_index(drop=True)
    )
    metric_rows = []
    for it, by_it in centers.groupby("iteration", sort=True):
        real = by_it[by_it["case"].eq("real")]
        for case, part in by_it.groupby("case", sort=False):
            metric_rows.append(
                {
                    "case": case,
                    "iteration": int(it),
                    "day": float(it / 1440.0),
                    "rmse_vs_real_over_R": 0.0 if case == "real" else _rmse_xy(part, real),
                    "corr_vs_real": 1.0 if case == "real" else _corr_xy(part, real),
                    "tilt_distance_over_R": _tilt_distance(part["x_over_R"].to_numpy(), part["y_over_R"].to_numpy()),
                }
            )
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(diag_dir / "modal_evolution_metrics.csv", index=False)
    metrics.to_parquet(diag_dir / "modal_evolution_metrics.parquet", index=False)
    _plot_tilt(summary, fig_dir)
    _plot_rmse(metrics, fig_dir)
    _write_summary(metrics, diag_dir)
    print(f"Wrote {diag_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
