"""Postprocess MITgcm runs with velocity-inverted streamfunction centers.

This experiment does not use temperature centers.  It reads existing MITgcm
U/V MDS output, computes relative vorticity, inverts a 2-D streamfunction for
each depth layer stack, and extracts centerlines from streamfunction extrema.
The result is kept separate from the velocity-speed-minimum diagnostics so the
center-definition sensitivity is explicit.
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

from src.Legacy.experiments.theory_validation.unified_math import streamfunction_from_zeta


CASES = ("real", "mode1", "mode2", "mode1_plus_mode2", "mode1_to_5")
REFERENCE_CASE = "real"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="/root/autodl-fs/kuroshiou_mitgcm_velocity_center_tilt_validation")
    parser.add_argument("--core-rmax", type=float, default=1.75)
    parser.add_argument("--smoke-days", default="", help="Optional comma-separated model days, e.g. 0,30,60.")
    parser.add_argument("--no-parquet", action="store_true", help="Write CSV/JSON/PNG only.")
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
    return x_m, y_m, depth, radius_m


def _relative_vorticity(u: np.ndarray, v: np.ndarray, x_m: np.ndarray, y_m: np.ndarray) -> np.ndarray:
    dvdx = np.gradient(v, x_m, axis=2, edge_order=1)
    dudy = np.gradient(u, y_m, axis=1, edge_order=1)
    return dvdx - dudy


def _core_mask(x_over_r: np.ndarray, y_over_r: np.ndarray, rmax: float) -> np.ndarray:
    xx, yy = np.meshgrid(x_over_r, y_over_r)
    return np.hypot(xx, yy) <= float(rmax)


def _centerline_from_scalar(
    field: np.ndarray,
    x_over_r: np.ndarray,
    y_over_r: np.ndarray,
    mask: np.ndarray,
    mode: str,
) -> pd.DataFrame:
    rows = []
    for k in range(field.shape[0]):
        layer = np.where(mask, field[k], np.nan)
        if not np.any(np.isfinite(layer)):
            rows.append((k, np.nan, np.nan, np.nan))
            continue
        if mode == "abs_max":
            idx = int(np.nanargmax(np.abs(layer)))
        elif mode == "signed_max":
            idx = int(np.nanargmax(layer))
        elif mode == "signed_min":
            idx = int(np.nanargmin(layer))
        elif mode == "speed_min":
            idx = int(np.nanargmin(layer))
        else:
            raise ValueError(f"Unsupported center mode: {mode}")
        j, i = np.unravel_index(idx, layer.shape)
        rows.append((k, float(x_over_r[i]), float(y_over_r[j]), float(layer[j, i])))
    return pd.DataFrame(rows, columns=["depth_index", "x_over_R", "y_over_R", "center_value"])


def _extract_centers(
    u: np.ndarray,
    v: np.ndarray,
    x_m: np.ndarray,
    y_m: np.ndarray,
    radius_m: float,
    core_rmax: float,
) -> tuple[pd.DataFrame, np.ndarray]:
    x_over_r = x_m / radius_m
    y_over_r = y_m / radius_m
    mask = _core_mask(x_over_r, y_over_r, core_rmax)
    zeta = _relative_vorticity(u, v, x_m, y_m)
    psi = streamfunction_from_zeta(zeta, y_m, x_m)
    definitions = {
        "psi_abs_extreme": (psi, "abs_max"),
        "psi_signed_max": (psi, "signed_max"),
        "psi_signed_min": (psi, "signed_min"),
        "speed_min": (np.hypot(u, v), "speed_min"),
    }
    parts = []
    for name, (field, mode) in definitions.items():
        part = _centerline_from_scalar(field, x_over_r, y_over_r, mask, mode)
        part.insert(0, "center_definition", name)
        parts.append(part)
    return pd.concat(parts, ignore_index=True), psi


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


def _center_part(centers: pd.DataFrame, definition: str) -> pd.DataFrame:
    return centers[centers["center_definition"].eq(definition)].copy()


def _metrics_for_definition(centers: pd.DataFrame, definition: str) -> pd.DataFrame:
    rows = []
    subset = centers[centers["center_definition"].eq(definition)].copy()
    for (case, iteration, day), part in subset.groupby(["case", "iteration", "day"], sort=True):
        rows.append(
            {
                "case": case,
                "iteration": int(iteration),
                "day": float(day),
                "center_definition": definition,
                "tilt_distance_over_R": _tilt_distance(
                    part["x_over_R"].to_numpy(dtype="f8"),
                    part["y_over_R"].to_numpy(dtype="f8"),
                ),
            }
        )
    metrics = pd.DataFrame(rows)
    cmp_rows = []
    for iteration, by_it in subset.groupby("iteration", sort=True):
        real = by_it[by_it["case"].eq(REFERENCE_CASE)]
        if real.empty:
            continue
        for case, part in by_it.groupby("case", sort=False):
            cmp_rows.append(
                {
                    "case": case,
                    "iteration": int(iteration),
                    "day": float(part["day"].iloc[0]),
                    "center_definition": definition,
                    "rmse_vs_real_over_R": 0.0 if case == REFERENCE_CASE else _rmse_xy(part, real),
                    "corr_vs_real": 1.0 if case == REFERENCE_CASE else _corr_xy(part, real),
                }
            )
    compare = pd.DataFrame(cmp_rows)
    return metrics.merge(compare, on=["case", "iteration", "day", "center_definition"], how="left")


def _sensitivity_metrics(centers: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (case, iteration, day), group in centers.groupby(["case", "iteration", "day"], sort=True):
        psi = _center_part(group, "psi_abs_extreme")
        speed = _center_part(group, "speed_min")
        rows.append(
            {
                "case": case,
                "iteration": int(iteration),
                "day": float(day),
                "psi_vs_speed_rmse_over_R": _rmse_xy(psi, speed),
                "psi_vs_speed_corr": _corr_xy(psi, speed),
                "psi_tilt_distance_over_R": _tilt_distance(
                    psi["x_over_R"].to_numpy(dtype="f8"),
                    psi["y_over_R"].to_numpy(dtype="f8"),
                ),
                "speed_tilt_distance_over_R": _tilt_distance(
                    speed["x_over_R"].to_numpy(dtype="f8"),
                    speed["y_over_R"].to_numpy(dtype="f8"),
                ),
            }
        )
    return pd.DataFrame(rows)


def _judge(metrics: pd.DataFrame) -> dict[str, object]:
    main = metrics[metrics["center_definition"].eq("psi_abs_extreme") & metrics["day"].gt(0)].copy()
    mean = main.groupby("case", as_index=False).agg(
        mean_rmse_vs_real_over_R=("rmse_vs_real_over_R", "mean"),
        mean_corr_vs_real=("corr_vs_real", "mean"),
        mean_tilt_distance_over_R=("tilt_distance_over_R", "mean"),
    )
    by_case = {row.case: row for row in mean.itertuples(index=False)}

    def value(case: str, field: str) -> float:
        row = by_case.get(case)
        return float(getattr(row, field)) if row is not None else np.nan

    rmse12 = value("mode1_plus_mode2", "mean_rmse_vs_real_over_R")
    rmse1 = value("mode1", "mean_rmse_vs_real_over_R")
    rmse2 = value("mode2", "mean_rmse_vs_real_over_R")
    corr12 = value("mode1_plus_mode2", "mean_corr_vs_real")
    corr1 = value("mode1", "mean_corr_vs_real")
    corr2 = value("mode2", "mean_corr_vs_real")
    rmse15 = value("mode1_to_5", "mean_rmse_vs_real_over_R")

    mode12_rmse_gain = min((rmse1 - rmse12) / rmse1, (rmse2 - rmse12) / rmse2) if rmse1 > 0 and rmse2 > 0 else np.nan
    mode12_corr_gain = min(corr12 - corr1, corr12 - corr2) if np.isfinite(corr12) else np.nan
    high_mode_gain = (rmse12 - rmse15) / rmse12 if rmse12 > 0 and np.isfinite(rmse15) else np.nan
    mode12_supported = bool(np.isfinite(mode12_rmse_gain) and mode12_rmse_gain >= 0.20 and mode12_corr_gain >= 0.20)
    low_modes_sufficient = bool(np.isfinite(high_mode_gain) and high_mode_gain < 0.10)
    if mode12_supported and low_modes_sufficient:
        verdict = "support"
    elif mode12_supported or (np.isfinite(mode12_rmse_gain) and mode12_rmse_gain > 0):
        verdict = "partial"
    else:
        verdict = "fail"
    return {
        "verdict": verdict,
        "mode12_supported": mode12_supported,
        "low_modes_sufficient": low_modes_sufficient,
        "mode12_rmse_gain_min_vs_single_modes": mode12_rmse_gain,
        "mode12_corr_gain_min_vs_single_modes": mode12_corr_gain,
        "mode1_to_5_rmse_gain_vs_mode12": high_mode_gain,
        "mean_metrics_excluding_day0": mean.to_dict(orient="records"),
    }


def _plot_tilt(metrics: pd.DataFrame, out_dir: Path) -> None:
    main = metrics[metrics["center_definition"].eq("psi_abs_extreme")]
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    for case, part in main.groupby("case", sort=False):
        ax.plot(part["day"], part["tilt_distance_over_R"], marker="o", markersize=2, linewidth=1.5, label=str(case))
    ax.set_xlabel("model day")
    ax.set_ylabel("psi-center tilt distance / R")
    ax.set_title("MITgcm velocity-inverted streamfunction-center tilt evolution")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.savefig(out_dir / "psi_tilt_distance_evolution.png", dpi=180)
    plt.close(fig)


def _plot_rmse(metrics: pd.DataFrame, out_dir: Path) -> None:
    main = metrics[metrics["center_definition"].eq("psi_abs_extreme")]
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    for case, part in main.groupby("case", sort=False):
        if case == REFERENCE_CASE:
            continue
        ax.plot(part["day"], part["rmse_vs_real_over_R"], marker="o", markersize=2, linewidth=1.5, label=str(case))
    ax.set_xlabel("model day")
    ax.set_ylabel("psi centerline RMSE vs real / R")
    ax.set_title("Modal reconstruction error relative to real psi-center evolution")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.savefig(out_dir / "psi_modal_centerline_rmse_vs_real.png", dpi=180)
    plt.close(fig)


def _plot_sensitivity(sensitivity: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    for case, part in sensitivity.groupby("case", sort=False):
        axes[0].plot(part["day"], part["psi_vs_speed_rmse_over_R"], marker="o", markersize=2, linewidth=1.3, label=str(case))
        axes[1].plot(part["day"], part["psi_tilt_distance_over_R"], linewidth=1.5, label=f"{case} psi")
        axes[1].plot(part["day"], part["speed_tilt_distance_over_R"], linestyle="--", linewidth=1.0, label=f"{case} speed")
    axes[0].set_xlabel("model day")
    axes[0].set_ylabel("psi-center vs speed-min RMSE / R")
    axes[0].set_title("Center-definition displacement")
    axes[1].set_xlabel("model day")
    axes[1].set_ylabel("tilt distance / R")
    axes[1].set_title("Psi center vs speed-min tilt")
    for ax in axes:
        ax.grid(True, alpha=0.25)
    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=7, ncol=2)
    fig.savefig(out_dir / "center_definition_sensitivity.png", dpi=180)
    plt.close(fig)


def _plot_3d_center_definition(centers: pd.DataFrame, out_dir: Path) -> None:
    available_days = sorted(float(v) for v in centers["day"].dropna().unique())
    if not available_days:
        return
    day = max(available_days)
    subset = centers[
        centers["day"].eq(day)
        & centers["center_definition"].isin(["psi_abs_extreme", "speed_min"])
        & centers["case"].isin(["real", "mode1_plus_mode2", "mode1_to_5"])
    ].copy()
    if subset.empty:
        return
    fig = plt.figure(figsize=(9, 7), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    colors = {"real": "k", "mode1_plus_mode2": "C2", "mode1_to_5": "C3"}
    styles = {"psi_abs_extreme": "-", "speed_min": "--"}
    labels = {"psi_abs_extreme": "psi center", "speed_min": "speed-min center"}
    for (case, definition), part in subset.groupby(["case", "center_definition"], sort=False):
        part = part.sort_values("depth_m")
        ax.plot(
            part["x_over_R"],
            part["y_over_R"],
            -part["depth_m"] / 1000.0,
            color=colors.get(str(case), "0.5"),
            linestyle=styles.get(str(definition), "-"),
            linewidth=2.0 if definition == "psi_abs_extreme" else 1.4,
            label=f"{case}: {labels.get(str(definition), definition)}",
        )
    ax.scatter([0], [0], [0], marker="+", s=120, color="red", linewidths=2.0, label="surface reference")
    ax.set_xlabel("x / R")
    ax.set_ylabel("y / R")
    ax.set_zlabel("depth (km, down negative)")
    ax.set_title(f"Psi-center vs speed-min centerlines, model day {day:.0f}")
    ax.legend(fontsize=8, loc="best")
    ax.view_init(elev=23, azim=-58)
    fig.savefig(out_dir / "psi_vs_speed_min_centerline_3d.png", dpi=180)
    plt.close(fig)


def _plot_examples(centers: pd.DataFrame, out_dir: Path) -> None:
    available_days = sorted(float(v) for v in centers["day"].dropna().unique())
    targets = [0.0, 30.0, 60.0]
    for target in targets:
        if not available_days:
            continue
        day = min(available_days, key=lambda value: abs(value - target))
        subset = centers[
            centers["day"].eq(day)
            & centers["center_definition"].eq("psi_abs_extreme")
            & centers["case"].isin(["real", "mode1", "mode2", "mode1_plus_mode2", "mode1_to_5"])
        ].copy()
        if subset.empty:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(10, 5.5), sharey=True, constrained_layout=True)
        for case, part in subset.groupby("case", sort=False):
            axes[0].plot(part["x_over_R"], part["depth_m"], marker="o", markersize=2, linewidth=1.4, label=str(case))
            axes[1].plot(part["y_over_R"], part["depth_m"], marker="o", markersize=2, linewidth=1.4, label=str(case))
        for ax, xlabel in zip(axes, ("x / R", "y / R")):
            ax.axvline(0.0, color="0.7", linewidth=0.8)
            ax.invert_yaxis()
            ax.set_xlabel(xlabel)
            ax.grid(True, alpha=0.25)
        axes[0].set_ylabel("depth (m)")
        axes[1].legend(fontsize=8)
        fig.suptitle(f"Psi-center centerlines, model day {day:.0f}")
        fig.savefig(out_dir / f"mode12_vs_real_psi_centerline_examples_day{day:.0f}.png", dpi=180)
        plt.close(fig)


def _write_summary(diag_dir: Path, judgment: dict[str, object], metrics: pd.DataFrame, sensitivity: pd.DataFrame) -> None:
    main_mean = pd.DataFrame(judgment["mean_metrics_excluding_day0"])
    sens_mean = sensitivity[sensitivity["day"].gt(0)].groupby("case", as_index=False).agg(
        mean_psi_vs_speed_rmse_over_R=("psi_vs_speed_rmse_over_R", "mean"),
        mean_psi_vs_speed_corr=("psi_vs_speed_corr", "mean"),
        mean_psi_tilt_distance_over_R=("psi_tilt_distance_over_R", "mean"),
        mean_speed_tilt_distance_over_R=("speed_tilt_distance_over_R", "mean"),
    )
    verdict_map = {
        "support": "鏀寔",
        "partial": "閮ㄥ垎鏀寔",
        "fail": "涓嶆敮鎸?,
    }
    lines = [
        "# 閫熷害鍙嶆紨娴佸嚱鏁颁腑蹇冮獙璇?Yang/Xu/Li 2026 妯℃€佸€炬枩鏈哄埗",
        "",
        "鏈瘖鏂彧浣跨敤 MITgcm 杈撳嚭鐨?`U,V`銆傛祦绋嬫槸 `U,V -> 鐩稿娑″害 zeta -> 娴佸嚱鏁?psi`锛屽啀鐢?`abs(psi)` 鏋佸€煎畾涔変腑蹇冪嚎锛涙病鏈変娇鐢ㄦ俯搴︿腑蹇冦€?,
        "",
        f"鎬诲垽瀹氾細**{verdict_map.get(str(judgment['verdict']), judgment['verdict'])}**銆?,
        "",
        "## 鍒ゆ嵁缁撴灉",
        "",
        f"- mode1+2 鐩稿鍗曟ā鎬佺殑鏈€灏?RMSE 鏀瑰杽锛歚{float(judgment['mode12_rmse_gain_min_vs_single_modes']):.3f}`銆?,
        f"- mode1+2 鐩稿鍗曟ā鎬佺殑鏈€灏忕浉鍏虫彁鍗囷細`{float(judgment['mode12_corr_gain_min_vs_single_modes']):.3f}`銆?,
        f"- mode1..5 鐩稿 mode1+2 鐨?RMSE 鏀瑰杽锛歚{float(judgment['mode1_to_5_rmse_gain_vs_mode12']):.3f}`銆?,
        "",
        "鍒ゅ畾瑙勫垯锛歮ode1+2 蹇呴』姣?mode1 鍜?mode2 鐨?60 澶╁钩鍧?RMSE 閮戒綆鑷冲皯 20%锛屼笖鐩稿叧閮芥彁楂樿嚦灏?0.2锛屾墠绠楁敮鎸侊紱mode1..5 鑻ュ彧姣?mode1+2 鏀瑰杽灏忎簬 10%锛屾墠璇存槑浣庢ā鎬佽冻澶熴€?,
        "",
        "## 60 澶╁钩鍧囨寚鏍囷紙鍘绘帀 day 0锛?,
        "",
        "```csv",
        main_mean.to_csv(index=False).strip(),
        "```",
        "",
        "## 涓績瀹氫箟鏁忔劅鎬?,
        "",
        "```csv",
        sens_mean.to_csv(index=False).strip(),
        "```",
        "",
        "瑙ｉ噴锛氬鏋?`psi_abs_extreme` 鏀寔 mode1+2锛岃€?`speed_min` 涓嶆敮鎸侊紝璇存槑璁烘枃鏈哄埗鏇存帴杩戞祦鍑芥暟/鍘嬪姏涓績鐨勬ā鎬佺浉浣嶆紨鍖栵紱杩欎笉鑳借嚜鍔ㄦ帹鍑?Hua/VG 閫熷害涓績绾夸篃鐢卞悓涓€鏈哄埗鎺у埗銆?,
    ]
    (diag_dir / "psi_center_validation_summary_zh.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (diag_dir / "psi_center_validation_summary.json").write_text(json.dumps(judgment, ensure_ascii=False, indent=2), encoding="utf-8")


def _requested_iterations(smoke_days: str, available: list[int]) -> set[int] | None:
    if not smoke_days.strip():
        return None
    days = [float(part.strip()) for part in smoke_days.split(",") if part.strip()]
    requested: set[int] = set()
    for day in days:
        target = int(round(day * 1440.0))
        requested.add(min(available, key=lambda value: abs(value - target)))
    return requested


def main() -> int:
    args = _parse_args()
    root = Path(args.output_root)
    diag_dir = root / "diagnostics" / "mitgcm_psi_center_evolution"
    fig_dir = root / "figures" / "mitgcm_psi_center_evolution"
    diag_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads((root / "experiment_manifest.json").read_text(encoding="utf-8"))
    x_m, y_m, depth, radius_m = _grid_from_run(root / "runs" / REFERENCE_CASE / "run", manifest)
    shape = (depth.size, y_m.size, x_m.size)
    all_iterations = sorted({_iteration_from_name(path) for path in (root / "runs" / REFERENCE_CASE / "run").glob("U.*.data")})
    keep_iterations = _requested_iterations(args.smoke_days, all_iterations)

    center_parts: list[pd.DataFrame] = []
    psi_stats: list[dict[str, float | int | str]] = []
    for case in CASES:
        run = root / "runs" / case / "run"
        files = sorted(run.glob("U.*.data"), key=_iteration_from_name)
        for u_path in files:
            iteration = _iteration_from_name(u_path)
            if keep_iterations is not None and iteration not in keep_iterations:
                continue
            v_path = run / f"V.{iteration:010d}.data"
            if not v_path.exists():
                continue
            u = _read_big_endian(u_path, shape)
            v = _read_big_endian(v_path, shape)
            centers, psi = _extract_centers(u, v, x_m, y_m, radius_m, args.core_rmax)
            centers.insert(0, "depth_m", depth[centers["depth_index"].to_numpy(dtype=int)])
            centers.insert(0, "day", iteration / 1440.0)
            centers.insert(0, "iteration", iteration)
            centers.insert(0, "case", case)
            center_parts.append(centers)
            finite = np.isfinite(psi)
            psi_stats.append(
                {
                    "case": case,
                    "iteration": int(iteration),
                    "day": float(iteration / 1440.0),
                    "psi_finite_fraction": float(np.mean(finite)),
                    "psi_abs_p95": float(np.nanpercentile(np.abs(psi), 95)) if finite.any() else np.nan,
                    "psi_abs_max": float(np.nanmax(np.abs(psi))) if finite.any() else np.nan,
                }
            )

    if not center_parts:
        raise RuntimeError("No MITgcm U/V frames were processed.")
    centers = pd.concat(center_parts, ignore_index=True)
    centers.to_csv(diag_dir / "psi_centerlines_evolution.csv", index=False)
    if not args.no_parquet:
        centers.to_parquet(diag_dir / "psi_centerlines_evolution.parquet", index=False)

    metrics = pd.concat(
        [_metrics_for_definition(centers, definition) for definition in ("psi_abs_extreme", "speed_min")],
        ignore_index=True,
    )
    metrics.to_csv(diag_dir / "psi_modal_evolution_metrics.csv", index=False)
    if not args.no_parquet:
        metrics.to_parquet(diag_dir / "psi_modal_evolution_metrics.parquet", index=False)

    sensitivity = _sensitivity_metrics(centers)
    sensitivity.to_csv(diag_dir / "center_definition_sensitivity.csv", index=False)
    if not args.no_parquet:
        sensitivity.to_parquet(diag_dir / "center_definition_sensitivity.parquet", index=False)

    pd.DataFrame(psi_stats).to_csv(diag_dir / "psi_reconstruction_quality.csv", index=False)
    judgment = _judge(metrics)
    _plot_tilt(metrics, fig_dir)
    _plot_rmse(metrics, fig_dir)
    _plot_sensitivity(sensitivity, fig_dir)
    _plot_3d_center_definition(centers, fig_dir)
    _plot_examples(centers, fig_dir)
    _write_summary(diag_dir, judgment, metrics, sensitivity)
    print(f"Wrote {diag_dir}")
    print(f"Wrote {fig_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
