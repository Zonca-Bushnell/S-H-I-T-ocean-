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

from src.Legacy.First_temp.tilted_ep_flux_validation import load_n2
from src.Legacy.experiments.theory_validation.unified_math import (
    geo_params,
    project_vertical_modes,
    streamfunction_from_zeta,
    velocity_from_psi,
    vertical_mode_decomposition,
    vertical_weights,
)


POLARITIES = ("anticyclonic", "cyclonic")
MODE_LABELS = ("mode1", "mode2", "mode1_plus_2", "mode1_to_5", "full")


def _decode_list(values: np.ndarray) -> list[str]:
    out: list[str] = []
    for value in values:
        if isinstance(value, bytes):
            out.append(value.decode("utf-8"))
        else:
            out.append(str(value))
    return out


def _load_azimuthal_npz(path: Path) -> dict[str, np.ndarray | list[str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    data = np.load(path, allow_pickle=True)
    required = {"polarities", "tau_grid", "depth", "radial", "theta", "u_mean", "v_mean", "speed_mean"}
    missing = sorted(required - set(data.files))
    if missing:
        raise KeyError(f"Missing arrays in {path}: {missing}")
    return {
        "polarities": _decode_list(data["polarities"]),
        "tau_grid": np.asarray(data["tau_grid"], dtype="f8"),
        "depth": np.asarray(data["depth"], dtype="f8"),
        "radial": np.asarray(data["radial"], dtype="f8"),
        "theta": np.asarray(data["theta"], dtype="f8"),
        "u_mean": np.asarray(data["u_mean"], dtype="f8"),
        "v_mean": np.asarray(data["v_mean"], dtype="f8"),
        "speed_mean": np.asarray(data["speed_mean"], dtype="f8"),
        "count": np.asarray(data["count"], dtype="f8") if "count" in data.files else None,
        "n_objects": np.asarray(data["n_objects"]) if "n_objects" in data.files else None,
        "n_tracks": np.asarray(data["n_tracks"]) if "n_tracks" in data.files else None,
    }


def _resolve_radial_seed_root(rv_root: Path, explicit: str) -> Path:
    if explicit:
        path = Path(explicit)
        if not path.exists():
            raise FileNotFoundError(path)
        return path
    candidates = [
        rv_root.parent / "representative_vortex_radial_seed",
        rv_root.parent / "representative_vortex",
        rv_root,
    ]
    for path in candidates:
        if (path / "axis" / "rotated_points.parquet").exists() and (
            path / "object_cache" / "selected_lifecycle_objects.parquet"
        ).exists():
            return path
    raise FileNotFoundError("Cannot locate radial-seed representative root with axis/object_cache")


def _resolve_n2_profile(rv_root: Path, radial_seed_root: Path, explicit: str) -> Path:
    if explicit:
        path = Path(explicit)
        if not path.exists():
            raise FileNotFoundError(path)
        return path
    candidates = [
        rv_root / "climatology" / "cmems_doy_climatology_1993_2022_31d_sigma0_dz_profile.npz",
        radial_seed_root / "climatology" / "cmems_doy_climatology_1993_2022_31d_sigma0_dz_profile.npz",
        rv_root.parent / "representative_vortex_radial_seed" / "climatology" / "cmems_doy_climatology_1993_2022_31d_sigma0_dz_profile.npz",
    ]
    for path in candidates:
        if path.exists():
            return path
    glob_candidates = sorted(rv_root.parent.glob("**/*sigma0_dz_profile.npz"))
    if glob_candidates:
        return glob_candidates[0]
    raise FileNotFoundError("Cannot locate sigma0_dz_profile.npz")


def _representative_radius_by_polarity(radial_seed_root: Path) -> dict[str, float]:
    objects = pd.read_parquet(radial_seed_root / "object_cache" / "selected_lifecycle_objects.parquet")
    if "shape_class" in objects:
        objects = objects[objects["shape_class"].astype(str).eq("coherent")].copy()
    if "mean_radius_m" not in objects:
        raise KeyError("selected_lifecycle_objects lacks mean_radius_m")
    radii: dict[str, float] = {}
    for polarity, part in objects.groupby(objects["polarity"].astype(str), sort=True):
        value = float(np.nanmedian(part["mean_radius_m"].to_numpy(dtype="f8")))
        if np.isfinite(value) and value > 0:
            radii[str(polarity)] = value
    return radii


def _polar_to_xy(field: np.ndarray, radial: np.ndarray, theta: np.ndarray, x_over_r: np.ndarray, y_over_r: np.ndarray) -> np.ndarray:
    r = np.hypot(x_over_r, y_over_r)
    phi = np.mod(np.arctan2(y_over_r, x_over_r), 2.0 * np.pi)
    ri = np.abs(radial[:, None, None] - r[None, :, :]).argmin(axis=0)
    ti = np.abs(np.angle(np.exp(1j * (theta[:, None, None] - phi[None, :, :])))).argmin(axis=0)
    out = field[:, ri, ti]
    out[:, r > float(np.nanmax(radial))] = np.nan
    return out


def _relative_vorticity_xy(u: np.ndarray, v: np.ndarray, x_m: np.ndarray, y_m: np.ndarray) -> np.ndarray:
    dvdx = np.gradient(v, x_m, axis=2, edge_order=1)
    dudy = np.gradient(u, y_m, axis=1, edge_order=1)
    return dvdx - dudy


def _mode_energy_fraction(psi: np.ndarray, mode_parts: dict[str, np.ndarray], depth: np.ndarray, x_m: np.ndarray, y_m: np.ndarray) -> dict[str, float]:
    w = vertical_weights(depth)
    area = float(np.nanmedian(np.diff(x_m)) * np.nanmedian(np.diff(y_m)))
    denom = float(np.nansum((psi * psi) * w[:, None, None]) * abs(area))
    out: dict[str, float] = {}
    for label in ("mode1", "mode2", "mode1_plus_2", "mode1_to_5"):
        part = mode_parts[label]
        value = float(np.nansum((part * part) * w[:, None, None]) * abs(area))
        out[f"{label}_psi_energy_fraction"] = value / denom if denom > 0 else np.nan
    return out


def _core_mask(x_over_r: np.ndarray, y_over_r: np.ndarray, rmax: float) -> np.ndarray:
    return np.hypot(x_over_r, y_over_r) <= float(rmax)


def _psi_extreme_centerline(psi: np.ndarray, x_over_r: np.ndarray, y_over_r: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    nz = psi.shape[0]
    x = np.full(nz, np.nan, dtype="f8")
    y = np.full(nz, np.nan, dtype="f8")
    for k in range(nz):
        layer = np.where(mask, psi[k], np.nan)
        if not np.any(np.isfinite(layer)):
            continue
        idx = np.nanargmax(np.abs(layer))
        j, i = np.unravel_index(idx, layer.shape)
        x[k] = x_over_r[j, i]
        y[k] = y_over_r[j, i]
    return x, y


def _speed_min_centerline(psi: np.ndarray, x_over_r: np.ndarray, y_over_r: np.ndarray, x_m: np.ndarray, y_m: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    u, v = velocity_from_psi(psi, y_m, x_m)
    speed = np.hypot(u, v)
    nz = psi.shape[0]
    x = np.full(nz, np.nan, dtype="f8")
    y = np.full(nz, np.nan, dtype="f8")
    for k in range(nz):
        layer = np.where(mask, speed[k], np.nan)
        if not np.any(np.isfinite(layer)):
            continue
        idx = np.nanargmin(layer)
        j, i = np.unravel_index(idx, layer.shape)
        x[k] = x_over_r[j, i]
        y[k] = y_over_r[j, i]
    return x, y


def _tilt_distance(x: np.ndarray, y: np.ndarray) -> float:
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 2:
        return np.nan
    x0 = float(x[valid][0])
    y0 = float(y[valid][0])
    return float(np.nanmax(np.hypot(x[valid] - x0, y[valid] - y0)))


def _rmse_xy(ax: np.ndarray, ay: np.ndarray, bx: np.ndarray, by: np.ndarray) -> float:
    valid = np.isfinite(ax) & np.isfinite(ay) & np.isfinite(bx) & np.isfinite(by)
    if valid.sum() < 2:
        return np.nan
    return float(np.sqrt(np.nanmean((ax[valid] - bx[valid]) ** 2 + (ay[valid] - by[valid]) ** 2)))


def _corr_xy(ax: np.ndarray, ay: np.ndarray, bx: np.ndarray, by: np.ndarray) -> float:
    valid = np.isfinite(ax) & np.isfinite(ay) & np.isfinite(bx) & np.isfinite(by)
    if valid.sum() < 3:
        return np.nan
    avec = np.column_stack([ax[valid], ay[valid]]).ravel()
    bvec = np.column_stack([bx[valid], by[valid]]).ravel()
    if np.nanstd(avec) <= 0 or np.nanstd(bvec) <= 0:
        return np.nan
    return float(np.corrcoef(avec, bvec)[0, 1])


def _observed_median_axis(radial_seed_root: Path, tau_grid: np.ndarray, radius_by_polarity: dict[str, float]) -> dict[tuple[str, int], pd.DataFrame]:
    objects = pd.read_parquet(radial_seed_root / "object_cache" / "selected_lifecycle_objects.parquet")
    if "shape_class" in objects:
        objects = objects[objects["shape_class"].astype(str).eq("coherent")].copy()
    objects["eddy3d_object_id"] = objects["eddy3d_object_id"].astype("int64")
    objects["tau_index"] = np.abs(objects["life_phase"].to_numpy(dtype="f8")[:, None] - tau_grid[None, :]).argmin(axis=1)
    obj_cols = ["eddy3d_object_id", "polarity", "tau_index"]
    points = pd.read_parquet(
        radial_seed_root / "axis" / "rotated_points.parquet",
        columns=["eddy3d_object_id", "depth_index", "depth_m", "x_rot_m", "y_rot_m", "axis_alignment_method"],
    )
    points["eddy3d_object_id"] = points["eddy3d_object_id"].astype("int64")
    if "axis_alignment_method" in points:
        points = points[points["axis_alignment_method"].astype(str).eq("global_ls_alpha")].copy()
    merged = points.merge(objects[obj_cols], on="eddy3d_object_id", how="inner")
    out: dict[tuple[str, int], pd.DataFrame] = {}
    for (polarity, tau_index), part in merged.groupby(["polarity", "tau_index"], sort=True):
        radius = radius_by_polarity.get(str(polarity), np.nan)
        med = (
            part.groupby("depth_index", sort=True)
            .agg(depth_m=("depth_m", "median"), x_rot_m=("x_rot_m", "median"), y_rot_m=("y_rot_m", "median"), n_objects=("eddy3d_object_id", "nunique"))
            .reset_index()
        )
        med["x_over_R"] = med["x_rot_m"].astype("f8") / radius
        med["y_over_R"] = med["y_rot_m"].astype("f8") / radius
        out[(str(polarity), int(tau_index))] = med
    return out


def _interp_observed_to_depth(obs: pd.DataFrame | None, depth: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if obs is None or obs.empty:
        return np.full_like(depth, np.nan), np.full_like(depth, np.nan)
    d = obs["depth_m"].to_numpy(dtype="f8")
    x = obs["x_over_R"].to_numpy(dtype="f8")
    y = obs["y_over_R"].to_numpy(dtype="f8")
    order = np.argsort(d)
    return (
        np.interp(depth, d[order], x[order], left=np.nan, right=np.nan),
        np.interp(depth, d[order], y[order], left=np.nan, right=np.nan),
    )


def _plot_vertical_modes(out_dir: Path, depth: np.ndarray, profiles: np.ndarray, radii: np.ndarray, orthogonality_error: float) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 7), constrained_layout=True)
    max_modes = min(6, profiles.shape[1])
    for n in range(max_modes):
        label = "barotropic" if n == 0 else f"mode {n}"
        rd = radii[n] / 1000.0 if np.isfinite(radii[n]) else np.inf
        ax.plot(profiles[:, n], depth, label=f"{label}, Rd={rd:.1f} km")
    ax.invert_yaxis()
    ax.set_xlabel("normalized vertical mode")
    ax.set_ylabel("depth (m)")
    ax.set_title(f"QG vertical modes, orthogonality error={orthogonality_error:.2e}")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.savefig(out_dir / "figures" / "vertical_modes.png", dpi=180)
    plt.close(fig)


def _plot_energy(out_dir: Path, energy: pd.DataFrame) -> None:
    for polarity, part in energy.groupby("polarity", sort=True):
        fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
        for label in ("mode1", "mode2", "mode1_plus_2", "mode1_to_5"):
            col = f"{label}_psi_energy_fraction"
            ax.plot(part["tau_center"], part[col], marker="o", ms=3, label=label)
        ax.set_ylim(bottom=0)
        ax.set_xlabel("tau")
        ax.set_ylabel("streamfunction energy fraction")
        ax.set_title(f"{polarity}: modal streamfunction energy fractions")
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.savefig(out_dir / "figures" / f"{polarity}_modal_energy_fraction_by_tau.png", dpi=180)
        plt.close(fig)
    pivot = energy.pivot_table(index="tau_center", columns="polarity", values="mode1_plus_2_psi_energy_fraction")
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    for col in pivot.columns:
        ax.plot(pivot.index, pivot[col], marker="o", label=str(col))
    ax.set_xlabel("tau")
    ax.set_ylabel("mode1+2 energy fraction")
    ax.set_title("Mode1+2 fraction by polarity")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.savefig(out_dir / "figures" / "modal_energy_fraction_by_tau.png", dpi=180)
    plt.close(fig)


def _plot_tilt(out_dir: Path, metrics: pd.DataFrame) -> None:
    for polarity, part in metrics.groupby("polarity", sort=True):
        fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
        for label, col in (
            ("observed Hua velocity axis", "observed_velocity_tilt_R"),
            ("mode1 psi center", "mode1_psi_tilt_R"),
            ("mode2 psi center", "mode2_psi_tilt_R"),
            ("mode1+2 psi center", "mode1_plus_2_psi_tilt_R"),
            ("mode1..5 psi center", "mode1_to_5_psi_tilt_R"),
        ):
            ax.plot(part["tau_center"], part[col], marker="o", ms=3, label=label)
        ax.set_xlabel("tau")
        ax.set_ylabel("max center displacement / R")
        ax.set_title(f"{polarity}: vertical tilt distance")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
        fig.savefig(out_dir / "figures" / f"{polarity}_tilt_distance_by_tau.png", dpi=180)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    for polarity, part in metrics.groupby("polarity", sort=True):
        ax.scatter(part["observed_velocity_tilt_R"], part["mode1_plus_2_psi_tilt_R"], label=polarity)
    lim = np.nanmax(
        np.abs(metrics[["observed_velocity_tilt_R", "mode1_plus_2_psi_tilt_R"]].to_numpy(dtype="f8"))
    )
    lim = float(lim) if np.isfinite(lim) and lim > 0 else 1.0
    ax.plot([0, lim], [0, lim], color="0.3", linestyle="--", linewidth=1)
    ax.set_xlabel("observed Hua velocity-axis tilt / R")
    ax.set_ylabel("mode1+2 psi-center tilt / R")
    ax.set_title("Mode1+2 reconstructed tilt vs observed velocity-center tilt")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.savefig(out_dir / "figures" / "mode12_vs_observed_tilt.png", dpi=180)
    plt.close(fig)


def _plot_centerline_overlays(out_dir: Path, depth: np.ndarray, center_rows: pd.DataFrame, tau_values: list[float]) -> None:
    for polarity, ppart in center_rows.groupby("polarity", sort=True):
        for tau in tau_values:
            tau_i = int(np.abs(ppart["tau_center"].to_numpy(dtype="f8") - tau).argmin())
            actual_tau = float(ppart.iloc[tau_i]["tau_center"])
            part = ppart[np.isclose(ppart["tau_center"].astype("f8"), actual_tau)].copy()
            fig, axes = plt.subplots(1, 2, figsize=(10, 5.5), sharey=True, constrained_layout=True)
            for label, style in (
                ("observed_velocity", "k-"),
                ("mode1_psi", "C0--"),
                ("mode2_psi", "C1--"),
                ("mode1_plus_2_psi", "C2-"),
                ("mode1_to_5_psi", "C3-"),
                ("full_psi", "C4:"),
            ):
                item = part[part["center_definition"].eq(label)]
                if item.empty:
                    continue
                axes[0].plot(item["x_over_R"], item["depth_m"], style, label=label)
                axes[1].plot(item["y_over_R"], item["depth_m"], style, label=label)
            for ax, xlabel in zip(axes, ("x_rot / R", "y_rot / R")):
                ax.axvline(0, color="0.7", linewidth=0.8)
                ax.invert_yaxis()
                ax.set_xlabel(xlabel)
                ax.grid(True, alpha=0.25)
            axes[0].set_ylabel("depth (m)")
            axes[1].legend(fontsize=7, loc="best")
            fig.suptitle(f"{polarity}: centerline overlay, tau={actual_tau:.2f}")
            fig.savefig(out_dir / "figures" / f"{polarity}_centerline_overlay_tau_{actual_tau:.2f}.png", dpi=180)
            plt.close(fig)


def _write_summary(out_dir: Path, metrics: pd.DataFrame, center_cmp: pd.DataFrame, energy: pd.DataFrame, meta: dict) -> None:
    mode12 = metrics["mode1_plus_2_explained_fraction_velocity_axis"].replace([np.inf, -np.inf], np.nan)
    mode15 = metrics["mode1_to_5_explained_fraction_velocity_axis"].replace([np.inf, -np.inf], np.nan)
    lines = [
        "# Yang/Xu/Li 2026 妯℃€佸€炬枩鏈哄埗楠岃瘉",
        "",
        "鏈疄楠屽彧璇荤幇鏈?Kuroshiou coherent-only 浠ｈ〃娑℃棆缁撴灉锛屾楠岀涓€銆佺浜屾枩鍘嬫ā鎬佹槸鍚﹁冻浠ヨВ閲婁唬琛ㄦ丁鏃嬬殑鍨傚悜鍊炬枩銆?,
        "",
        "## 鍏抽敭鍙ｅ緞",
        "",
        "- 璁烘枃鍙ｅ緞涓績锛氭祦鍑芥暟鏋佸€间腑蹇冿紝鐢ㄤ簬妯℃嫙 Yang/Xu/Li 2026 鐨勬ā鎬侀噸鏋勪腑蹇冪嚎銆?,
        "- 鎴戜滑鐨勭敓浜у彛寰勪腑蹇冿細Hua/Nencioli 閫熷害涓績绾匡紝鏉ヨ嚜 30-180 澶╁甫閫氶€熷害銆乻trict-contiguous 娣卞害鎵╁睍鍜?global_ls_alpha 瀵归綈銆?,
        "- 鍥犳鏈枃鍙垽鏂満鍒舵槸鍚︿竴鑷达紝涓嶆妸娓╁害/娴佸嚱鏁颁腑蹇冨畾涔夌洿鎺ョ瓑鍚屼簬閫熷害涓績瀹氫箟銆?,
        "",
        "## 涓昏鏁板€兼憳瑕?,
        "",
        f"- mode1+2 瀵归€熷害涓績绾跨殑涓綅瑙ｉ噴姣斾緥锛歚{float(np.nanmedian(mode12)):.3f}`銆?,
        f"- mode1..5 瀵归€熷害涓績绾跨殑涓綅瑙ｉ噴姣斾緥锛歚{float(np.nanmedian(mode15)):.3f}`銆?,
        f"- mode1+2 娴佸嚱鏁拌兘閲忓崰姣斾腑浣嶆暟锛歚{float(np.nanmedian(energy['mode1_plus_2_psi_energy_fraction'])):.3f}`銆?,
        f"- mode1..5 娴佸嚱鏁拌兘閲忓崰姣斾腑浣嶆暟锛歚{float(np.nanmedian(energy['mode1_to_5_psi_energy_fraction'])):.3f}`銆?,
        "",
        "## 鍒よ瑙勫垯",
        "",
        "- 濡傛灉 mode1-only 鍜?mode2-only 鍊炬枩寮憋紝鑰?mode1+2 鏄庢樉鎺ヨ繎瑙傛祴杞寸嚎锛屾敮鎸佲€滄ā鎬佷紶鎾樊寮傚鑷村€炬枩鈥濄€?,
        "- 濡傛灉 mode1..5 鐩告瘮 mode1+2 鍙皬骞呮敼鍠勶紝楂樻ā鎬佹槸娆＄骇淇銆?,
        "- 濡傛灉瑙傛祴閫熷害涓績鍊炬枩灏忎簬 mode1/mode2 鐙珛涓績鍒嗙锛岃鏄庨潪绾挎€ц€﹀悎鎴栭€熷害涓績瀹氫箟浼氬帇鍒跺彲瑙佸€炬枩銆?,
        "- 鍗曚竴鍖哄煙鍚堟垚涓嶈兘涓ユ牸楠岃瘉鍏ㄧ悆 N2/f2 鏍囧害寰嬶紱杩欓噷鍙仛鏈哄埗涓€鑷存€ф楠屻€?,
        "",
        "## 杈撳叆",
        "",
        f"- RV root: `{meta['rv_root']}`",
        f"- radial seed root: `{meta['radial_seed_root']}`",
        f"- azimuthal npz: `{meta['azimuthal_npz']}`",
        f"- N2 profile: `{meta['n2_profile']}`",
    ]
    (out_dir / "yang2026_mode_tilt_validation_summary_zh.md").write_text("\n".join(lines), encoding="utf-8")
    (out_dir / "metadata" / "run_metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    rv_root = Path(args.rv_root)
    out_dir = Path(args.output_dir)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)
    (out_dir / "metadata").mkdir(parents=True, exist_ok=True)

    azimuthal_npz = Path(args.azimuthal_npz) if args.azimuthal_npz else rv_root / "azimuthal_representative_velocity.npz"
    arrays = _load_azimuthal_npz(azimuthal_npz)
    radial_seed_root = _resolve_radial_seed_root(rv_root, args.radial_seed_root)
    n2_profile = _resolve_n2_profile(rv_root, radial_seed_root, args.n2_profile)

    polarities = [str(v) for v in arrays["polarities"]]
    tau_grid = arrays["tau_grid"]
    depth = arrays["depth"]
    radial = arrays["radial"]
    theta = arrays["theta"]
    u_mean = arrays["u_mean"]
    v_mean = arrays["v_mean"]

    keep_depth = np.isfinite(depth) & (depth <= float(args.max_depth_m) + 1.0e-6)
    depth = depth[keep_depth]
    u_mean = u_mean[:, :, keep_depth, :, :]
    v_mean = v_mean[:, :, keep_depth, :, :]

    radius_by_polarity = _representative_radius_by_polarity(radial_seed_root)
    latitude_ref = float(args.latitude_ref)
    if not np.isfinite(latitude_ref):
        objects = pd.read_parquet(radial_seed_root / "object_cache" / "selected_lifecycle_objects.parquet", columns=["surface_lat"])
        latitude_ref = float(np.nanmedian(objects["surface_lat"].to_numpy(dtype="f8")))
    geo = geo_params(latitude_ref)
    n2 = load_n2(n2_profile, depth)
    f_profile = (geo.f0 * geo.f0) / np.maximum(n2, 1.0e-12)
    values, profiles, radii = vertical_mode_decomposition(f_profile, depth, mode_count=int(args.mode_count))
    weights = vertical_weights(depth)
    gram = profiles.T @ (weights[:, None] * profiles)
    orthogonality_error = float(np.nanmax(np.abs(gram - np.eye(gram.shape[0]))))
    _plot_vertical_modes(out_dir, depth, profiles, radii, orthogonality_error)

    obs_axes = _observed_median_axis(radial_seed_root, tau_grid, radius_by_polarity)

    xy = np.linspace(-float(args.rmax), float(args.rmax), int(args.xy_size), dtype="f8")
    xx_r, yy_r = np.meshgrid(xy, xy)
    core = _core_mask(xx_r, yy_r, float(args.center_core_rmax))

    metric_rows: list[dict] = []
    center_rows: list[dict] = []
    energy_rows: list[dict] = []
    center_cmp_rows: list[dict] = []

    for polarity in POLARITIES:
        if polarity not in polarities:
            continue
        ip = polarities.index(polarity)
        radius_m = radius_by_polarity[polarity]
        x_m = xy * radius_m
        y_m = xy * radius_m
        for it, tau in enumerate(tau_grid):
            u_xy = _polar_to_xy(u_mean[ip, it], radial, theta, xx_r, yy_r)
            v_xy = _polar_to_xy(v_mean[ip, it], radial, theta, xx_r, yy_r)
            zeta = _relative_vorticity_xy(u_xy, v_xy, x_m, y_m)
            psi = streamfunction_from_zeta(zeta, y_m, x_m)
            mode_recon = project_vertical_modes(psi, profiles, depth)
            mode_parts = {
                "mode1": mode_recon[1] if mode_recon.shape[0] > 1 else np.full_like(psi, np.nan),
                "mode2": mode_recon[2] if mode_recon.shape[0] > 2 else np.full_like(psi, np.nan),
                "mode1_plus_2": np.nansum(mode_recon[1 : min(3, mode_recon.shape[0])], axis=0),
                "mode1_to_5": np.nansum(mode_recon[1 : min(6, mode_recon.shape[0])], axis=0),
                "full": psi,
            }
            energy = {
                "polarity": polarity,
                "tau_index": int(it),
                "tau_center": float(tau),
                **_mode_energy_fraction(psi, mode_parts, depth, x_m, y_m),
            }
            energy_rows.append(energy)

            obs_x, obs_y = _interp_observed_to_depth(obs_axes.get((polarity, int(it))), depth)
            centers: dict[str, tuple[np.ndarray, np.ndarray]] = {"observed_velocity": (obs_x, obs_y)}
            for label, part in mode_parts.items():
                centers[f"{label}_psi"] = _psi_extreme_centerline(part, xx_r, yy_r, core)
                centers[f"{label}_speed_min"] = _speed_min_centerline(part, xx_r, yy_r, x_m, y_m, core)

            for label, (cx, cy) in centers.items():
                for k, z in enumerate(depth):
                    if not (np.isfinite(cx[k]) and np.isfinite(cy[k])):
                        continue
                    center_rows.append(
                        {
                            "polarity": polarity,
                            "tau_index": int(it),
                            "tau_center": float(tau),
                            "depth_index": int(k),
                            "depth_m": float(z),
                            "center_definition": label,
                            "x_over_R": float(cx[k]),
                            "y_over_R": float(cy[k]),
                            "x_km": float(cx[k] * radius_m / 1000.0),
                            "y_km": float(cy[k] * radius_m / 1000.0),
                        }
                    )

            metric = {
                "polarity": polarity,
                "tau_index": int(it),
                "tau_center": float(tau),
                "representative_radius_km": float(radius_m / 1000.0),
                "observed_velocity_tilt_R": _tilt_distance(obs_x, obs_y),
            }
            zero_x = np.zeros_like(obs_x)
            zero_y = np.zeros_like(obs_y)
            obs_zero_rmse = _rmse_xy(obs_x, obs_y, zero_x, zero_y)
            for label in ("mode1", "mode2", "mode1_plus_2", "mode1_to_5", "full"):
                px, py = centers[f"{label}_psi"]
                vx, vy = centers[f"{label}_speed_min"]
                metric[f"{label}_psi_tilt_R"] = _tilt_distance(px, py)
                metric[f"{label}_speed_min_tilt_R"] = _tilt_distance(vx, vy)
                metric[f"{label}_psi_rmse_to_observed_velocity_R"] = _rmse_xy(px, py, obs_x, obs_y)
                metric[f"{label}_psi_corr_to_observed_velocity"] = _corr_xy(px, py, obs_x, obs_y)
                rmse = metric[f"{label}_psi_rmse_to_observed_velocity_R"]
                metric[f"{label}_explained_fraction_velocity_axis"] = (
                    1.0 - rmse / obs_zero_rmse if np.isfinite(rmse) and np.isfinite(obs_zero_rmse) and obs_zero_rmse > 0 else np.nan
                )
                center_cmp_rows.append(
                    {
                        "polarity": polarity,
                        "tau_index": int(it),
                        "tau_center": float(tau),
                        "reconstruction": label,
                        "psi_extreme_vs_speed_min_rmse_R": _rmse_xy(px, py, vx, vy),
                        "psi_extreme_tilt_R": _tilt_distance(px, py),
                        "speed_min_tilt_R": _tilt_distance(vx, vy),
                    }
                )
            metric["mode1_mode2_center_separation_R"] = _rmse_xy(
                centers["mode1_psi"][0], centers["mode1_psi"][1], centers["mode2_psi"][0], centers["mode2_psi"][1]
            )
            metric_rows.append(metric)

    metrics = pd.DataFrame(metric_rows)
    centers_df = pd.DataFrame(center_rows)
    energy_df = pd.DataFrame(energy_rows)
    center_cmp = pd.DataFrame(center_cmp_rows)

    metrics.to_csv(out_dir / "tables" / "modal_tilt_metrics.csv", index=False)
    centers_df.to_csv(out_dir / "tables" / "centerline_points.csv", index=False)
    energy_df.to_csv(out_dir / "tables" / "modal_energy_fractions.csv", index=False)
    center_cmp.to_csv(out_dir / "tables" / "center_definition_comparison.csv", index=False)

    _plot_energy(out_dir, energy_df)
    _plot_tilt(out_dir, metrics)
    _plot_centerline_overlays(out_dir, depth, centers_df, [float(v) for v in args.plot_taus.split(",") if v.strip()])

    meta = {
        "rv_root": str(rv_root),
        "radial_seed_root": str(radial_seed_root),
        "azimuthal_npz": str(azimuthal_npz),
        "n2_profile": str(n2_profile),
        "output_dir": str(out_dir),
        "latitude_ref": latitude_ref,
        "f0": geo.f0,
        "beta": geo.beta,
        "orthogonality_error": orthogonality_error,
        "mode_eigenvalues": values.tolist(),
        "mode_radius_like_m": radii.tolist(),
        "center_core_rmax": float(args.center_core_rmax),
        "xy_size": int(args.xy_size),
    }
    _write_summary(out_dir, metrics, center_cmp, energy_df, meta)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rv-root", required=True, help="ME_LIUTEX azimuth-preserved representative vortex root.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--azimuthal-npz", default="", help="Defaults to rv-root/azimuthal_representative_velocity.npz")
    parser.add_argument("--radial-seed-root", default="", help="Representative root containing axis/rotated_points and selected objects.")
    parser.add_argument("--n2-profile", default="")
    parser.add_argument("--latitude-ref", type=float, default=float("nan"))
    parser.add_argument("--max-depth-m", type=float, default=2000.0)
    parser.add_argument("--mode-count", type=int, default=5)
    parser.add_argument("--rmax", type=float, default=2.5)
    parser.add_argument("--xy-size", type=int, default=121)
    parser.add_argument("--center-core-rmax", type=float, default=1.5)
    parser.add_argument("--plot-taus", default="0.25,0.50,0.75")
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
