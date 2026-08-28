from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import shift as ndi_shift

from src.Legacy.First_temp.tilted_ep_flux_validation import load_n2
from src.Legacy.experiments.temp.run_mode_tilt_validation_yang2026 import (
    POLARITIES,
    _core_mask,
    _corr_xy,
    _decode_list,
    _interp_observed_to_depth,
    _load_azimuthal_npz,
    _mode_energy_fraction,
    _observed_median_axis,
    _plot_vertical_modes,
    _polar_to_xy,
    _relative_vorticity_xy,
    _representative_radius_by_polarity,
    _resolve_n2_profile,
    _resolve_radial_seed_root,
    _rmse_xy,
    _speed_min_centerline,
    _tilt_distance,
)
from src.Legacy.experiments.theory_validation.unified_math import (
    geo_params,
    project_vertical_modes,
    streamfunction_from_zeta,
    vertical_mode_decomposition,
    vertical_weights,
)


MODE_LABELS = ("mode1", "mode2", "mode1_plus_2", "mode1_to_5", "full")


def _rotation_core_centerline(
    u: np.ndarray,
    v: np.ndarray,
    x_m: np.ndarray,
    y_m: np.ndarray,
    x_over_r: np.ndarray,
    y_over_r: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    zeta = _relative_vorticity_xy(u, v, x_m, y_m)
    nz = zeta.shape[0]
    x = np.full(nz, np.nan, dtype="f8")
    y = np.full(nz, np.nan, dtype="f8")
    zeta_peak = np.full(nz, np.nan, dtype="f8")
    for k in range(nz):
        layer = np.where(mask, np.abs(zeta[k]), np.nan)
        if not np.any(np.isfinite(layer)):
            continue
        idx = np.nanargmax(layer)
        j, i = np.unravel_index(idx, layer.shape)
        x[k] = x_over_r[j, i]
        y[k] = y_over_r[j, i]
        zeta_peak[k] = zeta[k, j, i]
    return x, y, zeta_peak


def _psi_abs_centerline(psi: np.ndarray, x_over_r: np.ndarray, y_over_r: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    nz = psi.shape[0]
    x = np.full(nz, np.nan, dtype="f8")
    y = np.full(nz, np.nan, dtype="f8")
    for k in range(nz):
        layer = np.where(mask, np.abs(psi[k]), np.nan)
        if not np.any(np.isfinite(layer)):
            continue
        idx = np.nanargmax(layer)
        j, i = np.unravel_index(idx, layer.shape)
        x[k] = x_over_r[j, i]
        y[k] = y_over_r[j, i]
    return x, y


def _mode_parts(psi: np.ndarray, profiles: np.ndarray, depth: np.ndarray) -> dict[str, np.ndarray]:
    recon = project_vertical_modes(psi, profiles, depth)
    return {
        "mode1": recon[1] if recon.shape[0] > 1 else np.full_like(psi, np.nan),
        "mode2": recon[2] if recon.shape[0] > 2 else np.full_like(psi, np.nan),
        "mode1_plus_2": np.nansum(recon[1 : min(3, recon.shape[0])], axis=0),
        "mode1_to_5": np.nansum(recon[1 : min(6, recon.shape[0])], axis=0),
        "full": psi,
    }


def _center_layers_on_rotation_core(
    field: np.ndarray,
    omega_x: np.ndarray,
    omega_y: np.ndarray,
    xy: np.ndarray,
) -> np.ndarray:
    step = float(np.nanmedian(np.diff(xy)))
    out = np.full_like(field, np.nan, dtype="f8")
    for k in range(field.shape[0]):
        if not (np.isfinite(omega_x[k]) and np.isfinite(omega_y[k]) and step > 0):
            out[k] = field[k]
            continue
        # ndi_shift uses pixel units and positive shift moves values to larger indices.
        out[k] = ndi_shift(
            field[k],
            shift=(-omega_y[k] / step, -omega_x[k] / step),
            order=1,
            mode="constant",
            cval=np.nan,
            prefilter=False,
        )
    return out


def _axis_rows(
    polarity: str,
    tau_index: int,
    tau_center: float,
    depth: np.ndarray,
    label: str,
    x: np.ndarray,
    y: np.ndarray,
    radius_m: float,
    extra: dict[str, np.ndarray] | None = None,
) -> list[dict]:
    rows: list[dict] = []
    for k, z in enumerate(depth):
        if not (np.isfinite(x[k]) and np.isfinite(y[k])):
            continue
        row = {
            "polarity": polarity,
            "tau_index": int(tau_index),
            "tau_center": float(tau_center),
            "depth_index": int(k),
            "depth_m": float(z),
            "center_definition": label,
            "x_over_R": float(x[k]),
            "y_over_R": float(y[k]),
            "x_km": float(x[k] * radius_m / 1000.0),
            "y_km": float(y[k] * radius_m / 1000.0),
        }
        if extra:
            for key, values in extra.items():
                row[key] = float(values[k]) if k < len(values) and np.isfinite(values[k]) else np.nan
        rows.append(row)
    return rows


def _compare_to_target(
    polarity: str,
    tau_index: int,
    tau_center: float,
    chain: str,
    reconstruction: str,
    center_type: str,
    x: np.ndarray,
    y: np.ndarray,
    target_x: np.ndarray,
    target_y: np.ndarray,
    zero_rmse: float,
) -> dict:
    rmse = _rmse_xy(x, y, target_x, target_y)
    return {
        "polarity": polarity,
        "tau_index": int(tau_index),
        "tau_center": float(tau_center),
        "chain": chain,
        "reconstruction": reconstruction,
        "center_type": center_type,
        "rmse_to_rotation_core_R": rmse,
        "corr_to_rotation_core": _corr_xy(x, y, target_x, target_y),
        "tilt_distance_R": _tilt_distance(x, y),
        "target_tilt_distance_R": _tilt_distance(target_x, target_y),
        "explained_fraction_rotation_core": (
            1.0 - rmse / zero_rmse if np.isfinite(rmse) and np.isfinite(zero_rmse) and zero_rmse > 0 else np.nan
        ),
    }


def _decision(metrics: pd.DataFrame, chain: str) -> dict:
    part = metrics[(metrics["chain"].eq(chain)) & (metrics["center_type"].eq("rotation_core"))].copy()
    out = {"chain": chain, "decision": "insufficient", "reason": "no finite metrics"}
    if part.empty:
        return out
    rmse = part.pivot_table(
        index=["polarity", "tau_index"],
        columns="reconstruction",
        values="rmse_to_rotation_core_R",
        aggfunc="mean",
    )
    corr = part.pivot_table(
        index=["polarity", "tau_index"],
        columns="reconstruction",
        values="corr_to_rotation_core",
        aggfunc="mean",
    )
    required = {"mode1", "mode2", "mode1_plus_2", "mode1_to_5"}
    if not required.issubset(set(rmse.columns)):
        out["reason"] = "missing mode columns"
        return out
    for col in set(rmse.columns):
        if col not in corr.columns:
            corr[col] = np.nan
    corr = corr[list(rmse.columns)]
    better_rmse = ((rmse["mode1"] - rmse["mode1_plus_2"]) / rmse["mode1"] >= 0.20) & (
        (rmse["mode2"] - rmse["mode1_plus_2"]) / rmse["mode2"] >= 0.20
    )
    better_corr = (corr["mode1_plus_2"] - corr["mode1"] >= 0.20) & (corr["mode1_plus_2"] - corr["mode2"] >= 0.20)
    low_mode = ((rmse["mode1_plus_2"] - rmse["mode1_to_5"]) / rmse["mode1_plus_2"] < 0.10)
    support_fraction = float(np.nanmean((better_rmse & better_corr).to_numpy(dtype="f8")))
    low_mode_fraction = float(np.nanmean(low_mode.to_numpy(dtype="f8")))
    if support_fraction >= 0.50 and low_mode_fraction >= 0.50:
        decision = "support"
    elif support_fraction >= 0.20 or low_mode_fraction >= 0.50:
        decision = "partial"
    else:
        decision = "not_support"
    return {
        "chain": chain,
        "decision": decision,
        "support_fraction": support_fraction,
        "low_mode_fraction": low_mode_fraction,
        "median_mode12_rmse": float(np.nanmedian(rmse["mode1_plus_2"])),
        "median_mode15_rmse": float(np.nanmedian(rmse["mode1_to_5"])),
        "median_mode12_corr": float(np.nanmedian(corr["mode1_plus_2"])),
        "median_mode15_corr": float(np.nanmedian(corr["mode1_to_5"])),
    }


def _plot_center_overlay(out_dir: Path, centers: pd.DataFrame, depth: np.ndarray, plot_taus: list[float]) -> None:
    out_fig = out_dir / "figures"
    for polarity, pol_part in centers.groupby("polarity"):
        taus = sorted(pol_part["tau_center"].unique())
        for requested in plot_taus:
            tau = min(taus, key=lambda value: abs(float(value) - requested))
            part = pol_part[np.isclose(pol_part["tau_center"], tau)]
            fig, axes = plt.subplots(1, 2, figsize=(10, 5.3), sharey=True, constrained_layout=True)
            styles = {
                "observed_rotation_core": ("k-", 2.0),
                "observed_velocity_axis": ("C7--", 1.4),
                "r_omega_as_target_mode1_rotation_core": ("C0--", 1.3),
                "r_omega_as_target_mode2_rotation_core": ("C1--", 1.3),
                "r_omega_as_target_mode1_plus_2_rotation_core": ("C2-", 1.8),
                "r_omega_as_target_mode1_to_5_rotation_core": ("C3-", 1.5),
            }
            for label, (style, width) in styles.items():
                item = part[part["center_definition"].eq(label)].sort_values("depth_m")
                if item.empty:
                    continue
                axes[0].plot(item["x_over_R"], item["depth_m"], style, lw=width, label=label)
                axes[1].plot(item["y_over_R"], item["depth_m"], style, lw=width, label=label)
            for ax, xlabel in zip(axes, ("x / R", "y / R")):
                ax.axvline(0, color="0.75", lw=0.8)
                ax.invert_yaxis()
                ax.grid(True, alpha=0.25)
                ax.set_xlabel(xlabel)
            axes[0].set_ylabel("depth (m)")
            axes[1].legend(fontsize=6.8, loc="best")
            fig.suptitle(f"{polarity}: rotation-core modal centerlines, tau={tau:.2f}")
            fig.savefig(out_fig / f"{polarity}_r_omega_centerline_overlay_tau_{tau:.2f}.png", dpi=180)
            plt.close(fig)


def _plot_metric_summary(out_dir: Path, metrics: pd.DataFrame) -> None:
    out_fig = out_dir / "figures"
    for chain in sorted(metrics["chain"].unique()):
        part = metrics[(metrics["chain"].eq(chain)) & (metrics["center_type"].eq("rotation_core"))]
        if part.empty:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
        for polarity, pol_part in part.groupby("polarity"):
            for recon, style in (("mode1", "--"), ("mode2", ":"), ("mode1_plus_2", "-"), ("mode1_to_5", "-.")):
                item = pol_part[pol_part["reconstruction"].eq(recon)].sort_values("tau_center")
                if item.empty:
                    continue
                label = f"{polarity} {recon}"
                axes[0].plot(item["tau_center"], item["rmse_to_rotation_core_R"], style, label=label)
                axes[1].plot(item["tau_center"], item["corr_to_rotation_core"], style, label=label)
        axes[0].set_title(f"{chain}: RMSE to r_omega")
        axes[0].set_xlabel("tau")
        axes[0].set_ylabel("RMSE / R")
        axes[1].set_title(f"{chain}: correlation to r_omega")
        axes[1].set_xlabel("tau")
        axes[1].set_ylabel("corr")
        for ax in axes:
            ax.grid(True, alpha=0.25)
        axes[1].legend(fontsize=6.5, ncol=2, loc="best")
        fig.savefig(out_fig / f"{chain}_rmse_corr_vs_tau_by_mode.png", dpi=180)
        plt.close(fig)


def _plot_energy(out_dir: Path, energy: pd.DataFrame) -> None:
    out_fig = out_dir / "figures"
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    for ax, polarity in zip(axes, POLARITIES):
        part = energy[energy["polarity"].eq(polarity)]
        for col, style in (
            ("mode1_psi_energy_fraction", "--"),
            ("mode2_psi_energy_fraction", ":"),
            ("mode1_plus_2_psi_energy_fraction", "-"),
            ("mode1_to_5_psi_energy_fraction", "-."),
        ):
            item = part.sort_values("tau_center")
            if col in item:
                ax.plot(item["tau_center"], item[col], style, label=col.replace("_psi_energy_fraction", ""))
        ax.set_title(polarity)
        ax.set_xlabel("tau")
        ax.set_ylabel("psi energy fraction")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7, loc="best")
    fig.savefig(out_fig / "mode_energy_fraction_vs_tau.png", dpi=180)
    plt.close(fig)


def _write_summary(out_dir: Path, decision: pd.DataFrame, metrics: pd.DataFrame, meta: dict) -> None:
    decision_rows = decision.to_dict("records")
    lines = [
        "# 鏃嬭浆鏍稿績涓績绾挎ā鎬佸垎瑙ｉ獙璇?Li/Yang/Xu 2026",
        "",
        "鏈疄楠屽彧璇绘渶鏂?Kuroshiou boundary-monotonic coherent-only ME_LIUTEX 浠ｈ〃娑°€?,
        "涓績绾夸富鍙ｅ緞鏀逛负鏃嬭浆鏍稿績 r_omega锛屽嵆鐢变唬琛ㄦ丁閫熷害鍦虹殑鐩稿娑″害 |zeta| 鏍稿績鏋佸€煎畾涔夈€?,
        "",
        "## 涓ゆ潯楠岃瘉閾?,
        "",
        "- `r_omega_as_target`锛氫笉閲嶅畾蹇冮€熷害鍦猴紝妫€楠屾ā鎬侀噸鏋勪腑蹇冪嚎鏄惁鑳借В閲婂師濮嬫棆杞牳蹇冭酱绾裤€?,
        "- `rotation_core_centered`锛氭瘡灞傚厛浠?r_omega 閲嶅畾蹇冿紝鍐嶅仛妯℃€佸垎瑙ｏ紝妫€楠屽幓闄ゅ弻鏍稿績鍋忓績鍚庢槸鍚︿粛鏀寔浣庢ā鎬佸€炬枩鏈哄埗銆?,
        "",
        "## 鍒ゅ畾缁撴灉",
        "",
    ]
    for row in decision_rows:
        lines += [
            f"- `{row['chain']}`锛歚{row['decision']}`锛泂upport_fraction={row.get('support_fraction', float('nan')):.3f}锛?
            f"low_mode_fraction={row.get('low_mode_fraction', float('nan')):.3f}锛?
            f"median mode1+2 RMSE={row.get('median_mode12_rmse', float('nan')):.3f} R锛?
            f"median mode1..5 RMSE={row.get('median_mode15_rmse', float('nan')):.3f} R銆?,
        ]
    lines += [
        "",
        "## 瑙ｉ噴鍙ｅ緞",
        "",
        "鑻?`r_omega_as_target` 鏀寔鑰?`rotation_core_centered` 涓嶆敮鎸侊紝璇存槑浣庢ā鎬佹満鍒舵洿鍍忔槸鍦ㄨВ閲婃棆杞牳蹇冧綅缃紨鍖栵紝"
        "浣嗗弻鏍稿績鍋忓績鏈韩鏄澶栫粨鏋勩€傝嫢涓ゆ潯閮戒笉鏀寔锛屽垯璇存槑閫熷害瀹氫箟/鏃嬭浆鏍稿績瀹氫箟涓嬶紝Li 2026 鐨?mode1+2 鏈哄埗涓嶈冻浠ヨВ閲婃垜浠殑浠ｈ〃娑°€?,
        "",
        "鏈疄楠屼笉浣跨敤娓╁害涓績浣滀负涓诲畾涔夛紱閫熷害涓績鍜?psi 鏋佸€间腑蹇冨彧浣滀负瀹¤瀵圭収銆?,
        "",
        "## 杈撳叆",
        "",
        f"- RV root: `{meta['rv_root']}`",
        f"- radial seed root: `{meta['radial_seed_root']}`",
        f"- N2 profile: `{meta['n2_profile']}`",
        f"- output: `{meta['output_dir']}`",
    ]
    (out_dir / "rotation_core_modal_tilt_validation_summary_zh.md").write_text("\n".join(lines), encoding="utf-8")
    (out_dir / "metadata" / "run_metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    rv_root = Path(args.rv_root)
    out_dir = Path(args.output_dir)
    for sub in ("figures", "tables", "metadata"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    arrays = _load_azimuthal_npz(Path(args.azimuthal_npz) if args.azimuthal_npz else rv_root / "azimuthal_representative_velocity.npz")
    radial_seed_root = _resolve_radial_seed_root(rv_root, args.radial_seed_root)
    n2_profile = _resolve_n2_profile(rv_root, radial_seed_root, args.n2_profile)

    polarities = _decode_list(np.asarray(arrays["polarities"], dtype=object))
    tau_grid = np.asarray(arrays["tau_grid"], dtype="f8")
    depth = np.asarray(arrays["depth"], dtype="f8")
    radial = np.asarray(arrays["radial"], dtype="f8")
    theta = np.asarray(arrays["theta"], dtype="f8")
    u_mean = np.asarray(arrays["u_mean"], dtype="f8")
    v_mean = np.asarray(arrays["v_mean"], dtype="f8")

    keep_depth = np.isfinite(depth) & (depth <= float(args.max_depth_m) + 1e-6)
    depth = depth[keep_depth]
    u_mean = u_mean[:, :, keep_depth, :, :]
    v_mean = v_mean[:, :, keep_depth, :, :]

    radii = _representative_radius_by_polarity(radial_seed_root)
    latitude_ref = float(args.latitude_ref)
    if not np.isfinite(latitude_ref):
        objects = pd.read_parquet(radial_seed_root / "object_cache" / "selected_lifecycle_objects.parquet", columns=["surface_lat"])
        latitude_ref = float(np.nanmedian(objects["surface_lat"].to_numpy(dtype="f8")))
    geo = geo_params(latitude_ref)
    n2 = load_n2(n2_profile, depth)
    f_profile = (geo.f0 * geo.f0) / np.maximum(n2, 1e-12)
    values, profiles, mode_radii = vertical_mode_decomposition(f_profile, depth, mode_count=int(args.mode_count))
    weights = vertical_weights(depth)
    gram = profiles.T @ (weights[:, None] * profiles)
    orthogonality_error = float(np.nanmax(np.abs(gram - np.eye(gram.shape[0]))))
    _plot_vertical_modes(out_dir, depth, profiles, mode_radii, orthogonality_error)

    velocity_axes = _observed_median_axis(radial_seed_root, tau_grid, radii)
    xy = np.linspace(-float(args.rmax), float(args.rmax), int(args.xy_size), dtype="f8")
    xx_r, yy_r = np.meshgrid(xy, xy)
    core = _core_mask(xx_r, yy_r, float(args.center_core_rmax))

    center_rows: list[dict] = []
    metric_rows: list[dict] = []
    energy_rows: list[dict] = []

    for polarity in POLARITIES:
        if polarity not in polarities:
            continue
        ip = polarities.index(polarity)
        radius_m = radii[polarity]
        x_m = xy * radius_m
        y_m = xy * radius_m
        for it, tau in enumerate(tau_grid):
            if args.smoke and not (polarity == args.smoke_polarity and abs(float(tau) - float(args.smoke_tau)) < 1e-6):
                continue
            u_xy = _polar_to_xy(u_mean[ip, it], radial, theta, xx_r, yy_r)
            v_xy = _polar_to_xy(v_mean[ip, it], radial, theta, xx_r, yy_r)
            omega_x, omega_y, omega_zeta = _rotation_core_centerline(u_xy, v_xy, x_m, y_m, xx_r, yy_r, core)
            vel_x, vel_y = _interp_observed_to_depth(velocity_axes.get((polarity, int(it))), depth)
            center_rows += _axis_rows(polarity, it, float(tau), depth, "observed_rotation_core", omega_x, omega_y, radius_m, {"zeta_core_s-1": omega_zeta})
            center_rows += _axis_rows(polarity, it, float(tau), depth, "observed_velocity_axis", vel_x, vel_y, radius_m)

            psi = streamfunction_from_zeta(_relative_vorticity_xy(u_xy, v_xy, x_m, y_m), y_m, x_m)
            centered_u = _center_layers_on_rotation_core(u_xy, omega_x, omega_y, xy)
            centered_v = _center_layers_on_rotation_core(v_xy, omega_x, omega_y, xy)
            centered_psi = streamfunction_from_zeta(_relative_vorticity_xy(centered_u, centered_v, x_m, y_m), y_m, x_m)

            for chain, field in (("r_omega_as_target", psi), ("rotation_core_centered", centered_psi)):
                parts = _mode_parts(field, profiles, depth)
                energy = {"polarity": polarity, "tau_index": int(it), "tau_center": float(tau), "chain": chain}
                energy.update(_mode_energy_fraction(field, parts, depth, x_m, y_m))
                energy_rows.append(energy)

                target_x = omega_x if chain == "r_omega_as_target" else np.zeros_like(omega_x)
                target_y = omega_y if chain == "r_omega_as_target" else np.zeros_like(omega_y)
                zero_rmse = _rmse_xy(target_x, target_y, np.zeros_like(target_x), np.zeros_like(target_y))
                if chain == "rotation_core_centered":
                    zero_rmse = max(float(np.nanmedian(np.hypot(omega_x, omega_y))), 1e-6)
                for label, part in parts.items():
                    centers = {
                        "psi_abs_extreme": _psi_abs_centerline(part, xx_r, yy_r, core),
                        "speed_min": _speed_min_centerline(part, xx_r, yy_r, x_m, y_m, core),
                    }
                    u_part = centered_u if chain == "rotation_core_centered" and label == "full" else None
                    v_part = centered_v if chain == "rotation_core_centered" and label == "full" else None
                    if label == "full" and chain == "r_omega_as_target":
                        u_part, v_part = u_xy, v_xy
                    if u_part is not None and v_part is not None:
                        centers["rotation_core"] = _rotation_core_centerline(u_part, v_part, x_m, y_m, xx_r, yy_r, core)[:2]
                    else:
                        # For modal reconstructions, define rotation core through reconstructed geostrophic velocity from psi.
                        from src.Legacy.experiments.theory_validation.unified_math import velocity_from_psi

                        uu, vv = velocity_from_psi(part, y_m, x_m)
                        centers["rotation_core"] = _rotation_core_centerline(uu, vv, x_m, y_m, xx_r, yy_r, core)[:2]
                    for center_type, (cx, cy) in centers.items():
                        name = f"{chain}_{label}_{center_type}"
                        center_rows += _axis_rows(polarity, it, float(tau), depth, name, cx, cy, radius_m)
                        metric_rows.append(
                            _compare_to_target(
                                polarity, it, float(tau), chain, label, center_type, cx, cy, target_x, target_y, zero_rmse
                            )
                        )

    centers = pd.DataFrame(center_rows)
    metrics = pd.DataFrame(metric_rows)
    energy = pd.DataFrame(energy_rows)
    decisions = pd.DataFrame([_decision(metrics, "r_omega_as_target"), _decision(metrics, "rotation_core_centered")])

    centers.to_csv(out_dir / "tables" / "rotation_core_axis.csv", index=False)
    centers.to_csv(out_dir / "tables" / "center_definition_comparison.csv", index=False)
    metrics[metrics["chain"].eq("r_omega_as_target")].to_csv(out_dir / "tables" / "r_omega_as_target_metrics.csv", index=False)
    metrics[metrics["chain"].eq("rotation_core_centered")].to_csv(out_dir / "tables" / "rotation_core_centered_metrics.csv", index=False)
    energy.to_csv(out_dir / "tables" / "modal_projection_energy.csv", index=False)
    decisions.to_csv(out_dir / "tables" / "li2026_validation_decision_table.csv", index=False)

    _plot_center_overlay(out_dir, centers, depth, [float(v) for v in args.plot_taus.split(",") if v.strip()])
    _plot_metric_summary(out_dir, metrics)
    _plot_energy(out_dir, energy)

    meta = {
        "rv_root": str(rv_root),
        "radial_seed_root": str(radial_seed_root),
        "n2_profile": str(n2_profile),
        "output_dir": str(out_dir),
        "latitude_ref": latitude_ref,
        "f0": geo.f0,
        "beta": geo.beta,
        "orthogonality_error": orthogonality_error,
        "mode_eigenvalues": values.tolist(),
        "mode_radius_like_m": mode_radii.tolist(),
    }
    _write_summary(out_dir, decisions, metrics, meta)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rv-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--azimuthal-npz", default="")
    parser.add_argument("--radial-seed-root", default="")
    parser.add_argument("--n2-profile", default="")
    parser.add_argument("--latitude-ref", type=float, default=float("nan"))
    parser.add_argument("--max-depth-m", type=float, default=2000.0)
    parser.add_argument("--mode-count", type=int, default=5)
    parser.add_argument("--rmax", type=float, default=2.5)
    parser.add_argument("--xy-size", type=int, default=121)
    parser.add_argument("--center-core-rmax", type=float, default=1.5)
    parser.add_argument("--plot-taus", default="0.25,0.50,0.75")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-polarity", default="anticyclonic")
    parser.add_argument("--smoke-tau", type=float, default=0.5)
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
