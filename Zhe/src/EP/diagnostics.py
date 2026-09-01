from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .contracts import EPFluxConfig, RHO0
from .fields import RepresentativeVortexDataset
from .flux import EPFluxCalculator
from .geometry import AxisLine


def _json_ready(value: object) -> object:
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def resolve_n2_profile_path(config: EPFluxConfig, requested: str | None) -> Path | None:
    if requested in (None, "", "none"):
        return None
    if requested != "auto":
        return Path(requested)
    candidates = [
        config.radial_seed_root
        / "climatology"
        / "cmems_doy_climatology_1993_2022_31d_sigma0_dz_profile.npz",
        config.radial_seed_root / "N2" / "sigma0_dz_profile.npz",
        config.me_liutex_root.parent / "N2" / "sigma0_dz_profile.npz",
        config.me_liutex_root.parent.parent / "N2" / "sigma0_dz_profile.npz",
        Path("/root/autodl-fs/kuroshiou/N2/sigma0_dz_profile.npz"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def load_n2_profile(path: Path | None, depth_m: np.ndarray, constant_n2: float) -> np.ndarray:
    if path is None:
        return np.full_like(depth_m, constant_n2, dtype=float)
    if not path.exists():
        raise FileNotFoundError(path)
    data = np.load(path, allow_pickle=True)
    n2 = None
    src_depth = None
    for key in ("N2", "n2", "n2_profile", "N2_profile"):
        if key in data:
            n2 = np.asarray(data[key], dtype=float)
            break
    if n2 is None and "dsigma0_dz" in data:
        n2 = 9.81 * np.asarray(data["dsigma0_dz"], dtype=float) / RHO0
    if n2 is None:
        raise KeyError(f"No N2 array found in {path}")
    if "depth" in data:
        src_depth = np.asarray(data["depth"], dtype=float)
    elif "depth_m" in data:
        src_depth = np.asarray(data["depth_m"], dtype=float)
    else:
        src_depth = np.linspace(float(depth_m.min()), float(depth_m.max()), n2.size)
    return np.interp(depth_m, src_depth, n2)


def compute_ep_profiles(
    config: EPFluxConfig,
    *,
    n2_profile: str | None = "auto",
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    config.validate_contract()
    dataset = RepresentativeVortexDataset.load(config.vortex_npz, config.radial_seed_root)
    axis_path = config.axis_source_path
    if not axis_path.exists():
        raise FileNotFoundError(
            f"Axis source does not exist: {axis_path}. "
            "Run src.post.cli build-representative-axis-sources first."
        )

    all_profiles: list[pd.DataFrame] = []
    metric_rows: list[dict[str, object]] = []
    n2_profile_path = resolve_n2_profile_path(config, n2_profile)

    for polarity in dataset.polarities:
        rep = dataset.slice(polarity, config.tau)
        axis = AxisLine.from_csv(axis_path, polarity=polarity)
        n2 = load_n2_profile(n2_profile_path, rep.depth_m, config.constant_n2)
        result = EPFluxCalculator(
            rep,
            axis,
            f0=config.f0,
            n2=n2,
            buoyancy_source=config.buoyancy_source,
            curved_tube_mode=config.curved_tube_mode,
            large_curvature_threshold=config.large_curvature_threshold,
        ).compute()
        p_idx = dataset.polarity_index(polarity)
        t_idx = dataset.nearest_tau_index(config.tau)
        if dataset.n_objects is not None:
            result.profiles["n_objects"] = int(dataset.n_objects[p_idx, t_idx])
        if dataset.n_tracks is not None:
            result.profiles["n_tracks"] = int(dataset.n_tracks[p_idx, t_idx])
        all_profiles.append(result.profiles)
        metric_rows.append({"polarity": polarity, **result.metrics})

    profiles = pd.concat(all_profiles, ignore_index=True)
    metrics = pd.DataFrame(metric_rows)
    manifest = {
        **config.manifest(),
        "n2_profile_path": str(n2_profile_path) if n2_profile_path else None,
        "n2_source": "profile_npz" if n2_profile_path else "constant_smoke_N2",
        "polarity_count": len(dataset.polarities),
        "profile_rows": int(len(profiles)),
    }
    return profiles, metrics, manifest


def build_smoke(config: EPFluxConfig, *, n2_profile: str | None = "auto") -> dict[str, Path]:
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    profiles, metrics, manifest = compute_ep_profiles(config, n2_profile=n2_profile)
    profiles_path = output_dir / "ep_flux_smoke_profiles.csv"
    metrics_path = output_dir / "ep_flux_smoke_metrics.csv"
    geometry_metrics_path = output_dir / "ep_flux_geometry_scale_metrics.csv"
    curved_terms_path = output_dir / "ep_flux_curved_terms_profiles.csv"
    manifest_path = output_dir / "ep_flux_smoke_manifest.json"
    summary_path = output_dir / "ep_flux_smoke_summary_zh.md"
    curved_summary_path = output_dir / "curved_tube_scale_audit_summary_zh.md"

    profiles.to_csv(profiles_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    geometry_metrics = _geometry_scale_metrics(profiles)
    curved_terms = _curved_terms_profiles(profiles)
    geometry_metrics.to_csv(geometry_metrics_path, index=False)
    curved_terms.to_csv(curved_terms_path, index=False)
    manifest_path.write_text(json.dumps(_json_ready(manifest), ensure_ascii=False, indent=2), encoding="utf-8")
    summary_path.write_text(_summary_markdown(config, metrics), encoding="utf-8")
    curved_summary_path.write_text(
        _curved_scale_summary_markdown(config, metrics, geometry_metrics),
        encoding="utf-8",
    )

    figures = output_dir / "figures"
    figures.mkdir(exist_ok=True)
    div_path = figures / "classic_tilted_curved_divergence_tau_depth.png"
    tilt_path = figures / "axis_forcing_tilt_contribution.png"
    jacobian_path = figures / "jacobian_range_tau_depth.png"
    curved_terms_path_png = figures / "curved_terms_over_tilted.png"
    validity_path = figures / "metric_validity_mask.png"
    plot_divergence_comparison(profiles, config, div_path)
    plot_axis_tilt_contribution(profiles, config, tilt_path)
    plot_jacobian_range(profiles, config, jacobian_path)
    plot_curved_terms_over_tilted(profiles, config, curved_terms_path_png)
    plot_metric_validity(profiles, config, validity_path)

    return {
        "profiles": profiles_path,
        "metrics": metrics_path,
        "geometry_scale_metrics": geometry_metrics_path,
        "curved_terms_profiles": curved_terms_path,
        "manifest": manifest_path,
        "summary": summary_path,
        "curved_scale_summary": curved_summary_path,
        "divergence_figure": div_path,
        "tilt_figure": tilt_path,
        "jacobian_range_figure": jacobian_path,
        "curved_terms_figure": curved_terms_path_png,
        "metric_validity_figure": validity_path,
    }


def compare_classic_and_curved(output_dir: Path) -> dict[str, Path]:
    profiles_path = output_dir / "ep_flux_smoke_profiles.csv"
    if not profiles_path.exists():
        raise FileNotFoundError(f"Run build-smoke first; missing {profiles_path}")
    profiles = pd.read_csv(profiles_path)
    rows = []
    for polarity, sub in profiles.groupby("polarity"):
        core = sub[sub["radius_over_R"] <= 1.5]
        classic = core["divF_classic"].to_numpy(float)
        tilted = core["divF_tilted"].to_numpy(float)
        curved = core["divF_curved_tube_qg_approx"].to_numpy(float)
        sensitivity_field = (
            "divF_scale_upper_bound"
            if "divF_scale_upper_bound" in core.columns
            else "divF_curvature_sensitivity_upper"
        )
        sensitivity = core[sensitivity_field].to_numpy(float)
        rows.append(
            {
                "polarity": polarity,
                "median_abs_classic": float(np.nanmedian(np.abs(classic))),
                "median_abs_tilted": float(np.nanmedian(np.abs(tilted))),
                "median_abs_curved": float(np.nanmedian(np.abs(curved))),
                "median_abs_tilted_minus_classic": float(np.nanmedian(np.abs(tilted - classic))),
                "median_abs_curved_minus_tilted": float(np.nanmedian(np.abs(curved - tilted))),
                "median_abs_curvature_sensitivity_upper": float(np.nanmedian(np.abs(sensitivity))),
            }
        )
    table = pd.DataFrame(rows)
    out_path = output_dir / "classic_tilted_curved_comparison.csv"
    table.to_csv(out_path, index=False)
    return {"comparison": out_path}


def _curved_terms_profiles(profiles: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "polarity",
        "tau",
        "depth_m",
        "radius_m",
        "radius_over_R",
        "divF_tilted",
        "divF_jacobian",
        "divF_jacobian_correction",
        "divF_christoffel_qg_approx",
        "divF_curved_total",
        "divF_curved_tube_qg_approx",
        "divF_scale_upper_bound",
        "metric_valid_fraction",
        "metric_invalid_or_large_curvature",
    ]
    return profiles[[col for col in columns if col in profiles.columns]].copy()


def _geometry_scale_metrics(profiles: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (polarity, depth), sub in profiles.groupby(["polarity", "depth_m"], sort=True):
        core = sub[sub["radius_over_R"] <= 1.5]
        if core.empty:
            continue
        rows.append(
            {
                "polarity": polarity,
                "tau": float(np.nanmedian(core["tau"])),
                "depth_m": float(depth),
                "epsilon_tilt_median": float(np.nanmedian(np.abs(core["epsilon_tilt"]))),
                "epsilon_curvature_median": float(np.nanmedian(np.abs(core["epsilon_curvature"]))),
                "epsilon_curvature_p90": float(np.nanpercentile(np.abs(core["epsilon_curvature"]), 90)),
                "jacobian_min": float(np.nanmin(core["jacobian_min"])),
                "jacobian_max": float(np.nanmax(core["jacobian_max"])),
                "metric_valid_fraction": float(np.nanmean(core["metric_valid_fraction"])),
                "metric_invalid_or_large_curvature_fraction": float(
                    np.nanmean(core["metric_invalid_or_large_curvature"].astype(float))
                ),
            }
        )
    return pd.DataFrame(rows)


def plot_divergence_comparison(profiles: pd.DataFrame, config: EPFluxConfig, path: Path) -> None:
    import matplotlib.pyplot as plt

    polarities = list(profiles["polarity"].drop_duplicates())
    fields = ["divF_classic", "divF_tilted", "divF_curved_tube_qg_approx"]
    fig, axes = plt.subplots(1, len(polarities), figsize=(5.5 * len(polarities), 5.0), squeeze=False)
    for ax, polarity in zip(axes[0], polarities):
        sub = profiles[(profiles["polarity"] == polarity) & (profiles["radius_over_R"] <= 1.5)]
        depth = np.sort(sub["depth_m"].unique())
        image = np.full((depth.size, len(fields)), np.nan)
        for iz, z in enumerate(depth):
            zsub = sub[sub["depth_m"] == z]
            for jf, field in enumerate(fields):
                image[iz, jf] = np.nanmedian(zsub[field].to_numpy(float))
        vmax = np.nanpercentile(np.abs(image), 95)
        vmax = vmax if np.isfinite(vmax) and vmax > 0 else 1.0
        im = ax.imshow(image, aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
        ax.set_xticks(range(len(fields)), ["classic", "tilted", "curved"], rotation=20)
        ax.set_yticks(np.linspace(0, depth.size - 1, min(6, depth.size)).astype(int))
        ax.set_yticklabels([f"{depth[i]:.0f}" for i in ax.get_yticks().astype(int)])
        ax.set_ylabel("depth (m)")
        ax.set_title(f"{polarity}: median core divF")
        fig.colorbar(im, ax=ax, shrink=0.8, label="diagnostic units")
    fig.suptitle(
        f"EP smoke divergence comparison, {config.orientation}, {config.axis_source}, tau={config.tau:.2f}"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_axis_tilt_contribution(profiles: pd.DataFrame, config: EPFluxConfig, path: Path) -> None:
    import matplotlib.pyplot as plt

    polarities = list(profiles["polarity"].drop_duplicates())
    fig, axes = plt.subplots(1, len(polarities), figsize=(5.0 * len(polarities), 5.0), squeeze=False)
    for ax, polarity in zip(axes[0], polarities):
        sub = profiles[(profiles["polarity"] == polarity) & (profiles["radius_over_R"] <= 1.5)]
        grouped = sub.groupby("depth_m", as_index=False).agg(
            axis_tilt_km=("axis_tilt_km", "median"),
            F_z_ordinary=("F_z_ordinary", lambda s: np.nanmedian(np.abs(s))),
            F_z_tilt_correction=("F_z_tilt_correction", lambda s: np.nanmedian(np.abs(s))),
        )
        ratio = grouped["F_z_tilt_correction"] / (grouped["F_z_ordinary"] + 1e-30)
        ax.plot(grouped["axis_tilt_km"], grouped["depth_m"], label="axis tilt (km)", color="#3159a6")
        ax2 = ax.twiny()
        ax2.plot(ratio, grouped["depth_m"], label="|tilt correction| / |ordinary|", color="#c44e52")
        ax.invert_yaxis()
        ax.set_xlabel("axis tilt (km)")
        ax.set_ylabel("depth (m)")
        ax2.set_xlabel("tilt correction ratio")
        ax.set_title(polarity)
        ax.grid(True, alpha=0.25)
    fig.suptitle(f"Axis forcing and tilted EP contribution, {config.axis_source}")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_jacobian_range(profiles: pd.DataFrame, config: EPFluxConfig, path: Path) -> None:
    import matplotlib.pyplot as plt

    polarities = list(profiles["polarity"].drop_duplicates())
    fields = ["jacobian_min", "jacobian_max"]
    fig, axes = plt.subplots(1, len(polarities), figsize=(5.5 * len(polarities), 5.0), squeeze=False)
    for ax, polarity in zip(axes[0], polarities):
        sub = profiles[(profiles["polarity"] == polarity) & (profiles["radius_over_R"] <= 1.5)]
        depth = np.sort(sub["depth_m"].unique())
        image = np.full((depth.size, len(fields)), np.nan)
        for iz, z in enumerate(depth):
            zsub = sub[sub["depth_m"] == z]
            image[iz, 0] = np.nanmin(zsub["jacobian_min"].to_numpy(float))
            image[iz, 1] = np.nanmax(zsub["jacobian_max"].to_numpy(float))
        im = ax.imshow(image, aspect="auto", cmap="viridis", vmin=0.0, vmax=max(1.0, np.nanmax(image)))
        ax.set_xticks(range(len(fields)), ["J min", "J max"])
        ax.set_yticks(np.linspace(0, depth.size - 1, min(6, depth.size)).astype(int))
        ax.set_yticklabels([f"{depth[i]:.0f}" for i in ax.get_yticks().astype(int)])
        ax.set_ylabel("depth (m)")
        ax.set_title(polarity)
        fig.colorbar(im, ax=ax, shrink=0.8, label="Jacobian")
    fig.suptitle(f"Curved-tube first-order Jacobian range, {config.curved_tube_mode}")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_curved_terms_over_tilted(profiles: pd.DataFrame, config: EPFluxConfig, path: Path) -> None:
    import matplotlib.pyplot as plt

    fields = [
        ("divF_jacobian_correction", "Jacobian correction"),
        ("divF_christoffel_qg_approx", "Christoffel QG approx"),
        ("divF_curved_total", "Curved total - tilted"),
        ("divF_scale_upper_bound", "Scale upper bound"),
    ]
    polarities = list(profiles["polarity"].drop_duplicates())
    fig, axes = plt.subplots(1, len(polarities), figsize=(6.0 * len(polarities), 5.0), squeeze=False)
    for ax, polarity in zip(axes[0], polarities):
        sub = profiles[(profiles["polarity"] == polarity) & (profiles["radius_over_R"] <= 1.5)]
        depth = np.sort(sub["depth_m"].unique())
        image = np.full((depth.size, len(fields)), np.nan)
        for iz, z in enumerate(depth):
            zsub = sub[sub["depth_m"] == z]
            denom = np.nanmedian(np.abs(zsub["divF_tilted"].to_numpy(float))) + 1e-30
            for jf, (field, _) in enumerate(fields):
                values = zsub[field].to_numpy(float)
                if field == "divF_curved_total":
                    values = values - zsub["divF_tilted"].to_numpy(float)
                image[iz, jf] = np.nanmedian(np.abs(values)) / denom
        vmax = np.nanpercentile(image, 95)
        vmax = vmax if np.isfinite(vmax) and vmax > 0 else 1.0
        im = ax.imshow(image, aspect="auto", cmap="magma", vmin=0.0, vmax=vmax)
        ax.set_xticks(range(len(fields)), [label for _, label in fields], rotation=25, ha="right")
        ax.set_yticks(np.linspace(0, depth.size - 1, min(6, depth.size)).astype(int))
        ax.set_yticklabels([f"{depth[i]:.0f}" for i in ax.get_yticks().astype(int)])
        ax.set_ylabel("depth (m)")
        ax.set_title(polarity)
        fig.colorbar(im, ax=ax, shrink=0.8, label="median |term| / median |tilted divF|")
    fig.suptitle(f"Curved-tube term scale audit, {config.orientation}, {config.axis_source}")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_metric_validity(profiles: pd.DataFrame, config: EPFluxConfig, path: Path) -> None:
    import matplotlib.pyplot as plt

    polarities = list(profiles["polarity"].drop_duplicates())
    fig, axes = plt.subplots(1, len(polarities), figsize=(5.5 * len(polarities), 5.0), squeeze=False)
    for ax, polarity in zip(axes[0], polarities):
        sub = profiles[profiles["polarity"] == polarity]
        depth = np.sort(sub["depth_m"].unique())
        radius = np.sort(sub["radius_over_R"].unique())
        image = np.full((depth.size, radius.size), np.nan)
        for iz, z in enumerate(depth):
            zsub = sub[sub["depth_m"] == z]
            for ir, r in enumerate(radius):
                rsub = zsub[np.isclose(zsub["radius_over_R"], r)]
                if not rsub.empty:
                    image[iz, ir] = np.nanmean(rsub["metric_valid_fraction"].to_numpy(float))
        im = ax.imshow(image, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
        ax.set_xticks(np.linspace(0, radius.size - 1, min(5, radius.size)).astype(int))
        ax.set_xticklabels([f"{radius[i]:.1f}" for i in ax.get_xticks().astype(int)])
        ax.set_yticks(np.linspace(0, depth.size - 1, min(6, depth.size)).astype(int))
        ax.set_yticklabels([f"{depth[i]:.0f}" for i in ax.get_yticks().astype(int)])
        ax.set_xlabel("r/R")
        ax.set_ylabel("depth (m)")
        ax.set_title(polarity)
        fig.colorbar(im, ax=ax, shrink=0.8, label="valid metric fraction")
    fig.suptitle(f"Metric validity mask, kappa*r threshold={config.large_curvature_threshold:g}")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _summary_markdown(config: EPFluxConfig, metrics: pd.DataFrame) -> str:
    lines = [
        "# EP Flux Smoke Summary",
        "",
        "## 口径",
        f"- 代表涡：`{config.shape_label}` / `{config.orientation}` / `tau={config.tau:.2f}`",
        f"- Axis source：`{config.axis_source}`",
        f"- Buoyancy source：`{config.buoyancy_source}`",
        f"- Curved-tube mode：`{config.curved_tube_mode}`",
        "- Classic EP、tilted EP 与 curved-tube 几何审计项同时输出。",
        "- 本报告是 smoke 验证，不是多年全量结论。",
        "",
        "## 关键数值检查",
        "```text",
        metrics.to_string(index=False),
        "```",
        "",
        "## 解释",
        "- `F_z_tilt_correction` 衡量中心轴倾斜导致的垂向导数修正。",
        "- `thermal_wind` 口径用合成地转速度垂向切变反推浮力异常梯度，再恢复截面内的 \\(b'\\)。",
        "- `divF_jacobian` 是带一阶 Jacobian 的通量散度。",
        "- `divF_christoffel_qg_approx` 是 Bishop-frame 下的一阶 QG 几何联络近似。",
        "- `divF_scale_upper_bound` 是 \\(\\kappa F_n\\) 量级上界，只用于提示曲率可能重要。",
        "- `streamfunction_dz` 口径保留 \\(b'=f_0\\partial_z\\psi\\) 的 QG 对照。",
        "- `pv_flux_proxy` 来自代表流函数的 QG-like PV 代理，用于闭合关系的回归检查。",
        "",
        "## 注意",
        "- 第一版 curved-tube 中的张量几何和连接项仍是 QG 近似框架，完整 \\(T^{ia}\\) 需要后续理论实现。",
        "- 若 sensitivity 远大于 resolved divergence，应解释为“曲率项需要完整二维截面张量闭合”，而不是直接判定 curved-tube forcing 极大。",
    ]
    return "\n".join(lines) + "\n"


def _curved_scale_summary_markdown(
    config: EPFluxConfig,
    metrics: pd.DataFrame,
    geometry_metrics: pd.DataFrame,
) -> str:
    grouped = geometry_metrics.groupby("polarity", as_index=False).agg(
        metric_valid_fraction=("metric_valid_fraction", "median"),
        epsilon_curvature_median=("epsilon_curvature_median", "median"),
        epsilon_curvature_p90=("epsilon_curvature_p90", "median"),
        jacobian_min=("jacobian_min", "min"),
        jacobian_max=("jacobian_max", "max"),
    )
    lines = [
        "# Curved-Tube Scale Audit Summary",
        "",
        "## 审计口径",
        f"- Curved-tube mode：`{config.curved_tube_mode}`。",
        f"- 大曲率阈值：`kappa*r <= {config.large_curvature_threshold:g}` 才视为一阶 metric 有效。",
        "- 主物理解释仍以 `divF_tilted` 为基准；本文件只评估 Jacobian 与 Christoffel 项的尺度。",
        "",
        "## Polarity 汇总",
        "```text",
        metrics.to_string(index=False),
        "```",
        "",
        "## 几何尺度汇总",
        "```text",
        grouped.to_string(index=False),
        "```",
        "",
        "## 判读规则",
        "- `epsilon_curvature = kappa*r` 若接近或大于 1，说明一阶曲管展开不能作为强结论。",
        "- `jacobian_min <= 0` 表示一阶坐标映射局部失效，该层/半径只保留为风险提示。",
        "- `divF_christoffel_qg_approx` 只是一阶联络项近似；若其量级很大，下一步应回到完整截面张量。",
    ]
    return "\n".join(lines) + "\n"
