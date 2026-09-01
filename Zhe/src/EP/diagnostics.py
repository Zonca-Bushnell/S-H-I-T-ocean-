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


def build_smoke(config: EPFluxConfig, *, n2_profile: str | None = "auto") -> dict[str, Path]:
    config.validate_contract()
    dataset = RepresentativeVortexDataset.load(config.vortex_npz, config.radial_seed_root)
    axis_path = config.axis_source_path
    if not axis_path.exists():
        raise FileNotFoundError(
            f"Axis source does not exist: {axis_path}. "
            "Run src.post.cli build-representative-axis-sources first."
        )

    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
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
        ).compute()
        all_profiles.append(result.profiles)
        metric_rows.append({"polarity": polarity, **result.metrics})

    profiles = pd.concat(all_profiles, ignore_index=True)
    metrics = pd.DataFrame(metric_rows)
    profiles_path = output_dir / "ep_flux_smoke_profiles.csv"
    metrics_path = output_dir / "ep_flux_smoke_metrics.csv"
    manifest_path = output_dir / "ep_flux_smoke_manifest.json"
    summary_path = output_dir / "ep_flux_smoke_summary_zh.md"

    profiles.to_csv(profiles_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    manifest = {
        **config.manifest(),
        "n2_profile_path": str(n2_profile_path) if n2_profile_path else None,
        "n2_source": "profile_npz" if n2_profile_path else "constant_smoke_N2",
        "polarity_count": len(dataset.polarities),
        "profile_rows": int(len(profiles)),
    }
    manifest_path.write_text(json.dumps(_json_ready(manifest), ensure_ascii=False, indent=2), encoding="utf-8")
    summary_path.write_text(_summary_markdown(config, metrics), encoding="utf-8")

    figures = output_dir / "figures"
    figures.mkdir(exist_ok=True)
    div_path = figures / "classic_tilted_curved_divergence_tau_depth.png"
    tilt_path = figures / "axis_forcing_tilt_contribution.png"
    plot_divergence_comparison(profiles, config, div_path)
    plot_axis_tilt_contribution(profiles, config, tilt_path)

    return {
        "profiles": profiles_path,
        "metrics": metrics_path,
        "manifest": manifest_path,
        "summary": summary_path,
        "divergence_figure": div_path,
        "tilt_figure": tilt_path,
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
        sensitivity = core["divF_curvature_sensitivity_upper"].to_numpy(float)
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


def _summary_markdown(config: EPFluxConfig, metrics: pd.DataFrame) -> str:
    lines = [
        "# EP Flux Smoke Summary",
        "",
        "## 口径",
        f"- 代表涡：`{config.shape_label}` / `{config.orientation}` / `tau={config.tau:.2f}`",
        f"- Axis source：`{config.axis_source}`",
        f"- Buoyancy source：`{config.buoyancy_source}`",
        "- Classic EP、tilted EP 与 curved-tube EP-QG 近似同时输出。",
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
        "- `streamfunction_dz` 口径保留 \\(b'=f_0\\partial_z\\psi\\) 的 QG 对照。",
        "- `divF_curved_tube_qg_approx` 是一阶截面平均下的保守 curved-tube resolved 项；当前不把曲率上界项直接加进主散度。",
        "- `divF_curvature_sensitivity_upper` 是 \\(\\kappa F_n\\) 量级敏感性上界，用来提示曲率可能重要，但不能单独作为闭合结论。",
        "- `pv_flux_proxy` 来自代表流函数的 QG-like PV 代理，用于闭合关系的回归检查。",
        "",
        "## 注意",
        "- 第一版 curved-tube 中的张量几何和连接项仍是 QG 近似框架，完整 \\(T^{ia}\\) 需要后续理论实现。",
        "- 若 sensitivity 远大于 resolved divergence，应解释为“曲率项需要完整二维截面张量闭合”，而不是直接判定 curved-tube forcing 极大。",
    ]
    return "\n".join(lines) + "\n"
