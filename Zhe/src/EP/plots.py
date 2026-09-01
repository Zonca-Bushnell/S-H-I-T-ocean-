from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _matrix(table: pd.DataFrame, field: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tau = np.sort(table["tau"].dropna().unique().astype(float))
    depth = np.sort(table["depth_m"].dropna().unique().astype(float))
    image = np.full((depth.size, tau.size), np.nan)
    for iz, z in enumerate(depth):
        zsub = table[np.isclose(table["depth_m"].astype(float), z)]
        for it, value in enumerate(tau):
            sub = zsub[np.isclose(zsub["tau"].astype(float), value)]
            if not sub.empty:
                image[iz, it] = np.nanmedian(sub[field].to_numpy(float))
    return tau, depth, image


def _save_tau_depth(table: pd.DataFrame, field: str, path: Path, *, title: str, cmap: str = "viridis") -> None:
    import matplotlib.pyplot as plt

    polarities = list(table["polarity"].drop_duplicates())
    fig, axes = plt.subplots(1, len(polarities), figsize=(5.8 * len(polarities), 4.8), squeeze=False)
    for ax, polarity in zip(axes[0], polarities):
        sub = table[table["polarity"].astype(str).eq(str(polarity))]
        tau, depth, image = _matrix(sub, field)
        vmax = np.nanpercentile(np.abs(image), 95) if np.any(np.isfinite(image)) else 1.0
        if cmap == "coolwarm":
            im = ax.pcolormesh(tau, depth, image, shading="auto", cmap=cmap, vmin=-vmax, vmax=vmax)
        else:
            im = ax.pcolormesh(tau, depth, image, shading="auto", cmap=cmap)
        ax.invert_yaxis()
        ax.set_xlabel("life phase tau")
        ax.set_ylabel("depth (m)")
        ax.set_title(str(polarity))
        fig.colorbar(im, ax=ax, shrink=0.82)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_lifecycle_figures(profiles: pd.DataFrame, summary: pd.DataFrame, audit: pd.DataFrame, output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    core = profiles[profiles["radius_over_R"].astype(float) <= 1.5].copy()
    ratio_rows = []
    for keys, part in core.groupby(["polarity", "tau", "depth_m"]):
        ordinary = part["F_z_ordinary"].to_numpy(float)
        correction = part["F_z_tilt_correction"].to_numpy(float)
        ratio_rows.append(
            {
                "polarity": keys[0],
                "tau": float(keys[1]),
                "depth_m": float(keys[2]),
                "tilt_correction_ratio": float(
                    np.nanmedian(np.abs(correction)) / (np.nanmedian(np.abs(ordinary)) + 1e-30)
                ),
            }
        )
    ratio = pd.DataFrame(ratio_rows)
    path = output_dir / "tau_depth_tilt_correction_ratio.png"
    _save_tau_depth(ratio, "tilt_correction_ratio", path, title="|Fz tilt correction| / |Fz ordinary|")
    written.append(path)

    corr_rows = []
    for keys, part in core.groupby(["polarity", "tau", "depth_m"]):
        div = part["divF_tilted"].to_numpy(float)
        pv = part["pv_flux_proxy"].to_numpy(float)
        mask = np.isfinite(div) & np.isfinite(pv)
        corr = np.corrcoef(div[mask], pv[mask])[0, 1] if int(mask.sum()) > 2 else np.nan
        err = np.sqrt(np.nanmean((div - pv) ** 2))
        corr_rows.append({"polarity": keys[0], "tau": float(keys[1]), "depth_m": float(keys[2]), "divF_pv_corr": corr, "divF_pv_rmse": err})
    corr_table = pd.DataFrame(corr_rows)
    path = output_dir / "tau_depth_divF_pv_correlation.png"
    _save_tau_depth(corr_table, "divF_pv_corr", path, title="divF-PV proxy correlation by tau/depth", cmap="coolwarm")
    written.append(path)

    path = output_dir / "tau_depth_metric_valid_fraction.png"
    _save_tau_depth(audit, "metric_valid_fraction", path, title="Curved-tube metric valid fraction")
    written.append(path)

    path = output_dir / "tau_depth_epsilon_curvature_p90.png"
    _save_tau_depth(audit, "epsilon_curvature_p90", path, title="p90 epsilon_curvature = kappa*r")
    written.append(path)
    return written


def plot_cross_combo_summary(summary: pd.DataFrame, output_dir: Path) -> list[Path]:
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    if summary.empty:
        return written
    labels = summary.apply(
        lambda r: f"{r['shape']}|{r['axis_source']}|{r['orientation']}|{r['buoyancy_source']}|{r['polarity']}",
        axis=1,
    )
    grouped = summary.assign(label=labels).groupby("label", as_index=False).agg(
        tilt_ratio=("median_abs_tilt_correction_over_ordinary", "median"),
        closure_corr=("divF_pv_flux_corr_core", "median"),
        metric_valid=("metric_valid_fraction_median", "median"),
    )
    fig, ax = plt.subplots(figsize=(max(8.0, 0.35 * len(grouped)), 5.0))
    x = np.arange(len(grouped))
    ax.plot(x, grouped["tilt_ratio"], "o-", label="tilt correction ratio")
    ax.plot(x, grouped["closure_corr"], "s-", label="divF-PV corr")
    ax.plot(x, grouped["metric_valid"], "^-", label="metric valid fraction")
    ax.set_xticks(x, grouped["label"], rotation=75, ha="right", fontsize=7)
    ax.grid(True, alpha=0.25)
    ax.legend()
    ax.set_title("EP lifecycle cross-combo median diagnostics")
    fig.tight_layout()
    path = output_dir / "cross_combo_summary.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    written.append(path)
    return written

