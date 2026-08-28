from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .contracts import PostPaths, orientation_roots
from .reports import write_json, write_markdown


def _load_npz(root: Path) -> dict[str, np.ndarray]:
    path = root / "azimuthal_representative_velocity.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def _relative_vorticity_xy(u: np.ndarray, v: np.ndarray, x_m: np.ndarray, y_m: np.ndarray) -> np.ndarray:
    return np.gradient(v, x_m, axis=1, edge_order=1) - np.gradient(u, y_m, axis=0, edge_order=1)


def _polar_to_xy(field: np.ndarray, radial: np.ndarray, theta: np.ndarray, radius_m: float, xx: np.ndarray, yy: np.ndarray) -> np.ndarray:
    rr = np.sqrt(xx * xx + yy * yy) / radius_m
    tt = np.mod(np.arctan2(yy, xx), 2.0 * np.pi)
    ri = np.clip(np.searchsorted(radial, rr) - 1, 0, len(radial) - 1)
    ti = np.clip(np.searchsorted(theta, tt) - 1, 0, len(theta) - 1)
    out = np.asarray(field[ri, ti], dtype="f8")
    out[rr > float(np.nanmax(radial))] = np.nan
    return out


def analyze_double_core(
    *,
    result_root: Path,
    shape: str,
    orientation: str,
    output_dir: Path | None = None,
    core_rmax: float = 1.5,
) -> pd.DataFrame:
    paths = PostPaths(result_root=result_root, shape=shape)
    output = output_dir or paths.double_core_root
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for label, root in orientation_roots(paths, orientation):
        data = _load_npz(root)
        polarities = [str(x) for x in np.asarray(data["polarities"])]
        tau_grid = np.asarray(data["tau_grid"], dtype="f8")
        depth = np.asarray(data["depth"], dtype="f8")
        radial = np.asarray(data["radial"], dtype="f8")
        theta = np.asarray(data["theta"], dtype="f8")
        u = np.asarray(data["u_mean"], dtype="f8")
        v = np.asarray(data["v_mean"], dtype="f8")
        radius = np.asarray(data["radius_mean"], dtype="f8") if "radius_mean" in data else np.array([100000.0])
        for ip, polarity in enumerate(polarities):
            for it, tau in enumerate(tau_grid):
                radius_m = float(np.nanmedian(radius[ip, it])) if radius.ndim == 2 else float(np.nanmedian(radius))
                limit = float(np.nanmax(radial)) * radius_m
                x_axis = np.linspace(-limit, limit, len(radial) * 2 + 1)
                y_axis = x_axis.copy()
                xx, yy = np.meshgrid(x_axis, y_axis)
                core_mask = np.sqrt(xx * xx + yy * yy) <= core_rmax * radius_m
                for iz, depth_m in enumerate(depth):
                    u_xy = _polar_to_xy(u[ip, it, iz], radial, theta, radius_m, xx, yy)
                    v_xy = _polar_to_xy(v[ip, it, iz], radial, theta, radius_m, xx, yy)
                    zeta = _relative_vorticity_xy(u_xy, v_xy, x_axis, y_axis)
                    layer = np.where(core_mask, np.abs(zeta), np.nan)
                    if np.isfinite(layer).any():
                        jj, ii = np.unravel_index(np.nanargmax(layer), layer.shape)
                        omega_x = float(xx[jj, ii])
                        omega_y = float(yy[jj, ii])
                        d_r = float(np.hypot(omega_x, omega_y) / radius_m)
                    else:
                        omega_x = omega_y = d_r = np.nan
                    rows.append(
                        {
                            "shape": shape,
                            "orientation": label,
                            "polarity": polarity,
                            "tau_center": float(tau),
                            "depth_index": int(iz),
                            "depth_m": float(depth_m),
                            "rotation_core_x_m": omega_x,
                            "rotation_core_y_m": omega_y,
                            "D_omega_R": d_r,
                        }
                    )
    table = pd.DataFrame(rows)
    table.to_csv(output / "double_core_separation_metrics.csv", index=False)
    try:
        table.to_parquet(output / "double_core_separation_metrics.parquet", index=False)
    except Exception:
        pass
    _plot_heatmaps(table, output / "figures")
    write_json(output / "double_core_manifest.json", {"shape": shape, "orientation": orientation, "rows": int(len(table))})
    write_markdown(
        output / "double_core_summary_zh.md",
        [
            "# 双核心后处理",
            "",
            "- 速度中心轴线在代表涡合成坐标中作为原点。",
            "- 旋转核心由核心区 `|zeta|` 最大点定义。",
            "- `D_omega_R` 是旋转核心相对速度中心的无量纲距离。",
        ],
    )
    return table


def _plot_heatmaps(table: pd.DataFrame, fig_dir: Path) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    for (orientation, polarity), part in table.groupby(["orientation", "polarity"]):
        pivot = part.pivot_table(index="depth_index", columns="tau_center", values="D_omega_R", aggfunc="mean")
        fig, ax = plt.subplots(figsize=(8, 5))
        im = ax.imshow(pivot.to_numpy(), aspect="auto", origin="upper", cmap="viridis")
        ax.set_title(f"{orientation} {polarity}: rotation-core offset D_omega/R")
        ax.set_xlabel("tau")
        ax.set_ylabel("depth index")
        fig.colorbar(im, ax=ax, label="D_omega/R")
        fig.tight_layout()
        fig.savefig(fig_dir / f"{orientation}_{polarity}_D_omega_heatmap.png", dpi=180)
        plt.close(fig)

