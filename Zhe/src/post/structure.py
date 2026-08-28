from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .contracts import PostPaths, orientation_roots
from .reports import write_json, write_markdown


def _load_npz(root: Path) -> dict[str, np.ndarray]:
    path = root / "azimuthal_representative_velocity.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def _polar_mesh(radial: np.ndarray, theta: np.ndarray, radius_m: float) -> tuple[np.ndarray, np.ndarray]:
    rr, tt = np.meshgrid(radial * radius_m / 1000.0, theta, indexing="ij")
    return rr * np.cos(tt), rr * np.sin(tt)


def plot_structure(
    *,
    result_root: Path,
    shape: str,
    orientation: str,
    output_dir: Path | None = None,
) -> list[Path]:
    paths = PostPaths(result_root=result_root, shape=shape)
    output = output_dir or paths.figures_root
    output.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for label, root in orientation_roots(paths, orientation):
        data = _load_npz(root)
        polarities = [str(x) for x in np.asarray(data["polarities"])]
        tau_grid = np.asarray(data["tau_grid"], dtype="f8")
        depth = np.asarray(data["depth"], dtype="f8")
        radial = np.asarray(data["radial"], dtype="f8")
        theta = np.asarray(data["theta"], dtype="f8")
        speed = np.asarray(data["speed_mean"], dtype="f8")
        radius = np.asarray(data["radius_mean"], dtype="f8") if "radius_mean" in data else np.array([100000.0])
        tau_i = int(np.nanargmin(np.abs(tau_grid - 0.5)))
        finite_depth = np.where(np.isfinite(depth))[0]
        depth_i = int(finite_depth[len(finite_depth) // 2]) if finite_depth.size else 0
        for ip, polarity in enumerate(polarities):
            radius_m = float(np.nanmedian(radius[ip, tau_i])) if radius.ndim == 2 else float(np.nanmedian(radius))
            x, y = _polar_mesh(radial, theta, radius_m)
            layer = speed[ip, tau_i, depth_i]
            fig, ax = plt.subplots(figsize=(7, 6))
            im = ax.pcolormesh(x, y, layer, shading="auto", cmap="magma")
            ax.set_aspect("equal")
            ax.set_title(f"{shape}-only {label} speed, {polarity}, tau={tau_grid[tau_i]:.2f}")
            ax.set_xlabel("x_rot km" if label == "turned" else "east km")
            ax.set_ylabel("y_rot km" if label == "turned" else "north km")
            fig.colorbar(im, ax=ax, label="speed (m/s)")
            fig.tight_layout()
            out = output / f"{shape}_{label}_{polarity}_speed_tau050_middepth.png"
            fig.savefig(out, dpi=180)
            plt.close(fig)
            written.append(out)
    write_json(output / "structure_manifest.json", {"shape": shape, "orientation": orientation, "figures": [str(p) for p in written]})
    write_markdown(
        output / "structure_summary_zh.md",
        [
            "# 代表涡结构后处理",
            "",
            f"- shape: `{shape}`",
            f"- orientation: `{orientation}`",
            "- TURN 是 global_ls_alpha 转向后的合成坐标；UNTURN 是原始 east/north 坐标直接合成。",
            "- 本图展示结构场，不能替代 `aggregate-product` 输送诊断。",
        ],
    )
    return written

