from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def _plane(center: np.ndarray, direction: np.ndarray, half_width: float, z_top: float, z_bottom: float) -> list[np.ndarray]:
    top0 = center - half_width * direction + np.array([0.0, 0.0, z_top])
    top1 = center + half_width * direction + np.array([0.0, 0.0, z_top])
    bot1 = center + half_width * direction + np.array([0.0, 0.0, z_bottom])
    bot0 = center - half_width * direction + np.array([0.0, 0.0, z_bottom])
    return [top0, top1, bot1, bot0]


def plot_jump_section_geometry(output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    upper = np.array([-18.0, -8.0, -60.0])
    lower = np.array([32.0, 18.0, -90.0])
    midpoint = 0.5 * (upper + lower)
    jump = lower[:2] - upper[:2]
    jump = jump / np.linalg.norm(jump)
    normal = np.array([-jump[1], jump[0]])
    parallel3 = np.array([jump[0], jump[1], 0.0])
    normal3 = np.array([normal[0], normal[1], 0.0])

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    z_top, z_bottom = -35.0, -125.0
    p_parallel = _plane(midpoint * np.array([1, 1, 0]), parallel3, 95.0, z_top, z_bottom)
    p_normal = _plane(midpoint * np.array([1, 1, 0]), normal3, 80.0, z_top, z_bottom)
    ax.add_collection3d(Poly3DCollection([p_parallel], facecolor="tab:blue", alpha=0.18, edgecolor="tab:blue", linewidth=1.5))
    ax.add_collection3d(Poly3DCollection([p_normal], facecolor="tab:orange", alpha=0.20, edgecolor="tab:orange", linewidth=1.5))

    ax.plot([upper[0], lower[0]], [upper[1], lower[1]], [upper[2], lower[2]], color="crimson", lw=3, label="center jump vector")
    ax.scatter([upper[0]], [upper[1]], [upper[2]], s=80, c="cyan", edgecolor="k", label="upper/from center")
    ax.scatter([lower[0]], [lower[1]], [lower[2]], s=80, c="yellow", edgecolor="k", label="lower/to center")
    ax.quiver(upper[0], upper[1], upper[2], lower[0] - upper[0], lower[1] - upper[1], lower[2] - upper[2], color="crimson", arrow_length_ratio=0.18)
    ax.quiver(midpoint[0], midpoint[1], z_top, 45 * jump[0], 45 * jump[1], 0, color="tab:blue", arrow_length_ratio=0.15)
    ax.quiver(midpoint[0], midpoint[1], z_top, 45 * normal[0], 45 * normal[1], 0, color="tab:orange", arrow_length_ratio=0.15)
    ax.text(midpoint[0] + 52 * jump[0], midpoint[1] + 52 * jump[1], z_top, "jump-parallel", color="tab:blue")
    ax.text(midpoint[0] + 52 * normal[0], midpoint[1] + 52 * normal[1], z_top, "jump-normal", color="tab:orange")

    ax.set_title("Jump section geometry: parallel vs normal")
    ax.set_xlabel("east from surface center (km)")
    ax.set_ylabel("north from surface center (km)")
    ax.set_zlabel("depth (m)")
    ax.set_xlim(-110, 110)
    ax.set_ylim(-95, 95)
    ax.set_zlim(-140, -25)
    ax.view_init(elev=23, azim=-57)
    ax.legend(loc="upper left")
    fig.text(
        0.08,
        0.02,
        "jump-parallel follows the center displacement; jump-normal cuts across it through the midpoint.\n"
        "Section velocity zero lines are diagnostics, not Hua/VG center definitions.",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw a 3D schematic of jump-parallel and jump-normal sections.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plot_jump_section_geometry(args.output)


if __name__ == "__main__":
    main()
