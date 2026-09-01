from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


def _first_existing(columns: set[str], names: tuple[str, ...]) -> str | None:
    for name in names:
        if name in columns:
            return name
    return None


@dataclass(frozen=True)
class AxisLine:
    depth_m: np.ndarray
    x_km: np.ndarray
    y_km: np.ndarray
    source_path: Path | None = None

    @classmethod
    def from_csv(cls, path: Path, polarity: str | None = None) -> "AxisLine":
        table = pd.read_csv(path)
        if polarity and "polarity" in table.columns:
            table = table[table["polarity"].astype(str) == str(polarity)]
        if table.empty:
            raise ValueError(f"No axis rows found in {path} for polarity={polarity}")

        cols = set(table.columns)
        depth_col = _first_existing(cols, ("depth_m", "depth", "z_m", "z"))
        x_col = _first_existing(cols, ("x_km", "axis_x_km", "center_x_km", "x"))
        y_col = _first_existing(cols, ("y_km", "axis_y_km", "center_y_km", "y"))
        if depth_col is None or x_col is None or y_col is None:
            raise ValueError(
                f"Axis file {path} needs depth/x/y columns; found {list(table.columns)}"
            )

        out = table[[depth_col, x_col, y_col]].rename(
            columns={depth_col: "depth_m", x_col: "x_km", y_col: "y_km"}
        )
        out = out.dropna().sort_values("depth_m")
        return cls(
            depth_m=out["depth_m"].to_numpy(float),
            x_km=out["x_km"].to_numpy(float),
            y_km=out["y_km"].to_numpy(float),
            source_path=path,
        )

    @classmethod
    def zero(cls, depth_m: np.ndarray) -> "AxisLine":
        depth = np.asarray(depth_m, dtype=float)
        return cls(depth_m=depth, x_km=np.zeros_like(depth), y_km=np.zeros_like(depth))

    def interpolate_to(self, depth_m: np.ndarray) -> "AxisLine":
        target = np.asarray(depth_m, dtype=float)
        return AxisLine(
            depth_m=target,
            x_km=np.interp(target, self.depth_m, self.x_km),
            y_km=np.interp(target, self.depth_m, self.y_km),
            source_path=self.source_path,
        )

    @property
    def tilt_km(self) -> np.ndarray:
        return np.hypot(self.x_km - self.x_km[0], self.y_km - self.y_km[0])

    def slopes_m_per_m(self) -> tuple[np.ndarray, np.ndarray]:
        depth = np.asarray(self.depth_m, dtype=float)
        safe_depth = depth.copy()
        if safe_depth.size > 1 and np.nanmax(np.diff(safe_depth)) == 0:
            safe_depth = np.arange(safe_depth.size, dtype=float)
        dx_dz = np.gradient(self.x_km * 1000.0, safe_depth, edge_order=1)
        dy_dz = np.gradient(self.y_km * 1000.0, safe_depth, edge_order=1)
        return dx_dz, dy_dz

    def curvature_proxy_per_m(self) -> np.ndarray:
        depth = np.asarray(self.depth_m, dtype=float)
        if depth.size < 3:
            return np.zeros_like(depth)
        x_m = _moving_average(self.x_km * 1000.0, window=5)
        y_m = _moving_average(self.y_km * 1000.0, window=5)
        z_m = -depth
        curve = np.stack([x_m, y_m, z_m], axis=1)
        first = np.stack(
            [
                np.gradient(curve[:, 0], depth, edge_order=1),
                np.gradient(curve[:, 1], depth, edge_order=1),
                np.gradient(curve[:, 2], depth, edge_order=1),
            ],
            axis=1,
        )
        second = np.stack(
            [
                np.gradient(first[:, 0], depth, edge_order=1),
                np.gradient(first[:, 1], depth, edge_order=1),
                np.gradient(first[:, 2], depth, edge_order=1),
            ],
            axis=1,
        )
        numerator = np.linalg.norm(np.cross(first, second), axis=1)
        denominator = np.linalg.norm(first, axis=1) ** 3
        with np.errstate(divide="ignore", invalid="ignore"):
            curvature = numerator / denominator
        return np.where(np.isfinite(curvature), curvature, 0.0)


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if window <= 1 or values.size < 3:
        return values
    actual = min(window, values.size if values.size % 2 == 1 else values.size - 1)
    if actual < 3:
        return values
    pad = actual // 2
    padded = np.pad(values, pad, mode="edge")
    kernel = np.ones(actual, dtype=float) / actual
    return np.convolve(padded, kernel, mode="valid")


@dataclass(frozen=True)
class BishopFrame:
    axis: AxisLine
    tangent: np.ndarray
    normal_1: np.ndarray
    normal_2: np.ndarray
    curvature_proxy_per_m: np.ndarray


def build_bishop_frame(axis: AxisLine) -> BishopFrame:
    dx_dz, dy_dz = axis.slopes_m_per_m()
    raw = np.stack([dx_dz, dy_dz, np.ones_like(dx_dz)], axis=1)
    norm = np.linalg.norm(raw, axis=1)
    norm[norm == 0] = 1.0
    tangent = raw / norm[:, None]

    vertical = np.array([0.0, 0.0, 1.0])
    normal_1 = np.cross(tangent, vertical[None, :])
    n1_norm = np.linalg.norm(normal_1, axis=1)
    weak = n1_norm < 1e-8
    normal_1[weak] = np.array([1.0, 0.0, 0.0])
    n1_norm[weak] = 1.0
    normal_1 = normal_1 / n1_norm[:, None]
    normal_2 = np.cross(tangent, normal_1)
    n2_norm = np.linalg.norm(normal_2, axis=1)
    n2_norm[n2_norm == 0] = 1.0
    normal_2 = normal_2 / n2_norm[:, None]

    return BishopFrame(
        axis=axis,
        tangent=tangent,
        normal_1=normal_1,
        normal_2=normal_2,
        curvature_proxy_per_m=axis.curvature_proxy_per_m(),
    )
