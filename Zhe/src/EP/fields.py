from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


def _decode_labels(values: np.ndarray) -> list[str]:
    labels: list[str] = []
    for value in values:
        if isinstance(value, bytes):
            labels.append(value.decode("utf-8"))
        else:
            labels.append(str(value))
    return labels


def _array_by_key(data: np.lib.npyio.NpzFile, names: tuple[str, ...]) -> np.ndarray:
    for name in names:
        if name in data:
            return np.asarray(data[name])
    raise KeyError(f"Missing any of arrays: {names}")


@dataclass(frozen=True)
class RepresentativeSlice:
    polarity: str
    tau: float
    depth_m: np.ndarray
    radius_coord: np.ndarray
    theta_rad: np.ndarray
    radius_m: float
    u: np.ndarray
    v: np.ndarray
    speed: np.ndarray
    count: np.ndarray | None = None
    theta_prime: np.ndarray | None = None

    @property
    def radial_m(self) -> np.ndarray:
        coord = np.asarray(self.radius_coord, dtype=float)
        if np.nanmax(coord) <= 5.0:
            return coord * self.radius_m
        return coord * 1000.0

    @property
    def mesh_xy_km(self) -> tuple[np.ndarray, np.ndarray]:
        r_km = self.radial_m / 1000.0
        rr, tt = np.meshgrid(r_km, self.theta_rad, indexing="ij")
        return rr * np.cos(tt), rr * np.sin(tt)

    def polar_velocity(self) -> tuple[np.ndarray, np.ndarray]:
        tt = self.theta_rad[None, None, :]
        ur = self.u * np.cos(tt) + self.v * np.sin(tt)
        ut = -self.u * np.sin(tt) + self.v * np.cos(tt)
        return ur, ut


@dataclass(frozen=True)
class RepresentativeVortexDataset:
    npz_path: Path
    polarities: list[str]
    tau_grid: np.ndarray
    depth_m: np.ndarray
    radius_coord: np.ndarray
    theta_rad: np.ndarray
    u_mean: np.ndarray
    v_mean: np.ndarray
    speed_mean: np.ndarray
    count: np.ndarray | None
    n_objects: np.ndarray | None
    n_tracks: np.ndarray | None
    radius_by_polarity_m: dict[str, float]

    @classmethod
    def load(cls, npz_path: Path, radial_seed_root: Path | None = None) -> "RepresentativeVortexDataset":
        if not npz_path.exists():
            raise FileNotFoundError(npz_path)
        data = np.load(npz_path, allow_pickle=True)
        u = _array_by_key(data, ("u_mean", "u", "u_composite"))
        v = _array_by_key(data, ("v_mean", "v", "v_composite"))
        speed = data["speed_mean"] if "speed_mean" in data else np.hypot(u, v)
        count = data["count"] if "count" in data else None
        n_objects = data["n_objects"] if "n_objects" in data else None
        n_tracks = data["n_tracks"] if "n_tracks" in data else None
        tau_grid = _array_by_key(data, ("tau_grid", "tau"))
        depth = _array_by_key(data, ("depth", "depth_m", "z_m")).astype(float)
        radial = _array_by_key(data, ("radial", "radius", "r")).astype(float)
        theta = _array_by_key(data, ("theta", "theta_rad", "azimuth")).astype(float)
        if np.nanmax(np.abs(theta)) > 2.0 * np.pi + 1e-6:
            theta = np.deg2rad(theta)

        if "polarities" in data:
            polarities = _decode_labels(np.asarray(data["polarities"]))
        else:
            polarities = ["cyclonic", "anticyclonic"][: u.shape[0]]

        radius_by_polarity = _load_radii(radial_seed_root, polarities)
        return cls(
            npz_path=npz_path,
            polarities=polarities,
            tau_grid=tau_grid.astype(float),
            depth_m=depth,
            radius_coord=radial,
            theta_rad=theta,
            u_mean=u,
            v_mean=v,
            speed_mean=speed,
            count=count,
            n_objects=n_objects,
            n_tracks=n_tracks,
            radius_by_polarity_m=radius_by_polarity,
        )

    def nearest_tau_index(self, tau: float) -> int:
        return int(np.nanargmin(np.abs(self.tau_grid - tau)))

    def polarity_index(self, polarity: str) -> int:
        if polarity in self.polarities:
            return self.polarities.index(polarity)
        lowered = [p.lower() for p in self.polarities]
        return lowered.index(polarity.lower())

    def slice(self, polarity: str, tau: float) -> RepresentativeSlice:
        p_idx = self.polarity_index(polarity)
        t_idx = self.nearest_tau_index(tau)
        radius_m = self.radius_by_polarity_m.get(polarity, np.nan)
        if not np.isfinite(radius_m) or radius_m <= 0:
            radius_m = 100000.0
        count = None if self.count is None else self.count[p_idx, t_idx]
        return RepresentativeSlice(
            polarity=polarity,
            tau=float(self.tau_grid[t_idx]),
            depth_m=self.depth_m,
            radius_coord=self.radius_coord,
            theta_rad=self.theta_rad,
            radius_m=float(radius_m),
            u=self.u_mean[p_idx, t_idx],
            v=self.v_mean[p_idx, t_idx],
            speed=self.speed_mean[p_idx, t_idx],
            count=count,
        )


def _load_radii(radial_seed_root: Path | None, polarities: list[str]) -> dict[str, float]:
    fallback = {polarity: 100000.0 for polarity in polarities}
    if radial_seed_root is None:
        return fallback
    path = radial_seed_root / "representative_radii.csv"
    if not path.exists():
        return fallback
    table = pd.read_csv(path)
    if "polarity" not in table.columns:
        return fallback
    radius_col = None
    for candidate in ("radius_m", "representative_radius_m", "median_radius_m", "R_m"):
        if candidate in table.columns:
            radius_col = candidate
            break
    if radius_col is None:
        return fallback
    out = fallback.copy()
    for _, row in table.iterrows():
        polarity = str(row["polarity"])
        value = float(row[radius_col])
        if np.isfinite(value) and value > 0:
            out[polarity] = value
    return out
