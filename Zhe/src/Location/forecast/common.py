from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


G = 9.81
OMEGA = 7.2921159e-5
EARTH_RADIUS_M = 6_371_000.0


@dataclass(frozen=True)
class GeoParams:
    f0: float
    beta: float


def finite_or_nan(array: np.ndarray) -> np.ndarray:
    arr = np.asarray(array, dtype="f8")
    return np.where(np.isfinite(arr), arr, np.nan)


def gradient(array: np.ndarray, coord: np.ndarray, axis: int) -> np.ndarray:
    arr = np.asarray(array, dtype="f8")
    c = np.asarray(coord, dtype="f8")
    if c.size < 2:
        return np.zeros_like(arr, dtype="f8")
    return np.gradient(arr, c, axis=axis, edge_order=1)


def geo_params(latitude_ref: float = 30.0) -> GeoParams:
    lat = np.deg2rad(latitude_ref)
    return GeoParams(
        f0=float(2.0 * OMEGA * np.sin(lat)),
        beta=float(2.0 * OMEGA * np.cos(lat) / EARTH_RADIUS_M),
    )


def decode(values) -> list[str]:
    out = []
    for value in values:
        if isinstance(value, bytes):
            out.append(value.decode("utf-8"))
        else:
            out.append(str(value))
    return out


def composite_path(config: dict, shape: str) -> Path:
    root = Path(config["paths"]["output_dir"])
    start = str(config.get("composite", {}).get("start", "1993-01-01"))[:4]
    end = str(config.get("composite", {}).get("end", "2022-12-31"))[:4]
    return root / f"lifecycle_composites_{start}_{end}_{shape}" / "lifecycle_composite.nc"


def radius_lookup(config: dict, shape: str, polarity_names: list[str], phase_names: list[str]) -> dict[tuple[str, str], float]:
    root = Path(config["paths"]["output_dir"])
    start = str(config.get("composite", {}).get("start", "1993-01-01"))[:4]
    end = str(config.get("composite", {}).get("end", "2022-12-31"))[:4]
    index_path = root / f"lifecycle_composites_{start}_{end}_{shape}" / "lifecycle_composite_index.parquet"
    fallback = {(pol, phase): 50_000.0 for pol in polarity_names for phase in phase_names}
    if not index_path.exists():
        return fallback
    index = pd.read_parquet(index_path)
    if "phase_name" not in index.columns or "polarity" not in index.columns or "radius_m" not in index.columns:
        return fallback
    grouped = index.groupby(["polarity", "phase_name"])["radius_m"].median()
    for key, value in grouped.items():
        if np.isfinite(value):
            fallback[(str(key[0]), str(key[1]))] = float(value)
    return fallback


def thermal_wind_velocity(sigma: np.ndarray, adt: np.ndarray, depth: np.ndarray, x: np.ndarray, y: np.ndarray, radius_m: float, f0: float) -> tuple[np.ndarray, np.ndarray]:
    eta_x = gradient(adt, x, axis=1)
    eta_y = gradient(adt, y, axis=0)
    ug0 = -G / max(abs(f0) * radius_m, 1e-12) * eta_y
    vg0 = G / max(abs(f0) * radius_m, 1e-12) * eta_x
    sig_x = gradient(sigma, x, axis=2)
    sig_y = gradient(sigma, y, axis=1)
    du_dz = -G / max(1025.0 * abs(f0) * radius_m, 1e-12) * sig_y
    dv_dz = G / max(1025.0 * abs(f0) * radius_m, 1e-12) * sig_x
    u = np.zeros_like(sigma, dtype="f8")
    v = np.zeros_like(sigma, dtype="f8")
    u[0] = ug0
    v[0] = vg0
    for k in range(1, len(depth)):
        dz = float(depth[k] - depth[k - 1])
        u[k] = u[k - 1] + 0.5 * dz * (du_dz[k] + du_dz[k - 1])
        v[k] = v[k - 1] + 0.5 * dz * (dv_dz[k] + dv_dz[k - 1])
    return u, v


def load_sigma0_profile(config: dict, depth: np.ndarray) -> np.ndarray:
    root = Path(config["paths"]["output_dir"]) / "climatology"
    candidates = sorted(root.glob("*sigma0_dz_profile.npz"))
    if candidates:
        path = candidates[-1]
    else:
        path = root / "cmems_doy_climatology_1993_2022_31d_sigma0_dz_profile.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing sigma0 profile: {path}")
    data = np.load(path)
    src_depth = np.asarray(data["depth"], dtype="f8")
    if "sigma0_profile" in data:
        sigma = np.asarray(data["sigma0_profile"], dtype="f8")
    elif "sigma0" in data:
        sigma = np.asarray(data["sigma0"], dtype="f8")
    else:
        dsigma = np.asarray(data["dsigma0_dz"], dtype="f8")
        sigma = np.cumsum(np.gradient(src_depth) * dsigma)
    good = np.isfinite(src_depth) & np.isfinite(sigma)
    if np.count_nonzero(good) < 2:
        raise ValueError(f"Invalid sigma0 profile: {path}")
    return np.interp(np.asarray(depth, dtype="f8"), src_depth[good], sigma[good])


def write_summary(output_dir: Path, title: str, lines: list[str]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    text = [f"# {title}", ""]
    text.extend(f"- {line}" for line in lines)
    (output_dir / "validation_summary.md").write_text("\n".join(text) + "\n", encoding="utf-8")
