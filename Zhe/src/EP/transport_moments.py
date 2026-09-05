from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from .io import read_filter_day
from .numerics import (
    azimuth_second_derivative,
    bilinear_sample,
    ddz,
    radial_derivative,
    xy_to_lonlat,
)


def _load_objects(
    rv_root: Path,
    start: str,
    end: str,
    max_days: int,
    max_objects: int,
    shapes: tuple[str, ...],
) -> pd.DataFrame:
    objects = pd.read_parquet(Path(rv_root) / "object_cache" / "selected_lifecycle_objects.parquet")
    diag_path = Path(rv_root) / "axis" / "object_diagnostics.parquet"
    if diag_path.exists():
        diag = pd.read_parquet(diag_path)
        objects["eddy3d_object_id"] = objects["eddy3d_object_id"].astype("int64")
        diag["eddy3d_object_id"] = diag["eddy3d_object_id"].astype("int64")
        add_cols = [
            col
            for col in (
                "shape_class",
                "polarity",
                "axis_alignment_method",
                "global_alpha_ok",
                "global_deviate_angle_rad",
                "global_theta0_rad",
            )
            if col not in objects.columns and col in diag.columns
        ]
        if add_cols:
            objects = objects.merge(diag[["eddy3d_object_id", *add_cols]], on="eddy3d_object_id", how="left")
    if "global_deviate_angle_rad" not in objects and "global_deviate_angle_deg" in objects:
        objects["global_deviate_angle_rad"] = np.radians(objects["global_deviate_angle_deg"].astype("float64"))
    required = {
        "eddy3d_object_id",
        "track3d_id",
        "date",
        "life_phase",
        "polarity",
        "shape_class",
        "axis_alignment_method",
        "global_alpha_ok",
        "global_deviate_angle_rad",
        "surface_lon",
        "surface_lat",
        "mean_radius_m",
    }
    missing = sorted(required - set(objects.columns))
    if missing:
        raise KeyError(f"selected/axis object tables are missing required columns: {missing}")
    objects["date"] = pd.to_datetime(objects["date"]).dt.strftime("%Y-%m-%d")
    objects = objects[objects["shape_class"].astype(str).isin(shapes)].copy()
    objects = objects[objects["axis_alignment_method"].astype(str).eq("global_ls_alpha")].copy()
    objects = objects[objects["global_alpha_ok"].astype(bool)].copy()
    if start:
        objects = objects[objects["date"] >= pd.Timestamp(start).strftime("%Y-%m-%d")]
    if end:
        objects = objects[objects["date"] <= pd.Timestamp(end).strftime("%Y-%m-%d")]
    if max_days > 0 and not objects.empty:
        days = sorted(objects["date"].unique())[: int(max_days)]
        objects = objects[objects["date"].isin(days)].copy()
    if max_objects > 0 and len(objects) > max_objects:
        objects = objects.sort_values(["date", "polarity", "eddy3d_object_id"]).head(int(max_objects)).copy()
    objects["track3d_id"] = objects["track3d_id"].astype("int64")
    objects["eddy3d_object_id"] = objects["eddy3d_object_id"].astype("int64")
    return objects.sort_values(["date", "polarity", "eddy3d_object_id"]).reset_index(drop=True)


def _load_points(rv_root: Path, object_ids: set[int]) -> pd.DataFrame:
    cols = ["eddy3d_object_id", "track3d_id", "date", "depth_index", "depth_m", "x_rot_m", "y_rot_m"]
    points = pd.read_parquet(Path(rv_root) / "axis" / "rotated_points.parquet", columns=cols)
    points["eddy3d_object_id"] = points["eddy3d_object_id"].astype("int64")
    points = points[points["eddy3d_object_id"].isin(object_ids)].copy()
    points["date"] = pd.to_datetime(points["date"]).dt.strftime("%Y-%m-%d")
    points["depth_index"] = points["depth_index"].astype("int16")
    return points.sort_values(["eddy3d_object_id", "depth_index"]).reset_index(drop=True)


def _read_filter_day(
    filter_root: Path,
    template: str,
    day,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    lon, lat, depth, fields = read_filter_day(filter_root, template, day, ("uo_glor", "vo_glor", "thetao_glor"))
    return lon, lat, depth, fields["uo_glor"], fields["vo_glor"], fields["thetao_glor"]


def _center_lines(points: pd.DataFrame) -> dict[int, pd.DataFrame]:
    return {
        int(object_id): part.sort_values("depth_index").copy()
        for object_id, part in points.groupby("eddy3d_object_id", sort=False)
    }


def _sample_rotated_fields(
    obj,
    center_line: pd.DataFrame,
    lon: np.ndarray,
    lat: np.ndarray,
    psi: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    theta_field: np.ndarray,
    radial_mesh: np.ndarray,
    theta_mesh: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    if center_line is None or len(center_line) < 3:
        return None
    radius_m = float(obj.mean_radius_m)
    alpha = float(obj.global_deviate_angle_rad)
    cos_a = math.cos(alpha)
    sin_a = math.sin(alpha)
    local_x = radial_mesh * radius_m * np.cos(theta_mesh)
    local_y = radial_mesh * radius_m * np.sin(theta_mesh)
    depth_indices: list[int] = []
    depth_values: list[float] = []
    psi_layers: list[np.ndarray] = []
    vrot_layers: list[np.ndarray] = []
    theta_layers: list[np.ndarray] = []
    for row in center_line.itertuples(index=False):
        k = int(row.depth_index)
        if k < 0 or k >= u.shape[0]:
            continue
        x_rot = float(row.x_rot_m) + local_x
        y_rot = float(row.y_rot_m) + local_y
        x_orig = x_rot * cos_a + y_rot * sin_a
        y_orig = -x_rot * sin_a + y_rot * cos_a
        target_lon, target_lat = xy_to_lonlat(x_orig, y_orig, float(obj.surface_lon), float(obj.surface_lat))
        u_s = bilinear_sample(lon, lat, u[k], target_lon, target_lat)
        v_s = bilinear_sample(lon, lat, v[k], target_lon, target_lat)
        psi_layers.append(bilinear_sample(lon, lat, psi[k], target_lon, target_lat))
        theta_layers.append(bilinear_sample(lon, lat, theta_field[k], target_lon, target_lat))
        vrot_layers.append(u_s * sin_a + v_s * cos_a)
        depth_indices.append(k)
        depth_values.append(float(row.depth_m))
    if len(depth_indices) < 3:
        return None
    return (
        np.asarray(depth_indices, dtype="int64"),
        np.asarray(depth_values, dtype="float64"),
        np.asarray(psi_layers, dtype="float64"),
        np.asarray(vrot_layers, dtype="float64"),
        np.asarray(theta_layers, dtype="float64"),
    )


def _q_prime_from_psi(
    psi_sample: np.ndarray,
    depth: np.ndarray,
    radial: np.ndarray,
    theta: np.ndarray,
    radius_m: float,
    n2: np.ndarray,
    f0: float,
) -> np.ndarray:
    psi = np.where(np.abs(psi_sample) > 1.0e20, np.nan, psi_sample)
    psi_prime = psi - np.nanmean(psi, axis=2, keepdims=True)
    r_m = np.maximum(radial * float(radius_m), 1.0)
    dpsi_dr = radial_derivative(psi_prime, r_m)
    radial_lap = radial_derivative(dpsi_dr * r_m[None, :, None], r_m) / r_m[None, :, None]
    az_lap = azimuth_second_derivative(psi_prime, theta) / (r_m[None, :, None] ** 2)
    dpsi_dz = ddz(psi_prime, depth)
    strat = (f0 * f0 / n2)[:, None, None] * dpsi_dz
    q_total = radial_lap + az_lap + ddz(strat, depth)
    return q_total - np.nanmean(q_total, axis=2, keepdims=True)


def _tau_weights(life_phase: float, tau_grid: np.ndarray, bandwidth: float, min_weight: float) -> list[tuple[int, float]]:
    if not np.isfinite(life_phase):
        return []
    if bandwidth <= 0:
        index = int(np.argmin(np.abs(tau_grid - life_phase)))
        return [(index, 1.0)]
    raw = np.exp(-0.5 * ((tau_grid - float(life_phase)) / float(bandwidth)) ** 2)
    raw[raw < float(min_weight)] = 0.0
    total = float(raw.sum())
    if total <= 0:
        index = int(np.argmin(np.abs(tau_grid - life_phase)))
        return [(index, 1.0)]
    return [(int(i), float(w / total)) for i, w in enumerate(raw) if w > 0]


def _resolve_n2_profile(rv_root: Path, explicit: str) -> Path:
    if explicit:
        path = Path(explicit)
        if path.exists():
            return path
        raise FileNotFoundError(path)
    filename = "cmems_doy_climatology_1993_2022_31d_sigma0_dz_profile.npz"
    root = Path(rv_root)
    candidates = [
        root / "climatology" / filename,
        root.parent / "climatology" / filename,
        root.parent.parent / "climatology" / filename,
        root.parent.parent.parent / "climatology" / filename,
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("N2 profile not found in: " + ", ".join(str(path) for path in candidates))
