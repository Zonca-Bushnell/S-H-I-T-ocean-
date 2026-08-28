from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .axis_streamfunction import DEFAULT_AXIS_DIR, DEFAULT_CATALOG


DEFAULT_SHAPE_BY_SHAPE_DIR = Path(r"E:\Verify\outputs\kuroshio_cmems_3d\shape_classification_1993_2022\by_shape")
DEFAULT_LIFECYCLE_ROOT = Path(r"E:\Verify\outputs\kuroshio_cmems_3d\LIFE_CYCLE_REPRESENTATIVE_VOLOCITY")
DEFAULT_NEW_VORTICITY_ROOT = Path(r"E:\Verify\outputs\kuroshio_cmems_3d\NEW_vorticity")
PHASE_NAMES = ("birth", "growth", "mature", "decay", "death")
DEFAULT_SHAPES = ("coherent", "complex", "mixed", "transitional", "upright_like")
DEFAULT_POLARITIES = ("cyclonic", "anticyclonic")


def phase_index(life_phase: float, phase_count: int = 5) -> int:
    if not np.isfinite(life_phase):
        return -1
    return int(np.clip(np.floor(float(life_phase) * phase_count), 0, phase_count - 1))


def load_track_lifecycle(shape_dir: Path, shapes: tuple[str, ...]) -> pd.DataFrame:
    rows = []
    for shape in shapes:
        path = shape_dir / shape / "tracks.csv"
        if not path.exists():
            continue
        tracks = pd.read_csv(path)
        if tracks.empty:
            continue
        need = ["track3d_id", "shape_class", "polarity", "start_date", "end_date", "lifetime_days"]
        missing = [col for col in need if col not in tracks.columns]
        if missing:
            raise ValueError(f"{path} is missing columns: {missing}")
        rows.append(tracks[need].copy())
    if not rows:
        return pd.DataFrame(columns=["track3d_id", "shape_class", "polarity", "start_date", "end_date", "lifetime_days"])
    out = pd.concat(rows, ignore_index=True)
    out = out.drop_duplicates(subset=["track3d_id"]).copy()
    out["track3d_id"] = out["track3d_id"].astype("int64")
    out["start_date"] = pd.to_datetime(out["start_date"])
    out["end_date"] = pd.to_datetime(out["end_date"])
    out["lifetime_days"] = out["lifetime_days"].astype("float64")
    return out


def load_lifecycle_objects(
    axis_dir: Path = DEFAULT_AXIS_DIR,
    catalog_dir: Path = DEFAULT_CATALOG,
    shape_dir: Path = DEFAULT_SHAPE_BY_SHAPE_DIR,
    shapes: tuple[str, ...] = DEFAULT_SHAPES,
    polarities: tuple[str, ...] = DEFAULT_POLARITIES,
) -> pd.DataFrame:
    objects = pd.read_parquet(axis_dir / "object_diagnostics.parquet")
    objects = objects[objects["is_usable"] & objects["shape_class"].isin(shapes) & objects["polarity"].isin(polarities)].copy()
    tracks = load_track_lifecycle(shape_dir, shapes)
    if tracks.empty:
        return objects.iloc[0:0].copy()
    objects["track3d_id"] = objects["track3d_id"].astype("int64")
    data = objects.merge(
        tracks[["track3d_id", "start_date", "end_date", "lifetime_days"]],
        on="track3d_id",
        how="inner",
    )
    radii = pd.read_parquet(catalog_dir / "vertical_objects.parquet", columns=["eddy3d_object_id", "mean_radius_m"])
    data = data.merge(radii, on="eddy3d_object_id", how="left")
    data = data[np.isfinite(data["mean_radius_m"]) & (data["mean_radius_m"] > 0)].copy()
    data["date_ts"] = pd.to_datetime(data["date"])
    data["life_day"] = (data["date_ts"] - data["start_date"]).dt.days.astype("float64")
    denom = np.maximum(data["lifetime_days"].to_numpy(dtype="float64") - 1.0, 1.0)
    data["life_phase"] = np.clip(data["life_day"].to_numpy(dtype="float64") / denom, 0.0, 1.0)
    data["phase_index"] = data["life_phase"].map(phase_index).astype("int64")
    data = data[(data["phase_index"] >= 0) & (data["phase_index"] < len(PHASE_NAMES))].copy()
    data["phase_name"] = data["phase_index"].map(lambda value: PHASE_NAMES[int(value)])
    data["date"] = data["date_ts"].dt.strftime("%Y-%m-%d")
    return data


def apply_lifecycle_limits(objects: pd.DataFrame, max_days: int, max_objects_per_polarity: int, seed: int) -> pd.DataFrame:
    data = objects.copy()
    if max_days > 0 and not data.empty:
        keep = pd.Series(False, index=data.index)
        for (_, _), part in data.groupby(["polarity", "phase_name"], sort=False):
            dates = sorted(part["date"].unique())[:max_days]
            keep |= data.index.isin(part[part["date"].isin(dates)].index)
        data = data[keep].copy()
    if max_objects_per_polarity > 0 and not data.empty:
        rng = np.random.default_rng(seed)
        keep_ids: list[int] = []
        for (_, _), part in data.groupby(["polarity", "phase_name"], sort=False):
            ids = part["eddy3d_object_id"].to_numpy(dtype="int64")
            if ids.size > max_objects_per_polarity:
                ids = rng.choice(ids, size=max_objects_per_polarity, replace=False)
            keep_ids.extend(int(value) for value in ids)
        data = data[data["eddy3d_object_id"].isin(keep_ids)].copy()
    return data.sort_values(["date", "polarity", "phase_index", "eddy3d_object_id"]).copy()


def load_center_lines(axis_dir: Path, object_ids: set[int]) -> dict[int, pd.DataFrame]:
    if not object_ids:
        return {}
    points = pd.read_parquet(
        axis_dir / "rotated_points.parquet",
        columns=["eddy3d_object_id", "depth_index", "z_m", "x_rot_m", "y_rot_m"],
    )
    points = points[points["eddy3d_object_id"].isin(object_ids)].copy()
    return {int(object_id): part.sort_values("depth_index").copy() for object_id, part in points.groupby("eddy3d_object_id", sort=False)}


def representative_radii(objects: pd.DataFrame) -> dict[str, float]:
    data = objects[np.isfinite(objects["mean_radius_m"]) & (objects["mean_radius_m"] > 0)]
    return {str(polarity): float(part["mean_radius_m"].median()) for polarity, part in data.groupby("polarity")}


def write_lifecycle_root_summary(root: Path, lines: list[str]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

