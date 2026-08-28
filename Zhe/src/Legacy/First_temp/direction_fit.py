from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    from scipy.optimize import minimize_scalar
except Exception:  # pragma: no cover - scipy is available in production env
    minimize_scalar = None


EARTH_RADIUS_M = 6_371_000.0
DEFAULT_BY_SHAPE_DIR = Path(r"E:\Verify\outputs\kuroshio_cmems_3d\shape_classification_1993_2022\by_shape")
DEFAULT_CENTERS = Path(r"E:\Verify\outputs\kuroshio_cmems_3d\catalog\layer_centers_completed.parquet")
DEFAULT_OUTPUT = Path(r"E:\Verify\outputs\kuroshio_cmems_3d\first_temp_direction_fit_1993_2022_by_polarity")
DEFAULT_SHAPE_ORDER = ("coherent", "complex", "mixed", "transitional", "upright_like", "unknown")
AXIS_ALIGNMENT_SURFACE_TO_DEEP = "surface_to_deep"
AXIS_ALIGNMENT_GLOBAL_LS_ALPHA = "global_ls_alpha"


@dataclass(frozen=True)
class ShapeTracks:
    track_to_shape: dict[int, str]
    track_to_polarity: dict[int, str]
    counts: dict[tuple[str, str], int]
    requested_shapes: tuple[str, ...]


def parse_csv_list(value: str | None) -> tuple[str, ...]:
    if value is None or not value.strip():
        return DEFAULT_SHAPE_ORDER
    return tuple(part.strip() for part in value.split(",") if part.strip())


def local_xy_m(lon: np.ndarray, lat: np.ndarray, lon0: np.ndarray, lat0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    dlon = (lon - lon0 + 180.0) % 360.0 - 180.0
    x = np.radians(dlon) * EARTH_RADIUS_M * np.cos(np.radians(lat0))
    y = np.radians(lat - lat0) * EARTH_RADIUS_M
    return x, y


def wrap180_deg(value: np.ndarray | float) -> np.ndarray | float:
    return (np.asarray(value) + 180.0) % 360.0 - 180.0


def _weighted_circular_direction_deg(theta_deg: np.ndarray, weight: np.ndarray) -> float:
    sx = np.sum(weight * np.cos(np.radians(theta_deg)))
    sy = np.sum(weight * np.sin(np.radians(theta_deg)))
    if not np.isfinite(sx) or not np.isfinite(sy) or (sx == 0.0 and sy == 0.0):
        return np.nan
    return float(wrap180_deg(np.degrees(np.arctan2(sy, sx))))


def compute_global_alpha_ls(
    x_m: np.ndarray,
    y_m: np.ndarray,
    depth_m: np.ndarray,
    *,
    min_layers: int,
) -> dict[str, float | int | bool]:
    """Matlab-compatible per-object alpha: min sum w * wrap(theta + alpha)^2."""
    x = np.asarray(x_m, dtype="f8")
    y = np.asarray(y_m, dtype="f8")
    z = np.asarray(depth_m, dtype="f8")
    weight = np.hypot(x, y)
    theta = np.degrees(np.arctan2(y, x))
    valid = np.isfinite(z) & np.isfinite(theta) & np.isfinite(weight) & (weight > 0.0)
    n_valid = int(np.count_nonzero(valid))
    if n_valid < int(min_layers):
        return {
            "ok": False,
            "alpha_deg": np.nan,
            "theta0_deg": np.nan,
            "n_layers": n_valid,
            "weight_sum": float(np.nansum(weight[valid])) if n_valid else 0.0,
            "rmse_deg": np.nan,
        }

    theta = theta[valid]
    weight = weight[valid]
    weight_sum = float(np.sum(weight))
    theta0 = _weighted_circular_direction_deg(theta, weight)
    alpha0 = float(wrap180_deg(-theta0)) if np.isfinite(theta0) else 0.0

    def objective(alpha: float) -> float:
        residual = wrap180_deg(theta + alpha)
        return float(np.sum(weight * residual * residual))

    alpha = alpha0
    if minimize_scalar is not None:
        try:
            opt = minimize_scalar(objective, bounds=(-180.0, 180.0), method="bounded")
            candidates = [alpha0, alpha0 - 180.0, alpha0 + 180.0]
            if opt.success and np.isfinite(opt.x):
                candidates.append(float(opt.x))
            values = np.asarray([objective(value) for value in candidates], dtype="f8")
            alpha = float(wrap180_deg(candidates[int(np.nanargmin(values))]))
        except Exception:
            alpha = alpha0

    residual = wrap180_deg(theta + alpha)
    rmse = float(np.sqrt(np.sum(weight * residual * residual) / max(weight_sum, 1e-12)))
    return {
        "ok": bool(np.isfinite(alpha)),
        "alpha_deg": float(alpha) if np.isfinite(alpha) else np.nan,
        "theta0_deg": float(wrap180_deg(-alpha)) if np.isfinite(alpha) else np.nan,
        "n_layers": n_valid,
        "weight_sum": weight_sum,
        "rmse_deg": rmse if np.isfinite(rmse) else np.nan,
    }


def group_label(shape: str, polarity: str) -> str:
    return f"{shape}_{polarity}".replace(" ", "_")


def load_shape_tracks(by_shape_dir: Path, shapes: Iterable[str]) -> ShapeTracks:
    track_to_shape: dict[int, str] = {}
    track_to_polarity: dict[int, str] = {}
    counts: dict[tuple[str, str], int] = {}
    duplicates: list[int] = []
    requested = tuple(shapes)
    for shape in requested:
        path = by_shape_dir / shape / "tracks.csv"
        if not path.exists():
            continue
        tracks = pd.read_csv(path, usecols=["track3d_id", "polarity"])
        tracks = tracks.dropna(subset=["track3d_id", "polarity"]).copy()
        tracks["track3d_id"] = tracks["track3d_id"].astype("int64")
        tracks["polarity"] = tracks["polarity"].astype(str)
        for polarity, part in tracks.groupby("polarity"):
            ids = sorted(set(int(v) for v in part["track3d_id"].to_numpy()))
            counts[(shape, polarity)] = len(ids)
            for track_id in ids:
                if track_id in track_to_shape:
                    duplicates.append(track_id)
                    continue
                track_to_shape[track_id] = shape
                track_to_polarity[track_id] = polarity
    if duplicates:
        sample = ", ".join(str(v) for v in duplicates[:10])
        raise ValueError(f"Track ids appear in multiple shape folders, sample: {sample}")
    return ShapeTracks(
        track_to_shape=track_to_shape,
        track_to_polarity=track_to_polarity,
        counts=counts,
        requested_shapes=requested,
    )


def read_selected_centers(centers_path: Path, shape_tracks: ShapeTracks) -> pd.DataFrame:
    if not shape_tracks.track_to_shape:
        return pd.DataFrame()
    columns = [
        "date",
        "track3d_id",
        "eddy3d_object_id",
        "depth_index",
        "depth_m",
        "longitude",
        "latitude",
    ]
    centers = pd.read_parquet(centers_path, columns=columns)
    centers = centers[centers["track3d_id"].astype("int64").isin(shape_tracks.track_to_shape)].copy()
    centers["track3d_id"] = centers["track3d_id"].astype("int64")
    centers["eddy3d_object_id"] = centers["eddy3d_object_id"].astype("int64")
    centers["depth_index"] = centers["depth_index"].astype("int16")
    centers["shape_class"] = centers["track3d_id"].map(shape_tracks.track_to_shape)
    centers["polarity"] = centers["track3d_id"].map(shape_tracks.track_to_polarity)
    centers = centers.dropna(subset=["shape_class", "polarity", "longitude", "latitude", "depth_m"])
    return centers


def limit_objects(centers: pd.DataFrame, max_objects_per_group: int, seed: int) -> pd.DataFrame:
    if max_objects_per_group <= 0 or centers.empty:
        return centers
    rng = np.random.default_rng(seed)
    keep: list[int] = []
    unique = centers[["shape_class", "polarity", "eddy3d_object_id"]].drop_duplicates()
    for _, part in unique.groupby(["shape_class", "polarity"]):
        ids = part["eddy3d_object_id"].to_numpy(dtype="int64")
        if ids.size > max_objects_per_group:
            ids = rng.choice(ids, size=max_objects_per_group, replace=False)
        keep.extend(int(v) for v in ids)
    return centers[centers["eddy3d_object_id"].isin(keep)].copy()


def build_rotated_points(
    centers: pd.DataFrame,
    *,
    min_layers: int,
    min_depth_span_m: float,
    axis_alignment: str = AXIS_ALIGNMENT_SURFACE_TO_DEEP,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if centers.empty:
        return centers.copy(), pd.DataFrame()
    axis_alignment = str(axis_alignment or AXIS_ALIGNMENT_SURFACE_TO_DEEP)
    if axis_alignment not in {AXIS_ALIGNMENT_SURFACE_TO_DEEP, AXIS_ALIGNMENT_GLOBAL_LS_ALPHA}:
        raise ValueError(
            f"Unsupported axis_alignment={axis_alignment!r}; "
            f"use {AXIS_ALIGNMENT_SURFACE_TO_DEEP!r} or {AXIS_ALIGNMENT_GLOBAL_LS_ALPHA!r}."
        )

    centers = centers.sort_values(["eddy3d_object_id", "depth_index"]).copy()
    grouped = centers.groupby("eddy3d_object_id", sort=False)
    surface = grouped.first()
    deep = grouped.last()
    counts = grouped.size().rename("n_layers")
    object_info = surface[
        ["date", "track3d_id", "shape_class", "polarity", "depth_index", "depth_m", "longitude", "latitude"]
    ].rename(
        columns={
            "depth_index": "surface_depth_index",
            "depth_m": "surface_depth_m",
            "longitude": "surface_lon",
            "latitude": "surface_lat",
        }
    )
    object_info["deep_depth_index"] = deep["depth_index"].astype(int)
    object_info["deep_depth_m"] = deep["depth_m"].astype(float)
    object_info["deep_lon"] = deep["longitude"].astype(float)
    object_info["deep_lat"] = deep["latitude"].astype(float)
    object_info["n_layers"] = counts.astype(int)
    deep_x, deep_y = local_xy_m(
        object_info["deep_lon"].to_numpy(dtype="f8"),
        object_info["deep_lat"].to_numpy(dtype="f8"),
        object_info["surface_lon"].to_numpy(dtype="f8"),
        object_info["surface_lat"].to_numpy(dtype="f8"),
    )
    object_info["deep_x_m"] = deep_x
    object_info["deep_y_m"] = deep_y
    object_info["depth_span_m"] = object_info["deep_depth_m"] - object_info["surface_depth_m"]
    object_info["deep_distance_m"] = np.hypot(deep_x, deep_y)
    object_info["temp_direction_rad"] = np.arctan2(deep_y, deep_x)
    object_info["temp_direction_deg"] = np.degrees(object_info["temp_direction_rad"])
    object_info["axis_alignment_method"] = axis_alignment
    object_info["global_deviate_angle_deg"] = np.nan
    object_info["global_deviate_angle_rad"] = np.nan
    object_info["global_theta0_deg"] = np.nan
    object_info["global_theta0_rad"] = np.nan
    object_info["global_alpha_ok"] = False
    object_info["global_alpha_n_layers"] = 0
    object_info["global_alpha_weight_sum"] = 0.0
    object_info["global_alpha_rmse_deg"] = np.nan

    if axis_alignment == AXIS_ALIGNMENT_GLOBAL_LS_ALPHA:
        alpha_rows: list[dict[str, float | int | bool]] = []
        for object_id, part in grouped:
            base = object_info.loc[object_id]
            x_line, y_line = local_xy_m(
                part["longitude"].to_numpy(dtype="f8"),
                part["latitude"].to_numpy(dtype="f8"),
                np.full(part.shape[0], float(base["surface_lon"]), dtype="f8"),
                np.full(part.shape[0], float(base["surface_lat"]), dtype="f8"),
            )
            info = compute_global_alpha_ls(
                x_line,
                y_line,
                part["depth_m"].to_numpy(dtype="f8"),
                min_layers=int(min_layers),
            )
            info["eddy3d_object_id"] = int(object_id)
            alpha_rows.append(info)
        alpha = pd.DataFrame.from_records(alpha_rows).set_index("eddy3d_object_id")
        object_info["global_alpha_ok"] = alpha["ok"].astype(bool)
        object_info["global_deviate_angle_deg"] = alpha["alpha_deg"].astype(float)
        object_info["global_deviate_angle_rad"] = np.radians(object_info["global_deviate_angle_deg"].astype(float))
        object_info["global_theta0_deg"] = alpha["theta0_deg"].astype(float)
        object_info["global_theta0_rad"] = np.radians(object_info["global_theta0_deg"].astype(float))
        object_info["global_alpha_n_layers"] = alpha["n_layers"].astype(int)
        object_info["global_alpha_weight_sum"] = alpha["weight_sum"].astype(float)
        object_info["global_alpha_rmse_deg"] = alpha["rmse_deg"].astype(float)
        object_info["temp_direction_rad"] = object_info["global_theta0_rad"]
        object_info["temp_direction_deg"] = object_info["global_theta0_deg"]
    else:
        object_info["global_deviate_angle_deg"] = wrap180_deg(-object_info["temp_direction_deg"].to_numpy(dtype="f8"))
        object_info["global_deviate_angle_rad"] = np.radians(object_info["global_deviate_angle_deg"].astype(float))
        object_info["global_theta0_deg"] = object_info["temp_direction_deg"].astype(float)
        object_info["global_theta0_rad"] = object_info["temp_direction_rad"].astype(float)
        object_info["global_alpha_ok"] = np.isfinite(object_info["temp_direction_rad"])
        object_info["global_alpha_n_layers"] = object_info["n_layers"].astype(int)
        object_info["global_alpha_weight_sum"] = object_info["deep_distance_m"].astype(float)

    object_info["is_usable"] = (
        (object_info["n_layers"] >= int(min_layers))
        & np.isfinite(object_info["temp_direction_rad"])
        & (object_info["depth_span_m"] >= float(min_depth_span_m))
        & (object_info["deep_distance_m"] > 0.0)
    )
    if axis_alignment == AXIS_ALIGNMENT_GLOBAL_LS_ALPHA:
        object_info["is_usable"] = object_info["is_usable"] & object_info["global_alpha_ok"].astype(bool)

    use_info = object_info[object_info["is_usable"]].copy()
    points = centers.merge(
        use_info[
            [
                "surface_depth_m",
                "surface_lon",
                "surface_lat",
                "temp_direction_rad",
                "temp_direction_deg",
                "axis_alignment_method",
                "global_deviate_angle_deg",
                "global_deviate_angle_rad",
                "global_theta0_deg",
                "global_theta0_rad",
                "global_alpha_ok",
                "global_alpha_n_layers",
                "global_alpha_weight_sum",
                "global_alpha_rmse_deg",
            ]
        ],
        left_on="eddy3d_object_id",
        right_index=True,
        how="inner",
    )
    x, y = local_xy_m(
        points["longitude"].to_numpy(dtype="f8"),
        points["latitude"].to_numpy(dtype="f8"),
        points["surface_lon"].to_numpy(dtype="f8"),
        points["surface_lat"].to_numpy(dtype="f8"),
    )
    theta = points["temp_direction_rad"].to_numpy(dtype="f8")
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    points["x_m"] = x
    points["y_m"] = y
    points["z_m"] = points["depth_m"].astype(float) - points["surface_depth_m"].astype(float)
    points["x_rot_m"] = x * cos_t + y * sin_t
    points["y_rot_m"] = -x * sin_t + y * cos_t

    deep_rot = points.sort_values(["eddy3d_object_id", "depth_index"]).groupby("eddy3d_object_id").last()
    object_info["deep_x_rot_m"] = deep_rot["x_rot_m"]
    object_info["deep_y_rot_m"] = deep_rot["y_rot_m"]
    object_info["deep_rotation_abs_y_m"] = object_info["deep_y_rot_m"].abs()
    keep_columns = [
        "shape_class",
        "polarity",
        "track3d_id",
        "eddy3d_object_id",
        "date",
        "depth_index",
        "depth_m",
        "z_m",
        "longitude",
        "latitude",
        "x_m",
        "y_m",
        "x_rot_m",
        "y_rot_m",
        "temp_direction_rad",
        "temp_direction_deg",
        "axis_alignment_method",
        "global_deviate_angle_deg",
        "global_deviate_angle_rad",
        "global_theta0_deg",
        "global_theta0_rad",
        "global_alpha_ok",
        "global_alpha_n_layers",
        "global_alpha_weight_sum",
        "global_alpha_rmse_deg",
    ]
    return points[keep_columns].copy(), object_info.reset_index(names="eddy3d_object_id")


def fit_quadratic(points: pd.DataFrame, shape: str, polarity: str) -> dict:
    valid = points[np.isfinite(points["z_m"]) & np.isfinite(points["x_rot_m"]) & np.isfinite(points["y_rot_m"])]
    if valid.empty:
        return {"shape_class": shape, "polarity": polarity, "n_points": 0, "n_objects": 0}
    z = valid["z_m"].to_numpy(dtype="f8")
    design = np.column_stack([np.ones_like(z), z, z * z])
    x = valid["x_rot_m"].to_numpy(dtype="f8")
    y = valid["y_rot_m"].to_numpy(dtype="f8")
    cx, *_ = np.linalg.lstsq(design, x, rcond=None)
    cy, *_ = np.linalg.lstsq(design, y, rcond=None)
    err_x = x - design @ cx
    err_y = y - design @ cy
    return {
        "shape_class": shape,
        "polarity": polarity,
        "n_points": int(valid.shape[0]),
        "n_objects": int(valid["eddy3d_object_id"].nunique()),
        "z_min_m": float(np.nanmin(z)),
        "z_max_m": float(np.nanmax(z)),
        "c1": float(cx[0]),
        "c2": float(cx[1]),
        "c3": float(cx[2]),
        "c4": float(cy[0]),
        "c5": float(cy[1]),
        "c6": float(cy[2]),
        "rmse_x_m": float(np.sqrt(np.nanmean(err_x * err_x))),
        "rmse_y_m": float(np.sqrt(np.nanmean(err_y * err_y))),
        "rmse_2d_m": float(np.sqrt(np.nanmean(err_x * err_x + err_y * err_y))),
    }


def depth_distribution(points: pd.DataFrame) -> pd.DataFrame:
    if points.empty:
        return pd.DataFrame()
    rows: list[dict] = []
    for (shape, polarity, depth_index), part in points.groupby(["shape_class", "polarity", "depth_index"], sort=True):
        rows.append(
            {
                "shape_class": shape,
                "polarity": polarity,
                "depth_index": int(depth_index),
                "n_points": int(part.shape[0]),
                "z_m": float(part["z_m"].median()),
                "x_p10_m": float(part["x_rot_m"].quantile(0.10)),
                "x_p50_m": float(part["x_rot_m"].quantile(0.50)),
                "x_p90_m": float(part["x_rot_m"].quantile(0.90)),
                "y_p10_m": float(part["y_rot_m"].quantile(0.10)),
                "y_p50_m": float(part["y_rot_m"].quantile(0.50)),
                "y_p90_m": float(part["y_rot_m"].quantile(0.90)),
            }
        )
    return pd.DataFrame.from_records(rows)


def sample_object_ids(points: pd.DataFrame, max_lines: int, seed: int) -> np.ndarray:
    ids = points["eddy3d_object_id"].drop_duplicates().to_numpy(dtype="int64")
    if max_lines > 0 and ids.size > max_lines:
        ids = np.random.default_rng(seed).choice(ids, size=max_lines, replace=False)
    return ids


def curve_from_fit(fit: dict, z_grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = fit["c1"] + fit["c2"] * z_grid + fit["c3"] * z_grid * z_grid
    y = fit["c4"] + fit["c5"] * z_grid + fit["c6"] * z_grid * z_grid
    return x, y


def plot_group(
    shape: str,
    polarity: str,
    points: pd.DataFrame,
    dist: pd.DataFrame,
    fit: dict,
    figure_dir: Path,
    *,
    max_lines: int,
    seed: int,
) -> None:
    if points.empty or fit.get("n_points", 0) == 0:
        return
    safe = group_label(shape, polarity)
    z_grid = np.linspace(float(points["z_m"].min()), float(points["z_m"].max()), 180)
    x_fit, y_fit = curve_from_fit(fit, z_grid)

    fig = plt.figure(figsize=(9, 7), dpi=160)
    ax = fig.add_subplot(111, projection="3d")
    for object_id in sample_object_ids(points, max_lines, seed):
        line = points[points["eddy3d_object_id"] == object_id].sort_values("z_m")
        ax.plot(line["x_rot_m"], line["y_rot_m"], line["z_m"], color="0.55", alpha=0.12, linewidth=0.7)
    ax.plot(x_fit, y_fit, z_grid, color="#d62728", linewidth=3.0, label="quadratic fit")
    ax.set_title(f"{shape} {polarity}: rotated centerlines")
    ax.set_xlabel("x_rot eastward (m)")
    ax.set_ylabel("y_rot cross-track (m)")
    ax.set_zlabel("z depth from surface (m)")
    ax.set_zlim(float(points["z_m"].max()), 0.0)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(figure_dir / f"{safe}_3d.png")
    plt.close(fig)

    use_dist = dist[(dist["shape_class"] == shape) & (dist["polarity"] == polarity)].sort_values("z_m")
    for coord, prefix, label in (("x_rot_m", "x", "x_rot eastward"), ("y_rot_m", "y", "y_rot cross-track")):
        fig, ax = plt.subplots(figsize=(7, 8), dpi=160)
        for object_id in sample_object_ids(points, max_lines, seed):
            line = points[points["eddy3d_object_id"] == object_id].sort_values("z_m")
            ax.plot(line[coord], line["z_m"], color="0.55", alpha=0.08, linewidth=0.6)
        if not use_dist.empty:
            ax.fill_betweenx(
                use_dist["z_m"],
                use_dist[f"{prefix}_p10_m"],
                use_dist[f"{prefix}_p90_m"],
                facecolor="#9ecae1",
                alpha=0.22,
                hatch="///",
                edgecolor="#3182bd",
                linewidth=0.0,
                label="10-90% real distribution",
            )
            ax.plot(use_dist[f"{prefix}_p50_m"], use_dist["z_m"], color="#3182bd", linewidth=1.5, label="median")
        ax.plot(x_fit if coord == "x_rot_m" else y_fit, z_grid, color="#d62728", linewidth=2.5, label="quadratic fit")
        ax.axvline(0.0, color="0.2", linewidth=0.8)
        ax.set_title(f"{shape} {polarity}: {label}")
        ax.set_xlabel(f"{label} (m)")
        ax.set_ylabel("z depth from surface (m)")
        ax.invert_yaxis()
        ax.grid(True, color="0.9")
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(figure_dir / f"{safe}_{prefix}z.png")
        plt.close(fig)


def write_summary(
    output_dir: Path,
    *,
    shape_tracks: ShapeTracks,
    centers: pd.DataFrame,
    points: pd.DataFrame,
    object_info: pd.DataFrame,
    fits: pd.DataFrame,
    skipped_groups: list[str],
) -> None:
    fit_table = "No fits generated." if fits.empty else fits.to_csv(index=False)
    lines = [
        "# First_temp direction-fit by polarity summary",
        "",
        f"- Input shape folders: `{shape_tracks.requested_shapes}`",
        f"- Selected center rows before rotation: {len(centers):,}",
        f"- Rotated point rows: {len(points):,}",
        f"- Usable eddy3d objects: {points['eddy3d_object_id'].nunique() if not points.empty else 0:,}",
        f"- Skipped empty groups: {', '.join(skipped_groups) if skipped_groups else 'none'}",
        "",
        "## Track counts by shape and polarity",
    ]
    for (shape, polarity), count in sorted(shape_tracks.counts.items()):
        lines.append(f"- {shape} / {polarity}: {count:,}")
    if not object_info.empty and "deep_rotation_abs_y_m" in object_info:
        usable = object_info[object_info["is_usable"]].copy()
        if not usable.empty:
            lines.extend(
                [
                    "",
                    "## Rotation checks",
                    f"- Max |deep y_rot|: {usable['deep_rotation_abs_y_m'].max():.6g} m",
                    f"- Min deep x_rot: {usable['deep_x_rot_m'].min():.6g} m",
                    "- Surface z/x/y are constructed as zero per object.",
                ]
            )
    lines.extend(["", "## Fit coefficients", "```csv", fit_table.strip(), "```"])
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    shapes = parse_csv_list(args.shapes)
    shape_tracks = load_shape_tracks(Path(args.by_shape_dir), shapes)
    centers = read_selected_centers(Path(args.centers), shape_tracks)
    centers = limit_objects(centers, int(args.max_objects_per_shape), int(args.random_seed))
    points, object_info = build_rotated_points(
        centers,
        min_layers=int(args.min_layers),
        min_depth_span_m=float(args.min_depth_span_m),
        axis_alignment=str(args.axis_alignment),
    )
    points.to_parquet(output_dir / "rotated_points.parquet", index=False)
    object_info.to_parquet(output_dir / "object_diagnostics.parquet", index=False)

    dist = depth_distribution(points)
    dist.to_csv(output_dir / "depth_distribution.csv", index=False)

    fit_rows: list[dict] = []
    skipped_groups: list[str] = []
    group_keys = sorted(shape_tracks.counts)
    for shape, polarity in group_keys:
        part = points[(points["shape_class"] == shape) & (points["polarity"] == polarity)]
        if part.empty:
            skipped_groups.append(f"{shape}/{polarity}")
            continue
        fit_rows.append(fit_quadratic(part, shape, polarity))
    fits = pd.DataFrame.from_records(fit_rows)
    fits.to_csv(output_dir / "fit_coefficients.csv", index=False)
    fit_by_group = {(row["shape_class"], row["polarity"]): row for row in fit_rows}

    for shape, polarity in tqdm(group_keys, desc="Plot direction fits", unit="group"):
        part = points[(points["shape_class"] == shape) & (points["polarity"] == polarity)]
        fit = fit_by_group.get((shape, polarity))
        if fit is None:
            continue
        plot_group(shape, polarity, part, dist, fit, figure_dir, max_lines=int(args.max_lines), seed=int(args.random_seed))

    write_summary(
        output_dir,
        shape_tracks=shape_tracks,
        centers=centers,
        points=points,
        object_info=object_info,
        fits=fits,
        skipped_groups=skipped_groups,
    )
    print(f"Output: {output_dir}")
    print(f"Rotated points: {output_dir / 'rotated_points.parquet'}")
    print(f"Fit coefficients: {output_dir / 'fit_coefficients.csv'}")
    print(f"Figures: {figure_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rotate vertical eddy centerlines by shape+polarity and fit quadratic x(z), y(z).")
    parser.add_argument("--by-shape-dir", default=str(DEFAULT_BY_SHAPE_DIR))
    parser.add_argument("--centers", default=str(DEFAULT_CENTERS))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--shapes", default=",".join(DEFAULT_SHAPE_ORDER), help="Comma-separated shape classes to process.")
    parser.add_argument("--min-layers", type=int, default=3)
    parser.add_argument("--min-depth-span-m", type=float, default=10.0)
    parser.add_argument(
        "--axis-alignment",
        default=AXIS_ALIGNMENT_SURFACE_TO_DEEP,
        choices=[AXIS_ALIGNMENT_SURFACE_TO_DEEP, AXIS_ALIGNMENT_GLOBAL_LS_ALPHA],
    )
    parser.add_argument("--max-objects-per-shape", type=int, default=0, help="0 means no limit; useful for smoke tests.")
    parser.add_argument("--max-lines", type=int, default=300, help="Maximum real centerlines drawn per group figure.")
    parser.add_argument("--random-seed", type=int, default=20260708)
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
