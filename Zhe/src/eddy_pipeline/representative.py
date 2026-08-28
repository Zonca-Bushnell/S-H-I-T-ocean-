from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import netCDF4
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.utils.field_sampling import (
    bilinear_sample,
    make_polar_grid,
    sanitize_ocean_field,
    xy_to_lonlat,
)


EARTH_RADIUS_M = 6_371_000.0


def _parse_shapes(value: str) -> tuple[str, ...]:
    shapes = tuple(part.strip() for part in str(value).split(",") if part.strip())
    if not shapes:
        raise ValueError("--shapes must contain at least one shape class")
    return shapes


def _shape_label(shapes: tuple[str, ...]) -> str:
    return "-".join(shapes) + "-only" if len(shapes) == 1 else "-".join(shapes)


@dataclass(frozen=True)
class FilterDay:
    lon: np.ndarray
    lat: np.ndarray
    depth: np.ndarray
    u: np.ndarray
    v: np.ndarray


def _parse_date(value: str) -> date:
    return pd.Timestamp(value).date()


def _dates_from_nc(path: Path) -> dict[date, int]:
    with netCDF4.Dataset(path) as ds:
        values = ds.variables["time"][:]
        units = ds.variables["time"].units
        calendar = getattr(ds.variables["time"], "calendar", "standard")
        out = netCDF4.num2date(values, units=units, calendar=calendar)
    return {date(int(v.year), int(v.month), int(v.day)): i for i, v in enumerate(out)}


def _filter_path(root: Path, year: int, template: str) -> Path:
    path = root / template.format(year=year)
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _read_filter_day(root: Path, day: date, template: str, max_depth_m: float | None) -> FilterDay:
    path = _filter_path(root, day.year, template)
    day_index = _dates_from_nc(path)[day]
    with netCDF4.Dataset(path) as ds:
        depth_all = np.asarray(ds.variables["depth"][:], dtype="f8")
        if max_depth_m is None:
            depth_indexer = np.arange(depth_all.size)
        else:
            depth_indexer = np.where(depth_all <= float(max_depth_m) + 1.0e-6)[0]
        lon = np.asarray(ds.variables["longitude"][:], dtype="f8")
        lat = np.asarray(ds.variables["latitude"][:], dtype="f8")
        depth = depth_all[depth_indexer]
        u = np.ma.filled(ds.variables["uo_glor"][day_index, depth_indexer, :, :], np.nan).astype("f8", copy=False)
        v = np.ma.filled(ds.variables["vo_glor"][day_index, depth_indexer, :, :], np.nan).astype("f8", copy=False)
    u = sanitize_ocean_field(u)
    v = sanitize_ocean_field(v)
    return FilterDay(lon=lon, lat=lat, depth=depth, u=u, v=v)


def _load_objects(
    rv_root: Path,
    start: str,
    end: str,
    max_objects_per_polarity: int,
    seed: int,
    shapes: tuple[str, ...],
) -> pd.DataFrame:
    path = rv_root / "object_cache" / "selected_lifecycle_objects.parquet"
    objects = pd.read_parquet(path)
    diag_path = rv_root / "axis" / "object_diagnostics.parquet"
    if diag_path.exists():
        diag = pd.read_parquet(diag_path)
        objects["eddy3d_object_id"] = objects["eddy3d_object_id"].astype("int64")
        diag["eddy3d_object_id"] = diag["eddy3d_object_id"].astype("int64")
        add_cols = [
            col
            for col in (
                "axis_alignment_method",
                "global_alpha_ok",
                "global_deviate_angle_rad",
                "global_deviate_angle_deg",
            )
            if col not in objects.columns and col in diag.columns
        ]
        if add_cols:
            objects = objects.merge(diag[["eddy3d_object_id", *add_cols]], on="eddy3d_object_id", how="left")
    if "global_deviate_angle_rad" not in objects and "global_deviate_angle_deg" in objects:
        objects["global_deviate_angle_rad"] = np.radians(objects["global_deviate_angle_deg"].astype("float64"))
    objects = objects[objects["shape_class"].astype(str).isin(shapes)].copy()
    if "axis_alignment_method" in objects:
        objects = objects[objects["axis_alignment_method"].astype(str).eq("global_ls_alpha")].copy()
    if "global_alpha_ok" in objects:
        objects = objects[objects["global_alpha_ok"].astype(bool)].copy()
    objects["date"] = pd.to_datetime(objects["date"]).dt.strftime("%Y-%m-%d")
    if start:
        objects = objects[objects["date"] >= pd.Timestamp(start).strftime("%Y-%m-%d")].copy()
    if end:
        objects = objects[objects["date"] <= pd.Timestamp(end).strftime("%Y-%m-%d")].copy()
    required = {
        "eddy3d_object_id",
        "track3d_id",
        "polarity",
        "date",
        "life_phase",
        "mean_radius_m",
        "surface_lon",
        "surface_lat",
    }
    missing = sorted(required - set(objects.columns))
    if missing:
        raise KeyError(f"selected_lifecycle_objects missing columns: {missing}")
    if max_objects_per_polarity > 0:
        rng = np.random.default_rng(seed)
        keep: list[int] = []
        for _, part in objects.groupby("polarity", sort=False):
            ids = part["eddy3d_object_id"].to_numpy(dtype="i8")
            if len(ids) > max_objects_per_polarity:
                ids = rng.choice(ids, size=max_objects_per_polarity, replace=False)
            keep.extend(int(v) for v in ids)
        objects = objects[objects["eddy3d_object_id"].isin(keep)].copy()
    return objects.sort_values(["date", "polarity", "eddy3d_object_id"]).reset_index(drop=True)


def _load_center_lines(rv_root: Path, object_ids: set[int]) -> dict[int, pd.DataFrame]:
    cols = [
        "eddy3d_object_id",
        "track3d_id",
        "date",
        "polarity",
        "depth_index",
        "depth_m",
        "x_m",
        "y_m",
        "x_rot_m",
        "y_rot_m",
        "axis_alignment_method",
    ]
    points = pd.read_parquet(rv_root / "axis" / "rotated_points.parquet", columns=cols)
    points = points[points["eddy3d_object_id"].astype("int64").isin(object_ids)].copy()
    if "axis_alignment_method" in points:
        points = points[points["axis_alignment_method"].astype(str).eq("global_ls_alpha")].copy()
    points["eddy3d_object_id"] = points["eddy3d_object_id"].astype("int64")
    points["depth_index"] = points["depth_index"].astype("int16")
    return {
        int(object_id): part.sort_values("depth_index").reset_index(drop=True)
        for object_id, part in points.groupby("eddy3d_object_id", sort=False)
    }


def _tau_weights(tau_grid: np.ndarray, phase: float, bandwidth: float, weight_min: float) -> list[tuple[int, float]]:
    weights = np.exp(-0.5 * ((tau_grid - float(phase)) / max(float(bandwidth), 1.0e-12)) ** 2)
    return [(int(i), float(w)) for i, w in enumerate(weights) if float(w) >= float(weight_min)]


def _init_accum(shape: tuple[int, int, int]) -> dict[str, np.ndarray | set]:
    return {
        "sum_u": np.zeros(shape, dtype="f8"),
        "sum_v": np.zeros(shape, dtype="f8"),
        "sum_speed": np.zeros(shape, dtype="f8"),
        "count": np.zeros(shape, dtype="f8"),
        "objects": set(),
        "tracks": set(),
        "dates": set(),
    }


def _sample_object_velocity(
    obj,
    center_line: pd.DataFrame,
    day_data: FilterDay,
    rr: np.ndarray,
    tt: np.ndarray,
    orientation_mode: str,
) -> dict[str, np.ndarray] | None:
    if center_line.empty:
        return None
    radius = float(obj.mean_radius_m)
    alpha = float(getattr(obj, "global_deviate_angle_rad", getattr(obj, "temp_direction_rad", np.nan)))
    if orientation_mode == "global_alpha" and not np.isfinite(alpha):
        return None
    cos_a = math.cos(alpha)
    sin_a = math.sin(alpha)
    local_x = rr * radius * np.cos(tt)
    local_y = rr * radius * np.sin(tt)
    shape = (len(day_data.depth), rr.shape[0], rr.shape[1])
    u_arr = np.full(shape, np.nan, dtype="f8")
    v_arr = np.full(shape, np.nan, dtype="f8")
    filled = 0
    for row in center_line.itertuples(index=False):
        k = int(row.depth_index)
        if k < 0 or k >= len(day_data.depth):
            continue
        if orientation_mode == "global_alpha":
            x_rot = float(row.x_rot_m) + local_x
            y_rot = float(row.y_rot_m) + local_y
            x_orig = x_rot * cos_a - y_rot * sin_a
            y_orig = x_rot * sin_a + y_rot * cos_a
        elif orientation_mode == "unturned":
            x_orig = float(row.x_m) + local_x
            y_orig = float(row.y_m) + local_y
        else:
            raise ValueError(f"Unsupported orientation mode: {orientation_mode}")
        target_lon, target_lat = xy_to_lonlat(x_orig, y_orig, float(obj.surface_lon), float(obj.surface_lat))
        u_s = bilinear_sample(day_data.lon, day_data.lat, day_data.u[k], target_lon, target_lat)
        v_s = bilinear_sample(day_data.lon, day_data.lat, day_data.v[k], target_lon, target_lat)
        if orientation_mode == "global_alpha":
            u_arr[k] = u_s * cos_a + v_s * sin_a
            v_arr[k] = -u_s * sin_a + v_s * cos_a
        else:
            u_arr[k] = u_s
            v_arr[k] = v_s
        filled += 1
    if filled < 3:
        return None
    return {
        "u": u_arr,
        "v": v_arr,
        "speed": np.hypot(u_arr, v_arr),
    }


def _add_sample(accum: dict, sample: dict[str, np.ndarray], weight: float, object_id: int, track_id: int, date_text: str) -> None:
    valid = np.isfinite(sample["u"]) & np.isfinite(sample["v"])
    accum["sum_u"] += np.nan_to_num(sample["u"], nan=0.0) * valid * weight
    accum["sum_v"] += np.nan_to_num(sample["v"], nan=0.0) * valid * weight
    accum["sum_speed"] += np.nan_to_num(sample["speed"], nan=0.0) * valid * weight
    accum["count"] += valid.astype("f8") * weight
    accum["objects"].add(int(object_id))
    accum["tracks"].add(int(track_id))
    accum["dates"].add(str(date_text))


def _finalize(accum: dict, tau_grid: np.ndarray, depth: np.ndarray, radial: np.ndarray, theta: np.ndarray) -> dict[str, np.ndarray | list]:
    polarities = sorted({key[0] for key in accum})
    out: dict[str, np.ndarray | list] = {
        "polarities": np.asarray(polarities, dtype=object),
        "tau_grid": tau_grid.astype("f8"),
        "depth": depth.astype("f8"),
        "radial": radial.astype("f8"),
        "theta": theta.astype("f8"),
    }
    shape = (len(polarities), len(tau_grid), len(depth), len(radial), len(theta))
    for name in ("u_mean", "v_mean", "speed_mean", "count"):
        out[name] = np.full(shape, np.nan, dtype="f8")
    n_objects = np.zeros((len(polarities), len(tau_grid)), dtype="i8")
    n_tracks = np.zeros_like(n_objects)
    for ip, polarity in enumerate(polarities):
        for it in range(len(tau_grid)):
            item = accum.get((polarity, it))
            if item is None:
                continue
            count = item["count"]
            out["count"][ip, it] = count
            for target, source in (
                ("u_mean", "sum_u"),
                ("v_mean", "sum_v"),
                ("speed_mean", "sum_speed"),
            ):
                out[target][ip, it] = np.divide(
                    item[source],
                    count,
                    out=np.full_like(item[source], np.nan),
                    where=count > 0,
                )
            n_objects[ip, it] = len(item["objects"])
            n_tracks[ip, it] = len(item["tracks"])
    out["n_objects"] = n_objects
    out["n_tracks"] = n_tracks
    return out


def _load_radial_profiles(rv_root: Path) -> pd.DataFrame:
    path = rv_root / "streamfunction_templates" / "continuous_radial_psi_profiles.parquet"
    profiles = pd.read_parquet(path)
    if "axis_mode" in profiles.columns:
        profiles = profiles[profiles["axis_mode"].astype(str).eq("tilted")].copy()
    return profiles


def _old_axisymmetric_speed(profile: pd.DataFrame, polarity: str, tau: float, depth: np.ndarray, radial: np.ndarray) -> np.ndarray:
    part = profile[(profile["polarity"].astype(str).eq(polarity)) & np.isclose(profile["tau_center"].astype("f8"), tau)]
    if part.empty:
        return np.full((len(depth), len(radial)), np.nan)
    matrix = (
        part.pivot_table(index="depth_m", columns="r_over_R", values="psi_mean", aggfunc="mean")
        .reindex(index=depth, columns=radial)
        .to_numpy(dtype="f8")
    )
    speed = np.full_like(matrix, np.nan)
    for k in range(matrix.shape[0]):
        if np.sum(np.isfinite(matrix[k])) < 3:
            continue
        speed[k] = np.abs(np.gradient(matrix[k], radial, edge_order=1))
    finite = np.isfinite(speed)
    if np.any(finite):
        scale = np.nanpercentile(np.asarray(speed[finite]), 98)
        if scale > 0:
            speed = speed / scale
    return speed


def _polar_to_xy(field: np.ndarray, radial: np.ndarray, theta: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    r = np.hypot(x, y)
    phi = np.mod(np.arctan2(y, x), 2.0 * np.pi)
    ri = np.abs(radial[:, None, None] - r[None, :, :]).argmin(axis=0)
    ti = np.abs(np.angle(np.exp(1j * (theta[:, None, None] - phi[None, :, :])))).argmin(axis=0)
    return field[ri, ti]


def _plot_comparison(
    out_dir: Path,
    arrays: dict[str, np.ndarray | list],
    rv_root: Path,
    rmax: float,
    selected_tau: float,
    depth_targets: list[float],
    orientation_mode: str,
    shape_label: str,
) -> None:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    radial = arrays["radial"]
    theta = arrays["theta"]
    depth = arrays["depth"]
    tau_grid = arrays["tau_grid"]
    polarities = [str(v) for v in arrays["polarities"]]
    tau_i = int(np.abs(tau_grid - selected_tau).argmin())
    profiles = _load_radial_profiles(rv_root)
    xy = np.linspace(-rmax, rmax, 121)
    xx, yy = np.meshgrid(xy, xy)
    mask = np.hypot(xx, yy) <= rmax
    for ip, polarity in enumerate(polarities):
        depth_indices = [int(np.abs(depth - value).argmin()) for value in depth_targets]
        old_speed = _old_axisymmetric_speed(profiles, polarity, float(tau_grid[tau_i]), depth, radial)
        new_speed = arrays["speed_mean"][ip, tau_i]
        nrows = len(depth_indices)
        fig, axes = plt.subplots(nrows, 3, figsize=(12, 3.5 * nrows), constrained_layout=True)
        if nrows == 1:
            axes = axes[None, :]
        for row, k in enumerate(depth_indices):
            old_layer = np.interp(np.hypot(xx, yy), radial, old_speed[k], left=np.nan, right=np.nan)
            new_layer = _polar_to_xy(new_speed[k], radial, theta, xx, yy)
            old_layer = np.where(mask, old_layer, np.nan)
            new_layer = np.where(mask, new_layer, np.nan)
            vmax = np.nanpercentile(np.concatenate([old_layer[np.isfinite(old_layer)], new_layer[np.isfinite(new_layer)]]), 98)
            for ax, data, title, cmap in (
                (axes[row, 0], old_layer, "old radial/axisymmetric", "magma"),
                (axes[row, 1], new_layer, "new azimuth-preserved", "magma"),
                (axes[row, 2], new_layer - old_layer, "new - old", "RdBu_r"),
            ):
                if title == "new - old":
                    lim = np.nanpercentile(np.abs(data[np.isfinite(data)]), 98) if np.any(np.isfinite(data)) else 1.0
                    im = ax.imshow(data, extent=[-rmax, rmax, -rmax, rmax], origin="lower", cmap=cmap, vmin=-lim, vmax=lim)
                else:
                    im = ax.imshow(data, extent=[-rmax, rmax, -rmax, rmax], origin="lower", cmap=cmap, vmin=0, vmax=vmax)
                ax.contour(xx, yy, mask.astype(float), levels=[0.5], colors="white", linewidths=0.8)
                ax.set_title(f"{title}\nz={depth[k]:.0f} m")
                ax.set_xlabel("x_rot / R")
                ax.set_ylabel("y_rot / R")
                ax.set_aspect("equal")
                fig.colorbar(im, ax=ax, shrink=0.8)
        fig.suptitle(f"{polarity} {shape_label} tau={tau_grid[tau_i]:.2f}: {orientation_mode} speed composite comparison", fontsize=14)
        fig.savefig(fig_dir / f"{polarity}_tau{tau_i:02d}_speed_old_radial_vs_azimuthal.png", dpi=180)
        plt.close(fig)


def _axis_xy_for_plot(line: pd.DataFrame, alpha: float, orientation_mode: str) -> tuple[np.ndarray, np.ndarray]:
    if orientation_mode == "unturned":
        return line["x_m"].to_numpy(dtype="f8"), line["y_m"].to_numpy(dtype="f8")
    return line["x_rot_m"].to_numpy(dtype="f8"), line["y_rot_m"].to_numpy(dtype="f8")


def _plot_axis_3d(
    out_dir: Path,
    rv_root: Path,
    objects: pd.DataFrame,
    center_lines: dict[int, pd.DataFrame],
    orientation_mode: str,
    shape_label: str,
) -> None:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    for polarity, part in objects.groupby("polarity", sort=True):
        ids = part["eddy3d_object_id"].drop_duplicates().to_numpy(dtype="i8")
        alpha_by_id = {
            int(row.eddy3d_object_id): float(getattr(row, "global_deviate_angle_rad", getattr(row, "temp_direction_rad", 0.0)))
            for row in part.itertuples(index=False)
        }
        if len(ids) > 80:
            ids = rng.choice(ids, 80, replace=False)
        fig = plt.figure(figsize=(9, 7))
        ax = fig.add_subplot(111, projection="3d")
        for object_id in ids:
            line = center_lines.get(int(object_id))
            if line is None or line.empty:
                continue
            x_m, y_m = _axis_xy_for_plot(line, alpha_by_id.get(int(object_id), 0.0), orientation_mode)
            x = x_m / 1000.0
            y = y_m / 1000.0
            z = -line["depth_m"].to_numpy(dtype="f8") / 1000.0
            ax.plot(x, y, z, color="0.65", alpha=0.25, linewidth=0.8)
        all_lines = [center_lines[int(v)] for v in ids if int(v) in center_lines]
        if all_lines:
            transformed = []
            for object_id in ids:
                line = center_lines.get(int(object_id))
                if line is None or line.empty:
                    continue
                item = line.copy()
                x_m, y_m = _axis_xy_for_plot(item, alpha_by_id.get(int(object_id), 0.0), orientation_mode)
                item["plot_x_m"] = x_m
                item["plot_y_m"] = y_m
                transformed.append(item)
            merged = pd.concat(transformed, ignore_index=True)
            med = merged.groupby("depth_index", sort=True)[["x_rot_m", "y_rot_m", "plot_x_m", "plot_y_m", "depth_m"]].median().reset_index()
            ax.plot(
                med.get("plot_x_m", med["x_rot_m"]).to_numpy(dtype="f8") / 1000.0,
                med.get("plot_y_m", med["y_rot_m"]).to_numpy(dtype="f8") / 1000.0,
                -med["depth_m"].to_numpy(dtype="f8") / 1000.0,
                color="red",
                linewidth=3.0,
                label="median axis",
            )
        ax.scatter([0], [0], [0], color="black", s=50, marker="+", label="surface ref")
        axis_label = "unturned east/north axes" if orientation_mode == "unturned" else "global_ls_alpha axes"
        ax.set_title(f"{polarity} {shape_label} {axis_label}\nsame objects used by old and azimuth-preserved composites")
        ax.set_xlabel("east km" if orientation_mode == "unturned" else "x_rot km")
        ax.set_ylabel("north km" if orientation_mode == "unturned" else "y_rot km")
        ax.set_zlabel("depth km")
        ax.legend(loc="upper left")
        fig.savefig(fig_dir / f"{polarity}_{shape_label}_3d_axes.png", dpi=180)
        plt.close(fig)


def _write_summary(out_dir: Path, arrays: dict[str, np.ndarray | list], objects: pd.DataFrame, args: argparse.Namespace) -> None:
    polarities = [str(v) for v in arrays["polarities"]]
    shapes = _parse_shapes(args.shapes)
    shape_label = _shape_label(shapes)
    lines = [
        "# Azimuth-preserved representative vortex experiment",
        "",
        "This experiment keeps the azimuthal dimension in the requested orientation frame.",
        "It is meant to compare with the current radial representative vortex, where fields are stored as `(tau, depth, r)`.",
        "",
        f"- RV root: `{args.rv_root}`",
        f"- Filter root: `{args.filter_root}`",
        f"- Shape filter: `{', '.join(shapes)}`",
        f"- Objects used: `{len(objects)}` {shape_label} object-days",
        f"- Polarities: `{', '.join(polarities)}`",
        f"- Output array shape: `polarity x tau x depth x radius x azimuth = {arrays['speed_mean'].shape}`",
        f"- Orientation mode: `{args.orientation_mode}`",
        "",
        "Interpretation:",
        "- Old radial reconstruction is axisymmetric by construction and cannot show crescent/open-ring speed structure.",
        "- New azimuth-preserved composite can retain a systematic crescent only if the feature survives the chosen orientation frame.",
        f"- The 3D axis is not recomputed here; both composites use the same {shape_label} `global_ls_alpha` axis definition.",
    ]
    (out_dir / "azimuthal_representative_summary.md").write_text("\n".join(lines), encoding="utf-8")
    summary = {
        "rv_root": str(args.rv_root),
        "filter_root": str(args.filter_root),
        "shape_filter": list(shapes),
        "shape_label": shape_label,
        "objects": int(len(objects)),
        "polarities": polarities,
        "array_shape": list(arrays["speed_mean"].shape),
        "tau_nodes": int(len(arrays["tau_grid"])),
        "depth_layers": int(len(arrays["depth"])),
        "radial_bins": int(len(arrays["radial"])),
        "azimuth_bins": int(len(arrays["theta"])),
        "orientation_mode": str(args.orientation_mode),
    }
    (out_dir / "azimuthal_representative_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    rv_root = Path(args.rv_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tau_grid = np.arange(0.0, 1.0 + 0.5 * float(args.tau_grid_step), float(args.tau_grid_step), dtype="f8")
    radial, theta, rr, tt, _ = make_polar_grid(float(args.rmax), int(args.radial_bins), int(args.azimuth_bins))
    shapes = _parse_shapes(args.shapes)
    shape_label = _shape_label(shapes)
    objects = _load_objects(rv_root, args.start, args.end, int(args.max_objects_per_polarity), int(args.seed), shapes)
    if objects.empty:
        raise ValueError(f"No {shape_label} objects selected")
    center_lines = _load_center_lines(rv_root, set(objects["eddy3d_object_id"].astype("int64")))
    first_day = _read_filter_day(Path(args.filter_root), _parse_date(objects.iloc[0]["date"]), args.filter_template, args.max_depth_m)
    depth = first_day.depth
    shape = (len(depth), len(radial), len(theta))
    accum: dict[tuple[str, int], dict] = {}
    day_cache: dict[date, FilterDay] = {_parse_date(objects.iloc[0]["date"]): first_day}
    grouped = list(objects.groupby("date", sort=True))
    for date_text, day_objects in tqdm(grouped, desc=f"Azimuth-preserved {shape_label} composite"):
        day = _parse_date(date_text)
        day_data = day_cache.get(day)
        if day_data is None:
            day_data = _read_filter_day(Path(args.filter_root), day, args.filter_template, args.max_depth_m)
            day_cache.clear()
            day_cache[day] = day_data
        for obj in day_objects.itertuples(index=False):
            line = center_lines.get(int(obj.eddy3d_object_id))
            if line is None:
                continue
            sample = _sample_object_velocity(obj, line, day_data, rr, tt, str(args.orientation_mode))
            if sample is None or sample["u"].shape != shape:
                continue
            for tau_index, weight in _tau_weights(tau_grid, float(obj.life_phase), args.kernel_bandwidth, args.kernel_weight_min):
                key = (str(obj.polarity), tau_index)
                if key not in accum:
                    accum[key] = _init_accum(shape)
                _add_sample(accum[key], sample, weight, int(obj.eddy3d_object_id), int(obj.track3d_id), str(date_text))
    arrays = _finalize(accum, tau_grid, depth, radial, theta)
    np.savez_compressed(
        out_dir / "azimuthal_representative_velocity.npz",
        **{k: v for k, v in arrays.items() if isinstance(v, np.ndarray)},
    )
    _plot_comparison(out_dir, arrays, rv_root, float(args.rmax), float(args.plot_tau), [float(v) for v in args.plot_depths.split(",") if v.strip()], str(args.orientation_mode), shape_label)
    _plot_axis_3d(out_dir, rv_root, objects, center_lines, str(args.orientation_mode), shape_label)
    _write_summary(out_dir, arrays, objects, args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rv-root", required=True)
    parser.add_argument("--filter-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--shapes", default="coherent", help="Comma-separated shape classes to composite.")
    parser.add_argument("--filter-template", default="global_phy_{year}_bandpass_30_180d.nc")
    parser.add_argument("--start", default="")
    parser.add_argument("--end", default="")
    parser.add_argument("--max-depth-m", type=float, default=2000.0)
    parser.add_argument("--tau-grid-step", type=float, default=0.05)
    parser.add_argument("--kernel-bandwidth", type=float, default=0.075)
    parser.add_argument("--kernel-weight-min", type=float, default=1.0e-3)
    parser.add_argument("--radial-bins", type=int, default=40)
    parser.add_argument("--azimuth-bins", type=int, default=72)
    parser.add_argument("--rmax", type=float, default=2.5)
    parser.add_argument("--plot-tau", type=float, default=0.5)
    parser.add_argument("--plot-depths", default="0,200,500,1000,1500")
    parser.add_argument("--max-objects-per-polarity", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--orientation-mode", choices=["global_alpha", "unturned"], default="global_alpha")
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
