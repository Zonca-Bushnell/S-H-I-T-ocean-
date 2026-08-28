from __future__ import annotations

import argparse
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import netCDF4
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.utils.axis_streamfunction import grid_spacing_m, relative_vorticity, streamfunction_from_zeta
from src.utils.axis_alignment import (
    build_rotated_points,
    depth_distribution,
    fit_quadratic,
    load_shape_tracks,
    parse_csv_list as parse_shape_list,
)
from src.utils.representative_accumulation import (
    add_weighted_terms,
    build_continuous_mlrw_profiles,
    build_manifest,
    build_nonlinearity_tables,
    finalize_weighted,
    parse_tau_grid,
    representative_radii,
    rows_from_continuous_final,
)
from src.utils.ep_flux import add_wave_activity, compute_nondim_terms
from src.utils.field_sampling import (
    OMEGA,
    G,
    RHO0,
    load_n2,
    make_polar_grid,
    read_climatology_uv,
    sample_object_fields,
    sanitize_ocean_field,
)

from .utils.common import load_config, parse_ymd
from .utils.streaming_cmems import build_source_index, load_year_cache, read_day_data, source_paths_for_years
from .utils.table_io import read_table, read_table_or_partitions, write_table_fast


VALID_SHAPES = ("coherent", "complex", "mixed", "transitional", "upright_like")
VALID_POLARITIES = ("cyclonic", "anticyclonic")
SUBDIRS = (
    "axis",
    "object_cache",
    "streamfunction_templates",
    "ep_flux_terms",
    "wave_action_total_flux",
    "mlrw_applicability",
    "core_environment_sensitivity",
    "climatology",
    "partial_accum_parts",
    "logs",
)


@dataclass(frozen=True)
class ChunkTask:
    chunk_id: int
    start: str
    end: str
    objects: pd.DataFrame
    center_lines: pd.DataFrame
    output_path: Path


def _split_csv(value: str, default: tuple[str, ...]) -> tuple[str, ...]:
    if not value.strip():
        return default
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _ensure_tree(output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    out = {}
    for name in SUBDIRS:
        path = output_dir / name
        path.mkdir(parents=True, exist_ok=True)
        out[name] = path
    return out


def _write_summary(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _load_completed_centers(catalog_dir: Path) -> pd.DataFrame:
    return read_table_or_partitions(
        catalog_dir / "layer_centers_completed.parquet",
        catalog_dir / "layer_centers_completed_parts",
    )


def build_axis_products(
    results_root: Path,
    output_dir: Path,
    shapes: tuple[str, ...],
    seed: int,
    shape_dir_name: str,
    axis_alignment: str,
) -> Path:
    axis_dir = output_dir / "axis"
    by_shape_dir = results_root / shape_dir_name / "by_shape"
    catalog_dir = results_root / "catalog"
    centers = _load_completed_centers(catalog_dir)
    shape_tracks = load_shape_tracks(by_shape_dir, shapes)
    centers = centers[
        centers["track3d_id"].astype("int64").isin(shape_tracks.track_to_shape)
    ].copy()
    centers["track3d_id"] = centers["track3d_id"].astype("int64")
    centers["eddy3d_object_id"] = centers["eddy3d_object_id"].astype("int64")
    centers["depth_index"] = centers["depth_index"].astype("int16")
    centers["shape_class"] = centers["track3d_id"].map(shape_tracks.track_to_shape)
    centers["polarity"] = centers["track3d_id"].map(shape_tracks.track_to_polarity)
    centers = centers.dropna(subset=["shape_class", "polarity", "longitude", "latitude", "depth_m"])

    points, object_info = build_rotated_points(
        centers,
        min_layers=3,
        min_depth_span_m=10.0,
        axis_alignment=axis_alignment,
    )
    write_table_fast(points, axis_dir / "rotated_points.parquet")
    write_table_fast(object_info, axis_dir / "object_diagnostics.parquet")
    depth_distribution(points).to_csv(axis_dir / "depth_distribution.csv", index=False)

    fit_rows: list[dict] = []
    skipped: list[str] = []
    for shape, polarity in sorted(shape_tracks.counts):
        part = points[(points["shape_class"] == shape) & (points["polarity"] == polarity)]
        if part.empty:
            skipped.append(f"{shape}/{polarity}")
            continue
        fit_rows.append(fit_quadratic(part, shape, polarity))
    fits = pd.DataFrame.from_records(fit_rows)
    fits.to_csv(axis_dir / "fit_coefficients.csv", index=False)

    _write_summary(
        axis_dir / "summary.md",
        [
            "# Representative vortex axis products",
            "",
            f"- Center rows: {len(centers):,}",
            f"- Rotated point rows: {len(points):,}",
            f"- Object rows: {len(object_info):,}",
            f"- Usable objects: {int(object_info['is_usable'].sum()) if not object_info.empty else 0:,}",
            f"- Shapes: {','.join(shapes)}",
            f"- Axis alignment: {axis_alignment}",
            f"- Random seed: {seed}",
            f"- Skipped groups: {', '.join(skipped) if skipped else 'none'}",
        ],
    )
    return axis_dir


def _load_tracks_by_shape(shape_dir: Path, shapes: tuple[str, ...]) -> pd.DataFrame:
    frames = []
    for shape in shapes:
        path = shape_dir / shape / "tracks.csv"
        if path.exists():
            part = pd.read_csv(path)
            if not part.empty:
                part["shape_class"] = str(shape)
                frames.append(part)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out = out.drop_duplicates(subset=["track3d_id"]).copy()
    out["track3d_id"] = out["track3d_id"].astype("int64")
    out["start_date"] = pd.to_datetime(out["start_date"])
    out["end_date"] = pd.to_datetime(out["end_date"])
    out["lifetime_days"] = out["lifetime_days"].astype("float64")
    return out


def select_lifecycle_objects(
    results_root: Path,
    axis_dir: Path,
    shape_dir_name: str,
    shapes: tuple[str, ...],
    polarities: tuple[str, ...],
    max_days: int,
    max_objects_per_polarity: int,
    seed: int,
) -> pd.DataFrame:
    shape_dir = results_root / shape_dir_name / "by_shape"
    catalog_dir = results_root / "catalog"
    objects = read_table(axis_dir / "object_diagnostics.parquet")
    objects = objects[objects["is_usable"]].copy()
    objects = objects[objects["shape_class"].isin(shapes) & objects["polarity"].isin(polarities)].copy()
    tracks = _load_tracks_by_shape(shape_dir, shapes)
    if tracks.empty or objects.empty:
        return objects.iloc[0:0].copy()
    objects["track3d_id"] = objects["track3d_id"].astype("int64")
    objects = objects.merge(
        tracks[["track3d_id", "start_date", "end_date", "lifetime_days", "shape_class", "polarity"]],
        on=["track3d_id", "shape_class", "polarity"],
        how="inner",
    )
    radii = read_table(catalog_dir / "vertical_objects.parquet")[["eddy3d_object_id", "mean_radius_m"]]
    objects = objects.merge(radii, on="eddy3d_object_id", how="left")
    objects = objects[np.isfinite(objects["mean_radius_m"]) & (objects["mean_radius_m"] > 0)].copy()
    objects["date_ts"] = pd.to_datetime(objects["date"])
    objects["life_day"] = (objects["date_ts"] - objects["start_date"]).dt.days.astype("float64")
    denom = np.maximum(objects["lifetime_days"].to_numpy(dtype="float64") - 1.0, 1.0)
    objects["life_phase"] = np.clip(objects["life_day"].to_numpy(dtype="float64") / denom, 0.0, 1.0)
    objects["date"] = objects["date_ts"].dt.strftime("%Y-%m-%d")
    objects = objects.sort_values(["date", "polarity", "eddy3d_object_id"]).copy()

    if max_days > 0 and not objects.empty:
        dates = sorted(objects["date"].unique())[: int(max_days)]
        objects = objects[objects["date"].isin(dates)].copy()
    if max_objects_per_polarity > 0 and not objects.empty:
        rng = np.random.default_rng(seed)
        keep: list[int] = []
        unique = objects[["polarity", "eddy3d_object_id"]].drop_duplicates()
        for _, part in unique.groupby("polarity", sort=False):
            ids = part["eddy3d_object_id"].to_numpy(dtype="int64")
            if ids.size > max_objects_per_polarity:
                ids = rng.choice(ids, size=max_objects_per_polarity, replace=False)
            keep.extend(int(value) for value in ids)
        objects = objects[objects["eddy3d_object_id"].isin(keep)].copy()
    return objects


def _filter_objects_by_date(
    objects: pd.DataFrame,
    date_start: str,
    date_end: str,
) -> pd.DataFrame:
    if objects.empty:
        return objects
    out = objects
    if date_start:
        start = pd.Timestamp(date_start).strftime("%Y-%m-%d")
        out = out[out["date"] >= start]
    if date_end:
        end = pd.Timestamp(date_end).strftime("%Y-%m-%d")
        out = out[out["date"] <= end]
    return out.copy()


def build_n2_profile(climatology_nc: Path, out_path: Path, force: bool = False) -> Path:
    if out_path.exists() and not force:
        return out_path
    import gsw

    with netCDF4.Dataset(climatology_nc) as ds:
        depth = np.asarray(ds.variables["depth"][:], dtype="f8")
        lat = np.asarray(ds.variables["latitude"][:], dtype="f8")
        lon = np.asarray(ds.variables["longitude"][:], dtype="f8")
        theta_var = ds.variables["thetao_clim"]
        salt_var = ds.variables["so_clim"]
        theta_profile = np.full(depth.shape, np.nan, dtype="f8")
        salt_profile = np.full(depth.shape, np.nan, dtype="f8")
        for k in range(len(depth)):
            theta_layer = np.ma.filled(theta_var[:, k, :, :], np.nan).astype("f8", copy=False)
            salt_layer = np.ma.filled(salt_var[:, k, :, :], np.nan).astype("f8", copy=False)
            theta_profile[k] = float(np.nanmean(theta_layer))
            salt_profile[k] = float(np.nanmean(salt_layer))
    lat_ref = float(np.nanmedian(lat))
    lon_ref = float(np.nanmedian(lon))
    pressure = gsw.p_from_z(-depth, lat_ref)
    sa = gsw.SA_from_SP(salt_profile, pressure, lon_ref, lat_ref)
    ct = gsw.CT_from_pt(sa, theta_profile)
    sigma0 = gsw.sigma0(sa, ct)
    dsigma0_dz = np.gradient(sigma0, depth, edge_order=1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, depth=depth.astype("f8"), sigma0=sigma0.astype("f8"), dsigma0_dz=dsigma0_dz.astype("f8"))
    return out_path


def _source_path_by_year(config: dict, years: list[int]) -> dict[int, Path]:
    paths = source_paths_for_years(config, years)
    out: dict[int, Path] = {}
    for path in paths:
        for year in years:
            if str(year) in path.name:
                out[int(year)] = path
    if len(out) != len(set(years)):
        for year, path in zip(sorted(set(years)), paths):
            out.setdefault(int(year), Path(path))
    return out


def _read_depth_coordinate(config: dict, source_path: Path) -> np.ndarray:
    with netCDF4.Dataset(source_path) as ds:
        depth_name = "depth" if "depth" in ds.variables else "depth_glor"
        depth = np.asarray(ds.variables[depth_name][:], dtype="f8")
    max_depth = (
        config.get("region", {}).get("max_depth_m")
        or config.get("max_depth_m")
        or config.get("identification", {}).get("max_depth_m")
    )
    if max_depth is not None:
        depth = depth[depth <= float(max_depth) + 1e-6]
    return depth


def _center_lines_from_points(points: pd.DataFrame) -> dict[int, pd.DataFrame]:
    return {
        int(object_id): part.sort_values("depth_index").copy()
        for object_id, part in points.groupby("eddy3d_object_id", sort=False)
    }


def _merge_accum(target: dict, source: dict) -> None:
    for key, item in source.items():
        if key not in target:
            target[key] = {
                name: value.copy() if isinstance(value, np.ndarray) else set(value)
                for name, value in item.items()
            }
            continue
        for name, value in item.items():
            if isinstance(value, np.ndarray):
                target[key][name] += value
            elif isinstance(value, set):
                target[key][name].update(value)


def _worker_chunk(
    task: ChunkTask,
    config: dict,
    climatology_path: str,
    n2_profile_path: str,
    tau_grid: np.ndarray,
    kernel_bandwidth: float,
    kernel_weight_min: float,
    rmax: float,
    radial_bins: int,
    azimuth_bins: int,
    field_cache_mode: str,
) -> dict:
    objects = task.objects.copy()
    if objects.empty:
        task.output_path.parent.mkdir(parents=True, exist_ok=True)
        with task.output_path.open("wb") as handle:
            pickle.dump({}, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return {"chunk_id": task.chunk_id, "objects": 0, "days": 0, "path": str(task.output_path)}

    center_lines = _center_lines_from_points(task.center_lines)
    radial, theta, rr, tt, _ = make_polar_grid(float(rmax), int(radial_bins), int(azimuth_bins))
    lat_ref = float(objects["surface_lat"].median()) if "surface_lat" in objects else 27.5
    f0 = 2.0 * OMEGA * np.sin(np.radians(lat_ref))
    bandwidth = float(kernel_bandwidth)
    weight_min = float(kernel_weight_min)
    years = sorted({pd.Timestamp(value).year for value in objects["date"].unique()})
    accum: dict = {}
    processed = 0
    skipped = 0

    def process_day(date_text: str, day_objects: pd.DataFrame, lon: np.ndarray, lat: np.ndarray, depth: np.ndarray, u_all: np.ndarray, v_all: np.ndarray) -> None:
        nonlocal processed, skipped
        n2 = load_n2(Path(n2_profile_path), depth)
        _, dy, dx = grid_spacing_m(lon, lat)
        u = sanitize_ocean_field(u_all.astype("f8", copy=False))
        v = sanitize_ocean_field(v_all.astype("f8", copy=False))
        u_clim, v_clim = read_climatology_uv(Path(climatology_path), str(date_text))
        zeta = relative_vorticity(lon, lat, u, v)
        psi_prime = streamfunction_from_zeta(zeta, dx, dy)
        for obj in day_objects.itertuples(index=False):
            center_line = center_lines.get(int(obj.eddy3d_object_id))
            if center_line is None:
                skipped += 1
                continue
            fields = sample_object_fields(
                obj,
                center_line,
                lon,
                lat,
                depth,
                psi_prime,
                u,
                v,
                u_clim,
                v_clim,
                None,
                None,
                rr,
                tt,
            )
            if fields is None:
                skipped += 1
                continue
            terms = compute_nondim_terms(
                fields,
                center_line,
                depth,
                radial,
                theta,
                float(obj.mean_radius_m),
                n2,
                f0,
                axis_mode="tilted",
            )
            psi = np.where(np.abs(fields["psi_prime"]) > 1e20, np.nan, fields["psi_prime"])
            terms["psi_mean"] = np.nanmean(psi, axis=2)
            terms["psi_variance"] = np.nanvar(psi, axis=2)
            weights = np.exp(-0.5 * ((tau_grid - float(obj.life_phase)) / max(bandwidth, 1e-12)) ** 2)
            for tau_index, weight in enumerate(weights):
                if weight < weight_min:
                    continue
                key = ("tilted", str(obj.polarity), int(tau_index))
                add_weighted_terms(
                    accum,
                    key,
                    terms,
                    float(weight),
                    int(obj.eddy3d_object_id),
                    int(obj.track3d_id),
                    str(obj.date),
                )
            processed += 1

    mode = str(field_cache_mode or "year").lower()
    if mode == "day":
        for date_text, day_objects in objects.groupby("date", sort=True):
            day = pd.Timestamp(date_text).date()
            day_data = read_day_data(config, day)
            process_day(
                str(date_text),
                day_objects,
                day_data["lon"],
                day_data["lat"],
                day_data["depth"],
                day_data["u_all"],
                day_data["v_all"],
            )
            del day_data
    elif mode == "year":
        paths_by_year = _source_path_by_year(config, years)
        for year in years:
            cache = load_year_cache(config, paths_by_year[year], year=year)
            year_objects = objects[pd.to_datetime(objects["date"]).dt.year == year].copy()
            for date_text, day_objects in year_objects.groupby("date", sort=True):
                day = pd.Timestamp(date_text).date()
                if day not in cache.day_to_time_index:
                    skipped += len(day_objects)
                    continue
                time_i = cache.day_to_time_index[day]
                process_day(
                    str(date_text),
                    day_objects,
                    cache.lon,
                    cache.lat,
                    cache.depth,
                    cache.u_all[time_i],
                    cache.v_all[time_i],
                )
            del cache
    else:
        raise ValueError(f"Unsupported field_cache_mode={field_cache_mode!r}; use 'year' or 'day'.")

    task.output_path.parent.mkdir(parents=True, exist_ok=True)
    with task.output_path.open("wb") as handle:
        pickle.dump(accum, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return {
        "chunk_id": task.chunk_id,
        "objects": processed,
        "skipped": skipped,
        "days": int(objects["date"].nunique()),
        "path": str(task.output_path),
    }


def make_chunk_tasks(
    objects: pd.DataFrame,
    center_points: pd.DataFrame,
    partial_dir: Path,
    chunk_days: int,
    resume: bool,
    runner_id: str,
) -> list[ChunkTask]:
    dates = sorted(objects["date"].unique())
    tasks: list[ChunkTask] = []
    for chunk_id, start_i in enumerate(range(0, len(dates), int(chunk_days))):
        chunk_dates = dates[start_i : start_i + int(chunk_days)]
        chunk_objects = objects[objects["date"].isin(chunk_dates)].copy()
        object_ids = set(chunk_objects["eddy3d_object_id"].astype("int64"))
        chunk_points = center_points[center_points["eddy3d_object_id"].astype("int64").isin(object_ids)].copy()
        start_tag = chunk_dates[0].replace("-", "")
        end_tag = chunk_dates[-1].replace("-", "")
        if runner_id:
            safe_runner = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in runner_id)
            out_path = partial_dir / f"chunk_{safe_runner}_{chunk_id:04d}_{start_tag}_{end_tag}.pkl"
        else:
            out_path = partial_dir / f"chunk_{chunk_id:04d}_{start_tag}_{end_tag}.pkl"
        if resume and out_path.exists():
            continue
        tasks.append(
            ChunkTask(
                chunk_id=chunk_id,
                start=str(chunk_dates[0]),
                end=str(chunk_dates[-1]),
                objects=chunk_objects,
                center_lines=chunk_points,
                output_path=out_path,
            )
        )
    return tasks


def merge_partial_accums(partial_dir: Path) -> dict:
    accum: dict = {}
    for path in sorted(partial_dir.glob("chunk_*.pkl")):
        with path.open("rb") as handle:
            part = pickle.load(handle)
        _merge_accum(accum, part)
    return accum


def write_outputs(
    output_dir: Path,
    objects: pd.DataFrame,
    tau_grid: np.ndarray,
    final: dict,
    radial: np.ndarray,
    depth: np.ndarray,
    radii: dict[str, float],
    summaries: list[dict],
    args: argparse.Namespace,
) -> None:
    manifest, counts = build_manifest(objects)
    write_table_fast(manifest, output_dir / "object_cache" / "selected_lifecycle_objects.parquet")
    manifest.to_csv(output_dir / "object_cache" / "selected_lifecycle_objects.csv", index=False)
    counts.to_csv(output_dir / "continuous_lifecycle_object_counts.csv", index=False)
    pd.DataFrame({"tau": tau_grid}).to_csv(output_dir / "continuous_tau_grid.csv", index=False)
    pd.DataFrame(
        {"polarity": list(radii.keys()), "representative_radius_m": list(radii.values())}
    ).to_csv(output_dir / "representative_radii.csv", index=False)
    pd.DataFrame.from_records(summaries).to_csv(output_dir / "partial_accum_parts" / "chunk_summary.csv", index=False)

    profiles = rows_from_continuous_final(final, radial, depth, tau_grid, radii)
    profiles = add_wave_activity(profiles, radii)
    write_table_fast(profiles, output_dir / "ep_flux_terms" / "continuous_ep_flux_profiles.parquet")
    profiles.to_csv(output_dir / "ep_flux_terms" / "continuous_ep_flux_profiles.csv", index=False)
    write_table_fast(profiles, output_dir / "wave_action_total_flux" / "continuous_wave_action_profiles.parquet")
    profiles.to_csv(output_dir / "wave_action_total_flux" / "continuous_wave_action_profiles.csv", index=False)

    psi_cols = [
        "polarity",
        "phase_index",
        "phase_name",
        "tau_center",
        "depth_index",
        "depth_m",
        "r_over_R",
        "psi_mean",
        "psi_variance",
        "count",
        "n_objects",
        "n_tracks",
        "n_dates",
    ]
    stream = profiles[[col for col in psi_cols if col in profiles.columns]].copy()
    write_table_fast(stream, output_dir / "streamfunction_templates" / "continuous_radial_psi_profiles.parquet")
    stream.to_csv(output_dir / "streamfunction_templates" / "continuous_radial_psi_profiles.csv", index=False)

    nonlinearity, nonlinearity_metrics, nonlinearity_summary = build_nonlinearity_tables(profiles, radii)
    core_dir = output_dir / "core_environment_sensitivity"
    write_table_fast(nonlinearity, core_dir / "continuous_nonlinearity_profiles.parquet")
    nonlinearity.to_csv(core_dir / "continuous_nonlinearity_profiles.csv", index=False)
    nonlinearity_metrics.to_csv(core_dir / "continuous_nonlinearity_metrics.csv", index=False)
    nonlinearity_summary.to_csv(core_dir / "continuous_nonlinearity_summary_by_polarity.csv", index=False)

    mlrw_profiles, mlrw_metrics = build_continuous_mlrw_profiles(nonlinearity)
    mlrw_dir = output_dir / "mlrw_applicability"
    write_table_fast(mlrw_profiles, mlrw_dir / "continuous_mlrw_applicability_profiles.parquet")
    mlrw_profiles.to_csv(mlrw_dir / "continuous_mlrw_applicability_profiles.csv", index=False)
    mlrw_metrics.to_csv(mlrw_dir / "continuous_mlrw_applicability_metrics.csv", index=False)

    _write_summary(
        output_dir / "summary.md",
        [
            "# Kuroshio representative vortex continuous tau summary",
            "",
            f"- Output root: `{output_dir}`",
            f"- Tau nodes: {len(tau_grid)}",
            f"- Tau grid: {','.join(f'{value:.4g}' for value in tau_grid)}",
            f"- Kernel bandwidth: {float(args.kernel_bandwidth):.4g}",
            f"- Radial bins: {int(args.radial_bins)}",
            f"- Azimuth bins: {int(args.azimuth_bins)}",
            f"- Objects selected: {int(manifest['eddy3d_object_id'].nunique()) if not manifest.empty else 0:,}",
            f"- Tracks selected: {int(manifest['track3d_id'].nunique()) if not manifest.empty and 'track3d_id' in manifest else 0:,}",
            f"- Partial chunks: {len(list((output_dir / 'partial_accum_parts').glob('chunk_*.pkl'))):,}",
            "- Velocity basis: raw annual CMEMS uo_glor/vo_glor; no input_daily files.",
        ],
    )
    _write_summary(
        output_dir / "continuous_diagnostics_summary.md",
        [
            "# Continuous tau diagnostics",
            "",
            f"- Profile rows: {len(profiles):,}",
            f"- Streamfunction rows: {len(stream):,}",
            f"- Nonlinearity rows: {len(nonlinearity):,}",
            f"- MLRW rows: {len(mlrw_profiles):,}",
            f"- Output root: `{output_dir}`",
        ],
    )


def run(args: argparse.Namespace) -> Path:
    if args.partial_only and args.finalize_only:
        raise ValueError("--partial-only and --finalize-only are mutually exclusive.")

    results_root = Path(args.results_root)
    output_dir = Path(args.output_dir)
    subdirs = _ensure_tree(output_dir)
    config = load_config(args.config)
    shapes = _split_csv(args.shapes, VALID_SHAPES)
    polarities = _split_csv(args.polarities, VALID_POLARITIES)
    tau_grid = parse_tau_grid(args.tau_grid, float(args.tau_grid_step))

    shape_dir_name = str(args.shape_dir_name)
    if args.axis_dir:
        axis_dir = Path(args.axis_dir)
        if not (axis_dir / "object_diagnostics.parquet").exists():
            raise FileNotFoundError(axis_dir / "object_diagnostics.parquet")
        if not (axis_dir / "rotated_points.parquet").exists():
            raise FileNotFoundError(axis_dir / "rotated_points.parquet")
    else:
        axis_dir = build_axis_products(
            results_root,
            output_dir,
            shapes,
            int(args.random_seed),
            shape_dir_name,
            str(args.axis_alignment),
        )
    objects = select_lifecycle_objects(
        results_root,
        axis_dir,
        shape_dir_name,
        shapes,
        polarities,
        int(args.max_days),
        int(args.max_objects_per_polarity),
        int(args.random_seed),
    )
    objects = _filter_objects_by_date(objects, str(args.date_start), str(args.date_end))
    if objects.empty:
        raise RuntimeError("No representative vortex objects selected.")
    radii = representative_radii(objects)
    center_points = read_table(axis_dir / "rotated_points.parquet")

    climatology_nc = Path(args.climatology_path) if args.climatology_path else (
        results_root / "climatology" / "cmems_doy_climatology_1993_2022_31d.nc"
    )
    n2_path = build_n2_profile(
        climatology_nc,
        subdirs["climatology"] / f"{climatology_nc.stem}_sigma0_dz_profile.npz",
        force=bool(args.force_n2),
    )

    dates = pd.to_datetime(objects["date"])
    source_index = build_source_index(config, dates.min().date(), dates.max().date())
    first_day = min(source_index)
    first_path = source_index[first_day].path
    radial, _, _, _, _ = make_polar_grid(float(args.rmax), int(args.radial_bins), int(args.azimuth_bins))
    depth = _read_depth_coordinate(config, first_path)

    tasks = []
    if not args.finalize_only:
        tasks = make_chunk_tasks(
            objects,
            center_points,
            subdirs["partial_accum_parts"],
            int(args.chunk_days),
            resume=bool(args.resume),
            runner_id=str(args.runner_id),
        )
    summaries: list[dict] = []
    if tasks:
        if int(args.workers) <= 1:
            for task in tqdm(tasks, total=len(tasks), desc="Representative chunks", unit="chunk"):
                summaries.append(
                    _worker_chunk(
                        task,
                        config,
                        str(climatology_nc),
                        str(n2_path),
                        tau_grid,
                        float(args.kernel_bandwidth),
                        float(args.kernel_weight_min),
                        float(args.rmax),
                        int(args.radial_bins),
                        int(args.azimuth_bins),
                        str(args.field_cache_mode),
                    )
                )
        else:
            with ProcessPoolExecutor(max_workers=int(args.workers)) as pool:
                futures = [
                    pool.submit(
                        _worker_chunk,
                        task,
                        config,
                        str(climatology_nc),
                        str(n2_path),
                        tau_grid,
                        float(args.kernel_bandwidth),
                        float(args.kernel_weight_min),
                        float(args.rmax),
                        int(args.radial_bins),
                        int(args.azimuth_bins),
                        str(args.field_cache_mode),
                    )
                    for task in tasks
                ]
                for future in tqdm(as_completed(futures), total=len(futures), desc="Representative chunks", unit="chunk"):
                    summaries.append(future.result())

    if summaries:
        summary_name = f"chunk_summary_{args.runner_id}.csv" if args.runner_id else "chunk_summary_current.csv"
        pd.DataFrame.from_records(summaries).to_csv(subdirs["partial_accum_parts"] / summary_name, index=False)

    if args.partial_only:
        return output_dir

    accum = merge_partial_accums(subdirs["partial_accum_parts"])
    if not accum:
        raise RuntimeError("No partial accumulators were produced.")
    final = finalize_weighted(accum)
    write_outputs(output_dir, objects, tau_grid, final, radial, depth, radii, summaries, args)
    return output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build Kuroshio continuous-tau representative vortex products without input_daily files.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--axis-dir", default="")
    parser.add_argument("--shape-dir-name", default="shape_classification_1993_2022")
    parser.add_argument("--axis-alignment", default="surface_to_deep")
    parser.add_argument("--field-cache-mode", default="year", choices=["year", "day"])
    parser.add_argument("--climatology-path", default="")
    parser.add_argument("--shapes", default=",".join(VALID_SHAPES))
    parser.add_argument("--polarities", default=",".join(VALID_POLARITIES))
    parser.add_argument("--tau-grid", default="")
    parser.add_argument("--tau-grid-step", type=float, default=0.05)
    parser.add_argument("--kernel-bandwidth", type=float, default=0.075)
    parser.add_argument("--kernel-weight-min", type=float, default=1e-3)
    parser.add_argument("--rmax", type=float, default=2.5)
    parser.add_argument("--radial-bins", type=int, default=40)
    parser.add_argument("--azimuth-bins", type=int, default=72)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--chunk-days", type=int, default=30)
    parser.add_argument("--date-start", default="")
    parser.add_argument("--date-end", default="")
    parser.add_argument("--runner-id", default="")
    parser.add_argument("--partial-only", action="store_true")
    parser.add_argument("--finalize-only", action="store_true")
    parser.add_argument("--max-days", type=int, default=0)
    parser.add_argument("--max-objects-per-polarity", type=int, default=0)
    parser.add_argument("--random-seed", type=int, default=20260713)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force-n2", action="store_true")
    return parser


def main() -> None:
    out = run(build_parser().parse_args())
    print(f"Representative vortex output: {out}")


if __name__ == "__main__":
    main()
