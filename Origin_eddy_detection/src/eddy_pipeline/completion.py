from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import shutil
import tempfile
from pathlib import Path

import netCDF4
import numpy as np
import pandas as pd
from tqdm import tqdm

from .utils.common import ensure_dirs, load_config
from .utils.streaming_cmems import is_streaming_source_configured, read_day_data
from .utils.table_io import read_table, table_exists, write_table
from .utils.velocity3d_core import (
    closed_contour_around_core,
    equivalent_circle_lonlat,
    make_contour_context,
    polygon_area_m2,
    pseudo_streamfunction,
    select_layer_center_speed_leading,
)


def completed_output_paths(config: dict) -> tuple[Path, Path]:
    root = Path(config["paths"]["catalog_dir"])
    return root / "layer_centers_completed.parquet", root / "layer_contours_completed.parquet"


def _uv_path(config: dict, day) -> Path:
    return Path(config["paths"]["input_daily_dir"]) / f"uv_{pd.Timestamp(day):%Y%m%d}.nc"


def _load_catalog(config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    root = Path(config["paths"]["catalog_dir"])
    objects = read_table(root / "vertical_objects.parquet")
    layers = read_table(root / "layer_observations.parquet")
    objects["date"] = pd.to_datetime(objects["date"])
    layers["date"] = pd.to_datetime(layers["date"])
    return objects, layers


def _merge_existing(path: Path, new_df: pd.DataFrame, track_ids: set[int], force: bool) -> pd.DataFrame:
    if not table_exists(path):
        return new_df
    old = read_table(path)
    if old.empty or "track3d_id" not in old.columns:
        return new_df
    if new_df.empty:
        return old
    if not force:
        existing_ids = set(old["track3d_id"].astype(int).unique())
        append_df = new_df[~new_df["track3d_id"].astype(int).isin(existing_ids)].copy()
        if append_df.empty:
            return old
        return pd.concat([old, append_df], ignore_index=True)
    old = old[~old["track3d_id"].astype(int).isin(track_ids)].copy()
    return pd.concat([old, new_df], ignore_index=True)


def _parse_track_ids(value: str | None) -> set[int]:
    if value is None or not str(value).strip():
        return set()
    out: set[int] = set()
    for part in str(value).replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        out.add(int(part))
    return out


def _read_track_list(path: str | Path | None) -> set[int]:
    if path is None:
        return set()
    df = read_table(path)
    for col in ("track3d_id", "track_id", "id"):
        if col in df.columns:
            return set(df[col].dropna().astype(int).tolist())
    raise ValueError(f"Track list must contain a track3d_id column: {path}")


def _parse_object_ids(value: str | None) -> set[int]:
    if value is None or not str(value).strip():
        return set()
    out: set[int] = set()
    for part in str(value).replace(";", ",").split(","):
        part = part.strip()
        if part:
            out.add(int(part))
    return out


def _fallback_radius_m(k: int, object_layers: pd.DataFrame, obj_row: pd.Series, search_radius_km: float) -> float:
    same = object_layers[object_layers["depth_index"].astype(int) == int(k)]
    if not same.empty:
        radius = float(same.iloc[0].radius_m)
        if np.isfinite(radius) and radius > 0:
            return radius
    radius = float(obj_row.mean_radius_m)
    if np.isfinite(radius) and radius > 0:
        return radius
    return max(10_000.0, search_radius_km * 500.0)


def _fallback_radius_with_source(k: int, object_layers: pd.DataFrame, obj_row: pd.Series, search_radius_km: float) -> tuple[float, str]:
    same = object_layers[object_layers["depth_index"].astype(int) == int(k)]
    if not same.empty:
        radius = float(same.iloc[0].radius_m)
        if np.isfinite(radius) and radius > 0:
            return radius, "detected_layer_radius"
    radius = float(obj_row.mean_radius_m)
    if np.isfinite(radius) and radius > 0:
        return radius, "object_mean_radius"
    return max(10_000.0, search_radius_km * 500.0), "search_radius_fallback"


def _load_day_uv(config: dict, day: pd.Timestamp) -> dict:
    if is_streaming_source_configured(config) and not bool(config.get("conversion", {}).get("use_input_daily", True)):
        return read_day_data(config, pd.Timestamp(day).date())
    uv_path = _uv_path(config, day)
    if not uv_path.exists():
        raise FileNotFoundError(f"Missing UV subset: {uv_path}")
    with netCDF4.Dataset(uv_path) as ds:
        return {
            "lon": np.asarray(ds.variables["longitude"][:], dtype="f8"),
            "lat": np.asarray(ds.variables["latitude"][:], dtype="f8"),
            "depth": np.asarray(ds.variables["depth"][:], dtype="f8"),
            "u_all": np.ma.filled(ds.variables["u"][:, :, :], np.nan).astype("f8", copy=False),
            "v_all": np.ma.filled(ds.variables["v"][:, :, :], np.nan).astype("f8", copy=False),
            "adt": np.ma.filled(ds.variables["adt"][:, :], np.nan).astype("f8", copy=False),
        }


def _speed_layer(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    speed = np.empty(u.shape, dtype="f8")
    np.hypot(u, v, out=speed)
    speed[~(np.isfinite(u) & np.isfinite(v))] = np.nan
    return speed


def _layer_contour_inputs(day_data: dict, k: int, cache: dict[int, tuple[np.ndarray, np.ndarray, object]]) -> tuple[np.ndarray, np.ndarray, object]:
    if k in cache:
        return cache[k]
    u = day_data["u_all"][k, :, :]
    v = day_data["v_all"][k, :, :]
    speed = _speed_layer(u, v)
    if k == 0:
        scalar = day_data["adt"]
    else:
        scalar = pseudo_streamfunction(day_data["lon"], day_data["lat"], u, v)
    context = make_contour_context(scalar, np.isfinite(scalar), speed)
    cache[k] = (scalar, speed, context)
    return cache[k]


def _complete_object(
    config: dict,
    obj_row: pd.Series,
    object_layers: pd.DataFrame,
    day_data: dict | None = None,
    contour_cache: dict[int, tuple[np.ndarray, np.ndarray, object]] | None = None,
    output_mode: str = "centers-and-contours",
) -> tuple[list[dict], list[dict]]:
    day = pd.Timestamp(obj_row.date)

    ccfg = config.get("center_completion", {})
    icfg = config.get("identification", {})
    radius_factor = float(ccfg.get("search_radius_factor", 1.5))
    min_radius_km = float(ccfg.get("min_search_radius_km", 80.0))
    max_radius_km = float(ccfg.get("max_search_radius_km", 250.0))
    base_radius_km = float(obj_row.mean_radius_m) / 1000.0 if np.isfinite(float(obj_row.mean_radius_m)) else min_radius_km
    search_radius_km = min(max(base_radius_km * radius_factor, min_radius_km), max_radius_km)

    if day_data is None:
        day_data = _load_day_uv(config, day)
    if contour_cache is None:
        contour_cache = {}
    lon = day_data["lon"]
    lat = day_data["lat"]
    depth = day_data["depth"]
    u_all = day_data["u_all"]
    v_all = day_data["v_all"]

    object_layers = object_layers.sort_values("depth_m").copy()
    by_depth = {
        int(row.depth_index): row
        for _, row in object_layers.sort_values(["depth_index", "radius_m"], ascending=[True, False]).drop_duplicates("depth_index").iterrows()
    }
    if by_depth:
        seed_index = min(by_depth)
        seed = by_depth[seed_index]
        centers = {
            seed_index: {
                "longitude": float(seed.longitude),
                "latitude": float(seed.latitude),
                "center_method": "detected_seed",
                "speed_at_core": float(seed.core_speed) if np.isfinite(seed.core_speed) else np.nan,
                "n_exact_roots": 0,
            }
        }
    else:
        seed_index = 0
        centers = {0: {"longitude": float(obj_row.longitude), "latitude": float(obj_row.latitude), "center_method": "object_seed", "speed_at_core": np.nan, "n_exact_roots": 0}}

    def fill_one(k: int, ref_lon: float, ref_lat: float) -> dict:
        selected = select_layer_center_speed_leading(
            lon,
            lat,
            u_all[k, :, :],
            v_all[k, :, :],
            ref_lon,
            ref_lat,
            search_radius_km,
            zero_point_method=ccfg.get("zero_point_method", "hybrid"),
            multi_root_policy=ccfg.get("multi_root_policy", "depth_continuity"),
        )
        if not np.isfinite(selected["longitude"]) or not np.isfinite(selected["latitude"]):
            selected.update(longitude=ref_lon, latitude=ref_lat, center_method="carry_forward")
        return selected

    ref_lon = centers[seed_index]["longitude"]
    ref_lat = centers[seed_index]["latitude"]
    for k in range(seed_index + 1, len(depth)):
        centers[k] = fill_one(k, ref_lon, ref_lat)
        ref_lon = centers[k]["longitude"]
        ref_lat = centers[k]["latitude"]

    ref_lon = centers[seed_index]["longitude"]
    ref_lat = centers[seed_index]["latitude"]
    for k in range(seed_index - 1, -1, -1):
        centers[k] = fill_one(k, ref_lon, ref_lat)
        ref_lon = centers[k]["longitude"]
        ref_lat = centers[k]["latitude"]

    center_rows: list[dict] = []
    contour_rows: list[dict] = []
    output_mode = str(output_mode).lower().strip()
    if output_mode not in {"centers-only", "centers-and-contours"}:
        raise ValueError(f"Unsupported output_mode: {output_mode}")
    surface_min, surface_max = [int(v) for v in icfg.get("surface_pixel_limit", [5, 2000])]
    min_pixels = int(icfg.get("min_closed_contour_pixels", 12))
    max_pixels = int(icfg.get("max_closed_contour_pixels", 20000))
    contour_levels = int(icfg.get("contour_levels", 16))

    for k in range(len(depth)):
        c = centers[k]
        detected = k in by_depth
        source_det = by_depth.get(k)
        if output_mode == "centers-only":
            radius_m, radius_source = _fallback_radius_with_source(k, object_layers, obj_row, search_radius_km)
            area = float(np.pi * radius_m * radius_m)
            contour_found = False
            contour_source = "not_generated_centers_only"
            contour_lon = np.array([], dtype="f4")
            contour_lat = np.array([], dtype="f4")
        else:
            scalar, speed, contour_context = _layer_contour_inputs(day_data, k, contour_cache)
            if k == 0:
                cmin = surface_min
                cmax = max(surface_max, max_pixels)
                contour_method = "sla_speed_contour_completed_core"
            else:
                cmin = min_pixels
                cmax = max_pixels
                contour_method = "streamfunction_speed_contour_completed_core"
            contour = closed_contour_around_core(
                lon,
                lat,
                scalar,
                float(c["longitude"]),
                float(c["latitude"]),
                min_pixels=cmin,
                max_pixels=cmax,
                contour_levels=contour_levels,
                local_radius_km=search_radius_km,
                speed=speed,
                selection_mode="max_speed",
                contour_context=contour_context,
            )
            contour_found = contour is not None
            if contour_found:
                contour_lon, contour_lat = contour
                area = polygon_area_m2(contour_lon, contour_lat)
                radius_m = float(np.sqrt(area / np.pi)) if area > 0 else np.nan
                contour_source = contour_method
                radius_source = "completed_contour_radius"
            else:
                radius_m, radius_source = _fallback_radius_with_source(k, object_layers, obj_row, search_radius_km)
                contour_lon, contour_lat = equivalent_circle_lonlat(float(c["longitude"]), float(c["latitude"]), radius_m)
                area = float(np.pi * radius_m * radius_m)
                contour_source = "equivalent_circle_fallback"

        center_rows.append(
            {
                "date": f"{day:%Y-%m-%d}",
                "track3d_id": int(obj_row.track3d_id),
                "eddy3d_object_id": int(obj_row.eddy3d_object_id),
                "depth_index": int(k),
                "depth_m": float(depth[k]),
                "longitude": float(c["longitude"]),
                "latitude": float(c["latitude"]),
                "center_method": str(c["center_method"]),
                "center_is_detected": bool(detected),
                "center_is_completed": bool(not detected),
                "source_layer_detection_id": int(source_det.layer_detection_id) if detected else -1,
                "speed_at_core": float(c.get("speed_at_core", np.nan)),
                "n_exact_roots": int(c.get("n_exact_roots", 0)),
                "search_radius_km": float(search_radius_km),
                "contour_found": bool(contour_found),
                "contour_source": contour_source,
                "radius_source": radius_source,
                "area_m2": float(area),
                "radius_m": float(radius_m),
            }
        )
        if output_mode == "centers-and-contours":
            for point_index, (x, y) in enumerate(zip(contour_lon, contour_lat)):
                contour_rows.append(
                    {
                        "date": f"{day:%Y-%m-%d}",
                        "track3d_id": int(obj_row.track3d_id),
                        "eddy3d_object_id": int(obj_row.eddy3d_object_id),
                        "depth_index": int(k),
                        "depth_m": float(depth[k]),
                        "point_index": int(point_index),
                        "longitude": float(x),
                        "latitude": float(y),
                        "contour_source": contour_source,
                        "contour_found": bool(contour_found),
                    }
                )
    return center_rows, contour_rows


def _complete_day_worker(
    config_path: str,
    day_label: str,
    object_records: list[dict],
    layer_records: list[dict],
    output_mode: str,
) -> tuple[str, list[dict], list[dict]]:
    config = load_config(config_path)
    day_data = _load_day_uv(config, pd.Timestamp(day_label))
    layers = pd.DataFrame.from_records(layer_records)
    layers_by_object = {
        int(k): g.copy()
        for k, g in layers.groupby("eddy3d_object_id")
    } if not layers.empty else {}
    contour_cache: dict[int, tuple[np.ndarray, np.ndarray, object]] = {}
    center_rows: list[dict] = []
    contour_rows: list[dict] = []
    for record in object_records:
        obj_row = pd.Series(record)
        object_layers = layers_by_object.get(int(obj_row.eddy3d_object_id), pd.DataFrame())
        rows, contours = _complete_object(
            config,
            obj_row,
            object_layers,
            day_data=day_data,
            contour_cache=contour_cache,
            output_mode=output_mode,
        )
        center_rows.extend(rows)
        contour_rows.extend(contours)
    return day_label, center_rows, contour_rows


def complete_centers(
    config_path: str | Path,
    track3d_id: int | None = None,
    track_ids: set[int] | None = None,
    object_ids: set[int] | None = None,
    track_list: str | Path | None = None,
    all_tracks: bool = False,
    force: bool = False,
    output_mode: str = "centers-and-contours",
    workers: int = 1,
) -> None:
    config = load_config(config_path)
    ensure_dirs(config)
    output_mode = str(output_mode).lower().strip()
    if output_mode not in {"centers-only", "centers-and-contours"}:
        raise ValueError("--output-mode must be centers-only or centers-and-contours")
    workers = max(1, int(workers))
    objects, layers = _load_catalog(config)
    selected_ids: set[int] = set()
    if track3d_id is not None:
        selected_ids.add(int(track3d_id))
    if track_ids:
        selected_ids.update(int(v) for v in track_ids)
    object_ids = set(int(v) for v in object_ids) if object_ids else set()
    selected_ids.update(_read_track_list(track_list))
    if not selected_ids and not object_ids and not all_tracks:
        raise ValueError("Provide --track3d-id, --track-ids, --object-ids, --track-list, or --all-tracks.")
    if track3d_id is not None:
        objects = objects[objects["track3d_id"].astype(int).isin(selected_ids)].copy()
    elif selected_ids:
        objects = objects[objects["track3d_id"].astype(int).isin(selected_ids)].copy()
    if object_ids:
        objects = objects[objects["eddy3d_object_id"].astype(int).isin(object_ids)].copy()
    if objects.empty:
        raise ValueError("No objects matched requested track/object IDs.")

    centers_out, contours_out = completed_output_paths(config)
    requested_track_ids = set(objects["track3d_id"].astype(int).unique())
    if table_exists(centers_out) and not force:
        old_centers = read_table(centers_out)
        if object_ids and "eddy3d_object_id" in old_centers.columns:
            done_objects = set(old_centers["eddy3d_object_id"].dropna().astype(int).unique())
            remaining_objects = set(objects["eddy3d_object_id"].astype(int).unique()) - done_objects
            if not remaining_objects:
                print(f"All requested objects already have completed centers: {len(object_ids)}")
                return
            objects = objects[objects["eddy3d_object_id"].astype(int).isin(remaining_objects)].copy()
            requested_track_ids = set(objects["track3d_id"].astype(int).unique())
            print(f"Skipping existing completed objects; remaining objects: {len(remaining_objects)}")
        elif "track3d_id" in old_centers.columns:
            done_ids = set(old_centers["track3d_id"].dropna().astype(int).unique())
            remaining_ids = requested_track_ids - done_ids
            if not remaining_ids:
                print(f"All requested tracks already have completed centers: {len(requested_track_ids)}")
                return
            objects = objects[objects["track3d_id"].astype(int).isin(remaining_ids)].copy()
            requested_track_ids = remaining_ids
            print(f"Skipping existing completed tracks; remaining tracks: {len(requested_track_ids)}")

    part_dir = Path(tempfile.mkdtemp(prefix="completed_parts_", dir=str(Path(config["paths"]["catalog_dir"]))))
    center_parts: list[Path] = []
    contour_parts: list[Path] = []
    part_index = 0

    def write_part(center_rows: list[dict], contour_rows: list[dict]) -> None:
        nonlocal part_index
        if not center_rows and not contour_rows:
            return
        center_part = part_dir / f"centers_part_{part_index:05d}.parquet"
        contour_part = part_dir / f"contours_part_{part_index:05d}.parquet"
        write_table(pd.DataFrame.from_records(center_rows), center_part, index=False)
        if output_mode == "centers-and-contours":
            write_table(pd.DataFrame.from_records(contour_rows), contour_part, index=False)
            contour_parts.append(contour_part)
        center_parts.append(center_part)
        part_index += 1

    sorted_objects = objects.sort_values(["date", "track3d_id", "eddy3d_object_id"]).copy()
    selected_object_ids = set(sorted_objects["eddy3d_object_id"].astype(int))
    selected_layers = layers[layers["eddy3d_object_id"].astype(int).isin(selected_object_ids)].copy()
    day_jobs = []
    layers_by_day = {pd.Timestamp(k): g.copy() for k, g in selected_layers.groupby("date")}
    for day, day_objects in sorted_objects.groupby("date", sort=True):
        day_ts = pd.Timestamp(day)
        day_jobs.append((f"{day_ts:%Y-%m-%d}", day_objects.to_dict("records"), layers_by_day.get(day_ts, pd.DataFrame()).to_dict("records")))
    try:
        with tqdm(total=len(sorted_objects), desc=f"Complete speed-leading centers ({output_mode}, {workers} workers)", unit="object") as bar:
            if workers <= 1 or len(day_jobs) <= 1:
                for day_label, object_records, layer_records in day_jobs:
                    _, rows, contours = _complete_day_worker(str(config_path), day_label, object_records, layer_records, output_mode)
                    write_part(rows, contours)
                    bar.update(len(object_records))
            else:
                with ProcessPoolExecutor(max_workers=workers) as executor:
                    futures = {
                        executor.submit(_complete_day_worker, str(config_path), day_label, object_records, layer_records, output_mode): len(object_records)
                        for day_label, object_records, layer_records in day_jobs
                    }
                    for future in as_completed(futures):
                        count = futures[future]
                        _, rows, contours = future.result()
                        write_part(rows, contours)
                        bar.update(count)

        if center_parts:
            new_centers = pd.concat((read_table(path) for path in center_parts), ignore_index=True)
            if output_mode == "centers-and-contours" and contour_parts:
                new_contours = pd.concat((read_table(path) for path in contour_parts), ignore_index=True)
            else:
                new_contours = pd.DataFrame()
        else:
            new_centers = pd.DataFrame()
            new_contours = pd.DataFrame()
    finally:
        shutil.rmtree(part_dir, ignore_errors=True)

    track_ids = set(new_centers["track3d_id"].astype(int).unique()) if not new_centers.empty else set()
    centers_df = _merge_existing(centers_out, new_centers, track_ids, force)
    write_table(centers_df, centers_out, index=False)
    if output_mode == "centers-and-contours":
        contours_df = _merge_existing(contours_out, new_contours, track_ids, force)
        write_table(contours_df, contours_out, index=False)
    print(f"Completed centers: {centers_out}")
    if output_mode == "centers-and-contours":
        print(f"Completed contours: {contours_out}")
    else:
        print("Completed contours: skipped (centers-only)")
    print(f"New center rows: {len(new_centers)}, new contour rows: {len(new_contours)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Complete every depth layer center using speed_leading and derive contours around completed cores.")
    parser.add_argument("--config", default="config/config_3d.yaml")
    parser.add_argument("--track3d-id", type=int)
    parser.add_argument("--track-ids", help="Comma-separated track3d_id values.")
    parser.add_argument("--object-ids", help="Comma-separated eddy3d_object_id values.")
    parser.add_argument("--track-list", help="CSV/parquet/pkl table with a track3d_id column.")
    parser.add_argument("--all-tracks", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--output-mode", choices=["centers-only", "centers-and-contours"], default="centers-and-contours")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    complete_centers(
        args.config,
        track3d_id=args.track3d_id,
        track_ids=_parse_track_ids(args.track_ids),
        object_ids=_parse_object_ids(args.object_ids),
        track_list=args.track_list,
        all_tracks=args.all_tracks,
        force=args.force,
        output_mode=args.output_mode,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
