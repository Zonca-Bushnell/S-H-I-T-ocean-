from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .classify_3d_eddy_shape import (
    RULE_VERSION,
    SHAPE_CLASSES,
    _build_daily_metrics,
    _eligible_tracks,
    _track_metrics,
)
from .hua_b3_defaults import (
    completion_output_mode,
    default_shape_output_name,
    hua_method_name,
    strict_contiguous_passed_layers,
)
from .table_io import write_table_fast


def _read(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_parquet(path, engine="fastparquet")


def _make_object_map(frame_objects: pd.DataFrame) -> pd.DataFrame:
    objects = frame_objects[["hua_object_id", "date"]].drop_duplicates().copy()
    objects = objects.sort_values(["date", "hua_object_id"]).reset_index(drop=True)
    objects["eddy3d_object_id"] = np.arange(objects.shape[0], dtype=np.int64)
    return objects


def _build_catalog(detection_dir: Path, tracking_dir: Path, catalog_dir: Path, *, strict_contiguous: bool = True) -> dict[str, int]:
    catalog_dir.mkdir(parents=True, exist_ok=True)
    centers = _read(detection_dir / "centers_hua_style.parquet")
    structures = _read(detection_dir / "structures_hua_style.parquet")
    frame_objects = _read(tracking_dir / "feature_track_points.parquet")
    feature_tracks = _read(tracking_dir / "feature_tracks.parquet")

    if centers.empty or frame_objects.empty or feature_tracks.empty:
        raise RuntimeError("Hua detection/tracking outputs are empty; cannot build catalog.")

    obj_map = _make_object_map(frame_objects)
    frame_objects = frame_objects.merge(obj_map, on=["hua_object_id", "date"], how="inner")
    frame_objects = frame_objects.rename(columns={"feature_track_id": "track3d_id"})
    centers = centers.merge(obj_map[["hua_object_id", "eddy3d_object_id"]], on="hua_object_id", how="inner")

    if strict_contiguous:
        passed, strict_stats = strict_contiguous_passed_layers(centers)
    else:
        passed = centers[centers["hua_pass"].astype(bool)].copy()
        strict_stats = {
            "strict_contiguous": 0,
            "passed_rows_before_strict": int(len(passed)),
            "passed_rows_after_strict": int(len(passed)),
            "isolated_pass_rows_removed": 0,
            "objects_before_strict": int(passed[["date", "hua_object_id"]].drop_duplicates().shape[0]) if not passed.empty else 0,
            "objects_after_strict": int(passed[["date", "hua_object_id"]].drop_duplicates().shape[0]) if not passed.empty else 0,
            "objects_dropped_no_surface": 0,
        }
    if passed.empty:
        raise RuntimeError("No Hua-passed layer centers are available.")

    radius_key = ["date", "hua_object_id", "depth_index"]
    if not structures.empty and set(radius_key + ["radius_km"]).issubset(structures.columns):
        passed = passed.merge(
            structures[radius_key + ["radius_km"]],
            on=radius_key,
            how="left",
        )
    else:
        passed["radius_km"] = np.nan
    fallback_radius_m = passed["accepted_radius_cells"].astype(float) * 22_000.0
    passed["radius_m"] = passed["radius_km"].astype(float) * 1000.0
    bad_radius = ~np.isfinite(passed["radius_m"]) | (passed["radius_m"] <= 0)
    passed.loc[bad_radius, "radius_m"] = fallback_radius_m.loc[bad_radius]
    passed["area_m2"] = np.pi * passed["radius_m"].astype(float) ** 2

    layer_observations = pd.DataFrame(
        {
            "layer_detection_id": np.arange(passed.shape[0], dtype=np.int64),
            "date": passed["date"].astype(str),
            "depth_m": passed["depth_m"].astype(float),
            "depth_index": passed["depth_index"].astype(np.int16),
            "polarity": passed["polarity"].astype(str),
            "longitude": passed["center_lon"].astype(float),
            "latitude": passed["center_lat"].astype(float),
            "core_speed": passed["center_speed_ms"].astype(float),
            "vorticity": np.nan,
            "area_m2": passed["area_m2"].astype(float),
            "radius_m": passed["radius_m"].astype(float),
            "method": hua_method_name(strict_contiguous=strict_contiguous),
            "reversal_passed": passed["opposite_reversal_fraction"].astype(float) >= 0.55,
            "eddy3d_object_id": passed["eddy3d_object_id"].astype(np.int64),
        }
    ).sort_values(["date", "eddy3d_object_id", "depth_index"])

    centers_completed = layer_observations.copy()
    track_lookup = frame_objects.set_index("eddy3d_object_id")["track3d_id"].astype(np.int64)
    centers_completed["track3d_id"] = centers_completed["eddy3d_object_id"].map(track_lookup).astype(np.int64)
    centers_completed = centers_completed[
        [
            "date",
            "track3d_id",
            "eddy3d_object_id",
            "depth_index",
            "depth_m",
            "longitude",
            "latitude",
            "radius_m",
            "polarity",
        ]
    ].sort_values(["date", "track3d_id", "eddy3d_object_id", "depth_index"])

    object_stats = (
        centers_completed.groupby("eddy3d_object_id", as_index=False)
        .agg(
            date=("date", "first"),
            track3d_id=("track3d_id", "first"),
            polarity=("polarity", "first"),
            longitude=("longitude", "first"),
            latitude=("latitude", "first"),
            layer_count=("depth_index", "size"),
            min_depth_m=("depth_m", "min"),
            max_depth_m=("depth_m", "max"),
            mean_radius_m=("radius_m", "mean"),
        )
        .sort_values(["date", "track3d_id", "eddy3d_object_id"])
    )

    track_radius = object_stats.groupby("track3d_id")["mean_radius_m"].mean().rename("mean_radius_from_hua")
    track_layers = object_stats.groupby("track3d_id")["layer_count"].max().rename("max_layer_count_from_hua")
    tracks_3d = feature_tracks.rename(columns={"feature_track_id": "track3d_id"}).copy()
    tracks_3d["track3d_id"] = tracks_3d["track3d_id"].astype(np.int64)
    tracks_3d = tracks_3d.merge(track_radius, on="track3d_id", how="left")
    tracks_3d = tracks_3d.merge(track_layers, on="track3d_id", how="left")
    tracks_3d["mean_radius_m"] = tracks_3d["mean_radius_from_hua"].fillna(50_000.0).astype(float)
    tracks_3d["max_layer_count"] = tracks_3d["max_layer_count_from_hua"].fillna(0).astype(int)
    tracks_3d["observation_count"] = tracks_3d["n_objects"].astype(int)
    tracks_3d = tracks_3d[
        [
            "track3d_id",
            "polarity",
            "start_date",
            "end_date",
            "duration_days",
            "observation_count",
            "max_layer_count",
            "mean_radius_m",
        ]
    ].rename(columns={"duration_days": "lifetime_days"})
    tracks_3d = tracks_3d.sort_values("track3d_id")

    write_table_fast(layer_observations, catalog_dir / "layer_observations.parquet")
    write_table_fast(object_stats, catalog_dir / "vertical_objects.parquet")
    write_table_fast(tracks_3d, catalog_dir / "tracks_3d.parquet")
    write_table_fast(centers_completed, catalog_dir / "layer_centers_completed.parquet")
    tracks_3d.to_csv(catalog_dir / "tracks_3d.csv", index=False)

    return {
        "layer_observations": int(layer_observations.shape[0]),
        "vertical_objects": int(object_stats.shape[0]),
        "tracks_3d": int(tracks_3d.shape[0]),
        "completed_center_rows": int(centers_completed.shape[0]),
        **strict_stats,
    }


def _classify_partial_hua_centers(
    catalog_dir: Path,
    output_dir: Path,
    output_name: str,
    *,
    start: str,
    end: str,
    lifetime_min_days: int,
    radius_min_m: float,
    min_valid_layers: int,
    strict_contiguous: bool,
) -> dict[str, int]:
    shape_dir = output_dir / output_name
    shape_dir.mkdir(parents=True, exist_ok=True)
    tracks = _read(catalog_dir / "tracks_3d.parquet")
    objects = _read(catalog_dir / "vertical_objects.parquet")
    centers = _read(catalog_dir / "layer_centers_completed.parquet")
    objects["date"] = pd.to_datetime(objects["date"])
    centers["date"] = pd.to_datetime(centers["date"])
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    objects = objects[(objects["date"] >= start_ts) & (objects["date"] <= end_ts)].copy()
    centers = centers[(centers["date"] >= start_ts) & (centers["date"] <= end_ts)].copy()

    eligible = _eligible_tracks(tracks, lifetime_min_days=lifetime_min_days, radius_min_m=radius_min_m)
    eligible_objects = objects[objects["track3d_id"].astype(int).isin(set(eligible["track3d_id"].astype(int)))].copy()
    eligible_ids = set(eligible_objects["track3d_id"].astype(int))
    eligible = eligible[eligible["track3d_id"].astype(int).isin(eligible_ids)].copy()
    if eligible.empty:
        raise RuntimeError("No Hua tracks passed shape eligibility.")

    daily = _build_daily_metrics(eligible, eligible_objects, centers, min_valid_layers=min_valid_layers)
    valid_s = daily["S_rms"].astype(float)
    valid_s = valid_s[np.isfinite(valid_s)]
    if valid_s.size >= 10:
        upright_threshold = float(np.nanquantile(valid_s, 0.2))
        upright_mode = "eligible_daily_quantile"
    else:
        upright_threshold = 0.12
        upright_mode = "fixed_fallback"
    rule_opts = {
        "mono_coherent_threshold": 0.72,
        "dir_mean_coherent_max_deg": 35.0,
        "dir_max_complex_min_deg": 100.0,
        "mono_complex_max": 0.55,
    }
    shape_tracks = _track_metrics(daily, eligible, upright_threshold, rule_opts)

    write_table_fast(eligible, shape_dir / "eligible_tracks.parquet")
    write_table_fast(daily, shape_dir / "shape_daily_metrics.parquet")
    write_table_fast(shape_tracks, shape_dir / "shape_tracks.parquet")
    shape_tracks.to_csv(shape_dir / "shape_tracks.csv", index=False)
    by_shape = shape_dir / "by_shape"
    by_shape.mkdir(parents=True, exist_ok=True)
    for shape in SHAPE_CLASSES:
        folder = by_shape / shape
        folder.mkdir(parents=True, exist_ok=True)
        shape_tracks[shape_tracks["shape_class"] == shape].to_csv(folder / "tracks.csv", index=False)

    thresholds = {
        "rule_version": RULE_VERSION,
        "catalog_source": f"{hua_method_name(strict_contiguous=strict_contiguous)}_centers",
        "date_window": {"start": start, "end": end, "output_name": output_name},
        "eligibility": {
            "lifetime_days_cmp": ">",
            "lifetime_days_min": int(lifetime_min_days),
            "mean_radius_m_cmp": ">",
            "mean_radius_m_min": float(radius_min_m),
        },
        "shape_classes": SHAPE_CLASSES,
        "upright_mode": upright_mode,
        "upright_s_threshold_used": float(upright_threshold),
        "upright_s_quantile": 0.2,
        "min_valid_layers": int(min_valid_layers),
        "completion_output_mode": completion_output_mode(strict_contiguous=strict_contiguous),
        "mono_coherent_threshold": 0.72,
        "dir_mean_coherent_max_deg": 35.0,
        "dir_max_complex_min_deg": 100.0,
        "mono_complex_max": 0.55,
        "n_eligible_tracks": int(eligible.shape[0]),
        "n_daily_metric_rows": int(daily.shape[0]),
        "shape_counts": {str(k): int(v) for k, v in shape_tracks["shape_class"].value_counts().sort_index().items()},
    }
    (shape_dir / "shape_thresholds.yaml").write_text(yaml.safe_dump(thresholds, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return {
        "eligible_tracks": int(eligible.shape[0]),
        "daily_metrics": int(daily.shape[0]),
        "shape_tracks": int(shape_tracks.shape[0]),
        **{f"shape_{k}": int(v) for k, v in shape_tracks["shape_class"].value_counts().sort_index().items()},
    }


def run(args: argparse.Namespace) -> None:
    detection_dir = Path(args.detection_dir)
    tracking_dir = Path(args.tracking_dir)
    output_root = Path(args.output_root)
    catalog_dir = output_root / "catalog"
    strict_contiguous = not bool(args.allow_noncontiguous_depth)
    summary = {
        "catalog": _build_catalog(detection_dir, tracking_dir, catalog_dir, strict_contiguous=strict_contiguous),
        "shape": _classify_partial_hua_centers(
            catalog_dir,
            output_root,
            args.shape_output_name,
            start=args.start,
            end=args.end,
            lifetime_min_days=int(args.lifetime_min_days),
            radius_min_m=float(args.radius_min_m),
            min_valid_layers=int(args.min_valid_layers),
            strict_contiguous=strict_contiguous,
        ),
    }
    (output_root / "hua_b3_catalog_adapter_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Hua/Nencioli b3_start2 detections to slim catalog and shape outputs.")
    parser.add_argument("--detection-dir", required=True)
    parser.add_argument("--tracking-dir", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--shape-output-name", default="")
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2022-12-31")
    parser.add_argument("--lifetime-min-days", type=int, default=56)
    parser.add_argument("--radius-min-m", type=float, default=50_000.0)
    parser.add_argument("--min-valid-layers", type=int, default=6)
    parser.add_argument(
        "--allow-noncontiguous-depth",
        action="store_true",
        help="Legacy/diagnostic mode: keep all Hua-passed layers instead of only the surface-connected contiguous segment.",
    )
    args = parser.parse_args()
    if not args.shape_output_name:
        args.shape_output_name = default_shape_output_name(args.start, args.end, args.lifetime_min_days)
    run(args)


if __name__ == "__main__":
    main()
