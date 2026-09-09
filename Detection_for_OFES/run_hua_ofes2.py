from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage

from .ofes_io import (
    CtlMetadata,
    archive_candidates,
    ctl_path,
    data_path,
    expected_dta_bytes,
    open_dta_memmap,
    parse_ctl,
    prefetch_daily_files,
    read_ssh_latlon_daily_only,
    require_daily_file,
)


EARTH_RADIUS_M = 6_371_000.0
DEFAULT_DATA_ROOT = Path(r"F:\OFES\external_OFES2")
DEFAULT_OUTPUT_ROOT = Path(r"E:\DATA\01_Eddy_correspond\02_OFES")
SCIENCE_TAG = "raw_minus_jan_mean_diagnostic"


@dataclass(frozen=True)
class HuaParams:
    ssh_window_cells: int = 7
    max_candidates_per_day: int = 80
    candidate_selection: str = "global_topn"
    tile_lon_deg: float = 10.0
    tile_lat_deg: float = 10.0
    tile_top_n: int = 10
    surface_search_cells: int = 8
    deep_search_cells: int = 6
    start_radius_cells: int = 3
    max_radius_cells: int = 8
    speed_ratio_max: float = 3.0
    angle_jump_max_deg: float = 150.0
    tangent_tolerance_deg: float = 24.0
    symmetry_tolerance_deg: float = 120.0
    min_tangent_fraction: float = 0.70
    min_reversal_fraction: float = 0.70
    min_finite_fraction: float = 0.95
    boundary_monotonic_exception_limit: int = 0
    subgrid_target_degree: float = 1.0 / 24.0
    subgrid_window_radius_cells: int = 1
    subgrid_min_finite_fraction: float = 0.50


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    output_root = Path(args.output_root)
    paths = output_paths(output_root)
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)

    params = HuaParams(
        ssh_window_cells=args.ssh_window_cells,
        max_candidates_per_day=args.max_candidates_per_day,
        candidate_selection=args.candidate_selection,
        tile_lon_deg=args.tile_lon_deg,
        tile_lat_deg=args.tile_lat_deg,
        tile_top_n=args.tile_top_n,
        surface_search_cells=args.surface_search_cells,
        deep_search_cells=args.deep_search_cells,
        start_radius_cells=args.start_radius_cells,
        max_radius_cells=args.max_radius_cells,
        subgrid_target_degree=0.0 if args.disable_subgrid_center_refinement else args.subgrid_target_degree,
        subgrid_window_radius_cells=args.subgrid_window_radius_cells,
        subgrid_min_finite_fraction=args.subgrid_min_finite_fraction,
    )
    days = date_range(parse_iso_date(args.start), parse_iso_date(args.end))
    stages = [item.strip() for item in args.stages.split(",") if item.strip()]

    write_run_config(args, params, days, paths)
    if "metadata" in stages:
        write_metadata(Path(args.data_root), paths["metadata"], days)
    if "extract" in stages:
        extract_daily_uv(Path(args.data_root), paths["metadata"], days, int(args.extract_workers))
    if "means" in stages:
        build_monthly_means(
            Path(args.data_root),
            paths["anomaly"],
            paths["cache"],
            days,
            int(args.max_depth_layers),
            int(args.mean_depth_chunk),
            str(args.mean_strategy),
            int(args.extract_workers),
            int(args.mean_io_chunk_mb),
            bool(args.force),
        )
    if "detect" in stages:
        detect_month(Path(args.data_root), paths, days, params, int(args.max_depth_layers), bool(args.resume), args.mean_cache_root)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run OFES2 Hua hybrid detection on Jan monthly anomaly fields.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--start", default="1991-01-01")
    parser.add_argument("--end", default="1991-01-31")
    parser.add_argument("--stages", default="metadata,extract,means,detect", help="Comma-separated stages: metadata,extract,means,detect.")
    parser.add_argument("--max-depth-layers", type=int, default=105)
    parser.add_argument(
        "--mean-depth-chunk",
        type=int,
        default=8,
        help="Legacy fallback chunk size. The default volume accumulator reads each daily u/v file only once.",
    )
    parser.add_argument(
        "--mean-strategy",
        choices=["volume", "depth_chunk"],
        default="volume",
        help="Use volume for one-pass daily scans; use depth_chunk only on low-memory machines.",
    )
    parser.add_argument(
        "--extract-workers",
        type=int,
        default=4,
        help="Parallel archive workers for prefetching cached daily files. Independent .tgz files are scanned once each.",
    )
    parser.add_argument(
        "--mean-io-chunk-mb",
        type=int,
        default=256,
        help="Flat sequential chunk size for volume-mean accumulation, in MiB of float32 values.",
    )
    parser.add_argument("--ssh-window-cells", type=int, default=7)
    parser.add_argument("--max-candidates-per-day", type=int, default=80)
    parser.add_argument("--candidate-selection", choices=["global_topn", "tile_topn"], default="global_topn")
    parser.add_argument("--tile-lon-deg", type=float, default=10.0)
    parser.add_argument("--tile-lat-deg", type=float, default=10.0)
    parser.add_argument("--tile-top-n", type=int, default=10)
    parser.add_argument(
        "--mean-cache-root",
        type=Path,
        default=None,
        help="Optional root containing anomaly_jan1991. If omitted, the runner uses the output root, then the baseline refined OFES-grid cache when present.",
    )
    parser.add_argument("--surface-search-cells", type=int, default=8)
    parser.add_argument("--deep-search-cells", type=int, default=6)
    parser.add_argument("--start-radius-cells", type=int, default=3)
    parser.add_argument("--max-radius-cells", type=int, default=8)
    parser.add_argument("--disable-subgrid-center-refinement", action="store_true")
    parser.add_argument(
        "--subgrid-target-degree",
        type=float,
        default=0.025,
        help="OFES local refinement spacing in degrees. Default is one quarter of the 0.1 degree OFES grid.",
    )
    parser.add_argument(
        "--subgrid-window-radius-cells",
        type=int,
        default=2,
        help="OFES local refinement half-width in original grid cells.",
    )
    parser.add_argument("--subgrid-min-finite-fraction", type=float, default=0.50)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--force", action="store_true")
    return parser


def output_paths(output_root: Path) -> dict[str, Path]:
    return {
        "root": output_root,
        "metadata": output_root / "metadata",
        "anomaly": output_root / "anomaly_jan1991",
        "detection": output_root / "detection_hua_global_jan1991",
        "figures": output_root / "figures",
        "logs": output_root / "logs",
        "cache": output_root / "anomaly_jan1991" / "cache_daily_dta",
    }


def write_run_config(args: argparse.Namespace, params: HuaParams, days: list[date], paths: dict[str, Path]) -> None:
    cfg = {
        "science_tag": SCIENCE_TAG,
        "data_root": str(args.data_root),
        "output_root": str(args.output_root),
        "date_range": {"start": days[0].isoformat(), "end": days[-1].isoformat(), "n_days": len(days)},
        "domain": "global",
        "max_depth_layers": int(args.max_depth_layers),
        "mean_depth_chunk": int(args.mean_depth_chunk),
        "mean_strategy": str(args.mean_strategy),
        "extract_workers": int(args.extract_workers),
        "mean_io_chunk_mb": int(args.mean_io_chunk_mb),
        "mean_cache_root": str(args.mean_cache_root) if args.mean_cache_root else "",
        "method": "OFES2 Hua hybrid: SSH seed + velocity minimum + circular velocity checks + boundary monotonic + strict contiguous",
        "anomaly_policy": "Daily fields minus Jan 1991 monthly mean; not 30-180 day bandpass.",
        "parameters": asdict(params),
    }
    write_json(paths["metadata"] / "run_config.json", cfg)


def write_metadata(root: Path, output_dir: Path, days: list[date]) -> None:
    rows: list[dict[str, object]] = []
    for variable in ["eta", "pressur", "u", "v", "w", "temp", "salinity", "prho"]:
        meta = parse_ctl(ctl_path(root, variable))
        row = {
            "variable": variable,
            "ctl_path": str(meta.path),
            "source_name": meta.variable.name,
            "units": meta.variable.units,
            "description": meta.variable.description,
            "endian": meta.endian,
            "undef": meta.undef,
            **meta.grid_signature,
            "t_start": meta.t.start.isoformat(),
            "t_count": meta.t.count,
            "t_step_days": meta.t.step_days,
            "archive_count_1991": len(archive_candidates(root, variable, days[0])),
        }
        rows.append(row)
    write_csv(output_dir / "ctl_summary.csv", rows)
    write_json(output_dir / "ctl_summary.json", rows)


def extract_daily_uv(root: Path, metadata_dir: Path, days: list[date], extract_workers: int) -> None:
    rows: list[dict[str, object]] = []
    for variable in ["u", "v"]:
        meta = parse_ctl(ctl_path(root, variable))
        expected_bytes = expected_dta_bytes(meta)
        try:
            prefetch_daily_files(root, variable, days, cache_root=None, workers=max(1, int(extract_workers)))
        except FileNotFoundError as exc:
            print(f"[extract-warning] {variable} incomplete: {exc}", flush=True)
        for day in days:
            path = data_path(root, variable, day)
            archives = archive_candidates(root, variable, day)
            complete = path.exists() and path.stat().st_size == expected_bytes
            rows.append(
                {
                    "variable": variable,
                    "date": day.isoformat(),
                    "source_archive": str(archives[0]) if archives else "",
                    "target_path": str(path),
                    "expected_bytes": expected_bytes,
                    "actual_bytes": path.stat().st_size if path.exists() else 0,
                    "complete": bool(complete),
                }
            )
            print(f"[extract-ready] {variable} {day} complete={complete} {path}", flush=True)
        if any(not bool(row["complete"]) for row in rows if row["variable"] == variable):
            missing_days = [str(row["date"]) for row in rows if row["variable"] == variable and not bool(row["complete"])]
            print(f"[extract-missing] {variable} {', '.join(missing_days)}", flush=True)
    write_csv(metadata_dir / "daily_uv_extract_manifest.csv", rows)
    write_json(metadata_dir / "daily_uv_extract_manifest.json", rows)
    missing = [row for row in rows if not bool(row["complete"])]
    if missing:
        raise FileNotFoundError(
            "Daily u/v extraction is incomplete. Missing or incomplete days: "
            + ", ".join(f"{row['variable']}:{row['date']}" for row in missing)
        )


def build_monthly_means(
    root: Path,
    anomaly_dir: Path,
    cache_dir: Path,
    days: list[date],
    max_depth_layers: int,
    mean_depth_chunk: int,
    mean_strategy: str,
    extract_workers: int,
    mean_io_chunk_mb: int,
    force: bool,
) -> None:
    ssh_mean_path = anomaly_dir / "ssh_jan1991_mean_cm.npy"
    if force or not ssh_mean_path.exists():
        acc = None
        count = None
        for day in days:
            ssh = read_ssh_latlon_daily_only(root, day)
            finite = np.isfinite(ssh)
            if acc is None:
                acc = np.zeros(ssh.shape, dtype=np.float64)
                count = np.zeros(ssh.shape, dtype=np.uint16)
            acc[finite] += ssh[finite]
            count[finite] += 1
        mean = np.full(acc.shape, np.nan, dtype=np.float32)
        ok = count > 0
        mean[ok] = (acc[ok] / count[ok]).astype(np.float32)
        np.save(ssh_mean_path, mean)
        write_preview(mean, anomaly_dir / "ssh_jan1991_mean_preview.png", -150.0, 150.0)

    for variable in ["u", "v"]:
        meta = parse_ctl(ctl_path(root, variable))
        nlev = min(int(max_depth_layers), int(meta.z.count))
        mean_path = anomaly_dir / f"{variable}_jan1991_mean_cms.dat"
        shape_path = anomaly_dir / f"{variable}_jan1991_mean_cms.json"
        done_path = anomaly_dir / f"{variable}_jan1991_mean_cms.done.json"
        if mean_path.exists() and shape_path.exists() and done_path.exists() and not force:
            continue
        expected_bytes = expected_dta_bytes(meta)
        daily_paths = [require_daily_file(root, variable, day, expected_bytes) for day in days]
        for day, daily_path in zip(days, daily_paths):
            print(f"[daily-ready] {variable} {day} {daily_path}", flush=True)
        mean_mm = np.memmap(mean_path, dtype="float32", mode="w+", shape=(meta.x.count, meta.y.count, nlev), order="F")
        if mean_strategy == "volume":
            accumulate_volume_mean(variable, days, daily_paths, meta, nlev, mean_mm, int(mean_io_chunk_mb))
        else:
            chunk = max(1, int(mean_depth_chunk))
            for level_start in range(0, nlev, chunk):
                level_end = min(nlev, level_start + chunk)
                acc = np.zeros((meta.x.count, meta.y.count, level_end - level_start), dtype=np.float32, order="F")
                count = np.zeros(acc.shape, dtype=np.uint8, order="F")
                for path in daily_paths:
                    raw = open_dta_memmap(path, meta)
                    raw_volume = raw[:, :, level_start:level_end]
                    valid = np.abs(raw_volume) <= 1.0e30
                    np.add(acc, raw_volume, out=acc, where=valid, casting="unsafe")
                    count += valid.astype(np.uint8)
                out = np.full(acc.shape, np.nan, dtype=np.float32, order="F")
                np.divide(acc, count, out=out, where=count > 0)
                mean_mm[:, :, level_start:level_end] = out
                mean_mm.flush()
                print(f"[mean-depth-chunk] {variable} levels {level_start + 1}-{level_end}/{nlev}", flush=True)
        write_json(
            shape_path,
            {
                "dtype": "float32",
                "units": "cm/s",
                "shape": [meta.x.count, meta.y.count, nlev],
                "layout": "ofes_native_fortran_xyz",
                "mean_strategy": mean_strategy,
                "accumulator_dtype": "float32",
                "count_dtype": "uint8",
                "daily_file_reads_per_variable": len(daily_paths),
                "archive_prefetch_workers": int(extract_workers),
                "archive_prefetch_policy": "not_used_in_means_daily_files_required",
                "mean_io_pattern": "flat_fortran_order_sequential_chunks",
                "mean_io_chunk_mb": int(mean_io_chunk_mb),
            },
        )
        write_json(done_path, {"status": "complete", "variable": variable, "n_days": len(days), "n_levels": nlev})


def accumulate_volume_mean(
    variable: str,
    days: list[date],
    daily_paths: list[Path],
    meta: CtlMetadata,
    nlev: int,
    mean_mm: np.memmap,
    chunk_mb: int,
) -> None:
    n_values = int(meta.x.count * meta.y.count * nlev)
    chunk_values = max(1_000_000, int(chunk_mb) * 1024 * 1024 // np.dtype("float32").itemsize)
    acc = np.zeros(n_values, dtype=np.float32)
    count = np.zeros(n_values, dtype=np.uint8)

    for day, path in zip(days, daily_paths):
        raw = open_dta_memmap(path, meta)
        raw_flat = raw.ravel(order="F")[:n_values]
        for start in range(0, n_values, chunk_values):
            end = min(n_values, start + chunk_values)
            block = raw_flat[start:end]
            valid = np.abs(block) <= 1.0e30
            np.add(acc[start:end], block, out=acc[start:end], where=valid, casting="unsafe")
            count[start:end] += valid.astype(np.uint8, copy=False)
        print(f"[mean-volume] {variable} added {day}", flush=True)

    out_flat = mean_mm.ravel(order="F")
    for start in range(0, n_values, chunk_values):
        end = min(n_values, start + chunk_values)
        out_block = out_flat[start:end]
        out_block[:] = np.nan
        np.divide(acc[start:end], count[start:end], out=out_block, where=count[start:end] > 0)
    mean_mm.flush()


def detect_month(root: Path, paths: dict[str, Path], days: list[date], params: HuaParams, max_depth_layers: int, resume: bool, mean_cache_root: Path | None) -> None:
    detection_dir = paths["detection"]
    parts_dir = detection_dir / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)

    eta_meta = parse_ctl(ctl_path(root, "eta"))
    u_meta = parse_ctl(ctl_path(root, "u"))
    v_meta = parse_ctl(ctl_path(root, "v"))
    lon_scalar = eta_meta.x.values
    lat_scalar = eta_meta.y.values
    lon_uv = u_meta.x.values
    lat_uv = u_meta.y.values
    nlev = min(max_depth_layers, u_meta.z.count, v_meta.z.count)
    depth = u_meta.z.values[:nlev]
    anomaly_dir = resolve_anomaly_dir(paths["root"], paths["anomaly"], mean_cache_root)
    ssh_mean = np.load(anomaly_dir / "ssh_jan1991_mean_cm.npy")
    u_mean = np.memmap(anomaly_dir / "u_jan1991_mean_cms.dat", dtype="float32", mode="r", shape=(u_meta.x.count, u_meta.y.count, nlev), order="F")
    v_mean = np.memmap(anomaly_dir / "v_jan1991_mean_cms.dat", dtype="float32", mode="r", shape=(v_meta.x.count, v_meta.y.count, nlev), order="F")

    all_centers: list[pd.DataFrame] = []
    all_circle: list[pd.DataFrame] = []
    all_structures: list[pd.DataFrame] = []
    for day in days:
        centers_path = parts_dir / f"centers_{day:%Y%m%d}.csv"
        circle_path = parts_dir / f"circle_{day:%Y%m%d}.csv"
        structures_path = parts_dir / f"structures_{day:%Y%m%d}.csv"
        if resume and centers_path.exists() and circle_path.exists() and structures_path.exists():
            all_centers.append(pd.read_csv(centers_path))
            all_circle.append(pd.read_csv(circle_path))
            all_structures.append(pd.read_csv(structures_path))
            continue
        centers, circle, structures = detect_day(
            root,
            paths["cache"],
            day,
            ssh_mean,
            u_mean,
            v_mean,
            lon_scalar,
            lat_scalar,
            lon_uv,
            lat_uv,
            depth,
            u_meta,
            v_meta,
            params,
        )
        centers.to_csv(centers_path, index=False)
        circle.to_csv(circle_path, index=False)
        structures.to_csv(structures_path, index=False)
        write_day_figure(centers, day, paths["figures"])
        all_centers.append(centers)
        all_circle.append(circle)
        all_structures.append(structures)
        print(f"[detect] {day} centers={len(centers)} pass={int(centers['hua_pass'].sum()) if not centers.empty else 0}", flush=True)

    centers = pd.concat(all_centers, ignore_index=True) if all_centers else pd.DataFrame()
    circle = pd.concat(all_circle, ignore_index=True) if all_circle else pd.DataFrame()
    structures = pd.concat(all_structures, ignore_index=True) if all_structures else pd.DataFrame()
    write_table(centers, detection_dir / "centers_hua_style")
    write_table(circle, detection_dir / "circle_check_diagnostics")
    write_table(structures, detection_dir / "structures_hua_style")
    write_detection_summary(detection_dir, centers, circle, structures, params, days, nlev)


def resolve_anomaly_dir(output_root: Path, default_anomaly_dir: Path, mean_cache_root: Path | None) -> Path:
    candidates = [default_anomaly_dir]
    if mean_cache_root is not None:
        explicit = mean_cache_root / "anomaly_jan1991" if (mean_cache_root / "anomaly_jan1991").exists() else mean_cache_root
        candidates.insert(0, explicit)
    baseline = output_root.parent / "available_jan01_jan19_refined_ofes_grid" / "anomaly_jan1991"
    if baseline not in candidates:
        candidates.append(baseline)
    required = ["ssh_jan1991_mean_cm.npy", "u_jan1991_mean_cms.dat", "v_jan1991_mean_cms.dat"]
    for candidate in candidates:
        if all((candidate / name).exists() for name in required):
            print(f"[mean-cache] using {candidate}", flush=True)
            return candidate
    raise FileNotFoundError("Missing Jan 1991 mean cache. Checked: " + ", ".join(str(path) for path in candidates))


def detect_day(
    root: Path,
    cache_dir: Path,
    day: date,
    ssh_mean: np.ndarray,
    u_mean: np.ndarray,
    v_mean: np.ndarray,
    lon_scalar: np.ndarray,
    lat_scalar: np.ndarray,
    lon_uv: np.ndarray,
    lat_uv: np.ndarray,
    depth: np.ndarray,
    u_meta: CtlMetadata,
    v_meta: CtlMetadata,
    params: HuaParams,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ssh = read_ssh_latlon_daily_only(root, day)
    ssh_anom = ssh - ssh_mean
    extrema = local_extrema(ssh_anom, params.ssh_window_cells, params, lon_scalar, lat_scalar)
    if not extrema:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    u_path = require_daily_file(root, "u", day, expected_dta_bytes(u_meta))
    v_path = require_daily_file(root, "v", day, expected_dta_bytes(v_meta))
    u_raw = open_dta_memmap(u_path, u_meta)
    v_raw = open_dta_memmap(v_path, v_meta)
    dx_km, dy_km = grid_spacing_km(lon_uv, lat_uv)
    centers_rows: list[dict[str, object]] = []
    circle_rows: list[dict[str, object]] = []
    structure_rows: list[dict[str, object]] = []

    u0 = uv_anomaly_layer(u_raw, u_mean, 0)
    v0 = uv_anomaly_layer(v_raw, v_mean, 0)
    speed0 = np.hypot(u0, v0)
    for seed_order, seed in enumerate(extrema):
        seed_i_uv = nearest_index(lon_uv, lon_scalar[seed["i"]])
        seed_j_uv = nearest_index(lat_uv, lat_scalar[seed["j"]])
        center_i, center_j, center_speed, min_steps = seeded_speed_min(speed0, seed_i_uv, seed_j_uv, params.surface_search_cells)
        prev_i, prev_j = center_i, center_j
        object_id = f"{day:%Y%m%d}_{seed_order:05d}"
        for depth_index, depth_m in enumerate(depth):
            if depth_index == 0:
                u = u0
                v = v0
                speed = speed0
            else:
                u = uv_anomaly_layer(u_raw, u_mean, depth_index)
                v = uv_anomaly_layer(v_raw, v_mean, depth_index)
                speed = np.hypot(u, v)
                center_i, center_j, center_speed, min_steps = seeded_speed_min(speed, prev_i, prev_j, params.deep_search_cells)
            check = hua_verify_radius(u, v, center_i, center_j, params)
            refined = refine_speed_min_subgrid(
                speed,
                u,
                v,
                lon_uv,
                lat_uv,
                center_i,
                center_j,
                target_degree=params.subgrid_target_degree,
                window_radius_cells=params.subgrid_window_radius_cells,
                min_finite_fraction=params.subgrid_min_finite_fraction,
            ) if bool(check["hua_pass"]) else fallback_refined_center(speed, lon_uv, lat_uv, center_i, center_j, "not_attempted_failed_hua")
            if bool(check["hua_pass"]):
                prev_i, prev_j = center_i, center_j
            polarity = extremum_polarity(seed["kind"], float(lat_uv[center_j]), float(check.get("circulation_sign", np.nan)))
            row = {
                "science_tag": SCIENCE_TAG,
                "date": day.isoformat(),
                "hua_object_id": object_id,
                "seed_order": seed_order,
                "ssh_extremum_type": seed["kind"],
                "candidate_selection": str(seed.get("candidate_selection", params.candidate_selection)),
                "tile_lon_min": float(seed.get("tile_lon_min", np.nan)),
                "tile_lon_max": float(seed.get("tile_lon_max", np.nan)),
                "tile_lat_min": float(seed.get("tile_lat_min", np.nan)),
                "tile_lat_max": float(seed.get("tile_lat_max", np.nan)),
                "tile_rank": int(seed.get("tile_rank", seed_order + 1)),
                "polarity": polarity,
                "depth_index": depth_index,
                "depth_m": float(depth_m),
                "seed_i_scalar": int(seed["i"]),
                "seed_j_scalar": int(seed["j"]),
                "seed_lon": float(lon_scalar[seed["i"]]),
                "seed_lat": float(lat_scalar[seed["j"]]),
                "seed_i_uv": int(seed_i_uv),
                "seed_j_uv": int(seed_j_uv),
                "ssh_anomaly_cm": float(seed["value"]),
                "speed_min_i": int(center_i),
                "speed_min_j": int(center_j),
                "speed_min_i_grid": int(center_i),
                "speed_min_j_grid": int(center_j),
                "center_lon": float(lon_uv[center_i]),
                "center_lat": float(lat_uv[center_j]),
                "center_i_refined": float(refined["center_i_refined"]),
                "center_j_refined": float(refined["center_j_refined"]),
                "center_lon_refined": float(refined["center_lon_refined"]),
                "center_lat_refined": float(refined["center_lat_refined"]),
                "center_x_from_seed_km": float((center_i - seed_i_uv) * dx_km),
                "center_y_from_seed_km": float((center_j - seed_j_uv) * dy_km),
                "center_x_refined_from_seed_km": float((float(refined["center_i_refined"]) - seed_i_uv) * dx_km),
                "center_y_refined_from_seed_km": float((float(refined["center_j_refined"]) - seed_j_uv) * dy_km),
                "center_speed_ms": float(center_speed),
                "refined_speed_ms": float(refined["refined_speed_ms"]),
                "refined_offset_km": float(refined["refined_offset_km"]),
                "refined_ok": bool(refined["refined_ok"]),
                "subgrid_fit_quality": str(refined["subgrid_fit_quality"]),
                "local_min_steps": int(min_steps),
                **check,
            }
            centers_rows.append(row)
            circle_rows.append(row.copy())
            if bool(check["hua_pass"]):
                structure_rows.append(
                    {
                        "science_tag": SCIENCE_TAG,
                        "date": day.isoformat(),
                        "hua_object_id": object_id,
                        "candidate_selection": str(seed.get("candidate_selection", params.candidate_selection)),
                        "tile_lon_min": float(seed.get("tile_lon_min", np.nan)),
                        "tile_lon_max": float(seed.get("tile_lon_max", np.nan)),
                        "tile_lat_min": float(seed.get("tile_lat_min", np.nan)),
                        "tile_lat_max": float(seed.get("tile_lat_max", np.nan)),
                        "tile_rank": int(seed.get("tile_rank", seed_order + 1)),
                        "depth_index": depth_index,
                        "depth_m": float(depth_m),
                        "center_lon": float(lon_uv[center_i]),
                        "center_lat": float(lat_uv[center_j]),
                        "center_lon_refined": float(refined["center_lon_refined"]),
                        "center_lat_refined": float(refined["center_lat_refined"]),
                        "center_i_refined": float(refined["center_i_refined"]),
                        "center_j_refined": float(refined["center_j_refined"]),
                        "refined_ok": bool(refined["refined_ok"]),
                        "refined_offset_km": float(refined["refined_offset_km"]),
                        "subgrid_fit_quality": str(refined["subgrid_fit_quality"]),
                        "radius_km": float(check["accepted_radius_cells"]) * float(np.nanmean([dx_km, dy_km])),
                        "polarity": polarity,
                    }
                )
            else:
                break
    return pd.DataFrame(centers_rows), pd.DataFrame(circle_rows), pd.DataFrame(structure_rows)


def uv_anomaly_layer(raw: np.memmap, mean: np.ndarray, level_index: int) -> np.ndarray:
    layer_cms = np.asarray(raw[:, :, level_index].T, dtype=np.float32)
    layer_cms[np.abs(layer_cms) > 1.0e30] = np.nan
    return (layer_cms - np.asarray(mean[:, :, level_index].T, dtype=np.float32)) / 100.0


def local_extrema(field: np.ndarray, window: int, params: HuaParams, lon: np.ndarray, lat: np.ndarray) -> list[dict[str, object]]:
    finite = np.isfinite(field)
    if finite.sum() == 0:
        return []
    radius = max(1, int(window) // 2)
    maxima = finite.copy()
    minima = finite.copy()
    for dj in range(-radius, radius + 1):
        for di in range(-radius, radius + 1):
            if di == 0 and dj == 0:
                continue
            shifted, valid = shifted_with_lon_wrap(field, di, dj)
            maxima &= valid & (field >= shifted)
            minima &= valid & (field <= shifted)
    candidates: list[dict[str, object]] = []
    for kind, mask in [("ssh_max", maxima), ("ssh_min", minima)]:
        jj, ii = np.where(mask)
        for j, i in zip(jj.tolist(), ii.tolist()):
            value = float(field[j, i])
            if np.isfinite(value):
                candidates.append({"kind": kind, "i": int(i), "j": int(j), "value": value, "abs_value": abs(value)})
    return select_extrema_candidates(candidates, params, lon, lat)


def select_extrema_candidates(candidates: list[dict[str, object]], params: HuaParams, lon: np.ndarray, lat: np.ndarray) -> list[dict[str, object]]:
    candidates.sort(key=lambda item: item["abs_value"], reverse=True)
    if params.candidate_selection == "global_topn":
        selected = candidates[: params.max_candidates_per_day] if params.max_candidates_per_day > 0 else candidates
        for rank, item in enumerate(selected, start=1):
            item.update(
                {
                    "candidate_selection": "global_topn",
                    "tile_lon_min": np.nan,
                    "tile_lon_max": np.nan,
                    "tile_lat_min": np.nan,
                    "tile_lat_max": np.nan,
                    "tile_rank": rank,
                }
            )
        return selected

    if params.tile_lon_deg <= 0.0 or params.tile_lat_deg <= 0.0 or params.tile_top_n <= 0:
        raise ValueError("tile_topn requires positive tile-lon-deg, tile-lat-deg, and tile-top-n")

    tile_groups: dict[tuple[int, int], list[dict[str, object]]] = {}
    for item in candidates:
        item_lon = float(lon[int(item["i"])]) % 360.0
        item_lat = float(lat[int(item["j"])])
        lon_bin = int(math.floor(item_lon / params.tile_lon_deg))
        lat_bin = int(math.floor((item_lat + 90.0) / params.tile_lat_deg))
        tile_groups.setdefault((lon_bin, lat_bin), []).append(item)

    selected: list[dict[str, object]] = []
    for (lon_bin, lat_bin), group in sorted(tile_groups.items()):
        group.sort(key=lambda item: item["abs_value"], reverse=True)
        tile_lon_min = lon_bin * params.tile_lon_deg
        tile_lat_min = lat_bin * params.tile_lat_deg - 90.0
        for rank, item in enumerate(group[: params.tile_top_n], start=1):
            item.update(
                {
                    "candidate_selection": "tile_topn",
                    "tile_lon_min": float(tile_lon_min),
                    "tile_lon_max": float(min(tile_lon_min + params.tile_lon_deg, 360.0)),
                    "tile_lat_min": float(tile_lat_min),
                    "tile_lat_max": float(min(tile_lat_min + params.tile_lat_deg, 90.0)),
                    "tile_rank": int(rank),
                }
            )
            selected.append(item)
    selected.sort(key=lambda item: (item["tile_lon_min"], item["tile_lat_min"], item["tile_rank"], -item["abs_value"]))
    return selected


def shifted_with_lon_wrap(field: np.ndarray, di: int, dj: int) -> tuple[np.ndarray, np.ndarray]:
    shifted = np.roll(field, shift=(dj, di), axis=(0, 1))
    if dj > 0:
        shifted[:dj, :] = np.nan
    elif dj < 0:
        shifted[dj:, :] = np.nan
    valid = np.isfinite(shifted)
    return shifted, valid


def seeded_speed_min(speed: np.ndarray, seed_i: int, seed_j: int, radius_cells: int) -> tuple[int, int, float, int]:
    j0 = max(0, seed_j - radius_cells)
    j1 = min(speed.shape[0], seed_j + radius_cells + 1)
    i0 = seed_i - radius_cells
    i1 = seed_i + radius_cells + 1
    cols = np.arange(i0, i1) % speed.shape[1]
    rows = np.arange(j0, j1)
    yy, xx = np.meshgrid(rows, cols, indexing="ij")
    circle = (xx - seed_i) ** 2 + (yy - seed_j) ** 2 <= radius_cells**2
    values = speed[np.ix_(rows, cols)]
    mask = circle & np.isfinite(values)
    if not mask.any():
        return iterative_speed_min(speed, seed_i, seed_j)
    flat = np.where(mask.ravel())[0]
    pick = int(flat[np.nanargmin(values.ravel()[flat])])
    local_j, local_i = np.unravel_index(pick, values.shape)
    return iterative_speed_min(speed, int(cols[local_i]), int(rows[local_j]))


def iterative_speed_min(speed: np.ndarray, start_i: int, start_j: int, max_steps: int = 40) -> tuple[int, int, float, int]:
    ii = int(start_i) % speed.shape[1]
    jj = int(np.clip(start_j, 0, speed.shape[0] - 1))
    last = (-1, -1)
    steps = 0
    while (ii, jj) != last and steps < max_steps:
        last = (ii, jj)
        y0, y1 = max(0, jj - 2), min(speed.shape[0], jj + 3)
        cols = np.arange(ii - 2, ii + 3) % speed.shape[1]
        window = speed[np.ix_(np.arange(y0, y1), cols)]
        if not np.isfinite(window).any():
            break
        wy, wx = np.unravel_index(int(np.nanargmin(window)), window.shape)
        jj = y0 + wy
        ii = int(cols[wx])
        steps += 1
    value = float(speed[jj, ii]) if np.isfinite(speed[jj, ii]) else np.nan
    return ii, jj, value, steps


def fallback_refined_center(speed: np.ndarray, lon: np.ndarray, lat: np.ndarray, center_i: int, center_j: int, quality: str) -> dict[str, object]:
    grid_speed = float(speed[center_j, center_i]) if np.isfinite(speed[center_j, center_i]) else np.nan
    return {
        "center_i_refined": float(center_i),
        "center_j_refined": float(center_j),
        "center_lon_refined": float(lon[center_i]),
        "center_lat_refined": float(lat[center_j]),
        "refined_speed_ms": grid_speed,
        "refined_offset_km": 0.0,
        "refined_ok": False,
        "subgrid_fit_quality": quality,
    }


def refine_speed_min_subgrid(
    speed: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    lon: np.ndarray,
    lat: np.ndarray,
    center_i: int,
    center_j: int,
    *,
    target_degree: float,
    window_radius_cells: int,
    min_finite_fraction: float,
) -> dict[str, object]:
    fallback = fallback_refined_center(speed, lon, lat, center_i, center_j, "fallback_grid")
    if target_degree <= 0.0 or window_radius_cells < 1:
        return fallback | {"subgrid_fit_quality": "disabled"}
    x0 = max(0, int(center_i) - int(window_radius_cells))
    x1 = min(speed.shape[1] - 1, int(center_i) + int(window_radius_cells))
    y0 = max(0, int(center_j) - int(window_radius_cells))
    y1 = min(speed.shape[0] - 1, int(center_j) + int(window_radius_cells))
    if x1 <= x0 or y1 <= y0:
        return fallback | {"subgrid_fit_quality": "window_too_small"}

    speed_window = np.asarray(speed[y0 : y1 + 1, x0 : x1 + 1], dtype="float64")
    u_window = np.asarray(u[y0 : y1 + 1, x0 : x1 + 1], dtype="float64")
    v_window = np.asarray(v[y0 : y1 + 1, x0 : x1 + 1], dtype="float64")
    finite = np.isfinite(speed_window) & np.isfinite(u_window) & np.isfinite(v_window)
    if float(finite.mean()) < float(min_finite_fraction):
        return fallback | {"subgrid_fit_quality": "insufficient_finite"}

    filled_u, finite_u = fill_nearest_finite(u_window)
    filled_v, finite_v = fill_nearest_finite(v_window)
    finite_mask = finite & finite_u & finite_v
    dlon = float(np.nanmedian(np.abs(np.diff(lon)))) if lon.size > 1 else 0.0
    dlat = float(np.nanmedian(np.abs(np.diff(lat)))) if lat.size > 1 else 0.0
    if dlon <= 0.0 or dlat <= 0.0:
        return fallback | {"subgrid_fit_quality": "invalid_grid_spacing"}
    step_i = max(float(target_degree) / dlon, 1.0e-3)
    step_j = max(float(target_degree) / dlat, 1.0e-3)
    xi = np.arange(0.0, float(x1 - x0) + 0.5 * step_i, step_i, dtype="float64")
    yj = np.arange(0.0, float(y1 - y0) + 0.5 * step_j, step_j, dtype="float64")
    if xi.size < 3 or yj.size < 3:
        return fallback | {"subgrid_fit_quality": "refined_grid_too_small"}

    yy, xx = np.meshgrid(yj, xi, indexing="ij")
    coords = np.vstack([yy.ravel(), xx.ravel()])
    dense_u = ndimage.map_coordinates(filled_u, coords, order=1, mode="nearest").reshape(yy.shape)
    dense_v = ndimage.map_coordinates(filled_v, coords, order=1, mode="nearest").reshape(yy.shape)
    dense = np.hypot(dense_u, dense_v)
    valid_weight = ndimage.map_coordinates(finite_mask.astype("float64"), coords, order=1, mode="nearest").reshape(yy.shape)
    dense = np.where(valid_weight >= 0.999, dense, np.nan)
    fit_quality = "uv_vector_linear_interp_ofes_grid_aware"
    if not np.isfinite(dense).any():
        return fallback | {"subgrid_fit_quality": "no_refined_finite"}

    local_pick = int(np.nanargmin(dense))
    pick_j, pick_i = np.unravel_index(local_pick, dense.shape)
    if pick_i in (0, dense.shape[1] - 1) or pick_j in (0, dense.shape[0] - 1):
        return fallback | {"subgrid_fit_quality": "minimum_on_refined_boundary"}
    refined_i = float(x0 + xi[pick_i])
    refined_j = float(y0 + yj[pick_j])
    dx_km, dy_km = grid_spacing_km(lon, lat)
    return {
        "center_i_refined": refined_i,
        "center_j_refined": refined_j,
        "center_lon_refined": interp_1d_from_fraction(lon, refined_i),
        "center_lat_refined": interp_1d_from_fraction(lat, refined_j),
        "refined_speed_ms": float(dense[pick_j, pick_i]),
        "refined_offset_km": float(math.hypot((refined_i - center_i) * dx_km, (refined_j - center_j) * dy_km)),
        "refined_ok": True,
        "subgrid_fit_quality": fit_quality,
    }


def fill_nearest_finite(field: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(field, dtype="float64")
    finite = np.isfinite(arr)
    if finite.all():
        return arr, finite
    if not finite.any():
        return np.zeros_like(arr), finite
    _, indices = ndimage.distance_transform_edt(~finite, return_indices=True)
    filled = arr[tuple(indices)]
    return filled, finite


def interp_1d_from_fraction(coord: np.ndarray, index_fraction: float) -> float:
    idx = float(np.clip(index_fraction, 0.0, len(coord) - 1.0))
    i0 = int(np.floor(idx))
    i1 = min(len(coord) - 1, i0 + 1)
    frac = idx - i0
    return float((1.0 - frac) * coord[i0] + frac * coord[i1])


def quadratic_speed_surface(speed_window: np.ndarray, finite: np.ndarray, xi: np.ndarray, yj: np.ndarray) -> tuple[np.ndarray | None, str]:
    yy0, xx0 = np.meshgrid(np.arange(speed_window.shape[0]), np.arange(speed_window.shape[1]), indexing="ij")
    x = xx0[finite].astype("float64")
    y = yy0[finite].astype("float64")
    z = np.square(np.asarray(speed_window, dtype="float64")[finite].ravel())
    if z.size < 9:
        return None, "quadratic_insufficient_points"
    design = np.column_stack([x * x, y * y, x * y, x, y, np.ones_like(x)])
    try:
        coeff, *_ = np.linalg.lstsq(design, z, rcond=None)
    except np.linalg.LinAlgError:
        return None, "quadratic_lstsq_failed"
    a, b, c, *_ = coeff
    hessian = np.array([[2.0 * a, c], [c, 2.0 * b]], dtype="float64")
    eig = np.linalg.eigvalsh(hessian)
    if not np.all(np.isfinite(eig)) or float(np.nanmin(eig)) <= 0.0:
        return None, "quadratic_not_convex"
    yy, xx = np.meshgrid(yj, xi, indexing="ij")
    dense_sq = coeff[0] * xx * xx + coeff[1] * yy * yy + coeff[2] * xx * yy + coeff[3] * xx + coeff[4] * yy + coeff[5]
    if not np.isfinite(dense_sq).any():
        return None, "quadratic_no_finite"
    return np.sqrt(np.maximum(dense_sq, 0.0)), "quadratic_speed2_ofes_grid_aware"


def hua_verify_radius(u: np.ndarray, v: np.ndarray, center_i: int, center_j: int, params: HuaParams) -> dict[str, float | bool | str]:
    best = None
    first_fail = None
    for radius in range(params.start_radius_cells, params.max_radius_cells + 1):
        row = circle_check(u, v, center_i, center_j, radius, params)
        if bool(row["circle_passed"]):
            best = row
        else:
            first_fail = row
            break
    source = dict(best if best is not None else first_fail or {"circle_passed": False, "radius_cells": np.nan, "dominant_failure": "no_circle"})
    source["hua_pass"] = bool(best is not None)
    source["accepted_radius_cells"] = float(source["radius_cells"]) if best is not None else 0.0
    return source


def circle_check(u: np.ndarray, v: np.ndarray, center_i: int, center_j: int, radius_cells: int, params: HuaParams) -> dict[str, float | bool | str]:
    offsets = circle_offsets(radius_cells)
    ii = np.asarray([(center_i + dx) % u.shape[1] for dx, _ in offsets], dtype=int)
    jj = np.asarray([center_j + dy for _, dy in offsets], dtype=int)
    inside = (jj >= 0) & (jj < u.shape[0])
    uu = np.full(len(offsets), np.nan, dtype=np.float64)
    vv = np.full(len(offsets), np.nan, dtype=np.float64)
    uu[inside] = u[jj[inside], ii[inside]]
    vv[inside] = v[jj[inside], ii[inside]]
    sp = np.hypot(uu, vv)
    finite = np.isfinite(sp) & (sp > 1.0e-10)
    failures = {"invalid_velocity": 0, "velocity_ratio": 0, "angle_jump": 0, "rotation_direction": 0, "tangent_alignment": 0, "symmetry": 0, "opposite_reversal": 0, "boundary_monotonic_rotation": 0}
    rotation_failed = finite.mean() < params.min_finite_fraction
    if rotation_failed:
        failures["invalid_velocity"] += int((~finite).sum())
    angles = np.arctan2(vv, uu)
    max_ratio = 0.0
    max_angle = 0.0
    positive = 0
    negative = 0
    for n in range(len(offsets)):
        m = (n + 1) % len(offsets)
        if not (finite[n] and finite[m]):
            rotation_failed = True
            failures["invalid_velocity"] += 1
            continue
        ratio = float(sp[m] / sp[n])
        max_ratio = max(max_ratio, ratio, 1.0 / ratio if ratio > 0 else np.inf)
        if ratio > params.speed_ratio_max or ratio < 1.0 / params.speed_ratio_max:
            rotation_failed = True
            failures["velocity_ratio"] += 1
        dtheta = angle_diff(angles[n], angles[m])
        max_angle = max(max_angle, abs(math.degrees(dtheta)))
        if abs(math.degrees(dtheta)) > params.angle_jump_max_deg:
            rotation_failed = True
            failures["angle_jump"] += 1
        if dtheta > 0:
            positive += 1
        elif dtheta < 0:
            negative += 1
    direction_exceptions = min(positive, negative)
    if direction_exceptions > params.boundary_monotonic_exception_limit:
        rotation_failed = True
        failures["boundary_monotonic_rotation"] += int(direction_exceptions - params.boundary_monotonic_exception_limit)

    dx = np.asarray([p[0] for p in offsets], dtype=np.float64)
    dy = np.asarray([p[1] for p in offsets], dtype=np.float64)
    th = np.arctan2(dy, dx)
    tx = -np.sin(th)
    ty = np.cos(th)
    tangent_cos = np.abs((uu * tx + vv * ty) / np.maximum(sp, 1.0e-12))
    tangent_ok = finite & (tangent_cos >= math.cos(math.radians(params.tangent_tolerance_deg)))
    tangent_fraction = float(tangent_ok.sum() / finite.sum()) if finite.any() else 0.0
    if tangent_fraction < params.min_tangent_fraction:
        rotation_failed = True
        failures["tangent_alignment"] += 1

    half = len(offsets) // 2
    symmetry_ok = 0
    symmetry_total = 0
    reversal_ok = 0
    reversal_total = 0
    for n in range(half):
        m = (n + half) % len(offsets)
        if not (finite[n] and finite[m]):
            continue
        diff = abs(angle_diff(angles[n], angles[m]))
        symmetry_total += 1
        if abs(diff - math.pi) <= math.radians(params.symmetry_tolerance_deg):
            symmetry_ok += 1
        reversal_total += 1
        if uu[n] * uu[m] + vv[n] * vv[m] < 0:
            reversal_ok += 1
    symmetry_fraction = float(symmetry_ok / symmetry_total) if symmetry_total else 0.0
    reversal_fraction = float(reversal_ok / reversal_total) if reversal_total else 0.0
    if symmetry_total and symmetry_ok < symmetry_total:
        failures["symmetry"] += int(symmetry_total - symmetry_ok)
    if reversal_fraction < params.min_reversal_fraction:
        rotation_failed = True
        failures["opposite_reversal"] += 1

    tangential = uu * tx + vv * ty
    circulation_sign = float(np.sign(np.nanmedian(tangential[finite]))) if finite.any() else np.nan
    dominant_failure = max(failures.items(), key=lambda kv: kv[1])[0] if sum(failures.values()) else "none"
    return {
        "circle_passed": bool(not rotation_failed),
        "radius_cells": float(radius_cells),
        "finite_fraction": float(finite.mean()),
        "mean_circle_speed_ms": float(np.nanmean(sp[finite])) if finite.any() else np.nan,
        "max_velocity_ratio": float(max_ratio),
        "max_angle_jump_deg": float(max_angle),
        "direction_exception_count": float(direction_exceptions),
        "boundary_monotonic_required": True,
        "boundary_monotonic_passed": bool(direction_exceptions <= params.boundary_monotonic_exception_limit),
        "boundary_monotonic_exception_limit": float(params.boundary_monotonic_exception_limit),
        "tangent_pass_fraction": tangent_fraction,
        "symmetry_pass_fraction": symmetry_fraction,
        "opposite_reversal_fraction": reversal_fraction,
        "circulation_sign": circulation_sign,
        "dominant_failure": dominant_failure,
        **{f"failure_{name}_count": float(count) for name, count in failures.items()},
    }


def circle_offsets(radius_cells: int) -> list[tuple[int, int]]:
    points: list[tuple[int, int]] = []
    n = max(16, int(round(8 * radius_cells)))
    for theta in np.linspace(-math.pi / 2.0, 3.0 * math.pi / 2.0, n, endpoint=False):
        point = (int(round(radius_cells * math.cos(theta))), int(round(radius_cells * math.sin(theta))))
        if not points or points[-1] != point:
            points.append(point)
    if len(points) > 1 and points[0] == points[-1]:
        points.pop()
    return points


def angle_diff(a: float, b: float) -> float:
    return float((a - b + math.pi) % (2.0 * math.pi) - math.pi)


def extremum_polarity(extremum: str, lat_value: float, circulation_sign: float) -> str:
    if np.isfinite(circulation_sign) and circulation_sign != 0:
        f_sign = 1.0 if lat_value >= 0 else -1.0
        return "cyclonic" if circulation_sign == f_sign else "anticyclonic"
    return "cyclonic" if extremum == "ssh_min" else "anticyclonic"


def grid_spacing_km(lon: np.ndarray, lat: np.ndarray) -> tuple[float, float]:
    mid_lat = float(np.nanmedian(lat))
    dx = np.deg2rad(float(np.nanmedian(np.abs(np.diff(lon))))) * EARTH_RADIUS_M * math.cos(math.radians(mid_lat)) / 1000.0
    dy = np.deg2rad(float(np.nanmedian(np.abs(np.diff(lat))))) * EARTH_RADIUS_M / 1000.0
    return abs(dx), abs(dy)


def nearest_index(values: np.ndarray, value: float) -> int:
    return int(np.nanargmin(np.abs(np.asarray(values) - value)))


def date_range(start: date, end: date) -> list[date]:
    out = []
    current = start
    while current <= end:
        out.append(current)
        current += timedelta(days=1)
    return out


def parse_iso_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def write_detection_summary(detection_dir: Path, centers: pd.DataFrame, circle: pd.DataFrame, structures: pd.DataFrame, params: HuaParams, days: list[date], nlev: int) -> None:
    rejection = pd.DataFrame(columns=["failure_reason", "count"])
    if not centers.empty and "dominant_failure" in centers.columns:
        rejection = centers.loc[~centers["hua_pass"].astype(bool), "dominant_failure"].value_counts().rename_axis("failure_reason").reset_index(name="count")
    write_table(rejection, detection_dir / "rejection_reasons")
    summary = {
        "science_tag": SCIENCE_TAG,
        "date_range": {"start": days[0].isoformat(), "end": days[-1].isoformat(), "n_days": len(days)},
        "domain": "global",
        "depth_layers": int(nlev),
        "n_center_rows": int(len(centers)),
        "n_circle_rows": int(len(circle)),
        "n_structure_rows": int(len(structures)),
        "n_surface_candidates": int(centers[centers["depth_index"].eq(0)]["hua_object_id"].nunique()) if not centers.empty else 0,
        "n_pass_layers": int(centers["hua_pass"].sum()) if not centers.empty else 0,
        "pass_fraction": float(centers["hua_pass"].mean()) if not centers.empty else 0.0,
        "parameters": asdict(params),
    }
    summary["daily_surface_candidate_count"] = daily_surface_counts(centers)
    summary["daily_hua_pass_layer_count"] = daily_pass_layer_counts(centers)
    summary["daily_structure_object_count"] = daily_structure_object_counts(structures)
    summary["strict_1r_crossing_counts"] = strict_crossing_counts(structures, [20.0, 40.0])
    summary["baseline_global_top80_comparison"] = baseline_comparison(detection_dir, centers, structures)
    if not centers.empty and "refined_ok" in centers.columns:
        passed = centers[centers["hua_pass"].astype(bool)].copy()
        offsets = passed["refined_offset_km"].to_numpy(dtype="f8") if "refined_offset_km" in passed.columns else np.array([])
        finite_offsets = offsets[np.isfinite(offsets)]
        summary["subgrid_refined_enabled"] = bool(params.subgrid_target_degree > 0.0)
        summary["subgrid_target_degree"] = float(params.subgrid_target_degree)
        summary["subgrid_window_radius_cells"] = int(params.subgrid_window_radius_cells)
        summary["subgrid_refined_ok_rows"] = int(passed["refined_ok"].fillna(False).astype(bool).sum()) if not passed.empty else 0
        summary["subgrid_refined_ok_fraction_of_passed"] = float(passed["refined_ok"].fillna(False).astype(bool).mean()) if not passed.empty else 0.0
        summary["subgrid_refined_offset_km_median"] = float(np.nanmedian(finite_offsets)) if finite_offsets.size else 0.0
        summary["subgrid_refined_offset_km_p90"] = float(np.nanquantile(finite_offsets, 0.9)) if finite_offsets.size else 0.0
    write_json(detection_dir / "run_summary.json", summary)
    lines = [
        "# OFES2 Hua Hybrid Method Alignment",
        "",
        f"- Science tag: `{SCIENCE_TAG}`.",
        "- Input fields: OFES2 SSH/u/v daily fields minus Jan 1991 monthly mean.",
        "- This is not the 30-180 day bandpass production Kuroshio product.",
        "- Method: SSH extrema seed, velocity minimum, circular velocity checks, boundary monotonic, strict contiguous vertical extension.",
        f"- Date range: `{days[0].isoformat()}` to `{days[-1].isoformat()}`.",
        f"- Depth layers: `{nlev}`.",
        f"- Center rows: `{summary['n_center_rows']}`.",
        f"- Hua pass layers: `{summary['n_pass_layers']}`.",
        "",
        "## Rejection Reasons",
        "",
    ]
    if rejection.empty:
        lines.append("- none")
    else:
        for row in rejection.to_dict("records"):
            lines.append(f"- `{row['failure_reason']}`: `{row['count']}`")
    (detection_dir / "method_alignment_zh.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def daily_surface_counts(centers: pd.DataFrame) -> dict[str, int]:
    if centers.empty:
        return {}
    surface = centers[centers["depth_index"].eq(0)]
    return {str(day): int(count) for day, count in surface.groupby("date")["hua_object_id"].nunique().items()}


def daily_pass_layer_counts(centers: pd.DataFrame) -> dict[str, int]:
    if centers.empty:
        return {}
    passed = centers[centers["hua_pass"].astype(bool)]
    return {str(day): int(count) for day, count in passed.groupby("date").size().items()}


def daily_structure_object_counts(structures: pd.DataFrame) -> dict[str, int]:
    if structures.empty:
        return {}
    return {str(day): int(count) for day, count in structures.groupby("date")["hua_object_id"].nunique().items()}


def strict_crossing_counts(structures: pd.DataFrame, target_lats: list[float]) -> dict[str, dict[str, object]]:
    objects = representative_structure_objects(structures)
    out: dict[str, dict[str, object]] = {}
    for target_lat in target_lats:
        hits = 0
        nearest = None
        for row in objects:
            distance_km = abs(float(row["center_lat"]) - target_lat) * math.pi * EARTH_RADIUS_M / 180_000.0
            distance_over_r = distance_km / float(row["radius_km"]) if float(row["radius_km"]) > 0 else math.inf
            candidate = row | {"distance_over_r": distance_over_r}
            if distance_over_r <= 1.0:
                hits += 1
            if nearest is None or distance_over_r < float(nearest["distance_over_r"]):
                nearest = candidate
        out[f"{target_lat:g}N"] = {
            "strict_1r_count": int(hits),
            "nearest_hua_object_id": str(nearest["hua_object_id"]) if nearest else "",
            "nearest_center_lat": float(nearest["center_lat"]) if nearest else math.nan,
            "nearest_distance_over_r": float(nearest["distance_over_r"]) if nearest else math.nan,
        }
    return out


def representative_structure_objects(structures: pd.DataFrame) -> list[dict[str, object]]:
    if structures.empty:
        return []
    lon_col = "center_lon_refined" if "center_lon_refined" in structures.columns else "center_lon"
    lat_col = "center_lat_refined" if "center_lat_refined" in structures.columns else "center_lat"
    rows = []
    for object_id, part in structures.groupby("hua_object_id", sort=False):
        part = part.sort_values("depth_index")
        surface = part.iloc[0]
        rows.append(
            {
                "hua_object_id": str(object_id),
                "date": str(surface["date"]),
                "polarity": str(surface["polarity"]),
                "center_lon": float(surface[lon_col]),
                "center_lat": float(surface[lat_col]),
                "radius_km": float(np.nanmedian(part["radius_km"].to_numpy(dtype="f8"))),
                "pass_layers": int(len(part)),
            }
        )
    return rows


def baseline_comparison(detection_dir: Path, centers: pd.DataFrame, structures: pd.DataFrame) -> dict[str, object]:
    baseline_dir = detection_dir.parents[1] / "available_jan01_jan19_refined_ofes_grid" / "detection_hua_global_jan1991"
    current = {
        "n_surface_candidates": int(centers[centers["depth_index"].eq(0)]["hua_object_id"].nunique()) if not centers.empty else 0,
        "n_structure_objects": int(structures["hua_object_id"].nunique()) if not structures.empty else 0,
    }
    if not baseline_dir.exists():
        return {"baseline_dir": str(baseline_dir), "baseline_found": False, "current": current}
    baseline_summary_path = baseline_dir / "run_summary.json"
    baseline = {}
    if baseline_summary_path.exists():
        try:
            baseline = json.loads(baseline_summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            baseline = {}
    return {
        "baseline_dir": str(baseline_dir),
        "baseline_found": True,
        "baseline_n_surface_candidates": int(baseline.get("n_surface_candidates", 0)),
        "baseline_n_structure_objects": int(pd.read_csv(baseline_dir / "structures_hua_style.csv")["hua_object_id"].nunique()) if (baseline_dir / "structures_hua_style.csv").exists() else 0,
        "current": current,
    }


def write_day_figure(centers: pd.DataFrame, day: date, figure_dir: Path) -> None:
    if centers.empty:
        return
    surface = centers[centers["depth_index"].eq(0)]
    if surface.empty:
        return
    width, height = 1200, 520
    canvas = np.full((height, width, 3), 245, dtype=np.uint8)
    for _, row in surface.iterrows():
        x = int((float(row["center_lon"]) % 360.0) / 360.0 * (width - 1))
        y = int((90.0 - (float(row["center_lat"]) + 90.0)) / 180.0 * (height - 1))
        color = np.array([34, 197, 94], dtype=np.uint8) if bool(row["hua_pass"]) else np.array([156, 163, 175], dtype=np.uint8)
        canvas[max(0, y - 2) : min(height, y + 3), max(0, x - 2) : min(width, x + 3)] = color
    figure_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(canvas, mode="RGB").save(figure_dir / f"surface_candidates_{day:%Y%m%d}.png")


def write_preview(field: np.ndarray, path: Path, vmin: float, vmax: float) -> None:
    sampled = np.flipud(field[::4, ::4])
    finite = np.nan_to_num(sampled, nan=0.0, posinf=vmax, neginf=vmin)
    scaled = np.clip((finite - vmin) / (vmax - vmin), 0.0, 1.0)
    blue = np.array([49, 130, 189], dtype=np.float32)
    white = np.array([247, 247, 247], dtype=np.float32)
    red = np.array([203, 24, 29], dtype=np.float32)
    rgb = np.empty((*scaled.shape, 3), dtype=np.uint8)
    lower = scaled <= 0.5
    lt = scaled[lower] / 0.5
    ht = (scaled[~lower] - 0.5) / 0.5
    rgb[lower] = (blue * (1 - lt[:, None]) + white * lt[:, None]).astype(np.uint8)
    rgb[~lower] = (white * (1 - ht[:, None]) + red * ht[:, None]).astype(np.uint8)
    rgb[~np.isfinite(sampled)] = np.array([190, 190, 190], dtype=np.uint8)
    Image.fromarray(rgb, mode="RGB").save(path)


def write_table(df: pd.DataFrame, prefix: Path) -> None:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(prefix.with_suffix(".csv"), index=False)
    try:
        df.to_parquet(prefix.with_suffix(".parquet"), index=False)
    except Exception as exc:
        prefix.with_suffix(".parquet.txt").write_text(
            "Parquet output was skipped because the current environment could not write it.\n"
            f"Reason: {type(exc).__name__}: {exc}\n"
            "The CSV file with the same prefix is authoritative for this run.\n",
            encoding="utf-8",
        )


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
