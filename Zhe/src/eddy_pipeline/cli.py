from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from .contracts import (
    HUA_B3_START2_DETECTION_PARAMS,
    PRODUCTION_SCIENCE_MOUTHFUL,
    default_runtime_config_name,
    default_shape_output_name,
    strict_depth_policy_text,
)


DEFAULT_OUTPUT_ROOT = Path("/root/autodl-tmp/second_reslut")
DEFAULT_FILTER_ROOT = Path("/root/autodl-fs/kuroshiou/Filter")
DEFAULT_RAW_ROOT = Path("/root/autodl-fs/kuroshiou/raw")
DEFAULT_CLIMATOLOGY = Path("/root/autodl-fs/kuroshiou/result/climatology/cmems_doy_climatology_1993_2022_31d.nc")
DEFAULT_STALE_DIRS = (
    Path("/root/autodl-fs/kuroshiou/result"),
    Path("/root/autodl-fs/kuroshiou/result_strict_contiguous"),
    Path("/root/autodl-fs/kuroshiou/result_coherent_only"),
)


DRY_RUN = False


@dataclass(frozen=True)
class Paths:
    output_root: Path
    shape_name_value: str
    representative_relative_dir: Path
    radial_seed_relative_dir: Path

    @property
    def detection_dir(self) -> Path:
        return self.output_root / "hua_b3_start2_detection"

    @property
    def tracking_dir(self) -> Path:
        return self.detection_dir / "feature_group_tracking"

    @property
    def logs_dir(self) -> Path:
        return self.output_root / "logs"

    @property
    def catalog_dir(self) -> Path:
        return self.output_root / "catalog"

    @property
    def shape_name(self) -> str:
        return self.shape_name_value

    @property
    def representative_dir(self) -> Path:
        return self.output_root / self.representative_relative_dir

    @property
    def radial_seed_dir(self) -> Path:
        return self.output_root / self.radial_seed_relative_dir

    @property
    def config_path(self) -> Path:
        return self.output_root / default_runtime_config_name()


def _parse_date(value: str) -> datetime:
    return datetime.strptime(str(value)[:10], "%Y-%m-%d")


def _date_label(value: datetime) -> str:
    return value.strftime("%Y-%m-%d")


def _quarter_shards(start: str, end: str) -> list[tuple[str, str, str]]:
    start_dt = _parse_date(start)
    end_dt = _parse_date(end)
    shards: list[tuple[str, str, str]] = []
    current = start_dt
    while current <= end_dt:
        quarter = (current.month - 1) // 3 + 1
        q_end_month = quarter * 3
        next_q = datetime(current.year + (q_end_month == 12), 1 if q_end_month == 12 else q_end_month + 1, 1)
        shard_end = min(end_dt, next_q - timedelta(days=1))
        tag = f"{current.year}q{quarter}"
        shards.append((_date_label(current), _date_label(shard_end), tag))
        current = shard_end + timedelta(days=1)
    return shards


def _run(cmd: list[str], log_path: Path | None = None, *, cwd: Path | None = None) -> None:
    if DRY_RUN:
        print("[dry-run] " + " ".join(cmd), flush=True)
        return
    if log_path is None:
        subprocess.run(cmd, check=True, cwd=str(cwd) if cwd else None)
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=str(cwd) if cwd else None)
    if proc.returncode:
        raise subprocess.CalledProcessError(proc.returncode, cmd)


def _run_parallel(commands: list[tuple[list[str], Path]], max_parallel: int) -> None:
    if DRY_RUN:
        for cmd, log_path in commands:
            print(f"[dry-run] {' '.join(cmd)} > {log_path}", flush=True)
        return
    running: list[tuple[subprocess.Popen, Path, list[str]]] = []
    pending = list(commands)
    max_parallel = max(1, int(max_parallel))
    while pending or running:
        while pending and len(running) < max_parallel:
            cmd, log_path = pending.pop(0)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log = log_path.open("w", encoding="utf-8")
            proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
            log.close()
            running.append((proc, log_path, cmd))
        still: list[tuple[subprocess.Popen, Path, list[str]]] = []
        for proc, log_path, cmd in running:
            code = proc.poll()
            if code is None:
                still.append((proc, log_path, cmd))
            elif code != 0:
                raise subprocess.CalledProcessError(code, cmd, output=f"See log: {log_path}")
        running = still
        if running:
            time.sleep(2)


def _human_size(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024.0 or unit == "TiB":
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{value:.1f} TiB"


def _dir_size(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for root, _, files in os.walk(path):
        for name in files:
            try:
                total += (Path(root) / name).stat().st_size
            except OSError:
                continue
    return total


def _table_count(path: Path) -> dict[str, object]:
    if not path.exists():
        return {"exists": False}
    try:
        import pandas as pd

        df = pd.read_parquet(path, engine="fastparquet")
    except Exception as exc:
        return {"exists": True, "error": f"{type(exc).__name__}: {exc}"}
    out: dict[str, object] = {"exists": True, "rows": int(df.shape[0]), "columns": list(df.columns)}
    if "shape_class" in df.columns:
        out["shape_counts"] = {str(k): int(v) for k, v in df["shape_class"].value_counts().sort_index().items()}
    if "polarity" in df.columns:
        out["polarity_counts"] = {str(k): int(v) for k, v in df["polarity"].value_counts().sort_index().items()}
    return out


def _file_info(path: Path) -> dict[str, object]:
    if not path.exists():
        return {"exists": False}
    return {"exists": True, "size_bytes": int(path.stat().st_size)}


def write_runtime_config(args: argparse.Namespace, paths: Paths) -> None:
    bbox = [float(item) for item in str(args.bbox).split(",")]
    if len(bbox) != 4:
        raise ValueError("--bbox must contain four comma-separated values: lon_min,lon_max,lat_min,lat_max")
    cfg = {
        "project": {"name": str(args.project_name), "env_name": str(args.env_name)},
        "data_source": {
            "kind": "cmems_netcdf_timeseries",
            "input_nc_dir": str(args.filter_root),
            "annual_file_template": str(args.filter_template),
        },
        "paths": {
            "output_dir": str(paths.output_root),
            "input_daily_dir": str(paths.output_root / "input_daily_disabled"),
            "layer_dir": str(paths.output_root / "layers_disabled"),
            "catalog_dir": str(paths.catalog_dir),
            "logs_dir": str(paths.logs_dir),
            "temp_dir": str(paths.output_root / "work"),
        },
        "region": {"bbox": bbox, "max_depth_m": float(args.max_depth_m)},
        "date_range": {"start": str(args.start), "end": str(args.end)},
        "variables": {
            "source_lon": "longitude",
            "source_lat": "latitude",
            "source_depth": "depth",
            "source_time": "time",
            "source_height": "zos_glor",
            "source_u": "uo_glor",
            "source_v": "vo_glor",
            "output_height": str(args.output_height_name),
        },
        "conversion": {"use_input_daily": False, "netcdf_format": "NETCDF4", "compression_level": 1},
        "hua_b3": {
            "strict_contiguous_depth": not bool(args.allow_noncontiguous_depth),
            "depth_policy": strict_depth_policy_text(allow_noncontiguous_depth=bool(args.allow_noncontiguous_depth)),
            "require_boundary_monotonic_rotation": bool(args.require_boundary_monotonic_rotation),
            "boundary_monotonic_exception_limit": int(args.boundary_monotonic_exception_limit),
            "subgrid_center_refinement": not bool(args.disable_subgrid_center_refinement),
            "subgrid_target_degree": float(args.subgrid_target_degree),
            "subgrid_window_radius_cells": int(args.subgrid_window_radius_cells),
            "subgrid_min_finite_fraction": float(args.subgrid_min_finite_fraction),
            "production_science_mouthful": PRODUCTION_SCIENCE_MOUTHFUL,
        },
    }
    if DRY_RUN:
        print(f"[dry-run] write runtime config: {paths.config_path}", flush=True)
        print(f"[dry-run] production science: {PRODUCTION_SCIENCE_MOUTHFUL}", flush=True)
        return

    import yaml

    paths.output_root.mkdir(parents=True, exist_ok=True)
    paths.config_path.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")


def detection_command(args: argparse.Namespace, paths: Paths, start: str, end: str, extra: list[str]) -> list[str]:
    params = HUA_B3_START2_DETECTION_PARAMS
    cmd = [
        sys.executable,
        "-m",
        "src.eddy_pipeline.detection_hybrid",
        "--filter-root",
        str(args.filter_root),
        "--raw-root",
        str(args.raw_root),
        "--filter-template",
        str(args.filter_template),
        "--raw-template",
        str(args.raw_template),
        "--output-dir",
        str(paths.detection_dir),
        "--start",
        start,
        "--end",
        end,
        "--max-depth-m",
        str(args.max_depth_m),
        "--ssh-window-cells",
        "7",
        "--max-candidates-per-day",
        str(args.max_candidates_per_day),
        "--surface-search-cells",
        str(params["surface_search_cells"]),
        "--deep-search-cells",
        str(params["deep_search_cells"]),
        "--start-radius-cells",
        str(params["start_radius_cells"]),
        "--max-radius-cells",
        str(params["max_radius_cells"]),
        "--speed-ratio-max",
        str(params["speed_ratio_max"]),
        "--angle-jump-max-deg",
        str(params["angle_jump_max_deg"]),
        "--tangent-tolerance-deg",
        str(params["tangent_tolerance_deg"]),
        "--symmetry-tolerance-deg",
        str(params["symmetry_tolerance_deg"]),
        "--min-tangent-fraction",
        str(params["min_tangent_fraction"]),
        "--min-reversal-fraction",
        str(params["min_reversal_fraction"]),
        "--min-finite-fraction",
        str(params["min_finite_fraction"]),
        "--direction-exception-extra",
        str(params["direction_exception_extra"]),
        "--subgrid-target-degree",
        str(args.subgrid_target_degree),
        "--subgrid-window-radius-cells",
        str(args.subgrid_window_radius_cells),
        "--subgrid-min-finite-fraction",
        str(args.subgrid_min_finite_fraction),
        "--preload-day-uv",
        "--write-object-voxels",
        "--resume",
    ]
    if bool(args.disable_subgrid_center_refinement):
        cmd.append("--disable-subgrid-center-refinement")
    if not bool(args.allow_noncontiguous_depth):
        cmd.append("--stop-at-first-failed-layer")
    if bool(args.require_boundary_monotonic_rotation):
        cmd.extend(
            [
                "--require-boundary-monotonic-rotation",
                "--boundary-monotonic-exception-limit",
                str(args.boundary_monotonic_exception_limit),
            ]
        )
    cmd.extend(extra)
    return cmd


def run_pipeline(args: argparse.Namespace, paths: Paths) -> None:
    global DRY_RUN
    DRY_RUN = bool(args.dry_run)
    if not DRY_RUN:
        paths.logs_dir.mkdir(parents=True, exist_ok=True)
    write_runtime_config(args, paths)
    if "detect" in args.stages:
        shard_cmds = [
            (detection_command(args, paths, start, end, ["--partial-only"]), paths.logs_dir / f"detect_{tag}.log")
            for start, end, tag in _quarter_shards(args.start, args.end)
        ]
        _run_parallel(shard_cmds, int(args.detect_parallel))
        _run(
            detection_command(args, paths, args.start, args.end, ["--finalize-only"]),
            paths.logs_dir / "detect_finalize.log",
        )
        for csv_name in ("centers_hua_style.csv", "circle_check_diagnostics.csv", "object_voxels.csv", "structures_hua_style.csv"):
            (paths.detection_dir / csv_name).unlink(missing_ok=True)
    if "tracking" in args.stages:
        _run(
            [
                sys.executable,
                "-m",
                "src.eddy_pipeline.tracking",
                "--input-dir",
                str(paths.detection_dir),
                "--output-dir",
                str(paths.tracking_dir),
            ],
            paths.logs_dir / "feature_group_tracking.log",
        )
        for csv_path in paths.tracking_dir.glob("*.csv"):
            csv_path.unlink(missing_ok=True)
    if "catalog_shape" in args.stages:
        adapter_cmd = [
                sys.executable,
                "-m",
                "src.eddy_pipeline.catalog",
                "--detection-dir",
                str(paths.detection_dir),
                "--tracking-dir",
                str(paths.tracking_dir),
                "--output-root",
                str(paths.output_root),
                "--shape-output-name",
                paths.shape_name,
                "--start",
                args.start,
                "--end",
                args.end,
                "--lifetime-min-days",
                str(args.lifetime_min_days),
                "--radius-min-m",
                str(args.radius_min_m),
                "--min-valid-layers",
                str(args.min_valid_layers),
        ]
        if bool(args.allow_noncontiguous_depth):
            adapter_cmd.append("--allow-noncontiguous-depth")
        _run(adapter_cmd, paths.logs_dir / "catalog_shape_adapter.log")
    if "representative" in args.stages:
        radial_output_dir = paths.representative_dir
        if str(args.representative_composite_method) == "me_liutex":
            radial_output_dir = paths.radial_seed_dir
        _run(
            [
                sys.executable,
                "-m",
                "src.eddy_pipeline.radial_representative",
                "--config",
                str(paths.config_path),
                "--results-root",
                str(paths.output_root),
                "--output-dir",
                str(radial_output_dir),
                "--shape-dir-name",
                paths.shape_name,
                "--climatology-path",
                str(args.climatology_path),
                "--shapes",
                "coherent",
                "--polarities",
                "cyclonic,anticyclonic",
                "--tau-grid-step",
                "0.05",
                "--kernel-bandwidth",
                "0.075",
                "--radial-bins",
                "40",
                "--azimuth-bins",
                "72",
                "--rmax",
                "2.5",
                "--axis-alignment",
                "global_ls_alpha",
                "--field-cache-mode",
                "day",
                "--workers",
                str(args.representative_workers),
                "--chunk-days",
                str(args.representative_chunk_days),
                "--resume",
            ],
            paths.logs_dir / "representative_coherent_life30_lowmem.log",
        )
        patch_representative_summaries(radial_output_dir)
        if str(args.representative_composite_method) == "me_liutex":
            orientation_map = {
                "turned": [("global_alpha", paths.representative_dir, paths.logs_dir / "representative_coherent_life30_me_liutex.log")],
                "unturned": [("unturned", paths.representative_dir.with_name(paths.representative_dir.name + "_unturned"), paths.logs_dir / "representative_coherent_life30_me_liutex_unturned.log")],
                "both": [
                    ("global_alpha", paths.representative_dir, paths.logs_dir / "representative_coherent_life30_me_liutex.log"),
                    ("unturned", paths.representative_dir.with_name(paths.representative_dir.name + "_unturned"), paths.logs_dir / "representative_coherent_life30_me_liutex_unturned.log"),
                ],
            }
            for orientation_mode, output_dir, log_path in orientation_map[str(args.orientation)]:
                _run(
                    [
                    sys.executable,
                    "-m",
                    "src.eddy_pipeline.representative",
                    "--rv-root",
                    str(radial_output_dir),
                    "--filter-root",
                    str(args.filter_root),
                    "--output-dir",
                    str(output_dir),
                    "--filter-template",
                    str(args.filter_template),
                    "--orientation-mode",
                    orientation_mode,
                    "--max-depth-m",
                    str(args.max_depth_m),
                    "--tau-grid-step",
                    "0.05",
                    "--kernel-bandwidth",
                    "0.075",
                    "--radial-bins",
                    "40",
                    "--azimuth-bins",
                    "72",
                    "--rmax",
                    "2.5",
                    ],
                    log_path,
                )


def patch_representative_summaries(representative_dir: Path, region_name: str = "ACC") -> None:
    summary = representative_dir / "summary.md"
    if not summary.exists():
        return
    text = summary.read_text(encoding="utf-8")
    text = text.replace("# Representative vortex continuous tau summary", f"# {region_name} Hua b3 representative vortex continuous tau summary")
    text = text.replace("# Kuroshio representative vortex continuous tau summary", f"# {region_name} Hua b3 representative vortex continuous tau summary")
    text = text.replace("# ACC Hua b3 representative vortex continuous tau summary", f"# {region_name} Hua b3 representative vortex continuous tau summary")
    text = text.replace(
        "- Velocity basis: raw annual CMEMS uo_glor/vo_glor; no input_daily files.",
        "- Velocity basis: 30-180 day bandpass CMEMS uo_glor/vo_glor from Filter; no input_daily files.",
    )
    summary.write_text(text, encoding="utf-8")


def write_manifests(args: argparse.Namespace, paths: Paths) -> None:
    paths.output_root.mkdir(parents=True, exist_ok=True)
    result_manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "authority": f"{args.region_name} Hua/Nencioli b3_start2 life{int(args.lifetime_min_days)} coherent-only representative",
        "output_root": str(paths.output_root),
        "representative_root": str(paths.representative_dir),
        "defaults": {
            "date_start": args.start,
            "date_end_inclusive": args.end,
            "strict_contiguous_depth": not bool(args.allow_noncontiguous_depth),
            "require_boundary_monotonic_rotation": bool(args.require_boundary_monotonic_rotation),
            "boundary_monotonic_exception_limit": int(args.boundary_monotonic_exception_limit),
            "production_science_mouthful": PRODUCTION_SCIENCE_MOUTHFUL,
            "shape_lifetime_min_days": int(args.lifetime_min_days),
            "representative_field_cache_mode": "day",
            "representative_composite_method": str(args.representative_composite_method),
            "radial_seed_root": str(paths.radial_seed_dir),
            "representative_workers": int(args.representative_workers),
        },
        "tables": {
            "hua_centers": _table_count(paths.detection_dir / "centers_hua_style.parquet"),
            "feature_tracks": _table_count(paths.tracking_dir / "feature_tracks.parquet"),
            "layer_observations": _table_count(paths.catalog_dir / "layer_observations.parquet"),
            "vertical_objects": _table_count(paths.catalog_dir / "vertical_objects.parquet"),
            "tracks_3d": _table_count(paths.catalog_dir / "tracks_3d.parquet"),
            "completed_centers": _table_count(paths.catalog_dir / "layer_centers_completed.parquet"),
            "shape_tracks_life30": _table_count(paths.output_root / paths.shape_name / "shape_tracks.parquet"),
            "representative_psi": _table_count(paths.representative_dir / "streamfunction_templates" / "continuous_radial_psi_profiles.parquet"),
            "representative_ep_flux": _table_count(paths.representative_dir / "ep_flux_terms" / "continuous_ep_flux_profiles.parquet"),
            "representative_me_liutex_summary": _file_info(paths.representative_dir / "azimuthal_representative_summary.json"),
        },
    }
    (paths.output_root / "results_manifest.json").write_text(
        json.dumps(result_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (paths.output_root / "results_manifest.md").write_text(_manifest_markdown(result_manifest), encoding="utf-8")

    stale = []
    for path in DEFAULT_STALE_DIRS:
        size = _dir_size(path)
        stale.append(
            {
                "path": str(path),
                "exists": path.exists(),
                "size_bytes": int(size),
                "size_human": _human_size(size),
                "suggested_status": "stale_candidate_do_not_delete_without_explicit_confirmation",
                "suggested_command": f"rm -rf {path}",
            }
        )
    stale_manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "policy": "No deletion is performed by this manifest. Commands are suggestions only.",
        "stale_candidates": stale,
    }
    (paths.output_root / "stale_results_manifest.json").write_text(
        json.dumps(stale_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (paths.output_root / "stale_results_manifest.md").write_text(_stale_markdown(stale_manifest), encoding="utf-8")


def _manifest_markdown(manifest: dict[str, object]) -> str:
    lines = [
        "# Hua b3 Result Manifest",
        "",
        f"- Generated at: `{manifest['generated_at']}`",
        f"- Authority: {manifest['authority']}",
        f"- Output root: `{manifest['output_root']}`",
        f"- Representative root: `{manifest['representative_root']}`",
        "",
        "## Table Counts",
        "",
    ]
    for name, info in manifest["tables"].items():  # type: ignore[index,union-attr]
        if info.get("exists"):
            lines.append(f"- `{name}`: `{info.get('rows', 'unknown')}` rows")
        else:
            lines.append(f"- `{name}`: missing")
    lines.extend(["", "## Notes", "", f"- Production science mouthful: `{PRODUCTION_SCIENCE_MOUTHFUL}`."])
    return "\n".join(lines) + "\n"


def _stale_markdown(manifest: dict[str, object]) -> str:
    lines = [
        "# Stale Result Candidates",
        "",
        f"- Generated at: `{manifest['generated_at']}`",
        f"- Policy: {manifest['policy']}",
        "",
    ]
    for item in manifest["stale_candidates"]:  # type: ignore[index,union-attr]
        lines.append(f"- `{item['path']}`: exists={item['exists']}, size={item['size_human']}")
    lines.extend(["", "No deletion has been performed."])
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run or document the Hua/Nencioli b3_start2 CPU-only pipeline.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--filter-root", type=Path, default=DEFAULT_FILTER_ROOT)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--filter-template", default="global_phy_{year}_bandpass_30_180d.nc")
    parser.add_argument("--raw-template", default="global_phy_{year}.nc")
    parser.add_argument("--climatology-path", type=Path, default=DEFAULT_CLIMATOLOGY)
    parser.add_argument("--region-name", default="kuroshiou")
    parser.add_argument("--project-name", default="kuroshiou_hua_b3_start2")
    parser.add_argument("--env-name", default="eddy_verify")
    parser.add_argument("--bbox", default="120.0,145.0,20.0,35.0")
    parser.add_argument("--output-height-name", default="zos_bandpass")
    parser.add_argument("--shape-output-name", default="")
    parser.add_argument("--representative-relative-dir", type=Path, default=Path("result_coherent_only") / "representative_vortex_me_liutex")
    parser.add_argument("--radial-seed-relative-dir", type=Path, default=Path("result_coherent_only") / "representative_vortex_radial_seed")
    parser.add_argument("--representative-composite-method", choices=["me_liutex", "radial"], default="me_liutex")
    parser.add_argument("--start", default="1993-01-01")
    parser.add_argument("--end", default="2022-12-31")
    parser.add_argument("--max-depth-m", type=float, default=2000.0)
    parser.add_argument("--max-candidates-per-day", type=int, default=80)
    parser.add_argument("--detect-parallel", type=int, default=6)
    parser.add_argument("--lifetime-min-days", type=int, default=30)
    parser.add_argument("--radius-min-m", type=float, default=50_000.0)
    parser.add_argument("--min-valid-layers", type=int, default=6)
    parser.add_argument("--representative-workers", type=int, default=1)
    parser.add_argument("--representative-chunk-days", type=int, default=7)
    parser.add_argument("--orientation", choices=["turned", "unturned", "both"], default="turned")
    parser.add_argument(
        "--allow-noncontiguous-depth",
        action="store_true",
        help="Legacy/diagnostic mode. By default the Hua b3 pipeline stops each surface seed at the first failed layer.",
    )
    parser.add_argument(
        "--require-boundary-monotonic-rotation",
        action="store_true",
        default=True,
        help="Experimental mode: require monotonic rotation of velocity vectors along each accepted Hua boundary circle.",
    )
    parser.add_argument(
        "--allow-nonmonotonic-boundary",
        action="store_true",
        help="Legacy/diagnostic mode. Disable the production boundary-monotonic velocity-vector constraint.",
    )
    parser.add_argument("--boundary-monotonic-exception-limit", type=int, default=0)
    parser.add_argument(
        "--disable-subgrid-center-refinement",
        action="store_true",
        help="Legacy/diagnostic mode. Keep original grid-cell centers instead of local 1/24 degree refined velocity centers.",
    )
    parser.add_argument("--subgrid-target-degree", type=float, default=float(HUA_B3_START2_DETECTION_PARAMS["subgrid_target_degree"]))
    parser.add_argument("--subgrid-window-radius-cells", type=int, default=int(HUA_B3_START2_DETECTION_PARAMS["subgrid_window_radius_cells"]))
    parser.add_argument("--subgrid-min-finite-fraction", type=float, default=float(HUA_B3_START2_DETECTION_PARAMS["subgrid_min_finite_fraction"]))
    parser.add_argument(
        "--stages",
        default="detect,tracking,catalog_shape,representative",
        help="Comma-separated stages: detect,tracking,catalog_shape,representative.",
    )
    parser.add_argument("--manifest-only", action="store_true", help="Only write current result and stale-result manifests; do not run pipeline stages.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned commands without running them.")
    return parser


def main() -> None:
    argv = sys.argv[1:]
    command_aliases = {
        "run-detection-to-shape": "detect,tracking,catalog_shape",
        "build-representative": "representative",
        "run-all": "detect,tracking,catalog_shape,representative",
    }
    if argv and argv[0] in command_aliases:
        argv = ["--stages", command_aliases[argv[0]], *argv[1:]]
    args = build_parser().parse_args(argv)
    if bool(args.allow_nonmonotonic_boundary):
        args.require_boundary_monotonic_rotation = False
    args.stages = tuple(item.strip() for item in str(args.stages).split(",") if item.strip())
    if not args.shape_output_name:
        args.shape_output_name = default_shape_output_name(args.start, args.end, int(args.lifetime_min_days))
    paths = Paths(Path(args.output_root), str(args.shape_output_name), Path(args.representative_relative_dir), Path(args.radial_seed_relative_dir))
    if not args.manifest_only:
        run_pipeline(args, paths)
    if bool(args.dry_run):
        print(f"[dry-run] skip summary patching and manifest writes under: {paths.output_root}", flush=True)
        return
    patch_representative_summaries(paths.representative_dir, str(args.region_name))
    write_manifests(args, paths)
    print(f"{args.region_name} Hua b3 pipeline manifest written under: {paths.output_root}")


if __name__ == "__main__":
    main()
