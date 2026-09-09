from __future__ import annotations

import argparse
import json
import math
import os
import struct
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_ARGO_MAT = Path(r"F:\Argo_data\ArgoData_SA_CT_PT_PDen_sigma.mat")
DEFAULT_HISTORY_ARGO_MAT = Path(r"F:\Argo_data\Argo1000m_UVW_TSDen_199601_202306.mat")
DEFAULT_META_DIR = Path(r"F:\Eddy\Eddy\META4.0_DT_allsat")
DEFAULT_BOA_PDEN_ROOT = Path(r"F:\Argo_data\Self_BOA_Argo_PotentialDensity")
DEFAULT_CACHE_ROOT = Path(r"E:\DATA\01_Eddy_correspond\01_Vertical_asymmetric\_cache")
DEFAULT_OUTPUT_ROOT = Path(
    r"E:\DATA\01_Eddy_correspond\01_Vertical_asymmetric\META4_CoreArgo_vertical_transport_crossing_global_60S60N_10deg"
)
DEFAULT_BBOX = "0,360,-60,60"
DEFAULT_CROSSING_LATS = "-60,-50,-40,-30,-20,-10,0,10,20,30,40,50,60"
DEFAULT_LAT_BANDS = (
    "-60:-55,-55:-50,-50:-45,-45:-40,-40:-35,-35:-30,"
    "-30:-25,-25:-20,-20:-15,-15:-10,-10:-5,-5:0,"
    "0:5,5:10,10:15,15:20,20:25,25:30,"
    "30:35,35:40,40:45,45:50,50:55,55:60"
)


def parse_bbox(value: str) -> tuple[float, float, float, float]:
    parts = [float(item.strip()) for item in value.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("--bbox must be lon_min,lon_max,lat_min,lat_max")
    lon_min, lon_max, lat_min, lat_max = parts
    if lon_min >= lon_max or lat_min >= lat_max:
        raise argparse.ArgumentTypeError("--bbox ranges must be increasing")
    return lon_min, lon_max, lat_min, lat_max


def parse_lat_bands(value: str) -> list[tuple[float, float]]:
    bands: list[tuple[float, float]] = []
    for item in value.split(","):
        lo_text, hi_text = item.split(":")
        lo, hi = float(lo_text), float(hi_text)
        if lo >= hi:
            raise argparse.ArgumentTypeError("latitude bands must be increasing")
        bands.append((lo, hi))
    return bands


def parse_crossing_lats(value: str) -> list[float]:
    lats: list[float] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        lat = float(item)
        if lat < -90 or lat > 90:
            raise argparse.ArgumentTypeError("crossing latitudes must be within [-90, 90]")
        lats.append(lat)
    if not lats:
        raise argparse.ArgumentTypeError("--crossing-lats must contain at least one latitude")
    return lats


def parse_depth_levels(value: str) -> list[float]:
    value = value.strip()
    if ":" in value:
        parts = [float(item) for item in value.split(":")]
        if len(parts) != 3:
            raise argparse.ArgumentTypeError("--depth-levels range must be start:step:end")
        start, step, end = parts
        if step <= 0 or start <= 0 or end < start:
            raise argparse.ArgumentTypeError("--depth-levels range must have positive increasing depths")
        count = int(math.floor((end - start) / step + 0.5)) + 1
        levels = [start + i * step for i in range(count) if start + i * step <= end + step * 1e-6]
    else:
        levels = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not levels or any(level <= 0 for level in levels):
        raise argparse.ArgumentTypeError("--depth-levels must contain positive depths in meters")
    return levels


def parse_sensitivity_configs(value: str) -> list[str]:
    allowed = {"baseline", "recommended", "smoother", "strong_support", "low_res_smooth", "high_smooth"}
    names = [item.strip() for item in value.split(",") if item.strip()]
    if not names:
        raise argparse.ArgumentTypeError("--sensitivity-configs must contain at least one name")
    unknown = sorted(set(names) - allowed)
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown sensitivity config(s): {', '.join(unknown)}")
    return names


def matlab_quote(path: Path) -> str:
    return str(path).replace("\\", "\\\\").replace("'", "''")


def latitude_crossing_label(target_lat: float, intersect_radius_r: float) -> str:
    hemi = "N" if target_lat >= 0 else "S"
    lat_text = f"{abs(target_lat):02.0f}{hemi}"
    radius_text = f"{intersect_radius_r:g}".replace(".", "p")
    return f"cross_{lat_text}_{radius_text}R"


def npy_bytes_2d(values: list[list[float]]) -> bytes:
    rows = len(values)
    cols = len(values[0]) if rows else 0
    header = f"{{'descr': '<f8', 'fortran_order': False, 'shape': ({rows}, {cols}), }}"
    padding = 16 - ((10 + len(header) + 1) % 16)
    header = header + (" " * padding) + "\n"
    out = bytearray()
    out.extend(b"\x93NUMPY")
    out.extend(b"\x01\x00")
    out.extend(struct.pack("<H", len(header)))
    out.extend(header.encode("latin1"))
    for row in values:
        for value in row:
            out.extend(struct.pack("<d", float(value) if value is not None else math.nan))
    return bytes(out)


def _array_shape(values) -> tuple[int, ...]:
    shape = []
    item = values
    while isinstance(item, list):
        shape.append(len(item))
        item = item[0] if item else []
    return tuple(shape)


def _flatten_nested(values):
    if isinstance(values, list):
        for item in values:
            yield from _flatten_nested(item)
    else:
        yield values


def npy_bytes_nested(values) -> bytes:
    shape = _array_shape(values)
    shape_text = f"({shape[0]},)" if len(shape) == 1 else str(shape)
    header = f"{{'descr': '<f8', 'fortran_order': False, 'shape': {shape_text}, }}"
    padding = 16 - ((10 + len(header) + 1) % 16)
    header = header + (" " * padding) + "\n"
    out = bytearray()
    out.extend(b"\x93NUMPY")
    out.extend(b"\x01\x00")
    out.extend(struct.pack("<H", len(header)))
    out.extend(header.encode("latin1"))
    for value in _flatten_nested(values):
        out.extend(struct.pack("<d", float(value) if value is not None else math.nan))
    return bytes(out)


def write_npz_from_grid_json(json_path: Path, npz_path: Path) -> None:
    grid = json.loads(json_path.read_text(encoding="utf-8"))
    array_keys = [
        "x_over_R",
        "y_over_R",
        "z_rho_m",
        "z_rho_raw_m",
        "z_rho_anom_m",
        "u_argo_m_s",
        "v_argo_m_s",
        "wpk_observed_m_s",
        "sample_count",
        "mapped_support",
        "wpk_mapped_support",
        "term1_m_s",
        "term2_m_s",
        "rebuild_w_m_s",
        "term1_depth_positive_m_s",
        "term2_depth_positive_m_s",
        "rebuild_w_raw_depth_positive_m_s",
        "term1_plus_m_s",
        "term1_minus_m_s",
        "term2_abs_m_s",
        "term2_rel_m_s",
        "rebuild_plus_abs_m_s",
        "rebuild_minus_rel_m_s",
        "sample_term1_m_s",
        "sample_term2_m_s",
        "sample_rebuild_w_m_s",
    ]
    arrays = {key: grid[key] for key in array_keys if key in grid}
    with zipfile.ZipFile(npz_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, values in arrays.items():
            zf.writestr(f"{name}.npy", npy_bytes_2d(values))
        zf.writestr("metadata.json", json.dumps(grid.get("metadata", {}), ensure_ascii=False, indent=2))


def write_npz_from_w3d_json(json_path: Path, npz_path: Path) -> None:
    grid = json.loads(json_path.read_text(encoding="utf-8"))
    array_keys = [
        "depth_m",
        "x_over_R",
        "y_over_R",
        "w_3d_m_s",
        "term1_3d_m_s",
        "term2_3d_m_s",
        "z_rho_anom_3d_m",
        "sample_count_3d",
        "mapped_support_3d",
        "section_axis_coord_over_R",
        "section_depth_m",
        "section_w_m_s",
    ]
    arrays = {key: grid[key] for key in array_keys if key in grid}
    with zipfile.ZipFile(npz_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, values in arrays.items():
            zf.writestr(f"{name}.npy", npy_bytes_nested(values))
        zf.writestr("metadata.json", json.dumps(grid.get("metadata", {}), ensure_ascii=False, indent=2))


def run_matlab_pipeline(args: argparse.Namespace) -> list[Path]:
    args.output_root.mkdir(parents=True, exist_ok=True)
    log_path = args.output_root / "matlab_run.log"
    with tempfile.TemporaryDirectory(prefix="meta4_core_argo_") as tmp:
        tmp_dir = Path(tmp)
        manifest_path = tmp_dir / "manifest.json"
        script_path = tmp_dir / "run_meta4_core_argo.m"
        script_path.write_text(_matlab_script(args, manifest_path), encoding="utf-8")
        cmd = ["matlab", "-batch", f"run('{matlab_quote(script_path)}')"]
        with log_path.open("w", encoding="utf-8", errors="replace") as log:
            log.write(f"Command: {' '.join(cmd)}\n")
            log.flush()
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                log.write(line)
                log.flush()
            returncode = proc.wait()
        if returncode != 0:
            tail = "\n".join(log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-120:])
            raise RuntimeError(
                "MATLAB vertical-transport pipeline failed.\n"
                f"Command: {' '.join(cmd)}\n"
                f"Log: {log_path}\n"
                f"Log tail:\n{tail}"
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    output_root = Path(manifest.get("output_root", args.output_root))
    manifest_grid_paths = [Path(item) for item in manifest.get("grid_files", [])]
    if manifest_grid_paths:
        grid_paths = manifest_grid_paths
    else:
        grid_paths = [Path(item) for item in manifest.get("grid_json_files", [])]
        if not grid_paths:
            grid_paths = sorted(output_root.rglob("composite_grid.json"))
    npz_paths = []
    for grid_path in grid_paths:
        if grid_path.suffix.lower() == ".json":
            npz_path = grid_path.with_suffix(".npz")
            write_npz_from_grid_json(grid_path, npz_path)
            npz_paths.append(npz_path)
        else:
            npz_paths.append(grid_path)
    w3d_json_paths = []
    if not manifest_grid_paths:
        w3d_json_paths = sorted(output_root.rglob("w_3d_grid.json"))
    for grid_json in w3d_json_paths:
        npz_path = grid_json.with_suffix(".npz")
        write_npz_from_w3d_json(grid_json, npz_path)
        npz_paths.append(npz_path)
    return npz_paths


def _matlab_script(args: argparse.Namespace, manifest_path: Path) -> str:
    bbox = " ".join(f"{item:.12g}" for item in args.bbox)
    lat_bands = "; ".join(f"{lo:.12g} {hi:.12g}" for lo, hi in args.lat_bands)
    crossing_lats = " ".join(f"{lat:.12g}" for lat in args.crossing_lats)
    depth_levels = " ".join(f"{depth:.12g}" for depth in args.depth_levels)
    sensitivity_configs = " ".join(f"'{name}'" for name in args.sensitivity_configs)
    target_label = latitude_crossing_label(args.target_lat, args.intersect_radius_r)
    argo_mat = matlab_quote(args.argo_mat)
    history_argo_mat = matlab_quote(args.history_argo_mat)
    meta_dir = matlab_quote(args.meta_dir)
    boa_pden_root = matlab_quote(args.boa_pden_root)
    cache_root = matlab_quote(args.cache_root)
    output_root = matlab_quote(args.output_root)
    manifest = matlab_quote(manifest_path)
    max_matches = int(args.max_matches_per_group)
    template_path = SCRIPT_DIR / "matlab" / "run_meta4_core_argo_backend.m"
    template = template_path.read_text(encoding="utf-8")
    return (
        template.replace("@ARGO_MAT@", argo_mat)
        .replace("@HISTORY_ARGO_MAT@", history_argo_mat)
        .replace("@META_DIR@", meta_dir)
        .replace("@BOA_PDEN_ROOT@", boa_pden_root)
        .replace("@CACHE_ROOT@", cache_root)
        .replace("@OUTPUT_ROOT@", output_root)
        .replace("@BBOX@", bbox)
        .replace("@LAT_BANDS@", lat_bands)
        .replace("@CROSSING_LATS@", crossing_lats)
        .replace("@SELECTION_MODE@", str(args.selection_mode).replace("'", "''"))
        .replace("@TARGET_LAT@", f"{float(args.target_lat):.12g}")
        .replace("@INTERSECT_RADIUS_R@", f"{float(args.intersect_radius_r):.12g}")
        .replace("@TARGET_LABEL@", target_label.replace("'", "''"))
        .replace("@TIME_WINDOW_DAYS@", f"{float(args.time_window_days):.12g}")
        .replace("@GRID_N@", str(int(args.grid_n)))
        .replace("@MIN_BIN_COUNT@", str(int(args.min_bin_count)))
        .replace("@PLOT_FILLED_GRADIENT@", "true" if args.plot_filled_gradient else "false")
        .replace("@GRID_MAPPING@", str(args.grid_mapping).replace("'", "''"))
        .replace("@SMOOTH_PASSES@", str(int(args.smooth_passes)))
        .replace("@CRESSMAN_RADIUS_R@", f"{float(args.cressman_radius_r):.12g}")
        .replace("@CRESSMAN_MIN_OBS@", str(int(args.cressman_min_obs)))
        .replace("@SAMPLE_GRADIENT_MAX_PROFILES@", str(int(args.sample_gradient_max_profiles)))
        .replace("@RHO0_MODE@", str(args.rho0_mode).replace("'", "''"))
        .replace("@Z_MODE@", str(args.z_mode).replace("'", "''"))
        .replace("@VERTICAL_MODE@", str(args.vertical_mode).replace("'", "''"))
        .replace("@FAST_SENSITIVITY_2D@", "true" if args.fast_sensitivity_2d else "false")
        .replace("@SENSITIVITY_WORKERS@", str(int(args.workers)))
        .replace("@SENSITIVITY_CONFIGS@", sensitivity_configs)
        .replace("@COMPUTE_DEVICE@", str(args.compute_device).replace("'", "''"))
        .replace("@MATLAB_PROFILE@", "true" if args.matlab_profile else "false")
        .replace("@DIAGNOSE_REVERSAL_FACTORS@", "true" if args.diagnose_reversal_factors else "false")
        .replace("@COMPARE_Z_GEOMETRY_MODES@", "true" if args.compare_z_geometry_modes else "false")
        .replace("@Z_GEOMETRY_MODE@", str(args.z_geometry_mode).replace("'", "''"))
        .replace("@WRITE_MATCHED_CSV@", "true" if args.write_matched_csv else "false")
        .replace("@WRITE_GRID_JSON@", "true" if args.write_grid_json else "false")
        .replace("@WRITE_GRID_NC@", "true" if args.write_grid_nc else "false")
        .replace("@WRITE_SUMMARY_CSV@", "true" if args.write_summary_csv else "false")
        .replace("@DEPTH_LEVELS@", depth_levels)
        .replace("@SECTION_AXIS@", str(args.section_axis).replace("'", "''"))
        .replace("@SECTION_HALF_WIDTH_R@", f"{float(args.section_half_width_r):.12g}")
        .replace("@BOA_BACKGROUND_MODE@", str(args.boa_background_mode).replace("'", "''"))
        .replace("@VELOCITY_SOURCE@", str(args.velocity_source).replace("'", "''"))
        .replace("@MATCH_MODE@", str(args.match_mode).replace("'", "''"))
        .replace("@MAX_MATCHES@", str(max_matches))
        .replace("@CORE_MIN_M@", f"{float(args.core_min_m):.12g}")
        .replace("@CORE_MAX_M@", f"{float(args.core_max_m):.12g}")
        .replace("@Z_RHO_MIN_M@", f"{float(args.z_rho_min_m):.12g}")
        .replace("@Z_RHO_MAX_M@", f"{float(args.z_rho_max_m):.12g}")
        .replace("@MIN_DRHO_DZ@", f"{float(args.min_drho_dz):.12g}")
        .replace("@MAX_RHO_BRACKET_DZ_M@", f"{float(args.max_rho_bracket_dz_m):.12g}")
        .replace("@DENSITY_VARIABLE@", str(args.density_variable).replace("'", "''"))
        .replace("@MANIFEST@", manifest)
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Rebuild vertical transport terms from META4.0 eddies and Core Argo.")
    parser.add_argument("--argo-mat", type=Path, default=DEFAULT_ARGO_MAT)
    parser.add_argument("--history-argo-mat", type=Path, default=DEFAULT_HISTORY_ARGO_MAT)
    parser.add_argument("--meta-dir", type=Path, default=DEFAULT_META_DIR)
    parser.add_argument("--boa-pden-root", type=Path, default=DEFAULT_BOA_PDEN_ROOT)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--bbox", type=parse_bbox, default=parse_bbox(DEFAULT_BBOX))
    parser.add_argument("--lat-bands", type=parse_lat_bands, default=parse_lat_bands(DEFAULT_LAT_BANDS))
    parser.add_argument(
        "--crossing-lats",
        type=parse_crossing_lats,
        default=None,
        help="Comma-separated crossing latitudes. Defaults to -60,-50,...,60 unless --target-lat is explicitly used as a single-lat shortcut.",
    )
    parser.add_argument(
        "--selection-mode",
        choices=("lat_band", "crossing_lat"),
        default="crossing_lat",
        help="Choose standard latitude-band sampling or eddies whose radius crosses a target latitude.",
    )
    parser.add_argument("--target-lat", type=float, default=20.0)
    parser.add_argument("--intersect-radius-r", type=float, default=1.0)
    parser.add_argument("--time-window-days", type=float, default=1.0)
    parser.add_argument("--core-min-m", type=float, default=900.0)
    parser.add_argument("--core-max-m", type=float, default=1100.0)
    parser.add_argument("--z-rho-min-m", type=float, default=900.0)
    parser.add_argument("--z-rho-max-m", type=float, default=1100.0)
    parser.add_argument("--min-drho-dz", type=float, default=1e-5)
    parser.add_argument("--max-rho-bracket-dz-m", type=float, default=150.0)
    parser.add_argument("--density-variable", default="I_sigma1")
    parser.add_argument("--grid-n", type=int, default=81)
    parser.add_argument(
        "--min-bin-count",
        type=int,
        default=1,
        help="Minimum samples required for a composite grid cell to be displayed and saved in W terms.",
    )
    parser.add_argument(
        "--plot-filled-gradient",
        action="store_true",
        help="Keep the filled-gradient term1 field for diagnostics. By default W terms are masked to sampled cells.",
    )
    parser.add_argument(
        "--grid-mapping",
        choices=("cressman", "scattered", "bin"),
        default="cressman",
        help="Map profiles to the composite grid. Cressman is the default objective mapping; scattered is a diagnostic interpolation; bin keeps sampled cells only.",
    )
    parser.add_argument("--smooth-passes", type=int, default=2)
    parser.add_argument(
        "--cressman-radius-r",
        type=float,
        default=0.5,
        help="Cressman influence radius in eddy-radius units for --grid-mapping cressman.",
    )
    parser.add_argument(
        "--cressman-min-obs",
        type=int,
        default=3,
        help="Minimum profiles within the Cressman influence radius required to map a grid point.",
    )
    parser.add_argument(
        "--sample-gradient-max-profiles",
        type=int,
        default=1000,
        help="Deterministic cap for the sample-gradient-then-composite diagnostic. Use 0 for all profiles.",
    )
    parser.add_argument(
        "--rho0-mode",
        choices=("band_median", "profile"),
        default="band_median",
        help="Choose the target isopycnal. band_median uses one shared rho0 per latitude/polarity group; profile keeps the old per-profile parking-density target.",
    )
    parser.add_argument(
        "--z-mode",
        choices=("anomaly_boa_climatology", "anomaly_farfield_plane", "anomaly_farfield", "absolute"),
        default="anomaly_boa_climatology",
        help="Use BOA monthly climatology, far-field plane/scalar isopycnal displacement anomaly, or absolute z_rho.",
    )
    parser.add_argument(
        "--vertical-mode",
        choices=("single_isopycnal", "isopycnal_depth_stack", "thermal_wind_depth_stack"),
        default="single_isopycnal",
        help="single_isopycnal keeps the existing 2-D parking-depth W. isopycnal_depth_stack builds W(x/R,y/R,z). thermal_wind_depth_stack extends parking drift vertically with thermal-wind shear.",
    )
    parser.add_argument(
        "--fast-sensitivity-2d",
        action="store_true",
        help="Run the cached 2-D 20N-style sensitivity sweep: match once, then remap the six built-in Cressman/smoothing configurations.",
    )
    parser.add_argument(
        "--sensitivity-configs",
        type=parse_sensitivity_configs,
        default=parse_sensitivity_configs("baseline,recommended,smoother,strong_support,low_res_smooth,high_smooth"),
        help="Comma-separated subset of fast sensitivity configs to run, e.g. baseline,recommended.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Workers for the fast sensitivity remapping loop. Default is min(8, half of available CPU cores).",
    )
    parser.add_argument(
        "--compute-device",
        choices=("auto", "cpu", "gpu"),
        default="auto",
        help="Use GPU-accelerated Cressman mapping when available. auto tries GPU and falls back to CPU.",
    )
    parser.add_argument(
        "--matlab-profile",
        action="store_true",
        help="Save MATLAB profiler output under the output root for performance audits.",
    )
    parser.add_argument(
        "--diagnose-reversal-factors",
        action="store_true",
        help="Run the 20N-style controlled diagnosis for density-slope, thermal-wind velocity, and term2 choices.",
    )
    parser.add_argument(
        "--compare-z-geometry-modes",
        action="store_true",
        help="Run the controlled 20N comparison between BOA-referenced z_rho anomaly geometry and composite-density isosurface geometry.",
    )
    parser.add_argument(
        "--z-geometry-mode",
        choices=("boa_anomaly", "composite_density_isosurface"),
        default="boa_anomaly",
        help="Geometry used for z_rho gradients in diagnostic comparison modes.",
    )
    parser.add_argument(
        "--depth-levels",
        type=parse_depth_levels,
        default=parse_depth_levels("10:10:2000"),
        help="Depth levels for 3-D vertical modes, e.g. 10:10:2000 or 100,200,300.",
    )
    parser.add_argument(
        "--section-axis",
        choices=("x", "y"),
        default="x",
        help="Axis for the vertical section. x means x/R section averaged over |y/R| <= half-width.",
    )
    parser.add_argument(
        "--section-half-width-r",
        type=float,
        default=0.25,
        help="Half-width in eddy-radius units used to average the cross-section.",
    )
    parser.add_argument(
        "--boa-background-mode",
        choices=("monthly_climatology",),
        default="monthly_climatology",
        help="BOA background density mode. monthly_climatology averages PDen1000_YYYYMM files by calendar month.",
    )
    parser.add_argument(
        "--velocity-source",
        choices=("argo1000m_match", "profile_diff"),
        default="argo1000m_match",
        help="Use matched historical Argo1000m I_Upk/I_Vpk/I_Wpk, or the older profile-position-difference velocity proxy.",
    )
    parser.add_argument(
        "--match-mode",
        choices=("nearest", "all"),
        default="nearest",
        help="nearest assigns each Argo profile to the closest normalized eddy; all repeats it for every eddy within 4R and the time window.",
    )
    parser.add_argument(
        "--max-matches-per-group",
        type=int,
        default=0,
        help="Debug/smoke limit. Use 0 for all matched profiles.",
    )
    parser.add_argument(
        "--write-matched-csv",
        action="store_true",
        help="Opt in to large matched-table CSV files. Default writes matched tables as MAT only.",
    )
    parser.add_argument(
        "--write-grid-json",
        action="store_true",
        help="Opt in to large grid JSON files and Python NPZ conversion. Default writes MAT/NetCDF grids.",
    )
    parser.add_argument(
        "--write-summary-csv",
        action="store_true",
        help="Opt in to SUMMARY CSV files. Default writes summary tables as MAT.",
    )
    parser.add_argument(
        "--no-grid-nc",
        action="store_false",
        dest="write_grid_nc",
        help="Disable default NetCDF grid output and keep MAT-only grids.",
    )
    parser.set_defaults(write_grid_nc=True)
    raw_argv = sys.argv[1:]
    argv: list[str] = []
    i = 0
    while i < len(raw_argv):
        if raw_argv[i] == "--crossing-lats" and i + 1 < len(raw_argv):
            argv.append(f"--crossing-lats={raw_argv[i + 1]}")
            i += 2
        else:
            argv.append(raw_argv[i])
            i += 1
    args = parser.parse_args(argv)
    target_lat_explicit = any(item == "--target-lat" or item.startswith("--target-lat=") for item in raw_argv)
    if args.crossing_lats is None:
        if args.selection_mode == "crossing_lat" and target_lat_explicit:
            args.crossing_lats = [args.target_lat]
        else:
            args.crossing_lats = parse_crossing_lats(DEFAULT_CROSSING_LATS)
    if args.workers <= 0:
        args.workers = max(1, min(8, (os.cpu_count() or 2) // 2))
    npz_paths = run_matlab_pipeline(args)
    for path in npz_paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
