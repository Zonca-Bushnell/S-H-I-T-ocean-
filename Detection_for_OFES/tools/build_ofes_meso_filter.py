from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import ExitStack
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
from netCDF4 import Dataset
from scipy.ndimage import convolve1d, gaussian_filter1d
from scipy.signal import bessel, fftconvolve, sosfreqz


DEFAULT_INPUT_ROOT = Path(r"E:\DATA\01_Eddy_correspond\02_OFES\origin_compatible_filter")
DEFAULT_OUTPUT_ROOT = Path(r"E:\DATA\01_Eddy_correspond\02_OFES\origin_compatible_filter_meso10d_50_500km")
DEFAULT_ROSSBY_RADIUS_PATH = Path(
    r"E:\DATA\01_Eddy_correspond\02_OFES\rossby_radius_chelton1998"
    r"\unzip\fecampos-campos2025-a5d24c0\rossrad.nc"
)


def main() -> None:
    args = build_parser().parse_args()
    written = build_meso_filter(
        input_root=Path(args.input_root),
        output_root=Path(args.output_root),
        start=parse_date(args.start),
        end=parse_date(args.end),
        available_start=parse_date(args.available_start),
        available_end=parse_date(args.available_end),
        temporal_window_days=int(args.temporal_window_days),
        small_cutoff_km=float(args.small_cutoff_km),
        large_cutoff_km=float(args.large_cutoff_km),
        filter_mode=str(args.filter_mode),
        spatial_kernel=str(args.spatial_kernel),
        science_tag=str(args.science_tag) if args.science_tag else "",
        large_cutoff_mode=str(args.large_cutoff_mode),
        adaptive_large_cutoff_min_km=float(args.adaptive_large_cutoff_min_km),
        adaptive_large_cutoff_max_km=float(args.adaptive_large_cutoff_max_km),
        rossby_radius_path=Path(args.rossby_radius_path),
        rossby_small_factor=float(args.rossby_small_factor),
        rossby_large_factor=float(args.rossby_large_factor),
        rossby_min_km=float(args.rossby_min_km),
        rossby_max_km=float(args.rossby_max_km),
        rossby_large_fixed_km=float(args.rossby_large_fixed_km),
        zonal_scale_mode=str(args.zonal_scale_mode),
        meridional_scale_mode=str(args.meridional_scale_mode),
        min_valid_weight_fraction=float(args.min_valid_weight_fraction),
        max_depth_layers=int(args.max_depth_layers),
        workers=int(args.workers),
        convolution_engine=str(args.convolution_engine),
        compression_level=int(args.compression_level),
        overwrite=bool(args.overwrite),
    )
    print(json.dumps({"written": [str(path) for path in written]}, ensure_ascii=False, indent=2), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build Origin-compatible OFES daily NetCDF files with optional running mean and horizontal scale separation."
    )
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--start", default="1991-01-01")
    parser.add_argument("--end", default="1991-01-01")
    parser.add_argument("--available-start", default="1991-01-01")
    parser.add_argument("--available-end", default="1991-01-19")
    parser.add_argument("--temporal-window-days", type=int, default=10)
    parser.add_argument("--small-cutoff-km", type=float, default=50.0)
    parser.add_argument("--large-cutoff-km", type=float, default=500.0)
    parser.add_argument(
        "--filter-mode",
        choices=["bandpass", "highpass"],
        default="bandpass",
        help="bandpass computes LP_small - LP_large; highpass computes field - LP_large.",
    )
    parser.add_argument(
        "--spatial-kernel",
        choices=["gaussian", "lanczos", "bessel"],
        default="gaussian",
        help=(
            "Kernel used for each low-pass in the scale separation. Gaussian uses "
            "FWHM directly; Lanczos and order-3 Bessel are calibrated to the Gaussian "
            "half-power wavelength at the requested physical scale."
        ),
    )
    parser.add_argument("--science-tag", default="", help="Optional explicit science_tag written into NetCDF metadata.")
    parser.add_argument(
        "--large-cutoff-mode",
        choices=["fixed", "latitude_adaptive", "rossby_radius", "rossby_lower_latadaptive_upper"],
        default="fixed",
        help="fixed uses one scale everywhere; latitude_adaptive uses cos^2 latitude; "
        "rossby_radius scales both cutoffs with the local first-baroclinic deformation radius; "
        "rossby_lower_latadaptive_upper uses an R1 lower cutoff and a latitude-adaptive upper cutoff.",
    )
    parser.add_argument("--adaptive-large-cutoff-min-km", type=float, default=50.0)
    parser.add_argument("--adaptive-large-cutoff-max-km", type=float, default=180.0)
    parser.add_argument("--rossby-radius-path", type=Path, default=DEFAULT_ROSSBY_RADIUS_PATH)
    parser.add_argument("--rossby-small-factor", type=float, default=0.5)
    parser.add_argument("--rossby-large-factor", type=float, default=4.0)
    parser.add_argument("--rossby-min-km", type=float, default=10.0)
    parser.add_argument("--rossby-max-km", type=float, default=500.0)
    parser.add_argument(
        "--rossby-large-fixed-km",
        type=float,
        default=0.0,
        help="If positive, use this fixed upper cutoff while keeping the R1-scaled lower cutoff.",
    )
    parser.add_argument(
        "--zonal-scale-mode",
        choices=["km", "degree"],
        default="km",
        help="km converts cutoff to zonal grid sigma with cos(lat); degree uses a fixed longitude-degree sigma.",
    )
    parser.add_argument(
        "--meridional-scale-mode",
        choices=["median", "local"],
        default="median",
        help="median preserves the legacy global meridional sigma; local uses each latitude's physical cutoff.",
    )
    parser.add_argument(
        "--min-valid-weight-fraction",
        type=float,
        default=0.0,
        help="Leave a filtered cell missing when its NaN-aware convolution support is below this fraction.",
    )
    parser.add_argument(
        "--max-depth-layers",
        type=int,
        default=1,
        help="Number of velocity depth layers to export. Use 1 for surface-only detection smoke.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Independent dates to process concurrently. Full-depth global runs normally use 2-3.",
    )
    parser.add_argument(
        "--convolution-engine",
        choices=["direct", "fft"],
        default="fft",
        help="FFT uses the same discrete kernels and boundary modes as direct convolution.",
    )
    parser.add_argument(
        "--compression-level",
        type=int,
        default=1,
        help="Lossless NetCDF4 zlib compression level; 1 avoids compression dominating runtime.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def build_meso_filter(
    *,
    input_root: Path,
    output_root: Path,
    start: date,
    end: date,
    available_start: date,
    available_end: date,
    temporal_window_days: int,
    small_cutoff_km: float,
    large_cutoff_km: float,
    filter_mode: str,
    spatial_kernel: str,
    science_tag: str,
    large_cutoff_mode: str,
    adaptive_large_cutoff_min_km: float,
    adaptive_large_cutoff_max_km: float,
    rossby_radius_path: Path,
    rossby_small_factor: float,
    rossby_large_factor: float,
    rossby_min_km: float,
    rossby_max_km: float,
    rossby_large_fixed_km: float,
    zonal_scale_mode: str,
    meridional_scale_mode: str,
    min_valid_weight_fraction: float,
    max_depth_layers: int,
    workers: int = 1,
    convolution_engine: str = "fft",
    compression_level: int = 1,
    overwrite: bool,
    _manifest_name: str = "build_ofes_meso_filter_manifest.json",
) -> list[Path]:
    selected_days = date_range(start, end)
    if workers < 1:
        raise ValueError("workers must be >= 1")
    if not 0 <= compression_level <= 9:
        raise ValueError("compression_level must be in [0, 9]")
    if workers > 1 and len(selected_days) > 1:
        return build_meso_filter_parallel(
            input_root=input_root,
            output_root=output_root,
            selected_days=selected_days,
            available_start=available_start,
            available_end=available_end,
            temporal_window_days=temporal_window_days,
            small_cutoff_km=small_cutoff_km,
            large_cutoff_km=large_cutoff_km,
            filter_mode=filter_mode,
            spatial_kernel=spatial_kernel,
            science_tag=science_tag,
            large_cutoff_mode=large_cutoff_mode,
            adaptive_large_cutoff_min_km=adaptive_large_cutoff_min_km,
            adaptive_large_cutoff_max_km=adaptive_large_cutoff_max_km,
            rossby_radius_path=rossby_radius_path,
            rossby_small_factor=rossby_small_factor,
            rossby_large_factor=rossby_large_factor,
            rossby_min_km=rossby_min_km,
            rossby_max_km=rossby_max_km,
            rossby_large_fixed_km=rossby_large_fixed_km,
            zonal_scale_mode=zonal_scale_mode,
            meridional_scale_mode=meridional_scale_mode,
            min_valid_weight_fraction=min_valid_weight_fraction,
            max_depth_layers=max_depth_layers,
            workers=workers,
            convolution_engine=convolution_engine,
            compression_level=compression_level,
            overwrite=overwrite,
            manifest_name=_manifest_name,
        )
    output_root.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for target_day in selected_days:
        window_days = available_running_window(target_day, temporal_window_days, available_start, available_end)
        first_path = input_daily_path(input_root, window_days[0])
        with Dataset(first_path) as sample:
            lon = np.asarray(sample.variables["longitude"][:], dtype="f8")
            lat = np.asarray(sample.variables["latitude"][:], dtype="f8")
            depth_all = np.asarray(sample.variables["depth"][:], dtype="f4")
            depth_count = min(max_depth_layers, len(depth_all))
            depth = depth_all[:depth_count]
            attrs = read_source_attrs(sample)
        if large_cutoff_mode == "rossby_radius":
            r1_by_lat = load_rossby_radius_profile(rossby_radius_path, lat)
            small_cutoff_by_lat = rossby_cutoff_profile(
                r1_by_lat,
                factor=rossby_small_factor,
                min_km=rossby_min_km,
                max_km=rossby_max_km,
            )
            if rossby_large_fixed_km > 0:
                large_cutoff_by_lat = np.full(
                    len(lat),
                    float(rossby_large_fixed_km),
                    dtype="f8",
                )
            else:
                large_cutoff_by_lat = rossby_cutoff_profile(
                    r1_by_lat,
                    factor=rossby_large_factor,
                    min_km=rossby_min_km,
                    max_km=rossby_max_km,
                )
            large_cutoff_by_lat = np.maximum(
                large_cutoff_by_lat,
                np.minimum(small_cutoff_by_lat + 1.0, rossby_max_km),
            )
            small_cutoff_by_lat = np.minimum(small_cutoff_by_lat, large_cutoff_by_lat - 1.0)
            small_cutoff_for_filter: float | np.ndarray = small_cutoff_by_lat
        elif large_cutoff_mode == "rossby_lower_latadaptive_upper":
            r1_by_lat = load_rossby_radius_profile(rossby_radius_path, lat)
            small_cutoff_by_lat = rossby_cutoff_profile(
                r1_by_lat,
                factor=rossby_small_factor,
                min_km=rossby_min_km,
                max_km=rossby_max_km,
            )
            large_cutoff_by_lat = large_cutoff_profile(
                lat,
                mode="latitude_adaptive",
                fixed_km=large_cutoff_km,
                adaptive_min_km=adaptive_large_cutoff_min_km,
                adaptive_max_km=adaptive_large_cutoff_max_km,
            )
            large_cutoff_by_lat = np.maximum(
                large_cutoff_by_lat,
                np.minimum(small_cutoff_by_lat + 1.0, adaptive_large_cutoff_max_km),
            )
            small_cutoff_by_lat = np.minimum(small_cutoff_by_lat, large_cutoff_by_lat - 1.0)
            small_cutoff_for_filter = small_cutoff_by_lat
        else:
            small_cutoff_for_filter = small_cutoff_km
            large_cutoff_by_lat = large_cutoff_profile(
                lat,
                mode=large_cutoff_mode,
                fixed_km=large_cutoff_km,
                adaptive_min_km=adaptive_large_cutoff_min_km,
                adaptive_max_km=adaptive_large_cutoff_max_km,
            )
        science_tag = science_tag or science_tag_for(
            temporal_window_days,
            small_cutoff_km,
            large_cutoff_mode,
            large_cutoff_km,
            adaptive_large_cutoff_min_km,
            adaptive_large_cutoff_max_km,
            rossby_small_factor,
            rossby_large_factor,
            rossby_large_fixed_km,
            filter_mode,
            spatial_kernel,
        )

        out_path = output_root / f"global_phy_{target_day:%Y%m%d}.nc"
        if out_path.exists() and not overwrite:
            raise FileExistsError(f"{out_path} exists. Use --overwrite to replace it.")

        filter_plans = build_filter_plans(
            lon,
            lat,
            small_cutoff_for_filter,
            large_cutoff_by_lat,
            filter_mode,
            zonal_scale_mode,
            meridional_scale_mode,
            spatial_kernel,
            convolution_engine,
        )
        part_path = Path(f"{out_path}.part")
        part_path.unlink(missing_ok=True)
        try:
            with ExitStack() as stack:
                datasets = [stack.enter_context(Dataset(input_daily_path(input_root, day))) for day in window_days]
                slow_ssh = temporal_mean_surface_from_datasets(datasets, "zos_glor")
                meso_ssh = horizontal_scale_filter(
                    slow_ssh,
                    lon,
                    lat,
                    small_cutoff_for_filter,
                    large_cutoff_by_lat,
                    filter_mode,
                    zonal_scale_mode,
                    meridional_scale_mode,
                    min_valid_weight_fraction,
                    spatial_kernel,
                    convolution_engine,
                    filter_plans,
                )

                def filtered_velocity_layers():
                    for k in range(depth_count):
                        slow_u, slow_v = temporal_mean_velocity_pair_from_datasets(datasets, k)
                        meso_u, meso_v = horizontal_scale_filter_pair(
                            slow_u,
                            slow_v,
                            lon,
                            lat,
                            small_cutoff_for_filter,
                            large_cutoff_by_lat,
                            filter_mode,
                            zonal_scale_mode,
                            meridional_scale_mode,
                            min_valid_weight_fraction,
                            spatial_kernel,
                            convolution_engine,
                            filter_plans,
                        )
                        if (k + 1) % 8 == 0 or k + 1 == depth_count:
                            print(
                                f"[ofes-meso-filter] {target_day.isoformat()} depth {k + 1}/{depth_count}",
                                flush=True,
                            )
                        yield meso_u, meso_v

                write_daily_netcdf(
                    part_path,
                    target_day=target_day,
                    lon=lon,
                    lat=lat,
                    depth=depth,
                    ssh=meso_ssh,
                    u=None,
                    v=None,
                    velocity_layers=filtered_velocity_layers(),
                    compression_level=compression_level,
                    attrs=attrs,
                    window_days=window_days,
                    temporal_window_days=temporal_window_days,
                    small_cutoff_km=small_cutoff_km,
                    large_cutoff_km=large_cutoff_km,
                    filter_mode=filter_mode,
                    spatial_kernel=spatial_kernel,
                    large_cutoff_mode=large_cutoff_mode,
                    adaptive_large_cutoff_min_km=adaptive_large_cutoff_min_km,
                    adaptive_large_cutoff_max_km=adaptive_large_cutoff_max_km,
                    rossby_radius_path=rossby_radius_path,
                    rossby_small_factor=rossby_small_factor,
                    rossby_large_factor=rossby_large_factor,
                    rossby_min_km=rossby_min_km,
                    rossby_max_km=rossby_max_km,
                    rossby_large_fixed_km=rossby_large_fixed_km,
                    large_cutoff_by_lat=large_cutoff_by_lat,
                    small_cutoff_by_lat=small_cutoff_for_filter,
                    zonal_scale_mode=zonal_scale_mode,
                    meridional_scale_mode=meridional_scale_mode,
                    min_valid_weight_fraction=min_valid_weight_fraction,
                    science_tag=science_tag,
                    convolution_engine=convolution_engine,
                )
            part_path.replace(out_path)
        except BaseException:
            part_path.unlink(missing_ok=True)
            raise
        written.append(out_path)
        print(f"[ofes-meso-filter] wrote {target_day.isoformat()} -> {out_path}", flush=True)

    manifest = {
        "input_root": str(input_root),
        "output_root": str(output_root),
        "start": start.isoformat(),
        "end": end.isoformat(),
        "available_start": available_start.isoformat(),
        "available_end": available_end.isoformat(),
        "temporal_window_days": temporal_window_days,
        "small_cutoff_km": small_cutoff_km,
        "large_cutoff_km": large_cutoff_km,
        "filter_mode": filter_mode,
        "spatial_kernel": spatial_kernel,
        "large_cutoff_mode": large_cutoff_mode,
        "adaptive_large_cutoff_min_km": adaptive_large_cutoff_min_km,
        "adaptive_large_cutoff_max_km": adaptive_large_cutoff_max_km,
        "adaptive_large_cutoff_formula": "L_hi(lat)=min_km+(max_km-min_km)*cos(lat)^2",
        "rossby_radius_path": str(rossby_radius_path),
        "rossby_small_factor": rossby_small_factor,
        "rossby_large_factor": rossby_large_factor,
        "rossby_min_km": rossby_min_km,
        "rossby_max_km": rossby_max_km,
        "rossby_large_fixed_km": rossby_large_fixed_km,
        "zonal_scale_mode": zonal_scale_mode,
        "meridional_scale_mode": meridional_scale_mode,
        "min_valid_weight_fraction": min_valid_weight_fraction,
        "convolution_engine": convolution_engine,
        "date_workers": workers,
        "netcdf_compression_level": compression_level,
        "io_policy": "input datasets opened once per date; output written layer-by-layer without velocity memmaps",
        "large_cutoff_actual_min_km": float(np.nanmin(large_cutoff_by_lat)),
        "large_cutoff_actual_max_km": float(np.nanmax(large_cutoff_by_lat)),
        "spatial_band": spatial_band_label(
            small_cutoff_km,
            large_cutoff_km,
            large_cutoff_mode,
            adaptive_large_cutoff_max_km,
            filter_mode,
            rossby_small_factor,
            rossby_large_factor,
            rossby_large_fixed_km,
        ),
        "science_tag": science_tag,
        "baseline_definition": attrs.get("global_baseline_definition", "unspecified"),
        "baseline_formula": attrs.get("global_baseline_formula", "unspecified"),
        "annual_mss_path": attrs.get("global_annual_mss_path", ""),
        "seasonal_cycle_policy": attrs.get("global_seasonal_cycle_policy", "unspecified"),
        "daily_global_mean_removal": attrs.get("global_daily_global_mean_removal", "unspecified"),
        "max_depth_layers": max_depth_layers,
        "output_template": "global_phy_{yyyymmdd}.nc",
        "outputs": [str(path) for path in written],
        "note": "Diagnostic mesoscale filter: optional available-day running mean plus spatial scale separation. High-pass/band-pass suppresses large-scale barotropic/background signals but is not harmonic detiding and not a 30-180 day bandpass.",
    }
    (output_root / _manifest_name).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return written


def build_meso_filter_parallel(
    *,
    selected_days: list[date],
    workers: int,
    manifest_name: str,
    **kwargs,
) -> list[Path]:
    """Run independent dates in isolated processes and merge their manifests."""
    output_root = Path(kwargs["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    day_manifests: dict[date, Path] = {}
    written: list[Path] = []
    with ProcessPoolExecutor(max_workers=min(workers, len(selected_days))) as pool:
        futures = {}
        for target_day in selected_days:
            day_manifest = output_root / f".build_ofes_meso_filter_{target_day:%Y%m%d}.json"
            day_manifests[target_day] = day_manifest
            future = pool.submit(
                build_meso_filter,
                **kwargs,
                start=target_day,
                end=target_day,
                workers=1,
                _manifest_name=day_manifest.name,
            )
            futures[future] = target_day
        for future in as_completed(futures):
            target_day = futures[future]
            paths = future.result()
            written.extend(paths)
            print(f"[ofes-meso-filter] completed date worker {target_day.isoformat()}", flush=True)

    first_manifest = day_manifests[selected_days[0]]
    manifest = json.loads(first_manifest.read_text(encoding="utf-8"))
    manifest.update(
        {
            "start": selected_days[0].isoformat(),
            "end": selected_days[-1].isoformat(),
            "date_workers": min(workers, len(selected_days)),
            "outputs": [str(path) for path in sorted(written)],
            "parallel_schedule": "one isolated process per date; bounded process pool",
        }
    )
    (output_root / manifest_name).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    for day_manifest in day_manifests.values():
        day_manifest.unlink(missing_ok=True)
    return sorted(written)


def input_daily_path(root: Path, day: date) -> Path:
    path = root / f"global_phy_{day:%Y%m%d}.nc"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def read_source_attrs(ds: Dataset) -> dict[str, str]:
    attrs: dict[str, str] = {}
    for name in ["time", "zos_glor", "uo_glor", "vo_glor"]:
        if name not in ds.variables:
            continue
        var = ds.variables[name]
        for attr in ["units", "calendar", "long_name"]:
            if hasattr(var, attr):
                attrs[f"{name}_{attr}"] = str(getattr(var, attr))
    for attr in [
        "ssh_definition",
        "baseline_definition",
        "baseline_formula",
        "annual_mss_path",
        "annual_mss_months",
        "annual_mss_maximum_valid_day_weight",
        "seasonal_cycle_policy",
        "daily_global_mean_removal",
        "velocity_policy",
        "grid_note",
    ]:
        if hasattr(ds, attr):
            attrs[f"global_{attr}"] = str(getattr(ds, attr))
    return attrs


def temporal_mean_surface(root: Path, days: list[date], variable: str) -> np.ndarray:
    total: np.ndarray | None = None
    count: np.ndarray | None = None
    for day in days:
        with Dataset(input_daily_path(root, day)) as ds:
            layer = np.asarray(ds.variables[variable][0, :, :], dtype="f4")
        total, count = accumulate(total, count, layer)
    return divide_mean(total, count)


def temporal_mean_surface_from_datasets(datasets: list[Dataset], variable: str) -> np.ndarray:
    total: np.ndarray | None = None
    count: np.ndarray | None = None
    for ds in datasets:
        layer = np.asarray(ds.variables[variable][0, :, :], dtype="f4")
        total, count = accumulate(total, count, layer)
    return divide_mean(total, count)


def temporal_mean_velocity_layer(root: Path, days: list[date], variable: str, depth_index: int) -> np.ndarray:
    total: np.ndarray | None = None
    count: np.ndarray | None = None
    for day in days:
        with Dataset(input_daily_path(root, day)) as ds:
            layer = np.asarray(ds.variables[variable][0, depth_index, :, :], dtype="f4")
        total, count = accumulate(total, count, layer)
    return divide_mean(total, count)


def temporal_mean_velocity_pair_from_datasets(
    datasets: list[Dataset], depth_index: int
) -> tuple[np.ndarray, np.ndarray]:
    total_u: np.ndarray | None = None
    count_u: np.ndarray | None = None
    total_v: np.ndarray | None = None
    count_v: np.ndarray | None = None
    for ds in datasets:
        u = np.asarray(ds.variables["uo_glor"][0, depth_index, :, :], dtype="f4")
        v = np.asarray(ds.variables["vo_glor"][0, depth_index, :, :], dtype="f4")
        total_u, count_u = accumulate(total_u, count_u, u)
        total_v, count_v = accumulate(total_v, count_v, v)
    return divide_mean(total_u, count_u), divide_mean(total_v, count_v)


def accumulate(total: np.ndarray | None, count: np.ndarray | None, layer: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    layer = np.asarray(layer, dtype="f4")
    layer[np.abs(layer) > 1.0e20] = np.nan
    finite = np.isfinite(layer)
    if total is None or count is None:
        total = np.zeros(layer.shape, dtype="f8")
        count = np.zeros(layer.shape, dtype="f8")
    total[finite] += layer[finite]
    count[finite] += 1.0
    return total, count


def divide_mean(total: np.ndarray | None, count: np.ndarray | None) -> np.ndarray:
    if total is None or count is None:
        raise ValueError("No layers accumulated")
    return np.divide(total, count, out=np.full_like(total, np.nan, dtype="f8"), where=count > 0).astype("f4")


class SeparableFFTLowpassPlan:
    """Cached FFT representation of the existing discrete separable kernel."""

    def __init__(
        self,
        lon: np.ndarray,
        lat: np.ndarray,
        fwhm_km: float | np.ndarray,
        zonal_scale_mode: str,
        meridional_scale_mode: str,
        spatial_kernel: str,
    ) -> None:
        if meridional_scale_mode != "median":
            raise ValueError("FFT convolution currently requires meridional_scale_mode='median'")
        dlat = float(np.nanmedian(np.diff(lat)))
        dlon = float(np.nanmedian(np.diff(lon)))
        fwhm_by_lat = np.asarray(fwhm_km, dtype="f8")
        if fwhm_by_lat.ndim == 0:
            fwhm_by_lat = np.full(len(lat), float(fwhm_by_lat), dtype="f8")
        if fwhm_by_lat.shape != (len(lat),):
            raise ValueError(f"fwhm_km must be scalar or len(lat), got shape {fwhm_by_lat.shape}")
        sigma_km = fwhm_by_lat / 2.354820045
        sigma_y = float(np.nanmedian(np.maximum(0.01, sigma_km / (111.32 * abs(dlat)))))
        self.y_kernel = lowpass_kernel(spatial_kernel, sigma_y)

        nx = len(lon)
        circular_kernels = np.zeros((len(lat), nx), dtype="f8")
        for j, latitude in enumerate(lat):
            if zonal_scale_mode == "degree":
                sigma_x = max(0.01, float(sigma_km[j]) / (111.32 * abs(dlon)))
            elif zonal_scale_mode == "km":
                coslat = max(0.12, abs(float(np.cos(np.deg2rad(latitude)))))
                sigma_x = max(0.01, float(sigma_km[j]) / (111.32 * coslat * abs(dlon)))
            else:
                raise ValueError(f"Unsupported zonal_scale_mode {zonal_scale_mode!r}")
            kernel = lowpass_kernel(spatial_kernel, sigma_x)
            offsets = np.arange(-(kernel.size // 2), kernel.size // 2 + 1)
            np.add.at(circular_kernels[j], offsets % nx, kernel)
        self.zonal_spectrum = np.fft.rfft(circular_kernels, axis=1)
        self.nx = nx

    def convolve(self, values: np.ndarray) -> np.ndarray:
        radius = self.y_kernel.size // 2
        padded = np.pad(np.asarray(values, dtype="f8"), ((radius, radius), (0, 0)), mode="edge")
        meridional = fftconvolve(
            padded,
            self.y_kernel[:, np.newaxis],
            mode="valid",
            axes=0,
        )
        spectrum = np.fft.rfft(meridional, axis=1)
        return np.fft.irfft(spectrum * self.zonal_spectrum, n=self.nx, axis=1)


def build_filter_plans(
    lon: np.ndarray,
    lat: np.ndarray,
    small_km: float | np.ndarray,
    large_km: float | np.ndarray,
    filter_mode: str,
    zonal_scale_mode: str,
    meridional_scale_mode: str,
    spatial_kernel: str,
    convolution_engine: str,
) -> dict[str, SeparableFFTLowpassPlan]:
    if convolution_engine != "fft" or meridional_scale_mode != "median":
        return {}
    plans = {
        "large": SeparableFFTLowpassPlan(
            lon, lat, large_km, zonal_scale_mode, meridional_scale_mode, spatial_kernel
        )
    }
    if filter_mode == "bandpass":
        plans["small"] = SeparableFFTLowpassPlan(
            lon, lat, small_km, zonal_scale_mode, meridional_scale_mode, spatial_kernel
        )
    return plans


def horizontal_scale_filter(
    field: np.ndarray,
    lon: np.ndarray,
    lat: np.ndarray,
    small_km: float | np.ndarray,
    large_km: float | np.ndarray,
    filter_mode: str,
    zonal_scale_mode: str = "km",
    meridional_scale_mode: str = "median",
    min_valid_weight_fraction: float = 0.0,
    spatial_kernel: str = "gaussian",
    convolution_engine: str = "fft",
    filter_plans: dict[str, SeparableFFTLowpassPlan] | None = None,
) -> np.ndarray:
    if filter_mode == "highpass":
        large = nan_gaussian_lowpass(
            field, lon, lat, large_km, zonal_scale_mode,
            meridional_scale_mode, min_valid_weight_fraction, spatial_kernel,
            convolution_engine=convolution_engine,
            plan=(filter_plans or {}).get("large"),
        )
        out = np.asarray(field, dtype="f4") - large
        out[~np.isfinite(field) | ~np.isfinite(large)] = np.nan
        return out.astype("f4")
    if filter_mode != "bandpass":
        raise ValueError(f"Unsupported filter_mode {filter_mode!r}")
    small = nan_gaussian_lowpass(
        field, lon, lat, small_km, zonal_scale_mode,
        meridional_scale_mode, min_valid_weight_fraction, spatial_kernel,
        convolution_engine=convolution_engine,
        plan=(filter_plans or {}).get("small"),
    )
    large = nan_gaussian_lowpass(
        field, lon, lat, large_km, zonal_scale_mode,
        meridional_scale_mode, min_valid_weight_fraction, spatial_kernel,
        convolution_engine=convolution_engine,
        plan=(filter_plans or {}).get("large"),
    )
    out = small - large
    out[~np.isfinite(small) | ~np.isfinite(large)] = np.nan
    return out.astype("f4")


def nan_gaussian_lowpass(
    field: np.ndarray,
    lon: np.ndarray,
    lat: np.ndarray,
    fwhm_km: float | np.ndarray,
    zonal_scale_mode: str = "km",
    meridional_scale_mode: str = "median",
    min_valid_weight_fraction: float = 0.0,
    spatial_kernel: str = "gaussian",
    *,
    convolution_engine: str = "fft",
    plan: SeparableFFTLowpassPlan | None = None,
    denominator: np.ndarray | None = None,
    return_denominator: bool = False,
) -> np.ndarray:
    if not 0.0 <= min_valid_weight_fraction < 1.0:
        raise ValueError("min_valid_weight_fraction must be in [0, 1)")
    arr = np.asarray(field, dtype="f8")
    finite = np.isfinite(arr)
    values = np.where(finite, arr, 0.0)
    weights = finite.astype("f8")

    if convolution_engine == "fft" and meridional_scale_mode == "median":
        fft_plan = plan or SeparableFFTLowpassPlan(
            lon, lat, fwhm_km, zonal_scale_mode, meridional_scale_mode, spatial_kernel
        )
        out_num = fft_plan.convolve(values)
        out_den = denominator if denominator is not None else fft_plan.convolve(weights)
        valid = out_den > max(1.0e-6, min_valid_weight_fraction)
        result = np.divide(out_num, out_den, out=np.full_like(out_num, np.nan), where=valid)
        if return_denominator:
            return result, out_den
        return result
    if convolution_engine not in {"direct", "fft"}:
        raise ValueError(f"Unsupported convolution_engine {convolution_engine!r}")

    dlat = float(np.nanmedian(np.diff(lat)))
    dlon = float(np.nanmedian(np.diff(lon)))
    fwhm_by_lat = np.asarray(fwhm_km, dtype="f8")
    if fwhm_by_lat.ndim == 0:
        fwhm_by_lat = np.full(len(lat), float(fwhm_by_lat), dtype="f8")
    if fwhm_by_lat.shape != (len(lat),):
        raise ValueError(f"fwhm_km must be scalar or len(lat), got shape {fwhm_by_lat.shape}")
    sigma_km_by_lat = fwhm_by_lat / 2.354820045
    sigma_y_by_lat = np.maximum(0.01, sigma_km_by_lat / (111.32 * abs(dlat)))
    if spatial_kernel not in {"gaussian", "lanczos", "bessel"}:
        raise ValueError(f"Unsupported spatial_kernel {spatial_kernel!r}")
    if meridional_scale_mode == "median":
        sigma_y = float(np.nanmedian(sigma_y_by_lat))
        values = convolve_axis(values, lowpass_kernel(spatial_kernel, sigma_y), axis=0, mode="nearest")
        weights = convolve_axis(weights, lowpass_kernel(spatial_kernel, sigma_y), axis=0, mode="nearest")
    elif meridional_scale_mode == "local":
        values, weights = local_meridional_kernel(values, weights, sigma_y_by_lat, spatial_kernel)
    else:
        raise ValueError(f"Unsupported meridional_scale_mode {meridional_scale_mode!r}")

    out_num = np.empty_like(values)
    out_den = np.empty_like(weights)
    for j, latitude in enumerate(lat):
        if zonal_scale_mode == "degree":
            sigma_x = max(0.01, float(sigma_km_by_lat[j]) / (111.32 * abs(dlon)))
        elif zonal_scale_mode == "km":
            coslat = max(0.12, abs(float(np.cos(np.deg2rad(latitude)))))
            sigma_x = max(0.01, float(sigma_km_by_lat[j]) / (111.32 * coslat * abs(dlon)))
        else:
            raise ValueError(f"Unsupported zonal_scale_mode {zonal_scale_mode!r}")
        kernel = lowpass_kernel(spatial_kernel, sigma_x)
        out_num[j, :] = convolve_axis(values[j, :], kernel, axis=0, mode="wrap")
        out_den[j, :] = convolve_axis(weights[j, :], kernel, axis=0, mode="wrap")

    valid = out_den > max(1.0e-6, min_valid_weight_fraction)
    result = np.divide(out_num, out_den, out=np.full_like(out_num, np.nan), where=valid)
    if return_denominator:
        return result, out_den
    return result


def horizontal_scale_filter_pair(
    u: np.ndarray,
    v: np.ndarray,
    lon: np.ndarray,
    lat: np.ndarray,
    small_km: float | np.ndarray,
    large_km: float | np.ndarray,
    filter_mode: str,
    zonal_scale_mode: str,
    meridional_scale_mode: str,
    min_valid_weight_fraction: float,
    spatial_kernel: str,
    convolution_engine: str,
    filter_plans: dict[str, SeparableFFTLowpassPlan] | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Filter a velocity pair while sharing the land-mask denominator."""
    if filter_mode != "highpass" or convolution_engine != "fft" or meridional_scale_mode != "median":
        return (
            horizontal_scale_filter(
                u, lon, lat, small_km, large_km, filter_mode, zonal_scale_mode,
                meridional_scale_mode, min_valid_weight_fraction, spatial_kernel,
                convolution_engine, filter_plans,
            ),
            horizontal_scale_filter(
                v, lon, lat, small_km, large_km, filter_mode, zonal_scale_mode,
                meridional_scale_mode, min_valid_weight_fraction, spatial_kernel,
                convolution_engine, filter_plans,
            ),
        )
    plan = (filter_plans or {}).get("large")
    if plan is None:
        plan = SeparableFFTLowpassPlan(
            lon, lat, large_km, zonal_scale_mode, meridional_scale_mode, spatial_kernel
        )
    low_u, denominator_u = nan_gaussian_lowpass(
        u, lon, lat, large_km, zonal_scale_mode, meridional_scale_mode,
        min_valid_weight_fraction, spatial_kernel, convolution_engine="fft",
        plan=plan, return_denominator=True,
    )
    shared = denominator_u if np.array_equal(np.isfinite(u), np.isfinite(v)) else None
    low_v = nan_gaussian_lowpass(
        v, lon, lat, large_km, zonal_scale_mode, meridional_scale_mode,
        min_valid_weight_fraction, spatial_kernel, convolution_engine="fft",
        plan=plan, denominator=shared,
    )
    out_u = np.asarray(u, dtype="f4") - low_u
    out_v = np.asarray(v, dtype="f4") - low_v
    out_u[~np.isfinite(u) | ~np.isfinite(low_u)] = np.nan
    out_v[~np.isfinite(v) | ~np.isfinite(low_v)] = np.nan
    return out_u.astype("f4"), out_v.astype("f4")


def convolve_axis(values: np.ndarray, kernel: np.ndarray, *, axis: int, mode: str) -> np.ndarray:
    """One separable, zero-phase convolution used for values and validity weights."""
    return convolve1d(values, kernel, axis=axis, mode=mode)


def lowpass_kernel(kind: str, sigma_cells: float) -> np.ndarray:
    """Return a symmetric unit-sum low-pass kernel at a Gaussian-equivalent scale."""
    sigma = max(0.25, float(sigma_cells))
    if kind == "gaussian":
        radius = max(1, int(np.ceil(3.0 * sigma)))
        offsets = np.arange(-radius, radius + 1, dtype="f8")
        kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
    else:
        # The requested cutoff is expressed as Gaussian FWHM.  Convert it to
        # the Gaussian half-power wavelength so all three kernels use the same
        # physical scale convention.
        half_power_wavelength = 2.0 * np.pi * sigma / np.sqrt(np.log(2.0))
        if kind == "lanczos":
            radius = max(3, int(np.ceil(3.0 * half_power_wavelength)))
            offsets = np.arange(-radius, radius + 1, dtype="f8")
            cutoff = 1.0 / max(half_power_wavelength, 2.05)
            kernel = 2.0 * cutoff * np.sinc(2.0 * cutoff * offsets)
            kernel *= np.sinc(offsets / float(radius + 1))
        elif kind == "bessel":
            cutoff = min(0.49, 1.0 / max(half_power_wavelength, 2.05))
            sos = bessel(3, 2.0 * cutoff, btype="lowpass", output="sos", norm="mag")
            size = 1
            while size < max(257, int(np.ceil(24.0 * half_power_wavelength))):
                size *= 2
            _, response = sosfreqz(sos, worN=size // 2 + 1, fs=1.0)
            impulse = np.fft.fftshift(np.fft.irfft(np.abs(response) ** 2, n=size))
            center = size // 2
            keep = np.flatnonzero(np.abs(impulse) > np.max(np.abs(impulse)) * 1.0e-5)
            radius = max(2, min(center - int(keep.min()), int(keep.max()) - center))
            kernel = impulse[center - radius : center + radius + 1]
        else:
            raise ValueError(f"Unsupported spatial_kernel {kind!r}")
    total = float(kernel.sum())
    if not np.isfinite(total) or abs(total) < 1.0e-12:
        raise ValueError(f"Invalid {kind} low-pass kernel")
    return (kernel / total).astype("f8")


def local_meridional_kernel(
    values: np.ndarray,
    weights: np.ndarray,
    sigma_by_lat: np.ndarray,
    spatial_kernel: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a Gaussian y-kernel at the physical width of each output latitude."""
    nlat = values.shape[0]
    out_values = np.empty_like(values)
    out_weights = np.empty_like(weights)
    for j, sigma in enumerate(sigma_by_lat):
        kernel = lowpass_kernel(spatial_kernel, float(sigma))
        radius = kernel.size // 2
        indices = np.clip(np.arange(j - radius, j + radius + 1), 0, nlat - 1)
        out_values[j, :] = kernel @ values[indices, :]
        out_weights[j, :] = kernel @ weights[indices, :]
    return out_values, out_weights


def large_cutoff_profile(
    lat: np.ndarray,
    *,
    mode: str,
    fixed_km: float,
    adaptive_min_km: float,
    adaptive_max_km: float,
) -> np.ndarray:
    if mode == "fixed":
        return np.full(len(lat), float(fixed_km), dtype="f8")
    if adaptive_max_km < adaptive_min_km:
        raise ValueError("adaptive max cutoff must be >= adaptive min cutoff")
    cos2 = np.cos(np.deg2rad(np.abs(lat))) ** 2
    return adaptive_min_km + (adaptive_max_km - adaptive_min_km) * cos2


def load_rossby_radius_profile(path: Path, target_lat: np.ndarray) -> np.ndarray:
    """Return a Chelton 1998 first-baroclinic radius profile at target latitudes."""
    if not Path(path).exists():
        raise FileNotFoundError(path)
    with Dataset(path) as ds:
        source_lat = np.asarray(ds.variables["lat"][:], dtype="f8")
        radius = np.asarray(ds.variables["rossby_radius"][:], dtype="f8")
    radius = np.where(np.abs(radius) > 1.0e20, np.nan, radius)
    median = np.nanmedian(radius, axis=1)
    finite = np.isfinite(median)
    if not np.any(finite):
        raise ValueError(f"No finite Rossby radius values in {path}")
    source_lat = source_lat[finite]
    median = median[finite]
    order = np.argsort(source_lat)
    source_lat = source_lat[order]
    median = median[order]
    return np.interp(
        np.asarray(target_lat, dtype="f8"),
        source_lat,
        median,
        left=median[0],
        right=median[-1],
    )


def rossby_cutoff_profile(
    rossby_radius_km: np.ndarray,
    *,
    factor: float,
    min_km: float,
    max_km: float,
) -> np.ndarray:
    return np.clip(np.asarray(rossby_radius_km, dtype="f8") * float(factor), float(min_km), float(max_km))


def science_tag_for(
    temporal_window_days: int,
    small_cutoff_km: float,
    large_cutoff_mode: str,
    large_cutoff_km: float,
    adaptive_min_km: float,
    adaptive_max_km: float,
    rossby_small_factor: float,
    rossby_large_factor: float,
    rossby_large_fixed_km: float,
    filter_mode: str,
    spatial_kernel: str,
) -> str:
    suffix = f"_{spatial_kernel}"
    if large_cutoff_mode == "rossby_radius":
        if filter_mode == "bandpass" and rossby_large_fixed_km > 0:
            small = f"{rossby_small_factor:g}".replace(".", "p")
            upper = f"{rossby_large_fixed_km:g}".replace(".", "p")
            return f"rossby_radius_lower_r1x{small}_upper{upper}km{suffix}_diagnostic"
        if filter_mode == "highpass":
            factor = f"{rossby_large_factor:g}".replace(".", "p")
            return f"rossby_radius_highpass_r1x{factor}{suffix}_diagnostic"
        factors = f"{rossby_small_factor:g}_{rossby_large_factor:g}".replace(".", "p")
        return f"rossby_radius_bandpass_r1x{factors}{suffix}_diagnostic"
    if large_cutoff_mode == "rossby_lower_latadaptive_upper":
        small = f"{rossby_small_factor:g}".replace(".", "p")
        upper = f"{adaptive_min_km:g}_{adaptive_max_km:g}".replace(".", "p")
        return f"rossby_lower_r1x{small}_upper_latadaptive_{upper}{suffix}_diagnostic"
    if filter_mode == "highpass":
        return f"spatial_highpass_{large_cutoff_km:g}km{suffix}_diagnostic".replace(".", "p")
    if large_cutoff_mode == "latitude_adaptive":
        return f"meso{temporal_window_days}d_{small_cutoff_km:g}_{adaptive_max_km:g}km_lat_adaptive{suffix}_diagnostic".replace(".", "p")
    return f"meso{temporal_window_days}d_{small_cutoff_km:g}_{large_cutoff_km:g}km{suffix}_diagnostic".replace(".", "p")


def spatial_band_label(
    small_cutoff_km: float,
    large_cutoff_km: float,
    large_cutoff_mode: str,
    adaptive_large_cutoff_max_km: float,
    filter_mode: str,
    rossby_small_factor: float = 0.5,
    rossby_large_factor: float = 4.0,
    rossby_large_fixed_km: float = 0.0,
) -> str:
    if large_cutoff_mode == "rossby_radius":
        if filter_mode == "bandpass" and rossby_large_fixed_km > 0:
            return f"{rossby_small_factor:g}xR1-{rossby_large_fixed_km:g}km"
        return (
            f"{rossby_small_factor:g}xR1-{rossby_large_factor:g}xR1"
            if filter_mode == "bandpass"
            else f"highpass_{rossby_large_factor:g}xR1"
        )
    if large_cutoff_mode == "rossby_lower_latadaptive_upper":
        return f"{rossby_small_factor:g}xR1-{adaptive_large_cutoff_max_km:g}km_lat_adaptive"
    if filter_mode == "highpass":
        return f"highpass_{large_cutoff_km:g}km"
    if large_cutoff_mode == "fixed":
        return f"{small_cutoff_km:g}-{large_cutoff_km:g}km"
    return f"{small_cutoff_km:g}-{adaptive_large_cutoff_max_km:g}km_lat_adaptive"


def write_daily_netcdf(
    path: Path,
    *,
    target_day: date,
    lon: np.ndarray,
    lat: np.ndarray,
    depth: np.ndarray,
    ssh: np.ndarray,
    u: np.ndarray | None,
    v: np.ndarray | None,
    attrs: dict[str, str],
    window_days: list[date],
    temporal_window_days: int,
    small_cutoff_km: float,
    large_cutoff_km: float,
    filter_mode: str,
    spatial_kernel: str,
    large_cutoff_mode: str,
    adaptive_large_cutoff_min_km: float,
    adaptive_large_cutoff_max_km: float,
    rossby_radius_path: Path,
    rossby_small_factor: float,
    rossby_large_factor: float,
    rossby_min_km: float,
    rossby_max_km: float,
    rossby_large_fixed_km: float,
    large_cutoff_by_lat: np.ndarray,
    small_cutoff_by_lat: float | np.ndarray,
    zonal_scale_mode: str,
    meridional_scale_mode: str,
    min_valid_weight_fraction: float,
    science_tag: str,
    velocity_layers=None,
    compression_level: int = 1,
    convolution_engine: str = "direct",
) -> None:
    with Dataset(path, "w", format="NETCDF4") as ds:
        ds.createDimension("time", 1)
        ds.createDimension("depth", len(depth))
        ds.createDimension("latitude", len(lat))
        ds.createDimension("longitude", len(lon))

        time_var = ds.createVariable("time", "f8", ("time",))
        depth_var = ds.createVariable("depth", "f4", ("depth",))
        lat_var = ds.createVariable("latitude", "f4", ("latitude",))
        lon_var = ds.createVariable("longitude", "f4", ("longitude",))
        surface_chunks = (1, min(190, len(lat)), min(450, len(lon)))
        velocity_chunks = (1, 1, min(190, len(lat)), min(450, len(lon)))
        zos = ds.createVariable(
            "zos_glor", "f4", ("time", "latitude", "longitude"), zlib=True,
            complevel=compression_level, shuffle=True, chunksizes=surface_chunks,
        )
        uo = ds.createVariable(
            "uo_glor", "f4", ("time", "depth", "latitude", "longitude"), zlib=True,
            complevel=compression_level, shuffle=True, chunksizes=velocity_chunks,
        )
        vo = ds.createVariable(
            "vo_glor", "f4", ("time", "depth", "latitude", "longitude"), zlib=True,
            complevel=compression_level, shuffle=True, chunksizes=velocity_chunks,
        )

        time_var.units = attrs.get("time_units", f"days since {target_day.year:04d}-01-01 00:00:00")
        time_var.calendar = attrs.get("time_calendar", "standard")
        depth_var.units = "m"
        lat_var.units = "degrees_north"
        lon_var.units = "degrees_east"
        zos.units = attrs.get("zos_glor_units", "cm")
        uo.units = attrs.get("uo_glor_units", "m s-1")
        vo.units = attrs.get("vo_glor_units", "m s-1")
        zos.long_name = f"OFES SSH mesoscale diagnostic anomaly, running mean and horizontal {filter_mode}"
        uo.long_name = f"OFES zonal velocity mesoscale diagnostic anomaly, running mean and horizontal {filter_mode}"
        vo.long_name = f"OFES meridional velocity mesoscale diagnostic anomaly, running mean and horizontal {filter_mode}"

        ds.source = "Detection_for_OFES.tools.build_ofes_meso_filter"
        ds.science_tag = science_tag
        for attr in [
            "ssh_definition",
            "baseline_definition",
            "baseline_formula",
            "annual_mss_path",
            "annual_mss_months",
            "annual_mss_maximum_valid_day_weight",
            "seasonal_cycle_policy",
            "daily_global_mean_removal",
            "velocity_policy",
            "grid_note",
        ]:
            value = attrs.get(f"global_{attr}")
            if value is not None:
                ds.setncattr(attr, value)
        ds.temporal_filter = f"available-day running mean, nominal window {temporal_window_days} days"
        ds.temporal_filter_dates_used = ",".join(day.isoformat() for day in window_days)
        ds.horizontal_filter_mode = filter_mode
        ds.horizontal_filter_kernel = spatial_kernel
        ds.zonal_scale_mode = zonal_scale_mode
        ds.meridional_scale_mode = meridional_scale_mode
        ds.min_valid_weight_fraction = float(min_valid_weight_fraction)
        ds.convolution_engine = convolution_engine
        ds.convolution_equivalence = "same discrete kernel weights and boundary modes; FFT changes operation order only"
        ds.netcdf_compression_level = int(compression_level)
        if large_cutoff_mode == "rossby_radius":
            if filter_mode == "highpass":
                ds.horizontal_filter = (
                    f"field_minus_LP_{rossby_large_factor:g}xR1, "
                    f"Chelton 1998 first-baroclinic Rossby-radius profile, {spatial_kernel} Gaussian-equivalent scale"
                )
            else:
                if rossby_large_fixed_km > 0:
                    ds.horizontal_filter = (
                        f"LP_{rossby_small_factor:g}xR1_minus_LP_{rossby_large_fixed_km:g}km, "
                        f"Chelton 1998 first-baroclinic Rossby-radius lower cutoff, {spatial_kernel} Gaussian-equivalent scale"
                    )
                else:
                    ds.horizontal_filter = (
                        f"LP_{rossby_small_factor:g}xR1_minus_LP_{rossby_large_factor:g}xR1, "
                        f"Chelton 1998 first-baroclinic Rossby-radius profile, {spatial_kernel} Gaussian-equivalent scale"
                    )
        elif large_cutoff_mode == "rossby_lower_latadaptive_upper":
            ds.horizontal_filter = (
                f"LP_{rossby_small_factor:g}xR1_minus_LP_adaptive_{adaptive_large_cutoff_min_km:g}_{adaptive_large_cutoff_max_km:g}km, "
                f"Chelton 1998 lower cutoff with latitude-adaptive upper cutoff, {spatial_kernel} Gaussian-equivalent scale"
            )
        elif filter_mode == "highpass":
            ds.horizontal_filter = f"field_minus_LP_{large_cutoff_km:g}km, {spatial_kernel} Gaussian-equivalent scale"
        elif large_cutoff_mode == "latitude_adaptive":
            ds.horizontal_filter = (
                f"LP_{small_cutoff_km:g}km_minus_LP_adaptive_{adaptive_large_cutoff_min_km:g}_{adaptive_large_cutoff_max_km:g}km, "
                f"{spatial_kernel} Gaussian-equivalent scale"
            )
        else:
            ds.horizontal_filter = f"LP_{small_cutoff_km:g}km_minus_LP_{large_cutoff_km:g}km, {spatial_kernel} Gaussian-equivalent scale"
        ds.large_cutoff_mode = large_cutoff_mode
        ds.rossby_radius_path = str(rossby_radius_path)
        ds.rossby_small_factor = float(rossby_small_factor)
        ds.rossby_large_factor = float(rossby_large_factor)
        ds.rossby_min_km = float(rossby_min_km)
        ds.rossby_max_km = float(rossby_max_km)
        ds.rossby_large_fixed_km = float(rossby_large_fixed_km)
        ds.adaptive_large_cutoff_min_km = float(adaptive_large_cutoff_min_km)
        ds.adaptive_large_cutoff_max_km = float(adaptive_large_cutoff_max_km)
        ds.adaptive_large_cutoff_actual_min_km = float(np.nanmin(large_cutoff_by_lat))
        ds.adaptive_large_cutoff_actual_max_km = float(np.nanmax(large_cutoff_by_lat))
        if large_cutoff_mode in {"rossby_radius", "rossby_lower_latadaptive_upper"}:
            small_by_lat = np.asarray(small_cutoff_by_lat, dtype="f8")
            ds.rossby_small_cutoff_actual_min_km = float(np.nanmin(small_by_lat))
            ds.rossby_small_cutoff_actual_max_km = float(np.nanmax(small_by_lat))
            if large_cutoff_mode == "rossby_lower_latadaptive_upper":
                ds.adaptive_large_cutoff_formula = (
                    f"L(small)=clip({rossby_small_factor:g}*R1); "
                    f"L(large)=lat_adaptive({adaptive_large_cutoff_min_km:g},{adaptive_large_cutoff_max_km:g}); "
                    "R1 from Chelton et al. 1998"
                )
            elif rossby_large_fixed_km > 0:
                ds.adaptive_large_cutoff_formula = (
                    f"L(small)=clip({rossby_small_factor:g}*R1); "
                    f"L(large)=fixed {rossby_large_fixed_km:g} km; R1 from Chelton et al. 1998"
                )
            else:
                ds.adaptive_large_cutoff_formula = (
                    f"L(small)=clip({rossby_small_factor:g}*R1); "
                    f"L(large)=clip({rossby_large_factor:g}*R1); R1 from Chelton et al. 1998"
                )
        else:
            ds.adaptive_large_cutoff_formula = "L_hi(lat)=min_km+(max_km-min_km)*cos(lat)^2"
        ds.warning = "Diagnostic scale separation only; not strict detiding and not a 30-180 day bandpass."

        time_var[:] = np.asarray([(target_day - date(target_day.year, 1, 1)).days], dtype="f8")
        depth_var[:] = depth
        lat_var[:] = lat
        lon_var[:] = lon
        zos[0, :, :] = ssh
        if velocity_layers is not None:
            written_layers = 0
            for depth_index, (u_layer, v_layer) in enumerate(velocity_layers):
                if depth_index >= len(depth):
                    raise ValueError("velocity layer iterator produced too many layers")
                uo[0, depth_index, :, :] = u_layer
                vo[0, depth_index, :, :] = v_layer
                written_layers += 1
            if written_layers != len(depth):
                raise ValueError("velocity layer iterator produced too few layers")
        else:
            if u is None or v is None:
                raise ValueError("u/v arrays or velocity_layers are required")
            for depth_index in range(len(depth)):
                uo[0, depth_index, :, :] = u[depth_index]
                vo[0, depth_index, :, :] = v[depth_index]


def available_running_window(target: date, window_days: int, available_start: date, available_end: date) -> list[date]:
    before = (window_days - 1) // 2
    after = window_days - 1 - before
    start = max(available_start, target - timedelta(days=before))
    end = min(available_end, target + timedelta(days=after))
    return date_range(start, end)


def date_range(start: date, end: date) -> list[date]:
    if end < start:
        raise ValueError("end must be >= start")
    days: list[date] = []
    current = start
    while current <= end:
        days.append(current)
        current += timedelta(days=1)
    return days


def parse_date(value: str) -> date:
    return datetime.strptime(str(value)[:10], "%Y-%m-%d").date()


if __name__ == "__main__":
    main()
