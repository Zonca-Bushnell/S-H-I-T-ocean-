"""Canonical daily Gaussian 500-km high-pass for OFES SSH and velocity."""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
from netCDF4 import Dataset
from scipy.ndimage import convolve1d
from scipy.signal import fftconvolve


GAUSSIAN_FWHM_KM = 500.0


class SeparableFFTLowpassPlan:
    """Cached FFT representation of the canonical discrete Gaussian kernel."""

    def __init__(self, lon: np.ndarray, lat: np.ndarray, fwhm_km: float) -> None:
        dlat = float(np.nanmedian(np.diff(lat)))
        dlon = float(np.nanmedian(np.diff(lon)))
        sigma_km = float(fwhm_km) / 2.354820045
        self.y_kernel = gaussian_kernel(max(0.01, sigma_km / (111.32 * abs(dlat))))
        self.nx = len(lon)
        circular = np.zeros((len(lat), self.nx), dtype="f8")
        for j, latitude in enumerate(lat):
            coslat = max(0.12, abs(float(np.cos(np.deg2rad(latitude)))))
            sigma_x = max(0.01, sigma_km / (111.32 * coslat * abs(dlon)))
            kernel = gaussian_kernel(sigma_x)
            offsets = np.arange(-(kernel.size // 2), kernel.size // 2 + 1)
            np.add.at(circular[j], offsets % self.nx, kernel)
        self.zonal_spectrum = np.fft.rfft(circular, axis=1)

    def convolve(self, values: np.ndarray) -> np.ndarray:
        radius = self.y_kernel.size // 2
        padded = np.pad(np.asarray(values, dtype="f8"), ((radius, radius), (0, 0)), mode="edge")
        meridional = fftconvolve(padded, self.y_kernel[:, None], mode="valid", axes=0)
        return np.fft.irfft(
            np.fft.rfft(meridional, axis=1) * self.zonal_spectrum,
            n=self.nx,
            axis=1,
        )


def gaussian_kernel(sigma_cells: float) -> np.ndarray:
    sigma = max(0.25, float(sigma_cells))
    radius = max(1, int(np.ceil(3.0 * sigma)))
    offsets = np.arange(-radius, radius + 1, dtype="f8")
    kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
    return kernel / np.sum(kernel)


def nan_gaussian_lowpass(
    field: np.ndarray,
    lon: np.ndarray,
    lat: np.ndarray,
    fwhm_km: float = GAUSSIAN_FWHM_KM,
    zonal_scale_mode: str = "km",
    meridional_scale_mode: str = "median",
    min_valid_weight_fraction: float = 0.0,
    spatial_kernel: str = "gaussian",
    *,
    convolution_engine: str = "fft",
    plan: SeparableFFTLowpassPlan | None = None,
    denominator: np.ndarray | None = None,
    return_denominator: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """NaN-aware Gaussian low-pass with periodic longitude and edge latitude."""
    if zonal_scale_mode != "km" or meridional_scale_mode != "median" or spatial_kernel != "gaussian":
        raise ValueError("Canonical filtering requires km/median/Gaussian scale handling")
    if convolution_engine not in {"direct", "fft"}:
        raise ValueError("convolution_engine must be 'direct' or 'fft'")
    if not 0.0 <= min_valid_weight_fraction < 1.0:
        raise ValueError("min_valid_weight_fraction must be in [0, 1)")

    arr = np.asarray(field, dtype="f8")
    finite = np.isfinite(arr)
    values = np.where(finite, arr, 0.0)
    weights = finite.astype("f8")
    if convolution_engine == "fft":
        active_plan = plan or SeparableFFTLowpassPlan(lon, lat, fwhm_km)
        numerator = active_plan.convolve(values)
        support = denominator if denominator is not None else active_plan.convolve(weights)
    else:
        numerator = _direct_convolve(values, lon, lat, fwhm_km)
        support = denominator if denominator is not None else _direct_convolve(weights, lon, lat, fwhm_km)
    valid = support > max(1.0e-6, min_valid_weight_fraction)
    result = np.divide(numerator, support, out=np.full_like(numerator, np.nan), where=valid)
    return (result, support) if return_denominator else result


def _direct_convolve(values: np.ndarray, lon: np.ndarray, lat: np.ndarray, fwhm_km: float) -> np.ndarray:
    dlat = float(np.nanmedian(np.diff(lat)))
    dlon = float(np.nanmedian(np.diff(lon)))
    sigma_km = float(fwhm_km) / 2.354820045
    meridional = convolve1d(
        values,
        gaussian_kernel(max(0.01, sigma_km / (111.32 * abs(dlat)))),
        axis=0,
        mode="nearest",
    )
    output = np.empty_like(meridional)
    for j, latitude in enumerate(lat):
        coslat = max(0.12, abs(float(np.cos(np.deg2rad(latitude)))))
        kernel = gaussian_kernel(max(0.01, sigma_km / (111.32 * coslat * abs(dlon))))
        output[j] = convolve1d(meridional[j], kernel, axis=0, mode="wrap")
    return output


def gaussian_highpass(
    field: np.ndarray,
    lon: np.ndarray,
    lat: np.ndarray,
    *,
    cutoff_km: float = GAUSSIAN_FWHM_KM,
    engine: str = "fft",
    plan: SeparableFFTLowpassPlan | None = None,
    denominator: np.ndarray | None = None,
) -> np.ndarray:
    low = nan_gaussian_lowpass(
        field, lon, lat, cutoff_km, convolution_engine=engine, plan=plan, denominator=denominator
    )
    output = np.asarray(field, dtype="f4") - low
    output[~np.isfinite(field) | ~np.isfinite(low)] = np.nan
    return output.astype("f4")


def gaussian_highpass_pair(
    u: np.ndarray,
    v: np.ndarray,
    lon: np.ndarray,
    lat: np.ndarray,
    *,
    cutoff_km: float,
    engine: str,
    plan: SeparableFFTLowpassPlan,
) -> tuple[np.ndarray, np.ndarray]:
    low_u, support_u = nan_gaussian_lowpass(
        u, lon, lat, cutoff_km, convolution_engine=engine, plan=plan, return_denominator=True
    )
    shared = support_u if np.array_equal(np.isfinite(u), np.isfinite(v)) else None
    low_v = nan_gaussian_lowpass(
        v, lon, lat, cutoff_km, convolution_engine=engine, plan=plan, denominator=shared
    )
    out_u = np.asarray(u, dtype="f4") - low_u
    out_v = np.asarray(v, dtype="f4") - low_v
    out_u[~np.isfinite(u) | ~np.isfinite(low_u)] = np.nan
    out_v[~np.isfinite(v) | ~np.isfinite(low_v)] = np.nan
    return out_u.astype("f4"), out_v.astype("f4")


def _source_path(root: Path, current: date) -> Path:
    path = root / f"global_phy_{current:%Y%m%d}.nc"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _attrs(source: Dataset) -> dict[str, object]:
    return {name: source.getncattr(name) for name in source.ncattrs()}


def _build_day(
    input_root: Path,
    output_root: Path,
    current: date,
    cutoff_km: float,
    max_depth_layers: int,
    engine: str,
    compression_level: int,
    science_tag: str,
    overwrite: bool,
) -> Path:
    source_path = _source_path(input_root, current)
    output_root.mkdir(parents=True, exist_ok=True)
    output_path = output_root / source_path.name
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"{output_path} exists; pass --overwrite after contract invalidation")
    part_path = Path(f"{output_path}.part")
    part_path.unlink(missing_ok=True)
    try:
        with Dataset(source_path) as source:
            lon = np.asarray(source.variables["longitude"][:], dtype="f8")
            lat = np.asarray(source.variables["latitude"][:], dtype="f8")
            depth_all = np.asarray(source.variables["depth"][:], dtype="f4")
            depth = depth_all[: min(int(max_depth_layers), len(depth_all))]
            raw_ssh = np.asarray(source.variables["zos_glor"][0], dtype="f4")
            metadata = _attrs(source)
            plan = SeparableFFTLowpassPlan(lon, lat, cutoff_km)
            ssh = gaussian_highpass(raw_ssh, lon, lat, cutoff_km=cutoff_km, engine=engine, plan=plan)

            def layers():
                for index in range(len(depth)):
                    u = np.asarray(source.variables["uo_glor"][0, index], dtype="f4")
                    v = np.asarray(source.variables["vo_glor"][0, index], dtype="f4")
                    yield gaussian_highpass_pair(
                        u, v, lon, lat, cutoff_km=cutoff_km, engine=engine, plan=plan
                    )

            _write_netcdf(
                part_path, current, lon, lat, depth, ssh, layers(), metadata,
                cutoff_km, engine, compression_level, science_tag,
            )
        part_path.replace(output_path)
    except BaseException:
        part_path.unlink(missing_ok=True)
        raise
    return output_path


def _write_netcdf(
    path: Path,
    current: date,
    lon: np.ndarray,
    lat: np.ndarray,
    depth: np.ndarray,
    ssh: np.ndarray,
    layers,
    source_attrs: dict[str, object],
    cutoff_km: float,
    engine: str,
    compression_level: int,
    science_tag: str,
) -> None:
    with Dataset(path, "w", format="NETCDF4") as data:
        data.createDimension("time", 1)
        data.createDimension("depth", len(depth))
        data.createDimension("latitude", len(lat))
        data.createDimension("longitude", len(lon))
        time = data.createVariable("time", "f8", ("time",))
        dep = data.createVariable("depth", "f4", ("depth",))
        y = data.createVariable("latitude", "f4", ("latitude",))
        x = data.createVariable("longitude", "f4", ("longitude",))
        surface_chunks = (1, min(190, len(lat)), min(450, len(lon)))
        velocity_chunks = (1, 1, min(190, len(lat)), min(450, len(lon)))
        kwargs = dict(zlib=True, complevel=compression_level, shuffle=True)
        zos = data.createVariable("zos_glor", "f4", ("time", "latitude", "longitude"), chunksizes=surface_chunks, **kwargs)
        uo = data.createVariable("uo_glor", "f4", ("time", "depth", "latitude", "longitude"), chunksizes=velocity_chunks, **kwargs)
        vo = data.createVariable("vo_glor", "f4", ("time", "depth", "latitude", "longitude"), chunksizes=velocity_chunks, **kwargs)
        time.units = str(source_attrs.get("time_units", f"days since {current.year:04d}-01-01 00:00:00"))
        time.calendar = str(source_attrs.get("time_calendar", "standard"))
        dep.units, y.units, x.units = "m", "degrees_north", "degrees_east"
        zos.units = str(source_attrs.get("zos_glor_units", "cm"))
        uo.units = str(source_attrs.get("uo_glor_units", "m s-1"))
        vo.units = str(source_attrs.get("vo_glor_units", "m s-1"))
        data.source = "Detection_for_OFES.filters.gaussian_highpass"
        data.science_tag = science_tag
        data.ssh_definition = "ofes_eta_free_surface"
        data.baseline_definition = "OFES eta free-surface height; no atmospheric-pressure subtraction"
        data.horizontal_filter = f"field_minus_Gaussian_LP_{cutoff_km:g}km"
        data.horizontal_filter_mode = "highpass"
        data.horizontal_filter_kernel = "gaussian"
        data.convolution_engine = engine
        data.convolution_equivalence = "same discrete kernel weights and boundary modes; FFT changes operation order only"
        data.netcdf_compression_level = int(compression_level)
        time[:] = [(current - date(current.year, 1, 1)).days]
        dep[:], y[:], x[:] = depth, lat, lon
        zos[0] = ssh
        count = 0
        for count, (u_layer, v_layer) in enumerate(layers, start=1):
            uo[0, count - 1] = u_layer
            vo[0, count - 1] = v_layer
        if count != len(depth):
            raise ValueError(f"velocity layer iterator produced {count}, expected {len(depth)}")


def build_gaussian_highpass(
    input_root: Path,
    output_root: Path,
    start: date,
    end: date,
    *,
    cutoff_km: float = GAUSSIAN_FWHM_KM,
    max_depth_layers: int = 1,
    workers: int = 1,
    engine: str = "fft",
    compression_level: int = 1,
    science_tag: str = "eta_hp500_geometry_vertical_v1",
    overwrite: bool = False,
) -> list[Path]:
    if cutoff_km != GAUSSIAN_FWHM_KM:
        raise ValueError(f"Canonical cutoff is fixed at {GAUSSIAN_FWHM_KM:g} km")
    selected = date_range(start, end)
    arguments = [
        (input_root, output_root, current, cutoff_km, max_depth_layers, engine,
         compression_level, science_tag, overwrite)
        for current in selected
    ]
    if workers > 1 and len(arguments) > 1:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            outputs = list(executor.map(_build_day_star, arguments))
    else:
        outputs = [_build_day(*item) for item in arguments]
    manifest = {
        "profile": "eta_hp500_geometry_vertical_v1",
        "input_root": str(input_root),
        "output_root": str(output_root),
        "start": start.isoformat(),
        "end": end.isoformat(),
        "ssh_definition": "ofes_eta_free_surface",
        "filter": "field_minus_Gaussian_LP500km",
        "cutoff_km": cutoff_km,
        "convolution_engine": engine,
        "max_depth_layers": max_depth_layers,
        "date_workers": workers,
        "netcdf_compression_level": compression_level,
        "outputs": [str(path) for path in outputs],
    }
    (output_root / "gaussian_highpass_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return outputs


def _build_day_star(arguments: tuple) -> Path:
    return _build_day(*arguments)


def date_range(start: date, end: date) -> list[date]:
    if end < start:
        raise ValueError("end precedes start")
    return [start + timedelta(days=offset) for offset in range((end - start).days + 1)]


def parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--available-start", help=argparse.SUPPRESS)
    parser.add_argument("--available-end", help=argparse.SUPPRESS)
    parser.add_argument("--temporal-window-days", choices=["1"], default="1", help=argparse.SUPPRESS)
    parser.add_argument("--filter-mode", choices=["highpass"], default="highpass", help=argparse.SUPPRESS)
    parser.add_argument("--large-cutoff-km", type=float, choices=[GAUSSIAN_FWHM_KM], default=GAUSSIAN_FWHM_KM)
    parser.add_argument("--spatial-kernel", choices=["gaussian"], default="gaussian", help=argparse.SUPPRESS)
    parser.add_argument("--large-cutoff-mode", choices=["fixed"], default="fixed", help=argparse.SUPPRESS)
    parser.add_argument("--max-depth-layers", type=int, default=1)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--convolution-engine", choices=["direct", "fft"], default="fft")
    parser.add_argument("--compression-level", type=int, default=1)
    parser.add_argument("--science-tag", default="eta_hp500_geometry_vertical_v1")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    outputs = build_gaussian_highpass(
        args.input_root,
        args.output_root,
        parse_date(args.start),
        parse_date(args.end),
        cutoff_km=args.large_cutoff_km,
        max_depth_layers=args.max_depth_layers,
        workers=args.workers,
        engine=args.convolution_engine,
        compression_level=args.compression_level,
        science_tag=args.science_tag,
        overwrite=args.overwrite,
    )
    print(json.dumps({"written": [str(path) for path in outputs]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
