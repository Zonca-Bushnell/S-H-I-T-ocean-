"""Compare Gaussian, Lanczos, and Bessel SSH scale separation on one OFES day."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
from netCDF4 import Dataset
from scipy.ndimage import convolve1d
from scipy.signal import bessel, sosfreqz


DEFAULT_INPUT_ROOT = Path(r"E:\DATA\01_Eddy_correspond\02_OFES\origin_compatible_filter")
DEFAULT_OUTPUT_ROOT = Path(r"E:\DATA\01_Eddy_correspond\02_OFES\filter_kernel_compare_19910101")
GAUSSIAN_FWHM_TO_HALF_POWER_WAVELENGTH = 2.0 * np.pi / (2.354820045 * np.sqrt(np.log(2.0)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--day", default="1991-01-01")
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--available-start", default="1991-01-01")
    parser.add_argument("--available-end", default="1991-01-19")
    parser.add_argument("--temporal-window-days", type=int, default=10)
    parser.add_argument("--small-km", type=float, default=25.0)
    parser.add_argument("--large-km", type=float, default=180.0)
    parser.add_argument("--min-support", type=float, default=0.95)
    return parser.parse_args()


def daily_path(root: Path, day: date) -> Path:
    path = root / f"global_phy_{day:%Y%m%d}.nc"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def read_ssh(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with Dataset(path) as ds:
        lon = np.asarray(ds.variables["longitude"][:], dtype="f8")
        lat = np.asarray(ds.variables["latitude"][:], dtype="f8")
        ssh = np.ma.asarray(ds.variables["zos_glor"][0], dtype="f8").filled(np.nan)
    ssh[~np.isfinite(ssh)] = np.nan
    return lon, lat, ssh


def available_days(target: date, start: date, end: date, nominal_window: int) -> list[date]:
    half = (nominal_window - 1) // 2
    lo = max(start, target - timedelta(days=half))
    hi = min(end, target + timedelta(days=nominal_window - half - 1))
    return [lo + timedelta(days=index) for index in range((hi - lo).days + 1)]


def temporal_mean(root: Path, days: list[date]) -> np.ndarray:
    total = None
    count = None
    for day in days:
        _, _, field = read_ssh(daily_path(root, day))
        valid = np.isfinite(field)
        if total is None:
            total = np.zeros_like(field)
            count = np.zeros_like(field)
        total[valid] += field[valid]
        count[valid] += 1.0
    return np.divide(total, count, out=np.full_like(total, np.nan), where=count > 0)


def lanczos_kernel(cutoff_wavelength_cells: float, order: int = 2) -> np.ndarray:
    radius = max(2, int(np.ceil(order * cutoff_wavelength_cells)))
    offsets = np.arange(-radius, radius + 1, dtype="f8")
    cutoff = min(0.49, 1.0 / max(cutoff_wavelength_cells, 2.05))
    kernel = 2.0 * cutoff * np.sinc(2.0 * cutoff * offsets) * np.sinc(offsets / (radius + 1.0))
    kernel /= kernel.sum()
    return kernel


def bessel_kernel(cutoff_wavelength_cells: float, order: int = 3) -> np.ndarray:
    cutoff = min(0.49, 1.0 / max(cutoff_wavelength_cells, 2.05))
    sos = bessel(order, 2.0 * cutoff, btype="lowpass", output="sos", norm="mag")
    size = 1
    minimum = max(257, int(np.ceil(24.0 * cutoff_wavelength_cells)))
    while size < minimum:
        size *= 2
    _, response = sosfreqz(sos, worN=size // 2 + 1, fs=1.0)
    impulse = np.fft.irfft(np.abs(response) ** 2, n=size)
    impulse = np.fft.fftshift(impulse)
    center = size // 2
    keep = np.flatnonzero(np.abs(impulse) > np.max(np.abs(impulse)) * 1.0e-5)
    radius = min(center - keep.min(), keep.max() - center)
    radius = max(2, int(radius))
    kernel = impulse[center - radius : center + radius + 1]
    kernel /= kernel.sum()
    return kernel


def convolve_physical(
    field: np.ndarray,
    lon: np.ndarray,
    lat: np.ndarray,
    kernel_factory,
    wavelength_km: float,
    min_support: float,
) -> np.ndarray:
    """Separable physical-scale convolution with dynamic zonal kernels and strict land support."""
    values = np.where(np.isfinite(field), field, 0.0)
    mask = np.isfinite(field).astype("f8")
    dlat = abs(float(np.nanmedian(np.diff(lat))))
    dlon = abs(float(np.nanmedian(np.diff(lon))))
    ky = kernel_factory(wavelength_km / (111.32 * dlat))
    y_values = convolve1d(values, ky, axis=0, mode="nearest")
    y_support = convolve1d(mask, np.abs(ky) / np.abs(ky).sum(), axis=0, mode="nearest")
    output = np.empty_like(values)
    support = np.empty_like(values)
    for j, latitude in enumerate(lat):
        dx_km = 111.32 * max(0.12, abs(float(np.cos(np.deg2rad(latitude)))) * dlon)
        kx = kernel_factory(wavelength_km / dx_km)
        output[j] = convolve1d(y_values[j], kx, mode="wrap")
        support[j] = convolve1d(y_support[j], np.abs(kx) / np.abs(kx).sum(), mode="wrap")
    output[support < min_support] = np.nan
    return output


def bandpass(field: np.ndarray, lon: np.ndarray, lat: np.ndarray, kernel_factory, small_km: float, large_km: float, min_support: float) -> np.ndarray:
    small = convolve_physical(field, lon, lat, kernel_factory, small_km, min_support)
    large = convolve_physical(field, lon, lat, kernel_factory, large_km, min_support)
    result = small - large
    result[~np.isfinite(small) | ~np.isfinite(large)] = np.nan
    return result


def highpass(field: np.ndarray, lon: np.ndarray, lat: np.ndarray, kernel_factory, large_km: float, min_support: float) -> np.ndarray:
    low = convolve_physical(field, lon, lat, kernel_factory, large_km, min_support)
    result = field - low
    result[~np.isfinite(field) | ~np.isfinite(low)] = np.nan
    return result


def metrics(field: np.ndarray) -> dict[str, float]:
    finite = np.isfinite(field)
    gx = np.diff(field, axis=1)
    gy = np.diff(field, axis=0)
    lap = -4 * field[1:-1, 1:-1] + field[:-2, 1:-1] + field[2:, 1:-1] + field[1:-1, :-2] + field[1:-1, 2:]
    std = float(np.nanstd(field))
    return {
        "finite_fraction": float(finite.mean()),
        "std_cm": std,
        "q95_abs_cm": float(np.nanpercentile(np.abs(field), 95)),
        "gradient_x_std": float(np.nanstd(gx)),
        "gradient_y_std": float(np.nanstd(gy)),
        "relative_laplacian": float(np.nanstd(lap) / max(std, 1.0e-12)),
    }


def plot(fields: dict[str, np.ndarray], lon: np.ndarray, lat: np.ndarray, path: Path, title: str, bounds: tuple[float, float, float, float] | None) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if bounds is None:
        x0, x1, y0, y1 = float(lon.min()), float(lon.max()), float(lat.min()), float(lat.max())
    else:
        x0, x1, y0, y1 = bounds
    ix = np.flatnonzero((lon >= x0) & (lon <= x1))
    iy = np.flatnonzero((lat >= y0) & (lat <= y1))
    clipped = [field[np.ix_(iy, ix)] for field in fields.values()]
    vmax = max(float(np.nanpercentile(np.abs(item), 99.5)) for item in clipped)
    fig, axes = plt.subplots(1, len(fields), figsize=(7 * len(fields), 5), constrained_layout=True)
    for ax, (label, field), subset in zip(axes, fields.items(), clipped):
        mesh = ax.pcolormesh(lon[ix], lat[iy], subset, shading="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        mesh.set_rasterized(True)
        ax.set_title(label)
        ax.set_xlabel("longitude")
        ax.set_ylabel("latitude")
    fig.suptitle(f"{title}\nShared colour scale: +/- {vmax:.3g} cm")
    fig.colorbar(mesh, ax=axes.tolist(), label="SSHA (cm)", shrink=0.9)
    fig.savefig(path, dpi=160)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    args = parse_args()
    target = date.fromisoformat(args.day)
    start = date.fromisoformat(args.available_start)
    end = date.fromisoformat(args.available_end)
    args.output_root.mkdir(parents=True, exist_ok=True)
    lon, lat, _ = read_ssh(daily_path(args.input_root, target))
    days = available_days(target, start, end, args.temporal_window_days)
    mean_field = temporal_mean(args.input_root, days)

    small_equivalent = args.small_km * GAUSSIAN_FWHM_TO_HALF_POWER_WAVELENGTH
    large_equivalent = args.large_km * GAUSSIAN_FWHM_TO_HALF_POWER_WAVELENGTH
    fields = {
        "Gaussian DoG 25-180 km": bandpass(mean_field, lon, lat, gaussian_kernel, args.small_km, args.large_km, args.min_support),
        "Lanczos HP | Gaussian-equivalent 180 km": highpass(mean_field, lon, lat, lanczos_kernel, large_equivalent, args.min_support),
        "Bessel DoG | Gaussian-equivalent 25-180 km": bandpass(mean_field, lon, lat, bessel_kernel, small_equivalent, large_equivalent, args.min_support),
    }
    regions = {
        "global": None,
        "north_pacific": (150.0, 250.0, 20.0, 60.0),
        "south_pacific": (170.0, 290.0, -55.0, -20.0),
        "acc": (0.0, 360.0, -70.0, -40.0),
    }
    rows: list[dict[str, object]] = []
    for name, bounds in regions.items():
        plot(fields, lon, lat, args.output_root / f"spatial_kernel_compare_{name}_{target:%Y%m%d}.png", f"OFES spatial kernel comparison | {name} | {target}", bounds)
        if bounds is None:
            subset = slice(None), slice(None)
        else:
            x0, x1, y0, y1 = bounds
            subset = np.flatnonzero((lat >= y0) & (lat <= y1)), np.flatnonzero((lon >= x0) & (lon <= x1))
        for label, field in fields.items():
            selected = field if bounds is None else field[np.ix_(*subset)]
            rows.append({"region": name, "filter": label, **metrics(selected)})
    with (args.output_root / f"spatial_kernel_metrics_{target:%Y%m%d}.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        "day": target.isoformat(),
        "temporal_days_used": [value.isoformat() for value in days],
        "small_km": args.small_km,
        "large_km": args.large_km,
        "gaussian_fwhm_to_half_power_wavelength_factor": GAUSSIAN_FWHM_TO_HALF_POWER_WAVELENGTH,
        "lanczos_bessel_half_power_wavelength_km": {"small": small_equivalent, "large": large_equivalent},
        "min_support": args.min_support,
        "filters": list(fields),
        "status": "complete",
    }
    (args.output_root / f"spatial_kernel_manifest_{target:%Y%m%d}.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def gaussian_kernel(fwhm_cells: float) -> np.ndarray:
    sigma = max(0.25, fwhm_cells / 2.354820045)
    radius = max(1, int(np.ceil(3.0 * sigma)))
    offsets = np.arange(-radius, radius + 1, dtype="f8")
    kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
    return kernel / kernel.sum()


if __name__ == "__main__":
    main()
