"""Compare native OFES SSH and Copernicus SLA after shared spatial band-passes.

This is a diagnostic-only reader.  It does not write any detection input or
catalogue data.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import date, timedelta
from pathlib import Path
from typing import Callable

import numpy as np
from netCDF4 import Dataset
from scipy.ndimage import convolve1d
from scipy.signal import bessel, sosfreqz

from Detection_for_OFES.ofes_io import ctl_path, parse_ctl, read_ssh_latlon_daily_only


OFES_ROOT = Path(r"F:\OFES\external_OFES2")
COPERNICUS_PATH = Path(
    r"E:\DATA\01_Eddy_correspond\06_Copernicus_Marine\SEALEVEL_GLO_PHY_L4_MY_008_047"
    r"\19930101\copernicus_sla_19930101.nc"
)
OUTPUT_ROOT = Path(
    r"E:\DATA\01_Eddy_correspond\02_OFES\filter_diagnostics_19910101"
    r"\raw_ofes_vs_copernicus_1991_1993"
)
KM_PER_DEGREE = 111.32
GAUSSIAN_FWHM_TO_HALF_POWER_WAVELENGTH = 2.0 * np.pi / (2.354820045 * np.sqrt(np.log(2.0)))
KernelFactory = Callable[[float], np.ndarray]


def json_default(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ofes-root", type=Path, default=OFES_ROOT)
    parser.add_argument("--ofes-day", default="1991-01-01")
    parser.add_argument("--copernicus-path", type=Path, default=COPERNICUS_PATH)
    parser.add_argument("--copernicus-day", default="1993-01-01")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument(
        "--ofes-baselines",
        default="",
        help="Comma-separated OFES reference windows: jan01-jan31,jan01-jan19.",
    )
    parser.add_argument(
        "--baseline-stage",
        choices=("all", "prepare", "filter", "plot"),
        default="all",
        help="Run one resumable baseline-sensitivity stage; all runs every stage in order.",
    )
    parser.add_argument("--small-km", type=float, default=25.0)
    parser.add_argument("--large-km", type=float, default=200.0)
    parser.add_argument("--min-valid-weight-fraction", type=float, default=0.95)
    return parser.parse_args()


def gaussian_kernel(fwhm_cells: float) -> np.ndarray:
    sigma = max(0.25, fwhm_cells / 2.354820045)
    offsets = np.arange(-max(1, int(np.ceil(3.0 * sigma))), max(1, int(np.ceil(3.0 * sigma))) + 1)
    kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
    return kernel / kernel.sum()


def bessel_kernel(cutoff_wavelength_cells: float, order: int = 3) -> np.ndarray:
    cutoff = min(0.49, 1.0 / max(cutoff_wavelength_cells, 2.05))
    sos = bessel(order, 2.0 * cutoff, btype="lowpass", output="sos", norm="mag")
    size = 1
    while size < max(257, int(np.ceil(24.0 * cutoff_wavelength_cells))):
        size *= 2
    _, response = sosfreqz(sos, worN=size // 2 + 1, fs=1.0)
    impulse = np.fft.fftshift(np.fft.irfft(np.abs(response) ** 2, n=size))
    center = size // 2
    keep = np.flatnonzero(np.abs(impulse) > np.max(np.abs(impulse)) * 1.0e-5)
    radius = max(2, min(center - keep.min(), keep.max() - center))
    kernel = impulse[center - radius : center + radius + 1]
    return kernel / kernel.sum()


def read_ofes(root: Path, day_text: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    field = read_ssh_latlon_daily_only(root, date.fromisoformat(day_text)).astype("f8")
    eta_meta = parse_ctl(ctl_path(root, "eta"))
    return eta_meta.x.values, eta_meta.y.values, field, {
        "definition": "H_OFES = eta - (pressur - 1000)",
        "source_units": "cm",
        "grid": eta_meta.grid_signature,
    }


def baseline_days(day_text: str, name: str) -> list[date]:
    target = date.fromisoformat(day_text)
    if target.year != 1991 or target.month != 1:
        raise ValueError("The locally available OFES baseline windows are limited to January 1991.")
    stop_day = {"jan01-jan19": 19, "jan01-jan31": 31}.get(name)
    if stop_day is None:
        raise ValueError(f"Unsupported OFES baseline {name!r}.")
    return [date(1991, 1, 1) + timedelta(days=index) for index in range(stop_day)]


def ofes_baseline(root: Path, days: list[date]) -> np.ndarray:
    total: np.ndarray | None = None
    count: np.ndarray | None = None
    for index, current_day in enumerate(days, start=1):
        field = read_ssh_latlon_daily_only(root, current_day).astype("f8")
        valid = np.isfinite(field)
        if total is None:
            total = np.zeros_like(field)
            count = np.zeros_like(field)
        total[valid] += field[valid]
        count[valid] += 1.0
        print(f"[baseline] read {index}/{len(days)}: {current_day.isoformat()}", flush=True)
    assert total is not None and count is not None
    return np.divide(total, count, out=np.full_like(total, np.nan), where=count > 0)


def read_copernicus(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    with Dataset(path) as dataset:
        lon = np.asarray(dataset.variables["longitude"][:], dtype="f8")
        lat = np.asarray(dataset.variables["latitude"][:], dtype="f8")
        variable = dataset.variables["sla"]
        variable.set_auto_maskandscale(True)
        values = np.ma.asarray(variable[:]).filled(np.nan)
        while values.ndim > 2:
            values = values[0]
        units = str(getattr(variable, "units", ""))
        if units.lower() not in {"m", "meter", "meters", "metre", "metres"}:
            raise ValueError(f"Expected Copernicus SLA in metres, got {units!r}.")
        field_cm = np.asarray(values, dtype="f8") * 100.0
        # Reorder only the coordinate axis to the OFES 0-360 convention.
        # Values stay on their native 0.125 degree cells; no interpolation occurs.
        lon = np.mod(lon, 360.0)
        order = np.argsort(lon)
        lon = lon[order]
        field_cm = field_cm[:, order]
        metadata = {
            "definition": "Copernicus/DUACS L4 SLA above the product mean sea surface",
            "source_units": units,
            "plot_units": "cm",
            "fill_value": getattr(variable, "_FillValue", None),
            "scale_factor": getattr(variable, "scale_factor", None),
            "add_offset": getattr(variable, "add_offset", None),
            "grid": {"nx": int(lon.size), "ny": int(lat.size), "lon_step": float(np.median(np.diff(lon))), "lat_step": float(np.median(np.diff(lat)))},
            "longitude_coordinate_action": "reordered from -180..180 to 0..360 without interpolation",
        }
    field_cm[~np.isfinite(field_cm)] = np.nan
    return lon, lat, field_cm, metadata


def circular_kernel(kernel: np.ndarray, width: int) -> np.ndarray:
    result = np.zeros(width, dtype="f8")
    offsets = np.arange(-(kernel.size // 2), kernel.size // 2 + 1)
    np.add.at(result, offsets % width, kernel)
    return result


def circular_convolve_rows(values: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    spectrum = np.fft.rfft(circular_kernel(kernel, values.shape[1]))
    return np.fft.irfft(np.fft.rfft(values, axis=1) * spectrum[None, :], n=values.shape[1], axis=1)


def normalized_lowpass(field: np.ndarray, lon: np.ndarray, lat: np.ndarray, kernel_factory: KernelFactory, scale_km: float, min_support: float) -> np.ndarray:
    """Physical-scale separable lowpass with periodic longitude and NaN-aware weights."""
    finite = np.isfinite(field)
    values = np.where(finite, field, 0.0)
    mask = finite.astype("f8")
    dlat = abs(float(np.median(np.diff(lat))))
    dlon = abs(float(np.median(np.diff(lon))))
    kernel_y = kernel_factory(scale_km / (KM_PER_DEGREE * dlat))
    numerator_y = convolve1d(values, kernel_y, axis=0, mode="nearest")
    denominator_y = convolve1d(mask, kernel_y, axis=0, mode="nearest")
    support_y = convolve1d(mask, np.abs(kernel_y) / np.abs(kernel_y).sum(), axis=0, mode="nearest")
    output = np.full_like(field, np.nan, dtype="f8")
    support = np.zeros_like(field, dtype="f8")
    cell_scales = scale_km / (KM_PER_DEGREE * np.maximum(0.12, np.abs(np.cos(np.deg2rad(lat))) * dlon))
    keys = np.round(cell_scales * 2.0) / 2.0
    for key in np.unique(keys):
        rows = np.flatnonzero(keys == key)
        kernel_x = kernel_factory(float(key))
        numerator = circular_convolve_rows(numerator_y[rows], kernel_x)
        denominator = circular_convolve_rows(denominator_y[rows], kernel_x)
        weight = circular_convolve_rows(support_y[rows], np.abs(kernel_x) / np.abs(kernel_x).sum())
        valid = (weight >= min_support) & (np.abs(denominator) > 1.0e-10)
        output[rows] = np.divide(numerator, denominator, out=np.full_like(numerator, np.nan), where=valid)
        support[rows] = weight
    output[(support < min_support) | ~finite] = np.nan
    return output


def bandpass(field: np.ndarray, lon: np.ndarray, lat: np.ndarray, kernel_factory: KernelFactory, small_km: float, large_km: float, min_support: float) -> np.ndarray:
    small = normalized_lowpass(field, lon, lat, kernel_factory, small_km, min_support)
    large = normalized_lowpass(field, lon, lat, kernel_factory, large_km, min_support)
    result = small - large
    result[~np.isfinite(small) | ~np.isfinite(large)] = np.nan
    return result


def bounds_index(lon: np.ndarray, lat: np.ndarray, bounds: tuple[float, float, float, float] | None) -> tuple[np.ndarray, np.ndarray]:
    if bounds is None:
        return np.arange(lat.size), np.arange(lon.size)
    x0, x1, y0, y1 = bounds
    return np.flatnonzero((lat >= y0) & (lat <= y1)), np.flatnonzero((lon >= x0) & (lon <= x1))


def robust_limit(fields: list[np.ndarray], percentile: float = 99.5) -> float:
    values = [np.ravel(np.abs(field[np.isfinite(field)])) for field in fields if np.isfinite(field).any()]
    return max(1.0e-6, float(np.percentile(np.concatenate(values), percentile)))


def setup_map(ax, lon: np.ndarray, lat: np.ndarray, field: np.ndarray, title: str, vmax: float):
    cmap = __import__("matplotlib.pyplot", fromlist=["get_cmap"]).get_cmap("RdBu_r").copy()
    cmap.set_bad("#22262d")
    image = ax.pcolormesh(lon, lat, field, shading="auto", cmap=cmap, vmin=-vmax, vmax=vmax, rasterized=True)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Longitude (degrees east)")
    ax.set_ylabel("Latitude")
    ax.grid(True, color="k", alpha=0.2, linewidth=0.4)
    return image


def plot_raw(ofes: tuple[np.ndarray, np.ndarray, np.ndarray], copernicus: tuple[np.ndarray, np.ndarray, np.ndarray], output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(18, 7), constrained_layout=True)
    for ax, (name, lon, lat, field) in zip(axes, (("OFES raw H", *ofes), ("Copernicus raw SLA", *copernicus))):
        vmax = robust_limit([field])
        image = setup_map(ax, lon, lat, field, f"{name} | native definition | +/- {vmax:.1f} cm", vmax)
        fig.colorbar(image, ax=ax, pad=0.015, label="cm")
    fig.suptitle("Raw native surface-height fields | no temporal mean, no regridding, no spatial filtering", fontsize=14)
    fig.savefig(output, dpi=180)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)


def plot_bandpass(
    ofes: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    copernicus: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    output: Path,
    title: str,
    bounds: tuple[float, float, float, float] | None = None,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = ("Gaussian DoG", "Bessel order-3")
    selected = []
    for lon, lat, first, second in (ofes, copernicus):
        iy, ix = bounds_index(lon, lat, bounds)
        selected.extend((first[np.ix_(iy, ix)], second[np.ix_(iy, ix)]))
    vmax = robust_limit(selected)
    fig, axes = plt.subplots(2, 2, figsize=(18, 12), constrained_layout=True)
    for row, (source, data) in enumerate((("OFES H", ofes), ("Copernicus SLA", copernicus))):
        lon, lat, first, second = data
        iy, ix = bounds_index(lon, lat, bounds)
        for column, (label, field) in enumerate(zip(names, (first, second))):
            image = setup_map(axes[row, column], lon[ix], lat[iy], field[np.ix_(iy, ix)], f"{source} | {label}", vmax)
    fig.colorbar(image, ax=axes.ravel().tolist(), pad=0.012, label="25-200 km band-pass (cm)")
    fig.suptitle(f"{title}\nShared physical scales and shared colour scale: +/- {vmax:.2f} cm", fontsize=14)
    fig.savefig(output, dpi=180)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)


def plot_ofes_baselines(
    lon: np.ndarray,
    lat: np.ndarray,
    raw: np.ndarray,
    anomaly_31: np.ndarray,
    anomaly_19: np.ndarray,
    output: Path,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    anomaly_limit = robust_limit([anomaly_31, anomaly_19])
    fig, axes = plt.subplots(1, 3, figsize=(24, 7), constrained_layout=True)
    items = (
        ("OFES raw H", raw, robust_limit([raw])),
        ("OFES H' | H - Jan01-Jan31 mean", anomaly_31, anomaly_limit),
        ("OFES H' | H - Jan01-Jan19 mean", anomaly_19, anomaly_limit),
    )
    for axis, (label, field, vmax) in zip(axes, items):
        image = setup_map(axis, lon, lat, field, f"{label} | +/- {vmax:.1f} cm", vmax)
        fig.colorbar(image, ax=axis, pad=0.015, label="cm")
    fig.suptitle("OFES native-grid baseline sensitivity | 1991-01-01", fontsize=14)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_baseline_bandpass(
    datasets: tuple[
        tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ],
    output: Path,
    title: str,
    bounds: tuple[float, float, float, float] | None,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    selected: list[np.ndarray] = []
    for _, lon, lat, gaussian, bessel_field in datasets:
        iy, ix = bounds_index(lon, lat, bounds)
        selected.extend((gaussian[np.ix_(iy, ix)], bessel_field[np.ix_(iy, ix)]))
    vmax = robust_limit(selected)
    fig, axes = plt.subplots(2, 3, figsize=(24, 12), constrained_layout=True)
    for column, (source, lon, lat, gaussian, bessel_field) in enumerate(datasets):
        iy, ix = bounds_index(lon, lat, bounds)
        for row, (kernel, field) in enumerate((("Gaussian DoG", gaussian), ("Bessel order-3", bessel_field))):
            image = setup_map(axes[row, column], lon[ix], lat[iy], field[np.ix_(iy, ix)], f"{source} | {kernel}", vmax)
    fig.colorbar(image, ax=axes.ravel().tolist(), pad=0.012, label="25-200 km band-pass (cm)")
    fig.suptitle(f"{title}\nShared physical scales and shared colour scale: +/- {vmax:.2f} cm", fontsize=14)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def summary(field: np.ndarray) -> dict[str, float]:
    finite = np.isfinite(field)
    values = field[finite]
    return {
        "finite_fraction": float(finite.mean()),
        "std_cm": float(np.std(values)),
        "q95_abs_cm": float(np.percentile(np.abs(values), 95)),
        "q99_abs_cm": float(np.percentile(np.abs(values), 99)),
    }


def run_baseline_sensitivity(args: argparse.Namespace) -> None:
    requested = tuple(item.strip() for item in args.ofes_baselines.split(",") if item.strip())
    if set(requested) != {"jan01-jan31", "jan01-jan19"}:
        raise ValueError("--ofes-baselines must request jan01-jan31,jan01-jan19 for this comparison.")
    output_root = args.output_root / "baseline_sensitivity"
    output_root.mkdir(parents=True, exist_ok=True)
    cache = {
        "ofes_lon": output_root / "ofes_lon.npy", "ofes_lat": output_root / "ofes_lat.npy",
        "ofes_raw": output_root / "ofes_raw_h.npy", "ofes_31": output_root / "ofes_hprime_jan01_jan31.npy",
        "ofes_19": output_root / "ofes_hprime_jan01_jan19.npy", "cop_lon": output_root / "copernicus_lon.npy",
        "cop_lat": output_root / "copernicus_lat.npy", "cop_raw": output_root / "copernicus_sla.npy",
        "h31_gaussian": output_root / "hprime_jan01_jan31_gaussian.npy", "h31_bessel": output_root / "hprime_jan01_jan31_bessel.npy",
        "h19_gaussian": output_root / "hprime_jan01_jan19_gaussian.npy", "h19_bessel": output_root / "hprime_jan01_jan19_bessel.npy",
        "cop_gaussian": output_root / "copernicus_gaussian.npy", "cop_bessel": output_root / "copernicus_bessel.npy",
    }
    if args.baseline_stage in {"all", "prepare"}:
        ofes_lon, ofes_lat, ofes_raw, ofes_meta = read_ofes(args.ofes_root, args.ofes_day)
        cop_lon, cop_lat, cop_raw, cop_meta = read_copernicus(args.copernicus_path)
        means = {name: ofes_baseline(args.ofes_root, baseline_days(args.ofes_day, name)) for name in requested}
        arrays = {
            "ofes_lon": ofes_lon, "ofes_lat": ofes_lat, "ofes_raw": ofes_raw,
            "ofes_31": ofes_raw - means["jan01-jan31"], "ofes_19": ofes_raw - means["jan01-jan19"],
            "cop_lon": cop_lon, "cop_lat": cop_lat, "cop_raw": cop_raw,
        }
        for name, array in arrays.items():
            np.save(cache[name], array)
        manifest = {
            "status": "prepared", "comparison": "same calendar day cross-year qualitative comparison; not pointwise validation",
            "ofes": {"day": args.ofes_day, **ofes_meta},
            "copernicus": {"day": args.copernicus_day, "path": str(args.copernicus_path), **cop_meta},
            "ofes_baselines": {name: {"formula": f"H({args.ofes_day}) - mean(H over {name})", "days": [item.isoformat() for item in baseline_days(args.ofes_day, name)]} for name in requested},
            "grid_policy": "OFES baseline means and anomalies remain on the native SSH grid; Copernicus remains on its native L4 grid.",
        }
        (output_root / "baseline_sensitivity_manifest.json").write_text(json.dumps(manifest, indent=2, default=json_default), encoding="utf-8")
        print("[baseline] prepared native-grid OFES anomalies and Copernicus SLA cache", flush=True)
        if args.baseline_stage == "prepare":
            return
    required_prepare = ("ofes_lon", "ofes_lat", "ofes_raw", "ofes_31", "ofes_19", "cop_lon", "cop_lat", "cop_raw")
    missing = [name for name in required_prepare if not cache[name].exists()]
    if missing:
        raise FileNotFoundError(f"Run --baseline-stage prepare first; missing {missing}")
    ofes_lon, ofes_lat, ofes_raw, anomaly_31, anomaly_19, cop_lon, cop_lat, cop_raw = (np.load(cache[name]) for name in required_prepare)
    bessel_small = args.small_km * GAUSSIAN_FWHM_TO_HALF_POWER_WAVELENGTH
    bessel_large = args.large_km * GAUSSIAN_FWHM_TO_HALF_POWER_WAVELENGTH

    def fields(field: np.ndarray, lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return (
            bandpass(field, lon, lat, gaussian_kernel, args.small_km, args.large_km, args.min_valid_weight_fraction),
            bandpass(field, lon, lat, bessel_kernel, bessel_small, bessel_large, args.min_valid_weight_fraction),
        )

    if args.baseline_stage in {"all", "filter"}:
        for prefix, field, lon, lat in (("h31", anomaly_31, ofes_lon, ofes_lat), ("h19", anomaly_19, ofes_lon, ofes_lat), ("cop", cop_raw, cop_lon, cop_lat)):
            gaussian, bessel_field = fields(field, lon, lat)
            np.save(cache[f"{prefix}_gaussian"], gaussian)
            np.save(cache[f"{prefix}_bessel"], bessel_field)
            print(f"[baseline] completed {prefix} Gaussian and Bessel band-passes", flush=True)
        if args.baseline_stage == "filter":
            return
    required_filter = ("h31_gaussian", "h31_bessel", "h19_gaussian", "h19_bessel", "cop_gaussian", "cop_bessel")
    missing = [name for name in required_filter if not cache[name].exists()]
    if missing:
        raise FileNotFoundError(f"Run --baseline-stage filter first; missing {missing}")
    h31_fields = (np.load(cache["h31_gaussian"]), np.load(cache["h31_bessel"]))
    h19_fields = (np.load(cache["h19_gaussian"]), np.load(cache["h19_bessel"]))
    cop_fields = (np.load(cache["cop_gaussian"]), np.load(cache["cop_bessel"]))
    regions = {
        "global": None,
        "north_pacific_open_ocean": (190.0, 245.0, 25.0, 55.0),
        "south_pacific_open_ocean": (190.0, 280.0, -50.0, -30.0),
    }
    datasets = (
        ("OFES H' | Jan01-Jan31 reference", ofes_lon, ofes_lat, *h31_fields),
        ("OFES H' | Jan01-Jan19 reference", ofes_lon, ofes_lat, *h19_fields),
        ("Copernicus SLA | product reference", cop_lon, cop_lat, *cop_fields),
    )
    manifest_path = output_root / "baseline_sensitivity_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update({"status": "complete", "bandpass": {"formula": "LP_25km(field) - LP_200km(field)", "gaussian": {"small_fwhm_km": args.small_km, "large_fwhm_km": args.large_km}, "bessel_order_3": {"gaussian_equivalent_half_power_wavelength_km": {"small": bessel_small, "large": bessel_large}}, "longitude_boundary": "periodic", "latitude_boundary": "nearest/non-periodic", "nan_handling": "normalized convolution", "min_valid_weight_fraction": args.min_valid_weight_fraction}})
    manifest_path.write_text(json.dumps(manifest, indent=2, default=json_default), encoding="utf-8")
    plot_ofes_baselines(ofes_lon, ofes_lat, ofes_raw, anomaly_31, anomaly_19, output_root / "ofes_native_baseline_comparison.png")
    for region, bounds in regions.items():
        plot_baseline_bandpass(datasets, output_root / f"baseline_vs_copernicus_bandpass_{region}.png", f"OFES baseline sensitivity vs Copernicus | {region}", bounds)
    rows: list[dict[str, object]] = []
    for region, bounds in regions.items():
        for source, lon, lat, raw, filtered in (
            ("OFES_Hprime_Jan01_Jan31", ofes_lon, ofes_lat, anomaly_31, h31_fields),
            ("OFES_Hprime_Jan01_Jan19", ofes_lon, ofes_lat, anomaly_19, h19_fields),
            ("Copernicus_SLA", cop_lon, cop_lat, cop_raw, cop_fields),
        ):
            iy, ix = bounds_index(lon, lat, bounds)
            for label, field in (("reference_anomaly_or_sla", raw), ("gaussian_dog_25_200km", filtered[0]), ("bessel3_dog_25_200km", filtered[1])):
                rows.append({"region": region, "source": source, "field": label, **summary(field[np.ix_(iy, ix)])})
        iy, ix = bounds_index(ofes_lon, ofes_lat, bounds)
        rows.append({"region": region, "source": "OFES_baseline_difference", "field": "Hprime31_minus_Hprime19", **summary((anomaly_31 - anomaly_19)[np.ix_(iy, ix)])})
    with (output_root / "baseline_vs_copernicus_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[baseline] wrote {output_root}", flush=True)


def main() -> None:
    args = parse_args()
    if args.ofes_baselines:
        run_baseline_sensitivity(args)
        return
    args.output_root.mkdir(parents=True, exist_ok=True)
    ofes_lon, ofes_lat, ofes_raw, ofes_meta = read_ofes(args.ofes_root, args.ofes_day)
    cop_lon, cop_lat, cop_raw, cop_meta = read_copernicus(args.copernicus_path)
    print("[compare] loaded native OFES H and CF-decoded Copernicus SLA", flush=True)
    bessel_small = args.small_km * GAUSSIAN_FWHM_TO_HALF_POWER_WAVELENGTH
    bessel_large = args.large_km * GAUSSIAN_FWHM_TO_HALF_POWER_WAVELENGTH
    ofes_fields = (
        bandpass(ofes_raw, ofes_lon, ofes_lat, gaussian_kernel, args.small_km, args.large_km, args.min_valid_weight_fraction),
        bandpass(ofes_raw, ofes_lon, ofes_lat, bessel_kernel, bessel_small, bessel_large, args.min_valid_weight_fraction),
    )
    print("[compare] completed OFES Gaussian and Bessel band-passes", flush=True)
    cop_fields = (
        bandpass(cop_raw, cop_lon, cop_lat, gaussian_kernel, args.small_km, args.large_km, args.min_valid_weight_fraction),
        bandpass(cop_raw, cop_lon, cop_lat, bessel_kernel, bessel_small, bessel_large, args.min_valid_weight_fraction),
    )
    print("[compare] completed Copernicus Gaussian and Bessel band-passes", flush=True)
    regions = {
        "global": None,
        "north_pacific_open_ocean": (190.0, 245.0, 25.0, 55.0),
        "south_pacific_open_ocean": (190.0, 280.0, -50.0, -30.0),
    }
    manifest = {
        "status": "complete",
        "comparison": "same calendar day cross-year qualitative comparison; not pointwise validation",
        "ofes": {"day": args.ofes_day, **ofes_meta},
        "copernicus": {"day": args.copernicus_day, "path": str(args.copernicus_path), **cop_meta},
        "raw_processing": "native definitions only; no Jan01-Jan19 mean subtraction, regridding, time mean, or spatial filtering",
        "bandpass": {
            "formula": "LP_25km(field) - LP_200km(field)",
            "gaussian": {"small_fwhm_km": args.small_km, "large_fwhm_km": args.large_km},
            "bessel_order_3": {"gaussian_equivalent_half_power_wavelength_km": {"small": bessel_small, "large": bessel_large}},
            "longitude_boundary": "periodic", "latitude_boundary": "nearest/non-periodic", "nan_handling": "normalized convolution", "min_valid_weight_fraction": args.min_valid_weight_fraction,
        },
        "outputs": {"raw": "raw_native_ofes_h_vs_copernicus_sla.png", "bandpass_regions": [f"mesoscale_bandpass_{name}.png" for name in regions], "metrics": "raw_and_mesoscale_metrics.csv"},
    }
    (args.output_root / "raw_ofes_copernicus_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=json_default), encoding="utf-8"
    )
    plot_raw((ofes_lon, ofes_lat, ofes_raw), (cop_lon, cop_lat, cop_raw), args.output_root / "raw_native_ofes_h_vs_copernicus_sla.png")
    for name, bounds in regions.items():
        plot_bandpass(
            (ofes_lon, ofes_lat, *ofes_fields),
            (cop_lon, cop_lat, *cop_fields),
            args.output_root / f"mesoscale_bandpass_{name}.png",
            f"OFES 1991-01-01 vs Copernicus 1993-01-01 | {name}",
            bounds,
        )
    rows: list[dict[str, object]] = []
    for region, bounds in regions.items():
        for source, lon, lat, raw, fields in (("OFES", ofes_lon, ofes_lat, ofes_raw, ofes_fields), ("Copernicus", cop_lon, cop_lat, cop_raw, cop_fields)):
            iy, ix = bounds_index(lon, lat, bounds)
            for label, field in (("raw_native", raw), ("gaussian_dog_25_200km", fields[0]), ("bessel3_dog_25_200km", fields[1])):
                rows.append({"region": region, "source": source, "field": label, **summary(field[np.ix_(iy, ix)])})
    with (args.output_root / "raw_and_mesoscale_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[compare] wrote {args.output_root}", flush=True)


if __name__ == "__main__":
    main()
