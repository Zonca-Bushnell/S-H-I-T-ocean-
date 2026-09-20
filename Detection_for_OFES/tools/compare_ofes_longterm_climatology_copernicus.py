"""Compare OFES January-climatology anomalies with Copernicus SLA.

This is diagnostic-only. It never writes a detection input, catalogue, or
vertical-extension result.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
from pathlib import Path

import numpy as np

from Detection_for_OFES.tools.compare_raw_ofes_copernicus_mesoscale import (
    COPERNICUS_PATH,
    OFES_ROOT,
    bessel_kernel,
    bandpass,
    bounds_index,
    gaussian_kernel,
    read_copernicus,
    read_ofes,
    robust_limit,
    summary,
)


CLIMATOLOGY_PATH = Path(
    r"E:\DATA\01_Eddy_correspond\02_OFES\ofes2_monthly_climatology_1993_2012"
    r"\climatology\ofes2_eta_pair_h_monthly_climatology_1993_2012.npz"
)
OUTPUT_ROOT = Path(
    r"E:\DATA\01_Eddy_correspond\02_OFES\filter_diagnostics_19910101"
    r"\ofes_longterm_january_climatology_vs_copernicus_1991_1993"
)
GAUSSIAN_FWHM_TO_HALF_POWER_WAVELENGTH = 2.0 * np.pi / (2.354820045 * np.sqrt(np.log(2.0)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ofes-root", type=Path, default=OFES_ROOT)
    parser.add_argument("--ofes-day", default="1991-01-01")
    parser.add_argument("--copernicus-path", type=Path, default=COPERNICUS_PATH)
    parser.add_argument("--copernicus-day", default="1993-01-01")
    parser.add_argument("--climatology-path", type=Path, default=CLIMATOLOGY_PATH)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--small-km", type=float, default=25.0)
    parser.add_argument("--large-km", type=float, default=200.0)
    parser.add_argument("--min-valid-weight-fraction", type=float, default=0.95)
    parser.add_argument("--stage", choices=("all", "filter", "plot"), default="all")
    return parser.parse_args()


def read_january_climatology(path: Path, expected_lon: np.ndarray, expected_lat: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
    with np.load(path, allow_pickle=False) as data:
        lon = np.asarray(data["longitude"], dtype="f8")
        lat = np.asarray(data["latitude"], dtype="f8")
        months = np.asarray(data["months"])
        month_index = int(np.flatnonzero(months == 1)[0])
        h = np.asarray(data["h_monthly_mean_cm"][month_index], dtype="f8")
        count = np.asarray(data["h_valid_sample_count"][month_index])
    if not (
        lon.shape == expected_lon.shape
        and lat.shape == expected_lat.shape
        and np.allclose(lon, expected_lon, rtol=0.0, atol=1.0e-9)
        and np.allclose(lat, expected_lat, rtol=0.0, atol=1.0e-9)
    ):
        raise RuntimeError("January climatology grid does not match OFES native SSH grid")
    h[~np.isfinite(h)] = np.nan
    return h, {
        "path": str(path),
        "baseline": "January 1993-2012 climatology of H = eta - (pair - 1000)",
        "finite_fraction": float(np.isfinite(h).mean()),
        "valid_sample_count_min": int(count[np.isfinite(h)].min()),
        "valid_sample_count_max": int(count[np.isfinite(h)].max()),
    }


def plot_reference_fields(
    ofes_lon: np.ndarray,
    ofes_lat: np.ndarray,
    ofes_anomaly: np.ndarray,
    cop_lon: np.ndarray,
    cop_lat: np.ndarray,
    cop_sla: np.ndarray,
    output: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    limit = robust_limit([ofes_anomaly[::4, ::4], cop_sla[::4, ::4]])
    figure, axes = plt.subplots(1, 2, figsize=(18, 7), constrained_layout=True)
    for axis, (name, lon, lat, field) in zip(
        axes,
        (
            ("OFES H' | H(1991-01-01) - January climatology", ofes_lon, ofes_lat, ofes_anomaly),
            ("Copernicus/DUACS SLA | 1993-01-01", cop_lon, cop_lat, cop_sla),
        ),
    ):
        image = setup_image(axis, lon[::4], lat[::4], field[::4, ::4], name, limit)
        figure.colorbar(image, ax=axis, pad=0.015, label="cm")
    figure.suptitle(f"Long-term monthly-reference anomaly comparison | shared scale +/- {limit:.1f} cm", fontsize=14)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def setup_image(axis, lon: np.ndarray, lat: np.ndarray, field: np.ndarray, title: str, limit: float):
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#22262d")
    image = axis.imshow(
        field, origin="lower", aspect="auto", interpolation="nearest",
        extent=(float(lon[0]), float(lon[-1]), float(lat[0]), float(lat[-1])),
        cmap=cmap, vmin=-limit, vmax=limit,
    )
    axis.set_title(title, fontsize=10)
    axis.set_xlabel("Longitude (degrees east)")
    axis.set_ylabel("Latitude")
    axis.grid(True, color="k", alpha=0.2, linewidth=0.4)
    return image


def display_subset(
    lon: np.ndarray, lat: np.ndarray, field: np.ndarray, bounds: tuple[float, float, float, float] | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if bounds is None:
        return lon[::4], lat[::4], field[::4, ::4]
    iy, ix = bounds_index(lon, lat, bounds)
    return lon[ix], lat[iy], field[np.ix_(iy, ix)]


def plot_bandpass_comparison(
    ofes: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    copernicus: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    output: Path,
    title: str,
    bounds: tuple[float, float, float, float] | None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    selected: list[np.ndarray] = []
    for lon, lat, gaussian, bessel in (ofes, copernicus):
        _, _, gaussian_view = display_subset(lon, lat, gaussian, bounds)
        _, _, bessel_view = display_subset(lon, lat, bessel, bounds)
        selected.extend((gaussian_view, bessel_view))
    limit = robust_limit(selected)
    figure, axes = plt.subplots(2, 2, figsize=(18, 12), constrained_layout=True)
    for row, (source, data) in enumerate((("OFES H'", ofes), ("Copernicus SLA", copernicus))):
        lon, lat, gaussian, bessel = data
        lon_view, lat_view, gaussian_view = display_subset(lon, lat, gaussian, bounds)
        _, _, bessel_view = display_subset(lon, lat, bessel, bounds)
        for column, (label, field) in enumerate((("Gaussian DoG", gaussian), ("Bessel order-3", bessel))):
            field_view = gaussian_view if column == 0 else bessel_view
            image = setup_image(
                axes[row, column], lon_view, lat_view, field_view,
                f"{source} | {label}", limit,
            )
    figure.colorbar(image, ax=axes.ravel().tolist(), pad=0.012, label="25-200 km band-pass (cm)")
    figure.suptitle(f"{title}\nShared physical scales and colour scale: +/- {limit:.2f} cm", fontsize=14)
    figure.savefig(output, dpi=160)
    plt.close(figure)


def cache_paths(output_root: Path) -> dict[str, Path]:
    cache_root = output_root / "cache"
    return {name: cache_root / f"{name}.npy" for name in (
        "ofes_lon", "ofes_lat", "ofes_anomaly", "cop_lon", "cop_lat", "cop_sla",
        "ofes_gaussian", "ofes_bessel", "cop_gaussian", "cop_bessel",
    )}


def run_filter_stage(args: argparse.Namespace) -> None:
    paths = cache_paths(args.output_root)
    paths["ofes_lon"].parent.mkdir(parents=True, exist_ok=True)
    ofes_lon, ofes_lat, ofes_raw, _ = read_ofes(args.ofes_root, args.ofes_day)
    climatology, _ = read_january_climatology(args.climatology_path, ofes_lon, ofes_lat)
    ofes_anomaly = ofes_raw - climatology
    ofes_anomaly[~np.isfinite(ofes_raw) | ~np.isfinite(climatology)] = np.nan
    cop_lon, cop_lat, cop_sla, _ = read_copernicus(args.copernicus_path)
    for name, values in (("ofes_lon", ofes_lon), ("ofes_lat", ofes_lat), ("ofes_anomaly", ofes_anomaly), ("cop_lon", cop_lon), ("cop_lat", cop_lat), ("cop_sla", cop_sla)):
        np.save(paths[name], values)
    print("[longterm-compare] cached reference fields", flush=True)
    bessel_small = args.small_km * GAUSSIAN_FWHM_TO_HALF_POWER_WAVELENGTH
    bessel_large = args.large_km * GAUSSIAN_FWHM_TO_HALF_POWER_WAVELENGTH
    jobs = (
        ("ofes_gaussian", ofes_anomaly, ofes_lon, ofes_lat, gaussian_kernel, args.small_km, args.large_km),
        ("ofes_bessel", ofes_anomaly, ofes_lon, ofes_lat, bessel_kernel, bessel_small, bessel_large),
        ("cop_gaussian", cop_sla, cop_lon, cop_lat, gaussian_kernel, args.small_km, args.large_km),
        ("cop_bessel", cop_sla, cop_lon, cop_lat, bessel_kernel, bessel_small, bessel_large),
    )
    for name, field, lon, lat, kernel, small, large in jobs:
        result = bandpass(field, lon, lat, kernel, small, large, args.min_valid_weight_fraction).astype("f4")
        np.save(paths[name], result)
        del result
        gc.collect()
        print(f"[longterm-compare] cached {name}", flush=True)


def run_plot_stage(args: argparse.Namespace) -> None:
    paths = cache_paths(args.output_root)
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Run --stage filter first; missing cache files: {missing}")
    arrays = {name: np.load(path, mmap_mode="r") for name, path in paths.items()}
    ofes_lon, ofes_lat, ofes_anomaly = arrays["ofes_lon"], arrays["ofes_lat"], arrays["ofes_anomaly"]
    cop_lon, cop_lat, cop_sla = arrays["cop_lon"], arrays["cop_lat"], arrays["cop_sla"]
    ofes_fields = (arrays["ofes_gaussian"], arrays["ofes_bessel"])
    cop_fields = (arrays["cop_gaussian"], arrays["cop_bessel"])
    ofes_meta = {"definition": "H_OFES = eta - (pressur - 1000)", "grid": "native OFES SSH grid; cached for this diagnostic"}
    cop_meta = {"definition": "Copernicus/DUACS L4 SLA above the product mean sea surface", "grid": "native Copernicus L4 grid; cached for this diagnostic"}
    climatology_meta = {
        "path": str(args.climatology_path),
        "baseline": "January 1993-2012 climatology of H = eta - (pair - 1000)",
        "cached_field_policy": "same native OFES grid; coordinates validated during --stage filter",
    }
    regions: dict[str, tuple[float, float, float, float] | None] = {
        "global": None,
        "north_pacific_open_ocean": (190.0, 245.0, 25.0, 55.0),
        "south_pacific_open_ocean": (190.0, 280.0, -50.0, -30.0),
    }
    print("[longterm-compare] plotting reference fields", flush=True)
    plot_reference_fields(ofes_lon, ofes_lat, ofes_anomaly, cop_lon, cop_lat, cop_sla, args.output_root / "ofes_longterm_january_anomaly_vs_copernicus_sla.png")
    for name, bounds in regions.items():
        print(f"[longterm-compare] plotting {name}", flush=True)
        plot_bandpass_comparison(
            (ofes_lon, ofes_lat, *ofes_fields), (cop_lon, cop_lat, *cop_fields),
            args.output_root / f"ofes_longterm_january_climatology_bandpass_{name}.png",
            f"OFES H' (January 1993-2012 climatology) vs Copernicus SLA | {name}", bounds,
        )
    rows: list[dict[str, object]] = []
    for region, bounds in regions.items():
        for source, lon, lat, fields in (
            ("OFES_Hprime_Jan1993_2012", ofes_lon, ofes_lat, (ofes_anomaly, *ofes_fields)),
            ("Copernicus_SLA", cop_lon, cop_lat, (cop_sla, *cop_fields)),
        ):
            iy, ix = bounds_index(lon, lat, bounds)
            for label, field in zip(("reference_anomaly_or_sla", "gaussian_dog_25_200km", "bessel3_dog_25_200km"), fields):
                rows.append({"region": region, "source": source, "field": label, **summary(field[np.ix_(iy, ix)])})
    with (args.output_root / "ofes_longterm_january_climatology_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        "status": "complete", "comparison": "Cross-year, same-calendar-day qualitative comparison; not pointwise validation.",
        "ofes": {"day": args.ofes_day, **ofes_meta},
        "ofes_anomaly": {"formula": "H_OFES(1991-01-01) - H_climatology_January_1993_2012", **climatology_meta},
        "copernicus": {"day": args.copernicus_day, **cop_meta},
        "bandpass": {"formula": "LP_25km(field) - LP_200km(field)", "gaussian_fwhm_km": [args.small_km, args.large_km], "bessel_order": 3, "longitude_boundary": "periodic", "latitude_boundary": "nearest/non-periodic", "nan_handling": "normalized convolution"},
        "outputs": ["ofes_longterm_january_anomaly_vs_copernicus_sla.png", *[f"ofes_longterm_january_climatology_bandpass_{name}.png" for name in regions], "ofes_longterm_january_climatology_metrics.csv"],
    }
    (args.output_root / "ofes_longterm_january_climatology_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[longterm-compare] wrote {args.output_root}", flush=True)


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    if args.stage == "filter":
        run_filter_stage(args)
        return
    if args.stage == "plot":
        run_plot_stage(args)
        return
    ofes_lon, ofes_lat, ofes_raw, ofes_meta = read_ofes(args.ofes_root, args.ofes_day)
    climatology, climatology_meta = read_january_climatology(args.climatology_path, ofes_lon, ofes_lat)
    ofes_anomaly = ofes_raw - climatology
    ofes_anomaly[~np.isfinite(ofes_raw) | ~np.isfinite(climatology)] = np.nan
    cop_lon, cop_lat, cop_sla, cop_meta = read_copernicus(args.copernicus_path)
    print("[longterm-compare] loaded OFES H', January climatology, and Copernicus SLA", flush=True)

    bessel_small = args.small_km * GAUSSIAN_FWHM_TO_HALF_POWER_WAVELENGTH
    bessel_large = args.large_km * GAUSSIAN_FWHM_TO_HALF_POWER_WAVELENGTH
    ofes_fields = (
        bandpass(ofes_anomaly, ofes_lon, ofes_lat, gaussian_kernel, args.small_km, args.large_km, args.min_valid_weight_fraction).astype("f4"),
        bandpass(ofes_anomaly, ofes_lon, ofes_lat, bessel_kernel, bessel_small, bessel_large, args.min_valid_weight_fraction).astype("f4"),
    )
    cop_fields = (
        bandpass(cop_sla, cop_lon, cop_lat, gaussian_kernel, args.small_km, args.large_km, args.min_valid_weight_fraction).astype("f4"),
        bandpass(cop_sla, cop_lon, cop_lat, bessel_kernel, bessel_small, bessel_large, args.min_valid_weight_fraction).astype("f4"),
    )
    print("[longterm-compare] completed shared 25-200 km Gaussian and Bessel band-passes", flush=True)

    regions: dict[str, tuple[float, float, float, float] | None] = {
        "global": None,
        "north_pacific_open_ocean": (190.0, 245.0, 25.0, 55.0),
        "south_pacific_open_ocean": (190.0, 280.0, -50.0, -30.0),
    }
    plot_reference_fields(ofes_lon, ofes_lat, ofes_anomaly, cop_lon, cop_lat, cop_sla, args.output_root / "ofes_longterm_january_anomaly_vs_copernicus_sla.png")
    for name, bounds in regions.items():
        plot_bandpass_comparison(
            (ofes_lon, ofes_lat, *ofes_fields),
            (cop_lon, cop_lat, *cop_fields),
            args.output_root / f"ofes_longterm_january_climatology_bandpass_{name}.png",
            f"OFES H' (January 1993-2012 climatology) vs Copernicus SLA | {name}",
            bounds,
        )
    rows: list[dict[str, object]] = []
    for region, bounds in regions.items():
        for source, lon, lat, fields in (
            ("OFES_Hprime_Jan1993_2012", ofes_lon, ofes_lat, (ofes_anomaly, *ofes_fields)),
            ("Copernicus_SLA", cop_lon, cop_lat, (cop_sla, *cop_fields)),
        ):
            iy, ix = bounds_index(lon, lat, bounds)
            for label, field in zip(("reference_anomaly_or_sla", "gaussian_dog_25_200km", "bessel3_dog_25_200km"), fields):
                rows.append({"region": region, "source": source, "field": label, **summary(field[np.ix_(iy, ix)])})
    with (args.output_root / "ofes_longterm_january_climatology_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        "status": "complete",
        "comparison": "Cross-year, same-calendar-day qualitative comparison; not pointwise validation.",
        "ofes": {"day": args.ofes_day, **ofes_meta},
        "ofes_anomaly": {"formula": "H_OFES(1991-01-01) - H_climatology_January_1993_2012", **climatology_meta},
        "copernicus": {"day": args.copernicus_day, **cop_meta},
        "bandpass": {"formula": "LP_25km(field) - LP_200km(field)", "gaussian_fwhm_km": [args.small_km, args.large_km], "bessel_order": 3, "longitude_boundary": "periodic", "latitude_boundary": "nearest/non-periodic", "nan_handling": "normalized convolution"},
        "outputs": ["ofes_longterm_january_anomaly_vs_copernicus_sla.png", *[f"ofes_longterm_january_climatology_bandpass_{name}.png" for name in regions], "ofes_longterm_january_climatology_metrics.csv"],
    }
    (args.output_root / "ofes_longterm_january_climatology_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[longterm-compare] wrote {args.output_root}", flush=True)


if __name__ == "__main__":
    main()
