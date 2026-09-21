"""Plot native OFES SSH under raw, A, B, and annual-MSS references.

This diagnostic is deliberately isolated from all filtering and detection
pipelines.  Every field stays on the native OFES eta grid.
"""

from __future__ import annotations

import argparse
import calendar
import csv
import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np

from Detection_for_OFES.ofes_io import ctl_path, parse_ctl, read_ssh_latlon_daily_only


OFES_ROOT = Path(r"F:\OFES\external_OFES2")
CLIMATOLOGY_ROOT = Path(
    r"E:\DATA\01_Eddy_correspond\02_OFES\ofes2_monthly_climatology_1993_2012"
)
OUTPUT_ROOT = Path(
    r"E:\DATA\01_Eddy_correspond\02_OFES\filter_diagnostics_19910101"
    r"\ofes_ssh_baseline_ab_c_comparison"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ofes-root", type=Path, default=OFES_ROOT)
    parser.add_argument("--climatology-root", type=Path, default=CLIMATOLOGY_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--day", default="1991-01-01")
    parser.add_argument("--start-year", type=int, default=1993)
    parser.add_argument("--end-year", type=int, default=2012)
    parser.add_argument("--old-mean-start", default="1991-01-01")
    parser.add_argument("--old-mean-end", default="1991-01-19")
    parser.add_argument("--overwrite-mss", action="store_true")
    return parser.parse_args()


def json_default(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def day_range(start_text: str, end_text: str) -> list[date]:
    start = date.fromisoformat(start_text)
    end = date.fromisoformat(end_text)
    if end < start:
        raise ValueError("End date must not precede start date")
    return [start + timedelta(days=offset) for offset in range((end - start).days + 1)]


def native_mean(data_root: Path, days: list[date]) -> np.ndarray:
    total: np.ndarray | None = None
    count: np.ndarray | None = None
    for index, current_day in enumerate(days, start=1):
        field = np.asarray(read_ssh_latlon_daily_only(data_root, current_day), dtype="f8")
        finite = np.isfinite(field)
        if total is None:
            total = np.zeros(field.shape, dtype="f8")
            count = np.zeros(field.shape, dtype="u1")
        total[finite] += field[finite]
        count[finite] += 1
        print(f"[baseline-2x2] old baseline {index}/{len(days)}: {current_day}", flush=True)
    assert total is not None and count is not None
    return np.divide(total, count, out=np.full(total.shape, np.nan), where=count > 0)


def read_january_climatology(
    path: Path,
    expected_lon: np.ndarray,
    expected_lat: np.ndarray,
) -> tuple[np.ndarray, dict[str, object]]:
    with np.load(path, allow_pickle=False) as data:
        lon = np.asarray(data["longitude"], dtype="f8")
        lat = np.asarray(data["latitude"], dtype="f8")
        months = np.asarray(data["months"])
        january_index = int(np.flatnonzero(months == 1)[0])
        field = np.asarray(data["h_monthly_mean_cm"][january_index], dtype="f8")
        count = np.asarray(data["h_valid_sample_count"][january_index])
    validate_grid(lon, lat, expected_lon, expected_lat, "January climatology")
    field[~np.isfinite(field)] = np.nan
    finite = np.isfinite(field)
    return field, {
        "path": str(path),
        "definition": "Mean January H over 1993-2012; H = eta - (pair - 1000)",
        "valid_sample_count_min": int(count[finite].min()),
        "valid_sample_count_max": int(count[finite].max()),
    }


def validate_grid(
    lon: np.ndarray,
    lat: np.ndarray,
    expected_lon: np.ndarray,
    expected_lat: np.ndarray,
    label: str,
) -> None:
    if not (
        lon.shape == expected_lon.shape
        and lat.shape == expected_lat.shape
        and np.allclose(lon, expected_lon, rtol=0.0, atol=1.0e-9)
        and np.allclose(lat, expected_lat, rtol=0.0, atol=1.0e-9)
    ):
        raise RuntimeError(f"{label} grid does not match the native OFES eta grid")


def build_or_read_annual_mss(
    climatology_root: Path,
    expected_lon: np.ndarray,
    expected_lat: np.ndarray,
    start_year: int,
    end_year: int,
    overwrite: bool,
) -> tuple[np.ndarray, np.ndarray, Path, list[str]]:
    output = (
        climatology_root
        / "climatology"
        / f"ofes2_h_annual_mss_{start_year}_{end_year}.npz"
    )
    if output.exists() and not overwrite:
        with np.load(output, allow_pickle=False) as data:
            lon = np.asarray(data["longitude"], dtype="f8")
            lat = np.asarray(data["latitude"], dtype="f8")
            mss = np.asarray(data["h_annual_mss_cm"], dtype="f8")
            valid_day_weight = np.asarray(data["valid_day_weight"], dtype="f8")
            months_used = [str(value) for value in np.asarray(data["months_used"])]
        validate_grid(lon, lat, expected_lon, expected_lat, "Annual MSS cache")
        return mss, valid_day_weight, output, months_used

    total = np.zeros((expected_lat.size, expected_lon.size), dtype="f8")
    valid_day_weight = np.zeros(total.shape, dtype="f8")
    months_used: list[str] = []
    expected_months = (end_year - start_year + 1) * 12
    processed = 0
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            label = f"{year:04d}{month:02d}"
            month_root = climatology_root / "raw_monthly" / label
            eta_path = month_root / "eta.npy"
            pair_path = month_root / "pair.npy"
            if not eta_path.exists() or not pair_path.exists():
                raise FileNotFoundError(f"Missing monthly cache for {label}: {month_root}")
            eta = np.load(eta_path, mmap_mode="r")
            pair = np.load(pair_path, mmap_mode="r")
            if eta.shape != total.shape or pair.shape != total.shape:
                raise RuntimeError(f"Unexpected monthly shape for {label}: {eta.shape}, {pair.shape}")
            h = np.asarray(eta, dtype="f8") - (np.asarray(pair, dtype="f8") - 1000.0)
            finite = np.isfinite(h) & (np.abs(h) < 1.0e10)
            days = calendar.monthrange(year, month)[1]
            total[finite] += h[finite] * days
            valid_day_weight[finite] += days
            months_used.append(label)
            processed += 1
            print(f"[baseline-2x2] annual MSS {processed}/{expected_months}: {label}", flush=True)

    mss = np.divide(
        total,
        valid_day_weight,
        out=np.full(total.shape, np.nan, dtype="f8"),
        where=valid_day_weight > 0,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        longitude=expected_lon,
        latitude=expected_lat,
        h_annual_mss_cm=mss.astype("f4"),
        valid_day_weight=valid_day_weight.astype("u2"),
        months_used=np.asarray(months_used, dtype="U6"),
        start_year=np.asarray(start_year, dtype="i2"),
        end_year=np.asarray(end_year, dtype="i2"),
        weighting=np.asarray("actual calendar days per cached monthly mean"),
        definition=np.asarray("H = eta - (pair - 1000)"),
    )
    return mss, valid_day_weight, output, months_used


def field_summary(name: str, field: np.ndarray) -> dict[str, object]:
    finite = np.isfinite(field)
    values = field[finite]
    result: dict[str, object] = {
        "field": name,
        "shape": list(field.shape),
        "finite_fraction": float(finite.mean()),
        "finite_count": int(finite.sum()),
    }
    for key, value in zip(
        ("minimum_cm", "q01_cm", "q05_cm", "median_cm", "mean_cm", "q95_cm", "q99_cm", "maximum_cm", "std_cm"),
        (
            np.min(values),
            np.percentile(values, 1),
            np.percentile(values, 5),
            np.median(values),
            np.mean(values),
            np.percentile(values, 95),
            np.percentile(values, 99),
            np.max(values),
            np.std(values),
        ),
    ):
        result[key] = float(value)
    return result


def robust_limit(fields: list[np.ndarray], percentile: float = 99.0) -> float:
    samples = []
    for field in fields:
        sample = field[::4, ::4]
        sample = np.abs(sample[np.isfinite(sample)])
        if sample.size:
            samples.append(sample)
    return max(1.0e-6, float(np.percentile(np.concatenate(samples), percentile)))


def plot_fields(
    lon: np.ndarray,
    lat: np.ndarray,
    raw: np.ndarray,
    baseline_b: np.ndarray,
    baseline_a: np.ndarray,
    baseline_c: np.ndarray,
    output_png: Path,
    output_pdf: Path,
) -> dict[str, float]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    raw_limit = robust_limit([raw], percentile=99.0)
    anomaly_limit = robust_limit([baseline_a, baseline_b, baseline_c], percentile=99.0)
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#24282e")
    figure, axes = plt.subplots(2, 2, figsize=(20, 10.8), constrained_layout=True, sharex=True, sharey=True)
    panels = (
        (axes[0, 0], raw, "Raw OFES SSH", r"$H_{OFES}(1991-01-01)$", raw_limit),
        (axes[0, 1], baseline_b, "B: Long-term January reference", r"$H-\overline{H}_{Jan}^{1993-2012}$", anomaly_limit),
        (axes[1, 0], baseline_a, "A: Old Jan01-Jan19 reference", r"$H-\overline{H}_{Jan01-Jan19}^{1991}$", anomaly_limit),
        (axes[1, 1], baseline_c, "C: 20-year annual MSS reference", r"$H-MSS_{1993-2012}$", anomaly_limit),
    )
    images = []
    for axis, field, title, formula, limit in panels:
        image = axis.imshow(
            field,
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            extent=(float(lon[0]), float(lon[-1]), float(lat[0]), float(lat[-1])),
            cmap=cmap,
            vmin=-limit,
            vmax=limit,
        )
        images.append(image)
        axis.set_title(f"{title}\n{formula}", fontsize=12)
        axis.set_xlabel("Longitude (degrees east)")
        axis.set_ylabel("Latitude")
        axis.set_xticks(np.arange(0, 361, 60))
        axis.set_yticks(np.arange(-60, 61, 30))
        axis.grid(True, color="black", alpha=0.18, linewidth=0.45)
    figure.colorbar(images[0], ax=axes[0, 0], label="Raw SSH (cm)", pad=0.015, shrink=0.86)
    figure.colorbar(images[1], ax=[axes[0, 1], axes[1, 0], axes[1, 1]], label="SSH anomaly (cm)", pad=0.015, shrink=0.86)
    figure.suptitle(
        "OFES SSH baseline definitions | native grid | 1991-01-01\n"
        "No running mean, spatial filtering, scale separation, or regridding",
        fontsize=16,
    )
    output_png.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_png, dpi=180)
    figure.savefig(output_pdf)
    plt.close(figure)
    return {"raw_limit_cm": raw_limit, "shared_anomaly_limit_cm": anomaly_limit}


def write_metrics(path: Path, rows: list[dict[str, object]]) -> None:
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    target_day = date.fromisoformat(args.day)
    eta_meta = parse_ctl(ctl_path(args.ofes_root, "eta"))
    lon = np.asarray(eta_meta.x.values, dtype="f8")
    lat = np.asarray(eta_meta.y.values, dtype="f8")
    raw = np.asarray(read_ssh_latlon_daily_only(args.ofes_root, target_day), dtype="f8")
    if raw.shape != (lat.size, lon.size):
        raise RuntimeError(f"Native SSH shape {raw.shape} does not match coordinates {(lat.size, lon.size)}")

    old_days = day_range(args.old_mean_start, args.old_mean_end)
    old_mean = native_mean(args.ofes_root, old_days)
    monthly_path = (
        args.climatology_root
        / "climatology"
        / f"ofes2_eta_pair_h_monthly_climatology_{args.start_year}_{args.end_year}.npz"
    )
    january_mean, january_meta = read_january_climatology(monthly_path, lon, lat)
    annual_mss, valid_day_weight, annual_mss_path, months_used = build_or_read_annual_mss(
        args.climatology_root,
        lon,
        lat,
        args.start_year,
        args.end_year,
        args.overwrite_mss,
    )

    anomaly_a = raw - old_mean
    anomaly_b = raw - january_mean
    anomaly_c = raw - annual_mss
    for result, baseline in ((anomaly_a, old_mean), (anomaly_b, january_mean), (anomaly_c, annual_mss)):
        result[~np.isfinite(raw) | ~np.isfinite(baseline)] = np.nan

    identity_left = anomaly_c - anomaly_b
    identity_right = january_mean - annual_mss
    identity_valid = np.isfinite(identity_left) & np.isfinite(identity_right)
    identity_delta = np.abs(identity_left[identity_valid] - identity_right[identity_valid])
    identity_max = float(identity_delta.max()) if identity_delta.size else float("nan")

    output_png = args.output_root / f"ofes_ssh_baseline_2x2_{target_day:%Y%m%d}.png"
    output_pdf = args.output_root / f"ofes_ssh_baseline_2x2_{target_day:%Y%m%d}.pdf"
    color_limits = plot_fields(lon, lat, raw, anomaly_b, anomaly_a, anomaly_c, output_png, output_pdf)
    metrics = [
        field_summary("raw_h_ofes", raw),
        field_summary("baseline_b_longterm_january_anomaly", anomaly_b),
        field_summary("baseline_a_jan01_jan19_anomaly", anomaly_a),
        field_summary("baseline_c_annual_mss_anomaly", anomaly_c),
        field_summary("c_minus_b_january_seasonal_climatology", anomaly_c - anomaly_b),
    ]
    metrics_path = args.output_root / f"ofes_ssh_baseline_2x2_metrics_{target_day:%Y%m%d}.csv"
    write_metrics(metrics_path, metrics)

    expected_total_days = sum(
        calendar.monthrange(year, month)[1]
        for year in range(args.start_year, args.end_year + 1)
        for month in range(1, 13)
    )
    manifest = {
        "status": "complete",
        "target_day": target_day.isoformat(),
        "native_grid": {"shape": list(raw.shape), "longitude_count": int(lon.size), "latitude_count": int(lat.size)},
        "units": "cm",
        "processing": "No running mean, spatial filtering, scale separation, or regridding.",
        "definitions": {
            "raw": "H_OFES(day) = eta(day) - (pressur(day) - 1000)",
            "B": "H_OFES(day) - mean January H over 1993-2012",
            "A": f"H_OFES(day) - mean H over {args.old_mean_start}..{args.old_mean_end}",
            "C": "H_OFES(day) - day-count-weighted annual MSS over all monthly means in 1993-2012",
        },
        "annual_mss": {
            "path": str(annual_mss_path),
            "months_used_count": len(months_used),
            "first_month": months_used[0],
            "last_month": months_used[-1],
            "expected_total_calendar_days": expected_total_days,
            "maximum_valid_day_weight": int(np.nanmax(valid_day_weight)),
            "minimum_positive_valid_day_weight": int(np.nanmin(valid_day_weight[valid_day_weight > 0])),
            "weighting": "actual calendar days in each cached monthly mean, with per-cell finite-value support",
        },
        "january_climatology": january_meta,
        "identity_validation": {
            "equation": "C - B = January climatology - annual MSS",
            "common_finite_fraction": float(identity_valid.mean()),
            "max_abs_error_cm": identity_max,
        },
        "color_limits": color_limits,
        "metrics": metrics,
        "outputs": {"png": str(output_png), "pdf": str(output_pdf), "metrics_csv": str(metrics_path)},
    }
    manifest_path = args.output_root / f"ofes_ssh_baseline_2x2_manifest_{target_day:%Y%m%d}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=json_default), encoding="utf-8")
    print(json.dumps(manifest, indent=2, default=json_default), flush=True)


if __name__ == "__main__":
    main()
