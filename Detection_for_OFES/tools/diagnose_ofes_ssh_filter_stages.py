"""Diagnose OFES SSH filtering stages without changing production outputs."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

from Detection_for_OFES.tools.build_ofes_meso_filter import (
    horizontal_scale_filter,
    load_rossby_radius_profile,
    nan_gaussian_lowpass,
    rossby_cutoff_profile,
)


DEFAULT_INPUT_ROOT = Path(r"E:\DATA\01_Eddy_correspond\02_OFES\origin_compatible_filter")
DEFAULT_ROSSBY = Path(
    r"E:\DATA\01_Eddy_correspond\02_OFES\rossby_radius_chelton1998"
    r"\unzip\fecampos-campos2025-a5d24c0\rossrad.nc"
)
DEFAULT_CURRENT = Path(
    r"E:\DATA\01_Eddy_correspond\02_OFES\origin_compatible_filter_rossby_lower_upper180_full105"
)
DEFAULT_OUTPUT = Path(r"E:\DATA\01_Eddy_correspond\02_OFES\filter_diagnostics_19910101")
DEFAULT_CANDIDATE = Path(
    r"E:\DATA\01_Eddy_correspond\02_OFES"
    r"\origin_compatible_filter_10d_rossbymin25_upper180_local_y_jan01_smoke"
)


def daily_path(root: Path, day: date) -> Path:
    path = root / f"global_phy_{day:%Y%m%d}.nc"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def read_ssh(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with Dataset(path) as ds:
        lon = np.asarray(ds.variables["longitude"][:], dtype="f8")
        lat = np.asarray(ds.variables["latitude"][:], dtype="f8")
        field = np.ma.asarray(ds.variables["zos_glor"][0], dtype="f8").filled(np.nan)
    field[~np.isfinite(field)] = np.nan
    return lon, lat, field


def available_days(day: date, start: date, end: date, window: int = 10) -> list[date]:
    half = max(0, (window - 1) // 2)
    lo = max(start, day - timedelta(days=half))
    hi = min(end, day + timedelta(days=window - half - 1))
    return [day + timedelta(days=i) for i in range((hi - lo).days + 1) for day in [lo + timedelta(days=i)]]


def temporal_mean(root: Path, days: list[date]) -> np.ndarray:
    total = None
    count = None
    for day in days:
        _, _, field = read_ssh(daily_path(root, day))
        finite = np.isfinite(field)
        if total is None:
            total = np.zeros_like(field, dtype="f8")
            count = np.zeros_like(field, dtype="f8")
        total[finite] += field[finite]
        count[finite] += 1.0
    return np.divide(total, count, out=np.full_like(total, np.nan), where=count > 0)


def metric(field: np.ndarray, region: tuple[int, int, int, int] | None = None) -> dict[str, float]:
    a = field if region is None else field[region[2] : region[3], region[0] : region[1]]
    finite = np.isfinite(a)
    if not finite.any():
        return {"finite_fraction": 0.0}
    row = np.nanmean(a, axis=1)
    col = np.nanmean(a, axis=0)
    gx = np.diff(a, axis=1)
    gy = np.diff(a, axis=0)
    lap = -4.0 * a[1:-1, 1:-1] + a[:-2, 1:-1] + a[2:, 1:-1] + a[1:-1, :-2] + a[1:-1, 2:]
    gx_std = float(np.nanstd(gx))
    gy_std = float(np.nanstd(gy))
    return {
        "finite_fraction": float(finite.mean()),
        "std": float(np.nanstd(a)),
        "q95_abs": float(np.nanpercentile(np.abs(a), 95)),
        "q99_abs": float(np.nanpercentile(np.abs(a), 99)),
        "row_mean_std": float(np.nanstd(row)),
        "col_mean_std": float(np.nanstd(col)),
        "gradient_x_std": gx_std,
        "gradient_y_std": gy_std,
        "gradient_x_over_y": gx_std / gy_std if gy_std > 0 else float("nan"),
        "laplacian_std": float(np.nanstd(lap)),
        "relative_laplacian": float(np.nanstd(lap) / max(np.nanstd(a), 1.0e-12)),
    }


def plot_stages(
    stages: dict[str, np.ndarray],
    lon: np.ndarray,
    lat: np.ndarray,
    path: Path,
    title: str,
    bounds: tuple[float, float, float, float] | None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if bounds is None:
        x0, x1, y0, y1 = float(lon.min()), float(lon.max()), float(lat.min()), float(lat.max())
    else:
        x0, x1, y0, y1 = bounds
    ix = np.flatnonzero((lon >= x0) & (lon <= x1))
    iy = np.flatnonzero((lat >= y0) & (lat <= y1))
    if ix.size < 2 or iy.size < 2:
        raise ValueError(f"No grid in region {bounds}")
    fields = [field[np.ix_(iy, ix)] for field in stages.values()]
    vmax = max(float(np.nanpercentile(np.abs(field), 99.5)) for field in fields)
    vmax = max(vmax, 0.1)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
    for ax, (name, field), clipped in zip(axes.flat, stages.items(), fields):
        mesh = ax.pcolormesh(lon[ix], lat[iy], clipped, shading="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        mesh.set_rasterized(True)
        ax.set_title(name)
        x_tick_index = np.linspace(0, len(ix) - 1, min(6, len(ix)), dtype=int)
        y_tick_index = np.linspace(0, len(iy) - 1, min(6, len(iy)), dtype=int)
        ax.set_xticks(lon[ix[x_tick_index]])
        ax.set_xticklabels([f"{lon[ix[k]]:.0f}" for k in x_tick_index])
        ax.set_yticks(lat[iy[y_tick_index]])
        ax.set_yticklabels([f"{lat[iy[k]]:.0f}" for k in y_tick_index])
        ax.set_xlabel("longitude")
        ax.set_ylabel("latitude")
    for ax in axes.flat[len(stages) :]:
        ax.axis("off")
    fig.suptitle(f"{title}\nShared color scale: +/- {vmax:.3g} cm", fontsize=14)
    fig.colorbar(mesh, ax=axes.ravel().tolist(), label="SSHA (cm)", shrink=0.86)
    fig.savefig(path, dpi=150)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--day", default="1991-01-01")
    p.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    p.add_argument("--rossby-radius-path", type=Path, default=DEFAULT_ROSSBY)
    p.add_argument("--current-filter-root", type=Path, default=DEFAULT_CURRENT)
    p.add_argument("--candidate-filter-root", type=Path, default=DEFAULT_CANDIDATE)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    return p.parse_args()


def plot_current_candidate(
    current: np.ndarray,
    candidate: np.ndarray,
    lon: np.ndarray,
    lat: np.ndarray,
    path: Path,
    title: str,
    bounds: tuple[float, float, float, float] | None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if bounds is None:
        x0, x1, y0, y1 = float(lon.min()), float(lon.max()), float(lat.min()), float(lat.max())
    else:
        x0, x1, y0, y1 = bounds
    ix = np.flatnonzero((lon >= x0) & (lon <= x1))
    iy = np.flatnonzero((lat >= y0) & (lat <= y1))
    fields = [current[np.ix_(iy, ix)], candidate[np.ix_(iy, ix)]]
    vmax = max(float(np.nanpercentile(np.abs(field), 99.5)) for field in fields)
    vmax = max(vmax, 0.1)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), constrained_layout=True)
    labels = ["production | 1d, 0.5R1-180 km, median y", "candidate | 10d, max(0.5R1,25)-180 km, local y"]
    for ax, field, label in zip(axes, fields, labels):
        mesh = ax.pcolormesh(lon[ix], lat[iy], field, shading="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        mesh.set_rasterized(True)
        ax.set_title(label)
        ax.set_xlabel("longitude")
        ax.set_ylabel("latitude")
    fig.suptitle(f"{title}\nShared color scale: +/- {vmax:.3g} cm", fontsize=13)
    fig.colorbar(mesh, ax=axes.tolist(), label="SSHA (cm)", shrink=0.9)
    fig.savefig(path, dpi=160)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    args = parse_args()
    target = date.fromisoformat(args.day)
    args.output_root.mkdir(parents=True, exist_ok=True)
    lon, lat, input_field = read_ssh(daily_path(args.input_root, target))
    days = available_days(target, date(1991, 1, 1), date(1991, 1, 19), 10)
    mean10 = temporal_mean(args.input_root, days)
    r1 = load_rossby_radius_profile(args.rossby_radius_path, lat)
    lower = rossby_cutoff_profile(r1, factor=0.5, min_km=10.0, max_km=180.0)
    upper = np.full(lat.size, 180.0, dtype="f8")
    stages = {
        "1 input daily | raw-minus-JanMean": input_field,
        "2 available-day | 10d mean": mean10,
        "3 LP(0.5R1) | small-scale lowpass": nan_gaussian_lowpass(mean10, lon, lat, lower),
        "4 LP(0.5R1)-LP(180) | current DoG diagnostic": horizontal_scale_filter(mean10, lon, lat, lower, upper, "bandpass"),
        "5 LP(25)-LP(180) | fixed lower cutoff": horizontal_scale_filter(mean10, lon, lat, 25.0, 180.0, "bandpass"),
        "6 LP(50)-LP(180) | fixed lower cutoff": horizontal_scale_filter(mean10, lon, lat, 50.0, 180.0, "bandpass"),
    }
    current_path = daily_path(args.current_filter_root, target)
    _, _, production_current = read_ssh(current_path)
    candidate_path = daily_path(args.candidate_filter_root, target)
    _, _, candidate = read_ssh(candidate_path)
    regions = {
        "global": None,
        "north_pacific": (150.0, 250.0, 20.0, 60.0),
        "south_pacific": (170.0, 290.0, -55.0, -20.0),
        "acc": (0.0, 360.0, -70.0, -40.0),
    }
    metrics: list[dict[str, object]] = []
    for region_name, bounds in regions.items():
        plot_stages(stages, lon, lat, args.output_root / f"ssha_filter_stages_{region_name}_{target:%Y%m%d}.png", f"OFES SSHA filter stages | {region_name} | {target}", bounds)
        plot_current_candidate(
            production_current, candidate, lon, lat,
            args.output_root / f"ssha_production_vs_candidate_{region_name}_{target:%Y%m%d}.png",
            f"OFES production vs candidate filter | {region_name} | {target}", bounds,
        )
        for stage_name, field in {
            **stages,
            "production current | 1d 0.5R1-180": production_current,
            "candidate | 10d max(0.5R1,25)-180 local-y": candidate,
        }.items():
            row = {"region": region_name, "stage": stage_name, **metric(field, None if bounds is None else (int(np.searchsorted(lon, bounds[0], side="left")), int(np.searchsorted(lon, bounds[1], side="right")), int(np.searchsorted(lat, bounds[2], side="left")), int(np.searchsorted(lat, bounds[3], side="right"))))}
            metrics.append(row)
    csv_path = args.output_root / f"ssha_filter_stage_metrics_{target:%Y%m%d}.csv"
    keys = sorted({key for row in metrics for key in row})
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(metrics)
    manifest = {
        "status": "complete",
        "target_day": target.isoformat(),
        "input_root": str(args.input_root),
        "input_science_tag": "raw_minus_jan_mean_diagnostic",
        "temporal_days_used": [day.isoformat() for day in days],
        "rossby_radius_path": str(args.rossby_radius_path),
        "rossby_lower_cutoff_min_km": float(np.nanmin(lower)),
        "rossby_lower_cutoff_max_km": float(np.nanmax(lower)),
        "upper_cutoff_km": 180.0,
        "production_current_filter": str(current_path),
        "candidate_filter": str(candidate_path),
        "stages": list(stages),
        "regions": regions,
        "metrics_csv": str(csv_path),
    }
    (args.output_root / f"ssha_filter_diagnostic_manifest_{target:%Y%m%d}.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output_root": str(args.output_root), "metrics": str(csv_path), "days_used": [d.isoformat() for d in days]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
