"""Calibrate regional SSH amplitude profiles from saved OFES candidate tables.

This is intentionally an offline step: it never reads OFES NetCDF/.dta fields.
It uses the permissive candidate pool and optionally a META reference table to
choose a damped, regional density-matching threshold profile.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


BANDS = [
    ("equatorial", -15.0, 15.0, 0.80),
    ("subtropical", 15.0, 30.0, 0.60),
    ("midlatitude", 30.0, 45.0, 0.40),
    ("highlatitude", 45.0, 60.0, 0.25),
]

REGIONS = {
    "north_pacific_west_open": (145.0, 180.0, 20.0, 60.0),
    "north_pacific_central_open": (180.0, 220.0, 20.0, 60.0),
    "north_pacific_east_open": (220.0, 260.0, 20.0, 60.0),
    "south_pacific_open": (160.0, 280.0, -60.0, -20.0),
    "south_indian_open": (50.0, 110.0, -60.0, -20.0),
    "north_atlantic_open": (300.0, 360.0, 20.0, 60.0),
    "south_atlantic_open": (300.0, 360.0, -60.0, -20.0),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-profile", type=Path, required=True)
    parser.add_argument("--output-summary", type=Path, default=None)
    parser.add_argument("--meta-table", type=Path, default=None)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--feedback-gamma", type=float, default=0.35)
    parser.add_argument("--target-density-multiplier", type=float, default=1.0)
    return parser.parse_args()


def read_candidates(root: Path) -> pd.DataFrame:
    paths = sorted(root.glob("daily_runs/*/centers_hua_style.csv"))
    if not paths:
        paths = sorted(root.glob("raw_detection/daily_runs/*/centers_hua_style.csv"))
    if not paths:
        raise FileNotFoundError(f"No daily centers_hua_style.csv under {root}")
    frames = [pd.read_csv(path) for path in paths]
    data = pd.concat(frames, ignore_index=True)
    if "depth_index" in data:
        data = data[data["depth_index"].astype(int).eq(0)].copy()
    lon = numeric(data, ["ssh_contour_center_lon", "center_lon", "seed_lon"])
    lat = numeric(data, ["ssh_contour_center_lat", "center_lat", "seed_lat"])
    amp = numeric(data, ["ssh_contour_amplitude_cm"])
    data["_lon"] = lon
    data["_lat"] = lat
    data["_amp"] = amp
    return data[np.isfinite(lon) & np.isfinite(lat)].copy()


def numeric(data: pd.DataFrame, names: list[str]) -> pd.Series:
    for name in names:
        if name in data:
            return pd.to_numeric(data[name], errors="coerce")
    return pd.Series(np.nan, index=data.index, dtype="f8")


def area_km2(bbox: tuple[float, float, float, float]) -> float:
    lon0, lon1, lat0, lat1 = bbox
    radius = 6371.0
    return (math.pi / 180.0) * radius**2 * (lon1 - lon0) * (
        math.sin(math.radians(lat1)) - math.sin(math.radians(lat0))
    )


def subset(data: pd.DataFrame, bbox: tuple[float, float, float, float], band: tuple[str, float, float, float]) -> pd.DataFrame:
    _, _, lat0, lat1 = bbox
    name, band0, band1, _ = band
    intervals = [(band0, band1), (-band1, -band0)]
    if not any(max(lat0, lo) < min(lat1, hi) for lo, hi in intervals):
        return data.iloc[0:0]
    abs_lat = data["_lat"].abs()
    return data[
        data["_lon"].between(bbox[0], bbox[1])
        & data["_lat"].between(bbox[2], bbox[3])
        & abs_lat.between(band0, band1)
        & np.isfinite(data["_amp"])
    ]


def band_bbox(region_bbox: tuple[float, float, float, float], band: tuple[str, float, float, float]) -> tuple[float, float, float, float]:
    """Intersect a positive absolute-latitude band with a hemispheric box."""
    lon0, lon1, lat0, lat1 = region_bbox
    _, band0, band1, _ = band
    if lat1 <= 0:
        return lon0, lon1, max(lat0, -band1), min(lat1, -band0)
    return lon0, lon1, max(lat0, band0), min(lat1, band1)


def meta_density(meta_table: Path | None, region: str, bbox: tuple[float, float, float, float], days: int) -> float | None:
    if meta_table is None or not meta_table.exists():
        return None
    path = meta_table
    meta = pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)
    lon = numeric(meta, ["lon", "longitude", "center_lon"])
    lat = numeric(meta, ["lat", "latitude", "center_lat"])
    meta = meta[np.isfinite(lon) & np.isfinite(lat)].copy()
    mask = lon.between(bbox[0], bbox[1]) & lat.between(bbox[2], bbox[3])
    count = int(mask.sum())
    if "date" in meta:
        observed_days = int(pd.to_datetime(meta.loc[mask, "date"], errors="coerce").dt.date.nunique())
        days = max(observed_days, 1)
    return count / max(area_km2(bbox) * max(days, 1), 1.0)


def calibrate(data: pd.DataFrame, args: argparse.Namespace) -> tuple[dict[str, object], pd.DataFrame]:
    days = int(data["date"].nunique()) if "date" in data else 1
    rows: list[dict[str, object]] = []
    prelim: list[dict[str, object]] = []
    for region, bbox in REGIONS.items():
        for band in BANDS:
            part = subset(data, bbox, band)
            if part.empty:
                continue
            initial = float(band[3])
            amplitudes = part["_amp"].to_numpy(dtype="f8")
            initial_count = int((amplitudes >= initial).sum())
            density = initial_count / max(area_km2(band_bbox(bbox, band)) * days, 1.0)
            prelim.append({"region": region, "band": band[0], "density": density})
    initial_density = {(r["region"], r["band"]): float(r["density"]) for r in prelim}
    profile_regions: list[dict[str, object]] = []
    for region, bbox in REGIONS.items():
        region_entry = {"name": region, "lon_min": bbox[0], "lon_max": bbox[1], "lat_min": bbox[2], "lat_max": bbox[3], "bands": []}
        for band in BANDS:
            name, b0, b1, initial = band
            part = subset(data, bbox, band)
            if part.empty:
                continue
            amplitudes = part["_amp"].to_numpy(dtype="f8")
            bbox_band = band_bbox(bbox, band)
            days_in_band = max(days, 1)
            target = meta_density(args.meta_table, region, bbox_band, days_in_band)
            target_source = "meta_table" if target is not None else "pooled_open_ocean_reference"
            if target is None:
                # A pre-filtered OFES catalog cannot provide a cross-region
                # target. Keep this region's own density neutral until a META
                # reference table is supplied.
                target = initial_density.get((region, name), 0.0)
            target *= float(args.target_density_multiplier)
            threshold = float(initial)
            history: list[dict[str, object]] = []
            for iteration in range(max(1, int(args.iterations))):
                kept = amplitudes >= threshold
                count = int(kept.sum())
                observed = count / max(area_km2(bbox_band) * days_in_band, 1.0)
                q25 = float(np.nanpercentile(amplitudes, 25.0))
                frag = float((amplitudes < max(threshold, q25)).mean())
                history.append({"iteration": iteration, "threshold_cm": threshold, "density": observed, "count": count, "q25_amplitude_cm": q25, "fragmentation_proxy": frag})
                if observed > 0 and target > 0:
                    threshold *= math.exp(float(args.feedback_gamma) * math.log(observed / target))
                if target_source == "meta_table":
                    # META determines the density direction; local OFES
                    # amplitude distribution damps the update.
                    threshold = 0.75 * threshold + 0.25 * q25
            final = history[-1]
            reason = "density_feedback" if target_source == "meta_table" else "density_feedback_pooled_reference"
            entry = {"name": name, "lat_min": b0, "lat_max": b1, "threshold_cm": float(final["threshold_cm"]), "initial_threshold_cm": initial, "iteration": int(final["iteration"]), "target_density_per_km2_day": float(target), "target_density_source": target_source, "adjustment_reason": reason, "source": "regional_amplitude_density_feedback", "local_amplitude_percentile": 25.0, "local_amplitude_q25_cm": float(final["q25_amplitude_cm"]), "fragmentation_rate": float(final["fragmentation_proxy"]), "contour_quality_score": float(1.0 - final["fragmentation_proxy"]), "history": history}
            region_entry["bands"].append(entry)
            rows.append({"region": region, "lat_band": name, **{k: v for k, v in entry.items() if k != "history"}, "observed_initial_density_per_km2_day": next((p["density"] for p in prelim if p["region"] == region and p["band"] == name), np.nan)})
        if region_entry["bands"]:
            profile_regions.append(region_entry)
    profile = {"version": "regional_amplitude_density_feedback_v1", "days": days, "source_root": str(args.source_root), "threshold_policy": "initial values are seeds only; final thresholds are feedback-calibrated", "regions": profile_regions}
    return profile, pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    data = read_candidates(args.source_root)
    profile, summary = calibrate(data, args)
    args.output_profile.parent.mkdir(parents=True, exist_ok=True)
    args.output_profile.write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_path = args.output_summary or args.output_profile.with_suffix(".csv")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)
    print(json.dumps({"profile": str(args.output_profile), "summary": str(summary_path), "regions": len(profile["regions"]), "rows": len(summary)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
