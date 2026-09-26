"""Optional object-day shape classification from accepted vertical centers."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

EARTH_RADIUS_M = 6_371_000.0


def _xy_m(lon: np.ndarray, lat: np.ndarray, lon0: float, lat0: float) -> tuple[np.ndarray, np.ndarray]:
    dlon = (lon - lon0 + 180.0) % 360.0 - 180.0
    return (
        np.radians(dlon) * EARTH_RADIUS_M * np.cos(np.radians(lat0)),
        np.radians(lat - lat0) * EARTH_RADIUS_M,
    )


def _monotonic_ratio(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    if values.size < 3:
        return np.nan
    delta = np.diff(values)
    return float(max(np.mean(delta >= 0.0), np.mean(delta <= 0.0)))


def _turns(direction: np.ndarray, valid: np.ndarray) -> tuple[float, float]:
    use = direction[valid & np.isfinite(direction)]
    if use.size < 2:
        return np.nan, np.nan
    delta = np.abs((np.diff(use) + 180.0) % 360.0 - 180.0)
    return float(np.mean(delta)), float(np.max(delta))


def object_day_metrics(rows: pd.DataFrame, min_layers: int = 6) -> dict[str, object]:
    rows = rows.sort_values("depth_index").copy()
    rows = rows.loc[rows.get("hua_pass", True).fillna(False).astype(bool)]
    first = rows.iloc[0]
    lon = rows["center_lon"].astype(float).to_numpy()
    lat = rows["center_lat"].astype(float).to_numpy()
    x, y = _xy_m(lon, lat, float(first["center_lon"]), float(first["center_lat"]))
    if "radius_km" in rows:
        radius_m = rows["radius_km"].astype(float).to_numpy() * 1000.0
    else:
        radius_m = rows["accepted_radius_cells"].astype(float).to_numpy() * 10_000.0
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(radius_m) & (radius_m > 0)
    displacement = np.full(len(rows), np.nan)
    displacement[valid] = np.hypot(x[valid], y[valid]) / radius_m[valid]
    direction = np.degrees(np.arctan2(y, x))
    direction[displacement <= 1e-8] = np.nan
    n_valid = int(np.sum(np.isfinite(displacement)))
    enough = n_valid >= min_layers
    mean_turn, max_turn = _turns(direction, np.isfinite(displacement) & (displacement > 1e-8))
    return {
        "date": pd.Timestamp(first["date"]).strftime("%Y-%m-%d"),
        "hua_object_id": str(first["hua_object_id"]),
        "polarity": str(first.get("polarity", "")),
        "n_valid_layers": n_valid,
        "max_depth_m": float(rows.loc[valid, "depth_m"].max()) if np.any(valid) else np.nan,
        "S_rms": float(np.sqrt(np.nanmean(displacement ** 2))) if enough else np.nan,
        "S_max": float(np.nanmax(displacement)) if enough else np.nan,
        "monotonic_ratio": _monotonic_ratio(displacement) if enough else np.nan,
        "mean_turn_deg": mean_turn if enough else np.nan,
        "max_turn_deg": max_turn if enough else np.nan,
    }


def classify_object_days(
    centers: pd.DataFrame,
    *,
    min_layers: int = 6,
    upright_quantile: float = 0.20,
    upright_fallback: float = 0.12,
    coherent_monotonic_ratio: float = 0.72,
    coherent_mean_turn_deg: float = 35.0,
    complex_max_turn_deg: float = 100.0,
    complex_monotonic_ratio: float = 0.55,
    tracks: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = pd.DataFrame([
        object_day_metrics(rows, min_layers=min_layers)
        for _, rows in centers.groupby("hua_object_id", sort=True)
        if not rows.empty
    ])
    eligible = metrics["n_valid_layers"] >= min_layers
    threshold = (
        float(metrics.loc[eligible, "S_rms"].quantile(upright_quantile))
        if int(eligible.sum()) >= 10 else float(upright_fallback)
    )
    classes: list[str] = []
    for row in metrics.itertuples():
        if row.n_valid_layers < min_layers:
            classes.append("unknown")
            continue
        if row.S_rms <= threshold:
            classes.append("upright")
            continue
        coherent = row.monotonic_ratio >= coherent_monotonic_ratio and row.mean_turn_deg <= coherent_mean_turn_deg
        complex_ = row.max_turn_deg >= complex_max_turn_deg or row.monotonic_ratio <= complex_monotonic_ratio
        classes.append("mixed" if coherent and complex_ else "coherent" if coherent else "complex" if complex_ else "transitional")
    metrics["shape_class"] = classes
    metrics["upright_s_rms_threshold"] = threshold
    if tracks is not None and not tracks.empty:
        metrics = metrics.merge(tracks[["hua_object_id", "track_id"]].drop_duplicates(), on="hua_object_id", how="left")
        summary = (
            metrics.dropna(subset=["track_id"]).groupby(["track_id", "shape_class"], as_index=False)
            .size().sort_values(["track_id", "size"], ascending=[True, False])
        )
    else:
        summary = pd.DataFrame(columns=["track_id", "shape_class", "size"])
    return metrics, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--centers", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--tracks", type=Path)
    parser.add_argument("--min-layers", type=int, default=6)
    args = parser.parse_args()
    centers = pd.read_csv(args.centers, low_memory=False)
    tracks = pd.read_csv(args.tracks, low_memory=False) if args.tracks and args.tracks.exists() else None
    metrics, summary = classify_object_days(centers, min_layers=args.min_layers, tracks=tracks)
    args.output_root.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.output_root / "object_day_shape.csv", index=False)
    summary.to_csv(args.output_root / "track_shape_summary.csv", index=False)


if __name__ == "__main__":
    main()
