from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    args = parse_args()
    start = datetime.strptime(args.start, "%Y-%m-%d")
    end = datetime.strptime(args.end, "%Y-%m-%d")
    days = [(start + timedelta(days=i)).strftime("%Y-%m-%d") for i in range((end - start).days + 1)]
    all_daily = {day: read_day(args.source_root, day) for day in days if day_root(args.source_root, day).exists()}
    args.output_root.mkdir(parents=True, exist_ok=True)
    summaries = []
    for day in days:
        centers, structures = all_daily.get(day, (pd.DataFrame(), pd.DataFrame()))
        if centers.empty:
            continue
        out_centers = centers.copy()
        apply_persistence(out_centers, day, all_daily, args)
        ymd = day.replace("-", "")
        out_dir = args.output_root / "daily_runs" / ymd
        out_dir.mkdir(parents=True, exist_ok=True)
        out_centers.to_csv(out_dir / "centers_hua_style.csv", index=False)
        structures.to_csv(out_dir / "structures_hua_style.csv", index=False)
        summary = summarize(out_centers, day, args)
        summaries.append(summary)
        (out_dir / f"persistence_summary_{ymd}.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(summaries).to_csv(args.output_root / "surface_persistence_summary.csv", index=False)
    (args.output_root / "surface_persistence_summary.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output_root": str(args.output_root), "days": len(summaries)}, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Add short-term surface persistence labels to daily OFES eddy tables.")
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--start", default="1991-01-01")
    parser.add_argument("--end", default="1991-01-19")
    parser.add_argument("--lookahead-days", type=int, default=5)
    parser.add_argument("--distance-factor", type=float, default=1.5)
    parser.add_argument("--radius-ratio-max", type=float, default=2.0)
    return parser.parse_args()


def day_root(root: Path, day: str) -> Path:
    return root / "daily_runs" / day.replace("-", "")


def read_day(root: Path, day: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    dr = day_root(root, day)
    centers = read_table(dr / "centers_hua_style")
    structures = read_table(dr / "structures_hua_style")
    normalize_dates(centers)
    normalize_dates(structures)
    return centers, structures


def read_table(path_without_suffix: Path) -> pd.DataFrame:
    csv_path = path_without_suffix.with_suffix(".csv")
    parquet_path = path_without_suffix.with_suffix(".parquet")
    if csv_path.exists():
        return pd.read_csv(csv_path)
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    return pd.DataFrame()


def normalize_dates(df: pd.DataFrame) -> None:
    if not df.empty and "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")


def surface_pass(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out = df.copy()
    if "depth_index" in out.columns:
        out = out[out["depth_index"].astype(int).eq(0)].copy()
    if "hua_pass" in out.columns:
        out = out[out["hua_pass"].fillna(False).astype(bool)].copy()
    return out


def apply_persistence(out: pd.DataFrame, day: str, all_daily: dict[str, tuple[pd.DataFrame, pd.DataFrame]], args: argparse.Namespace) -> None:
    out["persistence_days_available"] = 0
    out["persistence_match_count"] = 0
    out["persistence_class"] = "not_evaluated"
    surf = surface_pass(out)
    if surf.empty:
        return
    day0 = datetime.strptime(day, "%Y-%m-%d")
    future_days = []
    future_surfaces = []
    for step in range(1, int(args.lookahead_days) + 1):
        fday = (day0 + timedelta(days=step)).strftime("%Y-%m-%d")
        if fday in all_daily:
            future_days.append(fday)
            future_surfaces.append(surface_pass(all_daily[fday][0]))
    if not future_surfaces:
        out.loc[surf.index, "persistence_class"] = "not_evaluated_no_future_days"
        return
    future = pd.concat(future_surfaces, ignore_index=True)
    out.loc[surf.index, "persistence_days_available"] = len(future_days)
    if future.empty:
        out.loc[surf.index, "persistence_class"] = "transient"
        return
    f_lon = np.asarray([row_lon(row) for _, row in future.iterrows()], dtype="f8")
    f_lat = np.asarray([row_lat(row) for _, row in future.iterrows()], dtype="f8")
    f_radius = np.asarray([row_radius(row) for _, row in future.iterrows()], dtype="f8")
    f_polarity = future["polarity"].astype(str).to_numpy() if "polarity" in future.columns else np.full(len(future), "", dtype=object)
    for idx, row in surf.iterrows():
        lon0, lat0, r0 = row_lon(row), row_lat(row), row_radius(row)
        if not all(np.isfinite(v) for v in [lon0, lat0, r0]) or r0 <= 0:
            out.at[idx, "persistence_class"] = "transient"
            continue
        same = f_polarity == str(row.get("polarity", ""))
        finite = np.isfinite(f_lon) & np.isfinite(f_lat) & np.isfinite(f_radius) & (f_radius > 0)
        cand = same & finite
        if not np.any(cand):
            out.at[idx, "persistence_class"] = "transient"
            continue
        dlon = ((f_lon[cand] - lon0 + 180.0) % 360.0) - 180.0
        dlat = f_lat[cand] - lat0
        lat_mid = 0.5 * (f_lat[cand] + lat0)
        dist = np.hypot(dlon * 111.2 * np.maximum(np.cos(np.deg2rad(lat_mid)), 0.2), dlat * 111.2)
        ratio = np.maximum(f_radius[cand] / r0, r0 / f_radius[cand])
        match = (dist <= float(args.distance_factor) * np.maximum(r0, f_radius[cand])) & (ratio <= float(args.radius_ratio_max))
        count = int(np.count_nonzero(match))
        out.at[idx, "persistence_match_count"] = count
        if count >= 2:
            out.at[idx, "persistence_class"] = "persistent_like"
        elif count == 1:
            out.at[idx, "persistence_class"] = "short_track"
        else:
            out.at[idx, "persistence_class"] = "transient"


def row_lon(row: pd.Series) -> float:
    for name in ["center_lon_refined", "center_lon", "ssh_contour_center_lon", "seed_lon"]:
        if name in row.index:
            val = numeric(row.get(name))
            if np.isfinite(val):
                return val % 360.0
    return np.nan


def row_lat(row: pd.Series) -> float:
    for name in ["center_lat_refined", "center_lat", "ssh_contour_center_lat", "seed_lat"]:
        if name in row.index:
            val = numeric(row.get(name))
            if np.isfinite(val):
                return val
    return np.nan


def row_radius(row: pd.Series) -> float:
    for name in ["radius_km", "streamline_radius_cells", "accepted_radius_cells", "radius_cells"]:
        if name in row.index:
            val = numeric(row.get(name))
            if np.isfinite(val) and val > 0:
                if name.endswith("cells"):
                    lat = row_lat(row)
                    return val * 111.2 * 0.1 if np.isfinite(lat) else val * 11.12
                return val
    return np.nan


def numeric(value: object) -> float:
    try:
        return float(value)
    except Exception:
        return np.nan


def summarize(out: pd.DataFrame, day: str, args: argparse.Namespace) -> dict[str, object]:
    surf = surface_pass(out)
    classes = surf["persistence_class"].astype(str) if "persistence_class" in surf.columns else pd.Series(dtype=str)
    return {
        "day": day,
        "surface_pass": int(len(surf)),
        "transient": int(classes.eq("transient").sum()),
        "short_track": int(classes.eq("short_track").sum()),
        "persistent_like": int(classes.eq("persistent_like").sum()),
        "lookahead_days": int(args.lookahead_days),
    }


if __name__ == "__main__":
    main()
