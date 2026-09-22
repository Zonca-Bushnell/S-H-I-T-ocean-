from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


REGIONS = {
    "global": (0.0, 360.0, -76.0, 76.0),
    "South Pacific STCC": (165.0, 230.0, -29.0, -21.0),
    "Kuroshio Extension": (140.0, 180.0, 28.0, 40.0),
    "Taiwan-Hawaii corridor": (122.0, 203.0, 18.0, 27.0),
    "North Pacific open ocean": (190.0, 245.0, 25.0, 55.0),
    "South Pacific open ocean": (190.0, 280.0, -50.0, -30.0),
    "North Atlantic interior": (320.0, 330.0, 25.0, 45.0),
    "South Atlantic interior": (335.0, 355.0, -45.0, -20.0),
    "ACC": (0.0, 360.0, -62.0, -40.0),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize geometry/overlap QC before and after by ocean region.")
    parser.add_argument("--qc-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--day", default="1991-01-01")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ymd = str(args.day).replace("-", "")
    source = args.qc_root / "daily_runs" / ymd / "centers_hua_style.csv"
    df = pd.read_csv(source)
    if "date" in df.columns:
        df = df[pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d").eq(str(args.day))].copy()
    if "depth_index" in df.columns:
        df = df[pd.to_numeric(df["depth_index"], errors="coerce").fillna(-1).eq(0)].copy()
    df["_lon"] = pd.to_numeric(df[best_col(df, ["center_lon_refined", "center_lon", "seed_lon"])], errors="coerce") % 360.0
    df["_lat"] = pd.to_numeric(df[best_col(df, ["center_lat_refined", "center_lat", "seed_lat"])], errors="coerce")
    raw = df[df.get("raw_hua_pass", pd.Series(False, index=df.index)).fillna(False).astype(bool)].copy()
    final = df[df.get("qc_pass", pd.Series(False, index=df.index)).fillna(False).astype(bool)].copy()
    rows = []
    for name, bbox in REGIONS.items():
        for stage, subset in [("before_geometry_qc", raw), ("after_geometry_overlap_qc", final)]:
            region = select_bbox(subset, bbox)
            rows.append(metrics(name, bbox, stage, region))
        region_all = select_bbox(df, bbox)
        rows.append(
            {
                "region": name,
                "lon_min": bbox[0], "lon_max": bbox[1], "lat_min": bbox[2], "lat_max": bbox[3],
                "stage": "qc_rejections",
                "object_count": int(len(region_all)),
                "raw_primary_count": int(region_all.get("raw_hua_pass", pd.Series(False, index=region_all.index)).fillna(False).astype(bool).sum()),
                "streamline_gate_removed_promoted_count": int(region_all.get("streamline_gate_removed", pd.Series(False, index=region_all.index)).fillna(False).astype(bool).sum()),
                "streamline_gate_removed_final_count": int((region_all.get("streamline_gate_removed", pd.Series(False, index=region_all.index)).fillna(False).astype(bool) & region_all.get("qc_pass", pd.Series(False, index=region_all.index)).fillna(False).astype(bool)).sum()),
                "shape_rejected_count": int(region_all.get("qc_class", pd.Series("", index=region_all.index)).astype(str).eq("shape_rejected").sum()),
                "overlap_duplicate_count": int(region_all.get("qc_class", pd.Series("", index=region_all.index)).astype(str).eq("overlap_duplicate").sum()),
            }
        )
    result = pd.DataFrame(rows)
    args.output_root.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_root / "geometry_qc_before_after_regional_summary.csv"
    json_path = args.output_root / "geometry_qc_before_after_regional_summary.json"
    result.to_csv(csv_path, index=False)
    payload = {
        "day": str(args.day),
        "source": str(source),
        "comparison": "500-km high-pass, no tile cap; geometry and same-day overlap QC only; persistence not applied",
        "before_definition": "surface rows with raw_hua_pass=True",
        "after_definition": "surface rows with qc_pass=True after geometry and overlap QC",
        "regions": REGIONS,
        "rows": result.replace({np.nan: None}).to_dict(orient="records"),
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(result.to_string(index=False))
    print(csv_path)


def best_col(df: pd.DataFrame, candidates: list[str]) -> str:
    for name in candidates:
        if name in df.columns:
            return name
    raise KeyError(f"No coordinate column among {candidates}")


def select_bbox(df: pd.DataFrame, bbox: tuple[float, float, float, float]) -> pd.DataFrame:
    lon0, lon1, lat0, lat1 = bbox
    if lon0 <= lon1:
        lon_mask = df["_lon"].between(lon0, lon1)
    else:
        lon_mask = (df["_lon"] >= lon0) | (df["_lon"] <= lon1)
    return df[lon_mask & df["_lat"].between(lat0, lat1)].copy()


def q(series: pd.Series, value: float) -> float:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype="f8")
    values = values[np.isfinite(values)]
    return float(np.nanquantile(values, value)) if values.size else np.nan


def metrics(name: str, bbox: tuple[float, float, float, float], stage: str, df: pd.DataFrame) -> dict[str, object]:
    return {
        "region": name,
        "lon_min": bbox[0], "lon_max": bbox[1], "lat_min": bbox[2], "lat_max": bbox[3],
        "stage": stage,
        "object_count": int(len(df)),
        "cyclonic_count": int(df.get("polarity", pd.Series("", index=df.index)).astype(str).eq("cyclonic").sum()),
        "anticyclonic_count": int(df.get("polarity", pd.Series("", index=df.index)).astype(str).eq("anticyclonic").sum()),
        "radius_median_km": q(df.get("radius_km", pd.Series(dtype=float)), 0.50),
        "radius_p10_km": q(df.get("radius_km", pd.Series(dtype=float)), 0.10),
        "radius_p90_km": q(df.get("radius_km", pd.Series(dtype=float)), 0.90),
        "shape_error_median_percent": q(df.get("shape_error_percent", pd.Series(dtype=float)), 0.50),
        "shape_error_p90_percent": q(df.get("shape_error_percent", pd.Series(dtype=float)), 0.90),
        "compactness_median": q(df.get("compactness", pd.Series(dtype=float)), 0.50),
        "compactness_p10": q(df.get("compactness", pd.Series(dtype=float)), 0.10),
        "amplitude_median_cm": q(df.get("ssh_contour_amplitude_cm", pd.Series(dtype=float)), 0.50),
        "streamline_gate_removed_count": int(df.get("streamline_gate_removed", pd.Series(False, index=df.index)).fillna(False).astype(bool).sum()),
    }


if __name__ == "__main__":
    main()
