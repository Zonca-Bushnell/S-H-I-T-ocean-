from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


REGIONS: dict[str, tuple[float, float, float, float]] = {
    "global": (0.0, 360.0, -76.0, 76.0),
    "northeast_pacific_subtropical": (210.0, 250.0, 20.0, 45.0),
    "northwest_pacific_subtropical": (145.0, 180.0, 20.0, 45.0),
    "south_pacific_subtropical": (190.0, 260.0, -45.0, -20.0),
    "south_indian_subtropical": (50.0, 110.0, -45.0, -20.0),
    "kuroshio_reference": (120.0, 145.0, 20.0, 35.0),
    "gulf_stream_reference": (280.0, 315.0, 25.0, 45.0),
    "acc_reference": (0.0, 360.0, -62.0, -40.0),
}


def main() -> None:
    args = parse_args()
    lon, lat = read_lon_lat(args.filter_root / f"global_phy_{args.day.replace('-', '')}.nc")
    rows: list[dict[str, object]] = []
    for case in args.case:
        label, root = parse_case(case)
        centers = read_centers(root, args.day)
        surface_all = surface_depth(centers)
        surface = accepted_surface(surface_all)
        ensure_shape_metrics(surface, lon, lat)
        for region, bbox in REGIONS.items():
            part = subset_bbox(surface, bbox)
            part_all = subset_bbox(surface_all, bbox)
            rows.append(summarize_case_region(label, root, region, part, part_all))
    out = pd.DataFrame(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / f"open_ocean_recovery_summary_{args.day.replace('-', '')}.csv"
    json_path = args.output_dir / f"open_ocean_recovery_summary_{args.day.replace('-', '')}.json"
    out.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"csv": str(csv_path), "json": str(json_path), "rows": rows}, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize OFES open-ocean recovery across SSH-primary amplitude-threshold smoke runs.")
    parser.add_argument("--filter-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--day", default="1991-01-01")
    parser.add_argument("--case", action="append", required=True, help="Label=ResultRoot. Repeat for amp0p4/amp1/amp2 cases.")
    return parser.parse_args()


def parse_case(text: str) -> tuple[str, Path]:
    if "=" not in text:
        raise ValueError(f"--case must be Label=Path, got {text!r}")
    label, path = text.split("=", 1)
    return label.strip(), Path(path.strip())


def read_lon_lat(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as ds:
        return ds["longitude"][:].astype("f8"), ds["latitude"][:].astype("f8")


def read_centers(root: Path, day: str) -> pd.DataFrame:
    ymd = day.replace("-", "")
    candidates = [root / "daily_runs" / ymd, root, root / "hua_b3_start2_detection"]
    for table_root in candidates:
        for suffix in [".parquet", ".csv"]:
            path = table_root / f"centers_hua_style{suffix}"
            if path.exists():
                df = pd.read_parquet(path) if suffix == ".parquet" else pd.read_csv(path)
                if "date" in df.columns:
                    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
                    df = df[df["date"].eq(day)].copy()
                return df
    raise FileNotFoundError(f"No centers_hua_style table for {day} under {root}")


def accepted_surface(df: pd.DataFrame) -> pd.DataFrame:
    out = surface_depth(df)
    if "hua_pass" in out.columns:
        out = out[out["hua_pass"].fillna(False).astype(bool)].copy()
    if "boundary_source" in out.columns:
        source = out["boundary_source"].astype(str)
        out = out[~source.str.contains("rejected|duplicate|shape", case=False, na=False)].copy()
    return out


def surface_depth(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "depth_index" in out.columns:
        out = out[out["depth_index"].astype(int).eq(0)].copy()
    return out


def ensure_shape_metrics(df: pd.DataFrame, lon: np.ndarray, lat: np.ndarray) -> None:
    if "shape_error_percent" not in df.columns:
        df["shape_error_percent"] = np.nan
    if "compactness" not in df.columns:
        df["compactness"] = np.nan
    if "boundary_point_count" not in df.columns:
        df["boundary_point_count"] = 0
    for idx, row in df.iterrows():
        if np.isfinite(to_float(row.get("shape_error_percent", np.nan))):
            continue
        metrics = contour_metrics(row, lon, lat)
        for key, value in metrics.items():
            df.at[idx, key] = value


def subset_bbox(df: pd.DataFrame, bbox: tuple[float, float, float, float]) -> pd.DataFrame:
    lon_col = first_col(df, ["center_lon_refined", "center_lon", "ssh_contour_center_lon", "seed_lon"])
    lat_col = first_col(df, ["center_lat_refined", "center_lat", "ssh_contour_center_lat", "seed_lat"])
    if lon_col is None or lat_col is None:
        return df.iloc[0:0].copy()
    lon_min, lon_max, lat_min, lat_max = bbox
    lon_values = pd.to_numeric(df[lon_col], errors="coerce")
    lat_values = pd.to_numeric(df[lat_col], errors="coerce")
    return df[lon_values.between(lon_min, lon_max) & lat_values.between(lat_min, lat_max)].copy()


def summarize_case_region(label: str, root: Path, region: str, part: pd.DataFrame, part_all: pd.DataFrame | None = None) -> dict[str, object]:
    if part_all is None:
        part_all = part
    amp = pd.to_numeric(part.get("ssh_contour_amplitude_cm", pd.Series(dtype=float)), errors="coerce")
    radius = radius_series(part)
    shape = pd.to_numeric(part.get("shape_error_percent", pd.Series(dtype=float)), errors="coerce")
    same_extrema = pd.to_numeric(part.get("ssh_contour_same_extrema_count", pd.Series(dtype=float)), errors="coerce")
    jet_flag = part.get("jet_meander_flag", pd.Series(dtype=bool)).fillna(False).astype(bool) if "jet_meander_flag" in part.columns else pd.Series(dtype=bool)
    weak_core = part.get("dynamical_core_class", pd.Series(dtype=str)).astype(str).str.contains("no_streamline|weak", case=False, na=False) if "dynamical_core_class" in part.columns else pd.Series(dtype=bool)
    qc_class = part_all.get("qc_class", pd.Series(dtype=str)).astype(str) if "qc_class" in part_all.columns else pd.Series(dtype=str)
    persistence = part.get("persistence_class", pd.Series(dtype=str)).astype(str) if "persistence_class" in part.columns else pd.Series(dtype=str)
    return {
        "case": label,
        "result_root": str(root),
        "region": region,
        "objects": int(len(part)),
        "cyclonic": int(part.get("polarity", pd.Series(dtype=str)).astype(str).eq("cyclonic").sum()) if not part.empty else 0,
        "anticyclonic": int(part.get("polarity", pd.Series(dtype=str)).astype(str).eq("anticyclonic").sum()) if not part.empty else 0,
        "radius_km_p25": q(radius, 25),
        "radius_km_median": q(radius, 50),
        "radius_km_p75": q(radius, 75),
        "amplitude_cm_median": q(amp, 50),
        "amplitude_cm_p25": q(amp, 25),
        "shape_error_percent_median": q(shape, 50),
        "single_extremum_fraction": float(np.nanmean(same_extrema <= 1.0)) if len(same_extrema) else np.nan,
        "jet_flag_count": int(jet_flag.sum()) if len(jet_flag) else 0,
        "jet_flag_fraction": float(jet_flag.mean()) if len(jet_flag) else np.nan,
        "weak_or_no_core_count": int(weak_core.sum()) if len(weak_core) else 0,
        "weak_or_no_core_fraction": float(weak_core.mean()) if len(weak_core) else np.nan,
        "overlap_duplicate_count": int(qc_class.eq("overlap_duplicate").sum()) if len(qc_class) else 0,
        "transient_fraction": float(persistence.eq("transient").mean()) if len(persistence) else np.nan,
    }


def radius_series(df: pd.DataFrame) -> pd.Series:
    if "radius_km" in df.columns:
        values = pd.to_numeric(df["radius_km"], errors="coerce")
    elif "ssh_contour_radius_cells" in df.columns:
        values = pd.to_numeric(df["ssh_contour_radius_cells"], errors="coerce") * 11.12
    elif "accepted_radius_cells" in df.columns:
        values = pd.to_numeric(df["accepted_radius_cells"], errors="coerce") * 11.12
    else:
        values = pd.Series(np.nan, index=df.index)
    return values


def contour_metrics(row: pd.Series, lon: np.ndarray, lat: np.ndarray) -> dict[str, float | int]:
    ii = parse_index_list(row.get("ssh_contour_boundary_i", ""))
    jj = parse_index_list(row.get("ssh_contour_boundary_j", ""))
    n = min(ii.size, jj.size)
    if n < 3:
        return {"shape_error_percent": np.nan, "compactness": np.nan, "boundary_point_count": int(n)}
    ii = ii[:n]
    jj = jj[:n]
    valid = (ii >= 0) & (ii < len(lon)) & (jj >= 0) & (jj < len(lat))
    ii = ii[valid]
    jj = jj[valid]
    if ii.size < 3:
        return {"shape_error_percent": np.nan, "compactness": np.nan, "boundary_point_count": int(ii.size)}
    lon0 = to_float(row.get("center_lon_refined", row.get("ssh_contour_center_lon", np.nan)))
    lat0 = to_float(row.get("center_lat_refined", row.get("ssh_contour_center_lat", np.nan)))
    if not np.isfinite(lon0):
        lon0 = float(np.nanmean(lon[ii]))
    if not np.isfinite(lat0):
        lat0 = float(np.nanmean(lat[jj]))
    x = (lon[ii].astype("f8") - lon0) * 111.2 * max(math.cos(math.radians(float(lat0))), 0.2)
    y = (lat[jj].astype("f8") - lat0) * 111.2
    order = np.argsort(np.arctan2(y, x))
    x = x[order]
    y = y[order]
    if x[0] != x[-1] or y[0] != y[-1]:
        x = np.r_[x, x[0]]
        y = np.r_[y, y[0]]
    dx = np.diff(x)
    dy = np.diff(y)
    perimeter = float(np.sum(np.hypot(dx, dy)))
    area = 0.5 * abs(float(np.sum(x[:-1] * y[1:] - x[1:] * y[:-1])))
    rr = np.hypot(x[:-1] - np.nanmean(x[:-1]), y[:-1] - np.nanmean(y[:-1]))
    mean_r = float(np.nanmean(rr)) if rr.size else np.nan
    shape = float(np.nanstd(rr) / mean_r * 100.0) if np.isfinite(mean_r) and mean_r > 0 else np.nan
    compactness = float(4.0 * math.pi * area / max(perimeter * perimeter, 1.0e-12)) if perimeter > 0 else np.nan
    return {"shape_error_percent": shape, "compactness": compactness, "boundary_point_count": int(ii.size)}


def parse_index_list(value: object) -> np.ndarray:
    text = "" if pd.isna(value) else str(value)
    out: list[int] = []
    for part in text.replace(",", ";").split(";"):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(int(float(part)))
        except ValueError:
            continue
    return np.asarray(out, dtype=int)


def first_col(df: pd.DataFrame, names: list[str]) -> str | None:
    for name in names:
        if name in df.columns:
            return name
    return None


def q(values: pd.Series, percentile: float) -> float:
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype="float64")
    arr = arr[np.isfinite(arr)]
    return float(np.nanpercentile(arr, percentile)) if arr.size else np.nan


def to_float(value: object) -> float:
    try:
        return float(value)
    except Exception:
        return np.nan


if __name__ == "__main__":
    main()
