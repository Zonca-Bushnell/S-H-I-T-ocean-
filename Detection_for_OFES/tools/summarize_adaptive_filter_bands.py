from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


SPECIAL_BOXES: dict[str, tuple[float, float, float, float]] = {
    "acc": (0.0, 360.0, -62.0, -40.0),
    "kuroshio": (120.0, 145.0, 20.0, 35.0),
    "gulf_stream": (280.0, 315.0, 25.0, 45.0),
    "east_australian": (150.0, 162.0, -45.0, -25.0),
    "agulhas": (18.0, 40.0, -45.0, -25.0),
    "brazil_malvinas": (300.0, 330.0, -50.0, -20.0),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize accepted SSH-primary objects by latitude bands, excluding ACC and WBC boxes.")
    parser.add_argument("--day", default="1991-01-01")
    parser.add_argument("--case", action="append", required=True, help="Label=ResultRoot")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def parse_case(text: str) -> tuple[str, Path]:
    label, path = text.split("=", 1)
    return label.strip(), Path(path.strip())


def read_centers(root: Path, day: str) -> pd.DataFrame:
    ymd = day.replace("-", "")
    candidates = [root / "daily_runs" / ymd, root]
    for table_root in candidates:
        for suffix in [".parquet", ".csv"]:
            path = table_root / f"centers_hua_style{suffix}"
            if path.exists():
                df = pd.read_parquet(path) if suffix == ".parquet" else pd.read_csv(path)
                if "date" in df.columns:
                    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
                    df = df[df["date"].eq(day)].copy()
                return df
    raise FileNotFoundError(f"No centers table under {root}")


def accepted_surface(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "depth_index" in out.columns:
        out = out[out["depth_index"].astype(int).eq(0)].copy()
    if "hua_pass" in out.columns:
        out = out[out["hua_pass"].fillna(False).astype(bool)].copy()
    if "boundary_source" in out.columns:
        source = out["boundary_source"].astype(str)
        out = out[~source.str.contains("rejected|duplicate|shape", case=False, na=False)].copy()
    return out


def lon_lat_series(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    lon_col = first_existing(df, ["center_lon_refined", "center_lon", "ssh_contour_center_lon", "seed_lon"])
    lat_col = first_existing(df, ["center_lat_refined", "center_lat", "ssh_contour_center_lat", "seed_lat"])
    if lon_col is None or lat_col is None:
        raise ValueError("No lon/lat columns")
    return pd.to_numeric(df[lon_col], errors="coerce"), pd.to_numeric(df[lat_col], errors="coerce")


def first_existing(df: pd.DataFrame, names: list[str]) -> str | None:
    for name in names:
        if name in df.columns:
            return name
    return None


def in_boxes(lon: pd.Series, lat: pd.Series, boxes: list[tuple[float, float, float, float]]) -> pd.Series:
    mask = pd.Series(False, index=lon.index)
    for lon_min, lon_max, lat_min, lat_max in boxes:
        mask |= lon.between(lon_min, lon_max) & lat.between(lat_min, lat_max)
    return mask


def summarize_case(label: str, root: Path, df: pd.DataFrame) -> dict[str, object]:
    lon, lat = lon_lat_series(df)
    abs_lat = lat.abs()
    special = in_boxes(lon, lat, list(SPECIAL_BOXES.values()))
    non_special = df.loc[~special].copy()
    ns_lon, ns_lat = lon_lat_series(non_special)
    ns_abs = ns_lat.abs()

    rows = {
        "case": label,
        "result_root": str(root),
        "global": int(len(df)),
        "low_lat_abs20": int((abs_lat < 20.0).sum()),
        "subtropics_20_45_non_wbc": int((ns_abs.between(20.0, 45.0)).sum()),
        "midhigh_45_76_non_acc": int((ns_abs.between(45.0, 76.0)).sum()),
        "north_midhigh_45_76": int((ns_lat.between(45.0, 76.0)).sum()),
        "south_midhigh_45_76_non_acc": int((ns_lat.between(-76.0, -45.0)).sum()),
        "acc_reference": int(in_boxes(lon, lat, [SPECIAL_BOXES["acc"]]).sum()),
        "wbc_reference_non_acc": int(in_boxes(lon, lat, list(SPECIAL_BOXES.values())[1:]).sum()),
    }
    return rows


def main() -> None:
    args = parse_args()
    rows = []
    for case in args.case:
        label, root = parse_case(case)
        df = accepted_surface(read_centers(root, args.day))
        rows.append(summarize_case(label, root, df))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
