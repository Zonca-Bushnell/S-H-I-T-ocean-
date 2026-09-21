"""Compare final Jan1 regional counts for legacy-A and DUACS-like E catalogs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


OLD_ROOT = Path(
    r"E:\DATA\01_Eddy_correspond\02_OFES"
    r"\origin_unified_ssh_primary_target_all_open_ocean_recovery_surface_jan01_jan19"
)
E_ROOT = Path(
    r"E:\DATA\01_Eddy_correspond\02_OFES"
    r"\origin_unified_duacs_like_mss_ssh_primary_target_all_open_ocean_surface_jan01_jan19"
)
REGIONS: dict[str, tuple[float, float, float, float]] = {
    "global": (0.0, 360.0, -76.0, 76.0),
    "north_pacific": (190.0, 245.0, 25.0, 55.0),
    "south_pacific": (190.0, 280.0, -50.0, -30.0),
    "north_atlantic": (320.0, 330.0, 25.0, 45.0),
    "south_atlantic": (335.0, 355.0, -45.0, -20.0),
    "acc": (0.0, 360.0, -62.0, -40.0),
    "kuroshio_reference": (120.0, 145.0, 20.0, 35.0),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-root", type=Path, default=OLD_ROOT)
    parser.add_argument("--e-root", type=Path, default=E_ROOT)
    parser.add_argument("--day", default="1991-01-01")
    parser.add_argument("--output-root", type=Path, default=None)
    return parser.parse_args()


def read_final(root: Path, day: str) -> pd.DataFrame:
    path = root / "final_catalog" / "daily_runs" / day.replace("-", "") / "centers_hua_style.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    if "depth_index" in frame.columns:
        frame = frame[pd.to_numeric(frame["depth_index"], errors="coerce").eq(0)].copy()
    if "hua_pass" in frame.columns:
        frame = frame[frame["hua_pass"].fillna(False).astype(bool)].copy()
    if "persistence_class" in frame.columns:
        frame = frame[~frame["persistence_class"].astype(str).eq("transient")].copy()
    return frame


def coordinate(frame: pd.DataFrame, primary: str, fallback: str) -> pd.Series:
    source = frame[primary] if primary in frame.columns else frame[fallback]
    return pd.to_numeric(source, errors="coerce")


def numeric(frame: pd.DataFrame, candidates: tuple[str, ...]) -> pd.Series:
    for name in candidates:
        if name in frame.columns:
            return pd.to_numeric(frame[name], errors="coerce")
    return pd.Series(np.nan, index=frame.index, dtype="f8")


def subset(frame: pd.DataFrame, bounds: tuple[float, float, float, float]) -> pd.DataFrame:
    lon = coordinate(frame, "center_lon", "seed_lon") % 360.0
    lat = coordinate(frame, "center_lat", "seed_lat")
    west, east, south, north = bounds
    return frame[lon.between(west, east) & lat.between(south, north)].copy()


def summarize(label: str, frame: pd.DataFrame) -> list[dict[str, object]]:
    rows = []
    for region, bounds in REGIONS.items():
        part = subset(frame, bounds)
        amplitude = numeric(part, ("ssh_contour_amplitude_cm", "regional_threshold_cm", "ssh_amplitude_cm"))
        radius = numeric(part, ("radius_km", "ssh_contour_radius_km", "accepted_radius_km"))
        polarity = part["polarity"].astype(str) if "polarity" in part.columns else pd.Series("", index=part.index)
        rows.append({
            "path": label,
            "region": region,
            "final_count": int(len(part)),
            "cyclonic_count": int(polarity.eq("cyclonic").sum()),
            "anticyclonic_count": int(polarity.eq("anticyclonic").sum()),
            "amplitude_median_cm": float(amplitude.median()) if amplitude.notna().any() else float("nan"),
            "radius_median_km": float(radius.median()) if radius.notna().any() else float("nan"),
        })
    return rows


def main() -> None:
    args = parse_args()
    output_root = args.output_root or (args.e_root / "comparison")
    output_root.mkdir(parents=True, exist_ok=True)
    old = read_final(args.old_root, args.day)
    e_path = read_final(args.e_root, args.day)
    rows = summarize("A_old_jan01_jan19", old) + summarize("E_duacs_like_annual_mss", e_path)
    table = pd.DataFrame(rows)
    old_counts = table[table["path"].eq("A_old_jan01_jan19")].set_index("region")["final_count"]
    e_counts = table[table["path"].eq("E_duacs_like_annual_mss")].set_index("region")["final_count"]
    table["delta_e_minus_a"] = table["region"].map((e_counts - old_counts).to_dict())
    stem = f"ofes_e_vs_old_a_regional_summary_{args.day.replace('-', '')}"
    table.to_csv(output_root / f"{stem}.csv", index=False)
    payload = {
        "day": args.day,
        "comparison_policy": "Final surface objects only: hua_pass, shape/overlap QC, non-transient persistence.",
        "old_root": str(args.old_root),
        "e_root": str(args.e_root),
        "regions": {name: list(bounds) for name, bounds in REGIONS.items()},
        "rows": table.replace({np.nan: None}).to_dict(orient="records"),
    }
    (output_root / f"{stem}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(table.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
