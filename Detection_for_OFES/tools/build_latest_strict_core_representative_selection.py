"""Select six latest-profile strict-core representative diagnostics by polarity."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


REGIONS = {
    "kuroshio_extension": (140.0, 180.0, 28.0, 40.0),
    "south_pacific_stcc": (165.0, 230.0, -29.0, -21.0),
    "taiwan_hawaii_corridor": (122.0, 203.0, 18.0, 27.0),
}
EXPECTED_CYCLONIC_OBJECTS = {
    "kuroshio_extension": ["19910107_00013", "19910113_00110"],
    "south_pacific_stcc": ["19910115_07191", "19910114_02949"],
    "taiwan_hawaii_corridor": ["19910119_06134", "19910114_05438"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog-root", type=Path, required=True)
    parser.add_argument("--surface-qc-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--polarity", choices=("cyclonic", "anticyclonic"), default="cyclonic")
    return parser.parse_args()


def read_catalogs(root: Path) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for path in sorted(root.glob("*/vertical_continuation_object_catalog.csv")):
        part = pd.read_csv(path, low_memory=False)
        part["composite_date"] = f"{path.parent.name[:4]}-{path.parent.name[4:6]}-{path.parent.name[6:8]}"
        parts.append(part)
    if not parts:
        raise FileNotFoundError(f"No daily strict-core catalogs below {root}")
    return pd.concat(parts, ignore_index=True)


def attach_surface_radius(rows: pd.DataFrame, qc_root: Path) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for day, part in rows.groupby("composite_date", sort=True):
        token = str(day).replace("-", "")
        path = qc_root / "daily_runs" / token / "centers_hua_style.csv"
        surface = pd.read_csv(path, low_memory=False)
        columns = [name for name in ("hua_object_id", "radius_km", "center_lon_refined", "center_lat_refined") if name in surface]
        if "radius_km" not in columns:
            raise ValueError(f"Surface QC table has no radius_km: {path}")
        joined = part.merge(surface[columns], on="hua_object_id", how="left", validate="one_to_one")
        if joined["radius_km"].isna().any():
            raise ValueError(f"Missing surface radius after join for {day}")
        parts.append(joined)
    return pd.concat(parts, ignore_index=True)


def main() -> None:
    args = parse_args()
    rows = read_catalogs(args.catalog_root)
    rows = rows.loc[
        rows["section_bipolar_class"].astype(str).eq("section_bipolar_strict_core")
        & rows["polarity"].astype(str).str.lower().eq(args.polarity)
    ].copy()
    selected: list[pd.DataFrame] = []
    for region, (lon_min, lon_max, lat_min, lat_max) in REGIONS.items():
        subset = rows.loc[
            rows["surface_lon"].between(lon_min, lon_max)
            & rows["surface_lat"].between(lat_min, lat_max)
        ].sort_values(["max_depth_m", "section_bipolarity_min_fraction", "hua_object_id"], ascending=[False, False, True])
        if len(subset) < 2:
            raise RuntimeError(f"{region} has fewer than two latest strict-core cyclones")
        take = subset.head(2).copy()
        take["region"] = region
        selected.append(take)
    selected_rows = attach_surface_radius(pd.concat(selected, ignore_index=True), args.surface_qc_root)
    selected_rows = selected_rows.sort_values(["region", "max_depth_m"], ascending=[True, False]).reset_index(drop=True)
    observed = {
        region: selected_rows.loc[selected_rows["region"].eq(region), "hua_object_id"].tolist()
        for region in REGIONS
    }
    if args.polarity == "cyclonic" and (
        {region: set(ids) for region, ids in observed.items()}
        != {region: set(ids) for region, ids in EXPECTED_CYCLONIC_OBJECTS.items()}
    ):
        raise RuntimeError(f"Latest-profile cyclonic representative selection changed: {observed}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    output = args.output_root / "selected_representative_object_days.csv"
    selected_rows.to_csv(output, index=False, encoding="utf-8-sig")
    manifest = {
        "source_catalog_root": str(args.catalog_root),
        "source_surface_qc_root": str(args.surface_qc_root),
        "vertical_profile": "tangent45_fraction35_then_near_closed_relaxed_v1",
        "polarity": args.polarity,
        "selection": "latest strict-core rows of the requested polarity, top two max_depth_m per fixed region; ties by section-bipolar score then hua_object_id",
        "regions": {name: {"lon_east": bounds[:2], "lat": bounds[2:]} for name, bounds in REGIONS.items()},
        "object_days": selected_rows[["region", "composite_date", "hua_object_id", "max_depth_m", "section_bipolarity_min_fraction"]].to_dict(orient="records"),
    }
    (args.output_root / "selection_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"selection": str(output), "object_days": len(selected_rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
