from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from Detection_for_OFES.tools.plot_latest_ssh_vector_overview import (
    draw_overview,
    read_detection_tables,
    read_surface_fields,
)


def main() -> None:
    args = parse_args()
    out_dir = args.output_root / "figures" / "velocity_streamline_four_overview"
    out_dir.mkdir(parents=True, exist_ok=True)
    lon, lat, ssh, u, v, attrs = read_surface_fields(args.filter_root, args.day)
    suffix = "_no_jet_flag" if args.exclude_jet_flagged else ""
    cases = [
        ("01_old_velocity_streamline", args.old_streamline_root, False, "velocity_streamline old input"),
        ("02_spatial25_200_velocity_streamline", args.spatial_streamline_root, False, "velocity_streamline + spatial25_200"),
        ("03_spatial25_200_effective", args.effective_root, False, "velocity_streamline + spatial25_200 + effective contour"),
        ("04_spatial25_200_effective_no_transient", args.persistence_root, True, "velocity_streamline + spatial25_200 + effective contour + no transient"),
    ]
    products = []
    summary_rows = []
    for stem, root, exclude_transient, title in cases:
        centers, structures, table_root = read_detection_tables(root, args.day)
        png = out_dir / f"{stem}_global_{args.day.replace('-', '')}{suffix}.png"
        product = draw_overview(
            lon,
            lat,
            ssh,
            u,
            v,
            centers,
            structures,
            attrs,
            args.day,
            png,
            bbox=(0.0, 360.0, -76.0, 76.0),
            title_region=title,
            vector_step=int(args.global_vector_step),
            catalog_layer="all",
            exclude_jet_flagged=bool(args.exclude_jet_flagged),
            exclude_transient=exclude_transient,
        )
        product["case"] = stem
        product["table_root"] = str(table_root)
        product["exclude_transient"] = bool(exclude_transient)
        products.append(product)
        summary_rows.extend(region_counts(stem, centers, structures, exclude_transient, bool(args.exclude_jet_flagged)))
    summary = {
        "day": args.day,
        "filter_root": str(args.filter_root),
        "background": "spatial25_200km SSH anomaly for all four panels",
        "exclude_jet_flagged": bool(args.exclude_jet_flagged),
        "products": products,
    }
    (out_dir / f"velocity_streamline_four_overview_manifest_{args.day.replace('-', '')}{suffix}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    pd.DataFrame(summary_rows).to_csv(out_dir / f"velocity_streamline_four_overview_counts_{args.day.replace('-', '')}{suffix}.csv", index=False)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot four global OFES velocity-streamline comparison overview figures.")
    parser.add_argument("--day", default="1991-01-01")
    parser.add_argument("--filter-root", type=Path, required=True)
    parser.add_argument("--old-streamline-root", type=Path, required=True)
    parser.add_argument("--spatial-streamline-root", type=Path, required=True)
    parser.add_argument("--effective-root", type=Path, required=True)
    parser.add_argument("--persistence-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--global-vector-step", type=int, default=120)
    parser.add_argument("--exclude-jet-flagged", action="store_true")
    return parser.parse_args()


def region_counts(case: str, centers: pd.DataFrame, structures: pd.DataFrame, exclude_transient: bool, exclude_jet_flagged: bool) -> list[dict[str, object]]:
    surface = centers.copy()
    if "depth_index" in surface.columns:
        surface = surface[surface["depth_index"].astype(int).eq(0)].copy()
    if "hua_pass" in surface.columns:
        surface = surface[surface["hua_pass"].fillna(False).astype(bool)].copy()
    if exclude_transient and "persistence_class" in surface.columns:
        surface = surface[~surface["persistence_class"].astype(str).str.lower().eq("transient")].copy()
    if exclude_jet_flagged and "jet_meander_flag" in surface.columns:
        surface = surface[~surface["jet_meander_flag"].fillna(False).astype(bool)].copy()
    lon_col = first_col(surface, ["center_lon_refined", "center_lon", "ssh_contour_center_lon", "seed_lon"])
    lat_col = first_col(surface, ["center_lat_refined", "center_lat", "ssh_contour_center_lat", "seed_lat"])
    if lon_col is None or lat_col is None:
        return []
    surface["plot_lon"] = pd.to_numeric(surface[lon_col], errors="coerce") % 360.0
    surface["plot_lat"] = pd.to_numeric(surface[lat_col], errors="coerce")
    regions = {
        "global": (0.0, 360.0, -76.0, 76.0),
        "NE_Pacific_subtropical": (190.0, 240.0, 20.0, 45.0),
        "NW_Pacific_subtropical": (145.0, 180.0, 20.0, 45.0),
        "North_Pacific_open_ocean": (145.0, 250.0, 20.0, 45.0),
        "ACC_reference": (0.0, 360.0, -62.0, -40.0),
    }
    rows = []
    for region, (lon0, lon1, lat0, lat1) in regions.items():
        sub = surface[surface["plot_lon"].between(lon0, lon1) & surface["plot_lat"].between(lat0, lat1)].copy()
        rows.append(
            {
                "case": case,
                "region": region,
                "objects": int(len(sub)),
                "cyclonic": int(sub["polarity"].astype(str).eq("cyclonic").sum()) if "polarity" in sub.columns else 0,
                "anticyclonic": int(sub["polarity"].astype(str).eq("anticyclonic").sum()) if "polarity" in sub.columns else 0,
                "transient_hidden": bool(exclude_transient),
                "jet_flag_hidden": bool(exclude_jet_flagged),
            }
        )
    return rows


def first_col(df: pd.DataFrame, names: list[str]) -> str | None:
    for name in names:
        if name in df.columns:
            return name
    return None


if __name__ == "__main__":
    main()
