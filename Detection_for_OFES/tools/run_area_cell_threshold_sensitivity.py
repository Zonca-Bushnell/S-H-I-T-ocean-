"""Sweep QC minimum contour area without rerunning OFES detection."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from Detection_for_OFES.tools.plot_latest_ssh_vector_overview import diverging_ramp, font, px, py
from Detection_for_OFES.tools.plot_ofes_raw_ssh_regions_pillow import draw_vectors
from Detection_for_OFES.tools.postprocess_ofes_eddy_qc import is_open_ocean
from Detection_for_OFES.tools.run_ofes_raw_ssh_five_treatment_comparison import (
    draw_seed_layer,
    grid,
    local_equal_distance_extent_km,
    read_surface,
    scalar_image,
    subset,
)


LEVELS = (
    (0, 16, 9, (92, 153, 255)),
    (20, 20, 11, (245, 132, 31)),
    (40, 23, 13, (225, 175, 24)),
    (60, 26, 15, (204, 60, 155)),
    (80, 29, 17, (210, 48, 48)),
)
REGIONS = {
    "global": {"label": "global", "bbox": (0.0, 360.0, -76.0, 76.0)},
    "kuroshio_extension": {"label": "Kuroshio Extension", "bbox": (140.0, 180.0, 28.0, 40.0)},
    "south_pacific_stcc": {"label": "South Pacific STCC", "bbox": (165.0, 230.0, -29.0, -21.0)},
    "taiwan_hawaii_stcc_hlcc": {"label": "Taiwan-Hawaii STCC-HLCC corridor", "bbox": (122.0, 203.0, 18.0, 27.0)},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-source-root", type=Path, required=True)
    parser.add_argument("--filter-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--day", default="1991-01-01")
    return parser.parse_args()


def frame(path: Path) -> pd.DataFrame:
    table = path / "daily_runs" / "19910101" / "centers_hua_style.csv"
    out = pd.read_csv(table)
    if "depth_index" in out:
        out = out[pd.to_numeric(out["depth_index"], errors="coerce").fillna(-1).eq(0)].copy()
    return out


def passed(rows: pd.DataFrame) -> pd.DataFrame:
    return rows[rows["qc_pass"].fillna(False).astype(bool)].copy()


def seed_cross(draw: ImageDraw.ImageDraw, rows: pd.DataFrame, *, box: tuple[int, int, int, int], bbox: tuple[float, float, float, float], color: tuple[int, int, int]) -> None:
    lo0, lo1, la0, la1 = bbox
    x0, y0, x1, y1 = box
    for _, row in rows.iterrows():
        lon = float(row.get("seed_lon", np.nan))
        lat = float(row.get("seed_lat", np.nan))
        if not (np.isfinite(lon) and np.isfinite(lat)):
            continue
        cx, cy = px(lon, lo0, lo1, x0, x1), py(lat, la0, la1, y0, y1)
        # The diagonals deliberately intersect at the original SSH extremum.
        draw.line((cx - 8, cy - 8, cx + 8, cy + 8), fill=(255, 255, 255), width=5)
        draw.line((cx - 8, cy + 8, cx + 8, cy - 8), fill=(255, 255, 255), width=5)
        draw.line((cx - 8, cy - 8, cx + 8, cy + 8), fill=color, width=3)
        draw.line((cx - 8, cy + 8, cx + 8, cy - 8), fill=color, width=3)


def variants_lost_by_area(baseline: pd.DataFrame, variant: pd.DataFrame) -> pd.DataFrame:
    base = baseline.set_index("hua_object_id")
    current = variant.set_index("hua_object_id")
    ids = base.index.intersection(current.index)
    base_pass = base.loc[ids, "qc_pass"].fillna(False).astype(bool)
    variant_pass = current.loc[ids, "qc_pass"].fillna(False).astype(bool)
    reason = current.loc[ids, "qc_reject_reason"].fillna("").astype(str)
    return current.loc[ids[base_pass & ~variant_pass & reason.str.contains("area_too_small", regex=False)]].reset_index()


def draw_map(
    ssh: np.ndarray,
    lon: np.ndarray,
    lat: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    baseline_rows: pd.DataFrame,
    variant_rows: pd.DataFrame,
    lost: pd.DataFrame,
    *,
    bbox: tuple[float, float, float, float],
    title: str,
    threshold_text: str,
    cross_color: tuple[int, int, int],
    output: Path,
) -> dict[str, object]:
    lo0, lo1, la0, la1 = bbox
    ix = np.where((lon >= lo0) & (lon <= lo1))[0]
    iy = np.where((lat >= la0) & (lat <= la1))[0]
    cropped = ssh[np.ix_(iy, ix)]
    limit = float(np.nanpercentile(np.abs(cropped), 99.0)) if np.isfinite(cropped).any() else 1.0
    global_map = (lo1 - lo0) > 100.0
    if global_map:
        map_width, map_height = 1780, 750
        left, top = 80, 110
        canvas_height = 1040
        spacing = 4.0
    else:
        width_km, height_km, _ = local_equal_distance_extent_km(bbox)
        map_width = 1840
        map_height = max(230, int(round(map_width * height_km / max(width_km, 1.0))))
        left, top = 110, 140
        canvas_height = map_height + 390
        spacing = 0.8
    right, bottom = left + map_width, top + map_height
    image = Image.new("RGB", (2200, canvas_height), "white")
    image.paste(scalar_image(cropped, max(limit, 1.0e-6), map_width, map_height), (left, top))
    draw = ImageDraw.Draw(image)
    grid(draw, (left, top, right, bottom), bbox)
    vector_info = draw_vectors(draw, lon=lon, lat=lat, u=u, v=v, box=(left, top, right, bottom), bbox=bbox, spacing_deg=spacing, arrow_pixels=12.0 if global_map else 18.0)
    # Baseline contours establish that an object was found; variant contours
    # are redrawn to keep the retained treatment visually prominent.
    draw_seed_layer(draw, subset(baseline_rows, bbox), lon, lat, (left, top, right, bottom), bbox)
    draw_seed_layer(draw, subset(variant_rows, bbox), lon, lat, (left, top, right, bottom), bbox)
    lost_region = subset(lost, bbox)
    seed_cross(draw, lost_region, box=(left, top, right, bottom), bbox=bbox, color=cross_color)
    draw.text((left, 28), title, fill=(18, 24, 38), font=font(29))
    draw.text((left, 74), f"{threshold_text}; retained={len(subset(variant_rows, bbox))}; new area-threshold removals={len(lost_region)}; colored X intersects the original SSH seed", fill=(65, 75, 90), font=font(16))
    footer_y = bottom + 28
    draw.line((left, footer_y + 8, left + 20, footer_y + 28), fill=cross_color, width=3)
    draw.line((left, footer_y + 28, left + 20, footer_y + 8), fill=cross_color, width=3)
    draw.text((left + 32, footer_y + 6), "X: saved closed SSH contour passed the 0% QC baseline, then failed this treatment specifically because pixel area is below its raised threshold.", fill=(45, 55, 68), font=font(15))
    draw.text((left, footer_y + 40), "No X means either the contour remains retained, or it was never a 0%-baseline QC pass; unclosed/undiscovered candidates are not represented as a removal.", fill=(65, 75, 90), font=font(14))
    colorbar_x0, colorbar_x1 = right + 45, right + 75
    values = np.linspace(limit, -limit, map_height, dtype="f4")[:, None]
    colorbar = Image.fromarray(np.asarray(diverging_ramp(values, -limit, limit), dtype="u1"), "RGB").resize((colorbar_x1 - colorbar_x0, map_height), Image.Resampling.NEAREST)
    image.paste(colorbar, (colorbar_x0, top))
    draw.rectangle((colorbar_x0, top, colorbar_x1, bottom), outline=(35, 45, 60), width=1)
    draw.text((colorbar_x0 - 2, top - 28), "SSH (cm)", fill=(35, 45, 60), font=font(16))
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)
    image.save(output.with_suffix(".pdf"), "PDF", resolution=180.0)
    return {"path": str(output), "retained": int(len(subset(variant_rows, bbox))), "new_area_removals": int(len(lost_region)), "vector_stride_cells": int(vector_info["stride_cells"])}


def regional_rows(baseline: pd.DataFrame, variant: pd.DataFrame, lost: pd.DataFrame, pct: int, regular: int, ocean: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for name, spec in REGIONS.items():
        bbox = spec["bbox"]
        for environment, selector in (("all", lambda x: x), ("open_ocean", lambda x: x[x.apply(is_open_ocean, axis=1)]), ("non_open_ocean", lambda x: x[~x.apply(is_open_ocean, axis=1)])):
            b = selector(subset(baseline, bbox))
            v = selector(subset(variant, bbox))
            x = selector(subset(lost, bbox))
            reactivated = v[~v["hua_object_id"].isin(b["hua_object_id"])]
            rows.append({
                "area_increase_percent": pct,
                "regular_min_area_cells": regular,
                "open_ocean_min_area_cells": ocean,
                "region": name,
                "environment": environment,
                "baseline_final_count": int(len(b)),
                "variant_final_count": int(len(v)),
                "new_area_threshold_removals": int(len(x)),
                "overlap_reactivated_count": int(len(reactivated)),
                "radius_median_km": float(pd.to_numeric(v.get("radius_km"), errors="coerce").median()) if not v.empty else np.nan,
                "pixel_area_median_cells": float(pd.to_numeric(v.get("pixel_area_cells"), errors="coerce").median()) if not v.empty else np.nan,
                "shape_error_median_percent": float(pd.to_numeric(v.get("shape_error_percent"), errors="coerce").median()) if not v.empty else np.nan,
                "compactness_median": float(pd.to_numeric(v.get("compactness"), errors="coerce").median()) if not v.empty else np.nan,
            })
    return rows


def main() -> None:
    args = parse_args()
    current = date.fromisoformat(args.day)
    ymd = current.strftime("%Y%m%d")
    args.output_root.mkdir(parents=True, exist_ok=True)
    for pct, regular, ocean, _ in LEVELS:
        target = args.output_root / f"area_plus_{pct:02d}"
        command = [
            sys.executable, "-m", "Detection_for_OFES.tools.postprocess_ofes_eddy_qc",
            "--source-root", str(args.raw_source_root), "--filter-root", str(args.filter_root),
            "--output-root", str(target), "--day", args.day, "--skip-persistence",
            "--accept-ssh-primary-without-streamline", "--min-area-cells", str(regular),
            "--open-ocean-min-area-cells", str(ocean),
        ]
        subprocess.run(command, check=True)
    baseline_root = args.output_root / "area_plus_00"
    baseline = frame(baseline_root)
    baseline_pass = passed(baseline)
    lon, lat, ssh, u, v = read_surface(args.filter_root, current)
    all_stats: list[dict[str, object]] = []
    products: dict[str, object] = {}
    for pct, regular, ocean, color in LEVELS:
        root = args.output_root / f"area_plus_{pct:02d}"
        variant = frame(root)
        variant_pass = passed(variant)
        lost = variants_lost_by_area(baseline, variant)
        all_stats.extend(regional_rows(baseline_pass, variant_pass, lost, pct, regular, ocean))
        products[str(pct)] = {}
        for key, region in REGIONS.items():
            output = root / "figures" / "area_threshold_sensitivity" / f"{key}_{ymd}.png"
            products[str(pct)][key] = draw_map(
                ssh, lon, lat, u, v, baseline_pass, variant_pass, lost,
                bbox=region["bbox"],
                title=f"500-km high-pass | no tile cap | area +{pct}% | {region['label']} | {args.day}",
                threshold_text=f"minimum contour area: regular={regular} cells; open ocean={ocean} cells",
                cross_color=color,
                output=output,
            )
    stats = pd.DataFrame(all_stats)
    stats.to_csv(args.output_root / "area_cell_threshold_sensitivity_summary.csv", index=False)
    manifest = {
        "day": args.day,
        "raw_source_root": str(args.raw_source_root),
        "filter_root": str(args.filter_root),
        "baseline_definition": "0% area threshold, same raw candidate table and same non-area QC",
        "persistence": "not applied",
        "levels": [{"increase_percent": pct, "regular_min_area_cells": regular, "open_ocean_min_area_cells": ocean, "cross_color_rgb": color} for pct, regular, ocean, color in LEVELS],
        "cross_rule": "Only baseline QC-pass objects newly rejected with area_too_small in that treatment; cross centre is seed_lon/seed_lat from seed_i/j.",
        "products": products,
    }
    (args.output_root / "area_cell_threshold_sensitivity_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
