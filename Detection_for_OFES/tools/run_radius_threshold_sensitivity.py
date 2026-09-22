"""Sweep minimum equivalent radius without rerunning OFES detection."""

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
from Detection_for_OFES.tools.run_area_cell_threshold_sensitivity import frame, passed, seed_cross
from Detection_for_OFES.tools.run_ofes_raw_ssh_five_treatment_comparison import (
    contour_indices,
    draw_seed_layer,
    grid,
    local_equal_distance_extent_km,
    read_surface,
    scalar_image,
    subset,
)


# Values are ceil(base_radius_km * (1 + increase / 100)) for each environment.
LEVELS = (
    (0, 25, 18, (92, 153, 255)),
    (40, 35, 26, (245, 132, 31)),
    (80, 45, 33, (225, 175, 24)),
    (120, 55, 40, (204, 60, 155)),
    (160, 65, 47, (164, 92, 205)),
    (200, 75, 54, (210, 48, 48)),
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


def closed_contours(rows: pd.DataFrame) -> pd.DataFrame:
    closed = rows.get("ssh_contour_closed", pd.Series(False, index=rows.index)).fillna(False).astype(bool)
    return rows.loc[closed].copy()


def lost_by_radius(baseline: pd.DataFrame, variant: pd.DataFrame) -> pd.DataFrame:
    base = baseline.set_index("hua_object_id")
    current = variant.set_index("hua_object_id")
    ids = base.index.intersection(current.index)
    base_pass = base.loc[ids, "qc_pass"].fillna(False).astype(bool)
    current_pass = current.loc[ids, "qc_pass"].fillna(False).astype(bool)
    reason = current.loc[ids, "qc_reject_reason"].fillna("").astype(str)
    only_radius = reason.eq("radius_too_small")
    return current.loc[ids[base_pass & ~current_pass & only_radius]].reset_index()


def draw_context_contours(
    draw: ImageDraw.ImageDraw,
    rows: pd.DataFrame,
    lon_axis: np.ndarray,
    lat_axis: np.ndarray,
    box: tuple[int, int, int, int],
    bbox: tuple[float, float, float, float],
) -> None:
    """Give every discovered closed contour a deliberately quiet background trace."""
    lo0, lo1, la0, la1 = bbox
    x0, y0, x1, y1 = box
    for _, row in rows.iterrows():
        ii, jj = contour_indices(row)
        valid = (ii >= 0) & (ii < lon_axis.size) & (jj >= 0) & (jj < lat_axis.size)
        ii, jj = ii[valid], jj[valid]
        if ii.size < 3:
            continue
        points = [(px(float(lon_axis[i]), lo0, lo1, x0, x1), py(float(lat_axis[j]), la0, la1, y0, y1)) for i, j in zip(ii, jj)]
        draw.line(points + [points[0]], fill=(150, 158, 170), width=1, joint="curve")


def draw_map(
    ssh: np.ndarray,
    lon: np.ndarray,
    lat: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    all_closed: pd.DataFrame,
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
        map_width, map_height, left, top, canvas_height, spacing = 1780, 750, 80, 110, 1040, 4.0
    else:
        width_km, height_km, _ = local_equal_distance_extent_km(bbox)
        map_width = 1840
        map_height = max(230, int(round(map_width * height_km / max(width_km, 1.0))))
        left, top, canvas_height, spacing = 110, 140, map_height + 390, 0.8
    right, bottom = left + map_width, top + map_height
    image = Image.new("RGB", (2200, canvas_height), "white")
    image.paste(scalar_image(cropped, max(limit, 1.0e-6), map_width, map_height), (left, top))
    draw = ImageDraw.Draw(image)
    grid(draw, (left, top, right, bottom), bbox)
    vector_info = draw_vectors(draw, lon=lon, lat=lat, u=u, v=v, box=(left, top, right, bottom), bbox=bbox, spacing_deg=spacing, arrow_pixels=12.0 if global_map else 18.0)
    draw_context_contours(draw, subset(all_closed, bbox), lon, lat, (left, top, right, bottom), bbox)
    draw_seed_layer(draw, subset(variant_rows, bbox), lon, lat, (left, top, right, bottom), bbox)
    lost_region = subset(lost, bbox)
    seed_cross(draw, lost_region, box=(left, top, right, bottom), bbox=bbox, color=cross_color)
    draw.text((left, 28), title, fill=(18, 24, 38), font=font(29))
    draw.text((left, 74), f"{threshold_text}; retained={len(subset(variant_rows, bbox))}; new radius-threshold removals={len(lost_region)}; colored X intersects the original SSH seed", fill=(65, 75, 90), font=font(16))
    footer_y = bottom + 28
    draw.line((left, footer_y + 8, left + 20, footer_y + 28), fill=cross_color, width=3)
    draw.line((left, footer_y + 28, left + 20, footer_y + 8), fill=cross_color, width=3)
    draw.text((left + 32, footer_y + 6), "X: saved closed SSH contour passed the 0% QC baseline, then failed this treatment specifically because equivalent radius is below its raised threshold.", fill=(45, 55, 68), font=font(15))
    draw.text((left, footer_y + 40), "Faint gray lines are all discovered closed SSH contours. No X means retained here, or never a 0%-baseline QC pass.", fill=(65, 75, 90), font=font(14))
    colorbar_x0, colorbar_x1 = right + 45, right + 75
    values = np.linspace(limit, -limit, map_height, dtype="f4")[:, None]
    colorbar = Image.fromarray(np.asarray(diverging_ramp(values, -limit, limit), dtype="u1"), "RGB").resize((colorbar_x1 - colorbar_x0, map_height), Image.Resampling.NEAREST)
    image.paste(colorbar, (colorbar_x0, top))
    draw.rectangle((colorbar_x0, top, colorbar_x1, bottom), outline=(35, 45, 60), width=1)
    draw.text((colorbar_x0 - 2, top - 28), "SSH (cm)", fill=(35, 45, 60), font=font(16))
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)
    image.save(output.with_suffix(".pdf"), "PDF", resolution=180.0)
    return {"path": str(output), "retained": int(len(subset(variant_rows, bbox))), "new_radius_removals": int(len(lost_region)), "vector_stride_cells": int(vector_info["stride_cells"])}


def regional_rows(baseline: pd.DataFrame, variant: pd.DataFrame, lost: pd.DataFrame, pct: int, regular: int, ocean: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for name, spec in REGIONS.items():
        bbox = spec["bbox"]
        for environment, selector in (("all", lambda x: x), ("open_ocean", lambda x: x[x.apply(is_open_ocean, axis=1)]), ("non_open_ocean", lambda x: x[~x.apply(is_open_ocean, axis=1)])):
            base_part = selector(subset(baseline, bbox))
            current_part = selector(subset(variant, bbox))
            lost_part = selector(subset(lost, bbox))
            reactivated = current_part[~current_part["hua_object_id"].isin(base_part["hua_object_id"])]
            rows.append({
                "radius_increase_percent": pct,
                "regular_min_radius_km": regular,
                "open_ocean_min_radius_km": ocean,
                "region": name,
                "environment": environment,
                "baseline_final_count": int(len(base_part)),
                "variant_final_count": int(len(current_part)),
                "new_radius_threshold_removals": int(len(lost_part)),
                "overlap_reactivated_count": int(len(reactivated)),
                "radius_median_km": float(pd.to_numeric(current_part.get("radius_km"), errors="coerce").median()) if not current_part.empty else np.nan,
                "pixel_area_median_cells": float(pd.to_numeric(current_part.get("pixel_area_cells"), errors="coerce").median()) if not current_part.empty else np.nan,
                "shape_error_median_percent": float(pd.to_numeric(current_part.get("shape_error_percent"), errors="coerce").median()) if not current_part.empty else np.nan,
                "compactness_median": float(pd.to_numeric(current_part.get("compactness"), errors="coerce").median()) if not current_part.empty else np.nan,
            })
    return rows


def main() -> None:
    args = parse_args()
    current = date.fromisoformat(args.day)
    ymd = current.strftime("%Y%m%d")
    args.output_root.mkdir(parents=True, exist_ok=True)
    for pct, regular, ocean, _ in LEVELS:
        target = args.output_root / f"radius_plus_{pct:02d}"
        command = [
            sys.executable, "-m", "Detection_for_OFES.tools.postprocess_ofes_eddy_qc",
            "--source-root", str(args.raw_source_root), "--filter-root", str(args.filter_root),
            "--output-root", str(target), "--day", args.day, "--skip-persistence",
            "--accept-ssh-primary-without-streamline", "--min-radius-km", str(regular),
            "--open-ocean-min-radius-km", str(ocean),
        ]
        subprocess.run(command, check=True)

    baseline = frame(args.output_root / "radius_plus_00")
    baseline_pass = passed(baseline)
    raw = frame(args.raw_source_root)
    all_closed = closed_contours(raw)
    lon, lat, ssh, u, v = read_surface(args.filter_root, current)
    all_stats: list[dict[str, object]] = []
    products: dict[str, object] = {}
    for pct, regular, ocean, color in LEVELS:
        root = args.output_root / f"radius_plus_{pct:02d}"
        variant = frame(root)
        variant_pass = passed(variant)
        lost = lost_by_radius(baseline, variant)
        all_stats.extend(regional_rows(baseline_pass, variant_pass, lost, pct, regular, ocean))
        products[str(pct)] = {}
        for key, region in REGIONS.items():
            output = root / "figures" / "radius_threshold_sensitivity" / f"{key}_{ymd}.png"
            products[str(pct)][key] = draw_map(
                ssh, lon, lat, u, v, all_closed, baseline_pass, variant_pass, lost,
                bbox=region["bbox"],
                title=f"500-km high-pass | no tile cap | radius +{pct}% | {region['label']} | {args.day}",
                threshold_text=f"minimum equivalent radius: regular={regular} km; open ocean={ocean} km",
                cross_color=color,
                output=output,
            )
    pd.DataFrame(all_stats).to_csv(args.output_root / "radius_threshold_sensitivity_summary.csv", index=False)
    manifest = {
        "day": args.day,
        "raw_source_root": str(args.raw_source_root),
        "filter_root": str(args.filter_root),
        "baseline_definition": "0% radius threshold, same raw candidate table and same non-radius QC",
        "fixed_thresholds": {"regular_min_area_cells": 16, "open_ocean_min_area_cells": 9},
        "persistence": "not applied",
        "levels": [{"increase_percent": pct, "regular_min_radius_km": regular, "open_ocean_min_radius_km": ocean, "cross_color_rgb": color} for pct, regular, ocean, color in LEVELS],
        "cross_rule": "Only baseline QC-pass objects newly rejected with exactly radius_too_small; cross centre is seed_lon/seed_lat from seed_i/j.",
        "products": products,
    }
    (args.output_root / "radius_threshold_sensitivity_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
