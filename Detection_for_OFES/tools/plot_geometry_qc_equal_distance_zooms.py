"""Render geometry-QC retained SSH contours in locally equal-distance coordinates."""

from __future__ import annotations

import argparse
import json
import math
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from Detection_for_OFES.tools.run_ofes_raw_ssh_five_treatment_comparison import (
    REGIONS,
    draw_dashed_ellipse,
    draw_equal_distance_candidate_map,
    draw_seed_layer,
    grid,
    local_equal_distance_extent_km,
    save_to_long_output,
    scalar_image,
    read_surface,
    subset,
)
from Detection_for_OFES.tools.plot_latest_ssh_vector_overview import diverging_ramp, font, px, py
from Detection_for_OFES.tools.plot_ofes_raw_ssh_regions_pillow import draw_vectors


FAILURE_STYLES = (
    ("streamline_gate_removed", "G", "SSH contour admitted after streamline gate removal", (33, 145, 104)),
    ("area_too_small", "S", "area below threshold", (245, 132, 31)),
    ("radius_too_small", "R", "equivalent radius below threshold", (225, 175, 24)),
    ("compactness_low", "C", "compactness below threshold", (30, 165, 190)),
    ("shape_error_high", "E", "shape error above threshold", (204, 60, 155)),
    ("overlap_duplicate", "O", "same-polarity overlap duplicate", (127, 92, 191)),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--filter-root", type=Path, required=True)
    parser.add_argument("--qc-root", type=Path, required=True)
    parser.add_argument("--day", default="1991-01-01")
    return parser.parse_args()


def callout_row(rows: pd.DataFrame, reason: str, chosen_ids: set[str]) -> pd.Series | None:
    if reason == "streamline_gate_removed":
        rejected = rows[rows.get("streamline_gate_removed", pd.Series(False, index=rows.index)).fillna(False).astype(bool)].copy()
        if rejected.empty:
            return None
        rejected["_radius"] = pd.to_numeric(rejected.get("ssh_contour_radius_cells", np.nan), errors="coerce")
        rejected = rejected.sort_values("_radius", ascending=False, na_position="last")
        for _, row in rejected.iterrows():
            object_id = str(row.get("hua_object_id", row.name))
            if object_id not in chosen_ids:
                chosen_ids.add(object_id)
                return row
        return None
    rejected = rows[rows.get("qc_class", pd.Series("", index=rows.index)).astype(str).eq("overlap_duplicate" if reason == "overlap_duplicate" else "shape_rejected")]
    reasons = rejected.get("qc_reject_reason", pd.Series("", index=rejected.index)).fillna("").astype(str)
    rejected = rejected[reasons.str.contains(reason, regex=False)].copy()
    if rejected.empty:
        return None
    rejected["_radius"] = pd.to_numeric(rejected.get("radius_km", np.nan), errors="coerce")
    rejected = rejected.sort_values("_radius", ascending=False, na_position="last")
    for _, row in rejected.iterrows():
        object_id = str(row.get("hua_object_id", row.name))
        if object_id not in chosen_ids:
            chosen_ids.add(object_id)
            return row
    return None


def draw_geometry_qc_map(
    ssh: np.ndarray,
    lon: np.ndarray,
    lat: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    accepted: pd.DataFrame,
    rejected: pd.DataFrame,
    all_closed_candidates: pd.DataFrame,
    *,
    bbox: tuple[float, float, float, float],
    title: str,
    output: Path,
) -> dict[str, object]:
    lo0, lo1, la0, la1 = bbox
    ix = np.where((lon >= lo0) & (lon <= lo1))[0]
    iy = np.where((lat >= la0) & (lat <= la1))[0]
    cropped = ssh[np.ix_(iy, ix)]
    limit = float(np.nanpercentile(np.abs(cropped), 99.0)) if np.isfinite(cropped).any() else 1.0
    width_km, height_km, lat_ref = local_equal_distance_extent_km(bbox)
    map_width = 1840
    map_height = max(1, int(round(map_width * height_km / max(width_km, 1.0))))
    left, top = 110, 140
    right, bottom = left + map_width, top + map_height
    image = Image.new("RGB", (2200, bottom + 250), "white")
    image.paste(scalar_image(cropped, max(limit, 1.0e-6), map_width, map_height), (left, top))
    draw = ImageDraw.Draw(image)
    grid(draw, (left, top, right, bottom), bbox)
    vector_info = draw_vectors(draw, lon=lon, lat=lat, u=u, v=v, box=(left, top, right, bottom), bbox=bbox, spacing_deg=0.8, arrow_pixels=18.0)
    accepted_region = subset(accepted, bbox)
    rejected_region = subset(rejected, bbox)
    closed_region = subset(all_closed_candidates, bbox)
    # Render every closed SSH contour first.  Retained objects are redrawn
    # immediately afterwards, making their boundaries visibly stronger.
    draw_seed_layer(draw, closed_region, lon, lat, (left, top, right, bottom), bbox)
    draw_seed_layer(draw, accepted_region, lon, lat, (left, top, right, bottom), bbox)
    # Mark every saved closed-contour seed, including those subsequently
    # rejected by geometry QC.  The marker is drawn from seed_i/j-derived
    # seed_lon/lat, rather than from the contour centroid.
    accepted_ids = set(accepted_region.get("hua_object_id", pd.Series(dtype=str)).astype(str))
    marked_rejected = 0
    for _, row in closed_region.iterrows():
        seed_lon = float(row.get("seed_lon", np.nan))
        seed_lat = float(row.get("seed_lat", np.nan))
        if not (np.isfinite(seed_lon) and np.isfinite(seed_lat)):
            continue
        cx = px(seed_lon, lo0, lo1, left, right)
        cy = py(seed_lat, la0, la1, top, bottom)
        color = (40, 105, 230) if str(row.get("polarity", "")).lower() == "cyclonic" else (225, 45, 45)
        object_id = str(row.get("hua_object_id", ""))
        if object_id not in accepted_ids:
            marked_rejected += 1
        # White underlay remains legible over both SSH colors and vectors.
        draw.line((cx - 6, cy, cx + 6, cy), fill=(255, 255, 255), width=3)
        draw.line((cx, cy - 6, cx, cy + 6), fill=(255, 255, 255), width=3)
        draw.line((cx - 5, cy, cx + 5, cy), fill=color, width=2)
        draw.line((cx, cy - 5, cx, cy + 5), fill=color, width=2)
    # A small neutral dot marks the geometrical centre used for R_eq.  It can
    # legitimately differ from the extremum marker for an asymmetric contour.
    for _, row in accepted_region.iterrows():
        lon0 = float(row.get("ssh_contour_center_lon", np.nan))
        lat0 = float(row.get("ssh_contour_center_lat", np.nan))
        if np.isfinite(lon0) and np.isfinite(lat0):
            cx = px(lon0, lo0, lo1, left, right)
            cy = py(lat0, la0, la1, top, bottom)
            draw.ellipse((cx - 2, cy - 2, cx + 2, cy + 2), fill=(30, 34, 42), outline=(255, 255, 255), width=1)
    chosen_ids: set[str] = set()
    shown: list[dict[str, object]] = []
    for reason, code, label, color in FAILURE_STYLES:
        row = callout_row(closed_region if reason == "streamline_gate_removed" else rejected_region, reason, chosen_ids)
        if row is None:
            continue
        seed_lon = float(row.get("seed_lon", row.get("center_lon_refined", np.nan)))
        seed_lat = float(row.get("seed_lat", row.get("center_lat_refined", np.nan)))
        if not (np.isfinite(seed_lon) and np.isfinite(seed_lat)):
            continue
        radius_km = 90.0
        cx = px(seed_lon, lo0, lo1, left, right)
        cy = py(seed_lat, la0, la1, top, bottom)
        dlon = radius_km / (111.32 * max(abs(math.cos(math.radians(lat_ref))), 0.1))
        dlat = radius_km / 111.32
        rx = abs(px(seed_lon + dlon, lo0, lo1, left, right) - cx)
        ry = abs(py(seed_lat + dlat, la0, la1, top, bottom) - cy)
        draw.ellipse((cx - rx, cy - ry, cx + rx, cy + ry), outline=color, width=4)
        draw.line((cx - 7, cy, cx + 7, cy), fill=color, width=3)
        draw.line((cx, cy - 7, cx, cy + 7), fill=color, width=3)
        draw.rectangle((cx + 8, cy - 25, cx + 31, cy - 2), fill=(255, 255, 255), outline=color, width=2)
        draw.text((cx + 13, cy - 24), code, fill=color, font=font(18))
        shown.append({"code": code, "reason": reason, "label": label, "color": color, "seed_lon": seed_lon, "seed_lat": seed_lat, "hua_object_id": str(row.get("hua_object_id", ""))})

    cyclonic = int(accepted_region.get("polarity", pd.Series(dtype=str)).astype(str).eq("cyclonic").sum())
    anticyclonic = int(accepted_region.get("polarity", pd.Series(dtype=str)).astype(str).eq("anticyclonic").sum())
    draw.text((left, 30), title, fill=(18, 24, 38), font=font(29))
    promoted_closed = int(closed_region.get("streamline_gate_removed", pd.Series(False, index=closed_region.index)).fillna(False).astype(bool).sum())
    geometry_rejected_closed = int((closed_region.get("raw_hua_pass", pd.Series(False, index=closed_region.index)).fillna(False).astype(bool) & ~closed_region.get("qc_pass", pd.Series(False, index=closed_region.index)).fillna(False).astype(bool)).sum())
    draw.text((left, 76), f"Closed-contour seeds={len(closed_region)}; retained={len(accepted_region)}; streamline-gate removal promoted={promoted_closed}; geometry/overlap rejects={geometry_rejected_closed}", fill=(65, 75, 90), font=font(16))
    legend_y = bottom + 34
    for index, item in enumerate(shown):
        x = left + index * 345
        draw.ellipse((x, legend_y, x + 18, legend_y + 18), outline=item["color"], width=3)
        draw.text((x + 27, legend_y - 2), f"{item['code']}: {item['label']}", fill=(35, 42, 55), font=font(15))
        draw.text((x + 27, legend_y + 21), f"{item['seed_lon']:.1f}E, {item['seed_lat']:.1f}N", fill=(65, 75, 90), font=font(13))
    draw.text((left, bottom + 94), "Thin colored contours/dashed R_eq = every saved SSH closed contour; retained contours are redrawn bolder. Colored +/- = original grid-extremum seed; dark dot = contour geometric centre. G marks one SSH contour newly admitted after removing the streamline gate.", fill=(45, 55, 68), font=font(15))
    draw.text((left, bottom + 122), f"u/v sampling: 0.8 degrees; local equal-distance map at {lat_ref:.1f} degrees; callout radius: 90 km", fill=(65, 75, 90), font=font(15))
    colorbar_x0, colorbar_x1 = right + 42, right + 72
    values = np.linspace(limit, -limit, map_height, dtype="f4")[:, None]
    colorbar = Image.fromarray(np.asarray(diverging_ramp(values, -limit, limit), dtype="u1"), "RGB").resize((colorbar_x1 - colorbar_x0, map_height), Image.Resampling.NEAREST)
    image.paste(colorbar, (colorbar_x0, top))
    draw.rectangle((colorbar_x0, top, colorbar_x1, bottom), outline=(35, 45, 60), width=1)
    draw.text((colorbar_x0 - 2, top - 28), "SSH (cm)", fill=(35, 45, 60), font=font(16))
    save_to_long_output(image, output, image_format="PNG")
    save_to_long_output(image, output.with_suffix(".pdf"), image_format="PDF", resolution=180.0)
    return {"objects": int(len(accepted_region)), "closed_contour_seed_count": int(len(closed_region)), "streamline_gate_removed_closed_count": promoted_closed, "closed_contour_geometry_rejected_count": geometry_rejected_closed, "closed_contour_rejected_seed_count": int(marked_rejected), "rejected_in_region": int(len(rejected_region)), "cyclonic": cyclonic, "anticyclonic": anticyclonic, "representative_rejections": shown, "vector_stride_cells": int(vector_info["stride_cells"]), "projection": "local_equirectangular_equal_distance", "reference_latitude_degrees": lat_ref, "map_width_km": width_km, "map_height_km": height_km}


def draw_global_qc_map(
    ssh: np.ndarray,
    lon: np.ndarray,
    lat: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    accepted: pd.DataFrame,
    rejected: pd.DataFrame,
    all_closed_candidates: pd.DataFrame,
    *,
    title: str,
    output: Path,
) -> dict[str, object]:
    """Draw the global retained field with one labeled ring per QC outcome."""
    bbox = (0.0, 360.0, -76.0, 76.0)
    lo0, lo1, la0, la1 = bbox
    limit = float(np.nanpercentile(np.abs(ssh), 99.0)) if np.isfinite(ssh).any() else 1.0
    left, top, map_width, map_height = 80, 110, 1780, 750
    right, bottom = left + map_width, top + map_height
    image = Image.new("RGB", (2200, 1150), "white")
    image.paste(scalar_image(ssh, max(limit, 1.0e-6), map_width, map_height), (left, top))
    draw = ImageDraw.Draw(image)
    grid(draw, (left, top, right, bottom), bbox)
    vector_info = draw_vectors(
        draw, lon=lon, lat=lat, u=u, v=v,
        box=(left, top, right, bottom), bbox=bbox, spacing_deg=4.0, arrow_pixels=12.0,
    )
    # The global figure is a QC overview: retained contours provide context,
    # while colored rings identify one representative object for each outcome.
    draw_seed_layer(draw, accepted, lon, lat, (left, top, right, bottom), bbox)
    chosen_ids: set[str] = set()
    shown: list[dict[str, object]] = []
    for reason, code, label, color in FAILURE_STYLES:
        row = callout_row(all_closed_candidates if reason == "streamline_gate_removed" else rejected, reason, chosen_ids)
        if row is None:
            continue
        seed_lon = float(row.get("seed_lon", row.get("center_lon_refined", np.nan)))
        seed_lat = float(row.get("seed_lat", row.get("center_lat_refined", np.nan)))
        if not (np.isfinite(seed_lon) and np.isfinite(seed_lat)):
            continue
        cx = px(seed_lon, lo0, lo1, left, right)
        cy = py(seed_lat, la0, la1, top, bottom)
        # A geographic 180-km callout is intentionally used only as a label.
        # The regional products retain the equal-distance true-circle display.
        radius_km = 180.0
        dlon = radius_km / (111.32 * max(abs(math.cos(math.radians(seed_lat))), 0.1))
        dlat = radius_km / 111.32
        rx = abs(px(seed_lon + dlon, lo0, lo1, left, right) - cx)
        ry = abs(py(seed_lat + dlat, la0, la1, top, bottom) - cy)
        draw.ellipse((cx - rx, cy - ry, cx + rx, cy + ry), outline=color, width=4)
        draw.rectangle((cx + 8, cy - 25, cx + 31, cy - 2), fill=(255, 255, 255), outline=color, width=2)
        draw.text((cx + 13, cy - 24), code, fill=color, font=font(18))
        shown.append({"code": code, "reason": reason, "label": label, "color": color, "seed_lon": seed_lon, "seed_lat": seed_lat, "hua_object_id": str(row.get("hua_object_id", ""))})
    draw.text((left, 28), title, fill=(18, 24, 38), font=font(29))
    draw.text((left, 74), f"Retained after geometry + overlap QC={len(accepted)}; colored rings identify representative QC outcomes; no persistence.", fill=(65, 75, 90), font=font(16))
    legend_y = bottom + 30
    for index, item in enumerate(shown):
        row_index, col_index = divmod(index, 3)
        x = left + col_index * 570
        y = legend_y + row_index * 48
        draw.ellipse((x, y, x + 18, y + 18), outline=item["color"], width=3)
        draw.text((x + 27, y - 2), f"{item['code']}: {item['label']} ({item['seed_lon']:.1f}E, {item['seed_lat']:.1f}N)", fill=(35, 42, 55), font=font(15))
    draw.text((left, bottom + 142), "Global callout rings are geographic labels; true physical R_eq circles and equal-distance geometry are shown in the three regional diagnostics.", fill=(65, 75, 90), font=font(14))
    colorbar_x0, colorbar_x1 = right + 48, right + 78
    values = np.linspace(limit, -limit, map_height, dtype="f4")[:, None]
    colorbar = Image.fromarray(np.asarray(diverging_ramp(values, -limit, limit), dtype="u1"), "RGB").resize((colorbar_x1 - colorbar_x0, map_height), Image.Resampling.NEAREST)
    image.paste(colorbar, (colorbar_x0, top))
    draw.rectangle((colorbar_x0, top, colorbar_x1, bottom), outline=(35, 45, 60), width=1)
    draw.text((colorbar_x0 - 2, top - 28), "SSH (cm)", fill=(35, 45, 60), font=font(16))
    save_to_long_output(image, output, image_format="PNG")
    save_to_long_output(image, output.with_suffix(".pdf"), image_format="PDF", resolution=180.0)
    return {"path": str(output), "objects": int(len(accepted)), "representative_qc_callouts": shown, "vector_stride_cells": int(vector_info["stride_cells"])}


def main() -> None:
    args = parse_args()
    current = date.fromisoformat(str(args.day))
    ymd = current.strftime("%Y%m%d")
    table = args.qc_root / "daily_runs" / ymd / "centers_hua_style.csv"
    centers = pd.read_csv(table)
    if "depth_index" in centers.columns:
        centers = centers[pd.to_numeric(centers["depth_index"], errors="coerce").fillna(-1).eq(0)].copy()
    accepted = centers[centers.get("qc_pass", pd.Series(False, index=centers.index)).fillna(False).astype(bool)].copy()
    rejected = centers[centers.get("raw_hua_pass", pd.Series(False, index=centers.index)).fillna(False).astype(bool) & ~centers.get("qc_pass", pd.Series(False, index=centers.index)).fillna(False).astype(bool)].copy()
    all_closed_candidates = centers[centers.get("ssh_contour_closed", pd.Series(False, index=centers.index)).fillna(False).astype(bool)].copy()
    lon, lat, ssh, u, v = read_surface(args.filter_root, current)
    destination = args.qc_root / "figures" / "geometry_qc_equivalent_radius"
    products: dict[str, object] = {}
    products["global_qc_callouts"] = draw_global_qc_map(
        ssh, lon, lat, u, v, accepted, rejected, all_closed_candidates,
        title=("500-km high-pass | no tile cap | SSH-contour geometry + overlap QC | global | " f"{current.isoformat()}"),
        output=destination / f"global_qc_callouts_{ymd}.png",
    )
    for key in ("south_pacific_stcc", "kuroshio_extension", "taiwan_hawaii_stcc_hlcc"):
        region = REGIONS[key]
        target = destination / f"{key}_{ymd}.png"
        products[key] = {
            "path": str(target),
            **draw_geometry_qc_map(
                ssh,
                lon,
                lat,
                u,
                v,
                accepted,
                rejected,
                all_closed_candidates,
                bbox=region["bbox"],
                title=(
                    "500-km high-pass | no tile cap | geometry + overlap QC only | "
                    f"{region['label']} | {current.isoformat()}"
                ),
                output=target,
            ),
        }
    manifest = {
        "day": current.isoformat(),
        "source_table": str(table),
        "selection": "surface qc_pass=True; persistence intentionally not run",
        "overlay": "retained: solid=true saved SSH contour, dashed=equivalent radius R_eq=sqrt(A/pi), cross=seed; colored rings: representative geometry/overlap rejections",
        "projection": "local equirectangular equal-distance at each regional center latitude; fixed physical km aspect",
        "vector_spacing_degrees": 0.8,
        "products": products,
    }
    destination.mkdir(parents=True, exist_ok=True)
    manifest_path = destination / f"geometry_qc_equal_distance_manifest_{ymd}.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
