"""Draw user-defined raw OFES SSH and regional surface-vector zooms with Pillow."""

from __future__ import annotations

import argparse
import json
import math
from datetime import date
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from Detection_for_OFES.ofes_io import ctl_path, parse_ctl, read_variable_latlon_daily_only
from Detection_for_OFES.tools.plot_latest_ssh_vector_overview import (
    diverging_ramp,
    draw_colorbar,
    font,
    lat_label,
    lon_label,
    px,
    py,
)


OFES_ROOT = Path(r"F:\OFES\external_OFES2")
OUTPUT_ROOT = Path(
    r"E:\DATA\01_Eddy_correspond\02_OFES\Fromthebeginning"
    r"\01_raw_ssh_region_vectors_19910101"
)
REGIONS = {
    "south_pacific_stcc": {
        "label": "South Pacific STCC",
        "bbox": (165.0, 230.0, -29.0, -21.0),
        "color": (245, 158, 11),
    },
    "kuroshio_extension": {
        "label": "Kuroshio Extension",
        "bbox": (140.0, 180.0, 28.0, 40.0),
        "color": (34, 197, 94),
    },
    "taiwan_hawaii_stcc_hlcc": {
        "label": "Taiwan-Hawaii STCC-HLCC corridor",
        "bbox": (122.0, 203.0, 18.0, 27.0),
        "color": (236, 72, 153),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ofes-root", type=Path, default=OFES_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--day", default="1991-01-01")
    parser.add_argument("--global-vector-spacing-deg", type=float, default=2.0)
    parser.add_argument("--zoom-vector-spacing-deg", type=float, default=0.3)
    return parser.parse_args()


def indices(coordinates: np.ndarray, lower: float, upper: float) -> np.ndarray:
    result = np.where((coordinates >= lower) & (coordinates <= upper))[0]
    if not result.size:
        raise ValueError(f"No coordinates within {lower}..{upper}")
    return result


def draw_arrow(draw: ImageDraw.ImageDraw, x0: float, y0: float, x1: float, y1: float) -> None:
    draw.line((x0, y0, x1, y1), fill=(15, 19, 25), width=2)
    angle = math.atan2(y1 - y0, x1 - x0)
    head = 6.5
    draw.polygon(
        [
            (x1, y1),
            (x1 - head * math.cos(angle - 0.48), y1 - head * math.sin(angle - 0.48)),
            (x1 - head * math.cos(angle + 0.48), y1 - head * math.sin(angle + 0.48)),
        ],
        fill=(15, 19, 25),
    )


def draw_grid(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], bbox: tuple[float, float, float, float], global_panel: bool) -> None:
    x0, y0, x1, y1 = box
    lon_min, lon_max, lat_min, lat_max = bbox
    lon_step = 60 if global_panel else 5
    lat_step = 30 if global_panel else 5
    for value in np.arange(math.ceil(lon_min / lon_step) * lon_step, lon_max + 0.1, lon_step):
        x = px(float(value), lon_min, lon_max, x0, x1)
        draw.line((x, y0, x, y1), fill=(45, 50, 60), width=1)
        draw.text((x - 18, y1 + 8), lon_label(float(value)), fill=(35, 40, 50), font=font(16))
    for value in np.arange(math.ceil(lat_min / lat_step) * lat_step, lat_max + 0.1, lat_step):
        y = py(float(value), lat_min, lat_max, y0, y1)
        draw.line((x0, y, x1, y), fill=(45, 50, 60), width=1)
        draw.text((x0 - 58, y - 9), lat_label(float(value)), fill=(35, 40, 50), font=font(16))
    draw.rectangle(box, outline=(12, 16, 24), width=2)


def draw_vectors(
    draw: ImageDraw.ImageDraw,
    *,
    lon: np.ndarray,
    lat: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    box: tuple[int, int, int, int],
    bbox: tuple[float, float, float, float],
    spacing_deg: float,
    arrow_pixels: float,
) -> dict[str, float | int]:
    x0, y0, x1, y1 = box
    lon_min, lon_max, lat_min, lat_max = bbox
    lon_idx = indices(lon, lon_min, lon_max)
    lat_idx = indices(lat, lat_min, lat_max)
    resolution = float(np.median(np.diff(lon)))
    step = max(1, int(round(spacing_deg / resolution)))
    lon_idx = lon_idx[step // 2::step]
    lat_idx = lat_idx[step // 2::step]
    u_subset = np.asarray(u[np.ix_(lat_idx, lon_idx)], dtype="f8")
    v_subset = np.asarray(v[np.ix_(lat_idx, lon_idx)], dtype="f8")
    latitude = lat[lat_idx, None]
    u_screen = u_subset / np.maximum(np.cos(np.deg2rad(latitude)), 0.2)
    speed = np.hypot(u_screen, v_subset)
    valid = np.isfinite(speed)
    q95 = float(np.nanpercentile(speed[valid], 95.0)) if valid.any() else 1.0
    scale = arrow_pixels / max(q95, 1.0e-6)
    for jj, latitude_value in enumerate(lat[lat_idx]):
        for ii, longitude_value in enumerate(lon[lon_idx]):
            uu = float(u_screen[jj, ii])
            vv = float(v_subset[jj, ii])
            if not np.isfinite(uu) or not np.isfinite(vv):
                continue
            x = px(float(longitude_value), lon_min, lon_max, x0, x1)
            y = py(float(latitude_value), lat_min, lat_max, y0, y1)
            # Keep deliberately prominent arrows inside the panel border.
            end_x = float(np.clip(x + uu * scale, x0 + 3, x1 - 3))
            end_y = float(np.clip(y - vv * scale, y0 + 3, y1 - 3))
            draw_arrow(draw, x, y, end_x, end_y)
    return {"spacing_deg": float(spacing_deg), "stride_cells": step, "arrow_q95_cm_s": q95, "count": int(valid.sum())}


def render_panel(
    *,
    canvas: Image.Image,
    box: tuple[int, int, int, int],
    bbox: tuple[float, float, float, float],
    title: str,
    ssh_lon: np.ndarray,
    ssh_lat: np.ndarray,
    raw_ssh: np.ndarray,
    vel_lon: np.ndarray,
    vel_lat: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    color_limit: float,
    vector_spacing_deg: float,
    arrow_pixels: float,
    global_panel: bool = False,
) -> dict[str, float | int]:
    draw = ImageDraw.Draw(canvas)
    x0, y0, x1, y1 = box
    lon_min, lon_max, lat_min, lat_max = bbox
    lon_idx = indices(ssh_lon, lon_min, lon_max)
    lat_idx = indices(ssh_lat, lat_min, lat_max)
    field = raw_ssh[np.ix_(lat_idx, lon_idx)]
    rgb = diverging_ramp(np.nan_to_num(field, nan=0.0), -color_limit, color_limit)
    rgb[~np.isfinite(field)] = np.array([36, 40, 48], dtype=np.uint8)
    map_image = Image.fromarray(rgb[::-1], "RGB").resize((x1 - x0, y1 - y0), Image.Resampling.BILINEAR)
    canvas.paste(map_image, (x0, y0))
    draw_grid(draw, box, bbox, global_panel)
    info = draw_vectors(
        draw,
        lon=vel_lon,
        lat=vel_lat,
        u=u,
        v=v,
        box=box,
        bbox=bbox,
        spacing_deg=vector_spacing_deg,
        arrow_pixels=arrow_pixels,
    )
    draw.text((x0, y0 - 32), title, fill=(20, 25, 34), font=font(24))
    return info


def draw_boxes(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], bbox: tuple[float, float, float, float]) -> None:
    x0, y0, x1, y1 = box
    lon_min, lon_max, lat_min, lat_max = bbox
    for region in REGIONS.values():
        left, right, bottom, top = region["bbox"]
        coords = (
            px(left, lon_min, lon_max, x0, x1),
            py(top, lat_min, lat_max, y0, y1),
            px(right, lon_min, lon_max, x0, x1),
            py(bottom, lat_min, lat_max, y0, y1),
        )
        draw.rectangle(coords, outline=region["color"], width=4)
        draw.text((coords[0] + 5, coords[1] + 5), region["label"], fill=region["color"], font=font(17))


def save_zoom(
    args: argparse.Namespace,
    *,
    target_day: date,
    region_key: str,
    region: dict[str, object],
    ssh_lon: np.ndarray,
    ssh_lat: np.ndarray,
    raw_ssh: np.ndarray,
    vel_lon: np.ndarray,
    vel_lat: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    color_limit: float,
) -> tuple[str, dict[str, float | int]]:
    image = Image.new("RGB", (2000, 1280), "white")
    info = render_panel(
        canvas=image,
        box=(110, 105, 1760, 1110),
        bbox=region["bbox"],
        title=f"OFES raw SSH | {region['label']} | {target_day.isoformat()}",
        ssh_lon=ssh_lon,
        ssh_lat=ssh_lat,
        raw_ssh=raw_ssh,
        vel_lon=vel_lon,
        vel_lat=vel_lat,
        u=u,
        v=v,
        color_limit=color_limit,
        vector_spacing_deg=args.zoom_vector_spacing_deg,
        arrow_pixels=31.0,
    )
    draw = ImageDraw.Draw(image)
    draw_colorbar(image, draw, 1820, 105, 30, 1005, -color_limit, color_limit, "cm")
    draw.text((110, 1160), "Raw SSH: H_OFES = eta - (pressur - 1000). Surface vectors: raw u/v at 2.5 m, sampled every 0.3 degrees.", fill=(45, 55, 70), font=font(18))
    output = args.output_root / f"raw_ssh_vector_zoom_{region_key}_{target_day:%Y%m%d}.png"
    image.save(output)
    image.save(output.with_suffix(".pdf"), "PDF", resolution=180.0)
    return str(output), info


def main() -> None:
    args = parse_args()
    target_day = date.fromisoformat(args.day)
    ssh_meta = parse_ctl(ctl_path(args.ofes_root, "eta"))
    velocity_meta = parse_ctl(ctl_path(args.ofes_root, "u"))
    ssh_lon = np.asarray(ssh_meta.x.values, dtype="f8")
    ssh_lat = np.asarray(ssh_meta.y.values, dtype="f8")
    vel_lon = np.asarray(velocity_meta.x.values, dtype="f8")
    vel_lat = np.asarray(velocity_meta.y.values, dtype="f8")
    eta = np.asarray(read_variable_latlon_daily_only(args.ofes_root, "eta", target_day), dtype="f8")
    pressure = np.asarray(read_variable_latlon_daily_only(args.ofes_root, "pressur", target_day), dtype="f8")
    u = np.asarray(read_variable_latlon_daily_only(args.ofes_root, "u", target_day, level_index=0), dtype="f8")
    v = np.asarray(read_variable_latlon_daily_only(args.ofes_root, "v", target_day, level_index=0), dtype="f8")
    raw_ssh = eta - (pressure - 1000.0)
    raw_ssh[~np.isfinite(eta) | ~np.isfinite(pressure)] = np.nan
    color_limit = float(np.nanpercentile(np.abs(raw_ssh), 99.0))
    args.output_root.mkdir(parents=True, exist_ok=True)

    image = Image.new("RGB", (3000, 1900), "white")
    global_box = (100, 140, 1475, 930)
    info = {
        "global": render_panel(
            canvas=image,
            box=global_box,
            bbox=(0.0, 360.0, -76.0, 76.0),
            title=f"Global context | raw H_OFES | {target_day.isoformat()}",
            ssh_lon=ssh_lon,
            ssh_lat=ssh_lat,
            raw_ssh=raw_ssh,
            vel_lon=vel_lon,
            vel_lat=vel_lat,
            u=u,
            v=v,
            color_limit=color_limit,
            vector_spacing_deg=args.global_vector_spacing_deg,
            arrow_pixels=24.0,
            global_panel=True,
        )
    }
    draw_boxes(ImageDraw.Draw(image), global_box, (0.0, 360.0, -76.0, 76.0))
    layouts = [
        ("south_pacific_stcc", (1640, 140, 2900, 730)),
        ("kuroshio_extension", (100, 1085, 1360, 1675)),
        ("taiwan_hawaii_stcc_hlcc", (1640, 1085, 2900, 1675)),
    ]
    for key, panel_box in layouts:
        region = REGIONS[key]
        info[key] = render_panel(
            canvas=image,
            box=panel_box,
            bbox=region["bbox"],
            title=region["label"],
            ssh_lon=ssh_lon,
            ssh_lat=ssh_lat,
            raw_ssh=raw_ssh,
            vel_lon=vel_lon,
            vel_lat=vel_lat,
            u=u,
            v=v,
            color_limit=color_limit,
            vector_spacing_deg=args.zoom_vector_spacing_deg,
            arrow_pixels=30.0,
        )
    draw = ImageDraw.Draw(image)
    draw.text((100, 35), "OFES raw SSH with three regional vector zooms", fill=(18, 24, 38), font=font(40))
    draw.text((100, 82), "H_OFES = eta - (pressur - 1000); raw 2.5 m u/v. Global vectors: 2.0 degrees; zoom vectors: 0.3 degrees.", fill=(70, 80, 95), font=font(20))
    draw_colorbar(image, draw, 2940, 140, 30, 1560, -color_limit, color_limit, "cm")
    combined = args.output_root / f"raw_ssh_three_regions_vectors_{target_day:%Y%m%d}.png"
    image.save(combined)
    image.save(combined.with_suffix(".pdf"), "PDF", resolution=180.0)
    zooms = {}
    for key, region in REGIONS.items():
        path, zoom_info = save_zoom(
            args,
            target_day=target_day,
            region_key=key,
            region=region,
            ssh_lon=ssh_lon,
            ssh_lat=ssh_lat,
            raw_ssh=raw_ssh,
            vel_lon=vel_lon,
            vel_lat=vel_lat,
            u=u,
            v=v,
            color_limit=color_limit,
        )
        zooms[key] = path
        info[f"{key}_zoom"] = zoom_info
    manifest = {
        "status": "complete",
        "day": target_day.isoformat(),
        "raw_ssh_definition": "H_OFES=eta-(pressur-1000)",
        "processing": "No temporal averaging, spatial filtering, regridding, eddy detection, QC, or persistence.",
        "vector_definition": "Raw OFES u/v at 2.5 m, arrows sampled at 0.3 degrees in regional zooms.",
        "regions": REGIONS,
        "color_limit_cm": color_limit,
        "sampling": info,
        "combined_png": str(combined),
        "zoom_png": zooms,
    }
    manifest_path = args.output_root / f"raw_ssh_three_regions_vectors_manifest_{target_day:%Y%m%d}.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
