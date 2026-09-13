from __future__ import annotations

import json
import math
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


RESULT_ROOT = Path(r"E:\DATA\01_Eddy_correspond\02_OFES\origin_streamline_cpu_jan01_jan19_life1")
FILTER_ROOT = Path(r"E:\DATA\01_Eddy_correspond\02_OFES\origin_compatible_filter")
DET_DIR = RESULT_ROOT / "hua_b3_start2_detection"
OUT_DIR = RESULT_ROOT / "figures" / "latest_velocity_streamline_surface_and_family_with_edges"


def font(size: int) -> ImageFont.ImageFont:
    for path in [r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\arial.ttf"]:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            pass
    return ImageFont.load_default()


def color_ramp(values: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    stops = np.asarray(
        [
            [40, 24, 85],
            [42, 88, 135],
            [34, 144, 140],
            [85, 190, 100],
            [250, 230, 55],
            [245, 130, 32],
            [185, 35, 20],
        ],
        dtype=np.float32,
    )
    t = np.clip((values - vmin) / max(vmax - vmin, 1e-12), 0.0, 1.0)
    scaled = t * (len(stops) - 1)
    lo = np.floor(scaled).astype(np.int32)
    hi = np.clip(lo + 1, 0, len(stops) - 1)
    frac = scaled - lo
    rgb = stops[lo] * (1.0 - frac[..., None]) + stops[hi] * frac[..., None]
    return np.clip(rgb, 0, 255).astype(np.uint8)


def read_surface_speed(day: str, lon_min: float, lon_max: float, lat_min: float, lat_max: float):
    ymd = day.replace("-", "")
    nc_path = FILTER_ROOT / f"global_phy_{ymd}.nc"
    with h5py.File(nc_path, "r") as ds:
        lon = ds["longitude"][:].astype(float)
        lat = ds["latitude"][:].astype(float)
        lon_idx = np.where((lon >= lon_min) & (lon <= lon_max))[0]
        lat_idx = np.where((lat >= lat_min) & (lat <= lat_max))[0]
        if lon_idx.size == 0 or lat_idx.size == 0:
            raise ValueError("requested Kuroshio box is outside NetCDF coordinates")
        i0, i1 = int(lon_idx[0]), int(lon_idx[-1]) + 1
        j0, j1 = int(lat_idx[0]), int(lat_idx[-1]) + 1
        u = ds["uo_glor"][0, 0, j0:j1, i0:i1].astype(np.float32)
        v = ds["vo_glor"][0, 0, j0:j1, i0:i1].astype(np.float32)
        lon_sub = lon[i0:i1]
        lat_sub = lat[j0:j1]
    u[np.abs(u) > 1e20] = np.nan
    v[np.abs(v) > 1e20] = np.nan
    speed = np.sqrt(u * u + v * v) * 100.0
    return lon_sub, lat_sub, speed


def px(lon: float, lon_min: float, lon_max: float, x0: int, x1: int) -> int:
    return x0 + int(round((lon - lon_min) / (lon_max - lon_min) * (x1 - x0)))


def py(lat: float, lat_min: float, lat_max: float, y0: int, y1: int) -> int:
    return y1 - int(round((lat - lat_min) / (lat_max - lat_min) * (y1 - y0)))


def draw_map(day: str = "1991-01-10") -> dict[str, str]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lon_min, lon_max, lat_min, lat_max = 120.0, 145.0, 20.0, 35.0
    lon, lat, speed = read_surface_speed(day, lon_min, lon_max, lat_min, lat_max)

    good = np.isfinite(speed)
    vmin = 0.0
    vmax = float(np.nanpercentile(speed[good], 98.5)) if np.any(good) else 1.0
    rgb = color_ramp(np.nan_to_num(speed, nan=vmin), vmin, vmax)
    rgb[~good] = np.array([34, 36, 42], dtype=np.uint8)
    rgb = rgb[::-1, :, :]

    map_box = (85, 95, 1665, 1045)
    map_img = Image.fromarray(rgb, "RGB").resize((map_box[2] - map_box[0], map_box[3] - map_box[1]), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (1840, 1160), "white")
    canvas.paste(map_img, map_box[:2])
    draw = ImageDraw.Draw(canvas)

    title = f"OFES velocity_streamline_contour eddies in the Kuroshio region | {day}"
    draw.text((50, 28), title, fill=(18, 24, 38), font=font(34))
    draw.text(
        (50, 66),
        "Background: surface velocity-anomaly speed (cm/s). Edges: accepted boundary radius proxies; colors show polarity.",
        fill=(75, 85, 100),
        font=font(18),
    )

    x0, y0, x1, y1 = map_box
    for glon in range(120, 146, 5):
        x = px(glon, lon_min, lon_max, x0, x1)
        draw.line((x, y0, x, y1), fill=(20, 24, 32), width=1)
        draw.text((x - 16, y1 + 8), f"{glon}E", fill=(35, 40, 50), font=font(16))
    for glat in range(20, 36, 5):
        y = py(glat, lat_min, lat_max, y0, y1)
        draw.line((x0, y, x1, y), fill=(20, 24, 32), width=1)
        draw.text((x0 - 62, y - 10), f"{glat}N", fill=(35, 40, 50), font=font(16))
    draw.rectangle(map_box, outline=(10, 15, 25), width=2)

    centers = pd.read_parquet(DET_DIR / "centers_hua_style.parquet")
    structures = pd.read_parquet(DET_DIR / "structures_hua_style.parquet")
    centers = centers[
        centers["boundary_mode"].astype(str).eq("velocity_streamline_contour")
        & centers["hua_pass"].astype(bool)
        & centers["depth_index"].astype(int).eq(0)
    ].copy()
    centers["date"] = pd.to_datetime(centers["date"]).dt.strftime("%Y-%m-%d")
    structures["date"] = pd.to_datetime(structures["date"]).dt.strftime("%Y-%m-%d")
    surface_radius = structures[structures["depth_index"].astype(int).eq(0)][["date", "hua_object_id", "radius_km"]]
    day_df = centers[centers["date"].eq(day)].merge(surface_radius, on=["date", "hua_object_id"], how="left")
    day_df = day_df[
        day_df["center_lon_refined"].between(lon_min, lon_max)
        & day_df["center_lat_refined"].between(lat_min, lat_max)
        & np.isfinite(day_df["radius_km"])
    ].copy()

    colors = {"cyclonic": (37, 99, 235), "anticyclonic": (220, 38, 38)}
    for _, row in day_df.iterrows():
        lon0 = float(row.center_lon_refined)
        lat0 = float(row.center_lat_refined)
        radius_km = float(row.radius_km)
        color = colors.get(str(row.polarity), (30, 30, 30))
        cx = px(lon0, lon_min, lon_max, x0, x1)
        cy = py(lat0, lat_min, lat_max, y0, y1)
        deg_lat = radius_km / 111.2
        deg_lon = deg_lat / max(math.cos(math.radians(lat0)), 0.2)
        rx = abs(px(lon0 + deg_lon, lon_min, lon_max, x0, x1) - cx)
        ry = abs(py(lat0 + deg_lat, lat_min, lat_max, y0, y1) - cy)
        draw.ellipse((cx - rx, cy - ry, cx + rx, cy + ry), outline=color, width=3)
        draw.ellipse((cx - 4, cy - 4, cx + 4, cy + 4), fill=color, outline=(255, 255, 255), width=1)

    cb_x0, cb_y0, cb_x1, cb_y1 = 1705, 125, 1742, 990
    grad = np.linspace(vmax, vmin, cb_y1 - cb_y0, dtype=np.float32)[:, None]
    cb = Image.fromarray(color_ramp(grad, vmin, vmax), "RGB").resize((cb_x1 - cb_x0, cb_y1 - cb_y0))
    canvas.paste(cb, (cb_x0, cb_y0))
    draw.rectangle((cb_x0, cb_y0, cb_x1, cb_y1), outline=(20, 24, 32), width=1)
    for frac in np.linspace(0, 1, 6):
        val = vmax * (1.0 - frac)
        yy = cb_y0 + int(frac * (cb_y1 - cb_y0))
        draw.line((cb_x1, yy, cb_x1 + 8, yy), fill=(20, 24, 32), width=1)
        draw.text((cb_x1 + 12, yy - 10), f"{val:.1f}", fill=(20, 24, 32), font=font(15))
    draw.text((1688, 1010), "cm/s", fill=(20, 24, 32), font=font(17))

    legend_y = 1068
    draw.ellipse((92, legend_y - 7, 106, legend_y + 7), fill=colors["cyclonic"])
    draw.text((116, legend_y - 11), "cyclonic", fill=(20, 24, 32), font=font(18))
    draw.ellipse((220, legend_y - 7, 234, legend_y + 7), fill=colors["anticyclonic"])
    draw.text((244, legend_y - 11), "anticyclonic", fill=(20, 24, 32), font=font(18))
    draw.text((380, legend_y - 11), f"surface objects in box: {len(day_df)}", fill=(20, 24, 32), font=font(18))

    png = OUT_DIR / f"ofes_velocity_streamline_kuroshio_lavd_style_{day.replace('-', '')}.png"
    pdf = png.with_suffix(".pdf")
    canvas.save(png)
    canvas.save(pdf, "PDF", resolution=180.0)
    manifest = {
        "day": day,
        "boundary_mode": "velocity_streamline_contour",
        "region": {"lon_min": lon_min, "lon_max": lon_max, "lat_min": lat_min, "lat_max": lat_max},
        "background": "surface velocity-anomaly speed from uo_glor/vo_glor, cm/s",
        "objects_in_box": int(len(day_df)),
        "edge_note": "Accepted boundary radius proxy, not persisted streamline vertices.",
        "png": str(png),
        "pdf": str(pdf),
    }
    (OUT_DIR / f"kuroshio_lavd_style_manifest_{day.replace('-', '')}.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


if __name__ == "__main__":
    print(json.dumps(draw_map(), ensure_ascii=False, indent=2))
