from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


def font(size: int):
    for p in [r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\arial.ttf"]:
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            pass
    return ImageFont.load_default()


def local_xy_km(lon: float, lat: float, lon0: float, lat0: float) -> tuple[float, float]:
    mx = 111.2 * math.cos(math.radians(lat0))
    return ((lon - lon0 + 180.0) % 360.0 - 180.0) * mx, (lat - lat0) * 111.2


def project(lon: float, lat: float, box: tuple[int, int, int, int]) -> tuple[int, int]:
    x0, y0, x1, y1 = box
    x = x0 + int(round(lon / 360.0 * (x1 - x0)))
    y = y1 - int(round((lat + 80.0) / 160.0 * (y1 - y0)))
    return x, y


def draw_overview_panel(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], day_df: pd.DataFrame, day: str) -> None:
    x0, y0, x1, y1 = box
    draw.rectangle(box, outline=(130, 138, 150), width=1)
    for lon in range(0, 361, 60):
        px, _ = project(lon, 0, box)
        draw.line((px, y0, px, y1), fill=(226, 230, 236), width=1)
        draw.text((px + 2, y1 + 4), str(lon), fill=(90, 96, 106), font=font(12))
    for lat in range(-60, 81, 20):
        _, py = project(0, lat, box)
        draw.line((x0, py, x1, py), fill=(226, 230, 236), width=1)
        draw.text((x0 - 34, py - 7), str(lat), fill=(90, 96, 106), font=font(12))
    colors = {"cyclonic": (37, 99, 235), "anticyclonic": (220, 38, 38)}
    for _, row in day_df.iterrows():
        lon = float(row.center_lon_refined)
        lat = float(row.center_lat_refined)
        radius_km = float(row.radius_km)
        color = colors.get(str(row.polarity), (70, 70, 70))
        px, py = project(lon, lat, box)
        deg_lat = radius_km / 111.2
        deg_lon = deg_lat / max(math.cos(math.radians(lat)), 0.18)
        px_e, _ = project(min(360.0, lon + deg_lon), lat, box)
        _, py_n = project(lon, min(80.0, lat + deg_lat), box)
        rx = max(1, abs(px_e - px))
        ry = max(1, abs(py_n - py))
        edge = (*color, 120)
        draw.ellipse((px - rx, py - ry, px + rx, py + ry), outline=color, width=1)
        draw.ellipse((px - 2, py - 2, px + 2, py + 2), fill=color)
    k0 = project(120, 20, box)
    k1 = project(145, 35, box)
    draw.rectangle((k0[0], k1[1], k1[0], k0[1]), outline=(0, 0, 0), width=2)
    draw.text((x0 + 8, y0 + 8), day, fill=(25, 30, 38), font=font(22))


def draw_xy_axes(draw, box, xmin, xmax, ymin, ymax, xlabel, ylabel):
    x0, y0, x1, y1 = box
    draw.rectangle(box, outline=(130, 138, 150), width=1)
    for frac in [0.25, 0.5, 0.75]:
        gx = x0 + int(frac * (x1 - x0))
        gy = y0 + int(frac * (y1 - y0))
        draw.line((gx, y0, gx, y1), fill=(226, 230, 236), width=1)
        draw.line((x0, gy, x1, gy), fill=(226, 230, 236), width=1)
    draw.text((x0, y1 + 4), xlabel, fill=(80, 88, 100), font=font(13))
    draw.text((x0 - 36, y0 + 4), ylabel, fill=(80, 88, 100), font=font(13))

    def clamp(v: int) -> int:
        return max(-10000, min(10000, v))

    def px(x):
        if not np.isfinite(x):
            return clamp(x0)
        denom = xmax - xmin
        if abs(denom) < 1e-12:
            return clamp((x0 + x1) // 2)
        return clamp(x0 + int(round((x - xmin) / denom * (x1 - x0))))

    def py(y):
        if not np.isfinite(y):
            return clamp(y1)
        denom = ymax - ymin
        if abs(denom) < 1e-12:
            return clamp((y0 + y1) // 2)
        # Support reversed axes, used for depth panels where shallow is top.
        if denom > 0:
            return clamp(y1 - int(round((y - ymin) / denom * (y1 - y0))))
        return clamp(y0 + int(round((y - ymax) / (-denom) * (y1 - y0))))

    return px, py


def draw_family_panel(out: Path, oid: str, row: pd.Series, part: pd.DataFrame, center_part: pd.DataFrame) -> None:
    surface_row = part.iloc[0]
    lon0 = float(surface_row.center_lon_refined)
    lat0 = float(surface_row.center_lat_refined)
    radius0 = float(surface_row.radius_km)
    dx, dy, edge_r, depth = [], [], [], []
    for _, rr in part.iterrows():
        x, y = local_xy_km(float(rr.center_lon_refined), float(rr.center_lat_refined), lon0, lat0)
        dx.append(x)
        dy.append(y)
        edge_r.append(float(rr.radius_km))
        depth.append(float(rr.depth_m))
    dx = np.asarray(dx)
    dy = np.asarray(dy)
    edge_r = np.asarray(edge_r)
    depth = np.asarray(depth)
    sp = center_part[center_part.depth_index.astype(int).eq(0)].head(1)
    points = int(float(sp.streamline_points.iloc[0])) if not sp.empty else -1

    canvas = Image.new("RGB", (1900, 1120), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((30, 20), f"OFES velocity_streamline family-panel with eddy edges | {oid} | {row.date} | {row.polarity}", fill=(20, 24, 32), font=font(30))
    draw.text((30, 58), f"layers={int(row.pass_layers)}; surface radius={radius0:.1f} km; surface streamline closed metadata points={points}", fill=(80, 88, 100), font=font(17))

    zmin, zmax = float(np.nanmin(depth)), float(np.nanmax(depth))
    xmin = min(float(np.nanmin(dx - edge_r)), -radius0 * 1.5)
    xmax = max(float(np.nanmax(dx + edge_r)), radius0 * 1.5)
    ymin = min(float(np.nanmin(dy - edge_r)), -radius0 * 1.5)
    ymax = max(float(np.nanmax(dy + edge_r)), radius0 * 1.5)
    dbox1 = (70, 130, 430, 1010)
    dbox2 = (510, 130, 870, 1010)
    px1, py1 = draw_xy_axes(draw, dbox1, xmin, xmax, zmax, zmin, "delta x km", "depth m")
    px2, py2 = draw_xy_axes(draw, dbox2, ymin, ymax, zmax, zmin, "delta y km", "depth m")
    draw.text((70, 100), "1 delta x from surface center", fill=(30, 36, 48), font=font(18))
    draw.text((510, 100), "2 delta y from surface center", fill=(30, 36, 48), font=font(18))
    for arr, edges, px, py, box in [(dx, edge_r, px1, py1, dbox1), (dy, edge_r, px2, py2, dbox2)]:
        pts = [(px(float(v)), py(float(z))) for v, z in zip(arr, depth)]
        if len(pts) > 1:
            draw.line(pts, fill=(31, 75, 153), width=3)
        for v, er, z in zip(arr, edges, depth):
            y = py(float(z))
            draw.line((px(float(v - er)), y, px(float(v + er)), y), fill=(31, 75, 153), width=1)
            draw.ellipse((px(float(v)) - 3, y - 3, px(float(v)) + 3, y + 3), fill=(31, 75, 153))
    # local projected map
    box3 = (1000, 150, 1760, 800)
    lim = max(4.0 * max(float(np.nanmax(edge_r)), 1.0), float(np.nanmax(np.hypot(dx, dy))) + float(np.nanmax(edge_r)))
    px, py = draw_xy_axes(draw, box3, -lim, lim, -lim, lim, "east km", "north km")
    draw.text((1000, 115), "3 projected layer centers and accepted edges", fill=(30, 36, 48), font=font(18))
    for x, y, r, z in zip(dx, dy, edge_r, depth):
        cx, cy = px(float(x)), py(float(y))
        rx = abs(px(float(x + r)) - cx)
        ry = abs(py(float(y + r)) - cy)
        shade = int(40 + 180 * (float(z) - zmin) / max(zmax - zmin, 1e-12))
        draw.ellipse((cx - rx, cy - ry, cx + rx, cy + ry), outline=(90, 90, 90), width=1)
        draw.ellipse((cx - 3, cy - 3, cx + 3, cy + 3), fill=(30, shade, 140))
    cx, cy = px(0), py(0)
    rx = abs(px(radius0) - cx)
    ry = abs(py(radius0) - cy)
    draw.ellipse((cx - rx, cy - ry, cx + rx, cy + ry), outline=(0, 0, 0), width=3)
    draw.ellipse((cx - 6, cy - 6, cx + 6, cy + 6), fill=(220, 38, 38))
    # lon-lat mini view
    box4 = (1000, 870, 1760, 1040)
    draw.rectangle(box4, outline=(130, 138, 150), width=1)
    draw.text((1000, 835), "4 lon-lat projected family with edges", fill=(30, 36, 48), font=font(18))
    lon_min, lon_max = float(part.center_lon_refined.min() - 1.0), float(part.center_lon_refined.max() + 1.0)
    lat_min, lat_max = float(part.center_lat_refined.min() - 1.0), float(part.center_lat_refined.max() + 1.0)

    def lonpx(lon):
        return box4[0] + int((lon - lon_min) / max(lon_max - lon_min, 1e-12) * (box4[2] - box4[0]))

    def latpy(lat):
        return box4[3] - int((lat - lat_min) / max(lat_max - lat_min, 1e-12) * (box4[3] - box4[1]))

    pts = [(lonpx(float(lon)), latpy(float(lat))) for lon, lat in zip(part.center_lon_refined, part.center_lat_refined)]
    if len(pts) > 1:
        draw.line(pts, fill=(31, 75, 153), width=3)
    for _, rr in part.iterrows():
        cx, cy = lonpx(float(rr.center_lon_refined)), latpy(float(rr.center_lat_refined))
        deg = float(rr.radius_km) / 111.2
        rx = abs(lonpx(float(rr.center_lon_refined) + deg) - cx)
        ry = abs(latpy(float(rr.center_lat_refined) + deg) - cy)
        draw.ellipse((cx - rx, cy - ry, cx + rx, cy + ry), outline=(90, 90, 90), width=1)
    canvas.save(out)
    canvas.save(out.with_suffix(".pdf"), "PDF", resolution=180.0)


def main() -> None:
    root = Path(r"E:\DATA\01_Eddy_correspond\02_OFES\origin_streamline_cpu_jan01_jan19_life1")
    det = root / "hua_b3_start2_detection"
    out = root / "figures" / "latest_velocity_streamline_surface_and_family_with_edges"
    out.mkdir(parents=True, exist_ok=True)
    centers = pd.read_parquet(det / "centers_hua_style.parquet")
    structures = pd.read_parquet(det / "structures_hua_style.parquet")
    centers = centers[centers["boundary_mode"].astype(str).eq("velocity_streamline_contour")].copy()
    centers["date"] = pd.to_datetime(centers["date"]).dt.strftime("%Y-%m-%d")
    structures["date"] = pd.to_datetime(structures["date"]).dt.strftime("%Y-%m-%d")

    surface = centers[(centers["depth_index"].astype(int) == 0) & (centers["hua_pass"].astype(bool))].copy()
    surf_struct = structures[structures["depth_index"].astype(int).eq(0)][["date", "hua_object_id", "radius_km"]].copy()
    surface = surface.merge(surf_struct, on=["date", "hua_object_id"], how="left")
    surface = surface.dropna(subset=["center_lon_refined", "center_lat_refined", "radius_km"])

    canvas = Image.new("RGB", (1900, 1450), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((35, 25), "OFES latest velocity_streamline_contour surface overview with eddy edges", fill=(20, 24, 32), font=font(32))
    draw.text((35, 65), "Edges are accepted boundary radius proxies from velocity-streamline metadata; Kuroshio box is outlined.", fill=(80, 88, 100), font=font(17))
    for idx, day in enumerate(["1991-01-01", "1991-01-10", "1991-01-19"]):
        day_df = surface[surface["date"].eq(day)].copy()
        kuro = day_df[(day_df.center_lon_refined.between(120, 145)) & (day_df.center_lat_refined.between(20, 35))]
        other = day_df.drop(kuro.index)
        if len(other) > 650:
            other = other.sample(650, random_state=20260913)
        show = pd.concat([other, kuro], ignore_index=True)
        box = (95, 135 + idx * 420, 1800, 485 + idx * 420)
        draw_overview_panel(draw, box, show, f"{day}: shown={len(show):,}/{len(day_df):,}; Kuroshio={len(kuro):,}")
    overview = out / "ofes_velocity_streamline_surface_overview_with_edges.png"
    canvas.save(overview)
    canvas.save(overview.with_suffix(".pdf"), "PDF", resolution=180.0)

    surface_struct = structures[structures.depth_index.astype(int).eq(0)].copy()
    surface_struct = surface_struct[(surface_struct.center_lon_refined.between(120, 145)) & (surface_struct.center_lat_refined.between(20, 35))]
    summary = structures.groupby("hua_object_id").agg(
        date=("date", "first"),
        polarity=("polarity", "first"),
        pass_layers=("depth_index", "nunique"),
        radius_km=("radius_km", "first"),
        max_depth=("depth_m", "max"),
    ).reset_index()
    summary = summary.merge(surface_struct[["hua_object_id"]], on="hua_object_id", how="inner")
    selected = []
    for pol in ["cyclonic", "anticyclonic"]:
        cand = summary[summary.polarity.eq(pol)].sort_values(["pass_layers", "max_depth", "radius_km"], ascending=False)
        if not cand.empty:
            selected.append(cand.iloc[0])
    panels = []
    for row in selected:
        oid = str(row.hua_object_id)
        panel = out / f"ofes_velocity_streamline_family_panel_with_edges_{oid}.png"
        draw_family_panel(panel, oid, row, structures[structures.hua_object_id.eq(oid)].sort_values("depth_index"), centers[centers.hua_object_id.eq(oid)])
        panels.append(str(panel))
    manifest = {
        "boundary_mode": "velocity_streamline_contour",
        "overview_png": str(overview),
        "family_panels": panels,
        "edge_note": "Edges use accepted boundary radius proxy from velocity_streamline_contour metadata; streamline vertex coordinates were not persisted in centers_hua_style.parquet.",
    }
    (out / "surface_family_with_edges_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
