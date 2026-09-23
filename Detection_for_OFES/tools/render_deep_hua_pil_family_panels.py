"""Render portable curve-section family panels without the Matplotlib backend.

The host currently blocks Matplotlib's native Agg renderer.  This renderer uses
Pillow only and consumes the accepted vertical tracks plus the bilateral section
diagnostics.  It does not perform or alter Hua acceptance.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from netCDF4 import Dataset
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import map_coordinates


EARTH_KM = 6371.0
KM_PER_DEGREE = 111.195
REGIONS = {
    "kuroshio_extension": (140.0, 180.0, 28.0, 40.0),
    "south_pacific_stcc": (165.0, 230.0, -29.0, -21.0),
    "taiwan_hawaii_corridor": (122.0, 203.0, 18.0, 27.0),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render Pillow curve-section panels from accepted deep Hua tracks.")
    parser.add_argument("--vertical-root", type=Path, required=True)
    parser.add_argument("--quality-summary", type=Path, required=True)
    parser.add_argument("--quality-layers", type=Path, required=True)
    parser.add_argument("--velocity-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-bipolar-fraction", type=float, default=0.60)
    parser.add_argument("--per-region", type=int, default=4)
    return parser.parse_args()


def read_structures(root: Path) -> pd.DataFrame:
    run = root / "raw_detection" / "daily_runs" / "19910101"
    parquet = run / "structures_hua_style.parquet"
    if parquet.exists():
        try:
            return pd.read_parquet(parquet)
        except (ImportError, OSError):
            pass
    return pd.read_csv(run / "structures_hua_style.csv")


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    names = ("segoeuib.ttf", "arialbd.ttf") if bold else ("segoeui.ttf", "arial.ttf")
    for name in names:
        path = Path(r"C:\Windows\Fonts") / name
        if path.exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def wrapped_lon_delta(lon: np.ndarray, ref: float) -> np.ndarray:
    return (lon - ref + 180.0) % 360.0 - 180.0


def in_region(lon: float, lat: float, box: tuple[float, float, float, float]) -> bool:
    west, east, south, north = box
    return west <= lon % 360.0 <= east and south <= lat <= north


def signed_rgb(values: np.ndarray, vmax: float) -> np.ndarray:
    norm = np.clip(values / max(vmax, 1.0e-12), -1.0, 1.0)
    out = np.full((*values.shape, 3), 235, dtype=np.uint8)
    neg, pos = norm < 0, norm > 0
    out[..., 0][neg] = (245 * (1.0 + norm[neg])).astype(np.uint8)
    out[..., 1][neg] = (245 * (1.0 + norm[neg])).astype(np.uint8)
    out[..., 2][neg] = 170
    out[..., 0][pos] = 185
    out[..., 1][pos] = (245 * (1.0 - norm[pos])).astype(np.uint8)
    out[..., 2][pos] = (245 * (1.0 - norm[pos])).astype(np.uint8)
    out[~np.isfinite(values)] = (90, 90, 90)
    return out


def draw_box(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], title: str) -> None:
    draw.rectangle(box, outline=(45, 50, 60), width=2)
    draw.text((box[0] + 8, box[1] + 6), title, fill=(20, 25, 35), font=font(20, True))


def xy_to_box(x: np.ndarray, y: np.ndarray, box: tuple[int, int, int, int], pad: int = 34) -> tuple[np.ndarray, np.ndarray]:
    finite = np.isfinite(x) & np.isfinite(y)
    if not finite.any():
        return np.full_like(x, np.nan), np.full_like(y, np.nan)
    xmin, xmax = np.nanmin(x[finite]), np.nanmax(x[finite])
    ymin, ymax = np.nanmin(y[finite]), np.nanmax(y[finite])
    dx, dy = max(xmax - xmin, 1.0), max(ymax - ymin, 1.0)
    xmin -= .08 * dx; xmax += .08 * dx; ymin -= .08 * dy; ymax += .08 * dy
    px = box[0] + pad + (x - xmin) / (xmax - xmin) * (box[2] - box[0] - 2 * pad)
    py = box[3] - pad - (y - ymin) / (ymax - ymin) * (box[3] - box[1] - 2 * pad)
    return px, py


def sample_section(
    rows: pd.DataFrame,
    ds: Dataset,
    lon0: float,
    dlon: float,
    lat0: float,
    dlat: float,
    velocity_cache: dict[int, tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = rows.sort_values("depth_index").copy()
    offsets = np.linspace(-1.2, 1.2, 61)
    section = np.full((len(rows), len(offsets)), np.nan, dtype="f8")
    for index, row in enumerate(rows.itertuples(index=False)):
        radius = float(row.radius_km)
        distance = offsets * max(radius, 2.0)
        tx, ty = float(row.axis_tx_east), float(row.axis_ty_north)
        nx, ny = -ty, tx
        lat = float(row.center_lat) + ny * distance / KM_PER_DEGREE
        cos_lat = max(np.cos(np.deg2rad(float(row.center_lat))), 0.05)
        lon = (float(row.center_lon) + nx * distance / (KM_PER_DEGREE * cos_lat)) % 360.0
        iy = (lat - lat0) / dlat
        ix = np.mod((lon - lon0) / dlon, len(ds.dimensions["longitude"]))
        depth_index = int(row.depth_index)
        if depth_index not in velocity_cache:
            velocity_cache[depth_index] = (
                np.asarray(ds.variables["uo_glor"][0, depth_index], dtype="f4"),
                np.asarray(ds.variables["vo_glor"][0, depth_index], dtype="f4"),
            )
        u, v = velocity_cache[depth_index]
        up = map_coordinates(u, np.vstack([iy, ix]), order=1, mode="nearest", prefilter=False)
        vp = map_coordinates(v, np.vstack([iy, ix]), order=1, mode="nearest", prefilter=False)
        section[index] = up * tx + vp * ty
    return offsets, rows["depth_m"].to_numpy(dtype="f8"), section


def draw_track(draw: ImageDraw.ImageDraw, rows: pd.DataFrame, box: tuple[int, int, int, int], component: str) -> None:
    depth = rows["depth_m"].to_numpy(dtype="f8")
    lon = np.unwrap(np.deg2rad(rows["center_lon"].to_numpy(dtype="f8")))
    lat = rows["center_lat"].to_numpy(dtype="f8")
    ref_lat = lat[0]
    if component == "east":
        values = (lon - lon[0]) * EARTH_KM * np.cos(np.deg2rad(ref_lat))
        label = "east offset from surface (km)"
    else:
        values = np.deg2rad(lat - lat[0]) * EARTH_KM
        label = "north offset from surface (km)"
    draw_box(draw, box, label)
    px, py = xy_to_box(values, -depth, box)
    points = [(int(x), int(y)) for x, y in zip(px, py) if np.isfinite(x) and np.isfinite(y)]
    if len(points) >= 2:
        draw.line(points, fill=(22, 79, 135), width=3)
    for point in points:
        draw.ellipse((point[0] - 3, point[1] - 3, point[0] + 3, point[1] + 3), fill=(22, 79, 135))
    draw.text((box[0] + 8, box[3] - 26), "depth increases downward", fill=(80, 80, 80), font=font(15))


def draw_section(image: Image.Image, draw: ImageDraw.ImageDraw, offsets: np.ndarray, depth: np.ndarray, section: np.ndarray, box: tuple[int, int, int, int]) -> None:
    draw_box(draw, box, "axis-curved signed velocity: normal section")
    vmax = float(np.nanpercentile(np.abs(section), 97)) if np.isfinite(section).any() else 1.0
    rgb = signed_rgb(section, vmax)
    raster = Image.fromarray(rgb, mode="RGB").resize((box[2] - box[0] - 40, box[3] - box[1] - 64), Image.Resampling.BILINEAR)
    image.paste(raster, (box[0] + 20, box[1] + 34))
    cx = box[0] + 20 + int((0.0 - offsets.min()) / (offsets.max() - offsets.min()) * (box[2] - box[0] - 40))
    draw.line((cx, box[1] + 34, cx, box[3] - 30), fill=(30, 30, 30), width=2)
    draw.text((box[0] + 22, box[3] - 26), "-1.2 R       center       +1.2 R", fill=(35, 35, 35), font=font(16))
    draw.text((box[2] - 170, box[1] + 8), f"+/- {vmax:.3f} m/s", fill=(35, 35, 35), font=font(15))


def render_object(object_id: str, structures: pd.DataFrame, layers: pd.DataFrame, ds: Dataset, lon0: float, dlon: float, lat0: float, dlat: float, velocity_cache: dict[int, tuple[np.ndarray, np.ndarray]], target: Path, rank: int, region: str) -> dict[str, object]:
    rows = structures.loc[structures["hua_object_id"].astype(str).eq(object_id)].copy()
    qrows = layers.loc[layers["hua_object_id"].astype(str).eq(object_id)].copy()
    rows = rows.merge(qrows[["depth_index", "axis_tx_east", "axis_ty_north", "section_bipolar_0p6r", "section_bipolar_1p0r"]], on="depth_index", how="inner")
    rows = rows.sort_values("depth_index")
    offsets, depth, section = sample_section(rows, ds, lon0, dlon, lat0, dlat, velocity_cache)
    image = Image.new("RGB", (1800, 1050), "white")
    draw = ImageDraw.Draw(image)
    draw.text((42, 25), f"Deep Hua continuous-center family panel | {region} | rank {rank:02d} | {object_id}", fill=(15, 25, 40), font=font(30, True))
    min06 = float(rows["section_bipolar_0p6r"].mean())
    min10 = float(rows["section_bipolar_1p0r"].mean())
    draw.text((44, 72), f"accepted layers={len(rows)}; max depth={float(rows.depth_m.max()):.0f} m; bilateral opposite-sign fraction: 0.6R={min06:.3f}, 1.0R={min10:.3f}", fill=(55, 65, 80), font=font(20))
    draw_track(draw, rows, (45, 125, 485, 915), "east")
    draw_track(draw, rows, (520, 125, 960, 915), "north")
    draw_section(image, draw, offsets, depth, section, (995, 125, 1755, 915))
    draw.text((45, 960), "Blue/red section pixels are signed along-axis velocity. A physically coherent section changes sign across the center line; this panel is diagnostic only and did not alter acceptance.", fill=(55, 65, 80), font=font(18))
    target.mkdir(parents=True, exist_ok=True)
    png = target / "curve_section_bilateral_family.png"
    image.save(png)
    image.save(target / "curve_section_bilateral_family.pdf", "PDF", resolution=150)
    return {"object_id": object_id, "region": region, "rank": rank, "png": str(png), "layers": int(len(rows)), "max_depth_m": float(rows.depth_m.max()), "bipolar_fraction_0p6r": min06, "bipolar_fraction_1p0r": min10}


def main() -> None:
    args = parse_args()
    structures = read_structures(args.vertical_root)
    layers = pd.read_csv(args.quality_layers)
    quality = pd.read_csv(args.quality_summary)
    first = structures.sort_values("depth_index").groupby("hua_object_id", as_index=False).first()[["hua_object_id", "center_lon", "center_lat"]]
    quality = quality.merge(first, on="hua_object_id", how="inner")
    quality["score"] = np.minimum(quality["bipolar_fraction_0p6r"], quality["bipolar_fraction_1p0r"])
    quality = quality.loc[quality["score"].ge(args.min_bipolar_fraction)].copy()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected: list[dict[str, object]] = []
    with Dataset(args.velocity_file) as ds:
        lon = np.asarray(ds.variables["longitude"][:], dtype="f8")
        lat = np.asarray(ds.variables["latitude"][:], dtype="f8")
        velocity_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for region, box in REGIONS.items():
            part = quality.loc[[in_region(float(row.center_lon), float(row.center_lat), box) for row in quality.itertuples(index=False)]].copy()
            part = part.sort_values(["accepted_below_surface_layers", "max_depth_m"], ascending=False).head(args.per_region)
            for rank, row in enumerate(part.itertuples(index=False), start=1):
                target = args.output_dir / f"{region}_rank_{rank:02d}_{row.hua_object_id}"
                selected.append(render_object(str(row.hua_object_id), structures, layers, ds, float(lon[0]), float(lon[1] - lon[0]), float(lat[0]), float(lat[1] - lat[0]), velocity_cache, target, rank, region))
    pd.DataFrame(selected).to_csv(args.output_dir / "selected_family_panels.csv", index=False)
    (args.output_dir / "selected_family_panels.json").write_text(json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"rendered": len(selected), "output_dir": str(args.output_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
