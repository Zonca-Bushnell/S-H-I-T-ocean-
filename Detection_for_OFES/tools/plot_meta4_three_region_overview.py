from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from netCDF4 import Dataset, date2num
from PIL import Image, ImageDraw, ImageFont


DEFAULT_META_ROOT = Path(r"F:\Eddy\Eddy\META4.0_DT_allsat")
DEFAULT_OUTPUT_ROOT = Path(
    r"E:\DATA\01_Eddy_correspond\02_OFES\origin_ssh_effective_contour_primary_meso10d_50_500km_smoke_19910101\figures\meta4_three_region_overview"
)


REGIONS = {
    "global": (0.0, 360.0, -76.0, 76.0),
    "kuroshio": (120.0, 145.0, 20.0, 35.0),
    "north_pacific_interior_dense": (170.0, 210.0, 20.0, 40.0),
}


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    day = datetime.strptime(args.day, "%Y-%m-%d")
    observations = []
    for polarity in ["cyclonic", "anticyclonic"]:
        observations.extend(read_meta_day(Path(args.meta_root), polarity, day))
    products = []
    for name, bbox in REGIONS.items():
        products.append(draw_region(observations, name, bbox, day, output_root))
    manifest = {
        "day": args.day,
        "meta_root": str(args.meta_root),
        "output_root": str(output_root),
        "note": "META4.0 effective-contour overview for qualitative comparison with OFES ssh_effective_contour_primary.",
        "products": products,
    }
    manifest_path = output_root / f"meta4_three_region_overview_manifest_{day:%Y%m%d}.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path), "products": products}, ensure_ascii=False, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot META4.0 effective-contour objects for the same three OFES overview regions.")
    parser.add_argument("--meta-root", type=Path, default=DEFAULT_META_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--day", default="1993-01-01")
    return parser.parse_args()


def read_meta_day(meta_root: Path, polarity: str, day: datetime) -> list[dict[str, object]]:
    path = meta_root / f"META4_DT_allsat_{polarity}_19930101_20230908.nc"
    if not path.exists():
        raise FileNotFoundError(path)
    rows: list[dict[str, object]] = []
    with Dataset(path) as ds:
        time_var = ds.variables["time"]
        target = date2num(day, units=time_var.units, calendar=getattr(time_var, "calendar", "standard"))
        indices = find_time_indices(time_var, target)
        if indices.size == 0:
            return rows
        lon = to_360(np.asarray(ds.variables["longitude"][indices], dtype="f8"))
        lat = np.asarray(ds.variables["latitude"][indices], dtype="f8")
        radius_m = np.asarray(ds.variables["effective_radius"][indices], dtype="f8")
        contour_lon = to_360(np.asarray(ds.variables["effective_contour_longitude"][indices, :], dtype="f8"))
        contour_lat = np.asarray(ds.variables["effective_contour_latitude"][indices, :], dtype="f8")
        for n in range(indices.size):
            rows.append(
                {
                    "polarity": polarity,
                    "lon": float(lon[n]),
                    "lat": float(lat[n]),
                    "effective_radius_m": float(radius_m[n]),
                    "contour_lon": contour_lon[n],
                    "contour_lat": contour_lat[n],
                }
            )
    return rows


def find_time_indices(time_var, target: float) -> np.ndarray:
    nobs = int(time_var.shape[0])
    start = lower_bound_time(time_var, target - 0.5, nobs)
    stop = lower_bound_time(time_var, target + 0.5, nobs)
    if stop <= start:
        return np.asarray([], dtype=int)
    values = np.asarray(time_var[start:stop], dtype="f8")
    local = np.flatnonzero(np.isclose(values, target))
    return (local + start).astype(int)


def lower_bound_time(time_var, value: float, nobs: int) -> int:
    lo = 0
    hi = nobs
    while lo < hi:
        mid = (lo + hi) // 2
        current = float(np.asarray(time_var[mid], dtype="f8"))
        if current < value:
            lo = mid + 1
        else:
            hi = mid
    return lo


def to_360(lon: np.ndarray) -> np.ndarray:
    return np.mod(lon, 360.0)


def draw_region(observations: list[dict[str, object]], name: str, bbox: tuple[float, float, float, float], day: datetime, output_root: Path) -> dict[str, object]:
    lon_min, lon_max, lat_min, lat_max = bbox
    shown = [row for row in observations if lon_min <= float(row["lon"]) <= lon_max and lat_min <= float(row["lat"]) <= lat_max]
    width, height = (2200, 1200) if name == "global" else (1840, 1160)
    box = (85, 110, width - 95, height - 115)
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((50, 30), f"META4.0 DT allsat effective contours | {name} | {day:%Y-%m-%d}", fill=(18, 24, 38), font=font(34))
    draw.text((50, 70), "Qualitative catalog comparison only: centers and effective-contour samples; no SSH background field in META files.", fill=(75, 85, 100), font=font(17))
    draw.rectangle(box, fill=(244, 248, 252), outline=(20, 24, 32), width=2)
    draw_grid(draw, box, bbox, name)
    counts = {"cyclonic": 0, "anticyclonic": 0}
    for row in shown:
        counts[str(row["polarity"])] += 1
        color = (32, 99, 220) if row["polarity"] == "cyclonic" else (215, 43, 43)
        draw_contour(draw, row, box, bbox, color)
        x, y = project(float(row["lon"]), float(row["lat"]), box, bbox)
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=color, outline="white", width=1)
    draw_legend(draw, box, counts)
    png_path = output_root / f"meta4_effective_contours_{name}_{day:%Y%m%d}.png"
    pdf_path = png_path.with_suffix(".pdf")
    image.save(png_path)
    image.save(pdf_path, "PDF", resolution=180.0)
    return {
        "region": name,
        "bbox": {"lon_min": lon_min, "lon_max": lon_max, "lat_min": lat_min, "lat_max": lat_max},
        "surface_objects": int(len(shown)),
        "cyclonic": int(counts["cyclonic"]),
        "anticyclonic": int(counts["anticyclonic"]),
        "png": str(png_path),
        "pdf": str(pdf_path),
    }


def draw_contour(draw: ImageDraw.ImageDraw, row: dict[str, object], box: tuple[int, int, int, int], bbox: tuple[float, float, float, float], color: tuple[int, int, int]) -> None:
    lons = np.asarray(row["contour_lon"], dtype="f8")
    lats = np.asarray(row["contour_lat"], dtype="f8")
    finite = np.isfinite(lons) & np.isfinite(lats)
    pts: list[tuple[float, float]] = []
    for lon, lat in zip(lons[finite], lats[finite], strict=False):
        if bbox[0] - 5.0 <= lon <= bbox[1] + 5.0 and bbox[2] - 5.0 <= lat <= bbox[3] + 5.0:
            pts.append(project(float(lon), float(lat), box, bbox))
    if len(pts) >= 2:
        draw.line(pts + [pts[0]], fill=color, width=2)


def draw_grid(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], bbox: tuple[float, float, float, float], name: str) -> None:
    lon_min, lon_max, lat_min, lat_max = bbox
    lon_step = 60 if name == "global" else 5
    lat_step = 30 if name == "global" else 5
    for lon in np.arange(np.ceil(lon_min / lon_step) * lon_step, lon_max + 0.1, lon_step):
        x, _ = project(float(lon), lat_min, box, bbox)
        draw.line((x, box[1], x, box[3]), fill=(170, 180, 195), width=1)
        draw.text((x - 18, box[3] + 8), lon_label(float(lon)), fill=(35, 40, 50), font=font(15))
    for lat in np.arange(np.ceil(lat_min / lat_step) * lat_step, lat_max + 0.1, lat_step):
        _, y = project(lon_min, float(lat), box, bbox)
        draw.line((box[0], y, box[2], y), fill=(170, 180, 195), width=1)
        draw.text((box[0] - 64, y - 9), lat_label(float(lat)), fill=(35, 40, 50), font=font(15))


def draw_legend(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], counts: dict[str, int]) -> None:
    y = box[3] + 38
    draw.ellipse((box[0] + 8, y - 8, box[0] + 24, y + 8), fill=(32, 99, 220))
    draw.text((box[0] + 32, y - 11), f"cyclonic={counts['cyclonic']}", fill=(20, 24, 32), font=font(18))
    draw.ellipse((box[0] + 188, y - 8, box[0] + 204, y + 8), fill=(215, 43, 43))
    draw.text((box[0] + 212, y - 11), f"anticyclonic={counts['anticyclonic']}", fill=(20, 24, 32), font=font(18))
    draw.text((box[0] + 450, y - 11), f"objects={counts['cyclonic'] + counts['anticyclonic']}", fill=(20, 24, 32), font=font(18))


def project(lon: float, lat: float, box: tuple[int, int, int, int], bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    lon_min, lon_max, lat_min, lat_max = bbox
    x = box[0] + (lon - lon_min) / (lon_max - lon_min) * (box[2] - box[0])
    y = box[3] - (lat - lat_min) / (lat_max - lat_min) * (box[3] - box[1])
    return x, y


def lon_label(value: float) -> str:
    return f"{int(round(value))}E" if value <= 180 else f"{int(round(360 - value))}W"


def lat_label(value: float) -> str:
    if abs(value) < 1.0e-6:
        return "0"
    return f"{abs(int(round(value)))}{'N' if value > 0 else 'S'}"


def font(size: int) -> ImageFont.ImageFont:
    for path in [
        r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\calibri.ttf",
        r"C:\Windows\Fonts\simhei.ttf",
    ]:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    return ImageFont.load_default()


if __name__ == "__main__":
    main()
