from __future__ import annotations

import json
import math
import re
import subprocess
import argparse
import array
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


RESULT_ROOT = Path(r"E:\DATA\01_Eddy_correspond\02_OFES\origin_streamline_cpu_jan01_jan19_life1")
OUT_DIR = RESULT_ROOT / "figures" / "latest_velocity_streamline_surface_and_family_with_edges"
BASE_PNG = OUT_DIR / "ofes_velocity_streamline_kuroshio_lavd_style_19910110.png"
META_ROOT = Path(r"F:\Eddy\Eddy\META4.0_DT_allsat")
H5DUMP = Path(r"D:\Util\lever\02_miniforge\envs\OFES_detection\Library\bin\h5dump.exe")

META_FILES = {
    "cyclonic": META_ROOT / "META4_DT_allsat_cyclonic_19930101_20230908.nc",
    "anticyclonic": META_ROOT / "META4_DT_allsat_anticyclonic_19930101_20230908.nc",
}
META_MAT_FILES = {
    "cyclonic": META_ROOT / "META4_DT_allsat_cyclonic_19930101_20230908_track_all_cols_opt_lifeGT56_erMeanGT50km_ampMeanGT2cm.mat",
    "anticyclonic": META_ROOT / "META4_DT_allsat_anticyclonic_19930101_20230908_track_all_cols_opt_lifeGT56_erMeanGT50km_ampMeanGT2cm.mat",
}

LON_MIN, LON_MAX, LAT_MIN, LAT_MAX = 120.0, 145.0, 20.0, 35.0
MAP_BOX = (85, 95, 1665, 1045)


def font(size: int) -> ImageFont.ImageFont:
    for path in [r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\arial.ttf"]:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            pass
    return ImageFont.load_default()


def px(lon: float) -> int:
    x0, _y0, x1, _y1 = MAP_BOX
    return x0 + int(round((lon - LON_MIN) / (LON_MAX - LON_MIN) * (x1 - x0)))


def py(lat: float) -> int:
    _x0, y0, _x1, y1 = MAP_BOX
    return y1 - int(round((lat - LAT_MIN) / (LAT_MAX - LAT_MIN) * (y1 - y0)))


def meta_raw_time_seconds(day: str) -> int:
    dt = datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    ref = datetime(1950, 1, 1, tzinfo=timezone.utc)
    return int((dt - ref).total_seconds())


def h5dump_dataset(path: Path, dataset: str, start: str, count: str) -> str:
    return subprocess.check_output(
        [str(H5DUMP), "-d", f"/{dataset}", "-s", start, "-c", count, str(path)],
        text=True,
        errors="ignore",
    )


def h5dump_binary(path: Path, dataset: str, start: str, count: str, out_path: Path) -> None:
    subprocess.check_call(
        [str(H5DUMP), "-d", f"/{dataset}", "-s", start, "-c", count, "-b", "LE", "-o", str(out_path), str(path)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def read_binary_array(path: Path, typecode: str) -> array.array:
    data = array.array(typecode)
    with path.open("rb") as handle:
        data.frombytes(handle.read())
    if data.itemsize > 1 and data.typecode not in ("b", "B"):
        # h5dump writes little-endian. array uses native endian, so byteswap on big-endian hosts.
        import sys

        if sys.byteorder != "little":
            data.byteswap()
    return data


def mat_row_length(path: Path, dataset: str) -> int:
    text = subprocess.check_output([str(H5DUMP), "-H", "-d", f"/{dataset}", str(path)], text=True, errors="ignore")
    match = re.search(r"DATASPACE\s+SIMPLE\s+\{\s+\(\s*1\s*,\s*(\d+)\s*\)", text)
    if not match:
        raise ValueError(f"could not parse row length for {dataset} from {path.name}")
    return int(match.group(1))


def meta_day_number(day: str) -> int:
    dt = datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    ref = datetime(1950, 1, 1, tzinfo=timezone.utc)
    return (dt - ref).days


def read_meta_mat_day(path: Path, polarity: str, day: str, chunk: int = 500_000) -> list[dict[str, object]]:
    target_day = meta_day_number(day)
    n = mat_row_length(path, "final_time")
    records: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="meta4_h5dump_") as td:
        tmp = Path(td)
        for start in range(0, n, chunk):
            count = min(chunk, n - start)
            time_bin = tmp / "time.bin"
            h5dump_binary(path, "final_time", f"0,{start}", f"1,{count}", time_bin)
            times = read_binary_array(time_bin, "d")
            local_hits = [i for i, value in enumerate(times) if abs(float(value) - target_day) < 0.25]
            if not local_hits:
                continue
            lon_bin = tmp / "lon.bin"
            lat_bin = tmp / "lat.bin"
            rad_bin = tmp / "radius.bin"
            h5dump_binary(path, "final_lon", f"0,{start}", f"1,{count}", lon_bin)
            h5dump_binary(path, "final_lat", f"0,{start}", f"1,{count}", lat_bin)
            h5dump_binary(path, "final_radius", f"0,{start}", f"1,{count}", rad_bin)
            lons = read_binary_array(lon_bin, "f")
            lats = read_binary_array(lat_bin, "f")
            radii = read_binary_array(rad_bin, "d")
            for i in local_hits:
                lon0 = float(lons[i])
                lat0 = float(lats[i])
                if not (LON_MIN <= lon0 <= LON_MAX and LAT_MIN <= lat0 <= LAT_MAX):
                    continue
                records.append(
                    {
                        "polarity": polarity,
                        "date": day,
                        "obs_index": start + i,
                        "lon": lon0,
                        "lat": lat0,
                        "effective_radius_km": float(radii[i]) / 1000.0,
                        "source": "META4 filtered track table final_lon/final_lat/final_radius",
                    }
                )
    return records


def h5dump_scalar(path: Path, dataset: str, index: int) -> int:
    text = h5dump_dataset(path, dataset, str(index), "1")
    vals = parse_1d_values(text)
    if not vals:
        raise ValueError(f"could not read {dataset}[{index}] from {path.name}")
    return int(vals[0])


def subset_data_block(text: str) -> str:
    marker = "   SUBSET {"
    start = text.find(marker)
    if start < 0:
        start = 0
    data_start = text.find("      DATA {", start)
    if data_start < 0:
        return ""
    data_start = text.find("\n", data_start) + 1
    data_end = text.find("      }", data_start)
    return text[data_start:data_end] if data_end > data_start else text[data_start:]


def parse_1d_values(text: str) -> list[float]:
    block = subset_data_block(text)
    values: list[float] = []
    for match in re.finditer(r"\((\d+)\):\s*([^\n]+)", block):
        values.extend(float(x) for x in re.findall(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", match.group(2)))
    return values


def parse_2d_values(text: str, nrow: int, ncol: int) -> list[list[float]]:
    block = subset_data_block(text)
    rows: list[list[float]] = [[] for _ in range(nrow)]
    row0 = None
    for match in re.finditer(r"\((\d+),(\d+)\):\s*([^\n]+)", block):
        row = int(match.group(1))
        if row0 is None:
            row0 = row
        local_row = row - row0
        if 0 <= local_row < nrow:
            rows[local_row].extend(float(x) for x in re.findall(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", match.group(3)))
    return [row[:ncol] for row in rows]


def dataset_obs_size(path: Path) -> int:
    text = subprocess.check_output([str(H5DUMP), "-H", "-d", "/time", str(path)], text=True, errors="ignore")
    match = re.search(r"DATASPACE\s+SIMPLE\s+\{\s+\(\s*(\d+)\s*\)", text)
    if not match:
        raise ValueError(f"could not parse obs size from {path.name}")
    return int(match.group(1))


def lower_bound_time(path: Path, raw_time: int, nobs: int) -> int:
    lo, hi = 0, nobs
    while lo < hi:
        mid = (lo + hi) // 2
        value = h5dump_scalar(path, "time", mid)
        if value < raw_time:
            lo = mid + 1
        else:
            hi = mid
    return lo


def upper_bound_time(path: Path, raw_time: int, nobs: int) -> int:
    lo, hi = 0, nobs
    while lo < hi:
        mid = (lo + hi) // 2
        value = h5dump_scalar(path, "time", mid)
        if value <= raw_time:
            lo = mid + 1
        else:
            hi = mid
    return lo


def find_time_range(path: Path, raw_time: int) -> tuple[int, int]:
    nobs = dataset_obs_size(path)
    first = lower_bound_time(path, raw_time, nobs)
    last_exclusive = upper_bound_time(path, raw_time, nobs)
    if first >= last_exclusive:
        raise ValueError(f"{path.name} has no META observations for raw time {raw_time}")
    return first, last_exclusive - 1


def read_meta_day(path: Path, polarity: str, day: str) -> list[dict[str, object]]:
    raw_time = meta_raw_time_seconds(day)
    first, last = find_time_range(path, raw_time)
    count = last - first + 1
    lon = parse_1d_values(h5dump_dataset(path, "longitude", str(first), str(count)))
    lat = parse_1d_values(h5dump_dataset(path, "latitude", str(first), str(count)))
    radius_raw = parse_1d_values(h5dump_dataset(path, "effective_radius", str(first), str(count)))
    contour_lon_raw = parse_2d_values(h5dump_dataset(path, "effective_contour_longitude", f"{first},0", f"{count},30"), count, 30)
    contour_lat_raw = parse_2d_values(h5dump_dataset(path, "effective_contour_latitude", f"{first},0", f"{count},30"), count, 30)

    records: list[dict[str, object]] = []
    for i in range(count):
        lon0 = float(lon[i])
        lat0 = float(lat[i])
        if not (LON_MIN <= lon0 <= LON_MAX and LAT_MIN <= lat0 <= LAT_MAX):
            continue
        contour = []
        for lo_raw, la_raw in zip(contour_lon_raw[i], contour_lat_raw[i]):
            lo = float(lo_raw) * 0.01 + 180.0
            la = float(la_raw) * 0.01
            if LON_MIN - 2 <= lo <= LON_MAX + 2 and LAT_MIN - 2 <= la <= LAT_MAX + 2:
                contour.append((lo, la))
        records.append(
            {
                "polarity": polarity,
                "date": day,
                "obs_index": first + i,
                "lon": lon0,
                "lat": lat0,
                "effective_radius_km": float(radius_raw[i]) * 50.0 / 1000.0,
                "effective_contour": contour,
            }
        )
    return records


def draw_dashed_polyline(draw: ImageDraw.ImageDraw, points: list[tuple[int, int]], fill: tuple[int, int, int], width: int = 3) -> None:
    if len(points) < 2:
        return
    for p0, p1 in zip(points, points[1:] + points[:1]):
        x0, y0 = p0
        x1, y1 = p1
        dist = math.hypot(x1 - x0, y1 - y0)
        if dist <= 0:
            continue
        dash = 10
        gap = 7
        n = max(1, int(dist // (dash + gap)) + 1)
        for k in range(n):
            a0 = min(1.0, k * (dash + gap) / dist)
            a1 = min(1.0, (k * (dash + gap) + dash) / dist)
            if a0 >= 1.0:
                break
            draw.line(
                (
                    int(round(x0 + (x1 - x0) * a0)),
                    int(round(y0 + (y1 - y0) * a0)),
                    int(round(x0 + (x1 - x0) * a1)),
                    int(round(y0 + (y1 - y0) * a1)),
                ),
                fill=fill,
                width=width,
            )


def draw_meta_overlay(ofes_day: str = "1991-01-10", meta_day: str = "1993-01-10") -> dict[str, object]:
    records: list[dict[str, object]] = []
    skipped: dict[str, str] = {}
    for polarity, path in META_MAT_FILES.items():
        try:
            records.extend(read_meta_mat_day(path, polarity, meta_day))
        except ValueError as exc:
            skipped[polarity] = str(exc)

    base = Image.open(BASE_PNG).convert("RGB")
    draw = ImageDraw.Draw(base)
    colors = {
        "cyclonic": (0, 255, 255),
        "anticyclonic": (255, 0, 255),
    }
    for rec in records:
        color = colors[str(rec["polarity"])]
        cx = px(float(rec["lon"]))
        cy = py(float(rec["lat"]))
        radius_km = float(rec["effective_radius_km"])
        deg_lat = radius_km / 111.2
        deg_lon = deg_lat / max(math.cos(math.radians(float(rec["lat"]))), 0.2)
        rx = abs(px(float(rec["lon"]) + deg_lon) - cx)
        ry = abs(py(float(rec["lat"]) + deg_lat) - cy)
        draw.ellipse((cx - rx, cy - ry, cx + rx, cy + ry), outline=color, width=4)
        draw.line((cx - 8, cy, cx + 8, cy), fill=color, width=3)
        draw.line((cx, cy - 8, cx, cy + 8), fill=color, width=3)
        draw.ellipse((cx - 4, cy - 4, cx + 4, cy + 4), outline=(255, 255, 255), width=2)

    # White strip and legend so the comparison date is not confused with OFES date.
    draw.rectangle((50, 1088, 1380, 1148), fill=(255, 255, 255))
    draw.text((58, 1092), f"META4.0 overlay: {meta_day} effective contours projected on OFES {ofes_day} result", fill=(18, 24, 38), font=font(18))
    draw.line((610, 1118, 660, 1118), fill=colors["cyclonic"], width=4)
    draw.text((670, 1105), "META cyclonic radius circle", fill=(18, 24, 38), font=font(16))
    draw.line((895, 1118, 945, 1118), fill=colors["anticyclonic"], width=4)
    draw.text((955, 1105), "META anticyclonic radius circle", fill=(18, 24, 38), font=font(16))
    draw.text((1260, 1105), f"META objects in box: {len(records)}", fill=(18, 24, 38), font=font(16))

    out_png = OUT_DIR / f"ofes_velocity_streamline_kuroshio_with_META4_overlay_META{meta_day.replace('-', '')}_OFES{ofes_day.replace('-', '')}.png"
    out_pdf = out_png.with_suffix(".pdf")
    base.save(out_png)
    base.save(out_pdf, "PDF", resolution=180.0)
    manifest = {
        "ofes_day": ofes_day,
        "meta_day": meta_day,
        "note": "META4.0_DT_allsat starts at 1993-01-01, so exact OFES 1991-01-10 same-day overlay is unavailable from this META4.0 source.",
        "region": {"lon_min": LON_MIN, "lon_max": LON_MAX, "lat_min": LAT_MIN, "lat_max": LAT_MAX},
        "meta_object_count": len(records),
        "skipped_polarities": skipped,
        "meta_counts_by_polarity": {
            "cyclonic": sum(1 for rec in records if rec["polarity"] == "cyclonic"),
            "anticyclonic": sum(1 for rec in records if rec["polarity"] == "anticyclonic"),
        },
        "png": str(out_png),
        "pdf": str(out_pdf),
        "records": records,
    }
    (OUT_DIR / f"meta4_overlay_manifest_META{meta_day.replace('-', '')}_OFES{ofes_day.replace('-', '')}.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Overlay META4.0 effective contours on the OFES Kuroshio velocity-streamline map.")
    parser.add_argument("--ofes-day", default="1991-01-10")
    parser.add_argument("--meta-day", default="2019-01-10")
    args = parser.parse_args()
    print(json.dumps(draw_meta_overlay(ofes_day=args.ofes_day, meta_day=args.meta_day), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
