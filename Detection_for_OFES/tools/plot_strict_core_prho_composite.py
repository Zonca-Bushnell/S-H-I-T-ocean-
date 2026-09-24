"""Plot the saved 19-day NH cyclonic strict-core density composite.

The source composite is read-only: this renderer does not resample OFES or
change the 435 object-day selection.  The second row is a local outer-ring
reference anomaly, not a climatological density anomaly.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import contourpy
import numpy as np
from PIL import Image, ImageDraw, ImageFont


DEFAULT_ROOT = Path(
    r"E:\DATA\01_Eddy_correspond\02_OFES\Fromthebeginning\05_TEMP"
    r"\nh_cyclonic_strict_core_native_w_pointwise_unrotated_eta_19910101_19910119"
    r"\paper_pointwise_no_rotation\NH_cyclone_strict_core_19d"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_ROOT / "isopycnal_composite.npz")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_ROOT / "figures")
    return parser.parse_args()


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    names = ("segoeuib.ttf", "arialbd.ttf") if bold else ("segoeui.ttf", "arial.ttf")
    for name in names:
        path = Path(r"C:\Windows\Fonts") / name
        if path.exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def coolwarm_rgb(values: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """A compact, dependency-free approximation of the ``coolwarm`` map."""
    t = np.nan_to_num(np.clip((values - lo) / max(hi - lo, 1e-12), 0, 1), nan=0.5)
    cool = np.asarray((59, 76, 192), dtype="f4")
    neutral = np.asarray((221, 220, 220), dtype="f4")
    warm = np.asarray((180, 4, 38), dtype="f4")
    lower = (2.0 * t[..., None]) * neutral + (1.0 - 2.0 * t[..., None]) * cool
    upper = (2.0 * (t[..., None] - 0.5)) * warm + (2.0 * (1.0 - t[..., None])) * neutral
    out = np.where(t[..., None] <= 0.5, lower, upper).astype(np.uint8)
    out[~np.isfinite(values)] = (58, 62, 70)
    return out


def paste(image: Image.Image, values: np.ndarray, box: tuple[int, int, int, int], rgb: np.ndarray) -> None:
    tile = Image.fromarray(rgb, mode="RGB").resize((box[2] - box[0], box[3] - box[1]), Image.Resampling.BILINEAR)
    image.paste(tile, (box[0], box[1]))


def draw_colorbar(draw: ImageDraw.ImageDraw, image: Image.Image, box: tuple[int, int, int, int], lo: float, hi: float, title: str) -> None:
    values = np.linspace(hi, lo, 256, dtype="f4")[:, None]
    strip = Image.fromarray(coolwarm_rgb(values, lo, hi), mode="RGB").resize((box[2] - box[0], box[3] - box[1]))
    image.paste(strip, (box[0], box[1]))
    draw.rectangle(box, outline=(30, 35, 45), width=1)
    draw.text((box[0] - 8, box[1] - 24), title, fill=(20, 30, 45), font=font(12, True))
    draw.text((box[2] + 6, box[1] - 5), f"{hi:.3f}", fill=(45, 50, 60), font=font(11))
    draw.text((box[2] + 6, (box[1] + box[3]) // 2 - 5), f"{(lo + hi) / 2:.3f}", fill=(45, 50, 60), font=font(11))
    draw.text((box[2] + 6, box[3] - 12), f"{lo:.3f}", fill=(45, 50, 60), font=font(11))


def finite_range(values: np.ndarray, signed: bool) -> tuple[float, float, str]:
    finite = values[np.isfinite(values)]
    if signed:
        bound = float(np.percentile(np.abs(finite), 98)) if finite.size else 1.0
        return -bound, bound, f"+/- {bound:.3f} kg m-3"
    lo, hi = (float(np.percentile(finite, 2)), float(np.percentile(finite, 98))) if finite.size else (0.0, 1.0)
    return lo, hi, f"{lo:.3f} to {hi:.3f} kg m-3"


def levels(values: np.ndarray, signed: bool) -> np.ndarray:
    lo, hi, _ = finite_range(values, signed)
    return np.linspace(lo, hi, 11)[1:-1]


def draw_contours(draw: ImageDraw.ImageDraw, field: np.ndarray, xs: np.ndarray, ys: np.ndarray, box: tuple[int, int, int, int], signed: bool, depth_down: bool) -> None:
    try:
        contour = contourpy.contour_generator(x=xs, y=ys, z=np.ma.masked_invalid(field), name="serial")
        x0, y0, x1, y1 = box
        for level in levels(field, signed):
            for segment in contour.lines(float(level)):
                if len(segment) < 2:
                    continue
                points = []
                for x, y in segment:
                    px = x0 + (x - xs[0]) / (xs[-1] - xs[0]) * (x1 - x0)
                    fy = (y - ys[0]) / (ys[-1] - ys[0])
                    py = y0 + fy * (y1 - y0) if depth_down else y1 - fy * (y1 - y0)
                    points.append((px, py))
                draw.line(points, fill=(20, 20, 25), width=1)
    except (RuntimeError, ValueError):
        return


def circle(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], radius: float, dashed: bool = False) -> None:
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    # Each map spans -2R to +2R, so a physical radius of 1R occupies one
    # quarter of the panel width/height.
    rx, ry = radius / 4 * (x1 - x0), radius / 4 * (y1 - y0)
    if not dashed:
        draw.ellipse((cx - rx, cy - ry, cx + rx, cy + ry), outline=(10, 10, 10), width=2)
        return
    for start in range(0, 360, 18):
        draw.arc((cx - rx, cy - ry, cx + rx, cy + ry), start=start, end=start + 9, fill=(25, 25, 25), width=1)


def draw_map(draw: ImageDraw.ImageDraw, image: Image.Image, field: np.ndarray, support: np.ndarray, x: np.ndarray, y: np.ndarray, box: tuple[int, int, int, int], label: str, signed: bool) -> str:
    lo, hi, scale = finite_range(field, signed)
    paste(image, field, box, coolwarm_rgb(field, lo, hi))
    draw_contours(draw, field, x, y, box, signed, depth_down=False)
    circle(draw, box, 1.0)
    circle(draw, box, 1.5, dashed=True)
    circle(draw, box, 2.0, dashed=True)
    draw.rectangle(box, outline=(25, 30, 40), width=2)
    x0, y0, x1, y1 = box
    draw.text((x0, y0 - 25), label, fill=(20, 30, 45), font=font(14, True))
    draw.text((x0 + 4, y0 + 4), f"{scale}; center N={int(support[len(y)//2, len(x)//2])}", fill=(12, 15, 20), font=font(11, True))
    draw.text((x0, y1 + 4), "x/R, y/R; solid 1R, dashed 1.5R/2R", fill=(75, 78, 85), font=font(11))
    return scale


def draw_section(draw: ImageDraw.ImageDraw, image: Image.Image, field: np.ndarray, support: np.ndarray, depth: np.ndarray, x: np.ndarray, box: tuple[int, int, int, int], label: str, signed: bool) -> str:
    section = field[:, field.shape[1] // 2, :]
    support_section = support[:, support.shape[1] // 2, :]
    lo, hi, scale = finite_range(section, signed)
    paste(image, section, box, coolwarm_rgb(section, lo, hi))
    draw_contours(draw, section, x, depth, box, signed, depth_down=True)
    draw.rectangle(box, outline=(25, 30, 40), width=2)
    x0, y0, x1, y1 = box
    draw.line(((x0 + x1) / 2, y0, (x0 + x1) / 2, y1), fill=(20, 20, 20), width=1)
    draw.text((x0, y0 - 25), label, fill=(20, 30, 45), font=font(14, True))
    draw.text((x0 + 4, y0 + 4), scale, fill=(12, 15, 20), font=font(11, True))
    draw.text((x0, y1 + 4), f"x/R; depth down; center support {int(np.nanmin(support_section[:, support_section.shape[1]//2]))}-{int(np.nanmax(support_section[:, support_section.shape[1]//2]))}", fill=(75, 78, 85), font=font(11))
    return scale


def nearest(depth: np.ndarray, target: float) -> int:
    return int(np.nanargmin(np.abs(depth - target)))


def main() -> None:
    args = parse_args()
    data = np.load(args.input)
    depth = np.asarray(data["depth_m"], dtype="f8")
    x = np.asarray(data["x_over_r"], dtype="f8")
    y = np.asarray(data["y_over_r"], dtype="f8")
    prho = np.asarray(data["composite_prho"], dtype="f4")
    anomaly = np.asarray(data["rho_anom_ring_kg_m3"], dtype="f4")
    support = np.asarray(data["support_prho_objects"], dtype="u2")
    depth_limit = depth <= 2000.0
    depth_section = depth[depth_limit]
    prho_section, anomaly_section, support_section = prho[depth_limit], anomaly[depth_limit], support[depth_limit]
    indices = [nearest(depth, target) for target in (2.5, 300.0, 500.0, 1000.0)]
    fields = [("Native composite prho", prho, prho_section, False), ("prho minus 1.5-2R outer-ring reference", anomaly, anomaly_section, True)]
    width, row_height = 2320, 410
    image = Image.new("RGB", (width, 135 + len(fields) * row_height + 55), "white")
    draw = ImageDraw.Draw(image)
    draw.text((42, 24), "NH cyclonic strict-core composite density | Jan1-Jan19 | 435 object-days", fill=(15, 25, 42), font=font(27, True))
    draw.text((44, 64), "Unrotated direct pointwise mean on [-2R, 2R], delta=0.04R. Top: native OFES prho. Bottom: local outer-ring density anomaly (not a climatological anomaly).", fill=(65, 70, 82), font=font(15))
    left, section_width, map_size, gap = 65, 400, 330, 45
    for row, (label, full_field, section_field, signed) in enumerate(fields):
        top = 125 + row * row_height
        lo, hi, _ = finite_range(full_field, signed)
        section_box = (left, top, left + section_width, top + map_size)
        draw_section(draw, image, section_field, support_section, depth_section, x, section_box, f"{label}: x-depth section", signed)
        for column, layer in enumerate(indices):
            x0 = left + section_width + gap + column * (map_size + gap)
            box = (x0, top, x0 + map_size, top + map_size)
            draw_map(draw, image, full_field[layer], support[layer], x, y, box, f"{label}: {depth[layer]:.1f} m", signed)
        draw_colorbar(draw, image, (2035, top, 2065, top + map_size), lo, hi, "prho (kg m-3)")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    png = args.output_dir / "composite_prho_and_outer_ring_anomaly.png"
    pdf = png.with_suffix(".pdf")
    image.save(png)
    image.save(pdf, "PDF", resolution=160)
    manifest = {
        "input": str(args.input), "object_days": 435, "date_range": "1991-01-01/1991-01-19",
        "selection": "NH cyclonic strict-core; min(f_0.6R, f_1.0R) >= 0.70",
        "composite_method": "unrotated pointwise finite-cell mean; [-2R,2R], delta=0.04R",
        "native_density": "composite_prho from the existing raw OFES sampling chain; no temporal/spatial filtering",
        "anomaly": "composite_prho minus same-depth median in 1.5-2R outer ring; not a monthly or annual climatology anomaly",
        "depths_m": [float(depth[index]) for index in indices], "png": str(png), "pdf": str(pdf),
    }
    manifest_path = args.output_dir / "composite_prho_and_outer_ring_anomaly_manifest.json"
    partial = manifest_path.with_suffix(".json.part")
    partial.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(partial, manifest_path)
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
