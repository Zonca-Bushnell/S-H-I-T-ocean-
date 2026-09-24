"""Render a contour-free strict-core composite of native density and W."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--selected-object-days", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--profile-label", required=True)
    parser.add_argument("--polarity", choices=("cyclonic", "anticyclonic"), required=True)
    return parser.parse_args()


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    names = ("segoeuib.ttf", "arialbd.ttf") if bold else ("segoeui.ttf", "arial.ttf")
    for name in names:
        path = Path(r"C:\Windows\Fonts") / name
        if path.exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def coolwarm(values: np.ndarray, lo: float, hi: float) -> np.ndarray:
    normalized = np.nan_to_num(np.clip((values - lo) / max(hi - lo, 1.0e-12), 0.0, 1.0), nan=0.5)
    cool = np.asarray((59, 76, 192), dtype="f4")
    neutral = np.asarray((221, 220, 220), dtype="f4")
    warm = np.asarray((180, 4, 38), dtype="f4")
    lower = 2.0 * normalized[..., None] * neutral + (1.0 - 2.0 * normalized[..., None]) * cool
    upper = 2.0 * (normalized[..., None] - 0.5) * warm + 2.0 * (1.0 - normalized[..., None]) * neutral
    rgb = np.where(normalized[..., None] <= 0.5, lower, upper).astype("u1")
    rgb[~np.isfinite(values)] = (55, 60, 68)
    return rgb


def paste(image: Image.Image, values: np.ndarray, lo: float, hi: float, box: tuple[int, int, int, int]) -> None:
    tile = Image.fromarray(coolwarm(values, lo, hi), mode="RGB")
    image.paste(tile.resize((box[2] - box[0], box[3] - box[1]), Image.Resampling.BILINEAR), box[:2])


def finite_limits(values: np.ndarray, signed: bool) -> tuple[float, float]:
    finite = np.asarray(values, dtype="f8")
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        return (-1.0, 1.0) if signed else (0.0, 1.0)
    if signed:
        bound = max(float(np.percentile(np.abs(finite), 98)), 1.0e-12)
        return -bound, bound
    return float(np.percentile(finite, 2)), float(np.percentile(finite, 98))


def draw_colorbar(draw: ImageDraw.ImageDraw, image: Image.Image, box: tuple[int, int, int, int], lo: float, hi: float, unit: str) -> None:
    values = np.linspace(hi, lo, 256, dtype="f4")[:, None]
    image.paste(Image.fromarray(coolwarm(values, lo, hi), mode="RGB").resize((box[2] - box[0], box[3] - box[1])), box[:2])
    draw.rectangle(box, outline=(30, 35, 45), width=1)
    label = font(11)
    draw.text((box[0] - 5, box[1] - 21), unit, fill=(20, 30, 45), font=font(11, True))
    draw.text((box[2] + 5, box[1] - 5), f"{hi:.3g}", fill=(45, 50, 60), font=label)
    draw.text((box[2] + 5, (box[1] + box[3]) // 2 - 5), f"{(lo + hi) / 2:.3g}", fill=(45, 50, 60), font=label)
    draw.text((box[2] + 5, box[3] - 12), f"{lo:.3g}", fill=(45, 50, 60), font=label)


def draw_ring(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int]) -> None:
    width, height = box[2] - box[0], box[3] - box[1]
    draw.ellipse((box[0] + width * 0.25, box[1] + height * 0.25, box[0] + width * 0.75, box[1] + height * 0.75), outline=(10, 10, 10), width=2)


def draw_map_axes(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int]) -> None:
    x0, y0, x1, y1 = box
    label = font(10)
    for value in (-2, 0, 2):
        fraction = (value + 2.0) / 4.0
        px = x0 + fraction * (x1 - x0)
        draw.line((px, y1, px, y1 + 4), fill=(25, 30, 40), width=1)
        draw.text((px - 8, y1 + 5), str(value), fill=(60, 65, 75), font=label)
        py = y1 - fraction * (y1 - y0)
        draw.line((x0 - 4, py, x0, py), fill=(25, 30, 40), width=1)
        draw.text((x0 - 18, py - 6), str(value), fill=(60, 65, 75), font=label)
    draw.text(((x0 + x1) // 2 - 10, y1 + 20), "x/R", fill=(60, 65, 75), font=label)
    draw.text((x0 - 35, (y0 + y1) // 2 - 5), "y/R", fill=(60, 65, 75), font=label)


def draw_section_axes(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], depth: np.ndarray) -> None:
    x0, y0, x1, y1 = box
    label = font(10)
    for value in (-2, 0, 2):
        fraction = (value + 2.0) / 4.0
        px = x0 + fraction * (x1 - x0)
        draw.line((px, y1, px, y1 + 4), fill=(25, 30, 40), width=1)
        draw.text((px - 8, y1 + 5), str(value), fill=(60, 65, 75), font=label)
    for value in (0.0, float(depth[-1]) / 2.0, float(depth[-1])):
        py = y0 + value / float(depth[-1]) * (y1 - y0)
        draw.line((x0 - 4, py, x0, py), fill=(25, 30, 40), width=1)
        draw.text((x0 - 38, py - 6), f"{value:.0f}", fill=(60, 65, 75), font=label)
    draw.text(((x0 + x1) // 2 - 10, y1 + 20), "x/R", fill=(60, 65, 75), font=label)
    draw.text((x0 - 55, (y0 + y1) // 2 - 5), "depth", fill=(60, 65, 75), font=label)


def layer_index(depth: np.ndarray, lower: float, upper: float) -> np.ndarray:
    use = np.flatnonzero((depth >= lower) & (depth <= upper))
    if not use.size:
        raise ValueError(f"No OFES layers in {lower}-{upper} m")
    return use


def field_slices(field: np.ndarray, depth: np.ndarray) -> list[tuple[str, np.ndarray]]:
    return [
        ("0-100 m mean", np.nanmean(field[layer_index(depth, 0.0, 100.0)], axis=0)),
        ("300-500 m mean", np.nanmean(field[layer_index(depth, 300.0, 500.0)], axis=0)),
        ("450 m slice", field[int(np.nanargmin(np.abs(depth - 450.0)))]),
    ]


def render(data: dict[str, np.ndarray], selected: pd.DataFrame, output: Path, profile: str, polarity: str) -> dict[str, object]:
    depth = np.asarray(data["depth_m"], dtype="f8")
    x = np.asarray(data["x_over_r"], dtype="f8")
    prho = np.asarray(data["composite_prho"], dtype="f4")
    ring = np.asarray(data["rho_anom_ring_kg_m3"], dtype="f4")
    native_w = np.asarray(data["native_w_m_s"], dtype="f4") * 1.0e6
    fields = [
        ("rho", "Native OFES prho", prho, False, "prho (kg m-3)"),
        ("rho_rel", "Outer-ring-relative density", ring, True, "delta prho (kg m-3)"),
        ("w", "Native OFES w", native_w, True, "w (10^-6 m s-1)"),
    ]
    width, row_height = 2260, 390
    image = Image.new("RGB", (width, 135 + len(fields) * row_height + 45), "white")
    draw = ImageDraw.Draw(image)
    draw.text((42, 23), f"NH {polarity} strict-core composite | {profile} | Jan1-Jan19 | {len(selected)} object-days", fill=(15, 25, 42), font=font(27, True))
    draw.text((44, 62), "Unrotated direct pointwise mean on [-2R, 2R], delta=0.04R. Field contours disabled; solid ring is 1R.", fill=(65, 70, 82), font=font(15))
    left, section_width, map_size, gap = 65, 420, 325, 42
    for row, (short_label, full_label, field, signed, unit) in enumerate(fields):
        top = 125 + row * row_height
        lo, hi = finite_limits(field, signed)
        section = np.nanmean(field, axis=1)
        section_box = (left, top, left + section_width, top + map_size)
        paste(image, section, lo, hi, section_box)
        draw.rectangle(section_box, outline=(25, 30, 40), width=2)
        x_mid = (section_box[0] + section_box[2]) // 2
        draw.line((x_mid, section_box[1], x_mid, section_box[3]), fill=(30, 35, 45), width=1)
        draw.text((section_box[0], section_box[1] - 23), f"{short_label} | x/R-depth", fill=(20, 30, 45), font=font(13, True))
        draw.text((section_box[0] + 125, section_box[1] - 23), full_label, fill=(75, 78, 85), font=font(10))
        draw_section_axes(draw, section_box, depth)
        for col, (slice_label, plane) in enumerate(field_slices(field, depth)):
            x0 = left + section_width + gap + col * (map_size + gap)
            box = (x0, top, x0 + map_size, top + map_size)
            paste(image, plane, lo, hi, box)
            draw_ring(draw, box)
            draw.rectangle(box, outline=(25, 30, 40), width=2)
            draw.text((x0, top - 23), f"{short_label} | {slice_label}", fill=(20, 30, 45), font=font(12, True))
            draw_map_axes(draw, box)
        draw_colorbar(draw, image, (2180, top, 2210, top + map_size), lo, hi, unit)
    output.parent.mkdir(parents=True, exist_ok=True)
    png, pdf = output.with_suffix(".png"), output.with_suffix(".pdf")
    image.save(png)
    image.save(pdf, "PDF", resolution=180)
    return {"png": str(png), "pdf": str(pdf), "object_days": int(len(selected)), "contours": "disabled (linestyle=NaN equivalent)"}


def main() -> None:
    args = parse_args()
    with np.load(args.input, allow_pickle=False) as archive:
        data = {name: archive[name] for name in archive.files}
    selected = pd.read_csv(args.selected_object_days, low_memory=False)
    if not selected.polarity.astype(str).str.lower().eq(args.polarity).all():
        raise ValueError("Selected object table polarity does not match --polarity")
    record = render(data, selected, args.output_root / "strict_core_native_prho_outer_ring_w", args.profile_label, args.polarity)
    manifest = {
        "input": str(args.input), "selected_object_days": str(args.selected_object_days), "profile": args.profile_label,
        "polarity": args.polarity, "density": "native OFES prho", "density_anomaly": "same-depth 1.5R-2R outer-ring median reference; not climatology",
        "vertical_velocity": "native OFES w aligned to density layer centers; no high-pass and no rebuild-W", "composite": "unrotated direct finite-cell mean on [-2R,2R], delta=0.04R", "axes": "maps: x/R and y/R; sections: x/R and depth in metres", **record,
    }
    (args.output_root / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
