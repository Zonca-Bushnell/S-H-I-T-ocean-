"""Compare two unrotated pointwise OFES native-W composites on one grid.

Pillow is used deliberately: the Windows Matplotlib backend can crash while
rendering multi-panel colour bars although the numerical arrays are valid.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--polarity", choices=("cyclonic", "anticyclonic"), default="cyclonic")
    return parser.parse_args()


def font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in (r"C:\Windows\Fonts\arial.ttf", r"C:\Windows\Fonts\segoeui.ttf"):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def payload(root: Path) -> dict[str, np.ndarray]:
    matches = list(root.glob("paper_pointwise_no_rotation/*/isopycnal_composite.npz"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected exactly one composite below {root}, found {len(matches)}")
    path = matches[0]
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def band_mean(data: dict[str, np.ndarray], lower: float, upper: float) -> np.ndarray:
    use = (data["depth_m"] >= lower) & (data["depth_m"] <= upper)
    if not np.any(use):
        raise ValueError(f"No depths in {lower}-{upper} m")
    return np.nanmean(data["native_w_m_s"][use], axis=0) * 1.0e6


def shared_limit(*fields: np.ndarray) -> float:
    values = np.concatenate([np.abs(field[np.isfinite(field)]) for field in fields])
    return max(float(np.nanpercentile(values, 99.0)), 1.0e-3)


def coolwarm(field: np.ndarray, limit: float) -> Image.Image:
    anchors = np.asarray([[59, 76, 192], [128, 164, 220], [245, 245, 245], [221, 123, 112], [180, 4, 38]], dtype="f8")
    value = np.clip((np.asarray(field, dtype="f8") / limit + 1.0) * 0.5, 0.0, 1.0)
    scaled = value * (len(anchors) - 1)
    low = np.floor(scaled).astype(int)
    high = np.clip(low + 1, 0, len(anchors) - 1)
    fraction = (scaled - low)[..., None]
    rgb = anchors[low] * (1.0 - fraction) + anchors[high] * fraction
    rgb[~np.isfinite(field)] = (220, 220, 220)
    return Image.fromarray(np.asarray(np.rint(rgb), dtype="u1"), mode="RGB")


def paste_field(canvas: Image.Image, field: np.ndarray, limit: float, box: tuple[int, int, int, int], *, circle: bool) -> None:
    image = coolwarm(field, limit).resize((box[2] - box[0], box[3] - box[1]), Image.Resampling.BILINEAR)
    canvas.paste(image, box[:2])
    draw = ImageDraw.Draw(canvas)
    draw.rectangle(box, outline="black", width=2)
    if circle:
        width, height = box[2] - box[0], box[3] - box[1]
        draw.ellipse((box[0] + width * 0.25, box[1] + height * 0.25, box[0] + width * 0.75, box[1] + height * 0.75), outline="black", width=2)


def colorbar(canvas: Image.Image, limit: float, x: int, y: int, height: int) -> None:
    values = np.linspace(limit, -limit, height, dtype="f8")[:, None]
    bar = coolwarm(values, limit).resize((20, height), Image.Resampling.NEAREST)
    canvas.paste(bar, (x, y))
    draw, label = ImageDraw.Draw(canvas), font(13)
    draw.rectangle((x, y, x + 20, y + height), outline="black")
    draw.text((x + 26, y - 7), f"{limit:.1f}", fill="black", font=label)
    draw.text((x + 26, y + height // 2 - 7), "0", fill="black", font=label)
    draw.text((x + 26, y + height - 14), f"{-limit:.1f}", fill="black", font=label)


def save(image: Image.Image, output: Path) -> None:
    image.save(output.with_suffix(".png"))
    image.save(output.with_suffix(".pdf"), "PDF", resolution=180.0)


def draw_plan_comparison(old: dict[str, np.ndarray], new: dict[str, np.ndarray], output: Path, polarity: str) -> None:
    labels = (("0-100 m", 0.0, 100.0), ("300-500 m", 300.0, 500.0), ("450 m", 425.0, 475.0))
    fields = [(band_mean(old, low, high), band_mean(new, low, high)) for _, low, high in labels]
    common = shared_limit(*(field for pair in fields for field in pair))
    diff_limit = shared_limit(*(new_field - old_field for old_field, new_field in fields))
    image = Image.new("RGB", (1800, 1680), "white")
    draw = ImageDraw.Draw(image)
    title, label = font(30), font(18)
    draw.text((55, 30), f"NH {polarity} strict-core native W: full versus tangent-then-near-closed", fill="black", font=title)
    titles = ("old full", "new hybrid", "new - old")
    for row, ((name, _, _), (old_field, new_field)) in enumerate(zip(labels, fields)):
        y0 = 110 + row * 510
        for column, (title_text, field, limit) in enumerate(zip(titles, (old_field, new_field, new_field - old_field), (common, common, diff_limit))):
            x0 = 70 + column * 580
            draw.text((x0, y0), f"{name}: {title_text}", fill="black", font=label)
            box = (x0, y0 + 30, x0 + 450, y0 + 480)
            paste_field(image, field, limit, box, circle=True)
            colorbar(image, limit, x0 + 470, y0 + 30, 450)
            draw.text((x0 + 180, y0 + 485), "x/R", fill="black", font=label)
            draw.text((x0, y0 + 250), "y/R", fill="black", font=label)
    save(image, output)


def draw_section_comparison(old: dict[str, np.ndarray], new: dict[str, np.ndarray], output: Path, polarity: str) -> None:
    old_section = np.nanmean(old["native_w_m_s"], axis=1) * 1.0e6
    new_section = np.nanmean(new["native_w_m_s"], axis=1) * 1.0e6
    common, diff_limit = shared_limit(old_section, new_section), shared_limit(new_section - old_section)
    image = Image.new("RGB", (1800, 820), "white")
    draw = ImageDraw.Draw(image)
    title, label = font(30), font(18)
    draw.text((55, 30), f"NH {polarity} native W x/R-depth: full versus tangent-then-near-closed", fill="black", font=title)
    for column, (name, field, limit) in enumerate((("old full", old_section, common), ("new hybrid", new_section, common), ("new - old", new_section - old_section, diff_limit))):
        x0 = 80 + column * 580
        draw.text((x0, 95), name, fill="black", font=label)
        box = (x0, 130, x0 + 450, 730)
        paste_field(image, field, limit, box, circle=False)
        draw.line((x0 + 225, 130, x0 + 225, 730), fill="black", width=2)
        colorbar(image, limit, x0 + 470, 130, 600)
        draw.text((x0 + 180, 745), "x/R", fill="black", font=label)
        draw.text((x0, 425), "depth", fill="black", font=label)
    save(image, output)


def metrics(root: Path) -> pd.DataFrame:
    table = pd.read_csv(root / "native_w_composite_metrics.csv")
    selected = pd.read_csv(root / "selected_object_days.csv")
    table.insert(0, "object_days", int(len(selected)))
    for field in ("radius_km", "max_depth_m", "section_bipolarity_min_fraction"):
        table[f"median_{field}"] = float(pd.to_numeric(selected.get(field), errors="coerce").median())
    return table


def field_difference_metrics(old: dict[str, np.ndarray], new: dict[str, np.ndarray]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for name, unit, scale in (
        ("composite_prho", "kg m-3", 1.0),
        ("rho_anom_ring_kg_m3", "kg m-3", 1.0),
        ("native_w_m_s", "m s-1", 1.0e6),
    ):
        old_values = np.asarray(old[name], dtype="f8") * scale
        new_values = np.asarray(new[name], dtype="f8") * scale
        difference = new_values - old_values
        valid = difference[np.isfinite(difference)]
        rows.append({
            "field": name, "unit": unit if scale == 1.0 else "10^-6 " + unit,
            "old_q95_abs": float(np.nanpercentile(np.abs(old_values), 95)),
            "new_q95_abs": float(np.nanpercentile(np.abs(new_values), 95)),
            "new_minus_old_q95_abs": float(np.nanpercentile(np.abs(valid), 95)) if valid.size else np.nan,
        })
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    old, new = payload(args.baseline_root), payload(args.candidate_root)
    for key in ("x_over_r", "y_over_r", "depth_m"):
        if not np.array_equal(old[key], new[key]):
            raise ValueError(f"Composite grids differ for {key}; a direct comparison would be invalid")
    args.output_root.mkdir(parents=True, exist_ok=True)
    draw_plan_comparison(old, new, args.output_root / "native_w_plan_comparison", args.polarity)
    draw_section_comparison(old, new, args.output_root / "native_w_cross_section_comparison", args.polarity)
    old_metrics, new_metrics = metrics(args.baseline_root), metrics(args.candidate_root)
    compare = old_metrics.merge(new_metrics, on="depth_band", suffixes=("_old_full", "_new_hybrid"))
    compare.to_csv(args.output_root / "native_w_comparison_metrics.csv", index=False, encoding="utf-8-sig")
    field_metrics = field_difference_metrics(old, new)
    field_metrics.to_csv(args.output_root / "native_fields_difference_metrics.csv", index=False, encoding="utf-8-sig")
    summary = {"baseline_root": str(args.baseline_root), "candidate_root": str(args.candidate_root), "polarity": args.polarity, "color_map": "coolwarm", "grid": "[-2R,2R], delta=0.04R, pointwise direct mean", "metrics": compare.to_dict(orient="records"), "field_difference_metrics": field_metrics.to_dict(orient="records")}
    (args.output_root / "native_w_comparison_manifest.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
