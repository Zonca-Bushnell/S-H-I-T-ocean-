"""Render raw OFES density and vertical-velocity diagnostics for selected deep cores.

This is intentionally a read-only diagnostic: it samples the native daily
``prho`` and ``w`` fields and never runs reconstruction, filtering, or QC.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from ..ofes_io import ctl_path, expected_dta_bytes, open_dta_memmap, parse_ctl, require_daily_file
from ..run_ofes_rebuild_w import align_native_w_vertical, meters_per_degree, parse_iso_date, sample_stack
from ..w_rebuild_config import DEFAULT_DATA_ROOT


DAY = "1991-01-01"
VERTICAL_ROOT = Path(
    r"E:\DATA\01_Eddy_correspond\02_OFES\Fromthebeginning\04_Vertical\hua_center_section_selection_19910101\local_step_2cells_section_bipolar_tiebreak"
)
DEFAULT_OUTPUT_ROOT = Path(
    r"E:\DATA\01_Eddy_correspond\02_OFES\Fromthebeginning\05_TEMP\jan01_strict_core_raw_prho_w_19910101"
)
SELECTED = [
    ("kuroshio_extension", "19910101_01073"),
    ("kuroshio_extension", "19910101_00131"),
    ("south_pacific_stcc", "19910101_03856"),
    ("south_pacific_stcc", "19910101_04637"),
    ("taiwan_hawaii_corridor", "19910101_05716"),
    ("taiwan_hawaii_corridor", "19910101_04887"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render raw OFES prho and native w for six selected strict-core tracks.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--vertical-root", type=Path, default=VERTICAL_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--grid-n", type=int, default=101)
    parser.add_argument("--extent-r", type=float, default=3.0)
    return parser.parse_args()


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    names = ("segoeuib.ttf", "arialbd.ttf") if bold else ("segoeui.ttf", "arial.ttf")
    for name in names:
        path = Path(r"C:\Windows\Fonts") / name
        if path.exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def finite_percentile(values: np.ndarray, q: float, fallback: float = 1.0) -> float:
    finite = np.asarray(values, dtype="f8")
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        return fallback
    return max(float(np.percentile(finite, q)), fallback * 1.0e-12)


def sequential_rgb(values: np.ndarray, lo: float, hi: float) -> np.ndarray:
    t = np.nan_to_num(np.clip((values - lo) / max(hi - lo, 1.0e-12), 0.0, 1.0), nan=0.0)
    out = np.empty((*values.shape, 3), dtype=np.uint8)
    out[..., 0] = (22 + 210 * t).astype(np.uint8)
    out[..., 1] = (55 + 155 * t).astype(np.uint8)
    out[..., 2] = (110 + 105 * (1.0 - t)).astype(np.uint8)
    out[~np.isfinite(values)] = (55, 60, 68)
    return out


def signed_rgb(values: np.ndarray, vmax: float) -> np.ndarray:
    t = np.nan_to_num(np.clip(values / max(vmax, 1.0e-12), -1.0, 1.0), nan=0.0)
    out = np.full((*values.shape, 3), 245, dtype=np.uint8)
    neg, pos = t < 0.0, t > 0.0
    out[..., 0][neg] = (245 * (1.0 + t[neg])).astype(np.uint8)
    out[..., 1][neg] = (245 * (1.0 + t[neg])).astype(np.uint8)
    out[..., 2][neg] = 175
    out[..., 0][pos] = 185
    out[..., 1][pos] = (245 * (1.0 - t[pos])).astype(np.uint8)
    out[..., 2][pos] = (245 * (1.0 - t[pos])).astype(np.uint8)
    out[~np.isfinite(values)] = (55, 60, 68)
    return out


def paste_raster(image: Image.Image, values: np.ndarray, box: tuple[int, int, int, int], rgb: np.ndarray) -> None:
    raster = Image.fromarray(rgb, mode="RGB").resize((box[2] - box[0], box[3] - box[1]), Image.Resampling.BILINEAR)
    image.paste(raster, (box[0], box[1]))


def draw_map_track(draw: ImageDraw.ImageDraw, rows: pd.DataFrame, k: int, box: tuple[int, int, int, int], extent_km: float) -> None:
    surface = rows.iloc[0]
    mx, my = meters_per_degree(float(surface.center_lat_refined))
    visible = rows.loc[rows.depth_index.le(k)].sort_values("depth_index")
    points = []
    for row in visible.itertuples(index=False):
        lon = float(getattr(row, "center_lon_refined"))
        lat = float(getattr(row, "center_lat_refined"))
        dx = ((lon - float(surface.center_lon_refined) + 180.0) % 360.0 - 180.0) * mx / 1000.0
        dy = (lat - float(surface.center_lat_refined)) * my / 1000.0
        x = box[0] + (dx + extent_km) / (2.0 * extent_km) * (box[2] - box[0])
        y = box[3] - (dy + extent_km) / (2.0 * extent_km) * (box[3] - box[1])
        points.append((int(x), int(y)))
    if len(points) > 1:
        draw.line(points, fill=(18, 18, 18), width=2)
    for idx, point in enumerate(points):
        radius = 4 if idx == len(points) - 1 else 2
        draw.ellipse((point[0] - radius, point[1] - radius, point[0] + radius, point[1] + radius), fill=(20, 20, 20))


def draw_section(image: Image.Image, draw: ImageDraw.ImageDraw, field: np.ndarray, depth: np.ndarray, rows: pd.DataFrame, extent_km: float, box: tuple[int, int, int, int], title: str, rgb_fn) -> None:
    if title.startswith("Raw density"):
        lo, hi = finite_percentile(field, 2), finite_percentile(field, 98)
        rgb = sequential_rgb(field, lo, hi)
        scale = f"{lo:.2f} to {hi:.2f} native density"
    else:
        vmax = finite_percentile(np.abs(field), 98)
        rgb = signed_rgb(field, vmax)
        scale = f"+/- {vmax:.2e} m/s"
    paste_raster(image, field, box, rgb)
    draw.rectangle(box, outline=(30, 35, 45), width=2)
    x0 = box[0] + (box[2] - box[0]) // 2
    draw.line((x0, box[1], x0, box[3]), fill=(25, 25, 25), width=1)
    track = rows.sort_values("depth_index")
    surface = track.iloc[0]
    mx, _ = meters_per_degree(float(surface.center_lat_refined))
    points = []
    max_depth = max(float(depth[-1]), 1.0)
    for row in track.itertuples(index=False):
        dx = ((float(row.center_lon_refined) - float(surface.center_lon_refined) + 180.0) % 360.0 - 180.0) * mx / 1000.0
        x = box[0] + (dx + extent_km) / (2.0 * extent_km) * (box[2] - box[0])
        y = box[1] + float(row.depth_m) / max_depth * (box[3] - box[1])
        points.append((int(x), int(y)))
    if len(points) > 1:
        draw.line(points, fill=(15, 15, 15), width=2)
    draw.text((box[0], box[1] - 25), title, fill=(20, 30, 45), font=font(17, True))
    draw.text((box[0], box[3] + 5), f"- {extent_km:.0f} km   surface-center east offset   + {extent_km:.0f} km; depth down", fill=(65, 70, 80), font=font(13))
    draw.text((box[2] - 180, box[1] + 5), scale, fill=(20, 25, 35), font=font(13))


def select_depth_indices(depth: np.ndarray, rows: pd.DataFrame) -> list[int]:
    max_k = int(rows.depth_index.max())
    available = sorted(set(int(k) for k in rows.depth_index if int(k) <= max_k))
    targets = [0, max_k / 3.0, 2.0 * max_k / 3.0, max_k]
    picked = []
    for target in targets:
        idx = min(available, key=lambda value: abs(value - target))
        if idx not in picked:
            picked.append(idx)
    while len(picked) < 4:
        picked.append(picked[-1])
    return picked[:4]


def render_object(output: Path, region: str, object_id: str, rows: pd.DataFrame, prho: np.ndarray, native_w: np.ndarray, depth: np.ndarray, extent_km: float) -> dict[str, object]:
    object_dir = output / "objects" / f"{region}_{object_id}"
    object_dir.mkdir(parents=True, exist_ok=True)
    depth_indices = select_depth_indices(depth, rows)
    rho_lo, rho_hi = finite_percentile(prho, 2), finite_percentile(prho, 98)
    w_vmax = finite_percentile(np.abs(native_w), 98)
    width, height = 2160, 1420
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    surface = rows.iloc[0]
    draw.text((45, 28), f"Raw OFES prho and native w | {region} | {object_id} | {DAY}", fill=(15, 25, 40), font=font(32, True))
    draw.text((48, 75), "No climatology, filtering, reconstruction, or persistence is applied. Black path: accepted vertical centers.", fill=(65, 70, 82), font=font(18))
    left, top, cell_w, cell_h, gap_x, gap_y = 110, 145, 440, 250, 45, 60
    labels = ["Raw OFES prho (native density)", "Raw OFES w (m/s)"]
    for col, k in enumerate(depth_indices):
        x0 = left + col * (cell_w + gap_x)
        draw.text((x0, 112), f"depth {float(depth[k]):.1f} m | layer {k}", fill=(25, 35, 50), font=font(18, True))
        for row_index, label in enumerate(labels):
            box = (x0, top + row_index * (cell_h + gap_y), x0 + cell_w, top + row_index * (cell_h + gap_y) + cell_h)
            if row_index == 0:
                paste_raster(image, prho[k], box, sequential_rgb(prho[k], rho_lo, rho_hi))
            else:
                paste_raster(image, native_w[k], box, signed_rgb(native_w[k], w_vmax))
            draw.rectangle(box, outline=(30, 35, 45), width=2)
            draw_map_track(draw, rows, k, box, extent_km)
            if col == 0:
                draw.text((22, box[1] + 90), label, fill=(20, 30, 45), font=font(17, True))
    section_top = 860
    section_w, section_h = 940, 350
    draw_section(image, draw, prho[:, prho.shape[1] // 2, :], depth, rows, extent_km, (120, section_top, 120 + section_w, section_top + section_h), "Raw density east-depth section", sequential_rgb)
    draw_section(image, draw, native_w[:, native_w.shape[1] // 2, :], depth, rows, extent_km, (1120, section_top, 1120 + section_w, section_top + section_h), "Raw native w east-depth section", signed_rgb)
    draw.text((50, 1270), f"Raw prho scale: {rho_lo:.3f} to {rho_hi:.3f} native units. Native w scale: +/- {w_vmax:.3e} m/s. Surface radius: {float(surface.radius_km):.1f} km; deepest accepted center: {float(rows.depth_m.max()):.1f} m.", fill=(60, 65, 78), font=font(17))
    png = object_dir / "raw_prho_native_w_family.png"
    image.save(png)
    image.save(object_dir / "raw_prho_native_w_family.pdf", "PDF", resolution=160)
    np.savez_compressed(
        object_dir / "raw_prho_native_w_data.npz",
        depth_m=depth.astype("f4"),
        prho_raw_native=prho.astype("f4"),
        native_w_m_s=native_w.astype("f4"),
        center_lon_refined=rows.center_lon_refined.to_numpy(dtype="f4"),
        center_lat_refined=rows.center_lat_refined.to_numpy(dtype="f4"),
        center_depth_index=rows.depth_index.to_numpy(dtype="i2"),
    )
    return {
        "region": region, "hua_object_id": object_id, "polarity": str(surface.polarity), "surface_lon": float(surface.center_lon_refined), "surface_lat": float(surface.center_lat_refined),
        "surface_radius_km": float(surface.radius_km), "accepted_layers": int(len(rows)), "max_depth_m": float(rows.depth_m.max()), "density_native_p02": rho_lo,
        "density_native_p98": rho_hi, "native_w_abs_p98_m_s": w_vmax, "figure_png": str(png), "data_npz": str(object_dir / "raw_prho_native_w_data.npz"),
    }


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    structures_path = args.vertical_root / "raw_detection" / "daily_runs" / "19910101" / "structures_hua_style.csv"
    strict_path = args.vertical_root / "section_bipolar_vertical_catalog_local_lat_scaled" / "section_bipolar_strict_core_bilateral_layer_catalog.csv"
    structures = pd.read_csv(structures_path)
    strict = pd.read_csv(strict_path)
    metas = {name: parse_ctl(ctl_path(args.data_root, name)) for name in ("prho", "w")}
    target_day = parse_iso_date(DAY)
    raw = {name: open_dta_memmap(require_daily_file(args.data_root, name, target_day, expected_dta_bytes(meta)), meta) for name, meta in metas.items()}
    nlev = min(105, metas["prho"].z.count)
    depth = np.asarray(metas["prho"].z.values[:nlev], dtype="f8")
    summaries = []
    for region, object_id in SELECTED:
        track = strict.loc[strict.hua_object_id.astype(str).eq(object_id)].copy()
        surface = structures.loc[(structures.hua_object_id.astype(str).eq(object_id)) & (structures.depth_index.eq(0))].copy()
        if track.empty or surface.empty:
            raise RuntimeError(f"Missing strict track or surface record for {object_id}")
        rows = pd.concat([surface, track], ignore_index=True).drop_duplicates("depth_index").sort_values("depth_index")
        lat0, lon0, radius = float(surface.iloc[0].center_lat_refined), float(surface.iloc[0].center_lon_refined), float(surface.iloc[0].radius_km)
        extent_km = max(120.0, radius * float(args.extent_r))
        x = np.linspace(-extent_km, extent_km, args.grid_n)
        y = np.linspace(-extent_km, extent_km, args.grid_n)
        xx, yy = np.meshgrid(x, y)
        mx, my = meters_per_degree(lat0)
        lon = (lon0 + xx * 1000.0 / mx) % 360.0
        lat = lat0 + yy * 1000.0 / my
        prho = sample_stack(raw["prho"], metas["prho"], lon, lat, nlev)
        w_raw = sample_stack(raw["w"], metas["w"], lon, lat, min(metas["w"].z.count, nlev + 1)) / 100.0
        native_w, alignment = align_native_w_vertical(w_raw, np.asarray(metas["w"].z.values[:w_raw.shape[0]], dtype="f8"), depth, "layer_center")
        summaries.append(render_object(args.output_root, region, object_id, rows, prho, native_w, depth, extent_km))
    pd.DataFrame(summaries).to_csv(args.output_root / "raw_prho_native_w_summary.csv", index=False)
    manifest = {
        "day": DAY, "objects": summaries, "density_definition": "native OFES prho, unfiltered and not climatology-subtracted", "vertical_velocity_definition": "native OFES w / 100, interpolated to prho layer-center depths", "density_anomaly_status": "not generated; no prho climatology was used", "source_vertical_root": str(args.vertical_root), "data_root": str(args.data_root), "grid_n": args.grid_n, "extent_r": args.extent_r,
    }
    (args.output_root / "raw_prho_native_w_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"rendered": len(summaries), "output_root": str(args.output_root)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
