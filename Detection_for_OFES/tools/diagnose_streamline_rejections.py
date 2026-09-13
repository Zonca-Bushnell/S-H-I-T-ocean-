from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


DEFAULT_RESULT_ROOT = Path(r"E:\DATA\01_Eddy_correspond\02_OFES\origin_streamline_cpu_jan01_jan19_life1")
KON_LON_MIN, KON_LON_MAX = 120.0, 145.0
KON_LAT_MIN, KON_LAT_MAX = 20.0, 35.0
MAP_BOX = (85, 95, 1665, 1045)


FASTPARQUET_CACHE_PATHS = [
    r"D:\Util\lever\02_miniforge\pkgs\cramjam-2.11.0-py312h7fb921c_2\Lib\site-packages",
    r"D:\Util\lever\02_miniforge\pkgs\fsspec-2026.7.0-pyhd8ed1ab_0\site-packages",
    r"D:\Util\lever\02_miniforge\pkgs\fsspec-2026.7.0-pyhd8ed1ab_0\Lib\site-packages",
    r"D:\Util\lever\02_miniforge\pkgs\fastparquet-2026.5.0-py312h196c9fc_0\Lib\site-packages",
]


FAILURE_COLORS = {
    "no_closed_streamline": (245, 158, 11),
    "velocity_ratio": (168, 85, 247),
    "boundary_monotonic_rotation": (14, 165, 233),
    "angle_jump": (236, 72, 153),
    "opposite_reversal": (34, 197, 94),
    "tangent_alignment": (20, 184, 166),
    "hua_pass": (255, 255, 255),
    "other": (120, 113, 108),
}


def add_fastparquet_bridge() -> tuple[bool, str]:
    for path in FASTPARQUET_CACHE_PATHS:
        if os.path.isdir(path) and path not in sys.path:
            sys.path.insert(0, path)
    try:
        import fastparquet  # noqa: F401

        return True, "fastparquet cache bridge available"
    except Exception as exc:  # pragma: no cover - diagnostic metadata
        return False, f"fastparquet unavailable: {type(exc).__name__}: {exc}"


def read_parquet(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    ok, msg = add_fastparquet_bridge()
    if not ok:
        raise RuntimeError(msg)
    import fastparquet

    return fastparquet.ParquetFile(str(path)).to_pandas(columns=columns)


def font(size: int) -> ImageFont.ImageFont:
    for path in [r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\arial.ttf"]:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            pass
    return ImageFont.load_default()


def px(lon: float) -> int:
    x0, _y0, x1, _y1 = MAP_BOX
    return x0 + int(round((lon - KON_LON_MIN) / (KON_LON_MAX - KON_LON_MIN) * (x1 - x0)))


def py(lat: float) -> int:
    _x0, y0, _x1, y1 = MAP_BOX
    return y1 - int(round((lat - KON_LAT_MIN) / (KON_LAT_MAX - KON_LAT_MIN) * (y1 - y0)))


def in_kuroshio_box(df: pd.DataFrame, lon_col: str, lat_col: str) -> pd.Series:
    return (
        df[lon_col].astype(float).between(KON_LON_MIN, KON_LON_MAX)
        & df[lat_col].astype(float).between(KON_LAT_MIN, KON_LAT_MAX)
    )


def norm_date(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series).dt.strftime("%Y-%m-%d")


def safe_reason(value: object) -> str:
    text = str(value) if value is not None else ""
    if not text or text.lower() in {"nan", "none"}:
        return "unknown"
    return text


def write_json(path: Path, obj: object) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def write_literature_note(path: Path) -> None:
    text = """# Velocity-streamline false positives and rejected OFES seeds

## 为什么 velocity streamline 会偏向黑潮流轴

Instantaneous velocity streamline closure is a dynamical-core diagnostic, not the
same object definition used by SSH/SLA eddy atlases. In a western boundary
current, jet meanders, recirculation tongues, and local shear minima can produce
closed or near-closed velocity streamlines along the current axis. These can
pass a streamline closure test even when they are not an isolated mesoscale eddy
interior.

## 文献口径

- Mason et al. (2014), *A New Sea Surface Height-Based Code for Oceanic
  Mesoscale Eddy Tracking*, identifies and tracks eddies from sea-surface-height
  closed contours. This is closer to META/py-eddy-tracker than to a pure
  velocity-streamline closure test.
  https://journals.ametsoc.org/downloadpdf/view/journals/atot/31/5/jtech-d-14-00019_1.pdf
- py-eddy-tracker documentation distinguishes the effective contour from the
  contour of maximum mean speed. This separation is useful here: OFES
  velocity-streamline boundaries should be checked against an SSH/effective
  contour, not treated as the complete eddy definition.
  https://py-eddy-tracker.readthedocs.io/en/stable/python_module/02_eddy_identification/pet_eddy_detection.html
- AVISO/META4.0 products provide eddy location, contours, amplitude, radius
  speed and metadata from altimetry, beginning in 1993. META is therefore a
  useful spatial/product-definition reference, but not a strict 1991 OFES truth
  set.
  https://www.aviso.altimetry.fr/
- Haller & Beron-Vera (2013) define coherent Lagrangian vortex boundaries as
  material/coherence-based curves, emphasizing that instantaneous Eulerian
  closed curves are not automatically coherent eddies.
  https://arxiv.org/abs/1308.2352
- Chelton, Schlax & Samelson (2011) describe global nonlinear mesoscale eddies
  from altimetry and provide the broader observational context for mesoscale
  eddy radius/amplitude/lifetime expectations.
  https://doi.org/10.1016/j.pocean.2011.01.002

## 对 OFES 下一步的直接含义

1. `velocity_streamline_contour` should be treated as a strong dynamical-core
   diagnostic, not as the only eddy catalog definition.
2. Add an SSH/effective-contour consensus gate before accepting a streamline
   object as an eddy interior.
3. Add a jet-axis/meander diagnostic: high overlap with the fast current core
   should be flagged separately from isolated eddies.
4. Audit `no_closed_streamline` seeds: many may be weak-velocity but
   SSH-coherent eddies that need a fallback SSH contour or fixed-circle check.
"""
    path.write_text(text, encoding="utf-8")


def summarize(df: pd.DataFrame, label: str) -> pd.DataFrame:
    reasons = df["seed_fate"].value_counts(dropna=False).rename_axis("seed_fate").reset_index(name="count")
    reasons["scope"] = label
    total = max(int(len(df)), 1)
    reasons["fraction"] = reasons["count"] / total
    return reasons[["scope", "seed_fate", "count", "fraction"]]


def select_samples(surface: pd.DataFrame, per_reason: int) -> pd.DataFrame:
    failed = surface[~surface["hua_pass"].astype(bool)].copy()
    if failed.empty:
        return failed
    failed["reason_rank"] = failed["seed_fate"].map(failed["seed_fate"].value_counts())
    parts = []
    for reason, group in failed.sort_values(["reason_rank", "seed_fate"], ascending=[False, True]).groupby("seed_fate"):
        group = group.copy()
        group["offset_abs_km"] = np.hypot(group["center_x_from_seed_km"].astype(float), group["center_y_from_seed_km"].astype(float))
        group["ssh_abs"] = np.abs(group["ssh_value_m"].astype(float))
        group = group.sort_values(["ssh_abs", "offset_abs_km"], ascending=[False, False])
        parts.append(group.head(per_reason))
    return pd.concat(parts, ignore_index=True) if parts else failed.head(0)


def draw_seed_fate_map(base_png: Path, day_surface: pd.DataFrame, meta_manifest: Path, out_png: Path) -> None:
    canvas = Image.open(base_png).convert("RGB")
    draw = ImageDraw.Draw(canvas)

    failed = day_surface[~day_surface["hua_pass"].astype(bool)].copy()
    passed = day_surface[day_surface["hua_pass"].astype(bool)].copy()

    # Failed seeds first, so accepted OFES centers remain visible.
    for _, row in failed.iterrows():
        lon = float(row["seed_lon"])
        lat = float(row["seed_lat"])
        if not (KON_LON_MIN <= lon <= KON_LON_MAX and KON_LAT_MIN <= lat <= KON_LAT_MAX):
            continue
        x, y = px(lon), py(lat)
        color = FAILURE_COLORS.get(str(row["seed_fate"]), FAILURE_COLORS["other"])
        r = 5 if row["seed_fate"] == "no_closed_streamline" else 4
        draw.rectangle((x - r, y - r, x + r, y + r), outline=color, width=2)

    for _, row in passed.iterrows():
        lon = float(row["center_lon_refined"])
        lat = float(row["center_lat_refined"])
        if not (KON_LON_MIN <= lon <= KON_LON_MAX and KON_LAT_MIN <= lat <= KON_LAT_MAX):
            continue
        x, y = px(lon), py(lat)
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=(255, 255, 255), outline=(20, 24, 32), width=1)

    meta_count = 0
    if meta_manifest.exists():
        try:
            meta = json.loads(meta_manifest.read_text(encoding="utf-8"))
            meta_count = int(meta.get("meta_object_count", 0))
        except Exception:
            meta_count = 0

    y0 = 1090
    draw.rectangle((45, y0 - 10, 1645, 1150), fill=(255, 255, 255))
    draw.text((55, y0), "Seed fate overlay: squares = rejected SSH seeds; white dots = accepted velocity-streamline surface objects", fill=(18, 24, 38), font=font(16))
    x = 55
    y = y0 + 30
    for reason in ["no_closed_streamline", "velocity_ratio", "boundary_monotonic_rotation", "angle_jump", "opposite_reversal"]:
        color = FAILURE_COLORS[reason]
        draw.rectangle((x, y, x + 14, y + 14), outline=color, width=2)
        draw.text((x + 20, y - 3), reason, fill=(18, 24, 38), font=font(14))
        x += 260 if reason == "boundary_monotonic_rotation" else 205
    draw.text((1185, y - 3), f"Kuroshio surface seeds={len(day_surface):,}; accepted={len(passed):,}; rejected={len(failed):,}; META ref={meta_count:,}", fill=(18, 24, 38), font=font(14))

    canvas.save(out_png)
    canvas.save(out_png.with_suffix(".pdf"), "PDF", resolution=180.0)


def draw_sample(seed: pd.Series, base_png: Path, out_png: Path) -> None:
    base = Image.open(base_png).convert("RGB")
    # Crop around the seed in map coordinates, clamped to the map box.
    sx, sy = px(float(seed["seed_lon"])), py(float(seed["seed_lat"]))
    half = 175
    x0 = max(MAP_BOX[0], sx - half)
    x1 = min(MAP_BOX[2], sx + half)
    y0 = max(MAP_BOX[1], sy - half)
    y1 = min(MAP_BOX[3], sy + half)
    crop = base.crop((x0, y0, x1, y1)).resize((700, 700), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (980, 760), "white")
    canvas.paste(crop, (30, 40))
    draw = ImageDraw.Draw(canvas)

    def local_xy(lon: float, lat: float) -> tuple[int, int]:
        gx = px(lon)
        gy = py(lat)
        return 30 + int((gx - x0) / max(x1 - x0, 1) * 700), 40 + int((gy - y0) / max(y1 - y0, 1) * 700)

    seed_xy = local_xy(float(seed["seed_lon"]), float(seed["seed_lat"]))
    center_xy = local_xy(float(seed["center_lon_refined"]), float(seed["center_lat_refined"]))
    reason = str(seed["seed_fate"])
    color = FAILURE_COLORS.get(reason, FAILURE_COLORS["other"])
    draw.line((*seed_xy, *center_xy), fill=(255, 255, 255), width=5)
    draw.line((*seed_xy, *center_xy), fill=color, width=3)
    draw.rectangle((seed_xy[0] - 8, seed_xy[1] - 8, seed_xy[0] + 8, seed_xy[1] + 8), outline=color, width=4)
    draw.ellipse((center_xy[0] - 7, center_xy[1] - 7, center_xy[0] + 7, center_xy[1] + 7), fill=color, outline=(255, 255, 255), width=2)

    radius_cells = seed.get("accepted_radius_cells", np.nan)
    if not np.isfinite(radius_cells):
        radius_cells = seed.get("radius_cells", np.nan)
    if np.isfinite(radius_cells):
        radius_km = float(radius_cells) * 11.12
        lat = float(seed["center_lat_refined"])
        deg_lat = radius_km / 111.2
        deg_lon = deg_lat / max(math.cos(math.radians(lat)), 0.2)
        edge_xy = local_xy(float(seed["center_lon_refined"]) + deg_lon, lat)
        rr = abs(edge_xy[0] - center_xy[0])
        draw.ellipse((center_xy[0] - rr, center_xy[1] - rr, center_xy[0] + rr, center_xy[1] + rr), outline=color, width=2)

    draw.text((755, 45), f"{seed['hua_object_id']}", fill=(18, 24, 38), font=font(18))
    draw.text((755, 75), f"date: {seed['date']}", fill=(18, 24, 38), font=font(15))
    draw.text((755, 105), f"reason: {reason}", fill=color, font=font(15))
    draw.text((755, 145), f"seed: {float(seed['seed_lon']):.2f}E, {float(seed['seed_lat']):.2f}N", fill=(18, 24, 38), font=font(14))
    draw.text((755, 175), f"center: {float(seed['center_lon_refined']):.2f}E, {float(seed['center_lat_refined']):.2f}N", fill=(18, 24, 38), font=font(14))
    draw.text((755, 205), f"SSH: {float(seed['ssh_value_m']):.3g} m", fill=(18, 24, 38), font=font(14))
    draw.text((755, 235), f"speed min: {float(seed['center_speed_ms']):.3g} m/s", fill=(18, 24, 38), font=font(14))
    draw.text((755, 265), f"offset: {math.hypot(float(seed['center_x_from_seed_km']), float(seed['center_y_from_seed_km'])):.1f} km", fill=(18, 24, 38), font=font(14))
    draw.text((755, 310), "□ seed", fill=color, font=font(14))
    draw.text((755, 335), "● velocity weak center", fill=color, font=font(14))
    draw.text((755, 380), "Background is OFES", fill=(75, 85, 100), font=font(13))
    draw.text((755, 402), "surface speed anomaly", fill=(75, 85, 100), font=font(13))
    canvas.save(out_png)
    canvas.save(out_png.with_suffix(".pdf"), "PDF", resolution=180.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit OFES velocity-streamline rejected SSH seeds.")
    parser.add_argument("--result-root", default=str(DEFAULT_RESULT_ROOT))
    parser.add_argument("--day", default="1991-01-10")
    parser.add_argument("--sample-per-reason", type=int, default=3)
    args = parser.parse_args()

    root = Path(args.result_root)
    det = root / "hua_b3_start2_detection"
    out = root / "diagnostics" / "streamline_rejection_audit"
    fig_dir = out / "figures"
    sample_dir = fig_dir / "rejected_seed_samples"
    out.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(parents=True, exist_ok=True)

    columns = [
        "date",
        "hua_object_id",
        "depth_index",
        "seed_lon",
        "seed_lat",
        "center_lon_refined",
        "center_lat_refined",
        "center_x_from_seed_km",
        "center_y_from_seed_km",
        "ssh_value_m",
        "center_speed_ms",
        "radius_cells",
        "accepted_radius_cells",
        "streamline_closed",
        "streamline_points",
        "max_velocity_ratio",
        "max_angle_jump_deg",
        "tangent_pass_fraction",
        "opposite_reversal_fraction",
        "first_hard_failure",
        "dominant_failure",
        "hua_pass",
        "boundary_mode",
    ]
    centers = read_parquet(det / "centers_hua_style.parquet", columns=columns)
    centers["date"] = norm_date(centers["date"])
    centers["seed_fate"] = np.where(centers["hua_pass"].astype(bool), "hua_pass", centers["first_hard_failure"].map(safe_reason))
    surface = centers[centers["depth_index"].astype(int).eq(0)].copy()

    global_summary = summarize(surface, "surface_global")
    kuro_mask = in_kuroshio_box(surface, "seed_lon", "seed_lat")
    kuro_surface = surface[kuro_mask].copy()
    kuro_summary = summarize(kuro_surface, "surface_kuroshio_seed_box")
    day_kuro = kuro_surface[kuro_surface["date"].eq(args.day)].copy()
    day_summary = summarize(day_kuro, f"surface_kuroshio_seed_box_{args.day}")
    summary = pd.concat([global_summary, kuro_summary, day_summary], ignore_index=True)
    summary.to_csv(out / "seed_fate_summary.csv", index=False)
    write_json(out / "seed_fate_summary.json", summary.to_dict("records"))

    # Numeric diagnostics by fate in the Kuroshio box.
    diag = (
        kuro_surface.assign(
            offset_km=np.hypot(kuro_surface["center_x_from_seed_km"].astype(float), kuro_surface["center_y_from_seed_km"].astype(float)),
            abs_ssh_m=np.abs(kuro_surface["ssh_value_m"].astype(float)),
        )
        .groupby("seed_fate", dropna=False)
        .agg(
            count=("hua_object_id", "size"),
            median_abs_ssh_m=("abs_ssh_m", "median"),
            median_center_offset_km=("offset_km", "median"),
            median_center_speed_ms=("center_speed_ms", "median"),
            median_velocity_ratio=("max_velocity_ratio", "median"),
            median_angle_jump_deg=("max_angle_jump_deg", "median"),
            median_tangent_fraction=("tangent_pass_fraction", "median"),
            closed_fraction=("streamline_closed", "mean"),
        )
        .reset_index()
        .sort_values("count", ascending=False)
    )
    diag.to_csv(out / "kuroshio_rejection_numeric_diagnostics.csv", index=False)
    write_json(out / "kuroshio_rejection_numeric_diagnostics.json", diag.to_dict("records"))

    samples = select_samples(day_kuro, args.sample_per_reason)
    samples.to_csv(out / f"sample_rejected_seeds_{args.day.replace('-', '')}.csv", index=False)
    write_json(out / f"sample_rejected_seeds_{args.day.replace('-', '')}.json", samples.to_dict("records"))
    all_day_samples = select_samples(kuro_surface, args.sample_per_reason)
    all_day_samples.to_csv(out / "sample_rejected_seeds_kuroshio_all_days.csv", index=False)
    write_json(out / "sample_rejected_seeds_kuroshio_all_days.json", all_day_samples.to_dict("records"))

    base_png = root / "figures" / "latest_velocity_streamline_surface_and_family_with_edges" / f"ofes_velocity_streamline_kuroshio_lavd_style_{args.day.replace('-', '')}.png"
    meta_manifest = root / "figures" / "latest_velocity_streamline_surface_and_family_with_edges" / "meta4_overlay_manifest_META20190701_OFES19910110.json"
    fate_png = fig_dir / f"kuroshio_seed_fate_{args.day.replace('-', '')}.png"
    draw_seed_fate_map(base_png, day_kuro, meta_manifest, fate_png)

    sample_paths = []
    for _, row in samples.iterrows():
        reason = str(row["seed_fate"])
        stem = f"sample_{args.day.replace('-', '')}_{reason}_{row['hua_object_id']}.png".replace("/", "_")
        path = sample_dir / stem
        draw_sample(row, base_png, path)
        sample_paths.append(str(path))
    all_day_sample_paths = []
    for _, row in all_day_samples.iterrows():
        reason = str(row["seed_fate"])
        stem = f"sample_all_days_{reason}_{row['date']}_{row['hua_object_id']}.png".replace("/", "_")
        path = sample_dir / stem
        draw_sample(row, base_png, path)
        all_day_sample_paths.append(str(path))

    write_literature_note(out / "streamline_vs_jet_meander_literature.md")

    metadata = {
        "result_root": str(root),
        "detection_dir": str(det),
        "output_dir": str(out),
        "day": args.day,
        "surface_global_count": int(len(surface)),
        "surface_kuroshio_count": int(len(kuro_surface)),
        "surface_kuroshio_day_count": int(len(day_kuro)),
        "figures": {
            "seed_fate_map": str(fate_png),
            "sample_rejected_seed_figures": sample_paths,
            "sample_rejected_seed_figures_all_days": all_day_sample_paths,
        },
        "parquet_reader": "bundled_python_fastparquet_cache_bridge",
        "environment_note": "OFES_detection conda NumPy is blocked by Windows application control; diagnostics used bundled Python plus cached fastparquet/cramjam/fsspec packages.",
    }
    write_json(out / "diagnostic_manifest.json", metadata)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
