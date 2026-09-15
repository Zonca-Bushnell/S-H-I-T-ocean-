from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from netCDF4 import Dataset, date2num
from PIL import Image, ImageDraw, ImageFont

from .plot_meta4_three_region_overview import find_time_indices


DEFAULT_OFES_ROOT = Path(r"E:\DATA\01_Eddy_correspond\02_OFES\origin_ssh_effective_contour_primary_meso10d_50_500km_smoke_19910101")
DEFAULT_META_ROOT = Path(r"F:\Eddy\Eddy\META4.0_DT_allsat")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir) if args.output_dir else Path(args.ofes_root) / "diagnostics" / "ofes_meta_filter_density"
    out_dir.mkdir(parents=True, exist_ok=True)
    ofes = load_ofes(Path(args.ofes_root), args.ofes_day)
    meta = load_meta(Path(args.meta_root), args.meta_day)
    summary = build_summary(ofes, meta)
    csv_path = out_dir / "ofes_meta_distribution_summary.csv"
    pd.DataFrame(summary).to_csv(csv_path, index=False)
    json_path = out_dir / "ofes_meta_distribution_summary.json"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    fig_path = draw_summary_figure(ofes, meta, out_dir, args.ofes_day, args.meta_day)
    print(json.dumps({"summary_csv": str(csv_path), "summary_json": str(json_path), "figure": str(fig_path)}, ensure_ascii=False, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare OFES SSH-primary meso-filter catalog density against META4.0 daily objects.")
    parser.add_argument("--ofes-root", type=Path, default=DEFAULT_OFES_ROOT)
    parser.add_argument("--meta-root", type=Path, default=DEFAULT_META_ROOT)
    parser.add_argument("--ofes-day", default="1991-01-01")
    parser.add_argument("--meta-day", default="1993-01-01")
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def load_ofes(root: Path, day: str) -> pd.DataFrame:
    centers = pd.read_csv(root / "centers_hua_style.csv")
    structures = pd.read_csv(root / "structures_hua_style.csv")
    centers["date"] = pd.to_datetime(centers["date"]).dt.strftime("%Y-%m-%d")
    structures["date"] = pd.to_datetime(structures["date"]).dt.strftime("%Y-%m-%d")
    surface = centers[(centers["date"].eq(day)) & (centers["depth_index"].astype(int).eq(0)) & (centers["hua_pass"].astype(bool))].copy()
    radius = structures[structures["date"].eq(day)][["hua_object_id", "radius_km"]].drop_duplicates("hua_object_id")
    surface = surface.merge(radius, on="hua_object_id", how="left")
    surface["dataset"] = "OFES_meso10d_50_500km"
    surface["amplitude_cm_proxy"] = (pd.to_numeric(surface["ssh_value_m"], errors="coerce") - pd.to_numeric(surface["ssh_contour_level"], errors="coerce")).abs()
    surface["radius_cap_flag"] = pd.to_numeric(surface["ssh_contour_radius_cells"], errors="coerce") >= 15.5
    return surface


def load_meta(meta_root: Path, day: str) -> pd.DataFrame:
    rows = []
    target_day = datetime.strptime(day, "%Y-%m-%d")
    for polarity in ["cyclonic", "anticyclonic"]:
        path = meta_root / f"META4_DT_allsat_{polarity}_19930101_20230908.nc"
        with Dataset(path) as ds:
            time_var = ds.variables["time"]
            target = date2num(target_day, units=time_var.units, calendar=getattr(time_var, "calendar", "standard"))
            idx = find_time_indices(time_var, target)
            if idx.size == 0:
                continue
            rows.append(
                pd.DataFrame(
                    {
                        "dataset": "META4.0_DT_allsat",
                        "polarity": polarity,
                        "center_lon_refined": np.mod(np.asarray(ds.variables["longitude"][idx], dtype="f8"), 360.0),
                        "center_lat_refined": np.asarray(ds.variables["latitude"][idx], dtype="f8"),
                        "radius_km": np.asarray(ds.variables["effective_radius"][idx], dtype="f8") / 1000.0,
                        "amplitude_cm_proxy": np.asarray(ds.variables["amplitude"][idx], dtype="f8") * 100.0,
                        "shape_error_percent": np.asarray(ds.variables["effective_contour_shape_error"][idx], dtype="f8"),
                    }
                )
            )
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def build_summary(ofes: pd.DataFrame, meta: pd.DataFrame) -> list[dict[str, object]]:
    boxes = {
        "global_76S_76N": (0.0, 360.0, -76.0, 76.0),
        "kuroshio": (120.0, 145.0, 20.0, 35.0),
        "npac_interior": (170.0, 210.0, 20.0, 40.0),
        "nh_mid_high": (0.0, 360.0, 35.0, 76.0),
        "sh_mid_high": (0.0, 360.0, -76.0, -35.0),
    }
    rows: list[dict[str, object]] = []
    for dataset, df in [("OFES_meso10d_50_500km", ofes), ("META4.0_DT_allsat", meta)]:
        for region, box in boxes.items():
            sub = subset_box(df, box)
            row = {
                "dataset": dataset,
                "region": region,
                "count": int(len(sub)),
                "radius_km_p05": q(sub, "radius_km", 5),
                "radius_km_p50": q(sub, "radius_km", 50),
                "radius_km_p95": q(sub, "radius_km", 95),
                "amplitude_cm_p05": q(sub, "amplitude_cm_proxy", 5),
                "amplitude_cm_p50": q(sub, "amplitude_cm_proxy", 50),
                "amplitude_cm_p95": q(sub, "amplitude_cm_proxy", 95),
            }
            if dataset.startswith("OFES"):
                row["radius_cap_fraction"] = float(np.nanmean(pd.to_numeric(sub.get("radius_cap_flag", pd.Series(dtype=float)), errors="coerce"))) if len(sub) else np.nan
                row["jet_core_overlap_gt_0p5_fraction"] = float(np.nanmean(pd.to_numeric(sub.get("jet_core_overlap_fraction", pd.Series(dtype=float)), errors="coerce") > 0.5)) if len(sub) else np.nan
            rows.append(row)
    return rows


def subset_box(df: pd.DataFrame, box: tuple[float, float, float, float]) -> pd.DataFrame:
    lon_min, lon_max, lat_min, lat_max = box
    lon_col = "center_lon_refined" if "center_lon_refined" in df else "longitude"
    lat_col = "center_lat_refined" if "center_lat_refined" in df else "latitude"
    return df[df[lon_col].between(lon_min, lon_max) & df[lat_col].between(lat_min, lat_max)].copy()


def q(df: pd.DataFrame, column: str, percentile: float) -> float:
    if column not in df or df.empty:
        return float("nan")
    vals = pd.to_numeric(df[column], errors="coerce").to_numpy("f8")
    return float(np.nanpercentile(vals, percentile)) if np.isfinite(vals).any() else float("nan")


def draw_summary_figure(ofes: pd.DataFrame, meta: pd.DataFrame, out_dir: Path, ofes_day: str, meta_day: str) -> Path:
    width, height = 1800, 1200
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    draw.text((45, 28), "OFES meso-filter vs META4.0 catalog-density diagnostics", fill=(18, 24, 38), font=font(34))
    draw.text((45, 70), f"OFES {ofes_day}: meso10d_50_500km ssh_primary; META {meta_day}: daily effective contour objects.", fill=(75, 85, 100), font=font(17))
    panels = [
        ((70, 130, 850, 500), "radius_km", "Effective radius / OFES radius (km)", (0, 250)),
        ((970, 130, 1750, 500), "amplitude_cm_proxy", "Amplitude proxy (cm)", (0, 12)),
        ((70, 620, 850, 990), "ssh_contour_radius_cells", "OFES SSH contour radius cells", (0, 17)),
        ((970, 620, 1750, 990), "ssh_contour_area_cells", "OFES SSH contour area cells", (0, 850)),
    ]
    for box, column, title, xlim in panels:
        draw_hist_panel(draw, box, ofes, meta, column, title, xlim)
    draw.text((70, 1045), "Key read: OFES Jan1 meso-filter creates many more small/intermediate closed contours than META; many OFES radii hit the current ~16-cell search cap.", fill=(20, 24, 32), font=font(22))
    out = out_dir / "ofes_meta_filter_density_diagnostics.png"
    img.save(out)
    img.save(out.with_suffix(".pdf"), "PDF", resolution=180.0)
    return out


def draw_hist_panel(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], ofes: pd.DataFrame, meta: pd.DataFrame, column: str, title: str, xlim: tuple[float, float]) -> None:
    x0, y0, x1, y1 = box
    draw.rectangle(box, outline=(35, 40, 50), width=2)
    draw.text((x0, y0 - 28), title, fill=(20, 24, 32), font=font(20))
    bins = np.linspace(xlim[0], xlim[1], 31)
    series = [("OFES", ofes, (32, 99, 220)), ("META", meta, (215, 43, 43))]
    hist_data = []
    ymax = 1.0
    for name, df, color in series:
        if column not in df:
            vals = np.asarray([], dtype="f8")
        else:
            vals = pd.to_numeric(df[column], errors="coerce").to_numpy("f8")
            vals = vals[np.isfinite(vals)]
        counts, _ = np.histogram(vals, bins=bins)
        frac = counts / max(1, counts.sum())
        ymax = max(ymax, float(frac.max()) if frac.size else 0.0)
        hist_data.append((name, frac, color))
    plot_h = y1 - y0 - 50
    plot_w = x1 - x0 - 55
    base_y = y1 - 35
    left_x = x0 + 45
    for gx in np.linspace(xlim[0], xlim[1], 6):
        px = left_x + (gx - xlim[0]) / (xlim[1] - xlim[0]) * plot_w
        draw.line((px, y0 + 18, px, base_y), fill=(220, 226, 235), width=1)
        draw.text((px - 16, base_y + 8), f"{gx:g}", fill=(75, 85, 100), font=font(13))
    for _, frac, color in hist_data:
        pts = []
        centers = 0.5 * (bins[:-1] + bins[1:])
        for xc, val in zip(centers, frac, strict=False):
            px = left_x + (xc - xlim[0]) / (xlim[1] - xlim[0]) * plot_w
            py = base_y - val / ymax * plot_h
            pts.append((px, py))
        if len(pts) > 1:
            draw.line(pts, fill=color, width=3)
    draw.line((left_x, base_y, left_x + plot_w, base_y), fill=(35, 40, 50), width=1)
    draw.text((x0 + 55, y0 + 25), "OFES", fill=(32, 99, 220), font=font(16))
    draw.text((x0 + 130, y0 + 25), "META", fill=(215, 43, 43), font=font(16))


def font(size: int) -> ImageFont.ImageFont:
    for path in [r"C:\Windows\Fonts\arial.ttf", r"C:\Windows\Fonts\calibri.ttf", r"C:\Windows\Fonts\simhei.ttf"]:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    return ImageFont.load_default()


if __name__ == "__main__":
    main()
