"""Summarize and plot a QC-sourced one-day Hua vertical-extension diagnostic."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


REGIONS = {
    "kuroshio_extension": (140.0, 180.0, 28.0, 40.0),
    "south_pacific_stcc": (165.0, 230.0, -29.0, -21.0),
    "taiwan_hawaii_corridor": (122.0, 203.0, 18.0, 27.0),
}


def as_bool(values: pd.Series) -> pd.Series:
    if values.dtype == bool:
        return values.fillna(False)
    return values.fillna(False).astype(str).str.strip().str.lower().isin(("1", "true", "yes"))


def wrapped_delta_lon(lon: np.ndarray, ref: float) -> np.ndarray:
    return (lon - ref + 180.0) % 360.0 - 180.0


def local_xy(lon: np.ndarray, lat: np.ndarray, lon0: float, lat0: float) -> tuple[np.ndarray, np.ndarray]:
    return 111.32 * np.cos(np.deg2rad(lat0)) * wrapped_delta_lon(lon, lon0), 111.32 * (lat - lat0)


def region_mask(frame: pd.DataFrame, box: tuple[float, float, float, float]) -> pd.Series:
    west, east, south, north = box
    lon = pd.to_numeric(frame["surface_lon"], errors="coerce") % 360.0
    return lon.between(west, east) & pd.to_numeric(frame["surface_lat"], errors="coerce").between(south, north)


def percentile(series: pd.Series, q: float) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
    return float(np.percentile(values, q)) if values.size else float("nan")


def summarize_region(frame: pd.DataFrame, name: str) -> dict[str, object]:
    failures = frame.loc[frame["first_hard_failure"].notna(), "first_hard_failure"].value_counts().to_dict()
    return {
        "region": name,
        "objects": int(len(frame)),
        "cyclonic": int((frame["polarity"].astype(str) == "cyclonic").sum()),
        "anticyclonic": int((frame["polarity"].astype(str) == "anticyclonic").sum()),
        "max_depth_m_p10": percentile(frame["max_depth_m"], 10),
        "max_depth_m_median": percentile(frame["max_depth_m"], 50),
        "max_depth_m_p90": percentile(frame["max_depth_m"], 90),
        "pass_layers_median": percentile(frame["pass_layers"], 50),
        "max_displacement_km_median": percentile(frame["max_displacement_km"], 50),
        "max_displacement_km_p90": percentile(frame["max_displacement_km"], 90),
        "first_failure_reasons": json.dumps(failures, ensure_ascii=False, sort_keys=True),
    }


def font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in (r"C:\Windows\Fonts\arial.ttf", r"C:\Windows\Fonts\segoeui.ttf"):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def depth_color(value: float, vmax: float) -> tuple[int, int, int]:
    t = float(np.clip(value / max(vmax, 1.0), 0.0, 1.0))
    return int(68 - 38 * t), int(25 + 205 * t), int(84 + 90 * (1.0 - t))


def save_image(image: Image.Image, output: Path) -> None:
    image.save(output.with_suffix(".png"))
    image.convert("RGB").save(output.with_suffix(".pdf"), "PDF", resolution=180)


def plot_global(summary: pd.DataFrame, output: Path, day: str) -> None:
    image = Image.new("RGB", (1800, 860), "white")
    draw = ImageDraw.Draw(image)
    title, label = font(30), font(18)
    box = (95, 100, 1660, 780)
    draw.rectangle(box, fill="#f3f5f7", outline="#17202a", width=2)
    for lon in range(0, 361, 60):
        x = box[0] + (box[2] - box[0]) * lon / 360
        draw.line((x, box[1], x, box[3]), fill="#c7ced4")
    for lat in range(-60, 61, 30):
        y = box[3] - (box[3] - box[1]) * (lat + 78) / 156
        draw.line((box[0], y, box[2], y), fill="#c7ced4")
    vmax = max(float(np.percentile(summary["max_depth_m"], 99)), 1.0)
    deepest = float(summary["max_depth_m"].max())
    for _, row in summary.iterrows():
        x = box[0] + (box[2] - box[0]) * (float(row["surface_lon"]) % 360) / 360
        y = box[3] - (box[3] - box[1]) * (float(row["surface_lat"]) + 78) / 156
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=depth_color(float(row["max_depth_m"]), vmax))
    draw.text((95, 28), f"QC-surface Hua vertical extension | maximum accepted depth | {day}", font=title, fill="#17202a")
    draw.text((95, 800), f"surface QC objects={len(summary)}; no persistence; centers colored by maximum accepted depth", font=label, fill="#334155")
    for k in range(5):
        y = 150 + k * 105
        draw.rectangle((1700, y, 1730, y + 95), fill=depth_color(vmax * (1 - (k + 0.5) / 5), vmax))
    draw.text((1680, 110), "depth", font=label, fill="#17202a")
    draw.text((1680, 680), f"0-{vmax:.0f} m (p99)", font=label, fill="#17202a")
    draw.text((95, 825), f"Color scale clipped at p99; deepest individual object={deepest:.0f} m", font=label, fill="#334155")
    save_image(image, output)


def plot_region(summary: pd.DataFrame, passed: pd.DataFrame, name: str, box: tuple[float, float, float, float], output: Path, day: str) -> None:
    subset = summary.loc[region_mask(summary, box)].copy()
    west, east, south, north = box
    lon0, lat0 = (west + east) / 2.0, (south + north) / 2.0
    xlim, ylim = local_xy(np.array([west, east]), np.array([lat0, lat0]), lon0, lat0)[0], local_xy(np.array([lon0, lon0]), np.array([south, north]), lon0, lat0)[1]
    vmax = max(float(np.percentile(summary["max_depth_m"], 99)), 1.0)
    image = Image.new("RGB", (2100, 800), "white")
    draw = ImageDraw.Draw(image)
    title, label = font(26), font(16)
    panels = [(70, 120, 770, 690), (850, 120, 1400, 690), (1480, 120, 2030, 690)]
    for panel in panels:
        draw.rectangle(panel, fill="#f7f8fa", outline="#17202a", width=2)
        draw.line((panel[0], (panel[1] + panel[3]) / 2, panel[2], (panel[1] + panel[3]) / 2), fill="#c7ced4")
        draw.line(((panel[0] + panel[2]) / 2, panel[1], (panel[0] + panel[2]) / 2, panel[3]), fill="#c7ced4")
    max_depth = max(float(passed["depth_m"].max()), 1.0)
    def map_point(panel: tuple[int, int, int, int], px: float, py: float, xmin: float, xmax: float, ymin: float, ymax: float) -> tuple[float, float]:
        return panel[0] + (panel[2] - panel[0]) * (px - xmin) / (xmax - xmin), panel[3] - (panel[3] - panel[1]) * (py - ymin) / (ymax - ymin)
    for _, obj in subset.iterrows():
        track = passed[passed["hua_object_id"].astype(str).eq(str(obj["hua_object_id"]))].sort_values("depth_index")
        if track.empty:
            continue
        x, y = local_xy(track["center_lon"].to_numpy(float), track["center_lat"].to_numpy(float), lon0, lat0)
        dx, dy = local_xy(track["center_lon"].to_numpy(float), track["center_lat"].to_numpy(float), float(obj["surface_lon"]), float(obj["surface_lat"]))
        color = depth_color(float(obj["max_depth_m"]), vmax)
        map_track = [map_point(panels[0], a, b, xlim.min(), xlim.max(), ylim.min(), ylim.max()) for a, b in zip(x, y)]
        draw.line(map_track, fill=color, width=3)
        draw.ellipse((map_track[0][0] - 4, map_track[0][1] - 4, map_track[0][0] + 4, map_track[0][1] + 4), fill=color)
        span = max(float(np.max(np.abs(np.r_[dx, dy]))), 1.0)
        ew_track = [map_point(panels[1], a, -b, -span, span, -max_depth, 0) for a, b in zip(dx, track["depth_m"])]
        ns_track = [map_point(panels[2], a, -b, -span, span, -max_depth, 0) for a, b in zip(dy, track["depth_m"])]
        draw.line(ew_track, fill=color, width=3)
        draw.line(ns_track, fill=color, width=3)
        for point in (ew_track[0], ns_track[0]):
            draw.ellipse((point[0] - 3, point[1] - 3, point[0] + 3, point[1] + 3), fill=color)
    draw.text((70, 32), f"{name.replace('_', ' ')} | QC-surface Hua vertical continuation | {day} | n={len(subset)}", font=title, fill="#17202a")
    draw.text((250, 710), "Center tracks (local equal-distance km)", font=label, fill="#334155")
    draw.text((920, 710), "East-west displacement vs depth", font=label, fill="#334155")
    draw.text((1530, 710), "North-south displacement vs depth", font=label, fill="#334155")
    save_image(image, output)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize a QC-surface Hua vertical-extension run.")
    parser.add_argument("--vertical-root", type=Path, required=True)
    parser.add_argument("--day", default="1991-01-01")
    args = parser.parse_args()
    compact = args.day.replace("-", "")
    run_dir = args.vertical_root / "raw_detection" / "daily_runs" / compact
    centers = pd.read_csv(run_dir / "centers_hua_style.csv")
    passed = centers[as_bool(centers["hua_pass"])].copy()
    passed["hua_object_id"] = passed["hua_object_id"].astype(str)
    passed = passed.sort_values(["hua_object_id", "depth_index"])
    all_rows = centers.copy()
    all_rows["hua_object_id"] = all_rows["hua_object_id"].astype(str)
    records: list[dict[str, object]] = []
    for object_id, track in passed.groupby("hua_object_id", sort=False):
        track = track.sort_values("depth_index")
        surface = track.iloc[0]
        failures = all_rows[(all_rows["hua_object_id"] == object_id) & ~as_bool(all_rows["hua_pass"])].sort_values("depth_index")
        terminal = failures.iloc[0] if not failures.empty else None
        dx, dy = local_xy(track["center_lon"].to_numpy(float), track["center_lat"].to_numpy(float), float(surface["center_lon"]), float(surface["center_lat"]))
        records.append({
            "hua_object_id": object_id, "polarity": str(surface.get("polarity", "")),
            "surface_lon": float(surface["center_lon"]), "surface_lat": float(surface["center_lat"]),
            "pass_layers": int(len(track)), "max_depth_m": float(track["depth_m"].max()),
            "final_hua_radius_cells": float(track.iloc[-1].get("accepted_radius_cells", np.nan)),
            "max_displacement_km": float(np.nanmax(np.hypot(dx, dy))),
            "first_failure_depth_m": float(terminal["depth_m"]) if terminal is not None else np.nan,
            "first_hard_failure": str(terminal.get("first_hard_failure", "")) if terminal is not None else "",
        })
    summary = pd.DataFrame(records)
    output = args.vertical_root
    output.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output / "vertical_object_summary.csv", index=False)
    region_rows = [summarize_region(summary, "global")]
    for name, box in REGIONS.items():
        region_rows.append(summarize_region(summary.loc[region_mask(summary, box)], name))
    region_summary = pd.DataFrame(region_rows)
    region_summary.to_csv(output / "vertical_region_summary.csv", index=False)
    payload = {"day": args.day, "surface_objects": int(len(summary)), "regions": region_rows}
    (output / "vertical_region_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "vertical_object_summary.json").write_text(summary.to_json(orient="records", force_ascii=False, indent=2), encoding="utf-8")
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    plot_global(summary, figures / f"global_max_depth_{compact}", args.day)
    for name, box in REGIONS.items():
        plot_region(summary, passed, name, box, figures / f"{name}_vertical_tracks_{compact}", args.day)
    print(json.dumps({"objects": len(summary), "figures": str(figures)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
