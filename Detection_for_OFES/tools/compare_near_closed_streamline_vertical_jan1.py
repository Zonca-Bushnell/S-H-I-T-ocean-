"""Compare tangent-gated and near-closed-streamline Jan1 vertical tracks."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from netCDF4 import Dataset
from PIL import Image, ImageDraw, ImageFont


REGIONS = {
    "kuroshio_extension": (140.0, 180.0, 28.0, 40.0),
    "south_pacific_stcc": (165.0, 230.0, -29.0, -21.0),
    "taiwan_hawaii_corridor": (122.0, 203.0, 18.0, 27.0),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--near-root", type=Path, required=True)
    parser.add_argument("--velocity-file", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--day", default="1991-01-01")
    return parser.parse_args()


def as_bool(values: pd.Series) -> pd.Series:
    return values.fillna(False).astype(str).str.strip().str.lower().isin(("1", "true", "yes"))


def read_centers(root: Path, day: str) -> pd.DataFrame:
    path = root / "raw_detection" / "daily_runs" / day.replace("-", "") / "centers_hua_style.csv"
    frame = pd.read_csv(path)
    frame["hua_object_id"] = frame["hua_object_id"].astype(str)
    return frame


def summaries(centers: pd.DataFrame) -> pd.DataFrame:
    accepted = centers.loc[as_bool(centers["hua_pass"])].copy()
    records: list[dict[str, object]] = []
    for object_id, track in accepted.groupby("hua_object_id", sort=False):
        track = track.sort_values("depth_index")
        all_rows = centers.loc[centers["hua_object_id"].eq(object_id)].sort_values("depth_index")
        failed = all_rows.loc[~as_bool(all_rows["hua_pass"])]
        terminal = failed.iloc[0] if not failed.empty else None
        surface = track.iloc[0]
        lon0, lat0 = float(surface["center_lon"]), float(surface["center_lat"])
        dx = ((track["center_lon"].to_numpy(float) - lon0 + 180.0) % 360.0 - 180.0) * 111.32 * np.cos(np.deg2rad(lat0))
        dy = (track["center_lat"].to_numpy(float) - lat0) * 111.32
        records.append({
            "hua_object_id": object_id,
            "polarity": str(surface.get("polarity", "")),
            "surface_lon": lon0,
            "surface_lat": lat0,
            "pass_layers": int(len(track)),
            "max_depth_m": float(track["depth_m"].max()),
            "max_displacement_km": float(np.nanmax(np.hypot(dx, dy))),
            "first_failure_depth_m": float(terminal["depth_m"]) if terminal is not None else np.nan,
            "first_hard_failure": str(terminal.get("first_hard_failure", "")) if terminal is not None else "",
        })
    return pd.DataFrame(records)


def region_mask(frame: pd.DataFrame, box: tuple[float, float, float, float]) -> pd.Series:
    west, east, south, north = box
    return (frame["surface_lon"] % 360.0).between(west, east) & frame["surface_lat"].between(south, north)


def local_xy(lon: np.ndarray, lat: np.ndarray, lon0: float, lat0: float) -> tuple[np.ndarray, np.ndarray]:
    x = ((lon - lon0 + 180.0) % 360.0 - 180.0) * 111.32 * np.cos(np.deg2rad(lat0))
    return x, (lat - lat0) * 111.32


def font(size: int) -> ImageFont.ImageFont:
    for candidate in (r"C:\Windows\Fonts\arial.ttf", r"C:\Windows\Fonts\segoeui.ttf"):
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def depth_color(value: float, vmax: float) -> tuple[int, int, int]:
    fraction = float(np.clip(value / max(vmax, 1.0), 0.0, 1.0))
    return int(68 * (1 - fraction) + 32 * fraction), int(1 + 170 * fraction), int(84 * (1 - fraction) + 100 * fraction)


def save_image(image: Image.Image, output: Path) -> None:
    image.save(output.with_suffix(".png"))
    image.convert("RGB").save(output.with_suffix(".pdf"), "PDF", resolution=180)


def mapped(point: tuple[float, float], box: tuple[int, int, int, int], xlim: tuple[float, float], ylim: tuple[float, float]) -> tuple[float, float]:
    x, y = point
    px = box[0] + (box[2] - box[0]) * (x - xlim[0]) / (xlim[1] - xlim[0])
    py = box[3] - (box[3] - box[1]) * (y - ylim[0]) / (ylim[1] - ylim[0])
    return px, py


def plot_global(summary: pd.DataFrame, output: Path, day: str) -> None:
    image = Image.new("RGB", (2100, 850), "white")
    draw = ImageDraw.Draw(image)
    title, label = font(28), font(16)
    panels = [(80, 110, 970, 740), (1080, 110, 1970, 740)]
    vmax = max(float(summary[["max_depth_m_baseline", "max_depth_m_near"]].quantile(0.99).max()), 1.0)
    for panel, mode, heading in zip(panels, ("baseline", "near"), ("tangent45_fraction35", "near_closed_streamline")):
        draw.rectangle(panel, fill="#f5f6f7", outline="#1f2933", width=2)
        for lon in range(0, 361, 60):
            x = panel[0] + (panel[2] - panel[0]) * lon / 360.0
            draw.line((x, panel[1], x, panel[3]), fill="#cbd5e1")
        for lat in range(-60, 61, 30):
            y = panel[3] - (panel[3] - panel[1]) * (lat + 78) / 156.0
            draw.line((panel[0], y, panel[2], y), fill="#cbd5e1")
        part = summary
        for _, row in part.iterrows():
            x = panel[0] + (panel[2] - panel[0]) * (float(row["surface_lon_near"]) % 360.0) / 360.0
            y = panel[3] - (panel[3] - panel[1]) * (float(row["surface_lat_near"]) + 78.0) / 156.0
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=depth_color(float(row[f"max_depth_m_{mode}"]), vmax))
        failure = part.loc[part[f"first_hard_failure_{mode}"].isin(("invalid_velocity", "no_closed_streamline"))]
        colors = {"invalid_velocity": "#d62728", "no_closed_streamline": "#ff8c00"}
        for reason, group in failure.groupby(f"first_hard_failure_{mode}"):
            for _, row in group.iterrows():
                x = panel[0] + (panel[2] - panel[0]) * (float(row["surface_lon_near"]) % 360.0) / 360.0
                y = panel[3] - (panel[3] - panel[1]) * (float(row["surface_lat_near"]) + 78.0) / 156.0
                draw.line((x - 4, y - 4, x + 4, y + 4), fill=colors[reason], width=2)
                draw.line((x - 4, y + 4, x + 4, y - 4), fill=colors[reason], width=2)
        draw.text((panel[0], 72), heading, font=font(22), fill="#17202a")
    draw.text((80, 25), f"Jan1 vertical continuation comparison | {day}", font=title, fill="#17202a")
    draw.text((80, 770), f"centers colored by max accepted depth; red x = invalid velocity; orange x = no closed streamline; scale 0-{vmax:.0f} m (p99)", font=label, fill="#334155")
    save_image(image, output)


def plot_region_tracks(
    baseline: pd.DataFrame, near: pd.DataFrame, summary: pd.DataFrame, name: str, box: tuple[float, float, float, float], output: Path
) -> None:
    part = summary.loc[region_mask(summary, box)]
    lon0, lat0 = (box[0] + box[1]) / 2.0, (box[2] + box[3]) / 2.0
    west_x, _ = local_xy(np.array([box[0], box[1]]), np.array([lat0, lat0]), lon0, lat0)
    _, south_y = local_xy(np.array([lon0, lon0]), np.array([box[2], box[3]]), lon0, lat0)
    image = Image.new("RGB", (1400, 800), "white")
    draw = ImageDraw.Draw(image)
    panel = (120, 100, 1320, 700)
    draw.rectangle(panel, fill="#f7f8fa", outline="#17202a", width=2)
    xlim, ylim = tuple(west_x), tuple(south_y)
    for _, row in part.iterrows():
        for source, style, alpha in ((baseline, "--", 0.35), (near, "-", 0.85)):
            track = source.loc[(source["hua_object_id"] == row["hua_object_id"]) & as_bool(source["hua_pass"])].sort_values("depth_index")
            if track.empty:
                continue
            x, y = local_xy(track["center_lon"].to_numpy(float), track["center_lat"].to_numpy(float), lon0, lat0)
            points = [mapped((a, b), panel, xlim, ylim) for a, b in zip(x, y)]
            color = depth_color(float(track["depth_m"].max()), 5000.0)
            draw.line(points, fill=(130, 130, 130) if style == "--" else color, width=1 if style == "--" else 2)
    draw.text((120, 35), f"{name.replace('_', ' ')} | gray=tangent45 tracks; color=near-closed tracks", font=font(24), fill="#17202a")
    draw.text((120, 725), "local equal-distance coordinates (km)", font=font(16), fill="#334155")
    save_image(image, output)


def _boundary_arrays(row: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    try:
        ii = np.asarray([float(value) for value in str(row["streamline_boundary_i"]).split(";") if value], dtype=float)
        jj = np.asarray([float(value) for value in str(row["streamline_boundary_j"]).split(";") if value], dtype=float)
        return ii, jj
    except (TypeError, ValueError):
        return np.asarray([]), np.asarray([])


def plot_representative(
    baseline: pd.DataFrame, near: pd.DataFrame, merged: pd.DataFrame, velocity_file: Path, name: str, box: tuple[float, float, float, float], output: Path
) -> dict[str, object] | None:
    available = merged.loc[region_mask(merged.rename(columns={"surface_lon_near": "surface_lon", "surface_lat_near": "surface_lat"}), box)].copy()
    available = available.loc[available["max_depth_gain_m"].gt(0)]
    if available.empty:
        return None
    target = available.sort_values(["max_depth_gain_m", "max_depth_m_near"], ascending=False).iloc[0]
    object_id = str(target["hua_object_id"])
    new_track = near.loc[(near["hua_object_id"] == object_id) & as_bool(near["hua_pass"])].sort_values("depth_index")
    old_track = baseline.loc[(baseline["hua_object_id"] == object_id) & as_bool(baseline["hua_pass"])].sort_values("depth_index")
    layer = new_track.iloc[-1]
    ii, jj = _boundary_arrays(layer)
    with Dataset(velocity_file) as ds:
        lon = np.asarray(ds.variables["longitude"][:], dtype=float)
        lat = np.asarray(ds.variables["latitude"][:], dtype=float)
    ci, cj = float(layer["center_i_refined"]), float(layer["center_j_refined"])
    lon0 = float(layer["center_lon"])
    lat0 = float(layer["center_lat"])
    radius = max(float(layer["accepted_radius_cells"]), 2.0)
    half = int(max(16, np.ceil(radius * 2.5)))
    x0, x1 = max(0, int(ci) - half), min(len(lon), int(ci) + half + 1)
    y0, y1 = max(0, int(cj) - half), min(len(lat), int(cj) + half + 1)
    xx, yy = np.meshgrid(lon[x0:x1], lat[y0:y1])
    x, y = local_xy(xx, yy, lon0, lat0)
    xlim = (float(np.nanmin(x)), float(np.nanmax(x)))
    ylim = (float(np.nanmin(y)), float(np.nanmax(y)))
    image = Image.new("RGB", (1800, 800), "white")
    draw = ImageDraw.Draw(image)
    panels = [(100, 120, 840, 700), (960, 120, 1700, 700)]
    for panel, track, title in ((panels[0], old_track, "tangent45_fraction35"), (panels[1], new_track, "near_closed_streamline")):
        draw.rectangle(panel, fill="#f7f8fa", outline="#17202a", width=2)
        draw.line((panel[0], (panel[1] + panel[3]) / 2, panel[2], (panel[1] + panel[3]) / 2), fill="#cbd5e1")
        draw.line(((panel[0] + panel[2]) / 2, panel[1], (panel[0] + panel[2]) / 2, panel[3]), fill="#cbd5e1")
        tx, ty = local_xy(track["center_lon"].to_numpy(float), track["center_lat"].to_numpy(float), lon0, lat0)
        draw.line([mapped((a, b), panel, xlim, ylim) for a, b in zip(tx, ty)], fill="#161616", width=3)
        if panel == panels[1] and len(ii) == len(jj) and len(ii) >= 3:
            bx, by = local_xy(lon[np.clip(np.rint(ii).astype(int), 0, len(lon) - 1)], lat[np.clip(np.rint(jj).astype(int), 0, len(lat) - 1)], lon0, lat0)
            draw.line([mapped((a, b), panel, xlim, ylim) for a, b in zip(bx, by)], fill="#c51b7d", width=3)
        cx, cy = mapped((0.0, 0.0), panel, xlim, ylim)
        draw.line((cx - 6, cy, cx + 6, cy), fill="#d62728", width=2)
        draw.line((cx, cy - 6, cx, cy + 6), fill="#d62728", width=2)
        draw.text((panel[0], 84), title, font=font(20), fill="#17202a")
    draw.text((100, 35), f"{name.replace('_', ' ')} representative {object_id} | near deepest layer {float(layer['depth_m']):.1f} m", font=font(24), fill="#17202a")
    draw.text((100, 730), "black: accepted center track; magenta: saved near-closed streamline; red cross: deepest near center", font=font(16), fill="#334155")
    save_image(image, output)
    return {"region": name, "hua_object_id": object_id, "near_deepest_depth_m": float(layer["depth_m"]), "depth_gain_m": float(target["max_depth_gain_m"])}


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    baseline = read_centers(args.baseline_root, args.day)
    near = read_centers(args.near_root, args.day)
    baseline_summary = summaries(baseline)
    near_summary = summaries(near)
    baseline_summary.to_csv(args.output_root / "tangent45_fraction35_object_summary.csv", index=False)
    near_summary.to_csv(args.output_root / "near_closed_streamline_object_summary.csv", index=False)
    merged = baseline_summary.merge(near_summary, on="hua_object_id", suffixes=("_baseline", "_near"), validate="one_to_one")
    merged["max_depth_gain_m"] = merged["max_depth_m_near"] - merged["max_depth_m_baseline"]
    merged["pass_layer_gain"] = merged["pass_layers_near"] - merged["pass_layers_baseline"]
    merged.to_csv(args.output_root / "tangent_vs_near_closed_object_comparison.csv", index=False)
    plot_global(merged, args.output_root / "global_tangent_vs_near_closed", args.day)
    selected: list[dict[str, object]] = []
    for name, box in REGIONS.items():
        plot_region_tracks(baseline, near, merged.rename(columns={"surface_lon_near": "surface_lon", "surface_lat_near": "surface_lat"}), name, box, args.output_root / f"{name}_track_comparison")
        picked = plot_representative(baseline, near, merged, args.velocity_file, name, box, args.output_root / f"{name}_representative_flowline_comparison")
        if picked:
            selected.append(picked)
    payload = {
        "day": args.day,
        "baseline_objects": int(len(baseline_summary)),
        "near_closed_objects": int(len(near_summary)),
        "median_depth_gain_m": float(merged["max_depth_gain_m"].median()),
        "representatives": selected,
    }
    (args.output_root / "comparison_manifest.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
