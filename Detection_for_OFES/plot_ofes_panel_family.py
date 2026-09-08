from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from .ofes_io import (
    ctl_path,
    expected_dta_bytes,
    open_dta_memmap,
    parse_ctl,
    read_ssh_latlon_daily_only,
    require_daily_file,
)


EARTH_RADIUS_M = 6_371_000.0
DEFAULT_RESULT_ROOT = Path(r"E:\DATA\01_Eddy_correspond\02_OFES\available_jan01_jan19")
DEFAULT_DATA_ROOT = Path(r"F:\OFES\external_OFES2")
SCIENCE_TAG = "raw_minus_jan_mean_diagnostic"


@dataclass(frozen=True)
class SelectedPanelObject:
    hua_object_id: str
    date: str
    polarity: str
    pass_layers: int
    max_jump_km: float
    max_jump_over_r: float
    first_from: int | None
    first_to: int | None
    second_from: int | None
    second_to: int | None


def main() -> None:
    args = build_parser().parse_args()
    data_root = Path(args.data_root)
    result_root = Path(args.result_root)
    output_dir = Path(args.output_dir) if args.output_dir else result_root / "figures" / "panel_family"
    output_dir.mkdir(parents=True, exist_ok=True)

    centers = pd.read_csv(result_root / "detection_hua_global_jan1991" / "centers_hua_style.csv")
    structures = pd.read_csv(result_root / "detection_hua_global_jan1991" / "structures_hua_style.csv")
    selected_rows = select_objects(centers, structures, args.hua_object_id, int(args.max_examples), float(args.jump_threshold_over_r))
    metadata_rows = []
    for idx, selected in enumerate(selected_rows, start=1):
        rows = centers[
            centers["hua_object_id"].astype(str).eq(selected.hua_object_id)
            & centers["hua_pass"].astype(bool)
        ].sort_values("depth_index").copy()
        if rows.empty:
            continue
        stem = f"ofes_panel_family_{idx:03d}_{selected.hua_object_id}_{args.right_panel_mode}"
        image_path = output_dir / f"{stem}.png"
        payload = plot_panel_family(
            data_root=data_root,
            result_root=result_root,
            selected=selected,
            object_layers=rows,
            image_path=image_path,
            half_width_deg=float(args.half_width_deg),
            section_half_width_km=float(args.section_half_width_km),
            section_points=int(args.section_points),
            right_panel_mode=str(args.right_panel_mode),
            backend=str(args.backend),
            horizontal_smooth_sigma_cells=float(args.horizontal_smooth_sigma_cells),
        )
        meta = {
            **asdict(selected),
            **payload,
            "jump_threshold_over_r": float(args.jump_threshold_over_r),
            "image_path": str(image_path),
        }
        metadata_rows.append(meta)
        (output_dir / f"{stem}.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[panel-family] {selected.hua_object_id} -> {image_path}", flush=True)
    if metadata_rows:
        pd.DataFrame(metadata_rows).to_csv(output_dir / "selected_objects_metadata.csv", index=False)
        (output_dir / "selected_objects_metadata.json").write_text(
            json.dumps(metadata_rows, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot OFES Hua object-day panel-family diagnostics.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--hua-object-id", default=None, help="Optional exact object id, e.g. 19910101_00001.")
    parser.add_argument("--max-examples", type=int, default=3)
    parser.add_argument("--jump-threshold-over-r", type=float, default=0.20)
    parser.add_argument("--half-width-deg", type=float, default=1.4)
    parser.add_argument("--section-half-width-km", type=float, default=180.0)
    parser.add_argument("--section-points", type=int, default=181)
    parser.add_argument(
        "--right-panel-mode",
        choices=["normal_horizontal_velocity", "horizontal_speed", "signed_horizontal_speed"],
        default="normal_horizontal_velocity",
        help="Zhe-style right panels 8/9/10/11.",
    )
    parser.add_argument("--horizontal-smooth-sigma-cells", type=float, default=0.8)
    parser.add_argument("--backend", choices=["matplotlib", "pillow"], default="matplotlib")
    return parser


def select_objects(
    centers: pd.DataFrame,
    structures: pd.DataFrame,
    requested_id: str | None,
    max_examples: int,
    jump_threshold_over_r: float,
) -> list[SelectedPanelObject]:
    if requested_id:
        part = structures[structures["hua_object_id"].astype(str).eq(str(requested_id))]
        if part.empty:
            raise ValueError(f"No passed structure rows for hua_object_id={requested_id}")
        return [summarize_object(part, jump_threshold_over_r)]

    candidates = []
    for _, part in structures.groupby("hua_object_id", sort=False):
        if len(part) < 2:
            continue
        candidates.append(summarize_object(part, jump_threshold_over_r))
    candidates.sort(key=lambda item: (item.max_jump_over_r, item.pass_layers), reverse=True)
    return candidates[: max(1, int(max_examples))]


def summarize_object(part: pd.DataFrame, jump_threshold_over_r: float) -> SelectedPanelObject:
    part = part.sort_values("depth_index").copy()
    lon_col = "center_lon_refined" if "center_lon_refined" in part.columns else "center_lon"
    lat_col = "center_lat_refined" if "center_lat_refined" in part.columns else "center_lat"
    dx_km, dy_km = meters_per_degree(float(part.iloc[0][lat_col]))
    lon = part[lon_col].to_numpy(dtype="f8")
    lat = part[lat_col].to_numpy(dtype="f8")
    radius = float(np.nanmedian(part["radius_km"].to_numpy(dtype="f8")))
    if not np.isfinite(radius) or radius <= 0:
        radius = 1.0
    jumps = []
    for a, b in zip(range(len(part) - 1), range(1, len(part))):
        dlon = lon_delta_deg(lon[b], lon[a])
        dlat = lat[b] - lat[a]
        dist = math.hypot(dlon * dx_km / 1000.0, dlat * dy_km / 1000.0)
        jumps.append((dist / radius, dist, int(part.iloc[a]["depth_index"]), int(part.iloc[b]["depth_index"])))
    jumps.sort(reverse=True)
    first = jumps[0] if jumps else (0.0, 0.0, None, None)
    second = next((j for j in jumps[1:] if j[0] >= jump_threshold_over_r), None)
    return SelectedPanelObject(
        hua_object_id=str(part.iloc[0]["hua_object_id"]),
        date=str(part.iloc[0]["date"]),
        polarity=str(part.iloc[0]["polarity"]),
        pass_layers=int(len(part)),
        max_jump_km=float(first[1]),
        max_jump_over_r=float(first[0]),
        first_from=first[2],
        first_to=first[3],
        second_from=second[2] if second else None,
        second_to=second[3] if second else None,
    )


def plot_panel_family(
    *,
    data_root: Path,
    result_root: Path,
    selected: SelectedPanelObject,
    object_layers: pd.DataFrame,
    image_path: Path,
    half_width_deg: float,
    section_half_width_km: float,
    section_points: int,
    right_panel_mode: str,
    backend: str,
    horizontal_smooth_sigma_cells: float,
) -> dict[str, object]:
    if backend == "pillow":
        return plot_panel_family_pillow(
            data_root=data_root,
            result_root=result_root,
            selected=selected,
            object_layers=object_layers,
            image_path=image_path,
            half_width_deg=half_width_deg,
            section_half_width_km=section_half_width_km,
            section_points=section_points,
        )
    return plot_panel_family_matplotlib(
        data_root=data_root,
        result_root=result_root,
        selected=selected,
        object_layers=object_layers,
        image_path=image_path,
        half_width_deg=half_width_deg,
        section_half_width_km=section_half_width_km,
        section_points=section_points,
        right_panel_mode=right_panel_mode,
        horizontal_smooth_sigma_cells=horizontal_smooth_sigma_cells,
    )


def plot_panel_family_matplotlib(
    *,
    data_root: Path,
    result_root: Path,
    selected: SelectedPanelObject,
    object_layers: pd.DataFrame,
    image_path: Path,
    half_width_deg: float,
    section_half_width_km: float,
    section_points: int,
    right_panel_mode: str,
    horizontal_smooth_sigma_cells: float,
) -> dict[str, object]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    u_meta = parse_ctl(ctl_path(data_root, "u"))
    v_meta = parse_ctl(ctl_path(data_root, "v"))
    lon = u_meta.x.values
    lat = u_meta.y.values
    depth = u_meta.z.values
    nlev = int(json.loads((result_root / "anomaly_jan1991" / "u_jan1991_mean_cms.json").read_text())["shape"][2])
    u_mean = np.memmap(result_root / "anomaly_jan1991" / "u_jan1991_mean_cms.dat", dtype="float32", mode="r", shape=(u_meta.x.count, u_meta.y.count, nlev), order="F")
    v_mean = np.memmap(result_root / "anomaly_jan1991" / "v_jan1991_mean_cms.dat", dtype="float32", mode="r", shape=(v_meta.x.count, v_meta.y.count, nlev), order="F")
    u_raw = open_dta_memmap(require_daily_file(data_root, "u", selected.date, expected_dta_bytes(u_meta)), u_meta)
    v_raw = open_dta_memmap(require_daily_file(data_root, "v", selected.date, expected_dta_bytes(v_meta)), v_meta)

    layers = object_offsets(object_layers)
    surface = layers.iloc[0]
    lon_col = "center_lon_refined" if "center_lon_refined" in layers.columns else "center_lon"
    lat_col = "center_lat_refined" if "center_lat_refined" in layers.columns else "center_lat"
    center_lon = float(surface[lon_col])
    center_lat = float(surface[lat_col])
    lon_idx, lat_idx = window_indices(lon, lat, center_lon, center_lat, half_width_deg)
    xx, yy = relative_mesh(lon[lon_idx], lat[lat_idx], center_lon, center_lat)

    fig = plt.figure(figsize=(32, 18), constrained_layout=True)
    gs = fig.add_gridspec(
        5,
        6,
        height_ratios=[1.0, 1.0, 1.0, 1.0, 0.85],
        width_ratios=[0.9, 0.9, 1.05, 1.05, 1.05, 1.05],
    )
    ax1 = fig.add_subplot(gs[0:4, 0])
    ax2 = fig.add_subplot(gs[0:4, 1])
    axes = {
        "j1_speed_from": fig.add_subplot(gs[0, 2]),
        "j1_pressure_from": fig.add_subplot(gs[0, 3]),
        "j1_speed_to": fig.add_subplot(gs[1, 2]),
        "j1_pressure_to": fig.add_subplot(gs[1, 3]),
        "j2_speed_from": fig.add_subplot(gs[2, 2]),
        "j2_pressure_from": fig.add_subplot(gs[2, 3]),
        "j2_speed_to": fig.add_subplot(gs[3, 2]),
        "j2_pressure_to": fig.add_subplot(gs[3, 3]),
        "j1_section_from": fig.add_subplot(gs[0:2, 4]),
        "j1_section_to": fig.add_subplot(gs[0:2, 5]),
        "j2_section_from": fig.add_subplot(gs[2:4, 4]),
        "j2_section_to": fig.add_subplot(gs[2:4, 5]),
    }
    ax7 = fig.add_subplot(gs[4, :])

    plot_offset_axes(ax1, ax2, layers, selected)
    add_jump_source_markers(ax1, ax2, layers, selected)

    section_payloads: list[dict[str, np.ndarray]] = []
    section_meshes = []
    section_axes = []
    for prefix, jump_from, jump_to, panel_keys in [
        ("J1", selected.first_from, selected.first_to, ("j1_speed_from", "j1_pressure_from", "j1_speed_to", "j1_pressure_to", "j1_section_from", "j1_section_to")),
        ("J2", selected.second_from, selected.second_to, ("j2_speed_from", "j2_pressure_from", "j2_speed_to", "j2_pressure_to", "j2_section_from", "j2_section_to")),
    ]:
        if jump_from is None or jump_to is None:
            for key in panel_keys:
                hatch_unavailable(axes[key], f"no {prefix} abrupt layer discontinuity detected")
            continue

        marks = center_marks_mpl(layers, center_lon, center_lat, int(jump_from), int(jump_to))
        jump_title = f"{prefix} {int(jump_from)}->{int(jump_to)}"
        for layer_index, speed_key, pressure_key, side in [
            (int(jump_from), panel_keys[0], panel_keys[1], "upper/from"),
            (int(jump_to), panel_keys[2], panel_keys[3], "lower/to"),
        ]:
            u = uv_anomaly_window(u_raw, u_mean, layer_index, lon_idx, lat_idx)
            v = uv_anomaly_window(v_raw, v_mean, layer_index, lon_idx, lat_idx)
            speed = nan_gaussian_smooth(np.hypot(u, v), horizontal_smooth_sigma_cells)
            pressure = nan_gaussian_smooth(pressure_proxy(u, v, xx, yy, center_lat), horizontal_smooth_sigma_cells)
            speed_mesh = plot_field_mpl(
                axes[speed_key],
                xx,
                yy,
                speed,
                f"{speed_key_label(prefix, speed_key)}  {jump_title} {side}: speed |u',v'|\nk={layer_index}, z={depth[layer_index]:.0f} m",
                "coolwarm",
                symmetric=False,
                quiver=(u, v),
                center_marks=marks,
                contour=False,
            )
            pressure_mesh = plot_field_mpl(
                axes[pressure_key],
                xx,
                yy,
                pressure,
                f"{pressure_key_label(prefix, pressure_key)}  {jump_title} {side}: geostrophic p' proxy\nk={layer_index}, z={depth[layer_index]:.0f} m",
                "RdBu_r",
                symmetric=True,
                center_marks=marks,
                contour=False,
            )
            fig.colorbar(speed_mesh, ax=axes[speed_key], shrink=0.82, label="m/s")
            fig.colorbar(pressure_mesh, ax=axes[pressure_key], shrink=0.82, label="Pa proxy")

        section = section_diagnostics(u_raw, v_raw, u_mean, v_mean, layers, depth[:nlev], int(jump_from), int(jump_to), section_half_width_km, section_points)
        section_payloads.append(section)
        for key, side in [(panel_keys[4], "upper/from"), (panel_keys[5], "lower/to")]:
            mesh = plot_right_section_mpl(axes[key], section, f"{right_section_label(key)}  {jump_title} {side}", right_panel_mode)
            section_meshes.append(mesh)
            section_axes.append(axes[key])

    if section_meshes:
        label = "m/s"
        fig.colorbar(section_meshes[0], ax=section_axes, shrink=0.82, label=label)

    plot_track_panel_mpl(ax7, layers, selected)
    jump_text = f"jump {selected.first_from}->{selected.first_to}, {selected.max_jump_km:.1f} km = {selected.max_jump_over_r:.2f} R"
    if selected.second_from is not None and selected.second_to is not None:
        jump_text += f"; second {selected.second_from}->{selected.second_to}"
    fig.suptitle(
        "OFES original-style eddy discontinuity diagnostic, not representative vortex\n"
        f"object {selected.hua_object_id}, {selected.date}, {selected.polarity}; {jump_text}; "
        f"{SCIENCE_TAG}; right_panel_mode={right_panel_mode}; smoothing sigma={horizontal_smooth_sigma_cells:g} grid",
        fontsize=15,
    )
    image_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(image_path, dpi=220)
    fig.savefig(image_path.with_suffix(".pdf"))
    plt.close(fig)
    return {
        "half_width_deg": half_width_deg,
        "section_half_width_km": section_half_width_km,
        "right_panel_mode": right_panel_mode,
        "horizontal_smooth_sigma_cells": horizontal_smooth_sigma_cells,
        "plot_backend": "matplotlib_ofes_self_contained_zhe_style",
        "trajectory_panel": "object_day_vertical_center_axis_not_lifecycle_tracking",
    }


def plot_panel_family_pillow(
    *,
    data_root: Path,
    result_root: Path,
    selected: SelectedPanelObject,
    object_layers: pd.DataFrame,
    image_path: Path,
    half_width_deg: float,
    section_half_width_km: float,
    section_points: int,
) -> dict[str, object]:
    u_meta = parse_ctl(ctl_path(data_root, "u"))
    v_meta = parse_ctl(ctl_path(data_root, "v"))
    lon = u_meta.x.values
    lat = u_meta.y.values
    depth = u_meta.z.values
    nlev = int(json.loads((result_root / "anomaly_jan1991" / "u_jan1991_mean_cms.json").read_text())["shape"][2])
    u_mean = np.memmap(result_root / "anomaly_jan1991" / "u_jan1991_mean_cms.dat", dtype="float32", mode="r", shape=(u_meta.x.count, u_meta.y.count, nlev), order="F")
    v_mean = np.memmap(result_root / "anomaly_jan1991" / "v_jan1991_mean_cms.dat", dtype="float32", mode="r", shape=(v_meta.x.count, v_meta.y.count, nlev), order="F")
    u_raw = open_dta_memmap(require_daily_file(data_root, "u", selected.date, expected_dta_bytes(u_meta)), u_meta)
    v_raw = open_dta_memmap(require_daily_file(data_root, "v", selected.date, expected_dta_bytes(v_meta)), v_meta)
    ssh_mean = np.load(result_root / "anomaly_jan1991" / "ssh_jan1991_mean_cm.npy")
    ssh_anom = read_ssh_latlon_daily_only(data_root, selected.date) - ssh_mean

    layers = object_offsets(object_layers)
    surface = layers.iloc[0]
    lon_col = "center_lon_refined" if "center_lon_refined" in layers.columns else "center_lon"
    lat_col = "center_lat_refined" if "center_lat_refined" in layers.columns else "center_lat"
    center_lon = float(surface[lon_col])
    center_lat = float(surface[lat_col])
    lon_idx, lat_idx = window_indices(lon, lat, center_lon, center_lat, half_width_deg)
    xx, yy = relative_mesh(lon[lon_idx], lat[lat_idx], center_lon, center_lat)

    canvas = Image.new("RGB", (2600, 1680), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = font(26)
    small_font = font(17)
    tiny_font = font(14)
    draw.text(
        (40, 22),
        f"OFES Hua panel-family | {selected.hua_object_id} | {selected.date} | {selected.polarity} | {SCIENCE_TAG}",
        fill=(20, 24, 32),
        font=title_font,
    )
    draw.text(
        (40, 58),
        f"pass_layers={selected.pass_layers}, max_jump={selected.max_jump_km:.1f} km ({selected.max_jump_over_r:.2f} R), daily-only u/v anomaly",
        fill=(70, 78, 92),
        font=small_font,
    )

    boxes = layout_boxes()
    draw_line_panel(canvas, boxes["dx"], layers["delta_x_km"].to_numpy("f8"), layers["depth_m"].to_numpy("f8"), "1 delta x from surface", "km", small_font, tiny_font)
    draw_line_panel(canvas, boxes["dy"], layers["delta_y_km"].to_numpy("f8"), layers["depth_m"].to_numpy("f8"), "2 delta y from surface", "km", small_font, tiny_font)

    payload: dict[str, object] = {
        "half_width_deg": half_width_deg,
        "section_half_width_km": section_half_width_km,
        "plot_backend": "pillow_ofes_self_contained_old_panel_style",
        "style_alignment": "Local renderer mirrors old panel-family colors, contours, quiver, and trajectory conventions; no runtime import from Zhe/src.",
        "trajectory_panel": "object_day_vertical_center_axis_not_lifecycle_tracking",
    }
    for prefix, jump_from, jump_to, panel_keys in [
        ("J1", selected.first_from, selected.first_to, ("j1_speed_from", "j1_pressure_from", "j1_speed_to", "j1_pressure_to", "j1_section_from", "j1_section_to")),
        ("J2", selected.second_from, selected.second_to, ("j2_speed_from", "j2_pressure_from", "j2_speed_to", "j2_pressure_to", "j2_section_from", "j2_section_to")),
    ]:
        if jump_from is None or jump_to is None:
            for key in panel_keys:
                draw_empty_panel(canvas, boxes[key], f"{prefix} unavailable", small_font)
            continue
        payload[f"{prefix.lower()}_from"] = int(jump_from)
        payload[f"{prefix.lower()}_to"] = int(jump_to)
        for layer_index, speed_key, pressure_key, label in [
            (int(jump_from), panel_keys[0], panel_keys[1], "upper/from"),
            (int(jump_to), panel_keys[2], panel_keys[3], "lower/to"),
        ]:
            u = uv_anomaly_window(u_raw, u_mean, int(layer_index), lon_idx, lat_idx)
            v = uv_anomaly_window(v_raw, v_mean, int(layer_index), lon_idx, lat_idx)
            speed = np.hypot(u, v)
            pressure = pressure_proxy(u, v, xx, yy, center_lat)
            marks = center_marks(layers, center_lon, center_lat, jump_from, jump_to)
            draw_heat_panel(canvas, boxes[speed_key], xx, yy, speed, f"{prefix} {label} speed z={depth[layer_index]:.1f}m", "speed", marks, quiver=(u, v), font_main=small_font, font_small=tiny_font)
            draw_heat_panel(canvas, boxes[pressure_key], xx, yy, pressure, f"{prefix} {label} p proxy z={depth[layer_index]:.1f}m", "diverging", marks, quiver=None, font_main=small_font, font_small=tiny_font)
        section = normal_velocity_section(u_raw, v_raw, u_mean, v_mean, layers, depth[:nlev], jump_from, jump_to, section_half_width_km, section_points)
        draw_section_panel(canvas, boxes[panel_keys[4]], section, f"{prefix} upper/from normal velocity", int(jump_from), small_font, tiny_font)
        draw_section_panel(canvas, boxes[panel_keys[5]], section, f"{prefix} lower/to normal velocity", int(jump_to), small_font, tiny_font)

    draw_trajectory_panel(canvas, boxes["track"], layers, object_layers, small_font, tiny_font)
    image_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(image_path)
    return payload


def speed_key_label(prefix: str, key: str) -> str:
    return "3U" if key == "j1_speed_from" else "3L" if key == "j1_speed_to" else "5U" if key == "j2_speed_from" else "5L"


def pressure_key_label(prefix: str, key: str) -> str:
    return "4U" if key == "j1_pressure_from" else "4L" if key == "j1_pressure_to" else "6U" if key == "j2_pressure_from" else "6L"


def right_section_label(key: str) -> str:
    return {
        "j1_section_from": "8",
        "j2_section_from": "9",
        "j1_section_to": "10",
        "j2_section_to": "11",
    }[key]


def plot_offset_axes(ax_dx, ax_dy, layers: pd.DataFrame, selected: SelectedPanelObject) -> None:
    offset_values = np.abs(layers[["delta_x_km", "delta_y_km"]].to_numpy(dtype="f8"))
    offset_values = offset_values[np.isfinite(offset_values)]
    offset_xlim = max(1.0, float(np.nanmax(offset_values)) * 1.08 if offset_values.size else 1.0)
    for ax, col, title in [
        (ax_dx, "delta_x_km", "1  delta x from surface center"),
        (ax_dy, "delta_y_km", "2  delta y from surface center"),
    ]:
        ax.plot(layers[col], layers["depth_m"], "-o", color="#244a9b", lw=1.8, ms=4)
        ax.axvline(0.0, color="0.75", lw=0.8)
        if selected.first_from is not None:
            depth_row = layers[layers["depth_index"].astype(int).eq(int(selected.first_from))]
            if not depth_row.empty:
                ax.axhline(float(depth_row.iloc[0]["depth_m"]), color="tab:red", ls="--", lw=1.0, alpha=0.8)
        if selected.first_to is not None:
            depth_row = layers[layers["depth_index"].astype(int).eq(int(selected.first_to))]
            if not depth_row.empty:
                ax.axhline(float(depth_row.iloc[0]["depth_m"]), color="tab:red", ls=":", lw=1.0, alpha=0.8)
        ax.invert_yaxis()
        ax.set_xlabel("offset (km)")
        ax.set_ylabel("depth (m)")
        ax.set_xlim(-offset_xlim, offset_xlim)
        ax.set_title(title)
        ax.grid(alpha=0.25)


def add_jump_source_markers(ax_dx, ax_dy, layers: pd.DataFrame, selected: SelectedPanelObject) -> None:
    from matplotlib.patches import Rectangle

    offset_values = np.abs(layers[["delta_x_km", "delta_y_km"]].to_numpy(dtype="f8"))
    offset_values = offset_values[np.isfinite(offset_values)]
    offset_xlim = max(1.0, float(np.nanmax(offset_values)) * 1.08 if offset_values.size else 1.0)

    def row_depth(level: int | None) -> float | None:
        if level is None:
            return None
        row = layers[layers["depth_index"].astype(int).eq(int(level))]
        return None if row.empty else float(row.iloc[0]["depth_m"])

    def marker_box(col: str, z_from: float, z_to: float) -> tuple[float, float, float, float]:
        picked = []
        for z in [z_from, z_to]:
            row = layers.iloc[[int(np.nanargmin(np.abs(layers["depth_m"].to_numpy(dtype="f8") - z)))]]
            picked.append(float(row.iloc[0][col]))
        x_pad = max(3.0, 0.055 * offset_xlim, 0.18 * abs(picked[1] - picked[0]))
        depth_pad = max(4.0, 0.006 * max(abs(z_from), abs(z_to), 1.0))
        x_min = max(-0.98 * offset_xlim, min(picked) - x_pad)
        x_max = min(0.98 * offset_xlim, max(picked) + x_pad)
        return x_min, min(z_from, z_to) - depth_pad, x_max - x_min, abs(z_to - z_from) + 2.0 * depth_pad

    for levels, color in [
        ((selected.first_from, selected.first_to), "#d62728"),
        ((selected.second_from, selected.second_to), "#2ca02c"),
    ]:
        z_from = row_depth(levels[0])
        z_to = row_depth(levels[1])
        if z_from is None or z_to is None:
            continue
        for ax, col in [(ax_dx, "delta_x_km"), (ax_dy, "delta_y_km")]:
            box = marker_box(col, z_from, z_to)
            ax.add_patch(Rectangle((box[0], box[1]), box[2], box[3], fill=True, facecolor=color, edgecolor=color, linewidth=1.35, alpha=0.18, zorder=8))
            ax.add_patch(Rectangle((box[0], box[1]), box[2], box[3], fill=False, edgecolor=color, linewidth=1.35, zorder=9))


def center_marks_mpl(layers: pd.DataFrame, lon0: float, lat0: float, jump_from: int, jump_to: int) -> list[tuple[float, float, str, str, str]]:
    mx, my = meters_per_degree(lat0)
    marks = [(0.0, 0.0, "surface", "red", "+")]
    lon_col = "center_lon_refined" if "center_lon_refined" in layers.columns else "center_lon"
    lat_col = "center_lat_refined" if "center_lat_refined" in layers.columns else "center_lat"
    for level, label, color in [(jump_from, "upper side", "cyan"), (jump_to, "lower side", "yellow")]:
        row = layers[layers["depth_index"].astype(int).eq(int(level))]
        if row.empty:
            continue
        marks.append((lon_delta_deg(row.iloc[0][lon_col], lon0) * mx / 1000.0, (float(row.iloc[0][lat_col]) - lat0) * my / 1000.0, label, color, "x"))
    return marks


def mark_center_mpl(ax, x_km: float, y_km: float, label: str, color: str, marker: str) -> None:
    ax.scatter([x_km], [y_km], s=90, c=color, marker=marker, linewidths=2.0, zorder=8, label=label)


def nan_gaussian_smooth(field: np.ndarray, sigma_cells: float) -> np.ndarray:
    if sigma_cells <= 0:
        return field
    from scipy import ndimage

    arr = np.asarray(field, dtype="f8")
    mask = np.isfinite(arr)
    filled = np.where(mask, arr, 0.0)
    weights = ndimage.gaussian_filter(mask.astype("f8"), sigma=sigma_cells, mode="nearest")
    smoothed = ndimage.gaussian_filter(filled, sigma=sigma_cells, mode="nearest")
    out = np.divide(smoothed, weights, out=np.full_like(smoothed, np.nan), where=weights > 1.0e-8)
    return out


def plot_field_mpl(
    ax,
    xx: np.ndarray,
    yy: np.ndarray,
    field: np.ndarray,
    title: str,
    cmap: str,
    *,
    symmetric: bool = False,
    quiver: tuple[np.ndarray, np.ndarray] | None = None,
    center_marks: list[tuple[float, float, str, str, str]] | None = None,
    contour: bool = True,
):
    if symmetric:
        vmin, vmax = finite_limits(field, symmetric=True)
    else:
        finite = field[np.isfinite(field)]
        vmin = float(np.nanquantile(finite, 0.02)) if finite.size else 0.0
        vmax = float(np.nanquantile(finite, 0.98)) if finite.size else 1.0
    mesh = ax.pcolormesh(xx, yy, field, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    if contour:
        finite = field[np.isfinite(field)]
        if finite.size:
            lo = float(np.nanquantile(finite, 0.10))
            hi = float(np.nanquantile(finite, 0.90))
            if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                ax.contour(xx, yy, field, levels=np.linspace(lo, hi, 7), colors="0.25", linewidths=0.55, alpha=0.7)
    if quiver is not None:
        u, v = quiver
        step = max(1, int(max(u.shape) / 18))
        ax.quiver(xx[::step, ::step], yy[::step, ::step], u[::step, ::step], v[::step, ::step], color="white", alpha=0.65, scale=2.5)
    if center_marks:
        for x_km, y_km, label, color, marker in center_marks:
            mark_center_mpl(ax, x_km, y_km, label, color, marker)
    ax.axhline(0.0, color="0.85", lw=0.8)
    ax.axvline(0.0, color="0.85", lw=0.8)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("east from surface center (km)")
    ax.set_ylabel("north from surface center (km)")
    return mesh


def section_diagnostics(
    u_raw: np.memmap,
    v_raw: np.memmap,
    u_mean: np.memmap,
    v_mean: np.memmap,
    layers: pd.DataFrame,
    depth: np.ndarray,
    jump_from: int,
    jump_to: int,
    half_width_km: float,
    n_points: int,
) -> dict[str, np.ndarray]:
    lower = layers[layers["depth_index"].astype(int).eq(int(jump_from))].iloc[0]
    upper = layers[layers["depth_index"].astype(int).eq(int(jump_to))].iloc[0]
    i_col = "center_i_refined" if "center_i_refined" in layers.columns else "speed_min_i"
    j_col = "center_j_refined" if "center_j_refined" in layers.columns else "speed_min_j"
    x0, y0 = float(lower[i_col]), float(lower[j_col])
    x1, y1 = float(upper[i_col]), float(upper[j_col])
    vec = np.array([x1 - x0, y1 - y0], dtype="f8")
    if not np.isfinite(vec).all() or np.hypot(*vec) < 0.5:
        vec = np.array([1.0, 0.0], dtype="f8")
    parallel = vec / np.hypot(*vec)
    normal = np.array([-parallel[1], parallel[0]], dtype="f8")
    s = np.linspace(-half_width_km, half_width_km, int(n_points))
    dl_cells = s / 11.119493
    base_i = 0.5 * (x0 + x1)
    base_j = 0.5 * (y0 + y1)
    ii = np.rint(base_i + dl_cells * parallel[0]).astype(int) % u_raw.shape[0]
    jj = np.clip(np.rint(base_j + dl_cells * parallel[1]).astype(int), 0, u_raw.shape[1] - 1)
    normal_velocity = np.empty((len(depth), len(s)), dtype="f4")
    horizontal_speed = np.empty_like(normal_velocity)
    centers_s, centers_z = [], []
    for k in range(len(depth)):
        u = (np.asarray(u_raw[ii, jj, k], dtype="f4") - np.asarray(u_mean[ii, jj, k], dtype="f4")) / 100.0
        v = (np.asarray(v_raw[ii, jj, k], dtype="f4") - np.asarray(v_mean[ii, jj, k], dtype="f4")) / 100.0
        bad = (np.abs(u) > 1.0e28) | (np.abs(v) > 1.0e28)
        normal_velocity[k] = u * normal[0] + v * normal[1]
        horizontal_speed[k] = np.hypot(u, v)
        normal_velocity[k, bad] = np.nan
        horizontal_speed[k, bad] = np.nan
    for _, row in layers.iterrows():
        delta = np.array([float(row[i_col]) - base_i, float(row[j_col]) - base_j])
        centers_s.append(float(np.dot(delta, parallel) * 11.119493))
        centers_z.append(float(row["depth_m"]))
    return {
        "section_coord_km": s,
        "depth": depth,
        "normal_horizontal_velocity_section": normal_velocity,
        "horizontal_speed_section": horizontal_speed,
        "signed_horizontal_speed_section": np.sign(normal_velocity) * horizontal_speed,
        "center_section_coord_km": np.array(centers_s),
        "center_depth_m": np.array(centers_z),
        "xlim_km": np.array([-half_width_km, half_width_km]),
        "zlim_m": np.array([float(np.nanmin(depth)), float(np.nanmax(depth))]),
        "section_axis": np.array("jump-parallel"),
        "velocity_label": np.array("normal horizontal velocity"),
        "coordinate_label": np.array("section distance from jump midpoint (km)"),
    }


def plot_right_section_mpl(ax, section: dict[str, np.ndarray], title: str, mode: str):
    s = section["section_coord_km"]
    depth = section["depth"]
    if mode == "horizontal_speed":
        values = section["horizontal_speed_section"]
        cmap = "coolwarm"
        vmin, vmax = finite_limits(values, symmetric=False)
        title = f"{title}: horizontal speed |u_h|\njump-parallel"
    elif mode == "signed_horizontal_speed":
        values = section["signed_horizontal_speed_section"]
        cmap = "RdBu_r"
        vmin, vmax = finite_limits(values, symmetric=True)
        title = f"{title}: signed horizontal speed\njump-parallel"
    else:
        values = section["normal_horizontal_velocity_section"]
        cmap = "RdBu_r"
        vmin, vmax = finite_limits(values, symmetric=True)
        title = f"{title}: normal horizontal velocity\njump-parallel"
    mesh = ax.pcolormesh(s, depth, values, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    finite = values[np.isfinite(values)]
    if finite.size:
        levels = np.linspace(float(vmin), float(vmax), 11)
        levels = levels[np.isfinite(levels)]
        levels = levels[np.abs(levels) > max(abs(float(vmax) - float(vmin)) * 0.01, 1.0e-12)]
        if levels.size >= 2:
            contours = ax.contour(s, depth, values, levels=levels, cmap=cmap, linewidths=0.55, alpha=0.72)
            for collection in contours.collections:
                collection.set_linewidth(1.0)
                collection.set_alpha(0.82)
    normal_velocity = section["normal_horizontal_velocity_section"]
    normal_finite = normal_velocity[np.isfinite(normal_velocity)]
    if normal_finite.size and float(np.nanmin(normal_finite)) < 0.0 < float(np.nanmax(normal_finite)):
        zero_lw = 2.8 if mode == "signed_horizontal_speed" else 1.8
        ax.contour(s, depth, normal_velocity, levels=[0.0], colors="0.05", linewidths=zero_lw, alpha=0.95)
    ax.invert_yaxis()
    ax.axvline(0.0, color="0.75", lw=0.8)
    ax.plot(section["center_section_coord_km"], section["center_depth_m"], "k.-", ms=4, lw=1.0, alpha=0.75, label="layer centers")
    ax.set_xlim(float(section["xlim_km"][0]), float(section["xlim_km"][1]))
    ax.set_ylim(float(section["zlim_m"][1]), float(section["zlim_m"][0]))
    ax.set_title(title, fontsize=9)
    ax.set_xlabel(str(section["coordinate_label"]))
    ax.set_ylabel("depth (m)")
    ax.grid(alpha=0.2)
    ax.legend(loc="best", fontsize=7)
    return mesh


def hatch_unavailable(ax, message: str) -> None:
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes, color="0.45", fontsize=9)
    ax.plot([0, 1], [0, 1], transform=ax.transAxes, color="0.85", lw=1.0)
    ax.plot([0, 1], [1, 0], transform=ax.transAxes, color="0.85", lw=1.0)
    ax.set_xticks([])
    ax.set_yticks([])


def plot_track_panel_mpl(ax, layers: pd.DataFrame, selected: SelectedPanelObject) -> None:
    lon_col = "center_lon_refined" if "center_lon_refined" in layers.columns else "center_lon"
    lat_col = "center_lat_refined" if "center_lat_refined" in layers.columns else "center_lat"
    ax.plot(layers[lon_col], layers[lat_col], "-", color="0.65", lw=1.0, label="vertical-center track")
    ax.scatter(layers[lon_col], layers[lat_col], c=np.arange(len(layers)), s=22, cmap="viridis", label="depth layers")
    ax.scatter([layers.iloc[0][lon_col]], [layers.iloc[0][lat_col]], s=130, c="red", marker="*", label="selected surface layer", zorder=9)
    ax.set_title("7  selected OFES object-day vertical trajectory")
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=9)
    ax.text(0.01, 0.96, "OFES v1 has no lifecycle tracking yet", transform=ax.transAxes, va="top", fontsize=9, color="0.35")


def layout_boxes() -> dict[str, tuple[int, int, int, int]]:
    x0, y0 = 40, 105
    w1, w, h, gap = 250, 360, 250, 26
    return {
        "dx": (x0, y0, x0 + w1, y0 + 4 * h + 3 * gap),
        "dy": (x0 + w1 + gap, y0, x0 + 2 * w1 + gap, y0 + 4 * h + 3 * gap),
        "j1_speed_from": (610, y0, 610 + w, y0 + h),
        "j1_pressure_from": (996, y0, 996 + w, y0 + h),
        "j1_speed_to": (610, y0 + h + gap, 610 + w, y0 + 2 * h + gap),
        "j1_pressure_to": (996, y0 + h + gap, 996 + w, y0 + 2 * h + gap),
        "j2_speed_from": (610, y0 + 2 * (h + gap), 610 + w, y0 + 3 * h + 2 * gap),
        "j2_pressure_from": (996, y0 + 2 * (h + gap), 996 + w, y0 + 3 * h + 2 * gap),
        "j2_speed_to": (610, y0 + 3 * (h + gap), 610 + w, y0 + 4 * h + 3 * gap),
        "j2_pressure_to": (996, y0 + 3 * (h + gap), 996 + w, y0 + 4 * h + 3 * gap),
        "j1_section_from": (1382, y0, 1382 + 560, y0 + 2 * h + gap),
        "j1_section_to": (1970, y0, 1970 + 560, y0 + 2 * h + gap),
        "j2_section_from": (1382, y0 + 2 * (h + gap), 1382 + 560, y0 + 4 * h + 3 * gap),
        "j2_section_to": (1970, y0 + 2 * (h + gap), 1970 + 560, y0 + 4 * h + 3 * gap),
        "track": (40, 1235, 2530, 1640),
    }


def object_offsets(layers: pd.DataFrame) -> pd.DataFrame:
    out = layers.sort_values("depth_index").copy()
    surface = out.iloc[0]
    lon_col = "center_lon_refined" if "center_lon_refined" in out.columns else "center_lon"
    lat_col = "center_lat_refined" if "center_lat_refined" in out.columns else "center_lat"
    mx, my = meters_per_degree(float(surface[lat_col]))
    out["delta_x_km"] = [lon_delta_deg(lon, float(surface[lon_col])) * mx / 1000.0 for lon in out[lon_col]]
    out["delta_y_km"] = (out[lat_col].astype(float) - float(surface[lat_col])) * my / 1000.0
    return out


def meters_per_degree(lat_deg: float) -> tuple[float, float]:
    lat_rad = math.radians(lat_deg)
    dx = math.pi * EARTH_RADIUS_M * math.cos(lat_rad) / 180.0
    dy = math.pi * EARTH_RADIUS_M / 180.0
    return dx, dy


def lon_delta_deg(lon: float, lon0: float) -> float:
    return ((float(lon) - float(lon0) + 180.0) % 360.0) - 180.0


def window_indices(lon: np.ndarray, lat: np.ndarray, lon0: float, lat0: float, half_width_deg: float) -> tuple[np.ndarray, np.ndarray]:
    dlon = np.asarray([abs(lon_delta_deg(v, lon0)) for v in lon])
    lon_idx = np.where(dlon <= half_width_deg)[0]
    lat_idx = np.where(np.abs(lat - lat0) <= half_width_deg)[0]
    if lon_idx.size < 8:
        center = int(np.nanargmin(dlon))
        lon_idx = np.arange(center - 8, center + 9) % len(lon)
    if lat_idx.size < 8:
        center = int(np.nanargmin(np.abs(lat - lat0)))
        lat_idx = np.arange(max(0, center - 8), min(len(lat), center + 9))
    return lon_idx.astype(int), lat_idx.astype(int)


def relative_mesh(lon: np.ndarray, lat: np.ndarray, lon0: float, lat0: float) -> tuple[np.ndarray, np.ndarray]:
    mx, my = meters_per_degree(lat0)
    x = np.asarray([lon_delta_deg(v, lon0) * mx / 1000.0 for v in lon], dtype="f8")
    y = (lat - lat0) * my / 1000.0
    return np.meshgrid(x, y)


def uv_anomaly_window(raw: np.memmap, mean: np.memmap, level: int, lon_idx: np.ndarray, lat_idx: np.ndarray) -> np.ndarray:
    block = np.asarray(raw[np.ix_(lon_idx, lat_idx, [level])][:, :, 0].T, dtype=np.float32)
    block[np.abs(block) > 1.0e30] = np.nan
    m = np.asarray(mean[np.ix_(lon_idx, lat_idx, [level])][:, :, 0].T, dtype=np.float32)
    return (block - m) / 100.0


def pressure_proxy(u: np.ndarray, v: np.ndarray, xx: np.ndarray, yy: np.ndarray, lat0: float) -> np.ndarray:
    f0 = 2.0 * 7.2921159e-5 * math.sin(math.radians(lat0))
    dx = np.nanmedian(np.abs(np.diff(xx, axis=1)))
    dy = np.nanmedian(np.abs(np.diff(yy, axis=0)))
    dx = max(float(dx) * 1000.0, 1.0)
    dy = max(float(dy) * 1000.0, 1.0)
    px = np.cumsum(np.nan_to_num(v, nan=0.0), axis=1) * f0 * dx
    py = -np.cumsum(np.nan_to_num(u, nan=0.0), axis=0) * f0 * dy
    p = 0.5 * (px + py)
    p -= np.nanmean(p)
    return p


def center_marks(layers: pd.DataFrame, lon0: float, lat0: float, jump_from: int, jump_to: int) -> list[tuple[float, float, str, tuple[int, int, int]]]:
    mx, my = meters_per_degree(lat0)
    marks = [(0.0, 0.0, "surface", (220, 40, 40))]
    lon_col = "center_lon_refined" if "center_lon_refined" in layers.columns else "center_lon"
    lat_col = "center_lat_refined" if "center_lat_refined" in layers.columns else "center_lat"
    for level, label, color in [(jump_from, "from", (0, 160, 210)), (jump_to, "to", (235, 190, 30))]:
        row = layers[layers["depth_index"].astype(int).eq(int(level))]
        if row.empty:
            continue
        marks.append((lon_delta_deg(row.iloc[0][lon_col], lon0) * mx / 1000.0, (float(row.iloc[0][lat_col]) - lat0) * my / 1000.0, label, color))
    return marks


def normal_velocity_section(
    u_raw: np.memmap,
    v_raw: np.memmap,
    u_mean: np.memmap,
    v_mean: np.memmap,
    layers: pd.DataFrame,
    depth: np.ndarray,
    jump_from: int,
    jump_to: int,
    half_width_km: float,
    n_points: int,
) -> dict[str, np.ndarray]:
    lower = layers[layers["depth_index"].astype(int).eq(int(jump_from))].iloc[0]
    upper = layers[layers["depth_index"].astype(int).eq(int(jump_to))].iloc[0]
    lat_col = "center_lat_refined" if "center_lat_refined" in layers.columns else "center_lat"
    mx, my = meters_per_degree(float(layers.iloc[0][lat_col]))
    i_col = "center_i_refined" if "center_i_refined" in layers.columns else "speed_min_i"
    j_col = "center_j_refined" if "center_j_refined" in layers.columns else "speed_min_j"
    x0 = float(lower[i_col])
    y0 = float(lower[j_col])
    x1 = float(upper[i_col])
    y1 = float(upper[j_col])
    vec = np.array([x1 - x0, y1 - y0], dtype="f8")
    if not np.isfinite(vec).all() or np.hypot(*vec) < 0.5:
        vec = np.array([1.0, 0.0], dtype="f8")
    parallel = vec / np.hypot(*vec)
    normal = np.array([-parallel[1], parallel[0]], dtype="f8")
    s = np.linspace(-half_width_km, half_width_km, int(n_points))
    dl_cells = s / 11.119493
    base_i = 0.5 * (x0 + x1)
    base_j = 0.5 * (y0 + y1)
    ii = np.rint(base_i + dl_cells * parallel[0]).astype(int) % u_raw.shape[0]
    jj = np.clip(np.rint(base_j + dl_cells * parallel[1]).astype(int), 0, u_raw.shape[1] - 1)
    values = np.empty((len(depth), len(s)), dtype="f4")
    centers_s = []
    centers_z = []
    for k in range(len(depth)):
        u = (np.asarray(u_raw[ii, jj, k], dtype="f4") - np.asarray(u_mean[ii, jj, k], dtype="f4")) / 100.0
        v = (np.asarray(v_raw[ii, jj, k], dtype="f4") - np.asarray(v_mean[ii, jj, k], dtype="f4")) / 100.0
        bad = (np.abs(u) > 1.0e28) | (np.abs(v) > 1.0e28)
        values[k] = u * normal[0] + v * normal[1]
        values[k, bad] = np.nan
    for _, row in layers.iterrows():
        delta = np.array([float(row[i_col]) - base_i, float(row[j_col]) - base_j])
        centers_s.append(float(np.dot(delta, parallel) * 11.119493))
        centers_z.append(float(row["depth_m"]))
    return {"s": s, "depth": depth, "values": values, "centers_s": np.array(centers_s), "centers_z": np.array(centers_z)}


def finite_limits(values: np.ndarray, symmetric: bool) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return (-1.0, 1.0)
    lo, hi = np.nanpercentile(finite, [2, 98])
    if symmetric:
        lim = max(abs(float(lo)), abs(float(hi)), 1.0e-9)
        return -lim, lim
    return float(lo), float(hi) if float(hi) > float(lo) else float(lo) + 1.0


def draw_heat_panel(canvas: Image.Image, box: tuple[int, int, int, int], xx: np.ndarray, yy: np.ndarray, values: np.ndarray, title: str, mode: str, marks, quiver, font_main, font_small) -> None:
    draw = ImageDraw.Draw(canvas)
    draw.rectangle(box, outline=(180, 186, 196), width=1)
    plot = inner_box(box, top=34, right=18, bottom=30, left=42)
    symmetric = mode == "diverging"
    vmin, vmax = finite_limits(values, symmetric)
    palette = "rdbu" if symmetric else "coolwarm"
    img = array_to_image(values, vmin, vmax, palette=palette).resize((plot[2] - plot[0], plot[3] - plot[1]), Image.Resampling.BILINEAR)
    canvas.paste(img, plot[:2])
    if symmetric:
        draw_contours(draw, plot, xx, yy, values, vmin, vmax, levels=9, fill=(64, 64, 64), width=1)
    if quiver is not None:
        draw_quiver(draw, plot, quiver[0], quiver[1], step=max(2, values.shape[0] // 12), fill=(255, 255, 255))
    for x, y, label, color in marks:
        px, py = map_xy(plot, x, y, np.nanmin(xx), np.nanmax(xx), np.nanmin(yy), np.nanmax(yy))
        draw.line((px - 6, py, px + 6, py), fill=color, width=2)
        draw.line((px, py - 6, px, py + 6), fill=color, width=2)
        draw.text((px + 7, py - 7), label, fill=color, font=font_small)
    draw.text((box[0] + 8, box[1] + 7), title, fill=(30, 36, 48), font=font_main)
    draw.text((plot[0], box[3] - 24), "km from surface center", fill=(80, 88, 100), font=font_small)
    draw.text((plot[2] - 110, box[3] - 24), f"{vmin:.2g}..{vmax:.2g}", fill=(80, 88, 100), font=font_small)


def draw_section_panel(canvas: Image.Image, box: tuple[int, int, int, int], section: dict[str, np.ndarray], title: str, highlight_level: int, font_main, font_small) -> None:
    draw = ImageDraw.Draw(canvas)
    draw.rectangle(box, outline=(180, 186, 196), width=1)
    plot = inner_box(box, top=40, right=18, bottom=38, left=58)
    values = section["values"]
    vmin, vmax = finite_limits(values, symmetric=True)
    img = array_to_image(values, vmin, vmax, palette="rdbu").resize((plot[2] - plot[0], plot[3] - plot[1]), Image.Resampling.BILINEAR)
    canvas.paste(img, plot[:2])
    s = section["s"]
    depth = section["depth"]
    ss, zz = np.meshgrid(s, depth)
    draw_contours(draw, plot, ss, zz, values, vmin, vmax, levels=11, fill=(55, 55, 55), width=1)
    finite = values[np.isfinite(values)]
    if finite.size and float(np.nanmin(finite)) < 0.0 < float(np.nanmax(finite)):
        draw_contours(draw, plot, ss, zz, values, -1.0, 1.0, levels=[0.0], fill=(5, 5, 5), width=2)
    for cs, cz in zip(section["centers_s"], section["centers_z"]):
        px, py = map_xy(plot, cs, cz, float(s[0]), float(s[-1]), float(depth[-1]), float(depth[0]))
        draw.ellipse((px - 2, py - 2, px + 2, py + 2), fill=(20, 20, 20))
    if 0 <= int(highlight_level) < len(depth):
        _, hy = map_xy(plot, 0.0, float(depth[int(highlight_level)]), float(s[0]), float(s[-1]), float(depth[-1]), float(depth[0]))
        draw.line((plot[0], hy, plot[2], hy), fill=(35, 35, 35), width=2)
    draw.line((plot[0] + (plot[2] - plot[0]) // 2, plot[1], plot[0] + (plot[2] - plot[0]) // 2, plot[3]), fill=(80, 80, 80), width=1)
    draw.text((box[0] + 8, box[1] + 8), title, fill=(30, 36, 48), font=font_main)
    draw.text((plot[0], box[3] - 28), "section distance (km)", fill=(80, 88, 100), font=font_small)
    draw.text((box[0] + 8, box[1] + 31), f"highlight layer {highlight_level}; normal velocity m/s; {vmin:.2g}..{vmax:.2g}", fill=(80, 88, 100), font=font_small)


def draw_line_panel(canvas: Image.Image, box: tuple[int, int, int, int], x: np.ndarray, depth: np.ndarray, title: str, xlabel: str, font_main, font_small) -> None:
    draw = ImageDraw.Draw(canvas)
    draw.rectangle(box, outline=(180, 186, 196), width=1)
    plot = inner_box(box, top=40, right=18, bottom=40, left=54)
    lim = max(1.0, float(np.nanmax(np.abs(x))) * 1.12)
    draw.line((plot[0], plot[3], plot[2], plot[3]), fill=(150, 156, 166))
    draw.line((plot[0], plot[1], plot[0], plot[3]), fill=(150, 156, 166))
    xzero, _ = map_xy(plot, 0.0, float(depth[0]), -lim, lim, float(depth[-1]), float(depth[0]))
    draw.line((xzero, plot[1], xzero, plot[3]), fill=(205, 210, 218))
    points = [map_xy(plot, float(a), float(b), -lim, lim, float(depth[-1]), float(depth[0])) for a, b in zip(x, depth)]
    if len(points) > 1:
        draw.line(points, fill=(36, 74, 155), width=3)
    for px, py in points:
        draw.ellipse((px - 4, py - 4, px + 4, py + 4), fill=(36, 74, 155))
    draw.text((box[0] + 8, box[1] + 8), title, fill=(30, 36, 48), font=font_main)
    draw.text((plot[0], box[3] - 28), xlabel, fill=(80, 88, 100), font=font_small)
    draw.text((box[0] + 8, box[1] + 31), f"+/- {lim:.1f} km", fill=(80, 88, 100), font=font_small)


def draw_trajectory_panel(canvas: Image.Image, box: tuple[int, int, int, int], layers: pd.DataFrame, object_layers: pd.DataFrame, font_main, font_small) -> None:
    draw = ImageDraw.Draw(canvas)
    draw.rectangle(box, outline=(180, 186, 196), width=1)
    plot = inner_box(box, top=36, right=30, bottom=34, left=48)
    x = layers["delta_x_km"].to_numpy("f8")
    y = layers["delta_y_km"].to_numpy("f8")
    lim = max(1.0, float(np.nanmax(np.abs(np.r_[x, y]))) * 1.20)
    draw.line((plot[0], (plot[1] + plot[3]) // 2, plot[2], (plot[1] + plot[3]) // 2), fill=(215, 220, 226))
    draw.line(((plot[0] + plot[2]) // 2, plot[1], (plot[0] + plot[2]) // 2, plot[3]), fill=(215, 220, 226))
    points = [map_xy(plot, float(a), float(b), -lim, lim, -lim, lim) for a, b in zip(x, y)]
    if len(points) > 1:
        draw.line(points, fill=(166, 166, 166), width=2)
    depth = layers["depth_m"].to_numpy("f8")
    dmin, dmax = float(np.nanmin(depth)), float(np.nanmax(depth))
    for idx, (px, py) in enumerate(points):
        color = (220, 40, 40) if idx == 0 else viridis_r((float(depth[idx]) - dmin) / max(dmax - dmin, 1.0e-12))
        r = 7 if idx == 0 else 5
        if idx == 0:
            draw_star(draw, px, py, 10, color)
        else:
            draw.ellipse((px - r, py - r, px + r, py + r), fill=color, outline=(30, 30, 30))
    draw.text((box[0] + 8, box[1] + 8), "7 vertical center trajectory within one object-day", fill=(30, 36, 48), font=font_main)
    draw.text((plot[0], box[3] - 26), f"old-style line + depth-colored centers; red star=surface; no lifecycle tracking in OFES v1; +/- {lim:.1f} km", fill=(80, 88, 100), font=font_small)


def draw_empty_panel(canvas: Image.Image, box: tuple[int, int, int, int], title: str, font_main) -> None:
    draw = ImageDraw.Draw(canvas)
    draw.rectangle(box, outline=(200, 205, 212), width=1)
    draw.line((box[0], box[1], box[2], box[3]), fill=(220, 224, 230), width=2)
    draw.line((box[0], box[3], box[2], box[1]), fill=(220, 224, 230), width=2)
    draw.text((box[0] + 10, box[1] + 10), title, fill=(100, 108, 120), font=font_main)


def array_to_image(values: np.ndarray, vmin: float, vmax: float, palette: str) -> Image.Image:
    arr = np.asarray(values, dtype="f4")
    nan = ~np.isfinite(arr)
    scaled = np.clip((arr - vmin) / max(vmax - vmin, 1.0e-12), 0.0, 1.0)
    scaled = np.nan_to_num(scaled, nan=0.5, posinf=1.0, neginf=0.0)
    if palette == "rdbu":
        colors = rdbu_r_colors(scaled)
    elif palette == "coolwarm":
        colors = coolwarm_colors(scaled)
    else:
        colors = sequential_colors(scaled)
    colors[nan] = np.array([198, 202, 208], dtype=np.uint8)
    return Image.fromarray(colors, mode="RGB")


def rdbu_r_colors(scaled: np.ndarray) -> np.ndarray:
    blue = np.array([5, 113, 176], dtype="f4")
    white = np.array([247, 247, 247], dtype="f4")
    red = np.array([202, 0, 32], dtype="f4")
    rgb = np.empty((*scaled.shape, 3), dtype=np.uint8)
    low = scaled <= 0.5
    lt = scaled[low] / 0.5
    ht = (scaled[~low] - 0.5) / 0.5
    rgb[low] = (blue * (1.0 - lt[:, None]) + white * lt[:, None]).astype(np.uint8)
    rgb[~low] = (white * (1.0 - ht[:, None]) + red * ht[:, None]).astype(np.uint8)
    return rgb


def coolwarm_colors(scaled: np.ndarray) -> np.ndarray:
    blue = np.array([59, 76, 192], dtype="f4")
    mid = np.array([221, 221, 221], dtype="f4")
    red = np.array([180, 4, 38], dtype="f4")
    rgb = np.empty((*scaled.shape, 3), dtype=np.uint8)
    low = scaled <= 0.5
    lt = scaled[low] / 0.5
    ht = (scaled[~low] - 0.5) / 0.5
    rgb[low] = (blue * (1.0 - lt[:, None]) + mid * lt[:, None]).astype(np.uint8)
    rgb[~low] = (mid * (1.0 - ht[:, None]) + red * ht[:, None]).astype(np.uint8)
    return rgb


def sequential_colors(scaled: np.ndarray) -> np.ndarray:
    a = np.array([236, 248, 255], dtype="f4")
    b = np.array([20, 83, 140], dtype="f4")
    return (a * (1.0 - scaled[..., None]) + b * scaled[..., None]).astype(np.uint8)


def viridis_r(t: float) -> tuple[int, int, int]:
    stops = [
        (253, 231, 37),
        (94, 201, 98),
        (33, 145, 140),
        (59, 82, 139),
        (68, 1, 84),
    ]
    t = float(np.clip(t, 0.0, 1.0)) * (len(stops) - 1)
    i = min(int(math.floor(t)), len(stops) - 2)
    f = t - i
    a = np.array(stops[i], dtype="f4")
    b = np.array(stops[i + 1], dtype="f4")
    return tuple((a * (1.0 - f) + b * f).astype(np.uint8).tolist())


def draw_star(draw: ImageDraw.ImageDraw, x: int, y: int, radius: int, fill: tuple[int, int, int]) -> None:
    points = []
    for idx in range(10):
        angle = -math.pi / 2.0 + idx * math.pi / 5.0
        r = radius if idx % 2 == 0 else radius * 0.42
        points.append((x + r * math.cos(angle), y + r * math.sin(angle)))
    draw.polygon(points, fill=fill, outline=(60, 0, 0))


def draw_contours(
    draw: ImageDraw.ImageDraw,
    plot: tuple[int, int, int, int],
    xx: np.ndarray,
    yy: np.ndarray,
    values: np.ndarray,
    vmin: float,
    vmax: float,
    *,
    levels: int | list[float],
    fill: tuple[int, int, int],
    width: int,
) -> None:
    if isinstance(levels, int):
        contour_levels = np.linspace(float(vmin), float(vmax), int(levels))
        contour_levels = contour_levels[np.abs(contour_levels) > max(abs(float(vmax) - float(vmin)) * 0.01, 1.0e-12)]
    else:
        contour_levels = np.asarray(levels, dtype="f8")
    contour_levels = contour_levels[np.isfinite(contour_levels)]
    if contour_levels.size == 0:
        return
    xmin, xmax = float(np.nanmin(xx)), float(np.nanmax(xx))
    ymin, ymax = float(np.nanmin(yy)), float(np.nanmax(yy))
    ny, nx = values.shape
    for level in contour_levels:
        for j in range(ny - 1):
            for i in range(nx - 1):
                vals = np.array([values[j, i], values[j, i + 1], values[j + 1, i + 1], values[j + 1, i]], dtype="f8")
                if not np.isfinite(vals).all():
                    continue
                pts_xy = [
                    (float(xx[j, i]), float(yy[j, i])),
                    (float(xx[j, i + 1]), float(yy[j, i + 1])),
                    (float(xx[j + 1, i + 1]), float(yy[j + 1, i + 1])),
                    (float(xx[j + 1, i]), float(yy[j + 1, i])),
                ]
                hits = []
                for a, b in [(0, 1), (1, 2), (2, 3), (3, 0)]:
                    va = vals[a] - level
                    vb = vals[b] - level
                    if va == 0.0 and vb == 0.0:
                        continue
                    if va == 0.0 or vb == 0.0 or (va < 0.0 < vb) or (vb < 0.0 < va):
                        denom = vals[b] - vals[a]
                        frac = 0.5 if abs(denom) < 1.0e-12 else (level - vals[a]) / denom
                        frac = float(np.clip(frac, 0.0, 1.0))
                        xa, ya = pts_xy[a]
                        xb, yb = pts_xy[b]
                        hits.append((xa + frac * (xb - xa), ya + frac * (yb - ya)))
                if len(hits) == 2:
                    draw.line((map_xy(plot, hits[0][0], hits[0][1], xmin, xmax, ymin, ymax), map_xy(plot, hits[1][0], hits[1][1], xmin, xmax, ymin, ymax)), fill=fill, width=width)
                elif len(hits) == 4:
                    draw.line((map_xy(plot, hits[0][0], hits[0][1], xmin, xmax, ymin, ymax), map_xy(plot, hits[1][0], hits[1][1], xmin, xmax, ymin, ymax)), fill=fill, width=width)
                    draw.line((map_xy(plot, hits[2][0], hits[2][1], xmin, xmax, ymin, ymax), map_xy(plot, hits[3][0], hits[3][1], xmin, xmax, ymin, ymax)), fill=fill, width=width)


def map_xy(box: tuple[int, int, int, int], x: float, y: float, xmin: float, xmax: float, ymin: float, ymax: float) -> tuple[int, int]:
    xspan = xmax - xmin
    yspan = ymax - ymin
    if abs(xspan) < 1.0e-12:
        xspan = 1.0
    if abs(yspan) < 1.0e-12:
        yspan = 1.0
    px = box[0] + int(round((x - xmin) / xspan * (box[2] - box[0] - 1)))
    py = box[3] - int(round((y - ymin) / yspan * (box[3] - box[1] - 1)))
    px = int(np.clip(px, min(box[0], box[2]) - 8, max(box[0], box[2]) + 8))
    py = int(np.clip(py, min(box[1], box[3]) - 8, max(box[1], box[3]) + 8))
    return px, py


def draw_quiver(draw: ImageDraw.ImageDraw, plot: tuple[int, int, int, int], u: np.ndarray, v: np.ndarray, step: int, fill: tuple[int, int, int]) -> None:
    ny, nx = u.shape
    mag = np.nanpercentile(np.hypot(u, v), 95)
    scale = 16.0 / max(float(mag), 1.0e-6)
    for j in range(step // 2, ny, step):
        for i in range(step // 2, nx, step):
            uu, vv = float(u[j, i]), float(v[j, i])
            if not (np.isfinite(uu) and np.isfinite(vv)):
                continue
            x = plot[0] + int(i / max(nx - 1, 1) * (plot[2] - plot[0]))
            y = plot[3] - int(j / max(ny - 1, 1) * (plot[3] - plot[1]))
            draw.line((x, y, x + uu * scale, y - vv * scale), fill=fill, width=1)


def inner_box(box: tuple[int, int, int, int], *, top: int, right: int, bottom: int, left: int) -> tuple[int, int, int, int]:
    return (box[0] + left, box[1] + top, box[2] - right, box[3] - bottom)


def font(size: int) -> ImageFont.ImageFont:
    for name in ["arial.ttf", "DejaVuSans.ttf"]:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


if __name__ == "__main__":
    main()
