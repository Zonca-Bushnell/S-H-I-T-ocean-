"""Plot user-defined raw OFES SSH with three regional vector zooms."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from Detection_for_OFES.ofes_io import ctl_path, parse_ctl, read_variable_latlon_daily_only


OFES_ROOT = Path(r"F:\OFES\external_OFES2")
OUTPUT_ROOT = Path(
    r"E:\DATA\01_Eddy_correspond\02_OFES\Fromthebeginning"
    r"\01_raw_ssh_region_vectors_19910101"
)
REGIONS = {
    "south_pacific_stcc": {
        "label": "South Pacific STCC",
        "bbox": (165.0, 230.0, -29.0, -21.0),
        "color": "#f59e0b",
    },
    "kuroshio_extension": {
        "label": "Kuroshio Extension",
        "bbox": (140.0, 180.0, 28.0, 40.0),
        "color": "#22c55e",
    },
    "taiwan_hawaii_stcc_hlcc": {
        "label": "Taiwan-Hawaii STCC-HLCC corridor",
        "bbox": (122.0, 203.0, 18.0, 27.0),
        "color": "#ec4899",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ofes-root", type=Path, default=OFES_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--day", default="1991-01-01")
    parser.add_argument("--global-vector-spacing-deg", type=float, default=2.0)
    parser.add_argument("--zoom-vector-spacing-deg", type=float, default=0.3)
    parser.add_argument("--global-arrow-length-deg", type=float, default=1.6)
    parser.add_argument("--zoom-arrow-length-deg", type=float, default=0.65)
    return parser.parse_args()


def subset_indices(values: np.ndarray, lower: float, upper: float) -> np.ndarray:
    indices = np.where((values >= lower) & (values <= upper))[0]
    if not indices.size:
        raise ValueError(f"No coordinates in requested range {lower}..{upper}")
    return indices


def vector_step(coordinates: np.ndarray, spacing_deg: float) -> int:
    resolution = float(np.median(np.diff(coordinates)))
    return max(1, int(round(float(spacing_deg) / resolution)))


def plot_field(
    axis,
    *,
    ssh_lon: np.ndarray,
    ssh_lat: np.ndarray,
    raw_ssh: np.ndarray,
    vel_lon: np.ndarray,
    vel_lat: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    bbox: tuple[float, float, float, float],
    color_limit: float,
    vector_spacing_deg: float,
    arrow_length_deg: float,
    title: str,
):
    lon_min, lon_max, lat_min, lat_max = bbox
    ssh_i = subset_indices(ssh_lon, lon_min, lon_max)
    ssh_j = subset_indices(ssh_lat, lat_min, lat_max)
    vel_i = subset_indices(vel_lon, lon_min, lon_max)
    vel_j = subset_indices(vel_lat, lat_min, lat_max)
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#252a31")
    image = axis.imshow(
        raw_ssh[np.ix_(ssh_j, ssh_i)],
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        extent=(ssh_lon[ssh_i[0]], ssh_lon[ssh_i[-1]], ssh_lat[ssh_j[0]], ssh_lat[ssh_j[-1]]),
        cmap=cmap,
        vmin=-color_limit,
        vmax=color_limit,
        zorder=0,
    )
    stride = vector_step(vel_lon, vector_spacing_deg)
    ii = vel_i[::stride]
    jj = vel_j[::stride]
    longitude, latitude = np.meshgrid(vel_lon[ii], vel_lat[jj])
    u_sub = np.asarray(u[np.ix_(jj, ii)], dtype="f8")
    v_sub = np.asarray(v[np.ix_(jj, ii)], dtype="f8")
    # In degree coordinates, compensate zonal displacement by cos(latitude)
    # so an arrow preserves the physical u/v direction on the map.
    u_display = u_sub / np.maximum(np.cos(np.deg2rad(latitude)), 0.2)
    display_speed = np.hypot(u_display, v_sub)
    valid = np.isfinite(display_speed)
    q95 = float(np.nanpercentile(display_speed[valid], 95.0)) if valid.any() else 1.0
    axis.quiver(
        longitude,
        latitude,
        u_display,
        v_sub,
        angles="xy",
        scale_units="xy",
        scale=max(q95 / max(arrow_length_deg, 1.0e-6), 1.0e-6),
        width=0.0038,
        headwidth=4.2,
        headlength=5.2,
        headaxislength=4.8,
        color="#15191f",
        alpha=0.9,
        zorder=3,
    )
    axis.set_xlim(lon_min, lon_max)
    axis.set_ylim(lat_min, lat_max)
    axis.set_title(title, fontsize=12)
    axis.set_xlabel("Longitude (degrees east)")
    axis.set_ylabel("Latitude")
    axis.grid(True, color="#20252c", alpha=0.18, linewidth=0.45)
    return image, {"vector_spacing_deg": vector_spacing_deg, "arrow_q95_cm_s": q95, "vector_count": int(valid.sum())}


def add_region_boxes(axis) -> None:
    for region in REGIONS.values():
        lon_min, lon_max, lat_min, lat_max = region["bbox"]
        axis.plot(
            [lon_min, lon_max, lon_max, lon_min, lon_min],
            [lat_min, lat_min, lat_max, lat_max, lat_min],
            color=region["color"],
            linewidth=2.2,
            zorder=5,
        )
        axis.text(
            lon_min + 0.8,
            lat_max - 1.2,
            region["label"],
            color=region["color"],
            fontsize=8.5,
            weight="bold",
            zorder=6,
        )


def save_individual_zooms(
    args: argparse.Namespace,
    *,
    ssh_lon: np.ndarray,
    ssh_lat: np.ndarray,
    raw_ssh: np.ndarray,
    vel_lon: np.ndarray,
    vel_lat: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    color_limit: float,
    day: date,
) -> dict[str, str]:
    outputs: dict[str, str] = {}
    for key, region in REGIONS.items():
        figure, axis = plt.subplots(figsize=(13, 7.4))
        image, _ = plot_field(
            axis,
            ssh_lon=ssh_lon,
            ssh_lat=ssh_lat,
            raw_ssh=raw_ssh,
            vel_lon=vel_lon,
            vel_lat=vel_lat,
            u=u,
            v=v,
            bbox=region["bbox"],
            color_limit=color_limit,
            vector_spacing_deg=args.zoom_vector_spacing_deg,
            arrow_length_deg=args.zoom_arrow_length_deg,
            title=f"OFES raw SSH | {region['label']} | {day.isoformat()}",
        )
        figure.colorbar(image, ax=axis, label="Raw SSH H_OFES (cm)", shrink=0.9)
        figure.subplots_adjust(left=0.07, right=0.91, bottom=0.10, top=0.90)
        output = args.output_root / f"raw_ssh_vector_zoom_{key}_{day:%Y%m%d}.png"
        figure.savefig(output, dpi=200)
        figure.savefig(output.with_suffix(".pdf"))
        plt.close(figure)
        outputs[key] = str(output)
    return outputs


def main() -> None:
    args = parse_args()
    target_day = date.fromisoformat(args.day)
    ssh_meta = parse_ctl(ctl_path(args.ofes_root, "eta"))
    velocity_meta = parse_ctl(ctl_path(args.ofes_root, "u"))
    ssh_lon = np.asarray(ssh_meta.x.values, dtype="f8")
    ssh_lat = np.asarray(ssh_meta.y.values, dtype="f8")
    vel_lon = np.asarray(velocity_meta.x.values, dtype="f8")
    vel_lat = np.asarray(velocity_meta.y.values, dtype="f8")
    eta = np.asarray(read_variable_latlon_daily_only(args.ofes_root, "eta", target_day), dtype="f8")
    pressure = np.asarray(read_variable_latlon_daily_only(args.ofes_root, "pressur", target_day), dtype="f8")
    u = np.asarray(read_variable_latlon_daily_only(args.ofes_root, "u", target_day, level_index=0), dtype="f8")
    v = np.asarray(read_variable_latlon_daily_only(args.ofes_root, "v", target_day, level_index=0), dtype="f8")
    raw_ssh = eta - (pressure - 1000.0)
    raw_ssh[~np.isfinite(eta) | ~np.isfinite(pressure)] = np.nan
    color_limit = float(np.nanpercentile(np.abs(raw_ssh), 99.0))
    args.output_root.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(2, 2, figsize=(22, 14))
    global_image, global_vector_info = plot_field(
        axes[0, 0],
        ssh_lon=ssh_lon,
        ssh_lat=ssh_lat,
        raw_ssh=raw_ssh,
        vel_lon=vel_lon,
        vel_lat=vel_lat,
        u=u,
        v=v,
        bbox=(0.0, 360.0, -76.0, 76.0),
        color_limit=color_limit,
        vector_spacing_deg=args.global_vector_spacing_deg,
        arrow_length_deg=args.global_arrow_length_deg,
        title=f"Global context | H_OFES | {target_day.isoformat()}",
    )
    add_region_boxes(axes[0, 0])
    vector_info: dict[str, dict[str, float | int]] = {"global": global_vector_info}
    for axis, (key, region) in zip(axes.flat[1:], REGIONS.items()):
        image, info = plot_field(
            axis,
            ssh_lon=ssh_lon,
            ssh_lat=ssh_lat,
            raw_ssh=raw_ssh,
            vel_lon=vel_lon,
            vel_lat=vel_lat,
            u=u,
            v=v,
            bbox=region["bbox"],
            color_limit=color_limit,
            vector_spacing_deg=args.zoom_vector_spacing_deg,
            arrow_length_deg=args.zoom_arrow_length_deg,
            title=region["label"],
        )
        vector_info[key] = info
    figure.colorbar(global_image, ax=axes.ravel().tolist(), label="Raw SSH H_OFES (cm)", shrink=0.78)
    figure.suptitle(
        r"OFES raw SSH and surface velocity | $H_{OFES}=\eta-(p_{air}-1000)$"
        + f" | {target_day.isoformat()}\n"
        + "Global vectors: 2.0 degrees; regional vectors: 0.3 degrees",
        fontsize=15,
    )
    figure.subplots_adjust(left=0.06, right=0.91, bottom=0.07, top=0.90, wspace=0.16, hspace=0.22)
    combined = args.output_root / f"raw_ssh_three_regions_vectors_{target_day:%Y%m%d}.png"
    figure.savefig(combined, dpi=220)
    figure.savefig(combined.with_suffix(".pdf"))
    plt.close(figure)
    zooms = save_individual_zooms(
        args,
        ssh_lon=ssh_lon,
        ssh_lat=ssh_lat,
        raw_ssh=raw_ssh,
        vel_lon=vel_lon,
        vel_lat=vel_lat,
        u=u,
        v=v,
        color_limit=color_limit,
        day=target_day,
    )
    manifest = {
        "status": "complete",
        "day": target_day.isoformat(),
        "raw_ssh_definition": "H_OFES=eta-(pressur-1000)",
        "processing": "No temporal mean, spatial filter, regridding, eddy detection, or QC.",
        "velocity": "Raw OFES u/v at 2.5 m; arrows retain physical direction with longitude cos(latitude) compensation.",
        "regions": REGIONS,
        "vector_sampling": vector_info,
        "color_limit_cm": color_limit,
        "combined_png": str(combined),
        "zoom_png": zooms,
    }
    manifest_path = args.output_root / f"raw_ssh_three_regions_vectors_manifest_{target_day:%Y%m%d}.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
