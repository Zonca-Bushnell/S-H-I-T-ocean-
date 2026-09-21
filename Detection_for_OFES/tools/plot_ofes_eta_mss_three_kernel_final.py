"""Plot final Jan1 catalogs for the corrected eta-MSS three-kernel run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from Detection_for_OFES.tools.plot_latest_ssh_vector_overview import parse_index_list


ROOT = Path(
    r"E:\DATA\01_Eddy_correspond\02_OFES"
    r"\origin_unified_eta_mss_three_kernel_surface_jan01_jan19"
)
KERNELS = ("gaussian", "lanczos", "bessel")
REGIONS = {
    "global": (0.0, 360.0, -76.0, 76.0),
    "north_pacific_open_ocean": (190.0, 245.0, 25.0, 55.0),
    "south_pacific_open_ocean": (190.0, 280.0, -50.0, -30.0),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--day", default="1991-01-01")
    return parser.parse_args()


def load_branch(root: Path, kernel: str, day: str):
    compact = day.replace("-", "")
    with h5py.File(root / kernel / "filtered_inputs" / f"global_phy_{compact}.nc", "r") as data:
        lon = np.asarray(data["longitude"][:], dtype="f8")
        lat = np.asarray(data["latitude"][:], dtype="f8")
        ssh = np.asarray(data["zos_glor"][0], dtype="f4")
    path = root / kernel / "catalog" / "final_catalog" / "daily_runs" / compact / "centers_hua_style.csv"
    centers = pd.read_csv(path)
    if centers.get("persistence_class", pd.Series(dtype=str)).astype(str).eq("transient").any():
        raise RuntimeError(f"Final catalog contains transient rows: {path}")
    return lon, lat, ssh, centers


def subset(frame: pd.DataFrame, bbox):
    lon = pd.to_numeric(frame.get("center_lon", frame.get("seed_lon")), errors="coerce")
    lat = pd.to_numeric(frame.get("center_lat", frame.get("seed_lat")), errors="coerce")
    return frame[lon.between(bbox[0], bbox[1]) & lat.between(bbox[2], bbox[3])]


def boundary(axis, row, lon, lat, color):
    ii = parse_index_list(row.get("ssh_contour_boundary_i", ""))
    jj = parse_index_list(row.get("ssh_contour_boundary_j", ""))
    n = min(ii.size, jj.size)
    if n < 3:
        return
    valid = (ii[:n] >= 0) & (ii[:n] < lon.size) & (jj[:n] >= 0) & (jj[:n] < lat.size)
    if valid.sum() >= 3:
        axis.plot(lon[ii[:n][valid]], lat[jj[:n][valid]], color=color, linewidth=0.45, alpha=0.85)


def draw(root: Path, day: str, region: str, bbox, branches) -> tuple[Path, dict[str, int]]:
    values = []
    for _, lon, lat, ssh, _ in branches:
        xi = np.where((lon >= bbox[0]) & (lon <= bbox[1]))[0]
        yi = np.where((lat >= bbox[2]) & (lat <= bbox[3]))[0]
        field = ssh[np.ix_(yi, xi)]
        values.append(field[np.isfinite(field)])
    limit = float(np.nanpercentile(np.abs(np.concatenate(values)), 99.0))
    figure, axes = plt.subplots(1, 3, figsize=(22, 7.8), constrained_layout=True, sharex=True, sharey=True)
    counts = {}
    image = None
    for axis, (kernel, lon, lat, ssh, centers) in zip(axes, branches):
        xi = np.where((lon >= bbox[0]) & (lon <= bbox[1]))[0]
        yi = np.where((lat >= bbox[2]) & (lat <= bbox[3]))[0]
        image = axis.imshow(
            ssh[np.ix_(yi, xi)], origin="lower", aspect="auto", interpolation="nearest",
            extent=(lon[xi[0]], lon[xi[-1]], lat[yi[0]], lat[yi[-1]]),
            cmap="RdBu_r", vmin=-limit, vmax=limit,
        )
        selected = subset(centers, bbox)
        counts[kernel] = int(len(selected))
        for _, row in selected.iterrows():
            color = "#2563eb" if str(row.get("polarity", "")) == "cyclonic" else "#dc2626"
            boundary(axis, row, lon, lat, color)
            axis.scatter(
                row.get("center_lon", row.get("seed_lon")), row.get("center_lat", row.get("seed_lat")),
                s=10, c=color, edgecolors="white", linewidths=0.25, zorder=3,
            )
        axis.set_title(f"{kernel.title()}\nfinal objects: {len(selected)}")
        axis.set_xlabel("Longitude (degrees east)")
        axis.grid(True, color="black", alpha=0.15, linewidth=0.4)
    axes[0].set_ylabel("Latitude")
    figure.colorbar(image, ax=axes.tolist(), label="Rossby-filtered SLA (cm)", shrink=0.86)
    figure.suptitle(
        f"OFES corrected eta-MSS final catalogs | {region} | {day}\n"
        "Same baseline, detection, QC, and persistence; only spatial kernel differs",
        fontsize=14,
    )
    output = root / "comparison" / f"eta_mss_three_kernel_final_{region}_{day.replace('-', '')}.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)
    return output, counts


def main() -> None:
    args = parse_args()
    branches = [(kernel, *load_branch(args.root, kernel, args.day)) for kernel in KERNELS]
    products = {}
    counts = {}
    for region, bbox in REGIONS.items():
        output, region_counts = draw(args.root, args.day, region, bbox, branches)
        products[region] = str(output)
        counts[region] = region_counts
    manifest = {
        "status": "complete",
        "day": args.day,
        "baseline": "eta(day)-annual eta MSS 1993-2012; no pair subtraction",
        "catalog_policy": "final QC and persistence pass only; transient excluded",
        "products": products,
        "counts": counts,
    }
    path = args.root / "comparison" / "eta_mss_three_kernel_final_manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
