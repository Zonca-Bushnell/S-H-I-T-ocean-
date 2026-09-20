"""Plot and summarize final Jan1 catalogs for SSH baseline x filter-kernel tests."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from Detection_for_OFES.tools.plot_latest_ssh_vector_overview import parse_index_list


DEFAULT_ROOT = Path(r"E:\DATA\01_Eddy_correspond\02_OFES\baseline_ab_longterm_january_climatology_filter6_19910101")
BASELINES = ("old_jan01_jan19_mean", "longterm_january_1993_2012")
KERNELS = ("gaussian", "lanczos", "bessel")
REGIONS: dict[str, tuple[float, float, float, float]] = {
    "global": (0.0, 360.0, -76.0, 76.0),
    "north_pacific_open_ocean": (190.0, 245.0, 25.0, 55.0),
    "south_pacific_open_ocean": (190.0, 280.0, -50.0, -30.0),
}
STATS_REGIONS: dict[str, tuple[float, float, float, float]] = {
    **REGIONS,
    "north_atlantic_open_ocean": (320.0, 330.0, 25.0, 45.0),
    "south_atlantic_open_ocean": (335.0, 355.0, -45.0, -20.0),
    "acc": (0.0, 360.0, -62.0, -40.0),
    "kuroshio_reference": (120.0, 145.0, 20.0, 35.0),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--day", default="1991-01-01")
    return parser.parse_args()


def branch_specs(root: Path) -> list[tuple[str, str, str]]:
    specs = []
    for baseline in BASELINES:
        for kernel in KERNELS:
            name = f"{baseline}__{kernel}"
            if (root / name / "catalog" / "jan1_final_catalog").exists():
                specs.append((name, baseline, kernel))
    if not specs:
        raise FileNotFoundError(f"No final kernel-comparison catalog under {root}")
    return specs


def read_filter(path: Path, day: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(path / f"global_phy_{day.replace('-', '')}.nc", "r") as data:
        return (
            np.asarray(data["longitude"][:], dtype="f8"),
            np.asarray(data["latitude"][:], dtype="f8"),
            np.asarray(data["zos_glor"][0], dtype="f4"),
        )


def read_centers(root: Path, relative: str, day: str) -> pd.DataFrame:
    path = root / relative / "daily_runs" / day.replace("-", "") / "centers_hua_style.csv"
    frame = pd.read_csv(path)
    if "depth_index" in frame.columns:
        frame = frame[frame["depth_index"].astype(int).eq(0)].copy()
    return frame


def in_region(frame: pd.DataFrame, bbox: tuple[float, float, float, float]) -> pd.DataFrame:
    lon = pd.to_numeric(frame.get("center_lon", frame.get("seed_lon")), errors="coerce")
    lat = pd.to_numeric(frame.get("center_lat", frame.get("seed_lat")), errors="coerce")
    lo0, lo1, la0, la1 = bbox
    return frame[lon.between(lo0, lo1) & lat.between(la0, la1)].copy()


def plot_boundary(axis, row: pd.Series, lon: np.ndarray, lat: np.ndarray, color: str) -> None:
    ii = parse_index_list(row.get("ssh_contour_boundary_i", ""))
    jj = parse_index_list(row.get("ssh_contour_boundary_j", ""))
    n = min(ii.size, jj.size)
    if n < 3:
        return
    ii, jj = ii[:n], jj[:n]
    valid = (ii >= 0) & (ii < len(lon)) & (jj >= 0) & (jj < len(lat))
    if valid.sum() >= 3:
        axis.plot(lon[ii[valid]], lat[jj[valid]], color=color, linewidth=0.55, alpha=0.9)


def draw_region(root: Path, day: str, region: str, bbox: tuple[float, float, float, float]) -> Path:
    specs = branch_specs(root)
    branches: list[tuple[str, str, str, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]] = []
    values: list[np.ndarray] = []
    for name, baseline, kernel in specs:
        branch = root / name
        lon, lat, ssh = read_filter(branch / "rossby_filter", day)
        centers = read_centers(branch / "catalog" / "jan1_final_catalog", "", day)
        # read_centers expects a relative daily-runs root; the Jan1-only output is exact.
        branches.append((name, baseline, kernel, lon, lat, ssh, centers))
        lat_idx = np.where((lat >= bbox[2]) & (lat <= bbox[3]))[0]
        lon_idx = np.where((lon >= bbox[0]) & (lon <= bbox[1]))[0]
        values.append(ssh[np.ix_(lat_idx, lon_idx)])
    finite = np.concatenate([value[np.isfinite(value)] for value in values if np.isfinite(value).any()])
    limit = float(np.nanpercentile(np.abs(finite), 99.0)) if finite.size else 1.0
    figure, axes = plt.subplots(2, 3, figsize=(19, 11), constrained_layout=True, sharex=True, sharey=True)
    for axis, (name, baseline, kernel, lon, lat, ssh, centers) in zip(axes.flat, branches):
        lon_idx = np.where((lon >= bbox[0]) & (lon <= bbox[1]))[0]
        lat_idx = np.where((lat >= bbox[2]) & (lat <= bbox[3]))[0]
        field = ssh[np.ix_(lat_idx, lon_idx)]
        image = axis.imshow(
            field, origin="lower", aspect="auto", interpolation="nearest",
            extent=(lon[lon_idx[0]], lon[lon_idx[-1]], lat[lat_idx[0]], lat[lat_idx[-1]]),
            cmap="RdBu_r", vmin=-limit, vmax=limit,
        )
        surface = in_region(centers, bbox)
        for _, row in surface.iterrows():
            color = "#2563eb" if str(row.get("polarity", "")) == "cyclonic" else "#dc2626"
            plot_boundary(axis, row, lon, lat, color)
            axis.scatter(row.get("center_lon", row.get("seed_lon")), row.get("center_lat", row.get("seed_lat")), s=15, c=color, edgecolors="white", linewidths=0.35, zorder=3)
        baseline_title = "A: Jan01-Jan19 mean" if baseline.startswith("old_") else "B: January 1993-2012 climatology"
        axis.set_title(f"{baseline_title} | {kernel.title()}\nfinal Jan1 objects: {len(surface)}")
        axis.set_xlabel("Longitude (degrees east)")
        axis.grid(True, color="black", alpha=0.18, linewidth=0.45)
    for axis in axes[:, 0]:
        axis.set_ylabel("Latitude")
    figure.colorbar(image, ax=axes.ravel().tolist(), label="Rossby-filtered SSH anomaly (cm)", shrink=0.86)
    figure.suptitle(f"OFES baseline x kernel | {region} | {day}\nSame scale band, detection, QC, and persistence; only SSH reference and LP kernel differ", fontsize=14)
    output = root / "comparison" / f"ofes_baseline_ab_{region}_{day.replace('-', '')}.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)
    return output


def region_metrics(frame: pd.DataFrame, bbox: tuple[float, float, float, float]) -> dict[str, float | int]:
    part = in_region(frame, bbox)
    amp = pd.to_numeric(part.get("ssh_contour_amplitude_cm", pd.Series(dtype=float)), errors="coerce")
    radius = pd.to_numeric(part.get("radius_km", pd.Series(dtype=float)), errors="coerce")
    return {
        "count": int(len(part)),
        "amplitude_median_cm": float(amp.median()) if amp.notna().any() else float("nan"),
        "radius_median_km": float(radius.median()) if radius.notna().any() else float("nan"),
    }


def main() -> None:
    args = parse_args()
    outputs = [str(draw_region(args.root, args.day, name, bbox)) for name, bbox in REGIONS.items()]
    rows: list[dict[str, object]] = []
    for branch, baseline, kernel in branch_specs(args.root):
        base = args.root / branch / "catalog"
        raw = read_centers(base / "raw_detection", "", args.day)
        qc = read_centers(base, "", args.day)
        final = read_centers(base / "jan1_final_catalog", "", args.day)
        for region, bbox in STATS_REGIONS.items():
            row: dict[str, object] = {"branch": branch, "baseline": baseline, "spatial_filter_kernel": kernel, "region": region}
            row["raw_seed_rows"] = int(len(in_region(raw, bbox)))
            row["ssh_discovery_pass"] = int(in_region(raw, bbox).get("ssh_primary_discovery_pass", pd.Series(dtype=bool)).fillna(False).astype(bool).sum())
            qpart = in_region(qc, bbox)
            row["shape_rejected"] = int(qpart.get("qc_class", pd.Series(dtype=str)).astype(str).eq("shape_rejected").sum())
            row["overlap_duplicate"] = int(qpart.get("qc_class", pd.Series(dtype=str)).astype(str).eq("overlap_duplicate").sum())
            row["transient"] = int(qpart.get("persistence_class", pd.Series(dtype=str)).astype(str).eq("transient").sum())
            row.update({f"final_{key}": value for key, value in region_metrics(final, bbox).items()})
            rows.append(row)
    csv_path = args.root / "comparison" / "baseline_ab_summary_19910101.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    manifest = {"status": "complete", "day": args.day, "final_catalog_policy": "Jan1 only; transient excluded after five-day support run.", "comparison": "2 SSH baselines x 3 filter kernels", "products": outputs, "summary": str(csv_path)}
    (args.root / "comparison" / "baseline_ab_manifest_19910101.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
