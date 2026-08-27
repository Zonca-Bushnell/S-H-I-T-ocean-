from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from .axis_streamfunction_separation import (
    DEFAULT_AXIS_DIR,
    DEFAULT_CATALOG,
    DEFAULT_INPUT_DAILY,
    fit_rank1,
    grid_spacing_m,
    local_xy_m,
    parse_csv_list,
    plot_group,
    read_daily_uv,
    relative_vorticity,
    streamfunction_from_zeta,
)
from .lifecycle_common import (
    DEFAULT_LIFECYCLE_ROOT,
    DEFAULT_POLARITIES,
    DEFAULT_SHAPE_BY_SHAPE_DIR,
    DEFAULT_SHAPES,
    PHASE_NAMES,
    apply_lifecycle_limits,
    load_center_lines,
    load_lifecycle_objects,
    representative_radii,
)


DEFAULT_OUTPUT = DEFAULT_LIFECYCLE_ROOT / "streamfunction_templates"


def object_axis_xy(obj, center_line: pd.DataFrame, depth: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    if len(center_line) != len(depth):
        return None
    theta = float(obj.temp_direction_rad)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    x_rot = center_line["x_rot_m"].to_numpy(dtype="f8")
    y_rot = center_line["y_rot_m"].to_numpy(dtype="f8")
    x_axis = x_rot * cos_t - y_rot * sin_t
    y_axis = x_rot * sin_t + y_rot * cos_t
    return x_axis, y_axis


def bin_object_on_own_axis(
    obj,
    center_line: pd.DataFrame,
    lon: np.ndarray,
    lat: np.ndarray,
    depth: np.ndarray,
    psi: np.ndarray,
    radial_edges: np.ndarray,
    rmax: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    axis = object_axis_xy(obj, center_line, depth)
    if axis is None:
        return None
    x_axis, y_axis = axis
    x_grid, y_grid = local_xy_m(lon[None, :], lat[:, None], float(obj.surface_lon), float(obj.surface_lat))
    radius = float(obj.mean_radius_m)
    n_depth = len(depth)
    n_bin = len(radial_edges) - 1
    sums = np.zeros((n_depth, n_bin), dtype="f8")
    counts = np.zeros((n_depth, n_bin), dtype="i8")
    for k in range(n_depth):
        r_norm = np.hypot(x_grid - x_axis[k], y_grid - y_axis[k]) / radius
        mask = r_norm <= rmax
        if not np.any(mask):
            continue
        layer = psi[k].astype("f8")
        core_mask = r_norm <= min(0.15, rmax)
        core = np.nanmean(layer[core_mask]) if np.any(core_mask) else np.nan
        if not np.isfinite(core):
            iy, ix = np.unravel_index(np.nanargmin(r_norm), r_norm.shape)
            core = layer[iy, ix]
        values = layer[mask] - core
        bins = np.searchsorted(radial_edges, r_norm[mask], side="right") - 1
        good = (bins >= 0) & (bins < n_bin) & np.isfinite(values)
        if np.any(good):
            np.add.at(sums[k], bins[good], values[good])
            np.add.at(counts[k], bins[good], 1)
    return sums, counts


def make_accumulator(shape: tuple[int, int]) -> dict:
    return {"sums": np.zeros(shape, dtype="f8"), "counts": np.zeros(shape, dtype="i8"), "objects": set(), "dates": set()}


def write_outputs(
    output_dir: Path,
    accum: dict[tuple[str, int, str], dict],
    depth: np.ndarray,
    radial: np.ndarray,
    objects: pd.DataFrame,
    radii: dict[str, float],
    args: argparse.Namespace,
) -> None:
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    profile_rows = []
    coeff_rows = []
    metric_rows = []
    count_rows = []
    for (polarity, phase_index, phase_name), item in sorted(accum.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        sums = item["sums"]
        counts = item["counts"]
        matrix = np.divide(sums, counts, out=np.full_like(sums, np.nan, dtype="f8"), where=counts > 0)
        recon, h, r_component, metrics = fit_rank1(matrix, counts)
        residual = matrix - recon
        label = f"all_shapes_{polarity}_{phase_name}"
        plot_group(label, matrix, recon, residual, h, r_component, radial, depth, figure_dir)
        metrics.update(
            {
                "shape_class": "all_shapes",
                "polarity": polarity,
                "phase_index": int(phase_index),
                "phase_name": phase_name,
                "n_objects": len(item["objects"]),
                "n_dates": len(item["dates"]),
                "n_valid_bins": int(np.sum(counts > 0)),
                "n_total_samples": int(np.sum(counts)),
                "median_radius_m": radii.get(polarity, np.nan),
            }
        )
        metric_rows.append(metrics)
        count_rows.append(
            {
                "shape_class": "all_shapes",
                "polarity": polarity,
                "phase_index": int(phase_index),
                "phase_name": phase_name,
                "n_objects": len(item["objects"]),
                "n_dates": len(item["dates"]),
                "n_total_samples": int(np.sum(counts)),
            }
        )
        for k, depth_m in enumerate(depth):
            for j, r in enumerate(radial):
                if counts[k, j] <= 0:
                    continue
                profile_rows.append(
                    {
                        "shape_class": "all_shapes",
                        "polarity": polarity,
                        "phase_index": int(phase_index),
                        "phase_name": phase_name,
                        "depth_index": k,
                        "depth_m": float(depth_m),
                        "r_over_R": float(r),
                        "psi_mean": float(matrix[k, j]),
                        "psi_rank1": float(recon[k, j]),
                        "psi_residual": float(residual[k, j]),
                        "count": int(counts[k, j]),
                    }
                )
        for k, depth_m in enumerate(depth):
            coeff_rows.append({"shape_class": "all_shapes", "polarity": polarity, "phase_index": int(phase_index), "phase_name": phase_name, "component": "H", "index": k, "coord": float(depth_m), "value": float(h[k])})
        for j, r in enumerate(radial):
            coeff_rows.append({"shape_class": "all_shapes", "polarity": polarity, "phase_index": int(phase_index), "phase_name": phase_name, "component": "R", "index": j, "coord": float(r), "value": float(r_component[j])})

    profiles = pd.DataFrame.from_records(profile_rows)
    coeffs = pd.DataFrame.from_records(coeff_rows)
    metrics = pd.DataFrame.from_records(metric_rows)
    counts = pd.DataFrame.from_records(count_rows)
    profiles.to_parquet(output_dir / "lifecycle_radial_psi_profiles.parquet", index=False)
    coeffs.to_parquet(output_dir / "lifecycle_separable_fit_coefficients.parquet", index=False)
    coeffs.to_csv(output_dir / "lifecycle_separable_fit_coefficients.csv", index=False)
    metrics.to_csv(output_dir / "lifecycle_separability_metrics.csv", index=False)
    counts.to_csv(output_dir / "lifecycle_object_counts.csv", index=False)
    plot_metric_summary(metrics, figure_dir)
    write_summary(output_dir, objects, metrics, counts, args)


def plot_metric_summary(metrics: pd.DataFrame, figure_dir: Path) -> None:
    if metrics.empty:
        return
    for polarity, part in metrics.groupby("polarity"):
        part = part.sort_values("phase_index")
        fig, ax = plt.subplots(figsize=(8, 5), dpi=160)
        ax.plot(part["phase_name"], part["rank1_energy_fraction"], marker="o", color="#4c78a8")
        ax.set_ylim(0, 1)
        ax.set_ylabel("rank-1 energy fraction")
        ax.set_title(f"{polarity}: lifecycle separability")
        ax.grid(True, color="0.9")
        fig.tight_layout()
        fig.savefig(figure_dir / f"{polarity}_lifecycle_rank1_energy_fraction.png")
        plt.close(fig)


def write_summary(output_dir: Path, objects: pd.DataFrame, metrics: pd.DataFrame, counts: pd.DataFrame, args: argparse.Namespace) -> None:
    missing = []
    for polarity in parse_csv_list(args.polarities, DEFAULT_POLARITIES):
        for index, phase_name in enumerate(PHASE_NAMES):
            hit = counts[(counts["polarity"] == polarity) & (counts["phase_index"] == index)] if not counts.empty else counts
            if hit.empty:
                missing.append(f"{polarity}:{phase_name}")
    lines = [
        "# Lifecycle representative streamfunction summary",
        "",
        f"- Objects selected: {len(objects):,}",
        f"- Output dir: `{output_dir}`",
        f"- Input daily dir: `{args.input_daily_dir}`",
        f"- Phase names: {', '.join(PHASE_NAMES)}",
        f"- Radial range: 0 <= r/R <= {args.rmax}, bins={args.radial_bins}",
        "- Velocity input is treated as eddy perturbation `u/v`; climatology is not used in this template step.",
        "",
        "## Object Counts",
        "```csv",
        counts.to_csv(index=False).strip() if not counts.empty else "No counts generated.",
        "```",
        "",
        "## Separability Metrics",
        "```csv",
        metrics.to_csv(index=False).strip() if not metrics.empty else "No metrics generated.",
        "```",
    ]
    if missing:
        lines.extend(["", "## Skipped Phase Groups", ", ".join(missing)])
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    shapes = parse_csv_list(args.shapes, DEFAULT_SHAPES)
    polarities = parse_csv_list(args.polarities, DEFAULT_POLARITIES)
    objects = load_lifecycle_objects(
        axis_dir=Path(args.axis_dir),
        catalog_dir=Path(args.catalog_dir),
        shape_dir=Path(args.shape_dir),
        shapes=shapes,
        polarities=polarities,
    )
    objects = apply_lifecycle_limits(objects, int(args.max_days), int(args.max_objects_per_polarity), int(args.random_seed))
    if objects.empty:
        raise RuntimeError("No lifecycle objects selected.")
    center_lines = load_center_lines(Path(args.axis_dir), set(objects["eddy3d_object_id"].astype(int)))
    radial_edges = np.linspace(0.0, float(args.rmax), int(args.radial_bins) + 1)
    radial = 0.5 * (radial_edges[:-1] + radial_edges[1:])
    accum: dict[tuple[str, int, str], dict] = {}
    depth_ref: np.ndarray | None = None
    for date, day_objects in tqdm(list(objects.groupby("date")), desc="Lifecycle psi", unit="day"):
        path = Path(args.input_daily_dir) / f"uv_{pd.Timestamp(date):%Y%m%d}.nc"
        if not path.exists():
            continue
        lon, lat, depth, u, v = read_daily_uv(path)
        _, dy, dx = grid_spacing_m(lon, lat)
        zeta = relative_vorticity(lon, lat, u, v)
        psi = streamfunction_from_zeta(zeta, dx, dy)
        depth_ref = depth if depth_ref is None else depth_ref
        for obj in day_objects.itertuples(index=False):
            center_line = center_lines.get(int(obj.eddy3d_object_id))
            if center_line is None:
                continue
            binned = bin_object_on_own_axis(obj, center_line, lon, lat, depth, psi, radial_edges, float(args.rmax))
            if binned is None:
                continue
            sums, counts = binned
            key = (str(obj.polarity), int(obj.phase_index), str(obj.phase_name))
            if key not in accum:
                accum[key] = make_accumulator(sums.shape)
            accum[key]["sums"] += sums
            accum[key]["counts"] += counts
            accum[key]["objects"].add(int(obj.eddy3d_object_id))
            accum[key]["dates"].add(str(obj.date))
    if depth_ref is None:
        raise RuntimeError("No daily uv files were processed.")
    write_outputs(output_dir, accum, depth_ref, radial, objects, representative_radii(objects), args)
    print(f"Output: {output_dir}")
    print(f"Profiles: {output_dir / 'lifecycle_radial_psi_profiles.parquet'}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build lifecycle-normalized all-shape representative streamfunction templates.")
    parser.add_argument("--axis-dir", default=str(DEFAULT_AXIS_DIR))
    parser.add_argument("--catalog-dir", default=str(DEFAULT_CATALOG))
    parser.add_argument("--shape-dir", default=str(DEFAULT_SHAPE_BY_SHAPE_DIR))
    parser.add_argument("--input-daily-dir", default=str(DEFAULT_INPUT_DAILY))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--shapes", default=",".join(DEFAULT_SHAPES))
    parser.add_argument("--polarities", default=",".join(DEFAULT_POLARITIES))
    parser.add_argument("--max-days", type=int, default=0, help="Maximum dates per polarity+phase for smoke runs.")
    parser.add_argument("--max-objects-per-polarity", type=int, default=0, help="Maximum objects per polarity+phase for smoke runs.")
    parser.add_argument("--rmax", type=float, default=2.5)
    parser.add_argument("--radial-bins", type=int, default=50)
    parser.add_argument("--random-seed", type=int, default=20260710)
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
