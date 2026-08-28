from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .axis_streamfunction_separation import DEFAULT_AXIS_DIR, DEFAULT_CATALOG
from .representative_velocity_stack_tilted import fit_pooled_axis


EARTH_RADIUS_M = 6_371_000.0
DEFAULT_REPRESENTATIVENESS_DIR = Path(r"E:\Verify\outputs\kuroshio_cmems_3d\axis_streamfunction_representativeness_1993_2022")
DEFAULT_INPUT_DAILY_DIR = Path(r"E:\Verify\outputs\kuroshio_cmems_3d\input_daily")
DEFAULT_OUTPUT = Path(r"E:\Verify\outputs\kuroshio_cmems_3d\weak_unrepresented_diagnostics_1993_2022")
DEFAULT_TEMPLATE_SCOPE = "polarity_all_shapes"
REASON_COLUMNS = (
    "data_coverage_limited",
    "data_coverage_severe",
    "boundary_limited",
    "near_boundary",
    "weak_intensity",
    "weak_intensity_severe",
    "complex_or_structural_anomaly",
    "axis_fit_large_error",
)


def read_grid_bounds(input_daily_dir: Path) -> dict[str, float]:
    files = sorted(input_daily_dir.glob("uv_*.nc"))
    if not files:
        return {"lon_min": 120.0, "lon_max": 145.0, "lat_min": 20.0, "lat_max": 35.0}
    try:
        import h5py

        with h5py.File(files[0], "r") as handle:
            lon = np.asarray(handle["longitude"][:], dtype="f8")
            lat = np.asarray(handle["latitude"][:], dtype="f8")
        return {"lon_min": float(np.nanmin(lon)), "lon_max": float(np.nanmax(lon)), "lat_min": float(np.nanmin(lat)), "lat_max": float(np.nanmax(lat))}
    except Exception:
        return {"lon_min": 120.0, "lon_max": 145.0, "lat_min": 20.0, "lat_max": 35.0}


def load_base_scores(representativeness_dir: Path, template_scope: str) -> pd.DataFrame:
    scores = pd.read_parquet(representativeness_dir / "object_template_scores.parquet")
    scores = scores[scores["template_scope"] == template_scope].copy()
    scores["is_weak"] = scores["representativeness_class"] == "weak_unrepresented"
    return scores


def load_object_context(axis_dir: Path, catalog_dir: Path) -> pd.DataFrame:
    obj = pd.read_parquet(
        axis_dir / "object_diagnostics.parquet",
        columns=[
            "eddy3d_object_id",
            "surface_lon",
            "surface_lat",
            "n_layers",
            "deep_distance_m",
            "depth_span_m",
            "temp_direction_deg",
        ],
    )
    vertical = pd.read_parquet(
        catalog_dir / "vertical_objects.parquet",
        columns=["eddy3d_object_id", "mean_radius_m", "layer_count", "longitude", "latitude"],
    )
    vertical = vertical.rename(columns={"longitude": "vertical_lon", "latitude": "vertical_lat"})
    return obj.merge(vertical, on="eddy3d_object_id", how="left")


def load_strength_context(catalog_dir: Path) -> pd.DataFrame:
    centers = pd.read_parquet(catalog_dir / "layer_centers_completed.parquet", columns=["eddy3d_object_id", "speed_at_core", "radius_m"])
    center_stats = (
        centers.groupby("eddy3d_object_id", sort=False)
        .agg(
            speed_at_core_mean=("speed_at_core", "mean"),
            speed_at_core_peak=("speed_at_core", "max"),
            center_radius_m_mean=("radius_m", "mean"),
        )
        .reset_index()
    )
    observations = pd.read_parquet(catalog_dir / "layer_observations.parquet", columns=["eddy3d_object_id", "core_speed", "vorticity"])
    obs_stats = (
        observations.groupby("eddy3d_object_id", sort=False)
        .agg(
            core_speed_mean=("core_speed", "mean"),
            core_speed_peak=("core_speed", "max"),
            abs_vorticity_mean=("vorticity", nanmean_abs),
        )
        .reset_index()
    )
    return center_stats.merge(obs_stats, on="eddy3d_object_id", how="outer")


def nanmean_abs(values: pd.Series) -> float:
    data = np.abs(values.to_numpy(dtype="f8"))
    data = data[np.isfinite(data)]
    return float(np.mean(data)) if data.size else np.nan


def fit_axes(axis_dir: Path, polarities: list[str]) -> dict[str, dict[str, float]]:
    return {polarity: fit_pooled_axis(axis_dir, polarity) for polarity in polarities}


def compute_axis_rmse(axis_dir: Path, axes: dict[str, dict[str, float]]) -> pd.DataFrame:
    points = pd.read_parquet(axis_dir / "rotated_points.parquet", columns=["eddy3d_object_id", "polarity", "z_m", "x_rot_m", "y_rot_m"])
    parts = []
    for polarity, axis in axes.items():
        part = points[points["polarity"] == polarity].copy()
        if part.empty:
            continue
        z = part["z_m"].to_numpy(dtype="f8")
        x_axis = axis["c2"] * z + axis["c3"] * z * z
        y_axis = axis["c5"] * z + axis["c6"] * z * z
        dx = part["x_rot_m"].to_numpy(dtype="f8") - x_axis
        dy = part["y_rot_m"].to_numpy(dtype="f8") - y_axis
        part["axis_residual_2d_sq_m2"] = dx * dx + dy * dy
        rmse = part.groupby("eddy3d_object_id", sort=False)["axis_residual_2d_sq_m2"].mean().pow(0.5).rename("pooled_axis_rmse_m")
        parts.append(rmse.reset_index())
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["eddy3d_object_id", "pooled_axis_rmse_m"])


def add_boundary_metrics(data: pd.DataFrame, bounds: dict[str, float]) -> pd.DataFrame:
    out = data.copy()
    lon = out["surface_lon"].astype("f8")
    lat = out["surface_lat"].astype("f8")
    radius = out["mean_radius_m"].astype("f8")
    dx_w = np.radians(lon - bounds["lon_min"]) * EARTH_RADIUS_M * np.cos(np.radians(lat))
    dx_e = np.radians(bounds["lon_max"] - lon) * EARTH_RADIUS_M * np.cos(np.radians(lat))
    dy_s = np.radians(lat - bounds["lat_min"]) * EARTH_RADIUS_M
    dy_n = np.radians(bounds["lat_max"] - lat) * EARTH_RADIUS_M
    out["edge_margin_m"] = np.minimum.reduce([dx_w, dx_e, dy_s, dy_n])
    out["edge_margin_over_R"] = out["edge_margin_m"] / radius
    return out


def add_strength_proxy(data: pd.DataFrame) -> pd.DataFrame:
    out = data.copy()
    out["strength_mean_m_s"] = out["speed_at_core_mean"].where(np.isfinite(out["speed_at_core_mean"]), out["core_speed_mean"])
    out["strength_peak_m_s"] = out["speed_at_core_peak"].where(np.isfinite(out["speed_at_core_peak"]), out["core_speed_peak"])
    return out


def add_thresholds_and_reasons(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = data.copy()
    threshold_rows = []
    for polarity, part in out.groupby("polarity", sort=True):
        thresholds = {
            "polarity": polarity,
            "strength_mean_q05": float(part["strength_mean_m_s"].quantile(0.05)),
            "strength_mean_q10": float(part["strength_mean_m_s"].quantile(0.10)),
            "strength_peak_q05": float(part["strength_peak_m_s"].quantile(0.05)),
            "strength_peak_q10": float(part["strength_peak_m_s"].quantile(0.10)),
            "object_rank1_q25": float(part["object_rank1_energy_fraction"].quantile(0.25)),
            "pooled_axis_rmse_q90": float(part["pooled_axis_rmse_m"].quantile(0.90)),
        }
        threshold_rows.append(thresholds)
        mask = out["polarity"] == polarity
        out.loc[mask, "strength_mean_q10"] = thresholds["strength_mean_q10"]
        out.loc[mask, "strength_mean_q05"] = thresholds["strength_mean_q05"]
        out.loc[mask, "strength_peak_q10"] = thresholds["strength_peak_q10"]
        out.loc[mask, "strength_peak_q05"] = thresholds["strength_peak_q05"]
        out.loc[mask, "object_rank1_q25"] = thresholds["object_rank1_q25"]
        out.loc[mask, "pooled_axis_rmse_q90"] = thresholds["pooled_axis_rmse_q90"]

    out["data_coverage_limited"] = out["valid_fraction"] < 0.5
    out["data_coverage_severe"] = out["valid_fraction"] < 0.4
    out["boundary_limited"] = out["edge_margin_over_R"] < 2.5
    out["near_boundary"] = out["edge_margin_over_R"] < 3.0
    out["weak_intensity"] = (out["strength_mean_m_s"] <= out["strength_mean_q10"]) | (out["strength_peak_m_s"] <= out["strength_peak_q10"])
    out["weak_intensity_severe"] = (out["strength_mean_m_s"] <= out["strength_mean_q05"]) | (out["strength_peak_m_s"] <= out["strength_peak_q05"])
    out["complex_or_structural_anomaly"] = (out["shape_class"] == "complex") | (out["object_rank1_energy_fraction"] <= out["object_rank1_q25"])
    out["axis_fit_large_error"] = out["pooled_axis_rmse_m"] >= out["pooled_axis_rmse_q90"]
    out["reason_labels"] = out.apply(reason_labels, axis=1)
    return out, pd.DataFrame.from_records(threshold_rows)


def reason_labels(row: pd.Series) -> str:
    labels = [column for column in REASON_COLUMNS if bool(row.get(column, False))]
    return ";".join(labels) if labels else "none"


def summarize_reasons(weak: pd.DataFrame) -> pd.DataFrame:
    total = len(weak)
    rows = []
    for column in REASON_COLUMNS:
        count = int(weak[column].sum())
        rows.append({"reason": column, "n_objects": count, "fraction_of_weak": count / total if total else np.nan})
    no_label = int((weak["reason_labels"] == "none").sum())
    rows.append({"reason": "none", "n_objects": no_label, "fraction_of_weak": no_label / total if total else np.nan})
    return pd.DataFrame.from_records(rows)


def summarize_comparison(data: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "valid_fraction",
        "weighted_corr",
        "scaled_relative_rmse",
        "object_rank1_energy_fraction",
        "mean_radius_m",
        "strength_mean_m_s",
        "strength_peak_m_s",
        "edge_margin_over_R",
        "pooled_axis_rmse_m",
        "n_layers",
        "deep_distance_m",
    ]
    rows = []
    for is_weak, part in data.groupby("is_weak", sort=True):
        group = "weak_unrepresented" if is_weak else "represented"
        for metric in metrics:
            values = part[metric].dropna().astype("f8")
            if values.empty:
                continue
            rows.append(
                {
                    "group": group,
                    "metric": metric,
                    "n": int(values.size),
                    "q05": float(values.quantile(0.05)),
                    "q25": float(values.quantile(0.25)),
                    "median": float(values.quantile(0.50)),
                    "q75": float(values.quantile(0.75)),
                    "q95": float(values.quantile(0.95)),
                }
            )
    return pd.DataFrame.from_records(rows)


def summarize_shape_polarity(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (shape, polarity), part in data.groupby(["shape_class", "polarity"], sort=True):
        weak = int(part["is_weak"].sum())
        total = int(len(part))
        rows.append({"shape_class": shape, "polarity": polarity, "n_objects": total, "weak_count": weak, "weak_fraction": weak / total if total else np.nan})
    return pd.DataFrame.from_records(rows)


def summarize_time(data: pd.DataFrame) -> pd.DataFrame:
    use = data.copy()
    dates = pd.to_datetime(use["date"])
    use["year"] = dates.dt.year
    use["month"] = dates.dt.month
    rows = []
    for (year, month), part in use.groupby(["year", "month"], sort=True):
        weak = int(part["is_weak"].sum())
        total = int(len(part))
        rows.append({"year": int(year), "month": int(month), "n_objects": total, "weak_count": weak, "weak_fraction": weak / total if total else np.nan})
    return pd.DataFrame.from_records(rows)


def plot_reason_summary(summary: pd.DataFrame, figure_dir: Path) -> None:
    use = summary[summary["reason"] != "none"].copy()
    fig, ax = plt.subplots(figsize=(9, 5), dpi=160)
    ax.bar(use["reason"], use["fraction_of_weak"], color="#4c78a8")
    ax.set_ylabel("fraction of weak objects")
    ax.set_title("Weak-unrepresented reason labels")
    ax.tick_params(axis="x", rotation=35)
    ax.set_ylim(0, max(1.0, float(use["fraction_of_weak"].max()) * 1.08 if not use.empty else 1.0))
    fig.tight_layout()
    fig.savefig(figure_dir / "weak_reason_summary.png")
    plt.close(fig)


def plot_distributions(data: pd.DataFrame, figure_dir: Path) -> None:
    metrics = [
        ("valid_fraction", "valid fraction"),
        ("mean_radius_m", "mean radius m"),
        ("strength_mean_m_s", "mean core speed m/s"),
        ("pooled_axis_rmse_m", "pooled axis RMSE m"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), dpi=160)
    for ax, (metric, title) in zip(axes.ravel(), metrics):
        for is_weak, label, color in ((False, "represented", "#54a24b"), (True, "weak", "#e45756")):
            values = data.loc[data["is_weak"] == is_weak, metric].dropna().astype("f8")
            if values.empty:
                continue
            ax.hist(values, bins=50, density=True, alpha=0.45, label=label, color=color)
        ax.set_title(title)
        ax.grid(True, color="0.9")
        ax.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "weak_vs_represented_distributions.png")
    plt.close(fig)


def plot_weak_map(weak: pd.DataFrame, bounds: dict[str, float], figure_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5), dpi=160)
    scatter = ax.scatter(weak["surface_lon"], weak["surface_lat"], c=weak["valid_fraction"], s=8, cmap="viridis", alpha=0.75)
    ax.plot(
        [bounds["lon_min"], bounds["lon_max"], bounds["lon_max"], bounds["lon_min"], bounds["lon_min"]],
        [bounds["lat_min"], bounds["lat_min"], bounds["lat_max"], bounds["lat_max"], bounds["lat_min"]],
        color="black",
        linewidth=1.2,
    )
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_title("Weak-unrepresented object locations")
    fig.colorbar(scatter, ax=ax, label="valid fraction")
    fig.tight_layout()
    fig.savefig(figure_dir / "weak_object_locations.png")
    plt.close(fig)


def plot_shape_heatmap(shape_summary: pd.DataFrame, figure_dir: Path) -> None:
    if shape_summary.empty:
        return
    pivot = shape_summary.pivot(index="shape_class", columns="polarity", values="weak_fraction")
    fig, ax = plt.subplots(figsize=(6, max(4, 0.45 * len(pivot))), dpi=160)
    image = ax.imshow(pivot.to_numpy(dtype="f8"), vmin=0, vmax=max(0.05, float(np.nanmax(pivot.to_numpy(dtype="f8")))), cmap="magma", aspect="auto")
    ax.set_xticks(np.arange(len(pivot.columns)), pivot.columns)
    ax.set_yticks(np.arange(len(pivot.index)), pivot.index)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            value = pivot.iloc[i, j]
            if np.isfinite(value):
                ax.text(j, i, f"{value:.1%}", ha="center", va="center", color="white")
    ax.set_title("Weak fraction by shape and polarity")
    fig.colorbar(image, ax=ax, label="weak fraction")
    fig.tight_layout()
    fig.savefig(figure_dir / "weak_by_shape_polarity_heatmap.png")
    plt.close(fig)


def write_summary(output_dir: Path, weak: pd.DataFrame, reason_summary: pd.DataFrame, comparison: pd.DataFrame, args: argparse.Namespace) -> None:
    def reason_fraction(name: str) -> float:
        row = reason_summary[reason_summary["reason"] == name]
        return float(row["fraction_of_weak"].iloc[0]) if not row.empty else np.nan

    def median_for(group: str, metric: str) -> float:
        row = comparison[(comparison["group"] == group) & (comparison["metric"] == metric)]
        return float(row["median"].iloc[0]) if not row.empty else np.nan

    lines = [
        "# Weak-unrepresented diagnostics summary",
        "",
        f"- Template scope: `{args.template_scope}`",
        f"- Weak objects diagnosed: {len(weak):,}",
        f"- Representativeness dir: `{args.representativeness_dir}`",
        "",
        "## Five Questions",
        "",
        f"- Boundary eddies? `{reason_fraction('boundary_limited'):.1%}` are within 2.5R of the grid boundary; near-boundary `<3R` is `{reason_fraction('near_boundary'):.1%}`.",
        f"- Very weak eddies? `{reason_fraction('weak_intensity'):.1%}` are below the same-polarity 10th percentile strength threshold.",
        f"- Complex/structural anomaly? `{reason_fraction('complex_or_structural_anomaly'):.1%}` are complex or have unusually low object rank-1 separability.",
        f"- Axis fit too large? `{reason_fraction('axis_fit_large_error'):.1%}` exceed the same-polarity 90th percentile pooled-axis RMSE.",
        f"- Data coverage insufficient? `{reason_fraction('data_coverage_limited'):.1%}` have valid_fraction < 0.5; severe coverage `<0.4` is `{reason_fraction('data_coverage_severe'):.1%}`.",
        "",
        "Reason labels are multi-label, so fractions can sum above 100%.",
        "",
        "## Median Comparison",
        "",
        f"- weak valid_fraction median: {median_for('weak_unrepresented', 'valid_fraction'):.3f}; represented median: {median_for('represented', 'valid_fraction'):.3f}",
        f"- weak mean radius median: {median_for('weak_unrepresented', 'mean_radius_m'):.1f} m; represented median: {median_for('represented', 'mean_radius_m'):.1f} m",
        f"- weak strength median: {median_for('weak_unrepresented', 'strength_mean_m_s'):.3f} m/s; represented median: {median_for('represented', 'strength_mean_m_s'):.3f} m/s",
        "",
        "## Reason Table",
        "",
        "```csv",
        reason_summary.to_csv(index=False).strip(),
        "```",
    ]
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    figure_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    scores = load_base_scores(Path(args.representativeness_dir), args.template_scope)
    context = load_object_context(Path(args.axis_dir), Path(args.catalog_dir))
    strength = load_strength_context(Path(args.catalog_dir))
    axes = fit_axes(Path(args.axis_dir), sorted(scores["polarity"].unique()))
    axis_rmse = compute_axis_rmse(Path(args.axis_dir), axes)
    bounds = read_grid_bounds(Path(args.input_daily_dir))

    data = scores.merge(context, on="eddy3d_object_id", how="left")
    data = data.merge(strength, on="eddy3d_object_id", how="left")
    data = data.merge(axis_rmse, on="eddy3d_object_id", how="left")
    data = add_boundary_metrics(data, bounds)
    data = add_strength_proxy(data)
    data, thresholds = add_thresholds_and_reasons(data)

    weak = data[data["is_weak"]].copy()
    if int(args.max_weak_objects) > 0:
        weak = weak.head(int(args.max_weak_objects)).copy()

    reason_summary = summarize_reasons(weak)
    comparison = summarize_comparison(data)
    shape_summary = summarize_shape_polarity(data)
    time_summary = summarize_time(data)

    weak.to_parquet(output_dir / "weak_object_diagnostics.parquet", index=False)
    weak.to_csv(output_dir / "weak_object_diagnostics.csv", index=False)
    reason_summary.to_csv(output_dir / "weak_reason_summary.csv", index=False)
    comparison.to_csv(output_dir / "weak_vs_represented_comparison.csv", index=False)
    shape_summary.to_csv(output_dir / "weak_by_shape_polarity.csv", index=False)
    time_summary.to_csv(output_dir / "weak_by_time.csv", index=False)
    thresholds.to_csv(output_dir / "diagnostic_thresholds_by_polarity.csv", index=False)

    plot_reason_summary(reason_summary, figure_dir)
    plot_distributions(data, figure_dir)
    plot_weak_map(weak, bounds, figure_dir)
    plot_shape_heatmap(shape_summary, figure_dir)
    write_summary(output_dir, weak, reason_summary, comparison, args)

    print(f"Output: {output_dir}")
    print(f"Weak diagnostics: {output_dir / 'weak_object_diagnostics.parquet'}")
    print(f"Weak rows: {len(weak):,}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Diagnose why weak_unrepresented eddies are not represented by the common streamfunction template.")
    parser.add_argument("--representativeness-dir", default=str(DEFAULT_REPRESENTATIVENESS_DIR))
    parser.add_argument("--axis-dir", default=str(DEFAULT_AXIS_DIR))
    parser.add_argument("--catalog-dir", default=str(DEFAULT_CATALOG))
    parser.add_argument("--input-daily-dir", default=str(DEFAULT_INPUT_DAILY_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--template-scope", default=DEFAULT_TEMPLATE_SCOPE)
    parser.add_argument("--max-weak-objects", type=int, default=0)
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
