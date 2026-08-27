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
    DEFAULT_OUTPUT as DEFAULT_TEMPLATE_DIR,
    DEFAULT_POLARITIES,
    DEFAULT_SHAPE_ORDER,
    bin_object_profiles,
    fit_rank1,
    grid_spacing_m,
    limit_objects,
    load_fits,
    load_objects,
    parse_csv_list,
    read_daily_uv,
    relative_vorticity,
    streamfunction_from_zeta,
)


DEFAULT_OUTPUT = Path(r"E:\Verify\outputs\kuroshio_cmems_3d\axis_streamfunction_representativeness_1993_2022")
SCORE_SCOPES = ("shape_polarity", "polarity_all_shapes", "polarity_normalized_all_shapes")
QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)


def polarity_sign(polarity: str) -> float:
    return -1.0 if str(polarity).lower().startswith("anti") else 1.0


def load_profile_templates(template_dir: Path) -> tuple[dict[tuple[str, str, str], dict[str, np.ndarray]], np.ndarray, np.ndarray]:
    profiles = pd.read_parquet(template_dir / "radial_psi_profiles.parquet")
    depth = np.sort(profiles["depth_m"].unique().astype("f8"))
    r = np.sort(profiles["r_over_R"].unique().astype("f8"))
    templates: dict[tuple[str, str, str], dict[str, np.ndarray]] = {}

    for (shape, polarity), part in profiles.groupby(["shape_class", "polarity"], sort=True):
        if polarity == "polarity_normalized":
            key = ("shape_polarity_normalized", str(shape), "polarity_normalized")
        else:
            key = ("shape_polarity", str(shape), str(polarity))
        templates[key] = profile_part_to_template(part, depth, r)

    raw = profiles[profiles["polarity"] != "polarity_normalized"]
    for polarity, part in raw.groupby("polarity", sort=True):
        templates[("polarity_all_shapes", "all_shapes", str(polarity))] = profile_part_to_template(part, depth, r)

    normalized = profiles[profiles["polarity"] == "polarity_normalized"]
    if not normalized.empty:
        templates[("polarity_normalized_all_shapes", "all_shapes", "polarity_normalized")] = profile_part_to_template(
            normalized,
            depth,
            r,
        )
    return templates, depth, r


def profile_part_to_template(part: pd.DataFrame, depth: np.ndarray, r: np.ndarray) -> dict[str, np.ndarray]:
    depth_index = {float(value): i for i, value in enumerate(depth)}
    r_index = {float(value): i for i, value in enumerate(r)}
    sums = np.zeros((len(depth), len(r)), dtype="f8")
    counts = np.zeros((len(depth), len(r)), dtype="f8")
    for row in part.itertuples(index=False):
        i = depth_index[float(row.depth_m)]
        j = r_index[float(row.r_over_R)]
        count = float(row.count)
        sums[i, j] += float(row.psi_mean) * count
        counts[i, j] += count
    return {"sums": sums, "counts": counts}


def object_matrix(sums: np.ndarray, counts: np.ndarray) -> np.ndarray:
    return np.divide(sums, counts, out=np.full_like(sums, np.nan, dtype="f8"), where=counts > 0)


def score_against_template(
    object_sums: np.ndarray,
    object_counts: np.ndarray,
    template_sums: np.ndarray,
    template_counts: np.ndarray,
) -> dict[str, float]:
    loo_sums = template_sums - object_sums
    loo_counts = template_counts - object_counts
    loo_sums = np.where(loo_counts > 0, loo_sums, 0.0)
    loo_counts = np.where(loo_counts > 0, loo_counts, 0.0)

    obj = object_matrix(object_sums, object_counts)
    tmpl = object_matrix(loo_sums, loo_counts)
    valid = (object_counts > 0) & (loo_counts > 0) & np.isfinite(obj) & np.isfinite(tmpl)
    total_bins = obj.size
    valid_fraction = float(np.sum(valid) / total_bins) if total_bins else 0.0
    if not np.any(valid):
        return {
            "weighted_corr": np.nan,
            "scale_factor": np.nan,
            "scaled_rmse": np.nan,
            "scaled_relative_rmse": np.nan,
            "explained_fraction": np.nan,
            "valid_fraction": valid_fraction,
            "n_valid_bins": 0,
        }

    weights = object_counts[valid].astype("f8")
    x = obj[valid].astype("f8")
    y = tmpl[valid].astype("f8")
    wsum = float(np.sum(weights))
    x_mean = float(np.sum(weights * x) / wsum)
    y_mean = float(np.sum(weights * y) / wsum)
    x0 = x - x_mean
    y0 = y - y_mean
    x_var = float(np.sum(weights * x0 * x0))
    y_var = float(np.sum(weights * y0 * y0))
    raw_corr = float(np.sum(weights * x0 * y0) / np.sqrt(x_var * y_var)) if x_var > 0 and y_var > 0 else np.nan
    corr = abs(raw_corr) if np.isfinite(raw_corr) else np.nan

    denom = float(np.sum(weights * y * y))
    scale = float(np.sum(weights * x * y) / denom) if denom > 0 else np.nan
    if np.isfinite(scale):
        residual = x - scale * y
        sse = float(np.sum(weights * residual * residual))
        rmse = float(np.sqrt(sse / wsum))
    else:
        sse = np.nan
        rmse = np.nan
    energy = float(np.sum(weights * x * x))
    rel_rmse = float(np.sqrt(sse / energy)) if np.isfinite(sse) and energy > 0 else np.nan
    explained = float(1.0 - sse / energy) if np.isfinite(sse) and energy > 0 else np.nan
    return {
        "weighted_corr": corr,
        "raw_weighted_corr": raw_corr,
        "scale_factor": scale,
        "scaled_rmse": rmse,
        "scaled_relative_rmse": rel_rmse,
        "explained_fraction": explained,
        "valid_fraction": valid_fraction,
        "n_valid_bins": int(np.sum(valid)),
    }


def representativeness_class(row: dict[str, float]) -> str:
    corr = row.get("weighted_corr", np.nan)
    rel = row.get("scaled_relative_rmse", np.nan)
    valid = row.get("valid_fraction", np.nan)
    if np.isfinite(corr) and np.isfinite(rel) and np.isfinite(valid):
        if corr >= 0.8 and rel <= 0.6 and valid >= 0.6:
            return "strong"
        if corr >= 0.6 and rel <= 0.8 and valid >= 0.5:
            return "moderate"
    return "weak_unrepresented"


def build_score_rows(
    obj,
    sums: np.ndarray,
    sums_norm: np.ndarray,
    counts: np.ndarray,
    templates: dict[tuple[str, str, str], dict[str, np.ndarray]],
    object_rank1_energy_fraction: float,
) -> list[dict]:
    rows: list[dict] = []
    object_keys = [
        ("shape_polarity", str(obj.shape_class), str(obj.polarity), sums),
        ("polarity_all_shapes", "all_shapes", str(obj.polarity), sums),
        ("polarity_normalized_all_shapes", "all_shapes", "polarity_normalized", sums_norm),
    ]
    for scope, template_shape, template_polarity, object_sums in object_keys:
        template = templates.get((scope, template_shape, template_polarity))
        if template is None:
            continue
        metrics = score_against_template(object_sums, counts, template["sums"], template["counts"])
        row = {
            "eddy3d_object_id": int(obj.eddy3d_object_id),
            "date": str(obj.date),
            "track3d_id": int(obj.track3d_id),
            "shape_class": str(obj.shape_class),
            "polarity": str(obj.polarity),
            "template_scope": scope,
            "template_shape_class": template_shape,
            "template_polarity": template_polarity,
            "object_rank1_energy_fraction": float(object_rank1_energy_fraction),
        }
        row.update(metrics)
        row["representativeness_class"] = representativeness_class(row)
        rows.append(row)
    return rows


def summarize_scores(scores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    group_cols = ["template_scope", "shape_class", "polarity"]
    summary = summarize_grouped(scores, group_cols)

    polarity_rows = []
    pooled = scores[scores["template_scope"] == "polarity_all_shapes"]
    if not pooled.empty:
        polarity_rows.append(summarize_grouped(pooled, ["template_scope", "polarity"]))
    normalized = scores[scores["template_scope"] == "polarity_normalized_all_shapes"]
    if not normalized.empty:
        overall = summarize_grouped(normalized.assign(polarity="all_polarities"), ["template_scope", "polarity"])
        polarity_rows.append(overall)
    by_polarity = pd.concat(polarity_rows, ignore_index=True) if polarity_rows else pd.DataFrame()

    quantiles = summarize_quantiles(scores)
    return summary, by_polarity, quantiles


def summarize_grouped(scores: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows = []
    for key, part in scores.groupby(group_cols, sort=True):
        if not isinstance(key, tuple):
            key = (key,)
        out = dict(zip(group_cols, key))
        n = int(part["eddy3d_object_id"].nunique())
        strong = int(np.sum(part["representativeness_class"] == "strong"))
        moderate = int(np.sum(part["representativeness_class"] == "moderate"))
        represented = strong + moderate
        out.update(
            {
                "n_objects": n,
                "strong_count": strong,
                "moderate_count": moderate,
                "represented_count": represented,
                "strong_fraction": strong / n if n else np.nan,
                "represented_fraction": represented / n if n else np.nan,
                "median_weighted_corr": float(part["weighted_corr"].median()),
                "median_scaled_relative_rmse": float(part["scaled_relative_rmse"].median()),
                "median_explained_fraction": float(part["explained_fraction"].median()),
                "median_object_rank1_energy_fraction": float(part["object_rank1_energy_fraction"].median()),
                "median_valid_fraction": float(part["valid_fraction"].median()),
            }
        )
        rows.append(out)
    return pd.DataFrame.from_records(rows)


def summarize_quantiles(scores: pd.DataFrame) -> pd.DataFrame:
    value_cols = [
        "weighted_corr",
        "raw_weighted_corr",
        "scaled_relative_rmse",
        "explained_fraction",
        "object_rank1_energy_fraction",
        "valid_fraction",
    ]
    rows = []
    for (scope, shape, polarity), part in scores.groupby(["template_scope", "shape_class", "polarity"], sort=True):
        for col in value_cols:
            values = part[col].dropna().astype("f8")
            if values.empty:
                continue
            for q in QUANTILES:
                rows.append(
                    {
                        "template_scope": scope,
                        "shape_class": shape,
                        "polarity": polarity,
                        "metric": col,
                        "quantile": q,
                        "value": float(values.quantile(q)),
                    }
                )
    return pd.DataFrame.from_records(rows)


def plot_outputs(scores: pd.DataFrame, by_polarity: pd.DataFrame, summary: pd.DataFrame, output_dir: Path) -> None:
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    plot_polarity_bars(by_polarity, figure_dir)
    plot_score_distributions(scores, figure_dir)
    plot_shape_polarity_heatmap(summary, figure_dir)
    plot_object_rank1_distribution(scores, figure_dir)


def plot_polarity_bars(by_polarity: pd.DataFrame, figure_dir: Path) -> None:
    if by_polarity.empty:
        return
    use = by_polarity.copy()
    use["label"] = use["template_scope"] + "\n" + use["polarity"]
    x = np.arange(len(use))
    fig, ax = plt.subplots(figsize=(max(8, 0.9 * len(use)), 5), dpi=160)
    ax.bar(x, use["strong_fraction"], label="strong", color="#4c78a8")
    ax.bar(x, use["represented_fraction"] - use["strong_fraction"], bottom=use["strong_fraction"], label="moderate", color="#f58518")
    ax.set_xticks(x, use["label"], rotation=30, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("object fraction")
    ax.set_title("Represented objects by pooled polarity template")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "representativeness_by_polarity.png")
    plt.close(fig)


def plot_score_distributions(scores: pd.DataFrame, figure_dir: Path) -> None:
    use = scores[scores["template_scope"] == "polarity_all_shapes"]
    if use.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), dpi=160)
    for polarity, part in use.groupby("polarity", sort=True):
        axes[0].hist(part["weighted_corr"].dropna(), bins=40, alpha=0.55, label=polarity)
        axes[1].hist(part["scaled_relative_rmse"].dropna(), bins=40, alpha=0.55, label=polarity)
    axes[0].set_xlabel("weighted corr")
    axes[0].set_ylabel("object count")
    axes[0].set_title("Template correlation")
    axes[1].set_xlabel("scaled relative RMSE")
    axes[1].set_title("Template error")
    for ax in axes:
        ax.grid(True, color="0.9")
        ax.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "pooled_polarity_score_distributions.png")
    plt.close(fig)


def plot_shape_polarity_heatmap(summary: pd.DataFrame, figure_dir: Path) -> None:
    use = summary[summary["template_scope"] == "polarity_all_shapes"]
    if use.empty:
        return
    pivot = use.pivot(index="shape_class", columns="polarity", values="represented_fraction")
    fig, ax = plt.subplots(figsize=(6, max(4, 0.45 * len(pivot))), dpi=160)
    image = ax.imshow(pivot.to_numpy(dtype="f8"), vmin=0, vmax=1, cmap="viridis", aspect="auto")
    ax.set_xticks(np.arange(len(pivot.columns)), pivot.columns)
    ax.set_yticks(np.arange(len(pivot.index)), pivot.index)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            value = pivot.iloc[i, j]
            if np.isfinite(value):
                ax.text(j, i, f"{value:.0%}", ha="center", va="center", color="white" if value < 0.55 else "black")
    ax.set_title("Represented fraction by shape against pooled polarity template")
    fig.colorbar(image, ax=ax, label="strong + moderate fraction")
    fig.tight_layout()
    fig.savefig(figure_dir / "shape_to_pooled_polarity_heatmap.png")
    plt.close(fig)


def plot_object_rank1_distribution(scores: pd.DataFrame, figure_dir: Path) -> None:
    use = scores[scores["template_scope"] == "shape_polarity"].drop_duplicates("eddy3d_object_id")
    if use.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5), dpi=160)
    ax.hist(use["object_rank1_energy_fraction"].dropna(), bins=50, color="#54a24b", alpha=0.85)
    ax.set_xlabel("object rank-1 energy fraction")
    ax.set_ylabel("object count")
    ax.set_title("Object-level separability")
    ax.grid(True, color="0.9")
    fig.tight_layout()
    fig.savefig(figure_dir / "object_rank1_energy_fraction_distribution.png")
    plt.close(fig)


def write_summary(
    output_dir: Path,
    *,
    objects: pd.DataFrame,
    scores: pd.DataFrame,
    by_polarity: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    lines = [
        "# Axis streamfunction representativeness summary",
        "",
        f"- Axis dir: `{args.axis_dir}`",
        f"- Template dir: `{args.template_dir}`",
        f"- Input daily dir: `{args.input_daily_dir}`",
        f"- Objects selected: {len(objects):,}",
        f"- Objects scored: {scores['eddy3d_object_id'].nunique() if not scores.empty else 0:,}",
        f"- Score rows: {len(scores):,}",
        f"- Radial range: 0 <= r/R <= {args.rmax}, bins={args.radial_bins}",
        "",
        "## Main pooled-polarity answer",
        "",
    ]
    if by_polarity.empty:
        lines.append("No pooled-polarity summary was generated.")
    else:
        for row in by_polarity.itertuples(index=False):
            lines.append(
                "- "
                f"{row.template_scope} / {row.polarity}: "
                f"strong={row.strong_fraction:.1%}, "
                f"strong+moderate={row.represented_fraction:.1%}, "
                f"n={int(row.n_objects):,}, "
                f"median_corr={row.median_weighted_corr:.3f}, "
                f"median_scaled_relative_rmse={row.median_scaled_relative_rmse:.3f}"
            )
    lines.extend(
        [
            "",
            "## Classification thresholds",
            "",
            "- strong: weighted_corr >= 0.8, scaled_relative_rmse <= 0.6, valid_fraction >= 0.6",
            "- moderate: weighted_corr >= 0.6, scaled_relative_rmse <= 0.8, valid_fraction >= 0.5",
            "- weak_unrepresented: all remaining objects",
        ]
    )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    shapes = parse_csv_list(args.shapes, DEFAULT_SHAPE_ORDER)
    polarities = parse_csv_list(args.polarities, DEFAULT_POLARITIES)
    objects = load_objects(Path(args.axis_dir), Path(args.catalog_dir), shapes, polarities)
    if args.start_date:
        objects = objects[objects["date"] >= args.start_date]
    if args.end_date:
        objects = objects[objects["date"] <= args.end_date]
    objects = limit_objects(objects, int(args.max_objects_per_group), int(args.random_seed))
    if int(args.max_days) > 0:
        keep_dates = sorted(objects["date"].unique())[: int(args.max_days)]
        objects = objects[objects["date"].isin(keep_dates)].copy()

    fits = load_fits(Path(args.axis_dir))
    templates, _, _ = load_profile_templates(Path(args.template_dir))
    radial_edges = np.linspace(0.0, float(args.rmax), int(args.radial_bins) + 1)

    score_rows: list[dict] = []
    skipped_missing_daily = 0
    skipped_missing_fit = 0
    for date, day_objects in tqdm(list(objects.groupby("date")), desc="Score daily objects", unit="day"):
        path = Path(args.input_daily_dir) / f"uv_{pd.Timestamp(date):%Y%m%d}.nc"
        if not path.exists():
            skipped_missing_daily += len(day_objects)
            continue
        lon, lat, depth, u, v = read_daily_uv(path)
        _, dy, dx = grid_spacing_m(lon, lat)
        zeta = relative_vorticity(lon, lat, u, v)
        psi = streamfunction_from_zeta(zeta, dx, dy)

        for obj in day_objects.itertuples(index=False):
            fit = fits.get((str(obj.shape_class), str(obj.polarity)))
            if fit is None:
                skipped_missing_fit += 1
                continue
            sums, sums_norm, counts = bin_object_profiles(
                obj,
                fit,
                lon,
                lat,
                depth,
                psi,
                rmax=float(args.rmax),
                radial_edges=radial_edges,
            )
            matrix = object_matrix(sums, counts)
            _, _, _, rank1_metrics = fit_rank1(matrix, counts)
            object_rank1 = rank1_metrics.get("rank1_energy_fraction", np.nan)
            score_rows.extend(build_score_rows(obj, sums, sums_norm, counts, templates, object_rank1))

    scores = pd.DataFrame.from_records(score_rows)
    summary, by_polarity, quantiles = summarize_scores(scores) if not scores.empty else (pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
    scores.to_parquet(output_dir / "object_template_scores.parquet", index=False)
    summary.to_csv(output_dir / "representativeness_summary.csv", index=False)
    by_polarity.to_csv(output_dir / "representativeness_by_polarity.csv", index=False)
    quantiles.to_csv(output_dir / "representativeness_quantiles.csv", index=False)
    plot_outputs(scores, by_polarity, summary, output_dir)
    write_summary(output_dir, objects=objects, scores=scores, by_polarity=by_polarity, args=args)

    if skipped_missing_daily or skipped_missing_fit:
        skipped = pd.DataFrame(
            [
                {"reason": "missing_daily_uv", "n_objects": skipped_missing_daily},
                {"reason": "missing_fit_coefficients", "n_objects": skipped_missing_fit},
            ]
        )
        skipped.to_csv(output_dir / "skipped_objects.csv", index=False)

    print(f"Output: {output_dir}")
    print(f"Scores: {output_dir / 'object_template_scores.parquet'}")
    print(f"By polarity: {output_dir / 'representativeness_by_polarity.csv'}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Quantify how many individual eddies are represented by common psi(r,z) templates.")
    parser.add_argument("--axis-dir", default=str(DEFAULT_AXIS_DIR))
    parser.add_argument("--catalog-dir", default=str(DEFAULT_CATALOG))
    parser.add_argument("--input-daily-dir", default=str(DEFAULT_INPUT_DAILY))
    parser.add_argument("--template-dir", default=str(DEFAULT_TEMPLATE_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--shapes", default=",".join(DEFAULT_SHAPE_ORDER))
    parser.add_argument("--polarities", default=",".join(DEFAULT_POLARITIES))
    parser.add_argument("--start-date", default="")
    parser.add_argument("--end-date", default="")
    parser.add_argument("--max-days", type=int, default=0)
    parser.add_argument("--max-objects-per-group", type=int, default=0)
    parser.add_argument("--rmax", type=float, default=2.5)
    parser.add_argument("--radial-bins", type=int, default=50)
    parser.add_argument("--random-seed", type=int, default=20260708)
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
