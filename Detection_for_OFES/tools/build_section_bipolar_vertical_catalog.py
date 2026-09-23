"""Materialize continuation and section-bipolar core views of one Hua run.

The continuation view retains every layer accepted by the calibrated vertical
tracker.  The core view is deliberately stricter: an object must demonstrate
opposite signed section velocity on both 0.6R and 1.0R over a configured share
of its accepted below-surface layers.  This is a classification layer, not a
replacement for the tracker or an undocumented hard stop.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build two-tier vertical catalogs from section-bipolarity diagnostics.")
    parser.add_argument("--vertical-root", type=Path, required=True)
    parser.add_argument("--bipolarity-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--core-min-fraction", type=float, default=0.60)
    parser.add_argument("--supported-min-fraction", type=float, default=0.50)
    parser.add_argument("--strict-core-min-fraction", type=float, default=0.70)
    return parser.parse_args()


def _as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.astype("string").str.strip().str.lower().isin(("1", "true", "yes")).fillna(False)


def annotate_bilateral_segments(layers: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Mark contiguous layers that pass both section radii for each object."""
    layers = layers.sort_values(["hua_object_id", "depth_index"]).copy()
    layers["section_bipolar_both_radii"] = (
        _as_bool(layers["section_bipolar_0p6r"]) & _as_bool(layers["section_bipolar_1p0r"])
    )
    layers["section_bipolar_segment_id"] = pd.Series(pd.NA, index=layers.index, dtype="Int64")
    rows: list[dict[str, object]] = []
    for object_id, part in layers.groupby("hua_object_id", sort=False):
        active = part.loc[part["section_bipolar_both_radii"]].copy()
        if active.empty:
            rows.append({"hua_object_id": object_id, "longest_bilateral_segment_layers": 0})
            continue
        starts = active["depth_index"].diff().fillna(1).ne(1).cumsum()
        active["section_bipolar_segment_id"] = starts.astype(int)
        layers.loc[active.index, "section_bipolar_segment_id"] = active["section_bipolar_segment_id"].astype("Int64")
        segments = active.groupby("section_bipolar_segment_id", as_index=False).agg(
            layers=("depth_index", "count"),
            start_depth_m=("depth_m", "min"),
            end_depth_m=("depth_m", "max"),
        )
        best = segments.sort_values(["layers", "end_depth_m"], ascending=[False, False]).iloc[0]
        rows.append({
            "hua_object_id": object_id,
            "longest_bilateral_segment_layers": int(best["layers"]),
            "longest_bilateral_segment_start_depth_m": float(best["start_depth_m"]),
            "longest_bilateral_segment_end_depth_m": float(best["end_depth_m"]),
            "longest_bilateral_segment_id": int(best["section_bipolar_segment_id"]),
        })
    return layers, pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.supported_min_fraction <= args.core_min_fraction <= args.strict_core_min_fraction <= 1.0:
        raise ValueError("Require 0 <= supported threshold <= core threshold <= strict-core threshold <= 1")
    run = args.vertical_root / "raw_detection" / "daily_runs" / "19910101"
    object_summary = pd.read_csv(args.vertical_root / "vertical_object_summary.csv")
    layer_diag = pd.read_csv(args.bipolarity_root / "section_bipolarity_layer_diagnostics.csv")
    bipolar_objects = pd.read_csv(args.bipolarity_root / "section_bipolarity_object_summary.csv")

    score = bipolar_objects[["hua_object_id", "bipolar_fraction_0p6r", "bipolar_fraction_1p0r"]].copy()
    score["section_bipolarity_min_fraction"] = score[["bipolar_fraction_0p6r", "bipolar_fraction_1p0r"]].min(axis=1)
    score["vertical_definition"] = "kinematic_continuation_tangent45_fraction35"
    score["section_bipolar_class"] = "continuation_only"
    score.loc[score["section_bipolarity_min_fraction"].ge(args.supported_min_fraction), "section_bipolar_class"] = "section_bipolar_supported"
    score.loc[score["section_bipolarity_min_fraction"].ge(args.core_min_fraction), "section_bipolar_class"] = "section_bipolar_core"
    score.loc[score["section_bipolarity_min_fraction"].ge(args.strict_core_min_fraction), "section_bipolar_class"] = "section_bipolar_strict_core"

    catalog = object_summary.merge(score, on="hua_object_id", how="left", validate="one_to_one")
    catalog["section_bipolar_class"] = catalog["section_bipolar_class"].fillna("surface_only")
    layer_columns = [
        "hua_object_id", "depth_index", "section_bipolar_0p6r", "section_bipolar_1p0r",
        "section_pair_finite_0p6r", "section_pair_finite_1p0r",
        "u_axis_minus_0p6r", "u_axis_plus_0p6r", "u_axis_minus_1p0r", "u_axis_plus_1p0r",
    ]
    layers = pd.read_parquet(run / "structures_hua_style.parquet").merge(
        layer_diag[layer_columns], on=["hua_object_id", "depth_index"], how="left", validate="one_to_one"
    )
    layers = layers.merge(score[["hua_object_id", "section_bipolarity_min_fraction", "section_bipolar_class"]], on="hua_object_id", how="left")
    layers, segment_summary = annotate_bilateral_segments(layers)
    catalog = catalog.merge(segment_summary, on="hua_object_id", how="left", validate="one_to_one")
    core_ids = score.loc[score["section_bipolar_class"].isin(("section_bipolar_core", "section_bipolar_strict_core")), "hua_object_id"]
    core_layers = layers.loc[layers["hua_object_id"].isin(core_ids)].copy()
    strict_ids = score.loc[score["section_bipolar_class"].eq("section_bipolar_strict_core"), "hua_object_id"]
    strict_layers = layers.loc[layers["hua_object_id"].isin(strict_ids)].copy()
    strict_bilateral_layers = strict_layers.loc[strict_layers["section_bipolar_both_radii"]].copy()

    args.output_root.mkdir(parents=True, exist_ok=True)
    catalog.to_csv(args.output_root / "vertical_continuation_object_catalog.csv", index=False)
    catalog.to_parquet(args.output_root / "vertical_continuation_object_catalog.parquet", index=False)
    layers.to_csv(args.output_root / "vertical_continuation_layer_catalog.csv", index=False)
    core_layers.to_csv(args.output_root / "section_bipolar_core_layer_catalog.csv", index=False)
    core_layers.to_parquet(args.output_root / "section_bipolar_core_layer_catalog.parquet", index=False)
    strict_layers.to_csv(args.output_root / "section_bipolar_strict_core_layer_catalog.csv", index=False)
    strict_layers.to_parquet(args.output_root / "section_bipolar_strict_core_layer_catalog.parquet", index=False)
    strict_bilateral_layers.to_csv(args.output_root / "section_bipolar_strict_core_bilateral_layer_catalog.csv", index=False)
    strict_bilateral_layers.to_parquet(args.output_root / "section_bipolar_strict_core_bilateral_layer_catalog.parquet", index=False)
    summary = {
        "vertical_definition": "kinematic_continuation_tangent45_fraction35",
        "section_bipolar_measure": "opposite sign of trajectory-axis velocity sampled on both sides of local-normal section",
        "radii": ["0.6R", "1.0R"],
        "supported_min_fraction": args.supported_min_fraction,
        "core_min_fraction": args.core_min_fraction,
        "strict_core_min_fraction": args.strict_core_min_fraction,
        "continuation_objects": int(len(catalog)),
        "supported_objects": int(catalog["section_bipolar_class"].isin(("section_bipolar_supported", "section_bipolar_core", "section_bipolar_strict_core")).sum()),
        "core_objects": int(catalog["section_bipolar_class"].isin(("section_bipolar_core", "section_bipolar_strict_core")).sum()),
        "strict_core_objects": int(catalog["section_bipolar_class"].eq("section_bipolar_strict_core").sum()),
        "core_median_max_depth_m": float(catalog.loc[catalog["section_bipolar_class"].isin(("section_bipolar_core", "section_bipolar_strict_core")), "max_depth_m"].median()),
        "core_p90_max_depth_m": float(catalog.loc[catalog["section_bipolar_class"].isin(("section_bipolar_core", "section_bipolar_strict_core")), "max_depth_m"].quantile(0.90)),
        "strict_core_median_max_depth_m": float(catalog.loc[catalog["section_bipolar_class"].eq("section_bipolar_strict_core"), "max_depth_m"].median()),
        "strict_core_bilateral_layers": int(len(strict_bilateral_layers)),
        "strict_core_longest_bilateral_segment_median_layers": float(catalog.loc[catalog["section_bipolar_class"].eq("section_bipolar_strict_core"), "longest_bilateral_segment_layers"].median()),
    }
    (args.output_root / "section_bipolar_catalog_manifest.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
