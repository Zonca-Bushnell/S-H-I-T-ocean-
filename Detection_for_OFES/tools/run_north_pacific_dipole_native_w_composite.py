"""Paper-style, unrotated native-W composites for configurable cyclonic samples."""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from ..ofes_io import ctl_path, expected_dta_bytes, open_dta_memmap, parse_ctl, require_daily_file
from ..run_ofes_rebuild_w import classify_native_w_multipole
from .postprocess_ofes_eddy_qc import is_open_ocean
from .run_ofes_isopycnal_composite import (
    load_table_objects,
    run_group,
    sample_object_geometry,
    write_group_outputs,
)


DEFAULT_OBJECT_TABLE = Path(
    r"E:\DATA\01_Eddy_correspond\02_OFES\Fromthebeginning\04_Vertical"
    r"\hua_center_section_selection_19910101\local_step_2cells_section_bipolar_tiebreak"
    r"\section_bipolar_vertical_catalog_local_lat_scaled\vertical_continuation_layer_catalog.csv"
)
DEFAULT_OUTPUT_ROOT = Path(
    r"E:\DATA\01_Eddy_correspond\02_OFES\Fromthebeginning\05_TEMP"
    r"\north_pacific_dipole_native_w_pointwise_unrotated_19910101"
)
DEFAULT_NH_OPEN_OCEAN_OUTPUT_ROOT = Path(
    r"E:\DATA\01_Eddy_correspond\02_OFES\Fromthebeginning\05_TEMP"
    r"\nh_open_ocean_cyclonic_dipole_native_w_pointwise_unrotated_19910101"
)
DEFAULT_DATA_ROOT = Path(r"F:\OFES\external_OFES2")
REGIONS = {
    "north_pacific_subtropical_stcc": (122.0, 203.0, 18.0, 27.0),
    "kuroshio_extension": (140.0, 180.0, 28.0, 40.0),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--object-table", type=Path, default=DEFAULT_OBJECT_TABLE)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument(
        "--selection-scope",
        choices=["north_pacific_regions", "nh_open_ocean"],
        default="north_pacific_regions",
        help="Candidate geography before the unchanged native-W dipole screen.",
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--day", default="1991-01-01")
    parser.add_argument("--max-depth-layers", type=int, default=105)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--min-objects", type=int, default=1,
        help="Display support threshold for this Jan-1 pilot; support counts remain in every output.",
    )
    parser.add_argument("--multipole-depth-min-m", type=float, default=300.0)
    parser.add_argument("--multipole-depth-max-m", type=float, default=500.0)
    parser.add_argument("--multipole-radius-inner-r", type=float, default=0.0)
    parser.add_argument("--multipole-radius-outer-r", type=float, default=1.0)
    parser.add_argument("--multipole-azimuth-count", type=int, default=60)
    parser.add_argument("--multipole-min-valid-azimuth-fraction", type=float, default=0.80)
    parser.add_argument("--multipole-min-sector-valid-fraction", type=float, default=0.35)
    parser.add_argument("--multipole-boundary-max-nan-fraction", type=float, default=0.35)
    parser.add_argument("--multipole-min-amp-1e6-m-s", type=float, default=0.5)
    parser.add_argument("--multipole-min-snr", type=float, default=2.0)
    parser.add_argument("--multipole-min-harmonic-dominance", type=float, default=1.1)
    return parser.parse_args()


def surface_input_rows(path: Path, day: str) -> pd.DataFrame:
    rows = pd.read_csv(path)
    if "date" in rows.columns:
        rows = rows[rows["date"].astype(str).eq(day)].copy()
    if "depth_index" in rows.columns:
        depth = pd.to_numeric(rows["depth_index"], errors="coerce").fillna(0)
        rows = rows[depth.eq(0)].copy()
    if "qc_pass" in rows.columns:
        passed = rows["qc_pass"]
        if passed.dtype == object:
            passed = passed.astype(str).str.strip().str.lower().isin({"1", "true", "yes", "y"})
        else:
            passed = passed.fillna(False).astype(bool)
        rows = rows[passed].copy()
    return rows


def object_is_nh_open_ocean(obj: object) -> bool:
    if float(obj.center_lat) < 0.0:
        return False
    return bool(is_open_ocean(pd.Series({
        "center_lon_refined": float(obj.center_lon),
        "center_lat_refined": float(obj.center_lat),
        "jet_meander_flag": False,
    })))


def region_for_object(obj: object, selection_scope: str) -> str:
    if selection_scope == "nh_open_ocean":
        return "nh_open_ocean" if object_is_nh_open_ocean(obj) else ""
    return next((
        name for name, (lo0, lo1, la0, la1) in REGIONS.items()
        if lo0 <= obj.center_lon <= lo1 and la0 <= obj.center_lat <= la1
    ), "")


def composite_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        max_depth_layers=args.max_depth_layers, grid_n=101, extent_r=2.0,
        day=args.day,
        cressman_radius_r=1.0, cressman_min_objects=args.min_objects,
        composite_method="pointwise_mean", max_objects_per_group=0, workers=args.workers,
        max_geometry_depth_m=2000.0, background_ring_min_r=1.5,
        background_ring_max_r=2.0, min_bracket_stratification=1.0e-4,
    )


def classify(sampled: dict[str, object], args: argparse.Namespace) -> dict[str, object]:
    grid = {
        "ofes_w_native_raw_m_s": np.asarray(sampled["native_w"], dtype="f4"),
        "depth_m": np.asarray(sampled["depth_m"], dtype="f4"),
        "x_over_r": np.asarray(sampled["x_over_r"], dtype="f4"),
        "y_over_r": np.asarray(sampled["y_over_r"], dtype="f4"),
    }
    return classify_native_w_multipole(grid, args)


def main() -> None:
    args = parse_args()
    if args.output_root is None:
        args.output_root = (
            DEFAULT_NH_OPEN_OCEAN_OUTPUT_ROOT
            if args.selection_scope == "nh_open_ocean"
            else DEFAULT_OUTPUT_ROOT
        )
    cargs = composite_args(args)
    groups = load_table_objects(args.object_table, args.day)
    source_rows = surface_input_rows(args.object_table, args.day)
    admitted_ids = set(source_rows["hua_object_id"].astype(str))
    all_cyclonic = [
        obj for obj in groups["NH_cyclonic"]
        if obj.hua_object_id in admitted_ids and region_for_object(obj, args.selection_scope)
    ]
    output_regions = ["nh_open_ocean"] if args.selection_scope == "nh_open_ocean" else list(REGIONS)
    metas = {name: parse_ctl(ctl_path(args.data_root, name)) for name in ("prho", "w")}
    day = datetime.strptime(args.day, "%Y-%m-%d").date()
    raw = {name: open_dta_memmap(require_daily_file(args.data_root, name, day, expected_dta_bytes(meta)), meta) for name, meta in metas.items()}
    rows: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(sample_object_geometry, raw, metas, obj, cargs): obj for obj in all_cyclonic}
        for future in as_completed(futures):
            obj = futures[future]
            region = region_for_object(obj, args.selection_scope)
            try:
                metrics = classify(future.result(), args)
                # The requested first-pass screen is deliberately limited to
                # a resolved azimuthal field, two zero crossings, and enough
                # native-W amplitude.  SNR/harmonic dominance remain recorded
                # diagnostics rather than adding another selection gate.
                selected = (
                    metrics["multipole_zero_crossings"] == 2
                    and np.isfinite(metrics["multipole_amp_1e6_m_s"])
                    and metrics["multipole_amp_1e6_m_s"] >= args.multipole_min_amp_1e6_m_s
                    and metrics["multipole_valid_azimuth_fraction"] >= args.multipole_min_valid_azimuth_fraction
                    and metrics["multipole_boundary_nan_fraction"] <= args.multipole_boundary_max_nan_fraction
                )
                status = "selected" if selected else "rejected"
                rows.append({
                    "hua_object_id": obj.hua_object_id, "region": region, "polarity": obj.polarity,
                    "center_lon": obj.center_lon, "center_lat": obj.center_lat, "radius_km": obj.radius_km,
                    "selection": status, **{key: value for key, value in metrics.items() if not isinstance(value, np.ndarray)},
                })
            except Exception as exc:
                rows.append({"hua_object_id": obj.hua_object_id, "region": region, "polarity": obj.polarity,
                             "center_lon": obj.center_lon, "center_lat": obj.center_lat, "radius_km": obj.radius_km,
                             "selection": "error", "multipole_class": "error", "error": str(exc)})
    selection = pd.DataFrame(rows)
    if selection.empty:
        selection = pd.DataFrame(columns=["hua_object_id", "region", "polarity", "selection"])
    else:
        selection = selection.sort_values(["region", "hua_object_id"])
    args.output_root.mkdir(parents=True, exist_ok=True)
    selection.to_csv(args.output_root / "dipole_classification_all_candidates.csv", index=False, encoding="utf-8-sig")
    selected_candidate_ids = {obj.hua_object_id for obj in all_cyclonic}
    source_rows[source_rows["hua_object_id"].astype(str).isin(selected_candidate_ids)].to_csv(
        args.output_root / "candidate_surface_objects.csv", index=False, encoding="utf-8-sig"
    )
    summaries: list[dict[str, object]] = []
    table = source_rows
    for region in output_regions:
        chosen = selection[(selection["region"] == region) & (selection["selection"] == "selected")]
        chosen_ids = set(chosen["hua_object_id"].astype(str))
        selected_rows = table[(table.get("depth_index", 0).eq(0)) & table["hua_object_id"].astype(str).isin(chosen_ids)].copy()
        selected_table = args.output_root / f"{region}_dipole_surface_objects.csv"
        selected_rows.to_csv(selected_table, index=False, encoding="utf-8-sig")
        selected_groups = load_table_objects(selected_table, args.day)
        payload = run_group(raw, metas, selected_groups["NH_cyclonic"], cargs)
        if payload is None:
            summaries.append({"region": region, "status": "no_selected_dipoles", "classified_objects": int((selection["region"] == region).sum())})
            continue
        payload.update({
            "region_definition": (
                "Northern Hemisphere open ocean: latitude >= 0 and current boundary-current/ACC exclusion mask"
                if region == "nh_open_ocean" else list(REGIONS[region])
            ),
            "orientation": "unrotated",
            "multipole_class": "dipole",
            "multipole_selection": "native_raw_w_300_500m_radial_integral_dipole_only",
            "pointwise_grid_definition": "[-2R,2R], delta=0.04R, direct finite-cell mean",
        })
        record = write_group_outputs(
            args.output_root, "paper_pointwise_no_rotation", region, payload,
            render_geometry_sections=False,
        )
        summaries.append({"region": region, "classified_objects": int((selection["region"] == region).sum()),
                          "selected_dipoles": int(len(chosen)), "status": "ok", **record})
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "day": args.day,
        "source_object_table": str(args.object_table), "source": "OFES native raw w; no rebuild-W",
        "selection_scope": args.selection_scope,
        "surface_input_policy": "depth_index=0; qc_pass=True when the source table provides qc_pass",
        "geographic_candidate_count": int(len(all_cyclonic)),
        "classification": "300-500m, 60-sector radial integration 0-1R, exactly two zero crossings and minimum native-W amplitude; SNR/harmonic diagnostics retained but not selected on",
        "composite": "unrotated direct pointwise mean on [-2R,2R] at 0.04R",
        "rotation": "not_applied", "summaries": summaries,
    }
    (args.output_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(summaries).to_csv(args.output_root / "SUMMARY.csv", index=False, encoding="utf-8-sig")
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
