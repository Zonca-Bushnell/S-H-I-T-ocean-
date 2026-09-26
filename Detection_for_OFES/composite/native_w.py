"""Composite hemisphere/polarity object-days from a chosen vertical profile."""
from __future__ import annotations

import argparse
import gc
import json
import shutil
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from Detection_for_OFES.io.ofes import ctl_path, expected_dta_bytes, open_dta_memmap, parse_ctl, require_daily_file
from Detection_for_OFES.core.regions import is_open_ocean
from Detection_for_OFES.composite.accumulator import (
    SelectedObject,
    finalize_accumulator,
    init_accumulator,
    sample_object_geometry,
    update_accumulator,
    write_group_outputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--surface-qc-root", type=Path, required=True,
                        help="Root containing daily_runs/YYYYMMDD/centers_hua_style.csv.")
    parser.add_argument("--vertical-run-root", type=Path, required=True,
                        help="Vertical root containing raw_detection/daily_runs/YYYYMMDD.")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--start", default="1991-01-01")
    parser.add_argument("--end", default="1991-01-19")
    parser.add_argument("--hemisphere", choices=("NH", "SH"), default="NH")
    parser.add_argument("--polarity", choices=("cyclonic", "anticyclonic"), default="cyclonic")
    parser.add_argument("--lat-min", type=float, help="Inclusive refined surface-latitude lower bound.")
    parser.add_argument("--lat-max", type=float, help="Surface-latitude upper bound (exclusive unless --lat-max-inclusive).")
    parser.add_argument("--lat-max-inclusive", action="store_true")
    parser.add_argument("--group-label", default="", help="Stable output label for a latitude-band composite.")
    parser.add_argument("--profile-label", default="")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-depth-layers", type=int, default=105)
    parser.add_argument("--min-objects", type=int, default=8)
    parser.add_argument(
        "--selection-mode", choices=["all_open_ocean", "all_qc_pass", "strict_core", "strict_core_nh"],
        default="all_open_ocean",
    )
    parser.add_argument(
        "--strict-core-catalog-root", type=Path,
        help=(
            "Canonical section-bipolar catalog root. Expected layout is "
            "YYYYMMDD/catalog/vertical_continuation_object_catalog.csv. "
            "When supplied, W selection consumes the existing classification and does not re-derive strict core."
        ),
    )
    parser.add_argument("--stage-raw", action="store_true",
                        help="Copy daily raw DTA files to --local-cache-root before sampling. Default reads source memmaps directly.")
    parser.add_argument("--local-cache-root", type=Path)
    parser.add_argument("--keep-local-cache", action="store_true")
    return parser.parse_args()


def selected_days(start: date, end: date) -> list[date]:
    return [start + timedelta(days=offset) for offset in range((end - start).days + 1)]


def bool_series(values: pd.Series) -> pd.Series:
    if values.dtype == object:
        return values.astype(str).str.strip().str.lower().isin({"1", "true", "yes", "y"})
    return values.fillna(False).astype(bool)


def strict_core_catalog_scores(path: Path) -> pd.DataFrame:
    """Read the canonical strict-core classification without recomputing it."""
    rows = pd.read_csv(path, low_memory=False)
    required = {"hua_object_id", "section_bipolar_class"}
    missing = required.difference(rows.columns)
    if missing:
        raise ValueError(f"Strict-core catalog is missing columns {sorted(missing)}: {path}")
    selected = rows.loc[
        rows["section_bipolar_class"].astype(str).eq("section_bipolar_strict_core")
    ].copy()
    columns = [
        name for name in (
            "hua_object_id", "accepted_below_surface_layers", "max_depth_m",
            "bipolar_fraction_0p6r", "bipolar_fraction_1p0r",
            "section_bipolarity_min_fraction", "section_bipolar_class",
        ) if name in selected.columns
    ]
    return selected[columns]


def day_objects(
    table: Path,
    current: date,
    *,
    selection_mode: str,
    strict_scores: pd.DataFrame | None = None,
    hemisphere: str = "NH",
    polarity: str = "cyclonic",
    lat_min: float | None = None,
    lat_max: float | None = None,
    lat_max_inclusive: bool = False,
) -> tuple[list[SelectedObject], pd.DataFrame]:
    rows = pd.read_csv(table, low_memory=False)
    if "depth_index" in rows.columns:
        rows = rows[pd.to_numeric(rows["depth_index"], errors="coerce").fillna(0).eq(0)].copy()
    if "qc_pass" in rows.columns:
        rows = rows[bool_series(rows["qc_pass"])].copy()
    lon_col = "center_lon_refined" if "center_lon_refined" in rows.columns else "center_lon"
    lat_col = "center_lat_refined" if "center_lat_refined" in rows.columns else "center_lat"
    selected_rows: list[pd.Series] = []
    objects: list[SelectedObject] = []
    strict_ids = (
        set(strict_scores["hua_object_id"].astype(str))
        if strict_scores is not None else set()
    )
    for _, row in rows.iterrows():
        lon = float(row[lon_col])
        lat = float(row[lat_col])
        radius = float(row["radius_km"])
        in_hemisphere = lat >= 0.0 if hemisphere == "NH" else lat < 0.0
        if str(row["polarity"]).strip().lower() != polarity or not in_hemisphere:
            continue
        if lat_min is not None and lat < lat_min:
            continue
        if lat_max is not None and (lat > lat_max if lat_max_inclusive else lat >= lat_max):
            continue
        if not (np.isfinite(lon) and np.isfinite(lat) and np.isfinite(radius) and radius > 0.0):
            continue
        object_id = str(row["hua_object_id"])
        if selection_mode == "all_open_ocean":
            if not is_open_ocean(pd.Series({
                "center_lon_refined": lon,
                "center_lat_refined": lat,
                "jet_meander_flag": False,
            })):
                continue
        elif selection_mode in {"strict_core", "strict_core_nh"} and object_id not in strict_ids:
            continue
        objects.append(SelectedObject(object_id, current.isoformat(), polarity, 1, 0.0, 0.0, radius, lon, lat))
        selected_rows.append(row)
    selected = pd.DataFrame(selected_rows)
    if not selected.empty:
        selected.insert(0, "composite_date", current.isoformat())
        if strict_scores is not None:
            selected = selected.merge(strict_scores, on="hua_object_id", how="left", validate="one_to_one")
    return objects, selected


def composite_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        max_depth_layers=int(args.max_depth_layers), grid_n=101, extent_r=2.0,
        day=f"{args.start}_{args.end}", cressman_radius_r=1.0,
        cressman_min_objects=int(args.min_objects), composite_method="pointwise_mean",
        max_objects_per_group=0, workers=int(args.workers), max_geometry_depth_m=2000.0,
        background_ring_min_r=1.5, background_ring_max_r=2.0,
        min_bracket_stratification=1.0e-4,
    )


def stage_daily_raw(
    data_root: Path,
    cache_root: Path,
    current: date,
    metas: dict[str, object],
) -> dict[str, Path]:
    daily = cache_root / f"{current:%Y%m%d}"
    daily.mkdir(parents=True, exist_ok=True)
    staged: dict[str, Path] = {}
    for name, meta in metas.items():
        expected = expected_dta_bytes(meta)
        source = require_daily_file(data_root, name, current, expected)
        target = daily / source.name
        if not target.exists() or target.stat().st_size != expected:
            partial = target.with_suffix(target.suffix + ".partial")
            partial.unlink(missing_ok=True)
            print(f"[stage] {current} {name} copy start", flush=True)
            shutil.copyfile(source, partial)
            if partial.stat().st_size != expected:
                raise OSError(f"Incomplete staged file: {partial}")
            partial.replace(target)
            print(f"[stage] {current} {name} copy complete", flush=True)
        staged[name] = target
    return staged


def native_w_metrics(payload: dict[str, object]) -> list[dict[str, object]]:
    w = np.asarray(payload["native_w_m_s"], dtype="f8")
    depth = np.asarray(payload["depth_m"], dtype="f8")
    x = np.asarray(payload["x_over_r"], dtype="f8")
    y = np.asarray(payload["y_over_r"], dtype="f8")
    xx, yy = np.meshgrid(x, y, indexing="xy")
    inside = np.hypot(xx, yy) <= 1.0
    west, east = inside & (xx < 0.0), inside & (xx > 0.0)
    rows: list[dict[str, object]] = []
    for name, lower, upper in (("surface_0_100m", 0.0, 100.0), ("mid_300_500m", 300.0, 500.0), ("paper_0_2000m", 0.0, 2000.0)):
        use = (depth >= lower) & (depth <= upper)
        field = np.nanmean(w[use], axis=0)
        west_mean = float(np.nanmean(field[west]))
        east_mean = float(np.nanmean(field[east]))
        values = field[inside]
        rows.append({
            "depth_band": name, "depth_levels": int(np.sum(use)),
            "west_mean_1e6_m_s": west_mean * 1.0e6,
            "east_mean_1e6_m_s": east_mean * 1.0e6,
            "east_west_dipole_index_1e6_m_s": abs(west_mean - east_mean) * 1.0e6,
            "inside_1R_peak_to_peak_1e6_m_s": float(np.nanmax(values) - np.nanmin(values)) * 1.0e6,
            "inside_1R_q95_abs_1e6_m_s": float(np.nanpercentile(np.abs(values), 95)) * 1.0e6,
        })
    return rows


def main() -> None:
    args = parse_args()
    start, end = date.fromisoformat(args.start), date.fromisoformat(args.end)
    days = selected_days(start, end)
    cargs = composite_args(args)
    if not 0.0 <= args.strict_core_min_fraction <= 1.0:
        raise ValueError("--strict-core-min-fraction must be between 0 and 1")
    if (args.lat_min is None) != (args.lat_max is None):
        raise ValueError("--lat-min and --lat-max must be supplied together")
    if args.lat_min is not None and args.lat_min >= args.lat_max:
        raise ValueError("--lat-min must be lower than --lat-max")
    surface_qc_root = args.surface_qc_root
    vertical_run_root = args.vertical_run_root
    metas = {name: parse_ctl(ctl_path(args.data_root, name)) for name in ("prho", "w")}
    args.output_root.mkdir(parents=True, exist_ok=True)
    cache_root = args.local_cache_root or (args.output_root / "_raw_day_cache")
    acc = None
    failures = 0
    selected_tables: list[pd.DataFrame] = []
    day_counts: list[dict[str, object]] = []

    for current in days:
        stem = f"{current:%Y%m%d}"
        table = surface_qc_root / "daily_runs" / stem / "centers_hua_style.csv"
        summary = (
            vertical_run_root
            / "raw_detection" / "daily_runs" / stem / "vertical_extension_summary.json"
        )
        if not summary.exists():
            raise FileNotFoundError(f"Vertical day is incomplete: {summary}")
        strict_scores = None
        if args.selection_mode in {"strict_core", "strict_core_nh"}:
            if not args.strict_core_catalog_root:
                raise ValueError("--strict-core-catalog-root is required for strict-core selection")
            catalog = args.strict_core_catalog_root / stem / "catalog" / "vertical_continuation_object_catalog.csv"
            if not catalog.exists():
                raise FileNotFoundError(f"Strict-core catalog is missing: {catalog}")
            strict_scores = strict_core_catalog_scores(catalog)
        objects, selected = day_objects(
            table,
            current,
            selection_mode=args.selection_mode,
            strict_scores=strict_scores,
            hemisphere=args.hemisphere,
            polarity=args.polarity,
            lat_min=args.lat_min,
            lat_max=args.lat_max,
            lat_max_inclusive=args.lat_max_inclusive,
        )
        selected_tables.append(selected)
        if args.stage_raw:
            raw_paths = stage_daily_raw(args.data_root, cache_root, current, metas)
        else:
            raw_paths = {
                name: require_daily_file(args.data_root, name, current, expected_dta_bytes(meta))
                for name, meta in metas.items()
            }
        raw = {name: open_dta_memmap(raw_paths[name], meta) for name, meta in metas.items()}
        day_failures = 0
        with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
            iterator = iter(objects)
            futures = {
                pool.submit(sample_object_geometry, raw, metas, obj, cargs)
                for _, obj in zip(range(max(1, int(args.workers))), iterator)
            }
            while futures:
                completed, _ = wait(futures, return_when=FIRST_COMPLETED)
                for future in completed:
                    futures.remove(future)
                    try:
                        next_object = next(iterator)
                    except StopIteration:
                        next_object = None
                    if next_object is not None:
                        futures.add(pool.submit(sample_object_geometry, raw, metas, next_object, cargs))
                    try:
                        sampled = future.result()
                    except Exception as exc:
                        failures += 1
                        day_failures += 1
                        print(f"[geometry] {current} object failed: {exc}", flush=True)
                        continue
                    if acc is None:
                        acc = init_accumulator(sampled, cargs)
                    update_accumulator(acc, sampled, None)
        day_counts.append({
            "date": current.isoformat(), "selected_object_days": len(objects),
            "successful_object_days": len(objects) - day_failures, "failed_object_days": day_failures,
        })
        print(f"[multiday] {current} objects={len(objects)} failed={day_failures}", flush=True)
        raw.clear()
        gc.collect()
        if args.stage_raw and not args.keep_local_cache:
            for path in raw_paths.values():
                path.unlink(missing_ok=True)
            next(iter(raw_paths.values())).parent.rmdir()

    if acc is None:
        raise RuntimeError(f"No {args.hemisphere} {args.polarity} objects were sampled")
    payload = finalize_accumulator(acc, failures, cargs)
    strict_mode = args.selection_mode in {"strict_core", "strict_core_nh"}
    payload.update({
        "region_definition": (
            f"{args.hemisphere}; no boundary-current exclusion"
            if strict_mode else
            f"{args.hemisphere}; all qc-pass surface objects without an open-ocean mask"
            if args.selection_mode == "all_qc_pass" else
            f"{args.hemisphere} open ocean using the current boundary-current/ACC exclusion mask"
        ),
        "orientation": "unrotated_geographic_east_west",
        "sample_definition": (
            f"strict_core_min_bilateral_fraction_{args.strict_core_min_fraction:.2f}_{args.polarity}_object_days"
            if strict_mode else
            f"all_qc_pass_{args.hemisphere}_{args.polarity}_object_days_no_persistence_no_dipole_prescreen"
            if args.selection_mode == "all_qc_pass" else
            f"all_open_ocean_{args.hemisphere}_{args.polarity}_object_days_no_persistence_no_dipole_prescreen"
        ),
        "pointwise_grid_definition": "[-2R,2R], delta=0.04R, direct finite-cell mean",
        "date_range": f"{args.start}/{args.end}",
    })
    group_name = args.group_label or f"{args.hemisphere}_{args.polarity}_{'strict_core' if strict_mode else 'all'}_{args.start.replace('-', '')}_{args.end.replace('-', '')}"
    record = write_group_outputs(
        args.output_root, "paper_pointwise_no_rotation", group_name,
        payload, render_geometry_sections=False,
    )
    selected_all = pd.concat(selected_tables, ignore_index=True) if selected_tables else pd.DataFrame()
    selected_all.to_csv(args.output_root / "selected_object_days.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(day_counts).to_csv(args.output_root / "object_counts_by_day.csv", index=False, encoding="utf-8-sig")
    metrics = native_w_metrics(payload)
    pd.DataFrame(metrics).to_csv(args.output_root / "native_w_composite_metrics.csv", index=False, encoding="utf-8-sig")
    (args.output_root / "native_w_composite_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "date_range": [args.start, args.end],
        "source_surface_qc_root": str(surface_qc_root),
        "source_vertical_run_root": str(vertical_run_root),
        "source": "OFES native raw w aligned to density layer centers; not rebuild-W",
        "selection": (
            f"daily qc_pass {args.hemisphere} {args.polarity} objects with both 0.6R and 1.0R section-bipolar fractions >= strict-core threshold"
            if strict_mode else
            f"all daily qc_pass {args.hemisphere} {args.polarity} surface objects; no strict-core or open-ocean mask"
            if args.selection_mode == "all_qc_pass" else
            f"all daily qc_pass {args.hemisphere} open-ocean {args.polarity} surface objects"
        ),
        "persistence": "not_used", "dipole_prescreen": "not_used", "strict_core": "not_used",
        "composite": "unrotated direct pointwise object-day mean on [-2R,2R] at 0.04R",
        "raw_read_mode": "staged_copy" if args.stage_raw else "direct_readonly_memmap",
        "profile_label": args.profile_label or None,
        "hemisphere": args.hemisphere,
        "polarity": args.polarity,
        "latitude_band": {
            "lat_min_inclusive": args.lat_min,
            "lat_max": args.lat_max,
            "lat_max_inclusive": bool(args.lat_max_inclusive),
        },
        "group_label": args.group_label or None,
        "object_day_count": int(payload["object_count"]),
        "failed_object_day_count": int(payload["failed_object_count"]),
        "days": day_counts, "native_w_metrics": metrics, "output": record,
    }
    if strict_mode:
        manifest["strict_core"] = {
            "used": True,
            "minimum_bilateral_fraction": float(args.strict_core_min_fraction),
            "radii": ["0.6R", "1.0R"],
            "source": "existing_section_bipolar_catalog" if args.strict_core_catalog_root else "diagnostic_rederivation",
            "diagnostics_root": str(args.strict_core_catalog_root or diagnostics_root),
        }
    (args.output_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
