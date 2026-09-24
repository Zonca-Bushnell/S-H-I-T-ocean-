"""Check raw-density core versus outer-ring signals for all 435 strict cores."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage

from ..ofes_io import ctl_path, expected_dta_bytes, open_dta_memmap, parse_ctl, require_daily_file
from ..run_ofes_rebuild_w import local_lon_lat_grid, parse_iso_date
from ..w_rebuild_config import DEFAULT_DATA_ROOT


DEFAULT_SELECTION = Path(
    r"E:\DATA\01_Eddy_correspond\02_OFES\Fromthebeginning\05_TEMP"
    r"\nh_cyclonic_strict_core_native_w_pointwise_unrotated_eta_19910101_19910119\selected_object_days.csv"
)
DEFAULT_OUTPUT = Path(
    r"E:\DATA\01_Eddy_correspond\02_OFES\Fromthebeginning\05_TEMP"
    r"\nh_cyclonic_strict_core_native_w_pointwise_unrotated_eta_19910101_19910119"
    r"\paper_pointwise_no_rotation\NH_cyclone_strict_core_19d\raw_object_density_ring_check"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-file", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--grid-n", type=int, default=61)
    return parser.parse_args()


def sample_layer(raw: np.memmap, meta, layer_index: int, lon_grid: np.ndarray, lat_grid: np.ndarray) -> np.ndarray:
    lon, lat = np.asarray(meta.x.values), np.asarray(meta.y.values)
    dx, dy = float(np.nanmedian(np.diff(lon))), float(np.nanmedian(np.diff(lat)))
    ii = ((lon_grid - lon[0]) / dx) % lon.size
    jj = np.clip((lat_grid - lat[0]) / dy, 0, lat.size - 1)
    layer = np.asarray(raw[:, :, layer_index], dtype="f4")
    layer[np.abs(layer) > 1.0e30] = np.nan
    result = ndimage.map_coordinates(layer, np.vstack([ii.ravel(), jj.ravel()]), order=1, mode="wrap", cval=np.nan).reshape(lon_grid.shape)
    finite = result[np.isfinite(result)]
    if finite.size and float(np.nanmedian(finite)) > 1000.0:
        result = result - 1000.0
    return result


def stats(values: pd.Series) -> dict[str, float | int]:
    valid = values[np.isfinite(values)]
    return {
        "n": int(valid.size), "positive_fraction": float((valid > 0.0).mean()) if valid.size else float("nan"),
        "median_kg_m3": float(valid.median()) if valid.size else float("nan"),
        "p25_kg_m3": float(valid.quantile(0.25)) if valid.size else float("nan"),
        "p75_kg_m3": float(valid.quantile(0.75)) if valid.size else float("nan"),
    }


def main() -> None:
    args = parse_args()
    selected = pd.read_csv(args.selection_file)
    selected["strict_min"] = selected[["bipolar_fraction_0p6r", "bipolar_fraction_1p0r"]].min(axis=1)
    selected = selected.loc[selected.strict_min.ge(0.70) & selected.polarity.astype(str).eq("cyclonic")].copy()
    if len(selected) != 435:
        raise RuntimeError(f"Expected 435 strict-core cyclonic object-days, found {len(selected)}")
    meta = parse_ctl(ctl_path(args.data_root, "prho"))
    depth = np.asarray(meta.z.values, dtype="f8")
    selected_layers = {"about_300m": int(np.nanargmin(np.abs(depth - 300.0))), "about_500m": int(np.nanargmin(np.abs(depth - 500.0)))}
    rows: list[dict[str, object]] = []
    axis = np.linspace(-2.0, 2.0, args.grid_n, dtype="f8")
    xx, yy = np.meshgrid(axis, axis)
    core = np.hypot(xx, yy) <= 0.6
    outer = (np.hypot(xx, yy) >= 1.5) & (np.hypot(xx, yy) <= 2.0)
    for day_value, group in selected.groupby("composite_date", sort=True):
        day = parse_iso_date(str(day_value))
        raw = open_dta_memmap(require_daily_file(args.data_root, "prho", day, expected_dta_bytes(meta)), meta)
        for _, obj in group.iterrows():
            radius = float(obj.radius_km)
            lon, lat = local_lon_lat_grid(xx, yy, float(obj.center_lon_refined), float(obj.center_lat_refined), radius)
            record: dict[str, object] = {"composite_date": str(day_value), "hua_object_id": str(obj.hua_object_id), "radius_km": radius}
            for label, layer in selected_layers.items():
                field = sample_layer(raw, meta, layer, lon, lat)
                core_mean, ring_mean = float(np.nanmean(field[core])), float(np.nanmean(field[outer]))
                record[f"{label}_depth_m"] = float(depth[layer])
                record[f"{label}_core_minus_outer_kg_m3"] = core_mean - ring_mean
                record[f"{label}_core_kg_m3"] = core_mean
                record[f"{label}_outer_kg_m3"] = ring_mean
            rows.append(record)
        print(f"[raw-prho-ring] {day_value} objects={len(group)}", flush=True)
    results = pd.DataFrame(rows)
    args.output_root.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output_root / "raw_prho_core_minus_outer_by_object_day.csv", index=False)
    summary = {
        "object_days": int(len(results)), "definition": "raw native OFES prho mean(r<=0.6R) minus mean(1.5R<=r<=2R)",
        "selection": "NH cyclonic strict core; min(f_0.6R, f_1.0R) >= 0.70", "grid": "[-2R,2R], 61x61 local equal-distance sample",
        "about_300m": stats(results["about_300m_core_minus_outer_kg_m3"]),
        "about_500m": stats(results["about_500m_core_minus_outer_kg_m3"]),
        "positive_at_both_depths_fraction": float(((results["about_300m_core_minus_outer_kg_m3"] > 0.0) & (results["about_500m_core_minus_outer_kg_m3"] > 0.0)).mean()),
    }
    path = args.output_root / "raw_prho_core_minus_outer_summary.json"
    partial = path.with_suffix(".json.part")
    partial.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(partial, path)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
