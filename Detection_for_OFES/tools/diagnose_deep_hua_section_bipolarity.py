"""Measure the opposite-sign property shown by axis-curved section panels.

This is intentionally a post-extension diagnostic: it never changes whether a
layer was accepted.  At every accepted layer, velocity is sampled on opposite
sides of the local trajectory-normal section through the tracked center.  The
two projected axis velocities should have opposite signs for a clean vortex
section, as in the reference curve-section panels.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import map_coordinates


KM_PER_DEGREE = 111.195


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose bilateral signed-velocity coherence for accepted Hua layers.")
    parser.add_argument("--velocity-file", type=Path, required=True)
    parser.add_argument("--vertical-root", type=Path, action="append", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--day", default="1991-01-01")
    return parser.parse_args()


def _trajectory_axes(records: pd.DataFrame) -> pd.DataFrame:
    """Attach an east/north tangent matching the axis-curved panel convention."""
    pieces: list[pd.DataFrame] = []
    for _, part in records.groupby("hua_object_id", sort=False):
        part = part.sort_values("depth_index").copy()
        lon = part["center_lon"].to_numpy(dtype="f8") % 360.0
        lat = part["center_lat"].to_numpy(dtype="f8")
        depth = part["depth_m"].to_numpy(dtype="f8")
        lat0 = float(lat[0]) if len(lat) else 0.0
        x = (np.unwrap(np.deg2rad(lon)) - np.deg2rad(lon[0])) * np.cos(np.deg2rad(lat0)) * 6371.0
        y = np.deg2rad(lat - lat[0]) * 6371.0
        if len(part) >= 2:
            dx = np.gradient(x, depth, edge_order=1)
            dy = np.gradient(y, depth, edge_order=1)
            norm = np.hypot(dx, dy)
            tx = np.divide(dx, norm, out=np.full_like(dx, np.nan), where=norm > 1.0e-8)
            ty = np.divide(dy, norm, out=np.full_like(dy, np.nan), where=norm > 1.0e-8)
        else:
            tx = np.array([1.0])
            ty = np.array([0.0])
        # Match original_eddy_panels: carry the nearest valid direction through
        # nearly vertical portions of a center trajectory.
        last = (1.0, 0.0)
        for i in range(len(tx)):
            if np.isfinite(tx[i]) and np.isfinite(ty[i]):
                last = (float(tx[i]), float(ty[i]))
            else:
                tx[i], ty[i] = last
        last = (1.0, 0.0)
        for i in range(len(tx) - 1, -1, -1):
            if np.isfinite(tx[i]) and np.isfinite(ty[i]):
                last = (float(tx[i]), float(ty[i]))
            else:
                tx[i], ty[i] = last
        part["axis_tx_east"] = tx
        part["axis_ty_north"] = ty
        pieces.append(part)
    return pd.concat(pieces, ignore_index=True)


def _bilinear(field: np.ndarray, lat_index: np.ndarray, lon_index: np.ndarray) -> np.ndarray:
    return map_coordinates(field, np.vstack([lat_index, lon_index]), order=1, mode="nearest", prefilter=False)


def read_vertical_table(run: Path, stem: str) -> pd.DataFrame:
    parquet = run / f"{stem}.parquet"
    csv = run / f"{stem}.csv"
    if parquet.exists():
        try:
            return pd.read_parquet(parquet)
        except (ImportError, OSError):
            if not csv.exists():
                raise
    return pd.read_csv(csv)


def diagnose_run(
    root: Path,
    dataset: xr.Dataset,
    lat0: float,
    dlat: float,
    lon0: float,
    dlon: float,
    output_root: Path,
    day: str = "1991-01-01",
) -> dict[str, object]:
    run = root / "raw_detection" / "daily_runs" / day.replace("-", "")
    structures = read_vertical_table(run, "structures_hua_style")
    structures = structures.loc[structures["depth_index"].astype(int).gt(0)].copy()
    if structures.empty:
        raise ValueError(f"No below-surface accepted layers in {root}")
    records = _trajectory_axes(structures)
    records["section_bipolar_0p6r"] = False
    records["section_bipolar_1p0r"] = False
    records["section_pair_finite_0p6r"] = False
    records["section_pair_finite_1p0r"] = False
    records["u_axis_minus_0p6r"] = np.nan
    records["u_axis_plus_0p6r"] = np.nan
    records["u_axis_minus_1p0r"] = np.nan
    records["u_axis_plus_1p0r"] = np.nan

    for depth_index, idx in records.groupby("depth_index", sort=True).groups.items():
        positions = np.asarray(list(idx), dtype=int)
        part = records.loc[positions]
        lat = part["center_lat"].to_numpy(dtype="f8")
        lon = part["center_lon"].to_numpy(dtype="f8") % 360.0
        tx = part["axis_tx_east"].to_numpy(dtype="f8")
        ty = part["axis_ty_north"].to_numpy(dtype="f8")
        radius = part["radius_km"].to_numpy(dtype="f8")
        u = dataset["uo_glor"].isel(time=0, depth=int(depth_index)).values.astype("f8", copy=False)
        v = dataset["vo_glor"].isel(time=0, depth=int(depth_index)).values.astype("f8", copy=False)
        for label, factor in (("0p6r", 0.6), ("1p0r", 1.0)):
            # The section itself is normal to the center trajectory.  Its signed
            # value is velocity projected along the trajectory, exactly like the
            # signed_horizontal_speed panel's sign(u_axis) component.
            distance = np.maximum(2.0, factor * radius)
            nx, ny = -ty, tx
            plus_lat = lat + ny * distance / KM_PER_DEGREE
            minus_lat = lat - ny * distance / KM_PER_DEGREE
            cos_lat = np.maximum(np.cos(np.deg2rad(lat)), 0.05)
            plus_lon = (lon + nx * distance / (KM_PER_DEGREE * cos_lat)) % 360.0
            minus_lon = (lon - nx * distance / (KM_PER_DEGREE * cos_lat)) % 360.0
            plus_lat_i = (plus_lat - lat0) / dlat
            minus_lat_i = (minus_lat - lat0) / dlat
            plus_lon_i = np.mod((plus_lon - lon0) / dlon, u.shape[1])
            minus_lon_i = np.mod((minus_lon - lon0) / dlon, u.shape[1])
            up = _bilinear(u, plus_lat_i, plus_lon_i)
            vp = _bilinear(v, plus_lat_i, plus_lon_i)
            um = _bilinear(u, minus_lat_i, minus_lon_i)
            vm = _bilinear(v, minus_lat_i, minus_lon_i)
            axis_plus = up * tx + vp * ty
            axis_minus = um * tx + vm * ty
            finite = np.isfinite(axis_plus) & np.isfinite(axis_minus)
            records.loc[positions, f"u_axis_plus_{label}"] = axis_plus
            records.loc[positions, f"u_axis_minus_{label}"] = axis_minus
            records.loc[positions, f"section_pair_finite_{label}"] = finite
            records.loc[positions, f"section_bipolar_{label}"] = finite & (axis_plus * axis_minus < 0.0)

    per_object = records.groupby("hua_object_id", as_index=False).agg(
        accepted_below_surface_layers=("depth_index", "count"),
        max_depth_m=("depth_m", "max"),
        bipolar_fraction_0p6r=("section_bipolar_0p6r", "mean"),
        bipolar_fraction_1p0r=("section_bipolar_1p0r", "mean"),
        finite_pair_fraction_0p6r=("section_pair_finite_0p6r", "mean"),
        finite_pair_fraction_1p0r=("section_pair_finite_1p0r", "mean"),
    )
    name = root.name
    target = output_root / name
    target.mkdir(parents=True, exist_ok=True)
    records.to_csv(target / "section_bipolarity_layer_diagnostics.csv", index=False)
    per_object.to_csv(target / "section_bipolarity_object_summary.csv", index=False)
    row = {
        "day": day,
        "profile": name,
        "objects_with_below_surface_layers": int(len(per_object)),
        "median_max_depth_m": float(per_object["max_depth_m"].median()),
        "median_bipolar_fraction_0p6r": float(per_object["bipolar_fraction_0p6r"].median()),
        "median_bipolar_fraction_1p0r": float(per_object["bipolar_fraction_1p0r"].median()),
        "objects_bipolar_0p6r_at_least_70pct": int(per_object["bipolar_fraction_0p6r"].ge(0.70).sum()),
        "objects_bipolar_1p0r_at_least_70pct": int(per_object["bipolar_fraction_1p0r"].ge(0.70).sum()),
    }
    (target / "section_bipolarity_manifest.json").write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
    return row


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    with xr.open_dataset(args.velocity_file) as dataset:
        latitude = dataset["latitude"].values.astype("f8")
        longitude = dataset["longitude"].values.astype("f8")
        rows = [
            diagnose_run(
                root,
                dataset,
                float(latitude[0]),
                float(latitude[1] - latitude[0]),
                float(longitude[0]),
                float(longitude[1] - longitude[0]),
                args.output_root,
                args.day,
            )
            for root in args.vertical_root
        ]
    pd.DataFrame(rows).to_csv(args.output_root / "section_bipolarity_comparison.csv", index=False)
    print(json.dumps({"profiles": rows, "output_root": str(args.output_root)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
