from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from netCDF4 import Dataset, num2date
from scipy.io import savemat

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eddy_pipeline.detection_hybrid import DetectionParams, _hua_verify_radius
from src.eddy_pipeline.utils.table_io import DEFAULT_PARQUET_ENGINE


PRODUCTION_PARAMS = {
    "ssh_window_cells": 7,
    "start_radius_cells": 2,
    "max_radius_cells": 8,
    "speed_ratio_max": 3.0,
    "angle_jump_max_deg": 150.0,
    "tangent_tolerance_deg": 24.0,
    "symmetry_tolerance_deg": 120.0,
    "min_tangent_fraction": 0.55,
    "min_reversal_fraction": 0.55,
    "min_finite_fraction": 0.75,
    "direction_exception_extra": 2,
    "surface_search_cells": 3,
    "deep_search_cells": 3,
    "require_boundary_monotonic_rotation": True,
    "boundary_monotonic_exception_limit": 0,
    "streamline_direction_exception_fraction": 0.10,
    "streamline_step_cells": 0.5,
    "streamline_max_steps": 180,
    "streamline_start_angles": 4,
    "streamline_closure_tolerance_cells": 1.75,
    "streamline_min_winding_turns": 0.75,
    "streamline_min_points": 16,
    "hua_backend": "python",
}


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path, engine=DEFAULT_PARQUET_ENGINE)
    return pd.read_csv(path)


def create_panel_bridge(detection_dir: Path, bridge_root: Path, *, shape_class: str = "detect_only") -> dict[str, object]:
    structures_path = detection_dir / "structures_hua_style.parquet"
    if not structures_path.exists():
        structures_path = detection_dir / "structures_hua_style.csv"
    structures = _read_table(structures_path)
    if structures.empty:
        raise ValueError(f"No passed structures found in {detection_dir}")
    structures = structures.copy()
    structures["date"] = structures["date"].astype(str)
    object_codes = {key: idx + 1 for idx, key in enumerate(sorted(structures["hua_object_id"].astype(str).unique()))}
    centers = pd.DataFrame(
        {
            "date": structures["date"],
            "eddy3d_object_id": structures["hua_object_id"].astype(str).map(object_codes).astype(int),
            "track3d_id": structures["hua_object_id"].astype(str).map(object_codes).astype(int),
            "depth_index": structures["depth_index"].astype(int),
            "depth_m": structures["depth_m"].astype(float),
            "longitude": structures["center_lon"].astype(float),
            "latitude": structures["center_lat"].astype(float),
            "radius_m": structures["radius_km"].astype(float) * 1000.0,
            "polarity": structures["polarity"].astype(str),
            "source_hua_object_id": structures["hua_object_id"].astype(str),
        }
    )
    centers = centers[np.isfinite(centers["radius_m"]) & centers["radius_m"].gt(0)].copy()
    catalog_dir = bridge_root / "catalog"
    shape_dir = bridge_root / "shape_classification_detect_only"
    catalog_dir.mkdir(parents=True, exist_ok=True)
    shape_dir.mkdir(parents=True, exist_ok=True)
    centers.to_parquet(catalog_dir / "layer_centers_completed.parquet", engine=DEFAULT_PARQUET_ENGINE, index=False)
    centers.to_csv(catalog_dir / "layer_centers_completed.csv", index=False)
    shape_tracks = (
        centers.groupby("track3d_id", as_index=False)
        .agg(
            n_object_days=("date", "nunique"),
            n_layers=("depth_index", "count"),
            source_hua_object_id=("source_hua_object_id", "first"),
        )
    )
    shape_tracks.insert(1, "shape_class", shape_class)
    shape_tracks.to_parquet(shape_dir / "shape_tracks.parquet", engine=DEFAULT_PARQUET_ENGINE, index=False)
    shape_tracks.to_csv(shape_dir / "shape_tracks.csv", index=False)
    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "detection_dir": str(detection_dir),
        "bridge_root": str(bridge_root),
        "shape_dir_name": "shape_classification_detect_only",
        "n_center_rows": int(len(centers)),
        "n_objects": int(centers["eddy3d_object_id"].nunique()),
        "note": "Detect-only bridge for panel-family diagnostics; not a formal tracking/catalog/shape product.",
    }
    (bridge_root / "panel_bridge_manifest.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata


def _time_index(nc_path: Path, target_date: str) -> int:
    with Dataset(nc_path) as ds:
        var = ds.variables["time"]
        dates = num2date(var[:], units=var.units, calendar=getattr(var, "calendar", "standard"))
        labels = [getattr(d, "strftime")("%Y-%m-%d") for d in dates]
    return labels.index(target_date)


def plot_surface_overview(detection_dir: Path, filter_root: Path, filter_template: str, output_dir: Path, *, date_text: str) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    centers_path = detection_dir / "centers_hua_style.parquet"
    if not centers_path.exists():
        centers_path = detection_dir / "centers_hua_style.csv"
    centers = _read_table(centers_path)
    surface = centers[centers["depth_index"].astype(int).eq(0)].copy()
    year = int(date_text[:4])
    nc_path = filter_root / filter_template.format(year=year)
    ti = _time_index(nc_path, date_text)
    with Dataset(nc_path) as ds:
        lon = np.asarray(ds.variables["longitude"][:], dtype="f8")
        lat = np.asarray(ds.variables["latitude"][:], dtype="f8")
        zos = np.asarray(ds.variables["zos_glor"][ti, :, :], dtype="f8")
    output_dir.mkdir(parents=True, exist_ok=True)
    mode = str(json.loads((detection_dir / "run_summary.json").read_text(encoding="utf-8")).get("boundary_mode", detection_dir.name))
    out = output_dir / f"surface_overview_{mode}_{date_text.replace('-', '')}.png"
    vmax = float(np.nanpercentile(np.abs(zos), 98))
    vmax = max(vmax, 1e-6)
    fig, ax = plt.subplots(figsize=(13.5, 5.0), constrained_layout=True)
    im = ax.pcolormesh(lon, lat, zos, shading="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    passed = surface["hua_pass"].astype(bool)
    ax.scatter(surface.loc[~passed, "seed_lon"], surface.loc[~passed, "seed_lat"], s=9, c="#8b8f98", alpha=0.55, label="failed surface seed")
    ax.scatter(surface.loc[passed, "center_lon"], surface.loc[passed, "center_lat"], s=16, c="#16a34a", edgecolors="black", linewidths=0.25, label="surface Hua pass")
    ax.set_title(f"{mode} surface overview {date_text}: {int(passed.sum())}/{len(surface)} passed")
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.grid(color="0.85", linewidth=0.6)
    ax.legend(loc="upper right", fontsize=8)
    cbar = fig.colorbar(im, ax=ax, pad=0.01)
    cbar.set_label("30-180d bandpass zos (m)")
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def run_panel_family(
    project_root: Path,
    bridge_root: Path,
    raw_root: Path,
    filter_root: Path,
    output_dir: Path,
    *,
    max_examples: int,
    python_exe: Path,
) -> None:
    cmd = [
        str(python_exe),
        "run_origin_eddy_pipeline.py",
        "plot-original-eddy-panels",
        "--results-root",
        str(bridge_root),
        "--shape-dir-name",
        "shape_classification_detect_only",
        "--raw-root",
        str(raw_root),
        "--filter-root",
        str(filter_root),
        "--output-dir",
        str(output_dir),
        "--preferred-shapes",
        "detect_only",
        "--min-layers",
        "8",
        "--max-examples",
        str(max_examples),
        "--right-panel-mode",
        "normal_horizontal_velocity",
        "--w-section-mode",
        "axis_curved",
        "--output-name-stem",
        "detect_only_curve_section_panel_family",
    ]
    subprocess.run(cmd, check=True, cwd=str(project_root))


def _load_velocity_layer(filter_root: Path, filter_template: str, date_text: str, depth_index: int) -> tuple[np.ndarray, np.ndarray]:
    year = int(date_text[:4])
    nc_path = filter_root / filter_template.format(year=year)
    ti = _time_index(nc_path, date_text)
    with Dataset(nc_path) as ds:
        u = np.asarray(ds.variables["uo_glor"][ti, depth_index, :, :], dtype="f8")
        v = np.asarray(ds.variables["vo_glor"][ti, depth_index, :, :], dtype="f8")
    return u, v


def python_matlab_diff(
    detection_dir: Path,
    filter_root: Path,
    filter_template: str,
    matlab_exe: Path,
    matlab_dir: Path,
    output_dir: Path,
    *,
    date_text: str,
    depth_index: int,
    max_centers: int,
    boundary_mode: str,
    matlab_batch_use_gpu: bool = False,
) -> tuple[Path, Path | None]:
    centers = _read_table(detection_dir / "centers_hua_style.parquet")
    part = centers[centers["date"].astype(str).eq(date_text) & centers["depth_index"].astype(int).eq(depth_index)].head(max_centers).copy()
    u, v = _load_velocity_layer(filter_root, filter_template, date_text, depth_index)
    day_layer = centers[centers["date"].astype(str).eq(date_text) & centers["depth_index"].astype(int).eq(depth_index)].copy()
    margin = int(PRODUCTION_PARAMS["max_radius_cells"]) + 2
    interior = day_layer[
        day_layer["speed_min_i_grid"].astype(int).between(margin, u.shape[1] - margin - 1)
        & day_layer["speed_min_j_grid"].astype(int).between(margin, u.shape[0] - margin - 1)
    ].copy()
    passed_interior = interior[interior["hua_pass"].astype(bool)].copy()
    source = passed_interior if len(passed_interior) >= max_centers else interior
    part = source.head(max_centers).copy()
    if part.empty:
        raise ValueError(f"No interior centers for {date_text} depth_index={depth_index} in {detection_dir}")
    params = DetectionParams(**{**PRODUCTION_PARAMS, "boundary_mode": boundary_mode})
    py_rows = []
    matlab_centers = []
    for idx, row in part.iterrows():
        ci_grid = int(row["speed_min_i_grid"])
        cj_grid = int(row["speed_min_j_grid"])
        ci = float(row["hua_center_i"]) if "hua_center_i" in row and pd.notna(row["hua_center_i"]) else float(ci_grid)
        cj = float(row["hua_center_j"]) if "hua_center_j" in row and pd.notna(row["hua_center_j"]) else float(cj_grid)
        matlab_centers.append([ci + 1.0, cj + 1.0])
        check = _hua_verify_radius(u, v, ci, cj, params)
        py_rows.append({"row_index": int(idx), "center_i": ci, "center_j": cj, "center_i_grid": ci_grid, "center_j_grid": cj_grid, **check})
    py = pd.DataFrame(py_rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    input_mat = output_dir / f"kernel_input_{boundary_mode}_{date_text.replace('-', '')}_z{depth_index}.mat"
    matlab_csv = output_dir / f"matlab_kernel_{boundary_mode}_{date_text.replace('-', '')}_z{depth_index}.csv"
    matlab_batch_csv = output_dir / f"matlab_batch_kernel_{boundary_mode}_{date_text.replace('-', '')}_z{depth_index}.csv"
    savemat(
        input_mat,
        {
            "u": u,
            "v": v,
            "centers": np.asarray(matlab_centers, dtype="f8"),
            "radii": np.arange(PRODUCTION_PARAMS["start_radius_cells"], PRODUCTION_PARAMS["max_radius_cells"] + 1, dtype="f8"),
        },
    )
    matlab_cmd = (
        f"addpath('{str(matlab_dir).replace("'", "''")}'); "
        f"load('{str(input_mat).replace("'", "''")}'); "
        "params=struct('start_radius_cells',2,'max_radius_cells',8,'speed_ratio_max',3,"
        "'angle_jump_max_deg',150,'tangent_tolerance_deg',24,'symmetry_tolerance_deg',120,"
        "'min_tangent_fraction',0.55,'min_reversal_fraction',0.55,'min_finite_fraction',0.75,"
        "'direction_exception_extra',2,'require_boundary_monotonic_rotation',true,"
        "'boundary_monotonic_exception_limit',0,'streamline_direction_exception_fraction',0.10,"
        "'streamline_step_cells',0.5,'streamline_max_steps',180,'streamline_start_angles',4,"
        "'streamline_closure_tolerance_cells',1.75,'streamline_min_winding_turns',0.75,"
        "'streamline_min_points',16); "
    )
    if boundary_mode == "velocity_streamline_contour":
        matlab_cmd += "rows=velocity_streamline_boundary_kernel(u,v,centers,radii,params); "
    else:
        matlab_cmd += "rows=hua_circle_batch_kernel(u,v,centers,params); "
    matlab_cmd += f"T=struct2table(rows); writetable(T,'{str(matlab_csv).replace("'", "''")}');"
    subprocess.run([str(matlab_exe), "-batch", matlab_cmd], check=True)
    ma = pd.read_csv(matlab_csv)
    batch_diff_out: Path | None = None
    mb: pd.DataFrame | None = None
    if boundary_mode == "velocity_streamline_contour":
        batch_cmd = (
            f"addpath('{str(matlab_dir).replace("'", "''")}'); "
            f"load('{str(input_mat).replace("'", "''")}'); "
            "params=struct('start_radius_cells',2,'max_radius_cells',8,'speed_ratio_max',3,"
            "'angle_jump_max_deg',150,'tangent_tolerance_deg',24,'symmetry_tolerance_deg',120,"
            "'min_tangent_fraction',0.55,'min_reversal_fraction',0.55,'min_finite_fraction',0.75,"
            "'direction_exception_extra',2,'require_boundary_monotonic_rotation',true,"
            "'boundary_monotonic_exception_limit',0,'streamline_direction_exception_fraction',0.10,"
            "'streamline_step_cells',0.5,'streamline_max_steps',180,'streamline_start_angles',4,"
            "'streamline_closure_tolerance_cells',1.75,'streamline_min_winding_turns',0.75,"
            "'streamline_min_points',16,"
            f"'use_gpu',{str(bool(matlab_batch_use_gpu)).lower()}); "
            "rows=streamline_gpu_batch_kernel(u,v,centers,radii,params); "
            f"T=struct2table(rows); writetable(T,'{str(matlab_batch_csv).replace("'", "''")}');"
        )
        subprocess.run([str(matlab_exe), "-batch", batch_cmd], check=True)
        mb = pd.read_csv(matlab_batch_csv)
    keys = [
        "circle_passed",
        "hua_pass",
        "accepted_radius_cells",
        "finite_fraction",
        "max_velocity_ratio",
        "max_angle_jump_deg",
        "tangent_pass_fraction",
        "opposite_reversal_fraction",
        "first_hard_failure_code",
        "first_hard_failure",
        "streamline_closed",
        "streamline_closure_error_cells",
        "streamline_winding_turns",
        "streamline_direction_exception_fraction",
    ]
    aliases = {
        "speed_ratio": "max_velocity_ratio",
        "tangent_fraction": "tangent_pass_fraction",
        "reversal_fraction": "opposite_reversal_fraction",
    }
    rows = []
    for ii in range(len(py)):
        for key in keys:
            py_key = key
            ma_key = next((k for k, vname in aliases.items() if vname == key and k in ma.columns), key)
            if py_key not in py.columns or ma_key not in ma.columns:
                continue
            py_val = py.iloc[ii][py_key]
            ma_val = ma.iloc[ii][ma_key]
            try:
                diff = abs(float(py_val) - float(ma_val))
            except Exception:
                diff = np.nan
            rows.append(
                {
                    "boundary_mode": boundary_mode,
                    "date": date_text,
                    "depth_index": depth_index,
                    "sample": ii,
                    "center_i": float(py.iloc[ii]["center_i"]),
                    "center_j": float(py.iloc[ii]["center_j"]),
                    "metric": key,
                    "python_value": py_val,
                    "matlab_value": ma_val,
                    "abs_diff": diff,
                    "match": bool(str(py_val) == str(ma_val) or (np.isfinite(diff) and diff <= 1e-6)),
                }
            )
    diff_df = pd.DataFrame(rows)
    out = output_dir / f"python_vs_matlab_diff_{boundary_mode}_{date_text.replace('-', '')}_z{depth_index}.csv"
    diff_df.to_csv(out, index=False, quoting=csv.QUOTE_MINIMAL)
    if mb is not None:
        batch_rows = []
        for ii in range(len(ma)):
            for key in keys:
                ref_key = next((k for k, vname in aliases.items() if vname == key and k in ma.columns), key)
                bat_key = next((k for k, vname in aliases.items() if vname == key and k in mb.columns), key)
                if ref_key not in ma.columns or bat_key not in mb.columns:
                    continue
                ref_val = ma.iloc[ii][ref_key]
                bat_val = mb.iloc[ii][bat_key]
                try:
                    diff = abs(float(ref_val) - float(bat_val))
                except Exception:
                    diff = np.nan
                batch_rows.append(
                    {
                        "boundary_mode": boundary_mode,
                        "date": date_text,
                        "depth_index": depth_index,
                        "sample": ii,
                        "center_i": float(py.iloc[ii]["center_i"]),
                        "center_j": float(py.iloc[ii]["center_j"]),
                        "metric": key,
                        "reference_kernel_value": ref_val,
                        "batch_kernel_value": bat_val,
                        "abs_diff": diff,
                        "match": bool(str(ref_val) == str(bat_val) or (np.isfinite(diff) and diff <= 1e-6)),
                        "batch_use_gpu": bool(matlab_batch_use_gpu),
                    }
                )
        batch_diff_df = pd.DataFrame(batch_rows)
        batch_diff_out = output_dir / f"matlab_reference_vs_batch_diff_{boundary_mode}_{date_text.replace('-', '')}_z{depth_index}.csv"
        batch_diff_df.to_csv(batch_diff_out, index=False, quoting=csv.QUOTE_MINIMAL)
    return out, batch_diff_out


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare ACC TEST overview, detect-only panel bridge, and MATLAB parity diagnostics.")
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--streamline-dir", type=Path, required=True)
    parser.add_argument("--filter-root", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--filter-template", default="global_phy_{year}_bandpass_30_180d.nc")
    parser.add_argument("--date", default="2018-01-01")
    parser.add_argument("--depth-index", type=int, default=0)
    parser.add_argument("--max-panel-examples", type=int, default=1)
    parser.add_argument("--max-diff-centers", type=int, default=6)
    parser.add_argument("--python-exe", type=Path, required=True)
    parser.add_argument("--matlab-exe", type=Path, default=Path(r"D:\Util\Ma\01_Matlab\bin\matlab.exe"))
    parser.add_argument("--matlab-batch-use-gpu", action="store_true")
    parser.add_argument("--diff-only", action="store_true", help="Only run kernel parity tables; skip matplotlib overview and panel-family plots.")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    matlab_dir = project_root / "matlab"
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {"created_at": datetime.now().isoformat(timespec="seconds"), "items": []}
    for label, detection_dir in [("baseline_circle_strict_original", args.baseline_dir), ("velocity_streamline_contour", args.streamline_dir)]:
        label_root = args.output_root / label
        bridge = label_root / "panel_bridge"
        panel_out = label_root / "panel_family_curve_section"
        overview = None
        bridge_meta = None
        if not bool(args.diff_only):
            overview = plot_surface_overview(detection_dir, args.filter_root, args.filter_template, label_root, date_text=args.date)
            bridge_meta = create_panel_bridge(detection_dir, bridge)
            run_panel_family(project_root, bridge, args.raw_root, args.filter_root, panel_out, max_examples=args.max_panel_examples, python_exe=args.python_exe)
        diff, batch_diff = python_matlab_diff(
            detection_dir,
            args.filter_root,
            args.filter_template,
            args.matlab_exe,
            matlab_dir,
            label_root / "python_vs_matlab",
            date_text=args.date,
            depth_index=args.depth_index,
            max_centers=args.max_diff_centers,
            boundary_mode="velocity_streamline_contour" if label == "velocity_streamline_contour" else "circle_strict_original",
            matlab_batch_use_gpu=bool(args.matlab_batch_use_gpu),
        )
        manifest["items"].append(
            {
                "label": label,
                "detection_dir": str(detection_dir),
                "surface_overview": str(overview) if overview else None,
                "panel_output_dir": str(panel_out) if bridge_meta else None,
                "bridge": bridge_meta,
                "python_vs_matlab_diff": str(diff),
                "matlab_reference_vs_batch_diff": str(batch_diff) if batch_diff else None,
            }
        )
    manifest_path = args.output_root / "acc_test_diagnostics_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
