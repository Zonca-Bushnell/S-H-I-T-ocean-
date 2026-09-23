"""Run Jan1 vertical Hua continuation without ratio/tangent hard rejection."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from Detection_for_OFES.object_identity import attach_source_identity
from Origin_eddy_detection.src.post.original_eddy_panels import _candidate_objects


DAY = "1991-01-01"
DAY_KEY = "19910101"
ROOT = Path(r"E:\DATA\01_Eddy_correspond\02_OFES\Fromthebeginning\04_Vertical")
STRICT_ROOT = ROOT / "500km_highpass_no_tilecap_streamline_gate_removed_geometry_qc_19910101"
OUTPUT_ROOT = ROOT / "500km_highpass_no_tilecap_streamline_gate_removed_geometry_qc_19910101_relaxed_ratio_tangent"
PANEL_OUTPUT_STEM = "relaxed_ratio_tangent_curve_section_family"
CURRENT_BRANCH_LABEL = "relaxed"
RIGHT_PANEL_MODE = "normal_horizontal_velocity"
QC_TABLE = Path(
    r"E:\DATA\01_Eddy_correspond\02_OFES\Fromthebeginning\03_QC"
    r"\500km_highpass_no_tilecap_streamline_gate_removed_geometry_qc_19910101"
    r"\daily_runs\19910101\centers_hua_style.csv"
)
FILTER_ROOT = STRICT_ROOT / "full_depth_velocity_highpass_500km"
REGIONS = {
    "kuroshio_extension": (140.0, 180.0, 28.0, 40.0),
    "south_pacific_stcc": (165.0, 230.0, -29.0, -21.0),
    "taiwan_hawaii_corridor": (122.0, 203.0, 18.0, 27.0),
}


def run_logged(command: list[str], stdout_path: Path, stderr_path: Path) -> None:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        result = subprocess.run(command, cwd=Path.cwd(), stdout=stdout, stderr=stderr, text=True, check=False)
    if result.returncode:
        raise RuntimeError(f"Command failed ({result.returncode}); see {stderr_path}")


def region_mask(frame: pd.DataFrame, box: tuple[float, float, float, float]) -> pd.Series:
    west, east, south, north = box
    lon = pd.to_numeric(frame["surface_lon"], errors="coerce") % 360.0
    lat = pd.to_numeric(frame["surface_lat"], errors="coerce")
    return lon.between(west, east) & lat.between(south, north)


def create_panel_bridge(detection_dir: Path, bridge_root: Path) -> None:
    """Create the minimum detect-only catalog required by original_eddy_panels."""
    structures_parquet = detection_dir / "structures_hua_style.parquet"
    try:
        structures = pd.read_parquet(structures_parquet) if structures_parquet.exists() else pd.read_csv(detection_dir / "structures_hua_style.csv")
    except (ImportError, OSError):
        structures = pd.read_csv(detection_dir / "structures_hua_style.csv")
    structures = attach_source_identity(structures)
    centers = pd.DataFrame({
        "date": structures["date"].astype(str),
        "eddy3d_object_id": structures["stable_object_id"].astype("int64"),
        "track3d_id": structures["stable_object_id"].astype("int64"),
        "depth_index": structures["depth_index"].astype(int),
        "depth_m": structures["depth_m"].astype(float),
        "longitude": structures["center_lon"].astype(float),
        "latitude": structures["center_lat"].astype(float),
        "radius_m": structures["radius_km"].astype(float) * 1000.0,
        "polarity": structures["polarity"].astype(str),
        "source_hua_object_id": structures["source_hua_object_id"],
    })
    centers = centers[np.isfinite(centers["radius_m"]) & centers["radius_m"].gt(0)].copy()
    catalog = bridge_root / "catalog"
    shape = bridge_root / "shape_classification_detect_only"
    catalog.mkdir(parents=True, exist_ok=True)
    shape.mkdir(parents=True, exist_ok=True)
    centers.to_csv(catalog / "layer_centers_completed.csv", index=False)
    shape_tracks = centers.groupby("track3d_id", as_index=False).agg(
        n_object_days=("date", "nunique"), n_layers=("depth_index", "count"), source_hua_object_id=("source_hua_object_id", "first"),
    )
    shape_tracks.insert(1, "shape_class", "detect_only")
    shape_tracks.to_csv(shape / "shape_tracks.csv", index=False)


def summarize_comparison() -> None:
    strict = pd.read_csv(STRICT_ROOT / "vertical_object_summary.csv")
    relaxed = pd.read_csv(OUTPUT_ROOT / "vertical_object_summary.csv")
    merged = strict.merge(relaxed, on="hua_object_id", suffixes=("_strict", "_relaxed"), validate="one_to_one")
    merged["max_depth_gain_m"] = merged["max_depth_m_relaxed"] - merged["max_depth_m_strict"]
    merged["layer_gain"] = merged["pass_layers_relaxed"] - merged["pass_layers_strict"]
    merged.to_csv(OUTPUT_ROOT / "strict_vs_relaxed_object_comparison.csv", index=False)
    rows: list[dict[str, object]] = []
    for name, box in {"global": (0.0, 360.0, -90.0, 90.0), **REGIONS}.items():
        part = merged if name == "global" else merged.loc[region_mask(merged.rename(columns={"surface_lon_strict": "surface_lon", "surface_lat_strict": "surface_lat"}), box)]
        rows.append({
            "region": name,
            "objects": int(len(part)),
            "strict_extend_below_surface": int((part["pass_layers_strict"] > 1).sum()),
            "relaxed_extend_below_surface": int((part["pass_layers_relaxed"] > 1).sum()),
            "median_layer_gain": float(part["layer_gain"].median()) if len(part) else np.nan,
            "median_depth_gain_m": float(part["max_depth_gain_m"].median()) if len(part) else np.nan,
            "max_depth_gain_m": float(part["max_depth_gain_m"].max()) if len(part) else np.nan,
        })
    pd.DataFrame(rows).to_csv(OUTPUT_ROOT / "strict_vs_relaxed_region_comparison.csv", index=False)


def make_family_panels(python: str) -> None:
    detection_dir = OUTPUT_ROOT / "raw_detection" / "daily_runs" / DAY_KEY
    bridge = OUTPUT_ROOT / "curve_section_panel_bridge"
    create_panel_bridge(detection_dir, bridge)
    centers = pd.read_csv(bridge / "catalog" / "layer_centers_completed.csv")
    shapes = pd.read_csv(bridge / "shape_classification_detect_only" / "shape_tracks.csv")
    _, candidates = _candidate_objects(centers, shapes, {"detect_only"}, min_layers=2, year_limit=1991)
    surface_lookup = pd.read_csv(OUTPUT_ROOT / "vertical_object_summary.csv").set_index("hua_object_id")
    rng = np.random.default_rng(19910101)
    selected_rows: list[dict[str, object]] = []
    for region, box in REGIONS.items():
        available = []
        for _, row in candidates.iterrows():
            source_id = str(shapes.loc[shapes["track3d_id"].eq(row["track3d_id"]), "source_hua_object_id"].iloc[0])
            if source_id not in surface_lookup.index:
                continue
            surface = surface_lookup.loc[source_id]
            if bool(region_mask(pd.DataFrame([surface]), box).iloc[0]):
                payload = row.to_dict()
                payload["source_hua_object_id"] = source_id
                payload["selection_region"] = region
                payload["strict_max_depth_m"] = float(pd.read_csv(STRICT_ROOT / "vertical_object_summary.csv").set_index("hua_object_id").loc[source_id, "max_depth_m"])
                payload[f"{CURRENT_BRANCH_LABEL}_max_depth_m"] = float(surface["max_depth_m"])
                payload["strict_pass_layers"] = int(pd.read_csv(STRICT_ROOT / "vertical_object_summary.csv").set_index("hua_object_id").loc[source_id, "pass_layers"])
                payload[f"{CURRENT_BRANCH_LABEL}_pass_layers"] = int(surface["pass_layers"])
                available.append(payload)
        if available:
            picks = rng.choice(len(available), size=min(4, len(available)), replace=False)
            for order, index in enumerate(np.sort(picks), start=1):
                item = available[int(index)]
                item["output_dir"] = str(
                    OUTPUT_ROOT / "curve_section_family" / f"{region}_sample_{order:02d}_{item['source_hua_object_id']}"
                )
                selected_rows.append(item)
    selected = pd.DataFrame(selected_rows)
    panel_root = OUTPUT_ROOT / "curve_section_family"
    panel_root.mkdir(parents=True, exist_ok=True)
    metadata = panel_root / "selected_objects_metadata.csv"
    selected.to_csv(metadata, index=False)
    (panel_root / "selection_manifest.json").write_text(
        json.dumps({"day": DAY, "random_seed": 19910101, "selection": selected_rows}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    link_root = OUTPUT_ROOT / "panel_filter_input"
    link_root.mkdir(exist_ok=True)
    target = FILTER_ROOT / f"global_phy_{DAY_KEY}.nc"
    alias = link_root / "global_phy_1991.nc"
    if alias.exists() or alias.is_symlink():
        alias.unlink()
    os.link(target, alias)
    command = [
        r"D:\Util\lever\02_miniforge\condabin\mamba.bat", "run", "-n", "OFES_detection", "python",
        "-m", "Origin_eddy_detection.src.post.original_eddy_panels",
        "--results-root", str(bridge), "--shape-dir-name", "shape_classification_detect_only",
        "--raw-root", str(OUTPUT_ROOT / "unused_raw_for_velocity_sections"), "--filter-root", str(link_root),
        "--output-dir", str(panel_root), "--selected-metadata", str(metadata), "--min-layers", "2",
        "--right-panel-mode", RIGHT_PANEL_MODE, "--w-section-mode", "axis_curved",
        "--output-name-stem", PANEL_OUTPUT_STEM,
    ]
    run_logged(command, OUTPUT_ROOT / "logs" / "curve_section_panels.stdout.log", OUTPUT_ROOT / "logs" / "curve_section_panels.stderr.log")


def main() -> None:
    if not QC_TABLE.exists() or not (FILTER_ROOT / f"global_phy_{DAY_KEY}.nc").exists():
        raise FileNotFoundError("Missing QC input or existing full-depth 500 km filter input")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    python = sys.executable
    vertical = [
        python, "-m", "Detection_for_OFES.tools.extend_final_surface_vertical", "--surface-table", str(QC_TABLE),
        "--filter-root", str(FILTER_ROOT), "--output-root", str(OUTPUT_ROOT), "--day", DAY,
        "--max-depth-layers", "105", "--deep-search-cells", "6", "--start-radius-cells", "2", "--max-radius-cells", "12", "--deep-hua-mode", "full",
    ]
    run_logged(vertical, OUTPUT_ROOT / "logs" / "vertical_extension.stdout.log", OUTPUT_ROOT / "logs" / "vertical_extension.stderr.log")
    summary = [python, "-m", "Detection_for_OFES.tools.summarize_qc_vertical_extension", "--vertical-root", str(OUTPUT_ROOT), "--day", DAY]
    run_logged(summary, OUTPUT_ROOT / "logs" / "summary_plots.stdout.log", OUTPUT_ROOT / "logs" / "summary_plots.stderr.log")
    summarize_comparison()
    make_family_panels(python)
    print(json.dumps({"status": "complete", "output_root": str(OUTPUT_ROOT)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
