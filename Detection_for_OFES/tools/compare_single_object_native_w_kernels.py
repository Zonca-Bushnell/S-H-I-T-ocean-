"""Compare direct pointwise and Cressman native-W fields for selected eddies."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from ..ofes_io import ctl_path, expected_dta_bytes, open_dta_memmap, parse_ctl, require_daily_file
from ..run_ofes_rebuild_w import classify_native_w_multipole, load_font
from .run_ofes_isopycnal_composite import load_table_objects, run_group, write_group_outputs


DEFAULT_SOURCE = Path(
    r"E:\DATA\01_Eddy_correspond\02_OFES\Fromthebeginning\05_TEMP"
    r"\nh_open_ocean_cyclonic_dipole_native_w_pointwise_unrotated_eta_19910101"
    r"\nh_open_ocean_dipole_surface_objects.csv"
)
DEFAULT_OUTPUT = Path(
    r"E:\DATA\01_Eddy_correspond\02_OFES\Fromthebeginning\05_TEMP"
    r"\jan01_three_single_eddies_pointwise_vs_cressman"
)
DEFAULT_DATA_ROOT = Path(r"F:\OFES\external_OFES2")
DEFAULT_IDS = "19910101_04613,19910101_11330,19910101_14217"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--object-table", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--day", default="1991-01-01")
    parser.add_argument("--object-ids", default=DEFAULT_IDS)
    parser.add_argument("--max-depth-layers", type=int, default=105)
    parser.add_argument("--grid-n", type=int, default=101)
    parser.add_argument("--extent-r", type=float, default=2.0)
    parser.add_argument("--cressman-radius-r", type=float, default=1.0)
    return parser.parse_args()


def composite_args(args: argparse.Namespace, method: str) -> SimpleNamespace:
    return SimpleNamespace(
        max_depth_layers=args.max_depth_layers,
        grid_n=args.grid_n,
        extent_r=args.extent_r,
        day=args.day,
        cressman_radius_r=args.cressman_radius_r,
        cressman_min_objects=1,
        composite_method=method,
        max_objects_per_group=0,
        workers=1,
        max_geometry_depth_m=2000.0,
        background_ring_min_r=1.5,
        background_ring_max_r=2.0,
        min_bracket_stratification=1.0e-4,
    )


def multipole_args() -> SimpleNamespace:
    return SimpleNamespace(
        multipole_depth_min_m=300.0,
        multipole_depth_max_m=500.0,
        multipole_radius_inner_r=0.0,
        multipole_radius_outer_r=1.0,
        multipole_azimuth_count=60,
        multipole_min_valid_azimuth_fraction=0.80,
        multipole_min_sector_valid_fraction=0.35,
        multipole_boundary_max_nan_fraction=0.35,
        multipole_min_amp_1e6_m_s=0.5,
        multipole_min_snr=2.0,
        multipole_min_harmonic_dominance=1.1,
    )


def field_metrics(payload: dict[str, object]) -> dict[str, object]:
    depth = np.asarray(payload["depth_m"], dtype="f8")
    x = np.asarray(payload["x_over_r"], dtype="f8")
    y = np.asarray(payload["y_over_r"], dtype="f8")
    w = np.asarray(payload["native_w_m_s"], dtype="f8")
    selected_depth = (depth >= 300.0) & (depth <= 500.0)
    mean_w = np.nanmean(w[selected_depth], axis=0)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    core = np.hypot(xx, yy) <= 1.0
    values = mean_w[core & np.isfinite(mean_w)] * 1.0e6
    dy = np.diff(mean_w, axis=0)
    dx = np.diff(mean_w, axis=1)
    roughness = np.sqrt(np.nanmean(dx * dx) + np.nanmean(dy * dy)) * 1.0e6
    classification = classify_native_w_multipole({
        "ofes_w_native_raw_m_s": w.astype("f4"),
        "depth_m": depth.astype("f4"),
        "x_over_r": x.astype("f4"),
        "y_over_r": y.astype("f4"),
    }, multipole_args())
    return {
        "w_300_500_core_min_1e6_m_s": float(np.nanmin(values)),
        "w_300_500_core_max_1e6_m_s": float(np.nanmax(values)),
        "w_300_500_core_peak_to_peak_1e6_m_s": float(np.nanmax(values) - np.nanmin(values)),
        "w_300_500_core_rms_1e6_m_s": float(np.sqrt(np.nanmean(values * values))),
        "w_300_500_grid_roughness_1e6_m_s": float(roughness),
        **{key: value for key, value in classification.items() if not isinstance(value, np.ndarray)},
    }


def pair_image(left_path: Path, right_path: Path, output: Path, title: str) -> None:
    left = Image.open(left_path).convert("RGB")
    right = Image.open(right_path).convert("RGB")
    height = max(left.height, right.height)
    canvas = Image.new("RGB", (left.width + right.width, height + 72), "white")
    canvas.paste(left, (0, 72))
    canvas.paste(right, (left.width, 72))
    draw = ImageDraw.Draw(canvas)
    draw.text((24, 18), title, fill=(20, 24, 32), font=load_font(28))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
    canvas.save(output.with_suffix(".pdf"), "PDF", resolution=180.0)


def main() -> None:
    args = parse_args()
    requested = [value.strip() for value in args.object_ids.split(",") if value.strip()]
    groups = load_table_objects(args.object_table, args.day)
    by_id = {obj.hua_object_id: obj for values in groups.values() for obj in values}
    missing = [object_id for object_id in requested if object_id not in by_id]
    if missing:
        raise ValueError(f"Object IDs absent from source table: {', '.join(missing)}")

    from datetime import datetime
    day = datetime.strptime(args.day, "%Y-%m-%d").date()
    metas = {name: parse_ctl(ctl_path(args.data_root, name)) for name in ("prho", "w")}
    raw = {
        name: open_dta_memmap(require_daily_file(args.data_root, name, day, expected_dta_bytes(meta)), meta)
        for name, meta in metas.items()
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for object_id in requested:
        obj = by_id[object_id]
        for method in ("pointwise_mean", "cressman"):
            cargs = composite_args(args, method)
            payload = run_group(raw, metas, [obj], cargs)
            if payload is None:
                raise RuntimeError(f"No sampled output for {object_id}/{method}")
            payload.update({
                "region": object_id,
                "orientation": "single_object_unrotated",
                "multipole_class": "single_object",
                "source_object_id": object_id,
            })
            record = write_group_outputs(
                args.output_root, method, object_id, payload, render_geometry_sections=False,
            )
            rows.append({
                "hua_object_id": object_id,
                "method": method,
                "center_lon": obj.center_lon,
                "center_lat": obj.center_lat,
                "radius_km": obj.radius_km,
                **field_metrics(payload),
                **record,
            })
        pair_image(
            args.output_root / "pointwise_mean" / object_id / "figures" / "native_w_focus.png",
            args.output_root / "cressman" / object_id / "figures" / "native_w_focus.png",
            args.output_root / "comparisons" / f"{object_id}_native_w_focus_pointwise_vs_cressman.png",
            f"{object_id} | pointwise mean (left) vs Cressman 1R (right)",
        )
        pair_image(
            args.output_root / "pointwise_mean" / object_id / "figures" / "native_w_cross_section.png",
            args.output_root / "cressman" / object_id / "figures" / "native_w_cross_section.png",
            args.output_root / "comparisons" / f"{object_id}_native_w_cross_section_pointwise_vs_cressman.png",
            f"{object_id} | pointwise mean (left) vs Cressman 1R (right)",
        )

    summary = pd.DataFrame(rows)
    summary.to_csv(args.output_root / "single_object_kernel_comparison.csv", index=False, encoding="utf-8-sig")
    manifest = {
        "day": args.day,
        "source": "OFES native raw W; no rebuild-W and no inter-object averaging",
        "object_ids": requested,
        "methods": ["pointwise_mean", "cressman"],
        "cressman_radius_r": args.cressman_radius_r,
        "grid": f"[-{args.extent_r}R,{args.extent_r}R], {args.grid_n} points per axis",
        "summary": str(args.output_root / "single_object_kernel_comparison.csv"),
    }
    (args.output_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
