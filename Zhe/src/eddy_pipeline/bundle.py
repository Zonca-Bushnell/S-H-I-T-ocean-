from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def _split_csv(value: str) -> list[str]:
    return [part.strip() for part in str(value).split(",") if part.strip()]


def _shape_dir_name(shapes: str, explicit_name: str) -> str:
    if explicit_name:
        return explicit_name
    parsed = _split_csv(shapes)
    if not parsed:
        raise ValueError("--shapes must contain at least one shape class")
    if parsed == ["coherent"]:
        return "result_coherent_only"
    return "result_" + "_".join(parsed)


def _run(command: list[str], dry_run: bool) -> None:
    print(" ".join(command), flush=True)
    if not dry_run:
        subprocess.run(command, check=True)


def _preflight_shape_tracks(results_root: Path, shape_dir_name: str, shape_root: Path, shapes: str) -> None:
    import pandas as pd

    requested = set(_split_csv(shapes))
    shape_tracks = results_root / shape_dir_name / "shape_tracks.parquet"
    if not shape_tracks.exists():
        raise FileNotFoundError(shape_tracks)

    table = pd.read_parquet(shape_tracks, columns=["shape_class"])
    counts = table["shape_class"].astype(str).value_counts().to_dict()
    missing = sorted(shape for shape in requested if int(counts.get(shape, 0)) == 0)
    shape_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "requested_shapes": sorted(requested),
        "shape_tracks_path": str(shape_tracks),
        "shape_track_counts": {str(key): int(value) for key, value in counts.items()},
        "missing_requested_shapes": missing,
        "status": "blocked_no_requested_shape_tracks" if missing else "ready",
    }
    (shape_root / "representative_bundle_preflight.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    lines = [
        "# Representative bundle preflight",
        "",
        f"Requested shapes: `{', '.join(sorted(requested))}`",
        f"Shape tracks table: `{shape_tracks}`",
        "",
        "Shape track counts:",
    ]
    for key, value in sorted(counts.items()):
        lines.append(f"- `{key}`: `{int(value)}`")
    if missing:
        lines.extend(
            [
                "",
                f"Result: blocked because these requested shapes have zero life30 tracks: `{', '.join(missing)}`.",
                "No fallback shape will be mixed in.",
            ]
        )
        (shape_root / "representative_bundle_preflight.md").write_text("\n".join(lines), encoding="utf-8")
        raise SystemExit(f"No life30 tracks for requested shapes: {', '.join(missing)}")
    lines.extend(["", "Result: ready."])
    (shape_root / "representative_bundle_preflight.md").write_text("\n".join(lines), encoding="utf-8")


def _filter_glob_pattern(template: str) -> str:
    return template.replace("{year}", "*") if "{year}" in template else template


def _preflight_filter_variables(filter_root: Path, filter_template: str, shape_root: Path) -> None:
    required = ("uo_glor", "vo_glor", "zos_glor", "thetao_glor")
    four_dimensional = ("uo_glor", "vo_glor", "thetao_glor")
    paths = sorted(filter_root.glob(_filter_glob_pattern(filter_template)))
    if not paths:
        raise FileNotFoundError(f"No filter files match {filter_root / _filter_glob_pattern(filter_template)}")

    try:
        from netCDF4 import Dataset
    except Exception as exc:  # pragma: no cover - environment specific
        raise RuntimeError("netCDF4 is required for filter preflight") from exc

    files: list[dict] = []
    missing_files: list[dict] = []
    dimension_mismatches: list[dict] = []
    for path in paths:
        with Dataset(path) as ds:
            variables = set(ds.variables.keys())
            missing = [name for name in required if name not in variables]
            dims = {name: tuple(ds.variables[name].dimensions) for name in required if name in variables}
            shape = {name: tuple(int(v) for v in ds.variables[name].shape) for name in required if name in variables}
            field_dims = {name: dims[name] for name in four_dimensional if name in dims}
            field_shape = {name: shape[name] for name in four_dimensional if name in shape}
            if len({value for value in field_dims.values()}) > 1 or len({value for value in field_shape.values()}) > 1:
                dimension_mismatches.append({"path": str(path), "dimensions": dims, "shape": shape})
            record = {
                "path": str(path),
                "variables_present": sorted(variables.intersection(required)),
                "missing_required_variables": missing,
                "shape": shape,
            }
            files.append(record)
            if missing:
                missing_files.append(record)

    manifest = {
        "filter_root": str(filter_root),
        "filter_template": filter_template,
        "required_variables": list(required),
        "n_files": len(files),
        "files": files,
        "missing_files": missing_files,
        "dimension_mismatches": dimension_mismatches,
        "status": "ready" if not missing_files and not dimension_mismatches else "blocked_filter_incomplete",
    }
    shape_root.mkdir(parents=True, exist_ok=True)
    (shape_root / "representative_bundle_filter_preflight.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# Representative bundle filter preflight",
        "",
        f"Filter root: `{filter_root}`",
        f"Template: `{filter_template}`",
        f"Files checked: `{len(files)}`",
        f"Required variables: `{', '.join(required)}`",
    ]
    if missing_files:
        lines.extend(["", "Files missing required variables:"])
        for record in missing_files[:20]:
            lines.append(f"- `{record['path']}` missing `{', '.join(record['missing_required_variables'])}`")
        if len(missing_files) > 20:
            lines.append(f"- ... `{len(missing_files) - 20}` more files omitted")
    if dimension_mismatches:
        lines.extend(["", f"Dimension/shape mismatches: `{len(dimension_mismatches)}`"])
    if missing_files or dimension_mismatches:
        lines.extend(
            [
                "",
                "Result: blocked. Generate the missing 30-180 day `thetao_glor` bandpass field before running stirring.",
            ]
        )
        (shape_root / "representative_bundle_filter_preflight.md").write_text("\n".join(lines), encoding="utf-8")
        raise SystemExit("Filter preflight failed: missing variables or inconsistent variable shapes.")

    lines.extend(["", "Result: ready."])
    (shape_root / "representative_bundle_filter_preflight.md").write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    output_root = Path(args.output_root)
    shape_root = output_root / _shape_dir_name(args.shapes, args.shape_output_name)
    radial_root = shape_root / "representative_vortex_radial_seed"
    azimuthal_root = shape_root / "representative_vortex_me_liutex"
    stirring_root = shape_root / "aggregate_product_stirring"

    if args.dry_run:
        print("[dry-run] skip representative bundle preflight checks", flush=True)
    elif not args.skip_preflight:
        _preflight_shape_tracks(Path(args.results_root), str(args.shape_dir_name), shape_root, str(args.shapes))
    if not args.dry_run and not args.skip_preflight and not args.skip_filter_preflight:
        _preflight_filter_variables(Path(args.filter_root), str(args.filter_template), shape_root)

    python = args.python
    common_radial = [
        python,
        "-m",
        "src.eddy_pipeline.radial_representative",
        "--config",
        str(args.config),
        "--results-root",
        str(args.results_root),
        "--output-dir",
        str(radial_root),
        "--shape-dir-name",
        str(args.shape_dir_name),
        "--shapes",
        str(args.shapes),
        "--polarities",
        str(args.polarities),
        "--axis-alignment",
        "global_ls_alpha",
        "--field-cache-mode",
        str(args.field_cache_mode),
        "--workers",
        str(args.radial_workers),
        "--chunk-days",
        str(args.chunk_days),
    ]
    if args.resume:
        common_radial.append("--resume")

    orientation_modes = {
        "turned": [("global_alpha", azimuthal_root)],
        "unturned": [("unturned", shape_root / "representative_vortex_me_liutex_unturned")],
        "both": [("global_alpha", azimuthal_root), ("unturned", shape_root / "representative_vortex_me_liutex_unturned")],
    }
    azimuthal_commands = []
    for orientation_mode, orientation_output in orientation_modes[str(args.orientation)]:
        azimuthal_commands.append([
        python,
        "-m",
        "src.eddy_pipeline.representative",
        "--rv-root",
        str(radial_root),
        "--filter-root",
        str(args.filter_root),
        "--filter-template",
        str(args.filter_template),
        "--output-dir",
        str(orientation_output),
        "--shapes",
        str(args.shapes),
        "--orientation-mode",
        orientation_mode,
        "--max-depth-m",
        str(args.max_depth_m),
        "--tau-grid-step",
        str(args.tau_grid_step),
        "--kernel-bandwidth",
        str(args.kernel_bandwidth),
        "--radial-bins",
        str(args.radial_bins),
        "--azimuth-bins",
        str(args.azimuth_bins),
        "--rmax",
        str(args.rmax),
    ])

    stirring = [
        python,
        "-m",
        "src.post.transport",
        "--rv-root",
        str(radial_root),
        "--filter-root",
        str(args.filter_root),
        "--filter-template",
        str(args.filter_template),
        "--output-dir",
        str(stirring_root),
        "--shapes",
        str(args.shapes),
        "--workers",
        str(args.stirring_workers),
        "--chunk-days",
        str(args.chunk_days),
        "--azimuth-bins",
        str(args.azimuth_bins),
        "--rmax",
        str(args.rmax),
        "--kernel-bandwidth",
        str(args.kernel_bandwidth),
    ]
    if args.resume:
        stirring.append("--resume")
    if args.force_stirring:
        stirring.append("--force")

    if not args.skip_radial:
        _run(common_radial, args.dry_run)
    if not args.skip_azimuthal:
        for command in azimuthal_commands:
            _run(command, args.dry_run)
    if not args.skip_stirring:
        _run(stirring, args.dry_run)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run radial seed, ME_LIUTEX azimuthal structure, and aggregate-product stirring for one shape set.",
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--filter-root", required=True)
    parser.add_argument("--filter-template", default="global_phy_{year}_bandpass_30_180d.nc")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--shape-dir-name", default="shape_classification_1993_2022_hua_b3_start2_life30")
    parser.add_argument("--shape-output-name", default="", help="Optional result directory name under --output-root.")
    parser.add_argument("--shapes", default="coherent")
    parser.add_argument("--polarities", default="cyclonic,anticyclonic")
    parser.add_argument("--field-cache-mode", choices=["day", "year"], default="day")
    parser.add_argument("--radial-workers", type=int, default=1)
    parser.add_argument("--stirring-workers", type=int, default=1)
    parser.add_argument("--chunk-days", type=int, default=7)
    parser.add_argument("--max-depth-m", type=float, default=2000.0)
    parser.add_argument("--tau-grid-step", type=float, default=0.05)
    parser.add_argument("--kernel-bandwidth", type=float, default=0.075)
    parser.add_argument("--radial-bins", type=int, default=40)
    parser.add_argument("--azimuth-bins", type=int, default=72)
    parser.add_argument("--rmax", type=float, default=2.5)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--orientation", choices=["turned", "unturned", "both"], default="turned")
    parser.add_argument("--force-stirring", action="store_true")
    parser.add_argument("--skip-radial", action="store_true")
    parser.add_argument("--skip-azimuthal", action="store_true")
    parser.add_argument("--skip-stirring", action="store_true")
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument("--skip-filter-preflight", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--python", default=sys.executable)
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
