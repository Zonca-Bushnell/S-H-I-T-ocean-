from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from .axis_streamfunction import (
    DEFAULT_AXIS_DIR,
    DEFAULT_CATALOG,
    DEFAULT_INPUT_DAILY,
    grid_spacing_m,
    parse_csv_list,
    read_daily_uv,
    relative_vorticity,
    streamfunction_from_zeta,
)
from .ep_flux import (
    MLRW_CATEGORY_CODE,
    MLRW_CATEGORY_ORDER,
    add_wave_activity,
    build_nonlinearity_tables,
    compute_nondim_terms,
)
from .lifecycle_common import (
    DEFAULT_NEW_VORTICITY_ROOT,
    DEFAULT_POLARITIES,
    DEFAULT_SHAPE_BY_SHAPE_DIR,
    DEFAULT_SHAPES,
    apply_lifecycle_limits,
    load_center_lines,
    load_lifecycle_objects,
    representative_radii,
)
from .field_sampling import (
    DEFAULT_CLIMATOLOGY,
    DEFAULT_CLIMATOLOGY_NC,
    OMEGA,
    divergence,
    load_n2,
    make_polar_grid,
    read_climatology_uv,
    sanitize_ocean_field,
    sample_object_fields,
)


DEFAULT_OUTPUT = DEFAULT_NEW_VORTICITY_ROOT / "continuous_lifecycle_representative_40r_72theta"
DEFAULT_SUBDIRS = (
    "object_cache",
    "streamfunction_templates",
    "ep_flux_terms",
    "wave_action_total_flux",
    "mlrw_applicability",
    "core_environment_sensitivity",
    "tilt_evolution_coupling",
    "velocity_stack_3d",
    "figures",
)


def parse_tau_grid(value: str | None, step: float) -> np.ndarray:
    if value:
        grid = np.asarray([float(item.strip()) for item in value.split(",") if item.strip()], dtype="f8")
    else:
        if step <= 0:
            raise ValueError("--tau-grid-step must be positive.")
        count = int(round(1.0 / step))
        grid = np.linspace(0.0, 1.0, count + 1)
    if grid.size == 0:
        raise ValueError("Tau grid is empty.")
    grid = np.unique(np.clip(grid, 0.0, 1.0))
    return grid


def ensure_output_tree(output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name in DEFAULT_SUBDIRS:
        path = output_dir / name
        path.mkdir(parents=True, exist_ok=True)
        paths[name] = path
    return paths


def select_lifecycle_objects(args: argparse.Namespace) -> pd.DataFrame:
    shapes = parse_csv_list(args.shapes, DEFAULT_SHAPES)
    polarities = parse_csv_list(args.polarities, DEFAULT_POLARITIES)
    objects = load_lifecycle_objects(
        axis_dir=Path(args.axis_dir),
        catalog_dir=Path(args.catalog_dir),
        shape_dir=Path(args.shape_dir),
        shapes=shapes,
        polarities=polarities,
    )
    return apply_lifecycle_limits(objects, int(args.max_days), int(args.max_objects_per_polarity), int(args.random_seed))


def build_manifest(objects: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if objects.empty:
        return pd.DataFrame(), pd.DataFrame()
    manifest = objects.copy()
    counts = (
        objects.groupby(["polarity"], as_index=False)
        .agg(
            n_objects=("eddy3d_object_id", "nunique"),
            n_tracks=("track3d_id", "nunique"),
            n_dates=("date", "nunique"),
            life_phase_min=("life_phase", "min"),
            life_phase_max=("life_phase", "max"),
        )
        .sort_values("polarity")
    )
    return manifest, counts


def make_weighted_accum(terms: dict[str, np.ndarray]) -> dict:
    out = {name: np.zeros_like(value, dtype="f8") for name, value in terms.items() if name != "valid"}
    out["count"] = np.zeros_like(terms["valid"], dtype="f8")
    out["objects"] = set()
    out["tracks"] = set()
    out["dates"] = set()
    return out


def add_weighted_terms(
    accum: dict,
    key: tuple[str, str, int],
    terms: dict[str, np.ndarray],
    weight: float,
    object_id: int,
    track_id: int,
    date: str,
) -> None:
    if weight <= 0 or not np.isfinite(weight):
        return
    if key not in accum:
        accum[key] = make_weighted_accum(terms)
    valid = np.isfinite(terms["valid"]) & (terms["valid"] > 0)
    for name, value in terms.items():
        if name == "valid":
            continue
        accum[key][name] += np.nan_to_num(value, nan=0.0) * valid * weight
    accum[key]["count"] += valid.astype("f8") * weight
    accum[key]["objects"].add(int(object_id))
    accum[key]["tracks"].add(int(track_id))
    accum[key]["dates"].add(str(date))


def finalize_weighted(accum: dict) -> dict:
    final = {}
    for key, item in accum.items():
        count = item["count"]
        out = {"count": count, "objects": item["objects"], "tracks": item["tracks"], "dates": item["dates"]}
        for name, value in item.items():
            if name in {"count", "objects", "tracks", "dates"}:
                continue
            out[name] = np.divide(value, count, out=np.full_like(value, np.nan), where=count > 0)
        final[key] = out
    return final


def rows_from_continuous_final(
    final: dict,
    radial: np.ndarray,
    depth: np.ndarray,
    tau_grid: np.ndarray,
    radii: dict[str, float],
) -> pd.DataFrame:
    rows = []
    mechanism_columns = (
        "axis_slope_x",
        "axis_slope_y",
        "axis_slope_mag",
        "N2",
        "stratification_factor",
        "raw_tilt_flux",
        "raw_tilt_flux_x",
        "raw_tilt_flux_y",
        "F_z_tilt_correction_const_N2",
        "tilt_projection_mean",
        "tilt_projection_rms",
        "un_prime_rms",
        "un_tilt_projection_cov",
        "psi_mean",
        "psi_variance",
    )
    for (mode, polarity, tau_index), item in sorted(final.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2])):
        radius = radii.get(polarity)
        if radius is None:
            continue
        divf = divergence(item["F_n"], item["F_z_tilted"], radial, depth, radius)
        divf_ordinary = divergence(item["F_n"], item["F_z_ordinary"], radial, depth, radius)
        tau_center = float(tau_grid[int(tau_index)])
        phase_name = f"tau_{int(tau_index):03d}"
        for k, depth_m in enumerate(depth):
            for j, r in enumerate(radial):
                fzt = item["F_z_tilted"][k, j]
                fz0 = item["F_z_ordinary"][k, j]
                fzc = item["F_z_tilt_correction"][k, j]
                row = {
                    "axis_mode": mode,
                    "shape_class": "all_shapes",
                    "polarity": polarity,
                    "phase_index": int(tau_index),
                    "phase_name": phase_name,
                    "tau_center": tau_center,
                    "depth_index": int(k),
                    "depth_m": float(depth_m),
                    "r_over_R": float(r),
                    "F_n": float(item["F_n"][k, j]),
                    "F_z_tilted": float(fzt),
                    "F_z_ordinary": float(fz0),
                    "F_z_tilt_correction": float(fzc),
                    "tilt_fraction": float(fzc / fzt) if np.isfinite(fzc) and np.isfinite(fzt) and abs(fzt) > 1e-14 else np.nan,
                    "ordinary_fraction": float(fz0 / fzt) if np.isfinite(fz0) and np.isfinite(fzt) and abs(fzt) > 1e-14 else np.nan,
                    "tilt_to_ordinary_ratio": float(fzc / fz0) if np.isfinite(fzc) and np.isfinite(fz0) and abs(fz0) > 1e-14 else np.nan,
                    "F_z_decomposition_residual": float(fzt - fz0 - fzc),
                    "divF": float(divf[k, j]),
                    "divF_ordinary": float(divf_ordinary[k, j]),
                    "pv_flux": float(item["pv_flux"][k, j]),
                    "q_mean": float(item["q_mean"][k, j]),
                    "q_prime_variance": float(item["q_prime_variance"][k, j]),
                    "Unbar": float(item["Unbar"][k, j]) if "Unbar" in item else np.nan,
                    "Ubar": float(item["Ubar"][k, j]),
                    "count": float(item["count"][k, j]),
                    "n_objects": len(item["objects"]),
                    "n_tracks": len(item["tracks"]),
                    "n_dates": len(item["dates"]),
                }
                for name in mechanism_columns:
                    if name in item:
                        row[name] = float(item[name][k, j])
                rows.append(row)
    return pd.DataFrame.from_records(rows)


def build_continuous_mlrw_profiles(nonlinearity: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if nonlinearity.empty:
        return pd.DataFrame(), pd.DataFrame()
    profiles = nonlinearity.copy()
    qn_valid = profiles["Q_n_valid"].to_numpy(dtype=bool)
    nlq = profiles["NL_q"].to_numpy(dtype="f8")
    residual = np.abs(profiles["wave_activity_residual_J_nd"].to_numpy(dtype="f8"))

    category = np.full(len(profiles), "weak_gradient_invalid", dtype=object)
    strict = qn_valid & np.isfinite(nlq) & np.isfinite(residual) & (nlq <= 1.0) & (residual <= 1.0)
    marginal = qn_valid & ~strict & np.isfinite(nlq) & np.isfinite(residual) & (nlq <= 10.0) & (residual <= 2.0)
    finite_amp = qn_valid & np.isfinite(nlq) & (nlq > 10.0)
    budget_bad = qn_valid & ~strict & ~marginal & ~finite_amp
    category[strict] = "strict_mlrw"
    category[marginal] = "marginal_mlrw"
    category[finite_amp] = "finite_amplitude_invalid"
    category[budget_bad] = "budget_nonclosure_invalid"

    nlq_factor = np.where(np.isfinite(nlq) & (nlq > 0), np.minimum(1.0, 1.0 / nlq), 0.0)
    residual_factor = np.where(np.isfinite(residual) & (residual > 0), np.minimum(1.0, 1.0 / residual), 1.0)
    profiles["mlrw_category"] = category
    profiles["mlrw_category_code"] = [MLRW_CATEGORY_CODE[str(name)] for name in category]
    profiles["mlrw_score"] = np.where(qn_valid, nlq_factor * residual_factor, 0.0)
    profiles["abs_wave_activity_residual_J_nd"] = residual

    metric_rows = []
    for (polarity, tau_index, phase_name), part in profiles.groupby(["polarity", "phase_index", "phase_name"], sort=True):
        row: dict[str, float | int | str] = {
            "polarity": str(polarity),
            "tau_index": int(tau_index),
            "phase_name": str(phase_name),
            "tau_center": float(part["tau_center"].iloc[0]),
            "n_bins": int(len(part)),
            "median_mlrw_score": float(np.nanmedian(part["mlrw_score"].to_numpy(dtype="f8"))),
            "Q_n_valid_fraction": float(np.mean(part["Q_n_valid"].to_numpy(dtype=bool))),
        }
        for name in MLRW_CATEGORY_ORDER:
            row[f"fraction_{name}"] = float(np.mean(part["mlrw_category"].to_numpy(dtype=object) == name))
        row["fraction_strict_plus_marginal"] = row["fraction_strict_mlrw"] + row["fraction_marginal_mlrw"]
        metric_rows.append(row)
    return profiles, pd.DataFrame(metric_rows)


def write_outputs(output_dir: Path, subdirs: dict[str, Path], objects: pd.DataFrame, tau_grid: np.ndarray, args: argparse.Namespace) -> None:
    manifest, counts = build_manifest(objects)
    radii = representative_radii(objects) if not objects.empty else {}
    manifest.to_parquet(subdirs["object_cache"] / "selected_lifecycle_objects.parquet", index=False)
    manifest.to_csv(subdirs["object_cache"] / "selected_lifecycle_objects.csv", index=False)
    counts.to_csv(output_dir / "continuous_lifecycle_object_counts.csv", index=False)
    pd.DataFrame({"tau": tau_grid}).to_csv(output_dir / "continuous_tau_grid.csv", index=False)
    pd.DataFrame({"polarity": list(radii.keys()), "representative_radius_m": list(radii.values())}).to_csv(output_dir / "representative_radii.csv", index=False)

    lines = [
        "# Continuous lifecycle representative workspace",
        "",
        f"- Output root: `{output_dir}`",
        f"- Object cache: `{subdirs['object_cache']}`",
        f"- Tau grid: {','.join(f'{value:.4g}' for value in tau_grid)}",
        f"- Kernel bandwidth: {float(args.kernel_bandwidth):.4g}",
        f"- Radial bins: {int(args.radial_bins)}",
        f"- Azimuth bins: {int(args.azimuth_bins)}",
        f"- Objects selected: {int(manifest['eddy3d_object_id'].nunique()) if not manifest.empty else 0:,}",
        f"- Tracks selected: {int(manifest['track3d_id'].nunique()) if not manifest.empty and 'track3d_id' in manifest else 0:,}",
        "",
        "This directory is the new output root for continuous lifecycle experiments. The current command prepares the directory tree, selected-object cache, tau grid, and metadata manifest without writing into the historical LIFE_CYCLE_REPRESENTATIVE_VOLOCITY tree.",
        "",
        "## Counts",
        "```csv",
        counts.to_csv(index=False).strip() if not counts.empty else "no_objects_selected",
        "```",
    ]
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def compute_continuous_diagnostics(
    output_dir: Path,
    subdirs: dict[str, Path],
    objects: pd.DataFrame,
    tau_grid: np.ndarray,
    args: argparse.Namespace,
) -> None:
    if objects.empty:
        return
    center_lines = load_center_lines(Path(args.axis_dir), set(objects["eddy3d_object_id"].astype(int)))
    radial, theta, rr, tt, _ = make_polar_grid(float(args.rmax), int(args.radial_bins), int(args.azimuth_bins))
    lat_ref = float(objects["surface_lat"].median()) if "surface_lat" in objects else 27.5
    f0 = 2.0 * OMEGA * np.sin(np.radians(lat_ref))
    radii = representative_radii(objects)
    bandwidth = float(args.kernel_bandwidth)
    weight_min = float(args.kernel_weight_min)

    accum: dict = {}
    processed_objects = 0
    skipped_objects = 0
    depth = None
    for date, day_objects in tqdm(list(objects.groupby("date")), desc="Continuous tau diagnostics", unit="day"):
        path = Path(args.input_daily_dir) / f"uv_{pd.Timestamp(date):%Y%m%d}.nc"
        if not path.exists():
            continue
        lon, lat, depth, u, v = read_daily_uv(path)
        u = sanitize_ocean_field(u)
        v = sanitize_ocean_field(v)
        u_clim, v_clim = read_climatology_uv(Path(args.climatology_path), str(date))
        n2 = load_n2(Path(args.n2_profile_path), depth)
        _, dy, dx = grid_spacing_m(lon, lat)
        zeta = relative_vorticity(lon, lat, u, v)
        psi_prime = streamfunction_from_zeta(zeta, dx, dy)
        for obj in day_objects.itertuples(index=False):
            center_line = center_lines.get(int(obj.eddy3d_object_id))
            if center_line is None:
                skipped_objects += 1
                continue
            fields = sample_object_fields(obj, center_line, lon, lat, depth, psi_prime, u, v, u_clim, v_clim, None, None, rr, tt)
            if fields is None:
                skipped_objects += 1
                continue
            terms = compute_nondim_terms(
                fields,
                center_line,
                depth,
                radial,
                theta,
                float(obj.mean_radius_m),
                n2,
                f0,
                axis_mode="tilted",
            )
            psi = np.where(np.abs(fields["psi_prime"]) > 1e20, np.nan, fields["psi_prime"])
            terms["psi_mean"] = np.nanmean(psi, axis=2)
            terms["psi_variance"] = np.nanvar(psi, axis=2)
            life_phase = float(obj.life_phase)
            weights = np.exp(-0.5 * ((tau_grid - life_phase) / max(bandwidth, 1e-12)) ** 2)
            for tau_index, weight in enumerate(weights):
                if weight < weight_min:
                    continue
                key = ("tilted", str(obj.polarity), int(tau_index))
                add_weighted_terms(
                    accum,
                    key,
                    terms,
                    float(weight),
                    int(obj.eddy3d_object_id),
                    int(obj.track3d_id),
                    str(obj.date),
                )
            processed_objects += 1

    if not accum or depth is None:
        return
    final = finalize_weighted(accum)
    profiles = rows_from_continuous_final(final, radial, depth, tau_grid, radii)
    profiles = add_wave_activity(profiles, radii)
    profiles.to_parquet(subdirs["ep_flux_terms"] / "continuous_ep_flux_profiles.parquet", index=False)
    profiles.to_csv(subdirs["ep_flux_terms"] / "continuous_ep_flux_profiles.csv", index=False)
    profiles.to_parquet(subdirs["wave_action_total_flux"] / "continuous_wave_action_profiles.parquet", index=False)
    profiles.to_csv(subdirs["wave_action_total_flux"] / "continuous_wave_action_profiles.csv", index=False)

    psi_cols = [
        "polarity",
        "phase_index",
        "phase_name",
        "tau_center",
        "depth_index",
        "depth_m",
        "r_over_R",
        "psi_mean",
        "psi_variance",
        "count",
        "n_objects",
        "n_tracks",
        "n_dates",
    ]
    stream = profiles[[col for col in psi_cols if col in profiles.columns]].copy()
    stream.to_parquet(subdirs["streamfunction_templates"] / "continuous_radial_psi_profiles.parquet", index=False)
    stream.to_csv(subdirs["streamfunction_templates"] / "continuous_radial_psi_profiles.csv", index=False)

    nonlinearity, nonlinearity_metrics, nonlinearity_summary = build_nonlinearity_tables(profiles, radii)
    core_dir = subdirs["core_environment_sensitivity"]
    nonlinearity.to_parquet(core_dir / "continuous_nonlinearity_profiles.parquet", index=False)
    nonlinearity.to_csv(core_dir / "continuous_nonlinearity_profiles.csv", index=False)
    nonlinearity_metrics.to_csv(core_dir / "continuous_nonlinearity_metrics.csv", index=False)
    nonlinearity_summary.to_csv(core_dir / "continuous_nonlinearity_summary_by_polarity.csv", index=False)

    mlrw_profiles, mlrw_metrics = build_continuous_mlrw_profiles(nonlinearity)
    mlrw_dir = subdirs["mlrw_applicability"]
    mlrw_profiles.to_parquet(mlrw_dir / "continuous_mlrw_applicability_profiles.parquet", index=False)
    mlrw_profiles.to_csv(mlrw_dir / "continuous_mlrw_applicability_profiles.csv", index=False)
    mlrw_metrics.to_csv(mlrw_dir / "continuous_mlrw_applicability_metrics.csv", index=False)

    lines = [
        "# Continuous tau diagnostics",
        "",
        f"- Processed objects: {processed_objects:,}",
        f"- Skipped objects: {skipped_objects:,}",
        f"- Tau nodes: {len(tau_grid)}",
        f"- Kernel bandwidth: {bandwidth:.4g}",
        f"- Radial bins: {len(radial)}",
        f"- Azimuth bins: {len(theta)}",
        f"- Output root: `{output_dir}`",
        "",
        "## Connected Outputs",
        f"- Streamfunction templates: `{subdirs['streamfunction_templates'] / 'continuous_radial_psi_profiles.parquet'}`",
        f"- E-P flux terms: `{subdirs['ep_flux_terms'] / 'continuous_ep_flux_profiles.parquet'}`",
        f"- J_T wave-action profiles: `{subdirs['wave_action_total_flux'] / 'continuous_wave_action_profiles.parquet'}`",
        f"- MLRW applicability: `{subdirs['mlrw_applicability'] / 'continuous_mlrw_applicability_profiles.parquet'}`",
    ]
    (output_dir / "continuous_diagnostics_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    subdirs = ensure_output_tree(output_dir)
    tau_grid = parse_tau_grid(args.tau_grid, float(args.tau_grid_step))

    if bool(args.from_cache):
        cache_path = subdirs["object_cache"] / "selected_lifecycle_objects.parquet"
        if cache_path.exists():
            objects = pd.read_parquet(cache_path)
            required = {"surface_lon", "surface_lat", "mean_radius_m", "temp_direction_rad", "life_phase"}
            if not required.issubset(objects.columns):
                objects = select_lifecycle_objects(args)
        else:
            objects = select_lifecycle_objects(args)
    else:
        objects = select_lifecycle_objects(args)

    write_outputs(output_dir, subdirs, objects, tau_grid, args)
    if not bool(args.cache_only):
        compute_continuous_diagnostics(output_dir, subdirs, objects, tau_grid, args)
    print(f"Output: {output_dir}")
    print(f"Summary: {output_dir / 'summary.md'}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare the NEW_vorticity continuous lifecycle representative workspace.")
    parser.add_argument("--axis-dir", default=str(DEFAULT_AXIS_DIR))
    parser.add_argument("--catalog-dir", default=str(DEFAULT_CATALOG))
    parser.add_argument("--shape-dir", default=str(DEFAULT_SHAPE_BY_SHAPE_DIR))
    parser.add_argument("--input-daily-dir", default=str(DEFAULT_INPUT_DAILY))
    parser.add_argument("--climatology-path", default=str(DEFAULT_CLIMATOLOGY_NC))
    parser.add_argument("--n2-profile-path", default=str(DEFAULT_CLIMATOLOGY))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--shapes", default=",".join(DEFAULT_SHAPES))
    parser.add_argument("--polarities", default=",".join(DEFAULT_POLARITIES))
    parser.add_argument("--rmax", type=float, default=2.5)
    parser.add_argument("--max-days", type=int, default=0)
    parser.add_argument("--max-objects-per-polarity", type=int, default=0)
    parser.add_argument("--random-seed", type=int, default=20260713)
    parser.add_argument("--tau-grid", default="")
    parser.add_argument("--tau-grid-step", type=float, default=0.05)
    parser.add_argument("--kernel-bandwidth", type=float, default=0.075)
    parser.add_argument("--kernel-weight-min", type=float, default=1e-3)
    parser.add_argument("--radial-bins", type=int, default=40)
    parser.add_argument("--azimuth-bins", type=int, default=72)
    parser.add_argument("--cache-only", action="store_true", help="Prepare object cache and metadata only.")
    parser.add_argument("--from-cache", action="store_true", help="Reuse selected_lifecycle_objects.parquet when it exists.")
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
