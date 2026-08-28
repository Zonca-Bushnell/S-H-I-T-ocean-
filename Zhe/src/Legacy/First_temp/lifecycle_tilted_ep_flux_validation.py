from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from .axis_streamfunction_separation import DEFAULT_AXIS_DIR, DEFAULT_CATALOG, DEFAULT_INPUT_DAILY, grid_spacing_m, parse_csv_list, read_daily_uv, relative_vorticity, streamfunction_from_zeta
from .lifecycle_common import (
    DEFAULT_LIFECYCLE_ROOT,
    DEFAULT_POLARITIES,
    DEFAULT_SHAPE_BY_SHAPE_DIR,
    DEFAULT_SHAPES,
    PHASE_NAMES,
    apply_lifecycle_limits,
    load_center_lines,
    load_lifecycle_objects,
)
from .tilted_ep_flux_validation import (
    DEFAULT_CLIMATOLOGY,
    DEFAULT_CLIMATOLOGY_NC,
    OMEGA,
    compute_q_and_flux_terms,
    divergence,
    load_n2,
    make_polar_grid,
    read_climatology_uv,
    sample_object_fields,
)


DEFAULT_OUTPUT = DEFAULT_LIFECYCLE_ROOT / "ep_flux_validation"


def add_to_accum(accum: dict, key: tuple[str, int, str], terms: dict[str, np.ndarray]) -> None:
    if key not in accum:
        accum[key] = {name: np.zeros_like(value, dtype="f8") for name, value in terms.items() if name != "valid"}
        accum[key]["count"] = np.zeros_like(terms["valid"], dtype="f8")
        accum[key]["objects"] = set()
        accum[key]["dates"] = set()
    valid = np.isfinite(terms["valid"]) & (terms["valid"] > 0)
    for name, value in terms.items():
        if name == "valid":
            continue
        accum[key][name] += np.nan_to_num(value, nan=0.0) * valid
    accum[key]["count"] += valid.astype("f8")


def finalize_accum(accum: dict) -> dict:
    final = {}
    for key, item in accum.items():
        count = item["count"]
        out = {"count": count, "objects": item["objects"], "dates": item["dates"]}
        for name, value in item.items():
            if name in {"count", "objects", "dates"}:
                continue
            out[name] = np.divide(value, count, out=np.full_like(value, np.nan), where=count > 0)
        final[key] = out
    return final


def build_profile_tables(final: dict, radial: np.ndarray, depth: np.ndarray, radii: dict[str, float]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    closure_rows = []
    feedback_rows = []
    for (polarity, phase_index, phase_name), item in sorted(final.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        radius = radii.get(polarity)
        if radius is None:
            continue
        divf = divergence(item["F_n"], item["F_z"], radial, depth, radius)
        residual = divf - item["pv_flux"]
        good = np.isfinite(divf) & np.isfinite(item["pv_flux"]) & (item["count"] > 0)
        corr = float(np.corrcoef(divf[good].ravel(), item["pv_flux"][good].ravel())[0, 1]) if np.sum(good) > 2 else np.nan
        rmse = float(np.sqrt(np.nanmean(residual[good] ** 2))) if np.any(good) else np.nan
        denom = float(np.sqrt(np.nanmean(item["pv_flux"][good] ** 2))) if np.any(good) else np.nan
        closure_rows.append(
            {
                "shape_class": "all_shapes",
                "polarity": polarity,
                "phase_index": int(phase_index),
                "phase_name": phase_name,
                "closure_corr": corr,
                "closure_rmse": rmse,
                "closure_relative_rmse": rmse / denom if denom and np.isfinite(denom) else np.nan,
                "closure_same_sign_fraction": float(np.mean(np.sign(divf[good]) == np.sign(item["pv_flux"][good]))) if np.any(good) else np.nan,
                "valid_bins": int(np.sum(good)),
                "n_objects": len(item["objects"]),
                "n_dates": len(item["dates"]),
            }
        )
        dudt = item.get("dUdt_clim", np.full_like(divf, np.nan))
        good_feedback = np.isfinite(divf) & np.isfinite(dudt) & (item["count"] > 0)
        feedback_rows.append(
            {
                "shape_class": "all_shapes",
                "polarity": polarity,
                "phase_index": int(phase_index),
                "phase_name": phase_name,
                "feedback_corr_divF_dUdt_clim": float(np.corrcoef(divf[good_feedback], dudt[good_feedback])[0, 1]) if np.sum(good_feedback) > 2 else np.nan,
                "feedback_slope_dUdt_per_divF": float(np.polyfit(divf[good_feedback], dudt[good_feedback], deg=1)[0]) if np.sum(good_feedback) > 2 else np.nan,
                "feedback_same_sign_fraction": float(np.mean(np.sign(divf[good_feedback]) == np.sign(dudt[good_feedback]))) if np.any(good_feedback) else np.nan,
                "valid_bins": int(np.sum(good_feedback)),
                "n_objects": len(item["objects"]),
                "n_dates": len(item["dates"]),
                "feedback_note": "dUdt is the next-day climatological along-axis flow tendency sampled on the same east-aligned tilted lifecycle coordinates; it is a trend check, not a complete momentum budget.",
            }
        )
        for k, depth_m in enumerate(depth):
            for j, r in enumerate(radial):
                rows.append(
                    {
                        "shape_class": "all_shapes",
                        "polarity": polarity,
                        "phase_index": int(phase_index),
                        "phase_name": phase_name,
                        "depth_index": k,
                        "depth_m": float(depth_m),
                        "r_over_R": float(r),
                        "F_n": float(item["F_n"][k, j]),
                        "F_z": float(item["F_z"][k, j]),
                        "divF": float(divf[k, j]),
                        "pv_flux": float(item["pv_flux"][k, j]),
                        "closure_residual": float(residual[k, j]),
                        "Ubar_prime_axisym": float(item["Ubar_prime_axisym"][k, j]),
                        "Ubar_clim": float(item["Ubar_clim"][k, j]),
                        "Ubar_clim_next": float(item.get("Ubar_clim_next", np.full_like(item["Ubar_clim"], np.nan))[k, j]),
                        "dUdt_clim": float(dudt[k, j]),
                        "count": float(item["count"][k, j]),
                    }
                )
    return pd.DataFrame.from_records(rows), pd.DataFrame.from_records(closure_rows), pd.DataFrame.from_records(feedback_rows)


def write_alignment(objects: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    cols = ["eddy3d_object_id", "polarity", "phase_index", "phase_name", "track3d_id", "date", "temp_direction_rad", "deep_x_rot_m", "deep_y_rot_m", "deep_rotation_abs_y_m"]
    diag = objects[cols].copy()
    diag["deep_points_east"] = diag["deep_x_rot_m"] > 0
    diag.to_csv(output_dir / "east_alignment_diagnostics.csv", index=False)
    return diag


def plot_outputs(profiles: pd.DataFrame, alignment: pd.DataFrame, output_dir: Path) -> None:
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    for (polarity, phase_name), part in profiles.groupby(["polarity", "phase_name"], sort=True):
        label = f"all_shapes_{polarity}_{phase_name}"
        pivot = lambda name: part.pivot(index="depth_m", columns="r_over_R", values=name).sort_index()
        divf = pivot("divF")
        pv = pivot("pv_flux")
        res = pivot("closure_residual")
        finite = pd.concat([divf.stack(), pv.stack()])
        vmax = np.nanpercentile(np.abs(finite), 98) if len(finite) else 1.0
        if not np.isfinite(vmax) or vmax <= 0:
            vmax = 1.0
        fig, axes = plt.subplots(1, 3, figsize=(14, 5), dpi=160)
        for ax, data, title in zip(axes, (divf, pv, res), ("div F_T", "mean u_n' q_T'", "residual")):
            mesh = ax.pcolormesh(data.columns.to_numpy(dtype="f8"), data.index.to_numpy(dtype="f8"), data.to_numpy(), shading="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
            ax.invert_yaxis()
            ax.set_xlabel("r/R")
            ax.set_title(title)
        axes[0].set_ylabel("depth m")
        fig.colorbar(mesh, ax=axes, label="diagnostic units")
        fig.suptitle(label)
        fig.tight_layout()
        fig.savefig(figure_dir / f"{label}_divF_pvflux_residual.png")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6, 5), dpi=160)
        good = np.isfinite(part["divF"]) & np.isfinite(part["pv_flux"])
        ax.scatter(part.loc[good, "pv_flux"], part.loc[good, "divF"], s=8, alpha=0.5)
        ax.set_xlabel("mean u_n' q_T'")
        ax.set_ylabel("div F_T")
        ax.set_title(f"{label}: closure")
        ax.grid(True, color="0.9")
        fig.tight_layout()
        fig.savefig(figure_dir / f"{label}_divF_vs_pv_flux.png")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6, 5), dpi=160)
        good = np.isfinite(part["divF"]) & np.isfinite(part["dUdt_clim"])
        ax.scatter(part.loc[good, "dUdt_clim"], part.loc[good, "divF"], s=8, alpha=0.5)
        ax.set_xlabel("d U_clim / dt")
        ax.set_ylabel("div F_T")
        ax.set_title(f"{label}: feedback trend")
        ax.grid(True, color="0.9")
        fig.tight_layout()
        fig.savefig(figure_dir / f"{label}_divF_vs_dUdt_clim.png")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5), dpi=160)
    ax.scatter(alignment["deep_x_rot_m"], alignment["deep_y_rot_m"], s=4, alpha=0.25)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xlabel("deep x_rot m")
    ax.set_ylabel("deep y_rot m")
    ax.set_title("Lifecycle east-alignment check")
    fig.tight_layout()
    fig.savefig(figure_dir / "east_alignment_check.png")
    plt.close(fig)


def write_summary(output_dir: Path, objects: pd.DataFrame, closure: pd.DataFrame, feedback: pd.DataFrame, alignment: pd.DataFrame, args: argparse.Namespace) -> None:
    counts = (
        objects.groupby(["polarity", "phase_index", "phase_name"], as_index=False)
        .agg(n_objects=("eddy3d_object_id", "nunique"), n_dates=("date", "nunique"))
        .sort_values(["polarity", "phase_index"])
    )
    counts.to_csv(output_dir / "lifecycle_object_counts.csv", index=False)
    missing = []
    for polarity in parse_csv_list(args.polarities, DEFAULT_POLARITIES):
        for index, phase_name in enumerate(PHASE_NAMES):
            if counts[(counts["polarity"] == polarity) & (counts["phase_index"] == index)].empty:
                missing.append(f"{polarity}:{phase_name}")
    lines = [
        "# Lifecycle tilted E-P flux validation summary",
        "",
        f"- Objects selected: {len(objects):,}",
        f"- Output dir: `{output_dir}`",
        f"- Perturbation field: `{args.input_daily_dir}` u/v",
        f"- Mean field: `{args.climatology_path}` u_clim/v_clim",
        f"- East alignment median |deep_y_rot|: {alignment['deep_y_rot_m'].abs().median():.3e} m",
        f"- East alignment positive deep_x fraction: {(alignment['deep_x_rot_m'] > 0).mean():.3f}",
        "- Perturbation q_T excludes planetary beta; beta belongs to the climatological mean PV.",
        "",
        "## Object Counts",
        "```csv",
        counts.to_csv(index=False).strip(),
        "```",
        "",
        "## Closure Metrics",
        "```csv",
        closure.to_csv(index=False).strip() if not closure.empty else "No closure metrics generated.",
        "```",
        "",
        "## Feedback Metrics",
        "```csv",
        feedback.to_csv(index=False).strip() if not feedback.empty else "No feedback metrics generated.",
        "```",
        "",
        "Note: feedback is a trend validation against next-day climatological along-axis flow tendency, not a full transformed momentum budget.",
    ]
    if missing:
        lines.extend(["", "## Skipped Phase Groups", ", ".join(missing)])
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    shapes = parse_csv_list(args.shapes, DEFAULT_SHAPES)
    polarities = parse_csv_list(args.polarities, DEFAULT_POLARITIES)
    objects = load_lifecycle_objects(
        axis_dir=Path(args.axis_dir),
        catalog_dir=Path(args.catalog_dir),
        shape_dir=Path(args.shape_dir),
        shapes=shapes,
        polarities=polarities,
    )
    objects = apply_lifecycle_limits(objects, int(args.max_days), int(args.max_objects_per_polarity), int(args.random_seed))
    if objects.empty:
        raise RuntimeError("No lifecycle objects selected.")
    center_lines = load_center_lines(Path(args.axis_dir), set(objects["eddy3d_object_id"].astype(int)))
    alignment = write_alignment(objects, output_dir)
    radial, theta, rr, tt, _ = make_polar_grid(float(args.rmax), int(args.radial_bins), int(args.azimuth_bins))
    lat_ref = float(objects["surface_lat"].median()) if not objects.empty else 27.5
    f0 = 2.0 * OMEGA * np.sin(np.radians(lat_ref))
    radii = {str(polarity): float(part["mean_radius_m"].median()) for polarity, part in objects.groupby("polarity")}

    accum: dict = {}
    depth_ref: np.ndarray | None = None
    for date, day_objects in tqdm(list(objects.groupby("date")), desc="Lifecycle EP flux", unit="day"):
        path = Path(args.input_daily_dir) / f"uv_{pd.Timestamp(date):%Y%m%d}.nc"
        if not path.exists():
            continue
        lon, lat, depth, u, v = read_daily_uv(path)
        u_clim, v_clim = read_climatology_uv(Path(args.climatology_path), str(date))
        next_date = (pd.Timestamp(date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        u_clim_next, v_clim_next = read_climatology_uv(Path(args.climatology_path), next_date)
        n2 = load_n2(Path(args.n2_profile_path), depth)
        _, dy, dx = grid_spacing_m(lon, lat)
        zeta = relative_vorticity(lon, lat, u, v)
        psi_prime = streamfunction_from_zeta(zeta, dx, dy)
        depth_ref = depth if depth_ref is None else depth_ref
        for obj in day_objects.itertuples(index=False):
            center_line = center_lines.get(int(obj.eddy3d_object_id))
            if center_line is None:
                continue
            fields = sample_object_fields(obj, center_line, lon, lat, depth, psi_prime, u, v, u_clim, v_clim, u_clim_next, v_clim_next, rr, tt)
            if fields is None:
                continue
            terms = compute_q_and_flux_terms(fields, depth, radial, theta, float(obj.mean_radius_m), n2, f0)
            key = (str(obj.polarity), int(obj.phase_index), str(obj.phase_name))
            add_to_accum(accum, key, terms)
            accum[key]["objects"].add(int(obj.eddy3d_object_id))
            accum[key]["dates"].add(str(obj.date))
    if depth_ref is None:
        raise RuntimeError("No daily uv files were processed.")
    final = finalize_accum(accum)
    profiles, closure, feedback = build_profile_tables(final, radial, depth_ref, radii)
    profiles.to_parquet(output_dir / "lifecycle_ep_flux_profiles.parquet", index=False)
    closure.to_csv(output_dir / "lifecycle_ep_flux_closure_metrics.csv", index=False)
    feedback.to_csv(output_dir / "lifecycle_ep_flux_feedback_metrics.csv", index=False)
    profiles[["shape_class", "polarity", "phase_index", "phase_name", "depth_index", "depth_m", "r_over_R", "count"]].to_csv(output_dir / "lifecycle_ep_flux_object_counts.csv", index=False)
    plot_outputs(profiles, alignment, output_dir)
    write_summary(output_dir, objects, closure, feedback, alignment, args)
    print(f"Output: {output_dir}")
    print(f"Profiles: {output_dir / 'lifecycle_ep_flux_profiles.parquet'}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate tilted E-P flux diagnostics by lifecycle phase.")
    parser.add_argument("--axis-dir", default=str(DEFAULT_AXIS_DIR))
    parser.add_argument("--catalog-dir", default=str(DEFAULT_CATALOG))
    parser.add_argument("--shape-dir", default=str(DEFAULT_SHAPE_BY_SHAPE_DIR))
    parser.add_argument("--input-daily-dir", default=str(DEFAULT_INPUT_DAILY))
    parser.add_argument("--climatology-path", default=str(DEFAULT_CLIMATOLOGY_NC))
    parser.add_argument("--n2-profile-path", default=str(DEFAULT_CLIMATOLOGY))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--shapes", default=",".join(DEFAULT_SHAPES))
    parser.add_argument("--polarities", default=",".join(DEFAULT_POLARITIES))
    parser.add_argument("--max-days", type=int, default=0, help="Maximum dates per polarity+phase for smoke runs.")
    parser.add_argument("--max-objects-per-polarity", type=int, default=0, help="Maximum objects per polarity+phase for smoke runs.")
    parser.add_argument("--rmax", type=float, default=2.5)
    parser.add_argument("--radial-bins", type=int, default=40)
    parser.add_argument("--azimuth-bins", type=int, default=72)
    parser.add_argument("--random-seed", type=int, default=20260710)
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
