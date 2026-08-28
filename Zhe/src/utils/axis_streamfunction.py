from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    from scipy.fft import dstn, idstn
except Exception:  # pragma: no cover
    dstn = None
    idstn = None

try:
    from netCDF4 import Dataset
except Exception:  # pragma: no cover
    Dataset = None

try:
    import h5py
except Exception:  # pragma: no cover
    h5py = None


EARTH_RADIUS_M = 6_371_000.0
DEFAULT_AXIS_DIR = Path(r"E:\Verify\outputs\kuroshio_cmems_3d\first_temp_direction_fit_1993_2022_by_polarity")
DEFAULT_CATALOG = Path(r"E:\Verify\outputs\kuroshio_cmems_3d\catalog")
DEFAULT_INPUT_DAILY = Path(r"E:\Verify\outputs\kuroshio_cmems_3d\input_daily")
DEFAULT_OUTPUT = Path(r"E:\Verify\outputs\kuroshio_cmems_3d\axis_streamfunction_separation_1993_2022")
DEFAULT_SHAPE_ORDER = ("coherent", "complex", "mixed", "transitional", "upright_like")
DEFAULT_POLARITIES = ("anticyclonic", "cyclonic")


def parse_csv_list(value: str | None, default: tuple[str, ...]) -> tuple[str, ...]:
    if value is None or not value.strip():
        return default
    return tuple(part.strip() for part in value.split(",") if part.strip())


def local_xy_m(lon: np.ndarray, lat: np.ndarray, lon0: float, lat0: float) -> tuple[np.ndarray, np.ndarray]:
    dlon = (lon - lon0 + 180.0) % 360.0 - 180.0
    x = np.radians(dlon) * EARTH_RADIUS_M * np.cos(np.radians(lat0))
    y = np.radians(lat - lat0) * EARTH_RADIUS_M
    return x, y


def grid_spacing_m(lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, float, float]:
    dlon = float(np.nanmedian(np.diff(lon)))
    dlat = float(np.nanmedian(np.diff(lat)))
    dx_by_lat = EARTH_RADIUS_M * np.cos(np.radians(lat)) * np.radians(dlon)
    dy = EARTH_RADIUS_M * np.radians(dlat)
    dx = float(np.nanmedian(np.abs(dx_by_lat[np.isfinite(dx_by_lat)])))
    return dx_by_lat.astype("f8"), abs(float(dy)), dx


def relative_vorticity(lon: np.ndarray, lat: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    dx_by_lat, dy, _ = grid_spacing_m(lon, lat)
    dvdx = np.gradient(v, axis=2) / dx_by_lat[None, :, None]
    dudy = np.gradient(u, axis=1) / dy
    return dvdx - dudy


def streamfunction_from_zeta(zeta: np.ndarray, dx: float, dy: float) -> np.ndarray:
    rhs = np.nan_to_num(zeta, nan=0.0, posinf=0.0, neginf=0.0)
    nz, ny, nx = rhs.shape
    psi = np.zeros_like(rhs, dtype="f8")
    if dstn is None or idstn is None or ny < 3 or nx < 3:
        return psi
    rhs_i = rhs[:, 1:-1, 1:-1]
    rhs_hat = dstn(rhs_i, type=1, axes=(1, 2), norm="ortho")
    nx_i = rhs_i.shape[2]
    ny_i = rhs_i.shape[1]
    mx = np.arange(1, nx_i + 1, dtype="f8")
    my = np.arange(1, ny_i + 1, dtype="f8")
    lam_x = -4.0 * np.sin(np.pi * mx / (2.0 * (nx_i + 1))) ** 2 / max(dx * dx, 1e-12)
    lam_y = -4.0 * np.sin(np.pi * my / (2.0 * (ny_i + 1))) ** 2 / max(dy * dy, 1e-12)
    denom = lam_y[None, :, None] + lam_x[None, None, :]
    psi_hat = np.divide(rhs_hat, denom, out=np.zeros_like(rhs_hat), where=np.abs(denom) > 1e-12)
    psi[:, 1:-1, 1:-1] = idstn(psi_hat, type=1, axes=(1, 2), norm="ortho")
    return psi


def read_daily_uv(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if Dataset is not None:
        try:
            with Dataset(path) as ds:
                lon = np.asarray(ds.variables["longitude"][:], dtype="f8")
                lat = np.asarray(ds.variables["latitude"][:], dtype="f8")
                depth = np.asarray(ds.variables["depth"][:], dtype="f8")
                u = read_variable_clean(ds.variables["u"])
                v = read_variable_clean(ds.variables["v"])
            return lon, lat, depth, u, v
        except Exception:
            pass
    if h5py is None:
        raise RuntimeError("Neither netCDF4 nor h5py can read daily uv files.")
    with h5py.File(path, "r") as f:
        lon = np.asarray(f["longitude"][:], dtype="f8")
        lat = np.asarray(f["latitude"][:], dtype="f8")
        depth = np.asarray(f["depth"][:], dtype="f8")
        u = read_variable_clean(f["u"])
        v = read_variable_clean(f["v"])
    return lon, lat, depth, u, v


def read_variable_clean(dataset) -> np.ndarray:
    values = np.asarray(dataset[:], dtype="f8")
    for attr_name in ("_FillValue", "missing_value"):
        if attr_name not in dataset.attrs:
            continue
        raw = np.asarray(dataset.attrs[attr_name]).ravel()
        for fill_value in raw:
            if np.isfinite(fill_value):
                values[np.isclose(values, float(fill_value), rtol=0.0, atol=max(abs(float(fill_value)) * 1e-7, 1.0))] = np.nan
    values[np.abs(values) > 1e20] = np.nan
    return values


def polarity_sign(polarity: str) -> float:
    return -1.0 if str(polarity).lower().startswith("anti") else 1.0


def load_objects(axis_dir: Path, catalog_dir: Path, shapes: tuple[str, ...], polarities: tuple[str, ...]) -> pd.DataFrame:
    obj = pd.read_parquet(axis_dir / "object_diagnostics.parquet")
    obj = obj[obj["is_usable"]].copy()
    obj = obj[obj["shape_class"].isin(shapes) & obj["polarity"].isin(polarities)].copy()
    vertical = pd.read_parquet(
        catalog_dir / "vertical_objects.parquet",
        columns=["eddy3d_object_id", "mean_radius_m"],
    )
    out = obj.merge(vertical, on="eddy3d_object_id", how="left")
    out = out[np.isfinite(out["mean_radius_m"]) & (out["mean_radius_m"] > 0)].copy()
    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    return out


def limit_objects(df: pd.DataFrame, max_objects_per_group: int, seed: int) -> pd.DataFrame:
    if max_objects_per_group <= 0 or df.empty:
        return df
    rng = np.random.default_rng(seed)
    keep: list[int] = []
    for _, part in df.groupby(["shape_class", "polarity"]):
        ids = part["eddy3d_object_id"].to_numpy(dtype="int64")
        if ids.size > max_objects_per_group:
            ids = rng.choice(ids, size=max_objects_per_group, replace=False)
        keep.extend(int(v) for v in ids)
    return df[df["eddy3d_object_id"].isin(keep)].copy()


def load_fits(axis_dir: Path) -> dict[tuple[str, str], dict]:
    fits = pd.read_csv(axis_dir / "fit_coefficients.csv")
    return {
        (str(row.shape_class), str(row.polarity)): row._asdict()
        for row in fits.itertuples(index=False)
    }


def axis_xy_for_object(obj, fit: dict, depth: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    z = np.asarray(depth, dtype="f8") - float(obj.surface_depth_m)
    x_rot = fit["c1"] + fit["c2"] * z + fit["c3"] * z * z
    y_rot = fit["c4"] + fit["c5"] * z + fit["c6"] * z * z
    theta = float(obj.temp_direction_rad)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    x = x_rot * cos_t - y_rot * sin_t
    y = x_rot * sin_t + y_rot * cos_t
    return x, y


def bin_object_profiles(
    obj,
    fit: dict,
    lon: np.ndarray,
    lat: np.ndarray,
    depth: np.ndarray,
    psi: np.ndarray,
    *,
    rmax: float,
    radial_edges: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_grid, y_grid = local_xy_m(lon[None, :], lat[:, None], float(obj.surface_lon), float(obj.surface_lat))
    x_axis, y_axis = axis_xy_for_object(obj, fit, depth)
    radius = float(obj.mean_radius_m)
    n_depth = len(depth)
    n_bin = len(radial_edges) - 1
    sums = np.zeros((n_depth, n_bin), dtype="f8")
    counts = np.zeros((n_depth, n_bin), dtype="i8")
    sums_norm = np.zeros((n_depth, n_bin), dtype="f8")
    sign = polarity_sign(str(obj.polarity))
    for k in range(n_depth):
        dx = x_grid - x_axis[k]
        dy = y_grid - y_axis[k]
        r_norm = np.hypot(dx, dy) / radius
        mask = r_norm <= rmax
        if not np.any(mask):
            continue
        layer = psi[k].astype("f8")
        core_mask = r_norm <= min(0.15, rmax)
        core = np.nan
        if np.any(core_mask):
            core = np.nanmean(layer[core_mask])
        if not np.isfinite(core):
            iy, ix = np.unravel_index(np.nanargmin(r_norm), r_norm.shape)
            core = layer[iy, ix]
        values = layer[mask] - core
        values_norm = values * sign
        bins = np.searchsorted(radial_edges, r_norm[mask], side="right") - 1
        good = (bins >= 0) & (bins < n_bin) & np.isfinite(values)
        if not np.any(good):
            continue
        np.add.at(sums[k], bins[good], values[good])
        np.add.at(sums_norm[k], bins[good], values_norm[good])
        np.add.at(counts[k], bins[good], 1)
    return sums, sums_norm, counts


def fit_rank1(matrix: np.ndarray, counts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    good = counts > 0
    filled = np.where(good, matrix, 0.0)
    if not np.any(good):
        return filled, np.zeros(filled.shape[0]), np.zeros(filled.shape[1]), {}
    row_mean = np.divide(filled.sum(axis=1), good.sum(axis=1), out=np.zeros(filled.shape[0]), where=good.sum(axis=1) > 0)
    filled = np.where(good, filled, row_mean[:, None])
    u, s, vt = np.linalg.svd(filled, full_matrices=False)
    recon = s[0] * np.outer(u[:, 0], vt[0])
    resid = np.where(good, matrix - recon, np.nan)
    total = np.nansum(np.where(good, matrix, np.nan) ** 2)
    res2 = np.nansum(resid**2)
    metrics = {
        "rank1_energy_fraction": float((s[0] ** 2) / np.sum(s * s)) if np.sum(s * s) > 0 else np.nan,
        "rmse": float(np.sqrt(np.nanmean(resid**2))),
        "relative_rmse": float(np.sqrt(res2 / total)) if total > 0 else np.nan,
        "valid_fraction": float(np.mean(good)),
    }
    return recon, u[:, 0] * s[0], vt[0], metrics


def plot_group(
    label: str,
    matrix: np.ndarray,
    recon: np.ndarray,
    residual: np.ndarray,
    h: np.ndarray,
    r_component: np.ndarray,
    r: np.ndarray,
    depth: np.ndarray,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    vmax = np.nanpercentile(np.abs(matrix), 98) if np.isfinite(matrix).any() else 1.0
    for name, data, title in (("psi", matrix, "Psi(r,z)"), ("rank1", recon, "rank-1 fit"), ("residual", residual, "residual")):
        fig, ax = plt.subplots(figsize=(8, 6), dpi=160)
        mesh = ax.pcolormesh(r, depth, data, shading="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.invert_yaxis()
        ax.set_xlabel("r/R")
        ax.set_ylabel("depth m")
        ax.set_title(f"{label}: {title}")
        fig.colorbar(mesh, ax=ax, label="psi relative")
        fig.tight_layout()
        fig.savefig(out_dir / f"{label}_{name}.png")
        plt.close(fig)
        plot_3d_surface(label, name, data, title, r, depth, out_dir, vmax)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), dpi=160)
    axes[0].plot(r, r_component, color="#d62728")
    axes[0].set_xlabel("r/R")
    axes[0].set_title("R(r), arbitrary scale")
    axes[0].grid(True, color="0.9")
    axes[1].plot(h, depth, color="#1f77b4")
    axes[1].invert_yaxis()
    axes[1].set_xlabel("H(z), arbitrary scale")
    axes[1].set_ylabel("depth m")
    axes[1].set_title("H(z)")
    axes[1].grid(True, color="0.9")
    fig.suptitle(label)
    fig.tight_layout()
    fig.savefig(out_dir / f"{label}_RH.png")
    plt.close(fig)


def plot_3d_surface(
    label: str,
    name: str,
    data: np.ndarray,
    title: str,
    r: np.ndarray,
    depth: np.ndarray,
    out_dir: Path,
    vmax: float,
) -> None:
    if not np.isfinite(data).any():
        return
    r_grid, depth_grid = np.meshgrid(r, depth)
    z_data = np.asarray(data, dtype="f8")
    z_plot = np.ma.masked_invalid(z_data)
    fig = plt.figure(figsize=(9, 7), dpi=160)
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(
        r_grid,
        depth_grid,
        z_plot,
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        linewidth=0,
        antialiased=True,
        shade=True,
        alpha=0.96,
    )
    ax.set_xlabel("r/R")
    ax.set_ylabel("depth m")
    ax.set_zlabel("psi relative")
    ax.set_title(f"{label}: {title} 3D surface")
    ax.invert_yaxis()
    ax.view_init(elev=28, azim=-132)
    fig.colorbar(surf, ax=ax, shrink=0.65, pad=0.12, label="psi relative")
    fig.tight_layout()
    fig.savefig(out_dir / f"{label}_{name}_3d.png")
    plt.close(fig)


def regenerate_3d_figures_from_profiles(output_dir: Path) -> int:
    profiles_path = output_dir / "radial_psi_profiles.parquet"
    if not profiles_path.exists():
        raise FileNotFoundError(profiles_path)
    profiles = pd.read_parquet(profiles_path)
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    made = 0
    for (shape, polarity), part in profiles.groupby(["shape_class", "polarity"], sort=True):
        depth = np.sort(part["depth_m"].unique().astype("f8"))
        r = np.sort(part["r_over_R"].unique().astype("f8"))
        depth_index = {float(v): i for i, v in enumerate(depth)}
        r_index = {float(v): i for i, v in enumerate(r)}
        matrix = np.full((len(depth), len(r)), np.nan, dtype="f8")
        recon = np.full_like(matrix, np.nan)
        residual = np.full_like(matrix, np.nan)
        for row in part.itertuples(index=False):
            i = depth_index[float(row.depth_m)]
            j = r_index[float(row.r_over_R)]
            matrix[i, j] = float(row.psi_mean)
            recon[i, j] = float(row.psi_rank1)
            residual[i, j] = float(row.psi_residual)
        vmax = np.nanpercentile(np.abs(matrix), 98) if np.isfinite(matrix).any() else 1.0
        label = f"{shape}_{polarity}".replace(" ", "_")
        for name, data, title in (("psi", matrix, "Psi(r,z)"), ("rank1", recon, "rank-1 fit"), ("residual", residual, "residual")):
            plot_3d_surface(label, name, data, title, r, depth, figure_dir, vmax)
            made += 1
    return made
def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    figure_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    shapes = parse_csv_list(args.shapes, DEFAULT_SHAPE_ORDER)
    polarities = parse_csv_list(args.polarities, DEFAULT_POLARITIES)
    objects = load_objects(Path(args.axis_dir), Path(args.catalog_dir), shapes, polarities)
    if args.start_date:
        objects = objects[objects["date"] >= args.start_date]
    if args.end_date:
        objects = objects[objects["date"] <= args.end_date]
    objects = limit_objects(objects, int(args.max_objects_per_group), int(args.random_seed))
    if int(args.max_days) > 0:
        keep_dates = sorted(objects["date"].unique())[: int(args.max_days)]
        objects = objects[objects["date"].isin(keep_dates)].copy()

    fits = load_fits(Path(args.axis_dir))
    radial_edges = np.linspace(0.0, float(args.rmax), int(args.radial_bins) + 1)
    radial_centers = 0.5 * (radial_edges[:-1] + radial_edges[1:])

    accum: dict[tuple[str, str], dict[str, np.ndarray | int | set]] = {}
    depth_ref: np.ndarray | None = None
    for date, day_objects in tqdm(list(objects.groupby("date")), desc="Process daily psi", unit="day"):
        path = Path(args.input_daily_dir) / f"uv_{pd.Timestamp(date):%Y%m%d}.nc"
        if not path.exists():
            continue
        lon, lat, depth, u, v = read_daily_uv(path)
        _, dy, dx = grid_spacing_m(lon, lat)
        zeta = relative_vorticity(lon, lat, u, v)
        psi = streamfunction_from_zeta(zeta, dx, dy)
        depth_ref = depth if depth_ref is None else depth_ref
        for obj in day_objects.itertuples(index=False):
            key = (str(obj.shape_class), str(obj.polarity))
            fit = fits.get(key)
            if fit is None:
                continue
            sums, sums_norm, counts = bin_object_profiles(
                obj,
                fit,
                lon,
                lat,
                depth,
                psi,
                rmax=float(args.rmax),
                radial_edges=radial_edges,
            )
            for out_key, use_sums in (
                (key, sums),
                ((str(obj.shape_class), "polarity_normalized"), sums_norm),
            ):
                if out_key not in accum:
                    accum[out_key] = {
                        "sums": np.zeros_like(use_sums, dtype="f8"),
                        "counts": np.zeros_like(counts, dtype="i8"),
                        "objects": set(),
                        "dates": set(),
                    }
                accum[out_key]["sums"] += use_sums
                accum[out_key]["counts"] += counts
                accum[out_key]["objects"].add(int(obj.eddy3d_object_id))
                accum[out_key]["dates"].add(str(obj.date))

    if depth_ref is None:
        raise RuntimeError("No daily uv files were processed.")

    profile_rows: list[dict] = []
    coeff_rows: list[dict] = []
    metric_rows: list[dict] = []
    for (shape, polarity), item in sorted(accum.items()):
        sums = item["sums"]
        counts = item["counts"]
        matrix = np.divide(sums, counts, out=np.full_like(sums, np.nan, dtype="f8"), where=counts > 0)
        recon, h, r_component, metrics = fit_rank1(matrix, counts)
        residual = matrix - recon
        metrics.update(
            {
                "shape_class": shape,
                "polarity": polarity,
                "n_objects": len(item["objects"]),
                "n_dates": len(item["dates"]),
                "n_valid_bins": int(np.sum(counts > 0)),
                "n_total_samples": int(np.sum(counts)),
            }
        )
        metric_rows.append(metrics)
        label = f"{shape}_{polarity}".replace(" ", "_")
        plot_group(label, matrix, recon, residual, h, r_component, radial_centers, depth_ref, figure_dir)
        for k, depth_m in enumerate(depth_ref):
            for j, r in enumerate(radial_centers):
                if counts[k, j] <= 0:
                    continue
                profile_rows.append(
                    {
                        "shape_class": shape,
                        "polarity": polarity,
                        "depth_index": k,
                        "depth_m": float(depth_m),
                        "r_over_R": float(r),
                        "psi_mean": float(matrix[k, j]),
                        "psi_rank1": float(recon[k, j]),
                        "psi_residual": float(residual[k, j]),
                        "count": int(counts[k, j]),
                    }
                )
        for k, depth_m in enumerate(depth_ref):
            coeff_rows.append({"shape_class": shape, "polarity": polarity, "component": "H", "index": k, "coord": float(depth_m), "value": float(h[k])})
        for j, r in enumerate(radial_centers):
            coeff_rows.append({"shape_class": shape, "polarity": polarity, "component": "R", "index": j, "coord": float(r), "value": float(r_component[j])})

    profiles = pd.DataFrame.from_records(profile_rows)
    coeffs = pd.DataFrame.from_records(coeff_rows)
    metrics = pd.DataFrame.from_records(metric_rows)
    profiles.to_parquet(output_dir / "radial_psi_profiles.parquet", index=False)
    coeffs.to_parquet(output_dir / "separable_fit_coefficients.parquet", index=False)
    coeffs.to_csv(output_dir / "separable_fit_coefficients.csv", index=False)
    metrics.to_csv(output_dir / "separability_metrics.csv", index=False)
    write_summary(output_dir, objects=objects, metrics=metrics, profiles=profiles, args=args)
    plot_metric_summary(metrics, figure_dir)
    print(f"Output: {output_dir}")
    print(f"Metrics: {output_dir / 'separability_metrics.csv'}")
    print(f"Profiles: {output_dir / 'radial_psi_profiles.parquet'}")


def plot_metric_summary(metrics: pd.DataFrame, figure_dir: Path) -> None:
    if metrics.empty:
        return
    use = metrics.sort_values(["shape_class", "polarity"])
    labels = [f"{r.shape_class}\n{r.polarity}" for r in use.itertuples(index=False)]
    fig, ax = plt.subplots(figsize=(max(8, 0.7 * len(labels)), 5), dpi=160)
    ax.bar(labels, use["rank1_energy_fraction"].astype(float), color="#4c78a8")
    ax.set_ylim(0, 1)
    ax.set_ylabel("rank-1 energy fraction")
    ax.set_title("Streamfunction separability by group")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(figure_dir / "separability_rank1_energy_fraction.png")
    plt.close(fig)


def write_summary(output_dir: Path, *, objects: pd.DataFrame, metrics: pd.DataFrame, profiles: pd.DataFrame, args: argparse.Namespace) -> None:
    lines = [
        "# Axis streamfunction separation summary",
        "",
        f"- Axis dir: `{args.axis_dir}`",
        f"- Input daily dir: `{args.input_daily_dir}`",
        f"- Objects selected: {len(objects):,}",
        f"- Dates selected: {objects['date'].nunique() if not objects.empty else 0:,}",
        f"- Radial range: 0 <= r/R <= {args.rmax}, bins={args.radial_bins}",
        f"- Radial profile rows: {len(profiles):,}",
        "",
        "## Metrics",
        "```csv",
        metrics.to_csv(index=False).strip() if not metrics.empty else "No metrics generated.",
        "```",
    ]
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Test psi(r,z)=R(r)H(z) around a shape+polarity quadratic vortex axis.")
    parser.add_argument("--axis-dir", default=str(DEFAULT_AXIS_DIR))
    parser.add_argument("--catalog-dir", default=str(DEFAULT_CATALOG))
    parser.add_argument("--input-daily-dir", default=str(DEFAULT_INPUT_DAILY))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--shapes", default=",".join(DEFAULT_SHAPE_ORDER))
    parser.add_argument("--polarities", default=",".join(DEFAULT_POLARITIES))
    parser.add_argument("--start-date", default="")
    parser.add_argument("--end-date", default="")
    parser.add_argument("--max-days", type=int, default=0)
    parser.add_argument("--max-objects-per-group", type=int, default=0)
    parser.add_argument("--rmax", type=float, default=2.5)
    parser.add_argument("--radial-bins", type=int, default=50)
    parser.add_argument("--random-seed", type=int, default=20260708)
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
