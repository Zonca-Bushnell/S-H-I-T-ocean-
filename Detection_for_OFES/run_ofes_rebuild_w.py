from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

from .ofes_io import (
    archive_candidates,
    ctl_path,
    data_path,
    expected_dta_bytes,
    open_dta_memmap,
    parse_ctl,
    prefetch_daily_files,
    require_daily_file,
)


EARTH_RADIUS_M = 6_371_000.0
DEFAULT_DATA_ROOT = Path(r"F:\OFES\external_OFES2")
DEFAULT_RESULT_ROOT = Path(r"E:\DATA\01_Eddy_correspond\02_OFES\available_jan01_jan19_refined_ofes_grid")
SCIENCE_TAG = "raw_minus_jan_mean_diagnostic"


@dataclass(frozen=True)
class SelectedObject:
    hua_object_id: str
    date: str
    polarity: str
    pass_layers: int
    max_jump_km: float
    max_jump_over_r: float
    radius_km: float
    center_lon: float
    center_lat: float


def main() -> None:
    args = build_parser().parse_args()
    data_root = Path(args.data_root)
    result_root = Path(args.result_root)
    output_root = Path(args.output_root) if args.output_root else result_root / "w_rebuild_diagnostics"
    metadata_dir = output_root / "metadata"
    grids_dir = output_root / "grids"
    figures_dir = output_root / "figures"
    for path in [metadata_dir, grids_dir, figures_dir]:
        path.mkdir(parents=True, exist_ok=True)

    days = date_range(parse_iso_date(args.start), parse_iso_date(args.end))
    if "extract" in args.stages or "all" in args.stages:
        extract_w_prho(data_root, metadata_dir, days, int(args.extract_workers))
    if "diagnose" in args.stages or "all" in args.stages:
        run_diagnostics(data_root, result_root, output_root, grids_dir, figures_dir, args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare OFES native W against Dipole-style rebuilt W.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--stages", default="all", help="Comma-separated: extract,diagnose,all.")
    parser.add_argument("--start", default="1991-01-01")
    parser.add_argument("--end", default="1991-01-19")
    parser.add_argument("--date", default="1991-01-10", help="Diagnostic object date.")
    parser.add_argument("--hua-object-id", default=None)
    parser.add_argument("--max-objects", type=int, default=8)
    parser.add_argument("--max-depth-layers", type=int, default=105)
    parser.add_argument("--grid-n", type=int, default=81)
    parser.add_argument("--extent-r", type=float, default=4.0)
    parser.add_argument("--farfield-inner-r", type=float, default=2.0)
    parser.add_argument("--farfield-outer-r", type=float, default=4.0)
    parser.add_argument("--smooth-sigma-cells", type=float, default=1.0)
    parser.add_argument("--extract-workers", type=int, default=1)
    parser.add_argument("--backend", choices=["matplotlib", "pillow"], default="matplotlib")
    return parser


def extract_w_prho(root: Path, metadata_dir: Path, days: list[date], workers: int) -> None:
    rows = []
    for variable in ["w", "prho"]:
        meta = parse_ctl(ctl_path(root, variable))
        expected_bytes = expected_dta_bytes(meta)
        try:
            prefetch_daily_files(root, variable, days, workers=max(1, workers))
        except FileNotFoundError as exc:
            print(f"[extract-warning] {variable}: {exc}", flush=True)
        for day in days:
            path = data_path(root, variable, day)
            archives = archive_candidates(root, variable, day)
            complete = path.exists() and path.stat().st_size == expected_bytes
            row = {
                "variable": variable,
                "date": day.isoformat(),
                "source_archive": str(archives[0]) if archives else "",
                "target_path": str(path),
                "expected_bytes": expected_bytes,
                "actual_bytes": path.stat().st_size if path.exists() else 0,
                "complete": bool(complete),
            }
            rows.append(row)
            print(f"[extract-ready] {variable} {day.isoformat()} complete={complete}", flush=True)
    write_csv(metadata_dir / "w_prho_extract_manifest_jan01_jan19.csv", rows)
    write_json(metadata_dir / "w_prho_extract_manifest_jan01_jan19.json", rows)
    missing = [row for row in rows if not row["complete"]]
    if missing:
        raise FileNotFoundError("w/prho extraction incomplete: " + ", ".join(f"{r['variable']}:{r['date']}" for r in missing))


def run_diagnostics(data_root: Path, result_root: Path, output_root: Path, grids_dir: Path, figures_dir: Path, args: argparse.Namespace) -> None:
    target_day = parse_iso_date(args.date)
    centers = pd.read_csv(result_root / "detection_hua_global_jan1991" / "centers_hua_style.csv")
    structures = pd.read_csv(result_root / "detection_hua_global_jan1991" / "structures_hua_style.csv")
    selected = select_objects(centers, structures, target_day, args.hua_object_id, int(args.max_objects))

    metas = {name: parse_ctl(ctl_path(data_root, name)) for name in ["u", "v", "w", "prho"]}
    expected = {name: expected_dta_bytes(meta) for name, meta in metas.items()}
    paths = {name: require_daily_file(data_root, name, target_day, expected[name]) for name in ["u", "v", "w", "prho"]}
    raw = {name: open_dta_memmap(paths[name], metas[name]) for name in ["u", "v", "w", "prho"]}
    prev_next = open_neighbor_center_tables(centers, target_day)

    rows = []
    for idx, obj in enumerate(selected, start=1):
        print(f"[w-rebuild] {idx}/{len(selected)} {obj.hua_object_id}", flush=True)
        grid = rebuild_object_w(raw, metas, centers, obj, prev_next, args)
        stem = f"ofes_w_rebuild_4panel_{target_day:%Y%m%d}_{obj.hua_object_id}"
        npz_path = grids_dir / f"w_rebuild_grid_{target_day:%Y%m%d}_{obj.hua_object_id}.npz"
        json_path = grids_dir / f"w_rebuild_grid_{target_day:%Y%m%d}_{obj.hua_object_id}.json"
        np.savez_compressed(npz_path, **{key: value for key, value in grid.items() if isinstance(value, np.ndarray)})
        scalar_payload = {key: json_safe(value) for key, value in grid.items() if not isinstance(value, np.ndarray)}
        write_json(json_path, scalar_payload)
        figure_path = figures_dir / f"{stem}.png"
        pdf_path = figures_dir / f"{stem}.pdf"
        plot_four_panel(grid, figure_path, pdf_path, backend=str(args.backend))
        rows.append({"image_path": str(figure_path), "grid_npz": str(npz_path), "grid_json": str(json_path), **scalar_payload})
    pd.DataFrame(rows).to_csv(output_root / "run_summary.csv", index=False)
    write_json(output_root / "run_summary.json", rows)
    write_method_doc(output_root / "METHOD_OFES_REBUILD_W_ZH.md", args, rows)


def select_objects(centers: pd.DataFrame, structures: pd.DataFrame, target_day: date, requested_id: str | None, max_objects: int) -> list[SelectedObject]:
    if requested_id:
        part = structures[structures["hua_object_id"].astype(str).eq(str(requested_id))]
        if part.empty:
            raise ValueError(f"No structure rows for hua_object_id={requested_id}")
        return [summarize_object(part)]
    day_part = structures[structures["date"].astype(str).eq(target_day.isoformat())]
    items = [summarize_object(part) for _, part in day_part.groupby("hua_object_id", sort=False) if len(part) >= 2]
    items.sort(key=lambda obj: (obj.pass_layers, obj.max_jump_over_r, obj.max_jump_km), reverse=True)
    return items[: max(1, max_objects)]


def summarize_object(part: pd.DataFrame) -> SelectedObject:
    part = part.sort_values("depth_index")
    lon_col = "center_lon_refined" if "center_lon_refined" in part.columns else "center_lon"
    lat_col = "center_lat_refined" if "center_lat_refined" in part.columns else "center_lat"
    surface = part.iloc[0]
    mx, my = meters_per_degree(float(surface[lat_col]))
    lon = part[lon_col].to_numpy(dtype="f8")
    lat = part[lat_col].to_numpy(dtype="f8")
    radius_km = float(np.nanmedian(part["radius_km"].to_numpy(dtype="f8")))
    if not np.isfinite(radius_km) or radius_km <= 0:
        radius_km = 100.0
    jumps = []
    for a, b in zip(range(len(part) - 1), range(1, len(part))):
        dist = math.hypot(lon_delta_deg(lon[b], lon[a]) * mx / 1000.0, (lat[b] - lat[a]) * my / 1000.0)
        jumps.append(dist)
    max_jump = max(jumps) if jumps else 0.0
    return SelectedObject(
        hua_object_id=str(surface["hua_object_id"]),
        date=str(surface["date"]),
        polarity=str(surface["polarity"]),
        pass_layers=int(len(part)),
        max_jump_km=float(max_jump),
        max_jump_over_r=float(max_jump / radius_km),
        radius_km=radius_km,
        center_lon=float(surface[lon_col]),
        center_lat=float(surface[lat_col]),
    )


def open_neighbor_center_tables(centers: pd.DataFrame, target_day: date) -> dict[str, pd.DataFrame]:
    out = {}
    for key, day in [("prev", target_day - timedelta(days=1)), ("next", target_day + timedelta(days=1))]:
        part = centers[centers["date"].astype(str).eq(day.isoformat()) & centers["hua_pass"].astype(bool)]
        out[key] = part.copy()
    return out


def rebuild_object_w(raw: dict[str, np.memmap], metas: dict[str, object], centers: pd.DataFrame, obj: SelectedObject, neighbors: dict[str, pd.DataFrame], args: argparse.Namespace) -> dict[str, object]:
    nlev = min(int(args.max_depth_layers), metas["u"].z.count, metas["w"].z.count, metas["prho"].z.count)
    x_over_r = np.linspace(-float(args.extent_r), float(args.extent_r), int(args.grid_n))
    y_over_r = np.linspace(-float(args.extent_r), float(args.extent_r), int(args.grid_n))
    xxr, yyr = np.meshgrid(x_over_r, y_over_r)
    lon_grid, lat_grid = local_lon_lat_grid(xxr, yyr, obj.center_lon, obj.center_lat, obj.radius_km)

    u = sample_stack(raw["u"], metas["u"], lon_grid, lat_grid, nlev) / 100.0
    v = sample_stack(raw["v"], metas["v"], lon_grid, lat_grid, nlev) / 100.0
    native_w = sample_stack(raw["w"], metas["w"], lon_grid, lat_grid, nlev)
    prho = sample_stack(raw["prho"], metas["prho"], lon_grid, lat_grid, nlev)
    prho = normalize_density_units(prho)

    r = np.hypot(xxr, yyr)
    far_mask = (r >= float(args.farfield_inner_r)) & (r <= float(args.farfield_outer_r))
    rho_bg = np.nanmedian(np.where(far_mask[None, :, :], prho, np.nan), axis=(1, 2))
    depth = metas["prho"].z.values[:nlev].astype("f8")
    drho_dz = np.gradient(rho_bg, depth)
    rho_prime = prho - rho_bg[:, None, None]
    z_anom = np.divide(-rho_prime, drho_dz[:, None, None], out=np.full_like(rho_prime, np.nan), where=np.abs(drho_dz[:, None, None]) >= 1.0e-8)
    z_anom = nan_gaussian_smooth_3d(z_anom, float(args.smooth_sigma_cells))

    dx_m = float(np.mean(np.diff(x_over_r))) * obj.radius_km * 1000.0
    dy_m = float(np.mean(np.diff(y_over_r))) * obj.radius_km * 1000.0
    dzdx = np.empty_like(z_anom)
    dzdy = np.empty_like(z_anom)
    for k in range(nlev):
        dzdy[k], dzdx[k] = np.gradient(z_anom[k], dy_m, dx_m)

    cx, cy, c_valid, c_method = estimate_translation(centers, neighbors, obj)
    support = np.isfinite(z_anom) & np.isfinite(u) & np.isfinite(v) & np.isfinite(native_w)
    term1 = cx * dzdx + cy * dzdy
    term2 = -((u - cx) * dzdx + (v - cy) * dzdy)
    rebuild_w = term1 + term2
    for arr in [term1, term2, rebuild_w]:
        arr[~support] = np.nan
    native_w[~support] = np.nan

    corr = spatial_corr(rebuild_w, native_w)
    corr_minus = spatial_corr(-rebuild_w, native_w)
    return {
        **asdict(obj),
        "science_tag": SCIENCE_TAG,
        "depth_m": depth,
        "x_over_r": x_over_r,
        "y_over_r": y_over_r,
        "term1_m_s": term1,
        "term2_m_s": term2,
        "rebuild_w_m_s": rebuild_w,
        "ofes_w_native_m_s": native_w,
        "z_rho_anom_m": z_anom,
        "rho_bg": rho_bg,
        "drho_dz": drho_dz,
        "cx_m_s": float(cx),
        "cy_m_s": float(cy),
        "c_valid": bool(c_valid),
        "c_method": c_method,
        "grid_n": int(args.grid_n),
        "extent_r": float(args.extent_r),
        "valid_grid_fraction": float(np.isfinite(rebuild_w).sum() / rebuild_w.size),
        "corr_rebuild_native": float(corr),
        "corr_minus_rebuild_native": float(corr_minus),
        "dipole_score_rebuild": float(dipole_score(np.nanmean(rebuild_w, axis=0), xxr, yyr)),
        "dipole_score_native": float(dipole_score(np.nanmean(native_w, axis=0), xxr, yyr)),
        "q95_abs_rebuild_1e6_m_s": q95_abs(rebuild_w) * 1.0e6,
        "q95_abs_native_1e6_m_s": q95_abs(native_w) * 1.0e6,
    }


def sample_stack(raw: np.memmap, meta, lon_grid: np.ndarray, lat_grid: np.ndarray, nlev: int) -> np.ndarray:
    lon = meta.x.values
    lat = meta.y.values
    lon_step = float(np.nanmedian(np.diff(lon)))
    lat_step = float(np.nanmedian(np.diff(lat)))
    ii = ((lon_grid - float(lon[0])) / lon_step) % len(lon)
    jj = np.clip((lat_grid - float(lat[0])) / lat_step, 0, len(lat) - 1)
    out = np.empty((nlev, lon_grid.shape[0], lon_grid.shape[1]), dtype="f4")
    coords = np.vstack([ii.ravel(), jj.ravel()])
    for k in range(nlev):
        layer = np.asarray(raw[:, :, k], dtype="f4")
        layer[np.abs(layer) > 1.0e30] = np.nan
        sampled = ndimage.map_coordinates(layer, coords, order=1, mode="wrap", cval=np.nan).reshape(lon_grid.shape)
        out[k] = sampled
    return out


def estimate_translation(centers: pd.DataFrame, neighbors: dict[str, pd.DataFrame], obj: SelectedObject) -> tuple[float, float, bool, str]:
    prev_row = nearest_neighbor(neighbors.get("prev", pd.DataFrame()), obj)
    next_row = nearest_neighbor(neighbors.get("next", pd.DataFrame()), obj)
    mx, my = meters_per_degree(obj.center_lat)
    if prev_row is not None and next_row is not None:
        cx = lon_delta_deg(float(next_row["center_lon_refined"]), float(prev_row["center_lon_refined"])) * mx / (2.0 * 86400.0)
        cy = (float(next_row["center_lat_refined"]) - float(prev_row["center_lat_refined"])) * my / (2.0 * 86400.0)
        return cx, cy, True, "central_neighbor"
    if next_row is not None:
        cx = lon_delta_deg(float(next_row["center_lon_refined"]), obj.center_lon) * mx / 86400.0
        cy = (float(next_row["center_lat_refined"]) - obj.center_lat) * my / 86400.0
        return cx, cy, True, "forward_neighbor"
    if prev_row is not None:
        cx = lon_delta_deg(obj.center_lon, float(prev_row["center_lon_refined"])) * mx / 86400.0
        cy = (obj.center_lat - float(prev_row["center_lat_refined"])) * my / 86400.0
        return cx, cy, True, "backward_neighbor"
    return 0.0, 0.0, False, "missing_neighbor"


def nearest_neighbor(rows: pd.DataFrame, obj: SelectedObject) -> pd.Series | None:
    if rows.empty:
        return None
    lon_col = "center_lon_refined" if "center_lon_refined" in rows.columns else "center_lon"
    lat_col = "center_lat_refined" if "center_lat_refined" in rows.columns else "center_lat"
    surface = rows[rows["depth_index"].astype(int).eq(0)].copy()
    if surface.empty:
        surface = rows.sort_values("depth_index").groupby("hua_object_id", as_index=False).first()
    same = surface[surface["polarity"].astype(str).eq(obj.polarity)]
    if not same.empty:
        surface = same
    mx, my = meters_per_degree(obj.center_lat)
    dist = np.hypot([lon_delta_deg(v, obj.center_lon) * mx / 1000.0 for v in surface[lon_col]], (surface[lat_col].astype(float) - obj.center_lat) * my / 1000.0)
    idx = int(np.nanargmin(dist))
    if float(dist[idx]) > max(250.0, 2.0 * obj.radius_km):
        return None
    return surface.iloc[idx]


def local_lon_lat_grid(xxr: np.ndarray, yyr: np.ndarray, lon0: float, lat0: float, radius_km: float) -> tuple[np.ndarray, np.ndarray]:
    mx, my = meters_per_degree(lat0)
    lon = (lon0 + xxr * radius_km * 1000.0 / mx) % 360.0
    lat = lat0 + yyr * radius_km * 1000.0 / my
    return lon, lat


def normalize_density_units(prho: np.ndarray) -> np.ndarray:
    finite = prho[np.isfinite(prho)]
    if finite.size and float(np.nanmedian(finite)) > 1000.0:
        return prho - 1000.0
    return prho


def nan_gaussian_smooth_3d(values: np.ndarray, sigma_cells: float) -> np.ndarray:
    if sigma_cells <= 0:
        return values
    out = np.empty_like(values, dtype="f4")
    for k in range(values.shape[0]):
        out[k] = nan_gaussian_smooth_2d(values[k], sigma_cells)
    return out


def nan_gaussian_smooth_2d(values: np.ndarray, sigma_cells: float) -> np.ndarray:
    arr = np.asarray(values, dtype="f8")
    valid = np.isfinite(arr)
    weights = ndimage.gaussian_filter(valid.astype("f8"), sigma=sigma_cells, mode="nearest")
    smoothed = ndimage.gaussian_filter(np.where(valid, arr, 0.0), sigma=sigma_cells, mode="nearest")
    return np.divide(smoothed, weights, out=np.full_like(smoothed, np.nan), where=weights > 1.0e-8).astype("f4")


def plot_four_panel(grid: dict[str, object], png_path: Path, pdf_path: Path, backend: str) -> None:
    if backend == "matplotlib":
        try:
            plot_four_panel_matplotlib(grid, png_path, pdf_path)
            return
        except BaseException as exc:
            print(f"[plot-warning] matplotlib failed, falling back to pillow: {exc}", flush=True)
    plot_four_panel_pillow(grid, png_path)


def plot_four_panel_matplotlib(grid: dict[str, object], png_path: Path, pdf_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = grid["x_over_r"]
    y = grid["y_over_r"]
    fields = [
        ("term1", np.nanmean(grid["term1_m_s"], axis=0)),
        ("term2", np.nanmean(grid["term2_m_s"], axis=0)),
        ("rebuild W", np.nanmean(grid["rebuild_w_m_s"], axis=0)),
        ("OFES native W", np.nanmean(grid["ofes_w_native_m_s"], axis=0)),
    ]
    vals = np.concatenate([arr[np.isfinite(arr)] for _, arr in fields if np.isfinite(arr).any()])
    lim = float(np.nanpercentile(np.abs(vals), 95)) * 1.0e6 if vals.size else 1.0
    lim = max(lim, 1.0e-12)
    fig, axes = plt.subplots(2, 2, figsize=(13, 11), constrained_layout=True)
    th = np.linspace(0, 2 * np.pi, 361)
    for ax, (title, data) in zip(axes.ravel(), fields):
        mesh = ax.pcolormesh(x, y, data * 1.0e6, shading="auto", cmap="RdBu_r", vmin=-lim, vmax=lim)
        ax.contour(x, y, data * 1.0e6, levels=np.linspace(-lim, lim, 9), colors="0.25", linewidths=0.55, alpha=0.7)
        ax.plot(np.cos(th), np.sin(th), "k-", lw=1.2)
        ax.plot(4 * np.cos(th), 4 * np.sin(th), "k-", lw=1.2)
        ax.plot(0, 0, "k.", ms=16)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(title)
        ax.set_xlabel("x/R")
        ax.set_ylabel("y/R")
        fig.colorbar(mesh, ax=ax, shrink=0.82)
    fig.suptitle(
        f"OFES W rebuild diagnostic | {grid['hua_object_id']} | {grid['date']} | corr={grid['corr_rebuild_native']:.3f} | q95 rebuild/native={grid['q95_abs_rebuild_1e6_m_s']:.2g}/{grid['q95_abs_native_1e6_m_s']:.2g} x10^-6 m/s"
    )
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=180)
    fig.savefig(pdf_path)
    plt.close(fig)


def plot_four_panel_pillow(grid: dict[str, object], png_path: Path) -> None:
    x = grid["x_over_r"]
    y = grid["y_over_r"]
    fields = [
        ("term1", np.nanmean(grid["term1_m_s"], axis=0)),
        ("term2", np.nanmean(grid["term2_m_s"], axis=0)),
        ("rebuild W", np.nanmean(grid["rebuild_w_m_s"], axis=0)),
        ("OFES native W", np.nanmean(grid["ofes_w_native_m_s"], axis=0)),
    ]
    vals = np.concatenate([arr[np.isfinite(arr)] for _, arr in fields if np.isfinite(arr).any()])
    lim = max(float(np.nanpercentile(np.abs(vals), 95)) * 1.0e6 if vals.size else 1.0, 1.0e-12)
    canvas = Image.new("RGB", (1800, 1450), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(30)
    main_font = load_font(20)
    small_font = load_font(15)
    draw.text((40, 30), f"OFES W rebuild diagnostic | {grid['hua_object_id']} | {grid['date']}", fill=(20, 24, 32), font=title_font)
    draw.text((40, 70), f"corr={grid['corr_rebuild_native']:.3f}, corr(-rebuild,W)={grid['corr_minus_rebuild_native']:.3f}, q95 rebuild/native={grid['q95_abs_rebuild_1e6_m_s']:.2g}/{grid['q95_abs_native_1e6_m_s']:.2g} x10^-6 m/s", fill=(80, 88, 100), font=small_font)
    boxes = [(60, 125, 860, 735), (940, 125, 1740, 735), (60, 800, 860, 1410), (940, 800, 1740, 1410)]
    for box, (name, data) in zip(boxes, fields):
        draw_field_pillow(canvas, box, x, y, data * 1.0e6, -lim, lim, name, main_font, small_font)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(png_path)


def draw_field_pillow(canvas: Image.Image, box: tuple[int, int, int, int], x: np.ndarray, y: np.ndarray, data: np.ndarray, vmin: float, vmax: float, title: str, main_font, small_font) -> None:
    draw = ImageDraw.Draw(canvas)
    draw.rectangle(box, outline=(180, 186, 196), width=1)
    plot = (box[0] + 55, box[1] + 38, box[2] - 30, box[3] - 45)
    img = array_to_rgb(data, vmin, vmax).resize((plot[2] - plot[0], plot[3] - plot[1]), Image.Resampling.BILINEAR)
    canvas.paste(img, plot[:2])
    th = np.linspace(0, 2 * np.pi, 361)
    for rr in [1.0, 4.0]:
        pts = [map_xy(plot, rr * math.cos(t), rr * math.sin(t), float(x[0]), float(x[-1]), float(y[0]), float(y[-1])) for t in th]
        draw.line(pts, fill=(0, 0, 0), width=2)
    cx, cy = map_xy(plot, 0.0, 0.0, float(x[0]), float(x[-1]), float(y[0]), float(y[-1]))
    draw.ellipse((cx - 4, cy - 4, cx + 4, cy + 4), fill=(0, 0, 0))
    draw.text((box[0] + 10, box[1] + 10), title, fill=(30, 36, 48), font=main_font)
    draw.text((plot[0], box[3] - 30), "x/R", fill=(80, 88, 100), font=small_font)
    draw.text((box[0] + 10, box[1] + 35), f"+/- {vmax:.2g} x10^-6 m/s", fill=(80, 88, 100), font=small_font)


def array_to_rgb(values: np.ndarray, vmin: float, vmax: float) -> Image.Image:
    arr = np.asarray(values, dtype="f4")
    nan = ~np.isfinite(arr)
    scaled = np.clip((arr - vmin) / max(vmax - vmin, 1.0e-12), 0.0, 1.0)
    colors = rdbu_r_colors(np.nan_to_num(scaled, nan=0.5))
    colors[nan] = np.array([255, 255, 255], dtype=np.uint8)
    return Image.fromarray(colors, mode="RGB")


def rdbu_r_colors(scaled: np.ndarray) -> np.ndarray:
    blue = np.array([5, 113, 176], dtype="f4")
    white = np.array([247, 247, 247], dtype="f4")
    red = np.array([202, 0, 32], dtype="f4")
    rgb = np.empty((*scaled.shape, 3), dtype=np.uint8)
    low = scaled <= 0.5
    rgb[low] = (blue * (1.0 - scaled[low, None] / 0.5) + white * (scaled[low, None] / 0.5)).astype(np.uint8)
    high = ~low
    rgb[high] = (white * (1.0 - (scaled[high, None] - 0.5) / 0.5) + red * ((scaled[high, None] - 0.5) / 0.5)).astype(np.uint8)
    return rgb


def spatial_corr(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if int(mask.sum()) < 3:
        return float("nan")
    av = np.asarray(a[mask], dtype="f8")
    bv = np.asarray(b[mask], dtype="f8")
    av -= av.mean()
    bv -= bv.mean()
    denom = math.sqrt(float(np.sum(av * av) * np.sum(bv * bv)))
    return float(np.sum(av * bv) / denom) if denom > 0 else float("nan")


def dipole_score(w: np.ndarray, x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(w) & (np.hypot(x, y) <= 4.0)
    if int(mask.sum()) < 3:
        return float("nan")
    east = np.nanmean(w[mask & (x > 0)])
    west = np.nanmean(w[mask & (x < 0)])
    north = np.nanmean(w[mask & (y > 0)])
    south = np.nanmean(w[mask & (y < 0)])
    scale = np.nanmean(np.abs(w[mask]))
    if not np.isfinite(scale) or scale <= 0:
        return float("nan")
    return float(max(abs(east - west), abs(north - south)) / scale)


def q95_abs(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(np.nanpercentile(np.abs(finite), 95)) if finite.size else float("nan")


def meters_per_degree(lat_deg: float) -> tuple[float, float]:
    lat_rad = math.radians(lat_deg)
    return math.pi * EARTH_RADIUS_M * math.cos(lat_rad) / 180.0, math.pi * EARTH_RADIUS_M / 180.0


def lon_delta_deg(lon: float, lon0: float) -> float:
    return ((float(lon) - float(lon0) + 180.0) % 360.0) - 180.0


def date_range(start: date, end: date) -> list[date]:
    current = start
    out = []
    while current <= end:
        out.append(current)
        current += timedelta(days=1)
    return out


def parse_iso_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def json_safe(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=True), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_method_doc(path: Path, args: argparse.Namespace, rows: list[dict[str, object]]) -> None:
    text = f"""# OFES 原生 W vs rebuild W 诊断

- 科学口径：`{SCIENCE_TAG}` 下的等密面异常法。
- 数据根目录：`{args.data_root}`
- 结果根目录：`{args.result_root}`
- 目标日期：`{args.date}`
- 重建公式：`term1 = c · grad(z'_rho)`，`term2 = -[(u-c_x),(v-c_y)] · grad(z'_rho)`，`rebuild_W = term1 + term2`。
- `z'_rho = -rho' / d rho_bg/dz`，其中 `rho_bg(z)` 来自局地 `2-4R` 远场环。
- OFES 原生 `w` 作为参照；图中单位为 `10^-6 m/s`，网格文件保存原始 `m/s`。
- 首版不做正式 tracking；传播速度来自相邻日 nearest-neighbor surface center，失败时用 `c=0` 并记录。
- 输出对象数：`{len(rows)}`
"""
    path.write_text(text, encoding="utf-8")


def load_font(size: int) -> ImageFont.ImageFont:
    for name in ["arial.ttf", "DejaVuSans.ttf"]:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def map_xy(box: tuple[int, int, int, int], x: float, y: float, xmin: float, xmax: float, ymin: float, ymax: float) -> tuple[int, int]:
    px = box[0] + int(round((x - xmin) / max(xmax - xmin, 1.0e-12) * (box[2] - box[0] - 1)))
    py = box[3] - int(round((y - ymin) / max(ymax - ymin, 1.0e-12) * (box[3] - box[1] - 1)))
    return px, py


if __name__ == "__main__":
    main()
