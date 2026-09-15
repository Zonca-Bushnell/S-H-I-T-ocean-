from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from ..ofes_io import ctl_path, expected_dta_bytes, open_dta_memmap, parse_ctl, require_daily_file
from ..run_ofes_rebuild_w import (
    SelectedObject,
    json_safe,
    load_detection_tables,
    meters_per_degree,
    lon_delta_deg,
    open_neighbor_center_tables,
    parse_iso_date,
    q95_abs,
    rebuild_object_w,
    resolve_detection_table_dir,
    spatial_corr,
    write_csv,
    write_json,
)
from .plot_ofes_theory_rebuild_w_composite import (
    DEFAULT_RESULT_ROOT,
    alpha_from_structure,
    build_runtime_args,
    draw_colorbar,
    draw_field_panel,
    load_coherent_selected_objects,
    rotate_scalar_stack,
    rotate_xy,
)
from ..w_rebuild_config import DEFAULT_DATA_ROOT


DEFAULT_OUTPUT_ROOT = DEFAULT_RESULT_ROOT / "w_rebuild_diagnostics" / "single_coherent_tilt_theory"


def main() -> None:
    cli = parse_args()
    output_root = cli.output_root
    figures_dir = output_root / "figures"
    grids_dir = output_root / "grids"
    figures_dir.mkdir(parents=True, exist_ok=True)
    grids_dir.mkdir(parents=True, exist_ok=True)

    detection_dir = resolve_detection_table_dir(cli.result_root)
    centers, structures = load_detection_tables(detection_dir)
    groups, alpha_info = load_coherent_selected_objects(cli.result_root, structures)
    objects = [obj for values in groups.values() for obj in values]
    if not objects:
        raise RuntimeError("No coherent OFES object-days were found in the current Origin result.")

    obj = select_object(objects, alpha_info, cli.hua_object_id)
    part = structures[structures["hua_object_id"].astype(str).eq(obj.hua_object_id)].copy()
    if part.empty:
        raise RuntimeError(f"Missing structures_hua_style rows for {obj.hua_object_id}")
    alpha = alpha_info.get(obj.hua_object_id) or alpha_from_structure(part, obj)

    runtime_args = runtime_args_from_cli(cli)
    metas = {name: parse_ctl(ctl_path(cli.data_root, name)) for name in ["u", "v", "w", "prho"]}
    expected = {name: expected_dta_bytes(meta) for name, meta in metas.items()}
    target_day = parse_iso_date(obj.date)
    paths = {name: require_daily_file(cli.data_root, name, target_day, expected[name]) for name in ["u", "v", "w", "prho"]}
    raw = {name: open_dta_memmap(paths[name], metas[name]) for name in ["u", "v", "w", "prho"]}
    neighbors = open_neighbor_center_tables(centers, target_day)

    print(f"[single-tilt-theory] rebuilding {obj.hua_object_id} ({obj.date}, {obj.polarity})", flush=True)
    grid = rebuild_object_w(raw, metas, centers, obj, neighbors, runtime_args)
    rotated = rotate_fields_for_theory(grid, float(alpha["alpha_deg"]))
    centerline = centerline_offsets(part, obj, len(np.asarray(grid["depth_m"])), float(alpha["alpha_deg"]))
    density = density_tilt_decomposition(rotated["rho_prime_for_rebuild"], np.asarray(grid["x_over_r"]), centerline["x_rot_r"])
    metrics = density_metrics(density)

    payload = {
        **{key: json_safe(value) for key, value in grid.items() if not isinstance(value, np.ndarray)},
        "alpha_deg": float(alpha["alpha_deg"]),
        "tilt_distance_r": float(alpha["tilt_distance_r"]),
        "tilt_dx_rot_r": float(alpha.get("tilt_dx_rot_r", np.nan)),
        "tilt_dy_rot_r": float(alpha.get("tilt_dy_rot_r", np.nan)),
        "alpha_source": "global_ls_centerline_single_object",
        "density_theory": "layer-centered rho_hat(xi,z)=rho_prime(xi+Xc,z); surface-fixed odd rho_prime_odd(x,z) ~= -Xc_rot(z) * d(rho_even_intrinsic)/d(x/R)",
        "single_object_scale_mode": str(cli.scale_mode),
        **metrics,
    }

    token = safe_object_token(obj)
    npz_path = grids_dir / f"single_coherent_tilt_theory_{token}.npz"
    json_path = grids_dir / f"single_coherent_tilt_theory_{token}.json"
    np.savez_compressed(
        npz_path,
        depth_m=np.asarray(grid["depth_m"], dtype="f4"),
        x_over_r=np.asarray(grid["x_over_r"], dtype="f4"),
        y_over_r=np.asarray(grid["y_over_r"], dtype="f4"),
        x_centerline_rot_r=centerline["x_rot_r"].astype("f4"),
        y_centerline_rot_r=centerline["y_rot_r"].astype("f4"),
        rho_prime_rot=density["rho"].astype("f4"),
        rho_layer_centered_rot=density["rho_layer_centered"].astype("f4"),
        rho_intrinsic_even_rot=density["rho_intrinsic_even"].astype("f4"),
        rho_intrinsic_odd_rot=density["rho_intrinsic_odd"].astype("f4"),
        rho_surface_fixed_odd_rot=density["rho_surface_odd"].astype("f4"),
        rho_tilt_theory_rot=density["rho_odd_theory"].astype("f4"),
        term1_rot_m_s=rotated["term1_m_s"].astype("f4"),
        term2_rot_m_s=rotated["term2_m_s"].astype("f4"),
        rebuild_w_rot_m_s=rotated["rebuild_w_m_s"].astype("f4"),
        native_w_rot_m_s=rotated["ofes_w_native_m_s"].astype("f4"),
        z_rho_anom_rot_m=rotated["z_rho_anom_m"].astype("f4"),
    )
    write_json(json_path, payload)

    density_png = figures_dir / f"single_coherent_density_tilt_theory_{token}.png"
    w_png = figures_dir / f"single_coherent_w_terms_{token}.png"
    slice_png = figures_dir / f"single_coherent_density_w_450m_slice_{token}.png"
    plot_density_decomposition(grid, density, centerline, payload, density_png)
    plot_w_terms(grid, rotated, payload, w_png)
    plot_450m_slice(grid, density, rotated, payload, slice_png)

    row = {
        "hua_object_id": obj.hua_object_id,
        "date": obj.date,
        "polarity": obj.polarity,
        "pass_layers": obj.pass_layers,
        "radius_km": obj.radius_km,
        "center_lon": obj.center_lon,
        "center_lat": obj.center_lat,
        "alpha_deg": payload["alpha_deg"],
        "tilt_distance_r": payload["tilt_distance_r"],
        "rho_odd_corr_actual_theory": payload["rho_surface_odd_corr_tilt_theory"],
        "rho_odd_q95_actual": payload["rho_surface_odd_q95"],
        "rho_odd_q95_theory": payload["rho_odd_q95_theory"],
        "rho_intrinsic_odd_q95": payload["rho_intrinsic_odd_q95"],
        "density_png": str(density_png),
        "w_png": str(w_png),
        "slice_png": str(slice_png),
        "grid_npz": str(npz_path),
        "grid_json": str(json_path),
    }
    write_csv(output_root / "single_coherent_tilt_theory_summary.csv", [row])
    write_json(output_root / "single_coherent_tilt_theory_summary.json", [row])
    print(f"[single-tilt-theory] wrote {output_root}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot one coherent OFES object-day to verify tilted-density theory and rebuild-W terms."
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--hua-object-id", default="", help="Optional object-day id. Empty selects a tilted coherent case.")
    parser.add_argument("--start", default="1991-01-01")
    parser.add_argument("--end", default="1991-01-19")
    parser.add_argument("--max-depth-layers", type=int, default=105)
    parser.add_argument("--grid-n", type=int, default=81)
    parser.add_argument("--extent-r", type=float, default=4.0)
    parser.add_argument("--cressman-radius-r", type=float, default=1.0)
    parser.add_argument("--cressman-min-objects", type=int, default=1)
    parser.add_argument("--max-objects-per-group", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--scale-mode",
        choices=["recommended", "raw_fast"],
        default="recommended",
        help="recommended keeps the current 10-day density/velocity scale separation; raw_fast uses same-day local fields for a quick original-object check.",
    )
    return parser.parse_args()


def runtime_args_from_cli(cli: argparse.Namespace) -> argparse.Namespace:
    args = build_runtime_args(cli)
    args.single_object_scale_mode = str(cli.scale_mode)
    if cli.scale_mode == "raw_fast":
        args.rebuild_density_filter = "none"
        args.rebuild_velocity_filter = "none"
        args.native_w_temporal_filter = "none"
        args.native_w_meso_filter = "none"
        args.precompute_object_temporal_blocks = False
        args.translation_profile_smooth_sigma_layers = 0.0
    return args


def select_object(objects: list[SelectedObject], alpha_info: dict[str, dict[str, float]], requested: str) -> SelectedObject:
    if requested:
        for obj in objects:
            if obj.hua_object_id == requested:
                return obj
        raise ValueError(f"--hua-object-id {requested} is not in the coherent object-day catalog")

    def score(obj: SelectedObject) -> tuple[float, int, float]:
        alpha = alpha_info.get(obj.hua_object_id, {})
        tilt = float(alpha.get("tilt_distance_r", 0.0))
        return (tilt * max(1, obj.pass_layers), obj.pass_layers, obj.radius_km)

    return max(objects, key=score)


def rotate_fields_for_theory(grid: dict[str, object], alpha_deg: float) -> dict[str, np.ndarray]:
    x = np.asarray(grid["x_over_r"], dtype="f8")
    y = np.asarray(grid["y_over_r"], dtype="f8")
    xx, yy = np.meshgrid(x, y)
    x_orig, y_orig = rotate_xy(xx, yy, -alpha_deg)
    keys = [
        "rho_prime_for_rebuild",
        "z_rho_anom_m",
        "term1_m_s",
        "term2_m_s",
        "rebuild_w_m_s",
        "ofes_w_native_m_s",
    ]
    return {
        key: rotate_scalar_stack(np.asarray(grid[key], dtype="f4"), x, y, x_orig, y_orig)
        for key in keys
        if key in grid
    }


def centerline_offsets(part: pd.DataFrame, obj: SelectedObject, nlev: int, alpha_deg: float) -> dict[str, np.ndarray]:
    rows = part.sort_values("depth_index").copy()
    lon_col = "center_lon_refined" if "center_lon_refined" in rows.columns else "center_lon"
    lat_col = "center_lat_refined" if "center_lat_refined" in rows.columns else "center_lat"
    surface = rows.iloc[0]
    mx, my = meters_per_degree(float(surface[lat_col]))
    radius_m = max(float(obj.radius_km) * 1000.0, 1.0)
    x = np.full(nlev, np.nan, dtype="f8")
    y = np.full(nlev, np.nan, dtype="f8")
    for _, row in rows.iterrows():
        k = int(row["depth_index"])
        if k < 0 or k >= nlev:
            continue
        if not np.isfinite(row[lon_col]) or not np.isfinite(row[lat_col]):
            continue
        x[k] = lon_delta_deg(float(row[lon_col]), float(surface[lon_col])) * mx / radius_m
        y[k] = (float(row[lat_col]) - float(surface[lat_col])) * my / radius_m
    x_rot, y_rot = rotate_xy(x, y, alpha_deg)
    return {"x_rot_r": x_rot, "y_rot_r": y_rot}


def density_tilt_decomposition(rho_rot: np.ndarray, x_axis: np.ndarray, x_centerline_r: np.ndarray) -> dict[str, np.ndarray]:
    rho = np.asarray(rho_rot, dtype="f8")
    rho_layer_centered = shift_stack_x(rho, x_axis, x_centerline_r)
    rho_intrinsic_even = 0.5 * (rho_layer_centered + np.flip(rho_layer_centered, axis=2))
    rho_intrinsic_odd = 0.5 * (rho_layer_centered - np.flip(rho_layer_centered, axis=2))
    rho_surface_odd = 0.5 * (rho - np.flip(rho, axis=2))
    dxr = float(np.nanmean(np.diff(x_axis)))
    drho_even_dx = np.empty_like(rho_intrinsic_even)
    for k in range(rho_intrinsic_even.shape[0]):
        drho_even_dx[k] = np.gradient(rho_intrinsic_even[k], dxr, axis=1)
    rho_odd_theory = -x_centerline_r[:, None, None] * drho_even_dx
    rho_odd_theory = np.where(np.isfinite(x_centerline_r)[:, None, None], rho_odd_theory, np.nan)
    return {
        "rho": rho,
        "rho_layer_centered": rho_layer_centered,
        "rho_intrinsic_even": rho_intrinsic_even,
        "rho_intrinsic_odd": rho_intrinsic_odd,
        "rho_surface_odd": rho_surface_odd,
        "rho_odd_theory": rho_odd_theory,
        "drho_even_dx_per_r": drho_even_dx,
    }


def shift_stack_x(stack: np.ndarray, x_axis: np.ndarray, x_centerline_r: np.ndarray) -> np.ndarray:
    out = np.full_like(stack, np.nan, dtype="f8")
    for k in range(stack.shape[0]):
        xc = float(x_centerline_r[k]) if k < len(x_centerline_r) else float("nan")
        if not np.isfinite(xc):
            continue
        query_x = np.asarray(x_axis, dtype="f8") + xc
        for j in range(stack.shape[1]):
            values = np.asarray(stack[k, j], dtype="f8")
            valid = np.isfinite(values) & np.isfinite(x_axis)
            if np.count_nonzero(valid) >= 2:
                out[k, j] = np.interp(query_x, x_axis[valid], values[valid], left=np.nan, right=np.nan)
    return out


def density_metrics(density: dict[str, np.ndarray]) -> dict[str, float]:
    actual = np.asarray(density["rho_surface_odd"], dtype="f8")
    theory = np.asarray(density["rho_odd_theory"], dtype="f8")
    intrinsic_odd = np.asarray(density["rho_intrinsic_odd"], dtype="f8")
    actual_q95 = q95_abs(actual)
    theory_q95 = q95_abs(theory)
    return {
        "rho_surface_odd_corr_tilt_theory": float(spatial_corr(actual, theory)),
        "rho_surface_odd_q95": float(actual_q95),
        "rho_odd_q95_theory": float(theory_q95),
        "rho_odd_q95_ratio_theory_over_surface_odd": float(theory_q95 / actual_q95) if actual_q95 > 0 else float("nan"),
        "rho_intrinsic_odd_q95": float(q95_abs(intrinsic_odd)),
    }


def plot_density_decomposition(
    grid: dict[str, object],
    density: dict[str, np.ndarray],
    centerline: dict[str, np.ndarray],
    payload: dict[str, object],
    png_path: Path,
) -> None:
    depth = np.asarray(grid["depth_m"], dtype="f8")
    x = np.asarray(grid["x_over_r"], dtype="f8")
    mid = int(len(np.asarray(grid["y_over_r"])) // 2)
    fields = [
        ("rho prime", density["rho"][:, mid, :]),
        ("layer-centered intrinsic even", density["rho_intrinsic_even"][:, mid, :]),
        ("surface-fixed actual odd", density["rho_surface_odd"][:, mid, :]),
        ("first-order tilt prediction", density["rho_odd_theory"][:, mid, :]),
    ]
    vmax = max(q95_abs(field) for _, field in fields)
    vmax = max(vmax, 1.0e-6)
    image, draw = make_canvas(
        2380,
        720,
        f"Single coherent OFES density tilt test | {payload['hua_object_id']} | alpha={payload['alpha_deg']:.1f} deg",
        "Rotated frame: vertical centerline tilt points to +x_rot. Units: potential density anomaly relative to rebuild background.",
    )
    margin_l, margin_t, panel_w, panel_h, gap = 90, 120, 500, 480, 42
    for col, (title, field) in enumerate(fields):
        x0 = margin_l + col * (panel_w + gap)
        draw_field_panel(image, draw, field, x, depth, (x0, margin_t, panel_w, panel_h), -vmax, vmax, "x_rot/R", "depth (m)", title)
        draw_centerline(draw, x, depth, centerline["x_rot_r"], (x0, margin_t, panel_w, panel_h))
    draw_colorbar(image, draw, image.width - 85, margin_t, 24, panel_h, -vmax, vmax, "rho")
    draw.text((margin_l, 78), f"surface-odd corr(actual,theory)={payload['rho_surface_odd_corr_tilt_theory']:.3f}; q95 odd={payload['rho_surface_odd_q95']:.3g}; q95 theory={payload['rho_odd_q95_theory']:.3g}; intrinsic-odd q95={payload['rho_intrinsic_odd_q95']:.3g}", fill=(55, 65, 80), font=ImageFont.load_default())
    png_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(png_path)


def plot_w_terms(grid: dict[str, object], rotated: dict[str, np.ndarray], payload: dict[str, object], png_path: Path) -> None:
    depth = np.asarray(grid["depth_m"], dtype="f8")
    x = np.asarray(grid["x_over_r"], dtype="f8")
    mid = int(len(np.asarray(grid["y_over_r"])) // 2)
    fields = [
        ("term1 = c_rel dot grad eta", rotated["term1_m_s"][:, mid, :] * 1.0e6),
        ("term2 = -u_rel dot grad eta", rotated["term2_m_s"][:, mid, :] * 1.0e6),
        ("term1 + term2", rotated["rebuild_w_m_s"][:, mid, :] * 1.0e6),
        ("OFES native w", rotated["ofes_w_native_m_s"][:, mid, :] * 1.0e6),
    ]
    vmax = max(q95_abs(field) for _, field in fields)
    vmax = max(vmax, 1.0e-6)
    image, draw = make_canvas(
        2380,
        720,
        f"Single coherent OFES W tilt test | {payload['hua_object_id']} | {payload['date']} | {payload['polarity']}",
        "x_rot-depth section at y_rot/R=0. Units: 10^-6 m/s.",
    )
    margin_l, margin_t, panel_w, panel_h, gap = 90, 120, 500, 480, 42
    for col, (title, field) in enumerate(fields):
        x0 = margin_l + col * (panel_w + gap)
        draw_field_panel(image, draw, field, x, depth, (x0, margin_t, panel_w, panel_h), -vmax, vmax, "x_rot/R", "depth (m)", title)
    draw_colorbar(image, draw, image.width - 85, margin_t, 24, panel_h, -vmax, vmax, "10^-6 m/s")
    png_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(png_path)


def plot_450m_slice(
    grid: dict[str, object],
    density: dict[str, np.ndarray],
    rotated: dict[str, np.ndarray],
    payload: dict[str, object],
    png_path: Path,
) -> None:
    depth = np.asarray(grid["depth_m"], dtype="f8")
    x = np.asarray(grid["x_over_r"], dtype="f8")
    y = np.asarray(grid["y_over_r"], dtype="f8")
    idx = int(np.nanargmin(np.abs(depth - 450.0)))
    fields = [
        ("surface actual odd", density["rho_surface_odd"][idx]),
        ("tilt prediction", density["rho_odd_theory"][idx]),
        ("rebuild W", rotated["rebuild_w_m_s"][idx] * 1.0e6),
        ("native W", rotated["ofes_w_native_m_s"][idx] * 1.0e6),
    ]
    vmax_rho = max(q95_abs(fields[0][1]), q95_abs(fields[1][1]), 1.0e-6)
    vmax_w = max(q95_abs(fields[2][1]), q95_abs(fields[3][1]), 1.0e-6)
    image, draw = make_canvas(
        1420,
        810,
        f"Single coherent OFES 450 m slice | {payload['hua_object_id']} | z={depth[idx]:.0f} m",
        "Rotated original object, not a composite. Circles mark 1R.",
    )
    margin_l, margin_t, panel, gap = 85, 120, 280, 42
    for col, (title, field) in enumerate(fields):
        x0 = margin_l + col * (panel + gap)
        vmax = vmax_rho if col < 2 else vmax_w
        unit = "rho" if col < 2 else "10^-6 m/s"
        draw_field_panel(image, draw, field, x, y, (x0, margin_t, panel, panel), -vmax, vmax, "x_rot/R", "y_rot/R", title, circle=True)
        draw.text((x0 + 70, margin_t + panel + 32), unit, fill=(55, 65, 80), font=ImageFont.load_default())
    png_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(png_path)


def draw_centerline(
    draw: ImageDraw.ImageDraw,
    x_axis: np.ndarray,
    depth: np.ndarray,
    x_centerline: np.ndarray,
    box: tuple[int, int, int, int],
) -> None:
    x0, y0, w, h = box
    valid = np.isfinite(x_centerline) & np.isfinite(depth)
    if np.count_nonzero(valid) < 2:
        return
    x_min, x_max = float(x_axis[0]), float(x_axis[-1])
    y_min, y_max = float(depth[0]), float(depth[-1])
    pts = []
    for x_r, z in zip(x_centerline[valid], depth[valid]):
        px = x0 + int((float(x_r) - x_min) / (x_max - x_min) * w)
        py = y0 + int((float(z) - y_min) / (y_max - y_min) * h)
        pts.append((px, py))
    if len(pts) >= 2:
        draw.line(pts, fill=(0, 0, 0), width=2)


def make_canvas(width: int, height: int, title: str, subtitle: str) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.text((80, 32), title, fill=(20, 30, 45), font=font)
    draw.text((80, 58), subtitle, fill=(65, 75, 90), font=font)
    return image, draw


def safe_object_token(obj: SelectedObject) -> str:
    return f"{obj.date.replace('-', '')}_{obj.hua_object_id}"


if __name__ == "__main__":
    main()
