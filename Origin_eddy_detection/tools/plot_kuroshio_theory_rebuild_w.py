from __future__ import annotations

import argparse
import json
from datetime import date, datetime, timedelta
from pathlib import Path

import netCDF4
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import gaussian_filter, gaussian_filter1d


RHO0 = 1027.0
ALPHA = 0.20
BETA = 0.80
SECONDS_PER_DAY = 86400.0


def parse_day(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def num2date_index(ds: netCDF4.Dataset, day: date) -> int:
    tvar = ds.variables["time"]
    times = netCDF4.num2date(tvar[:], units=tvar.units, calendar=getattr(tvar, "calendar", "standard"))
    for idx, item in enumerate(times):
        if date(int(item.year), int(item.month), int(item.day)) == day:
            return idx
    raise KeyError(f"{day} not found in {ds.filepath()}")


def filled(var) -> np.ndarray:
    arr = np.asarray(var[:], dtype="f8")
    if np.ma.isMaskedArray(var[:]):
        arr = np.asarray(var[:].filled(np.nan), dtype="f8")
    arr[np.abs(arr) > 1.0e20] = np.nan
    return arr


def linear_density(theta: np.ndarray, salt: np.ndarray) -> np.ndarray:
    return RHO0 - ALPHA * (theta - 10.0) + BETA * (salt - 35.0)


def lonlat_to_xy_m(lon: np.ndarray, lat: np.ndarray, lon0: float, lat0: float) -> tuple[np.ndarray, np.ndarray]:
    x = (lon - lon0) * 111_000.0 * np.cos(np.deg2rad(lat0))
    y = (lat - lat0) * 111_000.0
    return x, y


def nan_gaussian(field: np.ndarray, sigma_y: float, sigma_x: float) -> np.ndarray:
    out = np.empty_like(field, dtype="f8")
    for k in range(field.shape[0]):
        a = field[k]
        valid = np.isfinite(a).astype("f8")
        vals = np.where(np.isfinite(a), a, 0.0)
        num = gaussian_filter(vals, sigma=(sigma_y, sigma_x), mode="nearest")
        den = gaussian_filter(valid, sigma=(sigma_y, sigma_x), mode="nearest")
        out[k] = np.where(den > 1.0e-6, num / den, np.nan)
    return out


def smooth_profile(values: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    valid = np.isfinite(values).astype("f8")
    vals = np.where(np.isfinite(values), values, 0.0)
    num = gaussian_filter1d(vals, sigma=sigma, mode="nearest")
    den = gaussian_filter1d(valid, sigma=sigma, mode="nearest")
    return np.where(den > 1.0e-6, num / den, np.nan)


def centers_for_track(layers: pd.DataFrame, objects: pd.DataFrame, track_id: int, day: date) -> pd.DataFrame:
    obj_ids = objects[(objects["track3d_id"].eq(track_id)) & (objects["date"].astype(str).eq(str(day)))]["eddy3d_object_id"]
    return layers[layers["eddy3d_object_id"].isin(obj_ids)].copy()


def layerwise_c_ms(layers: pd.DataFrame, objects: pd.DataFrame, track_id: int, target_day: date, lat_ref: float) -> pd.DataFrame:
    current = centers_for_track(layers, objects, track_id, target_day)
    prev = centers_for_track(layers, objects, track_id, target_day - timedelta(days=1))
    nxt = centers_for_track(layers, objects, track_id, target_day + timedelta(days=1))
    rows = []
    for _, row in current.iterrows():
        didx = int(row["depth_index"])
        candidates = []
        pp = prev[prev["depth_index"].eq(didx)]
        nn = nxt[nxt["depth_index"].eq(didx)]
        if not pp.empty and not nn.empty:
            p = pp.iloc[0]
            n = nn.iloc[0]
            dt = 2.0 * SECONDS_PER_DAY
            candidates.append((p, n, dt))
        elif not nn.empty:
            n = nn.iloc[0]
            candidates.append((row, n, SECONDS_PER_DAY))
        elif not pp.empty:
            p = pp.iloc[0]
            candidates.append((p, row, SECONDS_PER_DAY))
        if not candidates:
            cx = cy = np.nan
        else:
            a, b, dt = candidates[0]
            dx = (float(b["longitude"]) - float(a["longitude"])) * 111_000.0 * np.cos(np.deg2rad(lat_ref))
            dy = (float(b["latitude"]) - float(a["latitude"])) * 111_000.0
            cx = dx / dt
            cy = dy / dt
        rows.append({"depth_index": didx, "c_x_m_s": cx, "c_y_m_s": cy})
    return pd.DataFrame(rows)


def interp_section(field: np.ndarray, x_km: np.ndarray, y_km: np.ndarray, theta: float, s_km: np.ndarray) -> np.ndarray:
    from scipy.interpolate import RegularGridInterpolator

    out = np.full((field.shape[0], s_km.size), np.nan, dtype="f8")
    xs = s_km * np.cos(theta)
    ys = s_km * np.sin(theta)
    points = np.column_stack([ys, xs])
    for k in range(field.shape[0]):
        interp = RegularGridInterpolator((y_km, x_km), field[k], bounds_error=False, fill_value=np.nan)
        out[k] = interp(points)
    return out


def font(size: int) -> ImageFont.ImageFont:
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            pass
    return ImageFont.load_default()


def colorize(data: np.ndarray, vmin: float, vmax: float) -> Image.Image:
    arr = np.asarray(data, dtype="f8")
    norm = np.clip((arr - vmin) / max(vmax - vmin, 1.0e-12), 0.0, 1.0)
    valid = np.isfinite(arr)
    r = np.empty(arr.shape, dtype="u1")
    g = np.empty(arr.shape, dtype="u1")
    b = np.empty(arr.shape, dtype="u1")
    low = norm < 0.5
    t1 = np.where(low, norm / 0.5, 1.0)
    t2 = np.where(low, 0.0, (norm - 0.5) / 0.5)
    r[:] = np.where(low, 50 + 205 * t1, 255).astype("u1")
    g[:] = np.where(low, 90 + 165 * t1, 255 - 210 * t2).astype("u1")
    b[:] = np.where(low, 190 + 65 * t1, 255 - 225 * t2).astype("u1")
    r[~valid] = 245
    g[~valid] = 245
    b[~valid] = 245
    return Image.fromarray(np.dstack([r, g, b]), mode="RGB")


def draw_colorbar(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], vmin: float, vmax: float, label: str, small) -> None:
    x0, y0, x1, y1 = box
    vals = np.linspace(vmax, vmin, max(2, y1 - y0))[:, None]
    bar = colorize(vals, vmin, vmax).resize((x1 - x0, y1 - y0))
    draw._image.paste(bar, (x0, y0))
    draw.rectangle(box, outline=(60, 60, 60))
    draw.text((x1 + 5, y0 - 5), f"{vmax:.2g}", fill=(20, 20, 20), font=small)
    draw.text((x1 + 5, y1 - 10), f"{vmin:.2g}", fill=(20, 20, 20), font=small)
    draw.text((x0 - 5, y1 + 5), label, fill=(20, 20, 20), font=small)


def paste_field(canvas: Image.Image, draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], data: np.ndarray, vmin: float, vmax: float, title: str, small) -> None:
    x0, y0, x1, y1 = box
    img = colorize(data, vmin, vmax).resize((x1 - x0, y1 - y0), Image.Resampling.BILINEAR)
    canvas.paste(img, (x0, y0))
    draw.rectangle(box, outline=(40, 40, 40), width=1)
    draw.text((x0, y0 - 22), title, fill=(20, 20, 20), font=small)


def plot_sections(out: Path, s_over_r: np.ndarray, depth: np.ndarray, term1: np.ndarray, term2: np.ndarray, total: np.ndarray, title: str) -> None:
    fields = [("term1", term1), ("term2", term2), ("term1 + term2", total)]
    lim = np.nanpercentile(np.abs(np.concatenate([f[np.isfinite(f)] for _, f in fields])), 97)
    lim = max(float(lim), 1.0e-9) * 1.0e6
    canvas = Image.new("RGB", (1500, 560), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = font(22)
    small = font(15)
    draw.text((30, 18), title, fill=(20, 24, 32), font=title_font)
    boxes = [(60, 95, 455, 485), (540, 95, 935, 485), (1020, 95, 1415, 485)]
    for box, (name, data) in zip(boxes, fields):
        paste_field(canvas, draw, box, data * 1.0e6, -lim, lim, name, small)
        x0, y0, x1, y1 = box
        zero = x0 + int((0.0 - float(s_over_r[0])) / (float(s_over_r[-1]) - float(s_over_r[0])) * (x1 - x0))
        draw.line((zero, y0, zero, y1), fill=(0, 0, 0), width=1)
        draw.text((x0, y1 + 8), "section distance / R", fill=(20, 20, 20), font=small)
        draw.text((x0 - 45, y0), f"{float(depth[0]):.0f} m", fill=(20, 20, 20), font=small)
        draw.text((x0 - 55, y1 - 15), f"{float(depth[-1]):.0f} m", fill=(20, 20, 20), font=small)
    draw_colorbar(draw, (1440, 115, 1465, 455), -lim, lim, "1e-6 m/s", small)
    canvas.save(out)


def plot_slices(out: Path, x_over_r: np.ndarray, y_over_r: np.ndarray, depth: np.ndarray, fields: dict[str, np.ndarray], title: str) -> None:
    target_depths = [100.0, 300.0, 500.0, 800.0]
    idxs = [int(np.nanargmin(np.abs(depth - d))) for d in target_depths]
    names = ["term1", "term2", "term1 + term2"]
    vals = [fields[n] for n in names]
    finite = np.concatenate([v[i][np.isfinite(v[i])] for v in vals for i in idxs])
    lim = max(float(np.nanpercentile(np.abs(finite), 97)), 1.0e-9) * 1.0e6
    canvas = Image.new("RGB", (1350, 1520), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = font(22)
    small = font(14)
    draw.text((30, 18), title, fill=(20, 24, 32), font=title_font)
    cell_w, cell_h = 355, 310
    x0_base, y0_base = 65, 95
    for r, idx in enumerate(idxs):
        for c, name in enumerate(names):
            x0 = x0_base + c * 405
            y0 = y0_base + r * 345
            box = (x0, y0, x0 + cell_w, y0 + cell_h)
            data = fields[name][idx] * 1.0e6
            paste_field(canvas, draw, box, data, -lim, lim, f"{name}, z={depth[idx]:.0f} m", small)
            cx = (box[0] + box[2]) // 2
            cy = (box[1] + box[3]) // 2
            draw.ellipse((cx - 3, cy - 3, cx + 3, cy + 3), fill=(0, 0, 0))
            for rr in (1.0, 2.0):
                rad_x = int(rr / max(abs(float(x_over_r[0])), abs(float(x_over_r[-1]))) * (cell_w / 2))
                rad_y = int(rr / max(abs(float(y_over_r[0])), abs(float(y_over_r[-1]))) * (cell_h / 2))
                draw.ellipse((cx - rad_x, cy - rad_y, cx + rad_x, cy + rad_y), outline=(0, 0, 0), width=1)
            if r == len(idxs) - 1:
                draw.text((x0 + 120, y0 + cell_h + 8), "x / R", fill=(20, 20, 20), font=small)
            if c == 0:
                draw.text((x0 - 42, y0 + 135), "y / R", fill=(20, 20, 20), font=small)
    draw_colorbar(draw, (1285, 150, 1310, 1370), -lim, lim, "1e-6 m/s", small)
    canvas.save(out)


def run(args: argparse.Namespace) -> None:
    result_root = Path(args.result_root)
    data_root = Path(args.data_root)
    filter_root = data_root / "FILTER"
    out_dir = Path(args.output_dir)
    fig_dir = out_dir / "figures"
    grid_dir = out_dir / "grids"
    fig_dir.mkdir(parents=True, exist_ok=True)
    grid_dir.mkdir(parents=True, exist_ok=True)

    shape = pd.read_parquet(result_root / "shape_classification_1993_2022_hua_b3_start2_life30" / "shape_tracks.parquet")
    if args.track_id is None:
        candidates = shape[shape["shape_class"].isin(["coherent", "mixed"])].sort_values(["shape_class", "mono_ratio"], ascending=[True, False])
        track_id = int(candidates.iloc[0]["track3d_id"])
    else:
        track_id = int(args.track_id)
    objects = pd.read_parquet(result_root / "catalog" / "vertical_objects.parquet")
    layers = pd.read_parquet(result_root / "catalog" / "layer_observations.parquet")
    track_objects = objects[objects["track3d_id"].eq(track_id)].sort_values("date").copy()
    target_day = parse_day(args.date) if args.date else parse_day(str(track_objects.iloc[len(track_objects) // 2]["date"]))
    obj = track_objects[track_objects["date"].astype(str).eq(str(target_day))].iloc[0]
    object_id = int(obj["eddy3d_object_id"])
    day_layers = layers[layers["eddy3d_object_id"].eq(object_id)].sort_values("depth_index").copy()
    lon0 = float(day_layers.iloc[0]["longitude"])
    lat0 = float(day_layers.iloc[0]["latitude"])
    radius_m = float(obj["mean_radius_m"])

    year = target_day.year
    raw = netCDF4.Dataset(data_root / f"global_phy_{year}.nc")
    fil = netCDF4.Dataset(filter_root / f"global_phy_{year}.nc")
    ti = num2date_index(raw, target_day)
    lon = np.asarray(raw.variables["longitude"][:], dtype="f8")
    lat = np.asarray(raw.variables["latitude"][:], dtype="f8")
    depth = np.asarray(raw.variables["depth"][:], dtype="f8")
    x_all_m, y_all_m = lonlat_to_xy_m(lon, lat, lon0, lat0)
    xmask = np.abs(x_all_m) <= args.window_r * radius_m
    ymask = np.abs(y_all_m) <= args.window_r * radius_m
    xi = np.where(xmask)[0]
    yi = np.where(ymask)[0]
    xs = slice(int(xi.min()), int(xi.max()) + 1)
    ys = slice(int(yi.min()), int(yi.max()) + 1)
    local_lon = lon[xs]
    local_lat = lat[ys]
    x_m, y_m = lonlat_to_xy_m(local_lon, local_lat, lon0, lat0)
    xx_m, yy_m = np.meshgrid(x_m, y_m)
    rr = np.hypot(xx_m, yy_m) / radius_m

    theta = filled(raw.variables["thetao_glor"][ti, :, ys, xs])
    salt = filled(raw.variables["so_glor"][ti, :, ys, xs])
    rho = linear_density(theta, salt)
    u = filled(fil.variables["uo_glor"][ti, :, ys, xs])
    v = filled(fil.variables["vo_glor"][ti, :, ys, xs])
    raw.close()
    fil.close()

    bg_mask = (rr >= args.bg_inner_r) & (rr <= args.bg_outer_r)
    rho_bg = np.array([np.nanmedian(rho[k][bg_mask]) for k in range(rho.shape[0])], dtype="f8")
    rho_bg = smooth_profile(rho_bg, args.rho_bg_smooth_sigma_layers)
    rho_z = np.gradient(rho_bg, depth)
    bad = (~np.isfinite(rho_z)) | (np.abs(rho_z) < args.rho_z_min)
    rho_prime = rho - rho_bg[:, None, None]
    eta = -rho_prime / rho_z[:, None, None]
    eta[bad] = np.nan
    eta = np.clip(eta, -args.eta_cap_m, args.eta_cap_m)
    dx = float(np.nanmedian(np.abs(np.diff(x_m))))
    dy = float(np.nanmedian(np.abs(np.diff(y_m))))
    sigma_cells = max(0.0, args.eta_smooth_r * radius_m / max(dx, dy))
    if sigma_cells > 0:
        eta = nan_gaussian(eta, sigma_cells, sigma_cells)
    grad_y, grad_x = np.gradient(eta, y_m, x_m, axis=(1, 2))

    u_bg = np.array([np.nanmedian(u[k][bg_mask]) for k in range(u.shape[0])], dtype="f8")
    v_bg = np.array([np.nanmedian(v[k][bg_mask]) for k in range(v.shape[0])], dtype="f8")
    u_rel = u - u_bg[:, None, None]
    v_rel = v - v_bg[:, None, None]

    ctab = layerwise_c_ms(layers, objects, track_id, target_day, lat0)
    cx = np.full(depth.shape, np.nan, dtype="f8")
    cy = np.full(depth.shape, np.nan, dtype="f8")
    for _, row in ctab.iterrows():
        idx = int(row["depth_index"])
        if 0 <= idx < cx.size:
            cx[idx] = row["c_x_m_s"]
            cy[idx] = row["c_y_m_s"]
    cx = smooth_profile(cx, args.c_smooth_sigma_layers)
    cy = smooth_profile(cy, args.c_smooth_sigma_layers)
    c_rel_x = cx - u_bg
    c_rel_y = cy - v_bg

    term1 = c_rel_x[:, None, None] * grad_x + c_rel_y[:, None, None] * grad_y
    term2 = -(u_rel * grad_x + v_rel * grad_y)
    total = term1 + term2

    layer_coords = day_layers[["depth_index", "depth_m", "longitude", "latitude", "radius_m", "polarity"]].copy()
    layer_coords["x_from_surface_km"] = (layer_coords["longitude"] - lon0) * 111.0 * np.cos(np.deg2rad(lat0))
    layer_coords["y_from_surface_km"] = (layer_coords["latitude"] - lat0) * 111.0
    deepest = layer_coords.iloc[-1]
    theta_axis = float(np.arctan2(deepest["y_from_surface_km"], deepest["x_from_surface_km"]))
    if not np.isfinite(theta_axis) or np.hypot(deepest["x_from_surface_km"], deepest["y_from_surface_km"]) < 1:
        theta_axis = float(np.arctan2(np.nanmedian(cy), np.nanmedian(cx)))
    s_km = np.linspace(-args.section_r, args.section_r, args.section_points) * radius_m / 1000.0
    x_km = x_m / 1000.0
    y_km = y_m / 1000.0
    sec1 = interp_section(term1, x_km, y_km, theta_axis, s_km)
    sec2 = interp_section(term2, x_km, y_km, theta_axis, s_km)
    sect = sec1 + sec2

    stem = f"track{track_id}_{target_day:%Y%m%d}_obj{object_id}"
    np.savez_compressed(
        grid_dir / f"theory_rebuild_w_{stem}.npz",
        x_over_r=x_m / radius_m,
        y_over_r=y_m / radius_m,
        depth=depth,
        term1_m_s=term1,
        term2_m_s=term2,
        rebuild_w_m_s=total,
        eta_rho_m=eta,
        rho_bg=rho_bg,
        rho_z=rho_z,
        c_rel_x_m_s=c_rel_x,
        c_rel_y_m_s=c_rel_y,
        u_bg_m_s=u_bg,
        v_bg_m_s=v_bg,
        section_s_over_r=s_km * 1000.0 / radius_m,
        section_term1_m_s=sec1,
        section_term2_m_s=sec2,
        section_rebuild_w_m_s=sect,
    )
    meta = {
        "track3d_id": track_id,
        "eddy3d_object_id": object_id,
        "date": str(target_day),
        "shape_class": str(shape[shape["track3d_id"].eq(track_id)].iloc[0]["shape_class"]),
        "polarity": str(obj["polarity"]),
        "surface_center_lon": lon0,
        "surface_center_lat": lat0,
        "radius_m": radius_m,
        "density_method": f"linear EOS rho={RHO0}- {ALPHA}(theta-10)+{BETA}(salt-35)",
        "velocity_source": str(filter_root / f"global_phy_{year}.nc"),
        "density_source": str(data_root / f"global_phy_{year}.nc"),
        "section_axis_rad": theta_axis,
        "eta_cap_m": args.eta_cap_m,
        "rho_z_min": args.rho_z_min,
        "q95_abs_term1_1e6_m_s": float(np.nanpercentile(np.abs(term1), 95) * 1e6),
        "q95_abs_term2_1e6_m_s": float(np.nanpercentile(np.abs(term2), 95) * 1e6),
        "q95_abs_rebuild_1e6_m_s": float(np.nanpercentile(np.abs(total), 95) * 1e6),
    }
    (grid_dir / f"theory_rebuild_w_{stem}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    title = f"Kuroshio theoretical rebuild W | {stem} | {meta['shape_class']} {meta['polarity']}"
    plot_sections(fig_dir / f"theory_rebuild_w_section_{stem}.png", s_km * 1000.0 / radius_m, depth, sec1, sec2, sect, title)
    plot_slices(
        fig_dir / f"theory_rebuild_w_slices_{stem}.png",
        x_m / radius_m,
        y_m / radius_m,
        depth,
        {"term1": term1, "term2": term2, "term1 + term2": total},
        title,
    )
    print(json.dumps(meta, indent=2))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Plot theoretical rebuild W for a Kuroshio velocity-streamline object-day.")
    p.add_argument("--result-root", default=r"E:\DATA\01_Eddy_correspond\03_Original_detection\TEST\Kuroshio_2018_streamline_daybatch_mat_cpu")
    p.add_argument("--data-root", default=r"F:\Global Ocean Ensemble Physics Reanalysis Kuroshio Current")
    p.add_argument("--output-dir", default=r"E:\DATA\01_Eddy_correspond\03_Original_detection\TEST\Kuroshio_2018_streamline_daybatch_mat_cpu\diagnostics\theory_rebuild_w")
    p.add_argument("--track-id", type=int, default=None)
    p.add_argument("--date", default="")
    p.add_argument("--window-r", type=float, default=3.0)
    p.add_argument("--bg-inner-r", type=float, default=2.0)
    p.add_argument("--bg-outer-r", type=float, default=3.0)
    p.add_argument("--rho-bg-smooth-sigma-layers", type=float, default=2.0)
    p.add_argument("--rho-z-min", type=float, default=2.0e-5)
    p.add_argument("--eta-cap-m", type=float, default=500.0)
    p.add_argument("--eta-smooth-r", type=float, default=0.5)
    p.add_argument("--c-smooth-sigma-layers", type=float, default=2.0)
    p.add_argument("--section-r", type=float, default=2.5)
    p.add_argument("--section-points", type=int, default=161)
    return p


if __name__ == "__main__":
    run(build_parser().parse_args())
