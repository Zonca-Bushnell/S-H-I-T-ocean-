from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


DEFAULT_RESULT_ROOT = Path(r"E:\DATA\01_Eddy_correspond\02_OFES\origin_ssh_effective_contour_primary_rawnc4_jan01_jan19_parallel")
DEFAULT_FILTER_ROOT = Path(r"E:\DATA\01_Eddy_correspond\02_OFES\origin_compatible_filter")


def main() -> None:
    args = parse_args()
    out_dir = args.result_root / "figures" / "ssh_vector_overview"
    out_dir.mkdir(parents=True, exist_ok=True)
    lon, lat, ssh, u, v, attrs = read_surface_fields(args.filter_root, args.day)
    centers, structures, table_root = read_detection_tables(args.result_root, args.day)
    layer_suffix = "" if str(args.catalog_layer) == "all" else f"_{args.catalog_layer}"
    jet_suffix = "_no_jet_flag" if args.exclude_jet_flagged else ""
    transient_suffix = "_no_transient" if args.exclude_transient else ""
    filter_suffix = f"{jet_suffix}{transient_suffix}"

    products = []
    products.append(
        draw_overview(
            lon,
            lat,
            ssh,
            u,
            v,
            centers,
            structures,
            attrs,
            args.day,
            out_dir / f"ofes_ssh_primary_global_ssh_uv_vectors_{args.day.replace('-', '')}{layer_suffix}{filter_suffix}.png",
            bbox=(0.0, 360.0, -76.0, 76.0),
            title_region="global",
            vector_step=int(args.global_vector_step),
            catalog_layer=str(args.catalog_layer),
            exclude_jet_flagged=bool(args.exclude_jet_flagged),
            exclude_transient=bool(args.exclude_transient),
            run_tag=str(args.run_tag),
        )
    )
    products.append(
        draw_overview(
            lon,
            lat,
            ssh,
            u,
            v,
            centers,
            structures,
            attrs,
            args.day,
            out_dir / f"ofes_ssh_primary_{slug(str(args.regional_zoom_name))}_ssh_uv_vectors_{args.day.replace('-', '')}{layer_suffix}{filter_suffix}.png",
            bbox=tuple(float(value) for value in args.regional_zoom_bbox),
            title_region=str(args.regional_zoom_name),
            vector_step=int(args.regional_vector_step),
            catalog_layer=str(args.catalog_layer),
            exclude_jet_flagged=bool(args.exclude_jet_flagged),
            exclude_transient=bool(args.exclude_transient),
            run_tag=str(args.run_tag),
        )
    )
    products.append(
        draw_overview(
            lon,
            lat,
            ssh,
            u,
            v,
            centers,
            structures,
            attrs,
            args.day,
            out_dir / f"ofes_ssh_primary_{slug(str(args.ocean_zoom_name))}_ssh_uv_vectors_{args.day.replace('-', '')}{layer_suffix}{filter_suffix}.png",
            bbox=tuple(float(value) for value in args.ocean_zoom_bbox),
            title_region=args.ocean_zoom_name,
            vector_step=int(args.ocean_vector_step),
            catalog_layer=str(args.catalog_layer),
            exclude_jet_flagged=bool(args.exclude_jet_flagged),
            exclude_transient=bool(args.exclude_transient),
            run_tag=str(args.run_tag),
        )
    )
    manifest = {
        "day": args.day,
        "result_root": str(args.result_root),
        "detection_table_root": str(table_root),
        "filter_root": str(args.filter_root),
        "background": "zos_glor SSH anomaly",
        "vector_fields": "surface uo_glor/vo_glor",
        "boundary_priority": "streamline_boundary_i/j when available, then ssh_contour_boundary_i/j, otherwise radius_km proxy",
        "catalog_layer": str(args.catalog_layer),
        "exclude_jet_flagged": bool(args.exclude_jet_flagged),
        "exclude_transient": bool(args.exclude_transient),
        "run_tag": str(args.run_tag),
        "products": products,
    }
    manifest_path = out_dir / f"ssh_vector_overview_manifest_{args.day.replace('-', '')}{layer_suffix}{filter_suffix}.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path), "products": products}, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot latest OFES SSH-primary eddy overview over SSH anomaly with velocity vectors.")
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--filter-root", type=Path, default=DEFAULT_FILTER_ROOT)
    parser.add_argument("--day", default="1991-01-10")
    parser.add_argument("--global-vector-step", type=int, default=120)
    parser.add_argument("--regional-vector-step", "--kuroshio-vector-step", dest="regional_vector_step", type=int, default=3)
    parser.add_argument("--regional-zoom-name", default="Kuroshio")
    parser.add_argument("--regional-zoom-bbox", type=float, nargs=4, metavar=("LON_MIN", "LON_MAX", "LAT_MIN", "LAT_MAX"), default=(120.0, 145.0, 20.0, 35.0))
    parser.add_argument("--ocean-vector-step", type=int, default=3)
    parser.add_argument("--ocean-zoom-name", default="North Pacific interior dense")
    parser.add_argument("--ocean-zoom-bbox", type=float, nargs=4, metavar=("LON_MIN", "LON_MAX", "LAT_MIN", "LAT_MAX"), default=(170.0, 210.0, 20.0, 40.0))
    parser.add_argument("--catalog-layer", choices=["all", "isolated", "jet_meander"], default="all")
    parser.add_argument("--exclude-jet-flagged", action="store_true", help="Hide rows with jet_meander_flag=True without changing the saved catalog.")
    parser.add_argument("--exclude-transient", action="store_true", help="Hide rows with persistence_class=transient without changing the saved catalog.")
    parser.add_argument("--run-tag", default="", help="Short diagnostic label included in figure titles and the manifest.")
    return parser.parse_args()


def slug(text: str) -> str:
    cleaned = []
    for ch in text.lower():
        if ch.isalnum():
            cleaned.append(ch)
        elif cleaned and cleaned[-1] != "_":
            cleaned.append("_")
    return "".join(cleaned).strip("_") or "region"


def read_surface_fields(filter_root: Path, day: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, str]]:
    ymd = day.replace("-", "")
    nc_path = filter_root / f"global_phy_{ymd}.nc"
    if not nc_path.exists():
        raise FileNotFoundError(nc_path)
    with h5py.File(nc_path, "r") as ds:
        lon = ds["longitude"][:].astype("f8")
        lat = ds["latitude"][:].astype("f8")
        ssh = ds["zos_glor"][0, :, :].astype("f4")
        u = ds["uo_glor"][0, 0, :, :].astype("f4")
        v = ds["vo_glor"][0, 0, :, :].astype("f4")
        attrs = {
            "zos_units": attr_text(ds["zos_glor"].attrs.get("units", "")),
            "u_units": attr_text(ds["uo_glor"].attrs.get("units", "")),
            "v_units": attr_text(ds["vo_glor"].attrs.get("units", "")),
            "science_tag": attr_text(ds.attrs.get("science_tag", "")),
            "source_file": str(nc_path),
        }
    for arr in [ssh, u, v]:
        arr[np.abs(arr) > 1.0e20] = np.nan
    return lon, lat, ssh, u, v, attrs


def attr_text(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def read_detection_tables(result_root: Path, day: str) -> tuple[pd.DataFrame, pd.DataFrame, Path]:
    ymd = day.replace("-", "")
    candidates = [
        result_root / "daily_runs" / ymd,
        result_root,
        result_root / "hua_b3_start2_detection",
    ]
    for root in candidates:
        centers_path = table_path(root, "centers_hua_style")
        structures_path = table_path(root, "structures_hua_style")
        if centers_path and structures_path:
            centers = read_table(centers_path)
            structures = read_table(structures_path)
            normalize_dates(centers)
            normalize_dates(structures)
            centers = centers[centers["date"].astype(str).eq(day)].copy() if "date" in centers.columns else centers
            structures = structures[structures["date"].astype(str).eq(day)].copy() if "date" in structures.columns else structures
            return centers, structures, root
    checked = "; ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Could not find centers_hua_style and structures_hua_style for {day}. Checked {checked}")


def table_path(root: Path, stem: str) -> Path | None:
    for suffix in [".parquet", ".csv"]:
        path = root / f"{stem}{suffix}"
        if path.exists():
            return path
    return None


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def normalize_dates(df: pd.DataFrame) -> None:
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")


def draw_overview(
    lon: np.ndarray,
    lat: np.ndarray,
    ssh: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    centers: pd.DataFrame,
    structures: pd.DataFrame,
    attrs: dict[str, str],
    day: str,
    png_path: Path,
    bbox: tuple[float, float, float, float],
    title_region: str,
    vector_step: int,
    catalog_layer: str,
    exclude_jet_flagged: bool = False,
    exclude_transient: bool = False,
    run_tag: str = "",
) -> dict[str, object]:
    lon_min, lon_max, lat_min, lat_max = bbox
    lon_idx = np.where((lon >= lon_min) & (lon <= lon_max))[0]
    lat_idx = np.where((lat >= lat_min) & (lat <= lat_max))[0]
    if lon_idx.size == 0 or lat_idx.size == 0:
        raise ValueError(f"bbox {bbox} outside lon/lat grid")
    i0, i1 = int(lon_idx[0]), int(lon_idx[-1]) + 1
    j0, j1 = int(lat_idx[0]), int(lat_idx[-1]) + 1
    lon_sub = lon[i0:i1]
    lat_sub = lat[j0:j1]
    ssh_sub = ssh[j0:j1, i0:i1]
    u_sub = u[j0:j1, i0:i1]
    v_sub = v[j0:j1, i0:i1]

    width, height = (2200, 1200) if title_region == "global" else (1840, 1160)
    map_box = (85, 110, width - 205, height - 115)
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    science_tag = str(attrs.get("science_tag") or "raw_minus_19day_mean_diagnostic")
    surface = surface_rows(
        centers,
        structures,
        bbox,
        catalog_layer=catalog_layer,
        exclude_jet_flagged=exclude_jet_flagged,
        exclude_transient=exclude_transient,
    )
    mode = infer_boundary_mode(surface)
    layer_suffix = f" | {catalog_layer}" if catalog_layer != "all" else ""
    tag_text = f" | {run_tag}" if run_tag else ""
    draw.text((50, 30), f"OFES {mode}{layer_suffix}{tag_text} | {title_region} | {day}", fill=(18, 24, 38), font=font(34))
    draw.text((50, 70), f"Input: {science_tag}. Background: SSH anomaly (zos_glor). Vectors: surface velocity anomaly. Edges: streamline/SSH contours when saved.", fill=(75, 85, 100), font=font(17))

    good = np.isfinite(ssh_sub)
    vmax = max(abs(float(np.nanpercentile(ssh_sub[good], 1.0))), abs(float(np.nanpercentile(ssh_sub[good], 99.0)))) if np.any(good) else 1.0
    vmax = max(vmax, 1.0e-6)
    bg = diverging_ramp(np.nan_to_num(ssh_sub, nan=0.0), -vmax, vmax)
    bg[~good] = np.array([35, 38, 45], dtype=np.uint8)
    bg = bg[::-1, :, :]
    map_img = Image.fromarray(bg, "RGB").resize((map_box[2] - map_box[0], map_box[3] - map_box[1]), Image.Resampling.BILINEAR)
    image.paste(map_img, map_box[:2])

    draw_grid(draw, map_box, bbox, title_region)
    draw_vectors(draw, lon_sub, lat_sub, u_sub, v_sub, map_box, bbox, vector_step)

    contour_count = draw_eddies(draw, surface, structures, lon, lat, map_box, bbox)
    draw_colorbar(image, draw, width - 150, map_box[1], 28, map_box[3] - map_box[1], -vmax, vmax, attrs.get("zos_units", ""))
    draw_legend(draw, map_box, surface, contour_count)

    png_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = png_path.with_suffix(".pdf")
    image.save(png_path)
    image.save(pdf_path, "PDF", resolution=180.0)
    return {
        "region": title_region,
        "bbox": {"lon_min": lon_min, "lon_max": lon_max, "lat_min": lat_min, "lat_max": lat_max},
        "surface_objects": int(len(surface)),
        "contour_boundary_count": int(contour_count),
        "radius_proxy_count": int(len(surface) - contour_count),
        "vector_step": int(vector_step),
        "science_tag": science_tag,
        "catalog_layer": catalog_layer,
        "exclude_jet_flagged": bool(exclude_jet_flagged),
        "exclude_transient": bool(exclude_transient),
        "run_tag": run_tag,
        "png": str(png_path),
        "pdf": str(pdf_path),
    }


def surface_rows(
    centers: pd.DataFrame,
    structures: pd.DataFrame,
    bbox: tuple[float, float, float, float],
    catalog_layer: str = "all",
    exclude_jet_flagged: bool = False,
    exclude_transient: bool = False,
) -> pd.DataFrame:
    if centers.empty:
        return centers.copy()
    surface = centers.copy()
    if "depth_index" in surface.columns:
        surface = surface[surface["depth_index"].astype(int).eq(0)].copy()
    if "hua_pass" in surface.columns:
        surface = surface[surface["hua_pass"].astype(bool)].copy()
    if "boundary_source" in surface.columns:
        source = surface["boundary_source"].astype(str)
        surface = surface[~source.str.contains("rejected", case=False, na=False)].copy()
    if "qc_class" in surface.columns:
        if catalog_layer == "isolated":
            surface = surface[surface["qc_class"].astype(str).eq("isolated_eddy_candidate")].copy()
        elif catalog_layer == "jet_meander":
            surface = surface[surface["qc_class"].astype(str).eq("jet_meander_candidate")].copy()
    if exclude_jet_flagged and "jet_meander_flag" in surface.columns:
        surface = surface[~surface["jet_meander_flag"].fillna(False).astype(bool)].copy()
    if exclude_transient and "persistence_class" in surface.columns:
        persistence_class = surface["persistence_class"].astype(str).str.lower()
        surface = surface[~persistence_class.eq("transient")].copy()
    lon_col = best_column(surface, ["center_lon_refined", "center_lon", "ssh_contour_center_lon", "seed_lon"])
    lat_col = best_column(surface, ["center_lat_refined", "center_lat", "ssh_contour_center_lat", "seed_lat"])
    if lon_col is None or lat_col is None:
        return surface.iloc[0:0].copy()
    surface["plot_lon"] = pd.to_numeric(surface[lon_col], errors="coerce")
    surface["plot_lat"] = pd.to_numeric(surface[lat_col], errors="coerce")
    if "radius_km" not in surface.columns and not structures.empty:
        surf_struct = structures.copy()
        if "depth_index" in surf_struct.columns:
            surf_struct = surf_struct[surf_struct["depth_index"].astype(int).eq(0)].copy()
        if "radius_km" in surf_struct.columns:
            surface = surface.merge(surf_struct[["date", "hua_object_id", "radius_km"]], on=["date", "hua_object_id"], how="left")
    lon_min, lon_max, lat_min, lat_max = bbox
    return surface[
        surface["plot_lon"].between(lon_min, lon_max)
        & surface["plot_lat"].between(lat_min, lat_max)
    ].copy()


def best_column(df: pd.DataFrame, names: list[str]) -> str | None:
    for name in names:
        if name in df.columns:
            return name
    return None


def draw_grid(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], bbox: tuple[float, float, float, float], region: str) -> None:
    x0, y0, x1, y1 = box
    lon_min, lon_max, lat_min, lat_max = bbox
    lon_step = 60 if region == "global" else 5
    lat_step = 30 if region == "global" else 5
    first_lon = math.ceil(lon_min / lon_step) * lon_step
    for glon in np.arange(first_lon, lon_max + 0.1, lon_step):
        x = px(float(glon), lon_min, lon_max, x0, x1)
        draw.line((x, y0, x, y1), fill=(45, 50, 60), width=1)
        draw.text((x - 18, y1 + 8), lon_label(float(glon)), fill=(35, 40, 50), font=font(15))
    first_lat = math.ceil(lat_min / lat_step) * lat_step
    for glat in np.arange(first_lat, lat_max + 0.1, lat_step):
        y = py(float(glat), lat_min, lat_max, y0, y1)
        draw.line((x0, y, x1, y), fill=(45, 50, 60), width=1)
        draw.text((x0 - 64, y - 9), lat_label(float(glat)), fill=(35, 40, 50), font=font(15))
    draw.rectangle(box, outline=(10, 15, 25), width=2)


def draw_vectors(
    draw: ImageDraw.ImageDraw,
    lon: np.ndarray,
    lat: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    box: tuple[int, int, int, int],
    bbox: tuple[float, float, float, float],
    step: int,
) -> None:
    x0, y0, x1, y1 = box
    lon_min, lon_max, lat_min, lat_max = bbox
    speed = np.hypot(u, v)
    scale = 14.0 / max(float(np.nanpercentile(speed[np.isfinite(speed)], 95.0)) if np.isfinite(speed).any() else 1.0, 1.0e-6)
    for jj in range(step // 2, len(lat), step):
        for ii in range(step // 2, len(lon), step):
            uu = float(u[jj, ii])
            vv = float(v[jj, ii])
            if not np.isfinite(uu) or not np.isfinite(vv):
                continue
            x = px(float(lon[ii]), lon_min, lon_max, x0, x1)
            y = py(float(lat[jj]), lat_min, lat_max, y0, y1)
            dx = uu * scale
            dy = -vv * scale
            draw.line((x, y, x + dx, y + dy), fill=(20, 20, 20), width=1)
            draw_arrowhead(draw, x, y, x + dx, y + dy)


def draw_arrowhead(draw: ImageDraw.ImageDraw, x0: float, y0: float, x1: float, y1: float) -> None:
    ang = math.atan2(y1 - y0, x1 - x0)
    size = 4.0
    pts = [
        (x1, y1),
        (x1 - size * math.cos(ang - 0.5), y1 - size * math.sin(ang - 0.5)),
        (x1 - size * math.cos(ang + 0.5), y1 - size * math.sin(ang + 0.5)),
    ]
    draw.polygon(pts, fill=(20, 20, 20))


def draw_eddies(
    draw: ImageDraw.ImageDraw,
    surface: pd.DataFrame,
    structures: pd.DataFrame,
    lon: np.ndarray,
    lat: np.ndarray,
    box: tuple[int, int, int, int],
    bbox: tuple[float, float, float, float],
) -> int:
    contour_count = 0
    colors = {"cyclonic": (37, 99, 235), "anticyclonic": (220, 38, 38)}
    x0, y0, x1, y1 = box
    lon_min, lon_max, lat_min, lat_max = bbox
    for _, row in surface.iterrows():
        color = colors.get(str(row.get("polarity", "")), (20, 20, 20))
        drew_contour = draw_saved_contour(draw, row, lon, lat, box, bbox, color)
        if drew_contour:
            contour_count += 1
        else:
            draw_radius_proxy(draw, row, box, bbox, color)
        cx = px(float(row["plot_lon"]), lon_min, lon_max, x0, x1)
        cy = py(float(row["plot_lat"]), lat_min, lat_max, y0, y1)
        draw.ellipse((cx - 4, cy - 4, cx + 4, cy + 4), fill=color, outline=(255, 255, 255), width=1)
    return contour_count


def draw_saved_contour(
    draw: ImageDraw.ImageDraw,
    row: pd.Series,
    lon: np.ndarray,
    lat: np.ndarray,
    box: tuple[int, int, int, int],
    bbox: tuple[float, float, float, float],
    color: tuple[int, int, int],
) -> bool:
    source = "ssh"
    if "streamline_boundary_i" in row and "streamline_boundary_j" in row:
        ii = parse_index_list(row.get("streamline_boundary_i", ""))
        jj = parse_index_list(row.get("streamline_boundary_j", ""))
        if ii.size >= 3 and jj.size >= 3:
            source = "streamline"
        else:
            ii = parse_index_list(row.get("ssh_contour_boundary_i", "")) if "ssh_contour_boundary_i" in row else np.asarray([], dtype=int)
            jj = parse_index_list(row.get("ssh_contour_boundary_j", "")) if "ssh_contour_boundary_j" in row else np.asarray([], dtype=int)
    elif "ssh_contour_boundary_i" in row and "ssh_contour_boundary_j" in row:
        ii = parse_index_list(row.get("ssh_contour_boundary_i", ""))
        jj = parse_index_list(row.get("ssh_contour_boundary_j", ""))
    else:
        return False
    if ii.size < 3 or jj.size < 3:
        return False
    n = min(ii.size, jj.size)
    ii = ii[:n]
    jj = jj[:n]
    valid = (ii >= 0) & (ii < len(lon)) & (jj >= 0) & (jj < len(lat))
    if np.count_nonzero(valid) < 3:
        return False
    x0, y0, x1, y1 = box
    lon_min, lon_max, lat_min, lat_max = bbox
    pts = [
        (px(float(lon[i]), lon_min, lon_max, x0, x1), py(float(lat[j]), lat_min, lat_max, y0, y1))
        for i, j in zip(ii[valid], jj[valid])
        if lon_min <= float(lon[i]) <= lon_max and lat_min <= float(lat[j]) <= lat_max
    ]
    if len(pts) < 3:
        return False
    pts = list(dict.fromkeys(pts))
    if source == "streamline":
        closed = pts + [pts[0]]
        draw.line(closed, fill=color, width=2)
    else:
        for x, y in pts:
            draw.ellipse((x - 1, y - 1, x + 1, y + 1), fill=color)
    return True


def infer_boundary_mode(surface: pd.DataFrame) -> str:
    if surface.empty or "boundary_mode" not in surface.columns:
        return "ssh_effective_contour_primary"
    values = surface["boundary_mode"].dropna().astype(str)
    if values.empty:
        return "ssh_effective_contour_primary"
    return values.mode().iloc[0]


def parse_index_list(value: object) -> np.ndarray:
    text = "" if pd.isna(value) else str(value)
    if not text:
        return np.asarray([], dtype=int)
    out: list[int] = []
    for part in text.replace(",", ";").split(";"):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(int(float(part)))
        except ValueError:
            continue
    return np.asarray(out, dtype=int)


def draw_radius_proxy(draw: ImageDraw.ImageDraw, row: pd.Series, box: tuple[int, int, int, int], bbox: tuple[float, float, float, float], color: tuple[int, int, int]) -> None:
    radius_km = float(row.get("radius_km", np.nan))
    if not np.isfinite(radius_km) and np.isfinite(row.get("ssh_contour_radius_cells", np.nan)):
        radius_km = float(row.get("ssh_contour_radius_cells")) * 11.12
    if not np.isfinite(radius_km) or radius_km <= 0:
        return
    x0, y0, x1, y1 = box
    lon_min, lon_max, lat_min, lat_max = bbox
    lon0 = float(row["plot_lon"])
    lat0 = float(row["plot_lat"])
    cx = px(lon0, lon_min, lon_max, x0, x1)
    cy = py(lat0, lat_min, lat_max, y0, y1)
    deg_lat = radius_km / 111.2
    deg_lon = deg_lat / max(math.cos(math.radians(lat0)), 0.2)
    rx = abs(px(lon0 + deg_lon, lon_min, lon_max, x0, x1) - cx)
    ry = abs(py(lat0 + deg_lat, lat_min, lat_max, y0, y1) - cy)
    draw.ellipse((cx - rx, cy - ry, cx + rx, cy + ry), outline=color, width=2)


def draw_legend(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], surface: pd.DataFrame, contour_count: int) -> None:
    _, _, _, y1 = box
    y = y1 + 36
    draw.ellipse((90, y - 7, 104, y + 7), fill=(37, 99, 235))
    draw.text((114, y - 11), "cyclonic", fill=(20, 24, 32), font=font(18))
    draw.ellipse((220, y - 7, 234, y + 7), fill=(220, 38, 38))
    draw.text((244, y - 11), "anticyclonic", fill=(20, 24, 32), font=font(18))
    weak = int(surface.get("dynamical_core_class", pd.Series([], dtype=str)).astype(str).str.contains("no_streamline|weak").sum()) if not surface.empty and "dynamical_core_class" in surface.columns else 0
    jet = int(surface.get("jet_meander_flag", pd.Series([], dtype=bool)).fillna(False).astype(bool).sum()) if not surface.empty and "jet_meander_flag" in surface.columns else 0
    isolated = int(surface["qc_class"].astype(str).eq("isolated_eddy_candidate").sum()) if "qc_class" in surface.columns else 0
    jet_qc = int(surface["qc_class"].astype(str).eq("jet_meander_candidate").sum()) if "qc_class" in surface.columns else 0
    transient = int(surface["persistence_class"].astype(str).eq("transient").sum()) if "persistence_class" in surface.columns else 0
    edge_label = "saved contours"
    if "qc_class" in surface.columns:
        msg = f"objects={len(surface)}; isolated={isolated}; jet-meander={jet_qc}; transient={transient}; {edge_label}={contour_count}; proxies={len(surface)-contour_count}"
    else:
        msg = f"objects={len(surface)}; {edge_label}={contour_count}; radius proxies={len(surface)-contour_count}; weak/no-core={weak}; jet-flag={jet}"
    draw.text((390, y - 11), msg, fill=(20, 24, 32), font=font(18))


def draw_colorbar(image: Image.Image, draw: ImageDraw.ImageDraw, x: int, y: int, w: int, h: int, vmin: float, vmax: float, units: str) -> None:
    values = np.linspace(vmax, vmin, h, dtype="f4")[:, None]
    bar = Image.fromarray(diverging_ramp(values, vmin, vmax), "RGB").resize((w, h))
    image.paste(bar, (x, y))
    draw.rectangle((x, y, x + w, y + h), outline=(20, 24, 32), width=1)
    for frac in np.linspace(0, 1, 7):
        val = vmax + frac * (vmin - vmax)
        yy = y + int(frac * h)
        draw.line((x + w, yy, x + w + 8, yy), fill=(20, 24, 32), width=1)
        draw.text((x + w + 12, yy - 9), f"{val:.2g}", fill=(20, 24, 32), font=font(14))
    draw.text((x - 4, y + h + 12), units or "SSH", fill=(20, 24, 32), font=font(15))


def diverging_ramp(values: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    arr = np.asarray(values, dtype="f4")
    norm = np.clip((arr - vmin) / max(vmax - vmin, 1.0e-12), 0.0, 1.0)
    blue = np.array([43, 95, 158], dtype="f4")
    white = np.array([248, 248, 246], dtype="f4")
    red = np.array([178, 24, 43], dtype="f4")
    rgb = np.empty(arr.shape + (3,), dtype="f4")
    low = norm <= 0.5
    high = ~low
    t_low = norm / 0.5
    t_high = (norm - 0.5) / 0.5
    rgb[low] = blue + (white - blue) * t_low[low][..., None]
    rgb[high] = white + (red - white) * t_high[high][..., None]
    return np.clip(rgb, 0, 255).astype("u1")


def px(lon: float, lon_min: float, lon_max: float, x0: int, x1: int) -> int:
    return x0 + int(round((lon - lon_min) / (lon_max - lon_min) * (x1 - x0)))


def py(lat: float, lat_min: float, lat_max: float, y0: int, y1: int) -> int:
    return y1 - int(round((lat - lat_min) / (lat_max - lat_min) * (y1 - y0)))


def lon_label(value: float) -> str:
    if value == 0 or value == 360:
        return "0E"
    if value <= 180:
        return f"{int(value)}E"
    return f"{int(360 - value)}W"


def lat_label(value: float) -> str:
    if value == 0:
        return "0"
    return f"{abs(int(value))}{'N' if value > 0 else 'S'}"


def font(size: int) -> ImageFont.ImageFont:
    for path in [r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\arial.ttf"]:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            pass
    return ImageFont.load_default()


if __name__ == "__main__":
    main()
