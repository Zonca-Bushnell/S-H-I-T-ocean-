from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import netCDF4
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from tqdm import tqdm

from .build_cmems_climatology import build_climatology, clim_doy_index, climatology_path
from .common import ensure_dirs, load_config, parse_ymd
from .table_io import read_table, read_table_or_partitions, write_table


EARTH_RADIUS_M = 6_371_000.0
PHASE_NAMES = ["birth", "growth", "mature", "decay", "death"]


@dataclass(frozen=True)
class SourceFields:
    u: np.ndarray
    v: np.ndarray
    thetao: np.ndarray
    so: np.ndarray
    sigma0: np.ndarray | None
    zos: np.ndarray
    mlotst_raw: np.ndarray
    mlotst: np.ndarray


def _date_to_index(ds: netCDF4.Dataset, time_name: str) -> dict[date, int]:
    time_var = ds.variables[time_name]
    values = netCDF4.num2date(
        time_var[:],
        time_var.units,
        getattr(time_var, "calendar", "standard"),
        only_use_cftime_datetimes=False,
        only_use_python_datetimes=True,
    )
    return {value.date(): i for i, value in enumerate(values)}


def _phase_index(life_phase: float, bins: int) -> int:
    if not np.isfinite(life_phase):
        return -1
    return int(np.clip(np.floor(life_phase * bins), 0, bins - 1))


def _xy_m(lon: np.ndarray, lat: np.ndarray, lon0: float, lat0: float) -> tuple[np.ndarray, np.ndarray]:
    x = EARTH_RADIUS_M * np.cos(np.radians(lat0)) * np.radians(np.asarray(lon) - lon0)
    y = EARTH_RADIUS_M * np.radians(np.asarray(lat) - lat0)
    return x, y


def _lonlat_from_xy_m(x: np.ndarray, y: np.ndarray, lon0: float, lat0: float) -> tuple[np.ndarray, np.ndarray]:
    lat = lat0 + np.degrees(y / EARTH_RADIUS_M)
    lon = lon0 + np.degrees(x / (EARTH_RADIUS_M * np.cos(np.radians(lat0))))
    return lon, lat


def _rotate(x: np.ndarray, y: np.ndarray, alpha_deg: float) -> tuple[np.ndarray, np.ndarray]:
    ca = np.cos(np.radians(alpha_deg))
    sa = np.sin(np.radians(alpha_deg))
    return x * ca - y * sa, x * sa + y * ca


def _alpha_from_centers(centers: pd.DataFrame, lon0: float, lat0: float, radius_m: float) -> float:
    rows = centers.sort_values("depth_index").copy()
    x, y = _xy_m(rows["longitude"].astype(float).to_numpy(), rows["latitude"].astype(float).to_numpy(), lon0, lat0)
    x = x / max(radius_m, 1.0)
    y = y / max(radius_m, 1.0)
    good = np.isfinite(x) & np.isfinite(y)
    if good.sum() < 2:
        return 0.0
    x = x[good]
    y = y[good]
    dx = float(x[-1] - x[0])
    dy = float(y[-1] - y[0])
    if np.hypot(dx, dy) < 0.02:
        return 0.0
    return float(-np.degrees(np.arctan2(dy, dx)))


def _interp2(lat: np.ndarray, lon: np.ndarray, field: np.ndarray, latq: np.ndarray, lonq: np.ndarray) -> np.ndarray:
    interp = RegularGridInterpolator((lat, lon), field, bounds_error=False, fill_value=np.nan)
    out = interp(np.column_stack([latq.ravel(), lonq.ravel()])).reshape(latq.shape)
    return out.astype("f4", copy=False)


def _sigma0_anomaly(
    thetao: np.ndarray,
    so: np.ndarray,
    thetao_clim: np.ndarray,
    so_clim: np.ndarray,
    lon: np.ndarray,
    lat: np.ndarray,
    depth: np.ndarray,
) -> np.ndarray:
    import gsw

    z = -np.asarray(depth, dtype="f8")[:, None, None]
    lat3 = np.asarray(lat, dtype="f8")[None, :, None]
    lon3 = np.asarray(lon, dtype="f8")[None, None, :]
    p = gsw.p_from_z(z, lat3)
    sa = gsw.SA_from_SP(so.astype("f8", copy=False), p, lon3, lat3)
    ct = gsw.CT_from_pt(sa, thetao.astype("f8", copy=False))
    sigma = gsw.sigma0(sa, ct)
    sa_clim = gsw.SA_from_SP(so_clim.astype("f8", copy=False), p, lon3, lat3)
    ct_clim = gsw.CT_from_pt(sa_clim, thetao_clim.astype("f8", copy=False))
    sigma_clim = gsw.sigma0(sa_clim, ct_clim)
    return (sigma - sigma_clim).astype("f4")


def _load_source_fields(
    src: netCDF4.Dataset,
    clim: netCDF4.Dataset,
    time_index: int,
    doy_index: int,
    lon: np.ndarray,
    lat: np.ndarray,
    depth: np.ndarray,
    include_sigma0: bool = False,
) -> SourceFields:
    u = np.ma.filled(src.variables["uo_glor"][time_index, ...], np.nan).astype("f4") - np.ma.filled(clim.variables["u_clim"][doy_index, ...], np.nan).astype("f4")
    v = np.ma.filled(src.variables["vo_glor"][time_index, ...], np.nan).astype("f4") - np.ma.filled(clim.variables["v_clim"][doy_index, ...], np.nan).astype("f4")
    thetao_raw = np.ma.filled(src.variables["thetao_glor"][time_index, ...], np.nan).astype("f4")
    so_raw = np.ma.filled(src.variables["so_glor"][time_index, ...], np.nan).astype("f4")
    thetao_clim = np.ma.filled(clim.variables["thetao_clim"][doy_index, ...], np.nan).astype("f4")
    so_clim = np.ma.filled(clim.variables["so_clim"][doy_index, ...], np.nan).astype("f4")
    thetao = thetao_raw - thetao_clim
    so = so_raw - so_clim
    sigma0 = _sigma0_anomaly(thetao_raw, so_raw, thetao_clim, so_clim, lon, lat, depth) if include_sigma0 else None
    zos = np.ma.filled(src.variables["zos_glor"][time_index, ...], np.nan).astype("f4") - np.ma.filled(clim.variables["zos_clim"][doy_index, ...], np.nan).astype("f4")
    mlotst = np.ma.filled(src.variables["mlotst_glor"][time_index, ...], np.nan).astype("f4")
    mlotst_anom = mlotst - np.ma.filled(clim.variables["mlotst_clim"][doy_index, ...], np.nan).astype("f4")
    return SourceFields(u=u, v=v, thetao=thetao, so=so, sigma0=sigma0, zos=zos, mlotst_raw=mlotst, mlotst=mlotst_anom)


def _add_sum_count(sum_arr: np.ndarray, count_arr: np.ndarray, idx: tuple, values: np.ndarray) -> None:
    good = np.isfinite(values)
    sum_arr[idx] += np.nan_to_num(values, nan=0.0)
    count_arr[idx] += good.astype("u4")


def _safe_mean(sum_arr: np.ndarray, count_arr: np.ndarray) -> np.ndarray:
    out = np.full(sum_arr.shape, np.nan, dtype="f4")
    np.divide(sum_arr, count_arr, out=out, where=count_arr > 0)
    return out


def _output_dir(config: dict, shape_dir: Path, start: date, end: date, shape: str) -> Path:
    return Path(config["paths"]["output_dir"]) / f"lifecycle_composites_{start:%Y}_{end:%Y}_{shape}"


def run_lifecycle_composite(
    config_path: str | Path,
    shape_dir: str | Path,
    start: str,
    end: str,
    shape: str = "coherent",
    phase_bins: int = 5,
    rmax: float = 2.0,
    ngrid: int = 61,
    smooth_days: int = 31,
    workers: int = 1,
    max_tracks_per_group: int | None = None,
    force_climatology: bool = False,
    include_sigma0: bool = False,
) -> Path:
    config = load_config(config_path)
    ensure_dirs(config)
    start_day = parse_ymd(start)
    end_day = parse_ymd(end)
    shape_dir = Path(shape_dir)
    if not shape_dir.is_absolute():
        shape_dir = Path.cwd() / shape_dir
    out_dir = _output_dir(config, shape_dir, start_day, end_day, shape)
    out_dir.mkdir(parents=True, exist_ok=True)

    clim_path = climatology_path(config, parse_ymd("1993-01-01"), parse_ymd("2022-12-31"), smooth_days)
    if not clim_path.exists() or force_climatology:
        clim_path = build_climatology(config_path, "1993-01-01", "2022-12-31", smooth_days, force=force_climatology)

    tracks = read_table(shape_dir / "shape_tracks.parquet")
    daily = read_table(shape_dir / "shape_daily_metrics.parquet")
    catalog_root = Path(config["paths"]["catalog_dir"])
    centers = read_table_or_partitions(
        catalog_root / "layer_centers_completed.parquet",
        catalog_root / "layer_centers_completed_parts",
    )
    tracks = tracks[tracks["shape_class"].astype(str) == str(shape)].copy()
    if max_tracks_per_group is not None:
        tracks = tracks.groupby(["shape_class", "polarity"], group_keys=False).head(int(max_tracks_per_group)).copy()
    track_ids = set(tracks["track3d_id"].astype(int))
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily[
        daily["track3d_id"].astype(int).isin(track_ids)
        & (daily["date"] >= pd.Timestamp(start_day))
        & (daily["date"] <= pd.Timestamp(end_day))
    ].copy()
    if daily.empty:
        raise RuntimeError(f"No daily samples for shape={shape} in {start_day}..{end_day}")

    track_info = tracks.set_index("track3d_id")
    daily["start_date"] = daily["track3d_id"].map(track_info["start_date"])
    daily["lifetime_days"] = daily["track3d_id"].map(track_info["lifetime_days"])
    daily["shape_class"] = daily["track3d_id"].map(track_info["shape_class"])
    daily["polarity"] = daily["track3d_id"].map(track_info["polarity"])
    daily["life_day"] = (daily["date"] - pd.to_datetime(daily["start_date"])).dt.days
    daily["life_phase"] = daily["life_day"].astype(float) / np.maximum(daily["lifetime_days"].astype(float) - 1.0, 1.0)
    daily["phase_index"] = daily["life_phase"].map(lambda value: _phase_index(float(value), phase_bins))
    daily = daily[(daily["phase_index"] >= 0) & (daily["phase_index"] < phase_bins)].copy()

    polarities = sorted(daily["polarity"].astype(str).unique())
    if phase_bins != 5:
        phase_names = [f"phase_{i:02d}" for i in range(phase_bins)]
    else:
        phase_names = PHASE_NAMES
    polarity_index = {p: i for i, p in enumerate(polarities)}

    sample_object_ids = set(daily["eddy3d_object_id"].astype(int))
    centers = centers[centers["eddy3d_object_id"].astype(int).isin(sample_object_ids)].copy()
    centers_by_object = {int(k): g.copy() for k, g in centers.groupby("eddy3d_object_id")}

    with netCDF4.Dataset(config["data_source"]["input_nc_file"]) as src:
        lon = np.asarray(src.variables["longitude"][:], dtype="f8")
        lat = np.asarray(src.variables["latitude"][:], dtype="f8")
        depth = np.asarray(src.variables["depth"][:], dtype="f8")
        time_index = _date_to_index(src, "time")

    xv = np.linspace(-float(rmax), float(rmax), int(ngrid), dtype="f8")
    yv = np.linspace(-float(rmax), float(rmax), int(ngrid), dtype="f8")
    xq, yq = np.meshgrid(xv, yv)
    mask = np.hypot(xq, yq) <= float(rmax)
    dims3 = (1, len(polarities), phase_bins, depth.size, ngrid, ngrid)
    dims2 = (1, len(polarities), phase_bins, ngrid, ngrid)
    names3 = ["u_anom", "v_anom", "speed_anom", "thetao_anom", "so_anom"]
    if include_sigma0:
        names3.append("sigma0_anom")
    sums3 = {name: np.zeros(dims3, dtype="f8") for name in names3}
    counts3 = {name: np.zeros(dims3, dtype="u4") for name in sums3}
    sums2 = {name: np.zeros(dims2, dtype="f8") for name in ("adt_anom", "mlotst", "mlotst_anom")}
    counts2 = {name: np.zeros(dims2, dtype="u4") for name in sums2}
    event_count = np.zeros((1, len(polarities), phase_bins), dtype="u4")
    index_rows: list[dict] = []

    with netCDF4.Dataset(config["data_source"]["input_nc_file"]) as src, netCDF4.Dataset(clim_path) as clim:
        time_index = _date_to_index(src, "time")
        for day, day_rows in tqdm(list(daily.groupby("date")), desc=f"Composite lifecycle {shape}", unit="day"):
            day_date = pd.Timestamp(day).date()
            if day_date not in time_index:
                continue
            fields = _load_source_fields(
                src,
                clim,
                time_index[day_date],
                clim_doy_index(day_date),
                lon,
                lat,
                depth,
                include_sigma0=include_sigma0,
            )
            for row in day_rows.itertuples(index=False):
                object_id = int(row.eddy3d_object_id)
                c = centers_by_object.get(object_id)
                if c is None or c.empty:
                    continue
                c = c.sort_values("depth_index")
                origin = c.iloc[0]
                lon0 = float(origin.longitude)
                lat0 = float(origin.latitude)
                radius_m = float(origin.radius_m) if np.isfinite(float(origin.radius_m)) and float(origin.radius_m) > 0 else float(c["radius_m"].median())
                if not np.isfinite(radius_m) or radius_m <= 0:
                    continue
                alpha = _alpha_from_centers(c, lon0, lat0, radius_m)
                xn, yn = _rotate(xq, yq, -alpha)
                lonq, latq = _lonlat_from_xy_m(xn * radius_m, yn * radius_m, lon0, lat0)
                pi = polarity_index[str(row.polarity)]
                ph = int(row.phase_index)
                event_count[0, pi, ph] += 1
                for k in range(depth.size):
                    idx = (0, pi, ph, k)
                    uq = _interp2(lat, lon, fields.u[k], latq, lonq)
                    vq = _interp2(lat, lon, fields.v[k], latq, lonq)
                    ur, vr = _rotate(uq, vq, alpha)
                    tq = _interp2(lat, lon, fields.thetao[k], latq, lonq)
                    sq = _interp2(lat, lon, fields.so[k], latq, lonq)
                    sigmaq = _interp2(lat, lon, fields.sigma0[k], latq, lonq) if fields.sigma0 is not None else None
                    for arr in (ur, vr, tq, sq):
                        arr[~mask] = np.nan
                    if sigmaq is not None:
                        sigmaq[~mask] = np.nan
                    speed = np.hypot(ur, vr).astype("f4")
                    speed[~mask] = np.nan
                    _add_sum_count(sums3["u_anom"], counts3["u_anom"], idx, ur)
                    _add_sum_count(sums3["v_anom"], counts3["v_anom"], idx, vr)
                    _add_sum_count(sums3["speed_anom"], counts3["speed_anom"], idx, speed)
                    _add_sum_count(sums3["thetao_anom"], counts3["thetao_anom"], idx, tq)
                    _add_sum_count(sums3["so_anom"], counts3["so_anom"], idx, sq)
                    if sigmaq is not None:
                        _add_sum_count(sums3["sigma0_anom"], counts3["sigma0_anom"], idx, sigmaq)
                idx2 = (0, pi, ph)
                adt = _interp2(lat, lon, fields.zos, latq, lonq)
                mld_raw = _interp2(lat, lon, fields.mlotst_raw, latq, lonq)
                mld = _interp2(lat, lon, fields.mlotst, latq, lonq)
                adt[~mask] = np.nan
                mld_raw[~mask] = np.nan
                mld[~mask] = np.nan
                _add_sum_count(sums2["adt_anom"], counts2["adt_anom"], idx2, adt)
                _add_sum_count(sums2["mlotst"], counts2["mlotst"], idx2, mld_raw)
                _add_sum_count(sums2["mlotst_anom"], counts2["mlotst_anom"], idx2, mld)
                index_rows.append(
                    {
                        "track3d_id": int(row.track3d_id),
                        "eddy3d_object_id": object_id,
                        "date": f"{day_date:%Y-%m-%d}",
                        "shape_class": str(row.shape_class),
                        "polarity": str(row.polarity),
                        "phase_index": ph,
                        "phase_name": phase_names[ph],
                        "life_phase": float(row.life_phase),
                        "radius_m": radius_m,
                        "alpha_deg": alpha,
                    }
                )

    out_nc = out_dir / "lifecycle_composite.nc"
    with netCDF4.Dataset(out_nc, "w", format="NETCDF4") as ds:
        ds.createDimension("shape", 1)
        ds.createDimension("polarity", len(polarities))
        ds.createDimension("phase", phase_bins)
        ds.createDimension("depth", depth.size)
        ds.createDimension("y", ngrid)
        ds.createDimension("x", ngrid)
        ds.title = f"Lifecycle-normalized 3D eddy composite for shape={shape}"
        ds.anomaly_definition = "field - 1993_2022_day_of_year_climatology_31d_smooth"
        ds.sigma0_definition = "TEOS-10 gsw.sigma0(SA_from_SP(so,p,lon,lat), CT_from_pt(SA,thetao)); anomaly is source sigma0 - climatological sigma0"
        ds.climatology_file = str(clim_path)
        ds.rmax = float(rmax)
        ds.ngrid = int(ngrid)
        ds.alignment = "daily centerline tilt rotated to +x"
        ds.createVariable("x_R", "f4", ("x",))[:] = xv.astype("f4")
        ds.createVariable("y_R", "f4", ("y",))[:] = yv.astype("f4")
        ds.createVariable("depth", "f4", ("depth",))[:] = depth.astype("f4")
        ds.createVariable("shape_name", str, ("shape",))[:] = np.array([shape], dtype=object)
        ds.createVariable("polarity_name", str, ("polarity",))[:] = np.array(polarities, dtype=object)
        ds.createVariable("phase_name", str, ("phase",))[:] = np.array(phase_names, dtype=object)
        ds.createVariable("event_count", "u4", ("shape", "polarity", "phase"))[:] = event_count
        for name, arr in sums3.items():
            var = ds.createVariable(name, "f4", ("shape", "polarity", "phase", "depth", "y", "x"), zlib=True, complevel=4)
            if name == "sigma0_anom":
                var.units = "kg m-3"
                var.long_name = "potential density anomaly sigma0 referenced to 0 dbar"
            var[:] = _safe_mean(arr, counts3[name])
            cvar = ds.createVariable(f"count_{name}", "u4", ("shape", "polarity", "phase", "depth", "y", "x"), zlib=True, complevel=4)
            cvar[:] = counts3[name]
        for name, arr in sums2.items():
            var = ds.createVariable(name, "f4", ("shape", "polarity", "phase", "y", "x"), zlib=True, complevel=4)
            var[:] = _safe_mean(arr, counts2[name])
            cvar = ds.createVariable(f"count_{name}", "u4", ("shape", "polarity", "phase", "y", "x"), zlib=True, complevel=4)
            cvar[:] = counts2[name]

    index_df = pd.DataFrame.from_records(index_rows)
    write_table(index_df, out_dir / "lifecycle_composite_index.parquet", index=False)
    summary = (
        index_df.groupby(["shape_class", "polarity", "phase_name"], as_index=False)
        .size()
        .rename(columns={"size": "event_count"})
        if not index_df.empty
        else pd.DataFrame(columns=["shape_class", "polarity", "phase_name", "event_count"])
    )
    write_table(summary, out_dir / "lifecycle_composite_summary.parquet", index=False)
    return out_nc


def main() -> None:
    parser = argparse.ArgumentParser(description="Build lifecycle-normalized 3D eddy composites from CMEMS fields.")
    parser.add_argument("--config", default="config/config_3d_cmems.yaml")
    parser.add_argument("--shape-dir", required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--shape", default="coherent")
    parser.add_argument("--phase-bins", type=int, default=5)
    parser.add_argument("--group-by", nargs="*", default=["shape", "polarity"])
    parser.add_argument("--rmax", type=float, default=2.0)
    parser.add_argument("--ngrid", type=int, default=61)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-tracks-per-group", type=int)
    parser.add_argument("--anomaly-source", choices=["doy-climatology"], default="doy-climatology")
    parser.add_argument("--climatology-smooth-days", type=int, default=31)
    parser.add_argument("--force-climatology", action="store_true")
    parser.add_argument("--include-sigma0", action="store_true", help="Also composite TEOS-10 potential density anomaly sigma0.")
    args = parser.parse_args()
    if args.group_by != ["shape", "polarity"]:
        raise ValueError("First version supports --group-by shape polarity.")
    out = run_lifecycle_composite(
        args.config,
        args.shape_dir,
        args.start,
        args.end,
        shape=args.shape,
        phase_bins=args.phase_bins,
        rmax=args.rmax,
        ngrid=args.ngrid,
        smooth_days=args.climatology_smooth_days,
        workers=args.workers,
        max_tracks_per_group=args.max_tracks_per_group,
        force_climatology=args.force_climatology,
        include_sigma0=args.include_sigma0,
    )
    print(f"Lifecycle composite: {out}")


if __name__ == "__main__":
    main()
