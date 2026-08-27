from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
import sys

import netCDF4
import numpy as np
import pandas as pd
from tqdm import tqdm

from .common import ensure_dirs, iter_days, load_config, parse_ymd
from .table_io import table_exists, write_table
from .velocity3d_core import LayerDetection, detect_surface_sla_fallback, detect_velocity_layer


ROOT = Path(__file__).resolve().parents[2]
PYEDDY_CANDIDATES = (
    ROOT / "vendor" / "py-eddy-tracker" / "src",
    Path(__file__).resolve().parents[1] / "vendor" / "py-eddy-tracker" / "src",
)
for PYEDDY_SRC in PYEDDY_CANDIDATES:
    if PYEDDY_SRC.exists() and str(PYEDDY_SRC) not in sys.path:
        sys.path.insert(0, str(PYEDDY_SRC))
        break


def layer_output_paths(config: dict, day) -> tuple[Path, Path]:
    root = Path(config["paths"]["layer_dir"])
    return root / f"layer_observations_{day:%Y%m%d}.parquet", root / f"contours_{day:%Y%m%d}.parquet"


def layer_depth_output_paths(config: dict, day, depth_index: int) -> tuple[Path, Path]:
    root = Path(config["paths"]["layer_dir"])
    suffix = f"{day:%Y%m%d}_depth{int(depth_index):03d}"
    return root / f"layer_observations_{suffix}.parquet", root / f"contours_{suffix}.parquet"


def uv_output_path(config: dict, day) -> Path:
    return Path(config["paths"]["input_daily_dir"]) / f"uv_{day:%Y%m%d}.nc"


def _valid_outputs(paths: tuple[Path, Path]) -> bool:
    return all(table_exists(path) for path in paths)


def _records_from_detection(det: LayerDetection) -> tuple[dict, list[dict]]:
    record = {
        "layer_detection_id": det.detection_id,
        "date": det.date,
        "depth_m": det.depth_m,
        "depth_index": det.depth_index,
        "polarity": det.polarity,
        "longitude": det.longitude,
        "latitude": det.latitude,
        "core_speed": det.core_speed,
        "vorticity": det.vorticity,
        "area_m2": det.area_m2,
        "radius_m": det.radius_m,
        "method": det.method,
        "reversal_passed": det.reversal_passed,
    }
    contour_rows = [
        {
            "layer_detection_id": det.detection_id,
            "date": det.date,
            "depth_m": det.depth_m,
            "depth_index": det.depth_index,
            "point_index": i,
            "longitude": float(x),
            "latitude": float(y),
        }
        for i, (x, y) in enumerate(zip(det.contour_lon, det.contour_lat))
    ]
    return record, contour_rows


def _surface_pyeddy_kwargs(config: dict) -> dict:
    id_cfg = config["identification"]
    return dict(
        step=float(id_cfg.get("surface_isoline_step_m", 0.002)),
        shape_error=float(id_cfg.get("surface_fit_errmax_percent", 55)),
        pixel_limit=tuple(int(v) for v in id_cfg.get("surface_pixel_limit", [5, 2000])),
        force_height_unit="m",
        force_speed_unit=None,
        nb_step_to_be_mle=0,
        nb_step_min=int(id_cfg.get("surface_nb_step_min", 2)),
        sampling=int(id_cfg.get("surface_sampling", 50)),
        sampling_method=id_cfg.get("surface_sampling_method", "visvalingam"),
    )


def _surface_from_pyeddy(config: dict, uv_path: Path, day, date_label: str, depth_m: float, depth_index: int) -> list[LayerDetection]:
    from py_eddy_tracker.appli.grid import identification

    id_cfg = config["identification"]
    height_name = config["variables"].get("output_height", "adt")
    pyeddy_date = datetime.combine(day, datetime.min.time())
    anticyclonic, cyclonic = identification(
        str(uv_path),
        "longitude",
        "latitude",
        pyeddy_date,
        height_name,
        "None",
        "None",
        unregular=False,
        cut_wavelength=float(id_cfg.get("surface_cut_wavelength_km", 500)),
        cut_highwavelength=float(id_cfg.get("surface_cut_highwavelength_km", 0)),
        lat_max=float(id_cfg.get("surface_lat_max", 80)),
        filter_order=int(id_cfg.get("surface_filter_order", 3)),
        indexs=None,
        **_surface_pyeddy_kwargs(config),
    )

    detections: list[LayerDetection] = []
    for polarity, eddies in (("anticyclonic", anticyclonic), ("cyclonic", cyclonic)):
        for n in range(len(eddies)):
            detections.append(
                LayerDetection(
                    detection_id=len(detections),
                    date=date_label,
                    depth_m=float(depth_m),
                    depth_index=int(depth_index),
                    polarity=polarity,
                    longitude=float(eddies.lon[n]),
                    latitude=float(eddies.lat[n]),
                    core_speed=float(eddies.speed_average[n]) if "speed_average" in eddies.fields else np.nan,
                    vorticity=np.nan,
                    contour_lon=np.asarray(eddies.contour_lon_e[n], dtype="f4"),
                    contour_lat=np.asarray(eddies.contour_lat_e[n], dtype="f4"),
                    area_m2=float(eddies.effective_area[n]) if "effective_area" in eddies.fields else np.nan,
                    radius_m=float(eddies.radius_e[n]) if "radius_e" in eddies.fields else np.nan,
                    method="sla_surface",
                    reversal_passed=False,
                )
            )
    return detections


def _surface_detections(config: dict, uv_path: Path, day, date_label: str, depth_m: float, depth_index: int) -> list[LayerDetection]:
    height_name = config["variables"].get("output_height", "adt")
    with netCDF4.Dataset(uv_path) as ds:
        lon = np.asarray(ds.variables["longitude"][:], dtype="f8")
        lat = np.asarray(ds.variables["latitude"][:], dtype="f8")
        adt = np.ma.filled(ds.variables[height_name][:], np.nan).astype("f8", copy=False)
    try:
        return _surface_from_pyeddy(config, uv_path, day, date_label, depth_m, depth_index)
    except Exception as exc:
        print(f"[surface fallback] py-eddy-tracker unavailable for {day}: {type(exc).__name__}: {exc}")
        return detect_surface_sla_fallback(lon, lat, adt, depth_m, depth_index, date_label, **config["identification"])


def identify_day(config_path: str | Path, day_iso: str, force: bool = False, depth_index: int | None = None) -> str:
    config = load_config(config_path)
    ensure_dirs(config)
    day = parse_ymd(day_iso)
    obs_path, contour_path = layer_depth_output_paths(config, day, depth_index) if depth_index is not None else layer_output_paths(config, day)
    if not force and not config["identification"].get("overwrite_existing", False) and _valid_outputs((obs_path, contour_path)):
        return day_iso

    uv_path = uv_output_path(config, day)
    if not uv_path.exists():
        raise FileNotFoundError(f"Missing UV subset: {uv_path}")

    with netCDF4.Dataset(uv_path) as ds:
        lon = np.asarray(ds.variables["longitude"][:], dtype="f8")
        lat = np.asarray(ds.variables["latitude"][:], dtype="f8")
        depth = np.asarray(ds.variables["depth"][:], dtype="f8")
        date_label = getattr(ds, "date", day_iso)
        depth_indices = [depth_index] if depth_index is not None else list(range(depth.size))
        surface_depth_index = int(config["identification"].get("surface_depth_index", 0))
        need_velocity = any(int(k) != surface_depth_index for k in depth_indices)
        if need_velocity:
            u_all = np.ma.filled(ds.variables["u"][:, :, :], np.nan).astype("f8", copy=False)
            v_all = np.ma.filled(ds.variables["v"][:, :, :], np.nan).astype("f8", copy=False)
        else:
            u_all = None
            v_all = None

    kwargs = dict(config["identification"])
    kwargs.pop("workers", None)
    kwargs.pop("overwrite_existing", None)
    kwargs.pop("surface_depth_index", None)
    for key in list(kwargs):
        if key.startswith("surface_") or key in {"surface_method", "subsurface_method"}:
            kwargs.pop(key)

    records = []
    contour_rows = []
    next_id = 0
    for k in depth_indices:
        if k is None or k < 0 or k >= depth.size:
            raise IndexError(f"depth_index {k} outside 0..{depth.size - 1}")
        if int(k) == surface_depth_index:
            detections = _surface_detections(config, uv_path, day, date_label, float(depth[k]), int(k))
        else:
            if u_all is None or v_all is None:
                raise RuntimeError("Velocity arrays were not loaded for a subsurface depth.")
            detections = detect_velocity_layer(lon, lat, float(depth[k]), int(k), u_all[int(k)], v_all[int(k)], date_label, **kwargs)
        for det in detections:
            det = det.__class__(**{**det.__dict__, "detection_id": next_id})
            next_id += 1
            record, rows = _records_from_detection(det)
            records.append(record)
            contour_rows.extend(rows)

    write_table(pd.DataFrame.from_records(records), obs_path, index=False)
    write_table(pd.DataFrame.from_records(contour_rows), contour_path, index=False)
    return day_iso


def identify_range(
    config_path: str | Path,
    start: str | None = None,
    end: str | None = None,
    force: bool = False,
    workers: int | None = None,
    depth_index: int | None = None,
) -> None:
    config = load_config(config_path)
    ensure_dirs(config)
    start_day = parse_ymd(start or config["date_range"]["start"])
    end_day = parse_ymd(end or config["date_range"]["end"])
    days = list(iter_days(start_day, end_day))
    workers = int(workers or config["identification"].get("workers", 1))

    if workers <= 1 or len(days) <= 1:
        for day in tqdm(days, desc="Hybrid 3D layer identification", unit="day"):
            identify_day(config_path, day.isoformat(), force=force, depth_index=depth_index)
        return

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(identify_day, str(config_path), day.isoformat(), force, depth_index): day
            for day in days
        }
        with tqdm(total=len(futures), desc=f"Hybrid 3D layer identification ({workers} workers)", unit="day") as bar:
            for future in as_completed(futures):
                day = futures[future]
                try:
                    future.result()
                except Exception as exc:
                    raise RuntimeError(f"Hybrid 3D identification failed for {day}") from exc
                bar.update(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid 3D eddy layer identification: SLA surface, velocity-reversal subsurface.")
    parser.add_argument("--config", default="config/config_3d.yaml")
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--depth-index", type=int)
    args = parser.parse_args()
    identify_range(args.config, args.start, args.end, force=args.force, workers=args.workers, depth_index=args.depth_index)


if __name__ == "__main__":
    main()
