from __future__ import annotations

import json
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    import netCDF4
except ModuleNotFoundError:  # pragma: no cover - dependency-light dry-run support.
    netCDF4 = None


def read_variable_clean(values) -> np.ndarray:
    arr = np.ma.filled(values, np.nan).astype("float64", copy=False)
    arr[np.abs(arr) > 1.0e20] = np.nan
    return arr


@lru_cache(maxsize=96)
def time_index(path_text: str) -> dict[date, int]:
    if netCDF4 is None:
        raise ModuleNotFoundError("netCDF4 is required to read Filter day files")
    with netCDF4.Dataset(path_text) as ds:
        tvar = ds.variables["time"]
        values = netCDF4.num2date(
            tvar[:],
            tvar.units,
            getattr(tvar, "calendar", "standard"),
            only_use_cftime_datetimes=False,
            only_use_python_datetimes=True,
        )
    return {value.date(): int(i) for i, value in enumerate(values)}


def read_filter_day(
    filter_root: Path,
    template: str,
    day: date,
    variables: Iterable[str] = ("uo_glor", "vo_glor", "thetao_glor"),
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    if netCDF4 is None:
        raise ModuleNotFoundError("netCDF4 is required to read Filter day files")
    path = Path(filter_root) / template.format(year=day.year)
    if not path.exists():
        raise FileNotFoundError(path)
    day_index = time_index(str(path)).get(day)
    if day_index is None:
        raise KeyError(f"{day} not found in {path}")
    with netCDF4.Dataset(path) as ds:
        lon = np.asarray(ds.variables["longitude"][:], dtype="float64")
        lat = np.asarray(ds.variables["latitude"][:], dtype="float64")
        depth = np.asarray(ds.variables["depth"][:], dtype="float64") if "depth" in ds.variables else np.array([])
        fields: dict[str, np.ndarray] = {}
        for variable in variables:
            if variable not in ds.variables:
                raise KeyError(f"{variable} not found in {path}")
            fields[variable] = read_variable_clean(ds.variables[variable][day_index])
    return lon, lat, depth, fields


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_table(table, csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(csv_path, index=False)
    try:
        table.to_parquet(csv_path.with_suffix(".parquet"), index=False)
    except Exception:
        pass
