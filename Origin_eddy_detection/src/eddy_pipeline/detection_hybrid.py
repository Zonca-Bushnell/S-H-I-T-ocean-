from __future__ import annotations

import argparse
import json
import math
import os
import uuid
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path


_MATLAB_DLL_HANDLES: list[object] = []


def _prepare_matlab_dll_search_path() -> None:
    """Register MATLAB runtime DLL directories before scientific DLLs are loaded."""
    if _MATLAB_DLL_HANDLES:
        return
    default_root = Path(r"D:\Util\Ma\01_Matlab")
    matlab_root = default_root if default_root.exists() else Path(os.environ.get("MATLABROOT", str(default_root)))
    dll_dirs = [
        matlab_root / "bin",
        matlab_root / "bin" / "win64",
        matlab_root / "runtime" / "win64",
        matlab_root / "extern" / "bin" / "win64",
        matlab_root / "sys" / "os" / "win64",
        matlab_root / "bin" / "win64" / "mvm_transport" / "mvm_transport",
    ]
    existing = [path for path in dll_dirs if path.exists()]
    if hasattr(os, "add_dll_directory"):
        for path in existing:
            _MATLAB_DLL_HANDLES.append(os.add_dll_directory(str(path)))
    os.environ["MATLABROOT"] = str(matlab_root)
    os.environ["PATH"] = os.pathsep.join([str(path) for path in existing] + [os.environ.get("PATH", "")])


_prepare_matlab_dll_search_path()

import numpy as np
import pandas as pd
from netCDF4 import Dataset, num2date
from scipy import ndimage

from .utils.table_io import DEFAULT_PARQUET_ENGINE


EARTH_RADIUS_M = 6_371_000.0
FAILURE_LABELS = {
    0: "invalid_velocity",
    1: "velocity_ratio",
    2: "angle_jump",
    3: "rotation_direction",
    4: "too_many_direction_exceptions",
    5: "dead_zone",
    6: "symmetry",
    7: "tangent_alignment",
    8: "opposite_reversal",
    9: "boundary_monotonic_rotation",
    10: "no_closed_streamline",
}
OBJECT_VOXEL_COLUMNS = [
    "date",
    "hua_object_id",
    "depth_index",
    "i",
    "j",
    "lon",
    "lat",
    "depth_m",
    "polarity",
    "accepted_radius_cells",
    "node_key_3d",
    "node_key_2d",
]

HARD_FAILURE_ORDER = (
    (0, "invalid_velocity"),
    (1, "velocity_ratio"),
    (2, "angle_jump"),
    (9, "boundary_monotonic_rotation"),
    (7, "tangent_alignment"),
    (8, "opposite_reversal"),
)


def _get_pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


@dataclass(frozen=True)
class DetectionParams:
    ssh_window_cells: int
    start_radius_cells: int
    max_radius_cells: int
    speed_ratio_max: float
    angle_jump_max_deg: float
    tangent_tolerance_deg: float
    symmetry_tolerance_deg: float
    min_tangent_fraction: float
    min_reversal_fraction: float
    min_finite_fraction: float
    direction_exception_extra: int
    surface_search_cells: int
    deep_search_cells: int
    require_boundary_monotonic_rotation: bool = False
    boundary_monotonic_exception_limit: int = 0
    boundary_mode: str = "circle_strict_original"
    streamline_direction_exception_fraction: float = 0.10
    streamline_step_cells: float = 0.5
    streamline_max_steps: int = 180
    streamline_start_angles: int = 4
    streamline_closure_tolerance_cells: float = 1.75
    streamline_min_winding_turns: float = 0.75
    streamline_min_points: int = 16
    hua_backend: str = "python"
    matlab_use_gpu: bool = False


class MatlabHuaBackend:
    """Call the MATLAB Hua kernels while keeping Python as the IO/orchestration layer."""

    def __init__(self, params: DetectionParams, cache_dir: Path) -> None:
        matlab_root = Path(os.environ.get("MATLABROOT", r"D:\Util\Ma\01_Matlab"))
        dll_dirs = [
            matlab_root / "bin" / "win64",
            matlab_root / "runtime" / "win64",
            matlab_root / "extern" / "bin" / "win64",
            matlab_root / "sys" / "os" / "win64",
            matlab_root / "bin" / "win64" / "mvm_transport" / "mvm_transport",
        ]
        existing_dll_dirs = [path for path in dll_dirs if path.exists()]
        if hasattr(os, "add_dll_directory"):
            self._dll_handles = [os.add_dll_directory(str(path)) for path in existing_dll_dirs]
        else:
            self._dll_handles = []
        os.environ["MATLABROOT"] = str(matlab_root)
        os.environ["PATH"] = os.pathsep.join([str(path) for path in existing_dll_dirs] + [os.environ.get("PATH", "")])
        try:
            import matlab  # type: ignore[import-not-found]
            import matlab.engine  # type: ignore[import-not-found]
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "MATLAB Hua backend requires the MATLAB Engine for Python in the active "
                "environment. Install it from D:\\Util\\Ma\\01_Matlab\\extern\\engines\\python "
                "before running --hua-backend matlab."
            ) from exc
        self.params = params
        self.matlab = matlab
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.eng = matlab.engine.start_matlab()
        matlab_dir = Path(__file__).resolve().parents[2] / "matlab"
        self.eng.addpath(str(matlab_dir), nargout=0)
        self._loaded_day: str | None = None
        self._loaded_grid: bool = False
        self._install_params()

    @staticmethod
    def _matlab_scalar(value: object) -> str:
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, (int, np.integer)):
            return str(int(value))
        if isinstance(value, (float, np.floating)):
            if not np.isfinite(float(value)):
                return "NaN"
            return repr(float(value))
        raise TypeError(f"Unsupported MATLAB parameter value: {value!r}")

    @staticmethod
    def _matlab_quote(path: Path) -> str:
        return "'" + str(path).replace("'", "''") + "'"

    def _install_params(self) -> None:
        fields = {
            "start_radius_cells": self.params.start_radius_cells,
            "max_radius_cells": self.params.max_radius_cells,
            "speed_ratio_max": self.params.speed_ratio_max,
            "angle_jump_max_deg": self.params.angle_jump_max_deg,
            "tangent_tolerance_deg": self.params.tangent_tolerance_deg,
            "symmetry_tolerance_deg": self.params.symmetry_tolerance_deg,
            "min_tangent_fraction": self.params.min_tangent_fraction,
            "min_reversal_fraction": self.params.min_reversal_fraction,
            "min_finite_fraction": self.params.min_finite_fraction,
            "direction_exception_extra": self.params.direction_exception_extra,
            "require_boundary_monotonic_rotation": self.params.require_boundary_monotonic_rotation,
            "boundary_monotonic_exception_limit": self.params.boundary_monotonic_exception_limit,
            "streamline_direction_exception_fraction": self.params.streamline_direction_exception_fraction,
            "streamline_step_cells": self.params.streamline_step_cells,
            "streamline_max_steps": self.params.streamline_max_steps,
            "streamline_start_angles": self.params.streamline_start_angles,
            "streamline_closure_tolerance_cells": self.params.streamline_closure_tolerance_cells,
            "streamline_min_winding_turns": self.params.streamline_min_winding_turns,
            "streamline_min_points": self.params.streamline_min_points,
            "surface_search_cells": self.params.surface_search_cells,
            "deep_search_cells": self.params.deep_search_cells,
            "use_gpu": self.params.matlab_use_gpu,
        }
        items = []
        for name, value in fields.items():
            items.append(f"'{name}'")
            items.append(self._matlab_scalar(value))
        radii = f"{int(self.params.start_radius_cells)}:{int(self.params.max_radius_cells)}"
        self.eng.eval(f"hua_params=struct({','.join(items)}); hua_radii={radii};", nargout=0)
        boundary_mode = str(self.params.boundary_mode).replace("'", "''")
        self.eng.eval(f"hua_params.boundary_mode='{boundary_mode}';", nargout=0)

    def load_day(self, day: date, u_day: np.ndarray, v_day: np.ndarray) -> None:
        day_key = day.isoformat()
        if self._loaded_day == day_key:
            return
        from scipy.io import savemat

        mat_path = self.cache_dir / f"matlab_hua_uv_{day:%Y%m%d}.mat"
        if not mat_path.exists():
            savemat(
                mat_path,
                {
                    "u_day": np.asarray(u_day, dtype="float32"),
                    "v_day": np.asarray(v_day, dtype="float32"),
                },
                do_compression=False,
            )
        quoted = self._matlab_quote(mat_path)
        self.eng.eval(f"S=load({quoted}); u_day=S.u_day; v_day=S.v_day; clear S;", nargout=0)
        if self.params.matlab_use_gpu:
            self.eng.eval("gpuDevice; u_day=gpuArray(double(u_day)); v_day=gpuArray(double(v_day));", nargout=0)
        self._loaded_day = day_key

    def load_grid(self, lon: np.ndarray, lat: np.ndarray) -> None:
        """Stage lon/lat vectors for MATLAB-side subgrid refinement."""
        if self._loaded_grid:
            return
        self.eng.workspace["hua_lon_vec"] = self.matlab.double(np.asarray(lon, dtype="float64").ravel().tolist())
        self.eng.workspace["hua_lat_vec"] = self.matlab.double(np.asarray(lat, dtype="float64").ravel().tolist())
        self._loaded_grid = True

    def refine_many(
        self,
        depth_index: int,
        centers_ij: np.ndarray,
        *,
        target_degree: float,
        window_radius_cells: int,
        min_finite_fraction: float,
    ) -> list[dict[str, object]]:
        centers = np.asarray(centers_ij, dtype="float64")
        if centers.ndim != 2 or centers.shape[1] != 2:
            raise ValueError("centers_ij must be an n-by-2 array of [center_i, center_j] grid anchors")
        if len(centers) == 0:
            return []
        matlab_centers = centers.copy()
        matlab_centers[:, 0] += 1.0
        matlab_centers[:, 1] += 1.0
        k = int(depth_index) + 1
        self.eng.workspace["refine_centers"] = self.matlab.double(matlab_centers.tolist())
        self.eng.workspace["refine_target_degree"] = float(target_degree)
        self.eng.workspace["refine_window_radius_cells"] = float(window_radius_cells)
        self.eng.workspace["refine_min_finite_fraction"] = float(min_finite_fraction)
        command = (
            "refine_rows=refine_speed_min_batch_kernel("
            f"hypot(squeeze(u_day({k},:,:)),squeeze(v_day({k},:,:))),"
            f"squeeze(u_day({k},:,:)),squeeze(v_day({k},:,:)),"
            "hua_lon_vec,hua_lat_vec,refine_centers,"
            "refine_target_degree,refine_window_radius_cells,refine_min_finite_fraction,hua_params);"
        )
        self.eng.eval(command, nargout=0)
        encoded = self.eng.eval("jsonencode(refine_rows)", nargout=1)
        payload = json.loads(str(encoded))
        if isinstance(payload, dict):
            payloads = [payload]
        elif isinstance(payload, list):
            payloads = payload
        else:
            payloads = []
        if len(payloads) != len(centers):
            raise RuntimeError(f"MATLAB refinement backend returned {len(payloads)} rows for {len(centers)} centers.")
        rows: list[dict[str, object]] = []
        for item in payloads:
            rows.append(
                {
                    "center_i_refined": _payload_float(item, "center_i_refined", np.nan) - 1.0,
                    "center_j_refined": _payload_float(item, "center_j_refined", np.nan) - 1.0,
                    "center_lon_refined": _payload_float(item, "center_lon_refined", np.nan),
                    "center_lat_refined": _payload_float(item, "center_lat_refined", np.nan),
                    "refined_speed_ms": _payload_float(item, "refined_speed_ms", np.nan),
                    "refined_offset_km": _payload_float(item, "refined_offset_km", np.nan),
                    "refined_ok": bool(item.get("refined_ok", False)),
                    "subgrid_fit_quality": str(item.get("subgrid_fit_quality", "matlab_unknown")),
                }
            )
        return rows

    def check(self, depth_index: int, center_i: float, center_j: float) -> dict[str, object]:
        return self.check_many(depth_index, np.asarray([[center_i, center_j]], dtype="float64"))[0]

    def check_many(self, depth_index: int, centers_ij: np.ndarray) -> list[dict[str, object]]:
        centers = np.asarray(centers_ij, dtype="float64")
        if centers.ndim != 2 or centers.shape[1] != 2:
            raise ValueError("centers_ij must be an n-by-2 array of [center_i, center_j] values")
        if len(centers) == 0:
            return []
        matlab_centers = centers.copy()
        matlab_centers[:, 0] += 1.0
        matlab_centers[:, 1] += 1.0
        k = int(depth_index) + 1
        self.eng.workspace["hua_centers"] = self.matlab.double(matlab_centers.tolist())
        if self.params.boundary_mode == "velocity_streamline_contour":
            command = (
                "rows=streamline_gpu_batch_kernel("
                f"squeeze(u_day({k},:,:)),squeeze(v_day({k},:,:)),hua_centers,hua_radii,hua_params);"
            )
        else:
            command = (
                "rows=hua_circle_batch_kernel("
                f"squeeze(u_day({k},:,:)),squeeze(v_day({k},:,:)),hua_centers,hua_params);"
            )
        self.eng.eval(command, nargout=0)
        encoded = self.eng.eval("jsonencode(rows)", nargout=1)
        payload = json.loads(str(encoded))
        if isinstance(payload, dict):
            payloads = [payload]
        elif isinstance(payload, list):
            payloads = payload
        else:
            payloads = []
        if len(payloads) != len(centers):
            raise RuntimeError(f"MATLAB Hua backend returned {len(payloads)} rows for {len(centers)} centers.")
        return [_matlab_hua_payload_to_check(item, self.params) for item in payloads]

    def detect_day_many(
        self,
        seed_centers_ij: np.ndarray,
        *,
        target_degree: float,
        window_radius_cells: int,
        min_finite_fraction: float,
        stop_at_first_failed_layer: bool,
        keep_bridge_file: bool = False,
    ) -> list[dict[str, object]]:
        """Run same-day vertical continuation inside MATLAB and read a temporary MAT bridge."""
        centers = np.asarray(seed_centers_ij, dtype="float64")
        if centers.ndim != 2 or centers.shape[1] != 2:
            raise ValueError("seed_centers_ij must be an n-by-2 array of [seed_i, seed_j] values")
        if len(centers) == 0:
            return []
        matlab_centers = centers.copy()
        matlab_centers[:, 0] += 1.0
        matlab_centers[:, 1] += 1.0
        self.eng.workspace["day_seed_centers"] = self.matlab.double(matlab_centers.tolist())
        self.eng.workspace["day_target_degree"] = float(target_degree)
        self.eng.workspace["day_window_radius_cells"] = float(window_radius_cells)
        self.eng.workspace["day_min_finite_fraction"] = float(min_finite_fraction)
        self.eng.workspace["day_stop_at_first_failed"] = bool(stop_at_first_failed_layer)
        bridge_path = self.cache_dir / f"hua_day_bridge_{uuid.uuid4().hex}.mat"
        self.eng.workspace["day_bridge_path"] = str(bridge_path)
        command = (
            "hua_day_batch_to_mat("
            "day_bridge_path,u_day,v_day,hua_lon_vec,hua_lat_vec,day_seed_centers,hua_radii,hua_params,"
            "day_target_degree,day_window_radius_cells,day_min_finite_fraction,day_stop_at_first_failed);"
        )
        self.eng.eval(command, nargout=0)
        try:
            from scipy.io import loadmat

            payloads = _load_matlab_bridge_rows(loadmat(bridge_path, simplify_cells=True))
            rows: list[dict[str, object]] = []
            for item in payloads:
                rows.append({**item, "_hua_check": _matlab_hua_payload_to_check(item, self.params)})
            return rows
        finally:
            if not keep_bridge_file:
                try:
                    bridge_path.unlink(missing_ok=True)
                except OSError:
                    pass

    def close(self) -> None:
        try:
            self.eng.quit()
        except Exception:
            pass
        for handle in getattr(self, "_dll_handles", []):
            try:
                handle.close()
            except Exception:
                pass


def _payload_float(payload: dict[str, object], name: str, default: float = np.nan) -> float:
    value = payload.get(name, default)
    if value is None or value == "":
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _payload_bool(payload: dict[str, object], name: str, default: bool = False) -> bool:
    value = payload.get(name, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"true", "1", "yes"}
    return bool(value)


def _payload_text(payload: dict[str, object], name: str, default: str = "none") -> str:
    value = payload.get(name, default)
    return default if value is None else str(value)


def _load_matlab_bridge_rows(mat_data: dict[str, object]) -> list[dict[str, object]]:
    n_rows = int(float(mat_data.get("bridge_n_rows", 0) or 0))
    if n_rows <= 0:
        return []
    fields = [name for name in mat_data if not name.startswith("__") and name != "bridge_n_rows"]
    rows: list[dict[str, object]] = []
    for idx in range(n_rows):
        row: dict[str, object] = {}
        for field in fields:
            row[field] = _mat_bridge_item(mat_data[field], idx, n_rows)
        rows.append(row)
    return rows


def _mat_bridge_item(value: object, idx: int, n_rows: int) -> object:
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return value.item()
        flat = value.reshape(-1)
        if flat.size == n_rows:
            item = flat[idx]
            return item.item() if isinstance(item, np.generic) else item
        return value.tolist()
    if isinstance(value, (list, tuple)):
        if len(value) == n_rows:
            return value[idx]
        return value
    return value


def _matlab_hua_payload_to_check(payload: dict[str, object], params: DetectionParams) -> dict[str, object]:
    dominant_code = _payload_float(payload, "dominant_failure_code", -1.0)
    first_code = _payload_float(payload, "first_hard_failure_code", -1.0)
    boundary_mode = params.boundary_mode
    row: dict[str, object] = {
        "circle_passed": _payload_bool(payload, "circle_passed"),
        "hua_pass": _payload_bool(payload, "hua_pass"),
        "radius_cells": _payload_float(payload, "radius_cells"),
        "finite_fraction": _payload_float(payload, "finite_fraction"),
        "mean_circle_speed_ms": np.nan,
        "max_velocity_ratio": _payload_float(payload, "speed_ratio"),
        "max_angle_jump_deg": _payload_float(payload, "max_angle_jump_deg"),
        "direction_exception_count": _payload_float(payload, "direction_exception_count"),
        "positive_angle_diff_count": np.nan,
        "negative_angle_diff_count": np.nan,
        "direction_exception_limit": float(params.streamline_direction_exception_fraction)
        if boundary_mode == "velocity_streamline_contour"
        else np.nan,
        "boundary_monotonic_required": bool(
            boundary_mode == "velocity_streamline_contour" or params.require_boundary_monotonic_rotation
        ),
        "boundary_monotonic_passed": first_code != 9.0,
        "boundary_monotonic_exception_limit": float(params.streamline_direction_exception_fraction)
        if boundary_mode == "velocity_streamline_contour"
        else float(params.boundary_monotonic_exception_limit),
        "tangent_pass_fraction": _payload_float(payload, "tangent_fraction"),
        "symmetry_pass_fraction": _payload_float(payload, "symmetry_fraction"),
        "opposite_reversal_fraction": _payload_float(payload, "reversal_fraction"),
        "circulation_sign": _payload_float(payload, "circulation_sign"),
        "dominant_failure_code": dominant_code,
        "dominant_failure": _payload_text(payload, "dominant_failure", FAILURE_LABELS.get(int(dominant_code), "none") if dominant_code >= 0 else "none"),
        "first_hard_failure_code": first_code,
        "first_hard_failure": _payload_text(payload, "first_hard_failure", FAILURE_LABELS.get(int(first_code), "none") if first_code >= 0 else "none"),
        "hard_failure_order": _payload_text(payload, "hard_failure_order", "finite->velocity_ratio->angle_jump->boundary_monotonic->tangent->opposite_reversal"),
        "boundary_mode": boundary_mode,
        "accepted_radius_cells": _payload_float(payload, "accepted_radius_cells", 0.0),
    }
    if boundary_mode == "velocity_streamline_contour":
        row.update(
            {
                "streamline_closed": _payload_bool(payload, "streamline_closed"),
                "streamline_points": _payload_float(payload, "streamline_points", 0.0),
                "streamline_closure_error_cells": _payload_float(payload, "streamline_closure_error_cells"),
                "streamline_winding_turns": _payload_float(payload, "streamline_winding_turns"),
                "streamline_direction_exception_fraction": _payload_float(payload, "streamline_direction_exception_fraction"),
                "tangent_fraction_24deg": _payload_float(payload, "tangent_fraction_24deg"),
                "tangent_fraction_30deg": _payload_float(payload, "tangent_fraction_30deg"),
                "tangent_fraction_36deg": _payload_float(payload, "tangent_fraction_36deg"),
                "tangent_fraction_45deg": _payload_float(payload, "tangent_fraction_45deg"),
            }
        )
    else:
        row.update(
            {
                "streamline_closed": False,
                "streamline_points": np.nan,
                "streamline_closure_error_cells": np.nan,
                "streamline_winding_turns": np.nan,
                "streamline_direction_exception_fraction": np.nan,
                "failure_10_no_closed_streamline_count": 0.0,
            }
        )
    return row


def _parse_date(value: str) -> date:
    return datetime.strptime(str(value)[:10], "%Y-%m-%d").date()


def _date_range(start: date, end: date) -> list[date]:
    days: list[date] = []
    current = start
    while current <= end:
        days.append(current)
        current += timedelta(days=1)
    return days


def _wrap_lon_delta_deg(lon: np.ndarray, lon0: float) -> np.ndarray:
    return (lon - lon0 + 180.0) % 360.0 - 180.0


def _time_lookup(ds: Dataset) -> dict[date, int]:
    tvar = ds.variables["time"]
    times = num2date(tvar[:], units=tvar.units, calendar=getattr(tvar, "calendar", "standard"))
    return {date(int(t.year), int(t.month), int(t.day)): i for i, t in enumerate(times)}


def _candidate_cache_path(cache_dir: Path, day: date) -> Path:
    return cache_dir / f"candidates_{day:%Y%m%d}.csv"


def _load_cached_extrema(cache_dir: str | Path | None, day: date, max_candidates: int) -> pd.DataFrame | None:
    if not cache_dir:
        return None
    path = _candidate_cache_path(Path(cache_dir), day)
    if not path.exists():
        return None
    out = pd.read_csv(path)
    if out.empty:
        return out
    required = {"ssh_extremum_type", "seed_i", "seed_j", "ssh_value_m"}
    missing = required.difference(out.columns)
    if missing:
        raise ValueError(f"Candidate cache {path} is missing columns: {sorted(missing)}")
    out["seed_i"] = out["seed_i"].astype("int64")
    out["seed_j"] = out["seed_j"].astype("int64")
    out["ssh_value_m"] = out["ssh_value_m"].astype("float64")
    if "component_pixels" not in out.columns:
        out["component_pixels"] = 1
    if "abs_ssh_value_m" not in out.columns:
        out["abs_ssh_value_m"] = out["ssh_value_m"].abs()
    out = out.sort_values("abs_ssh_value_m", ascending=False).reset_index(drop=True)
    out["candidate_selection"] = "candidate_cache"
    out["tile_lon_min"] = np.nan
    out["tile_lon_max"] = np.nan
    out["tile_lat_min"] = np.nan
    out["tile_lat_max"] = np.nan
    out["tile_rank"] = np.arange(1, len(out) + 1, dtype=int)
    if max_candidates > 0:
        out = out.head(max_candidates).copy()
    return out


def _grid_spacing_km(lon: np.ndarray, lat: np.ndarray) -> tuple[float, float]:
    mid_lat = float(np.nanmedian(lat))
    dx = np.deg2rad(float(np.nanmedian(np.abs(np.diff(lon))))) * EARTH_RADIUS_M * math.cos(math.radians(mid_lat)) / 1000.0
    dy = np.deg2rad(float(np.nanmedian(np.abs(np.diff(lat))))) * EARTH_RADIUS_M / 1000.0
    return abs(dx), abs(dy)


def _select_extrema_candidates(
    extrema: pd.DataFrame,
    lon: np.ndarray,
    lat: np.ndarray,
    *,
    selection: str,
    max_candidates: int,
    tile_lon_deg: float,
    tile_lat_deg: float,
    tile_top_n: int,
) -> pd.DataFrame:
    if extrema.empty:
        return extrema
    selection = str(selection).lower().strip()
    out = extrema.sort_values("abs_ssh_value_m", ascending=False).reset_index(drop=True)
    if selection == "global_topn":
        out["candidate_selection"] = "global_topn"
        out["tile_lon_min"] = np.nan
        out["tile_lon_max"] = np.nan
        out["tile_lat_min"] = np.nan
        out["tile_lat_max"] = np.nan
        out["tile_rank"] = np.arange(1, len(out) + 1, dtype=int)
        if max_candidates > 0:
            out = out.head(max_candidates).copy()
        return out.reset_index(drop=True)
    if selection != "tile_topn":
        raise ValueError(f"Unsupported candidate selection mode: {selection}")
    if tile_lon_deg <= 0 or tile_lat_deg <= 0 or tile_top_n <= 0:
        raise ValueError("tile_topn requires positive tile_lon_deg, tile_lat_deg, and tile_top_n")

    seed_i = out["seed_i"].astype(int).to_numpy()
    seed_j = out["seed_j"].astype(int).to_numpy()
    seed_lon = np.asarray(lon[seed_i], dtype="float64")
    seed_lat = np.asarray(lat[seed_j], dtype="float64")
    tile_lon_min = np.floor(seed_lon / float(tile_lon_deg)) * float(tile_lon_deg)
    tile_lat_min = np.floor(seed_lat / float(tile_lat_deg)) * float(tile_lat_deg)
    out["tile_lon_min"] = tile_lon_min
    out["tile_lon_max"] = tile_lon_min + float(tile_lon_deg)
    out["tile_lat_min"] = tile_lat_min
    out["tile_lat_max"] = tile_lat_min + float(tile_lat_deg)
    out["candidate_selection"] = "tile_topn"
    selected_parts = []
    group_cols = ["tile_lon_min", "tile_lat_min"]
    for _, part in out.groupby(group_cols, sort=True, dropna=False):
        part = part.sort_values("abs_ssh_value_m", ascending=False).head(int(tile_top_n)).copy()
        part["tile_rank"] = np.arange(1, len(part) + 1, dtype=int)
        selected_parts.append(part)
    if not selected_parts:
        return out.iloc[0:0].copy()
    selected = pd.concat(selected_parts, ignore_index=True)
    selected = selected.sort_values(["tile_lon_min", "tile_lat_min", "tile_rank", "abs_ssh_value_m"], ascending=[True, True, True, False])
    return selected.reset_index(drop=True)


def _local_extrema(
    zos: np.ndarray,
    window: int,
    *,
    lon: np.ndarray,
    lat: np.ndarray,
    selection: str,
    max_candidates: int,
    tile_lon_deg: float,
    tile_lat_deg: float,
    tile_top_n: int,
) -> pd.DataFrame:
    finite = np.isfinite(zos)
    fill_max = np.where(finite, zos, -np.inf)
    fill_min = np.where(finite, zos, np.inf)
    max_mask = finite & (fill_max == ndimage.maximum_filter(fill_max, size=window, mode="nearest"))
    min_mask = finite & (fill_min == ndimage.minimum_filter(fill_min, size=window, mode="nearest"))
    rows = []
    structure = np.ones((3, 3), dtype=bool)
    for kind, mask in (("ssh_max", max_mask), ("ssh_min", min_mask)):
        labels, count = ndimage.label(mask, structure=structure)
        for label in range(1, count + 1):
            yy, xx = np.where(labels == label)
            n = len(xx)
            if n == 0 or n > 100:
                continue
            values = zos[yy, xx]
            if kind == "ssh_max":
                pick = int(np.nanargmax(values))
            else:
                pick = int(np.nanargmin(values))
            rows.append(
                {
                    "ssh_extremum_type": kind,
                    "seed_i": int(xx[pick]),
                    "seed_j": int(yy[pick]),
                    "ssh_value_m": float(values[pick]),
                    "component_pixels": int(n),
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["abs_ssh_value_m"] = out["ssh_value_m"].abs()
    return _select_extrema_candidates(
        out,
        lon,
        lat,
        selection=selection,
        max_candidates=max_candidates,
        tile_lon_deg=tile_lon_deg,
        tile_lat_deg=tile_lat_deg,
        tile_top_n=tile_top_n,
    )


def _circle_offsets(radius_cells: int) -> list[tuple[int, int]]:
    points: list[tuple[int, int]] = []
    n = max(16, int(round(8 * radius_cells)))
    for theta in np.linspace(-math.pi / 2.0, 3.0 * math.pi / 2.0, n, endpoint=False):
        point = (int(round(radius_cells * math.cos(theta))), int(round(radius_cells * math.sin(theta))))
        if not points or points[-1] != point:
            points.append(point)
    if len(points) > 1 and points[0] == points[-1]:
        points.pop()
    return points


def _angle_diff(a: np.ndarray | float, b: np.ndarray | float) -> np.ndarray | float:
    return (np.asarray(a) - np.asarray(b) + math.pi) % (2.0 * math.pi) - math.pi


def _iterative_speed_min(speed: np.ndarray, start_i: int, start_j: int, *, max_steps: int = 40) -> tuple[int, int, float, int]:
    ii = int(np.clip(start_i, 0, speed.shape[1] - 1))
    jj = int(np.clip(start_j, 0, speed.shape[0] - 1))
    last = (-1, -1)
    steps = 0
    while (ii, jj) != last and steps < max_steps:
        last = (ii, jj)
        x0, x1 = max(0, ii - 2), min(speed.shape[1], ii + 3)
        y0, y1 = max(0, jj - 2), min(speed.shape[0], jj + 3)
        window = speed[y0:y1, x0:x1]
        if not np.isfinite(window).any():
            break
        local = int(np.nanargmin(window))
        wy, wx = np.unravel_index(local, window.shape)
        ii = x0 + wx
        jj = y0 + wy
        steps += 1
    val = float(speed[jj, ii]) if np.isfinite(speed[jj, ii]) else np.nan
    return ii, jj, val, steps


def _seeded_speed_min(speed: np.ndarray, seed_i: int, seed_j: int, radius_cells: int) -> tuple[int, int, float, int]:
    radius = max(0, int(radius_cells))
    x0 = max(0, int(seed_i) - radius)
    x1 = min(speed.shape[1] - 1, int(seed_i) + radius)
    y0 = max(0, int(seed_j) - radius)
    y1 = min(speed.shape[0] - 1, int(seed_j) + radius)
    if x1 < x0 or y1 < y0:
        return _iterative_speed_min(speed, seed_i, seed_j)
    window = speed[y0 : y1 + 1, x0 : x1 + 1]
    yy, xx = np.ogrid[y0 : y1 + 1, x0 : x1 + 1]
    mask = (xx - int(seed_i)) ** 2 + (yy - int(seed_j)) ** 2 <= radius**2
    mask &= np.isfinite(window)
    if not np.any(mask):
        return _iterative_speed_min(speed, seed_i, seed_j)
    flat = np.where(mask.ravel())[0]
    pick = int(flat[np.nanargmin(window.ravel()[flat])])
    local_j, local_i = np.unravel_index(pick, window.shape)
    ii = x0 + int(local_i)
    jj = y0 + int(local_j)
    return _iterative_speed_min(speed, int(ii), int(jj))


def _fill_nearest_finite(field: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    finite = np.isfinite(field)
    if not finite.any():
        return np.asarray(field, dtype="float64"), finite
    if finite.all():
        return np.asarray(field, dtype="float64"), finite
    _, indices = ndimage.distance_transform_edt(~finite, return_indices=True)
    filled = np.asarray(field, dtype="float64")[tuple(indices)]
    return filled, finite


def _interp_1d_from_fraction(coord: np.ndarray, index_fraction: float) -> float:
    grid = np.arange(coord.size, dtype="float64")
    return float(np.interp(float(index_fraction), grid, np.asarray(coord, dtype="float64")))


def _quadratic_speed_surface(
    speed_window: np.ndarray,
    finite: np.ndarray,
    xi: np.ndarray,
    yj: np.ndarray,
) -> tuple[np.ndarray | None, str]:
    yy0, xx0 = np.indices(speed_window.shape, dtype="float64")
    x = xx0[finite].ravel()
    y = yy0[finite].ravel()
    z = np.square(np.asarray(speed_window, dtype="float64")[finite].ravel())
    if z.size < 9:
        return None, "quadratic_insufficient_points"
    normal = np.zeros((6, 6), dtype="float64")
    rhs = np.zeros(6, dtype="float64")
    for xx, yy, zz in zip(x, y, z):
        row = (xx * xx, yy * yy, xx * yy, xx, yy, 1.0)
        for ii in range(6):
            rhs[ii] += row[ii] * zz
            for jj in range(ii, 6):
                normal[ii, jj] += row[ii] * row[jj]
    for ii in range(6):
        for jj in range(ii):
            normal[ii, jj] = normal[jj, ii]
    scale = float(np.nanmax(np.abs(np.diag(normal)))) if normal.size else 1.0
    normal = normal + np.eye(normal.shape[0], dtype="float64") * max(scale, 1.0) * 1.0e-10
    coeff = _solve_small_linear_system(normal, rhs)
    if coeff is None:
        return None, "quadratic_lstsq_failed"
    a, b, c, *_ = coeff
    h00 = 2.0 * float(a)
    h11 = 2.0 * float(b)
    det = h00 * h11 - float(c) * float(c)
    if not np.all(np.isfinite([h00, h11, det])) or h00 <= 0.0 or det <= 0.0:
        return None, "quadratic_not_convex"
    yy, xx = np.meshgrid(yj, xi, indexing="ij")
    dense_sq = (
        coeff[0] * xx * xx
        + coeff[1] * yy * yy
        + coeff[2] * xx * yy
        + coeff[3] * xx
        + coeff[4] * yy
        + coeff[5]
    )
    if not np.isfinite(dense_sq).any():
        return None, "quadratic_no_finite"
    return np.sqrt(np.maximum(dense_sq, 0.0)), "quadratic_speed2_1_24deg"


def _solve_small_linear_system(a: np.ndarray, b: np.ndarray) -> np.ndarray | None:
    """Solve a tiny dense system without calling platform LAPACK.

    The refined-center fit is called many times on 6x6 normal equations. Using
    a local Gaussian elimination avoids recurring `np.linalg.lstsq` overhead and
    sidesteps fragile Windows BLAS/LAPACK crashes seen in this environment.
    """
    mat = np.asarray(a, dtype="float64").copy()
    rhs = np.asarray(b, dtype="float64").copy()
    n = int(rhs.size)
    if mat.shape != (n, n) or not (np.isfinite(mat).all() and np.isfinite(rhs).all()):
        return None
    for col in range(n):
        pivot = col + int(np.argmax(np.abs(mat[col:, col])))
        pivot_value = float(mat[pivot, col])
        if not np.isfinite(pivot_value) or abs(pivot_value) < 1.0e-14:
            return None
        if pivot != col:
            mat[[col, pivot], :] = mat[[pivot, col], :]
            rhs[[col, pivot]] = rhs[[pivot, col]]
        for row in range(col + 1, n):
            factor = mat[row, col] / mat[col, col]
            if factor == 0.0:
                continue
            mat[row, col:] -= factor * mat[col, col:]
            rhs[row] -= factor * rhs[col]
    out = np.zeros(n, dtype="float64")
    for row in range(n - 1, -1, -1):
        denom = float(mat[row, row])
        if not np.isfinite(denom) or abs(denom) < 1.0e-14:
            return None
        accum = 0.0
        for col in range(row + 1, n):
            accum += float(mat[row, col]) * float(out[col])
        out[row] = (rhs[row] - accum) / denom
    return out


def _refine_speed_min_subgrid(
    speed: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    lon: np.ndarray,
    lat: np.ndarray,
    center_i: int,
    center_j: int,
    *,
    target_degree: float,
    window_radius_cells: int,
    min_finite_fraction: float,
) -> dict[str, object]:
    grid_lon = float(lon[center_i])
    grid_lat = float(lat[center_j])
    grid_speed = float(speed[center_j, center_i]) if np.isfinite(speed[center_j, center_i]) else np.nan
    fallback = {
        "center_i_refined": float(center_i),
        "center_j_refined": float(center_j),
        "center_lon_refined": grid_lon,
        "center_lat_refined": grid_lat,
        "refined_speed_ms": grid_speed,
        "refined_offset_km": 0.0,
        "refined_ok": False,
        "subgrid_fit_quality": "fallback_grid",
    }
    if target_degree <= 0 or window_radius_cells < 1:
        return fallback | {"subgrid_fit_quality": "disabled"}
    x0 = max(0, int(center_i) - int(window_radius_cells))
    x1 = min(speed.shape[1] - 1, int(center_i) + int(window_radius_cells))
    y0 = max(0, int(center_j) - int(window_radius_cells))
    y1 = min(speed.shape[0] - 1, int(center_j) + int(window_radius_cells))
    if x1 <= x0 or y1 <= y0:
        return fallback | {"subgrid_fit_quality": "window_too_small"}
    speed_window = np.asarray(speed[y0 : y1 + 1, x0 : x1 + 1], dtype="float64")
    u_window = np.asarray(u[y0 : y1 + 1, x0 : x1 + 1], dtype="float64")
    v_window = np.asarray(v[y0 : y1 + 1, x0 : x1 + 1], dtype="float64")
    finite = np.isfinite(speed_window) & np.isfinite(u_window) & np.isfinite(v_window)
    if float(finite.mean()) < float(min_finite_fraction):
        return fallback | {"subgrid_fit_quality": "insufficient_finite"}

    filled_u, finite_u = _fill_nearest_finite(u_window)
    filled_v, finite_v = _fill_nearest_finite(v_window)
    finite_mask = finite & finite_u & finite_v
    dlon = float(np.nanmedian(np.abs(np.diff(lon)))) if lon.size > 1 else 0.0
    dlat = float(np.nanmedian(np.abs(np.diff(lat)))) if lat.size > 1 else 0.0
    if dlon <= 0 or dlat <= 0:
        return fallback | {"subgrid_fit_quality": "invalid_grid_spacing"}
    step_i = max(float(target_degree) / dlon, 1.0e-3)
    step_j = max(float(target_degree) / dlat, 1.0e-3)
    xi = np.arange(0.0, float(x1 - x0) + 0.5 * step_i, step_i, dtype="float64")
    yj = np.arange(0.0, float(y1 - y0) + 0.5 * step_j, step_j, dtype="float64")
    if xi.size < 3 or yj.size < 3:
        return fallback | {"subgrid_fit_quality": "refined_grid_too_small"}
    yy, xx = np.meshgrid(yj, xi, indexing="ij")
    coords = np.vstack([yy.ravel(), xx.ravel()])
    dense_u = ndimage.map_coordinates(filled_u, coords, order=1, mode="nearest").reshape(yy.shape)
    dense_v = ndimage.map_coordinates(filled_v, coords, order=1, mode="nearest").reshape(yy.shape)
    dense = np.hypot(dense_u, dense_v)
    valid_weight = ndimage.map_coordinates(finite_mask.astype("float64"), coords, order=1, mode="nearest").reshape(yy.shape)
    dense = np.where(valid_weight >= 0.999, dense, np.nan)
    quadratic_dense, quadratic_quality = _quadratic_speed_surface(speed_window, finite, xi, yj)
    if quadratic_dense is not None:
        dense = np.where(np.isfinite(dense), quadratic_dense, np.nan)
        fit_quality = quadratic_quality
    else:
        fit_quality = "uv_vector_linear_interp_1_24deg"
    if not np.isfinite(dense).any():
        return fallback | {"subgrid_fit_quality": "no_refined_finite"}
    local_pick = int(np.nanargmin(dense))
    pick_j, pick_i = np.unravel_index(local_pick, dense.shape)
    if pick_i in (0, dense.shape[1] - 1) or pick_j in (0, dense.shape[0] - 1):
        return fallback | {"subgrid_fit_quality": "minimum_on_refined_boundary"}

    refined_i = float(x0 + xi[pick_i])
    refined_j = float(y0 + yj[pick_j])
    refined_lon = _interp_1d_from_fraction(lon, refined_i)
    refined_lat = _interp_1d_from_fraction(lat, refined_j)
    dx_km, dy_km = _grid_spacing_km(lon, lat)
    offset_km = math.hypot((refined_i - center_i) * dx_km, (refined_j - center_j) * dy_km)
    return {
        "center_i_refined": refined_i,
        "center_j_refined": refined_j,
        "center_lon_refined": refined_lon,
        "center_lat_refined": refined_lat,
        "refined_speed_ms": float(dense[pick_j, pick_i]),
        "refined_offset_km": float(offset_km),
        "refined_ok": True,
        "subgrid_fit_quality": fit_quality,
    }


def _circle_check(
    u: np.ndarray,
    v: np.ndarray,
    center_i: float,
    center_j: float,
    radius_cells: int,
    params: DetectionParams,
) -> dict[str, float | bool | str]:
    offsets = _circle_offsets(radius_cells)
    uu = np.full(len(offsets), np.nan, dtype="float64")
    vv = np.full(len(offsets), np.nan, dtype="float64")
    use_grid_sampling = abs(float(center_i) - round(float(center_i))) < 1.0e-9 and abs(float(center_j) - round(float(center_j))) < 1.0e-9
    if use_grid_sampling:
        ii = np.asarray([int(round(float(center_i))) + dx for dx, _ in offsets], dtype=int)
        jj = np.asarray([int(round(float(center_j))) + dy for _, dy in offsets], dtype=int)
        inside = (ii >= 0) & (ii < u.shape[1]) & (jj >= 0) & (jj < u.shape[0])
        uu[inside] = u[jj[inside], ii[inside]]
        vv[inside] = v[jj[inside], ii[inside]]
    else:
        for idx, (dx, dy) in enumerate(offsets):
            uu[idx], vv[idx] = _sample_uv_at(u, v, float(center_i) + float(dx), float(center_j) + float(dy))
    sp = np.hypot(uu, vv)
    finite = np.isfinite(sp) & (sp > 1e-10)
    failure_counts = {k: 0 for k in FAILURE_LABELS}
    if finite.mean() < params.min_finite_fraction:
        failure_counts[0] += int((~finite).sum())

    angles = np.arctan2(vv, uu)
    angle_diffs: list[float] = []
    max_ratio = 0.0
    max_angle = 0.0
    positive_diffs = 0
    negative_diffs = 0
    rotation_failed = False
    for n in range(len(offsets)):
        m = (n + 1) % len(offsets)
        if not (finite[n] and finite[m]):
            rotation_failed = True
            failure_counts[0] += 1
            continue
        ratio = float(sp[m] / sp[n])
        max_ratio = max(max_ratio, ratio, 1.0 / ratio if ratio > 0 else np.inf)
        if ratio > params.speed_ratio_max or ratio < 1.0 / params.speed_ratio_max:
            rotation_failed = True
            failure_counts[1] += 1
        dtheta = float(_angle_diff(angles[n], angles[m]))
        angle_diffs.append(dtheta)
        max_angle = max(max_angle, abs(math.degrees(dtheta)))
        if abs(math.degrees(dtheta)) > params.angle_jump_max_deg:
            rotation_failed = True
            failure_counts[2] += 1
        if dtheta > 0:
            positive_diffs += 1
        elif dtheta < 0:
            negative_diffs += 1
    max_exceptions = int(math.floor(radius_cells / 5.0) + 1 + params.direction_exception_extra)
    direction_exceptions = min(positive_diffs, negative_diffs)
    monotonic_exception_limit = (
        int(params.boundary_monotonic_exception_limit)
        if params.require_boundary_monotonic_rotation
        else max_exceptions
    )
    boundary_monotonic_passed = direction_exceptions <= monotonic_exception_limit
    if direction_exceptions > max_exceptions:
        rotation_failed = True
        failure_counts[4] += int(direction_exceptions - max_exceptions)
    if params.require_boundary_monotonic_rotation and not boundary_monotonic_passed:
        rotation_failed = True
        failure_counts[9] += int(direction_exceptions - monotonic_exception_limit)

    dx = np.asarray([p[0] for p in offsets], dtype="float64")
    dy = np.asarray([p[1] for p in offsets], dtype="float64")
    th = np.arctan2(dy, dx)
    tx = -np.sin(th)
    ty = np.cos(th)
    tangent_cos = np.abs((uu * tx + vv * ty) / np.maximum(sp, 1e-12))
    tangent_ok = finite & (tangent_cos >= math.cos(math.radians(params.tangent_tolerance_deg)))
    tangent_fraction = float(tangent_ok.sum() / finite.sum()) if finite.any() else 0.0
    if tangent_fraction < params.min_tangent_fraction:
        rotation_failed = True
        failure_counts[7] += int(max(1, round((params.min_tangent_fraction - tangent_fraction) * len(offsets))))

    half = len(offsets) // 2
    symmetry_ok = 0
    symmetry_total = 0
    reversal_ok = 0
    reversal_total = 0
    for n in range(half):
        m = (n + half) % len(offsets)
        if not (finite[n] and finite[m]):
            continue
        diff = abs(float(_angle_diff(angles[n], angles[m])))
        symmetry_total += 1
        if abs(diff - math.pi) <= math.radians(params.symmetry_tolerance_deg):
            symmetry_ok += 1
        reversal_total += 1
        if uu[n] * uu[m] + vv[n] * vv[m] < 0:
            reversal_ok += 1
    symmetry_fraction = float(symmetry_ok / symmetry_total) if symmetry_total else 0.0
    reversal_fraction = float(reversal_ok / reversal_total) if reversal_total else 0.0
    if symmetry_total and symmetry_ok < symmetry_total:
        failure_counts[6] += int(symmetry_total - symmetry_ok)
    if reversal_fraction < params.min_reversal_fraction:
        rotation_failed = True
        failure_counts[8] += int(max(1, round((params.min_reversal_fraction - reversal_fraction) * max(reversal_total, 1))))

    tangential = uu * tx + vv * ty
    circulation_sign = float(np.sign(np.nanmedian(tangential[finite]))) if finite.any() else np.nan
    dominant = max(failure_counts.items(), key=lambda kv: kv[1])[0] if sum(failure_counts.values()) else -1
    first_hard_code = -1
    hard_failures = {
        0: finite.mean() < params.min_finite_fraction,
        1: failure_counts[1] > 0,
        2: failure_counts[2] > 0,
        9: params.require_boundary_monotonic_rotation and not boundary_monotonic_passed,
        7: tangent_fraction < params.min_tangent_fraction,
        8: reversal_fraction < params.min_reversal_fraction,
    }
    for code, _label in HARD_FAILURE_ORDER:
        if hard_failures.get(code, False):
            first_hard_code = code
            break
    return {
        "circle_passed": bool(not rotation_failed),
        "radius_cells": float(radius_cells),
        "finite_fraction": float(finite.mean()),
        "mean_circle_speed_ms": float(np.nanmean(sp[finite])) if finite.any() else np.nan,
        "max_velocity_ratio": float(max_ratio),
        "max_angle_jump_deg": float(max_angle),
        "direction_exception_count": float(direction_exceptions),
        "positive_angle_diff_count": float(positive_diffs),
        "negative_angle_diff_count": float(negative_diffs),
        "direction_exception_limit": float(max_exceptions),
        "boundary_monotonic_required": bool(params.require_boundary_monotonic_rotation),
        "boundary_monotonic_passed": bool(boundary_monotonic_passed),
        "boundary_monotonic_exception_limit": float(monotonic_exception_limit),
        "tangent_pass_fraction": tangent_fraction,
        "symmetry_pass_fraction": symmetry_fraction,
        "opposite_reversal_fraction": reversal_fraction,
        "circulation_sign": circulation_sign,
        "dominant_failure_code": float(dominant),
        "dominant_failure": FAILURE_LABELS.get(int(dominant), "none") if dominant >= 0 else "none",
        "first_hard_failure_code": float(first_hard_code),
        "first_hard_failure": FAILURE_LABELS.get(int(first_hard_code), "none") if first_hard_code >= 0 else "none",
        "hard_failure_order": "finite->velocity_ratio->angle_jump->boundary_monotonic->tangent->opposite_reversal",
        **{f"failure_{k}_{label}_count": float(failure_counts[k]) for k, label in FAILURE_LABELS.items()},
    }


def _sample_uv_at(u: np.ndarray, v: np.ndarray, x: float, y: float) -> tuple[float, float]:
    if x < 0.0 or y < 0.0 or x >= u.shape[1] - 1 or y >= u.shape[0] - 1:
        return np.nan, np.nan
    x0 = int(math.floor(float(x)))
    y0 = int(math.floor(float(y)))
    wx = float(x) - float(x0)
    wy = float(y) - float(y0)
    weights = (
        (1.0 - wx) * (1.0 - wy),
        wx * (1.0 - wy),
        (1.0 - wx) * wy,
        wx * wy,
    )
    u00, u10, u01, u11 = float(u[y0, x0]), float(u[y0, x0 + 1]), float(u[y0 + 1, x0]), float(u[y0 + 1, x0 + 1])
    v00, v10, v01, v11 = float(v[y0, x0]), float(v[y0, x0 + 1]), float(v[y0 + 1, x0]), float(v[y0 + 1, x0 + 1])
    vals = (u00, u10, u01, u11, v00, v10, v01, v11)
    if not all(np.isfinite(value) for value in vals):
        return np.nan, np.nan
    uu = weights[0] * u00 + weights[1] * u10 + weights[2] * u01 + weights[3] * u11
    vv = weights[0] * v00 + weights[1] * v10 + weights[2] * v01 + weights[3] * v11
    return float(uu), float(vv)


def _trace_streamline_candidate(
    u: np.ndarray,
    v: np.ndarray,
    center_i: float,
    center_j: float,
    radius_cells: int,
    start_angle: float,
    direction: float,
    params: DetectionParams,
) -> dict[str, object]:
    start_x = float(center_i) + float(radius_cells) * math.cos(float(start_angle))
    start_y = float(center_j) + float(radius_cells) * math.sin(float(start_angle))
    points: list[tuple[float, float]] = [(start_x, start_y)]
    prev_angle = math.atan2(start_y - float(center_j), start_x - float(center_i))
    winding = 0.0
    closed = False
    failed_finite = False
    step = max(float(params.streamline_step_cells), 0.05)
    for _ in range(max(8, int(params.streamline_max_steps))):
        x, y = points[-1]
        uu, vv = _sample_uv_at(u, v, x, y)
        sp = math.hypot(uu, vv) if np.isfinite(uu) and np.isfinite(vv) else np.nan
        if not np.isfinite(sp) or sp <= 1.0e-10:
            failed_finite = True
            break
        ux = float(direction) * uu / sp
        uy = float(direction) * vv / sp
        mid_x = x + 0.5 * step * ux
        mid_y = y + 0.5 * step * uy
        mu, mv = _sample_uv_at(u, v, mid_x, mid_y)
        msp = math.hypot(mu, mv) if np.isfinite(mu) and np.isfinite(mv) else np.nan
        if np.isfinite(msp) and msp > 1.0e-10:
            ux = float(direction) * mu / msp
            uy = float(direction) * mv / msp
        new_x = x + step * ux
        new_y = y + step * uy
        if new_x < 0.0 or new_y < 0.0 or new_x > u.shape[1] - 1 or new_y > u.shape[0] - 1:
            failed_finite = True
            break
        new_angle = math.atan2(new_y - float(center_j), new_x - float(center_i))
        winding += float(_angle_diff(new_angle, prev_angle))
        prev_angle = new_angle
        points.append((new_x, new_y))
        closure = math.hypot(new_x - start_x, new_y - start_y)
        if (
            len(points) >= int(params.streamline_min_points)
            and abs(winding) >= 2.0 * math.pi * float(params.streamline_min_winding_turns)
            and closure <= float(params.streamline_closure_tolerance_cells)
        ):
            closed = True
            break
    arr = np.asarray(points, dtype="float64")
    if arr.size == 0:
        arr = np.empty((0, 2), dtype="float64")
    closure_error = float(math.hypot(arr[-1, 0] - start_x, arr[-1, 1] - start_y)) if len(arr) else np.inf
    radial = np.hypot(arr[:, 0] - float(center_i), arr[:, 1] - float(center_j)) if len(arr) else np.asarray([], dtype="float64")
    radial_cv = float(np.nanstd(radial) / max(np.nanmean(radial), 1.0e-6)) if radial.size else np.inf
    winding_turns = float(abs(winding) / (2.0 * math.pi))
    score = closure_error + 2.0 * abs(1.0 - min(winding_turns, 1.5)) + radial_cv
    if failed_finite:
        score += 5.0
    return {
        "points": arr,
        "closed": bool(closed),
        "closure_error_cells": closure_error,
        "winding_turns": winding_turns,
        "radial_cv": radial_cv,
        "score": float(score),
    }


def _best_streamline_contour(
    u: np.ndarray,
    v: np.ndarray,
    center_i: float,
    center_j: float,
    radius_cells: int,
    params: DetectionParams,
) -> dict[str, object] | None:
    best: dict[str, object] | None = None
    n_angles = max(1, int(params.streamline_start_angles))
    for angle in np.linspace(0.0, 2.0 * math.pi, n_angles, endpoint=False):
        for direction in (1.0, -1.0):
            candidate = _trace_streamline_candidate(u, v, center_i, center_j, radius_cells, float(angle), direction, params)
            if best is None or float(candidate["score"]) < float(best["score"]):
                best = candidate
    if best is None or not bool(best["closed"]):
        return None
    return best


def _streamline_contour_check(
    u: np.ndarray,
    v: np.ndarray,
    center_i: float,
    center_j: float,
    radius_cells: int,
    params: DetectionParams,
) -> dict[str, float | bool | str]:
    failure_counts = {k: 0 for k in FAILURE_LABELS}
    contour = _best_streamline_contour(u, v, center_i, center_j, radius_cells, params)
    if contour is None:
        failure_counts[10] = 1
        dominant = 10
        return {
            "circle_passed": False,
            "radius_cells": float(radius_cells),
            "finite_fraction": 0.0,
            "mean_circle_speed_ms": np.nan,
            "max_velocity_ratio": np.nan,
            "max_angle_jump_deg": np.nan,
            "direction_exception_count": np.nan,
            "positive_angle_diff_count": np.nan,
            "negative_angle_diff_count": np.nan,
            "direction_exception_limit": np.nan,
            "boundary_monotonic_required": True,
            "boundary_monotonic_passed": False,
            "boundary_monotonic_exception_limit": float(params.streamline_direction_exception_fraction),
            "tangent_pass_fraction": 0.0,
            "symmetry_pass_fraction": 0.0,
            "opposite_reversal_fraction": 0.0,
            "circulation_sign": np.nan,
            "dominant_failure_code": float(dominant),
            "dominant_failure": FAILURE_LABELS[dominant],
            "first_hard_failure_code": float(dominant),
            "first_hard_failure": FAILURE_LABELS[dominant],
            "hard_failure_order": "no_closed_streamline->finite->velocity_ratio->angle_jump->boundary_monotonic->tangent->opposite_reversal",
            "boundary_mode": "velocity_streamline_contour",
            "streamline_closed": False,
            "streamline_points": 0.0,
            "streamline_closure_error_cells": np.nan,
            "streamline_winding_turns": 0.0,
            "streamline_direction_exception_fraction": 1.0,
            "tangent_fraction_24deg": 0.0,
            "tangent_fraction_30deg": 0.0,
            "tangent_fraction_36deg": 0.0,
            "tangent_fraction_45deg": 0.0,
            **{f"failure_{k}_{label}_count": float(failure_counts[k]) for k, label in FAILURE_LABELS.items()},
        }

    points = np.asarray(contour["points"], dtype="float64")
    uu = np.empty(len(points), dtype="float64")
    vv = np.empty(len(points), dtype="float64")
    for idx, (x, y) in enumerate(points):
        uu[idx], vv[idx] = _sample_uv_at(u, v, float(x), float(y))
    sp = np.hypot(uu, vv)
    finite = np.isfinite(sp) & (sp > 1e-10)
    if finite.mean() < params.min_finite_fraction:
        failure_counts[0] += int((~finite).sum())

    x_s = ndimage.gaussian_filter1d(points[:, 0], sigma=1.0, mode="wrap")
    y_s = ndimage.gaussian_filter1d(points[:, 1], sigma=1.0, mode="wrap")
    tx = np.gradient(x_s)
    ty = np.gradient(y_s)
    tnorm = np.hypot(tx, ty)
    tx = tx / np.maximum(tnorm, 1.0e-12)
    ty = ty / np.maximum(tnorm, 1.0e-12)
    tangent_cos = np.abs((uu * tx + vv * ty) / np.maximum(sp, 1.0e-12))
    tangent_ok = finite & (tangent_cos >= math.cos(math.radians(params.tangent_tolerance_deg)))
    tangent_fraction = float(tangent_ok.sum() / finite.sum()) if finite.any() else 0.0
    tangent_fraction_by_tolerance = {
        int(tol): float((finite & (tangent_cos >= math.cos(math.radians(float(tol))))).sum() / finite.sum()) if finite.any() else 0.0
        for tol in (24, 30, 36, 45)
    }
    if tangent_fraction < params.min_tangent_fraction:
        failure_counts[7] += int(max(1, round((params.min_tangent_fraction - tangent_fraction) * len(points))))

    polar = np.unwrap(np.arctan2(points[:, 1] - float(center_j), points[:, 0] - float(center_i)))
    dpolar = np.diff(polar)
    positive_diffs = int(np.sum(dpolar > 1.0e-9))
    negative_diffs = int(np.sum(dpolar < -1.0e-9))
    direction_exceptions = min(positive_diffs, negative_diffs)
    direction_total = max(1, positive_diffs + negative_diffs)
    direction_exception_fraction = float(direction_exceptions / direction_total)
    boundary_monotonic_passed = direction_exception_fraction <= float(params.streamline_direction_exception_fraction)
    if not boundary_monotonic_passed:
        failure_counts[9] += int(max(1, round((direction_exception_fraction - float(params.streamline_direction_exception_fraction)) * len(points))))

    angles = np.arctan2(vv, uu)
    max_ratio = 0.0
    max_angle = 0.0
    for n in range(len(points)):
        m = (n + 1) % len(points)
        if not (finite[n] and finite[m]):
            failure_counts[0] += 1
            continue
        ratio = float(sp[m] / sp[n])
        max_ratio = max(max_ratio, ratio, 1.0 / ratio if ratio > 0 else np.inf)
        if ratio > params.speed_ratio_max or ratio < 1.0 / params.speed_ratio_max:
            failure_counts[1] += 1
        dtheta = float(_angle_diff(angles[n], angles[m]))
        max_angle = max(max_angle, abs(math.degrees(dtheta)))
        if abs(math.degrees(dtheta)) > params.angle_jump_max_deg:
            failure_counts[2] += 1

    half = len(points) // 2
    symmetry_ok = 0
    symmetry_total = 0
    reversal_ok = 0
    reversal_total = 0
    for n in range(half):
        m = (n + half) % len(points)
        if not (finite[n] and finite[m]):
            continue
        diff = abs(float(_angle_diff(angles[n], angles[m])))
        symmetry_total += 1
        if abs(diff - math.pi) <= math.radians(params.symmetry_tolerance_deg):
            symmetry_ok += 1
        reversal_total += 1
        if uu[n] * uu[m] + vv[n] * vv[m] < 0:
            reversal_ok += 1
    symmetry_fraction = float(symmetry_ok / symmetry_total) if symmetry_total else 0.0
    reversal_fraction = float(reversal_ok / reversal_total) if reversal_total else 0.0
    if symmetry_total and symmetry_ok < symmetry_total:
        failure_counts[6] += int(symmetry_total - symmetry_ok)
    if reversal_fraction < params.min_reversal_fraction:
        failure_counts[8] += int(max(1, round((params.min_reversal_fraction - reversal_fraction) * max(reversal_total, 1))))

    rotation_failed = False
    if finite.mean() < params.min_finite_fraction:
        rotation_failed = True
    if failure_counts[1] or failure_counts[2] or failure_counts[8]:
        rotation_failed = True
    if not boundary_monotonic_passed:
        rotation_failed = True
    if tangent_fraction < params.min_tangent_fraction:
        rotation_failed = True

    tangential = uu * tx + vv * ty
    circulation_sign = float(np.sign(np.nanmedian(tangential[finite]))) if finite.any() else np.nan
    dominant = max(failure_counts.items(), key=lambda kv: kv[1])[0] if sum(failure_counts.values()) else -1
    first_hard_code = -1
    hard_failures = {
        0: finite.mean() < params.min_finite_fraction,
        1: failure_counts[1] > 0,
        2: failure_counts[2] > 0,
        9: not boundary_monotonic_passed,
        7: tangent_fraction < params.min_tangent_fraction,
        8: reversal_fraction < params.min_reversal_fraction,
    }
    for code, _label in HARD_FAILURE_ORDER:
        if hard_failures.get(code, False):
            first_hard_code = code
            break
    return {
        "circle_passed": bool(not rotation_failed),
        "radius_cells": float(radius_cells),
        "finite_fraction": float(finite.mean()),
        "mean_circle_speed_ms": float(np.nanmean(sp[finite])) if finite.any() else np.nan,
        "max_velocity_ratio": float(max_ratio),
        "max_angle_jump_deg": float(max_angle),
        "direction_exception_count": float(direction_exceptions),
        "positive_angle_diff_count": float(positive_diffs),
        "negative_angle_diff_count": float(negative_diffs),
        "direction_exception_limit": float(params.streamline_direction_exception_fraction),
        "boundary_monotonic_required": True,
        "boundary_monotonic_passed": bool(boundary_monotonic_passed),
        "boundary_monotonic_exception_limit": float(params.streamline_direction_exception_fraction),
        "tangent_pass_fraction": tangent_fraction,
        "symmetry_pass_fraction": symmetry_fraction,
        "opposite_reversal_fraction": reversal_fraction,
        "circulation_sign": circulation_sign,
        "dominant_failure_code": float(dominant),
        "dominant_failure": FAILURE_LABELS.get(int(dominant), "none") if dominant >= 0 else "none",
        "first_hard_failure_code": float(first_hard_code),
        "first_hard_failure": FAILURE_LABELS.get(int(first_hard_code), "none") if first_hard_code >= 0 else "none",
        "hard_failure_order": "finite->velocity_ratio->angle_jump->boundary_monotonic->tangent->opposite_reversal",
        "boundary_mode": "velocity_streamline_contour",
        "streamline_closed": True,
        "streamline_points": float(len(points)),
        "streamline_closure_error_cells": float(contour["closure_error_cells"]),
        "streamline_winding_turns": float(contour["winding_turns"]),
        "streamline_direction_exception_fraction": direction_exception_fraction,
        "tangent_fraction_24deg": tangent_fraction_by_tolerance[24],
        "tangent_fraction_30deg": tangent_fraction_by_tolerance[30],
        "tangent_fraction_36deg": tangent_fraction_by_tolerance[36],
        "tangent_fraction_45deg": tangent_fraction_by_tolerance[45],
        **{f"failure_{k}_{label}_count": float(failure_counts[k]) for k, label in FAILURE_LABELS.items()},
    }


def _hua_verify_radius(
    u: np.ndarray,
    v: np.ndarray,
    center_i: float,
    center_j: float,
    params: DetectionParams,
) -> dict[str, float | bool | str]:
    best: dict[str, float | bool | str] | None = None
    first_fail: dict[str, float | bool | str] | None = None
    for radius in range(params.start_radius_cells, params.max_radius_cells + 1):
        if params.boundary_mode == "velocity_streamline_contour":
            row = _streamline_contour_check(u, v, center_i, center_j, radius, params)
        else:
            row = _circle_check(u, v, center_i, center_j, radius, params)
            row = {
                **row,
                "boundary_mode": "circle_strict_original",
                "streamline_closed": False,
                "streamline_points": np.nan,
                "streamline_closure_error_cells": np.nan,
                "streamline_winding_turns": np.nan,
                "streamline_direction_exception_fraction": np.nan,
                "failure_10_no_closed_streamline_count": 0.0,
            }
        if bool(row["circle_passed"]):
            best = row
        else:
            first_fail = row
            break
    source_row = best if best is not None else first_fail
    if source_row is None:
        source_row = {"circle_passed": False, "radius_cells": np.nan, "dominant_failure": "no_circle"}
    source_row = dict(source_row)
    source_row["hua_pass"] = bool(best is not None)
    source_row["accepted_radius_cells"] = float(source_row["radius_cells"]) if best is not None else 0.0
    return source_row


def _object_voxels_for_layer(
    u: np.ndarray,
    v: np.ndarray,
    lon: np.ndarray,
    lat: np.ndarray,
    depth_m: float,
    *,
    day: date,
    object_id: str,
    depth_index: int,
    center_i: int,
    center_j: int,
    radius_cells: float,
    polarity: str,
) -> list[dict[str, object]]:
    """Return the finite component connected to the Hua center inside its accepted circle."""
    if not np.isfinite(radius_cells) or radius_cells <= 0:
        return []
    radius = int(math.ceil(float(radius_cells)))
    x0, x1 = max(0, center_i - radius), min(u.shape[1], center_i + radius + 1)
    y0, y1 = max(0, center_j - radius), min(u.shape[0], center_j + radius + 1)
    if x0 >= x1 or y0 >= y1:
        return []

    yy, xx = np.ogrid[y0:y1, x0:x1]
    circle = (xx - center_i) ** 2 + (yy - center_j) ** 2 <= float(radius_cells) ** 2
    finite = np.isfinite(u[y0:y1, x0:x1]) & np.isfinite(v[y0:y1, x0:x1])
    mask = circle & finite
    cj = center_j - y0
    ci = center_i - x0
    if cj < 0 or cj >= mask.shape[0] or ci < 0 or ci >= mask.shape[1] or not mask[cj, ci]:
        return []
    labels, _ = ndimage.label(mask, structure=np.ones((3, 3), dtype=bool))
    component_label = int(labels[cj, ci])
    if component_label <= 0:
        return []
    comp_y, comp_x = np.where(labels == component_label)
    rows: list[dict[str, object]] = []
    ny, nx = u.shape
    for ly, lx in zip(comp_y.tolist(), comp_x.tolist()):
        jj = int(y0 + ly)
        ii = int(x0 + lx)
        rows.append(
            {
                "date": day.isoformat(),
                "hua_object_id": object_id,
                "depth_index": int(depth_index),
                "i": ii,
                "j": jj,
                "lon": float(lon[ii]),
                "lat": float(lat[jj]),
                "depth_m": float(depth_m),
                "polarity": polarity,
                "accepted_radius_cells": float(radius_cells),
                "node_key_3d": int(depth_index * ny * nx + jj * nx + ii),
                "node_key_2d": int(jj * nx + ii),
            }
        )
    return rows


def _parse_float_list(value: str, default: list[float]) -> list[float]:
    text = str(value or "").strip()
    if not text:
        return list(default)
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def _pass_rate_breakdown(centers: pd.DataFrame, params: DetectionParams) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    required_reasons = [
        "invalid_velocity",
        "velocity_ratio",
        "angle_jump",
        "boundary_monotonic_rotation",
        "tangent_alignment",
        "opposite_reversal",
        "no_closed_streamline",
        "symmetry",
    ]
    reason_aliases = {
        "boundary_monotonic_rotation": "modified_boundary_failure"
        if params.boundary_mode == "velocity_streamline_contour"
        else "boundary_monotonic_rotation",
        "tangent_alignment": "modified_tangent_failure"
        if params.boundary_mode == "velocity_streamline_contour"
        else "tangent_alignment",
    }
    scopes = {
        "all_layers": centers,
        "surface": centers[centers["depth_index"].eq(0)] if "depth_index" in centers.columns else centers.iloc[0:0],
    }
    for scope, part in scopes.items():
        total = int(len(part))
        passed = int(part["hua_pass"].astype(bool).sum()) if total and "hua_pass" in part.columns else 0
        rows.append(
            {
                "scope": scope,
                "reason": "hua_pass",
                "count": passed,
                "total": total,
                "fraction_of_total": float(passed / total) if total else 0.0,
                "fraction_of_failed": np.nan,
                "boundary_mode": params.boundary_mode,
                "reason_alias": "hua_pass",
            }
        )
        failed = part[~part["hua_pass"].astype(bool)] if total and "hua_pass" in part.columns else part.iloc[0:0]
        fail_total = int(len(failed))
        reason_column = "first_hard_failure" if "first_hard_failure" in failed.columns else "dominant_failure"
        counts = failed[reason_column].fillna("none").value_counts(dropna=False).to_dict() if fail_total else {}
        for reason in required_reasons:
            count = int(counts.get(reason, 0))
            rows.append(
                {
                    "scope": scope,
                    "reason": str(reason),
                    "reason_alias": reason_aliases.get(str(reason), str(reason)),
                    "count": count,
                    "total": total,
                    "fraction_of_total": float(count / total) if total else 0.0,
                    "fraction_of_failed": float(count / fail_total) if fail_total else 0.0,
                    "boundary_mode": params.boundary_mode,
                }
            )
    return pd.DataFrame(rows)


def _streamline_pass_mask(part: pd.DataFrame, *, tangent_fraction: float, tangent_tolerance_deg: float, direction_exception_fraction: float, params: DetectionParams) -> pd.Series:
    tolerance_key = int(round(float(tangent_tolerance_deg)))
    tangent_column = f"tangent_fraction_{tolerance_key}deg"
    tangent_values = part[tangent_column].astype(float) if tangent_column in part.columns else part["tangent_pass_fraction"].astype(float)
    finite = part["finite_fraction"].astype(float).ge(float(params.min_finite_fraction))
    velocity_ratio = part["max_velocity_ratio"].astype(float).le(float(params.speed_ratio_max))
    angle_jump = part["max_angle_jump_deg"].astype(float).le(float(params.angle_jump_max_deg))
    reversal = part["opposite_reversal_fraction"].astype(float).ge(float(params.min_reversal_fraction))
    closed = part["streamline_closed"].fillna(False).astype(bool)
    boundary = part["streamline_direction_exception_fraction"].astype(float).le(float(direction_exception_fraction))
    tangent = tangent_values.ge(float(tangent_fraction))
    return closed & finite & velocity_ratio & angle_jump & reversal & boundary & tangent


def _write_detection_diagnostics(output_dir: Path, centers: pd.DataFrame, params: DetectionParams, args: argparse.Namespace) -> None:
    breakdown = _pass_rate_breakdown(centers, params)
    breakdown.to_csv(output_dir / "pass_rate_breakdown.csv", index=False)
    (output_dir / "pass_rate_breakdown.json").write_text(
        json.dumps(breakdown.to_dict(orient="records"), ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    if params.boundary_mode != "velocity_streamline_contour" or centers.empty:
        return
    baseline_pass_fraction = np.nan
    baseline_pass_rows = np.nan
    baseline_layer_rows = np.nan
    baseline_path = getattr(args, "baseline_summary_path", None)
    if baseline_path:
        try:
            baseline_summary = json.loads(Path(baseline_path).read_text(encoding="utf-8"))
            baseline_pass_fraction = float(baseline_summary.get("pass_fraction", np.nan))
            baseline_pass_rows = float(baseline_summary.get("n_pass_layers", np.nan))
            baseline_layer_rows = float(baseline_summary.get("n_center_rows", np.nan))
        except Exception:
            baseline_pass_fraction = np.nan
    fractions = _parse_float_list(args.sensitivity_tangent_fractions, [0.50, 0.60, 0.70])
    tolerances = _parse_float_list(args.sensitivity_tangent_tolerances_deg, [24.0, 30.0, 36.0, 45.0])
    direction_fractions = _parse_float_list(args.sensitivity_direction_exception_fractions, [0.05, 0.10, 0.15])
    rows: list[dict[str, object]] = []
    for min_fraction in fractions:
        for tolerance in tolerances:
            for direction_fraction in direction_fractions:
                passed = _streamline_pass_mask(
                    centers,
                    tangent_fraction=float(min_fraction),
                    tangent_tolerance_deg=float(tolerance),
                    direction_exception_fraction=float(direction_fraction),
                    params=params,
                )
                total = int(len(centers))
                rows.append(
                    {
                        "boundary_mode": params.boundary_mode,
                        "baseline_summary_path": str(baseline_path) if baseline_path else "",
                        "baseline_pass_fraction": baseline_pass_fraction,
                        "baseline_hua_pass_rows": baseline_pass_rows,
                        "baseline_layer_attempt_rows": baseline_layer_rows,
                        "min_tangent_fraction": float(min_fraction),
                        "tangent_tolerance_deg": float(tolerance),
                        "streamline_direction_exception_fraction": float(direction_fraction),
                        "layer_attempt_rows": total,
                        "hua_pass_rows_proxy": int(passed.sum()),
                        "pass_fraction_proxy": float(passed.mean()) if total else 0.0,
                        "pass_fraction_delta_vs_baseline": float(passed.mean() - baseline_pass_fraction)
                        if total and np.isfinite(baseline_pass_fraction)
                        else np.nan,
                        "pass_rows_delta_vs_baseline": float(passed.sum() - baseline_pass_rows)
                        if np.isfinite(baseline_pass_rows)
                        else np.nan,
                        "no_closed_streamline_rows": int((~centers["streamline_closed"].fillna(False).astype(bool)).sum()),
                        "modified_boundary_failure_rows": int((centers["streamline_closed"].fillna(False).astype(bool) & centers["streamline_direction_exception_fraction"].astype(float).gt(float(direction_fraction))).sum()),
                        "modified_tangent_failure_rows": int((centers[f"tangent_fraction_{int(round(float(tolerance)))}deg"].astype(float).lt(float(min_fraction))).sum())
                        if f"tangent_fraction_{int(round(float(tolerance)))}deg" in centers.columns
                        else int((centers["tangent_pass_fraction"].astype(float).lt(float(min_fraction))).sum()),
                        "invalid_velocity_rows": int((centers["finite_fraction"].astype(float).lt(float(params.min_finite_fraction))).sum()),
                        "velocity_ratio_rows": int((centers["max_velocity_ratio"].astype(float).gt(float(params.speed_ratio_max))).sum()),
                        "angle_jump_rows": int((centers["max_angle_jump_deg"].astype(float).gt(float(params.angle_jump_max_deg))).sum()),
                        "opposite_reversal_rows": int((centers["opposite_reversal_fraction"].astype(float).lt(float(params.min_reversal_fraction))).sum()),
                    }
                )
    sensitivity = pd.DataFrame(rows)
    sensitivity.to_csv(output_dir / "sensitivity_matrix.csv", index=False)
    sensitivity.to_parquet(output_dir / "sensitivity_matrix.parquet", engine=DEFAULT_PARQUET_ENGINE, index=False)


def _extremum_polarity(extremum: str, lat_value: float, circulation_sign: float) -> str:
    if np.isfinite(circulation_sign) and circulation_sign != 0:
        # Positive tangential circulation is counterclockwise. In the Southern
        # Hemisphere cyclonic rotation is clockwise, so use f sign.
        f_sign = 1.0 if lat_value >= 0 else -1.0
        return "cyclonic" if circulation_sign == f_sign else "anticyclonic"
    if lat_value < 0:
        return "cyclonic" if extremum == "ssh_min" else "anticyclonic"
    return "cyclonic" if extremum == "ssh_min" else "anticyclonic"


def _format_year_template(template: str, year: int) -> str:
    return str(template).format(year=year)


def _configure_var_chunk_cache(ds: Dataset, variable_names: tuple[str, ...], cache_mb: int) -> None:
    cache_size = int(cache_mb) * 1024 * 1024
    if cache_size <= 0:
        return
    for name in variable_names:
        if name not in ds.variables:
            continue
        var = ds.variables[name]
        try:
            chunking = var.chunking()
            if isinstance(chunking, list) and chunking:
                chunk_values = 1
                for item in chunking:
                    chunk_values *= int(item)
                dtype_size = np.dtype(var.dtype).itemsize
                chunk_bytes = max(1, chunk_values * dtype_size)
                nelems = max(1009, min(1_000_003, cache_size // chunk_bytes * 2 + 1))
            else:
                nelems = 1009
            var.set_var_chunk_cache(size=cache_size, nelems=int(nelems), preemption=0.75)
        except Exception:
            continue


def _load_year_arrays(args: argparse.Namespace, year: int) -> tuple[Dataset, Dataset | None, np.ndarray, np.ndarray, np.ndarray]:
    filter_root = Path(args.filter_root)
    raw_root = Path(args.raw_root)
    filt = Dataset(filter_root / _format_year_template(args.filter_template, year))
    raw_path = raw_root / _format_year_template(args.raw_template, year)
    raw = Dataset(raw_path) if raw_path.exists() else None
    _configure_var_chunk_cache(filt, ("zos_glor", "uo_glor", "vo_glor"), int(args.netcdf_chunk_cache_mb))
    if raw is not None:
        _configure_var_chunk_cache(raw, ("zos_glor", "uo_glor", "vo_glor"), max(1, int(args.netcdf_chunk_cache_mb) // 4))
    lon = np.asarray(filt.variables["longitude"][:], dtype="float64")
    lat = np.asarray(filt.variables["latitude"][:], dtype="float64")
    depth = np.asarray(filt.variables["depth"][:], dtype="float64")
    return filt, raw, lon, lat, depth


def _detect_day(
    day: date,
    filt: Dataset,
    raw: Dataset | None,
    lon: np.ndarray,
    lat: np.ndarray,
    depth: np.ndarray,
    params: DetectionParams,
    args: argparse.Namespace,
    output_dir: Path,
    matlab_backend: MatlabHuaBackend | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    time_index = _time_lookup(filt)[day]
    zos = np.asarray(filt.variables["zos_glor"][time_index, :, :], dtype="float64")
    cached_extrema = _load_cached_extrema(args.candidate_cache_dir, day, args.max_candidates_per_day)
    extrema = (
        cached_extrema
        if cached_extrema is not None
        else _local_extrema(
            zos,
            params.ssh_window_cells,
            lon=lon,
            lat=lat,
            selection=args.candidate_selection,
            max_candidates=args.max_candidates_per_day,
            tile_lon_deg=args.tile_lon_deg,
            tile_lat_deg=args.tile_lat_deg,
            tile_top_n=args.tile_top_n,
        )
    )
    if extrema.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(columns=OBJECT_VOXEL_COLUMNS)

    if float(args.max_depth_m) <= 0.0 or not np.isfinite(float(args.max_depth_m)):
        max_depth_idx = len(depth)
    else:
        max_depth_idx = int(np.searchsorted(depth, args.max_depth_m, side="right"))
        max_depth_idx = min(max_depth_idx, len(depth))
    depth_indices = np.arange(max_depth_idx, dtype=int)
    dx_km, dy_km = _grid_spacing_km(lon, lat)
    centers_rows: list[dict[str, object]] = []
    circle_rows: list[dict[str, object]] = []
    structure_rows: list[dict[str, object]] = []
    voxel_rows: list[dict[str, object]] = []

    u_day = None
    v_day = None
    if args.preload_day_uv or matlab_backend is not None:
        # ACC windows are small enough that loading one day of u/v once is
        # cheaper than thousands of HDF5 layer reads during candidate checks.
        u_day = np.asarray(filt.variables["uo_glor"][time_index, :max_depth_idx, :, :], dtype="float32")
        v_day = np.asarray(filt.variables["vo_glor"][time_index, :max_depth_idx, :, :], dtype="float32")
    if matlab_backend is not None:
        if u_day is None or v_day is None:
            raise RuntimeError("MATLAB Hua backend requires preloaded daily u/v arrays.")
        matlab_backend.load_day(day, u_day, v_day)
        matlab_backend.load_grid(lon, lat)

    if u_day is None:
        u0 = np.asarray(filt.variables["uo_glor"][time_index, 0, :, :], dtype="float32")
        v0 = np.asarray(filt.variables["vo_glor"][time_index, 0, :, :], dtype="float32")
    else:
        u0 = u_day[0]
        v0 = v_day[0]
    speed0 = np.hypot(u0, v0)
    if matlab_backend is not None:
        states: list[dict[str, object]] = []
        for seed_order, seed in extrema.iterrows():
            seed_i = int(seed["seed_i"])
            seed_j = int(seed["seed_j"])
            states.append(
                {
                    "seed_order": int(seed_order),
                    "seed": seed,
                    "seed_i": seed_i,
                    "seed_j": seed_j,
                    "object_id": f"{day:%Y%m%d}_{int(seed_order):05d}",
                }
            )
        day_rows = matlab_backend.detect_day_many(
            np.asarray([(state["seed_i"], state["seed_j"]) for state in states], dtype="float64"),
            target_degree=float(args.subgrid_target_degree),
            window_radius_cells=int(args.subgrid_window_radius_cells),
            min_finite_fraction=float(args.subgrid_min_finite_fraction),
            stop_at_first_failed_layer=bool(args.stop_at_first_failed_layer),
            keep_bridge_file=bool(args.keep_matlab_bridge_files),
        )
        for item in day_rows:
            check = item.pop("_hua_check")
            state_idx = int(_payload_float(item, "state_index", 0.0)) - 1
            if state_idx < 0 or state_idx >= len(states):
                continue
            state = states[state_idx]
            seed = state["seed"]
            seed_i = int(state["seed_i"])
            seed_j = int(state["seed_j"])
            depth_index = int(_payload_float(item, "depth_index", 0.0))
            if depth_index < 0 or depth_index >= len(depth):
                continue
            center_i = int(round(_payload_float(item, "speed_min_i_grid", np.nan)))
            center_j = int(round(_payload_float(item, "speed_min_j_grid", np.nan)))
            if center_i < 0 or center_i >= len(lon) or center_j < 0 or center_j >= len(lat):
                continue
            hua_center_i = _payload_float(item, "hua_center_i", np.nan)
            hua_center_j = _payload_float(item, "hua_center_j", np.nan)
            lat_value = _payload_float(item, "center_lat_refined", np.nan)
            lon_value = _payload_float(item, "center_lon_refined", np.nan)
            polarity = _extremum_polarity(str(seed["ssh_extremum_type"]), lat_value, float(check.get("circulation_sign", np.nan)))
            stopped = not bool(check["hua_pass"])
            row = {
                "date": day.isoformat(),
                "hua_object_id": str(state["object_id"]),
                "seed_order": int(state["seed_order"]),
                "ssh_extremum_type": str(seed["ssh_extremum_type"]),
                "polarity": polarity,
                "time_index": int(time_index),
                "candidate_selection": str(seed.get("candidate_selection", args.candidate_selection)),
                "tile_lon_min": float(seed.get("tile_lon_min", np.nan)),
                "tile_lon_max": float(seed.get("tile_lon_max", np.nan)),
                "tile_lat_min": float(seed.get("tile_lat_min", np.nan)),
                "tile_lat_max": float(seed.get("tile_lat_max", np.nan)),
                "tile_rank": int(seed.get("tile_rank", int(state["seed_order"]) + 1)) if pd.notna(seed.get("tile_rank", np.nan)) else int(state["seed_order"]) + 1,
                "depth_index": int(depth_index),
                "depth_m": float(depth[depth_index]),
                "seed_i": seed_i,
                "seed_j": seed_j,
                "seed_lon": float(lon[seed_i]),
                "seed_lat": float(lat[seed_j]),
                "ssh_value_m": float(seed["ssh_value_m"]),
                "speed_min_i": int(round(hua_center_i)) if np.isfinite(hua_center_i) else center_i,
                "speed_min_j": int(round(hua_center_j)) if np.isfinite(hua_center_j) else center_j,
                "speed_min_i_grid": int(center_i),
                "speed_min_j_grid": int(center_j),
                "center_lon_grid": float(lon[center_i]),
                "center_lat_grid": float(lat[center_j]),
                "center_lon": lon_value,
                "center_lat": lat_value,
                "center_i_refined": hua_center_i,
                "center_j_refined": hua_center_j,
                "hua_center_i": hua_center_i,
                "hua_center_j": hua_center_j,
                "hua_center_source": "refined_search_stage",
                "refined_before_hua": True,
                "center_refinement_stage": "pre_hua",
                "center_lon_refined": lon_value,
                "center_lat_refined": lat_value,
                "center_x_from_seed_km": float((hua_center_i - seed_i) * dx_km) if np.isfinite(hua_center_i) else np.nan,
                "center_y_from_seed_km": float((hua_center_j - seed_j) * dy_km) if np.isfinite(hua_center_j) else np.nan,
                "center_x_from_seed_grid_km": float((center_i - seed_i) * dx_km),
                "center_y_from_seed_grid_km": float((center_j - seed_j) * dy_km),
                "center_speed_ms": _payload_float(item, "refined_speed_ms", np.nan),
                "center_speed_grid_ms": _payload_float(item, "center_speed_grid_ms", np.nan),
                "refined_offset_km": _payload_float(item, "refined_offset_km", np.nan),
                "refined_ok": _payload_bool(item, "refined_ok", False),
                "subgrid_fit_quality": _payload_text(item, "subgrid_fit_quality", "matlab_unknown"),
                "local_min_steps": int(_payload_float(item, "local_min_steps", 0.0)),
                "stopped_after_failure": bool(stopped),
                "matlab_backend_mode": "day_batch",
                "matlab_batch_size": int(_payload_float(item, "matlab_batch_size", np.nan)),
                "matlab_refine_batch_size": int(_payload_float(item, "matlab_refine_batch_size", np.nan)),
                "matlab_unique_refine_batch_size": int(_payload_float(item, "matlab_unique_refine_batch_size", np.nan)),
                "matlab_unique_hua_batch_size": int(_payload_float(item, "matlab_unique_hua_batch_size", np.nan)),
                "matlab_refine_dedup_count": int(_payload_float(item, "matlab_refine_dedup_count", 0.0)),
                "matlab_hua_dedup_count": int(_payload_float(item, "matlab_hua_dedup_count", 0.0)),
                "matlab_day_batch": _payload_bool(item, "matlab_day_batch", True),
                **check,
            }
            centers_rows.append(row)
            circle_rows.append({k: v for k, v in row.items() if k not in {"center_lon", "center_lat"}})
            if bool(check["hua_pass"]):
                structure_rows.append(
                    {
                        "date": day.isoformat(),
                        "hua_object_id": str(state["object_id"]),
                        "depth_index": int(depth_index),
                        "depth_m": float(depth[depth_index]),
                        "center_lon": lon_value,
                        "center_lat": lat_value,
                        "center_lon_grid": float(lon[center_i]),
                        "center_lat_grid": float(lat[center_j]),
                        "center_lon_refined": lon_value,
                        "center_lat_refined": lat_value,
                        "refined_ok": _payload_bool(item, "refined_ok", False),
                        "refined_offset_km": _payload_float(item, "refined_offset_km", np.nan),
                        "radius_km": float(check["accepted_radius_cells"]) * float(np.nanmean([dx_km, dy_km])),
                        "polarity": polarity,
                    }
                )
        centers = pd.DataFrame(centers_rows)
        circle = pd.DataFrame(circle_rows)
        structures = pd.DataFrame(structure_rows)
        voxels = pd.DataFrame(voxel_rows, columns=OBJECT_VOXEL_COLUMNS)
        if args.write_day_figures and not centers.empty:
            _plot_day_summary(day, centers, zos, lon, lat, output_dir / "figures")
        return centers, circle, structures, voxels

        active = list(range(len(states)))
        for depth_index in depth_indices:
            if not active:
                break
            u = u_day[depth_index] if u_day is not None else np.asarray(filt.variables["uo_glor"][time_index, depth_index, :, :], dtype="float32")
            v = v_day[depth_index] if v_day is not None else np.asarray(filt.variables["vo_glor"][time_index, depth_index, :, :], dtype="float32")
            speed = np.hypot(u, v)
            pending: list[dict[str, object]] = []
            grid_centers: list[tuple[float, float]] = []
            for state_idx in active:
                state = states[state_idx]
                if int(depth_index) == 0:
                    center_i = int(state["surface_center_i"])
                    center_j = int(state["surface_center_j"])
                    center_speed = float(state["surface_center_speed"])
                    min_steps = int(state["surface_min_steps"])
                else:
                    center_i, center_j, center_speed, min_steps = _seeded_speed_min(
                        speed,
                        int(state["prev_i"]),
                        int(state["prev_j"]),
                        params.deep_search_cells,
                    )
                grid_lat_value = float(lat[center_j])
                grid_lon_value = float(lon[center_i])
                grid_centers.append((float(center_i), float(center_j)))
                pending.append(
                    {
                        "state_idx": state_idx,
                        "center_i": int(center_i),
                        "center_j": int(center_j),
                        "center_speed": float(center_speed),
                        "min_steps": int(min_steps),
                        "grid_lon_value": grid_lon_value,
                        "grid_lat_value": grid_lat_value,
                    }
                )
            unique_grid_centers: list[tuple[float, float]] = []
            grid_inverse: list[int] = []
            grid_lookup: dict[tuple[float, float], int] = {}
            for center in grid_centers:
                key = (float(center[0]), float(center[1]))
                unique_idx = grid_lookup.get(key)
                if unique_idx is None:
                    unique_idx = len(unique_grid_centers)
                    grid_lookup[key] = unique_idx
                    unique_grid_centers.append(key)
                grid_inverse.append(unique_idx)
            unique_refined_rows = matlab_backend.refine_many(
                int(depth_index),
                np.asarray(unique_grid_centers, dtype="float64"),
                target_degree=float(args.subgrid_target_degree),
                window_radius_cells=int(args.subgrid_window_radius_cells),
                min_finite_fraction=float(args.subgrid_min_finite_fraction),
            )
            refined_rows = [unique_refined_rows[idx] for idx in grid_inverse]
            hua_centers = [
                (float(refined["center_i_refined"]), float(refined["center_j_refined"]))
                for refined in refined_rows
            ]
            unique_hua_centers: list[tuple[float, float]] = []
            hua_inverse: list[int] = []
            hua_lookup: dict[tuple[float, float], int] = {}
            for center in hua_centers:
                key = (float(center[0]), float(center[1]))
                unique_idx = hua_lookup.get(key)
                if unique_idx is None:
                    unique_idx = len(unique_hua_centers)
                    hua_lookup[key] = unique_idx
                    unique_hua_centers.append(key)
                hua_inverse.append(unique_idx)
            unique_checks = matlab_backend.check_many(int(depth_index), np.asarray(unique_hua_centers, dtype="float64"))
            checks = [unique_checks[idx] for idx in hua_inverse]
            next_active: list[int] = []
            for item, refined, check in zip(pending, refined_rows, checks, strict=True):
                state_idx = int(item["state_idx"])
                state = states[state_idx]
                seed = state["seed"]
                seed_i = int(state["seed_i"])
                seed_j = int(state["seed_j"])
                center_i = int(item["center_i"])
                center_j = int(item["center_j"])
                center_speed = float(item["center_speed"])
                hua_center_i = float(refined["center_i_refined"])
                hua_center_j = float(refined["center_j_refined"])
                stopped = not bool(check["hua_pass"])
                if bool(check["hua_pass"]):
                    state["prev_i"] = int(np.clip(round(float(refined["center_i_refined"])), 0, speed.shape[1] - 1))
                    state["prev_j"] = int(np.clip(round(float(refined["center_j_refined"])), 0, speed.shape[0] - 1))
                    next_active.append(state_idx)
                elif not args.stop_at_first_failed_layer:
                    next_active.append(state_idx)
                lat_value = float(refined["center_lat_refined"])
                lon_value = float(refined["center_lon_refined"])
                polarity = _extremum_polarity(str(seed["ssh_extremum_type"]), lat_value, float(check.get("circulation_sign", np.nan)))
                row = {
                    "date": day.isoformat(),
                    "hua_object_id": str(state["object_id"]),
                    "seed_order": int(state["seed_order"]),
                    "ssh_extremum_type": str(seed["ssh_extremum_type"]),
                    "polarity": polarity,
                    "time_index": int(time_index),
                    "candidate_selection": str(seed.get("candidate_selection", args.candidate_selection)),
                    "tile_lon_min": float(seed.get("tile_lon_min", np.nan)),
                    "tile_lon_max": float(seed.get("tile_lon_max", np.nan)),
                    "tile_lat_min": float(seed.get("tile_lat_min", np.nan)),
                    "tile_lat_max": float(seed.get("tile_lat_max", np.nan)),
                    "tile_rank": int(seed.get("tile_rank", int(state["seed_order"]) + 1)) if pd.notna(seed.get("tile_rank", np.nan)) else int(state["seed_order"]) + 1,
                    "depth_index": int(depth_index),
                    "depth_m": float(depth[depth_index]),
                    "seed_i": seed_i,
                    "seed_j": seed_j,
                    "seed_lon": float(lon[seed_i]),
                    "seed_lat": float(lat[seed_j]),
                    "ssh_value_m": float(seed["ssh_value_m"]),
                    "speed_min_i": int(round(float(refined["center_i_refined"]))),
                    "speed_min_j": int(round(float(refined["center_j_refined"]))),
                    "speed_min_i_grid": int(center_i),
                    "speed_min_j_grid": int(center_j),
                    "center_lon_grid": float(item["grid_lon_value"]),
                    "center_lat_grid": float(item["grid_lat_value"]),
                    "center_lon": lon_value,
                    "center_lat": lat_value,
                    "center_i_refined": float(refined["center_i_refined"]),
                    "center_j_refined": float(refined["center_j_refined"]),
                    "hua_center_i": hua_center_i,
                    "hua_center_j": hua_center_j,
                    "hua_center_source": "refined_search_stage",
                    "refined_before_hua": True,
                    "center_refinement_stage": "pre_hua",
                    "center_lon_refined": float(refined["center_lon_refined"]),
                    "center_lat_refined": float(refined["center_lat_refined"]),
                    "center_x_from_seed_km": float((float(refined["center_i_refined"]) - seed_i) * dx_km),
                    "center_y_from_seed_km": float((float(refined["center_j_refined"]) - seed_j) * dy_km),
                    "center_x_from_seed_grid_km": float((center_i - seed_i) * dx_km),
                    "center_y_from_seed_grid_km": float((center_j - seed_j) * dy_km),
                    "center_speed_ms": float(refined["refined_speed_ms"] if bool(refined["refined_ok"]) else center_speed),
                    "center_speed_grid_ms": float(center_speed),
                    "refined_offset_km": float(refined["refined_offset_km"]),
                    "refined_ok": bool(refined["refined_ok"]),
                    "subgrid_fit_quality": str(refined["subgrid_fit_quality"]),
                    "local_min_steps": int(item["min_steps"]),
                    "stopped_after_failure": bool(stopped),
                    "matlab_batch_size": int(len(pending)),
                    "matlab_refine_batch_size": int(len(pending)),
                    "matlab_unique_refine_batch_size": int(len(unique_grid_centers)),
                    "matlab_unique_hua_batch_size": int(len(unique_hua_centers)),
                    "matlab_refine_dedup_count": int(len(pending) - len(unique_grid_centers)),
                    "matlab_hua_dedup_count": int(len(pending) - len(unique_hua_centers)),
                    **check,
                }
                centers_rows.append(row)
                circle_rows.append({k: v for k, v in row.items() if k not in {"center_lon", "center_lat"}})
                if bool(check["hua_pass"]):
                    structure_rows.append(
                        {
                            "date": day.isoformat(),
                            "hua_object_id": str(state["object_id"]),
                            "depth_index": int(depth_index),
                            "depth_m": float(depth[depth_index]),
                            "center_lon": lon_value,
                            "center_lat": lat_value,
                            "center_lon_grid": float(item["grid_lon_value"]),
                            "center_lat_grid": float(item["grid_lat_value"]),
                            "center_lon_refined": float(refined["center_lon_refined"]),
                            "center_lat_refined": float(refined["center_lat_refined"]),
                            "refined_ok": bool(refined["refined_ok"]),
                            "refined_offset_km": float(refined["refined_offset_km"]),
                            "radius_km": float(check["accepted_radius_cells"]) * float(np.nanmean([dx_km, dy_km])),
                            "polarity": polarity,
                        }
                    )
                    if args.write_object_voxels:
                        voxel_rows.extend(
                            _object_voxels_for_layer(
                                u,
                                v,
                                lon,
                                lat,
                                float(depth[depth_index]),
                                day=day,
                                object_id=str(state["object_id"]),
                                depth_index=int(depth_index),
                                center_i=int(center_i),
                                center_j=int(center_j),
                                radius_cells=float(check["accepted_radius_cells"]),
                                polarity=polarity,
                            )
                        )
            active = next_active
        centers = pd.DataFrame(centers_rows)
        circle = pd.DataFrame(circle_rows)
        structures = pd.DataFrame(structure_rows)
        voxels = pd.DataFrame(voxel_rows, columns=OBJECT_VOXEL_COLUMNS)
        if args.write_day_figures and not centers.empty:
            _plot_day_summary(day, centers, zos, lon, lat, output_dir / "figures")
        return centers, circle, structures, voxels

    for seed_order, seed in extrema.iterrows():
        seed_i = int(seed["seed_i"])
        seed_j = int(seed["seed_j"])
        center_i, center_j, center_speed, min_steps = _seeded_speed_min(
            speed0,
            seed_i,
            seed_j,
            params.surface_search_cells,
        )
        prev_i, prev_j = center_i, center_j
        object_id = f"{day:%Y%m%d}_{int(seed_order):05d}"
        stopped = False
        for depth_index in depth_indices:
            if u_day is None:
                u = np.asarray(filt.variables["uo_glor"][time_index, depth_index, :, :], dtype="float32")
                v = np.asarray(filt.variables["vo_glor"][time_index, depth_index, :, :], dtype="float32")
            else:
                u = u_day[depth_index]
                v = v_day[depth_index]
            speed = np.hypot(u, v)
            if depth_index > 0:
                center_i, center_j, center_speed, min_steps = _seeded_speed_min(speed, prev_i, prev_j, params.deep_search_cells)
            grid_lat_value = float(lat[center_j])
            grid_lon_value = float(lon[center_i])
            refinement_enabled = True
            refinement_stage = "pre_hua"
            fallback_refined = {
                "center_i_refined": float(center_i),
                "center_j_refined": float(center_j),
                "center_lon_refined": grid_lon_value,
                "center_lat_refined": grid_lat_value,
                "refined_speed_ms": float(center_speed),
                "refined_offset_km": 0.0,
                "refined_ok": False,
                "subgrid_fit_quality": "not_attempted",
            }
            pre_hua_refined = (
                _refine_speed_min_subgrid(
                    speed,
                    u,
                    v,
                    lon,
                    lat,
                    center_i,
                    center_j,
                    target_degree=float(args.subgrid_target_degree),
                    window_radius_cells=int(args.subgrid_window_radius_cells),
                    min_finite_fraction=float(args.subgrid_min_finite_fraction),
                )
                if refinement_enabled
                else fallback_refined
            )
            hua_center_i = float(pre_hua_refined["center_i_refined"])
            hua_center_j = float(pre_hua_refined["center_j_refined"])
            if matlab_backend is not None:
                check = matlab_backend.check(int(depth_index), hua_center_i, hua_center_j)
            else:
                check = _hua_verify_radius(u, v, hua_center_i, hua_center_j, params)
            refined = pre_hua_refined
            if bool(check["hua_pass"]):
                prev_i = int(np.clip(round(float(refined["center_i_refined"])), 0, speed.shape[1] - 1))
                prev_j = int(np.clip(round(float(refined["center_j_refined"])), 0, speed.shape[0] - 1))
            else:
                stopped = True
            lat_value = float(refined["center_lat_refined"])
            lon_value = float(refined["center_lon_refined"])
            polarity = _extremum_polarity(str(seed["ssh_extremum_type"]), lat_value, float(check.get("circulation_sign", np.nan)))
            row = {
                "date": day.isoformat(),
                "hua_object_id": object_id,
                "seed_order": int(seed_order),
                "ssh_extremum_type": str(seed["ssh_extremum_type"]),
                "polarity": polarity,
                "time_index": int(time_index),
                "candidate_selection": str(seed.get("candidate_selection", args.candidate_selection)),
                "tile_lon_min": float(seed.get("tile_lon_min", np.nan)),
                "tile_lon_max": float(seed.get("tile_lon_max", np.nan)),
                "tile_lat_min": float(seed.get("tile_lat_min", np.nan)),
                "tile_lat_max": float(seed.get("tile_lat_max", np.nan)),
                "tile_rank": int(seed.get("tile_rank", seed_order + 1)) if pd.notna(seed.get("tile_rank", np.nan)) else int(seed_order + 1),
                "depth_index": int(depth_index),
                "depth_m": float(depth[depth_index]),
                "seed_i": seed_i,
                "seed_j": seed_j,
                "seed_lon": float(lon[seed_i]),
                "seed_lat": float(lat[seed_j]),
                "ssh_value_m": float(seed["ssh_value_m"]),
                "speed_min_i": int(round(float(refined["center_i_refined"]))),
                "speed_min_j": int(round(float(refined["center_j_refined"]))),
                "speed_min_i_grid": int(center_i),
                "speed_min_j_grid": int(center_j),
                "center_lon_grid": grid_lon_value,
                "center_lat_grid": grid_lat_value,
                "center_lon": lon_value,
                "center_lat": lat_value,
                "center_i_refined": float(refined["center_i_refined"]),
                "center_j_refined": float(refined["center_j_refined"]),
                "hua_center_i": float(hua_center_i),
                "hua_center_j": float(hua_center_j),
                "hua_center_source": "refined_search_stage",
                "refined_before_hua": True,
                "center_refinement_stage": refinement_stage,
                "center_lon_refined": float(refined["center_lon_refined"]),
                "center_lat_refined": float(refined["center_lat_refined"]),
                "center_x_from_seed_km": float((float(refined["center_i_refined"]) - seed_i) * dx_km),
                "center_y_from_seed_km": float((float(refined["center_j_refined"]) - seed_j) * dy_km),
                "center_x_from_seed_grid_km": float((center_i - seed_i) * dx_km),
                "center_y_from_seed_grid_km": float((center_j - seed_j) * dy_km),
                "center_speed_ms": float(refined["refined_speed_ms"] if bool(refined["refined_ok"]) else center_speed),
                "center_speed_grid_ms": float(center_speed),
                "refined_offset_km": float(refined["refined_offset_km"]),
                "refined_ok": bool(refined["refined_ok"]),
                "subgrid_fit_quality": str(refined["subgrid_fit_quality"]),
                "local_min_steps": int(min_steps),
                "stopped_after_failure": bool(stopped),
                **check,
            }
            centers_rows.append(row)
            circle_rows.append({k: v for k, v in row.items() if k not in {"center_lon", "center_lat"}})
            if bool(check["hua_pass"]):
                structure_rows.append(
                    {
                        "date": day.isoformat(),
                        "hua_object_id": object_id,
                        "depth_index": int(depth_index),
                        "depth_m": float(depth[depth_index]),
                        "center_lon": lon_value,
                        "center_lat": lat_value,
                        "center_lon_grid": grid_lon_value,
                        "center_lat_grid": grid_lat_value,
                        "center_lon_refined": float(refined["center_lon_refined"]),
                        "center_lat_refined": float(refined["center_lat_refined"]),
                        "refined_ok": bool(refined["refined_ok"]),
                        "refined_offset_km": float(refined["refined_offset_km"]),
                        "radius_km": float(check["accepted_radius_cells"]) * float(np.nanmean([dx_km, dy_km])),
                        "polarity": polarity,
                    }
                )
                if args.write_object_voxels:
                    voxel_rows.extend(
                        _object_voxels_for_layer(
                            u,
                            v,
                            lon,
                            lat,
                            float(depth[depth_index]),
                            day=day,
                            object_id=object_id,
                            depth_index=int(depth_index),
                            center_i=int(center_i),
                            center_j=int(center_j),
                            radius_cells=float(check["accepted_radius_cells"]),
                            polarity=polarity,
                        )
                    )
            if args.stop_at_first_failed_layer and stopped:
                break

    centers = pd.DataFrame(centers_rows)
    circle = pd.DataFrame(circle_rows)
    structures = pd.DataFrame(structure_rows)
    voxels = pd.DataFrame(voxel_rows, columns=OBJECT_VOXEL_COLUMNS)
    if args.write_day_figures and not centers.empty:
        _plot_day_summary(day, centers, zos, lon, lat, output_dir / "figures")
    return centers, circle, structures, voxels


def _plot_day_summary(day: date, centers: pd.DataFrame, zos: np.ndarray, lon: np.ndarray, lat: np.ndarray, figure_dir: Path) -> None:
    plt = _get_pyplot()
    figure_dir.mkdir(parents=True, exist_ok=True)
    surface = centers[centers["depth_index"].eq(0)].copy()
    if surface.empty:
        return
    fig, ax = plt.subplots(figsize=(13, 4.8))
    vmax = float(np.nanpercentile(np.abs(zos), 98))
    vmax = max(vmax, 1e-6)
    im = ax.pcolormesh(lon, lat, zos, shading="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    passed = surface["hua_pass"].astype(bool)
    ax.scatter(surface.loc[~passed, "seed_lon"], surface.loc[~passed, "seed_lat"], s=12, c="#9ca3af", label="SSH seed failed")
    ax.scatter(surface.loc[passed, "center_lon"], surface.loc[passed, "center_lat"], s=18, c="#22c55e", label="Hua passed center")
    ax.set_title(f"Hua SSH+velocity surface candidates {day:%Y-%m-%d}")
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.legend(loc="upper right", fontsize=8)
    cbar = fig.colorbar(im, ax=ax, pad=0.01)
    cbar.set_label("30-180d bandpass zos (m)")
    fig.savefig(figure_dir / f"surface_candidates_{day:%Y%m%d}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _write_parts(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, index=False, engine=DEFAULT_PARQUET_ENGINE)
    tmp.replace(path)


def _read_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path, engine=DEFAULT_PARQUET_ENGINE)


def _write_parquet(df: pd.DataFrame, path: Path, index: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=index, engine=DEFAULT_PARQUET_ENGINE)


def _merge_parts(parts_dir: Path, name: str, output_dir: Path) -> pd.DataFrame:
    existing = output_dir / f"{name}.parquet"
    if existing.exists():
        return _read_parquet(existing)
    parts = sorted(parts_dir.rglob("*.parquet"))
    if not parts:
        return pd.DataFrame()
    frames = [_read_parquet(path) for path in parts]
    merged = pd.concat(frames, ignore_index=True)
    _write_parquet(merged, output_dir / f"{name}.parquet", index=False)
    merged.to_csv(output_dir / f"{name}.csv", index=False)
    return merged


def _voxel_stats_from_parts(parts_dir: Path) -> pd.DataFrame:
    parts = sorted(parts_dir.rglob("*.parquet"))
    if not parts:
        return pd.DataFrame()
    cols = ["hua_object_id", "depth_index", "i", "j", "node_key_2d", "node_key_3d"]
    stats = []
    for path in parts:
        voxels = pd.read_parquet(path, columns=cols, engine=DEFAULT_PARQUET_ENGINE)
        if voxels.empty:
            continue
        surface = voxels[voxels["depth_index"].eq(0)]
        total = (
            voxels.groupby("hua_object_id")
            .agg(
                voxel_count_3d=("node_key_3d", "nunique"),
                min_i=("i", "min"),
                max_i=("i", "max"),
                min_j=("j", "min"),
                max_j=("j", "max"),
                min_depth_index=("depth_index", "min"),
                max_depth_index=("depth_index", "max"),
            )
            .reset_index()
        )
        surf = surface.groupby("hua_object_id")["node_key_2d"].nunique().rename("surface_voxel_count_2d").reset_index()
        stats.append(total.merge(surf, on="hua_object_id", how="left"))
    if not stats:
        return pd.DataFrame()
    merged = pd.concat(stats, ignore_index=True)
    reduced = (
        merged.groupby("hua_object_id")
        .agg(
            voxel_count_3d=("voxel_count_3d", "sum"),
            surface_voxel_count_2d=("surface_voxel_count_2d", "sum"),
            min_i=("min_i", "min"),
            max_i=("max_i", "max"),
            min_j=("min_j", "min"),
            max_j=("max_j", "max"),
            min_depth_index=("min_depth_index", "min"),
            max_depth_index=("max_depth_index", "max"),
        )
        .reset_index()
    )
    reduced["surface_voxel_count_2d"] = reduced["surface_voxel_count_2d"].fillna(0).astype("int64")
    return reduced


def _write_frame_object_summary(centers: pd.DataFrame, voxels: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    if centers.empty:
        summary = pd.DataFrame()
    else:
        passed = centers[centers["hua_pass"].astype(bool)].copy()
        if passed.empty:
            summary = pd.DataFrame()
        else:
            layer_stats = (
                passed.groupby("hua_object_id")
                .agg(
                    date=("date", "first"),
                    polarity=("polarity", "first"),
                    surface_seed_i=("seed_i", "first"),
                    surface_seed_j=("seed_j", "first"),
                    surface_seed_lon=("seed_lon", "first"),
                    surface_seed_lat=("seed_lat", "first"),
                    surface_center_i=("speed_min_i", "first"),
                    surface_center_j=("speed_min_j", "first"),
                    surface_center_lon=("center_lon", "first"),
                    surface_center_lat=("center_lat", "first"),
                    ssh_value_m=("ssh_value_m", "first"),
                    pass_layers=("depth_index", "size"),
                    max_depth_m=("depth_m", "max"),
                    mean_radius_cells=("accepted_radius_cells", "mean"),
                    min_center_speed_ms=("center_speed_ms", "min"),
                    mean_center_speed_ms=("center_speed_ms", "mean"),
                )
                .reset_index()
            )
            voxel_stats = _voxel_stats_from_parts(output_dir / "object_voxels_parts") if voxels.empty else pd.DataFrame()
            if voxels.empty and voxel_stats.empty:
                summary = layer_stats
                summary["voxel_count_3d"] = 0
                summary["surface_voxel_count_2d"] = 0
            else:
                if voxel_stats.empty:
                    surface = voxels[voxels["depth_index"].eq(0)]
                    voxel_stats = (
                        voxels.groupby("hua_object_id")
                        .agg(
                            voxel_count_3d=("node_key_3d", "nunique"),
                            min_i=("i", "min"),
                            max_i=("i", "max"),
                            min_j=("j", "min"),
                            max_j=("j", "max"),
                            min_depth_index=("depth_index", "min"),
                            max_depth_index=("depth_index", "max"),
                        )
                        .reset_index()
                    )
                    surf = surface.groupby("hua_object_id")["node_key_2d"].nunique().rename("surface_voxel_count_2d").reset_index()
                    voxel_stats = voxel_stats.merge(surf, on="hua_object_id", how="left")
                summary = layer_stats.merge(voxel_stats, on="hua_object_id", how="left")
    _write_parquet(summary, output_dir / "frame_object_summary.parquet", index=False)
    summary.to_csv(output_dir / "frame_object_summary.csv", index=False)
    return summary


def _plot_axis_examples(centers: pd.DataFrame, output_dir: Path, max_examples: int = 8) -> None:
    plt = _get_pyplot()
    if centers.empty:
        return
    figure_dir = output_dir / "axis_velocity_stack_examples"
    figure_dir.mkdir(parents=True, exist_ok=True)
    ranked = (
        centers.groupby("hua_object_id")
        .agg(n_layers=("depth_index", "size"), n_pass=("hua_pass", "sum"), date=("date", "first"), polarity=("polarity", "first"))
        .sort_values(["n_pass", "n_layers"], ascending=False)
        .head(max_examples)
        .reset_index()
    )
    for _, item in ranked.iterrows():
        obj = centers[centers["hua_object_id"].eq(item["hua_object_id"])].sort_values("depth_index")
        z = -obj["depth_m"].to_numpy(dtype="float64") / 1000.0
        x = obj["center_x_from_seed_km"].to_numpy(dtype="float64")
        y = obj["center_y_from_seed_km"].to_numpy(dtype="float64")
        passed = obj["hua_pass"].astype(bool).to_numpy()
        fig = plt.figure(figsize=(8, 6.5))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot(x, y, z, color="#334155", linewidth=2.0, label="Hua candidate axis")
        ax.scatter(x[passed], y[passed], z[passed], c="#22c55e", s=36, label="pass")
        ax.scatter(x[~passed], y[~passed], z[~passed], c="#ef4444", s=32, marker="x", label="fail")
        ax.scatter([0], [0], [z[0] if len(z) else 0], marker="+", c="red", s=150, linewidth=3, label="SSH seed")
        ax.set_xlabel("east from SSH seed (km)")
        ax.set_ylabel("north from SSH seed (km)")
        ax.set_zlabel("depth (km, down)")
        ax.set_title(f"Hua replicated axis {item['hua_object_id']}\n{item['polarity']}, {item['date']}, pass {int(item['n_pass'])}/{int(item['n_layers'])}")
        ax.view_init(elev=22, azim=-55)
        ax.legend(fontsize=8)
        stem = figure_dir / f"axis_{item['hua_object_id']}"
        fig.savefig(stem.with_suffix(".png"), dpi=180, bbox_inches="tight")
        fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)


def _write_docs(output_dir: Path, args: argparse.Namespace, summary: dict[str, object], rejection: pd.DataFrame) -> None:
    lines = [
        "# Hua 2023 SSH+Velocity Hybrid 方法对齐说明",
        "",
        "本目录是 Hua et al. 2023 方法在 ACC 30-180 天带通场上的论文复刻实验，不读取现有 catalog/tracks/completed centers。",
        "",
        "## 方法链条",
        "",
        "1. 用 `zos_glor` 带通信号在表层做局地极大/极小搜索，对应论文的 SSH minima/maxima candidate centers。",
        "2. 在 SSH candidate 附近寻找 `sqrt(u'^2+v'^2)` 局部低值，得到速度中心候选。",
        "3. 从 `STARTRADIUS=3` 个网格点开始沿圆周路径检查速度模连续性、方向连续性、切向性、对心对称性和两侧反转。",
        "4. 表层通过后，以下一层上方中心为 seed 向深层扩展；失败层记录原因，不硬跳到远处中心。",
        "",
        "## 边界模式",
        "",
        f"- 当前边界模式：`{getattr(args, 'boundary_mode', 'circle_strict_original')}`。",
        f"- Hua 检验后端：`{getattr(args, 'hua_backend', 'python')}`。",
        f"- 中心加密阶段：`{getattr(args, 'center_refinement_stage', 'pre_hua')}`。",
        "- `circle_strict_original` 是修改前原标准：固定半径圆周上的 Hua 几何检验。",
        "- `velocity_streamline_contour` 是 ACC TEST 实验口径：每层围绕速度弱中心寻找闭合速度流线轮廓，并在该轮廓上评估方向一致性和切向对齐。",
        "- `pre_hua` 是唯一中心加密顺序：速度弱中心先做连续坐标加密，再用 refined center 进入 Hua 检验。",
        "- `--hua-backend matlab` 使用 MATLAB Engine 执行同口径边界 kernel；Python 仍负责 IO、垂向延展、表格输出和后续 shape 链条。",
        "- `pass_rate_breakdown.csv/json` 给出 surface/all-layer 通过率与失败占比；`sensitivity_matrix.csv/parquet` 给出流线模式下 tangent/monotonic 参数敏感度。",
        "",
        "## ACC 适配",
        "",
        "- 主速度口径为 `u'=u_{30-180d}, v'=v_{30-180d}`；SSH 口径为 `zos_{30-180d}`。",
        "- 未复制 raw 年文件，未写 `input_daily/`。",
        "- SSH 搜索窗口、深层搜索半径是 ACC 网格适配参数，已写入 `run_summary.json`。",
        "",
        "## 运行摘要",
        "",
        f"- 日期范围：`{args.start}` 到 `{args.end}`。",
        f"- 总中心记录：`{summary.get('n_center_rows', 0)}`。",
        f"- 表层候选数：`{summary.get('n_surface_candidates', 0)}`。",
        f"- Hua pass 层数：`{summary.get('n_pass_layers', 0)}`。",
        f"- Hua pass fraction：`{summary.get('pass_fraction', 0.0):.4f}`。",
        "",
        "## 失败原因",
        "",
    ]
    if rejection.empty:
        lines.append("没有失败原因统计。")
    else:
        for row in rejection.to_dict("records"):
            lines.append(f"- `{row['failure_reason']}`：`{row['count']}`")
    (output_dir / "method_alignment_zh.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output_dir / "hua_acc_replication_summary_zh.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_synthetic_tests(output_dir: Path, params: DetectionParams) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    yy, xx = np.mgrid[-40:41, -40:41]
    r = np.hypot(xx, yy)
    tang_u = -yy / np.maximum(r, 1)
    tang_v = xx / np.maximum(r, 1)
    amp = np.exp(-(r / 15) ** 2)
    cases = []
    saddle_u = xx * np.exp(-(r / 18) ** 2)
    saddle_v = -yy * np.exp(-(r / 18) ** 2)
    for name, u, v, expected in [
        ("gaussian_vortex", tang_u * amp, tang_v * amp, True),
        ("pure_shear", np.ones_like(xx, dtype=float) * 0.05, yy * 0.0, False),
        ("double_core_saddle", saddle_u, saddle_v, False),
    ]:
        check = _hua_verify_radius(u, v, 40, 40, params)
        cases.append({"case": name, "expected_pass": expected, **check})
    df = pd.DataFrame(cases)
    df.to_csv(output_dir / "synthetic_hua_tests.csv", index=False)
    _write_parquet(df, output_dir / "synthetic_hua_tests.parquet", index=False)
    return df


def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if bool(args.disable_subgrid_center_refinement):
        raise SystemExit(
            "--disable-subgrid-center-refinement has been retired for ACC TEST. "
            "Search-stage pre-Hua subgrid center refinement is now mandatory."
        )
    params = DetectionParams(
        ssh_window_cells=args.ssh_window_cells,
        start_radius_cells=args.start_radius_cells,
        max_radius_cells=args.max_radius_cells,
        speed_ratio_max=args.speed_ratio_max,
        angle_jump_max_deg=args.angle_jump_max_deg,
        tangent_tolerance_deg=args.tangent_tolerance_deg,
        symmetry_tolerance_deg=args.symmetry_tolerance_deg,
        min_tangent_fraction=args.min_tangent_fraction,
        min_reversal_fraction=args.min_reversal_fraction,
        min_finite_fraction=args.min_finite_fraction,
        direction_exception_extra=args.direction_exception_extra,
        surface_search_cells=args.surface_search_cells,
        deep_search_cells=args.deep_search_cells,
        require_boundary_monotonic_rotation=args.require_boundary_monotonic_rotation,
        boundary_monotonic_exception_limit=args.boundary_monotonic_exception_limit,
        boundary_mode=args.boundary_mode,
        streamline_direction_exception_fraction=args.streamline_direction_exception_fraction,
        streamline_step_cells=args.streamline_step_cells,
        streamline_max_steps=args.streamline_max_steps,
        streamline_start_angles=args.streamline_start_angles,
        streamline_closure_tolerance_cells=args.streamline_closure_tolerance_cells,
        streamline_min_winding_turns=args.streamline_min_winding_turns,
        streamline_min_points=args.streamline_min_points,
        hua_backend=args.hua_backend,
        matlab_use_gpu=bool(args.matlab_use_gpu),
    )
    if params.hua_backend != "python":
        print(
            "[hua-backend] MATLAB Engine will execute the Hua boundary kernels; "
            "Python remains responsible for IO, vertical continuation, output tables, tracking, catalog, and shape."
        )
    if args.synthetic_tests:
        run_synthetic_tests(output_dir / "synthetic_tests", params)

    days = _date_range(_parse_date(args.start), _parse_date(args.end))
    part_root = output_dir / "parts"
    if not args.finalize_only:
        years = sorted({d.year for d in days})
        for year in years:
            filt, raw, lon, lat, depth = _load_year_arrays(args, year)
            matlab_backend = (
                MatlabHuaBackend(params, output_dir / "matlab_hua_cache" / f"year={year}")
                if params.hua_backend == "matlab"
                else None
            )
            try:
                year_days = [d for d in days if d.year == year]
                for day in year_days:
                    centers_path = part_root / "centers" / f"date={day:%Y%m%d}.parquet"
                    circle_path = part_root / "circle" / f"date={day:%Y%m%d}.parquet"
                    structures_path = part_root / "structures" / f"date={day:%Y%m%d}.parquet"
                    voxels_path = output_dir / "object_voxels_parts" / f"year={day.year}" / f"date={day:%Y%m%d}.parquet"
                    voxel_ready = (not args.write_object_voxels) or voxels_path.exists()
                    if args.resume and centers_path.exists() and circle_path.exists() and structures_path.exists() and voxel_ready:
                        continue
                    centers, circle, structures, voxels = _detect_day(
                        day,
                        filt,
                        raw,
                        lon,
                        lat,
                        depth,
                        params,
                        args,
                        output_dir,
                        matlab_backend=matlab_backend,
                    )
                    _write_parts(centers, centers_path)
                    _write_parts(circle, circle_path)
                    _write_parts(structures, structures_path)
                    if args.write_object_voxels:
                        _write_parts(voxels, voxels_path)
                    print(f"[hua] {day} centers={len(centers)} pass={int(centers['hua_pass'].sum()) if not centers.empty else 0}", flush=True)
            finally:
                if matlab_backend is not None:
                    matlab_backend.close()
                filt.close()
                if raw is not None:
                    raw.close()
    if args.partial_only:
        return

    centers = _merge_parts(part_root / "centers", "centers_hua_style", output_dir)
    _merge_parts(part_root / "circle", "circle_check_diagnostics", output_dir)
    _merge_parts(part_root / "structures", "structures_hua_style", output_dir)
    if args.write_object_voxels:
        voxels = pd.DataFrame(columns=OBJECT_VOXEL_COLUMNS)
    else:
        voxels = pd.DataFrame(columns=OBJECT_VOXEL_COLUMNS)
    _write_frame_object_summary(centers, voxels, output_dir)
    if centers.empty:
        rejection = pd.DataFrame(columns=["failure_reason", "count"])
        summary = {"n_center_rows": 0, "n_surface_candidates": 0, "n_pass_layers": 0, "pass_fraction": 0.0}
    else:
        reason_column = "first_hard_failure" if "first_hard_failure" in centers.columns else "dominant_failure"
        rejection = (
            centers.loc[~centers["hua_pass"].astype(bool), reason_column]
            .value_counts()
            .rename_axis("failure_reason")
            .reset_index(name="count")
        )
        rejection.to_csv(output_dir / "rejection_reasons.csv", index=False)
        if not bool(args.skip_axis_examples):
            _plot_axis_examples(centers, output_dir)
        summary = {
            "n_center_rows": int(len(centers)),
            "n_surface_candidates": int(centers[centers["depth_index"].eq(0)]["hua_object_id"].nunique()),
            "n_pass_layers": int(centers["hua_pass"].sum()),
            "pass_fraction": float(centers["hua_pass"].mean()),
            "n_days": int(centers["date"].nunique()),
            "n_objects": int(centers["hua_object_id"].nunique()),
            "boundary_mode": str(params.boundary_mode),
            "hua_backend": str(params.hua_backend),
            "matlab_use_gpu": bool(params.matlab_use_gpu),
            "center_refinement_stage": str(getattr(args, "center_refinement_stage", "pre_hua")),
            "refined_before_hua_rows": int(centers["refined_before_hua"].fillna(False).astype(bool).sum())
            if "refined_before_hua" in centers.columns
            else 0,
            "parameters": {**vars(args), "detection_params": params.__dict__},
        }
        if "streamline_closed" in centers.columns:
            closed = centers["streamline_closed"].fillna(False).astype(bool)
            summary.update(
                {
                    "streamline_closed_rows": int(closed.sum()),
                    "streamline_closed_fraction": float(closed.mean()) if len(closed) else 0.0,
                }
            )
        if "refined_ok" in centers.columns:
            passed_refined = centers[centers["hua_pass"].astype(bool)].copy()
            offsets = passed_refined["refined_offset_km"].astype(float) if "refined_offset_km" in passed_refined.columns else pd.Series(dtype=float)
            finite_offsets = offsets[np.isfinite(offsets)]
            summary.update(
                {
                    "subgrid_refined_enabled": bool(not args.disable_subgrid_center_refinement),
                    "subgrid_target_degree": float(args.subgrid_target_degree),
                    "subgrid_refined_ok_rows": int(passed_refined["refined_ok"].fillna(False).astype(bool).sum()),
                    "subgrid_refined_ok_fraction_of_passed": float(
                        passed_refined["refined_ok"].fillna(False).astype(bool).mean()
                    )
                    if len(passed_refined)
                    else 0.0,
                    "subgrid_refined_offset_km_median": float(np.nanmedian(finite_offsets)) if finite_offsets.size else 0.0,
                    "subgrid_refined_offset_km_p90": float(np.nanquantile(finite_offsets, 0.9)) if finite_offsets.size else 0.0,
                }
            )
    (output_dir / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    if rejection.empty:
        rejection.to_csv(output_dir / "rejection_reasons.csv", index=False)
    if not centers.empty:
        _write_detection_diagnostics(output_dir, centers, params, args)
    _write_docs(output_dir, args, summary, rejection)


def main() -> None:
    parser = argparse.ArgumentParser(description="Replicate Hua 2023 SSH+velocity hybrid eddy detection on bandpass fields.")
    parser.add_argument("--filter-root", default=r"F:\Global Ocean Ensemble Physics Reanalysis Kuroshio Current\FILTER")
    parser.add_argument("--raw-root", default=r"F:\Global Ocean Ensemble Physics Reanalysis Kuroshio Current")
    parser.add_argument("--filter-template", default="global_phy_{year}.nc")
    parser.add_argument("--raw-template", default="global_phy_{year}.nc")
    parser.add_argument("--output-dir", default="/root/autodl-fs/2020_2022_acc/hua_paper_replication/smoke_20200101_20200107")
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2020-01-07")
    parser.add_argument("--max-depth-m", type=float, default=0.0, help="Maximum depth in meters. Use <=0 for all source depth levels.")
    parser.add_argument("--ssh-window-cells", type=int, default=7)
    parser.add_argument("--max-candidates-per-day", type=int, default=0)
    parser.add_argument("--candidate-selection", choices=["global_topn", "tile_topn"], default="tile_topn")
    parser.add_argument("--tile-lon-deg", type=float, default=10.0)
    parser.add_argument("--tile-lat-deg", type=float, default=10.0)
    parser.add_argument("--tile-top-n", type=int, default=15)
    parser.add_argument("--surface-search-cells", type=int, default=8)
    parser.add_argument("--deep-search-cells", type=int, default=6)
    parser.add_argument("--start-radius-cells", type=int, default=3)
    parser.add_argument("--max-radius-cells", type=int, default=8)
    parser.add_argument("--speed-ratio-max", type=float, default=3.0)
    parser.add_argument("--angle-jump-max-deg", type=float, default=150.0)
    parser.add_argument("--tangent-tolerance-deg", type=float, default=24.0)
    parser.add_argument("--symmetry-tolerance-deg", type=float, default=120.0)
    parser.add_argument("--min-tangent-fraction", type=float, default=0.70)
    parser.add_argument("--min-reversal-fraction", type=float, default=0.70)
    parser.add_argument("--min-finite-fraction", type=float, default=0.95)
    parser.add_argument("--direction-exception-extra", type=int, default=0)
    parser.add_argument("--require-boundary-monotonic-rotation", action="store_true")
    parser.add_argument("--boundary-monotonic-exception-limit", type=int, default=0)
    parser.add_argument("--boundary-mode", choices=["circle_strict_original", "velocity_streamline_contour"], default="circle_strict_original")
    parser.add_argument(
        "--hua-backend",
        choices=["python", "matlab"],
        default="python",
        help="Hua backend. matlab uses day-level MATLAB Engine kernels while Python keeps IO, seed selection, outputs, tracking, catalog, and shape.",
    )
    parser.add_argument("--matlab-use-gpu", action="store_true", help="Use gpuArray inside the MATLAB streamline batch kernel when --hua-backend matlab.")
    parser.add_argument("--keep-matlab-bridge-files", action="store_true", help="Keep temporary MATLAB day-batch .mat bridge files for debugging. Defaults to deleting them after Python reads the rows.")
    parser.add_argument("--streamline-direction-exception-fraction", type=float, default=0.10)
    parser.add_argument("--streamline-step-cells", type=float, default=0.5)
    parser.add_argument("--streamline-max-steps", type=int, default=180)
    parser.add_argument("--streamline-start-angles", type=int, default=4)
    parser.add_argument("--streamline-closure-tolerance-cells", type=float, default=1.75)
    parser.add_argument("--streamline-min-winding-turns", type=float, default=0.75)
    parser.add_argument("--streamline-min-points", type=int, default=16)
    parser.add_argument("--sensitivity-tangent-fractions", default="0.50,0.60,0.70")
    parser.add_argument("--sensitivity-tangent-tolerances-deg", default="24,30,36,45")
    parser.add_argument("--sensitivity-direction-exception-fractions", default="0.05,0.10,0.15")
    parser.add_argument(
        "--baseline-summary-path",
        type=Path,
        default=None,
        help="Optional baseline run_summary.json used to add delta-vs-baseline fields to streamline sensitivity_matrix outputs.",
    )
    parser.add_argument(
        "--disable-subgrid-center-refinement",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--center-refinement-stage",
        choices=["pre_hua"],
        default="pre_hua",
        help="Compatibility no-op. The only supported order is pre_hua: refine the velocity-minimum center before Hua checks.",
    )
    parser.add_argument(
        "--subgrid-target-degree",
        type=float,
        default=1.0 / 24.0,
        help="Local refined-center interpolation spacing in degrees; default is 1/24 degree.",
    )
    parser.add_argument(
        "--subgrid-window-radius-cells",
        type=int,
        default=2,
        help="Half-width, in original grid cells, used around each passed Hua center for local refinement.",
    )
    parser.add_argument(
        "--subgrid-min-finite-fraction",
        type=float,
        default=0.6,
        help="Minimum finite-data fraction in the local refinement window.",
    )
    parser.add_argument("--preload-day-uv", action="store_true")
    parser.add_argument("--write-object-voxels", action="store_true")
    parser.add_argument("--partial-only", action="store_true")
    parser.add_argument("--finalize-only", action="store_true")
    parser.add_argument("--stop-at-first-failed-layer", action="store_true")
    parser.add_argument("--write-day-figures", action="store_true")
    parser.add_argument("--skip-axis-examples", action="store_true", help="Skip 3-D diagnostic axis-example plots during finalize.")
    parser.add_argument("--synthetic-tests", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--candidate-cache-dir",
        type=Path,
        default=None,
        help="Optional directory containing MATLAB/Python precomputed candidates_YYYYMMDD.csv files. Cache rows bypass live candidate selection.",
    )
    parser.add_argument(
        "--netcdf-chunk-cache-mb",
        type=int,
        default=512,
        help="Per-variable NetCDF chunk cache. Local yearly sharding benefits from reusing 46-day HDF5 time chunks.",
    )
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
