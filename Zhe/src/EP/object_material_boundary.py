from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import date
from functools import lru_cache
from pathlib import Path

import netCDF4
import numpy as np
try:
    import pandas as pd
except ModuleNotFoundError:  # pragma: no cover - dependency-light dry-run support.
    pd = None

from .contracts import DEFAULT_RESULT_ROOT, RHO0, shape_output_name
from .fields import RepresentativeSlice
from .geometry import AxisLine
from .material_volume import (
    DEFAULT_DYNAMIC_BOUNDARY_V2_OUTPUT_ROOT,
    MaterialVolumeRequest,
    _compute_one_slice,
    _json_ready,
    _write_table,
)

try:
    from src.utils.field_sampling import bilinear_sample, xy_to_lonlat
except ModuleNotFoundError:  # pragma: no cover - import checked at run time.
    bilinear_sample = None
    xy_to_lonlat = None


DEFAULT_OBJECT_BOUNDARY_OUTPUT_ROOT = DEFAULT_DYNAMIC_BOUNDARY_V2_OUTPUT_ROOT.parent / "object_material_boundary_validation"


@dataclass(frozen=True)
class ObjectBoundaryRequest:
    result_root: Path = DEFAULT_RESULT_ROOT
    filter_root: Path = Path("/root/autodl-fs/kuroshiou/Filter")
    output_root: Path = DEFAULT_OBJECT_BOUNDARY_OUTPUT_ROOT
    shapes: tuple[str, ...] = ("coherent", "upright_like")
    orientations: tuple[str, ...] = ("turned",)
    buoyancy_sources: tuple[str, ...] = ("thermal_wind",)
    filter_template: str = "global_phy_{year}_bandpass_30_180d.nc"
    radial_bins: int = 24
    azimuth_bins: int = 48
    rmax: float = 1.5
    reference_lat: float = 30.0
    constant_n2: float = 2.0e-5
    core_radius_over_R: float = 1.5
    speed_core_quantile: float = 0.45
    pv_core_quantile: float = 0.70
    min_mask_fraction: float = 0.01
    boundary_mode: str = "levelset_v2"
    active_contour_iterations: int = 14
    leakage_weight: float = 1.0
    smoothness_weight: float = 0.08
    containment_weight: float = 0.35
    area_weight: float = 0.12
    vertical_continuity_weight: float = 0.18
    time_continuity_weight: float = 0.08
    levelset_sigma_cells: float = 1.0
    min_core_retention: float = 0.75
    min_area_fraction: float = 0.15
    max_area_fraction: float = 0.65
    max_tracks_per_shape: int = 0
    max_objectdays: int = 0
    skip_missing: bool = False
    dry_run: bool = False


def _require_runtime() -> None:
    if pd is None:
        raise ModuleNotFoundError("pandas is required for object-level material-boundary validation")
    if bilinear_sample is None or xy_to_lonlat is None:
        raise ModuleNotFoundError("src.utils.field_sampling is required for object-level material-boundary validation")


def _split_csv(value: str | tuple[str, ...] | list[str]) -> tuple[str, ...]:
    if isinstance(value, (tuple, list)):
        return tuple(str(v).strip() for v in value if str(v).strip())
    return tuple(part.strip() for part in str(value).split(",") if part.strip())


@lru_cache(maxsize=96)
def _time_index(path_text: str) -> dict[date, int]:
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


def _clean(values) -> np.ndarray:
    arr = np.ma.filled(values, np.nan).astype("float64", copy=False)
    arr[np.abs(arr) > 1.0e20] = np.nan
    return arr


def _read_filter_day(filter_root: Path, template: str, day: date) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    path = filter_root / template.format(year=day.year)
    if not path.exists():
        raise FileNotFoundError(path)
    day_index = _time_index(str(path)).get(day)
    if day_index is None:
        raise KeyError(f"{day} not found in {path}")
    with netCDF4.Dataset(path) as ds:
        for variable in ("uo_glor", "vo_glor", "thetao_glor"):
            if variable not in ds.variables:
                raise KeyError(f"{variable} not found in {path}")
        lon = np.asarray(ds.variables["longitude"][:], dtype="float64")
        lat = np.asarray(ds.variables["latitude"][:], dtype="float64")
        u = _clean(ds.variables["uo_glor"][day_index])
        v = _clean(ds.variables["vo_glor"][day_index])
        theta = _clean(ds.variables["thetao_glor"][day_index])
    return lon, lat, u, v, theta


def _polar_grid(rmax: float, radial_bins: int, azimuth_bins: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    radial = np.linspace(0.0, float(rmax), int(radial_bins), dtype="float64")
    theta = np.linspace(0.0, 2.0 * np.pi, int(azimuth_bins), endpoint=False, dtype="float64")
    rr, tt = np.meshgrid(radial, theta, indexing="ij")
    return radial, theta, rr, tt


def _load_shape_objects(result_root: Path, shape: str, max_tracks: int, max_objectdays: int) -> tuple[Path, pd.DataFrame, pd.DataFrame]:
    rv_root = result_root / shape_output_name(shape) / "representative_vortex_radial_seed"
    objects_path = rv_root / "object_cache" / "selected_lifecycle_objects.parquet"
    points_path = rv_root / "axis" / "rotated_points.parquet"
    if not objects_path.exists() or not points_path.exists():
        raise FileNotFoundError(f"Missing radial seed object cache for shape={shape}: {rv_root}")
    objects = pd.read_parquet(objects_path)
    points = pd.read_parquet(points_path)
    objects["date"] = pd.to_datetime(objects["date"]).dt.strftime("%Y-%m-%d")
    points["date"] = pd.to_datetime(points["date"]).dt.strftime("%Y-%m-%d")
    objects = objects[objects["shape_class"].astype(str).eq(shape)].copy()
    if max_tracks > 0:
        keep_tracks = list(objects["track3d_id"].drop_duplicates().astype(int).head(int(max_tracks)))
        objects = objects[objects["track3d_id"].astype(int).isin(keep_tracks)].copy()
    if max_objectdays > 0:
        objects = objects.sort_values(["track3d_id", "date", "eddy3d_object_id"]).head(int(max_objectdays)).copy()
    keep_ids = set(objects["eddy3d_object_id"].astype(int))
    points = points[points["eddy3d_object_id"].astype(int).isin(keep_ids)].copy()
    return rv_root, objects, points


def _sample_object_slice(
    obj,
    points: pd.DataFrame,
    lon: np.ndarray,
    lat: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    radial: np.ndarray,
    theta: np.ndarray,
    radial_mesh: np.ndarray,
    theta_mesh: np.ndarray,
    orientation: str,
) -> tuple[RepresentativeSlice, AxisLine] | None:
    center_line = points[points["eddy3d_object_id"].astype(int).eq(int(obj.eddy3d_object_id))].sort_values("depth_index")
    if center_line.shape[0] < 3:
        return None
    radius_m = float(obj.mean_radius_m)
    alpha = float(getattr(obj, "global_deviate_angle_rad", 0.0)) if orientation == "turned" else 0.0
    cos_a = math.cos(alpha)
    sin_a = math.sin(alpha)
    local_x = radial_mesh * radius_m * np.cos(theta_mesh)
    local_y = radial_mesh * radius_m * np.sin(theta_mesh)
    u_layers: list[np.ndarray] = []
    v_layers: list[np.ndarray] = []
    depth_values: list[float] = []
    for row in center_line.itertuples(index=False):
        k = int(row.depth_index)
        if k < 0 or k >= u.shape[0]:
            continue
        x_rot = float(row.x_rot_m) + local_x
        y_rot = float(row.y_rot_m) + local_y
        x_orig = x_rot * cos_a + y_rot * sin_a
        y_orig = -x_rot * sin_a + y_rot * cos_a
        target_lon, target_lat = xy_to_lonlat(x_orig, y_orig, float(obj.surface_lon), float(obj.surface_lat))
        u_s = bilinear_sample(lon, lat, u[k], target_lon, target_lat)
        v_s = bilinear_sample(lon, lat, v[k], target_lon, target_lat)
        u_layers.append(u_s * cos_a - v_s * sin_a)
        v_layers.append(u_s * sin_a + v_s * cos_a)
        depth_values.append(float(row.depth_m))
    if len(u_layers) < 3:
        return None
    u_arr = np.asarray(u_layers, dtype="float64")
    v_arr = np.asarray(v_layers, dtype="float64")
    depth_arr = np.asarray(depth_values, dtype="float64")
    rep = RepresentativeSlice(
        polarity=str(obj.polarity),
        tau=float(getattr(obj, "life_phase", np.nan)),
        depth_m=depth_arr,
        radius_coord=radial,
        theta_rad=theta,
        radius_m=radius_m,
        u=u_arr,
        v=v_arr,
        speed=np.hypot(u_arr, v_arr),
        count=None,
    )
    axis = AxisLine.zero(depth_arr)
    return rep, axis


def _summary_table(profiles: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["shape", "polarity", "track3d_id"]
    for keys, sub in profiles.groupby(group_cols, sort=True):
        shape, polarity, track_id = keys
        rows.append(
            {
                "shape": shape,
                "polarity": polarity,
                "track3d_id": int(track_id),
                "n_objectdays": int(sub["eddy3d_object_id"].nunique()),
                "n_rows": int(sub.shape[0]),
                "leakage_median_ms": float(np.nanmedian(sub["leakage_mean_abs_ms"].to_numpy(float))),
                "leakage_reduction_fraction_median": float(np.nanmedian(sub["leakage_reduction_fraction"].to_numpy(float))),
                "boundary_flux_over_internal_flux_median": float(np.nanmedian(sub["boundary_flux_over_internal_flux"].to_numpy(float))),
                "vertical_mask_roughness_median": float(np.nanmedian(sub.get("vertical_mask_roughness", pd.Series(dtype=float)).to_numpy(float))),
                "pv_core_retention_median": float(np.nanmedian(sub["pv_core_retention"].to_numpy(float))),
                "weak_core_retention_median": float(np.nanmedian(sub["weak_core_retention"].to_numpy(float))),
            }
        )
    return pd.DataFrame(rows)


def _plot_object_summaries(output_root: Path, profiles: pd.DataFrame, summary: pd.DataFrame) -> list[Path]:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return []

    figures = output_root / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    finite = profiles.replace([np.inf, -np.inf], np.nan)

    if not finite.empty and {"shape", "leakage_mean_abs_ms", "leakage_reduction_fraction"}.issubset(finite.columns):
        shapes = list(dict.fromkeys(finite["shape"].astype(str)))
        leakage_groups = [
            finite.loc[finite["shape"].astype(str) == shape, "leakage_mean_abs_ms"].dropna().to_numpy(float)
            for shape in shapes
        ]
        reduction_groups = [
            finite.loc[finite["shape"].astype(str) == shape, "leakage_reduction_fraction"].dropna().to_numpy(float)
            for shape in shapes
        ]
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
        axes[0].boxplot(leakage_groups, labels=shapes, showfliers=False)
        axes[0].set_ylabel("boundary leakage |u_n| (m/s)")
        axes[0].set_title("Object-level leakage")
        axes[0].grid(True, alpha=0.25)
        axes[1].boxplot(reduction_groups, labels=shapes, showfliers=False)
        axes[1].axhline(0.0, color="0.2", lw=1.0)
        axes[1].set_ylabel("leakage reduction fraction")
        axes[1].set_title("Level-set V2 improvement")
        axes[1].grid(True, alpha=0.25)
        path = figures / "object_leakage_reduction_by_shape.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        written.append(path)

    flux_col = "boundary_flux_over_internal_flux"
    if not finite.empty and {"leakage_mean_abs_ms", flux_col, "shape"}.issubset(finite.columns):
        fig, ax = plt.subplots(figsize=(6.5, 5), constrained_layout=True)
        for shape, sub in finite.groupby("shape", sort=True):
            ax.scatter(sub["leakage_mean_abs_ms"], sub[flux_col], s=12, alpha=0.35, label=str(shape))
        ax.set_xlabel("boundary leakage |u_n| (m/s)")
        ax.set_ylabel("boundary flux / internal flux")
        ax.set_title("Boundary exchange diagnostic")
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False)
        path = figures / "object_boundary_flux_budget.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        written.append(path)

    if not summary.empty and {"shape", "track3d_id", "leakage_reduction_fraction_median"}.issubset(summary.columns):
        plot_table = summary.copy()
        plot_table["track_label"] = plot_table["shape"].astype(str) + "/" + plot_table["track3d_id"].astype(str)
        plot_table = plot_table.sort_values("leakage_reduction_fraction_median", ascending=False)
        fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
        ax.bar(np.arange(plot_table.shape[0]), plot_table["leakage_reduction_fraction_median"].to_numpy(float), color="#4c78a8")
        ax.set_xticks(np.arange(plot_table.shape[0]))
        ax.set_xticklabels(plot_table["track_label"], rotation=45, ha="right")
        ax.set_ylabel("median leakage reduction")
        ax.set_title("Track-level material-boundary optimization")
        ax.grid(True, axis="y", alpha=0.25)
        path = figures / "object_track_leakage_reduction.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        written.append(path)

    if not finite.empty and {"shape", "weak_core_retention", "pv_core_retention"}.issubset(finite.columns):
        rows = []
        for shape, sub in finite.groupby("shape", sort=True):
            rows.append(
                {
                    "shape": str(shape),
                    "weak": float(np.nanmedian(sub["weak_core_retention"].to_numpy(float))),
                    "pv": float(np.nanmedian(sub["pv_core_retention"].to_numpy(float))),
                }
            )
        ret = pd.DataFrame(rows)
        fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
        x = np.arange(ret.shape[0])
        ax.bar(x - 0.18, ret["weak"], width=0.36, label="weak-core retention")
        ax.bar(x + 0.18, ret["pv"], width=0.36, label="PV-core retention")
        ax.axhline(0.75, color="0.25", lw=1.0, ls="--")
        ax.set_xticks(x)
        ax.set_xticklabels(ret["shape"])
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("retention fraction")
        ax.set_title("Core retention under dynamic boundary search")
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend(frameon=False)
        path = figures / "object_core_retention_by_shape.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        written.append(path)

    return written


def _write_summary_md(path: Path, summary: pd.DataFrame, request: ObjectBoundaryRequest) -> None:
    lines = [
        "# Object-Level Material Boundary Validation Summary",
        "",
        "## 目标",
        "本诊断把材料边界验证从代表涡平均场推进到原始 object-day/track。代表涡平均场本身不严格守恒，因此 object-level 结果优先用于判断材料体闭合。",
        "",
        "## 方法",
        f"- boundary mode: `{request.boundary_mode}`",
        f"- shapes: `{','.join(request.shapes)}`",
        f"- orientations: `{','.join(request.orientations)}`",
        f"- buoyancy sources: `{','.join(request.buoyancy_sources)}`",
        f"- radial grid: `{request.radial_bins} x {request.azimuth_bins}`, r/R <= `{request.rmax}`",
        "- 每个 object-day 使用自己的三维中心线和当天 30-180d Filter 速度场构造局地材料体。",
        "- 第一版时间连续性作为 track-level mask/leakage 稳定性审计；真正的拉格朗日边界平流将在下一版进一步加强。",
        "",
        "## 结果摘要",
        "```text",
        summary.to_string(index=False) if not summary.empty else "empty",
        "```",
        "",
        "## 判读",
        "- 若 object-level leakage 明显低于代表涡 leakage，说明代表涡平均混合了不同材料边界。",
        "- 若 object-level leakage 仍高，说明当前速度/PV/低速核心 mask 还不是严格材料体。",
        "- heat/PV/momentum 边界通量不为零时，体内 EP forcing 不能单独解释为闭合材料体的全部动力。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_summary_md(path: Path, summary: pd.DataFrame, request: ObjectBoundaryRequest) -> None:
    lines = [
        "# Object-Level Material Boundary Validation Summary",
        "",
        "## 目标",
        "本诊断把材料边界验证从代表涡平均场推进到原始 object-day/track。代表涡平均场本身不严格守恒，因此 object-level 结果优先用于判断材料体闭合。",
        "",
        "## 方法",
        f"- boundary mode: `{request.boundary_mode}`",
        f"- shapes: `{','.join(request.shapes)}`",
        f"- orientations: `{','.join(request.orientations)}`",
        f"- buoyancy sources: `{','.join(request.buoyancy_sources)}`",
        f"- radial grid: `{request.radial_bins} x {request.azimuth_bins}`, r/R <= `{request.rmax}`",
        "- 每个 object-day 使用自己的三维中心线和当天 30-180d Filter 速度场构造局地材料体。",
        "- levelset_v2 从阈值连通 mask 出发，加入低法向泄漏、核心保留、平滑、面积漂移和垂向连续约束。",
        "- 当前时间连续性仍是 track-level 稳定性审计；真正的拉格朗日边界平流将在下一版进一步加强。",
        "",
        "## 结果摘要",
        "```text",
        summary.to_string(index=False) if not summary.empty else "empty",
        "```",
        "",
        "## 判读",
        "- 若 object-level leakage 明显低于代表涡 leakage，说明代表涡平均混合了不同材料边界，不能直接当作严格材料体。",
        "- 若 object-level leakage 仍高，说明当前速度/PV/低速核心 mask 还不是严格材料体。",
        "- heat/PV/momentum 边界通量不为零时，体内 EP forcing 不能单独解释为闭合材料体的全部动力。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_object_material_boundary_validation(request: ObjectBoundaryRequest) -> dict[str, Path]:
    if request.dry_run:
        print("Object-level material-boundary validation dry-run")
        print(f"result_root: {request.result_root}")
        print(f"filter_root: {request.filter_root}")
        print(f"output_root: {request.output_root}")
        print(f"shapes: {','.join(request.shapes)}")
        print(f"boundary_mode: {request.boundary_mode}")
        print(f"max_tracks_per_shape: {request.max_tracks_per_shape}")
        print(f"max_objectdays: {request.max_objectdays}")
        return {}
    _require_runtime()
    f0 = 2.0 * 7.2921159e-5 * math.sin(math.radians(request.reference_lat))
    radial, theta, radial_mesh, theta_mesh = _polar_grid(request.rmax, request.radial_bins, request.azimuth_bins)
    outputs: dict[str, Path] = {}
    all_profiles: list[pd.DataFrame] = []
    all_summaries: list[pd.DataFrame] = []

    for shape in request.shapes:
        try:
            rv_root, objects, points = _load_shape_objects(
                request.result_root,
                shape,
                request.max_tracks_per_shape,
                request.max_objectdays,
            )
        except FileNotFoundError:
            if request.skip_missing:
                continue
            raise
        if objects.empty:
            continue
        for orientation in request.orientations:
            for buoyancy_source in request.buoyancy_sources:
                combo_dir = request.output_root / shape / orientation / buoyancy_source
                combo_dir.mkdir(parents=True, exist_ok=True)
                rows: list[pd.DataFrame] = []
                day_cache: dict[date, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
                for obj in objects.sort_values(["track3d_id", "date", "eddy3d_object_id"]).itertuples(index=False):
                    day = pd.Timestamp(obj.date).date()
                    if day not in day_cache:
                        day_cache[day] = _read_filter_day(request.filter_root, request.filter_template, day)
                    lon, lat, u, v, _theta = day_cache[day]
                    sampled = _sample_object_slice(
                        obj,
                        points,
                        lon,
                        lat,
                        u,
                        v,
                        radial,
                        theta,
                        radial_mesh,
                        theta_mesh,
                        orientation,
                    )
                    if sampled is None:
                        continue
                    rep, axis = sampled
                    material_request = MaterialVolumeRequest(
                        result_root=request.result_root,
                        output_root=request.output_root,
                        shapes=(shape,),
                        orientations=(orientation,),
                        buoyancy_sources=(buoyancy_source,),
                        tau_values=(float(rep.tau),) if np.isfinite(rep.tau) else None,
                        reference_lat=request.reference_lat,
                        constant_n2=request.constant_n2,
                        core_radius_over_R=request.core_radius_over_R,
                        speed_core_quantile=request.speed_core_quantile,
                        pv_core_quantile=request.pv_core_quantile,
                        min_mask_fraction=request.min_mask_fraction,
                        boundary_mode=request.boundary_mode,
                        active_contour_iterations=request.active_contour_iterations,
                        leakage_weight=request.leakage_weight,
                        smoothness_weight=request.smoothness_weight,
                        containment_weight=request.containment_weight,
                        area_weight=request.area_weight,
                        vertical_continuity_weight=request.vertical_continuity_weight,
                        time_continuity_weight=request.time_continuity_weight,
                        levelset_sigma_cells=request.levelset_sigma_cells,
                        min_core_retention=request.min_core_retention,
                        min_area_fraction=request.min_area_fraction,
                        max_area_fraction=request.max_area_fraction,
                    )
                    n2 = np.full(rep.depth_m.shape, float(request.constant_n2), dtype="float64")
                    table, _debug = _compute_one_slice(
                        rep=rep,
                        axis=axis,
                        f0=f0,
                        n2=n2,
                        buoyancy_source=buoyancy_source,
                        request=material_request,
                    )
                    table["shape"] = shape
                    table["orientation"] = orientation
                    table["buoyancy_source"] = buoyancy_source
                    table["polarity"] = str(obj.polarity)
                    table["tau"] = float(rep.tau) if np.isfinite(rep.tau) else np.nan
                    table["date"] = str(obj.date)
                    table["track3d_id"] = int(obj.track3d_id)
                    table["eddy3d_object_id"] = int(obj.eddy3d_object_id)
                    table["mean_radius_m"] = float(obj.mean_radius_m)
                    rows.append(table)
                if not rows:
                    continue
                profiles = pd.concat(rows, ignore_index=True)
                summary = _summary_table(profiles)
                _write_table(profiles, combo_dir / "object_material_boundary_profiles.csv")
                _write_table(summary, combo_dir / "object_track_materiality_summary.csv")
                _write_table(profiles, combo_dir / "boundary_v2_profiles.csv")
                flux_cols = [
                    col
                    for col in profiles.columns
                    if col
                    in {
                        "shape",
                        "orientation",
                        "buoyancy_source",
                        "polarity",
                        "tau",
                        "date",
                        "track3d_id",
                        "eddy3d_object_id",
                        "depth_m",
                        "leakage_mean_abs_ms",
                        "leakage_reduction_fraction",
                        "boundary_flux_over_internal_flux",
                    }
                    or col.endswith("_boundary_flux_proxy")
                    or col.endswith("_boundary_flux_integral_proxy")
                ]
                _write_table(profiles[flux_cols], combo_dir / "boundary_flux_budget.csv")
                (combo_dir / "object_material_boundary_manifest.json").write_text(
                    json.dumps(
                        _json_ready(
                            {
                                "shape": shape,
                                "orientation": orientation,
                                "buoyancy_source": buoyancy_source,
                                "rv_root": rv_root,
                                "boundary_mode": request.boundary_mode,
                                "radial_bins": request.radial_bins,
                                "azimuth_bins": request.azimuth_bins,
                                "rmax": request.rmax,
                                "n_objectdays": int(profiles["eddy3d_object_id"].nunique()),
                                "n_tracks": int(profiles["track3d_id"].nunique()),
                                "diagnostic_status": "object-level validation; center-following local section; no representative averaging",
                            }
                        ),
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                all_profiles.append(profiles)
                all_summaries.append(summary.assign(shape=shape, orientation=orientation, buoyancy_source=buoyancy_source))
                outputs[str(combo_dir)] = combo_dir
    if all_profiles:
        request.output_root.mkdir(parents=True, exist_ok=True)
        root_profiles = pd.concat(all_profiles, ignore_index=True)
        root_summary = pd.concat(all_summaries, ignore_index=True)
        _write_table(root_profiles, request.output_root / "object_material_boundary_profiles.csv")
        _write_table(root_summary, request.output_root / "object_track_materiality_summary.csv")
        for plot_path in _plot_object_summaries(request.output_root, root_profiles, root_summary):
            outputs[f"figure:{plot_path.name}"] = plot_path
        _write_summary_md(request.output_root / "object_material_boundary_validation_summary_zh.md", root_summary, request)
        (request.output_root / "object_material_boundary_manifest.json").write_text(
            json.dumps(_json_ready({"request": request.__dict__}), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        outputs["root_summary"] = request.output_root / "object_track_materiality_summary.csv"
    return outputs


def request_from_args(args) -> ObjectBoundaryRequest:
    return ObjectBoundaryRequest(
        result_root=Path(args.result_root),
        filter_root=Path(args.filter_root),
        output_root=Path(args.output_root),
        shapes=_split_csv(args.shapes),
        orientations=_split_csv(args.orientations),
        buoyancy_sources=_split_csv(args.buoyancy_sources),
        filter_template=str(args.filter_template),
        radial_bins=int(args.radial_bins),
        azimuth_bins=int(args.azimuth_bins),
        rmax=float(args.rmax),
        reference_lat=float(args.reference_lat),
        constant_n2=float(args.constant_n2),
        core_radius_over_R=float(args.core_radius_over_R),
        speed_core_quantile=float(args.speed_core_quantile),
        pv_core_quantile=float(args.pv_core_quantile),
        min_mask_fraction=float(args.min_mask_fraction),
        boundary_mode=str(args.boundary_mode),
        active_contour_iterations=int(args.active_contour_iterations),
        leakage_weight=float(args.leakage_weight),
        smoothness_weight=float(args.smoothness_weight),
        containment_weight=float(args.containment_weight),
        area_weight=float(args.area_weight),
        vertical_continuity_weight=float(args.vertical_continuity_weight),
        time_continuity_weight=float(args.time_continuity_weight),
        levelset_sigma_cells=float(args.levelset_sigma_cells),
        min_core_retention=float(args.min_core_retention),
        min_area_fraction=float(args.min_area_fraction),
        max_area_fraction=float(args.max_area_fraction),
        max_tracks_per_shape=int(args.max_tracks_per_shape),
        max_objectdays=int(args.max_objectdays),
        skip_missing=bool(args.skip_missing),
        dry_run=bool(args.dry_run),
    )
