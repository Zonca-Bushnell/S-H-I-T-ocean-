from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .contracts import (
    AXIS_SOURCES,
    BUOYANCY_SOURCES,
    CURVED_TUBE_MODES,
    DEFAULT_FULL_OUTPUT_ROOT,
    DEFAULT_RESULT_ROOT,
    ORIENTATIONS,
    EPFluxConfig,
    default_me_liutex_root,
    default_radial_seed_root,
    shape_output_name,
)


@dataclass(frozen=True)
class LifecycleRequest:
    result_root: Path = DEFAULT_RESULT_ROOT
    output_root: Path = DEFAULT_FULL_OUTPUT_ROOT
    shapes: tuple[str, ...] = ("coherent",)
    axis_sources: tuple[str, ...] = ("radial_seed",)
    orientations: tuple[str, ...] = ("turned",)
    buoyancy_sources: tuple[str, ...] = ("thermal_wind",)
    tau_values: tuple[float, ...] | None = None
    reference_lat: float = 30.0
    constant_n2: float = 2.0e-5
    n2_profile: str | None = "auto"
    curved_tube_mode: str = "scale_audit"
    large_curvature_threshold: float = 1.0
    bootstrap_samples: int = 0
    bootstrap_unit: str = "track"
    ensure_axis_sources: bool = False
    dry_run: bool = False
    skip_missing: bool = False


def _split_csv(value: str | tuple[str, ...] | list[str]) -> tuple[str, ...]:
    if isinstance(value, (tuple, list)):
        return tuple(str(v).strip() for v in value if str(v).strip())
    return tuple(part.strip() for part in str(value).split(",") if part.strip())


def _parse_tau_values(value: str | None) -> tuple[float, ...] | None:
    if value in (None, ""):
        return None
    return tuple(float(part.strip()) for part in str(value).split(",") if part.strip())


def _combo_output(root: Path, shape: str, axis_source: str, orientation: str, buoyancy_source: str) -> Path:
    return root / shape / axis_source / orientation / buoyancy_source


def _shape_label(shape: str) -> str:
    return f"{shape}-only"


def _ensure_axis_source(
    *,
    result_root: Path,
    shape: str,
    orientation: str,
    tau: float,
    axis_source: str,
) -> None:
    if axis_source not in AXIS_SOURCES:
        raise ValueError(f"Unsupported axis source: {axis_source}")
    output_name = shape_output_name(shape)
    me_root = default_me_liutex_root(result_root, output_name, orientation)
    path = me_root / "axis_sources" / f"{axis_source}_axis_tau{int(round(tau * 100)):03d}.csv"
    if path.exists():
        return
    from .axis_sources import build_representative_axis_sources_for_root

    turned_root = default_me_liutex_root(result_root, output_name, "turned")
    unturned_root = default_me_liutex_root(result_root, output_name, "unturned")
    radial_root = default_radial_seed_root(result_root, output_name)
    root = turned_root if orientation == "turned" else unturned_root
    build_representative_axis_sources_for_root(
        me_liutex_root=root,
        radial_seed_root=radial_root,
        tau=float(tau),
        orientation=orientation,
    )


def _tau_grid_for_combo(result_root: Path, shape: str, orientation: str, tau_values: tuple[float, ...] | None) -> np.ndarray:
    import numpy as np

    if tau_values is not None:
        return np.asarray(tau_values, dtype=float)
    from .fields import RepresentativeVortexDataset

    output_name = shape_output_name(shape)
    me_root = default_me_liutex_root(result_root, output_name, orientation)
    dataset = RepresentativeVortexDataset.load(
        me_root / "azimuthal_representative_velocity.npz",
        default_radial_seed_root(result_root, output_name),
    )
    return np.asarray(dataset.tau_grid, dtype=float)


def _write_table(table, csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(csv_path, index=False)
    try:
        table.to_parquet(csv_path.with_suffix(".parquet"), index=False)
    except Exception:
        pass


def _summary_markdown(
    request: LifecycleRequest,
    combo: dict[str, str],
    summary,
    cache_reason: str,
    bootstrap_done: bool,
) -> str:
    lines = [
        "# EP Full Lifecycle Validation Summary",
        "",
        "## 口径",
        f"- shape: `{combo['shape']}`",
        f"- axis source: `{combo['axis_source']}`",
        f"- orientation: `{combo['orientation']}`",
        f"- buoyancy source: `{combo['buoyancy_source']}`",
        f"- curved-tube mode: `{request.curved_tube_mode}`",
        "",
        "## 结果表",
        "```text",
        summary.to_string(index=False),
        "```",
        "",
        "## Bootstrap / Jackknife",
    ]
    if bootstrap_done:
        lines.append("- 已基于 track-level velocity accumulator 计算严格 bootstrap/jackknife。")
    else:
        lines.append(f"- 未计算严格 bootstrap/jackknife：{cache_reason}")
        lines.append("- 不能把最终代表涡均值场重新抽样当作严格置信区间。")
    lines.extend(
        [
            "",
            "## 判读",
            "- 主结论仍优先看 `turned + radial_seed + thermal_wind`。",
            "- `F_z_tilt_correction / F_z_ordinary` 若在多个 tau 保持同量级，说明倾斜修正不是 tau=0.50 的偶然现象。",
            "- `divF_tilted` 与 `pv_flux_proxy` 的相关和误差用于检查 EP-PV closure。",
            "- `epsilon_curvature` 大或 `metric_valid_fraction` 低时，只能说明曲管项需要更完整理论闭合。",
        ]
    )
    return "\n".join(lines) + "\n"


def run_lifecycle_validation(request: LifecycleRequest) -> dict[str, Path]:
    bad_axis = sorted(set(request.axis_sources) - set(AXIS_SOURCES))
    bad_orient = sorted(set(request.orientations) - set(ORIENTATIONS))
    bad_buoy = sorted(set(request.buoyancy_sources) - set(BUOYANCY_SOURCES))
    if bad_axis or bad_orient or bad_buoy:
        raise ValueError(f"Bad options: axis={bad_axis}, orientation={bad_orient}, buoyancy={bad_buoy}")
    if request.curved_tube_mode not in CURVED_TUBE_MODES:
        raise ValueError(f"curved_tube_mode must be one of {CURVED_TUBE_MODES}")

    if not request.dry_run:
        import pandas as pd

        from .diagnostics import compute_ep_profiles
        from .plots import plot_cross_combo_summary, plot_lifecycle_figures
        from .statistics import lifecycle_summary, metric_audit, write_placeholder_strict_ci, write_summary_json
        from .track_accumulator import inspect_track_cache, write_track_cache_manifest

    written: dict[str, Path] = {}
    all_summaries: list[pd.DataFrame] = []
    manifest_rows: list[dict[str, object]] = []
    for shape in request.shapes:
        output_name = shape_output_name(shape)
        radial_root = default_radial_seed_root(request.result_root, output_name)
        for orientation in request.orientations:
            me_root = default_me_liutex_root(request.result_root, output_name, orientation)
            if not request.dry_run and not me_root.exists():
                if request.skip_missing:
                    continue
                raise FileNotFoundError(me_root)
            if request.dry_run and request.tau_values is None:
                tau_grid = []
                tau_count_text = "all tau nodes from representative npz"
            else:
                tau_grid = _tau_grid_for_combo(request.result_root, shape, orientation, request.tau_values)
                tau_count_text = str(len(tau_grid))
            for axis_source in request.axis_sources:
                for buoyancy_source in request.buoyancy_sources:
                    combo_dir = _combo_output(request.output_root, shape, axis_source, orientation, buoyancy_source)
                    combo = {
                        "shape": shape,
                        "axis_source": axis_source,
                        "orientation": orientation,
                        "buoyancy_source": buoyancy_source,
                    }
                    if request.dry_run:
                        print(f"[dry-run] {combo_dir}")
                        print(f"  me_liutex_root={me_root}")
                        print(f"  radial_seed_root={radial_root}")
                        print(f"  tau_count={tau_count_text}")
                        continue
                    combo_dir.mkdir(parents=True, exist_ok=True)
                    profiles_list: list[pd.DataFrame] = []
                    manifests: list[dict[str, object]] = []
                    for tau in tau_grid:
                        if request.ensure_axis_sources:
                            _ensure_axis_source(
                                result_root=request.result_root,
                                shape=shape,
                                orientation=orientation,
                                tau=float(tau),
                                axis_source=axis_source,
                            )
                        config = EPFluxConfig(
                            me_liutex_root=me_root,
                            radial_seed_root=radial_root,
                            output_dir=combo_dir,
                            orientation=orientation,
                            axis_source=axis_source,
                            tau=float(tau),
                            reference_lat=request.reference_lat,
                            constant_n2=request.constant_n2,
                            buoyancy_source=buoyancy_source,
                            curved_tube_mode=request.curved_tube_mode,
                            large_curvature_threshold=request.large_curvature_threshold,
                            shape_label=_shape_label(shape),
                            run_label="full_lifecycle_validation",
                        )
                        profiles, _, manifest = compute_ep_profiles(config, n2_profile=request.n2_profile)
                        profiles["shape"] = shape
                        profiles["axis_source"] = axis_source
                        profiles["orientation"] = orientation
                        profiles["buoyancy_source"] = buoyancy_source
                        profiles_list.append(profiles)
                        manifests.append(manifest)
                    profiles_all = pd.concat(profiles_list, ignore_index=True)
                    summary = lifecycle_summary(profiles_all)
                    audit = metric_audit(profiles_all)
                    _write_table(profiles_all, combo_dir / "ep_lifecycle_profiles.csv")
                    _write_table(summary, combo_dir / "ep_lifecycle_summary.csv")
                    _write_table(audit, combo_dir / "ep_metric_audit.csv")

                    cache_status = inspect_track_cache(me_root)
                    write_track_cache_manifest(combo_dir, cache_status)
                    bootstrap_done = False
                    write_placeholder_strict_ci(
                        combo_dir / "ep_lifecycle_bootstrap_ci.csv",
                        bootstrap_samples=request.bootstrap_samples,
                        bootstrap_unit=request.bootstrap_unit,
                        reason=cache_status.reason,
                    )
                    pd.DataFrame(
                        [
                            {
                                "status": "not_computed",
                                "bootstrap_unit": request.bootstrap_unit,
                                "reason": cache_status.reason,
                            }
                        ]
                    ).to_csv(combo_dir / "ep_lifecycle_jackknife.csv", index=False)

                    manifest = {
                        "combo": combo,
                        "tau_values": [float(v) for v in tau_grid],
                        "result_root": str(request.result_root),
                        "output_dir": str(combo_dir),
                        "strict_bootstrap_requested": int(request.bootstrap_samples) > 0,
                        "strict_bootstrap_done": bootstrap_done,
                        "track_cache_status": cache_status.reason,
                        "manifests": manifests,
                    }
                    write_summary_json(combo_dir / "ep_lifecycle_summary.json", summary, manifest)
                    (combo_dir / "ep_lifecycle_validation_summary_zh.md").write_text(
                        _summary_markdown(request, combo, summary, cache_status.reason, bootstrap_done),
                        encoding="utf-8",
                    )
                    figures = plot_lifecycle_figures(profiles_all, summary, audit, combo_dir / "figures")
                    written[str(combo_dir)] = combo_dir
                    all_summaries.append(summary)
                    manifest_rows.append(manifest)
                    for fig in figures:
                        written[str(fig)] = fig

    if not request.dry_run and all_summaries:
        root_summary = pd.concat(all_summaries, ignore_index=True)
        _write_table(root_summary, request.output_root / "ep_lifecycle_all_combo_summary.csv")
        (request.output_root / "ep_lifecycle_manifest.json").write_text(
            json.dumps(manifest_rows, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        for fig in plot_cross_combo_summary(root_summary, request.output_root / "figures"):
            written[str(fig)] = fig
    return written


def request_from_args(args) -> LifecycleRequest:
    return LifecycleRequest(
        result_root=Path(args.result_root),
        output_root=Path(args.output_root),
        shapes=_split_csv(args.shapes),
        axis_sources=_split_csv(args.axis_sources),
        orientations=_split_csv(args.orientations),
        buoyancy_sources=_split_csv(args.buoyancy_sources),
        tau_values=_parse_tau_values(args.tau_values),
        reference_lat=float(args.reference_lat),
        constant_n2=float(args.constant_n2),
        n2_profile=args.n2_profile,
        curved_tube_mode=args.curved_tube_mode,
        large_curvature_threshold=float(args.large_curvature_threshold),
        bootstrap_samples=int(args.bootstrap_samples),
        bootstrap_unit=str(args.bootstrap_unit),
        ensure_axis_sources=bool(args.ensure_axis_sources),
        dry_run=bool(args.dry_run),
        skip_missing=bool(args.skip_missing),
    )
