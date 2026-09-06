"""Formal dual-zone EP diagnostics.

This module promotes the tested core-shell V2 workflow into the default
interpretation contract used by the EP package.  It keeps the numerical kernel
in :mod:`src.EP.core_shell_runner`, then writes stable dual-zone table names,
figures, and a LaTeX report.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import pandas as pd
except ModuleNotFoundError:  # pragma: no cover - dry-run can still work.
    pd = None

from .contracts import DEFAULT_RESULT_ROOT
from .core_shell_runner import CoreShellRequest, run_core_shell_ep_validation
from .material_volume import _json_ready, _write_table


DEFAULT_DUAL_ZONE_OUTPUT_ROOT = Path(
    "/root/autodl-fs/kuroshiou/EP-FLUX/dual_zone_ep_diagnostics"
)

LOCAL_MEDIUM_AUDIT_ROOT = Path(
    "G:/TEMP/kuroshiou_ep_object_material_pv_retention_validation_medium_8tracks"
)
SERVER_MEDIUM_AUDIT_ROOT = Path(
    "/root/autodl-fs/kuroshiou/EP-FLUX/object_material_pv_retention_ep_validation_medium_8tracks"
)


@dataclass(frozen=True)
class DualZoneRequest:
    result_root: Path = DEFAULT_RESULT_ROOT
    output_root: Path = DEFAULT_DUAL_ZONE_OUTPUT_ROOT
    filter_root: Path = Path("/root/autodl-fs/kuroshiou/Filter")
    filter_template: str = "global_phy_{year}_bandpass_30_180d.nc"
    shapes: tuple[str, ...] = ("coherent", "upright_like")
    axis_sources: tuple[str, ...] = ("radial_seed",)
    orientations: tuple[str, ...] = ("turned",)
    buoyancy_sources: tuple[str, ...] = ("thermal_wind",)
    tau_values: tuple[float, ...] | None = None
    reference_lat: float = 30.0
    constant_n2: float = 2.0e-5
    n2_profile: str | None = "auto"
    inner_boundary_mode: str = "levelset_v2"
    boundary_budget: str = "full_3d"
    core_radius_over_R: float = 1.2
    shell_outer_radius_over_R: float = 2.5
    speed_core_quantile: float = 0.45
    pv_core_quantile: float = 0.70
    pv_shell_quantile: float = 0.80
    shell_dilation_cells: int = 2
    min_mask_fraction: float = 0.01
    min_core_retention: float = 0.75
    object_aggregate_transport: bool = True
    object_aggregate_max_days: int = 0
    object_aggregate_max_objects: int = 0
    object_boundary_audit: str = "medium_8tracks"
    object_boundary_audit_root: Path | None = None
    compile_pdf: bool = True
    skip_missing: bool = False
    dry_run: bool = False


def _split_csv(value: str | tuple[str, ...] | list[str]) -> tuple[str, ...]:
    if isinstance(value, (tuple, list)):
        return tuple(str(v).strip() for v in value if str(v).strip())
    return tuple(part.strip() for part in str(value).split(",") if part.strip())


def _parse_tau_values(value: str | None) -> tuple[float, ...] | None:
    if value in (None, ""):
        return None
    return tuple(float(part.strip()) for part in str(value).split(",") if part.strip())


def _require_pandas() -> None:
    if pd is None:
        raise ModuleNotFoundError("pandas is required for dual-zone EP diagnostics")


def _core_shell_request(request: DualZoneRequest) -> CoreShellRequest:
    return CoreShellRequest(
        result_root=request.result_root,
        output_root=request.output_root,
        filter_root=request.filter_root,
        filter_template=request.filter_template,
        shapes=request.shapes,
        axis_sources=request.axis_sources,
        orientations=request.orientations,
        buoyancy_sources=request.buoyancy_sources,
        tau_values=request.tau_values,
        reference_lat=request.reference_lat,
        constant_n2=request.constant_n2,
        n2_profile=request.n2_profile,
        inner_boundary_mode=request.inner_boundary_mode,
        boundary_budget=request.boundary_budget,
        core_radius_over_R=request.core_radius_over_R,
        shell_outer_radius_over_R=request.shell_outer_radius_over_R,
        speed_core_quantile=request.speed_core_quantile,
        pv_core_quantile=request.pv_core_quantile,
        pv_shell_quantile=request.pv_shell_quantile,
        shell_dilation_cells=request.shell_dilation_cells,
        min_mask_fraction=request.min_mask_fraction,
        min_core_retention=request.min_core_retention,
        object_aggregate_transport=request.object_aggregate_transport,
        object_aggregate_max_days=request.object_aggregate_max_days,
        object_aggregate_max_objects=request.object_aggregate_max_objects,
        skip_missing=request.skip_missing,
        dry_run=request.dry_run,
    )


def _resolve_audit_root(request: DualZoneRequest) -> Path | None:
    if request.object_boundary_audit == "none":
        return None
    if request.object_boundary_audit_root is not None:
        return request.object_boundary_audit_root
    if LOCAL_MEDIUM_AUDIT_ROOT.exists():
        return LOCAL_MEDIUM_AUDIT_ROOT
    if SERVER_MEDIUM_AUDIT_ROOT.exists():
        return SERVER_MEDIUM_AUDIT_ROOT
    return None


def _read_csv(path: Path):
    _require_pandas()
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _copy_existing(src: Path, dst: Path, written: dict[str, Path], key: str) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        written[key] = dst


def _select_columns(frame, tokens: tuple[str, ...]):
    keep = []
    id_cols = (
        "shape",
        "axis_source",
        "orientation",
        "buoyancy_source",
        "tau",
        "polarity",
        "depth_index",
        "depth_m",
        "region",
    )
    for col in frame.columns:
        if col in id_cols or any(token in col for token in tokens):
            keep.append(col)
    return frame[keep] if keep else frame


def _write_standard_tables(output_root: Path) -> dict[str, Path]:
    _require_pandas()
    written: dict[str, Path] = {}
    profiles = _read_csv(output_root / "core_shell_all_profiles.csv")
    summary = _read_csv(output_root / "core_shell_all_summary.csv")
    if profiles.empty:
        return written

    table_specs = [
        ("dual_zone_partition_profiles", profiles),
        (
            "region_transport_moments",
            _select_columns(
                profiles,
                (
                    "theta",
                    "q",
                    "heat",
                    "pv",
                    "product_mean",
                    "mean_product",
                    "covariance",
                    "object_aggregate",
                    "fraction_of_total_abs",
                ),
            ),
        ),
        (
            "region_ep_tilt_profiles",
            _select_columns(
                profiles,
                (
                    "F_z",
                    "tilt",
                    "ordinary",
                    "tilted",
                    "region_fraction_of_total_abs_tilt",
                ),
            ),
        ),
        (
            "boundary_exchange_budget",
            _select_columns(
                profiles,
                (
                    "boundary",
                    "exchange",
                    "volume_flux",
                    "heat_flux",
                    "pv_flux",
                    "buoyancy_flux",
                    "momentum",
                    "lateral",
                    "top",
                    "bottom",
                    "core_shell_interface",
                ),
            ),
        ),
    ]
    for stem, frame in table_specs:
        path = output_root / f"{stem}.csv"
        _write_table(frame, path)
        written[stem] = path
    if not summary.empty:
        path = output_root / "dual_zone_summary_source.csv"
        _write_table(summary, path)
        written["dual_zone_summary_source"] = path
    return written


def _median_or_nan(frame, column: str) -> float:
    if frame.empty or column not in frame.columns:
        return float("nan")
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    if values.empty:
        return float("nan")
    return float(values.median())


def _write_tradeoff_summary(request: DualZoneRequest, output_root: Path) -> tuple[Path, dict[str, Any]]:
    _require_pandas()
    audit_root = _resolve_audit_root(request)
    rows: list[dict[str, Any]] = []
    status: dict[str, Any] = {
        "object_boundary_audit": request.object_boundary_audit,
        "audit_root": str(audit_root) if audit_root is not None else None,
        "audit_found": False,
    }
    if audit_root is None or not audit_root.exists():
        rows.append(
            {
                "audit_status": "missing",
                "interpretation": "object-level boundary audit not found; dual-zone report uses representative core-shell diagnostics only",
            }
        )
    else:
        source = audit_root / "shape_materiality_comparison.csv"
        audit = _read_csv(source)
        if audit.empty:
            rows.append({"audit_status": "empty", "source_file": str(source)})
        else:
            status["audit_found"] = True
            group_cols = [col for col in ["shape", "boundary_mode"] if col in audit.columns]
            grouped = audit.groupby(group_cols, dropna=False) if group_cols else [((), audit)]
            for keys, sub in grouped:
                if not isinstance(keys, tuple):
                    keys = (keys,)
                row = {"audit_status": "ok", "source_file": str(source)}
                for col, key in zip(group_cols, keys):
                    row[col] = key
                row.update(
                    {
                        "pv_core_retention_median": _median_or_nan(sub, "pv_core_retention_median"),
                        "pv_abs_retention_median": _median_or_nan(sub, "pv_abs_retention_median"),
                        "pv_high_quantile_retention_median": _median_or_nan(
                            sub, "pv_high_quantile_retention_median"
                        ),
                        "leakage_median_ms": _median_or_nan(sub, "leakage_median_ms"),
                        "boundary_flux_over_internal_flux_median": _median_or_nan(
                            sub, "boundary_flux_over_internal_flux_median"
                        ),
                        "closure_residual_proxy_median": _median_or_nan(
                            sub, "closure_residual_proxy_median"
                        ),
                        "particle_retention_median": _median_or_nan(sub, "particle_retention_median"),
                    }
                )
                rows.append(row)
    tradeoff = pd.DataFrame(rows)
    path = output_root / "materiality_tradeoff_summary.csv"
    _write_table(tradeoff, path)
    return path, status


def _write_decision_table(output_root: Path, tradeoff_path: Path) -> Path:
    _require_pandas()
    profiles = _read_csv(output_root / "dual_zone_partition_profiles.csv")
    tradeoff = _read_csv(tradeoff_path)
    rows: list[dict[str, Any]] = []
    if not profiles.empty:
        for shape, sub in profiles.groupby("shape", dropna=False):
            inner = sub[sub["region"].astype(str).eq("inner_core")]
            shell = sub[sub["region"].astype(str).eq("pv_shell")]
            rows.append(
                {
                    "shape": shape,
                    "diagnostic": "dual_zone_partition",
                    "inner_material_core_mean_abs_speed_ms": _median_or_nan(inner, "mean_speed_ms"),
                    "inner_material_core_pv_retention": _median_or_nan(
                        inner, "pv_high_quantile_retention"
                    ),
                    "pv_shell_pv_retention": _median_or_nan(shell, "pv_high_quantile_retention"),
                    "shell_heat_covariance_fraction": _median_or_nan(
                        shell, "object_aggregate_region_fraction_of_total_abs_heat_covariance"
                    ),
                    "shell_pv_covariance_fraction": _median_or_nan(
                        shell, "object_aggregate_region_fraction_of_total_abs_pv_covariance"
                    ),
                    "shell_ep_tilt_fraction": _median_or_nan(
                        shell, "region_fraction_of_total_abs_tilt_correction"
                    ),
                    "interpretation": (
                        "inner core is read as trapping/material-coherence region; "
                        "PV-active shell is read as stirring and tilt-correction region"
                    ),
                }
            )
    if not tradeoff.empty and "audit_status" in tradeoff.columns:
        ok = tradeoff[tradeoff["audit_status"].astype(str).eq("ok")]
        if not ok.empty:
            rows.append(
                {
                    "shape": "object_medium_audit",
                    "diagnostic": "pv_retention_vs_leakage_tradeoff",
                    "inner_material_core_mean_abs_speed_ms": float("nan"),
                    "inner_material_core_pv_retention": float("nan"),
                    "pv_shell_pv_retention": float("nan"),
                    "shell_heat_covariance_fraction": float("nan"),
                    "shell_pv_covariance_fraction": float("nan"),
                    "shell_ep_tilt_fraction": float("nan"),
                    "interpretation": (
                        "PV-retention-aware boundaries are used as an audit: if PV retention "
                        "rises with leakage/exchange, PV core and material rotation core are separated"
                    ),
                }
            )
    decision = pd.DataFrame(rows)
    path = output_root / "dual_zone_decision_table.csv"
    _write_table(decision, path)
    return path


def _copy_standard_figures(output_root: Path, audit_root: Path | None) -> dict[str, Path]:
    written: dict[str, Path] = {}
    fig_dir = output_root / "figures"
    mapping = {
        "object_aggregate_heat_core_vs_shell_partition.png": "heat_pv_core_shell_partition.png",
        "ep_tilt_correction_core_vs_shell.png": "ep_tilt_core_shell_partition.png",
        "core_shell_exchange_budget.png": "boundary_exchange_budget.png",
        "coherent_vs_upright_like_partition.png": "coherent_vs_upright_like_dual_zone.png",
        "dual_zone_framework_summary.png": "dual_zone_framework_summary.png",
    }
    for src_name, dst_name in mapping.items():
        _copy_existing(fig_dir / src_name, fig_dir / dst_name, written, f"figure:{dst_name}")
    if audit_root is not None:
        _copy_existing(
            audit_root / "figures" / "pv_retention_vs_leakage.png",
            fig_dir / "pv_retention_vs_leakage_tradeoff.png",
            written,
            "figure:pv_retention_vs_leakage_tradeoff.png",
        )
        _copy_existing(
            audit_root / "figures" / "heat_pv_momentum_boundary_budget.png",
            fig_dir / "object_boundary_exchange_budget.png",
            written,
            "figure:object_boundary_exchange_budget.png",
        )
    return written


def _safe_float_text(value: Any, fmt: str = ".3g") -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "NA"
    if not pd.notna(number):
        return "NA"
    return format(number, fmt)


def _read_key_metrics(output_root: Path, tradeoff_path: Path) -> dict[str, str]:
    _require_pandas()
    profiles = _read_csv(output_root / "dual_zone_partition_profiles.csv")
    tradeoff = _read_csv(tradeoff_path)
    metrics: dict[str, str] = {
        "shell_heat_fraction": "NA",
        "shell_pv_fraction": "NA",
        "shell_tilt_fraction": "NA",
        "pv_retention_tradeoff": "medium audit unavailable",
    }
    if not profiles.empty:
        shell = profiles[profiles["region"].astype(str).eq("pv_shell")]
        metrics["shell_heat_fraction"] = _safe_float_text(
            _median_or_nan(shell, "object_aggregate_region_fraction_of_total_abs_heat_covariance"),
            ".2f",
        )
        metrics["shell_pv_fraction"] = _safe_float_text(
            _median_or_nan(shell, "object_aggregate_region_fraction_of_total_abs_pv_covariance"),
            ".2f",
        )
        metrics["shell_tilt_fraction"] = _safe_float_text(
            _median_or_nan(shell, "region_fraction_of_total_abs_tilt_correction"),
            ".2f",
        )
    if not tradeoff.empty and {"boundary_mode", "pv_core_retention_median", "leakage_median_ms"}.issubset(
        tradeoff.columns
    ):
        ok = tradeoff[tradeoff.get("audit_status", "ok").astype(str).eq("ok")]
        if not ok.empty:
            pieces = []
            for _, row in ok.iterrows():
                mode = str(row.get("boundary_mode", "boundary"))
                shape = str(row.get("shape", "shape"))
                pv = _safe_float_text(row.get("pv_core_retention_median"), ".2f")
                leak = _safe_float_text(row.get("leakage_median_ms"), ".3f")
                pieces.append(f"{shape}/{mode}: PV retention {pv}, leakage {leak} m/s")
            metrics["pv_retention_tradeoff"] = "; ".join(pieces[:6])
    return metrics


def _write_markdown_report(request: DualZoneRequest, output_root: Path, tradeoff_path: Path) -> Path:
    metrics = _read_key_metrics(output_root, tradeoff_path)
    lines = [
        "# Dual-Zone EP Diagnostic Report",
        "",
        "## 正式诊断口径",
        "",
        "本报告把 EP 诊断的默认解释从单一材料涡体积改为双区结构：",
        "",
        r"\\[",
        r"\\mathcal{T}_{total}=\\mathcal{T}_{core}^{trap}+\\mathcal{T}_{shell}^{stir}+\\mathcal{T}_{exchange}.",
        r"\\]",
        "",
        "- `inner_material_core`：低泄漏、Hua/LAVD 近同位的运动学材料核，用于解释 trapping 和 material coherence。",
        "- `pv_active_shell`：高 `|q'|`、高 `|\\nabla q'|`、强剪切或月牙强速带所在区域，用于解释 heat/PV stirring 和 EP 倾斜修正。",
        "- `exchange_layer`：inner core 与 PV-active shell 的接触带，单独列账 heat/PV/momentum boundary exchange。",
        "",
        "## 当前默认参数",
        "",
        f"- result root: `{request.result_root}`",
        f"- shapes: `{','.join(request.shapes)}`",
        f"- axis sources: `{','.join(request.axis_sources)}`",
        f"- orientations: `{','.join(request.orientations)}`",
        f"- buoyancy sources: `{','.join(request.buoyancy_sources)}`",
        f"- inner core: `r/R <= {request.core_radius_over_R}`",
        f"- PV-active shell: `r/R <= {request.shell_outer_radius_over_R}`, `|q'|` quantile >= `{request.pv_shell_quantile}`",
        f"- object boundary audit: `{request.object_boundary_audit}`",
        "",
        "## 关键判定",
        "",
        f"- shell heat covariance fraction median: `{metrics['shell_heat_fraction']}`。",
        f"- shell PV covariance fraction median: `{metrics['shell_pv_fraction']}`。",
        f"- shell EP tilt-correction fraction median: `{metrics['shell_tilt_fraction']}`。",
        f"- object-level PV-retention tradeoff: {metrics['pv_retention_tradeoff']}。",
        "",
        "若 PV-retention-aware 边界提高了 PV core retention，但 leakage、boundary exchange 或 closure residual 同步增大，",
        "本框架将其解释为 PV 动力核心与低泄漏旋转材料核的分离，而不是继续强迫 `PV core subset LAVD core`。",
        "",
        "## 输出文件",
        "",
        "- `dual_zone_partition_profiles.csv`：tau/depth/polarity/region 的分区属性。",
        "- `region_transport_moments.csv`：分区 heat/PV aggregate-product moments 与 covariance。",
        "- `region_ep_tilt_profiles.csv`：分区 ordinary/tilted EP vertical flux 与 tilt correction。",
        "- `boundary_exchange_budget.csv`：core/shell/exchange 的边界交换项。",
        "- `materiality_tradeoff_summary.csv`：LAVD/geodesic/PV-retention 审计对照。",
        "- `dual_zone_decision_table.csv`：最终判定表。",
        "",
    ]
    path = output_root / "dual_zone_ep_diagnostic_report_zh.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _figure_latex(path: Path, caption: str) -> str:
    if not path.exists():
        return ""
    tex_path = str(path).replace("\\", "/")
    return "\n".join(
        [
            r"\begin{figure}[htbp]",
            r"\centering",
            rf"\includegraphics[width=0.92\linewidth]{{{tex_path}}}",
            rf"\caption{{{caption}}}",
            r"\end{figure}",
            "",
        ]
    )


def _write_latex_report(request: DualZoneRequest, output_root: Path, tradeoff_path: Path) -> Path:
    metrics = _read_key_metrics(output_root, tradeoff_path)
    fig_dir = output_root / "figures"
    body = rf"""
\documentclass[UTF8]{{ctexart}}
\usepackage[a4paper,margin=2.2cm]{{geometry}}
\usepackage{{amsmath,amssymb,booktabs,graphicx,hyperref,float}}
\hypersetup{{colorlinks=true,linkcolor=blue,urlcolor=blue,citecolor=blue}}
\title{{Dual-Zone EP 正式诊断口径报告}}
\author{{S-H-I-T-ocean / src.EP}}
\date{{\today}}

\begin{{document}}
\maketitle

\section{{为什么从单一材料体改为双区结构}}

已有验证显示，低泄漏材料核、LAVD 旋转相干核和 PV anomaly 动力核心并不总是同位。
若继续要求一个边界同时满足低泄漏、强 PV retention、强 heat/PV stirring 和 EP 闭合，
会把不同物理过程硬塞进同一个体积。新的正式口径写为
\[
\mathcal{{T}}_{{total}}
=\mathcal{{T}}_{{core}}^{{trap}}
+\mathcal{{T}}_{{shell}}^{{stir}}
+\mathcal{{T}}_{{exchange}} .
\]
其中 inner material core 负责 trapping/material coherence；PV-active shell 负责热、PV 与动量扰动输送；
exchange layer 单独记录 core 与 shell 之间的边界交换。

\section{{分区判据}}

默认 inner material core 取 $r/R\le {request.core_radius_over_R:.2f}$ 的弱速连通材料核，
并要求低 leakage 与高 weak-core retention。PV-active shell 取 inner core 外侧、
$r/R\le {request.shell_outer_radius_over_R:.2f}$ 且高 $|q'|$ 或高 $|\nabla q'|$ 的连通区域；
默认高 PV 阈值为分位数 {request.pv_shell_quantile:.2f}。exchange layer 为 inner core 边界外侧
1--2 个格点和 shell 接触带，用于独立列账 boundary exchange。

\section{{输送与 EP 量}}

热与 PV stirring 不用平均结构相乘，而用 aggregate-product：
\[
H_M=\rho_0 C_p\langle v'_\mathrm{{rot}}\theta'\rangle_M,\qquad
P_M=\langle v'_\mathrm{{rot}}q'\rangle_M .
\]
同时保存
\[
\langle vX\rangle_M,\quad \langle v\rangle_M\langle X\rangle_M,\quad
\mathrm{{cov}}_M=\langle vX\rangle_M-\langle v\rangle_M\langle X\rangle_M .
\]
垂向 EP 项继续区分 ordinary 与 tilted-coordinate correction：
\[
F_z^{{tilted}}=F_z^{{ordinary}}+F_z^{{tilt\ correction}} .
\]
已完成全生命周期 EP 验证表明
$|F_z^{{tilt\ correction}}|/|F_z^{{ordinary}}|$ 的中位数量级约为 0.47，
因此倾斜坐标修正不是小量。

\section{{当前结果摘要}}

PV-active shell 的 heat covariance 中位贡献比例约为 {metrics["shell_heat_fraction"]}；
PV covariance 中位贡献比例约为 {metrics["shell_pv_fraction"]}；
EP tilt correction 中位贡献比例约为 {metrics["shell_tilt_fraction"]}。
object-level medium audit 显示：{metrics["pv_retention_tradeoff"]}。
因此，当 PV retention 提高伴随 leakage 或 boundary exchange 增大时，
应解释为 PV-active shell 与材料旋转核分离，而不是 EP 公式本身已经闭合失败。

\section{{曲管几何项的地位}}

thin curved tube 的 metric/Jacobian/Christoffel 仍保留为审计项，而不是强解释项。
此前全生命周期结果中 metric valid fraction 中位数为 0，且
$\epsilon_{{curvature}}=\kappa r$ 中位数约 10--35，远超小曲率近似。
因此，曲管几何项说明“需要曲线几何”，但当前一阶 thin-tube 近似不能作为闭合强结论。

\section{{图像证据}}
"""
    figures = [
        ("dual_zone_framework_summary.png", "Dual-zone 解释框架：material core、PV-active shell 与 exchange layer。"),
        ("heat_pv_core_shell_partition.png", "heat/PV aggregate-product covariance 的 core-shell 分区。"),
        ("ep_tilt_core_shell_partition.png", "EP 倾斜修正的 core-shell 分区。"),
        ("boundary_exchange_budget.png", "边界交换预算：core、shell 与 combined volume 的对照。"),
        ("pv_retention_vs_leakage_tradeoff.png", "object-level 审计：PV retention 与 leakage 的权衡。"),
        ("coherent_vs_upright_like_dual_zone.png", "coherent 与 upright-like 的 dual-zone 差异。"),
    ]
    body += "\n".join(_figure_latex(fig_dir / name, caption) for name, caption in figures)
    body += r"""
\section{结论}

当前正式诊断口径接受双区解释：inner material core 是低泄漏、低交换的 trapping 核；
PV-active shell 是 heat/PV stirring、月牙强速带和 EP tilt correction 的主要候选区域；
exchange layer 是闭合残差和边界热、PV、动量交换必须单独列账的区域。
single material eddy volume、LAVD/geodesic/PV-retention 边界继续作为审计对照，
但不再作为唯一闭合目标。

\end{document}
"""
    path = output_root / "dual_zone_ep_diagnostic_report_zh.tex"
    path.write_text(body, encoding="utf-8")
    return path


def _compile_latex(tex_path: Path) -> Path | None:
    xelatex = shutil.which("xelatex")
    if xelatex is None:
        return None
    for _ in range(2):
        subprocess.run(
            [xelatex, "-interaction=nonstopmode", "-halt-on-error", tex_path.name],
            cwd=tex_path.parent,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    pdf_path = tex_path.with_suffix(".pdf")
    return pdf_path if pdf_path.exists() else None


def _write_manifest(
    request: DualZoneRequest,
    output_root: Path,
    outputs: dict[str, Path],
    audit_status: dict[str, Any],
) -> Path:
    manifest = {
        "diagnostic_contract": "dual_zone_ep_diagnostics_v1",
        "interpretation": {
            "inner_material_core": "trapping/material coherence/low leakage",
            "pv_active_shell": "heat-PV stirring, crescent high-speed band, EP tilt correction",
            "exchange_layer": "heat/PV/momentum boundary exchange",
            "total_transport": "T_total = T_core_trap + T_shell_stir + T_exchange",
            "single_material_volume_status": "audit comparator, not the sole closure target",
        },
        "runtime": {
            "result_root": str(request.result_root),
            "output_root": str(request.output_root),
            "filter_root": str(request.filter_root),
            "shapes": list(request.shapes),
            "axis_sources": list(request.axis_sources),
            "orientations": list(request.orientations),
            "buoyancy_sources": list(request.buoyancy_sources),
            "tau_values": list(request.tau_values) if request.tau_values is not None else "all",
            "inner_boundary_mode": request.inner_boundary_mode,
            "boundary_budget": request.boundary_budget,
            "core_radius_over_R": request.core_radius_over_R,
            "shell_outer_radius_over_R": request.shell_outer_radius_over_R,
            "pv_shell_quantile": request.pv_shell_quantile,
            "object_boundary_audit": request.object_boundary_audit,
        },
        "object_boundary_audit_status": audit_status,
        "outputs": {key: str(path) for key, path in outputs.items()},
    }
    path = output_root / "dual_zone_manifest.json"
    path.write_text(json.dumps(_json_ready(manifest), ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def run_dual_zone_ep_diagnostics(request: DualZoneRequest) -> dict[str, Path]:
    if request.dry_run:
        print("Dual-zone EP diagnostics dry-run")
        print(f"output root: {request.output_root}")
        print(f"shapes: {','.join(request.shapes)}")
        print(f"axis sources: {','.join(request.axis_sources)}")
        print(f"orientations: {','.join(request.orientations)}")
        print(f"buoyancy sources: {','.join(request.buoyancy_sources)}")
        print(f"inner material core: r/R <= {request.core_radius_over_R}")
        print(f"PV-active shell: r/R <= {request.shell_outer_radius_over_R}; |q'| quantile >= {request.pv_shell_quantile}")
        print(f"exchange layer: inner-core boundary plus adjacent shell cells")
        print(f"object boundary audit: {request.object_boundary_audit}")
        return {}

    _require_pandas()
    request.output_root.mkdir(parents=True, exist_ok=True)
    outputs = run_core_shell_ep_validation(_core_shell_request(request))
    standard = _write_standard_tables(request.output_root)
    outputs.update({f"dual_zone:{key}": path for key, path in standard.items()})
    tradeoff_path, audit_status = _write_tradeoff_summary(request, request.output_root)
    outputs["dual_zone:materiality_tradeoff_summary"] = tradeoff_path
    outputs["dual_zone:decision_table"] = _write_decision_table(request.output_root, tradeoff_path)
    audit_root = _resolve_audit_root(request)
    outputs.update(_copy_standard_figures(request.output_root, audit_root))
    outputs["dual_zone:markdown_report"] = _write_markdown_report(request, request.output_root, tradeoff_path)
    tex_path = _write_latex_report(request, request.output_root, tradeoff_path)
    outputs["dual_zone:latex_report"] = tex_path
    if request.compile_pdf:
        pdf_path = _compile_latex(tex_path)
        if pdf_path is not None:
            outputs["dual_zone:pdf_report"] = pdf_path
    outputs["dual_zone:manifest"] = _write_manifest(request, request.output_root, outputs, audit_status)
    return outputs


def request_from_args(args) -> DualZoneRequest:
    compile_pdf = not bool(getattr(args, "no_compile_pdf", False))
    audit_root_arg = getattr(args, "object_boundary_audit_root", "")
    audit_root = Path(audit_root_arg) if audit_root_arg else None
    return DualZoneRequest(
        result_root=Path(args.result_root),
        output_root=Path(args.output_root),
        filter_root=Path(args.filter_root),
        filter_template=args.filter_template,
        shapes=_split_csv(args.shapes),
        axis_sources=_split_csv(args.axis_sources),
        orientations=_split_csv(args.orientations),
        buoyancy_sources=_split_csv(args.buoyancy_sources),
        tau_values=_parse_tau_values(args.tau_values),
        reference_lat=float(args.reference_lat),
        constant_n2=float(args.constant_n2),
        n2_profile=args.n2_profile,
        inner_boundary_mode=args.inner_boundary_mode,
        boundary_budget=args.boundary_budget,
        core_radius_over_R=float(args.core_radius_over_R),
        shell_outer_radius_over_R=float(args.shell_outer_radius_over_R),
        speed_core_quantile=float(args.speed_core_quantile),
        pv_core_quantile=float(args.pv_core_quantile),
        pv_shell_quantile=float(args.pv_shell_quantile),
        shell_dilation_cells=int(args.shell_dilation_cells),
        min_mask_fraction=float(args.min_mask_fraction),
        min_core_retention=float(args.min_core_retention),
        object_aggregate_transport=not bool(args.no_object_aggregate_transport),
        object_aggregate_max_days=int(args.object_aggregate_max_days),
        object_aggregate_max_objects=int(args.object_aggregate_max_objects),
        object_boundary_audit=args.object_boundary_audit,
        object_boundary_audit_root=audit_root,
        compile_pdf=compile_pdf,
        skip_missing=bool(args.skip_missing),
        dry_run=bool(args.dry_run),
    )
