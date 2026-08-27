from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm


PHASE_ORDER = ["birth", "growth", "mature", "decay", "death"]
DEFAULT_VARIABLES = ["thetao_anom", "so_anom", "adt_anom", "mlotst_anom", "speed_anom"]
DEFAULT_DEPTHS_M = [0.0, 50.0, 100.0, 300.0, 700.0, 1000.0, 1500.0]
DIVERGING_VARIABLES = {"u_anom", "v_anom", "thetao_anom", "so_anom", "sigma0_anom", "adt_anom", "mlotst_anom"}
VARIABLE_LABELS = {
    "u_anom": "u anomaly (m/s)",
    "v_anom": "v anomaly (m/s)",
    "speed_anom": "speed anomaly (m/s)",
    "thetao_anom": "potential temperature anomaly (deg C)",
    "so_anom": "salinity anomaly (psu)",
    "sigma0_anom": "potential density anomaly sigma0 (kg/m^3)",
    "adt_anom": "ADT anomaly (m)",
    "mlotst_anom": "mixed layer depth anomaly (m)",
    "mlotst": "mixed layer depth (m)",
}


def _split_csv(value: str | None, default: list[str]) -> list[str]:
    if value is None or value.strip() == "":
        return list(default)
    return [item.strip() for item in value.split(",") if item.strip()]


def _split_float_csv(value: str | None, default: list[float]) -> list[float]:
    if value is None or value.strip() == "":
        return list(default)
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def _decode_names(values) -> list[str]:
    names = []
    for value in np.asarray(values).ravel():
        if isinstance(value, bytes):
            names.append(value.decode("utf-8"))
        else:
            names.append(str(value))
    return names


def _phase_indices(ds: xr.Dataset) -> list[int]:
    names = _decode_names(ds["phase_name"].values)
    order = {name: i for i, name in enumerate(PHASE_ORDER)}
    return sorted(range(len(names)), key=lambda i: order.get(names[i], i))


def _nearest_depth_indices(depth: np.ndarray, requested: list[float]) -> list[int]:
    indices = []
    for target in requested:
        idx = int(np.nanargmin(np.abs(depth - target)))
        if idx not in indices:
            indices.append(idx)
    return indices


def _variable_dims(ds: xr.Dataset, variable: str) -> tuple[str, ...]:
    if variable not in ds:
        raise KeyError(f"Variable not found in composite file: {variable}")
    return tuple(ds[variable].dims)


def _is_3d_variable(ds: xr.Dataset, variable: str) -> bool:
    return "depth" in _variable_dims(ds, variable)


def _count_name(variable: str) -> str:
    return f"count_{variable}"


def _as_float(values) -> np.ndarray:
    return np.asarray(values, dtype="f8")


def _global_limits(ds: xr.Dataset, variable: str, symmetric: bool = True) -> tuple[float, float]:
    arr = _as_float(ds[variable].values)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return -1.0, 1.0
    if variable == "speed_anom" or not symmetric:
        vmin, vmax = np.nanpercentile(finite, [2, 98])
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
            vmin, vmax = float(np.nanmin(finite)), float(np.nanmax(finite))
        if vmin == vmax:
            vmax = vmin + 1.0
        return float(vmin), float(vmax)
    vmax = float(np.nanpercentile(np.abs(finite), 98))
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = float(np.nanmax(np.abs(finite)))
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0
    return -vmax, vmax


def _cmap_for(variable: str, coverage: bool = False):
    if coverage:
        return "viridis"
    if variable == "speed_anom":
        return "turbo"
    if variable in DIVERGING_VARIABLES:
        return "RdBu_r"
    return "viridis"


def _contour_levels(field: np.ndarray, variable: str, contour_lines: int) -> np.ndarray:
    if contour_lines <= 0:
        return np.array([], dtype="f8")
    values = np.asarray(field, dtype="f8")
    finite = values[np.isfinite(values)]
    if finite.size < 3:
        return np.array([], dtype="f8")
    if variable in DIVERGING_VARIABLES:
        vmax = float(np.nanpercentile(np.abs(finite), 96))
        if not np.isfinite(vmax) or vmax <= 0:
            return np.array([], dtype="f8")
        levels = np.linspace(-vmax, vmax, contour_lines)
        return levels[np.abs(levels) > vmax * 1e-6]
    lo, hi = np.nanpercentile(finite, [5, 95])
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        return np.array([], dtype="f8")
    return np.linspace(float(lo), float(hi), contour_lines)


def _select_slice(ds: xr.Dataset, variable: str, shape_i: int, polarity_i: int, phase_i: int, depth_i: int | None):
    selection = {"shape": shape_i, "polarity": polarity_i, "phase": phase_i}
    if depth_i is not None and "depth" in ds[variable].dims:
        selection["depth"] = depth_i
    return ds[variable].isel(**selection).values


def _event_count(ds: xr.Dataset, shape_i: int, polarity_i: int, phase_i: int) -> int:
    if "event_count" not in ds:
        return 0
    return int(ds["event_count"].isel(shape=shape_i, polarity=polarity_i, phase=phase_i).values)


def _panel_title(phase_name: str, count: int) -> str:
    return f"{phase_name}\nn={count}"


def _shape_label(ds: xr.Dataset, shape_i: int = 0) -> str:
    if "shape_name" not in ds:
        return "shape"
    return _decode_names(ds["shape_name"].values)[shape_i]


def _polarity_names(ds: xr.Dataset) -> list[str]:
    if "polarity_name" not in ds:
        return [str(i) for i in range(ds.sizes["polarity"])]
    return _decode_names(ds["polarity_name"].values)


def _phase_names(ds: xr.Dataset) -> list[str]:
    if "phase_name" not in ds:
        return [str(i) for i in range(ds.sizes["phase"])]
    return _decode_names(ds["phase_name"].values)


def _xy(ds: xr.Dataset) -> tuple[np.ndarray, np.ndarray]:
    return _as_float(ds["x_R"].values), _as_float(ds["y_R"].values)


def _plot_event_summary(ds: xr.Dataset, summary_path: Path, out_path: Path) -> None:
    phase_names = _phase_names(ds)
    polarities = _polarity_names(ds)
    phase_idx = _phase_indices(ds)
    phase_labels = [phase_names[i] for i in phase_idx]
    counts = np.zeros((len(polarities), len(phase_idx)), dtype="i8")
    for pi in range(len(polarities)):
        for j, ph in enumerate(phase_idx):
            counts[pi, j] = _event_count(ds, 0, pi, ph)

    tracks_text = ""
    if summary_path.exists():
        try:
            summary = pd.read_csv(summary_path)
            total = int(summary["event_count"].sum()) if "event_count" in summary else int(counts.sum())
            tracks_text = f"total events={total}"
        except Exception:
            tracks_text = f"total events={int(counts.sum())}"
    else:
        tracks_text = f"total events={int(counts.sum())}"

    fig, ax = plt.subplots(figsize=(9, 4.8), constrained_layout=True)
    im = ax.imshow(counts, cmap="YlGnBu")
    ax.set_xticks(range(len(phase_labels)), phase_labels)
    ax.set_yticks(range(len(polarities)), polarities)
    ax.set_title(f"{_shape_label(ds)} lifecycle composite event counts\n{tracks_text}")
    for i in range(counts.shape[0]):
        for j in range(counts.shape[1]):
            ax.text(j, i, str(counts[i, j]), ha="center", va="center", color="black", fontsize=10)
    fig.colorbar(im, ax=ax, shrink=0.8, label="event count")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _plot_horizontal(
    ds: xr.Dataset,
    variable: str,
    polarity_i: int,
    polarity: str,
    depth_i: int | None,
    out_path: Path,
    *,
    arrow_step: int,
    with_arrows: bool,
    contour_lines: int,
) -> None:
    x, y = _xy(ds)
    phase_names = _phase_names(ds)
    phase_idx = _phase_indices(ds)
    vmin, vmax = _global_limits(ds, variable, symmetric=variable in DIVERGING_VARIABLES)
    fig, axes = plt.subplots(1, len(phase_idx), figsize=(3.4 * len(phase_idx), 3.7), sharex=True, sharey=True, constrained_layout=True)
    axes = np.atleast_1d(axes)

    mappable = None
    for ax, ph in zip(axes, phase_idx):
        field = _select_slice(ds, variable, 0, polarity_i, ph, depth_i)
        mappable = ax.pcolormesh(x, y, field, shading="auto", cmap=_cmap_for(variable), vmin=vmin, vmax=vmax)
        if contour_lines > 0:
            levels = _contour_levels(field, variable, contour_lines)
            if levels.size:
                try:
                    cs = ax.contour(x, y, field, levels=levels, colors="0.15", linewidths=0.55, alpha=0.75)
                    ax.clabel(cs, inline=True, fontsize=6, fmt="%.2g")
                except Exception:
                    pass
        try:
            ax.contour(x, y, field, levels=[0.0], colors="0.15", linewidths=0.8, alpha=0.8)
        except Exception:
            pass
        if with_arrows and depth_i is not None and {"u_anom", "v_anom"}.issubset(ds.data_vars):
            u = _select_slice(ds, "u_anom", 0, polarity_i, ph, depth_i)
            v = _select_slice(ds, "v_anom", 0, polarity_i, ph, depth_i)
            sl = (slice(None, None, arrow_step), slice(None, None, arrow_step))
            ax.quiver(
                x[sl[1]],
                y[sl[0]],
                u[sl],
                v[sl],
                color="black",
                alpha=0.55,
                scale=3.0,
                width=0.003,
                headwidth=3,
            )
        ax.axhline(0, color="0.25", lw=0.6, alpha=0.6)
        ax.axvline(0, color="0.25", lw=0.6, alpha=0.6)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(float(np.nanmin(x)), float(np.nanmax(x)))
        ax.set_ylim(float(np.nanmin(y)), float(np.nanmax(y)))
        ax.set_title(_panel_title(phase_names[ph], _event_count(ds, 0, polarity_i, ph)), fontsize=10)
        ax.set_xlabel("x/R")
    axes[0].set_ylabel("y/R")
    depth_text = ""
    if depth_i is not None:
        depth_text = f", depth={float(ds['depth'].isel(depth=depth_i).values):.1f} m"
    label = VARIABLE_LABELS.get(variable, variable)
    fig.suptitle(f"{_shape_label(ds)} {polarity} {label}{depth_text}\nlifecycle-normalized, tilt-aligned to +x", fontsize=12)
    if mappable is not None:
        fig.colorbar(mappable, ax=axes.tolist(), shrink=0.82, label=label)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _plot_section(
    ds: xr.Dataset,
    variable: str,
    polarity_i: int,
    polarity: str,
    section: str,
    out_path: Path,
    *,
    contour_lines: int,
) -> None:
    if not _is_3d_variable(ds, variable):
        return
    x, y = _xy(ds)
    depth = _as_float(ds["depth"].values)
    phase_names = _phase_names(ds)
    phase_idx = _phase_indices(ds)
    center_index = int(np.nanargmin(np.abs(y if section == "xz" else x)))
    axis_values = x if section == "xz" else y
    axis_label = "x/R" if section == "xz" else "y/R"
    vmin, vmax = _global_limits(ds, variable, symmetric=variable in DIVERGING_VARIABLES)
    fig, axes = plt.subplots(1, len(phase_idx), figsize=(3.4 * len(phase_idx), 4.4), sharex=True, sharey=True, constrained_layout=True)
    axes = np.atleast_1d(axes)
    mappable = None
    for ax, ph in zip(axes, phase_idx):
        cube = _select_slice(ds, variable, 0, polarity_i, ph, None)
        if section == "xz":
            field = cube[:, center_index, :]
        else:
            field = cube[:, :, center_index]
        mappable = ax.pcolormesh(axis_values, depth, field, shading="auto", cmap=_cmap_for(variable), vmin=vmin, vmax=vmax)
        if contour_lines > 0:
            levels = _contour_levels(field, variable, contour_lines)
            if levels.size:
                try:
                    cs = ax.contour(axis_values, depth, field, levels=levels, colors="0.15", linewidths=0.5, alpha=0.75)
                    ax.clabel(cs, inline=True, fontsize=6, fmt="%.2g")
                except Exception:
                    pass
        try:
            ax.contour(axis_values, depth, field, levels=[0.0], colors="0.15", linewidths=0.8, alpha=0.8)
        except Exception:
            pass
        ax.axvline(0, color="0.25", lw=0.6, alpha=0.6)
        ax.invert_yaxis()
        ax.set_title(_panel_title(phase_names[ph], _event_count(ds, 0, polarity_i, ph)), fontsize=10)
        ax.set_xlabel(axis_label)
    axes[0].set_ylabel("depth (m)")
    label = VARIABLE_LABELS.get(variable, variable)
    fig.suptitle(f"{_shape_label(ds)} {polarity} {label} {section.upper()} section\nlifecycle-normalized, tilt-aligned to +x", fontsize=12)
    if mappable is not None:
        fig.colorbar(mappable, ax=axes.tolist(), shrink=0.82, label=label)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _plot_coverage(
    ds: xr.Dataset,
    variable: str,
    polarity_i: int,
    polarity: str,
    depth_i: int | None,
    out_path: Path,
) -> None:
    count_var = _count_name(variable)
    if count_var not in ds:
        return
    x, y = _xy(ds)
    phase_names = _phase_names(ds)
    phase_idx = _phase_indices(ds)
    vmax = float(np.nanmax(ds[count_var].values))
    fig, axes = plt.subplots(1, len(phase_idx), figsize=(3.4 * len(phase_idx), 3.7), sharex=True, sharey=True, constrained_layout=True)
    axes = np.atleast_1d(axes)
    mappable = None
    for ax, ph in zip(axes, phase_idx):
        field = _select_slice(ds, count_var, 0, polarity_i, ph, depth_i)
        mappable = ax.pcolormesh(x, y, field, shading="auto", cmap=_cmap_for(variable, coverage=True), vmin=0, vmax=vmax)
        ax.axhline(0, color="0.25", lw=0.6, alpha=0.6)
        ax.axvline(0, color="0.25", lw=0.6, alpha=0.6)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(_panel_title(phase_names[ph], _event_count(ds, 0, polarity_i, ph)), fontsize=10)
        ax.set_xlabel("x/R")
    axes[0].set_ylabel("y/R")
    depth_text = ""
    if depth_i is not None:
        depth_text = f", depth={float(ds['depth'].isel(depth=depth_i).values):.1f} m"
    fig.suptitle(f"{_shape_label(ds)} {polarity} {variable} valid sample count{depth_text}", fontsize=12)
    if mappable is not None:
        fig.colorbar(mappable, ax=axes.tolist(), shrink=0.82, label="valid samples")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _topview_quiver_scale(ds: xr.Dataset) -> float:
    if "u_anom" not in ds or "v_anom" not in ds:
        return 1.0
    speed = np.hypot(_as_float(ds["u_anom"].values), _as_float(ds["v_anom"].values))
    finite = speed[np.isfinite(speed)]
    if finite.size == 0:
        return 1.0
    u95 = float(np.nanpercentile(finite, 95))
    if not np.isfinite(u95) or u95 <= 0:
        return 1.0
    x, _ = _xy(ds)
    xr = float(np.nanmax(x) - np.nanmin(x))
    return (0.12 * xr) / u95


def _plot_topview_one(
    ds: xr.Dataset,
    variable: str,
    polarity_i: int,
    polarity: str,
    phase_i: int,
    phase_name: str,
    depth_i: int | None,
    out_path: Path,
    *,
    arrow_step: int,
    with_arrows: bool,
    quiver_scale: float,
) -> None:
    x, y = _xy(ds)
    field = _select_slice(ds, variable, 0, polarity_i, phase_i, depth_i)
    vmin, vmax = _global_limits(ds, variable, symmetric=variable in DIVERGING_VARIABLES)
    label = VARIABLE_LABELS.get(variable, variable)
    event_count = _event_count(ds, 0, polarity_i, phase_i)
    depth_text = "surface"
    if depth_i is not None:
        depth_text = f"{float(ds['depth'].isel(depth=depth_i).values):.1f} m"

    fig, ax = plt.subplots(figsize=(7.6, 6.4), constrained_layout=True)
    mesh = ax.pcolormesh(x, y, field, shading="auto", cmap=_cmap_for(variable), vmin=vmin, vmax=vmax)
    try:
        ax.contour(x, y, field, levels=[0.0], colors="0.15", linewidths=0.85, alpha=0.85)
    except Exception:
        pass
    if with_arrows and depth_i is not None and {"u_anom", "v_anom"}.issubset(ds.data_vars):
        u = _select_slice(ds, "u_anom", 0, polarity_i, phase_i, depth_i) * quiver_scale
        v = _select_slice(ds, "v_anom", 0, polarity_i, phase_i, depth_i) * quiver_scale
        sl = (slice(None, None, arrow_step), slice(None, None, arrow_step))
        xs = x[sl[1]]
        ys = y[sl[0]]
        us = u[sl]
        vs = v[sl]
        good = np.isfinite(us) & np.isfinite(vs)
        if np.any(good):
            xx, yy = np.meshgrid(xs, ys)
            ax.quiver(xx[good], yy[good], us[good], vs[good], color="black", alpha=0.65, scale=1.0, scale_units="xy", angles="xy", width=0.003)
    ax.axhline(0, color="0.25", lw=0.7, alpha=0.7)
    ax.axvline(0, color="0.25", lw=0.7, alpha=0.7)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(float(np.nanmin(x)), float(np.nanmax(x)))
    ax.set_ylim(float(np.nanmin(y)), float(np.nanmax(y)))
    ax.grid(True, color="0.82", linewidth=0.6)
    ax.set_xlabel("x/R")
    ax.set_ylabel("y/R")
    ax.set_title(
        f"{_shape_label(ds)} {polarity} {phase_name} {variable} top view @ {depth_text}\n"
        f"n={event_count}, lifecycle-normalized, tilt-aligned to +x",
        fontsize=11,
    )
    fig.colorbar(mesh, ax=ax, shrink=0.86, label=label)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _write_depth_index(depth_values: np.ndarray, depth_indices: list[int], out_dir: Path) -> None:
    rows = [{"depth_index": int(idx), "depth_m": float(depth_values[idx])} for idx in depth_indices]
    pd.DataFrame(rows).to_csv(out_dir / "depth_index.csv", index=False)


def _plot_topview(
    ds: xr.Dataset,
    variables: list[str],
    depth_indices: list[int],
    polarity_indices: list[int],
    output_dir: Path,
    *,
    arrow_step: int,
    with_arrows: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    x, _ = _xy(ds)
    qstep = max(1, round(x.size / 18)) if arrow_step <= 0 else arrow_step
    phase_names = _phase_names(ds)
    phase_indices = _phase_indices(ds)
    polarity_names = _polarity_names(ds)
    depth_values = _as_float(ds["depth"].values) if "depth" in ds else np.array([])
    quiver_scale = _topview_quiver_scale(ds)
    if depth_values.size:
        _write_depth_index(depth_values, depth_indices, output_dir)

    tasks: list[tuple[str, int, str, int, str, int | None]] = []
    for variable in variables:
        if variable not in ds:
            raise KeyError(f"Variable not found: {variable}")
        is_3d = _is_3d_variable(ds, variable)
        selected_depths: list[int | None] = list(depth_indices) if is_3d else [None]
        for polarity_i in polarity_indices:
            polarity = polarity_names[polarity_i]
            for phase_i in phase_indices:
                phase_name = phase_names[phase_i]
                for depth_i in selected_depths:
                    tasks.append((variable, polarity_i, polarity, phase_i, phase_name, depth_i))

    manifest_rows = []
    for variable, polarity_i, polarity, phase_i, phase_name, depth_i in tqdm(tasks, desc="Plot topview composite"):
        pol = _safe_name(polarity)
        phase = _safe_name(phase_name)
        if depth_i is None:
            depth_tag = "surface"
            depth_m = np.nan
            depth_index = -1
        else:
            depth_m = float(depth_values[depth_i])
            depth_index = int(depth_i)
            depth_tag = f"z{depth_m:08.1f}m"
        file_name = f"{pol}_{phase}_{variable}_{depth_tag}.png"
        out_path = output_dir / pol / phase / variable / file_name
        _plot_topview_one(
            ds,
            variable,
            polarity_i,
            polarity,
            phase_i,
            phase_name,
            depth_i,
            out_path,
            arrow_step=qstep,
            with_arrows=with_arrows,
            quiver_scale=quiver_scale,
        )
        manifest_rows.append(
            {
                "shape": _shape_label(ds),
                "polarity": polarity,
                "phase": phase_name,
                "variable": variable,
                "depth_index": depth_index,
                "depth_m": depth_m,
                "event_count": _event_count(ds, 0, polarity_i, phase_i),
                "file_path": str(out_path),
            }
        )
    pd.DataFrame(manifest_rows).to_csv(output_dir / "manifest.csv", index=False)
    print(f"Topview figures written to: {output_dir}")


def _safe_name(text: str) -> str:
    return text.replace(" ", "_").replace("/", "_")


def plot_composite(
    input_dir: Path,
    variables: list[str],
    requested_depths: list[float],
    polarities: list[str] | None,
    plots: set[str],
    *,
    arrow_step: int,
    with_arrows: bool,
    contour_lines: int = 0,
    topview_output_dir: Path | None = None,
    topview_all_depths: bool = False,
) -> None:
    nc_path = input_dir / "lifecycle_composite.nc"
    if not nc_path.exists():
        raise FileNotFoundError(f"Composite NetCDF not found: {nc_path}")
    ds = xr.open_dataset(nc_path)
    out_root = input_dir / "figures"
    summary_path = input_dir / "lifecycle_composite_summary.csv"
    polarity_names = _polarity_names(ds)
    if polarities is None:
        polarity_indices = list(range(len(polarity_names)))
    else:
        wanted = set(polarities)
        polarity_indices = [i for i, name in enumerate(polarity_names) if name in wanted]
        missing = sorted(wanted - {polarity_names[i] for i in polarity_indices})
        if missing:
            raise ValueError(f"Requested polarities not found: {missing}; available={polarity_names}")

    depth_values = _as_float(ds["depth"].values) if "depth" in ds else np.array([])
    depth_indices = _nearest_depth_indices(depth_values, requested_depths) if depth_values.size else []
    topview_depth_indices = list(range(depth_values.size)) if topview_all_depths and depth_values.size else depth_indices

    wrote_standard_figures = False
    if "summary" in plots:
        _plot_event_summary(ds, summary_path, out_root / "summary_event_counts.png")
        wrote_standard_figures = True
    if "topview" in plots:
        tv_out = topview_output_dir if topview_output_dir is not None else input_dir / "topview_figures"
        _plot_topview(ds, variables, topview_depth_indices, polarity_indices, tv_out, arrow_step=arrow_step, with_arrows=with_arrows)

    tasks: list[tuple[str, str, int, str, int | None]] = []
    for variable in variables:
        if variable not in ds:
            raise KeyError(f"Variable not found: {variable}")
        for polarity_i in polarity_indices:
            polarity = polarity_names[polarity_i]
            if "horizontal" in plots:
                if _is_3d_variable(ds, variable):
                    for depth_i in depth_indices:
                        tasks.append(("horizontal", variable, polarity_i, polarity, depth_i))
                else:
                    tasks.append(("horizontal", variable, polarity_i, polarity, None))
            if "coverage" in plots:
                if _is_3d_variable(ds, variable):
                    for depth_i in depth_indices:
                        tasks.append(("coverage", variable, polarity_i, polarity, depth_i))
                else:
                    tasks.append(("coverage", variable, polarity_i, polarity, None))
            if "sections" in plots and _is_3d_variable(ds, variable):
                tasks.append(("section_xz", variable, polarity_i, polarity, None))
                tasks.append(("section_yz", variable, polarity_i, polarity, None))

    if tasks:
        for kind, variable, polarity_i, polarity, depth_i in tqdm(tasks, desc="Plot lifecycle composite"):
            wrote_standard_figures = True
            pol = _safe_name(polarity)
            if kind == "horizontal":
                if depth_i is None:
                    out = out_root / "horizontal" / f"{pol}_{variable}_surface.png"
                else:
                    depth_m = int(round(float(depth_values[depth_i])))
                    out = out_root / "horizontal" / f"{pol}_{variable}_depth{depth_m}m.png"
                _plot_horizontal(
                    ds,
                    variable,
                    polarity_i,
                    polarity,
                    depth_i,
                    out,
                    arrow_step=arrow_step,
                    with_arrows=with_arrows,
                    contour_lines=contour_lines,
                )
            elif kind == "coverage":
                if depth_i is None:
                    out = out_root / "coverage" / f"{pol}_{variable}_count_surface.png"
                else:
                    depth_m = int(round(float(depth_values[depth_i])))
                    out = out_root / "coverage" / f"{pol}_{variable}_count_depth{depth_m}m.png"
                _plot_coverage(ds, variable, polarity_i, polarity, depth_i, out)
            elif kind == "section_xz":
                out = out_root / "sections" / f"{pol}_{variable}_xz.png"
                _plot_section(ds, variable, polarity_i, polarity, "xz", out, contour_lines=contour_lines)
            elif kind == "section_yz":
                out = out_root / "sections" / f"{pol}_{variable}_yz.png"
                _plot_section(ds, variable, polarity_i, polarity, "yz", out, contour_lines=contour_lines)

    if wrote_standard_figures:
        print(f"Figures written to: {out_root}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot lifecycle-normalized 3D eddy composite NetCDF outputs.")
    parser.add_argument("--input-dir", required=True, help="Directory containing lifecycle_composite.nc")
    parser.add_argument("--variables", help="Comma-separated variable names.")
    parser.add_argument("--depths", help="Comma-separated requested depth values in meters.")
    parser.add_argument("--polarities", help="Comma-separated polarity names.")
    parser.add_argument(
        "--plot",
        default="summary,horizontal,sections,coverage",
        help="Comma-separated plot groups: summary,horizontal,sections,coverage,topview.",
    )
    parser.add_argument("--all", action="store_true", help="Generate the standard full plot set.")
    parser.add_argument("--arrow-step", type=int, default=5, help="Horizontal vector arrow thinning step.")
    parser.add_argument("--no-arrows", action="store_true", help="Disable u/v vector arrows on 3D horizontal plots.")
    parser.add_argument("--contour-lines", type=int, default=0, help="Overlay this many contour lines on filled horizontal/section plots.")
    parser.add_argument("--topview-all-depths", action="store_true", help="For topview, export every depth layer instead of only --depths.")
    parser.add_argument("--topview-output-dir", help="Optional independent output directory for topview PNGs.")
    args = parser.parse_args()

    variables = _split_csv(args.variables, DEFAULT_VARIABLES)
    depths = _split_float_csv(args.depths, DEFAULT_DEPTHS_M)
    polarities = _split_csv(args.polarities, []) if args.polarities else None
    plots = set(_split_csv(args.plot, ["summary", "horizontal", "sections", "coverage"]))
    if args.all:
        plots = {"summary", "horizontal", "sections", "coverage"}
    valid = {"summary", "horizontal", "sections", "coverage", "topview"}
    unknown = plots - valid
    if unknown:
        raise ValueError(f"Unknown plot group(s): {sorted(unknown)}")
    plot_composite(
        Path(args.input_dir),
        variables,
        depths,
        polarities,
        plots,
        arrow_step=max(1, args.arrow_step),
        with_arrows=not args.no_arrows,
        contour_lines=max(0, args.contour_lines),
        topview_output_dir=Path(args.topview_output_dir) if args.topview_output_dir else None,
        topview_all_depths=args.topview_all_depths,
    )


if __name__ == "__main__":
    main()
