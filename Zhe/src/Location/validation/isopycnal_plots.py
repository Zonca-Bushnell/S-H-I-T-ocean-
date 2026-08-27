from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from .plotting import contour_levels, p2_p98_limits


def plot_sections(ds: xr.Dataset, output_dir: Path) -> None:
    fig_dir = output_dir / "figures" / "sections"
    fig_dir.mkdir(parents=True, exist_ok=True)
    shapes = _names(ds["shape_name"].values)
    polarities = _names(ds["polarity_name"].values)
    phases = _names(ds["phase_name"].values)
    methods = _names(ds["method_name"].values)
    depth = ds["depth"].values
    x = ds["x"].values
    y = ds["y"].values
    x0 = int(np.nanargmin(np.abs(x)))
    y0 = int(np.nanargmin(np.abs(y)))
    for si, shape in enumerate(shapes):
        for pi, pol in enumerate(polarities):
            for ph_i, phase in enumerate(phases):
                if not _has_data(ds["u_m_s"].isel(shape=si, polarity=pi, phase=ph_i).values):
                    continue
                out_dir = fig_dir / shape / pol
                out_dir.mkdir(parents=True, exist_ok=True)
                fig, axes = plt.subplots(2, len(methods), figsize=(4.0 * len(methods), 7.0), constrained_layout=True)
                for mi, method in enumerate(methods):
                    u = ds["u_m_s"].isel(method=mi, shape=si, polarity=pi, phase=ph_i).values[:, :, x0]
                    v = ds["v_m_s"].isel(method=mi, shape=si, polarity=pi, phase=ph_i).values[:, y0, :]
                    _plot_panel(axes[0, mi], y, depth, u, f"{method}: u YZ", "m/s")
                    _plot_panel(axes[1, mi], x, depth, v, f"{method}: v XZ", "m/s")
                fig.suptitle(f"{shape} {pol} {phase}: physical isopycnal-A velocity sections")
                fig.savefig(out_dir / f"{phase}_velocity_sections.png", dpi=180)
                plt.close(fig)

                fig, axes = plt.subplots(2, len(methods), figsize=(4.0 * len(methods), 7.0), constrained_layout=True)
                for mi, method in enumerate(methods):
                    sig = ds["sigma0_kg_m3"].isel(method=mi, shape=si, polarity=pi, phase=ph_i).values
                    _plot_panel(axes[0, mi], y, depth, sig[:, :, x0], f"{method}: sigma0 YZ", "kg/m^3")
                    _plot_panel(axes[1, mi], x, depth, sig[:, y0, :], f"{method}: sigma0 XZ", "kg/m^3")
                fig.suptitle(f"{shape} {pol} {phase}: physical isopycnal-A density sections")
                fig.savefig(out_dir / f"{phase}_sigma0_sections.png", dpi=180)
                plt.close(fig)


def plot_topviews(ds: xr.Dataset, output_dir: Path, requested_depths: list[float]) -> None:
    fig_dir = output_dir / "figures" / "topview"
    fig_dir.mkdir(parents=True, exist_ok=True)
    shapes = _names(ds["shape_name"].values)
    polarities = _names(ds["polarity_name"].values)
    phases = _names(ds["phase_name"].values)
    methods = _names(ds["method_name"].values)
    depth = ds["depth"].values
    x = ds["x"].values
    y = ds["y"].values
    xx, yy = np.meshgrid(x, y)
    depth_ids = [int(np.nanargmin(np.abs(depth - d))) for d in requested_depths]
    for si, shape in enumerate(shapes):
        for pi, pol in enumerate(polarities):
            for ph_i, phase in enumerate(phases):
                if not _has_data(ds["u_m_s"].isel(shape=si, polarity=pi, phase=ph_i).values):
                    continue
                for di in depth_ids:
                    out_dir = fig_dir / shape / pol / phase
                    out_dir.mkdir(parents=True, exist_ok=True)
                    fig, axes = plt.subplots(1, len(methods), figsize=(4.0 * len(methods), 3.8), constrained_layout=True)
                    if len(methods) == 1:
                        axes = [axes]
                    for mi, method in enumerate(methods):
                        u = ds["u_m_s"].isel(method=mi, shape=si, polarity=pi, phase=ph_i, depth=di).values
                        v = ds["v_m_s"].isel(method=mi, shape=si, polarity=pi, phase=ph_i, depth=di).values
                        speed = np.sqrt(u * u + v * v)
                        _plot_top(axes[mi], x, y, speed, f"{method}: speed {depth[di]:.0f} m", "m/s")
                        step = max(1, len(x) // 16)
                        axes[mi].quiver(xx[::step, ::step], yy[::step, ::step], u[::step, ::step], v[::step, ::step], color="k", scale=3.5, width=0.003)
                    fig.suptitle(f"{shape} {pol} {phase}: physical isopycnal-A topview velocity")
                    fig.savefig(out_dir / f"velocity_topview_depth_{depth[di]:07.1f}m.png", dpi=180)
                    plt.close(fig)


def plot_isopycnal_geometry(ds: xr.Dataset, output_dir: Path) -> None:
    fig_dir = output_dir / "figures" / "isopycnal_geometry"
    fig_dir.mkdir(parents=True, exist_ok=True)
    shapes = _names(ds["shape_name"].values)
    polarities = _names(ds["polarity_name"].values)
    phases = _names(ds["phase_name"].values)
    depth = ds["depth"].values
    x = ds["x"].values
    y = ds["y"].values
    x0 = int(np.nanargmin(np.abs(x)))
    for si, shape in enumerate(shapes):
        for pi, pol in enumerate(polarities):
            for ph_i, phase in enumerate(phases):
                if not _has_data(ds["isopycnal_z_anom_m"].isel(shape=si, polarity=pi, phase=ph_i).values):
                    continue
                z_anom = ds["isopycnal_z_anom_m"].isel(shape=si, polarity=pi, phase=ph_i).values[:, :, x0]
                h = ds["curvature_H_1_m"].isel(shape=si, polarity=pi, phase=ph_i).values[:, :, x0]
                fig, axes = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)
                _plot_panel(axes[0], y, depth, z_anom, "isopycnal depth anomaly YZ", "m")
                _plot_panel(axes[1], y, depth, h, "mean curvature H YZ", "1/m")
                fig.suptitle(f"{shape} {pol} {phase}: isopycnal geometry")
                out_dir = fig_dir / shape / pol
                out_dir.mkdir(parents=True, exist_ok=True)
                fig.savefig(out_dir / f"{phase}_isopycnal_geometry.png", dpi=180)
                plt.close(fig)


def _plot_panel(ax, xcoord, depth, field, title: str, label: str) -> None:
    vmin, vmax = p2_p98_limits(field)
    mesh = ax.pcolormesh(xcoord, depth, field, shading="auto", cmap="RdBu_r", vmin=vmin, vmax=vmax)
    levels = contour_levels(field, vmin, vmax, 7)
    if levels.size:
        ax.contour(xcoord, depth, field, levels=levels, colors="0.35", linewidths=0.6)
    if vmin < 0 < vmax:
        ax.contour(xcoord, depth, field, levels=[0], colors="k", linewidths=1.0)
    ax.invert_yaxis()
    ax.set_title(title)
    ax.set_xlabel("x/R or y/R")
    ax.set_ylabel("depth (m)")
    plt.colorbar(mesh, ax=ax, label=label)


def _plot_top(ax, xcoord, ycoord, field, title: str, label: str) -> None:
    vmin, vmax = p2_p98_limits(field)
    mesh = ax.pcolormesh(xcoord, ycoord, field, shading="auto", cmap="viridis", vmin=vmin, vmax=vmax)
    levels = contour_levels(field, vmin, vmax, 7)
    if levels.size:
        ax.contour(xcoord, ycoord, field, levels=levels, colors="0.25", linewidths=0.5)
    ax.axhline(0, color="0.7", lw=0.7)
    ax.axvline(0, color="0.7", lw=0.7)
    ax.set_title(title)
    ax.set_xlabel("x/R")
    ax.set_ylabel("y/R")
    plt.colorbar(mesh, ax=ax, label=label)


def _names(values) -> list[str]:
    arr = np.asarray(values)
    if arr.ndim > 1:
        arr = arr[0]
    return [str(v) for v in arr]


def _has_data(values) -> bool:
    arr = np.asarray(values, dtype="f8")
    return bool(np.any(np.isfinite(arr)))
