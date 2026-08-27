from __future__ import annotations

from pathlib import Path
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from ..validation.plotting import contour_levels, p2_p98_limits


def plot_sections(ds: xr.Dataset, output_dir: Path, quick: bool = False, raw: bool = False) -> None:
    fig_root = output_dir / ("figures_quick" if quick else "figures")
    fig_dir = fig_root / ("raw_forecast_position" if raw else "") / "sections"
    if fig_dir.exists():
        shutil.rmtree(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    methods = [str(v) for v in ds.method_name.values]
    phases = [str(v) for v in ds.phase_name.values]
    polarities = [str(v) for v in ds.polarity_name.values]
    depth, x, y = ds.depth.values, ds.x.values, ds.y.values
    x0 = int(np.nanargmin(np.abs(x)))
    y0 = int(np.nanargmin(np.abs(y)))
    for pi, polarity in enumerate(polarities):
        for ph_i, phase in enumerate(phases):
            fig, axes = plt.subplots(2, len(methods), figsize=(4 * len(methods), 7), constrained_layout=True)
            for mi, method in enumerate(methods):
                u3 = _field(ds, "u_m_s", raw).isel(method=mi, polarity=pi, phase=ph_i).values
                v3 = _field(ds, "v_m_s", raw).isel(method=mi, polarity=pi, phase=ph_i).values
                title_note = _driver_confidence_note(ds, mi, pi, ph_i)
                _panel(axes[0, mi], y, depth, u3[:, :, x0], f"{method}: u YZ{title_note}", "m/s")
                _panel(axes[1, mi], x, depth, v3[:, y0, :], f"{method}: v XZ{title_note}", "m/s")
            fig.suptitle(f"{polarity} {phase}: {'raw-position' if raw else 'velocity-centroid recentered'} forecast-diagnosed velocity")
            out = fig_dir / polarity
            out.mkdir(parents=True, exist_ok=True)
            fig.savefig(out / f"{phase}_velocity_sections.png", dpi=180)
            plt.close(fig)

            fig, axes = plt.subplots(2, len(methods), figsize=(4 * len(methods), 7), constrained_layout=True)
            for mi, method in enumerate(methods):
                sig = _field(ds, "sigma0_kg_m3", raw).isel(method=mi, polarity=pi, phase=ph_i).values
                title_note = _driver_confidence_note(ds, mi, pi, ph_i)
                _panel(axes[0, mi], y, depth, sig[:, :, x0], f"{method}: sigma0 YZ{title_note}", "kg/m^3")
                _panel(axes[1, mi], x, depth, sig[:, y0, :], f"{method}: sigma0 XZ{title_note}", "kg/m^3")
            fig.suptitle(f"{polarity} {phase}: {'raw-position' if raw else 'velocity-centroid recentered'} forecast sigma0 anomaly")
            fig.savefig(out / f"{phase}_sigma0_sections.png", dpi=180)
            plt.close(fig)


def plot_topviews(ds: xr.Dataset, output_dir: Path, depths: list[float], quick: bool = False, raw: bool = False) -> None:
    fig_root = output_dir / ("figures_quick" if quick else "figures")
    fig_dir = fig_root / ("raw_forecast_position" if raw else "") / "topview"
    if fig_dir.exists():
        shutil.rmtree(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    methods = [str(v) for v in ds.method_name.values]
    phases = [str(v) for v in ds.phase_name.values]
    polarities = [str(v) for v in ds.polarity_name.values]
    x, y, z = ds.x.values, ds.y.values, ds.depth.values
    xx, yy = np.meshgrid(x, y)
    depth_ids = [int(np.nanargmin(np.abs(z - d))) for d in depths]
    for pi, polarity in enumerate(polarities):
        for ph_i, phase in enumerate(phases):
            for di in depth_ids:
                fig, axes = plt.subplots(1, len(methods), figsize=(4 * len(methods), 3.8), constrained_layout=True)
                for mi, method in enumerate(methods):
                    u = _field(ds, "u_m_s", raw).isel(method=mi, polarity=pi, phase=ph_i, depth=di).values
                    v = _field(ds, "v_m_s", raw).isel(method=mi, polarity=pi, phase=ph_i, depth=di).values
                    speed = np.sqrt(u * u + v * v)
                    title_note = _driver_confidence_note(ds, mi, pi, ph_i, di)
                    _top(axes[mi], x, y, speed, f"{method}: {z[di]:.0f} m{title_note}", "m/s")
                    step = max(1, len(x) // 16)
                    axes[mi].quiver(xx[::step, ::step], yy[::step, ::step], u[::step, ::step], v[::step, ::step], color="k", scale=3.5, width=0.003)
                fig.suptitle(f"{polarity} {phase}: {'raw-position' if raw else 'velocity-centroid recentered'} forecast-diagnosed topview velocity")
                out = fig_dir / polarity / phase
                out.mkdir(parents=True, exist_ok=True)
                fig.savefig(out / f"velocity_topview_depth_{z[di]:07.1f}m.png", dpi=180)
                plt.close(fig)


def plot_tilt_growth(observed: pd.DataFrame, model: pd.DataFrame, output_dir: Path) -> None:
    if observed.empty or model.empty:
        return
    fig_dir = output_dir / "figures" / "tilt_growth"
    fig_dir.mkdir(parents=True, exist_ok=True)
    for polarity, obs in observed.groupby("polarity", dropna=False):
        fig, ax = plt.subplots(figsize=(5.2, 6), constrained_layout=True)
        ax.plot(obs["obs_TD_star_growth_per_phase"], obs["depth_m"], color="k", lw=2.2, label="completed-center observed")
        for model_name, group in model[model["polarity"] == polarity].groupby("model", dropna=False):
            ax.plot(group["TD_velocity_growth_per_phase"], group["depth_index"], lw=1.5, label=f"{model_name} velocity centroid")
        ax.axvline(0, color="0.5", lw=0.8)
        ax.invert_yaxis()
        ax.set_xlabel("TD* growth per phase")
        ax.set_ylabel("depth index")
        ax.set_title(f"{polarity}: non-circular velocity-centroid forecast tilt growth")
        ax.legend()
        fig.savefig(fig_dir / f"TD_growth_forecast_{polarity}.png", dpi=180)
        plt.close(fig)


def _panel(ax, xcoord, depth, field, title: str, label: str) -> None:
    vmin, vmax = p2_p98_limits(field)
    mesh = ax.pcolormesh(xcoord, depth, field, shading="auto", cmap="RdBu_r", vmin=vmin, vmax=vmax)
    levels = contour_levels(field, vmin, vmax, 7)
    if levels.size:
        ax.contour(xcoord, depth, field, levels=levels, colors="0.35", linewidths=0.55)
    if vmin < 0 < vmax:
        ax.contour(xcoord, depth, field, levels=[0], colors="k", linewidths=1.0)
    ax.invert_yaxis()
    ax.set_title(title)
    ax.set_xlabel("x/R or y/R")
    ax.set_ylabel("depth (m)")
    plt.colorbar(mesh, ax=ax, label=label)


def _top(ax, xcoord, ycoord, field, title: str, label: str) -> None:
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


def _field(ds: xr.Dataset, base_name: str, raw: bool) -> xr.DataArray:
    if raw:
        return ds[base_name]
    view_name = f"{base_name}_view"
    return ds[view_name] if view_name in ds else ds[base_name]


def _driver_confidence_note(ds: xr.Dataset, method_i: int, polarity_i: int, phase_i: int, depth_i: int | None = None) -> str:
    if "driver_confidence" not in ds:
        return ""
    conf = ds["driver_confidence"].isel(method=method_i, polarity=polarity_i, phase=phase_i)
    vals = conf.values if depth_i is None else np.asarray([float(conf.isel(depth=depth_i).values)])
    low = int(np.count_nonzero(np.isfinite(vals) & (vals < 0.25)))
    return f"\nlow-conf={low}" if low else ""
