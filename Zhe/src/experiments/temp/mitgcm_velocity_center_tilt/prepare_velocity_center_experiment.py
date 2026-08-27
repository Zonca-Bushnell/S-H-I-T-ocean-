from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.experiments.temp.run_mode_tilt_validation_yang2026 import (
    _load_azimuthal_npz,
    _polar_to_xy,
    _relative_vorticity_xy,
    _representative_radius_by_polarity,
    _resolve_n2_profile,
    _resolve_radial_seed_root,
    _speed_min_centerline,
)
from src.experiments.theory_validation.unified_math import (
    geo_params,
    project_vertical_modes,
    streamfunction_from_zeta,
    velocity_from_psi,
    vertical_mode_decomposition,
    vertical_weights,
)
from src.First_temp.tilted_ep_flux_validation import load_n2


G = 9.81
RHO0 = 1025.0
CP = 3990.0
T_ALPHA = 2.0e-4
DEFAULT_OUTPUT_ROOT = Path("/root/autodl-fs/kuroshiou_mitgcm_velocity_center_tilt_validation")


@dataclass(frozen=True)
class ExperimentGrid:
    x_m: np.ndarray
    y_m: np.ndarray
    depth_m: np.ndarray
    x_over_r: np.ndarray
    y_over_r: np.ndarray
    radius_m: float


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a standalone MITgcm idealized experiment from a Kuroshiou "
            "velocity-center representative vortex."
        )
    )
    parser.add_argument("--rv-root", required=True, help="ME_LIUTEX representative vortex directory.")
    parser.add_argument("--radial-seed-root", default="", help="Radial-seed representative root with axis/object cache.")
    parser.add_argument("--n2-profile", default="", help="sigma0_dz_profile.npz. Auto-detected when omitted.")
    parser.add_argument("--climatology-nc", default="", help="Optional CMEMS climatology NetCDF for T/S background.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--polarity", default="anticyclonic", choices=("anticyclonic", "cyclonic"))
    parser.add_argument("--tau", type=float, default=0.50)
    parser.add_argument("--latitude-ref", type=float, default=30.0)
    parser.add_argument("--nx", type=int, default=96)
    parser.add_argument("--ny", type=int, default=96)
    parser.add_argument("--max-depth-m", type=float, default=2000.0)
    parser.add_argument("--mode-count", type=int, default=5)
    parser.add_argument("--horizontal-visc-m2s", type=float, default=80.0)
    parser.add_argument("--horizontal-diff-m2s", type=float, default=40.0)
    parser.add_argument("--vertical-visc-m2s", type=float, default=1.0e-4)
    parser.add_argument("--vertical-diff-m2s", type=float, default=1.0e-5)
    parser.add_argument("--delta-t-seconds", type=float, default=300.0)
    parser.add_argument("--run-days", type=float, default=90.0)
    parser.add_argument("--dump-days", type=float, default=1.0)
    parser.add_argument("--sponge-fraction", type=float, default=0.18)
    parser.add_argument("--theta-anomaly-clip-c", type=float, default=3.0)
    parser.add_argument("--write-binary", action="store_true", help="Write MITgcm big-endian binary fields.")
    return parser.parse_args()


def _nearest_index(values: np.ndarray, target: float) -> int:
    return int(np.nanargmin(np.abs(np.asarray(values, dtype="f8") - float(target))))


def _safe_interp(source_x: np.ndarray, source_y: np.ndarray, target_x: np.ndarray, default: float) -> np.ndarray:
    source_x = np.asarray(source_x, dtype="f8")
    source_y = np.asarray(source_y, dtype="f8")
    valid = np.isfinite(source_x) & np.isfinite(source_y)
    if valid.sum() < 2:
        return np.full_like(target_x, float(default), dtype="f8")
    order = np.argsort(source_x[valid])
    return np.interp(target_x, source_x[valid][order], source_y[valid][order])


def _load_background_profiles(clim_path: Path | None, target_depth: np.ndarray) -> tuple[np.ndarray, np.ndarray, str]:
    if clim_path is None or not clim_path.exists():
        theta = 20.0 - 0.010 * target_depth
        salt = np.full_like(target_depth, 34.7, dtype="f8")
        return theta, salt, "linear_fallback_no_climatology"
    import netCDF4

    with netCDF4.Dataset(clim_path) as ds:
        depth_name = "depth" if "depth" in ds.variables else "depth_m"
        source_depth = np.asarray(ds.variables[depth_name][:], dtype="f8")
        theta_name = "thetao_clim" if "thetao_clim" in ds.variables else "thetao_glor"
        salt_name = "so_clim" if "so_clim" in ds.variables else "so_glor"
        theta_var = ds.variables[theta_name]
        theta_raw = np.ma.asarray(theta_var[:], dtype="f8").filled(np.nan)
        salt_raw = np.ma.asarray(ds.variables[salt_name][:], dtype="f8").filled(np.nan) if salt_name in ds.variables else np.nan
        theta_dims = theta_var.dimensions
        depth_axis = theta_dims.index(depth_name) if depth_name in theta_dims else 1
        salt_depth_axis = ds.variables[salt_name].dimensions.index(depth_name) if salt_name in ds.variables and depth_name in ds.variables[salt_name].dimensions else depth_axis
    theta_raw[np.abs(theta_raw) > 1e10] = np.nan
    reduce_axes = tuple(i for i in range(theta_raw.ndim) if i != depth_axis)
    theta_profile = np.nanmean(theta_raw, axis=reduce_axes)
    if np.ndim(salt_raw) == 0:
        salt_profile = np.full_like(source_depth, 34.7, dtype="f8")
    else:
        salt_raw[np.abs(salt_raw) > 1e10] = np.nan
        salt_reduce_axes = tuple(i for i in range(salt_raw.ndim) if i != salt_depth_axis)
        salt_profile = np.nanmean(salt_raw, axis=salt_reduce_axes)
    theta = _safe_interp(source_depth, np.ravel(theta_profile), target_depth, default=15.0)
    salt = _safe_interp(source_depth, np.ravel(salt_profile), target_depth, default=34.7)
    return theta, salt, f"climatology:{clim_path}"


def _make_grid(radius_m: float, depth: np.ndarray, nx: int, ny: int) -> ExperimentGrid:
    extent = 2.5 * float(radius_m)
    x_m = np.linspace(-extent, extent, int(nx), dtype="f8")
    y_m = np.linspace(-extent, extent, int(ny), dtype="f8")
    xx, yy = np.meshgrid(x_m / radius_m, y_m / radius_m)
    return ExperimentGrid(x_m=x_m, y_m=y_m, depth_m=depth, x_over_r=xx, y_over_r=yy, radius_m=radius_m)


def _edge_taper(x_over_r: np.ndarray, y_over_r: np.ndarray, sponge_fraction: float) -> np.ndarray:
    edge = np.maximum(np.abs(x_over_r) / np.nanmax(np.abs(x_over_r)), np.abs(y_over_r) / np.nanmax(np.abs(y_over_r)))
    start = max(0.0, 1.0 - float(sponge_fraction))
    taper = np.ones_like(edge, dtype="f8")
    ramp = edge > start
    taper[ramp] = 0.5 * (1.0 + np.cos(np.pi * np.clip((edge[ramp] - start) / max(1.0 - start, 1e-6), 0.0, 1.0)))
    taper[edge >= 1.0] = 0.0
    return taper


def _thermal_wind_theta_anomaly(psi: np.ndarray, depth: np.ndarray, f0: float, clip_c: float) -> np.ndarray:
    dpsi_dz = np.gradient(psi, depth, axis=0, edge_order=1)
    buoyancy = f0 * dpsi_dz
    theta_prime = buoyancy / max(G * T_ALPHA, 1e-12)
    return np.clip(theta_prime, -abs(float(clip_c)), abs(float(clip_c)))


def _write_big_endian(path: Path, values: np.ndarray) -> None:
    arr = np.asarray(values, dtype="f8")
    arr.astype(">f8", copy=False).tofile(path)


def _make_mitgcm_data_text(case: str, grid: ExperimentGrid, args: argparse.Namespace, f0: float, beta: float) -> str:
    nsteps = max(1, int(round(float(args.run_days) * 86400.0 / float(args.delta_t_seconds))))
    dump_freq = float(args.dump_days) * 86400.0
    dx = float(np.nanmedian(np.diff(grid.x_m)))
    dy = float(np.nanmedian(np.diff(grid.y_m)))
    delz = vertical_weights(grid.depth_m)
    delz_values = [f"{max(float(v), 1.0):.6g}" for v in delz]
    delz_lines = []
    for start in range(0, len(delz_values), 8):
        prefix = " delZ=" if start == 0 else "      "
        delz_lines.append(prefix + ", ".join(delz_values[start : start + 8]) + ",")
    delz_text = "\n".join(delz_lines)
    return f"""# Auto-generated Kuroshiou velocity-center tilt validation case: {case}
 &PARM01
 readBinaryPrec=64,
 writeBinaryPrec=64,
 globalFiles=.TRUE.,
 viscAh={float(args.horizontal_visc_m2s):.6g},
 viscAz={float(args.vertical_visc_m2s):.6g},
 diffKhT={float(args.horizontal_diff_m2s):.6g},
 diffKzT={float(args.vertical_diff_m2s):.6g},
 no_slip_sides=.FALSE.,
 no_slip_bottom=.FALSE.,
 implicitFreeSurface=.TRUE.,
 eosType='LINEAR',
 tAlpha={T_ALPHA:.6g},
 sBeta=0.,
 gravity={G:.6g},
 rhoConst={RHO0:.6g},
 rhoNil={RHO0:.6g},
 heatCapacity_Cp={CP:.6g},
 f0={f0:.10g},
 beta={beta:.10g},
 saltStepping=.FALSE.,
 &
 &PARM02
 cg2dMaxIters=1000,
 cg2dTargetResidual=1.E-10,
 &
 &PARM03
 nIter0=0,
 nTimeSteps={nsteps},
 deltaT={float(args.delta_t_seconds):.6g},
 abEps=0.1,
 dumpFreq={dump_freq:.6g},
 dumpInitAndLast=.TRUE.,
 monitorFreq=0.,
 &
 &PARM04
 usingCartesianGrid=.TRUE.,
 dXspacing={dx:.12g},
 dYspacing={dy:.12g},
{delz_text}
 &
&PARM05
 hydrogThetaFile='T_init.bin',
 hydrogSaltFile='S_init.bin',
 pSurfInitFile='Eta_init.bin',
 uVelInitFile='U_init.bin',
 vVelInitFile='V_init.bin',
 &
"""


def _make_size_h(nx: int, ny: int, nz: int) -> str:
    return f"""CBOP
C    !ROUTINE: SIZE.h
C    !DESCRIPTION: Local Kuroshiou MITgcm smoke grid.
CEOP
      INTEGER sNx
      INTEGER sNy
      INTEGER OLx
      INTEGER OLy
      INTEGER nSx
      INTEGER nSy
      INTEGER nPx
      INTEGER nPy
      INTEGER Nx
      INTEGER Ny
      INTEGER Nr
      PARAMETER (
     &           sNx = {int(nx):4d},
     &           sNy = {int(ny):4d},
     &           OLx =    2,
     &           OLy =    2,
     &           nSx =    1,
     &           nSy =    1,
     &           nPx =    1,
     &           nPy =    1,
     &           Nx  = sNx*nSx*nPx,
     &           Ny  = sNy*nSy*nPy,
     &           Nr  = {int(nz):4d})
      INTEGER MAX_OLX
      INTEGER MAX_OLY
      PARAMETER ( MAX_OLX = OLx,
     &            MAX_OLY = OLy )
"""


def _case_fields(
    psi: np.ndarray,
    theta_bg: np.ndarray,
    salt_bg: np.ndarray,
    depth: np.ndarray,
    x_m: np.ndarray,
    y_m: np.ndarray,
    f0: float,
    taper: np.ndarray,
    theta_clip_c: float,
) -> dict[str, np.ndarray]:
    u, v = velocity_from_psi(psi, y_m, x_m)
    u = np.where(np.isfinite(u), u * taper[None, :, :], 0.0)
    v = np.where(np.isfinite(v), v * taper[None, :, :], 0.0)
    theta_prime = _thermal_wind_theta_anomaly(psi, depth, f0, theta_clip_c)
    theta = theta_bg[:, None, None] + np.where(np.isfinite(theta_prime), theta_prime * taper[None, :, :], 0.0)
    salt = np.broadcast_to(salt_bg[:, None, None], theta.shape).copy()
    eta = f0 * np.nan_to_num(psi[0], nan=0.0) * taper / G
    bathy = -float(np.nanmax(depth)) * np.ones_like(taper, dtype="f8")
    return {"U": u, "V": v, "T": theta, "S": salt, "Eta": eta, "bathy": bathy}


def _diagnose_centerline(psi: np.ndarray, grid: ExperimentGrid, label: str, polarity: str) -> pd.DataFrame:
    mask = np.hypot(grid.x_over_r, grid.y_over_r) <= 1.75
    x, y = _speed_min_centerline(psi, grid.x_over_r, grid.y_over_r, grid.x_m, grid.y_m, mask)
    return pd.DataFrame(
        {
            "case": label,
            "polarity": polarity,
            "depth_m": grid.depth_m,
            "x_over_R": x,
            "y_over_R": y,
            "tilt_distance_over_R": np.hypot(x - x[0], y - y[0]) if np.isfinite(x[0]) and np.isfinite(y[0]) else np.nan,
        }
    )


def _plot_centerlines(out_dir: Path, centerlines: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(8.5, 6.5), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    colors = {
        "real": "#111111",
        "mode1": "#2a9d8f",
        "mode2": "#e76f51",
        "mode1_plus_mode2": "#264653",
        "mode1_to_5": "#7b2cbf",
    }
    for case, part in centerlines.groupby("case", sort=False):
        ax.plot(
            part["x_over_R"],
            part["y_over_R"],
            -part["depth_m"] / 1000.0,
            marker="o",
            markersize=3,
            linewidth=2,
            color=colors.get(str(case), None),
            label=str(case),
        )
    ax.scatter([0], [0], [0], marker="+", s=130, color="red", label="surface reference")
    ax.set_xlabel("x_rot / R")
    ax.set_ylabel("y_rot / R")
    ax.set_zlabel("depth (km, down)")
    ax.set_title("Velocity-center centerlines in MITgcm initial states")
    ax.legend(loc="upper left")
    fig.savefig(out_dir / "velocity_centerline_comparison_3d.png", dpi=180)
    plt.close(fig)


def _plot_mode_panel(out_dir: Path, depth: np.ndarray, profiles: np.ndarray, radii: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 7), constrained_layout=True)
    for n in range(min(6, profiles.shape[1])):
        label = "barotropic" if n == 0 else f"mode {n}"
        rd = radii[n] / 1000.0 if np.isfinite(radii[n]) else np.inf
        ax.plot(profiles[:, n], depth, label=f"{label}, Rd={rd:.1f} km")
    ax.invert_yaxis()
    ax.grid(True, alpha=0.25)
    ax.set_xlabel("normalized mode amplitude")
    ax.set_ylabel("depth (m)")
    ax.set_title("QG vertical modes from Kuroshiou N2")
    ax.legend(fontsize=8)
    fig.savefig(out_dir / "vertical_modes.png", dpi=180)
    plt.close(fig)


def _write_case(case_dir: Path, case: str, fields: dict[str, np.ndarray], args: argparse.Namespace, grid: ExperimentGrid, f0: float, beta: float) -> None:
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "input").mkdir(exist_ok=True)
    (case_dir / "code").mkdir(exist_ok=True)
    (case_dir / "build").mkdir(exist_ok=True)
    (case_dir / "input" / "data").write_text(_make_mitgcm_data_text(case, grid, args, f0, beta), encoding="utf-8")
    (case_dir / "input" / "eedata").write_text(" &EEPARMS\n nTx=1,\n nTy=1,\n &\n", encoding="utf-8")
    (case_dir / "input" / "data.pkg").write_text(" &PACKAGES\n useDiagnostics=.TRUE.,\n &\n", encoding="utf-8")
    (case_dir / "input" / "data.diagnostics").write_text(
        " &DIAGNOSTICS_LIST\n dumpAtLast=.TRUE.,\n fields(1:4,1)='UVEL    ','VVEL    ','THETA   ','ETAN    ',\n"
        " fileName(1)='diag_state',\n frequency(1)=-86400.,\n &\n &DIAG_STATIS_PARMS\n &\n",
        encoding="utf-8",
    )
    (case_dir / "code" / "SIZE.h").write_text(_make_size_h(grid.x_m.size, grid.y_m.size, grid.depth_m.size), encoding="utf-8")
    (case_dir / "code" / "DIAGNOSTICS_SIZE.h").write_text(
        """C Diagnostic storage for Kuroshiou MITgcm tilt-validation cases.
      INTEGER    ndiagMax
      INTEGER    numlists, numperlist, numLevels
      INTEGER    numDiags
      INTEGER    nRegions, sizRegMsk, nStats
      INTEGER    diagSt_size
      PARAMETER( ndiagMax = 500 )
      PARAMETER( numlists = 10, numperlist = 50, numLevels = 2*Nr )
      PARAMETER( numDiags = 15*Nr )
      PARAMETER( nRegions = 0, sizRegMsk = 1, nStats = 4 )
      PARAMETER( diagSt_size = 10*Nr )
""",
        encoding="utf-8",
    )
    (case_dir / "code" / "packages.conf").write_text(
        "mom_fluxform\ngeneric_advdiff\nmdsio\ndiagnostics\n",
        encoding="utf-8",
    )
    (case_dir / "build_case.sh").write_text(
        (
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n"
            "MITGCM_ROOT=${MITGCM_ROOT:-/root/Verify/vendor/MITgcm}\n"
            "mkdir -p build run\n"
            "cd build\n"
            "bash \"${MITGCM_ROOT}/tools/genmake2\" -rootdir \"${MITGCM_ROOT}\" -mods ../code\n"
            "make depend\n"
            "make -j \"${MITGCM_BUILD_JOBS:-4}\"\n"
            "cd ../run\n"
            "ln -sf ../build/mitgcmuv .\n"
            "cp -f ../input/* .\n"
            "echo \"Run with: ./mitgcmuv > run.log 2>&1\"\n"
        ),
        encoding="utf-8",
    )
    (case_dir / "README_run.md").write_text(
        (
            "# MITgcm run notes\n\n"
            "This directory contains generated initial fields and MITgcm namelists. "
            "Build with the local MITgcm `tools/genmake2` and copy/link `input/*.bin` "
            "into the run directory before executing `mitgcmuv`.\n"
        ),
        encoding="utf-8",
    )
    if not args.write_binary:
        return
    _write_big_endian(case_dir / "input" / "U_init.bin", fields["U"])
    _write_big_endian(case_dir / "input" / "V_init.bin", fields["V"])
    _write_big_endian(case_dir / "input" / "T_init.bin", fields["T"])
    _write_big_endian(case_dir / "input" / "S_init.bin", fields["S"])
    _write_big_endian(case_dir / "input" / "Eta_init.bin", fields["Eta"])
    _write_big_endian(case_dir / "input" / "bathy.bin", fields["bathy"])


def main() -> int:
    args = _parse_args()
    rv_root = Path(args.rv_root)
    radial_seed_root = _resolve_radial_seed_root(rv_root, args.radial_seed_root)
    n2_profile = _resolve_n2_profile(rv_root, radial_seed_root, args.n2_profile)
    out_root = Path(args.output_root)
    (out_root / "figures").mkdir(parents=True, exist_ok=True)
    (out_root / "diagnostics").mkdir(exist_ok=True)
    (out_root / "runs").mkdir(exist_ok=True)

    data = _load_azimuthal_npz(rv_root / "azimuthal_representative_velocity.npz")
    polarity_index = data["polarities"].index(args.polarity)
    tau_index = _nearest_index(np.asarray(data["tau_grid"]), args.tau)
    depth_all = np.asarray(data["depth"], dtype="f8")
    depth_mask = depth_all <= float(args.max_depth_m)
    depth = depth_all[depth_mask]
    radius = _representative_radius_by_polarity(radial_seed_root)[args.polarity]
    grid = _make_grid(radius, depth, int(args.nx), int(args.ny))

    u_polar = np.asarray(data["u_mean"])[polarity_index, tau_index, depth_mask]
    v_polar = np.asarray(data["v_mean"])[polarity_index, tau_index, depth_mask]
    u = _polar_to_xy(u_polar, np.asarray(data["radial"]), np.asarray(data["theta"]), grid.x_over_r, grid.y_over_r)
    v = _polar_to_xy(v_polar, np.asarray(data["radial"]), np.asarray(data["theta"]), grid.x_over_r, grid.y_over_r)
    u = np.where(np.isfinite(u), u, 0.0)
    v = np.where(np.isfinite(v), v, 0.0)

    geo = geo_params(float(args.latitude_ref))
    zeta = _relative_vorticity_xy(u, v, grid.x_m, grid.y_m)
    psi = streamfunction_from_zeta(zeta, grid.y_m, grid.x_m)
    n2 = load_n2(n2_profile, depth)
    f_profile = (geo.f0 * geo.f0) / np.maximum(n2, 1.0e-12)
    eigvals, profiles, deformation_radii = vertical_mode_decomposition(f_profile, depth, mode_count=int(args.mode_count))
    mode_recon = project_vertical_modes(psi, profiles, depth)
    cases = {
        "real": psi,
        "mode1": mode_recon[1] if mode_recon.shape[0] > 1 else np.zeros_like(psi),
        "mode2": mode_recon[2] if mode_recon.shape[0] > 2 else np.zeros_like(psi),
        "mode1_plus_mode2": np.nansum(mode_recon[1:3], axis=0) if mode_recon.shape[0] > 2 else np.zeros_like(psi),
        "mode1_to_5": np.nansum(mode_recon[1 : min(6, mode_recon.shape[0])], axis=0)
        if mode_recon.shape[0] > 1
        else np.zeros_like(psi),
    }

    clim_path = Path(args.climatology_nc) if args.climatology_nc else None
    theta_bg, salt_bg, background_source = _load_background_profiles(clim_path, depth)
    taper = _edge_taper(grid.x_over_r, grid.y_over_r, float(args.sponge_fraction))

    centerline_parts = []
    case_summaries = []
    for case, case_psi in cases.items():
        fields = _case_fields(case_psi, theta_bg, salt_bg, depth, grid.x_m, grid.y_m, geo.f0, taper, float(args.theta_anomaly_clip_c))
        _write_case(out_root / "runs" / case, case, fields, args, grid, geo.f0, geo.beta)
        centers = _diagnose_centerline(case_psi, grid, case, args.polarity)
        centerline_parts.append(centers)
        case_summaries.append(
            {
                "case": case,
                "max_speed_m_s": float(np.nanmax(np.hypot(fields["U"], fields["V"]))),
                "theta_anomaly_p95_c": float(np.nanpercentile(np.abs(fields["T"] - theta_bg[:, None, None]), 95)),
                "surface_eta_p95_m": float(np.nanpercentile(np.abs(fields["Eta"]), 95)),
                "tilt_distance_over_R_max": float(np.nanmax(centers["tilt_distance_over_R"])),
            }
        )

    centerlines = pd.concat(centerline_parts, ignore_index=True)
    centerlines.to_parquet(out_root / "diagnostics" / "velocity_center_tracks.parquet", index=False)
    centerlines.to_csv(out_root / "diagnostics" / "velocity_center_tracks.csv", index=False)
    pd.DataFrame(case_summaries).to_csv(out_root / "diagnostics" / "modal_tilt_metrics.csv", index=False)

    weights = vertical_weights(depth)
    orth = profiles.T @ (weights[:, None] * profiles)
    initial_dir = out_root / "initial_conditions"
    initial_dir.mkdir(exist_ok=True)
    np.savez_compressed(
        initial_dir / "modal_initial_states.npz",
        depth=depth,
        x_m=grid.x_m,
        y_m=grid.y_m,
        psi_real=cases["real"],
        psi_mode1=cases["mode1"],
        psi_mode2=cases["mode2"],
        psi_mode1_plus_mode2=cases["mode1_plus_mode2"],
        vertical_mode_profiles=profiles,
        vertical_mode_eigenvalues=eigvals,
        deformation_radii_m=deformation_radii,
    )

    _plot_centerlines(out_root / "figures", centerlines)
    _plot_mode_panel(out_root / "figures", depth, profiles, deformation_radii)

    manifest = {
        "experiment": "kuroshiou_mitgcm_velocity_center_tilt_validation",
        "rv_root": str(rv_root),
        "radial_seed_root": str(radial_seed_root),
        "n2_profile": str(n2_profile),
        "background_source": background_source,
        "polarity": args.polarity,
        "tau_requested": float(args.tau),
        "tau_used": float(np.asarray(data["tau_grid"])[tau_index]),
        "radius_m": float(radius),
        "grid": {"nx": int(args.nx), "ny": int(args.ny), "nz": int(depth.size), "max_depth_m": float(np.nanmax(depth))},
        "mitgcm_notes": {
            "binary_precision": "big_endian_float64",
            "field_order": "numpy depth,y,x order written contiguously; verify against MITgcm read layout before production science",
            "center_definition": "velocity anomaly speed minimum in the global_ls_alpha aligned representative frame",
            "yang2026_difference": "Yang/Xu/Li center diagnostics are temperature/streamfunction based; this experiment uses velocity centers as the primary diagnostic.",
        },
    }
    (out_root / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_root / "summary_zh.md").write_text(
        "\n".join(
            [
                "# Kuroshiou MITgcm 速度中心倾斜验证实验",
                "",
                "本目录是独立 MITgcm 理想化/观测约束实验脚手架，不覆盖现有识别结果。",
                "",
                f"- 输入代表涡：`{rv_root}`",
                f"- 径向种子/速度中心：`{radial_seed_root}`",
                f"- 极性与生命周期：`{args.polarity}`, tau={float(np.asarray(data['tau_grid'])[tau_index]):.2f}",
                f"- 半径：{float(radius)/1000:.1f} km",
                f"- 背景场：{background_source}",
                "",
                "核心设计是：用代表涡速度异常反演流函数，按 QG 垂向模态分解成 real、mode1、mode2、mode1+mode2 四组初值；后处理时只把速度异常中心线作为主诊断。",
                "",
                "需要特别注意：Yang/Xu/Li 2026 的中心口径主要来自温度异常/流函数结构；这里的主口径是速度弱中心加旋转约束，因此验证的是同一模态机制在速度中心定义下是否仍成立。",
            ]
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
