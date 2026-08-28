from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


OBS = "obs"
BASELINE = "baseline_li_depth_layer"
MODEL_A = "model_A_isopycnal"
MODEL_B = "model_B_isopycnal_streamfunction"
MODEL_C = "model_C_PE_isopycnal_PV_closure"
METHODS_A_ONLY = [OBS, BASELINE, MODEL_A]
METHODS = [OBS, BASELINE, MODEL_A, MODEL_B]
METHODS_WITH_C = [OBS, BASELINE, MODEL_A, MODEL_B, MODEL_C]
PHASE_ORDER = ["birth", "growth", "mature", "decay", "death"]
PHASE_TAU = np.asarray([0.0, 0.25, 0.5, 0.75, 1.0], dtype="f8")

BASELINE_DEFINITION = (
    "Li-style fixed-depth/fixed-layer baseline: uses birth adt_anom, "
    "sigma0_anom and climatological stratification on the fixed depth grid; "
    "it does not construct or project S_i isopycnal surfaces."
)
MODEL_A_DEFINITION = (
    "Unified S_eta+S_i deformable-isopycnal A model: uses the free-surface "
    "node S_eta and internal S_i control-volume diagnostics, then projects "
    "the predicted state back to fixed depth for comparison."
)
MODEL_B_DEFINITION = (
    "Model B isopycnal streamfunction model: initializes Psi_i from the "
    "birth velocity condition, then evolves S_eta, S_i, Psi_i, q_i and eta "
    "with a PV-eta streamfunction network before projecting tangent "
    "velocities back to fixed depth for comparison."
)
MODEL_C_DEFINITION = (
    "Model C PE-isopycnal PV-closure model: initializes Psi_i from the birth "
    "velocity condition, then advances S_eta, S_i, nonuniform normal layer "
    "thickness, eta, Psi_i, hydrostatic pressure-gradient Montgomery closure "
    "and closed Morel-style control-volume PV before projecting "
    "surface/isopycnal velocities back to fixed depth."
)


@dataclass
class ForecastState:
    sigma_birth: np.ndarray
    adt_birth: np.ndarray
    sigma_clim: np.ndarray
    depth: np.ndarray
    x: np.ndarray
    y: np.ndarray
    radius_m: float
    f0: float
    center_x_R: np.ndarray
    center_y_R: np.ndarray
    driver_source: np.ndarray
    driver_confidence: np.ndarray


def filled_centers(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype="f8").copy()
    good = np.isfinite(arr)
    if not np.any(good):
        return np.zeros_like(arr)
    idx = np.arange(arr.size)
    arr[~good] = np.interp(idx[~good], idx[good], arr[good])
    first = np.flatnonzero(np.isfinite(arr))[0]
    arr -= arr[first]
    return arr


def assert_model_definitions(forecast_root: Path) -> list[str]:
    """Return consistency-check failures without mutating any files."""
    failures: list[str] = []
    baseline_path = forecast_root / "baseline_li.py"
    model_a_path = forecast_root / "model_a_isopycnal.py"
    text_base = baseline_path.read_text(encoding="utf-8")
    text_a = model_a_path.read_text(encoding="utf-8")
    forbidden = ("build_isopycnal_surfaces", "project_surface_field_to_depth")
    for token in forbidden:
        if token in text_base:
            failures.append(f"baseline_li.py must not use {token}")
    required = ("build_isopycnal_surfaces", "compute_isopycnal_control_volume_pv")
    for token in required:
        if token not in text_a:
            failures.append(f"model_a_isopycnal.py must use {token}")
    if "velocity_centroid_profile" in text_a:
        failures.append("model_a_isopycnal.py must not use velocity_centroid_profile for A-driver construction")
    model_b_path = forecast_root / "model_b_streamfunction.py"
    if not model_b_path.exists():
        failures.append("model_b_streamfunction.py is required for formal Model B")
    else:
        text_b = model_b_path.read_text(encoding="utf-8")
        for token in ("u_anom", "v_anom"):
            if token in text_b:
                failures.append(f"model_b_streamfunction.py must not read validation target {token}")
        if "thermal_wind_velocity" in text_b:
            failures.append("model_b_streamfunction.py must not use thermal_wind_velocity as final velocity closure")
        for token in ("build_isopycnal_surfaces", "project_surface_field_to_depth"):
            if token not in text_b:
                failures.append(f"model_b_streamfunction.py must use {token}")
        for token in (
            "build_initial_psi_from_birth_velocity",
            "step_model_b_pv_eta_network",
            "solve_surface_internal_pv_network",
            "project_model_b_to_fixed_depth",
        ):
            if token not in text_b:
                failures.append(f"model_b_streamfunction.py must define/use {token}")
        if "_shift_with_streamfunction_centers" in text_b:
            failures.append("model_b_streamfunction.py must not use center-shift persistence for Model B")
    model_c_path = forecast_root / "model_c_pe_isopycnal.py"
    if not model_c_path.exists():
        failures.append("model_c_pe_isopycnal.py is required for formal Model C")
    else:
        text_c = model_c_path.read_text(encoding="utf-8")
        for token in ("u_anom", "v_anom"):
            if token in text_c:
                failures.append(f"model_c_pe_isopycnal.py must not read validation target {token}")
        if "thermal_wind_velocity" in text_c:
            failures.append("model_c_pe_isopycnal.py must not use thermal_wind_velocity as final velocity closure")
        for token in (
            "normal_thickness_m",
            "surface_pressure_pa",
            "pressure_on_isopycnal_pa",
            "hydrostatic_pressure_increment_pa",
            "compute_montgomery_pressure_closure",
            "pv_closure_residual",
            "eta_tendency",
            "update_eta_continuity",
            "montgomery_potential",
            "q_closed_control_volume",
            "pv_inversion_rhs",
            "pv_closure_used_mask",
            "compute_isopycnal_control_volume_pv",
            "build_initial_psi_from_birth_velocity",
            "project_surface_field_to_depth",
        ):
            if token not in text_c:
                failures.append(f"model_c_pe_isopycnal.py must define/use {token}")
    return failures
