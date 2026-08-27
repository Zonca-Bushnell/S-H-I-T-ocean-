from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import yaml

from ..table_io import write_table
from ..validation.io import OUTPUT_ROOT, resolve_path, split_csv, write_run_metadata

from .common import composite_path, decode, finite_or_nan, load_sigma0_profile, radius_lookup, write_summary
from .observations import compute_completed_center_offsets, estimate_observed_growth, load_completed_centers, load_event_index
from .plots import plot_sections, plot_tilt_growth, plot_topviews
from .pv_budget import pv_growth, velocity_centroid_growth
from .baseline_li import (
    baseline_velocity,
    build_baseline_state,
    predict_baseline_state,
    recenter_2d_by_surface_velocity,
    recenter_3d_by_velocity,
    velocity_centroid_profile,
)
from .model_a_isopycnal import (
    build_model_a_state,
    diagnose_model_a_pv,
    predict_model_a_state,
)
from .model_b_streamfunction import (
    build_model_b_state,
    forecast_model_b_phase,
)
from .model_c_pe_isopycnal import (
    build_model_c_state,
    forecast_model_c_phase,
)
from .models import (
    BASELINE,
    BASELINE_DEFINITION,
    METHODS,
    METHODS_A_ONLY,
    METHODS_WITH_C,
    MODEL_A,
    MODEL_A_DEFINITION,
    MODEL_B,
    MODEL_B_DEFINITION,
    MODEL_C,
    MODEL_C_DEFINITION,
    OBS,
    PHASE_ORDER,
    PHASE_TAU,
)


def add_arguments(parser) -> None:
    parser.add_argument("--config", default=str(Path("config") / "config_3d_cmems.yaml"))
    parser.add_argument("--shape", default="coherent")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--centers-path", default=str(OUTPUT_ROOT / "catalog" / "layer_centers_completed.parquet"))
    parser.add_argument("--latitude-ref", type=float, default=30.0)
    parser.add_argument("--depths", default="0,100,300,700,1000,1500")
    parser.add_argument("--forecast-strength", type=float, default=1.0)
    parser.add_argument("--plot-sections", action="store_true")
    parser.add_argument("--plot-topview", action="store_true")
    parser.add_argument("--plot-tilt-growth", action="store_true")
    parser.add_argument("--quick", action="store_true")


def run(args) -> Path:
    config = yaml.safe_load(resolve_path(args.config).read_text(encoding="utf-8"))
    shape = str(args.shape)
    command = getattr(args, "command", "isopycnal-a")
    include_model_b = command in {"model-b", "model-c"}
    include_model_c = command == "model-c"
    default_branch = "baseline_A_B_model_C" if include_model_c else ("baseline_A_model_B" if include_model_b else "li_baseline_vs_isopycnal_A")
    out_dir = resolve_path(args.output_dir) if str(args.output_dir).strip() else Path(config["paths"]["output_dir"]) / "forecast" / default_branch / shape
    out_dir.mkdir(parents=True, exist_ok=True)
    depths = [float(v) for v in split_csv(args.depths, ["0", "100", "300", "700", "1000", "1500"])]
    ds, pv_tables, velocity_centers, driver_centers = build_forecast_dataset(
        config,
        shape,
        float(args.latitude_ref),
        float(args.forecast_strength),
        quick=bool(args.quick),
        include_model_b=include_model_b,
        include_model_c=include_model_c,
    )
    ds.to_netcdf(out_dir / "forecast_state.nc")
    pv_df = pd.concat(pv_tables, ignore_index=True) if pv_tables else pd.DataFrame()
    write_table(pv_df, out_dir / "pv_budget_by_isopycnal.parquet", index=False)
    model_growth = pv_growth(pv_df)
    write_table(model_growth, out_dir / "forecast_pv_tilt_growth.parquet", index=False)
    velocity_centers_df = pd.concat(velocity_centers, ignore_index=True) if velocity_centers else pd.DataFrame()
    write_table(velocity_centers_df, out_dir / "forecast_velocity_centers.parquet", index=False)
    velocity_growth = velocity_centroid_growth(velocity_centers_df)
    write_table(velocity_growth, out_dir / "forecast_velocity_tilt_growth.parquet", index=False)
    driver_centers_df = pd.concat(driver_centers, ignore_index=True) if driver_centers else pd.DataFrame()
    write_table(driver_centers_df, out_dir / "forecast_driver_centers.parquet", index=False)
    observed_growth = completed_center_growth(resolve_path(args.centers_path), config, shape, quick=bool(args.quick))
    write_table(observed_growth, out_dir / "completed_center_tilt_growth.parquet", index=False)
    skill = tilt_growth_skill(observed_growth, velocity_growth)
    write_table(skill, out_dir / "tilt_growth_skill.parquet", index=False)
    if args.plot_sections:
        plot_sections(ds, out_dir, quick=bool(args.quick), raw=False)
        plot_sections(ds, out_dir, quick=bool(args.quick), raw=True)
    if args.plot_topview:
        plot_topviews(ds, out_dir, depths, quick=bool(args.quick), raw=False)
        plot_topviews(ds, out_dir, depths, quick=bool(args.quick), raw=True)
    if args.plot_tilt_growth:
        plot_tilt_growth(observed_growth, velocity_growth, out_dir)
    write_summary(
        out_dir,
        f"Non-circular Physical Isopycnal Forecast: {shape}",
        [
            "Forecast states are generated from one birth initial condition only.",
            "At birth, all requested forecast branches share identical u/v, sigma0, and adt initial fields; model differences begin after phase stepping.",
            "Formal models are unique across src: baseline_li_depth_layer and model_A_isopycnal" + (" and model_B_isopycnal_streamfunction" if include_model_b else "") + (" and model_C_PE_isopycnal_PV_closure." if include_model_c else "."),
            "When command=model-b, model_B_isopycnal_streamfunction is added as the formal streamfunction/PV-network upgrade.",
            BASELINE_DEFINITION,
            MODEL_A_DEFINITION,
            MODEL_B_DEFINITION if include_model_b else "Model B was not requested in this run.",
            MODEL_C_DEFINITION if include_model_c else "Model C was not requested in this run.",
            "Model B initializes psi from the birth velocity condition once; growth/mature/decay/death do not read target u/v as predictors.",
            "Model B updates eta/adt through a linearized PV-network source and saves eta_increment for inspection.",
            "Model C advances eta, nonuniform normal thickness, Psi_i and Morel-gated control-volume PV diagnostics; fixed-depth plots are projections.",
            "Target velocities are read only for obs panels and skill evaluation.",
            "baseline_li_depth_layer remains on the fixed depth grid and never constructs S_i.",
            "model_A_isopycnal constructs S_eta+S_i and uses surface/isopycnal control-volume driver centers before projection back to fixed depth.",
            "Velocity centroids are used for validation and recentered plotting only; they do not drive model_A_isopycnal.",
            "This is a phase-level diagnostic forecast, not a primitive-equation model.",
        ],
    )
    write_run_metadata(out_dir, f"forecast {getattr(args, 'command', 'isopycnal-a')}", vars(args))
    return out_dir


def build_forecast_dataset(
    config: dict,
    shape: str,
    latitude_ref: float,
    strength: float,
    quick: bool,
    include_model_b: bool = False,
    include_model_c: bool = False,
) -> tuple[xr.Dataset, list[pd.DataFrame], list[pd.DataFrame], list[pd.DataFrame]]:
    comp = xr.open_dataset(composite_path(config, shape)).load()
    try:
        methods = METHODS_WITH_C if include_model_c else (METHODS if include_model_b else METHODS_A_ONLY)
        method_index = {name: i for i, name in enumerate(methods)}
        x = finite_or_nan(comp["x_R"].values)
        y = finite_or_nan(comp["y_R"].values)
        depth = finite_or_nan(comp["depth"].values)
        sigma_clim = load_sigma0_profile(config, depth)
        polarities = decode(comp["polarity_name"].values)
        source_phases = decode(comp["phase_name"].values)
        phase_ids = [source_phases.index("birth"), source_phases.index("mature")] if quick and "mature" in source_phases else [source_phases.index(p) for p in PHASE_ORDER if p in source_phases]
        phases = [source_phases[i] for i in phase_ids]
        radius = radius_lookup(config, shape, polarities, source_phases)
        shape_out = (len(methods), len(polarities), len(phases), len(depth), len(y), len(x))
        u_out = np.full(shape_out, np.nan, dtype="f4")
        v_out = np.full(shape_out, np.nan, dtype="f4")
        sigma_out = np.full(shape_out, np.nan, dtype="f4")
        adt_out = np.full((len(methods), len(polarities), len(phases), len(y), len(x)), np.nan, dtype="f4")
        z_out = np.full((len(polarities), len(phases), len(depth), len(y), len(x)), np.nan, dtype="f4")
        driver_conf_out = np.full((len(methods), len(polarities), len(phases), len(depth)), np.nan, dtype="f4")
        psi_b_out = np.full((len(polarities), len(phases), len(depth), len(y), len(x)), np.nan, dtype="f4") if include_model_b else None
        q_b_out = np.full_like(psi_b_out, np.nan, dtype="f4") if include_model_b else None
        coupling_b_out = np.full_like(psi_b_out, np.nan, dtype="f4") if include_model_b else None
        eta_increment_b_out = np.full((len(polarities), len(phases), len(y), len(x)), np.nan, dtype="f4") if include_model_b else None
        psi_initial_b_out = np.full((len(polarities), len(depth), len(y), len(x)), np.nan, dtype="f4") if include_model_b else None
        psi_c_out = np.full((len(polarities), len(phases), len(depth), len(y), len(x)), np.nan, dtype="f4") if include_model_c else None
        q_c_out = np.full_like(psi_c_out, np.nan, dtype="f4") if include_model_c else None
        normal_thickness_c_out = np.full_like(psi_c_out, np.nan, dtype="f4") if include_model_c else None
        area_factor_c_out = np.full_like(psi_c_out, np.nan, dtype="f4") if include_model_c else None
        control_volume_c_out = np.full((len(polarities), len(phases), max(len(depth) - 1, 0), len(y), len(x)), np.nan, dtype="f4") if include_model_c else None
        eta_tendency_c_out = np.full((len(polarities), len(phases), len(y), len(x)), np.nan, dtype="f4") if include_model_c else None
        eta_mass_residual_c_out = np.full((len(polarities), len(phases), len(y), len(x)), np.nan, dtype="f4") if include_model_c else None
        h_tendency_c_out = np.full_like(psi_c_out, np.nan, dtype="f4") if include_model_c else None
        q_closed_c_out = np.full_like(psi_c_out, np.nan, dtype="f4") if include_model_c else None
        pv_rhs_c_out = np.full_like(psi_c_out, np.nan, dtype="f4") if include_model_c else None
        surface_pressure_c_out = np.full((len(polarities), len(phases), len(y), len(x)), np.nan, dtype="f4") if include_model_c else None
        pressure_c_out = np.full_like(psi_c_out, np.nan, dtype="f4") if include_model_c else None
        hydro_increment_c_out = np.full_like(psi_c_out, np.nan, dtype="f4") if include_model_c else None
        montgomery_c_out = np.full_like(psi_c_out, np.nan, dtype="f4") if include_model_c else None
        montgomery_gradient_residual_c_out = np.full_like(psi_c_out, np.nan, dtype="f4") if include_model_c else None
        montgomery_residual_c_out = np.full_like(psi_c_out, np.nan, dtype="f4") if include_model_c else None
        h_repaired_c_out = np.full((len(polarities), len(phases), len(depth)), np.nan, dtype="f4") if include_model_c else None
        pv_residual_c_out = np.full((len(polarities), len(phases), len(depth)), np.nan, dtype="f4") if include_model_c else None
        pv_gate_c_out = np.full((len(polarities), len(phases), len(depth)), np.nan, dtype="f4") if include_model_c else None
        psi_initial_c_out = np.full((len(polarities), len(depth), len(y), len(x)), np.nan, dtype="f4") if include_model_c else None
        pv_tables: list[pd.DataFrame] = []
        velocity_tables: list[pd.DataFrame] = []
        driver_tables: list[pd.DataFrame] = []
        birth_i = source_phases.index("birth")
        for pi, polarity in enumerate(polarities):
            r_birth = radius[(polarity, "birth")]
            sigma_birth = finite_or_nan(comp["sigma0_anom"].isel(shape=0, polarity=pi, phase=birth_i).values)
            adt_birth = finite_or_nan(comp["adt_anom"].isel(shape=0, polarity=pi, phase=birth_i).values)
            u_birth = finite_or_nan(comp["u_anom"].isel(shape=0, polarity=pi, phase=birth_i).values)
            v_birth = finite_or_nan(comp["v_anom"].isel(shape=0, polarity=pi, phase=birth_i).values)
            baseline_state = build_baseline_state(sigma_birth, adt_birth, sigma_clim, depth, x, y, r_birth, latitude_ref)
            model_a_state = build_model_a_state(sigma_birth, adt_birth, sigma_clim, depth, x, y, r_birth, latitude_ref)
            model_b_state = build_model_b_state(sigma_birth, adt_birth, u_birth, v_birth, sigma_clim, depth, x, y, r_birth, latitude_ref) if include_model_b else None
            model_c_state = build_model_c_state(sigma_birth, adt_birth, u_birth, v_birth, sigma_clim, depth, x, y, r_birth, latitude_ref) if include_model_c else None
            if include_model_b and model_b_state is not None and psi_initial_b_out is not None:
                psi_initial_b_out[pi] = model_b_state.psi_initial_from_birth_velocity.astype("f4")
            if include_model_c and model_c_state is not None and psi_initial_c_out is not None:
                psi_initial_c_out[pi] = model_c_state.psi_initial_from_birth_velocity.astype("f4")
            for out_i, ph_i in enumerate(phase_ids):
                phase = source_phases[ph_i]
                tau = float(PHASE_TAU[PHASE_ORDER.index(phase)]) if phase in PHASE_ORDER else 0.0
                is_initial_phase = abs(tau) < 1e-12
                # Validation target panels only; not used to build model states. At birth, every
                # forecast branch must share this same initial state before model evolution begins.
                obs_u = finite_or_nan(comp["u_anom"].isel(shape=0, polarity=pi, phase=ph_i).values)
                obs_v = finite_or_nan(comp["v_anom"].isel(shape=0, polarity=pi, phase=ph_i).values)
                obs_sigma = finite_or_nan(comp["sigma0_anom"].isel(shape=0, polarity=pi, phase=ph_i).values)
                obs_adt = finite_or_nan(comp["adt_anom"].isel(shape=0, polarity=pi, phase=ph_i).values)
                sigma_base, adt_base = predict_baseline_state(baseline_state, tau, strength=strength)
                sigma_a, adt_a = predict_model_a_state(model_a_state, tau, strength=strength)
                u_base, v_base = baseline_velocity(baseline_state, sigma_base, adt_base)
                u_a, v_a, z_anom, pv = diagnose_model_a_pv(model_a_state, sigma_a, adt_a, shape, polarity, phase, ph_i, MODEL_A)
                if is_initial_phase:
                    sigma_base, adt_base = obs_sigma, obs_adt
                    sigma_a, adt_a = obs_sigma, obs_adt
                    u_base, v_base = obs_u, obs_v
                    u_a, v_a = obs_u, obs_v
                pv_tables.append(pv)
                driver_tables.append(_driver_center_table(baseline_state, shape, polarity, phase, ph_i, tau, BASELINE, strength))
                driver_tables.append(_driver_center_table(model_a_state, shape, polarity, phase, ph_i, tau, MODEL_A, strength))
                driver_conf_out[method_index[BASELINE], pi, out_i] = baseline_state.driver_confidence.astype("f4")
                driver_conf_out[method_index[MODEL_A], pi, out_i] = model_a_state.driver_confidence.astype("f4")
                velocity_tables.append(_velocity_center_table(u_base, v_base, x, y, shape, polarity, phase, ph_i, BASELINE))
                velocity_tables.append(_velocity_center_table(u_a, v_a, x, y, shape, polarity, phase, ph_i, MODEL_A))
                u_out[method_index[OBS], pi, out_i] = obs_u.astype("f4")
                v_out[method_index[OBS], pi, out_i] = obs_v.astype("f4")
                sigma_out[method_index[OBS], pi, out_i] = obs_sigma.astype("f4")
                adt_out[method_index[OBS], pi, out_i] = obs_adt.astype("f4")
                u_out[method_index[BASELINE], pi, out_i] = u_base.astype("f4")
                v_out[method_index[BASELINE], pi, out_i] = v_base.astype("f4")
                sigma_out[method_index[BASELINE], pi, out_i] = sigma_base.astype("f4")
                adt_out[method_index[BASELINE], pi, out_i] = adt_base.astype("f4")
                u_out[method_index[MODEL_A], pi, out_i] = u_a.astype("f4")
                v_out[method_index[MODEL_A], pi, out_i] = v_a.astype("f4")
                sigma_out[method_index[MODEL_A], pi, out_i] = sigma_a.astype("f4")
                adt_out[method_index[MODEL_A], pi, out_i] = adt_a.astype("f4")
                z_out[pi, out_i] = z_anom.astype("f4")
                if include_model_b and model_b_state is not None:
                    sigma_b, adt_b, u_b, v_b, z_b, psi_b, q_b, coupling_b, eta_increment_b, pv_b = forecast_model_b_phase(
                        model_b_state,
                        tau,
                        strength,
                        shape,
                        polarity,
                        phase,
                        ph_i,
                        MODEL_B,
                    )
                    if is_initial_phase:
                        sigma_b, adt_b = obs_sigma, obs_adt
                        u_b, v_b = obs_u, obs_v
                    pv_tables.append(pv_b)
                    driver_tables.append(_driver_center_table(model_b_state.base, shape, polarity, phase, ph_i, tau, MODEL_B, strength))
                    driver_conf_out[method_index[MODEL_B], pi, out_i] = model_b_state.base.driver_confidence.astype("f4")
                    velocity_tables.append(_velocity_center_table(u_b, v_b, x, y, shape, polarity, phase, ph_i, MODEL_B))
                    u_out[method_index[MODEL_B], pi, out_i] = u_b.astype("f4")
                    v_out[method_index[MODEL_B], pi, out_i] = v_b.astype("f4")
                    sigma_out[method_index[MODEL_B], pi, out_i] = sigma_b.astype("f4")
                    adt_out[method_index[MODEL_B], pi, out_i] = adt_b.astype("f4")
                    if psi_b_out is not None and q_b_out is not None and coupling_b_out is not None:
                        psi_b_out[pi, out_i] = psi_b.astype("f4")
                        q_b_out[pi, out_i] = q_b.astype("f4")
                        coupling_b_out[pi, out_i] = coupling_b.astype("f4")
                        z_out[pi, out_i] = z_b.astype("f4")
                    if eta_increment_b_out is not None:
                        eta_increment_b_out[pi, out_i] = eta_increment_b.astype("f4")
                if include_model_c and model_c_state is not None:
                    (
                        sigma_c,
                        adt_c,
                        u_c,
                        v_c,
                        z_c,
                        psi_c,
                        q_c,
                        normal_thickness_c,
                        area_factor_c,
                        control_volume_c,
                        eta_tendency_c,
                        h_tendency_c,
                        q_closed_c,
                        pv_rhs_c,
                        pv_used_c,
                        eta_mass_residual_c,
                        h_pred_c,
                        h_repaired_c,
                        surface_pressure_c,
                        pressure_c,
                        hydro_increment_c,
                        montgomery_c,
                        montgomery_gradient_residual_c,
                        montgomery_residual_c,
                        pv_c,
                    ) = forecast_model_c_phase(
                        model_c_state,
                        tau,
                        strength,
                        shape,
                        polarity,
                        phase,
                        ph_i,
                        MODEL_C,
                    )
                    if is_initial_phase:
                        sigma_c, adt_c = obs_sigma, obs_adt
                        u_c, v_c = obs_u, obs_v
                    pv_tables.append(pv_c)
                    driver_tables.append(_driver_center_table(model_c_state.base, shape, polarity, phase, ph_i, tau, MODEL_C, strength))
                    driver_conf_out[method_index[MODEL_C], pi, out_i] = model_c_state.model_c_driver_confidence.astype("f4")
                    velocity_tables.append(_velocity_center_table(u_c, v_c, x, y, shape, polarity, phase, ph_i, MODEL_C))
                    u_out[method_index[MODEL_C], pi, out_i] = u_c.astype("f4")
                    v_out[method_index[MODEL_C], pi, out_i] = v_c.astype("f4")
                    sigma_out[method_index[MODEL_C], pi, out_i] = sigma_c.astype("f4")
                    adt_out[method_index[MODEL_C], pi, out_i] = adt_c.astype("f4")
                    z_out[pi, out_i] = z_c.astype("f4")
                    if psi_c_out is not None and q_c_out is not None and normal_thickness_c_out is not None and area_factor_c_out is not None:
                        psi_c_out[pi, out_i] = psi_c.astype("f4")
                        q_c_out[pi, out_i] = q_c.astype("f4")
                        normal_thickness_c_out[pi, out_i] = normal_thickness_c.astype("f4")
                        area_factor_c_out[pi, out_i] = area_factor_c.astype("f4")
                    if control_volume_c_out is not None:
                        control_volume_c_out[pi, out_i] = control_volume_c.astype("f4")
                    if eta_tendency_c_out is not None:
                        eta_tendency_c_out[pi, out_i] = eta_tendency_c.astype("f4")
                    if h_tendency_c_out is not None:
                        h_tendency_c_out[pi, out_i] = h_tendency_c.astype("f4")
                    if q_closed_c_out is not None:
                        q_closed_c_out[pi, out_i] = q_closed_c.astype("f4")
                    if pv_rhs_c_out is not None:
                        pv_rhs_c_out[pi, out_i] = pv_rhs_c.astype("f4")
                    if surface_pressure_c_out is not None:
                        surface_pressure_c_out[pi, out_i] = surface_pressure_c.astype("f4")
                    if pressure_c_out is not None:
                        pressure_c_out[pi, out_i] = pressure_c.astype("f4")
                    if hydro_increment_c_out is not None:
                        hydro_increment_c_out[pi, out_i] = hydro_increment_c.astype("f4")
                    if montgomery_c_out is not None:
                        montgomery_c_out[pi, out_i] = montgomery_c.astype("f4")
                    if montgomery_gradient_residual_c_out is not None:
                        montgomery_gradient_residual_c_out[pi, out_i] = montgomery_gradient_residual_c.astype("f4")
                    if montgomery_residual_c_out is not None:
                        montgomery_residual_c_out[pi, out_i] = montgomery_residual_c.astype("f4")
                    if eta_mass_residual_c_out is not None:
                        eta_mass_residual_c_out[pi, out_i] = eta_mass_residual_c.astype("f4")
                    if h_repaired_c_out is not None:
                        h_repaired_c_out[pi, out_i] = h_repaired_c.astype("f4")
                    if pv_residual_c_out is not None and "pv_balance_residual" in pv_c:
                        vals = _table_values_to_layers(pv_c["pv_balance_residual"].to_numpy(dtype="f8"), len(depth))
                        pv_residual_c_out[pi, out_i] = vals.astype("f4")
                    if pv_gate_c_out is not None and "pv_gate_closed" in pv_c:
                        vals = _table_values_to_layers(pv_c["pv_gate_closed"].astype("f8").to_numpy(dtype="f8"), len(depth))
                        if np.isfinite(pv_used_c).any():
                            vals = pv_used_c
                        pv_gate_c_out[pi, out_i] = vals.astype("f4")
        u_view, v_view, sigma_view, adt_view = _build_recentered_view(u_out, v_out, sigma_out, adt_out, x, y)
        data_vars = {
                "u_m_s": (("method", "polarity", "phase", "depth", "y", "x"), u_out),
                "v_m_s": (("method", "polarity", "phase", "depth", "y", "x"), v_out),
                "sigma0_kg_m3": (("method", "polarity", "phase", "depth", "y", "x"), sigma_out),
                "adt_m": (("method", "polarity", "phase", "y", "x"), adt_out),
                "u_m_s_view": (("method", "polarity", "phase", "depth", "y", "x"), u_view),
                "v_m_s_view": (("method", "polarity", "phase", "depth", "y", "x"), v_view),
                "sigma0_kg_m3_view": (("method", "polarity", "phase", "depth", "y", "x"), sigma_view),
                "adt_m_view": (("method", "polarity", "phase", "y", "x"), adt_view),
                "isopycnal_z_anom_m": (("polarity", "phase", "depth", "y", "x"), z_out),
                "driver_confidence": (("method", "polarity", "phase", "depth"), driver_conf_out),
                "method_name": ("method", np.asarray(methods, dtype="U64")),
                "polarity_name": ("polarity", np.asarray(polarities, dtype="U32")),
                "phase_name": ("phase", np.asarray(phases, dtype="U32")),
        }
        if include_model_b and psi_b_out is not None and q_b_out is not None and coupling_b_out is not None:
            data_vars["psi_on_isopycnal"] = (("polarity", "phase", "depth", "y", "x"), psi_b_out)
            data_vars["q_model_B"] = (("polarity", "phase", "depth", "y", "x"), q_b_out)
            data_vars["surface_internal_coupling"] = (("polarity", "phase", "depth", "y", "x"), coupling_b_out)
            if eta_increment_b_out is not None:
                data_vars["eta_increment"] = (("polarity", "phase", "y", "x"), eta_increment_b_out)
            if psi_initial_b_out is not None:
                data_vars["psi_initial_from_birth_velocity"] = (("polarity", "depth", "y", "x"), psi_initial_b_out)
        if include_model_c and psi_c_out is not None and q_c_out is not None:
            data_vars["psi_model_C"] = (("polarity", "phase", "depth", "y", "x"), psi_c_out)
            data_vars["q_model_C"] = (("polarity", "phase", "depth", "y", "x"), q_c_out)
            if normal_thickness_c_out is not None:
                data_vars["normal_thickness_m"] = (("polarity", "phase", "depth", "y", "x"), normal_thickness_c_out)
                data_vars["h_pred"] = (("polarity", "phase", "depth", "y", "x"), normal_thickness_c_out)
            if area_factor_c_out is not None:
                data_vars["surface_area_factor"] = (("polarity", "phase", "depth", "y", "x"), area_factor_c_out)
            if control_volume_c_out is not None:
                data_vars["control_volume_dV"] = (("polarity", "phase", "rho_bin", "y", "x"), control_volume_c_out)
            if eta_tendency_c_out is not None:
                data_vars["eta_tendency"] = (("polarity", "phase", "y", "x"), eta_tendency_c_out)
                data_vars["eta_tendency_continuity"] = (("polarity", "phase", "y", "x"), eta_tendency_c_out)
            data_vars["eta_pred"] = (("method", "polarity", "phase", "y", "x"), adt_out)
            if h_tendency_c_out is not None:
                data_vars["h_tendency"] = (("polarity", "phase", "depth", "y", "x"), h_tendency_c_out)
                data_vars["h_tendency_continuity"] = (("polarity", "phase", "depth", "y", "x"), h_tendency_c_out)
            if q_closed_c_out is not None:
                data_vars["q_closed_control_volume"] = (("polarity", "phase", "depth", "y", "x"), q_closed_c_out)
            if pv_rhs_c_out is not None:
                data_vars["pv_inversion_rhs"] = (("polarity", "phase", "depth", "y", "x"), pv_rhs_c_out)
            if surface_pressure_c_out is not None:
                data_vars["surface_pressure_pa"] = (("polarity", "phase", "y", "x"), surface_pressure_c_out)
            if pressure_c_out is not None:
                data_vars["pressure_on_isopycnal_pa"] = (("polarity", "phase", "depth", "y", "x"), pressure_c_out)
            if hydro_increment_c_out is not None:
                data_vars["hydrostatic_pressure_increment_pa"] = (("polarity", "phase", "depth", "y", "x"), hydro_increment_c_out)
            if montgomery_c_out is not None:
                data_vars["montgomery_potential"] = (("polarity", "phase", "depth", "y", "x"), montgomery_c_out)
            if montgomery_gradient_residual_c_out is not None:
                data_vars["montgomery_pressure_gradient_residual"] = (("polarity", "phase", "depth", "y", "x"), montgomery_gradient_residual_c_out)
            if montgomery_residual_c_out is not None:
                data_vars["montgomery_streamfunction_residual"] = (("polarity", "phase", "depth", "y", "x"), montgomery_residual_c_out)
            if eta_mass_residual_c_out is not None:
                data_vars["eta_mass_residual"] = (("polarity", "phase", "y", "x"), eta_mass_residual_c_out)
            if h_repaired_c_out is not None:
                data_vars["h_repaired_fraction"] = (("polarity", "phase", "depth"), h_repaired_c_out)
            if pv_residual_c_out is not None:
                data_vars["pv_closure_residual_C"] = (("polarity", "phase", "depth"), pv_residual_c_out)
            if pv_gate_c_out is not None:
                data_vars["pv_gate_closed"] = (("polarity", "phase", "depth"), pv_gate_c_out)
                data_vars["pv_closure_used_mask"] = (("polarity", "phase", "depth"), pv_gate_c_out)
            if psi_initial_c_out is not None:
                data_vars["psi_initial_model_C_from_birth_velocity"] = (("polarity", "depth", "y", "x"), psi_initial_c_out)
        ds = xr.Dataset(
            data_vars,
            coords={
                "method": np.arange(len(methods)),
                "polarity": np.arange(len(polarities)),
                "phase": np.arange(len(phases)),
                "depth": depth,
                "rho_bin": np.arange(max(len(depth) - 1, 0)),
                "y": y,
                "x": x,
            },
        )
        ds.attrs["forecast_policy"] = "non-circular birth-initialized diagnostic forecast with Li baseline, isopycnal A, and optional Model B streamfunction network"
        ds.attrs["baseline_definition"] = BASELINE_DEFINITION
        ds.attrs["model_A_definition"] = MODEL_A_DEFINITION
        ds.attrs["model_B_definition"] = MODEL_B_DEFINITION if include_model_b else "not included"
        ds.attrs["model_C_definition"] = MODEL_C_DEFINITION if include_model_c else "not included"
        ds.attrs["model_B_initial_condition_policy"] = "birth velocity initializes psi once; post-birth predictors use PV-eta streamfunction network only"
        ds.attrs["model_B_eta_update_source"] = "pv_network_linearized"
        ds.attrs["model_C_initial_condition_policy"] = "birth velocity initializes psi once; post-birth predictors use PE-isopycnal PV closure only"
        ds.attrs["model_C_eta_update_source"] = "free_surface_continuity_zero_external_mass_flux"
        ds.attrs["model_C_h_update_source"] = "adiabatic_isopycnal_thickness_continuity_with_column_correction"
        ds.attrs["model_C_psi_update_source"] = "closed_control_volume_pv_rhs_plus_montgomery_potential_constraint"
        ds.attrs["model_C_montgomery_source"] = "hydrostatic pressure-gradient closure from eta, z_i, h_i^n and rho_i; layer median pressure offsets removed"
        ds.attrs["model_C_pv_gate_policy"] = "control volumes with pv_balance_residual > 0.75 are diagnostics-only and excluded from formal Model C PV inversion RHS"
        ds.attrs["model_A_driver_policy"] = "surface/isopycnal control-volume network; velocity centroids are diagnostics-only"
        ds.attrs["target_policy"] = "u/v target fields are stored for validation panels only"
        ds.attrs["plot_view_policy"] = "main figures use velocity-centroid recentered *_view variables; raw fields remain in base variables"
        ds.attrs["shape_class"] = shape
        return ds, pv_tables, velocity_tables, driver_tables
    finally:
        comp.close()


def completed_center_growth(centers_path: Path, config: dict, shape: str, quick: bool) -> pd.DataFrame:
    root = Path(config["paths"]["output_dir"])
    events = load_event_index(root, [shape])
    if quick:
        events = events.groupby(["shape_class", "polarity"], group_keys=False).head(80).copy()
    centers = load_completed_centers(centers_path, events)
    offsets = compute_completed_center_offsets(events, centers)
    return estimate_observed_growth(offsets)


def tilt_growth_skill(observed: pd.DataFrame, model: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    if observed.empty or model.empty:
        return pd.DataFrame(columns=["polarity", "model", "rmse_TD_growth", "corr_TD_growth", "n_layers"])
    for (polarity, model_name), pred in model.groupby(["polarity", "model"], dropna=False):
        merged = observed.merge(pred, on=["polarity", "depth_index"], how="inner")
        if merged.empty:
            continue
        obs = merged["obs_TD_star_growth_per_phase"].to_numpy(dtype="f8")
        est = merged["TD_velocity_growth_per_phase"].to_numpy(dtype="f8")
        good = np.isfinite(obs) & np.isfinite(est)
        corr = float(np.corrcoef(obs[good], est[good])[0, 1]) if np.count_nonzero(good) > 2 else np.nan
        rmse = float(np.sqrt(np.nanmean((obs[good] - est[good]) ** 2))) if np.any(good) else np.nan
        rows.append({"polarity": polarity, "model": model_name, "rmse_TD_growth": rmse, "corr_TD_growth": corr, "n_layers": int(np.count_nonzero(good))})
    return pd.DataFrame(rows)


def _velocity_center_table(u: np.ndarray, v: np.ndarray, x: np.ndarray, y: np.ndarray, shape: str, polarity: str, phase: str, phase_index: int, model: str) -> pd.DataFrame:
    cx, cy = velocity_centroid_profile(u, v, x, y)
    x0 = cx[0] if cx.size else 0.0
    y0 = cy[0] if cy.size else 0.0
    rows = []
    for k, (xk, yk) in enumerate(zip(cx, cy)):
        td = float(np.hypot(xk - x0, yk - y0)) if np.isfinite(xk) and np.isfinite(yk) else np.nan
        rows.append({
            "shape": shape,
            "polarity": polarity,
            "model": model,
            "phase": phase,
            "phase_index": int(phase_index),
            "depth_index": int(k),
            "center_x_velocity_R": float(xk) if np.isfinite(xk) else np.nan,
            "center_y_velocity_R": float(yk) if np.isfinite(yk) else np.nan,
            "TD_velocity_star": td,
        })
    return pd.DataFrame(rows)


def _driver_center_table(state, shape: str, polarity: str, phase: str, phase_index: int, tau: float, model: str, strength: float) -> pd.DataFrame:
    x0 = state.center_x_R[0] if state.center_x_R.size else 0.0
    y0 = state.center_y_R[0] if state.center_y_R.size else 0.0
    rows = []
    for k, (xk, yk) in enumerate(zip(state.center_x_R, state.center_y_R)):
        px = float(tau * strength * xk)
        py = float(tau * strength * yk)
        sx = float(tau * strength * x0)
        sy = float(tau * strength * y0)
        source = str(state.driver_source[k]) if k < len(state.driver_source) else ""
        confidence = float(state.driver_confidence[k]) if k < len(state.driver_confidence) else np.nan
        rows.append({
            "shape": shape,
            "polarity": polarity,
            "model": model,
            "phase": phase,
            "phase_index": int(phase_index),
            "depth_index": int(k),
            "driver_center_x_R": px,
            "driver_center_y_R": py,
            "driver_TD_star": float(np.hypot(px - sx, py - sy)),
            "driver_source": source,
            "driver_confidence": confidence,
            "driver_low_confidence": bool(np.isfinite(confidence) and confidence < 0.25),
        })
    return pd.DataFrame(rows)


def _table_values_to_layers(values: np.ndarray, n_layers: int) -> np.ndarray:
    arr = np.asarray(values, dtype="f8")
    arr = arr[np.isfinite(arr)]
    if arr.size == n_layers:
        return arr.copy()
    if arr.size == n_layers - 1 and arr.size:
        out = np.full(n_layers, np.nan, dtype="f8")
        out[0] = arr[0]
        out[-1] = arr[-1]
        if n_layers > 2:
            out[1:-1] = 0.5 * (arr[:-1] + arr[1:])
        return out
    if arr.size == 0:
        return np.full(n_layers, np.nan, dtype="f8")
    x_old = np.linspace(0.0, 1.0, arr.size)
    x_new = np.linspace(0.0, 1.0, n_layers)
    return np.interp(x_new, x_old, arr)


def _build_recentered_view(
    u: np.ndarray,
    v: np.ndarray,
    sigma: np.ndarray,
    adt: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    u_view = np.full_like(u, np.nan, dtype="f4")
    v_view = np.full_like(v, np.nan, dtype="f4")
    sigma_view = np.full_like(sigma, np.nan, dtype="f4")
    adt_view = np.full_like(adt, np.nan, dtype="f4")
    for mi in range(u.shape[0]):
        for pi in range(u.shape[1]):
            for ph_i in range(u.shape[2]):
                uu = u[mi, pi, ph_i].astype("f8")
                vv = v[mi, pi, ph_i].astype("f8")
                if not np.isfinite(uu).any() or not np.isfinite(vv).any():
                    continue
                u_view[mi, pi, ph_i] = recenter_3d_by_velocity(uu, uu, vv, x, y).astype("f4")
                v_view[mi, pi, ph_i] = recenter_3d_by_velocity(vv, uu, vv, x, y).astype("f4")
                sigma_view[mi, pi, ph_i] = recenter_3d_by_velocity(sigma[mi, pi, ph_i].astype("f8"), uu, vv, x, y).astype("f4")
                adt_view[mi, pi, ph_i] = recenter_2d_by_surface_velocity(adt[mi, pi, ph_i].astype("f8"), uu, vv, x, y).astype("f4")
    return u_view, v_view, sigma_view, adt_view
