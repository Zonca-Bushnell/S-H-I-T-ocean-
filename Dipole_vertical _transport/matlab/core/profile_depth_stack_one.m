function [z_row, rho_row, rho_abs_row, boa_rho_row, profile_valid, boa_valid] = profile_depth_stack_one(argo_index, lon, lat, time_value, rho, depth, boa_clim, depth_levels, min_drho_dz, max_rho_bracket_dz_m)
    nz = numel(depth_levels);
    z_row = nan(1, nz);
    rho_row = nan(1, nz);
    rho_abs_row = nan(1, nz);
    boa_rho_row = nan(1, nz);
    profile_valid = false(1, nz);
    boa_valid = false(1, nz);
    [~, month_id, ~] = datevec(time_value);
    boa_profile = boa_density_profile_at(boa_clim, lon, lat, month_id);
    target_profile_raw = double(rho(argo_index,:));
    for zz = 1:nz
        z0 = depth_levels(zz);
        rho0 = interp1(boa_clim.pres, boa_profile, z0, 'linear', NaN);
        if ~isfinite(rho0)
            continue
        end
        boa_rho_row(zz) = rho0;
        target_profile = align_density_units(target_profile_raw, rho0);
        boa_profile_aligned = align_density_units(boa_profile, rho0);
        rho_profile_z0 = interp1(depth, target_profile, z0, 'linear', NaN);
        if isfinite(rho_profile_z0)
            rho_abs_row(zz) = rho_profile_z0;
            rho_row(zz) = rho_profile_z0 - rho0;
        end
        [z_rho, ~, bracket_dz, local_drho_dz] = isopycnal_depth_qc(depth, target_profile, rho0, z0);
        [z_bg, ~, bg_bracket_dz, bg_drho_dz] = isopycnal_depth_qc(boa_clim.pres, boa_profile_aligned, rho0, z0);
        profile_ok = isfinite(z_rho) && isfinite(bracket_dz) && bracket_dz <= max_rho_bracket_dz_m && ...
            isfinite(local_drho_dz) && abs(local_drho_dz) >= min_drho_dz;
        bg_ok = isfinite(z_bg) && isfinite(bg_bracket_dz) && bg_bracket_dz <= max_rho_bracket_dz_m && ...
            isfinite(bg_drho_dz) && abs(bg_drho_dz) >= min_drho_dz;
        profile_valid(zz) = profile_ok;
        boa_valid(zz) = bg_ok;
        if profile_ok && bg_ok
            z_row(zz) = z_rho - z_bg;
        end
    end
end
