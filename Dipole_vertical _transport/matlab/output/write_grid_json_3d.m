function write_grid_json_3d(path, grid3d, polarity, band_label)
    G = struct();
    G.metadata = struct('polarity', polarity, 'lat_band', band_label, 'vertical_mode', grid3d.vertical_mode, ...
        'w_positive_direction', 'upward', 'depth_positive_direction', 'downward', 'section_axis', grid3d.section_axis, ...
        'section_half_width_r', grid3d.section_half_width_r, 'mean_cx_raw_m_s', grid3d.mean_cx_raw, ...
        'mean_u_bg_m_s', grid3d.mean_u_bg, 'cx_rel_m_s', grid3d.cx_rel, 'mean_radius_m', grid3d.mean_radius_m, ...
        'match_count', grid3d.match_count, 'unique_argo_count', grid3d.unique_argo_count, ...
        'duplicate_match_count', grid3d.duplicate_match_count, 'match_mode', grid3d.match_mode, ...
        'z_mode', grid3d.z_mode, 'min_drho_dz', grid3d.min_drho_dz, ...
        'max_rho_bracket_dz_m', grid3d.max_rho_bracket_dz_m, ...
        'thermal_wind_anchor_depth_m', grid3d.thermal_wind_anchor_depth_m, 'thermal_wind_f_s_1', grid3d.thermal_wind_f_s_1);
    G.depth_m = grid3d.depth_levels(:)';
    G.x_over_R = grid3d.x;
    G.y_over_R = grid3d.y;
    G.w_3d_m_s = permute(grid3d.w, [3 1 2]);
    G.term1_3d_m_s = permute(grid3d.term1, [3 1 2]);
    G.term2_3d_m_s = permute(grid3d.term2, [3 1 2]);
    G.z_rho_anom_3d_m = permute(grid3d.z_anom, [3 1 2]);
    G.rho_anom_3d = permute(grid3d.rho_anom, [3 1 2]);
    if isfield(grid3d, 'rho_abs')
        G.rho_abs_3d = permute(grid3d.rho_abs, [3 1 2]);
    end
    G.u_thermal_wind_3d_m_s = permute(grid3d.u_tw, [3 1 2]);
    G.v_thermal_wind_3d_m_s = permute(grid3d.v_tw, [3 1 2]);
    G.sample_count_3d = permute(grid3d.count, [3 1 2]);
    G.mapped_support_3d = permute(grid3d.mapped_support, [3 1 2]);
    G.valid_profile_count_by_depth = grid3d.valid_profile_count(:)';
    G.boa_bg_valid_count_by_depth = grid3d.boa_bg_valid_count(:)';
    G.section_axis_coord_over_R = grid3d.section_coord(:)';
    G.section_depth_m = grid3d.depth_levels(:)';
    G.section_w_m_s = grid3d.section_w;
    fid = fopen(path, 'w');
    fwrite(fid, jsonencode(G), 'char');
    fclose(fid);
end
