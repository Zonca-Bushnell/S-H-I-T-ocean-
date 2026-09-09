function write_grid_json(path, grid, polarity, band_label)
    G = struct();
    G.metadata = struct('polarity', polarity, 'lat_band', band_label, 'w_positive_direction', 'upward', 'depth_positive_direction', 'downward', ...
        'mean_cx_raw_m_s', grid.mean_cx_raw, ...
        'mean_u_bg_m_s', grid.mean_u_bg, 'cx_rel_m_s', grid.cx_rel, 'mean_radius_m', grid.mean_radius_m, ...
        'mean_wpk_observed_m_s', grid.mean_wpk_observed, 'corr_rebuild_wpk', grid.corr_rebuild_wpk, ...
        'grid_mapping', grid.grid_mapping, 'cressman_radius_r', grid.cressman_radius_r, 'cressman_min_obs', grid.cressman_min_obs, ...
        'match_mode', grid.match_mode, 'z_mode', grid.z_mode, ...
        'z_rho_min_m', grid.z_rho_min_m, 'z_rho_max_m', grid.z_rho_max_m, 'min_drho_dz', grid.min_drho_dz, ...
        'max_rho_bracket_dz_m', grid.max_rho_bracket_dz_m, ...
        'match_count', grid.match_count, 'unique_argo_count', grid.unique_argo_count, 'duplicate_match_count', grid.duplicate_match_count, ...
        'boa_bg_valid_count', grid.boa_bg_valid_count, ...
        'valid_grid_cells', sum(isfinite(grid.rebuild_w(:))), 'total_grid_cells', numel(grid.count), ...
        'corr_sample_rebuild_wpk', grid.corr_sample_rebuild_wpk);
    G.x_over_R = grid.x;
    G.y_over_R = grid.y;
    G.z_rho_m = grid.z;
    G.z_rho_raw_m = grid.z_raw;
    G.z_rho_anom_m = grid.z_anom;
    G.u_argo_m_s = grid.u;
    G.v_argo_m_s = grid.v;
    G.wpk_observed_m_s = grid.wpk;
    G.sample_count = grid.count;
    G.mapped_support = grid.mapped_support;
    G.wpk_mapped_support = grid.wpk_mapped_support;
    G.term1_m_s = grid.term1;
    G.term2_m_s = grid.term2;
    G.rebuild_w_m_s = grid.rebuild_w;
    G.term1_depth_positive_m_s = grid.term1_depth_positive;
    G.term2_depth_positive_m_s = grid.term2_depth_positive;
    G.rebuild_w_raw_depth_positive_m_s = grid.rebuild_w_raw_depth_positive;
    G.term1_plus_m_s = grid.term1_plus;
    G.term1_minus_m_s = grid.term1_minus;
    G.term2_abs_m_s = grid.term2_abs;
    G.term2_rel_m_s = grid.term2_rel;
    G.rebuild_plus_abs_m_s = grid.rebuild_plus_abs;
    G.rebuild_minus_rel_m_s = grid.rebuild_minus_rel;
    G.sample_term1_m_s = grid.sample_term1;
    G.sample_term2_m_s = grid.sample_term2;
    G.sample_rebuild_w_m_s = grid.sample_rebuild_w;
    fid = fopen(path, 'w');
    fwrite(fid, jsonencode(G), 'char');
    fclose(fid);
end
