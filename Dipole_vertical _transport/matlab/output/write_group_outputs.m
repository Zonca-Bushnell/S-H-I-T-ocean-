function grid_file = write_group_outputs(group_dir, matches, grid, polarity, band_label, write_matches)
    if nargin < 6
        write_matches = true;
    end
    grid.match_count = size(matches, 1);
    if isempty(matches)
        grid.unique_argo_count = 0;
        grid.boa_bg_valid_count = 0;
    else
        grid.unique_argo_count = numel(unique(cell2mat(matches(:,3))));
        if size(matches, 2) >= 33
            grid.boa_bg_valid_count = sum(cell2mat(matches(:,33)));
        else
            grid.boa_bg_valid_count = 0;
        end
    end
    grid.duplicate_match_count = grid.match_count - grid.unique_argo_count;
    header = {'polarity','lat_band','argo_index','platform','argo_time','argo_lon','argo_lat','parking_depth_m','u_argo_m_s','v_argo_m_s','wpk_observed_m_s','rho0','z_rho_m','z_rho_bg_m','z_rho_anom_m','rho_crossing_count','rho_bracket_dz_m','local_drho_dz','eddy_track','eddy_time','eddy_lon','eddy_lat','eddy_radius_m','x_over_R','y_over_R','r_over_R','ring','cx_raw_m_s','history_velocity_matched','boa_rho_crossing_count','boa_rho_bracket_dz_m','boa_local_drho_dz','boa_bg_valid'};
    global WRITE_MATCHED_CSV WRITE_GRID_JSON WRITE_GRID_NC;
    if write_matches
        write_matched_table_mat(fullfile(group_dir, 'matched_core_argo.mat'), header, matches);
        if WRITE_MATCHED_CSV
            writecell(clean_write_cells([header; matches]), fullfile(group_dir, 'matched_core_argo.csv'));
        end
    end
    grid_file = fullfile(group_dir, 'composite_grid.mat');
    write_grid_mat(grid_file, grid, polarity, band_label);
    if WRITE_GRID_NC
        write_grid_nc_file(fullfile(group_dir, 'composite_grid.nc'), grid, polarity, band_label);
    end
    if WRITE_GRID_JSON
        write_grid_json(fullfile(group_dir, 'composite_grid.json'), grid, polarity, band_label);
    end
    plot_three_panel(fullfile(group_dir, 'vertical_transport_terms.png'), grid, [polarity ' ' band_label]);
    plot_sensitivity(fullfile(group_dir, 'velocity_sign_sensitivity.png'), grid, [polarity ' ' band_label]);
    plot_wpk_validation(fullfile(group_dir, 'wpk_validation.png'), grid, [polarity ' ' band_label]);
    plot_gradient_order_comparison(fullfile(group_dir, 'gradient_order_comparison.png'), grid, [polarity ' ' band_label]);
    write_group_doc(fullfile(group_dir, 'METHOD_ASSUMPTIONS_ZH.md'), matches, grid, polarity, band_label);
end
