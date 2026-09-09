function grid_file = write_group_outputs_3d(group_dir, matches, grid3d, polarity, band_label)
    grid3d.match_count = size(matches, 1);
    if isempty(matches)
        grid3d.unique_argo_count = 0;
    else
        grid3d.unique_argo_count = numel(unique(cell2mat(matches(:,3))));
    end
    grid3d.duplicate_match_count = grid3d.match_count - grid3d.unique_argo_count;
    header = {'polarity','lat_band','argo_index','platform','argo_time','argo_lon','argo_lat','parking_depth_m','u_argo_m_s','v_argo_m_s','wpk_observed_m_s','rho0_parking','z_rho_m','z_rho_bg_m','z_rho_anom_m','rho_crossing_count','rho_bracket_dz_m','local_drho_dz','eddy_track','eddy_time','eddy_lon','eddy_lat','eddy_radius_m','x_over_R','y_over_R','r_over_R','ring','cx_raw_m_s','history_velocity_matched','boa_rho_crossing_count','boa_rho_bracket_dz_m','boa_local_drho_dz','boa_bg_valid'};
    global WRITE_MATCHED_CSV WRITE_GRID_JSON WRITE_GRID_NC;
    io_timer = tic;
    stage_timer = tic;
    write_matched_table_mat(fullfile(group_dir, 'matched_core_argo_3d.mat'), header, matches);
    log_step(sprintf('%s %s wrote matched MAT in %.1f s', polarity, band_label, toc(stage_timer)));
    if WRITE_MATCHED_CSV
        stage_timer = tic;
        writecell(clean_write_cells([header; matches]), fullfile(group_dir, 'matched_core_argo_3d.csv'));
        log_step(sprintf('%s %s wrote matched CSV in %.1f s', polarity, band_label, toc(stage_timer)));
    end
    grid_file = fullfile(group_dir, 'w_3d_grid.mat');
    stage_timer = tic;
    write_grid_3d_mat(grid_file, grid3d, polarity, band_label);
    log_step(sprintf('%s %s wrote 3D grid MAT in %.1f s', polarity, band_label, toc(stage_timer)));
    if WRITE_GRID_NC
        stage_timer = tic;
        write_grid_3d_nc_file(fullfile(group_dir, 'w_3d_grid.nc'), grid3d, polarity, band_label);
        log_step(sprintf('%s %s wrote 3D grid NetCDF in %.1f s', polarity, band_label, toc(stage_timer)));
    end
    if WRITE_GRID_JSON
        stage_timer = tic;
        write_grid_json_3d(fullfile(group_dir, 'w_3d_grid.json'), grid3d, polarity, band_label);
        log_step(sprintf('%s %s wrote 3D grid JSON in %.1f s', polarity, band_label, toc(stage_timer)));
    end
    stage_timer = tic;
    plot_3d_section(fullfile(group_dir, ['w_3d_section_' grid3d.section_axis '.png']), grid3d, [polarity ' ' band_label]);
    log_step(sprintf('%s %s wrote section PNG in %.1f s', polarity, band_label, toc(stage_timer)));
    stage_timer = tic;
    plot_3d_depth_slices(fullfile(group_dir, 'w_3d_depth_slices.png'), grid3d, [polarity ' ' band_label]);
    log_step(sprintf('%s %s wrote depth-slice PNG in %.1f s', polarity, band_label, toc(stage_timer)));
    write_group_doc_3d(fullfile(group_dir, 'METHOD_3D_W_ZH.md'), matches, grid3d, polarity, band_label);
    log_step(sprintf('%s %s total 3D output write time %.1f s', polarity, band_label, toc(io_timer)));
end
