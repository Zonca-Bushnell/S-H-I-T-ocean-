function grid_files = run_fast_sensitivity_2d(output_root, meta_dir, bbox, crossing_lats, target_lat, intersect_radius_r, ...
    argo_base_mask, argo_lon, argo_lat, argo_time, argo_park, argo_pf, argo_u, argo_v, argo_wpk, history_match_mask, rho, depth, ...
    time_window_days, min_bin_count, plot_filled_gradient, sample_gradient_max_profiles, boa_clim, ...
    z_rho_min_m, z_rho_max_m, min_drho_dz, max_rho_bracket_dz_m, match_mode, max_matches_per_group, deg_m, sensitivity_workers, sensitivity_config_names)

    if isempty(crossing_lats)
        sensitivity_lat = target_lat;
    else
        sensitivity_lat = crossing_lats(1);
    end
    band_label = crossing_label(sensitivity_lat, intersect_radius_r);
    configs = select_sensitivity_configs(sensitivity_configs(), sensitivity_config_names);
    polarities = {'cyclonic','anticyclonic'};
    summary_rows = {};
    grid_files = {};
    summary_header = {'polarity','config','grid_n','cressman_radius_r','cressman_min_obs','smooth_passes','match_count','unique_argo_count','duplicate_match_count','valid_grid_fraction','mapped_support_median','mapped_support_p10','corr_rebuild_wpk','q95_abs_rebuild_1e6_m_s','q95_abs_wpk_1e6_m_s','roughness_score','dipole_score','output_dir'};
    for p = 1:numel(polarities)
        polarity = polarities{p};
        stage_timer = tic;
        meta_file = find_meta_file(meta_dir, polarity);
        log_step(sprintf('Fast sensitivity loading META %s from %s', polarity, meta_file));
        M = load(meta_file, 'final_lon', 'final_lat', 'final_time', 'final_track', 'final_radius');
        meta_lon = double(M.final_lon);
        meta_lon(meta_lon < 0) = meta_lon(meta_lon < 0) + 360;
        meta_lat = double(M.final_lat);
        meta_time = double(M.final_time);
        meta_track = double(M.final_track);
        meta_radius = double(M.final_radius);
        meta_cx = track_cx(meta_lon, meta_lat, meta_time, meta_track, deg_m);
        cross_distance_m = abs(meta_lat - sensitivity_lat) * deg_m;
        meta_band = find(meta_lon >= bbox(1) & meta_lon <= bbox(2) & isfinite(meta_radius) & meta_radius > 0 & ...
            cross_distance_m <= meta_radius * intersect_radius_r);
        if isempty(meta_band)
            argo_band = [];
        else
            argo_lat_window_m = (intersect_radius_r + 4) * max(meta_radius(meta_band));
            argo_band = find(argo_base_mask & abs(argo_lat - sensitivity_lat) * deg_m <= argo_lat_window_m);
        end
        log_step(sprintf('%s %s candidates: %d Argo profiles, %d META snapshots', polarity, band_label, numel(argo_band), numel(meta_band)));
        match_timer = tic;
        matches = build_group_matches_only(argo_band, meta_band, polarity, band_label, ...
            argo_lon, argo_lat, argo_time, argo_park, argo_pf, argo_u, argo_v, argo_wpk, history_match_mask, rho, depth, ...
            meta_lon, meta_lat, meta_time, meta_track, meta_radius, meta_cx, ...
            time_window_days, boa_clim, z_rho_min_m, z_rho_max_m, min_drho_dz, max_rho_bracket_dz_m, match_mode, max_matches_per_group, deg_m);
        log_step(sprintf('%s %s cache built: %d QC matches, %d unique Argo, %.1f s', polarity, band_label, size(matches, 1), count_unique_argo(matches), toc(match_timer)));
        cache_dir = fullfile(output_root, polarity, band_label, '_cache');
        if exist(cache_dir, 'dir') ~= 7
            mkdir(cache_dir);
        end
        header = {'polarity','lat_band','argo_index','platform','argo_time','argo_lon','argo_lat','parking_depth_m','u_argo_m_s','v_argo_m_s','wpk_observed_m_s','rho0','z_rho_m','z_rho_bg_m','z_rho_anom_m','rho_crossing_count','rho_bracket_dz_m','local_drho_dz','eddy_track','eddy_time','eddy_lon','eddy_lat','eddy_radius_m','x_over_R','y_over_R','r_over_R','ring','cx_raw_m_s','history_velocity_matched','boa_rho_crossing_count','boa_rho_bracket_dz_m','boa_local_drho_dz','boa_bg_valid'};
        write_matched_table_mat(fullfile(cache_dir, 'matched_core_argo_cache.mat'), header, matches);
        global WRITE_MATCHED_CSV;
        if WRITE_MATCHED_CSV
            writecell(clean_write_cells([header; matches]), fullfile(cache_dir, 'matched_core_argo_cache.csv'));
        end
        grids = cell(numel(configs), 1);
        global USE_GPU_CRESSMAN;
        use_parallel = ~USE_GPU_CRESSMAN && maybe_start_parallel_pool(sensitivity_workers);
        map_timer = tic;
        if use_parallel
            parfor c = 1:numel(configs)
                cfg = configs(c);
                grids{c} = composite_grid(matches, cfg.grid_n, min_bin_count, plot_filled_gradient, cfg.smooth_passes, cfg.cressman_radius_r, cfg.cressman_min_obs, sample_gradient_max_profiles);
            end
        else
            for c = 1:numel(configs)
                cfg = configs(c);
                grids{c} = composite_grid(matches, cfg.grid_n, min_bin_count, plot_filled_gradient, cfg.smooth_passes, cfg.cressman_radius_r, cfg.cressman_min_obs, sample_gradient_max_profiles);
            end
        end
        log_step(sprintf('%s %s remapped %d sensitivity configs in %.1f s', polarity, band_label, numel(configs), toc(map_timer)));
        for c = 1:numel(configs)
            cfg = configs(c);
            grid = grids{c};
            grid.match_mode = match_mode;
            grid.z_mode = 'anomaly_boa_climatology';
            grid.z_rho_min_m = z_rho_min_m;
            grid.z_rho_max_m = z_rho_max_m;
            grid.min_drho_dz = min_drho_dz;
            grid.max_rho_bracket_dz_m = max_rho_bracket_dz_m;
            cfg_dir = fullfile(output_root, polarity, band_label, cfg.name);
            if exist(cfg_dir, 'dir') ~= 7
                mkdir(cfg_dir);
            end
            grid_file = write_group_outputs(cfg_dir, matches, grid, polarity, [band_label ' ' cfg.name], false);
            grid_files{end+1} = grid_file; %#ok<AGROW>
            summary_rows(end+1,:) = sensitivity_summary_row(matches, grid, polarity, cfg, cfg_dir); %#ok<AGROW>
        end
        plot_sensitivity_montage(fullfile(output_root, [polarity '_sensitivity_montage.png']), grids, configs, polarity, band_label);
        log_step(sprintf('%s %s fast sensitivity finished in %.1f s', polarity, band_label, toc(stage_timer)));
    end
    write_summary_table_mat(fullfile(output_root, 'SENSITIVITY_SUMMARY.mat'), summary_header, summary_rows);
    global WRITE_SUMMARY_CSV;
    if WRITE_SUMMARY_CSV
        writecell([summary_header; summary_rows], fullfile(output_root, 'SENSITIVITY_SUMMARY.csv'));
    end
    write_best_sensitivity_doc(fullfile(output_root, 'BEST_PARAMETER_RECOMMENDATION_ZH.md'), summary_rows);
end
