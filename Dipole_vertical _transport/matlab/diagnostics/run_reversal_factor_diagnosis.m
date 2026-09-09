function grid_files = run_reversal_factor_diagnosis(output_root, meta_dir, bbox, target_lat, intersect_radius_r, ...
    argo_base_mask, argo_lon, argo_lat, argo_time, argo_park, argo_pf, argo_u, argo_v, argo_wpk, history_match_mask, rho, depth, ...
    time_window_days, depth_levels, deg_m, cache_root, boa_clim, max_matches_per_group)

    grid_files = {};
    polarities = {'cyclonic','anticyclonic'};
    band_label = crossing_label(target_lat, intersect_radius_r);
    summary_header = {'polarity','variant','match_count','unique_argo_count','median_corr_w_vs_1000m','deep_reversal_score','first_zero_crossing_depth_m','q95_abs_w_1e6_m_s','output_dir'};
    summary_rows = {};
    grid_n_diag = 61;
    min_bin_count_diag = 1;
    smooth_passes_diag = 4;
    cressman_radius_r_diag = 1.0;
    cressman_min_obs_diag = 8;
    match_mode_diag = 'all';
    vertical_mode_diag = 'thermal_wind_depth_stack';
    section_axis_diag = 'x';
    section_half_width_r_diag = 0.25;
    max_matches_diag = max_matches_per_group;

    for p = 1:numel(polarities)
        polarity = polarities{p};
        meta_file = find_meta_file(meta_dir, polarity);
        log_step(sprintf('Reversal diagnosis loading META %s from %s', polarity, meta_file));
        M = load(meta_file, 'final_lon', 'final_lat', 'final_time', 'final_track', 'final_radius');
        meta_lon = double(M.final_lon);
        meta_lon(meta_lon < 0) = meta_lon(meta_lon < 0) + 360;
        meta_lat = double(M.final_lat);
        meta_time = double(M.final_time);
        meta_track = double(M.final_track);
        meta_radius = double(M.final_radius);
        meta_cx = track_cx(meta_lon, meta_lat, meta_time, meta_track, deg_m);
        cross_distance_m = abs(meta_lat - target_lat) * deg_m;
        meta_band = find(meta_lon >= bbox(1) & meta_lon <= bbox(2) & isfinite(meta_radius) & meta_radius > 0 & ...
            cross_distance_m <= meta_radius * intersect_radius_r);
        if isempty(meta_band)
            argo_band = [];
        else
            argo_lat_window_m = (intersect_radius_r + 4) * max(meta_radius(meta_band));
            argo_band = find(argo_base_mask & abs(argo_lat - target_lat) * deg_m <= argo_lat_window_m);
        end
        log_step(sprintf('%s %s reversal diagnosis candidates: %d Argo profiles, %d META snapshots', polarity, band_label, numel(argo_band), numel(meta_band)));
        [matches, grid3d] = build_group_3d(argo_band, meta_band, polarity, band_label, ...
            argo_lon, argo_lat, argo_time, argo_park, argo_pf, argo_u, argo_v, argo_wpk, history_match_mask, rho, depth, ...
            meta_lon, meta_lat, meta_time, meta_track, meta_radius, meta_cx, ...
            time_window_days, grid_n_diag, min_bin_count_diag, smooth_passes_diag, cressman_radius_r_diag, cressman_min_obs_diag, ...
            'anomaly_boa_climatology', boa_clim, depth_levels, 1e-5, 150, match_mode_diag, max_matches_diag, deg_m, section_axis_diag, section_half_width_r_diag, vertical_mode_diag, cache_root);
        diag = reversal_factor_terms(grid3d);
        group_dir = fullfile(output_root, polarity, band_label);
        if exist(group_dir, 'dir') ~= 7
            mkdir(group_dir);
        end
        save(fullfile(group_dir, 'reversal_factor_terms.mat'), 'diag', 'grid3d', 'matches', '-v7.3');
        plot_reversal_factor_4panel(fullfile(group_dir, 'reversal_factor_4panel.png'), diag, polarity, band_label);
        write_reversal_factor_doc(fullfile(group_dir, 'REVERSION_FACTOR_DIAGNOSIS_ZH.md'), diag, polarity, band_label);
        grid_files{end+1} = fullfile(group_dir, 'reversal_factor_terms.mat'); %#ok<AGROW>
        match_count = size(matches, 1);
        unique_argo_count = count_unique_argo(matches);
        for vv = 1:numel(diag.variant_names)
            stats = diag.variant_stats(vv);
            summary_rows(end+1,:) = {polarity, diag.variant_names{vv}, match_count, unique_argo_count, ...
                stats.median_corr_w_vs_1000m, stats.deep_reversal_score, stats.first_zero_crossing_depth_m, stats.q95_abs_w_1e6_m_s, group_dir}; %#ok<AGROW>
        end
    end
    write_summary_table_mat(fullfile(output_root, 'REVERSAL_FACTOR_SUMMARY.mat'), summary_header, summary_rows);
    write_reversal_factor_summary_doc(fullfile(output_root, 'REVERSAL_FACTOR_SUMMARY_ZH.md'), summary_rows);
end
