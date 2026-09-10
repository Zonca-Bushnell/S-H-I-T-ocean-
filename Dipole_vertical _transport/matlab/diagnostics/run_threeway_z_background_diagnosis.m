function grid_files = run_threeway_z_background_diagnosis(output_root, meta_dir, bbox, target_lat, intersect_radius_r, ...
    argo_base_mask, argo_lon, argo_lat, argo_time, argo_park, argo_pf, argo_u, argo_v, argo_wpk, history_match_mask, rho, depth, ...
    time_window_days, depth_levels, deg_m, cache_root, boa_clim, max_matches_per_group, isas_density_mat)

    grid_files = {};
    polarities = {'cyclonic','anticyclonic'};
    band_label = crossing_label(target_lat, intersect_radius_r);
    summary_header = {'polarity','panel','match_count','unique_argo_count','q95_abs_w_1e6_m_s','q95_abs_term2_1e6_m_s', ...
        'deep_q95_abs_term2_1e6_m_s','term2_corr_vs_A','background_term2_ratio_vs_A','median_corr_w_vs_1000m','deep_reversal_score','first_zero_crossing_depth_m','output_dir'};
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

    for p = 1:numel(polarities)
        polarity = polarities{p};
        meta_file = find_meta_file(meta_dir, polarity);
        log_step(sprintf('Three-way z/background diagnosis loading META %s from %s', polarity, meta_file));
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
        log_step(sprintf('%s %s three-way candidates: %d Argo profiles, %d META snapshots', polarity, band_label, numel(argo_band), numel(meta_band)));
        [matches, grid3d] = build_group_3d(argo_band, meta_band, polarity, band_label, ...
            argo_lon, argo_lat, argo_time, argo_park, argo_pf, argo_u, argo_v, argo_wpk, history_match_mask, rho, depth, ...
            meta_lon, meta_lat, meta_time, meta_track, meta_radius, meta_cx, ...
            time_window_days, grid_n_diag, min_bin_count_diag, smooth_passes_diag, cressman_radius_r_diag, cressman_min_obs_diag, ...
            'anomaly_boa_climatology', boa_clim, depth_levels, 1e-5, 150, match_mode_diag, max_matches_per_group, deg_m, section_axis_diag, section_half_width_r_diag, vertical_mode_diag, cache_root);
        comparison = threeway_z_background_terms(grid3d, polarity, isas_density_mat);
        group_dir = fullfile(output_root, polarity, band_label);
        if exist(group_dir, 'dir') ~= 7
            mkdir(group_dir);
        end
        plot_threeway_z_background_4panel(fullfile(group_dir, 'threeway_w_section_4panel.png'), comparison, polarity, band_label, 'w');
        plot_threeway_z_background_4panel(fullfile(group_dir, 'term2_background_compare.png'), comparison, polarity, band_label, 'term2');
        write_threeway_z_background_doc(fullfile(group_dir, 'THREEWAY_Z_BACKGROUND_DIAGNOSIS_ZH.md'), comparison, polarity, band_label, size(matches,1), count_unique_argo(matches));
        comparison_light = threeway_z_background_light(comparison);
        grid3d_metadata_only = grid3d_metadata(grid3d, polarity, band_label);
        save(fullfile(group_dir, 'threeway_terms.mat'), 'comparison_light', 'grid3d_metadata_only', '-v7.3');
        grid_files{end+1} = fullfile(group_dir, 'threeway_terms.mat'); %#ok<AGROW>
        for vv = 1:numel(comparison.panel_names)
            stats = comparison.panels(vv).stats;
            summary_rows(end+1,:) = {polarity, comparison.panel_names{vv}, size(matches,1), count_unique_argo(matches), ...
                stats.q95_abs_w_1e6_m_s, stats.q95_abs_term2_1e6_m_s, stats.deep_q95_abs_term2_1e6_m_s, ...
                stats.term2_corr_vs_A, stats.background_term2_ratio_vs_A, stats.median_corr_w_vs_1000m, ...
                stats.deep_reversal_score, stats.first_zero_crossing_depth_m, group_dir}; %#ok<AGROW>
        end
    end
    write_summary_table_mat(fullfile(output_root, 'THREEWAY_Z_BACKGROUND_SUMMARY.mat'), summary_header, summary_rows);
    write_threeway_z_background_summary_doc(fullfile(output_root, 'THREEWAY_Z_BACKGROUND_SUMMARY_ZH.md'), summary_rows);
end
