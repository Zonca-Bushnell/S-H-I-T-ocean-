function grid_files = run_zgeometry_comparison(output_root, meta_dir, bbox, target_lat, intersect_radius_r, ...
    argo_base_mask, argo_lon, argo_lat, argo_time, argo_park, argo_pf, argo_u, argo_v, argo_wpk, history_match_mask, rho, depth, ...
    time_window_days, depth_levels, deg_m, cache_root, boa_clim, max_matches_per_group, z_geometry_mode)

    grid_files = {};
    polarities = {'cyclonic','anticyclonic'};
    band_label = crossing_label(target_lat, intersect_radius_r);
    summary_header = {'polarity','panel','z_geometry_mode','match_count','unique_argo_count','median_corr_w_vs_1000m','deep_reversal_score','first_zero_crossing_depth_m','q95_abs_w_1e6_m_s','output_dir'};
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
        log_step(sprintf('Z-geometry comparison loading META %s from %s', polarity, meta_file));
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
        log_step(sprintf('%s %s z-geometry comparison candidates: %d Argo profiles, %d META snapshots', polarity, band_label, numel(argo_band), numel(meta_band)));
        [matches, grid3d] = build_group_3d(argo_band, meta_band, polarity, band_label, ...
            argo_lon, argo_lat, argo_time, argo_park, argo_pf, argo_u, argo_v, argo_wpk, history_match_mask, rho, depth, ...
            meta_lon, meta_lat, meta_time, meta_track, meta_radius, meta_cx, ...
            time_window_days, grid_n_diag, min_bin_count_diag, smooth_passes_diag, cressman_radius_r_diag, cressman_min_obs_diag, ...
            'anomaly_boa_climatology', boa_clim, depth_levels, 1e-5, 150, match_mode_diag, max_matches_per_group, deg_m, section_axis_diag, section_half_width_r_diag, vertical_mode_diag, cache_root);
        comparison = zgeometry_comparison_terms(grid3d, z_geometry_mode);
        group_dir = fullfile(output_root, polarity, band_label);
        if exist(group_dir, 'dir') ~= 7
            mkdir(group_dir);
        end
        save(fullfile(group_dir, 'zgeometry_comparison_terms.mat'), 'comparison', 'grid3d', 'matches', '-v7.3');
        plot_zgeometry_4panel(fullfile(group_dir, 'zgeometry_w_4panel.png'), comparison, polarity, band_label);
        write_zgeometry_comparison_doc(fullfile(group_dir, 'ZGEOMETRY_COMPARISON_ZH.md'), comparison, polarity, band_label, size(matches,1), count_unique_argo(matches));
        grid_files{end+1} = fullfile(group_dir, 'zgeometry_comparison_terms.mat'); %#ok<AGROW>
        for vv = 1:numel(comparison.panel_names)
            stats = comparison.panel_stats(vv);
            summary_rows(end+1,:) = {polarity, comparison.panel_names{vv}, comparison.panels(vv).z_geometry_mode, size(matches,1), count_unique_argo(matches), ...
                stats.median_corr_w_vs_1000m, stats.deep_reversal_score, stats.first_zero_crossing_depth_m, stats.q95_abs_w_1e6_m_s, group_dir}; %#ok<AGROW>
        end
    end
    write_summary_table_mat(fullfile(output_root, 'ZGEOMETRY_COMPARISON_SUMMARY.mat'), summary_header, summary_rows);
    write_zgeometry_summary_doc(fullfile(output_root, 'ZGEOMETRY_COMPARISON_SUMMARY_ZH.md'), summary_rows);
end
