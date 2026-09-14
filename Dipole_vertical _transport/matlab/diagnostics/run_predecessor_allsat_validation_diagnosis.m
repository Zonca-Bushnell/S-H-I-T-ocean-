function grid_files = run_predecessor_allsat_validation_diagnosis(output_root, meta32_allsat_dir, bbox, crossing_lats, intersect_radius_r, ...
    argo_base_mask, argo_lon, argo_lat, argo_time, argo_park, argo_pf, argo_u, argo_v, argo_wpk, history_match_mask, rho, depth, ...
    time_window_days, depth_levels, deg_m, cache_root, max_matches_per_group, isas_density_mat)

    %#ok<INUSD> cache_root is kept in the signature to match other diagnostics runners.
    grid_files = {};
    polarities = {'cyclonic','anticyclonic'};
    summary_header = {'polarity','crossing_lat','lat_label','meta_source','meta_total_count','meta_selected_count', ...
        'meta_crossing_count','match_count','unique_argo_count','duplicate_match_count','isas_path','isas_available','isas_strict_polarity_match', ...
        'q95_abs_term1_1e6_m_s','q95_abs_term2_1e6_m_s','q95_abs_w_1e6_m_s','deep_q95_abs_term2_1e6_m_s', ...
        'median_corr_w_vs_1000m','deep_reversal_score','first_zero_crossing_depth_m','output_dir'};
    summary_rows = {};

    grid_n_diag = 61;
    smooth_passes_diag = 4;
    cressman_radius_r_diag = 1.0;
    cressman_min_obs_diag = 8;
    match_mode_diag = 'all';
    section_axis_diag = 'x';
    section_half_width_r_diag = 0.25;

    meta_cache_dir = fullfile(output_root, '_meta32_allsat_inputs');
    if exist(meta_cache_dir, 'dir') ~= 7
        mkdir(meta_cache_dir);
    end

    for p = 1:numel(polarities)
        polarity = polarities{p};
        meta = load_meta32_allsat_predecessor_snapshots(meta32_allsat_dir, polarity, meta_cache_dir, deg_m);

        meta_lon = meta.lon;
        meta_lat = meta.lat;
        meta_time = meta.time;
        meta_track = meta.track;
        meta_radius = meta.radius;
        meta_cx = meta.cx;

        for b = 1:numel(crossing_lats)
            target_lat = crossing_lats(b);
            band_label = crossing_label(target_lat, intersect_radius_r);
            cross_distance_m = abs(meta_lat - target_lat) * deg_m;
            meta_band = find(meta_lon >= bbox(1) & meta_lon <= bbox(2) & isfinite(meta_radius) & meta_radius > 0 & ...
                cross_distance_m <= meta_radius * intersect_radius_r);

            if isempty(meta_band)
                argo_band = [];
            else
                argo_lat_window_m = (intersect_radius_r + 4) * max(meta_radius(meta_band));
                argo_band = find(argo_base_mask & abs(argo_lat - target_lat) * deg_m <= argo_lat_window_m);
            end
            log_step(sprintf('%s %s predecessor-allsat candidates: %d Argo profiles, %d META seven-snapshots', ...
                polarity, band_label, numel(argo_band), numel(meta_band)));

            [matches, grid3d] = build_group_3d_absolute_density(argo_band, meta_band, polarity, band_label, ...
                argo_lon, argo_lat, argo_time, argo_park, argo_pf, argo_u, argo_v, argo_wpk, history_match_mask, rho, depth, ...
                meta_lon, meta_lat, meta_time, meta_track, meta_radius, meta_cx, ...
                time_window_days, depth_levels, grid_n_diag, smooth_passes_diag, cressman_radius_r_diag, cressman_min_obs_diag, ...
                match_mode_diag, max_matches_per_group, deg_m, section_axis_diag, section_half_width_r_diag);

            hybrid = argo_absolute_term1_isas_term2_terms(grid3d, polarity, isas_density_mat);
            hybrid.meta32_allsat = meta;
            hybrid.validation_mode = 'predecessor algorithm with META3.2 allsat substitute for missing twosat';
            group_dir = fullfile(output_root, polarity, band_label);
            if exist(group_dir, 'dir') ~= 7
                mkdir(group_dir);
            end

            plot_predecessor_allsat_validation_4panel(fullfile(group_dir, 'predecessor_allsat_section_4panel.png'), hybrid, polarity, band_label);
            plot_hybrid_depth_slices(fullfile(group_dir, 'predecessor_allsat_depth_slices.png'), hybrid, polarity, band_label);
            write_predecessor_allsat_validation_doc(fullfile(group_dir, 'PREDECESSOR_ALLSAT_VALIDATION_ZH.md'), hybrid, polarity, band_label, matches, meta);
            save(fullfile(group_dir, 'predecessor_allsat_terms.mat'), 'hybrid', 'matches', '-v7.3');
            grid_files{end+1} = fullfile(group_dir, 'predecessor_allsat_terms.mat'); %#ok<AGROW>

            stats = hybrid.stats;
            summary_rows(end+1,:) = {polarity, target_lat, band_label, meta.source_file, meta.total_count, meta.keep_count, ...
                numel(meta_band), size(matches,1), count_unique_argo(matches), size(matches,1) - count_unique_argo(matches), ...
                hybrid.isas_info.path, hybrid.isas_info.available, hybrid.isas_info.strict_polarity_match, stats.q95_abs_term1_1e6_m_s, stats.q95_abs_term2_1e6_m_s, ...
                stats.q95_abs_w_1e6_m_s, stats.deep_q95_abs_term2_1e6_m_s, stats.median_corr_w_vs_1000m, ...
                stats.deep_reversal_score, stats.first_zero_crossing_depth_m, group_dir}; %#ok<AGROW>
        end
    end

    write_summary_table_mat(fullfile(output_root, 'PREDECESSOR_ALLSAT_VALIDATION_SUMMARY.mat'), summary_header, summary_rows);
    write_predecessor_allsat_validation_summary_doc(fullfile(output_root, 'PREDECESSOR_ALLSAT_VALIDATION_SUMMARY_ZH.md'), summary_header, summary_rows);
end
