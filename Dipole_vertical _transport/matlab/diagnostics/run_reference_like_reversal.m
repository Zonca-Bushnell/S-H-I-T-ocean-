function grid_files = run_reference_like_reversal(output_root, meta_dir, bbox, target_lat, intersect_radius_r, ...
    argo_base_mask, argo_lon, argo_lat, argo_time, argo_park, argo_pf, argo_u, argo_v, argo_wpk, history_match_mask, rho, depth, ...
    time_window_days, depth_levels, deg_m, cache_root, boa_clim, max_matches_per_group)

    grid_files = {};
    polarities = {'cyclonic','anticyclonic'};
    band_label = crossing_label(target_lat, intersect_radius_r);
    summary_header = {'polarity','match_count','unique_argo_count','valid_fraction','q95_abs_w_1e6_m_s','median_corr_w_vs_1000m','deep_reversal_score','first_zero_crossing_depth_m','slope_cap','output_dir'};
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

    opts = struct();
    opts.horizontal_smooth_passes = 10;
    opts.vertical_smooth_passes = 8;
    opts.min_abs_rho_D = 5e-5;
    opts.slope_cap_quantile = 0.99;
    opts.default_slope_cap = 8e-4;

    for p = 1:numel(polarities)
        polarity = polarities{p};
        meta_file = find_meta_file(meta_dir, polarity);
        log_step(sprintf('Reference-like reversal loading META %s from %s', polarity, meta_file));
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

        log_step(sprintf('%s %s reference-like candidates: %d Argo profiles, %d META snapshots', polarity, band_label, numel(argo_band), numel(meta_band)));
        [matches, grid3d] = build_group_3d(argo_band, meta_band, polarity, band_label, ...
            argo_lon, argo_lat, argo_time, argo_park, argo_pf, argo_u, argo_v, argo_wpk, history_match_mask, rho, depth, ...
            meta_lon, meta_lat, meta_time, meta_track, meta_radius, meta_cx, ...
            time_window_days, grid_n_diag, min_bin_count_diag, smooth_passes_diag, cressman_radius_r_diag, cressman_min_obs_diag, ...
            'anomaly_boa_climatology', boa_clim, depth_levels, 1e-5, 150, match_mode_diag, max_matches_per_group, deg_m, section_axis_diag, section_half_width_r_diag, vertical_mode_diag, cache_root);

        ref_timer = tic;
        result = reference_like_reversal_terms(grid3d, opts);
        log_step(sprintf('%s %s reference-like implicit-slope terms ready in %.1f s', polarity, band_label, toc(ref_timer)));
        group_dir = fullfile(output_root, polarity, band_label);
        if exist(group_dir, 'dir') ~= 7
            mkdir(group_dir);
        end
        match_count = size(matches, 1);
        unique_argo_count = count_unique_argo(matches);
        grid_info = struct('x', grid3d.x, 'y', grid3d.y, 'depth_levels', grid3d.depth_levels, ...
            'mean_cx_raw', grid3d.mean_cx_raw, 'mean_u_bg', grid3d.mean_u_bg, 'cx_rel', grid3d.cx_rel, ...
            'mean_radius_m', grid3d.mean_radius_m, 'section_axis', grid3d.section_axis, ...
            'section_half_width_r', grid3d.section_half_width_r, 'match_count', match_count, ...
            'unique_argo_count', unique_argo_count, 'duplicate_match_count', match_count - unique_argo_count);
        save_timer = tic;
        save(fullfile(group_dir, 'reference_like_reversal_terms.mat'), 'result', 'grid_info', '-v7.3');
        log_step(sprintf('%s %s reference-like MAT saved in %.1f s', polarity, band_label, toc(save_timer)));
        plot_timer = tic;
        plot_reference_like_reversal( ...
            fullfile(group_dir, 'reference_like_w_section_x.png'), ...
            fullfile(group_dir, 'reference_like_terms_section.png'), ...
            grid3d, result, polarity, band_label);
        log_step(sprintf('%s %s reference-like PNGs saved in %.1f s', polarity, band_label, toc(plot_timer)));
        doc_timer = tic;
        write_reference_like_reversal_doc(fullfile(group_dir, 'REFERENCE_LIKE_REVERSAL_ZH.md'), result, polarity, band_label, match_count, unique_argo_count);
        log_step(sprintf('%s %s reference-like doc saved in %.1f s', polarity, band_label, toc(doc_timer)));
        grid_files{end+1} = fullfile(group_dir, 'reference_like_reversal_terms.mat'); %#ok<AGROW>
        summary_rows(end+1,:) = {polarity, match_count, unique_argo_count, result.stats.valid_fraction, ...
            result.stats.q95_abs_w_1e6_m_s, result.stats.median_corr_w_vs_1000m, result.stats.deep_reversal_score, ...
            result.stats.first_zero_crossing_depth_m, result.stats.slope_cap, group_dir}; %#ok<AGROW>
    end

    write_summary_table_mat(fullfile(output_root, 'REFERENCE_LIKE_REVERSAL_SUMMARY.mat'), summary_header, summary_rows);
    writecell([summary_header; summary_rows], fullfile(output_root, 'REFERENCE_LIKE_REVERSAL_SUMMARY.csv'));
    write_reference_like_reversal_summary_doc(fullfile(output_root, 'REFERENCE_LIKE_REVERSAL_SUMMARY_ZH.md'), summary_rows);
end
