function row = summary_from_matches(matches, grid, polarity, band_label, group_dir)
    if isempty(matches)
        rings = {};
        n = 0;
        unique_argo_count = 0;
        history_velocity_match_count = 0;
        boa_bg_valid_count = 0;
    else
        rings = matches(:,27);
        n = size(matches, 1);
        unique_argo_count = numel(unique(cell2mat(matches(:,3))));
        history_velocity_match_count = sum(cell2mat(matches(:,29)));
        if size(matches, 2) >= 33
            boa_bg_valid_count = sum(cell2mat(matches(:,33)));
        else
            boa_bg_valid_count = 0;
        end
    end
    duplicate_match_count = n - unique_argo_count;
    valid_cells = sum(isfinite(grid.rebuild_w(:)));
    valid_fraction = valid_cells / numel(grid.count);
    rebuild = grid.rebuild_w(:);
    sample_rebuild = grid.sample_rebuild_w(:);
    wpk = grid.wpk(:);
    if any(isfinite(rebuild))
        q95_rebuild = prctile(abs(rebuild(isfinite(rebuild))) * 1e6, 95);
    else
        q95_rebuild = NaN;
    end
    if any(isfinite(sample_rebuild))
        q95_sample_rebuild = prctile(abs(sample_rebuild(isfinite(sample_rebuild))) * 1e6, 95);
    else
        q95_sample_rebuild = NaN;
    end
    if any(isfinite(wpk))
        q95_wpk = prctile(abs(wpk(isfinite(wpk))) * 1e6, 95);
    else
        q95_wpk = NaN;
    end
    row = {polarity, band_label, n, unique_argo_count, duplicate_match_count, sum(strcmp(rings,'0-1R')), sum(strcmp(rings,'1-2R')), sum(strcmp(rings,'2-4R')), ...
        valid_cells, valid_fraction, grid.mean_cx_raw, grid.mean_u_bg, grid.cx_rel, grid.mean_radius_m / 1000, ...
        history_velocity_match_count, boa_bg_valid_count, safe_fraction(boa_bg_valid_count, n), grid.mean_wpk_observed, grid.corr_rebuild_wpk, ...
        grid.corr_sample_rebuild_wpk, q95_rebuild, q95_sample_rebuild, q95_wpk, group_dir};
end
