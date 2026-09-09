function row = summary_from_matches_3d(matches, grid3d, polarity, band_label, group_dir)
    if isempty(matches)
        rings = [0 0 0];
        unique_count = 0;
        history_count = 0;
    else
        r = cell2mat(matches(:,26));
        rings = [sum(r <= 1), sum(r > 1 & r <= 2), sum(r > 2 & r <= 4)];
        unique_count = numel(unique(cell2mat(matches(:,3))));
        history_count = sum(cell2mat(matches(:,29)));
    end
    match_count = size(matches, 1);
    duplicate_count = match_count - unique_count;
    valid_voxels = sum(isfinite(grid3d.w(:)));
    valid_fraction = safe_fraction(valid_voxels, numel(grid3d.w));
    valid_by_depth = squeeze(sum(sum(isfinite(grid3d.w), 1), 2));
    mean_valid_cells = mean(valid_by_depth, 'omitnan');
    boa_fraction = mean(grid3d.boa_bg_valid_count ./ max(match_count, 1), 'omitnan');
    profile_fraction = mean(grid3d.valid_profile_count ./ max(match_count, 1), 'omitnan');
    q95_w = quantile(abs(grid3d.w(isfinite(grid3d.w))) * 1e6, 0.95);
    if isempty(q95_w) || ~isfinite(q95_w)
        q95_w = NaN;
    end
    row = {polarity, band_label, match_count, unique_count, duplicate_count, rings(1), rings(2), rings(3), ...
        numel(grid3d.depth_levels), valid_voxels, valid_fraction, mean_valid_cells, grid3d.mean_cx_raw, grid3d.mean_u_bg, ...
        grid3d.cx_rel, grid3d.mean_radius_m / 1000, history_count, boa_fraction, profile_fraction, q95_w, group_dir};
end
