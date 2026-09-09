function row = sensitivity_summary_row(matches, grid, polarity, cfg, group_dir)
    n = size(matches, 1);
    if isempty(matches)
        unique_count = 0;
    else
        unique_count = numel(unique(cell2mat(matches(:,3))));
    end
    support = grid.mapped_support(:);
    support = support(isfinite(support) & support > 0);
    if isempty(support)
        support_median = NaN;
        support_p10 = NaN;
    else
        support_median = median(support, 'omitnan');
        support_p10 = quantile(support, 0.10);
    end
    rebuild = grid.rebuild_w(:);
    wpk = grid.wpk(:);
    q95_rebuild = quantile(abs(rebuild(isfinite(rebuild))) * 1e6, 0.95);
    q95_wpk = quantile(abs(wpk(isfinite(wpk))) * 1e6, 0.95);
    if isempty(q95_rebuild), q95_rebuild = NaN; end
    if isempty(q95_wpk), q95_wpk = NaN; end
    rough = roughness_score(grid.rebuild_w);
    dipole = dipole_score(grid.rebuild_w, grid.x, grid.y);
    row = {polarity, cfg.name, cfg.grid_n, cfg.cressman_radius_r, cfg.cressman_min_obs, cfg.smooth_passes, ...
        n, unique_count, n - unique_count, safe_fraction(sum(isfinite(grid.rebuild_w(:))), numel(grid.rebuild_w)), ...
        support_median, support_p10, grid.corr_rebuild_wpk, q95_rebuild, q95_wpk, rough, dipole, group_dir};
end
