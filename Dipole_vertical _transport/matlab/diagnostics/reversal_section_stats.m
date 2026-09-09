function stats = reversal_section_stats(section_w, depth_levels)
    anchor = nearest_depth_index(depth_levels, 1000);
    base = section_w(anchor,:);
    corr_by_depth = nan(numel(depth_levels), 1);
    for kk = 1:numel(depth_levels)
        a = section_w(kk,:);
        good = isfinite(a) & isfinite(base);
        if nnz(good) >= 5
            C = corrcoef(a(good), base(good));
            corr_by_depth(kk) = C(1,2);
        end
    end
    shallow_mask = depth_levels <= 700;
    deep_mask = depth_levels >= 1200;
    shallow_pattern = median(section_w(shallow_mask,:), 1, 'omitnan');
    deep_pattern = median(section_w(deep_mask,:), 1, 'omitnan');
    good = isfinite(shallow_pattern) & isfinite(deep_pattern);
    if nnz(good) >= 5
        C = corrcoef(shallow_pattern(good), deep_pattern(good));
        deep_reversal_score = -C(1,2);
    else
        deep_reversal_score = NaN;
    end
    profile = median(section_w, 2, 'omitnan');
    first_zero = NaN;
    for kk = 2:numel(profile)
        if isfinite(profile(kk-1)) && isfinite(profile(kk)) && profile(kk-1) * profile(kk) < 0
            first_zero = 0.5 * (depth_levels(kk-1) + depth_levels(kk));
            break
        end
    end
    stats = struct('corr_w_vs_1000m', corr_by_depth, ...
        'median_corr_w_vs_1000m', median(corr_by_depth, 'omitnan'), ...
        'deep_reversal_score', deep_reversal_score, ...
        'first_zero_crossing_depth_m', first_zero, ...
        'q95_abs_w_1e6_m_s', NaN);
end
