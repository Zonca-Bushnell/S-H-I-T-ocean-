function grid = composite_grid(matches, grid_n, min_bin_count, plot_filled_gradient, smooth_passes, cressman_radius_r, cressman_min_obs, sample_gradient_max_profiles)
    x_vec = linspace(-4, 4, grid_n);
    y_vec = linspace(-4, 4, grid_n);
    [X, Y] = meshgrid(x_vec, y_vec);
    nan_grid = nan(size(X));
    count_grid = zeros(size(X));
    grid = struct('x', X, 'y', Y, 'z', nan_grid, 'z_raw', nan_grid, 'z_anom', nan_grid, 'u', nan_grid, 'v', nan_grid, 'wpk', nan_grid, ...
        'mapped_support', count_grid, 'wpk_mapped_support', count_grid, ...
        'count', count_grid, 'term1', nan_grid, 'term2', nan_grid, 'rebuild_w', nan_grid, ...
        'term1_depth_positive', nan_grid, 'term2_depth_positive', nan_grid, 'rebuild_w_raw_depth_positive', nan_grid, ...
        'term1_plus', nan_grid, 'term1_minus', nan_grid, 'term2_abs', nan_grid, 'term2_rel', nan_grid, ...
        'rebuild_plus_abs', nan_grid, 'rebuild_minus_rel', nan_grid, ...
        'sample_term1', nan_grid, 'sample_term2', nan_grid, 'sample_rebuild_w', nan_grid, 'corr_sample_rebuild_wpk', NaN, ...
        'mean_cx_raw', NaN, 'mean_u_bg', NaN, 'cx_rel', NaN, 'mean_radius_m', NaN, ...
        'mean_wpk_observed', NaN, 'corr_rebuild_wpk', NaN, ...
        'grid_mapping', 'cressman', 'cressman_radius_r', cressman_radius_r, 'cressman_min_obs', cressman_min_obs, ...
        'z_mode', '', 'z_rho_min_m', NaN, 'z_rho_max_m', NaN, 'min_drho_dz', NaN, 'max_rho_bracket_dz_m', NaN);
    if isempty(matches)
        return
    end
    x = cell2mat(matches(:,24)); x = x(:);
    y = cell2mat(matches(:,25)); y = y(:);
    z_raw = cell2mat(matches(:,13)); z_raw = z_raw(:);
    z = cell2mat(matches(:,15)); z = z(:);
    u = cell2mat(matches(:,9)); u = u(:);
    v = cell2mat(matches(:,10)); v = v(:);
    wpk = cell2mat(matches(:,11)); wpk = wpk(:);
    cx_raw = cell2mat(matches(:,28)); cx_raw = cx_raw(:);
    radius = cell2mat(matches(:,23)); radius = radius(:);
    sample_term1 = nan(size(z));
    sample_term2 = nan(size(z));
    sample_rebuild_w = nan(size(z));
    grid.mean_cx_raw = mean(cx_raw, 'omitnan');
    grid.mean_u_bg = mean(u, 'omitnan');
    grid.cx_rel = grid.mean_cx_raw - grid.mean_u_bg;
    grid.mean_radius_m = mean(radius, 'omitnan');
    grid.mean_wpk_observed = mean(wpk, 'omitnan');
    edges = linspace(-4, 4, grid_n + 1);
    xb = discretize(x, edges);
    yb = discretize(y, edges);
    n = min([numel(xb), numel(yb), numel(z), numel(z_raw), numel(u), numel(v), numel(wpk)]);
    xb = xb(1:n); yb = yb(1:n); z = z(1:n); z_raw = z_raw(1:n); u = u(1:n); v = v(1:n); wpk = wpk(1:n);
    cx_raw = cx_raw(1:n); radius = radius(1:n);
    [sample_term1, sample_term2, sample_rebuild_w] = sample_gradient_terms(x(1:n), y(1:n), z, u, v, cx_raw, radius, grid.cx_rel, cressman_radius_r, cressman_min_obs, sample_gradient_max_profiles);
    valid = isfinite(xb) & isfinite(yb);
    subs = [yb(valid), xb(valid)];
    grid.count = accumarray(subs, 1, [grid_n grid_n], @sum, 0);
    grid.z = accumarray(subs, z(valid), [grid_n grid_n], @(q) median(q, 'omitnan'), NaN);
    grid.z_raw = accumarray(subs, z_raw(valid), [grid_n grid_n], @(q) median(q, 'omitnan'), NaN);
    grid.z_anom = grid.z;
    grid.u = accumarray(subs, u(valid), [grid_n grid_n], @(q) median(q, 'omitnan'), NaN);
    grid.v = accumarray(subs, v(valid), [grid_n grid_n], @(q) median(q, 'omitnan'), NaN);
    grid.wpk = accumarray(subs, wpk(valid), [grid_n grid_n], @(q) median(q, 'omitnan'), NaN);
    grid.sample_term1 = accumarray(subs, sample_term1(valid), [grid_n grid_n], @(q) median(q, 'omitnan'), NaN);
    grid.sample_term2 = accumarray(subs, sample_term2(valid), [grid_n grid_n], @(q) median(q, 'omitnan'), NaN);
    grid.sample_rebuild_w = accumarray(subs, sample_rebuild_w(valid), [grid_n grid_n], @(q) median(q, 'omitnan'), NaN);
    [mapped, support_all] = cressman_map_multi(x, y, [z, z_raw, u, v, wpk], X, Y, cressman_radius_r, cressman_min_obs);
    grid.z = mapped(:,:,1);
    grid.z_raw = mapped(:,:,2);
    grid.z_anom = grid.z;
    grid.u = mapped(:,:,3);
    grid.v = mapped(:,:,4);
    grid.wpk = mapped(:,:,5);
    grid.mapped_support = support_all;
    grid.wpk_mapped_support = support_all;
    if smooth_passes > 0
        support = grid.mapped_support >= cressman_min_obs;
        grid.z = smooth2_supported(grid.z, support, smooth_passes);
        grid.z_raw = smooth2_supported(grid.z_raw, support, smooth_passes);
        grid.z_anom = grid.z;
        grid.u = smooth2_supported(grid.u, support, smooth_passes);
        grid.v = smooth2_supported(grid.v, support, smooth_passes);
        grid.wpk = smooth2_supported(grid.wpk, grid.wpk_mapped_support >= cressman_min_obs, smooth_passes);
    end
    dx_m = mean(diff(x_vec)) * grid.mean_radius_m;
    dy_m = mean(diff(y_vec)) * grid.mean_radius_m;
    if isfinite(dx_m) && dx_m > 0 && isfinite(dy_m) && dy_m > 0
        [dzdx, dzdy] = gradient(fillmissing2(grid.z), dx_m, dy_m);
        support = grid.mapped_support >= cressman_min_obs;
        grid.term1_plus = mask_to_support(grid.cx_rel .* dzdx, support);
        grid.term1_minus = mask_to_support(-grid.cx_rel .* dzdx, support);
        grid.term2_abs = mask_to_support(grid.u .* dzdx + grid.v .* dzdy, support);
        grid.term2_rel = mask_to_support((grid.u - grid.mean_cx_raw) .* dzdx + grid.v .* dzdy, support);
        grid.rebuild_plus_abs = mask_to_support(grid.term1_plus + grid.term2_abs, support);
        grid.rebuild_minus_rel = mask_to_support(grid.term1_minus + grid.term2_rel, support);
        grid.term1_depth_positive = grid.term1_minus;
        grid.term2_depth_positive = grid.term2_rel;
        grid.rebuild_w_raw_depth_positive = grid.rebuild_minus_rel;
        if plot_filled_gradient
            grid.term1 = grid.cx_rel .* dzdx;
        else
            grid.term1 = grid.term1_plus;
        end
        grid.term2 = mask_to_support(-grid.term2_rel, support);
        grid.rebuild_w = mask_to_support(grid.term1 + grid.term2, support);
        grid.corr_rebuild_wpk = spatial_corr(grid.rebuild_w, grid.wpk);
        grid.corr_sample_rebuild_wpk = spatial_corr(grid.sample_rebuild_w, grid.wpk);
    end
end
