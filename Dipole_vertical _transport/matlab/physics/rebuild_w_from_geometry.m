function grid = rebuild_w_from_geometry(grid, plot_filled_gradient)
    nan_grid = nan(size(grid.x));
    fields = {'term1','term2','rebuild_w','term1_depth_positive','term2_depth_positive', ...
        'rebuild_w_raw_depth_positive','term1_plus','term1_minus','term2_abs','term2_rel', ...
        'rebuild_plus_abs','rebuild_minus_rel'};
    for k = 1:numel(fields)
        grid.(fields{k}) = nan_grid;
    end
    grid.corr_rebuild_wpk = NaN;
    grid.w_stage = 'physics_rebuild_from_isopycnal_geometry';
    x_vec = grid.x(1,:);
    y_vec = grid.y(:,1);
    dx_m = mean(diff(x_vec)) * grid.mean_radius_m;
    dy_m = mean(diff(y_vec)) * grid.mean_radius_m;
    if ~isfinite(dx_m) || dx_m <= 0 || ~isfinite(dy_m) || dy_m <= 0
        return
    end
    support = grid.mapped_support >= grid.cressman_min_obs;
    [dDdx, dDdy] = gradient_xy(fillmissing2(grid.z), dx_m, dy_m);
    grid.dDdx = mask_to_support(dDdx, support);
    grid.dDdy = mask_to_support(dDdy, support);

    grid.term1_plus = mask_to_support(grid.cx_rel .* dDdx, support);
    grid.term1_minus = mask_to_support(-grid.cx_rel .* dDdx, support);
    grid.term2_abs = mask_to_support(grid.u .* dDdx + grid.v .* dDdy, support);
    grid.term2_rel = mask_to_support((grid.u - grid.mean_cx_raw) .* dDdx + grid.v .* dDdy, support);
    grid.rebuild_plus_abs = mask_to_support(grid.term1_plus + grid.term2_abs, support);
    grid.rebuild_minus_rel = mask_to_support(grid.term1_minus + grid.term2_rel, support);

    grid.term1_depth_positive = grid.term1_minus;
    grid.term2_depth_positive = grid.term2_rel;
    grid.rebuild_w_raw_depth_positive = grid.rebuild_minus_rel;
    if plot_filled_gradient
        grid.term1 = grid.cx_rel .* dDdx;
    else
        grid.term1 = grid.term1_plus;
    end
    grid.term2 = mask_to_support(-grid.term2_rel, support);
    grid.rebuild_w = mask_to_support(grid.term1 + grid.term2, support);
    grid.corr_rebuild_wpk = spatial_corr(grid.rebuild_w, grid.wpk);
end
