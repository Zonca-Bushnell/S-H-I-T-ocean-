function grid3d = composite_grid_3d_absolute_density(matches, rho, depth, depth_levels, grid_n, smooth_passes, cressman_radius_r, cressman_min_obs, section_axis, section_half_width_r)
    x_vec = linspace(-4, 4, grid_n);
    y_vec = linspace(-4, 4, grid_n);
    [X, Y] = meshgrid(x_vec, y_vec);
    nz = numel(depth_levels);
    nan3 = nan(grid_n, grid_n, nz);
    count3 = zeros(grid_n, grid_n, nz);
    grid3d = struct('x', X, 'y', Y, 'depth_levels', depth_levels(:), 'w', nan3, 'term1', nan3, 'term2', nan3, ...
        'z_anom', nan3, 'rho_anom', nan3, 'rho_abs', nan3, 'rho_boa', nan3, 'u_tw', nan3, 'v_tw', nan3, ...
        'count', count3, 'mapped_support', count3, 'valid_profile_count', zeros(nz,1), 'boa_bg_valid_count', zeros(nz,1), ...
        'mean_cx_raw', NaN, 'mean_u_bg', NaN, 'cx_rel', NaN, 'mean_radius_m', NaN, ...
        'section_axis', section_axis, 'section_half_width_r', section_half_width_r, 'section_coord', [], 'section_w', [], ...
        'match_count', 0, 'unique_argo_count', 0, 'duplicate_match_count', 0, 'match_mode', '', 'z_mode', '', 'vertical_mode', '', ...
        'thermal_wind_anchor_depth_m', 1000, 'thermal_wind_f_s_1', NaN, 'min_drho_dz', NaN, 'max_rho_bracket_dz_m', NaN);
    if isempty(matches)
        return
    end
    x = cell2mat(matches(:,24)); x = x(:);
    y = cell2mat(matches(:,25)); y = y(:);
    u = cell2mat(matches(:,9)); u = u(:);
    v = cell2mat(matches(:,10)); v = v(:);
    cx_raw = cell2mat(matches(:,28)); cx_raw = cx_raw(:);
    radius = cell2mat(matches(:,23)); radius = radius(:);
    argo_indices = cell2mat(matches(:,3)); argo_indices = argo_indices(:);
    argo_lat_match = cell2mat(matches(:,7)); argo_lat_match = argo_lat_match(:);
    grid3d.mean_cx_raw = mean(cx_raw, 'omitnan');
    grid3d.mean_u_bg = mean(u, 'omitnan');
    grid3d.cx_rel = grid3d.mean_cx_raw - grid3d.mean_u_bg;
    grid3d.mean_radius_m = mean(radius, 'omitnan');
    grid3d.thermal_wind_f_s_1 = 2 * 7.2921159e-5 * sind(mean(argo_lat_match, 'omitnan'));
    base_good = isfinite(x) & isfinite(y) & isfinite(u) & isfinite(v) & isfinite(cx_raw) & hypot(x, y) <= 4;

    base_timer = tic;
    [base_uv, base_support_count] = cressman_map_multi(x(base_good), y(base_good), [u(base_good), v(base_good)], X, Y, cressman_radius_r, cressman_min_obs);
    grid3d.u_tw = repmat(base_uv(:,:,1), 1, 1, nz);
    grid3d.v_tw = repmat(base_uv(:,:,2), 1, 1, nz);
    base_support = base_support_count >= cressman_min_obs;
    for zz = 1:nz
        grid3d.u_tw(:,:,zz) = mask_to_support(grid3d.u_tw(:,:,zz), base_support);
        grid3d.v_tw(:,:,zz) = mask_to_support(grid3d.v_tw(:,:,zz), base_support);
    end
    log_step(sprintf('absolute-density base velocity Cressman ready in %.1f s', toc(base_timer)));

    sample_timer = tic;
    rho_abs_samples = nan(numel(argo_indices), nz);
    for zz = 1:nz
        rho_abs_samples(:,zz) = profile_values_at_depth_by_index(depth, rho, argo_indices, repmat(depth_levels(zz), numel(argo_indices), 1));
    end
    rho_abs_samples(~base_good,:) = NaN;
    log_step(sprintf('absolute-density profile interpolation ready: %d matches x %d depths in %.1f s', numel(argo_indices), nz, toc(sample_timer)));

    map_timer = tic;
    [rho_abs_grid, support_stack] = cressman_map_multi_missing(x, y, rho_abs_samples, X, Y, cressman_radius_r, cressman_min_obs);
    edges = linspace(-4, 4, grid_n + 1);
    for zz = 1:nz
        support = support_stack(:,:,zz) >= cressman_min_obs;
        grid3d.mapped_support(:,:,zz) = support_stack(:,:,zz);
        grid3d.valid_profile_count(zz) = nnz(isfinite(rho_abs_samples(:,zz)));
        good = base_good & isfinite(rho_abs_samples(:,zz));
        xb = discretize(x(good), edges);
        yb = discretize(y(good), edges);
        bin_ok = isfinite(xb) & isfinite(yb);
        if any(bin_ok)
            grid3d.count(:,:,zz) = accumarray([yb(bin_ok), xb(bin_ok)], 1, [grid_n grid_n], @sum, 0);
        end
        rho_grid = rho_abs_grid(:,:,zz);
        if smooth_passes > 0
            rho_grid = smooth2_supported(rho_grid, support, smooth_passes);
        end
        grid3d.rho_abs(:,:,zz) = mask_to_support(rho_grid, support);
    end
    log_step(sprintf('absolute-density Cressman mapped %d layers in %.1f s', nz, toc(map_timer)));
end
