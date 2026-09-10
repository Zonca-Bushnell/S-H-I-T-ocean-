function grid3d = composite_grid_3d(matches, rho, depth, boa_clim, depth_levels, grid_n, min_bin_count, smooth_passes, cressman_radius_r, cressman_min_obs, min_drho_dz, max_rho_bracket_dz_m, section_axis, section_half_width_r, vertical_mode, cache_root, polarity, band_label, match_mode)
    x_vec = linspace(-4, 4, grid_n);
    y_vec = linspace(-4, 4, grid_n);
    [X, Y] = meshgrid(x_vec, y_vec);
    nz = numel(depth_levels);
    nan3 = nan(grid_n, grid_n, nz);
    count3 = zeros(grid_n, grid_n, nz);
    grid3d = struct('x', X, 'y', Y, 'depth_levels', depth_levels(:), 'w', nan3, 'term1', nan3, 'term2', nan3, ...
        'z_anom', nan3, 'rho_anom', nan3, 'rho_abs', nan3, 'rho_boa', nan3, 'u_tw', nan3, 'v_tw', nan3, 'count', count3, 'mapped_support', count3, 'valid_profile_count', zeros(nz,1), ...
        'boa_bg_valid_count', zeros(nz,1), 'mean_cx_raw', NaN, 'mean_u_bg', NaN, 'cx_rel', NaN, ...
        'mean_radius_m', NaN, 'section_axis', section_axis, 'section_half_width_r', section_half_width_r, ...
        'section_coord', [], 'section_w', [], 'match_count', 0, 'unique_argo_count', 0, 'duplicate_match_count', 0, ...
        'match_mode', '', 'z_mode', '', 'vertical_mode', vertical_mode, 'thermal_wind_anchor_depth_m', 1000, ...
        'thermal_wind_f_s_1', NaN, 'min_drho_dz', NaN, 'max_rho_bracket_dz_m', NaN);
    if isempty(matches) || isempty(fieldnames(boa_clim))
        return
    end
    x = cell2mat(matches(:,24)); x = x(:);
    y = cell2mat(matches(:,25)); y = y(:);
    u = cell2mat(matches(:,9)); u = u(:);
    v = cell2mat(matches(:,10)); v = v(:);
    cx_raw = cell2mat(matches(:,28)); cx_raw = cx_raw(:);
    radius = cell2mat(matches(:,23)); radius = radius(:);
    argo_indices = cell2mat(matches(:,3)); argo_indices = argo_indices(:);
    argo_lon_match = cell2mat(matches(:,6)); argo_lon_match = argo_lon_match(:);
    argo_lat_match = cell2mat(matches(:,7)); argo_lat_match = argo_lat_match(:);
    argo_time_match = cell2mat(matches(:,5)); argo_time_match = argo_time_match(:);
    grid3d.mean_cx_raw = mean(cx_raw, 'omitnan');
    grid3d.mean_u_bg = mean(u, 'omitnan');
    grid3d.cx_rel = grid3d.mean_cx_raw - grid3d.mean_u_bg;
    grid3d.mean_radius_m = mean(radius, 'omitnan');
    grid3d.thermal_wind_f_s_1 = 2 * 7.2921159e-5 * sind(mean(argo_lat_match, 'omitnan'));
    edges = linspace(-4, 4, grid_n + 1);
    dx_m = mean(diff(x_vec)) * grid3d.mean_radius_m;
    dy_m = mean(diff(y_vec)) * grid3d.mean_radius_m;
    profile_timer = tic;
    profile_cache = profile_depth_stack_cache(matches, rho, depth, boa_clim, depth_levels, min_drho_dz, max_rho_bracket_dz_m, cache_root, polarity, band_label, match_mode);
    log_step(sprintf('%s %s profile depth cache ready in %.1f s', polarity, band_label, toc(profile_timer)));
    [~, unique_pos] = ismember(argo_indices, profile_cache.argo_indices);
    finite_unique = unique_pos > 0;
    base_timer = tic;
    [base_uv, base_support] = cressman_map_multi(x(finite_unique), y(finite_unique), [u(finite_unique), v(finite_unique)], X, Y, cressman_radius_r, cressman_min_obs);
    log_step(sprintf('%s %s base velocity Cressman ready in %.1f s', polarity, band_label, toc(base_timer)));
    u_base = base_uv(:,:,1);
    v_base = base_uv(:,:,2);
    base_support = base_support >= cressman_min_obs;
    rho_grids = nan3;
    support3 = false(grid_n, grid_n, nz);
    z_samples = nan(numel(x), nz);
    rho_samples = nan(numel(x), nz);
    rho_abs_samples = nan(numel(x), nz);
    boa_rho_samples = nan(numel(x), nz);
    z_samples(finite_unique,:) = profile_cache.z_anom(unique_pos(finite_unique), :);
    rho_samples(finite_unique,:) = profile_cache.rho_anom(unique_pos(finite_unique), :);
    if isfield(profile_cache, 'rho_abs')
        rho_abs_samples(finite_unique,:) = profile_cache.rho_abs(unique_pos(finite_unique), :);
    end
    if isfield(profile_cache, 'boa_rho')
        boa_rho_samples(finite_unique,:) = profile_cache.boa_rho(unique_pos(finite_unique), :);
    end
    base_good = isfinite(x) & isfinite(y) & isfinite(u) & isfinite(v) & isfinite(cx_raw) & hypot(x, y) <= 4;
    z_samples(~base_good,:) = NaN;
    rho_samples(~base_good,:) = NaN;
    rho_abs_samples(~base_good,:) = NaN;
    boa_rho_samples(~base_good,:) = NaN;
    valid_pair = isfinite(z_samples) & isfinite(rho_samples);
    z_samples(~valid_pair) = NaN;
    rho_samples(~valid_pair) = NaN;
    rho_abs_samples(~valid_pair) = NaN;
    boa_rho_samples(~valid_pair) = NaN;
    map_timer = tic;
    [mapped_stack, support_stack] = cressman_map_multi_missing(x, y, [z_samples, rho_samples, rho_abs_samples, boa_rho_samples], X, Y, cressman_radius_r, cressman_min_obs);
    for zz = 1:nz
        grid3d.valid_profile_count(zz) = nnz(profile_cache.profile_valid(:,zz));
        grid3d.boa_bg_valid_count(zz) = nnz(profile_cache.boa_valid(:,zz));
        good = base_good & valid_pair(:,zz);
        xb = discretize(x(good), edges);
        yb = discretize(y(good), edges);
        bin_ok = isfinite(xb) & isfinite(yb);
        if any(bin_ok)
            subs = [yb(bin_ok), xb(bin_ok)];
            grid3d.count(:,:,zz) = accumarray(subs, 1, [grid_n grid_n], @sum, 0);
        end
        z_grid = mapped_stack(:,:,zz);
        rho_grid = mapped_stack(:,:,nz+zz);
        rho_abs_grid = mapped_stack(:,:,2*nz+zz);
        rho_boa_grid = mapped_stack(:,:,3*nz+zz);
        support = support_stack(:,:,zz) >= cressman_min_obs & support_stack(:,:,nz+zz) >= cressman_min_obs & ...
            support_stack(:,:,2*nz+zz) >= cressman_min_obs & support_stack(:,:,3*nz+zz) >= cressman_min_obs;
        grid3d.mapped_support(:,:,zz) = min(min(support_stack(:,:,zz), support_stack(:,:,nz+zz)), min(support_stack(:,:,2*nz+zz), support_stack(:,:,3*nz+zz)));
        if smooth_passes > 0
            z_grid = smooth2_supported(z_grid, support, smooth_passes);
            rho_grid = smooth2_supported(rho_grid, support, smooth_passes);
            rho_abs_grid = smooth2_supported(rho_abs_grid, support, smooth_passes);
            rho_boa_grid = smooth2_supported(rho_boa_grid, support, smooth_passes);
        end
        grid3d.z_anom(:,:,zz) = mask_to_support(z_grid, support);
        grid3d.rho_anom(:,:,zz) = mask_to_support(rho_grid, support);
        grid3d.rho_abs(:,:,zz) = mask_to_support(rho_abs_grid, support);
        grid3d.rho_boa(:,:,zz) = mask_to_support(rho_boa_grid, support);
        rho_grids(:,:,zz) = rho_grid;
        support3(:,:,zz) = support;
    end
    log_step(sprintf('%s %s depth-stack Cressman mapped %d layers in %.1f s', polarity, band_label, nz, toc(map_timer)));
    if strcmp(vertical_mode, 'thermal_wind_depth_stack')
        tw_timer = tic;
        [grid3d.u_tw, grid3d.v_tw] = thermal_wind_velocity_stack(rho_grids, support3, u_base, v_base, base_support, depth_levels, dx_m, dy_m, grid3d.thermal_wind_f_s_1);
        log_step(sprintf('%s %s thermal-wind velocity stack ready in %.1f s', polarity, band_label, toc(tw_timer)));
    else
        grid3d.u_tw = repmat(u_base, 1, 1, nz);
        grid3d.v_tw = repmat(v_base, 1, 1, nz);
    end
    w_timer = tic;
    for zz = 1:nz
        z_grid = grid3d.z_anom(:,:,zz);
        support = support3(:,:,zz) & isfinite(grid3d.u_tw(:,:,zz)) & isfinite(grid3d.v_tw(:,:,zz));
        if isfinite(dx_m) && dx_m > 0 && isfinite(dy_m) && dy_m > 0
            [dzdx, dzdy] = gradient_xy(fillmissing2(z_grid), dx_m, dy_m);
            [term1, term2, w_up] = w_terms_from_depth_geometry(dzdx, dzdy, grid3d.u_tw(:,:,zz), grid3d.v_tw(:,:,zz), grid3d.cx_rel, grid3d.mean_u_bg, support);
            grid3d.term1(:,:,zz) = term1;
            grid3d.term2(:,:,zz) = term2;
            grid3d.w(:,:,zz) = w_up;
        end
    end
    log_step(sprintf('%s %s W terms computed for %d layers in %.1f s', polarity, band_label, nz, toc(w_timer)));
    [grid3d.section_coord, grid3d.section_w] = section_from_grid3d(grid3d, section_axis, section_half_width_r);
end
