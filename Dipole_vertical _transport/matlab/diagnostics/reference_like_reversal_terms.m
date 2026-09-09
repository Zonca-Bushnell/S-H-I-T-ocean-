function result = reference_like_reversal_terms(grid3d, opts)
    rho_abs = grid3d.rho_abs;
    support3 = isfinite(rho_abs) & isfinite(grid3d.u_tw) & isfinite(grid3d.v_tw);
    radius_m = double(grid3d.mean_radius_m);
    x_vec = grid3d.x(1,:);
    y_vec = grid3d.y(:,1);
    dx_m = median(diff(x_vec), 'omitnan') * radius_m;
    dy_m = median(diff(y_vec), 'omitnan') * radius_m;
    depth_levels = grid3d.depth_levels(:);
    dD = median(diff(depth_levels), 'omitnan');

    rho_work = rho_abs;
    for kk = 1:numel(depth_levels)
        rho_work(:,:,kk) = fillmissing2(rho_work(:,:,kk));
    end
    h_kernel = reshape([1 2 1; 2 4 2; 1 2 1] ./ 16, 3, 3, 1);
    z_kernel = reshape([0.25 0.5 0.25], 1, 1, 3);
    for pass = 1:opts.horizontal_smooth_passes
        rho_work = nan_weighted_convn(rho_work, h_kernel);
    end
    for pass = 1:opts.vertical_smooth_passes
        rho_work = nan_weighted_convn(rho_work, z_kernel);
    end

    [rho_x, rho_y, rho_D] = gradient_density_stack(rho_work, dx_m, dy_m, dD);
    strat_ok = isfinite(rho_D) & abs(rho_D) >= opts.min_abs_rho_D;
    dzdx = -rho_x ./ rho_D;
    dzdy = -rho_y ./ rho_D;
    slope_values = [dzdx(strat_ok); dzdy(strat_ok)];
    slope_cap = quantile(abs(slope_values(isfinite(slope_values))), opts.slope_cap_quantile);
    if ~isfinite(slope_cap) || slope_cap <= 0
        slope_cap = opts.default_slope_cap;
    end
    dzdx = max(min(dzdx, slope_cap), -slope_cap);
    dzdy = max(min(dzdy, slope_cap), -slope_cap);
    support = support3 & strat_ok & isfinite(dzdx) & isfinite(dzdy);

    term1 = grid3d.cx_rel .* dzdx;
    term2 = -((grid3d.u_tw - grid3d.mean_cx_raw) .* dzdx + grid3d.v_tw .* dzdy);
    w = mask_stack(term1 + term2, support);
    term1 = mask_stack(term1, support);
    term2 = mask_stack(term2, support);

    result = struct();
    result.method = 'reference_like_absolute_density_implicit_slope';
    result.rho_abs_smoothed = rho_work;
    result.rho_x = mask_stack(rho_x, support);
    result.rho_y = mask_stack(rho_y, support);
    result.rho_D = mask_stack(rho_D, support);
    result.dzdx = mask_stack(dzdx, support);
    result.dzdy = mask_stack(dzdy, support);
    result.term1 = term1;
    result.term2 = term2;
    result.w = w;
    result.support = support;
    result.section_w = section_stack(w, grid3d.y(:,1), grid3d.section_half_width_r);
    result.section_term1 = section_stack(term1, grid3d.y(:,1), grid3d.section_half_width_r);
    result.section_term2 = section_stack(term2, grid3d.y(:,1), grid3d.section_half_width_r);
    result.stats = reversal_section_stats(result.section_w, depth_levels);
    result.stats.q95_abs_w_1e6_m_s = q95_abs(w(:) * 1e6);
    result.stats.valid_fraction = nnz(isfinite(w)) / numel(w);
    result.stats.slope_cap = slope_cap;
    result.options = opts;
end

function out = nan_weighted_convn(in, kernel)
    mask = isfinite(in);
    values = in;
    values(~mask) = 0;
    weight = convn(double(mask), kernel, 'same');
    summed = convn(values, kernel, 'same');
    out = summed ./ weight;
    out(weight <= 0) = NaN;
end

function [rho_x, rho_y, rho_D] = gradient_density_stack(rho_work, dx_m, dy_m, dD)
    [ny, nx, nz] = size(rho_work);
    rho_x = nan(ny, nx, nz);
    rho_y = nan(ny, nx, nz);
    rho_D = nan(ny, nx, nz);
    for kk = 1:nz
        [gx, gy] = gradient(rho_work(:,:,kk), dx_m, dy_m);
        rho_x(:,:,kk) = gx;
        rho_y(:,:,kk) = gy;
    end
    for kk = 1:nz
        if kk == 1
            rho_D(:,:,kk) = (rho_work(:,:,kk+1) - rho_work(:,:,kk)) ./ dD;
        elseif kk == nz
            rho_D(:,:,kk) = (rho_work(:,:,kk) - rho_work(:,:,kk-1)) ./ dD;
        else
            rho_D(:,:,kk) = (rho_work(:,:,kk+1) - rho_work(:,:,kk-1)) ./ (2 * dD);
        end
    end
end
