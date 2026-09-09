function diag = reversal_factor_terms(grid3d)
    rho_abs = grid3d.rho_abs;
    if ~isfield(grid3d, 'rho_abs') || ~any(isfinite(rho_abs(:)))
        rho_abs = grid3d.rho_anom;
    end
    support3 = isfinite(grid3d.z_anom) & isfinite(grid3d.rho_anom);
    radius_m = double(grid3d.mean_radius_m);
    x_vec = grid3d.x(1,:);
    y_vec = grid3d.y(:,1);
    dx_m = median(diff(x_vec), 'omitnan') * radius_m;
    dy_m = median(diff(y_vec), 'omitnan') * radius_m;
    depth_levels = grid3d.depth_levels(:);
    [dzdx_current, dzdy_current] = gradient_stack_depth_positive(grid3d.z_anom, dx_m, dy_m);
    [dzdx_comp_down, dzdy_comp_down] = predecessor_isopycnal_slope(rho_abs, depth_levels, dx_m, dy_m, false);
    [dzdx_comp_up, dzdy_comp_up] = predecessor_isopycnal_slope(rho_abs, depth_levels, dx_m, dy_m, true);
    [u_tw_comp, v_tw_comp] = thermal_wind_velocity_stack(rho_abs, support3, grid3d.u_tw(:,:,nearest_depth_index(depth_levels, 1000)), ...
        grid3d.v_tw(:,:,nearest_depth_index(depth_levels, 1000)), isfinite(grid3d.u_tw(:,:,nearest_depth_index(depth_levels, 1000))) & isfinite(grid3d.v_tw(:,:,nearest_depth_index(depth_levels, 1000))), ...
        depth_levels, dx_m, dy_m, grid3d.thermal_wind_f_s_1);

    variants = struct('name', {}, 'description', {}, 'term1', {}, 'term2', {}, 'w', {}, 'section_w', {}, 'stats', {});
    variants(1) = make_reversal_variant_from_factors('A_current', '当前：BOA z''_rho anomaly 梯度 + rho_anom 热成风 + 当前相对速度 term2', ...
        dzdx_current, dzdy_current, -dzdx_current, -dzdy_current, grid3d.u_tw, grid3d.v_tw, 'current', support3, grid3d);
    variants(2) = make_reversal_variant_from_factors('B_comp_isoslope', '只换前辈式：composite rho 反插得到等密面斜率；速度和 term2 仍用当前口径', ...
        dzdx_comp_down, dzdy_comp_down, dzdx_comp_up, dzdy_comp_up, grid3d.u_tw, grid3d.v_tw, 'current', support3, grid3d);
    variants(3) = make_reversal_variant_from_factors('C_comp_isoslope_comp_tw', '前辈式等密面斜率 + 用 composite density 梯度积分热成风；term2 仍用当前相对速度符号', ...
        dzdx_comp_down, dzdy_comp_down, dzdx_comp_up, dzdy_comp_up, u_tw_comp, v_tw_comp, 'current', support3, grid3d);
    variants(4) = make_reversal_variant_from_factors('D_predecessor_like', '尽量接近前辈：composite rho 等密面斜率 + composite density 热成风 + c0/绝对速度 term2', ...
        dzdx_comp_down, dzdy_comp_down, dzdx_comp_up, dzdy_comp_up, u_tw_comp, v_tw_comp, 'predecessor', support3, grid3d);
    variants(5) = make_reversal_variant_from_factors('E_current_slope_comp_tw', '当前 z''_rho 梯度 + composite density 热成风 + 当前 term2', ...
        dzdx_current, dzdy_current, -dzdx_current, -dzdy_current, u_tw_comp, v_tw_comp, 'current', support3, grid3d);
    variants(6) = make_reversal_variant_from_factors('F_current_slope_pred_term2', '当前 z''_rho 梯度 + 当前热成风 + 前辈式 c0/绝对速度 term2', ...
        dzdx_current, dzdy_current, -dzdx_current, -dzdy_current, grid3d.u_tw, grid3d.v_tw, 'predecessor', support3, grid3d);
    variants(7) = make_reversal_variant_from_factors('G_comp_isoslope_pred_term2', '前辈式等密面斜率 + 当前热成风 + 前辈式 c0/绝对速度 term2', ...
        dzdx_comp_down, dzdy_comp_down, dzdx_comp_up, dzdy_comp_up, grid3d.u_tw, grid3d.v_tw, 'predecessor', support3, grid3d);
    variants(8) = make_reversal_variant_from_factors('H_current_slope_comp_tw_pred_term2', '当前 z''_rho 梯度 + composite density 热成风 + 前辈式 c0/绝对速度 term2', ...
        dzdx_current, dzdy_current, -dzdx_current, -dzdy_current, u_tw_comp, v_tw_comp, 'predecessor', support3, grid3d);

    diag = struct();
    diag.x = grid3d.x;
    diag.y = grid3d.y;
    diag.depth_levels = depth_levels;
    diag.section_axis = grid3d.section_axis;
    diag.section_half_width_r = grid3d.section_half_width_r;
    diag.dzdx_current = dzdx_current;
    diag.dzdy_current = dzdy_current;
    diag.dzdx_comp_density_down = dzdx_comp_down;
    diag.dzdy_comp_density_down = dzdy_comp_down;
    diag.dzdx_comp_density_up = dzdx_comp_up;
    diag.dzdy_comp_density_up = dzdy_comp_up;
    diag.u_tw_current = grid3d.u_tw;
    diag.v_tw_current = grid3d.v_tw;
    diag.u_tw_comp_density = u_tw_comp;
    diag.v_tw_comp_density = v_tw_comp;
    diag.z_rho_anom = grid3d.z_anom;
    diag.rho_abs = rho_abs;
    diag.rho_anom = grid3d.rho_anom;
    diag.variant_names = {variants.name};
    diag.variant_descriptions = {variants.description};
    diag.variants = variants;
    diag.variant_stats = [variants.stats];
end
