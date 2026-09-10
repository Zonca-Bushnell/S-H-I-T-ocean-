function comparison = threeway_z_background_terms(grid3d, polarity, isas_density_mat)
    radius_m = double(grid3d.mean_radius_m);
    x_vec = grid3d.x(1,:);
    y_vec = grid3d.y(:,1);
    dx_m = median(diff(x_vec), 'omitnan') * radius_m;
    dy_m = median(diff(y_vec), 'omitnan') * radius_m;
    depth_levels = grid3d.depth_levels(:);
    anchor = nearest_depth_index(depth_levels, 1000);
    base_u = grid3d.u_tw(:,:,anchor);
    base_v = grid3d.v_tw(:,:,anchor);
    base_support = isfinite(base_u) & isfinite(base_v);
    support_anom = isfinite(grid3d.z_anom) & isfinite(grid3d.rho_anom);
    support_abs = support_anom & isfinite(grid3d.rho_abs);
    support_boa = support_anom & isfield(grid3d, 'rho_boa') & isfinite(grid3d.rho_boa);

    t = tic;
    [dzdx_anom, dzdy_anom] = gradient_stack_depth_positive(grid3d.z_anom, dx_m, dy_m);
    log_step(sprintf('three-way current anomaly gradient ready in %.1f s', toc(t)));
    t = tic;
    [dzdx_abs, dzdy_abs] = predecessor_isopycnal_slope(grid3d.rho_abs, depth_levels, dx_m, dy_m, false);
    log_step(sprintf('three-way composite-density isopycnal slope ready in %.1f s', toc(t)));
    t = tic;
    [u_abs, v_abs] = thermal_wind_velocity_stack(grid3d.rho_abs, support_abs, base_u, base_v, base_support, depth_levels, dx_m, dy_m, grid3d.thermal_wind_f_s_1);
    log_step(sprintf('three-way composite-density thermal wind ready in %.1f s', toc(t)));
    t = tic;
    [dzdx_boa, dzdy_boa] = predecessor_isopycnal_slope(grid3d.rho_boa, depth_levels, dx_m, dy_m, false);
    log_step(sprintf('three-way BOA background isopycnal slope ready in %.1f s', toc(t)));
    t = tic;
    [u_boa, v_boa] = thermal_wind_velocity_stack(grid3d.rho_boa, support_boa, base_u, base_v, base_support, depth_levels, dx_m, dy_m, grid3d.thermal_wind_f_s_1);
    log_step(sprintf('three-way BOA background thermal wind ready in %.1f s', toc(t)));

    panels = struct('name', {}, 'title', {}, 'description', {}, 'term1', {}, 'term2', {}, 'w', {}, 'section_w', {}, 'section_term2', {}, 'stats', {});
    panels(1) = make_threeway_panel('A_boa_anomaly', 'A BOA anomaly z''_\rho', ...
        '当前正式口径：term1 和 term2 都使用 BOA 背景扣除后的 z''_\rho 几何。', ...
        grid3d.cx_rel .* dzdx_anom, -((grid3d.u_tw - grid3d.mean_u_bg) .* dzdx_anom + grid3d.v_tw .* dzdy_anom), support_anom, grid3d, []);
    panels(2) = make_threeway_panel('B_absolute_composite_density', 'B composite-density absolute z_\rho', ...
        'term1 和 term2 都使用涡旋合成密度场反插得到的整体等密面集合。', ...
        grid3d.cx_rel .* dzdx_abs, -((u_abs - grid3d.mean_u_bg) .* dzdx_abs + v_abs .* dzdy_abs), support_abs, grid3d, panels(1));
    panels(3) = make_threeway_panel('C_boa_background_term2', 'C eddy z_\rho + BOA background term2', ...
        'term1 使用涡旋合成 absolute z_\rho；term2 使用 BOA 背景合成密度场的等密面斜率和热成风速度。', ...
        grid3d.cx_rel .* dzdx_abs, -((u_boa - grid3d.mean_u_bg) .* dzdx_boa + v_boa .* dzdy_boa), support_abs & support_boa, grid3d, panels(1));

    isas_info = struct('path', '', 'available', false, 'message', 'ISAS predecessor density field not available for this polarity.');
    rho_isas = load_predecessor_isas_density_stack(isas_density_mat, polarity, grid3d.x, grid3d.y, depth_levels);
    if any(isfinite(rho_isas(:)))
        isas_info.path = resolved_isas_density_path(isas_density_mat, polarity);
        isas_info.available = true;
        isas_info.message = 'Loaded predecessor ISAS-derived density composite.';
        support_isas = support_anom & isfinite(rho_isas);
        t = tic;
        [dzdx_isas, dzdy_isas] = predecessor_isopycnal_slope(rho_isas, depth_levels, dx_m, dy_m, false);
        log_step(sprintf('three-way ISAS-derived background isopycnal slope ready in %.1f s', toc(t)));
        t = tic;
        [u_isas, v_isas] = thermal_wind_velocity_stack(rho_isas, support_isas, base_u, base_v, base_support, depth_levels, dx_m, dy_m, grid3d.thermal_wind_f_s_1);
        log_step(sprintf('three-way ISAS-derived background thermal wind ready in %.1f s', toc(t)));
        panels(4) = make_threeway_panel('C_isas_background_term2', 'C eddy z_\rho + ISAS background term2', ...
            'term1 使用涡旋合成 absolute z_\rho；term2 使用前辈 ISAS-derived 背景合成密度场。', ...
            grid3d.cx_rel .* dzdx_abs, -((u_isas - grid3d.mean_u_bg) .* dzdx_isas + v_isas .* dzdy_isas), support_abs & support_isas, grid3d, panels(1));
    else
        panels(4) = make_threeway_panel('C_isas_background_term2_missing', 'C ISAS background unavailable', ...
            isas_info.message, nan(size(grid3d.w)), nan(size(grid3d.w)), false(size(grid3d.w)), grid3d, panels(1));
    end

    comparison = struct();
    comparison.x = grid3d.x;
    comparison.y = grid3d.y;
    comparison.depth_levels = depth_levels;
    comparison.section_axis = grid3d.section_axis;
    comparison.section_half_width_r = grid3d.section_half_width_r;
    comparison.z_rho_anom = grid3d.z_anom;
    comparison.rho_anom = grid3d.rho_anom;
    comparison.rho_abs = grid3d.rho_abs;
    comparison.rho_boa = grid3d.rho_boa;
    comparison.isas_info = isas_info;
    comparison.dzdx_anom = dzdx_anom;
    comparison.dzdy_anom = dzdy_anom;
    comparison.dzdx_abs = dzdx_abs;
    comparison.dzdy_abs = dzdy_abs;
    comparison.dzdx_boa = dzdx_boa;
    comparison.dzdy_boa = dzdy_boa;
    comparison.panel_names = {panels.name};
    comparison.panels = panels;
    comparison.panel_stats = [panels.stats];
end

function panel = make_threeway_panel(name, title_text, description, term1, term2, support3, grid3d, reference_panel)
    term1 = mask_stack(term1, support3);
    term2 = mask_stack(term2, support3);
    w = mask_stack(term1 + term2, support3);
    section_w = section_stack(w, grid3d.y(:,1), grid3d.section_half_width_r);
    section_term2 = section_stack(term2, grid3d.y(:,1), grid3d.section_half_width_r);
    stats = reversal_section_stats(section_w, grid3d.depth_levels(:));
    stats.q95_abs_w_1e6_m_s = q95_abs(w(:) * 1e6);
    stats.q95_abs_term2_1e6_m_s = q95_abs(term2(:) * 1e6);
    deep_mask = reshape(grid3d.depth_levels(:) >= 1000 & grid3d.depth_levels(:) <= 2000, 1, 1, []);
    stats.deep_q95_abs_term2_1e6_m_s = q95_abs(term2(repmat(deep_mask, size(term2,1), size(term2,2), 1)) * 1e6);
    stats.term2_corr_vs_A = NaN;
    stats.background_term2_ratio_vs_A = NaN;
    if ~isempty(reference_panel)
        stats.term2_corr_vs_A = corr_finite(term2(:), reference_panel.term2(:));
        ref_q95 = q95_abs(reference_panel.term2(:) * 1e6);
        if isfinite(ref_q95) && ref_q95 > 0
            stats.background_term2_ratio_vs_A = stats.q95_abs_term2_1e6_m_s / ref_q95;
        end
    end
    panel = struct('name', name, 'title', title_text, 'description', description, ...
        'term1', term1, 'term2', term2, 'w', w, 'section_w', section_w, 'section_term2', section_term2, 'stats', stats);
end
