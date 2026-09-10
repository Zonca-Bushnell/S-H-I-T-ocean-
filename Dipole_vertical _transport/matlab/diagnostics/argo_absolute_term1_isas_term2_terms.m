function hybrid = argo_absolute_term1_isas_term2_terms(grid3d, polarity, isas_density_mat)
    radius_m = double(grid3d.mean_radius_m);
    x_vec = grid3d.x(1,:);
    y_vec = grid3d.y(:,1);
    dx_m = median(diff(x_vec), 'omitnan') * radius_m;
    dy_m = median(diff(y_vec), 'omitnan') * radius_m;
    depth_levels = grid3d.depth_levels(:);
    support_abs = isfinite(grid3d.rho_abs);

    t = tic;
    [dzdx_abs, dzdy_abs] = predecessor_isopycnal_slope(grid3d.rho_abs, depth_levels, dx_m, dy_m, false);
    log_step(sprintf('hybrid Argo absolute isopycnal slope ready in %.1f s', toc(t)));

    isas_info = struct('path', '', 'available', false, 'message', 'ISAS density field not available.');
    rho_isas = load_predecessor_isas_density_stack(isas_density_mat, polarity, grid3d.x, grid3d.y, depth_levels);
    if any(isfinite(rho_isas(:)))
        isas_info.path = resolved_isas_density_path(isas_density_mat, polarity);
        isas_info.available = true;
        isas_info.message = 'Loaded predecessor ISAS-derived background density composite.';
    end
    support_isas = isfinite(rho_isas);

    t = tic;
    [dzdx_isas, dzdy_isas] = predecessor_isopycnal_slope(rho_isas, depth_levels, dx_m, dy_m, false);
    log_step(sprintf('hybrid ISAS background isopycnal slope ready in %.1f s', toc(t)));

    anchor = nearest_depth_index(depth_levels, 1000);
    base_u = grid3d.u_tw(:,:,anchor);
    base_v = grid3d.v_tw(:,:,anchor);
    base_support = isfinite(base_u) & isfinite(base_v);
    t = tic;
    [u_isas, v_isas] = thermal_wind_velocity_stack(rho_isas, support_isas, base_u, base_v, base_support, depth_levels, dx_m, dy_m, grid3d.thermal_wind_f_s_1);
    log_step(sprintf('hybrid ISAS background thermal wind ready in %.1f s', toc(t)));

    support = support_abs & support_isas & isfinite(u_isas) & isfinite(v_isas);
    [term1, ~] = w_terms_from_depth_geometry(dzdx_abs, dzdy_abs, u_isas, v_isas, grid3d.cx_rel, grid3d.mean_u_bg, support);
    [~, term2] = w_terms_from_depth_geometry(dzdx_isas, dzdy_isas, u_isas, v_isas, grid3d.cx_rel, grid3d.mean_u_bg, support);
    w = mask_stack(term1 + term2, support);

    section_w = section_stack(w, grid3d.y(:,1), grid3d.section_half_width_r);
    section_term1 = section_stack(term1, grid3d.y(:,1), grid3d.section_half_width_r);
    section_term2 = section_stack(term2, grid3d.y(:,1), grid3d.section_half_width_r);
    stats = reversal_section_stats(section_w, depth_levels);
    stats.q95_abs_term1_1e6_m_s = q95_abs(term1(:) * 1e6);
    stats.q95_abs_term2_1e6_m_s = q95_abs(term2(:) * 1e6);
    stats.q95_abs_w_1e6_m_s = q95_abs(w(:) * 1e6);
    deep_mask = reshape(depth_levels >= 1000 & depth_levels <= 2000, 1, 1, []);
    stats.deep_q95_abs_term2_1e6_m_s = q95_abs(term2(repmat(deep_mask, size(term2,1), size(term2,2), 1)) * 1e6);

    hybrid = struct();
    hybrid.x = grid3d.x;
    hybrid.y = grid3d.y;
    hybrid.depth_levels = depth_levels;
    hybrid.section_axis = grid3d.section_axis;
    hybrid.section_half_width_r = grid3d.section_half_width_r;
    hybrid.term1 = term1;
    hybrid.term2 = term2;
    hybrid.w = w;
    hybrid.section_term1 = section_term1;
    hybrid.section_term2 = section_term2;
    hybrid.section_w = section_w;
    hybrid.dzdx_abs_argo = dzdx_abs;
    hybrid.dzdy_abs_argo = dzdy_abs;
    hybrid.dzdx_isas = dzdx_isas;
    hybrid.dzdy_isas = dzdy_isas;
    hybrid.u_isas = u_isas;
    hybrid.v_isas = v_isas;
    hybrid.rho_abs_argo = grid3d.rho_abs;
    hybrid.rho_isas = rho_isas;
    hybrid.isas_info = isas_info;
    hybrid.stats = stats;
    hybrid.mean_radius_m = grid3d.mean_radius_m;
    hybrid.cx_rel = grid3d.cx_rel;
    hybrid.mean_u_bg = grid3d.mean_u_bg;
    hybrid.thermal_wind_f_s_1 = grid3d.thermal_wind_f_s_1;
    hybrid.coordinate_convention = 'D_rho positive downward; W positive upward; matrix columns are east-west x/R and rows are north-south y/R.';
    hybrid.term1_geometry = 'Argo composite absolute density isosurface slope';
    hybrid.term2_geometry = 'ISAS/background density isosurface slope';
end
