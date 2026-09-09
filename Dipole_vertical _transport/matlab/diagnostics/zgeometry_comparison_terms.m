function comparison = zgeometry_comparison_terms(grid3d, requested_mode)
    diag = reversal_factor_terms(grid3d);
    source_names = {'A_current','B_comp_isoslope','C_comp_isoslope_comp_tw','D_predecessor_like'};
    panel_names = {'A_boa_anomaly','B_composite_density_isosurface','C_isosurface_composite_density_tw','D_isosurface_predecessor_term2'};
    panel_modes = {'boa_anomaly','composite_density_isosurface','composite_density_isosurface','composite_density_isosurface'};
    panel_titles = {'A current BOA z''_\rho anomaly', 'B composite rho isosurface', ...
        'C isosurface + composite-density TW', 'D isosurface + predecessor term2'};
    panel_descriptions = { ...
        '当前正式口径：BOA monthly climatology 给定 rho0，先合成 z''_\rho，再求梯度并重建向上为正 W。', ...
        '只替换几何：从合成后的 rho(x,y,z) 反插整体等密面集合，再求 dz_\rho/dx 和 dz_\rho/dy。', ...
        '在整体等密面几何基础上，热成风速度也改用 composite density 的水平密度梯度积分。', ...
        '尽量接近前辈程序：整体等密面几何、composite-density 热成风、绝对速度形式 term2。'};

    panels = struct('name', {}, 'title', {}, 'description', {}, 'z_geometry_mode', {}, ...
        'term1', {}, 'term2', {}, 'w', {}, 'section_w', {}, 'stats', {});
    for ii = 1:numel(source_names)
        src = find(strcmp(diag.variant_names, source_names{ii}), 1);
        if isempty(src)
            error('Missing z-geometry source variant: %s', source_names{ii});
        end
        v = diag.variants(src);
        panels(ii).name = panel_names{ii}; %#ok<AGROW>
        panels(ii).title = panel_titles{ii};
        panels(ii).description = panel_descriptions{ii};
        panels(ii).z_geometry_mode = panel_modes{ii};
        panels(ii).term1 = v.term1;
        panels(ii).term2 = v.term2;
        panels(ii).w = v.w;
        panels(ii).section_w = v.section_w;
        panels(ii).stats = v.stats;
    end

    comparison = struct();
    comparison.requested_z_geometry_mode = requested_mode;
    comparison.x = diag.x;
    comparison.y = diag.y;
    comparison.depth_levels = diag.depth_levels;
    comparison.section_axis = diag.section_axis;
    comparison.section_half_width_r = diag.section_half_width_r;
    comparison.z_rho_anom = diag.z_rho_anom;
    comparison.rho_abs = diag.rho_abs;
    comparison.rho_anom = diag.rho_anom;
    comparison.dzdx_boa_anomaly = diag.dzdx_current;
    comparison.dzdy_boa_anomaly = diag.dzdy_current;
    comparison.dzdx_composite_density_down = diag.dzdx_comp_density_down;
    comparison.dzdy_composite_density_down = diag.dzdy_comp_density_down;
    comparison.dzdx_composite_density_up = diag.dzdx_comp_density_up;
    comparison.dzdy_composite_density_up = diag.dzdy_comp_density_up;
    comparison.u_tw_current = diag.u_tw_current;
    comparison.v_tw_current = diag.v_tw_current;
    comparison.u_tw_comp_density = diag.u_tw_comp_density;
    comparison.v_tw_comp_density = diag.v_tw_comp_density;
    comparison.panel_names = {panels.name};
    comparison.panel_descriptions = {panels.description};
    comparison.panels = panels;
    comparison.panel_stats = [panels.stats];
end
