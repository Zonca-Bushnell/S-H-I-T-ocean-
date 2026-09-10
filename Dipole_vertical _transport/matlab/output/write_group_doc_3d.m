function write_group_doc_3d(path, matches, grid3d, polarity, band_label)
    fid = fopen(path, 'w');
    fprintf(fid, '# %s %s Core Argo 三维 W 重建摘要\n\n', polarity, band_label);
    fprintf(fid, '- 垂向模式：`%s`，逐名义深度层重建 `W(x/R,y/R,z)`。\n', grid3d.vertical_mode);
    fprintf(fid, '- 深度层：');
    for i = 1:numel(grid3d.depth_levels)
        fprintf(fid, '`%.0f m` ', grid3d.depth_levels(i));
    end
    fprintf(fid, '\n- 匹配样本数：`%d`，唯一 Argo profile：`%d`。\n', size(matches,1), grid3d.unique_argo_count);
    fprintf(fid, '- mean c_x_raw：`%.6g m/s`，mean u_bg：`%.6g m/s`，c_x_rel：`%.6g m/s`。\n', grid3d.mean_cx_raw, grid3d.mean_u_bg, grid3d.cx_rel);
    if strcmp(grid3d.vertical_mode, 'thermal_wind_depth_stack')
        fprintf(fid, '- 热成风口径：以 1000 m `I_Upk/I_Vpk` 为锚定速度，使用 `rho_anom` 的水平梯度按正深度向下积分得到 `u_tw(z), v_tw(z)`。\n');
        fprintf(fid, '- 热成风剪切：`du/dD = g/(f*rho_ref) * d rho''/dy`，`dv/dD = -g/(f*rho_ref) * d rho''/dx`，`D` 为正深度向下，`f=%.6g s^-1`。\n', grid3d.thermal_wind_f_s_1);
        fprintf(fid, '- 公式：`term1 = +c_x_rel dz''_rho/dx`，`term2 = -[(u_tw-u_bg,v_tw)·grad(z''_rho)]`，`W = term1 + term2`。\n');
    else
        fprintf(fid, '- 公式：`term1 = +c_x_rel dz''_rho/dx`，`term2 = -[(u_pk-u_bg,v_pk)·grad(z''_rho)]`，`W = term1 + term2`。\n');
    end
    fprintf(fid, '- W 符号：向上为正；深度和 `z_rho_anom` 按正深度向下保存和标注。\n');
    fprintf(fid, '- 横截面：`%s` 方向，半宽 `%.3gR`，纵坐标显示正深度数值。\n', grid3d.section_axis, grid3d.section_half_width_r);
    fprintf(fid, '- 输出：默认 `matched_core_argo_3d.mat`、`w_3d_grid.mat`、`w_3d_grid.nc`、`w_3d_section_%s.png`、`w_3d_depth_slices.png`；网格内包含 `z_anom/rho_anom/u_tw/v_tw/term1/term2/w`。\n', grid3d.section_axis);
    fclose(fid);
end
