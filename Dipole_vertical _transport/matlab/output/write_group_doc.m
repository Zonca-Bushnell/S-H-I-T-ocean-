function write_group_doc(path, matches, grid, polarity, band_label)
    fid = fopen(path, 'w');
    fprintf(fid, '# %s %s Core Argo 垂直速度重建摘要\n\n', polarity, band_label);
    fprintf(fid, '- 匹配样本数：`%d`\n', size(matches,1));
    fprintf(fid, '- mean c_x_raw：`%.6g m/s`\n', grid.mean_cx_raw);
    fprintf(fid, '- mean u_bg：`%.6g m/s`\n', grid.mean_u_bg);
    fprintf(fid, '- c_x_rel：`%.6g m/s`\n', grid.cx_rel);
    fprintf(fid, '- mean observed I_Wpk：`%.6g m/s`\n', grid.mean_wpk_observed);
    fprintf(fid, '- corr(rebuild_W, I_Wpk)：`%.6g`\n', grid.corr_rebuild_wpk);
    fprintf(fid, '- corr(sample-gradient W, I_Wpk)：`%.6g`\n', grid.corr_sample_rebuild_wpk);
    fprintf(fid, '- BOA 背景有效样本：`%d / %d`\n', grid.boa_bg_valid_count, size(matches,1));
    fprintf(fid, '- mean radius：`%.3f km`\n', grid.mean_radius_m / 1000);
    valid_cells = sum(isfinite(grid.rebuild_w(:)));
    valid_fraction = valid_cells / numel(grid.count);
    fprintf(fid, '- 有样本支撑网格：`%d / %d (%.2f%%)`。\n', valid_cells, numel(grid.count), valid_fraction * 100);
    fprintf(fid, '- 符号约定：正式几何统一为 `D_rho` 正深度向下；历史字段名 `z_rho/z_rho_anom` 按 `D_rho/D''_rho` 解释。W 向上为正，即 `W_up = -D_t`。\n');
    fprintf(fid, '- 输出：默认 `matched_core_argo.mat`、`composite_grid.mat`、`composite_grid.nc`、`vertical_transport_terms.png`、`wpk_validation.png`、`gradient_order_comparison.png`、`velocity_sign_sensitivity.png`。图像显示为 `10^-6 m/s`，网格文件保存原始 `m/s`，白色为空样本格点。\n');
    fclose(fid);
end
