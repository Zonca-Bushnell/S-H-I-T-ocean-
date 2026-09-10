function write_summary_doc(path, summary_rows, output_root)
    fid = fopen(path, 'w');
    fprintf(fid, '# META4.0 + Core Argo 垂直速度重建运行摘要\n\n');
    fprintf(fid, '- 输出根目录：`%s`\n', output_root);
    fprintf(fid, '- 主图变量：W 向上为正，`term1 = +c_x_rel dz''_rho/dx`，`term2 = -[(u_pk-u_bg, v_pk) · grad(z''_rho)]`，`rebuild_W = term1 + term2`。\n');
    fprintf(fid, '- 深度变量：`z_rho_m`、`z_rho_bg_m`、`z_rho_anom_m` 仍为正深度向下；默认 MAT/NetCDF 保存原始 `m/s`；PNG 色标显示为 `10^-6 m/s`；白色为空样本格点。\n');
    fprintf(fid, '- 若使用 `--max-matches-per-group` 做 smoke run，覆盖率会很低；正式结果应使用默认 `0` 读取全部匹配。\n\n');
    fprintf(fid, '| polarity | lat_band | matches | unique Argo | duplicated | 0-1R | 1-2R | 2-4R | valid grid %% | BOA bg %% | c_x_rel m/s | corr W/Wpk | corr sample/Wpk | q95 rebuild | q95 sample | q95 Wpk |\n');
    fprintf(fid, '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n');
    for i=1:size(summary_rows,1)
        fprintf(fid, '| %s | %s | %d | %d | %d | %d | %d | %d | %.2f | %.2f | %.6g | %.3g | %.3g | %.3g | %.3g | %.3g |\n', summary_rows{i,1}, summary_rows{i,2}, summary_rows{i,3}, summary_rows{i,4}, summary_rows{i,5}, summary_rows{i,6}, summary_rows{i,7}, summary_rows{i,8}, summary_rows{i,10} * 100, summary_rows{i,17} * 100, summary_rows{i,13}, summary_rows{i,19}, summary_rows{i,20}, summary_rows{i,21}, summary_rows{i,22}, summary_rows{i,23});
    end
    fclose(fid);
end
