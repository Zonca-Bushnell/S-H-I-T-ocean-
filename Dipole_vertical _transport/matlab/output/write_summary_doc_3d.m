function write_summary_doc_3d(path, summary_rows, output_root, depth_levels, section_axis, section_half_width_r)
    fid = fopen(path, 'w');
    fprintf(fid, '# META4.0 + Core Argo 三维 W 重建运行摘要\n\n');
    fprintf(fid, '- 输出根目录：`%s`\n', output_root);
    fprintf(fid, '- 深度层：');
    for i = 1:numel(depth_levels)
        fprintf(fid, '`%.0f m` ', depth_levels(i));
    end
    fprintf(fid, '\n- 横截面：`%s` 方向，半宽 `%.3gR`。纵坐标为正深度数值，W 向上为正。\n', section_axis, section_half_width_r);
    fprintf(fid, '- 公式：`term1 = +c_x_rel dz''_rho/dx`，`term2 = -[(u_pk-u_bg,v_pk)·grad(z''_rho)]`，`W = term1 + term2`。\n\n');
    fprintf(fid, '| polarity | lat_band | matches | unique Argo | depth count | valid voxels | valid voxel %% | mean cells/depth | BOA bg %% | profile valid %% | c_x_rel m/s | q95 W |\n');
    fprintf(fid, '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n');
    for i=1:size(summary_rows,1)
        fprintf(fid, '| %s | %s | %d | %d | %d | %d | %.2f | %.1f | %.2f | %.2f | %.6g | %.3g |\n', ...
            summary_rows{i,1}, summary_rows{i,2}, summary_rows{i,3}, summary_rows{i,4}, summary_rows{i,9}, summary_rows{i,10}, ...
            summary_rows{i,11} * 100, summary_rows{i,12}, summary_rows{i,18} * 100, summary_rows{i,19} * 100, summary_rows{i,15}, summary_rows{i,20});
    end
    fclose(fid);
end
