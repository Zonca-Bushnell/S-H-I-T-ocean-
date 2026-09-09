function write_zgeometry_comparison_doc(path, comparison, polarity, band_label, match_count, unique_argo_count)
    fid = fopen(path, 'w');
    fprintf(fid, '# %s %s 等密面几何口径对照\n\n', polarity, band_label);
    fprintf(fid, '本输出从 matched Argo 重新计算三维 W，而不是只对已有 MAT 做后处理。匹配口径固定为 `20N crossing / 1R / match-mode all / recommended`，不生成 combined。\n\n');
    fprintf(fid, '- 匹配记录数：`%d`\n', match_count);
    fprintf(fid, '- 唯一 Argo profile：`%d`\n', unique_argo_count);
    fprintf(fid, '- 请求的 z_geometry_mode：`%s`\n', comparison.requested_z_geometry_mode);
    fprintf(fid, '- W 符号：向上为正；深度显示：正深度向下。\n\n');
    fprintf(fid, '| panel | 几何/速度/term2 口径 | median corr W/1000m | deep reversal score | first zero depth m | q95 W |\n');
    fprintf(fid, '|---|---|---:|---:|---:|---:|\n');
    for ii = 1:numel(comparison.panels)
        s = comparison.panel_stats(ii);
        fprintf(fid, '| %s | %s | %.3g | %.3g | %.3g | %.3g |\n', comparison.panel_names{ii}, ...
            comparison.panel_descriptions{ii}, s.median_corr_w_vs_1000m, s.deep_reversal_score, ...
            s.first_zero_crossing_depth_m, s.q95_abs_w_1e6_m_s);
    end
    fprintf(fid, '\n判读规则：如果 B 相对 A 明显出现深层反转，主因就是 `z_rho` 从 BOA 相对异常几何换成整体等密面集合；如果 B 不明显而 C/D 改变明显，则分别指向热成风速度口径或 term2 速度/符号组合。\n');
    fclose(fid);
end
