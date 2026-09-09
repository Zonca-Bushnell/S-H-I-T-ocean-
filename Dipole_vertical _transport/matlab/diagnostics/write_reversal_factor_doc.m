function write_reversal_factor_doc(path, diag, polarity, band_label)
    fid = fopen(path, 'w');
    fprintf(fid, '# %s %s 三因素反转诊断\n\n', polarity, band_label);
    fprintf(fid, '本诊断用于判断深层 W 反转是否由 `等密面斜率`、`热成风速度` 或 `term2` 组合方式造成。\n\n');
    fprintf(fid, '| variant | 口径 | median corr W/1000m | deep reversal score | first zero depth m | q95 W |\n');
    fprintf(fid, '|---|---|---:|---:|---:|---:|\n');
    for i = 1:numel(diag.variant_names)
        s = diag.variant_stats(i);
        fprintf(fid, '| %s | %s | %.3g | %.3g | %.3g | %.3g |\n', diag.variant_names{i}, diag.variant_descriptions{i}, ...
            s.median_corr_w_vs_1000m, s.deep_reversal_score, s.first_zero_crossing_depth_m, s.q95_abs_w_1e6_m_s);
    end
    fprintf(fid, '\n判读：如果 B 开始反转，主因是 `composite rho -> isopycnal slope`；如果 C 开始反转，主因偏热成风速度；如果 D 开始反转，主因偏 term2 的符号/速度/斜率组合。\n');
    fclose(fid);
end
