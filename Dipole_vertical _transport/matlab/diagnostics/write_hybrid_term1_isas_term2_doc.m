function write_hybrid_term1_isas_term2_doc(path, hybrid, polarity, band_label, matches)
    fid = fopen(path, 'w');
    if fid < 0
        warning('Could not write %s', path);
        return
    end
    cleaner = onCleanup(@() fclose(fid));
    fprintf(fid, '# Argo absolute term1 + ISAS term2 诊断\n\n');
    fprintf(fid, '- 极性：`%s`\n', polarity);
    fprintf(fid, '- 纬线：`%s`\n', band_label);
    fprintf(fid, '- 匹配记录数：`%d`\n', size(matches,1));
    fprintf(fid, '- 唯一 Argo profile 数：`%d`\n', count_unique_argo(matches));
    fprintf(fid, '- 重复匹配数：`%d`\n', size(matches,1) - count_unique_argo(matches));
    fprintf(fid, '- 坐标约定：`D_rho` 为正深度向下，`W` 为向上为正；矩阵列是东西向 `x/R`，矩阵行是南北向 `y/R`。\n\n');
    fprintf(fid, '## 口径\n\n');
    fprintf(fid, '本诊断按前辈程序思路重组，但不改变正式默认生产流程：\n\n');
    fprintf(fid, '- `term1`：使用 Argo 合成 absolute density 场，通过整条相邻密度柱反插同一 `rho0` 得到整体等密面斜率，再计算 `+ c_x^{rel} dD_rho/dx`。\n');
    fprintf(fid, '- `term2`：使用 ISAS/background density 场，通过同样的整体等密面反插法得到背景 `dD_rho/dx, dD_rho/dy`，再计算 `-[(u-u_bg)dD_rho/dx + v dD_rho/dy]`。\n');
    fprintf(fid, '- `W`：`term1 + term2`，单位 `m s^-1`，图中用 `10^-6 m s^-1`。\n\n');
    fprintf(fid, '## ISAS 状态\n\n');
    fprintf(fid, '- available：`%d`\n', hybrid.isas_info.available);
    fprintf(fid, '- path：`%s`\n', hybrid.isas_info.path);
    fprintf(fid, '- message：%s\n\n', hybrid.isas_info.message);
    fprintf(fid, '## 指标\n\n');
    fprintf(fid, '| 指标 | 数值 |\n');
    fprintf(fid, '|---|---:|\n');
    fprintf(fid, '| q95 abs term1 (10^-6 m/s) | %.6g |\n', hybrid.stats.q95_abs_term1_1e6_m_s);
    fprintf(fid, '| q95 abs term2 (10^-6 m/s) | %.6g |\n', hybrid.stats.q95_abs_term2_1e6_m_s);
    fprintf(fid, '| q95 abs W (10^-6 m/s) | %.6g |\n', hybrid.stats.q95_abs_w_1e6_m_s);
    fprintf(fid, '| deep q95 abs term2 (10^-6 m/s) | %.6g |\n', hybrid.stats.deep_q95_abs_term2_1e6_m_s);
    fprintf(fid, '| median corr W(z), W(1000m) | %.6g |\n', hybrid.stats.median_corr_w_vs_1000m);
    fprintf(fid, '| deep reversal score | %.6g |\n', hybrid.stats.deep_reversal_score);
    fprintf(fid, '| first zero-crossing depth (m) | %.6g |\n', hybrid.stats.first_zero_crossing_depth_m);
end
