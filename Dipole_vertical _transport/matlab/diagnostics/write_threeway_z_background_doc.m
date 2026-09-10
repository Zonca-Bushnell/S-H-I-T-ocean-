function write_threeway_z_background_doc(path, comparison, polarity, band_label, match_count, unique_argo_count)
    fid = fopen(path, 'w');
    cleaner = onCleanup(@() fclose(fid));
    fprintf(fid, '# 三分 z_rho / 背景密度几何诊断\n\n');
    fprintf(fid, '- 极性：`%s`\n', polarity);
    fprintf(fid, '- 纬线：`%s`\n', band_label);
    fprintf(fid, '- 匹配记录数：`%d`\n', match_count);
    fprintf(fid, '- 唯一 Argo profile 数：`%d`\n', unique_argo_count);
    fprintf(fid, '- 坐标口径：矩阵列为东西向 `x/R`，矩阵行为南北向 `y/R`；水平梯度统一经 `gradient_xy`。\n');
    fprintf(fid, '- 符号口径：W 向上为正；深度和等密面深度仍按正深度向下显示。\n\n');
    fprintf(fid, '## 三组含义\n\n');
    fprintf(fid, 'A：当前正式口径，term1 和 term2 都使用 BOA 气候态扣除后的 `z''_rho` 异常几何。\n\n');
    fprintf(fid, 'B：整体等密面口径，先合成 `rho(x,y,z)`，再在相邻整条密度柱中反插同一个 `rho0` 得到 absolute `z_rho` 斜率。\n\n');
    fprintf(fid, 'C：混合口径，term1 使用 B 的涡旋合成等密面斜率，term2 使用背景密度场的等密面斜率；BOA 背景一定输出，ISAS-derived 前辈背景只在本地文件与极性匹配时输出。\n\n');
    fprintf(fid, '## ISAS 状态\n\n');
    fprintf(fid, '- available：`%d`\n', comparison.isas_info.available);
    fprintf(fid, '- path：`%s`\n', comparison.isas_info.path);
    fprintf(fid, '- message：%s\n\n', comparison.isas_info.message);
    fprintf(fid, '## 统计\n\n');
    fprintf(fid, '| panel | q95 W | q95 term2 | deep q95 term2 | corr term2 vs A | term2 ratio vs A | deep reversal score | zero crossing depth |\n');
    fprintf(fid, '|---|---:|---:|---:|---:|---:|---:|---:|\n');
    for ii = 1:numel(comparison.panels)
        s = comparison.panels(ii).stats;
        fprintf(fid, '| %s | %.4g | %.4g | %.4g | %.4g | %.4g | %.4g | %.4g |\n', ...
            comparison.panels(ii).name, s.q95_abs_w_1e6_m_s, s.q95_abs_term2_1e6_m_s, s.deep_q95_abs_term2_1e6_m_s, ...
            s.term2_corr_vs_A, s.background_term2_ratio_vs_A, s.deep_reversal_score, s.first_zero_crossing_depth_m);
    end
    fprintf(fid, '\n判读重点：若 C 的 `deep q95 term2` 相对 A 明显放大，且深层 W 结构改变，说明 BOA anomaly 扣除确实移除了会进入 term2 的深层背景等密面几何。若 ISAS C 比 BOA C 更强，则说明前辈 ISAS-derived 背景可能保留了更强深层大尺度等密面坡度。\n');
end
