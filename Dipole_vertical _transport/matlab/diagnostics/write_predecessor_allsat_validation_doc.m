function write_predecessor_allsat_validation_doc(path, hybrid, polarity, band_label, matches, meta)
    lines = {};
    lines{end+1} = '# META3.2 allsat 前辈算法验证说明';
    lines{end+1} = '';
    lines{end+1} = sprintf('- 极性：`%s`', polarity);
    lines{end+1} = sprintf('- 目标：`%s`', band_label);
    lines{end+1} = '- 目的：验证前辈算法口径在 `twosat` 缺失时，使用 META3.2 allsat 替代是否仍能产生深层反转或规整剖面。';
    lines{end+1} = '';
    lines{end+1} = '## 数据与替代关系';
    lines{end+1} = '';
    lines{end+1} = sprintf('- META 源：`%s`', meta.source_file);
    lines{end+1} = sprintf('- META 抽样：%s', meta.selection);
    lines{end+1} = sprintf('- META 原始记录数：%d；七阶段抽样后记录数：%d；原始 track 数：%d。', meta.total_count, meta.keep_count, meta.track_count);
    lines{end+1} = sprintf('- Argo 匹配记录数：%d；唯一 Argo profile 数：%d；重复匹配数：%d。', size(matches,1), count_unique_argo(matches), size(matches,1) - count_unique_argo(matches));
    lines{end+1} = sprintf('- ISAS 背景文件：`%s`', hybrid.isas_info.path);
    lines{end+1} = sprintf('- ISAS 可用性：`%d`；极性严格匹配：`%d`；说明：%s', hybrid.isas_info.available, hybrid.isas_info.strict_polarity_match, hybrid.isas_info.message);
    lines{end+1} = '';
    lines{end+1} = '## 前辈式口径';
    lines{end+1} = '';
    lines{end+1} = '- term1：从 Argo composite absolute density 得到整体等密面集合，再用相邻密度柱反插同一密度面的深度差计算 `dD_rho/dx`。';
    lines{end+1} = '- 热成风：优先从 Argo composite absolute density 的水平密度梯度积分，1000 m 由 matched Argo parking drift 锚定。';
    lines{end+1} = '- term2：使用 ISAS background density 的整体等密面斜率，速度使用 Argo absolute density 热成风速度。';
    lines{end+1} = '- 坐标：矩阵列是东西向 `x/R`，矩阵行是南北向 `y/R`；深度 `D` 正向下；`W` 向上为正。';
    lines{end+1} = '';
    lines{end+1} = '## 反转诊断';
    lines{end+1} = '';
    lines{end+1} = sprintf('- `q95(|term1|)`：%.3g ×10^-6 m s^-1', hybrid.stats.q95_abs_term1_1e6_m_s);
    lines{end+1} = sprintf('- `q95(|term2|)`：%.3g ×10^-6 m s^-1', hybrid.stats.q95_abs_term2_1e6_m_s);
    lines{end+1} = sprintf('- `q95(|W|)`：%.3g ×10^-6 m s^-1', hybrid.stats.q95_abs_w_1e6_m_s);
    lines{end+1} = sprintf('- 深层 term2 q95：%.3g ×10^-6 m s^-1', hybrid.stats.deep_q95_abs_term2_1e6_m_s);
    lines{end+1} = sprintf('- W 与 1000 m 结构的中位相关：%.3f', hybrid.stats.median_corr_w_vs_1000m);
    lines{end+1} = sprintf('- 深层反转分数：%.3f', hybrid.stats.deep_reversal_score);
    lines{end+1} = sprintf('- 第一处剖面零交叉深度：%.1f m', hybrid.stats.first_zero_crossing_depth_m);
    lines{end+1} = '';
    lines{end+1} = '## 解释边界';
    lines{end+1} = '';
    lines{end+1} = '- 这不是原始 `twosat` 的严格复现，因为缺少前辈原始 `META3.2_DT_twosat_*` 文件。';
    lines{end+1} = '- 若本结果出现深层反转，说明反转更可能来自前辈算法的密度几何和 term2 背景斜率口径，而不是 `twosat` 文件本身。';
    lines{end+1} = '- 若本结果不出现深层反转，优先怀疑缺失的原始 twosat、seven-snapshot 中间文件、ISAS 背景极性文件或样本筛选差异。';
    write_lines_utf8(path, lines);
end

function write_lines_utf8(path, lines)
    fid = fopen(path, 'w', 'n', 'UTF-8');
    if fid < 0
        error('Cannot write %s', path);
    end
    cleaner = onCleanup(@() fclose(fid));
    for ii = 1:numel(lines)
        fprintf(fid, '%s\n', lines{ii});
    end
    clear cleaner
end
