function write_reference_like_reversal_summary_doc(path, summary_rows)
    fid = fopen(path, 'w');
    cleanup = onCleanup(@() fclose(fid));
    fprintf(fid, '# Reference-like 深层反转诊断汇总\n\n');
    fprintf(fid, '本次诊断固定 `20N crossing / 1R / match-mode all / recommended Cressman`，只测试 `triangle z_rho` 的几何口径是否能让 W 剖面更接近参考图中的深层反相结构。\n\n');
    fprintf(fid, '## 方法摘要\n\n');
    fprintf(fid, '正式 BOA anomaly 口径是先计算 `z''_rho = z_profile - z_BOA_bg`，再合成异常等密面起伏并求梯度。reference-like 口径改为先合成总密度场 `rho_abs(x/R,y/R,D)`，随后用隐式等密面关系：\n\n');
    fprintf(fid, '```text\n');
    fprintf(fid, 'dD/dx|rho = -rho_x / rho_D\n');
    fprintf(fid, 'dD/dy|rho = -rho_y / rho_D\n');
    fprintf(fid, '```\n\n');
    fprintf(fid, '这里 `D` 为正深度向下，W 仍按向上为正输出。该方法会保留合成总密度场中的背景斜率和垂向结构，因此最直接检验“深层反转是否来自整体等密面集合”。\n\n');
    fprintf(fid, '## 结果表\n\n');
    fprintf(fid, '| polarity | matches | unique Argo | valid fraction | q95 W | median corr W/1000m | deep reversal score | first zero depth | slope cap |\n');
    fprintf(fid, '|---|---:|---:|---:|---:|---:|---:|---:|---:|\n');
    for ii = 1:size(summary_rows, 1)
        r = summary_rows(ii,:);
        fprintf(fid, '| %s | %d | %d | %.3f | %.3g | %.3g | %.3g | %.3g | %.3g |\n', ...
            r{1}, r{2}, r{3}, r{4}, r{5}, r{6}, r{7}, r{8}, r{9});
    end
    fprintf(fid, '\n## 判读\n\n');
    fprintf(fid, '若该图比正式 BOA anomaly 版本更连续并出现深层反相，说明当前最该修改的不是 Argo-META 匹配，而是 `triangle z_rho` 的定义：从“异常等密面起伏”改为“合成总密度场中的整体等密面斜率”会显著改变深层 W 的相位。\n');
end
