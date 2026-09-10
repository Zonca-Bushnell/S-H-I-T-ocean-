function write_threeway_z_background_summary_doc(path, summary_rows)
    fid = fopen(path, 'w');
    cleaner = onCleanup(@() fclose(fid));
    fprintf(fid, '# 20N 三分 z_rho / 背景密度几何诊断汇总\n\n');
    fprintf(fid, '本诊断用于判断当前 BOA anomaly 几何是否扣除了深层背景等密面坡度，以及 BOA/ISAS 背景项是否会显著改变 term2 和 W 的深层结构。\n\n');
    fprintf(fid, '| polarity | panel | matches | unique Argo | q95 W | q95 term2 | deep q95 term2 | corr term2 vs A | term2 ratio vs A | reversal | zero depth |\n');
    fprintf(fid, '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n');
    for ii = 1:size(summary_rows, 1)
        fprintf(fid, '| %s | %s | %d | %d | %.4g | %.4g | %.4g | %.4g | %.4g | %.4g | %.4g |\n', ...
            summary_rows{ii,1}, summary_rows{ii,2}, summary_rows{ii,3}, summary_rows{ii,4}, summary_rows{ii,5}, ...
            summary_rows{ii,6}, summary_rows{ii,7}, summary_rows{ii,8}, summary_rows{ii,9}, summary_rows{ii,11}, summary_rows{ii,12});
    end
    fprintf(fid, '\nA 是正式 BOA `z''_rho` anomaly；B 是涡旋合成 absolute `z_rho`；C_BOA/C_ISAS 是 term1 用涡旋 absolute 斜率、term2 用背景密度斜率的混合口径。注意 C_ISAS 只有在本地存在对应极性的前辈 ISAS-derived 密度场时才有效。\n');
end
