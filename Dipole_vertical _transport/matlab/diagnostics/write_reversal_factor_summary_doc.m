function write_reversal_factor_summary_doc(path, summary_rows)
    fid = fopen(path, 'w');
    fprintf(fid, '# 20N 三因素反转诊断汇总\n\n');
    fprintf(fid, '| polarity | variant | matches | unique Argo | median corr W/1000m | deep reversal score | first zero depth m | q95 W | output |\n');
    fprintf(fid, '|---|---|---:|---:|---:|---:|---:|---:|---|\n');
    for i = 1:size(summary_rows,1)
        fprintf(fid, '| %s | %s | %d | %d | %.3g | %.3g | %.3g | %.3g | `%s` |\n', ...
            summary_rows{i,1}, summary_rows{i,2}, summary_rows{i,3}, summary_rows{i,4}, summary_rows{i,5}, summary_rows{i,6}, summary_rows{i,7}, summary_rows{i,8}, summary_rows{i,9});
    end
    fclose(fid);
end
