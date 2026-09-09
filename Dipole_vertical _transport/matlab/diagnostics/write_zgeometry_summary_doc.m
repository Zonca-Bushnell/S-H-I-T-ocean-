function write_zgeometry_summary_doc(path, summary_rows)
    fid = fopen(path, 'w');
    fprintf(fid, '# 20N 等密面几何口径对照汇总\n\n');
    fprintf(fid, '| polarity | panel | z geometry | matches | unique Argo | median corr W/1000m | deep reversal score | first zero depth m | q95 W | output |\n');
    fprintf(fid, '|---|---|---|---:|---:|---:|---:|---:|---:|---|\n');
    for ii = 1:size(summary_rows,1)
        fprintf(fid, '| %s | %s | %s | %d | %d | %.3g | %.3g | %.3g | %.3g | `%s` |\n', ...
            summary_rows{ii,1}, summary_rows{ii,2}, summary_rows{ii,3}, summary_rows{ii,4}, summary_rows{ii,5}, ...
            summary_rows{ii,6}, summary_rows{ii,7}, summary_rows{ii,8}, summary_rows{ii,9}, summary_rows{ii,10});
    end
    fprintf(fid, '\nA 是当前 BOA `z''_rho` 相对异常几何；B/C/D 逐步替换为前辈式整体等密面集合、composite-density 热成风和前辈式 term2。\n');
    fclose(fid);
end
