function write_hybrid_term1_isas_term2_summary_doc(path, header, rows)
    fid = fopen(path, 'w');
    if fid < 0
        warning('Could not write %s', path);
        return
    end
    cleaner = onCleanup(@() fclose(fid));
    fprintf(fid, '# 20N/30N/40N Argo absolute term1 + ISAS term2 汇总\n\n');
    fprintf(fid, '本轮只检查 crossing 纬线 `20N, 30N, 40N`，cyclonic 和 anticyclonic 分开，不生成 combined。\n\n');
    fprintf(fid, '核心口径：`term1` 使用 Argo 合成 absolute density 的整体等密面斜率；热成风速度以 1000 m Argo parking drift 为锚点，并用 Argo 合成 absolute density 的水平密度梯度积分；`term2` 使用 ISAS/background density 的整体等密面斜率。`D_rho` 正深度向下，`W` 向上为正。\n\n');
    if isempty(rows)
        fprintf(fid, '没有生成有效行。\n');
        return
    end
    fprintf(fid, '| %s |\n', strjoin(header, ' | '));
    fprintf(fid, '|%s|\n', strjoin(repmat({'---'}, 1, numel(header)), '|'));
    for r = 1:size(rows,1)
        vals = cell(1, numel(header));
        for c = 1:numel(header)
            item = rows{r,c};
            if isnumeric(item) || islogical(item)
                vals{c} = sprintf('%.6g', double(item));
            else
                vals{c} = char(string(item));
            end
        end
        fprintf(fid, '| %s |\n', strjoin(vals, ' | '));
    end
end
