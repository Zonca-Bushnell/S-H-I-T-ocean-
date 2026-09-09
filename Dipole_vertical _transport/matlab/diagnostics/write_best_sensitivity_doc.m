function write_best_sensitivity_doc(path, summary_rows)
    fid = fopen(path, 'w');
    fprintf(fid, '# 2D 20N W 参数敏感度推荐\n\n');
    fprintf(fid, '本轮使用缓存后的 20N crossing 样本表，只重做 Cressman、平滑、梯度和出图，用于快速判断碎片/锯齿来源。\n\n');
    if isempty(summary_rows)
        fprintf(fid, '没有可用结果。\n');
        fclose(fid);
        return
    end
    names = unique(summary_rows(:,2), 'stable');
    best_name = '';
    best_score = Inf;
    for i = 1:numel(names)
        name = names{i};
        rows = summary_rows(strcmp(summary_rows(:,2), name), :);
        rough = cell2mat(rows(:,15));
        corrv = abs(cell2mat(rows(:,12)));
        valid = cell2mat(rows(:,9));
        score = mean(rough, 'omitnan') ./ max(mean(valid, 'omitnan'), eps) - 0.1 * mean(corrv, 'omitnan');
        if isfinite(score) && score < best_score
            best_score = score;
            best_name = name;
        end
    end
    fprintf(fid, '- 自动推荐组合：`%s`。\n', best_name);
    fprintf(fid, '- 推荐依据：优先降低 `roughness_score`，同时保留有效覆盖并避免 `corr(rebuild_W,I_Wpk)` 明显恶化。\n');
    fprintf(fid, '- 最终仍需人工看 `cyclonic_sensitivity_montage.png` 和 `anticyclonic_sensitivity_montage.png`，确认是否保留中心东西偶极。\n\n');
    fprintf(fid, '| polarity | config | valid grid %% | support median | support p10 | corr W/Wpk | q95 W | roughness | dipole score |\n');
    fprintf(fid, '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n');
    for i=1:size(summary_rows,1)
        fprintf(fid, '| %s | %s | %.2f | %.1f | %.1f | %.3g | %.3g | %.3g | %.3g |\n', ...
            summary_rows{i,1}, summary_rows{i,2}, summary_rows{i,9} * 100, summary_rows{i,10}, summary_rows{i,11}, ...
            summary_rows{i,12}, summary_rows{i,13}, summary_rows{i,15}, summary_rows{i,16});
    end
    fclose(fid);
end
