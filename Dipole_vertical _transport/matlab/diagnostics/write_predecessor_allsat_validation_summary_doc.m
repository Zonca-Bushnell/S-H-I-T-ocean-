function write_predecessor_allsat_validation_summary_doc(path, summary_header, summary_rows)
    lines = {};
    lines{end+1} = '# META3.2 allsat 前辈算法验证汇总';
    lines{end+1} = '';
    lines{end+1} = '## 任务定义';
    lines{end+1} = '';
    lines{end+1} = '- 目标：判断前辈算法口径在 `META3.2 allsat` 替代缺失 `twosat` 轨迹后，是否仍能产生前辈式深层反转和较规整 W 剖面。';
    lines{end+1} = '- 注意：这不是原始 `twosat` 输入的完全复现，而是 allsat 替代验证。';
    lines{end+1} = '- 极性：cyclonic 与 anticyclonic 分开；不生成 combined。';
    lines{end+1} = '';
    lines{end+1} = '## 汇总表';
    lines{end+1} = '';
    lines{end+1} = strjoin(summary_header, ' | ');
    lines{end+1} = strjoin(repmat({'---'}, 1, numel(summary_header)), ' | ');
    for rr = 1:size(summary_rows, 1)
        parts = cell(1, numel(summary_header));
        for cc = 1:numel(summary_header)
            val = summary_rows{rr, cc};
            if islogical(val)
                parts{cc} = char(string(val));
            elseif isnumeric(val)
                if isscalar(val)
                    parts{cc} = sprintf('%.6g', val);
                else
                    parts{cc} = mat2str(val);
                end
            else
                parts{cc} = char(string(val));
            end
        end
        lines{end+1} = strjoin(parts, ' | ');
    end
    lines{end+1} = '';
    lines{end+1} = '## 判读规则';
    lines{end+1} = '';
    lines{end+1} = '- 若 `deep_reversal_score > 0` 且深层 W 与 1000 m 结构相关显著转负，说明出现深层反转。';
    lines{end+1} = '- 若 term2 的深层 q95 明显接近或超过 term1，说明 ISAS 背景等密面坡度对深层结构贡献不可忽略。';
    lines{end+1} = '- 若 anticyclonic 的 ISAS 文件路径不是对应 AE 文件，必须把它视为背景文件不足下的敏感性结果。';
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
