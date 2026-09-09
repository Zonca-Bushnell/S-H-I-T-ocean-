function write_summary_table_mat(path, header, rows)
    summary = struct();
    summary.header = header;
    summary.row_count = size(rows, 1);
    field_names = matlab.lang.makeValidName(header);
    for c = 1:numel(header)
        if isempty(rows) || c > size(rows, 2)
            summary.(field_names{c}) = [];
            continue
        end
        col = rows(:, c);
        numeric_col = true;
        for r = 1:numel(col)
            if ~(isnumeric(col{r}) || islogical(col{r})) || ~isscalar(col{r})
                numeric_col = false;
                break
            end
        end
        if numeric_col
            summary.(field_names{c}) = cell2mat(col);
        else
            summary.(field_names{c}) = col;
        end
    end
    save(path, 'summary');
end
