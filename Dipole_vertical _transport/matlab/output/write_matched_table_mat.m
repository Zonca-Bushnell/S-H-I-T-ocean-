function write_matched_table_mat(path, header, matches)
    matched = struct();
    matched.header = header;
    matched.row_count = size(matches, 1);
    matched.note = 'Large matched tables are stored as MAT by default; use --write-matched-csv to opt in to CSV.';
    field_names = matlab.lang.makeValidName(header);
    for c = 1:numel(header)
        if isempty(matches)
            matched.(field_names{c}) = [];
            continue
        end
        col = matches(:, c);
        numeric_col = true;
        for r = 1:numel(col)
            if ~(isnumeric(col{r}) || islogical(col{r})) || ~isscalar(col{r})
                numeric_col = false;
                break
            end
        end
        if numeric_col
            matched.(field_names{c}) = cell2mat(col);
        else
            matched.(field_names{c}) = col;
        end
    end
    save(path, 'matched');
end
