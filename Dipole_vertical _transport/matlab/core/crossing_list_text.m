function text = crossing_list_text(crossing_lats)
    parts = cell(1, numel(crossing_lats));
    for i = 1:numel(crossing_lats)
        parts{i} = lat_token(crossing_lats(i));
    end
    text = strjoin(parts, ',');
end
