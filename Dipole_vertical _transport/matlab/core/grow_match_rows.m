function rows = grow_match_rows(rows)
    rows(end + size(rows, 1), 33) = {[]};
end
