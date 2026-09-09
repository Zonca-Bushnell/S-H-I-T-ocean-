function rows = preallocate_match_rows(n_argo)
    n_rows = max(1024, min(max(1, n_argo) * 2, 200000));
    rows = cell(n_rows, 33);
end
