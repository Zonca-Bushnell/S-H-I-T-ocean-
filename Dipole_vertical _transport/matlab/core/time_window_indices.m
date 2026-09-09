function idx = time_window_indices(sorted_time, t0, window_days)
    if ~isfinite(t0) || isempty(sorted_time)
        idx = [];
        return
    end
    lo = lower_bound(sorted_time, t0 - window_days);
    hi = upper_bound(sorted_time, t0 + window_days) - 1;
    if lo > hi
        idx = [];
    else
        idx = lo:hi;
    end
end
