function idx = upper_bound(values, target)
    lo = 1;
    hi = numel(values) + 1;
    while lo < hi
        mid = floor((lo + hi) / 2);
        if values(mid) <= target
            lo = mid + 1;
        else
            hi = mid;
        end
    end
    idx = lo;
end
