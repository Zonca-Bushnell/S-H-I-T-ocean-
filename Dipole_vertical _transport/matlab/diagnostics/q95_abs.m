function q = q95_abs(values)
    values = abs(values(isfinite(values)));
    if isempty(values)
        q = NaN;
    else
        q = quantile(values, 0.95);
    end
end
