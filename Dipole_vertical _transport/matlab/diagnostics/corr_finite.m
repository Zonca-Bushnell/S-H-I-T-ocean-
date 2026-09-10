function r = corr_finite(a, b)
    good = isfinite(a) & isfinite(b);
    if nnz(good) < 5
        r = NaN;
        return
    end
    C = corrcoef(a(good), b(good));
    r = C(1,2);
end
