function c = spatial_corr(A, B)
    a = A(:);
    b = B(:);
    good = isfinite(a) & isfinite(b);
    if nnz(good) < 3
        c = NaN;
        return
    end
    R = corrcoef(a(good), b(good));
    c = R(1,2);
end
