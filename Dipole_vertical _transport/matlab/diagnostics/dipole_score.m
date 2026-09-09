function score = dipole_score(W, X, Y)
    mask = hypot(X, Y) <= 2 & isfinite(W);
    ideal = X;
    score = spatial_corr(W(mask), ideal(mask));
end
