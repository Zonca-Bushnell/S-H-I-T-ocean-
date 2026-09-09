function [Z, support_count] = cressman_map(x, y, v, X, Y, radius_r, min_obs)
    Z = NaN(size(X));
    support_count = zeros(size(X));
    good = isfinite(x) & isfinite(y) & isfinite(v) & hypot(x, y) <= 4;
    x = x(good); y = y(good); v = v(good);
    if isempty(x) || ~isfinite(radius_r) || radius_r <= 0
        return
    end
    r2_limit = radius_r ^ 2;
    for ii = 1:numel(X)
        d2 = (x - X(ii)).^2 + (y - Y(ii)).^2;
        inside = d2 < r2_limit;
        n_inside = nnz(inside);
        support_count(ii) = n_inside;
        if n_inside >= min_obs
            w = (r2_limit - d2(inside)) ./ (r2_limit + d2(inside));
            ok = isfinite(w) & w > 0;
            if any(ok)
                vals = v(inside);
                vals = vals(ok);
                w = w(ok);
                Z(ii) = sum(w .* vals) ./ sum(w);
            end
        end
    end
    Z(hypot(X, Y) > 4) = NaN;
    support_count(hypot(X, Y) > 4) = 0;
end
