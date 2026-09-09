function [Z, support_count] = cressman_map_multi_missing(x, y, V, X, Y, radius_r, min_obs)
    n_var = size(V, 2);
    Z = NaN([size(X), n_var]);
    support_count = zeros([size(X), n_var]);
    good_xy = isfinite(x) & isfinite(y) & any(isfinite(V), 2) & hypot(x, y) <= 4;
    x = x(good_xy);
    y = y(good_xy);
    V = V(good_xy,:);
    if isempty(x) || ~isfinite(radius_r) || radius_r <= 0
        return
    end
    global USE_GPU_CRESSMAN;
    if USE_GPU_CRESSMAN
        try
            [Z, support_count] = cressman_map_multi_missing_gpu(x, y, V, X, Y, radius_r, min_obs);
            return
        catch ME
            warning('GPU missing-aware Cressman failed; falling back to CPU for this map: %s', ME.message);
        end
    end
    r2_limit = radius_r ^ 2;
    finite_v = isfinite(V);
    V0 = V;
    V0(~finite_v) = 0;
    for ii = 1:numel(X)
        d2 = (x - X(ii)).^2 + (y - Y(ii)).^2;
        inside = d2 < r2_limit;
        if ~any(inside)
            continue
        end
        w = (r2_limit - d2(inside)) ./ (r2_limit + d2(inside));
        ok_w = isfinite(w) & w > 0;
        if ~any(ok_w)
            continue
        end
        w = w(ok_w);
        finite_inside = finite_v(inside,:);
        finite_inside = finite_inside(ok_w,:);
        vals = V0(inside,:);
        vals = vals(ok_w,:);
        counts = sum(finite_inside, 1);
        denom = sum(w .* finite_inside, 1);
        mapped = sum(w .* vals, 1) ./ denom;
        ok = counts >= min_obs & denom > 0;
        support_count(ii + (0:n_var-1) * numel(X)) = counts;
        if any(ok)
            Z(ii + find(ok) * numel(X) - numel(X)) = mapped(ok);
        end
    end
    outside = hypot(X, Y) > 4;
    for kk = 1:n_var
        tmp = Z(:,:,kk);
        tmp(outside) = NaN;
        Z(:,:,kk) = tmp;
        tmp_count = support_count(:,:,kk);
        tmp_count(outside) = 0;
        support_count(:,:,kk) = tmp_count;
    end
end
