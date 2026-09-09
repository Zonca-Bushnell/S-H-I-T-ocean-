function [Z, support_count] = cressman_map_multi(x, y, V, X, Y, radius_r, min_obs)
    Z = NaN([size(X), size(V, 2)]);
    support_count = zeros(size(X));
    good = isfinite(x) & isfinite(y) & all(isfinite(V), 2) & hypot(x, y) <= 4;
    x = x(good);
    y = y(good);
    V = V(good,:);
    if isempty(x) || ~isfinite(radius_r) || radius_r <= 0
        return
    end
    global USE_GPU_CRESSMAN;
    if USE_GPU_CRESSMAN
        try
            [Z, support_count] = cressman_map_multi_gpu(x, y, V, X, Y, radius_r, min_obs);
            return
        catch ME
            warning('GPU Cressman failed; falling back to CPU for this map: %s', ME.message);
        end
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
                vals = V(inside,:);
                vals = vals(ok,:);
                w = w(ok);
                for kk = 1:size(V, 2)
                    Z(ii + (kk-1) * numel(X)) = sum(w .* vals(:,kk)) ./ sum(w);
                end
            end
        end
    end
    outside = hypot(X, Y) > 4;
    for kk = 1:size(V, 2)
        tmp = Z(:,:,kk);
        tmp(outside) = NaN;
        Z(:,:,kk) = tmp;
    end
    support_count(outside) = 0;
end
