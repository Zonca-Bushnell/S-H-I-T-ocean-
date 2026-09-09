function [Z, support_count] = cressman_map_multi_gpu(x, y, V, X, Y, radius_r, min_obs)
    Z = NaN([size(X), size(V, 2)]);
    support_count = zeros(size(X));
    xg = gpuArray(single(x(:)));
    yg = gpuArray(single(y(:)));
    Vg = gpuArray(single(V));
    Xv = single(X(:)');
    Yv = single(Y(:)');
    r2_limit = single(radius_r ^ 2);
    n_grid = numel(X);
    n_var = size(V, 2);
    chunk = 2048;
    for start_idx = 1:chunk:n_grid
        stop_idx = min(n_grid, start_idx + chunk - 1);
        cols = start_idx:stop_idx;
        Xg = gpuArray(Xv(cols));
        Yg = gpuArray(Yv(cols));
        d2 = (xg - Xg) .^ 2 + (yg - Yg) .^ 2;
        inside = d2 < r2_limit;
        support = sum(inside, 1);
        W = (r2_limit - d2) ./ (r2_limit + d2);
        W(~inside) = 0;
        denom = sum(W, 1);
        support_cpu = gather(support);
        support_count(cols) = double(support_cpu);
        ok_cols = support_cpu >= min_obs & gather(denom) > 0;
        if any(ok_cols)
            for kk = 1:n_var
                val = Vg(:,kk);
                mapped = sum(W .* val, 1) ./ denom;
                mapped_cpu = double(gather(mapped));
                out = nan(1, numel(cols));
                out(ok_cols) = mapped_cpu(ok_cols);
                Z(cols + (kk-1) * n_grid) = out;
            end
        end
    end
    outside = hypot(X, Y) > 4;
    for kk = 1:n_var
        tmp = Z(:,:,kk);
        tmp(outside) = NaN;
        Z(:,:,kk) = tmp;
    end
    support_count(outside) = 0;
end
