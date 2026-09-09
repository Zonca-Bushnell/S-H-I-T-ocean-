function [Z, support_count] = cressman_map_multi_missing_gpu(x, y, V, X, Y, radius_r, min_obs)
    n_grid = numel(X);
    n_var = size(V, 2);
    Z = NaN([size(X), n_var]);
    support_count = zeros([size(X), n_var]);
    xg = gpuArray(single(x(:)));
    yg = gpuArray(single(y(:)));
    valid_cpu = isfinite(V);
    V0 = single(V);
    V0(~valid_cpu) = 0;
    Vg = gpuArray(V0);
    valid_g = gpuArray(single(valid_cpu));
    Xv = single(X(:)');
    Yv = single(Y(:)');
    r2_limit = single(radius_r ^ 2);
    grid_chunk = 1024;
    var_chunk = 64;
    for start_idx = 1:grid_chunk:n_grid
        stop_idx = min(n_grid, start_idx + grid_chunk - 1);
        cols = start_idx:stop_idx;
        Xg = gpuArray(Xv(cols));
        Yg = gpuArray(Yv(cols));
        d2 = (xg - Xg) .^ 2 + (yg - Yg) .^ 2;
        inside = d2 < r2_limit;
        W = (r2_limit - d2) ./ (r2_limit + d2);
        W(~inside) = 0;
        Wt = W';
        inside_t = single(inside');
        for var0 = 1:var_chunk:n_var
            var1 = min(n_var, var0 + var_chunk - 1);
            vars = var0:var1;
            valid_chunk = valid_g(:,vars);
            counts = inside_t * valid_chunk;
            denom = Wt * valid_chunk;
            numerator = Wt * Vg(:,vars);
            mapped = numerator ./ denom;
            counts_cpu = double(gather(counts));
            denom_cpu = double(gather(denom));
            mapped_cpu = double(gather(mapped));
            ok = counts_cpu >= min_obs & denom_cpu > 0;
            block = nan(numel(cols), numel(vars));
            block(ok) = mapped_cpu(ok);
            linear_idx = cols(:) + (vars - 1) * n_grid;
            Z(linear_idx) = block;
            support_count(linear_idx) = counts_cpu;
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
