function [term1, term2, rebuild] = sample_gradient_terms(x, y, z, u, v, cx_raw, radius, cx_rel, radius_r, min_obs, max_profiles)
    term1 = nan(size(z));
    term2 = nan(size(z));
    rebuild = nan(size(z));
    good_all = isfinite(x) & isfinite(y) & isfinite(z) & isfinite(u) & isfinite(v) & isfinite(cx_raw) & isfinite(radius) & radius > 0 & hypot(x, y) <= 4;
    if nnz(good_all) < min_obs || ~isfinite(radius_r) || radius_r <= 0
        return
    end
    eval_idx = find(good_all);
    if isfinite(max_profiles) && max_profiles > 0 && numel(eval_idx) > max_profiles
        eval_idx = eval_idx(unique(round(linspace(1, numel(eval_idx), max_profiles))));
    end
    for kk = 1:numel(eval_idx)
        ii = eval_idx(kk);
        if ~good_all(ii)
            continue
        end
        d2 = (x - x(ii)).^2 + (y - y(ii)).^2;
        inside = good_all & d2 < radius_r ^ 2;
        if nnz(inside) < max(min_obs, 6)
            continue
        end
        Xfit = [ones(nnz(inside), 1), x(inside) - x(ii), y(inside) - y(ii)];
        coef = Xfit \ z(inside);
        dzdx = coef(2) / radius(ii);
        dzdy = coef(3) / radius(ii);
        term1(ii) = cx_rel * dzdx;
        term2(ii) = -((u(ii) - cx_raw(ii)) * dzdx + v(ii) * dzdy);
        rebuild(ii) = term1(ii) + term2(ii);
    end
end
