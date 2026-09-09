function z = interp_density_to_depth(profile, z_axis, rho_targets)
    z = nan(size(rho_targets));
    good = isfinite(profile) & isfinite(z_axis);
    if nnz(good) < 3
        return
    end
    p = profile(good);
    z_good = z_axis(good);
    [p, ia] = unique(p, 'stable');
    z_good = z_good(ia);
    if numel(p) < 3
        return
    end
    z = interp1(p, z_good, rho_targets, 'linear', NaN);
end
