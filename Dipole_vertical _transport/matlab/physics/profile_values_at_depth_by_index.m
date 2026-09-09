function values = profile_values_at_depth_by_index(depth, rho, profile_idx, target_depth)
    target_depth = target_depth(:);
    values = nan(numel(target_depth), 1);
    for kk = 1:numel(depth)-1
        z1 = depth(kk);
        z2 = depth(kk+1);
        if z2 == z1
            continue
        end
        if kk == numel(depth)-1
            mask = target_depth >= z1 & target_depth <= z2;
        else
            mask = target_depth >= z1 & target_depth < z2;
        end
        if ~any(mask)
            continue
        end
        w = (target_depth(mask) - z1) ./ (z2 - z1);
        rows = profile_idx(mask);
        v1 = double(rho(rows, kk));
        v2 = double(rho(rows, kk+1));
        values(mask) = v1 .* (1 - w) + v2 .* w;
    end
end
