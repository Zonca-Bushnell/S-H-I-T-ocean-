function [z, crossing_count, bracket_dz, local_drho_dz] = isopycnal_depth_qc(depth, profile, rho0, target_depth)
    z = NaN;
    crossing_count = 0;
    bracket_dz = NaN;
    local_drho_dz = NaN;
    depth = depth(:);
    profile = profile(:);
    good = isfinite(depth) & isfinite(profile);
    depth = depth(good);
    profile = profile(good);
    if numel(depth) < 3 || ~isfinite(rho0) || ~isfinite(target_depth)
        return
    end
    [depth, order] = sort(depth);
    profile = profile(order);
    [depth, ia] = unique(depth, 'stable');
    profile = profile(ia);
    hits = nan(0, 4);
    for k = 1:numel(depth)-1
        r1 = profile(k) - rho0;
        r2 = profile(k+1) - rho0;
        if r1 == 0 && depth(k) > 0
            dz = abs(depth(k+1) - depth(k));
            drhodz = (profile(k+1) - profile(k)) / (depth(k+1) - depth(k));
            hits(end+1,:) = [depth(k), dz, drhodz, k]; %#ok<AGROW>
            continue
        elseif r2 == 0 && depth(k+1) > 0
            dz = abs(depth(k+1) - depth(k));
            drhodz = (profile(k+1) - profile(k)) / (depth(k+1) - depth(k));
            hits(end+1,:) = [depth(k+1), dz, drhodz, k]; %#ok<AGROW>
            continue
        end
        if r1 * r2 > 0 || profile(k) == profile(k+1)
            continue
        end
        frac = (rho0 - profile(k)) / (profile(k+1) - profile(k));
        z_hit = depth(k) + frac * (depth(k+1) - depth(k));
        dz = abs(depth(k+1) - depth(k));
        drhodz = (profile(k+1) - profile(k)) / (depth(k+1) - depth(k));
        hits(end+1,:) = [z_hit, dz, drhodz, k]; %#ok<AGROW>
    end
    crossing_count = size(hits, 1);
    if ~isempty(hits)
        [~, idx] = min(abs(hits(:,1) - target_depth));
        z = hits(idx,1);
        bracket_dz = hits(idx,2);
        local_drho_dz = hits(idx,3);
    end
end
