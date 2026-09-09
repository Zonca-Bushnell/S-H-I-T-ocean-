function profile = monotonic_density_profile(profile)
    profile = profile(:);
    for kk = 2:numel(profile)
        if isfinite(profile(kk-1)) && isfinite(profile(kk)) && profile(kk) <= profile(kk-1)
            profile(kk) = profile(kk-1) + max(abs(profile(kk-1)) * 1e-9, 1e-6);
        end
    end
end
