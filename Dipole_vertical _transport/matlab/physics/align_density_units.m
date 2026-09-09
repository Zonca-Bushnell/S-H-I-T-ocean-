function profile = align_density_units(profile, rho0)
    med = median(profile, 'omitnan');
    if isfinite(med) && isfinite(rho0)
        if med > 1000 && rho0 < 100
            profile = profile - 1000;
        elseif med < 100 && rho0 > 1000
            profile = profile + 1000;
        end
    end
end
