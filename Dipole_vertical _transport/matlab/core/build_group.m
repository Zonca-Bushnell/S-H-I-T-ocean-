function [matches, grid] = build_group(argo_idx, meta_idx, polarity, band_label, ...
    argo_lon, argo_lat, argo_time, argo_park, argo_pf, argo_u, argo_v, argo_wpk, history_match_mask, rho, depth, ...
    meta_lon, meta_lat, meta_time, meta_track, meta_radius, meta_cx, ...
    time_window_days, grid_n, min_bin_count, plot_filled_gradient, smooth_passes, cressman_radius_r, cressman_min_obs, sample_gradient_max_profiles, ...
    boa_clim, z_rho_min_m, z_rho_max_m, min_drho_dz, max_rho_bracket_dz_m, match_mode, max_matches_per_group, deg_m)

    if strcmp(match_mode, 'all')
        matches = build_group_matches_only(argo_idx, meta_idx, polarity, band_label, ...
            argo_lon, argo_lat, argo_time, argo_park, argo_pf, argo_u, argo_v, argo_wpk, history_match_mask, rho, depth, ...
            meta_lon, meta_lat, meta_time, meta_track, meta_radius, meta_cx, ...
            time_window_days, boa_clim, z_rho_min_m, z_rho_max_m, min_drho_dz, max_rho_bracket_dz_m, match_mode, max_matches_per_group, deg_m);
        grid = composite_grid(matches, grid_n, min_bin_count, plot_filled_gradient, smooth_passes, cressman_radius_r, cressman_min_obs, sample_gradient_max_profiles);
        grid.match_mode = match_mode;
        grid.z_mode = 'anomaly_boa_climatology';
        grid.z_rho_min_m = z_rho_min_m;
        grid.z_rho_max_m = z_rho_max_m;
        grid.min_drho_dz = min_drho_dz;
        grid.max_rho_bracket_dz_m = max_rho_bracket_dz_m;
        return
    end

    rows = {};
    row_count = 0;
    meta_time_band = meta_time(meta_idx);
    for a = 1:numel(argo_idx)
        ii = argo_idx(a);
        candidate_local = find(abs(meta_time_band - argo_time(ii)) <= time_window_days);
        if isempty(candidate_local)
            continue
        end
        candidates = meta_idx(candidate_local);
        dx = local_dx_m(argo_lon(ii), meta_lon(candidates), argo_lat(ii), deg_m);
        dy = (argo_lat(ii) - meta_lat(candidates)) * deg_m;
        r_norm = hypot(dx, dy) ./ meta_radius(candidates);
        rho0 = interp1(depth, double(rho(ii,:)), argo_park(ii), 'linear', NaN);
        if ~isfinite(rho0)
            continue
        end
        if strcmp(match_mode, 'all')
            use_pos = find(isfinite(r_norm) & r_norm <= 4);
        else
            [best_r, best_pos] = min(r_norm);
            if isfinite(best_r) && best_r <= 4
                use_pos = best_pos;
            else
                use_pos = [];
            end
        end
        for pp = 1:numel(use_pos)
            pos = use_pos(pp);
            jj = candidates(pos);
            best_dx = dx(pos);
            best_dy = dy(pos);
            this_r = r_norm(pos);
            row_count = row_count + 1;
            ring = ring_label(this_r);
            rows(row_count,:) = {polarity, band_label, ii, argo_pf(ii), argo_time(ii), argo_lon(ii), argo_lat(ii), ...
                argo_park(ii), argo_u(ii), argo_v(ii), argo_wpk(ii), rho0, NaN, NaN, NaN, NaN, NaN, NaN, ...
                meta_track(jj), meta_time(jj), meta_lon(jj), meta_lat(jj), meta_radius(jj), ...
                best_dx / meta_radius(jj), best_dy / meta_radius(jj), this_r, ring, meta_cx(jj), history_match_mask(ii), ...
                NaN, NaN, NaN, false}; %#ok<AGROW>
            if max_matches_per_group > 0 && row_count >= max_matches_per_group
                break
            end
        end
        if max_matches_per_group > 0 && row_count >= max_matches_per_group
            break
        end
    end
    rows = apply_rho0_mode(rows, rho, depth, argo_park, boa_clim, z_rho_min_m, z_rho_max_m, min_drho_dz, max_rho_bracket_dz_m);
    matches = rows;
    grid = composite_grid(matches, grid_n, min_bin_count, plot_filled_gradient, smooth_passes, cressman_radius_r, cressman_min_obs, sample_gradient_max_profiles);
    grid.match_mode = match_mode;
    grid.z_mode = 'anomaly_boa_climatology';
    grid.z_rho_min_m = z_rho_min_m;
    grid.z_rho_max_m = z_rho_max_m;
    grid.min_drho_dz = min_drho_dz;
    grid.max_rho_bracket_dz_m = max_rho_bracket_dz_m;
end
