function rows = build_match_rows_nearest(argo_idx, meta_idx, polarity, band_label, ...
    argo_lon, argo_lat, argo_time, argo_park, argo_pf, argo_u, argo_v, argo_wpk, history_match_mask, rho, depth, ...
    meta_lon, meta_lat, meta_time, meta_track, meta_radius, meta_cx, ...
    time_window_days, max_matches_per_group, deg_m)

    rows = preallocate_match_rows(numel(argo_idx));
    row_count = 0;
    meta_time_band = meta_time(meta_idx);
    [meta_time_sorted, meta_order] = sort(meta_time_band);
    meta_idx_sorted = meta_idx(meta_order);
    for a = 1:numel(argo_idx)
        ii = argo_idx(a);
        candidate_sorted = time_window_indices(meta_time_sorted, argo_time(ii), time_window_days);
        if isempty(candidate_sorted)
            continue
        end
        candidates = meta_idx_sorted(candidate_sorted);
        dx = local_dx_m(argo_lon(ii), meta_lon(candidates), argo_lat(ii), deg_m);
        dy = (argo_lat(ii) - meta_lat(candidates)) * deg_m;
        r_norm = hypot(dx, dy) ./ meta_radius(candidates);
        rho0 = interp1(depth, double(rho(ii,:)), argo_park(ii), 'linear', NaN);
        if ~isfinite(rho0)
            continue
        end
        [best_r, best_pos] = min(r_norm);
        if ~isfinite(best_r) || best_r > 4
            continue
        end
        jj = candidates(best_pos);
        row_count = row_count + 1;
        if row_count > size(rows, 1)
            rows = grow_match_rows(rows);
        end
        rows(row_count,:) = {polarity, band_label, ii, argo_pf(ii), argo_time(ii), argo_lon(ii), argo_lat(ii), ...
            argo_park(ii), argo_u(ii), argo_v(ii), argo_wpk(ii), rho0, NaN, NaN, NaN, NaN, NaN, NaN, ...
            meta_track(jj), meta_time(jj), meta_lon(jj), meta_lat(jj), meta_radius(jj), ...
            dx(best_pos) / meta_radius(jj), dy(best_pos) / meta_radius(jj), best_r, ring_label(best_r), meta_cx(jj), history_match_mask(ii), ...
            NaN, NaN, NaN, false};
        if max_matches_per_group > 0 && row_count >= max_matches_per_group
            break
        end
    end
    rows = rows(1:row_count,:);
end
