function rows = build_match_rows_time_blocks(argo_idx, meta_idx, polarity, band_label, ...
    argo_lon, argo_lat, argo_time, argo_park, argo_pf, argo_u, argo_v, argo_wpk, history_match_mask, rho, depth, ...
    meta_lon, meta_lat, meta_time, meta_track, meta_radius, meta_cx, ...
    time_window_days, max_matches_per_group, deg_m)

    rows = preallocate_match_rows(numel(argo_idx));
    row_count = 0;
    if isempty(argo_idx) || isempty(meta_idx)
        rows = rows(1:0,:);
        return
    end

    block_days = 3;
    sub_block_size = 1024;
    [argo_time_sorted, argo_order] = sort(argo_time(argo_idx));
    argo_idx_sorted = argo_idx(argo_order);
    [meta_time_sorted, meta_order] = sort(meta_time(meta_idx));
    meta_idx_sorted = meta_idx(meta_order);
    rho0_sorted = profile_values_at_depth_by_index(depth, rho, argo_idx_sorted, argo_park(argo_idx_sorted));
    t_min = floor(min(argo_time_sorted, [], 'omitnan'));
    t_max = ceil(max(argo_time_sorted, [], 'omitnan'));
    if ~isfinite(t_min) || ~isfinite(t_max)
        rows = rows(1:0,:);
        return
    end

    block_starts = t_min:block_days:t_max;
    for bb = 1:numel(block_starts)
        t0 = block_starts(bb);
        t1 = min(t0 + block_days, t_max + 1);
        a_lo = lower_bound(argo_time_sorted, t0);
        a_hi = lower_bound(argo_time_sorted, t1) - 1;
        m_lo = lower_bound(meta_time_sorted, t0 - time_window_days);
        m_hi = lower_bound(meta_time_sorted, t1 + time_window_days) - 1;
        if a_lo > a_hi || m_lo > m_hi
            continue
        end
        a_block = argo_idx_sorted(a_lo:a_hi);
        rho0_block = rho0_sorted(a_lo:a_hi);
        m_block = meta_idx_sorted(m_lo:m_hi);
        if isempty(a_block) || isempty(m_block)
            continue
        end
        for a0 = 1:sub_block_size:numel(a_block)
            a1 = min(a0 + sub_block_size - 1, numel(a_block));
            a_sub = a_block(a0:a1);
            rho0_sub = rho0_block(a0:a1);

            dt_ok = abs(argo_time(a_sub(:)) - meta_time(m_block(:))') <= time_window_days;
            dlon = argo_lon(a_sub(:)) - meta_lon(m_block(:))';
            dlon(dlon > 180) = dlon(dlon > 180) - 360;
            dlon(dlon < -180) = dlon(dlon < -180) + 360;
            dx = dlon .* deg_m .* cosd(argo_lat(a_sub(:)));
            dy = (argo_lat(a_sub(:)) - meta_lat(m_block(:))') .* deg_m;
            r_norm = hypot(dx, dy) ./ meta_radius(m_block(:))';
            mask = dt_ok & isfinite(r_norm) & r_norm <= 4 & isfinite(rho0_sub);
            if ~any(mask(:))
                continue
            end
            [ai, mi] = find(mask);
            lin = sub2ind(size(mask), ai, mi);
            n_add = numel(ai);
            while row_count + n_add > size(rows, 1)
                rows = grow_match_rows(rows);
            end
            for kk = 1:n_add
                ii = a_sub(ai(kk));
                jj = m_block(mi(kk));
                rr = row_count + kk;
                this_r = r_norm(lin(kk));
                rows(rr,:) = {polarity, band_label, ii, argo_pf(ii), argo_time(ii), argo_lon(ii), argo_lat(ii), ...
                    argo_park(ii), argo_u(ii), argo_v(ii), argo_wpk(ii), rho0_sub(ai(kk)), NaN, NaN, NaN, NaN, NaN, NaN, ...
                    meta_track(jj), meta_time(jj), meta_lon(jj), meta_lat(jj), meta_radius(jj), ...
                    dx(lin(kk)) / meta_radius(jj), dy(lin(kk)) / meta_radius(jj), this_r, ring_label(this_r), meta_cx(jj), history_match_mask(ii), ...
                    NaN, NaN, NaN, false};
            end
            row_count = row_count + n_add;
            if max_matches_per_group > 0 && row_count >= max_matches_per_group
                rows = rows(1:max_matches_per_group,:);
                return
            end
        end
        if max_matches_per_group == 0 && (mod(bb, 500) == 0 || bb == numel(block_starts))
            log_step(sprintf('%s %s match blocks: %d/%d, raw rows so far: %d', polarity, band_label, bb, numel(block_starts), row_count));
        end
    end
    rows = rows(1:row_count,:);
end
