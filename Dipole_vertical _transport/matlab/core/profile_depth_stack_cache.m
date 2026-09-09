function profile_cache = profile_depth_stack_cache(matches, rho, depth, boa_clim, depth_levels, min_drho_dz, max_rho_bracket_dz_m, cache_root, polarity, band_label, match_mode)
    argo_indices_all = cell2mat(matches(:,3));
    [argo_indices, first_pos] = unique(argo_indices_all(:), 'stable');
    lon = cell2mat(matches(first_pos,6));
    lat = cell2mat(matches(first_pos,7));
    time = cell2mat(matches(first_pos,5));
    n_unique = numel(argo_indices);
    nz = numel(depth_levels);
    depth_tag = sprintf('%gm_%gm_%03dlev', min(depth_levels), max(depth_levels), nz);
    sample_hash = mod(sum(double(argo_indices(:))) + 1000003 * n_unique, 2147483647);
    sample_tag = sprintf('n%d_h%d', n_unique, sample_hash);
    cache_file = fullfile(cache_root, ['boa_profile_qc_' sanitize_filename(band_label) '_' sanitize_filename(match_mode) '_' depth_tag '_' sample_tag '.mat']);
    key = matlab.lang.makeValidName([polarity '_' band_label '_' match_mode '_' depth_tag '_' sample_tag]);
    if exist(cache_file, 'file') == 2
        C = load(cache_file, 'profile_cache_store');
        if isfield(C, 'profile_cache_store') && isfield(C.profile_cache_store, key)
            cached = C.profile_cache_store.(key);
            if isequal(cached.argo_indices(:), argo_indices(:)) && isequal(cached.depth_levels(:), depth_levels(:)) && isfield(cached, 'rho_abs')
                profile_cache = cached;
                log_step(sprintf('3D profile BOA/QC cache hit: %s [%s]', cache_file, key));
                return
            end
        end
    end
    log_step(sprintf('3D profile BOA/QC cache build: %d unique Argo x %d depths', n_unique, nz));
    z_anom = nan(n_unique, nz);
    rho_anom = nan(n_unique, nz);
    rho_abs = nan(n_unique, nz);
    boa_rho = nan(n_unique, nz);
    profile_valid = false(n_unique, nz);
    boa_valid = false(n_unique, nz);
    global QC_WORKERS;
    use_parallel = maybe_start_parallel_pool(QC_WORKERS);
    if use_parallel
        parfor uu = 1:n_unique
            [z_row, rho_row, rho_abs_row, boa_rho_row, pv_row, bv_row] = profile_depth_stack_one(argo_indices(uu), lon(uu), lat(uu), time(uu), rho, depth, boa_clim, depth_levels, min_drho_dz, max_rho_bracket_dz_m);
            z_anom(uu,:) = z_row;
            rho_anom(uu,:) = rho_row;
            rho_abs(uu,:) = rho_abs_row;
            boa_rho(uu,:) = boa_rho_row;
            profile_valid(uu,:) = pv_row;
            boa_valid(uu,:) = bv_row;
        end
    else
        for uu = 1:n_unique
            [z_row, rho_row, rho_abs_row, boa_rho_row, pv_row, bv_row] = profile_depth_stack_one(argo_indices(uu), lon(uu), lat(uu), time(uu), rho, depth, boa_clim, depth_levels, min_drho_dz, max_rho_bracket_dz_m);
            z_anom(uu,:) = z_row;
            rho_anom(uu,:) = rho_row;
            rho_abs(uu,:) = rho_abs_row;
            boa_rho(uu,:) = boa_rho_row;
            profile_valid(uu,:) = pv_row;
            boa_valid(uu,:) = bv_row;
        end
    end
    profile_cache = struct('argo_indices', argo_indices(:), 'depth_levels', depth_levels(:), ...
        'z_anom', z_anom, 'rho_anom', rho_anom, 'rho_abs', rho_abs, 'boa_rho', boa_rho, 'profile_valid', profile_valid, 'boa_valid', boa_valid);
    if exist(cache_file, 'file') == 2
        C = load(cache_file, 'profile_cache_store');
        if isfield(C, 'profile_cache_store')
            profile_cache_store = C.profile_cache_store; %#ok<NASGU>
        else
            profile_cache_store = struct(); %#ok<NASGU>
        end
    else
        profile_cache_store = struct(); %#ok<NASGU>
    end
    profile_cache_store.(key) = profile_cache; %#ok<STRNU>
    save(cache_file, 'profile_cache_store', '-v7.3');
    log_step(sprintf('3D profile BOA/QC cache saved: %s [%s]', cache_file, key));
end
