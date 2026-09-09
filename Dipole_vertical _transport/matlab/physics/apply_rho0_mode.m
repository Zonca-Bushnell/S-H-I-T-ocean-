function rows = apply_rho0_mode(rows, rho, depth, argo_park, boa_clim, z_rho_min_m, z_rho_max_m, min_drho_dz, max_rho_bracket_dz_m)
    if isempty(rows)
        return
    end
    profile_rho0 = cell2mat(rows(:,12));
    target_rho0 = profile_rho0;
    keep = false(size(rows, 1), 1);
    argo_ids = cell2mat(rows(:,3));
    [unique_argo_ids, first_row, row_to_unique] = unique(argo_ids);
    log_step(sprintf('profile rho QC start: %d raw rows, %d unique Argo', size(rows, 1), numel(unique_argo_ids)));
    profile_timer = tic;
    unique_z_rho = nan(numel(unique_argo_ids), 1);
    unique_crossing_count = nan(numel(unique_argo_ids), 1);
    unique_bracket_dz = nan(numel(unique_argo_ids), 1);
    unique_drho_dz = nan(numel(unique_argo_ids), 1);
    unique_ok = false(numel(unique_argo_ids), 1);
    unique_rho0 = nan(numel(unique_argo_ids), 1);
    global QC_WORKERS;
    use_qc_parallel = maybe_start_parallel_pool(QC_WORKERS);
    if use_qc_parallel
        parfor uu = 1:numel(unique_argo_ids)
            rr0 = first_row(uu);
            ii = unique_argo_ids(uu);
            this_rho0 = target_rho0(rr0);
            [z_rho, crossing_count, bracket_dz, local_drho_dz] = isopycnal_depth_qc(depth, double(rho(ii,:)), this_rho0, argo_park(ii));
            unique_rho0(uu) = this_rho0;
            unique_z_rho(uu) = z_rho;
            unique_crossing_count(uu) = crossing_count;
            unique_bracket_dz(uu) = bracket_dz;
            unique_drho_dz(uu) = local_drho_dz;
            unique_ok(uu) = isfinite(z_rho) && z_rho >= z_rho_min_m && z_rho <= z_rho_max_m && ...
                isfinite(bracket_dz) && bracket_dz <= max_rho_bracket_dz_m && ...
                isfinite(local_drho_dz) && abs(local_drho_dz) >= min_drho_dz;
        end
    else
        for uu = 1:numel(unique_argo_ids)
            rr0 = first_row(uu);
            ii = unique_argo_ids(uu);
            unique_rho0(uu) = target_rho0(rr0);
            [z_rho, crossing_count, bracket_dz, local_drho_dz] = isopycnal_depth_qc(depth, double(rho(ii,:)), unique_rho0(uu), argo_park(ii));
            unique_z_rho(uu) = z_rho;
            unique_crossing_count(uu) = crossing_count;
            unique_bracket_dz(uu) = bracket_dz;
            unique_drho_dz(uu) = local_drho_dz;
            unique_ok(uu) = isfinite(z_rho) && z_rho >= z_rho_min_m && z_rho <= z_rho_max_m && ...
                isfinite(bracket_dz) && bracket_dz <= max_rho_bracket_dz_m && ...
                isfinite(local_drho_dz) && abs(local_drho_dz) >= min_drho_dz;
        end
    end
    log_step(sprintf('profile rho QC done: %d unique kept, %.1f s', nnz(unique_ok), toc(profile_timer)));
    for rr = 1:size(rows, 1)
        uu = row_to_unique(rr);
        if unique_ok(uu)
            rows{rr,12} = unique_rho0(uu);
            rows{rr,13} = unique_z_rho(uu);
            rows{rr,16} = unique_crossing_count(uu);
            rows{rr,17} = unique_bracket_dz(uu);
            rows{rr,18} = unique_drho_dz(uu);
            keep(rr) = true;
        end
    end
    rows = rows(keep,:);
    if isempty(rows)
        return
    end
    z_rho = cell2mat(rows(:,13));
    r_norm = cell2mat(rows(:,26));
    z_bg_all = nan(size(z_rho));
    z_anom = nan(size(z_rho));
    keep_boa = false(size(rows, 1), 1);
    argo_ids = cell2mat(rows(:,3));
    [unique_argo_ids, first_row, row_to_unique] = unique(argo_ids);
    unique_z_bg = nan(numel(unique_argo_ids), 1);
    unique_bg_crossing_count = nan(numel(unique_argo_ids), 1);
    unique_bg_bracket_dz = nan(numel(unique_argo_ids), 1);
    unique_bg_drho_dz = nan(numel(unique_argo_ids), 1);
    unique_bg_ok = false(numel(unique_argo_ids), 1);
    log_step(sprintf('BOA rho QC start: %d unique Argo', numel(unique_argo_ids)));
    boa_timer = tic;
    if use_qc_parallel
        parfor uu = 1:numel(unique_argo_ids)
            rr0 = first_row(uu);
            [~, month_id, ~] = datevec(rows{rr0,5});
            boa_profile = boa_density_profile_at(boa_clim, rows{rr0,6}, rows{rr0,7}, month_id);
            boa_profile = align_density_units(boa_profile, rows{rr0,12});
            [z_bg, bg_crossing_count, bg_bracket_dz, bg_drho_dz] = isopycnal_depth_qc(boa_clim.pres, boa_profile, rows{rr0,12}, rows{rr0,8});
            unique_z_bg(uu) = z_bg;
            unique_bg_crossing_count(uu) = bg_crossing_count;
            unique_bg_bracket_dz(uu) = bg_bracket_dz;
            unique_bg_drho_dz(uu) = bg_drho_dz;
            unique_bg_ok(uu) = isfinite(z_bg) && z_bg >= z_rho_min_m && z_bg <= z_rho_max_m && ...
                isfinite(bg_bracket_dz) && bg_bracket_dz <= max_rho_bracket_dz_m && ...
                isfinite(bg_drho_dz) && abs(bg_drho_dz) >= min_drho_dz;
        end
    else
        for uu = 1:numel(unique_argo_ids)
            rr0 = first_row(uu);
            [~, month_id, ~] = datevec(rows{rr0,5});
            boa_profile = boa_density_profile_at(boa_clim, rows{rr0,6}, rows{rr0,7}, month_id);
            boa_profile = align_density_units(boa_profile, rows{rr0,12});
            [z_bg, bg_crossing_count, bg_bracket_dz, bg_drho_dz] = isopycnal_depth_qc(boa_clim.pres, boa_profile, rows{rr0,12}, rows{rr0,8});
            unique_z_bg(uu) = z_bg;
            unique_bg_crossing_count(uu) = bg_crossing_count;
            unique_bg_bracket_dz(uu) = bg_bracket_dz;
            unique_bg_drho_dz(uu) = bg_drho_dz;
            unique_bg_ok(uu) = isfinite(z_bg) && z_bg >= z_rho_min_m && z_bg <= z_rho_max_m && ...
                isfinite(bg_bracket_dz) && bg_bracket_dz <= max_rho_bracket_dz_m && ...
                isfinite(bg_drho_dz) && abs(bg_drho_dz) >= min_drho_dz;
        end
    end
    log_step(sprintf('BOA rho QC done: %d unique kept, %.1f s', nnz(unique_bg_ok), toc(boa_timer)));
    for rr = 1:size(rows, 1)
        uu = row_to_unique(rr);
        rows{rr,30} = unique_bg_crossing_count(uu);
        rows{rr,31} = unique_bg_bracket_dz(uu);
        rows{rr,32} = unique_bg_drho_dz(uu);
        rows{rr,33} = unique_bg_ok(uu);
        if unique_bg_ok(uu)
            z_bg_all(rr) = unique_z_bg(uu);
            z_anom(rr) = z_rho(rr) - unique_z_bg(uu);
            keep_boa(rr) = true;
        end
    end
    rows = rows(keep_boa,:);
    if isempty(rows)
        return
    end
    z_bg_all = z_bg_all(keep_boa);
    z_anom = z_anom(keep_boa);
    for rr = 1:size(rows, 1)
        rows{rr,14} = z_bg_all(rr);
        rows{rr,15} = z_anom(rr);
        if size(rows, 2) < 33 || isempty(rows{rr,33})
            rows{rr,33} = false;
        end
    end
end
