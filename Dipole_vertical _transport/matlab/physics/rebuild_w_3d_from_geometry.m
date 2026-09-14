function grid3d = rebuild_w_3d_from_geometry(grid3d)
    [ny, nx, nz] = size(grid3d.z_anom);
    grid3d.term1 = nan(ny, nx, nz);
    grid3d.term2 = nan(ny, nx, nz);
    grid3d.w = nan(ny, nx, nz);
    grid3d.dDdx = nan(ny, nx, nz);
    grid3d.dDdy = nan(ny, nx, nz);
    grid3d.w_stage = 'physics_rebuild_from_isopycnal_geometry';
    x_vec = grid3d.x(1,:);
    y_vec = grid3d.y(:,1);
    dx_m = mean(diff(x_vec)) * grid3d.mean_radius_m;
    dy_m = mean(diff(y_vec)) * grid3d.mean_radius_m;
    if ~isfinite(dx_m) || dx_m <= 0 || ~isfinite(dy_m) || dy_m <= 0
        return
    end
    for zz = 1:nz
        support = grid3d.mapped_support(:,:,zz) >= grid3d.cressman_min_obs & ...
            isfinite(grid3d.u_tw(:,:,zz)) & isfinite(grid3d.v_tw(:,:,zz));
        [dDdx, dDdy] = gradient_xy(fillmissing2(grid3d.z_anom(:,:,zz)), dx_m, dy_m);
        grid3d.dDdx(:,:,zz) = mask_to_support(dDdx, support);
        grid3d.dDdy(:,:,zz) = mask_to_support(dDdy, support);
        term1 = mask_to_support(grid3d.cx_rel .* dDdx, support);
        term2 = mask_to_support(-((grid3d.u_tw(:,:,zz) - grid3d.mean_cx_raw) .* dDdx + grid3d.v_tw(:,:,zz) .* dDdy), support);
        grid3d.term1(:,:,zz) = term1;
        grid3d.term2(:,:,zz) = term2;
        grid3d.w(:,:,zz) = mask_to_support(term1 + term2, support);
    end
end
