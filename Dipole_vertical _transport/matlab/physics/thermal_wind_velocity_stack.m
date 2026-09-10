function [u_tw, v_tw] = thermal_wind_velocity_stack(rho_grids, support3, u_base, v_base, base_support, depth_levels, dx_m, dy_m, f)
    [ny, nx, nz] = size(rho_grids);
    u_tw = nan(ny, nx, nz);
    v_tw = nan(ny, nx, nz);
    if ~isfinite(f) || abs(f) < 1e-7 || ~isfinite(dx_m) || dx_m <= 0 || ~isfinite(dy_m) || dy_m <= 0
        return
    end
    g = 9.81;
    rho_ref = 1025;
    du_dD = nan(ny, nx, nz);
    dv_dD = nan(ny, nx, nz);
    for zz = 1:nz
        rho_grid = fillmissing2(rho_grids(:,:,zz));
        [drhodx, drhody] = gradient_xy(rho_grid, dx_m, dy_m);
        du_dD(:,:,zz) = mask_to_support(g ./ (f * rho_ref) .* drhody, support3(:,:,zz));
        dv_dD(:,:,zz) = mask_to_support(-g ./ (f * rho_ref) .* drhodx, support3(:,:,zz));
    end
    [~, anchor] = min(abs(depth_levels(:) - 1000));
    u_tw(:,:,anchor) = mask_to_support(u_base, base_support & support3(:,:,anchor));
    v_tw(:,:,anchor) = mask_to_support(v_base, base_support & support3(:,:,anchor));
    for zz = anchor+1:nz
        dD = depth_levels(zz) - depth_levels(zz-1);
        u_tw(:,:,zz) = u_tw(:,:,zz-1) + 0.5 .* (du_dD(:,:,zz-1) + du_dD(:,:,zz)) .* dD;
        v_tw(:,:,zz) = v_tw(:,:,zz-1) + 0.5 .* (dv_dD(:,:,zz-1) + dv_dD(:,:,zz)) .* dD;
        u_tw(:,:,zz) = mask_to_support(u_tw(:,:,zz), base_support & support3(:,:,zz));
        v_tw(:,:,zz) = mask_to_support(v_tw(:,:,zz), base_support & support3(:,:,zz));
    end
    for zz = anchor-1:-1:1
        dD = depth_levels(zz+1) - depth_levels(zz);
        u_tw(:,:,zz) = u_tw(:,:,zz+1) - 0.5 .* (du_dD(:,:,zz+1) + du_dD(:,:,zz)) .* dD;
        v_tw(:,:,zz) = v_tw(:,:,zz+1) - 0.5 .* (dv_dD(:,:,zz+1) + dv_dD(:,:,zz)) .* dD;
        u_tw(:,:,zz) = mask_to_support(u_tw(:,:,zz), base_support & support3(:,:,zz));
        v_tw(:,:,zz) = mask_to_support(v_tw(:,:,zz), base_support & support3(:,:,zz));
    end
end
