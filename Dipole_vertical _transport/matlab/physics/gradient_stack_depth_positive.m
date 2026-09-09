function [dzdx_stack, dzdy_stack] = gradient_stack_depth_positive(z_stack, dx_m, dy_m)
    [ny, nx, nz] = size(z_stack);
    dzdx_stack = nan(ny, nx, nz);
    dzdy_stack = nan(ny, nx, nz);
    for zz = 1:nz
        z_grid = fillmissing2(z_stack(:,:,zz));
        [dzdx, dzdy] = gradient(z_grid, dx_m, dy_m);
        dzdx_stack(:,:,zz) = dzdx;
        dzdy_stack(:,:,zz) = dzdy;
    end
end
