function [dzdx_stack, dzdy_stack] = predecessor_isopycnal_slope(rho_stack, depth_levels, dx_m, dy_m, upward_coordinate)
    [ny, nx, nz] = size(rho_stack);
    rho_clean = nan(size(rho_stack));
    parfor ii = 1:ny
        row_clean = nan(1, nx, nz);
        for jj = 1:nx
            row_clean(1,jj,:) = reshape(monotonic_density_profile(squeeze(rho_stack(ii,jj,:))), 1, 1, nz);
        end
        rho_clean(ii,:,:) = row_clean;
    end
    z_axis = depth_levels(:);
    if upward_coordinate
        z_axis = -z_axis;
    end
    dzdx_stack = nan(ny, nx, nz);
    dzdy_stack = nan(ny, nx, nz);
    parfor ii = 1:ny
        dzdx_row = nan(1, nx, nz);
        dzdy_row = nan(1, nx, nz);
        for jj = 1:nx
            center = squeeze(rho_clean(ii,jj,:));
            if nnz(isfinite(center)) < 3
                continue
            end
            if jj == 1
                left = squeeze(rho_clean(ii,jj,:));
                right = squeeze(rho_clean(ii,jj+1,:));
                scale_x = 1 / dx_m;
            elseif jj == nx
                left = squeeze(rho_clean(ii,jj-1,:));
                right = squeeze(rho_clean(ii,jj,:));
                scale_x = 1 / dx_m;
            else
                left = squeeze(rho_clean(ii,jj-1,:));
                right = squeeze(rho_clean(ii,jj+1,:));
                scale_x = 0.5 / dx_m;
            end
            z_left = interp_density_to_depth(left, z_axis, center);
            z_right = interp_density_to_depth(right, z_axis, center);
            dzdx_row(1,jj,:) = reshape((z_right - z_left) .* scale_x, 1, 1, nz);
            if ii == 1
                south = squeeze(rho_clean(ii,jj,:));
                north = squeeze(rho_clean(ii+1,jj,:));
                scale_y = 1 / dy_m;
            elseif ii == ny
                south = squeeze(rho_clean(ii-1,jj,:));
                north = squeeze(rho_clean(ii,jj,:));
                scale_y = 1 / dy_m;
            else
                south = squeeze(rho_clean(ii-1,jj,:));
                north = squeeze(rho_clean(ii+1,jj,:));
                scale_y = 0.5 / dy_m;
            end
            z_south = interp_density_to_depth(south, z_axis, center);
            z_north = interp_density_to_depth(north, z_axis, center);
            dzdy_row(1,jj,:) = reshape((z_north - z_south) .* scale_y, 1, 1, nz);
        end
        dzdx_stack(ii,:,:) = dzdx_row;
        dzdy_stack(ii,:,:) = dzdy_row;
    end
end
