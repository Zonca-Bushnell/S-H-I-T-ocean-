function write_grid_3d_nc_file(path, grid3d, polarity, band_label)
    if exist(path, 'file') == 2
        delete(path);
    end
    [ny, nx, nz] = size(grid3d.w);
    nccreate(path, 'depth_m', 'Dimensions', {'depth', nz}, 'Datatype', 'double', 'DeflateLevel', 4);
    ncwrite(path, 'depth_m', double(grid3d.depth_levels(:)));
    write_nc_2d(path, 'x_over_R', grid3d.x);
    write_nc_2d(path, 'y_over_R', grid3d.y);
    write_nc_3d(path, 'w_3d_m_s', grid3d.w);
    write_nc_3d(path, 'term1_3d_m_s', grid3d.term1);
    write_nc_3d(path, 'term2_3d_m_s', grid3d.term2);
    write_nc_3d(path, 'z_rho_anom_3d_m', grid3d.z_anom);
    write_nc_3d(path, 'rho_anom_3d', grid3d.rho_anom);
    write_nc_3d(path, 'u_thermal_wind_3d_m_s', grid3d.u_tw);
    write_nc_3d(path, 'v_thermal_wind_3d_m_s', grid3d.v_tw);
    write_nc_3d(path, 'sample_count_3d', grid3d.count);
    write_nc_3d(path, 'mapped_support_3d', grid3d.mapped_support);
    ncwriteatt(path, '/', 'polarity', polarity);
    ncwriteatt(path, '/', 'lat_band', band_label);
    ncwriteatt(path, '/', 'w_positive_direction', 'upward');
    ncwriteatt(path, '/', 'depth_positive_direction', 'downward');
end
