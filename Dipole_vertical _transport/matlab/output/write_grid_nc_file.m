function write_grid_nc_file(path, grid, polarity, band_label)
    if exist(path, 'file') == 2
        delete(path);
    end
    [ny, nx] = size(grid.x);
    write_nc_2d(path, 'x_over_R', grid.x);
    write_nc_2d(path, 'y_over_R', grid.y);
    vars = {'z','z_raw','z_anom','u','v','wpk','count','mapped_support','wpk_mapped_support', ...
        'term1','term2','rebuild_w','term1_depth_positive','term2_depth_positive','rebuild_w_raw_depth_positive', ...
        'term1_plus','term1_minus','term2_abs','term2_rel','rebuild_plus_abs','rebuild_minus_rel', ...
        'sample_term1','sample_term2','sample_rebuild_w'};
    names = {'z_rho_m','z_rho_raw_m','z_rho_anom_m','u_argo_m_s','v_argo_m_s','wpk_observed_m_s','sample_count','mapped_support','wpk_mapped_support', ...
        'term1_m_s','term2_m_s','rebuild_w_m_s','term1_depth_positive_m_s','term2_depth_positive_m_s','rebuild_w_raw_depth_positive_m_s', ...
        'term1_plus_m_s','term1_minus_m_s','term2_abs_m_s','term2_rel_m_s','rebuild_plus_abs_m_s','rebuild_minus_rel_m_s', ...
        'sample_term1_m_s','sample_term2_m_s','sample_rebuild_w_m_s'};
    for i = 1:numel(vars)
        if isfield(grid, vars{i})
            write_nc_2d(path, names{i}, grid.(vars{i}));
        end
    end
    ncwriteatt(path, '/', 'polarity', polarity);
    ncwriteatt(path, '/', 'lat_band', band_label);
    ncwriteatt(path, '/', 'w_positive_direction', 'upward');
    ncwriteatt(path, '/', 'depth_positive_direction', 'downward');
    ncwriteatt(path, '/', 'nx', nx);
    ncwriteatt(path, '/', 'ny', ny);
end
