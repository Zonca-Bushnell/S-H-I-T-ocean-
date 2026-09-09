function variant = make_reversal_variant_from_factors(name, description, dzdx_down, dzdy_down, dzdx_up, dzdy_up, u_field, v_field, term2_mode, support3, grid3d)
    if strcmp(term2_mode, 'predecessor')
        c0 = abs(grid3d.mean_cx_raw);
        term1 = c0 .* dzdx_up;
        term2 = u_field .* dzdx_up + v_field .* dzdy_up;
    else
        term1 = grid3d.cx_rel .* dzdx_down;
        term2 = -((u_field - grid3d.mean_cx_raw) .* dzdx_down + v_field .* dzdy_down);
    end
    variant = make_reversal_variant(name, description, term1, term2, support3, grid3d);
end
