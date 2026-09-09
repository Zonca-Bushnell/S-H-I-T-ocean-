function variant = make_reversal_variant(name, description, term1, term2, support3, grid3d)
    term1 = mask_stack(term1, support3);
    term2 = mask_stack(term2, support3);
    w = mask_stack(term1 + term2, support3);
    section_w = section_stack(w, grid3d.y(:,1), grid3d.section_half_width_r);
    stats = reversal_section_stats(section_w, grid3d.depth_levels(:));
    stats.q95_abs_w_1e6_m_s = q95_abs(w(:) * 1e6);
    variant = struct('name', name, 'description', description, 'term1', term1, 'term2', term2, 'w', w, 'section_w', section_w, 'stats', stats);
end
