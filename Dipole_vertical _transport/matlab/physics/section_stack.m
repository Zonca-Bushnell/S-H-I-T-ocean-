function section = section_stack(field3d, y_vec, half_width_r)
    if ~isfinite(half_width_r) || half_width_r <= 0
        half_width_r = 0.25;
    end
    y_mask = abs(y_vec) <= half_width_r;
    tmp = field3d(y_mask,:,:);
    section = squeeze(median(tmp, 1, 'omitnan'))';
end
