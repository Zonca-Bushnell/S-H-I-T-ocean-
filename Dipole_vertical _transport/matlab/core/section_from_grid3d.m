function [coord, section_w] = section_from_grid3d(grid3d, section_axis, half_width_r)
    if strcmp(section_axis, 'y')
        mask = abs(grid3d.x(1,:)) <= half_width_r;
        coord = grid3d.y(:,1);
        section_w = squeeze(median(grid3d.w(:,mask,:), 2, 'omitnan'))';
    else
        mask = abs(grid3d.y(:,1)) <= half_width_r;
        coord = grid3d.x(1,:);
        section_w = squeeze(median(grid3d.w(mask,:,:), 1, 'omitnan'))';
    end
end
