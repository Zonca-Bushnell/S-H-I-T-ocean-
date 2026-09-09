function plot_filled_2d_field(grid, data, lim_micro)
    if any(isfinite(data(:)))
        contourf(grid.x(1,:), grid.y(:,1), data, 24, 'LineStyle', 'none');
    else
        h = imagesc(grid.x(1,:), grid.y(:,1), data);
        set(h, 'AlphaData', isfinite(data));
    end
    set(gca, 'YDir', 'normal');
    set(gca, 'Color', [1 1 1]);
    axis image;
    xlim([-4 4]); ylim([-4 4]);
    clim([-lim_micro lim_micro]);
    colormap(redblue_colormap());
    colorbar;
end
