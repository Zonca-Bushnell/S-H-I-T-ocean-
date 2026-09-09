function plot_zgeometry_4panel(path, comparison, polarity, band_label)
    vals = [];
    for ii = 1:numel(comparison.panels)
        vals = [vals; comparison.panels(ii).section_w(:) * 1e6]; %#ok<AGROW>
    end
    lim = q95_abs(vals);
    if ~isfinite(lim) || lim <= 0
        lim = 2.5;
    end
    fig = figure('Visible','off','Color','w','Position',[100 100 1500 1000]);
    tl = tiledlayout(fig, 2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
    x = comparison.x(1,:);
    depth_plot = comparison.depth_levels(:);
    for ii = 1:numel(comparison.panels)
        ax = nexttile(tl);
        data = comparison.panels(ii).section_w * 1e6;
        contourf(ax, x, depth_plot, data, 28, 'LineStyle', 'none');
        set(ax, 'YDir', 'reverse');
        colormap(ax, redblue_colormap());
        clim(ax, [-lim lim]);
        cb = colorbar(ax);
        ylabel(cb, '10^{-6} m s^{-1}');
        xlabel(ax, 'x/R');
        ylabel(ax, 'Depth (m)');
        title(ax, comparison.panels(ii).title, 'Interpreter', 'none');
    end
    sgtitle(tl, [polarity ' ' band_label ' z_\rho geometry comparison'], 'Interpreter', 'none');
    exportgraphics(fig, path, 'Resolution', 180);
    close(fig);
end
