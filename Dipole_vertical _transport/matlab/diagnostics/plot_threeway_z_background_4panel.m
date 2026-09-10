function plot_threeway_z_background_4panel(path, comparison, polarity, band_label, field_name)
    vals = [];
    for ii = 1:numel(comparison.panels)
        if strcmp(field_name, 'term2')
            vals = [vals; comparison.panels(ii).section_term2(:) * 1e6]; %#ok<AGROW>
        else
            vals = [vals; comparison.panels(ii).section_w(:) * 1e6]; %#ok<AGROW>
        end
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
        if strcmp(field_name, 'term2')
            data = comparison.panels(ii).section_term2 * 1e6;
        else
            data = comparison.panels(ii).section_w * 1e6;
        end
        if any(isfinite(data(:)))
            contourf(ax, x, depth_plot, data, 28, 'LineStyle', 'none');
            set(ax, 'YDir', 'reverse');
            colormap(ax, redblue_colormap());
            clim(ax, [-lim lim]);
            cb = colorbar(ax);
            ylabel(cb, '10^{-6} m s^{-1}');
        else
            axis(ax, [-4 4 min(depth_plot) max(depth_plot)]);
            set(ax, 'YDir', 'reverse');
            text(ax, 0, median(depth_plot), 'No valid field', 'HorizontalAlignment', 'center');
        end
        xlabel(ax, 'x/R');
        ylabel(ax, 'Depth (m)');
        title(ax, comparison.panels(ii).title, 'Interpreter', 'none');
    end
    sgtitle(tl, [polarity ' ' band_label ' three-way ' field_name], 'Interpreter', 'none');
    exportgraphics(fig, path, 'Resolution', 180);
    close(fig);
end
