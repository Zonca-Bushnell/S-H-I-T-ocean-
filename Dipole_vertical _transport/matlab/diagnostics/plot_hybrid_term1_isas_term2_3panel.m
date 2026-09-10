function plot_hybrid_term1_isas_term2_3panel(path, hybrid, polarity, band_label)
    vals = [hybrid.section_term1(:); hybrid.section_term2(:); hybrid.section_w(:)] * 1e6;
    lim = q95_abs(vals);
    if ~isfinite(lim) || lim <= 0
        lim = 2.5;
    end
    fig = figure('Visible','off','Color','w','Position',[100 100 1650 520]);
    tl = tiledlayout(fig, 1, 3, 'TileSpacing', 'compact', 'Padding', 'compact');
    x = hybrid.x(1,:);
    depth_plot = hybrid.depth_levels(:);
    panels = {hybrid.section_term1, hybrid.section_term2, hybrid.section_w};
    titles = {'term1: Argo composite absolute z_\rho', 'term2: ISAS background z_\rho', 'rebuild W'};
    for ii = 1:3
        ax = nexttile(tl);
        data = panels{ii} * 1e6;
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
        title(ax, titles{ii}, 'Interpreter', 'tex');
    end
    sgtitle(tl, [polarity ' ' band_label ' Argo absolute term1 + ISAS term2'], 'Interpreter', 'none');
    export_png_safe(fig, path, 180);
    close(fig);
end
