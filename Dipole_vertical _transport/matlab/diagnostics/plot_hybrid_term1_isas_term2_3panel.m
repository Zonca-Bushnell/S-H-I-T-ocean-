function plot_hybrid_term1_isas_term2_3panel(path, hybrid, polarity, band_label)
    [section_term1, info1] = regularize_field_for_plot(hybrid.section_term1, 2, 3);
    [section_term2, info2] = regularize_field_for_plot(hybrid.section_term2, 2, 3);
    [section_w, info3] = regularize_field_for_plot(hybrid.section_w, 2, 3);
    vals = [section_term1(:); section_term2(:); section_w(:)] * 1e6;
    lim = q95_abs(vals);
    if ~isfinite(lim) || lim <= 0
        lim = 2.5;
    end
    fig = figure('Visible','off','Color','w','Position',[100 100 1650 520]);
    tl = tiledlayout(fig, 1, 3, 'TileSpacing', 'compact', 'Padding', 'compact');
    x = hybrid.x(1,:);
    depth_plot = hybrid.depth_levels(:);
    panels = {section_term1, section_term2, section_w};
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
    sgtitle(tl, [polarity ' ' band_label ' Argo absolute term1 + ISAS term2 (display-regularized)'], 'Interpreter', 'none');
    annotation(fig, 'textbox', [0.01 0.01 0.98 0.04], 'String', ...
        sprintf('PNG only: support-limited gap fill + x-depth smoothing. filled cells term1/term2/W = %d/%d/%d; MAT fields unchanged.', ...
        info1.filled_count, info2.filled_count, info3.filled_count), ...
        'EdgeColor', 'none', 'HorizontalAlignment', 'center', 'FontSize', 8, 'Interpreter', 'none');
    export_png_safe(fig, path, 180);
    close(fig);
end
