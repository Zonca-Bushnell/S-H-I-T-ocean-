function plot_predecessor_allsat_validation_4panel(path, hybrid, polarity, band_label)
    [section_term1, info1] = regularize_field_for_plot(hybrid.section_term1, 8, 2, 'sigma', [10.0 3.0], 'min_neighbors', 2);
    [section_term2, info2] = regularize_field_for_plot(hybrid.section_term2, 8, 2, 'sigma', [10.0 3.0], 'min_neighbors', 2);
    [section_w, info3] = regularize_field_for_plot(hybrid.section_w, 8, 2, 'sigma', [10.0 3.0], 'min_neighbors', 2);
    vals = [section_term1(:); section_term2(:); section_w(:)] * 1e6;
    lim = q95_abs(vals);
    if ~isfinite(lim) || lim <= 0
        lim = 2.5;
    end

    fig = figure('Visible','off','Color','w','Position',[100 100 1800 820]);
    tl = tiledlayout(fig, 2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
    x = hybrid.x(1,:);
    depth_plot = hybrid.depth_levels(:);
    panels = {section_term1, section_term2, section_w};
    titles = {'term1: Argo composite absolute D_\rho', 'term2: ISAS background D_\rho', 'rebuild W'};
    for ii = 1:3
        ax = nexttile(tl);
        data = panels{ii} * 1e6;
        if any(isfinite(data(:)))
            [xq, depth_q, data_q] = upsample_field_for_plot(x, depth_plot, data, 4, 2);
            contourf(ax, xq, depth_q, data_q, 22, 'LineStyle', 'none');
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

    ax = nexttile(tl);
    stats = hybrid.stats;
    corr_depth = stats.corr_w_vs_1000m;
    plot(ax, corr_depth, depth_plot, 'k-', 'LineWidth', 1.8);
    hold(ax, 'on');
    plot(ax, [0 0], [min(depth_plot) max(depth_plot)], 'Color', [0.65 0.65 0.65], 'LineStyle', '--');
    plot(ax, [-1 1], [1000 1000], 'Color', [0.65 0.65 0.65], 'LineStyle', ':');
    set(ax, 'YDir', 'reverse');
    xlim(ax, [-1 1]);
    ylim(ax, [min(depth_plot) max(depth_plot)]);
    xlabel(ax, 'corr(W(D), W(1000 m))');
    ylabel(ax, 'Depth (m)');
    title(ax, sprintf('deep reversal score = %.2f', stats.deep_reversal_score));
    grid(ax, 'on');

    sgtitle(tl, [polarity ' ' band_label ' predecessor algorithm with META3.2 allsat substitute'], 'Interpreter', 'none');
    annotation(fig, 'textbox', [0.01 0.01 0.98 0.04], 'String', ...
        sprintf('PNG display only: regularized for readability; filled cells term1/term2/W = %d/%d/%d. MAT fields remain support-limited.', ...
        info1.filled_count, info2.filled_count, info3.filled_count), ...
        'EdgeColor', 'none', 'HorizontalAlignment', 'center', 'FontSize', 8, 'Interpreter', 'none');
    export_png_safe(fig, path, 180);
    close(fig);
end
