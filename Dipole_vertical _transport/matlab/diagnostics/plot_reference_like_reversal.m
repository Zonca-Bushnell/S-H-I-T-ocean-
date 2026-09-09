function plot_reference_like_reversal(section_path, terms_path, grid3d, result, polarity, band_label)
    x = grid3d.x(1,:);
    depth_plot = grid3d.depth_levels(:);
    vals = result.w(:) * 1e6;
    lim = q95_abs(vals);
    if ~isfinite(lim) || lim <= 0
        lim = 2.5;
    end

    fig = figure('Visible','off','Color','w','Position',[100 100 900 820]);
    contourf(x, depth_plot, result.section_w * 1e6, 32, 'LineStyle', 'none');
    set(gca, 'YDir', 'reverse');
    colormap(redblue_colormap());
    clim([-lim lim]);
    colorbar;
    xlabel('x/R');
    ylabel('Depth (m)');
    title(sprintf('%s %s reference-like W', polarity, band_label), 'Interpreter', 'none');
    exportgraphics(fig, section_path, 'Resolution', 180);
    close(fig);

    fig = figure('Visible','off','Color','w','Position',[100 100 1800 760]);
    tl = tiledlayout(fig, 1, 3, 'TileSpacing', 'compact', 'Padding', 'compact');
    names = {'term1', 'term2', 'W'};
    fields = {'section_term1', 'section_term2', 'section_w'};
    vals = [result.section_term1(:); result.section_term2(:); result.section_w(:)] * 1e6;
    term_lim = q95_abs(vals);
    if ~isfinite(term_lim) || term_lim <= 0
        term_lim = lim;
    end
    for ii = 1:3
        ax = nexttile(tl);
        contourf(ax, x, depth_plot, result.(fields{ii}) * 1e6, 32, 'LineStyle', 'none');
        set(ax, 'YDir', 'reverse');
        colormap(ax, redblue_colormap());
        clim(ax, [-term_lim term_lim]);
        colorbar(ax);
        xlabel(ax, 'x/R');
        ylabel(ax, 'Depth (m)');
        title(ax, names{ii}, 'Interpreter', 'none');
    end
    sgtitle(tl, sprintf('%s %s reference-like absolute-density implicit slope terms', polarity, band_label), 'Interpreter', 'none');
    exportgraphics(fig, terms_path, 'Resolution', 180);
    close(fig);
end
