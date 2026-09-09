function plot_reversal_factor_4panel(path, diag, polarity, band_label)
    panel_names = {'A_current','B_comp_isoslope','C_comp_isoslope_comp_tw','D_predecessor_like'};
    idx = zeros(1, numel(panel_names));
    vals = [];
    for i = 1:numel(panel_names)
        idx(i) = find(strcmp(diag.variant_names, panel_names{i}), 1);
        vals = [vals; diag.variants(idx(i)).section_w(:) * 1e6]; %#ok<AGROW>
    end
    lim = q95_abs(vals);
    if ~isfinite(lim) || lim <= 0
        lim = 2.5;
    end
    fig = figure('Visible','off','Color','w','Position',[100 100 1500 1000]);
    tl = tiledlayout(fig, 2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
    x = diag.x(1,:);
    depth = diag.depth_levels(:);
    titles = {'A current', 'B comp-rho isoslope', 'C comp-rho isoslope + TW', 'D predecessor-like'};
    for i = 1:numel(idx)
        ax = nexttile(tl);
        data = diag.variants(idx(i)).section_w * 1e6;
        contourf(ax, x, depth, data, 28, 'LineStyle', 'none');
        set(ax, 'YDir', 'reverse');
        colormap(ax, redblue_colormap());
        clim(ax, [-lim lim]);
        cb = colorbar(ax);
        ylabel(cb, '10^{-6} m s^{-1}');
        xlabel(ax, 'x/R');
        ylabel(ax, 'Depth (m)');
        title(ax, titles{i}, 'Interpreter', 'none');
    end
    sgtitle(tl, [polarity ' ' band_label ' reversal factor W sections'], 'Interpreter', 'none');
    exportgraphics(fig, path, 'Resolution', 180);
    close(fig);
end
