function plot_hybrid_depth_slices(path, hybrid, polarity, band_label)
    wanted = [200 500 1000 1500 1900];
    idx = arrayfun(@(d) nearest_depth_index(hybrid.depth_levels(:), d), wanted);
    [w_plot, info] = regularize_stack_for_plot(hybrid.w(:,:,idx), 2, 2);
    vals = w_plot * 1e6;
    lim = q95_abs(vals(:));
    if ~isfinite(lim) || lim <= 0
        lim = 2.5;
    end
    fig = figure('Visible','off','Color','w','Position',[100 100 1700 700]);
    tl = tiledlayout(fig, 2, 3, 'TileSpacing', 'compact', 'Padding', 'compact');
    x = hybrid.x(1,:);
    y = hybrid.y(:,1);
    for ii = 1:numel(idx)
        ax = nexttile(tl);
        data = w_plot(:,:,ii) * 1e6;
        if any(isfinite(data(:)))
            contourf(ax, x, y, data, 28, 'LineStyle', 'none');
            axis(ax, 'equal');
            axis(ax, [-4 4 -4 4]);
            colormap(ax, redblue_colormap());
            clim(ax, [-lim lim]);
            hold(ax, 'on');
            th = linspace(0, 2*pi, 240);
            plot(ax, cos(th), sin(th), 'k-', 'LineWidth', 1.2);
            plot(ax, 4*cos(th), 4*sin(th), 'k-', 'LineWidth', 1.2);
            plot(ax, 0, 0, 'k.', 'MarkerSize', 16);
        else
            axis(ax, [-4 4 -4 4]);
            text(ax, 0, 0, 'No valid field', 'HorizontalAlignment', 'center');
        end
        xlabel(ax, 'x/R');
        ylabel(ax, 'y/R');
        title(ax, sprintf('%g m', hybrid.depth_levels(idx(ii))));
    end
    ax = nexttile(tl);
    axis(ax, 'off');
    cb = colorbar(ax);
    cb.Layout.Tile = 'east';
    ylabel(cb, '10^{-6} m s^{-1}');
    colormap(fig, redblue_colormap());
    clim(ax, [-lim lim]);
    sgtitle(tl, [polarity ' ' band_label ' rebuild W depth slices (display-regularized)'], 'Interpreter', 'none');
    annotation(fig, 'textbox', [0.01 0.01 0.98 0.04], 'String', ...
        sprintf('PNG only: support-limited gap fill + horizontal smoothing; filled cells = %d; MAT fields unchanged.', info.filled_count), ...
        'EdgeColor', 'none', 'HorizontalAlignment', 'center', 'FontSize', 8, 'Interpreter', 'none');
    export_png_safe(fig, path, 180);
    close(fig);
end
