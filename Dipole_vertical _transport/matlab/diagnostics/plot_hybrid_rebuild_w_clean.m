function plot_hybrid_rebuild_w_clean(section_path, slices_path, hybrid, polarity, band_label)
%PLOT_HYBRID_REBUILD_W_CLEAN Create presentation-only rebuild-W figures.
%   The saved MAT fields are not modified. This function applies the same
%   support-limited display regularization used by the diagnostic figures, then
%   renders only rebuild W to reduce visual clutter in reports.
    plot_clean_section(section_path, hybrid, polarity, band_label);
    plot_clean_slices(slices_path, hybrid, polarity, band_label);
end

function plot_clean_section(path, hybrid, polarity, band_label)
    [section_w, info] = regularize_field_for_plot(hybrid.section_w, 8, 2, ...
        'sigma', [11.0 3.2], 'min_neighbors', 2);
    data = section_w * 1e6;
    lim = q95_abs(data(:));
    if ~isfinite(lim) || lim <= 0
        lim = 2.5;
    end

    x = hybrid.x(1,:);
    depth_plot = hybrid.depth_levels(:);
    fig = figure('Visible','off','Color','w','Position',[100 100 760 620]);
    ax = axes(fig);
    if any(isfinite(data(:)))
        [xq, depth_q, data_q] = upsample_field_for_plot(x, depth_plot, data, 5, 2);
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
    title(ax, sprintf('%s %s rebuild W', polarity, band_label), 'Interpreter', 'none');
    annotation(fig, 'textbox', [0.08 0.01 0.84 0.04], 'String', ...
        sprintf('Display-regularized PNG; filled cells = %d; MAT fields unchanged.', info.filled_count), ...
        'EdgeColor', 'none', 'HorizontalAlignment', 'center', 'FontSize', 8, 'Interpreter', 'none');
    export_png_safe(fig, path, 220);
    close(fig);
end

function plot_clean_slices(path, hybrid, polarity, band_label)
    wanted = [200 500 1000 1500 1900];
    idx = arrayfun(@(d) nearest_depth_index(hybrid.depth_levels(:), d), wanted);
    [w_plot, info] = regularize_stack_for_plot(hybrid.w(:,:,idx), 10, 2, ...
        'sigma', [6.0 6.0], 'min_neighbors', 2);
    vals = w_plot * 1e6;
    lim = q95_abs(vals(:));
    if ~isfinite(lim) || lim <= 0
        lim = 2.5;
    end

    x = hybrid.x(1,:);
    y = hybrid.y(:,1);
    fig = figure('Visible','off','Color','w','Position',[100 100 1440 880]);
    tl = tiledlayout(fig, 2, 3, 'TileSpacing', 'compact', 'Padding', 'compact');
    th = linspace(0, 2*pi, 240);
    for ii = 1:numel(idx)
        ax = nexttile(tl);
        data = w_plot(:,:,ii) * 1e6;
        if any(isfinite(data(:)))
            [xq, yq, data_q] = upsample_field_for_plot(x, y, data, 5, 5);
            contourf(ax, xq, yq, data_q, 14, 'LineStyle', 'none');
            axis(ax, 'equal');
            axis(ax, [-4 4 -4 4]);
            colormap(ax, redblue_colormap());
            clim(ax, [-lim lim]);
            hold(ax, 'on');
            plot(ax, cos(th), sin(th), 'k-', 'LineWidth', 1.1);
            plot(ax, 4*cos(th), 4*sin(th), 'k-', 'LineWidth', 1.1);
            plot(ax, 0, 0, 'k.', 'MarkerSize', 15);
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
    sgtitle(tl, sprintf('%s %s rebuild W depth slices', polarity, band_label), 'Interpreter', 'none');
    annotation(fig, 'textbox', [0.08 0.01 0.84 0.04], 'String', ...
        sprintf('Display-regularized PNG; filled cells = %d; MAT fields unchanged.', info.filled_count), ...
        'EdgeColor', 'none', 'HorizontalAlignment', 'center', 'FontSize', 8, 'Interpreter', 'none');
    export_png_safe(fig, path, 220);
    close(fig);
end
