function plot_3d_depth_slices(path, grid3d, title_prefix)
    fig = figure('Visible','off','Position',[100 100 1500 660]);
    requested = [100 500 1000 1500 1900];
    idx = zeros(size(requested));
    for i = 1:numel(requested)
        [~, idx(i)] = min(abs(grid3d.depth_levels - requested(i)));
    end
    idx = unique(idx, 'stable');
    vals = grid3d.w(:,:,idx) * 1e6;
    lim = max(abs(vals(isfinite(vals))));
    if isempty(lim) || ~isfinite(lim) || lim == 0
        lim = 2.5;
    end
    lim = max(lim, 2.5);
    for k = 1:numel(idx)
        subplot(2, ceil(numel(idx)/2), k);
        data = grid3d.w(:,:,idx(k)) * 1e6;
        plot_filled_2d_field(grid3d, data, lim);
        hold on;
        th = linspace(0, 2*pi, 240);
        plot(cos(th), sin(th), 'k-', 'LineWidth', 1.0);
        plot(4*cos(th), 4*sin(th), 'k-', 'LineWidth', 1.0);
        plot(0, 0, 'k.', 'MarkerSize', 14);
        title(sprintf('%.0f m', grid3d.depth_levels(idx(k))));
        xlabel('x/R'); ylabel('y/R');
    end
    sgtitle([title_prefix ' W depth slices  (10^{-6} m s^{-1})'], 'Interpreter', 'tex');
    tmp_path = [tempname(fileparts(path)) '.png'];
    try
        exportgraphics(fig, tmp_path, 'Resolution', 180);
        if exist(path, 'file') == 2
            delete(path);
        end
        movefile(tmp_path, path, 'f');
    catch ME
        fallback_path = fullfile(fileparts(path), ['w_3d_depth_slices_' datestr(now, 'yyyymmdd_HHMMSS') '.png']);
        if exist(tmp_path, 'file') == 2
            movefile(tmp_path, fallback_path, 'f');
        end
        warning('Could not replace %s: %s. Wrote %s instead.', path, ME.message, fallback_path);
    end
    close(fig);
end
