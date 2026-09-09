function plot_sensitivity_montage(path, grids, configs, polarity, band_label)
    fig = figure('Visible','off','Position',[100 100 1500 940]);
    vals = [];
    for c = 1:numel(grids)
        vals = [vals; grids{c}.rebuild_w(:)]; %#ok<AGROW>
    end
    lim = max(abs(vals(isfinite(vals))));
    if isempty(lim) || ~isfinite(lim) || lim == 0
        lim = 2.5e-6;
    end
    lim = max(lim, 2.5e-6);
    for c = 1:numel(grids)
        subplot(2, 3, c);
        grid = grids{c};
        data = grid.rebuild_w * 1e6;
        plot_filled_2d_field(grid, data, lim * 1e6);
        hold on;
        th = linspace(0, 2*pi, 240);
        plot(cos(th), sin(th), 'k-', 'LineWidth', 1.0);
        plot(4*cos(th), 4*sin(th), 'k-', 'LineWidth', 1.0);
        plot(0, 0, 'k.', 'MarkerSize', 14);
        cfg = configs(c);
        title(sprintf('%s: N=%d Rc=%.2g min=%d sm=%d', cfg.name, cfg.grid_n, cfg.cressman_radius_r, cfg.cressman_min_obs, cfg.smooth_passes), 'Interpreter', 'none');
        xlabel('x/R'); ylabel('y/R');
    end
    sgtitle([polarity ' ' band_label ' rebuild W sensitivity  (10^{-6} m s^{-1})'], 'Interpreter', 'tex');
    tmp_path = [tempname(fileparts(path)) '.png'];
    try
        exportgraphics(fig, tmp_path, 'Resolution', 180);
        if exist(path, 'file') == 2
            delete(path);
        end
        movefile(tmp_path, path, 'f');
    catch ME
        fallback_path = fullfile(fileparts(path), [polarity '_sensitivity_montage_' datestr(now, 'yyyymmdd_HHMMSS') '.png']);
        if exist(tmp_path, 'file') == 2
            movefile(tmp_path, fallback_path, 'f');
        end
        warning('Could not replace %s: %s. Wrote %s instead.', path, ME.message, fallback_path);
    end
    close(fig);
end
