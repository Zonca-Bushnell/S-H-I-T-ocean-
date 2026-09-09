function plot_sensitivity(path, grid, title_prefix)
    fig = figure('Visible','off','Position',[100 100 1180 900]);
    fields = {'rebuild_w','rebuild_w_raw_depth_positive','term1','term2'};
    titles = {'main W upward-positive','raw W depth-positive','term1 W up+','term2 W up+'};
    vals = [];
    for k = 1:numel(fields)
        vals = [vals; grid.(fields{k})(:)]; %#ok<AGROW>
    end
    lim = max(abs(vals(isfinite(vals))));
    if isempty(lim) || ~isfinite(lim) || lim == 0
        lim = 2.5e-6;
    end
    lim = max(lim, 2.5e-6);
    for k = 1:4
        subplot(2,2,k);
        data = grid.(fields{k}) * 1e6;
        plot_filled_2d_field(grid, data, lim * 1e6);
        hold on;
        th = linspace(0, 2*pi, 240);
        plot(cos(th), sin(th), 'k-', 'LineWidth', 1.2);
        plot(4*cos(th), 4*sin(th), 'k-', 'LineWidth', 1.2);
        plot(0, 0, 'k.', 'MarkerSize', 16);
        title(titles{k}, 'Interpreter', 'tex');
        xlabel('x/R'); ylabel('y/R');
    end
    sgtitle([title_prefix ' velocity/sign sensitivity  (10^{-6} m s^{-1})'], 'Interpreter', 'tex');
    tmp_path = [tempname(fileparts(path)) '.png'];
    try
        exportgraphics(fig, tmp_path, 'Resolution', 180);
        if exist(path, 'file') == 2
            delete(path);
        end
        movefile(tmp_path, path, 'f');
    catch ME
        fallback_path = fullfile(fileparts(path), ['velocity_sign_sensitivity_' datestr(now, 'yyyymmdd_HHMMSS') '.png']);
        if exist(tmp_path, 'file') == 2
            movefile(tmp_path, fallback_path, 'f');
        end
        warning('Could not replace %s: %s. Wrote %s instead.', path, ME.message, fallback_path);
    end
    close(fig);
end
