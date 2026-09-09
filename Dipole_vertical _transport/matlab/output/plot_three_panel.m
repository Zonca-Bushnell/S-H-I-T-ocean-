function plot_three_panel(path, grid, title_prefix)
    fig = figure('Visible','off','Position',[100 100 1500 430]);
    fields = {'term1','term2','rebuild_w'};
    titles = {'+c_x^{rel} dz''_\rho/dx  (W up+)','-(u_{pk}-c_x, v_{pk}) \cdot \nabla z''_\rho','rebuild W  (up+)'};
    vals = [grid.term1(:); grid.term2(:); grid.rebuild_w(:)];
    lim = max(abs(vals(isfinite(vals))));
    if isempty(lim) || ~isfinite(lim) || lim == 0
        lim = 2.5e-6;
    end
    lim = max(lim, 2.5e-6);
    for k = 1:3
        subplot(1,3,k);
        data = grid.(fields{k}) * 1e6;
        plot_filled_2d_field(grid, data, lim * 1e6);
        hold on;
        th = linspace(0, 2*pi, 240);
        plot(cos(th), sin(th), 'k-', 'LineWidth', 1.2);
        plot(4*cos(th), 4*sin(th), 'k-', 'LineWidth', 1.2);
        plot(0, 0, 'k.', 'MarkerSize', 16);
        title(titles{k});
        xlabel('x/R'); ylabel('y/R');
    end
    sgtitle([title_prefix '  W upward-positive (10^{-6} m s^{-1})'], 'Interpreter', 'tex');
    tmp_path = [tempname(fileparts(path)) '.png'];
    try
        exportgraphics(fig, tmp_path, 'Resolution', 180);
        if exist(path, 'file') == 2
            delete(path);
        end
        movefile(tmp_path, path, 'f');
    catch ME
        fallback_path = fullfile(fileparts(path), ['vertical_transport_terms_' datestr(now, 'yyyymmdd_HHMMSS') '.png']);
        if exist(tmp_path, 'file') == 2
            movefile(tmp_path, fallback_path, 'f');
        end
        warning('Could not replace %s: %s. Wrote %s instead.', path, ME.message, fallback_path);
    end
    close(fig);
end
