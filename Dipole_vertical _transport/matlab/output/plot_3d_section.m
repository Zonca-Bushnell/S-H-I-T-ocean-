function plot_3d_section(path, grid3d, title_prefix)
    fig = figure('Visible','off','Position',[100 100 760 760]);
    data = grid3d.section_w * 1e6;
    coord = grid3d.section_coord(:)';
    depth_plot = grid3d.depth_levels(:);
    lim = max(abs(data(isfinite(data))));
    if isempty(lim) || ~isfinite(lim) || lim == 0
        lim = 2.5;
    end
    lim = max(lim, 2.5);
    contourf(coord, depth_plot, data, 24, 'LineStyle', 'none');
    set(gca, 'YDir', 'reverse');
    set(gca, 'Color', [1 1 1]);
    colormap(redblue_colormap());
    clim([-lim lim]);
    colorbar;
    if strcmp(grid3d.section_axis, 'y')
        xlabel('y/R');
    else
        xlabel('x/R');
    end
    ylabel('Depth (m)');
    display_label = strrep(title_prefix, 'cross_', '');
    display_label = strrep(display_label, '_', ' ');
    title({display_label, 'W upward-positive (10^{-6} m s^{-1})'}, 'Interpreter', 'tex');
    tmp_path = [tempname(fileparts(path)) '.png'];
    try
        exportgraphics(fig, tmp_path, 'Resolution', 180);
        if exist(path, 'file') == 2
            delete(path);
        end
        movefile(tmp_path, path, 'f');
    catch ME
        fallback_path = fullfile(fileparts(path), ['w_3d_section_' datestr(now, 'yyyymmdd_HHMMSS') '.png']);
        if exist(tmp_path, 'file') == 2
            movefile(tmp_path, fallback_path, 'f');
        end
        warning('Could not replace %s: %s. Wrote %s instead.', path, ME.message, fallback_path);
    end
    close(fig);
end
