function export_png_safe(fig, path, resolution)
    if nargin < 3 || ~isfinite(resolution)
        resolution = 180;
    end
    folder = fileparts(path);
    if exist(folder, 'dir') ~= 7
        mkdir(folder);
    end
    if exist(path, 'file') == 2
        try
            delete(path);
        catch
            pause(0.2);
        end
    end
    try
        exportgraphics(fig, path, 'Resolution', resolution);
    catch ME
        warning('exportgraphics failed for %s, falling back to print: %s', path, ME.message);
        print(fig, path, '-dpng', ['-r' num2str(resolution)]);
    end
end
