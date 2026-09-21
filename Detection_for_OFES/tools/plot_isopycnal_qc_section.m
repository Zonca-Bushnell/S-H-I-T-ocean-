function plot_isopycnal_qc_section(input_mat, output_png, title_text, colorbar_label)
% Plot one QC-masked direct-isopycnal section with filled colors and contours.
% D is positive downward; NaN values remain blank and are not contoured.

if nargin < 4
    colorbar_label = 'D_{rho} - D0 (m)';
end

data = load(input_mat, 'x_over_r', 'depth_m', 'section_m');
fig = figure('Visible', 'off', 'Color', 'w', 'Position', [100, 100, 1200, 760]);
ax = axes(fig);
valid = abs(data.section_m(isfinite(data.section_m)));
if isempty(valid)
    error('plot_isopycnal_qc_section:NoValidData', 'The QC section has no finite values.');
end
limit = max(prctile(valid, 95), 1);
imagesc(ax, data.x_over_r, data.depth_m, data.section_m, [-limit, limit]);
set(ax, 'YDir', 'reverse');
axis(ax, 'tight');
hold(ax, 'on');
levels = linspace(-limit, limit, 11);
[contours, handles] = contour(ax, data.x_over_r, data.depth_m, data.section_m, levels, ...
    'Color', [0.18, 0.18, 0.18], 'LineWidth', 0.55);
contour(ax, data.x_over_r, data.depth_m, data.section_m, [0, 0], 'k', 'LineWidth', 1.0);
xline(ax, 0, 'k-', 'LineWidth', 0.7);
colormap(ax, redbluecmap(256));
cb = colorbar(ax);
cb.Label.String = colorbar_label;
xlabel(ax, 'x/R (east-west)');
ylabel(ax, 'Depth (m)');
title(ax, title_text, 'Interpreter', 'none');
exportgraphics(fig, output_png, 'Resolution', 180);
close(fig);
end

function cmap = redbluecmap(n)
base = [0.0196, 0.1882, 0.3804; 0.2627, 0.5765, 0.7647; 0.9686, 0.9686, 0.9686; 0.8392, 0.3765, 0.3020; 0.4039, 0.0000, 0.1216];
cmap = interp1(linspace(0, 1, size(base, 1)), base, linspace(0, 1, n));
end
