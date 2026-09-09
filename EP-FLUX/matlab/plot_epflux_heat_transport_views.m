function plot_epflux_heat_transport_views(varargin)
%PLOT_EPFLUX_HEAT_TRANSPORT_VIEWS Plot W*T' sections and depth slices.

p = inputParser;
addParameter(p, 'InputRoot', '', @ischar);
addParameter(p, 'OutputRoot', '', @ischar);
addParameter(p, 'SliceDepths', [100, 300, 700, 1000, 1500, 1900], @isnumeric);
parse(p, varargin{:});
opt = p.Results;

if isempty(opt.InputRoot)
    error('InputRoot is required.');
end
if isempty(opt.OutputRoot)
    opt.OutputRoot = opt.InputRoot;
end
if ~exist(opt.OutputRoot, 'dir')
    mkdir(opt.OutputRoot);
end

polarities = {'cyclonic', 'anticyclonic'};
for ip = 1:numel(polarities)
    polarity = polarities{ip};
    in_file = fullfile(opt.InputRoot, polarity, 'cross_20N_1R', 'epflux_20n_validation.mat');
    if ~isfile(in_file)
        error('Missing diagnostic file: %s', in_file);
    end
    S = load(in_file, 'diagnostics');
    D = S.diagnostics;
    out_dir = fullfile(opt.OutputRoot, polarity, 'cross_20N_1R');
    if ~exist(out_dir, 'dir')
        mkdir(out_dir);
    end
    plot_one_polarity(D, polarity, opt.SliceDepths, out_dir);
end
end

function plot_one_polarity(D, polarity, slice_depths, out_dir)
X = double(D.X);
Y = double(D.Y);
depths = double(D.depth_m(:));
heat = double(D.Bz_heat_K_m_s);
if isvector(X) && isvector(Y)
    [X, Y] = meshgrid(X, Y);
end

rows = abs(Y(:, 1)) <= 0.25;
if ~any(rows)
    [~, mid] = min(abs(Y(:, 1)));
    rows(mid) = true;
end
section = squeeze(median(heat(rows, :, :), 1, 'omitnan'))';

fig1 = figure('Visible', 'off', 'Color', 'w', 'Position', [100, 100, 900, 760]);
lim = robust_limit(section);
contourf(X(1, :), depths, section, 35, 'LineStyle', 'none');
set(gca, 'YDir', 'reverse');
colormap(gca, redblue_colormap(256));
caxis([-lim, lim]);
colorbar;
xlabel('x/R');
ylabel('Depth (m)');
title(sprintf('%s 20N W T'' section (K m s^-1)', polarity), 'Interpreter', 'none');
exportgraphics(fig1, fullfile(out_dir, 'heat_transport_section_x.png'), 'Resolution', 240);
close(fig1);

fig2 = figure('Visible', 'off', 'Color', 'w', 'Position', [100, 100, 1400, 900]);
tiledlayout(2, 3, 'TileSpacing', 'compact', 'Padding', 'compact');
lim3d = robust_limit(heat);
for k = 1:numel(slice_depths)
    [~, iz] = min(abs(depths - slice_depths(k)));
    nexttile;
    layer = heat(:, :, iz);
    contourf(X, Y, layer, 35, 'LineStyle', 'none');
    axis equal tight;
    colormap(gca, redblue_colormap(256));
    caxis([-lim3d, lim3d]);
    hold on;
    th = linspace(0, 2*pi, 300);
    plot(cos(th), sin(th), 'k-', 'LineWidth', 1.2);
    plot(4*cos(th), 4*sin(th), 'k-', 'LineWidth', 1.2);
    plot(0, 0, 'k.', 'MarkerSize', 18);
    xlabel('x/R');
    ylabel('y/R');
    title(sprintf('%g m', depths(iz)));
end
cb = colorbar;
cb.Layout.Tile = 'east';
sgtitle(sprintf('%s 20N W T'' depth slices (K m s^-1)', polarity), 'Interpreter', 'none');
exportgraphics(fig2, fullfile(out_dir, 'heat_transport_depth_slices.png'), 'Resolution', 240);
close(fig2);

write_summary(D, polarity, section, heat, out_dir);
end

function lim = robust_limit(F)
vals = abs(F(isfinite(F)));
if isempty(vals)
    lim = 1;
else
    lim = prctile(vals, 95);
    if ~isfinite(lim) || lim <= 0
        lim = max(vals);
    end
    if ~isfinite(lim) || lim <= 0
        lim = 1;
    end
end
end

function write_summary(D, polarity, section, heat, out_dir)
fid = fopen(fullfile(out_dir, 'HEAT_TRANSPORT_VIEWS_ZH.md'), 'w');
cleanup = onCleanup(@() fclose(fid));
fprintf(fid, '# 20N 垂直热输送视图：%s\n\n', polarity);
fprintf(fid, '这里展示的是当前理论链条中的三维 `W T''` 场本身，不讨论其是否闭合解释水平动量通量。\n\n');
fprintf(fid, '- `W`：向上为正。\n');
fprintf(fid, '- `T'' = -rho''/(rho_ref alpha)`：由线性 EOS 从密度异常反推。\n');
fprintf(fid, '- `W T''` 单位：`K m s^-1`；若后续乘以 `rho0 Cp`，可转换为热通量量纲。\n');
fprintf(fid, '- 截面：沿 `|y/R| <= 0.25` 做中位数。\n\n');
fprintf(fid, '## 量级\n\n');
fprintf(fid, '- section q95(|WT''|) = %.4g K m s^-1。\n', robust_limit(section));
fprintf(fid, '- full 3D q95(|WT''|) = %.4g K m s^-1。\n\n', robust_limit(heat));
fprintf(fid, '## 图像\n\n');
fprintf(fid, '![section](%s)\n\n', fullfile(out_dir, 'heat_transport_section_x.png'));
fprintf(fid, '![slices](%s)\n', fullfile(out_dir, 'heat_transport_depth_slices.png'));
end

function cmap = redblue_colormap(n)
if nargin < 1
    n = 256;
end
x = linspace(-1, 1, n)';
r = min(1, max(0, 1.5 + 1.5*x));
b = min(1, max(0, 1.5 - 1.5*x));
g = 1 - abs(x);
g = 0.92 * max(g, 0);
cmap = [r, g, b];
end
