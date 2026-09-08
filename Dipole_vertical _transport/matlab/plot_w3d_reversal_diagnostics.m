function plot_w3d_reversal_diagnostics(input_root, output_root, target_lat)
%PLOT_W3D_REVERSAL_DIAGNOSTICS Diagnose vertical phase reversal in 3-D W.
%
% This reader does not recompute the pipeline. It inspects existing
% w_3d_grid.mat files and writes section plots for z'_rho, its zonal
% gradient, thermal-wind velocity, term1, term2, and rebuilt W.

if nargin < 1 || strlength(string(input_root)) == 0
    input_root = 'E:\DATA\01_Eddy_correspond\01_Vertical_asymmetric\META4_CoreArgo_W_3D_crossing_global_60S60N_10deg_repeat_all_recommended_thermalwind';
end
if nargin < 2 || strlength(string(output_root)) == 0
    output_root = fullfile(input_root, '_diagnostics_reversal');
end
if nargin < 3 || isempty(target_lat)
    target_lat = 20;
end

input_root = char(input_root);
output_root = char(output_root);
if ~exist(output_root, 'dir')
    mkdir(output_root);
end

label = crossing_label(target_lat);
polarities = {'cyclonic', 'anticyclonic'};
summary = cell(numel(polarities) + 1, 10);
summary(1,:) = {'polarity','lat_label','w_zero_crossing_count','term1_zero_crossing_count', ...
    'term2_zero_crossing_count','zonal_gradient_zero_crossing_count', ...
    'median_corr_w_vs_1000m','median_corr_zanom_vs_1000m', ...
    'q95_abs_w_1e6_m_s','output_png'};

for pp = 1:numel(polarities)
    polarity = polarities{pp};
    grid_path = fullfile(input_root, polarity, label, 'w_3d_grid.mat');
    if ~isfile(grid_path)
        warning('Missing grid file: %s', grid_path);
        continue;
    end

    S = load(grid_path, 'grid3d');
    G = S.grid3d;
    x = squeeze(G.x(1,:));
    y = squeeze(G.y(:,1));
    depth = G.depth_levels(:);
    radius_m = double(G.mean_radius_m);
    dx_m = median(diff(x), 'omitnan') * radius_m;
    dy_m = median(diff(y), 'omitnan') * radius_m;

    dzdx = nan(size(G.z_anom));
    for kk = 1:numel(depth)
        z_layer = G.z_anom(:,:,kk);
        if any(isfinite(z_layer(:)))
            [dzy, dzx] = gradient(z_layer, dy_m, dx_m);
            dzdx(:,:,kk) = dzx;
        end
    end

    y_width = double(G.section_half_width_r);
    if ~isfinite(y_width) || y_width <= 0
        y_width = 0.25;
    end
    y_mask = abs(y) <= y_width;

    sec_z = section_median(G.z_anom, y_mask);
    sec_dzdx = section_median(dzdx, y_mask);
    sec_u_rel = section_median(G.u_tw - G.mean_cx_raw, y_mask);
    sec_v = section_median(G.v_tw, y_mask);
    sec_t1 = section_median(G.term1, y_mask) * 1e6;
    sec_t2 = section_median(G.term2, y_mask) * 1e6;
    sec_w = section_median(G.w, y_mask) * 1e6;

    anchor = find_closest(depth, 1000);
    corr_w = depth_corr_to_anchor(sec_w, anchor);
    corr_z = depth_corr_to_anchor(sec_z, anchor);

    out_png = fullfile(output_root, [polarity '_' label '_reversal_diagnostics.png']);
    plot_one(out_png, x, depth, sec_z, sec_dzdx, sec_u_rel, sec_v, sec_t1, sec_t2, sec_w, corr_w, corr_z, polarity, target_lat);

    out_mat = fullfile(output_root, [polarity '_' label '_reversal_diagnostics.mat']);
    diagnostics = struct();
    diagnostics.x_over_R = x;
    diagnostics.depth_m = depth;
    diagnostics.z_rho_anom_section_m = sec_z;
    diagnostics.dzdx_section = sec_dzdx;
    diagnostics.u_rel_section_m_s = sec_u_rel;
    diagnostics.v_section_m_s = sec_v;
    diagnostics.term1_section_1e6_m_s = sec_t1;
    diagnostics.term2_section_1e6_m_s = sec_t2;
    diagnostics.w_section_1e6_m_s = sec_w;
    diagnostics.corr_w_vs_1000m = corr_w;
    diagnostics.corr_zanom_vs_1000m = corr_z;
    save(out_mat, 'diagnostics', '-v7.3');

    summary{pp+1,1} = polarity;
    summary{pp+1,2} = label;
    summary{pp+1,3} = count_depth_reversals(sec_w);
    summary{pp+1,4} = count_depth_reversals(sec_t1);
    summary{pp+1,5} = count_depth_reversals(sec_t2);
    summary{pp+1,6} = count_depth_reversals(sec_dzdx);
    summary{pp+1,7} = median(corr_w, 'omitnan');
    summary{pp+1,8} = median(corr_z, 'omitnan');
    summary{pp+1,9} = quantile(abs(sec_w(isfinite(sec_w))), 0.95);
    summary{pp+1,10} = out_png;
end

summary_path = fullfile(output_root, ['REVERSAL_DIAGNOSTICS_' label '.mat']);
save(summary_path, 'summary', '-v7.3');
write_markdown(fullfile(output_root, ['REVERSAL_DIAGNOSTICS_' label '_ZH.md']), summary, target_lat);
disp(['W reversal diagnostics written to: ' output_root]);
end

function label = crossing_label(lat)
abs_lat = abs(lat);
if lat < 0
    hemi = 'S';
elseif lat > 0
    hemi = 'N';
else
    hemi = 'N';
end
label = sprintf('cross_%02.0f%s_1R', abs_lat, hemi);
end

function idx = find_closest(v, target)
[~, idx] = min(abs(v(:) - target));
end

function section = section_median(field3d, y_mask)
tmp = field3d(y_mask,:,:);
section = squeeze(median(tmp, 1, 'omitnan'))';
end

function corr_by_depth = depth_corr_to_anchor(section, anchor)
nz = size(section, 1);
corr_by_depth = nan(nz, 1);
base = section(anchor, :);
for kk = 1:nz
    a = section(kk, :);
    good = isfinite(a) & isfinite(base);
    if nnz(good) >= 5
        C = corrcoef(a(good), base(good));
        corr_by_depth(kk) = C(1,2);
    end
end
end

function n = count_depth_reversals(section)
profile = median(section, 2, 'omitnan');
profile(abs(profile) < 1e-12) = NaN;
sgn = sign(profile);
sgn = sgn(isfinite(sgn));
if numel(sgn) < 2
    n = 0;
else
    n = sum(sgn(1:end-1) .* sgn(2:end) < 0);
end
end

function plot_one(path, x, depth, zsec, dzdxsec, urel, vsec, term1, term2, w, corr_w, corr_z, polarity, target_lat)
fig = figure('Visible', 'off', 'Color', 'w', 'Position', [100 100 1700 1150]);
tl = tiledlayout(fig, 3, 3, 'TileSpacing', 'compact', 'Padding', 'compact');

draw_panel(nexttile(tl), x, depth, zsec, 'z''_\rho anomaly (m)', symmetric_limit(zsec), 'm');
draw_panel(nexttile(tl), x, depth, dzdxsec, '\partial z''_\rho / \partial x', symmetric_limit(dzdxsec), '');
draw_panel(nexttile(tl), x, depth, urel * 100, 'u_{tw}-c_x^{raw} (cm s^{-1})', symmetric_limit(urel * 100), 'cm s^{-1}');
draw_panel(nexttile(tl), x, depth, vsec * 100, 'v_{tw} (cm s^{-1})', symmetric_limit(vsec * 100), 'cm s^{-1}');
draw_panel(nexttile(tl), x, depth, term1, 'term1 (10^{-6} m s^{-1})', symmetric_limit(term1), '10^{-6} m s^{-1}');
draw_panel(nexttile(tl), x, depth, term2, 'term2 (10^{-6} m s^{-1})', symmetric_limit(term2), '10^{-6} m s^{-1}');
draw_panel(nexttile(tl), x, depth, w, 'rebuild W upward+ (10^{-6} m s^{-1})', symmetric_limit(w), '10^{-6} m s^{-1}');

ax = nexttile(tl);
plot(ax, corr_w, depth, 'LineWidth', 1.8); hold(ax, 'on');
plot(ax, corr_z, depth, 'LineWidth', 1.8);
xline(ax, 0, 'k-');
set(ax, 'YDir', 'reverse');
ylim(ax, [min(depth) max(depth)]);
xlim(ax, [-1 1]);
grid(ax, 'on');
xlabel(ax, 'corr with 1000 m');
ylabel(ax, 'Depth (m)');
legend(ax, {'W','z''_\rho'}, 'Location', 'best');
title(ax, 'vertical phase');

ax = nexttile(tl);
plot(ax, median(w, 2, 'omitnan'), depth, 'k-', 'LineWidth', 1.8); hold(ax, 'on');
plot(ax, median(term1, 2, 'omitnan'), depth, 'Color', [0.75 0 0], 'LineWidth', 1.4);
plot(ax, median(term2, 2, 'omitnan'), depth, 'Color', [0 0.2 0.75], 'LineWidth', 1.4);
xline(ax, 0, 'k-');
set(ax, 'YDir', 'reverse');
ylim(ax, [min(depth) max(depth)]);
grid(ax, 'on');
xlabel(ax, 'section median');
ylabel(ax, 'Depth (m)');
legend(ax, {'W','term1','term2'}, 'Location', 'best');
title(ax, 'depth sign check');

sgtitle(tl, sprintf('%s %.0fN crossing W reversal diagnostics', polarity, abs(target_lat)), 'Interpreter', 'none');
exportgraphics(fig, path, 'Resolution', 180);
close(fig);
end

function lim = symmetric_limit(A)
vals = abs(A(isfinite(A)));
if isempty(vals)
    lim = 1;
else
    lim = quantile(vals, 0.98);
    if ~isfinite(lim) || lim <= 0
        lim = max(vals);
    end
    if ~isfinite(lim) || lim <= 0
        lim = 1;
    end
end
end

function draw_panel(ax, x, depth, data, title_text, lim, cb_label)
contourf(ax, x, depth, data, 24, 'LineStyle', 'none');
set(ax, 'YDir', 'reverse');
caxis(ax, [-lim lim]);
colormap(ax, redblue_colormap(48));
cb = colorbar(ax);
if strlength(string(cb_label)) > 0
    ylabel(cb, cb_label);
end
xlabel(ax, 'x/R');
ylabel(ax, 'Depth (m)');
title(ax, title_text, 'Interpreter', 'tex');
end

function cmap = redblue_colormap(n)
if nargin < 1
    n = 64;
end
x = linspace(0, 1, n)';
blue = [0.05 0.15 0.85];
white = [0.96 0.96 0.96];
red = [0.75 0 0.05];
cmap = zeros(n, 3);
for ii = 1:n
    if x(ii) <= 0.5
        t = x(ii) / 0.5;
        cmap(ii,:) = (1 - t) * blue + t * white;
    else
        t = (x(ii) - 0.5) / 0.5;
        cmap(ii,:) = (1 - t) * white + t * red;
    end
end
end

function write_markdown(path, summary, target_lat)
fid = fopen(path, 'w');
if fid < 0
    warning('Cannot write markdown: %s', path);
    return;
end
cleanup = onCleanup(@() fclose(fid));
fprintf(fid, '# %.0fN Crossing W 垂向反转诊断\n\n', abs(target_lat));
fprintf(fid, '本诊断只读取既有 `w_3d_grid.mat`，不重新计算 Argo-META 匹配、BOA 背景或 W。\n\n');
fprintf(fid, '## 判读重点\n\n');
fprintf(fid, '- 如果 `z''_rho anomaly` 或 `partial z''_rho / partial x` 随深度相位不变，term1 通常不会反转。\n');
fprintf(fid, '- 如果 `u_tw-c_x_raw` 没有跨零或剪切很弱，term2 也不容易制造深层反转。\n');
fprintf(fid, '- `vertical phase` 面板中，相关系数长期保持正值表示该变量相对 1000 m 没有发生主要相位翻转。\n\n');
fprintf(fid, '## 摘要\n\n');
fprintf(fid, '| polarity | W zero crossings | term1 zero crossings | term2 zero crossings | dzdx zero crossings | median corr W/1000m | median corr z/1000m | q95 W |\n');
fprintf(fid, '|---|---:|---:|---:|---:|---:|---:|---:|\n');
for rr = 2:size(summary,1)
    if isempty(summary{rr,1})
        continue;
    end
    fprintf(fid, '| %s | %.0f | %.0f | %.0f | %.0f | %.3g | %.3g | %.3g |\n', ...
        summary{rr,1}, summary{rr,3}, summary{rr,4}, summary{rr,5}, summary{rr,6}, ...
        summary{rr,7}, summary{rr,8}, summary{rr,9});
end
end
