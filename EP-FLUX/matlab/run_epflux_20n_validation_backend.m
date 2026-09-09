function run_epflux_20n_validation_backend(varargin)
%RUN_EPFLUX_20N_VALIDATION_BACKEND First-pass material-volume EP-flux check.
%
% The diagnostic intentionally computes a local proxy budget from existing
% 20N W/rho/u/v grids. It does not claim to solve the non-local QG inversion
% term L_x^B or the material momentum tendency G_x.

p = inputParser;
addParameter(p, 'ThermalRoot', '', @ischar);
addParameter(p, 'ReferenceRoot', '', @ischar);
addParameter(p, 'OutputRoot', '', @ischar);
addParameter(p, 'DepthLevels', '10:10:2000', @ischar);
addParameter(p, 'MaxDepthLevels', 0, @isnumeric);
addParameter(p, 'BootstrapN', 300, @isnumeric);
addParameter(p, 'Alpha', 2.0e-4, @isnumeric);
addParameter(p, 'RhoRef', 1025.0, @isnumeric);
parse(p, varargin{:});
opt = p.Results;

if isempty(opt.OutputRoot)
    error('OutputRoot is required.');
end
if ~exist(opt.OutputRoot, 'dir')
    mkdir(opt.OutputRoot);
end

requested_depths = parse_depth_levels(opt.DepthLevels);
if opt.MaxDepthLevels > 0 && numel(requested_depths) > opt.MaxDepthLevels
    requested_depths = requested_depths(1:opt.MaxDepthLevels);
end

polarities = {'cyclonic', 'anticyclonic'};
summary = cell(numel(polarities), 1);

for ip = 1:numel(polarities)
    polarity = polarities{ip};
    fprintf('Running EP-flux validation for %s...\n', polarity);
    summary{ip} = run_one_polarity(polarity, opt, requested_depths);
end

summary_table = struct2table([summary{:}]);
summary_csv = fullfile(opt.OutputRoot, 'EP_FLUX_20N_VALIDATION_SUMMARY.csv');
writetable(summary_table, summary_csv);
write_root_markdown(opt.OutputRoot, summary_table, opt);
fprintf('EP-flux validation finished: %s\n', opt.OutputRoot);
end

function out = run_one_polarity(polarity, opt, requested_depths)
src_dir = fullfile(opt.ThermalRoot, polarity, 'cross_20N_1R');
grid_file = fullfile(src_dir, 'w_3d_grid.mat');
if ~isfile(grid_file)
    error('Missing thermal-wind grid: %s', grid_file);
end

S = load(grid_file, 'grid3d');
G = S.grid3d;

depth_all = double(G.depth_levels(:));
[depths, idx] = select_depths(depth_all, requested_depths);

W = double(G.w(:, :, idx));
U = double(G.u_tw(:, :, idx));
V = double(G.v_tw(:, :, idx));
rho_anom = double(G.rho_anom(:, :, idx));
X = double(G.x);
Y = double(G.y);
if isvector(X) && isvector(Y)
    [X, Y] = meshgrid(X, Y);
end

radius_m = double(G.mean_radius_m);
if ~isfinite(radius_m) || radius_m <= 0
    radius_m = 1;
end
dx_m = median(diff(X(1, :)), 'omitnan') * radius_m;
dy_m = median(diff(Y(:, 1)), 'omitnan') * radius_m;
if ~isfinite(dx_m) || dx_m == 0 || ~isfinite(dy_m) || dy_m == 0
    error('Invalid x/y spacing in %s', grid_file);
end
if numel(depths) > 1
    dD_m = median(diff(depths), 'omitnan');
else
    dD_m = 1;
end

Rnorm = hypot(X, Y);
inside = Rnorm <= 4;
farfield = Rnorm >= 2 & Rnorm <= 4;

Uprime = remove_depthwise_farfield(U, farfield);
Vprime = remove_depthwise_farfield(V, farfield);
Tprime = -rho_anom ./ (double(opt.RhoRef) * double(opt.Alpha));
bprime = -9.81 .* rho_anom ./ double(opt.RhoRef);

Rxz = Uprime .* W;
Rxy = Uprime .* Vprime;
Rxx = Uprime .* Uprime;
Bz_heat = W .* Tprime;
Bz_buoy = W .* bprime;

[dRxy_dy, ~, ~] = gradient(Rxy, dy_m, dx_m, dD_m);
[~, dRxx_dx, ~] = gradient(Rxx, dy_m, dx_m, dD_m);
[~, ~, dRxz_dD] = gradient(Rxz, dy_m, dx_m, dD_m);

% z is positive upward, D is positive downward: d/dz = -d/dD.
AxR_partial = -dRxy_dy + dRxz_dD;
AxR_full = -dRxx_dx - dRxy_dy + dRxz_dD;

[beta_heat_partial, residual_heat_partial, corr_heat_partial, ci_heat_partial] = regress_field(AxR_partial, Bz_heat, inside, opt.BootstrapN);
[beta_buoy_partial, residual_buoy_partial, corr_buoy_partial, ci_buoy_partial] = regress_field(AxR_partial, Bz_buoy, inside, opt.BootstrapN);
[beta_heat_full, residual_heat_full, corr_heat_full, ci_heat_full] = regress_field(AxR_full, Bz_heat, inside, opt.BootstrapN);
[beta_buoy_full, residual_buoy_full, corr_buoy_full, ci_buoy_full] = regress_field(AxR_full, Bz_buoy, inside, opt.BootstrapN);

out_dir = fullfile(opt.OutputRoot, polarity, 'cross_20N_1R');
if ~exist(out_dir, 'dir')
    mkdir(out_dir);
end

section = make_sections(X, Y, depths, Rxz, AxR_full, Bz_heat, residual_heat_full);
fig_path = fullfile(out_dir, 'epflux_20n_validation_4panel.png');
plot_four_panel(section, polarity, fig_path);

mat_path = fullfile(out_dir, 'epflux_20n_validation.mat');
diagnostics = struct();
diagnostics.X = X;
diagnostics.Y = Y;
diagnostics.depth_m = depths;
diagnostics.W_upward_m_s = W;
diagnostics.Uprime_m_s = Uprime;
diagnostics.Vprime_m_s = Vprime;
diagnostics.Tprime_C_proxy = Tprime;
diagnostics.bprime_m_s2 = bprime;
diagnostics.Rxz_m2_s2 = Rxz;
diagnostics.Rxy_m2_s2 = Rxy;
diagnostics.Rxx_m2_s2 = Rxx;
diagnostics.Bz_heat_K_m_s = Bz_heat;
diagnostics.Bz_buoy_m3_s4 = Bz_buoy;
diagnostics.AxR_partial_m_s2 = AxR_partial;
diagnostics.AxR_full_m_s2 = AxR_full;
diagnostics.beta_heat_to_AxR_partial = beta_heat_partial;
diagnostics.residual_heat_partial_m_s2 = residual_heat_partial;
diagnostics.beta_buoy_to_AxR_partial = beta_buoy_partial;
diagnostics.residual_buoy_partial_m_s2 = residual_buoy_partial;
diagnostics.beta_heat_to_AxR_full = beta_heat_full;
diagnostics.residual_heat_full_m_s2 = residual_heat_full;
diagnostics.beta_buoy_to_AxR_full = beta_buoy_full;
diagnostics.residual_buoy_full_m_s2 = residual_buoy_full;
diagnostics.source_grid_file = grid_file;
diagnostics.notes = 'AxR_full = -dRxx/dx - dRxy/dy + dRxz/dD. Gx and nonlocal LxB are not available in this check.';
save(mat_path, 'diagnostics', '-v7.3');

write_polarity_markdown(out_dir, polarity, grid_file, fig_path, ...
    corr_heat_partial, ci_heat_partial, corr_buoy_partial, ci_buoy_partial, ...
    beta_heat_partial, beta_buoy_partial, AxR_partial, residual_heat_partial, ...
    corr_heat_full, ci_heat_full, corr_buoy_full, ci_buoy_full, ...
    beta_heat_full, beta_buoy_full, AxR_full, Bz_heat, residual_heat_full, inside);

out = struct();
out.polarity = string(polarity);
out.depth_count = numel(depths);
out.match_count = get_optional_scalar(G, 'match_count');
out.unique_argo_count = get_optional_scalar(G, 'unique_argo_count');
out.duplicate_match_count = get_optional_scalar(G, 'duplicate_match_count');
out.valid_fraction = nnz(isfinite(W) & repmat(inside, 1, 1, numel(depths))) / nnz(repmat(inside, 1, 1, numel(depths)));
out.corr_AxR_partial_WT = corr_heat_partial;
out.corr_AxR_partial_Wb = corr_buoy_partial;
out.beta_WT_to_AxR_partial = beta_heat_partial;
out.residual_partial_heat_q95_over_AxR_q95 = q95_abs(residual_heat_partial, inside) / max(q95_abs(AxR_partial, inside), eps);
out.corr_AxR_full_WT = corr_heat_full;
out.corr_AxR_full_Wb = corr_buoy_full;
out.beta_WT_to_AxR_full = beta_heat_full;
out.residual_full_heat_q95_over_AxR_q95 = q95_abs(residual_heat_full, inside) / max(q95_abs(AxR_full, inside), eps);
out.Rxz_q95_m2_s2 = q95_abs(Rxz, inside);
out.Rxx_q95_m2_s2 = q95_abs(Rxx, inside);
out.AxR_partial_q95_m_s2 = q95_abs(AxR_partial, inside);
out.AxR_full_q95_m_s2 = q95_abs(AxR_full, inside);
out.WT_q95_K_m_s = q95_abs(Bz_heat, inside);
out.figure = string(fig_path);
out.mat_file = string(mat_path);
end

function depths = parse_depth_levels(text)
parts = split(string(text), ':');
if numel(parts) == 3
    a = str2double(parts(1));
    b = str2double(parts(2));
    c = str2double(parts(3));
    depths = (a:b:c)';
else
    depths = str2double(split(string(text), ','));
end
depths = depths(isfinite(depths));
if isempty(depths)
    error('No valid depth levels parsed from %s', text);
end
end

function [selected_depths, idx] = select_depths(depth_all, requested)
idx = zeros(numel(requested), 1);
for k = 1:numel(requested)
    [~, idx(k)] = min(abs(depth_all - requested(k)));
end
idx = unique(idx, 'stable');
selected_depths = depth_all(idx);
end

function Fprime = remove_depthwise_farfield(F, farfield)
Fprime = F;
for k = 1:size(F, 3)
    layer = F(:, :, k);
    bg = median(layer(farfield & isfinite(layer)), 'omitnan');
    if ~isfinite(bg)
        bg = 0;
    end
    Fprime(:, :, k) = layer - bg;
end
end

function [beta, residual, c, ci] = regress_field(target, predictor, mask2d, bootstrap_n)
mask3d = repmat(mask2d, 1, 1, size(target, 3));
ok = mask3d & isfinite(target) & isfinite(predictor);
a = target(ok);
b = predictor(ok);
if numel(a) < 10 || all(abs(b) < eps)
    beta = NaN;
    residual = NaN(size(target));
    c = NaN;
    ci = [NaN, NaN];
    return;
end
beta = (b(:)' * a(:)) / (b(:)' * b(:));
residual = target - beta .* predictor;
c = corr(a(:), b(:), 'rows', 'complete');
ci = bootstrap_corr_ci(a(:), b(:), bootstrap_n);
end

function ci = bootstrap_corr_ci(a, b, nboot)
if nboot <= 0 || numel(a) < 20
    ci = [NaN, NaN];
    return;
end
rng(20260909);
n = numel(a);
vals = NaN(nboot, 1);
for ib = 1:nboot
    pick = randi(n, n, 1);
    vals(ib) = corr(a(pick), b(pick), 'rows', 'complete');
end
vals = vals(isfinite(vals));
if isempty(vals)
    ci = [NaN, NaN];
else
    ci = prctile(vals, [2.5, 97.5]);
end
end

function section = make_sections(X, Y, depths, Rxz, AxR, WT, residual)
rows = abs(Y(:, 1)) <= 0.25;
if ~any(rows)
    [~, mid] = min(abs(Y(:, 1)));
    rows(mid) = true;
end
section.x = X(1, :);
section.depths = depths(:);
section.Rxz = squeeze(median(Rxz(rows, :, :), 1, 'omitnan'))';
section.AxR = squeeze(median(AxR(rows, :, :), 1, 'omitnan'))';
section.WT = squeeze(median(WT(rows, :, :), 1, 'omitnan'))';
section.residual = squeeze(median(residual(rows, :, :), 1, 'omitnan'))';
end

function plot_four_panel(section, polarity, fig_path)
fig = figure('Visible', 'off', 'Color', 'w', 'Position', [100, 100, 1500, 820]);
tiledlayout(2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
plot_one(section, section.Rxz, 'R_{xz}=u''W (m^2 s^{-2})');
plot_one(section, section.AxR, 'A_x^R full (m s^{-2})');
plot_one(section, section.WT, 'W T'' (K m s^{-1})');
plot_one(section, section.residual, 'A_x^R - \beta WT'' (m s^{-2})');
sgtitle(sprintf('%s 20N EP-flux validation proxy', polarity), 'Interpreter', 'none');
exportgraphics(fig, fig_path, 'Resolution', 220);
close(fig);
end

function plot_one(section, F, title_text)
nexttile;
vals = abs(F(isfinite(F)));
if isempty(vals)
    lim = 1;
else
    lim = prctile(vals, 95);
    if ~isfinite(lim) || lim == 0
        lim = 1;
    end
end
contourf(section.x, section.depths, F, 31, 'LineStyle', 'none');
set(gca, 'YDir', 'reverse');
colormap(gca, redblue_colormap(256));
caxis([-lim, lim]);
colorbar;
xlabel('x/R');
ylabel('Depth (m)');
title(title_text);
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

function v = get_optional_scalar(S, name)
if isfield(S, name)
    val = S.(name);
    if isnumeric(val) && ~isempty(val)
        v = double(val(1));
        return;
    end
end
v = NaN;
end

function v = q95_abs(F, mask2d)
mask3d = repmat(mask2d, 1, 1, size(F, 3));
vals = abs(F(mask3d & isfinite(F)));
if isempty(vals)
    v = NaN;
else
    v = prctile(vals, 95);
end
end

function write_polarity_markdown(out_dir, polarity, grid_file, fig_path, ...
    corr_heat_partial, ci_heat_partial, corr_buoy_partial, ci_buoy_partial, ...
    beta_heat_partial, beta_buoy_partial, AxR_partial, residual_heat_partial, ...
    corr_heat_full, ci_heat_full, corr_buoy_full, ci_buoy_full, ...
    beta_heat_full, beta_buoy_full, AxR_full, Bz_heat, residual_heat_full, inside)
fid = fopen(fullfile(out_dir, 'EP_FLUX_20N_VALIDATION_ZH.md'), 'w');
cleanup = onCleanup(@() fclose(fid));
fprintf(fid, '# EP-flux 20N 数值验证：%s\n\n', polarity);
fprintf(fid, '输入三维网格：`%s`\n\n', grid_file);
fprintf(fid, '本轮补入了完整可解析 Reynolds stress 强迫 `A_x^R = -partial_x R_xx - partial_y R_xy + partial_D R_xz`。但它仍然不是完整闭合验证，因为 `G_x` 和非局地 `L_x^B` 尚未定义和求解。\n\n');
fprintf(fid, '## 已计算量\n\n');
fprintf(fid, '- `R_xz = u''W`，其中 `u''` 为热成风延拓速度去除 2-4R 远场中位数后的异常。\n');
fprintf(fid, '- `R_xx = u''u''`，`R_xy = u''v''`，`R_xz = u''W`。\n');
fprintf(fid, '- `A_x^R(full) = -partial_x R_xx - partial_y R_xy + partial_D R_xz`。这里 `D` 为正深度向下，因 `z=-D`，所以垂向通量项写作 `+partial_D R_xz`。\n');
fprintf(fid, '- 同时保留 `A_x^R(partial) = -partial_y R_xy + partial_D R_xz`，用于比较补入 `-partial_x R_xx` 前后的变化。\n');
fprintf(fid, '- `T'' = -rho''/(rho_ref alpha)`，线性 EOS 反推温度异常；`W T''` 作为垂直热输运诊断量。\n');
fprintf(fid, '- `residual_proxy = A_x^R(full) - beta W T''`，其中 beta 为最小二乘回归系数。它只是热输运解释动量强迫的代理残差，不是真正的 QG 反演残差。\n\n');
fprintf(fid, '## 指标\n\n');
fprintf(fid, '- partial: `corr(A_x^R, W T'') = %.4g`，bootstrap 95%% CI `[%.4g, %.4g]`，`beta=%.4g`，`residual_q95/AxR_q95=%.4g`。\n', ...
    corr_heat_partial, ci_heat_partial(1), ci_heat_partial(2), beta_heat_partial, q95_abs(residual_heat_partial, inside) / max(q95_abs(AxR_partial, inside), eps));
fprintf(fid, '- full: `corr(A_x^R, W T'') = %.4g`，bootstrap 95%% CI `[%.4g, %.4g]`，`beta=%.4g`，`residual_q95/AxR_q95=%.4g`。\n', ...
    corr_heat_full, ci_heat_full(1), ci_heat_full(2), beta_heat_full, q95_abs(residual_heat_full, inside) / max(q95_abs(AxR_full, inside), eps));
fprintf(fid, '- partial: `corr(A_x^R, W b'') = %.4g`，bootstrap 95%% CI `[%.4g, %.4g]`，`beta=%.4g`。\n', corr_buoy_partial, ci_buoy_partial(1), ci_buoy_partial(2), beta_buoy_partial);
fprintf(fid, '- full: `corr(A_x^R, W b'') = %.4g`，bootstrap 95%% CI `[%.4g, %.4g]`，`beta=%.4g`。\n', corr_buoy_full, ci_buoy_full(1), ci_buoy_full(2), beta_buoy_full);
fprintf(fid, '\n');
fprintf(fid, '- `q95(|W T''|) = %.4g K m s^-1`。\n\n', q95_abs(Bz_heat, inside));
fprintf(fid, '## 图像\n\n');
fprintf(fid, '![EP-flux 20N validation](%s)\n\n', fig_path);
fprintf(fid, '## 解释限制\n\n');
fprintf(fid, '如果 `W T''` 与完整 `A_x^R` 的相关稳定，只能说明垂直热输运与可解析 Reynolds stress 强迫存在同位相/反位相关系。要验证理论闭合，还需要下一步定义 `G_x` 和非局地响应算子 `L_x^B=M^{-1}S_B`。\n');
end

function write_root_markdown(output_root, T, opt)
fid = fopen(fullfile(output_root, 'EP_FLUX_20N_VALIDATION_SUMMARY_ZH.md'), 'w');
cleanup = onCleanup(@() fclose(fid));
fprintf(fid, '# EP-flux 新理论 20N 数值验证汇总\n\n');
fprintf(fid, '本次使用已有 `20N crossing / 1R / match-mode all / recommended` 三维热成风 W 产品，只输出 cyclonic 和 anticyclonic，不生成 combined。\n\n');
fprintf(fid, '## 理论定位\n\n');
fprintf(fid, '验证目标不是证明 `W T''` 与水平动量通量简单相等，而是检查它们是否能在同一材料体预算中呈现稳定关系。本版已补入 `R_xx` 并计算完整可解析 `A_x^R`，但仍没有 `G_x` 目标量，也没有真正求解 QG/PV 非局地反演 `L_x^B`。\n\n');
fprintf(fid, '线性 EOS 采用 `T'' = -rho''/(rho_ref alpha)`，其中 `rho_ref = %.1f kg m^-3`，`alpha = %.3g K^-1`。\n\n', opt.RhoRef, opt.Alpha);
fprintf(fid, '## 数值结果\n\n');
for i = 1:height(T)
    fprintf(fid, '- %s: match=%g, unique=%g, corr(AxR_full,WT)=%.4g, residual_q95/AxR_q95=%.4g, figure=`%s`\n', ...
        T.polarity(i), T.match_count(i), T.unique_argo_count(i), T.corr_AxR_full_WT(i), ...
        T.residual_full_heat_q95_over_AxR_q95(i), T.figure(i));
end
fprintf(fid, '\n## 下一步\n\n');
fprintf(fid, '若要把这一步升级为理论闭合验证，还需要补两件事：定义材料体动量变化 `G_x`；实现 `L_x^B=M^{-1}S_B` 的 QG/PV 反演，而不是用 `W T''` 的线性回归代理它。\n');
end
