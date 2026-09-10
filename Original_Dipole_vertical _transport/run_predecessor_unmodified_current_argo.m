function run_predecessor_unmodified_current_argo(varargin)
%RUN_PREDECESSOR_UNMODIFIED_CURRENT_ARGO Feed current Argo composites to the
%unmodified predecessor scripts, then plot sections/slices.

p = inputParser;
addParameter(p, 'SourceRoot', 'E:\DATA\01_Eddy_correspond\01_Vertical_asymmetric\META4_CoreArgo_W_3D_20N_repeat_all_recommended_thermalwind', @ischar);
addParameter(p, 'ReferenceRoot', 'E:\DATA\01_Eddy_correspond\01_Vertical_asymmetric\META4_CoreArgo_W_3D_20N_reference_like_reversal_worktree', @ischar);
addParameter(p, 'OutputRoot', 'E:\DATA\01_Eddy_correspond\05_Original_Dipole_vertical _transport\predecessor_unmodified_current_argo_20N', @ischar);
addParameter(p, 'Polarities', {'cyclonic','anticyclonic'});
parse(p, varargin{:});
opt = p.Results;

repo_dir = fileparts(mfilename('fullpath'));
addpath(fullfile(repo_dir, 'matlab_compat'));
compat_roots = {fullfile('D:\Users\Root'), fullfile('E:\Users\Root')};
for k_root = 1:numel(compat_roots)
    prepare_fixed_paths(compat_roots{k_root});
end

if ~exist(opt.OutputRoot, 'dir')
    mkdir(opt.OutputRoot);
end

polarities = opt.Polarities;
for ip = 1:numel(polarities)
    polarity = polarities{ip};
    fprintf('Preparing predecessor-compatible inputs for %s...\n', polarity);
    src_file = fullfile(opt.SourceRoot, polarity, 'cross_20N_1R', 'w_3d_grid.mat');
    ref_file = fullfile(opt.ReferenceRoot, polarity, 'cross_20N_1R', 'reference_like_reversal_terms.mat');
    if ~isfile(src_file)
        error('Missing source grid: %s', src_file);
    end
    if ~isfile(ref_file)
        error('Missing reference-like density grid: %s', ref_file);
    end
    out_dir = fullfile(opt.OutputRoot, polarity, 'cross_20N_1R');
    if ~exist(out_dir, 'dir')
        mkdir(out_dir);
    end
    for k_root = 1:numel(compat_roots)
        prepare_compat_inputs(src_file, ref_file, compat_roots{k_root});
    end
    copyfile(fullfile(repo_dir, 'rebuild_eddy_W.m'), fullfile(out_dir, 'rebuild_eddy_W.m'));
    copyfile(fullfile(repo_dir, 'test01_dzdt_induced_W.m'), fullfile(out_dir, 'test01_dzdt_induced_W.m'));
    copyfile(fullfile(repo_dir, 'test02_UV_induced_W.m'), fullfile(out_dir, 'test02_UV_induced_W.m'));
    old_dir = pwd;
    cleanup = onCleanup(@() cd(old_dir));
    cd(out_dir);
    addpath(fullfile(repo_dir, 'matlab_compat'));
    run(fullfile(out_dir, 'rebuild_eddy_W.m'));
    clear cleanup
    plot_outputs(out_dir, polarity);
end
write_root_doc(opt.OutputRoot);
end

function prepare_fixed_paths(root_dir)
dirs = {
    fullfile(root_dir, 'Output', 'Eddy Heat Flux')
    fullfile(root_dir, 'Output', 'Eddy Heat Flux', 'eddy_features', 'eddy_moving_speed')
    fullfile(root_dir, 'Output', 'Eddy Heat Flux', 'rebuild_W', 'UV_induced_W')
    fullfile(root_dir, 'Data', 'AVISO_Eddy', 'META3.2_DT_twosat')
    fullfile(root_dir, 'Data', 'Argo_Data', 'ISAS_Argo', 'field', '2004')
    };
for k = 1:numel(dirs)
    if ~exist(dirs{k}, 'dir')
        mkdir(dirs{k});
    end
end
end

function prepare_compat_inputs(src_file, ref_file, root_dir)
S = load(src_file, 'grid3d');
G = S.grid3d;
R = load(ref_file, 'result');
rho_abs_source = double(R.result.rho_abs_smoothed);
Depth1 = (0:25:2000)';
depth_isas = linspace(0, 2000, 152)';
[Den_compound, U1000_compound, V1000_compound, W1000_compound, Den_compound_all] = resample_current_grid(G, rho_abs_source, Depth1, depth_isas);

E_radius = repmat(double(G.mean_radius_m), 200, 1);
E_mspeed = repmat(double(G.mean_cx_raw), 200, 1);
E_lat = repmat(20, 200, 1);
moving_speed_zonal = E_mspeed;

out_root = fullfile(root_dir, 'Output', 'Eddy Heat Flux');
save(fullfile(out_root, 'Argo02_data_point_ce_North_twosat.mat'), 'Depth1', 'E_lat', 'E_radius', 'E_mspeed');
save(fullfile(out_root, 'Argo02_data_point_ae_North_twosat.mat'), 'Depth1', 'E_lat', 'E_radius', 'E_mspeed');
save(fullfile(out_root, 'Argo03_compound_ce_North_res0.6_median.mat'), ...
    'Den_compound', 'U1000_compound', 'V1000_compound', 'W1000_compound', 'Depth1');
save(fullfile(out_root, 'Argo04_data_DenField_ce_North_twosat_ISAS_7Sample.mat'), 'Den_compound_all');
save(fullfile(out_root, 'eddy_features', 'eddy_moving_speed', 'Twosat_CE_eddy_zonal_moving_speed.mat'), 'moving_speed_zonal');

U_thw = fillnan_nearest(double(G.u_tw));
V_thw = fillnan_nearest(double(G.v_tw));
[U_thw, V_thw] = resample_uv_to_original(G, U_thw, V_thw, Depth1);
save(fullfile(out_root, 'rebuild_W', 'UV_induced_W', 'AE_North_ThermalWind_UV_smooth1.mat'), 'U_thw', 'V_thw');

write_meta_nc(fullfile(root_dir, 'Data', 'AVISO_Eddy', 'META3.2_DT_twosat', 'META3.2_DT_twosat_Cyclonic_long_19930101_20220209.nc'), E_radius, E_lat);
write_isas_nc(fullfile(root_dir, 'Data', 'Argo_Data', 'ISAS_Argo', 'field', '2004', 'ISAS20_ARGO_20040615_fld_TEMP.nc'), depth_isas);
end

function [Den81, U1000, V1000, W1000, DenAll] = resample_current_grid(G, rho_abs_source, depth1, depth_isas)
x_old = double(G.x(1,:));
y_old = double(G.y(:,1));
z_old = double(G.depth_levels(:));
x_new = linspace(-4, 4, 81);
y_new = linspace(-4, 4, 81);
[Yq, Xq, Zq] = ndgrid(y_new, x_new, depth1);
rho_abs = fillnan_nearest(rho_abs_source);
F = griddedInterpolant({y_old, x_old, z_old}, rho_abs, 'linear', 'nearest');
Den81 = F(Yq, Xq, Zq);
Den81 = enforce_monotonic_depth(Den81);

[Y2, X2, Z2] = ndgrid(y_new, x_new, depth_isas);
DenAll = F(Y2, X2, Z2);
DenAll = enforce_monotonic_depth(DenAll);
DenAll = reshape(DenAll, 81, 81, numel(depth_isas), 1);

iz = nearest_index(z_old, 1000);
U1000 = interp_layer(G, double(G.u_tw(:,:,iz)), x_new, y_new);
V1000 = interp_layer(G, double(G.v_tw(:,:,iz)), x_new, y_new);
W1000 = interp_layer(G, double(G.w(:,:,iz)), x_new, y_new);
end

function [U81, V81] = resample_uv_to_original(G, U, V, depth1)
x_old = double(G.x(1,:));
y_old = double(G.y(:,1));
z_old = double(G.depth_levels(:));
x_new = linspace(-4, 4, 81);
y_new = linspace(-4, 4, 81);
[Yq, Xq, Zq] = ndgrid(y_new, x_new, depth1);
FU = griddedInterpolant({y_old, x_old, z_old}, U, 'linear', 'nearest');
FV = griddedInterpolant({y_old, x_old, z_old}, V, 'linear', 'nearest');
U81 = FU(Yq, Xq, Zq);
V81 = FV(Yq, Xq, Zq);
end

function layer_new = interp_layer(G, layer, x_new, y_new)
x_old = double(G.x(1,:));
y_old = double(G.y(:,1));
layer = fillnan_nearest(layer);
F = griddedInterpolant({y_old, x_old}, layer, 'linear', 'nearest');
[Yq, Xq] = ndgrid(y_new, x_new);
layer_new = F(Yq, Xq);
end

function A = fillnan_nearest(A)
if all(isfinite(A(:)))
    return
end
for dim = 1:ndims(A)
    A = fillmissing(A, 'nearest', dim, 'EndValues', 'nearest');
end
A(~isfinite(A)) = median(A(isfinite(A)), 'omitnan');
if any(~isfinite(A(:)))
    A(~isfinite(A)) = 0;
end
end

function A = enforce_monotonic_depth(A)
for k = 2:size(A, 3)
    prev = A(:,:,k-1);
    cur = A(:,:,k);
    bad = cur <= prev;
    cur(bad) = prev(bad) + 1e-6;
    A(:,:,k) = cur;
end
end

function idx = nearest_index(values, target)
[~, idx] = min(abs(values(:) - target));
end

function write_meta_nc(path, radius, lat)
if isfile(path)
    delete(path);
end
nccreate(path, 'speed_radius', 'Dimensions', {'obs', numel(radius)});
nccreate(path, 'latitude', 'Dimensions', {'obs', numel(lat)});
ncwrite(path, 'speed_radius', double(radius(:)));
ncwrite(path, 'latitude', double(lat(:)));
end

function write_isas_nc(path, depth)
if isfile(path)
    delete(path);
end
nccreate(path, 'depth', 'Dimensions', {'depth', numel(depth)});
ncwrite(path, 'depth', double(depth(:)));
end

function plot_outputs(out_dir, polarity)
S = load(fullfile(out_dir, 'test_AE_North_rebuild_W.mat'), 'W_is', 'W_dzdt', 'W', 'Depth1');
depth = double(S.Depth1(:));
x = linspace(-4, 4, size(S.W, 2));
y = linspace(-4, 4, size(S.W, 1));
rows = abs(y) <= 0.25;
if ~any(rows)
    [~, mid] = min(abs(y));
    rows(mid) = true;
end
sec_dzdt = squeeze(median(S.W_dzdt(rows,:,:), 1, 'omitnan'))';
sec_is = squeeze(median(S.W_is(rows,:,:), 1, 'omitnan'))';
sec_w = squeeze(median(S.W(rows,:,:), 1, 'omitnan'))';
iz1000 = nearest_index(depth, 1000);

fig = figure('Visible', 'off', 'Color', 'w', 'Position', [100, 100, 1500, 900]);
tiledlayout(2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
plot_section(x, depth, sec_dzdt, 'W_{dzdt}=c dz_\rho/dx');
plot_section(x, depth, sec_is, 'W_{is}=U dz_\rho/dx+V dz_\rho/dy');
plot_section(x, depth, sec_w, 'W rebuilt');
plot_slice(x, y, S.W(:,:,iz1000), sprintf('W slice %.0f m', depth(iz1000)));
sgtitle(sprintf('%s original scripts on current Argo-compatible inputs', polarity), 'Interpreter', 'none');
exportgraphics(fig, fullfile(out_dir, 'original_unmodified_4panel.png'), 'Resolution', 240);
close(fig);

slice_depths = [100, 300, 700, 1000, 1500, 1900];
fig2 = figure('Visible', 'off', 'Color', 'w', 'Position', [100, 100, 1400, 900]);
tiledlayout(2, 3, 'TileSpacing', 'compact', 'Padding', 'compact');
lim = robust_lim(S.W);
for k = 1:numel(slice_depths)
    [~, iz] = min(abs(depth - slice_depths(k)));
    nexttile;
    contourf(x, y, S.W(:,:,iz) .* 1e6, 35, 'LineStyle', 'none');
    axis equal tight;
    caxis([-lim, lim] .* 1e6);
    colormap(gca, redblue_colormap(256));
    colorbar;
    hold on;
    th = linspace(0, 2*pi, 300);
    plot(cos(th), sin(th), 'k-', 'LineWidth', 1.1);
    plot(4*cos(th), 4*sin(th), 'k-', 'LineWidth', 1.1);
    plot(0, 0, 'k.', 'MarkerSize', 16);
    xlabel('x/R');
    ylabel('y/R');
    title(sprintf('%.0f m', depth(iz)));
end
sgtitle(sprintf('%s original-script W slices (10^{-6} m s^{-1})', polarity), 'Interpreter', 'none');
exportgraphics(fig2, fullfile(out_dir, 'original_unmodified_w_slices.png'), 'Resolution', 240);
close(fig2);

write_doc(out_dir, polarity, S);
end

function plot_section(x, depth, F, ttl)
nexttile;
lim = robust_lim(F);
contourf(x, depth, F .* 1e6, 35, 'LineStyle', 'none');
set(gca, 'YDir', 'reverse');
caxis([-lim, lim] .* 1e6);
colormap(gca, redblue_colormap(256));
colorbar;
xlabel('x/R');
ylabel('Depth (m)');
title([ttl ' (10^{-6} m s^{-1})']);
end

function plot_slice(x, y, F, ttl)
nexttile;
lim = robust_lim(F);
contourf(x, y, F .* 1e6, 35, 'LineStyle', 'none');
axis equal tight;
caxis([-lim, lim] .* 1e6);
colormap(gca, redblue_colormap(256));
colorbar;
hold on;
th = linspace(0, 2*pi, 300);
plot(cos(th), sin(th), 'k-', 'LineWidth', 1.1);
plot(4*cos(th), 4*sin(th), 'k-', 'LineWidth', 1.1);
plot(0, 0, 'k.', 'MarkerSize', 16);
xlabel('x/R');
ylabel('y/R');
title([ttl ' (10^{-6} m s^{-1})']);
end

function lim = robust_lim(F)
vals = abs(F(isfinite(F)));
if isempty(vals)
    lim = 1e-6;
else
    lim = prctile(vals, 95);
    if ~isfinite(lim) || lim <= 0
        lim = max(vals);
    end
    if ~isfinite(lim) || lim <= 0
        lim = 1e-6;
    end
end
end

function cmap = redblue_colormap(n)
if nargin < 1
    n = 256;
end
if mod(n, 2) ~= 0
    n = n + 1;
end
r = [(0:(n/2-1))/(n/2), ones(1,n/2)];
g = [(0:(n/2-1))/(n/2), (n/2-1:-1:0)/(n/2)];
b = [ones(1,n/2), (n/2-1:-1:0)/(n/2)];
cmap = [r(:), g(:), b(:)];
end

function write_doc(out_dir, polarity, S)
fid = fopen(fullfile(out_dir, 'ORIGINAL_UNMODIFIED_RUN_ZH.md'), 'w');
cleanup = onCleanup(@() fclose(fid));
fprintf(fid, '# 前辈原始程序兼容运行：%s\n\n', polarity);
fprintf(fid, '本目录结果来自未修改的 `rebuild_eddy_W.m`。外部步骤只负责把我们当前 20N Argo 三维产品转换为前辈脚本硬编码路径需要的同名中间 MAT/NC 文件，并提供兼容 `ndnanfilter.m`。\n\n');
fprintf(fid, '## 输出\n\n');
fprintf(fid, '- `test_AE_North_rebuild_W.mat`：原脚本直接输出。\n');
fprintf(fid, '- `original_unmodified_4panel.png`：`W_dzdt`、`W_is`、`W` 东西向截面与 1000 m 水平 slice。\n');
fprintf(fid, '- `original_unmodified_w_slices.png`：多深度水平 slice。\n\n');
fprintf(fid, '## 量级\n\n');
fprintf(fid, '- `q95(|W_dzdt|)=%.4g m/s`\n', robust_lim(S.W_dzdt));
fprintf(fid, '- `q95(|W_is|)=%.4g m/s`\n', robust_lim(S.W_is));
fprintf(fid, '- `q95(|W|)=%.4g m/s`\n', robust_lim(S.W));
end

function write_root_doc(output_root)
fid = fopen(fullfile(output_root, 'README_ORIGINAL_UNMODIFIED_CURRENT_ARGO_ZH.md'), 'w');
cleanup = onCleanup(@() fclose(fid));
fprintf(fid, '# 前辈原始程序套用当前 Argo 数据\n\n');
fprintf(fid, '本目录用于回答：如果不修改前辈三个程序，而把我们当前 Argo 20N 三维产品转换成它们所需的同名中间输入，输出图像长什么样。\n\n');
fprintf(fid, '注意：这不是从原始 Argo/META 重新跑前辈完整前置流程，因为前辈三个程序本身不包含 Argo-META 匹配和 composite 生成代码。\n');
end
