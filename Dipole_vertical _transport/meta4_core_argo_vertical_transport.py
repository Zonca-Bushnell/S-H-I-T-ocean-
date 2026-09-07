from __future__ import annotations

import argparse
import json
import math
import struct
import subprocess
import tempfile
import zipfile
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_ARGO_MAT = Path(r"F:\Argo_data\ArgoData_SA_CT_PT_PDen_sigma.mat")
DEFAULT_META_DIR = Path(r"F:\Eddy\Eddy\META4.0_DT_allsat")
DEFAULT_OUTPUT_ROOT = Path(r"E:\DATA\01_Eddy_correspond\01_Vertical_asymmetric\META4_CoreArgo_vertical_transport")
DEFAULT_BBOX = "120,145,20,35"
DEFAULT_LAT_BANDS = "20:25,25:30,30:35"


def parse_bbox(value: str) -> tuple[float, float, float, float]:
    parts = [float(item.strip()) for item in value.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("--bbox must be lon_min,lon_max,lat_min,lat_max")
    lon_min, lon_max, lat_min, lat_max = parts
    if lon_min >= lon_max or lat_min >= lat_max:
        raise argparse.ArgumentTypeError("--bbox ranges must be increasing")
    return lon_min, lon_max, lat_min, lat_max


def parse_lat_bands(value: str) -> list[tuple[float, float]]:
    bands: list[tuple[float, float]] = []
    for item in value.split(","):
        lo_text, hi_text = item.split(":")
        lo, hi = float(lo_text), float(hi_text)
        if lo >= hi:
            raise argparse.ArgumentTypeError("latitude bands must be increasing")
        bands.append((lo, hi))
    return bands


def matlab_quote(path: Path) -> str:
    return str(path).replace("\\", "\\\\").replace("'", "''")


def npy_bytes_2d(values: list[list[float]]) -> bytes:
    rows = len(values)
    cols = len(values[0]) if rows else 0
    header = f"{{'descr': '<f8', 'fortran_order': False, 'shape': ({rows}, {cols}), }}"
    padding = 16 - ((10 + len(header) + 1) % 16)
    header = header + (" " * padding) + "\n"
    out = bytearray()
    out.extend(b"\x93NUMPY")
    out.extend(b"\x01\x00")
    out.extend(struct.pack("<H", len(header)))
    out.extend(header.encode("latin1"))
    for row in values:
        for value in row:
            out.extend(struct.pack("<d", float(value) if value is not None else math.nan))
    return bytes(out)


def write_npz_from_grid_json(json_path: Path, npz_path: Path) -> None:
    grid = json.loads(json_path.read_text(encoding="utf-8"))
    arrays = {
        "x_over_R": grid["x_over_R"],
        "y_over_R": grid["y_over_R"],
        "z_rho_m": grid["z_rho_m"],
        "u_argo_m_s": grid["u_argo_m_s"],
        "v_argo_m_s": grid["v_argo_m_s"],
        "sample_count": grid["sample_count"],
        "term1_m_s": grid["term1_m_s"],
        "term2_m_s": grid["term2_m_s"],
        "rebuild_w_m_s": grid["rebuild_w_m_s"],
    }
    with zipfile.ZipFile(npz_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, values in arrays.items():
            zf.writestr(f"{name}.npy", npy_bytes_2d(values))
        zf.writestr("metadata.json", json.dumps(grid.get("metadata", {}), ensure_ascii=False, indent=2))


def run_matlab_pipeline(args: argparse.Namespace) -> list[Path]:
    with tempfile.TemporaryDirectory(prefix="meta4_core_argo_") as tmp:
        tmp_dir = Path(tmp)
        manifest_path = tmp_dir / "manifest.json"
        script_path = tmp_dir / "run_meta4_core_argo.m"
        script_path.write_text(_matlab_script(args, manifest_path), encoding="utf-8")
        cmd = ["matlab", "-batch", f"run('{matlab_quote(script_path)}')"]
        proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", check=False)
        if proc.returncode != 0:
            raise RuntimeError(
                "MATLAB vertical-transport pipeline failed.\n"
                f"Command: {' '.join(cmd)}\n"
                f"STDOUT:\n{proc.stdout}\n"
                f"STDERR:\n{proc.stderr}"
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    output_root = Path(manifest.get("output_root", args.output_root))
    grid_json_paths = sorted(output_root.rglob("composite_grid.json"))
    if not grid_json_paths:
        grid_json_paths = [Path(item) for item in manifest.get("grid_json_files", [])]
    npz_paths = []
    for grid_json in grid_json_paths:
        npz_path = grid_json.with_suffix(".npz")
        write_npz_from_grid_json(grid_json, npz_path)
        npz_paths.append(npz_path)
    return npz_paths


def _matlab_script(args: argparse.Namespace, manifest_path: Path) -> str:
    bbox = " ".join(f"{item:.12g}" for item in args.bbox)
    lat_bands = "; ".join(f"{lo:.12g} {hi:.12g}" for lo, hi in args.lat_bands)
    argo_mat = matlab_quote(args.argo_mat)
    meta_dir = matlab_quote(args.meta_dir)
    output_root = matlab_quote(args.output_root)
    manifest = matlab_quote(manifest_path)
    max_matches = int(args.max_matches_per_group)
    template = """
argo_mat = '@ARGO_MAT@';
meta_dir = '@META_DIR@';
output_root = '@OUTPUT_ROOT@';
bbox = [@BBOX@];
lat_bands = [@LAT_BANDS@];
time_window_days = @TIME_WINDOW_DAYS@;
grid_n = @GRID_N@;
min_bin_count = @MIN_BIN_COUNT@;
plot_filled_gradient = @PLOT_FILLED_GRADIENT@;
max_matches_per_group = @MAX_MATCHES@;
core_min_m = @CORE_MIN_M@;
core_max_m = @CORE_MAX_M@;
earth_radius_m = 6371000;
deg_m = earth_radius_m * pi / 180;
density_variable = '@DENSITY_VARIABLE@';

if exist(output_root, 'dir') ~= 7
    mkdir(output_root);
end

method_md = fullfile(output_root, 'METHOD_ASSUMPTIONS_ZH.md');
write_method_doc(method_md, argo_mat, meta_dir, output_root, bbox, lat_bands, time_window_days, core_min_m, core_max_m, density_variable);

fprintf('Loading Argo vectors from %s\\n', argo_mat);
A = load(argo_mat, 'I_Time', 'I_Lon', 'I_Lat', 'I_ParkDepth', 'I_PF', density_variable, 'Depth');
rho = A.(density_variable);
argo_lon = double(A.I_Lon);
argo_lon(argo_lon < 0) = argo_lon(argo_lon < 0) + 360;
argo_lat = double(A.I_Lat);
argo_time = double(A.I_Time);
argo_park = double(A.I_ParkDepth);
argo_pf = double(A.I_PF);
depth = double(A.Depth(:));

fprintf('Computing Argo parking drift\\n');
argo_u = nan(size(argo_time));
argo_v = nan(size(argo_time));
for i = 2:numel(argo_time)-1
    if argo_pf(i-1) == argo_pf(i+1)
        dt = (argo_time(i+1) - argo_time(i-1)) * 86400;
        if isfinite(dt) && dt > 0
            dlon = argo_lon(i+1) - argo_lon(i-1);
            if dlon > 180
                dlon = dlon - 360;
            elseif dlon < -180
                dlon = dlon + 360;
            end
            argo_u(i) = dlon * deg_m * cosd(argo_lat(i)) / dt;
            argo_v(i) = (argo_lat(i+1) - argo_lat(i-1)) * deg_m / dt;
        end
    end
end

argo_base_mask = argo_lon >= bbox(1) & argo_lon <= bbox(2) & argo_lat >= bbox(3) & argo_lat <= bbox(4) & ...
    argo_park >= core_min_m & argo_park <= core_max_m & isfinite(argo_u) & isfinite(argo_v);

polarities = {'cyclonic','anticyclonic'};
grid_json_files = {};
summary_rows = {};
summary_header = {'polarity','lat_band','match_count','ring_0_1R','ring_1_2R','ring_2_4R','valid_grid_cells','valid_grid_fraction','mean_cx_raw_m_s','mean_u_bg_m_s','cx_rel_m_s','mean_radius_km','output_dir'};
combined_matches = cell(size(lat_bands, 1), 1);
for b = 1:size(lat_bands, 1)
    combined_matches{b} = cell(0, 22);
end

for p = 1:numel(polarities)
    polarity = polarities{p};
    meta_file = find_meta_file(meta_dir, polarity);
    fprintf('Loading META %s from %s\\n', polarity, meta_file);
    M = load(meta_file, 'final_lon', 'final_lat', 'final_time', 'final_track', 'final_radius');
    meta_lon = double(M.final_lon);
    meta_lon(meta_lon < 0) = meta_lon(meta_lon < 0) + 360;
    meta_lat = double(M.final_lat);
    meta_time = double(M.final_time);
    meta_track = double(M.final_track);
    meta_radius = double(M.final_radius);
    meta_cx = track_cx(meta_lon, meta_lat, meta_time, meta_track, deg_m);

    for b = 1:size(lat_bands, 1)
        lat_min = lat_bands(b,1);
        lat_max = lat_bands(b,2);
        band_label = sprintf('%02.0f_%02.0fN', lat_min, lat_max);
        argo_band = find(argo_base_mask & argo_lat >= lat_min & argo_lat < lat_max);
        meta_band = find(meta_lon >= bbox(1) & meta_lon <= bbox(2) & meta_lat >= lat_min & meta_lat < lat_max & isfinite(meta_radius) & meta_radius > 0);
        group_dir = fullfile(output_root, polarity, band_label);
        if exist(group_dir, 'dir') ~= 7
            mkdir(group_dir);
        end
        [matches, grid] = build_group(argo_band, meta_band, polarity, band_label, ...
            argo_lon, argo_lat, argo_time, argo_park, argo_pf, argo_u, argo_v, rho, depth, ...
            meta_lon, meta_lat, meta_time, meta_track, meta_radius, meta_cx, ...
            time_window_days, grid_n, min_bin_count, plot_filled_gradient, max_matches_per_group, deg_m);
        write_group_outputs(group_dir, matches, grid, polarity, band_label);
        grid_json_files{end+1} = fullfile(group_dir, 'composite_grid.json'); %#ok<SAGROW>
        summary_rows(end+1,:) = summary_from_matches(matches, grid, polarity, band_label, group_dir); %#ok<SAGROW>
        combined_matches{b} = [combined_matches{b}; matches]; %#ok<SAGROW>
    end
end

combined_rows = write_combined_outputs(output_root, lat_bands, combined_matches, grid_n, min_bin_count, plot_filled_gradient);
summary_rows = [summary_rows; combined_rows];
summary_path = fullfile(output_root, 'SUMMARY.csv');
writecell([summary_header; summary_rows], summary_path);
write_summary_doc(fullfile(output_root, 'RUN_SUMMARY_ZH.md'), summary_rows, output_root);

manifest = struct();
manifest.grid_json_files = grid_json_files;
manifest.output_root = output_root;
text = jsonencode(manifest);
fid = fopen('@MANIFEST@', 'w');
fwrite(fid, text, 'char');
fclose(fid);

function file = find_meta_file(meta_dir, polarity)
    files = dir(fullfile(meta_dir, ['*' polarity '*track*.mat']));
    if strcmp(polarity, 'cyclonic')
        keep = true(size(files));
        for k = 1:numel(files)
            keep(k) = isempty(strfind(files(k).name, 'anticyclonic'));
        end
        files = files(keep);
    end
    if isempty(files)
        error('No META track mat found for %s in %s', polarity, meta_dir);
    end
    [~, idx] = max([files.bytes]);
    file = fullfile(files(idx).folder, files(idx).name);
end

function cx = track_cx(lon, lat, time, track, deg_m)
    cx = nan(size(time));
    for i = 2:numel(time)-1
        if track(i-1) == track(i+1)
            dt = (time(i+1) - time(i-1)) * 86400;
            if isfinite(dt) && dt > 0
                dlon = lon(i+1) - lon(i-1);
                if dlon > 180
                    dlon = dlon - 360;
                elseif dlon < -180
                    dlon = dlon + 360;
                end
                cx(i) = dlon * deg_m * cosd(lat(i)) / dt;
            end
        end
    end
end

function [matches, grid] = build_group(argo_idx, meta_idx, polarity, band_label, ...
    argo_lon, argo_lat, argo_time, argo_park, argo_pf, argo_u, argo_v, rho, depth, ...
    meta_lon, meta_lat, meta_time, meta_track, meta_radius, meta_cx, ...
    time_window_days, grid_n, min_bin_count, plot_filled_gradient, max_matches_per_group, deg_m)

    rows = {};
    row_count = 0;
    meta_time_band = meta_time(meta_idx);
    for a = 1:numel(argo_idx)
        ii = argo_idx(a);
        candidate_local = find(abs(meta_time_band - argo_time(ii)) <= time_window_days);
        if isempty(candidate_local)
            continue
        end
        candidates = meta_idx(candidate_local);
        dx = local_dx_m(argo_lon(ii), meta_lon(candidates), argo_lat(ii), deg_m);
        dy = (argo_lat(ii) - meta_lat(candidates)) * deg_m;
        r_norm = hypot(dx, dy) ./ meta_radius(candidates);
        [best_r, best_pos] = min(r_norm);
        if ~isfinite(best_r) || best_r > 4
            continue
        end
        jj = candidates(best_pos);
        rho0 = interp1(depth, double(rho(ii,:)), argo_park(ii), 'linear', NaN);
        z_rho = isopycnal_depth(depth, double(rho(ii,:)), rho0);
        if ~isfinite(z_rho)
            continue
        end
        row_count = row_count + 1;
        ring = ring_label(best_r);
        rows(row_count,:) = {polarity, band_label, ii, argo_pf(ii), argo_time(ii), argo_lon(ii), argo_lat(ii), ...
            argo_park(ii), argo_u(ii), argo_v(ii), rho0, z_rho, meta_track(jj), meta_time(jj), meta_lon(jj), ...
            meta_lat(jj), meta_radius(jj), dx / meta_radius(jj), dy / meta_radius(jj), best_r, ring, meta_cx(jj)}; %#ok<AGROW>
        if max_matches_per_group > 0 && row_count >= max_matches_per_group
            break
        end
    end
    matches = rows;
    grid = composite_grid(matches, grid_n, min_bin_count, plot_filled_gradient);
end

function dx = local_dx_m(lon_a, lon_b, lat_ref, deg_m)
    dlon = lon_a - lon_b;
    dlon(dlon > 180) = dlon(dlon > 180) - 360;
    dlon(dlon < -180) = dlon(dlon < -180) + 360;
    dx = dlon .* deg_m .* cosd(lat_ref);
end

function z = isopycnal_depth(depth, profile, rho0)
    z = NaN;
    depth = depth(:);
    profile = profile(:);
    good = isfinite(depth) & isfinite(profile);
    depth = depth(good);
    profile = profile(good);
    if numel(depth) < 3 || ~isfinite(rho0)
        return
    end
    [profile_unique, ia] = unique(profile, 'stable');
    depth_unique = depth(ia);
    if numel(profile_unique) < 3 || rho0 < min(profile_unique) || rho0 > max(profile_unique)
        return
    end
    z = interp1(profile_unique, depth_unique, rho0, 'linear', NaN);
end

function label = ring_label(r_norm)
    if r_norm <= 1
        label = '0-1R';
    elseif r_norm <= 2
        label = '1-2R';
    else
        label = '2-4R';
    end
end

function grid = composite_grid(matches, grid_n, min_bin_count, plot_filled_gradient)
    x_vec = linspace(-4, 4, grid_n);
    y_vec = linspace(-4, 4, grid_n);
    [X, Y] = meshgrid(x_vec, y_vec);
    nan_grid = nan(size(X));
    count_grid = zeros(size(X));
    grid = struct('x', X, 'y', Y, 'z', nan_grid, 'u', nan_grid, 'v', nan_grid, ...
        'count', count_grid, 'term1', nan_grid, 'term2', nan_grid, 'rebuild_w', nan_grid, ...
        'mean_cx_raw', NaN, 'mean_u_bg', NaN, 'cx_rel', NaN, 'mean_radius_m', NaN);
    if isempty(matches)
        return
    end
    x = cell2mat(matches(:,18)); x = x(:);
    y = cell2mat(matches(:,19)); y = y(:);
    z = cell2mat(matches(:,12)); z = z(:);
    u = cell2mat(matches(:,9)); u = u(:);
    v = cell2mat(matches(:,10)); v = v(:);
    cx_raw = cell2mat(matches(:,22)); cx_raw = cx_raw(:);
    radius = cell2mat(matches(:,17)); radius = radius(:);
    grid.mean_cx_raw = mean(cx_raw, 'omitnan');
    grid.mean_u_bg = mean(u, 'omitnan');
    grid.cx_rel = grid.mean_cx_raw - grid.mean_u_bg;
    grid.mean_radius_m = mean(radius, 'omitnan');
    edges = linspace(-4, 4, grid_n + 1);
    xb = discretize(x, edges);
    yb = discretize(y, edges);
    n = min([numel(xb), numel(yb), numel(z), numel(u), numel(v)]);
    xb = xb(1:n); yb = yb(1:n); z = z(1:n); u = u(1:n); v = v(1:n);
    valid = isfinite(xb) & isfinite(yb);
    subs = [yb(valid), xb(valid)];
    grid.count = accumarray(subs, 1, [grid_n grid_n], @sum, 0);
    grid.z = accumarray(subs, z(valid), [grid_n grid_n], @(q) median(q, 'omitnan'), NaN);
    grid.u = accumarray(subs, u(valid), [grid_n grid_n], @(q) median(q, 'omitnan'), NaN);
    grid.v = accumarray(subs, v(valid), [grid_n grid_n], @(q) median(q, 'omitnan'), NaN);
    dx_m = mean(diff(x_vec)) * grid.mean_radius_m;
    dy_m = mean(diff(y_vec)) * grid.mean_radius_m;
    if isfinite(dx_m) && dx_m > 0 && isfinite(dy_m) && dy_m > 0
        [dzdy, dzdx] = gradient(fillmissing2(grid.z), dy_m, dx_m);
        support = grid.count >= min_bin_count;
        if plot_filled_gradient
            grid.term1 = grid.cx_rel .* dzdx;
        else
            grid.term1 = mask_to_support(grid.cx_rel .* dzdx, support);
        end
        grid.term2 = mask_to_support(grid.u .* dzdx + grid.v .* dzdy, support);
        grid.rebuild_w = mask_to_support(grid.term1 + grid.term2, support);
    end
end

function out = mask_to_support(A, support)
    out = A;
    out(~support) = NaN;
end

function out = fillmissing2(A)
    out = A;
    for n = 1:4
        B = out;
        for i = 1:size(out,1)
            for j = 1:size(out,2)
                if ~isfinite(out(i,j))
                    i0 = max(1, i-1); i1 = min(size(out,1), i+1);
                    j0 = max(1, j-1); j1 = min(size(out,2), j+1);
                    vals = out(i0:i1, j0:j1);
                    if any(isfinite(vals), 'all')
                        B(i,j) = mean(vals(isfinite(vals)), 'omitnan');
                    end
                end
            end
        end
        out = B;
    end
end

function write_group_outputs(group_dir, matches, grid, polarity, band_label)
    header = {'polarity','lat_band','argo_index','platform','argo_time','argo_lon','argo_lat','parking_depth_m','u_argo_m_s','v_argo_m_s','rho0','z_rho_m','eddy_track','eddy_time','eddy_lon','eddy_lat','eddy_radius_m','x_over_R','y_over_R','r_over_R','ring','cx_raw_m_s'};
    writecell(clean_write_cells([header; matches]), fullfile(group_dir, 'matched_core_argo.csv'));
    write_grid_json(fullfile(group_dir, 'composite_grid.json'), grid, polarity, band_label);
    plot_three_panel(fullfile(group_dir, 'vertical_transport_terms.png'), grid, [polarity ' ' band_label]);
    write_group_doc(fullfile(group_dir, 'METHOD_ASSUMPTIONS_ZH.md'), matches, grid, polarity, band_label);
end

function C = clean_write_cells(C)
    for ii = 1:numel(C)
        try
            if ismissing(C{ii})
                C{ii} = '';
            end
        catch
        end
    end
end

function row = summary_from_matches(matches, grid, polarity, band_label, group_dir)
    if isempty(matches)
        rings = {};
        n = 0;
    else
        rings = matches(:,21);
        n = size(matches, 1);
    end
    valid_cells = sum(isfinite(grid.rebuild_w(:)));
    valid_fraction = valid_cells / numel(grid.count);
    row = {polarity, band_label, n, sum(strcmp(rings,'0-1R')), sum(strcmp(rings,'1-2R')), sum(strcmp(rings,'2-4R')), ...
        valid_cells, valid_fraction, grid.mean_cx_raw, grid.mean_u_bg, grid.cx_rel, grid.mean_radius_m / 1000, group_dir};
end

function combined_rows = write_combined_outputs(output_root, lat_bands, combined_matches, grid_n, min_bin_count, plot_filled_gradient)
    combined_rows = {};
    for b = 1:size(lat_bands, 1)
        band_label = sprintf('%02.0f_%02.0fN', lat_bands(b,1), lat_bands(b,2));
        combined_dir = fullfile(output_root, 'combined', band_label);
        if exist(combined_dir, 'dir') ~= 7
            mkdir(combined_dir);
        end
        all_matches = combined_matches{b};
        grid = composite_grid(all_matches, grid_n, min_bin_count, plot_filled_gradient);
        write_group_outputs(combined_dir, all_matches, grid, 'combined', band_label);
        combined_rows(end+1,:) = summary_from_matches(all_matches, grid, 'combined', band_label, combined_dir); %#ok<AGROW>
    end
end

function write_grid_json(path, grid, polarity, band_label)
    G = struct();
    G.metadata = struct('polarity', polarity, 'lat_band', band_label, 'mean_cx_raw_m_s', grid.mean_cx_raw, ...
        'mean_u_bg_m_s', grid.mean_u_bg, 'cx_rel_m_s', grid.cx_rel, 'mean_radius_m', grid.mean_radius_m, ...
        'valid_grid_cells', sum(isfinite(grid.rebuild_w(:))), 'total_grid_cells', numel(grid.count));
    G.x_over_R = grid.x;
    G.y_over_R = grid.y;
    G.z_rho_m = grid.z;
    G.u_argo_m_s = grid.u;
    G.v_argo_m_s = grid.v;
    G.sample_count = grid.count;
    G.term1_m_s = grid.term1;
    G.term2_m_s = grid.term2;
    G.rebuild_w_m_s = grid.rebuild_w;
    fid = fopen(path, 'w');
    fwrite(fid, jsonencode(G), 'char');
    fclose(fid);
end

function plot_three_panel(path, grid, title_prefix)
    fig = figure('Visible','off','Position',[100 100 1500 430]);
    fields = {'term1','term2','rebuild_w'};
    titles = {'c_x^{rel} dz_\\rho/dx','u_{Argo} \\cdot \\nabla z_\\rho','rebuild W'};
    vals = [grid.term1(:); grid.term2(:); grid.rebuild_w(:)];
    lim = max(abs(vals(isfinite(vals))));
    if isempty(lim) || ~isfinite(lim) || lim == 0
        lim = 2.5e-6;
    end
    lim = max(lim, 2.5e-6);
    for k = 1:3
        subplot(1,3,k);
        data = grid.(fields{k}) * 1e6;
        h = imagesc(grid.x(1,:), grid.y(:,1), data);
        set(h, 'AlphaData', isfinite(data));
        set(gca, 'YDir', 'normal');
        set(gca, 'Color', [1 1 1]);
        axis image;
        xlim([-4 4]); ylim([-4 4]);
        clim([-lim lim] * 1e6);
        colormap(redblue_colormap());
        colorbar;
        hold on;
        th = linspace(0, 2*pi, 240);
        plot(cos(th), sin(th), 'k-', 'LineWidth', 1.2);
        plot(4*cos(th), 4*sin(th), 'k-', 'LineWidth', 1.2);
        plot(0, 0, 'k.', 'MarkerSize', 16);
        title(titles{k});
        xlabel('x/R'); ylabel('y/R');
    end
    sgtitle([title_prefix '  (10^{-6} m s^{-1})'], 'Interpreter', 'tex');
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

function cmap = redblue_colormap()
    n = 256;
    r = [(0:(n/2-1))/(n/2), ones(1,n/2)];
    g = [(0:(n/2-1))/(n/2), (n/2-1:-1:0)/(n/2)];
    b = [ones(1,n/2), (n/2-1:-1:0)/(n/2)];
    cmap = [r(:), g(:), b(:)];
end

function write_method_doc(path, argo_mat, meta_dir, output_root, bbox, lat_bands, time_window_days, core_min_m, core_max_m, density_variable)
    fid = fopen(path, 'w');
    fprintf(fid, '# META4.0 + Core Argo 垂直速度重建方法与假定\\n\\n');
    fprintf(fid, '- Argo 主数据源：`%s`\\n', argo_mat);
    fprintf(fid, '- META4.0 涡旋源：`%s`\\n', meta_dir);
    fprintf(fid, '- 输出根目录：`%s`\\n', output_root);
    fprintf(fid, '- 黑潮区域：`%.1fE-%.1fE, %.1fN-%.1fN`\\n', bbox(1), bbox(2), bbox(3), bbox(4));
    fprintf(fid, '- 纬度带：');
    for i=1:size(lat_bands,1), fprintf(fid, '`%.0f-%.0fN` ', lat_bands(i,1), lat_bands(i,2)); end
    fprintf(fid, '\\n- Core Argo 限定：`%.0f-%.0f m` parking depth。\\n', core_min_m, core_max_m);
    fprintf(fid, '- 时间匹配：Argo profile 与 META 轨迹点相差不超过 `%.1f day`。\\n', time_window_days);
    fprintf(fid, '- 空间分区：保存 `0-1R`、`1-2R`、`2-4R`，图像网格为 `x/R, y/R = -4..4`。\\n');
    fprintf(fid, '- 密度变量：`%s`。`rho0` 为每条 Core Argo 在实际 parking depth 处的密度。\\n', density_variable);
    fprintf(fid, '- `c_x_raw` 来自 META track 相邻点中央差分；`u_bg` 为同纬度带、同极性、匹配 Core Argo 的 parking drift 纬向均值；`c_x_rel = mean(c_x_raw) - mean(u_bg)`。\\n');
    fprintf(fid, '- BOA_Argo 只作为 gridded 温盐/密度背景可行性假设记录，不在第一版中直接推导背景速度。\\n\\n');
    fprintf(fid, '参考：NOAA AOML Argo overview, NOAA Argo best practices, Lin et al. 2019 Remote Sensing, Zhou et al. 2023 JGR Oceans, JAMSTEC Argo gridded products。\\n');
    fclose(fid);
end

function write_group_doc(path, matches, grid, polarity, band_label)
    fid = fopen(path, 'w');
    fprintf(fid, '# %s %s Core Argo 垂直速度重建摘要\\n\\n', polarity, band_label);
    fprintf(fid, '- 匹配样本数：`%d`\\n', size(matches,1));
    fprintf(fid, '- mean c_x_raw：`%.6g m/s`\\n', grid.mean_cx_raw);
    fprintf(fid, '- mean u_bg：`%.6g m/s`\\n', grid.mean_u_bg);
    fprintf(fid, '- c_x_rel：`%.6g m/s`\\n', grid.cx_rel);
    fprintf(fid, '- mean radius：`%.3f km`\\n', grid.mean_radius_m / 1000);
    valid_cells = sum(isfinite(grid.rebuild_w(:)));
    valid_fraction = valid_cells / numel(grid.count);
    fprintf(fid, '- 有样本支撑网格：`%d / %d (%.2f%%)`。\\n', valid_cells, numel(grid.count), valid_fraction * 100);
    fprintf(fid, '- 输出：`matched_core_argo.csv`、`composite_grid.npz`、`vertical_transport_terms.png`。图像显示为 `10^-6 m/s`，网格文件保存原始 `m/s`，白色为空样本格点。\\n');
    fclose(fid);
end

function write_summary_doc(path, summary_rows, output_root)
    fid = fopen(path, 'w');
    fprintf(fid, '# META4.0 + Core Argo 垂直速度重建运行摘要\\n\\n');
    fprintf(fid, '- 输出根目录：`%s`\\n', output_root);
    fprintf(fid, '- 主图变量：`term1 = c_x_rel dz_rho/dx`，`term2 = u_Argo · grad(z_rho)`，`rebuild_W = term1 + term2`。\\n');
    fprintf(fid, '- 单位：JSON/NPZ 保存原始 `m/s`；PNG 色标显示为 `10^-6 m/s`；白色为空样本格点。\\n');
    fprintf(fid, '- 若使用 `--max-matches-per-group` 做 smoke run，覆盖率会很低；正式结果应使用默认 `0` 读取全部匹配。\\n\\n');
    fprintf(fid, '| polarity | lat_band | matches | 0-1R | 1-2R | 2-4R | valid grid %% | c_x_rel m/s |\\n');
    fprintf(fid, '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |\\n');
    for i=1:size(summary_rows,1)
        fprintf(fid, '| %s | %s | %d | %d | %d | %d | %.2f | %.6g |\\n', summary_rows{i,1}, summary_rows{i,2}, summary_rows{i,3}, summary_rows{i,4}, summary_rows{i,5}, summary_rows{i,6}, summary_rows{i,8} * 100, summary_rows{i,11});
    end
    fclose(fid);
end
"""
    return (
        template.replace("@ARGO_MAT@", argo_mat)
        .replace("@META_DIR@", meta_dir)
        .replace("@OUTPUT_ROOT@", output_root)
        .replace("@BBOX@", bbox)
        .replace("@LAT_BANDS@", lat_bands)
        .replace("@TIME_WINDOW_DAYS@", f"{float(args.time_window_days):.12g}")
        .replace("@GRID_N@", str(int(args.grid_n)))
        .replace("@MIN_BIN_COUNT@", str(int(args.min_bin_count)))
        .replace("@PLOT_FILLED_GRADIENT@", "true" if args.plot_filled_gradient else "false")
        .replace("@MAX_MATCHES@", str(max_matches))
        .replace("@CORE_MIN_M@", f"{float(args.core_min_m):.12g}")
        .replace("@CORE_MAX_M@", f"{float(args.core_max_m):.12g}")
        .replace("@DENSITY_VARIABLE@", str(args.density_variable).replace("'", "''"))
        .replace("@MANIFEST@", manifest)
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Rebuild vertical transport terms from META4.0 eddies and Core Argo.")
    parser.add_argument("--argo-mat", type=Path, default=DEFAULT_ARGO_MAT)
    parser.add_argument("--meta-dir", type=Path, default=DEFAULT_META_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--bbox", type=parse_bbox, default=parse_bbox(DEFAULT_BBOX))
    parser.add_argument("--lat-bands", type=parse_lat_bands, default=parse_lat_bands(DEFAULT_LAT_BANDS))
    parser.add_argument("--time-window-days", type=float, default=1.0)
    parser.add_argument("--core-min-m", type=float, default=900.0)
    parser.add_argument("--core-max-m", type=float, default=1100.0)
    parser.add_argument("--density-variable", default="I_sigma1")
    parser.add_argument("--grid-n", type=int, default=81)
    parser.add_argument(
        "--min-bin-count",
        type=int,
        default=1,
        help="Minimum samples required for a composite grid cell to be displayed and saved in W terms.",
    )
    parser.add_argument(
        "--plot-filled-gradient",
        action="store_true",
        help="Keep the filled-gradient term1 field for diagnostics. By default W terms are masked to sampled cells.",
    )
    parser.add_argument(
        "--max-matches-per-group",
        type=int,
        default=0,
        help="Debug/smoke limit. Use 0 for all matched profiles.",
    )
    args = parser.parse_args()
    npz_paths = run_matlab_pipeline(args)
    for path in npz_paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
