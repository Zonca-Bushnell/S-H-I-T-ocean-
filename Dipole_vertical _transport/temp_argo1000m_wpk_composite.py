from __future__ import annotations

import argparse
import json
import struct
import subprocess
import tempfile
import zipfile
from pathlib import Path


DEFAULT_ARGO_MAT = Path(r"F:\Argo_data\Argo1000m_UVW_TSDen_199601_202306.mat")
DEFAULT_META_DIR = Path(r"F:\Eddy\Eddy\META4.0_DT_allsat")
DEFAULT_OUTPUT_ROOT = Path(
    r"E:\DATA\01_Eddy_correspond\01_Vertical_asymmetric\TEMP_Argo1000m_Wpk_crossing_10N_1R"
)


def parse_bbox(value: str) -> tuple[float, float, float, float]:
    parts = [float(item.strip()) for item in value.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("--bbox must be lon_min,lon_max,lat_min,lat_max")
    lon_min, lon_max, lat_min, lat_max = parts
    if lon_min >= lon_max or lat_min >= lat_max:
        raise argparse.ArgumentTypeError("--bbox ranges must be increasing")
    return lon_min, lon_max, lat_min, lat_max


def matlab_quote(path: Path) -> str:
    return str(path).replace("\\", "\\\\").replace("'", "''")


def latitude_crossing_label(target_lat: float, intersect_radius_r: float) -> str:
    hemi = "N" if target_lat >= 0 else "S"
    lat_text = f"{abs(target_lat):02.0f}{hemi}"
    radius_text = f"{intersect_radius_r:g}".replace(".", "p")
    return f"cross_{lat_text}_{radius_text}R"


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
            out.extend(struct.pack("<d", float(value) if value is not None else float("nan")))
    return bytes(out)


def write_npz_from_grid_json(json_path: Path, npz_path: Path) -> None:
    grid = json.loads(json_path.read_text(encoding="utf-8"))
    arrays = {
        "x_over_R": grid["x_over_R"],
        "y_over_R": grid["y_over_R"],
        "wpk_m_s": grid["wpk_m_s"],
        "sample_count": grid["sample_count"],
        "mapped_support": grid["mapped_support"],
    }
    with zipfile.ZipFile(npz_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, values in arrays.items():
            zf.writestr(f"{name}.npy", npy_bytes_2d(values))
        zf.writestr("metadata.json", json.dumps(grid.get("metadata", {}), ensure_ascii=False, indent=2))


def run_matlab(args: argparse.Namespace) -> list[Path]:
    with tempfile.TemporaryDirectory(prefix="temp_argo1000m_wpk_") as tmp:
        tmp_dir = Path(tmp)
        manifest_path = tmp_dir / "manifest.json"
        script_path = tmp_dir / "run_temp_wpk.m"
        script_path.write_text(matlab_script(args, manifest_path), encoding="utf-8")
        proc = subprocess.run(
            ["matlab", "-batch", f"run('{matlab_quote(script_path)}')"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"MATLAB failed\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    out = []
    for item in manifest["grid_json_files"]:
        p = Path(item)
        npz = p.with_suffix(".npz")
        write_npz_from_grid_json(p, npz)
        out.append(npz)
    return out


def matlab_script(args: argparse.Namespace, manifest_path: Path) -> str:
    lon_min, lon_max, lat_min, lat_max = args.bbox
    target_label = latitude_crossing_label(args.target_lat, args.intersect_radius_r)
    replacements = {
        "@ARGO_MAT@": matlab_quote(args.argo_mat),
        "@META_DIR@": matlab_quote(args.meta_dir),
        "@OUTPUT_ROOT@": matlab_quote(args.output_root),
        "@MANIFEST@": matlab_quote(manifest_path),
        "@BBOX@": f"{lon_min:.12g} {lon_max:.12g} {lat_min:.12g} {lat_max:.12g}",
        "@TARGET_LAT@": f"{float(args.target_lat):.12g}",
        "@INTERSECT_RADIUS_R@": f"{float(args.intersect_radius_r):.12g}",
        "@TARGET_LABEL@": target_label.replace("'", "''"),
        "@GRID_N@": str(int(args.grid_n)),
        "@TIME_WINDOW_DAYS@": f"{float(args.time_window_days):.12g}",
        "@CORE_MIN_M@": f"{float(args.core_min_m):.12g}",
        "@CORE_MAX_M@": f"{float(args.core_max_m):.12g}",
        "@CRESSMAN_RADIUS_R@": f"{float(args.cressman_radius_r):.12g}",
        "@CRESSMAN_MIN_OBS@": str(int(args.cressman_min_obs)),
    }
    script = r"""
argo_mat = '@ARGO_MAT@';
meta_dir = '@META_DIR@';
output_root = '@OUTPUT_ROOT@';
bbox = [@BBOX@];
target_lat = @TARGET_LAT@;
intersect_radius_r = @INTERSECT_RADIUS_R@;
target_label = '@TARGET_LABEL@';
grid_n = @GRID_N@;
time_window_days = @TIME_WINDOW_DAYS@;
core_min_m = @CORE_MIN_M@;
core_max_m = @CORE_MAX_M@;
cressman_radius_r = @CRESSMAN_RADIUS_R@;
cressman_min_obs = @CRESSMAN_MIN_OBS@;
earth_radius_m = 6371000;
deg_m = earth_radius_m * pi / 180;

if exist(output_root, 'dir') ~= 7
    mkdir(output_root);
end

fprintf('Loading Argo Wpk from %s\n', argo_mat);
A = load(argo_mat, 'I_Time', 'I_Lon', 'I_Lat', 'I_ParkDepth', 'I_PF', 'I_Wpk');
argo_lon = double(A.I_Lon);
argo_lon(argo_lon < 0) = argo_lon(argo_lon < 0) + 360;
argo_lat = double(A.I_Lat);
argo_time = double(A.I_Time);
argo_park = double(A.I_ParkDepth);
argo_pf = double(A.I_PF);
argo_w = double(A.I_Wpk);

argo_mask = argo_lon >= bbox(1) & argo_lon <= bbox(2) & ...
    argo_park >= core_min_m & argo_park <= core_max_m & isfinite(argo_w);

polarities = {'cyclonic','anticyclonic'};
grid_json_files = {};
summary = {'polarity','lat_band','match_count','ring_0_1R','ring_1_2R','ring_2_4R','valid_grid_cells','max_support','median_support_valid','q95_abs_w_1e6_m_s'};
for p = 1:numel(polarities)
    polarity = polarities{p};
    meta_file = find_meta_file(meta_dir, polarity);
    fprintf('Loading META %s from %s\n', polarity, meta_file);
    M = load(meta_file, 'final_lon', 'final_lat', 'final_time', 'final_track', 'final_radius');
    meta_lon = double(M.final_lon);
    meta_lon(meta_lon < 0) = meta_lon(meta_lon < 0) + 360;
    meta_lat = double(M.final_lat);
    meta_time = double(M.final_time);
    meta_track = double(M.final_track);
    meta_radius = double(M.final_radius);
    cross_distance_m = abs(meta_lat - target_lat) * deg_m;
    meta_idx = find(meta_lon >= bbox(1) & meta_lon <= bbox(2) & ...
        isfinite(meta_radius) & meta_radius > 0 & cross_distance_m <= meta_radius * intersect_radius_r);
    argo_idx = find(argo_mask);
    matches = match_argo_to_meta(argo_idx, argo_lon, argo_lat, argo_time, argo_park, argo_pf, argo_w, ...
        meta_idx, meta_lon, meta_lat, meta_time, meta_track, meta_radius, time_window_days, deg_m);
    group_dir = fullfile(output_root, polarity, target_label);
    if exist(group_dir, 'dir') ~= 7
        mkdir(group_dir);
    end
    grid = composite_wpk(matches, grid_n, cressman_radius_r, cressman_min_obs);
    write_outputs(group_dir, matches, grid, polarity, target_label, target_lat, intersect_radius_r, cressman_radius_r, cressman_min_obs);
    grid_json_files{end+1} = fullfile(group_dir, 'wpk_composite_grid.json'); %#ok<SAGROW>
    w = grid.wpk(:);
    valid = isfinite(w);
    support = grid.support(:);
    q95 = NaN;
    med_support = NaN;
    if any(valid)
        q95 = prctile(abs(w(valid))*1e6, 95);
        med_support = median(support(valid), 'omitnan');
    end
    rings = matches(:,15);
    summary(end+1,:) = {polarity, target_label, size(matches,1), ...
        sum(strcmp(rings,'0-1R')), sum(strcmp(rings,'1-2R')), sum(strcmp(rings,'2-4R')), ...
        sum(valid), max(support), med_support, q95}; %#ok<SAGROW>
end
writecell(summary, fullfile(output_root, 'SUMMARY_WPK.csv'));

manifest = struct('grid_json_files', {grid_json_files}, 'output_root', output_root);
fid = fopen('@MANIFEST@', 'w');
fwrite(fid, jsonencode(manifest), 'char');
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

function rows = match_argo_to_meta(argo_idx, argo_lon, argo_lat, argo_time, argo_park, argo_pf, argo_w, ...
    meta_idx, meta_lon, meta_lat, meta_time, meta_track, meta_radius, time_window_days, deg_m)
    rows = {};
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
        best_dx = dx(best_pos);
        best_dy = dy(best_pos);
        rows(end+1,:) = {ii, argo_pf(ii), argo_time(ii), argo_lon(ii), argo_lat(ii), argo_park(ii), argo_w(ii), ...
            meta_track(jj), meta_time(jj), meta_lon(jj), meta_lat(jj), meta_radius(jj), ...
            best_dx / meta_radius(jj), best_dy / meta_radius(jj), ring_label(best_r)}; %#ok<AGROW>
    end
end

function dx = local_dx_m(lon_a, lon_b, lat_ref, deg_m)
    dlon = lon_a - lon_b;
    dlon(dlon > 180) = dlon(dlon > 180) - 360;
    dlon(dlon < -180) = dlon(dlon < -180) + 360;
    dx = dlon .* deg_m .* cosd(lat_ref);
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

function grid = composite_wpk(matches, grid_n, radius_r, min_obs)
    x_vec = linspace(-4, 4, grid_n);
    y_vec = linspace(-4, 4, grid_n);
    [X, Y] = meshgrid(x_vec, y_vec);
    grid = struct('x', X, 'y', Y, 'wpk', NaN(size(X)), 'count', zeros(size(X)), 'support', zeros(size(X)));
    if isempty(matches)
        return
    end
    x = cell2mat(matches(:,13));
    y = cell2mat(matches(:,14));
    w = cell2mat(matches(:,7));
    edges = linspace(-4, 4, grid_n + 1);
    xb = discretize(x, edges);
    yb = discretize(y, edges);
    valid = isfinite(xb) & isfinite(yb);
    grid.count = accumarray([yb(valid), xb(valid)], 1, [grid_n grid_n], @sum, 0);
    [grid.wpk, grid.support] = cressman_map(x, y, w, X, Y, radius_r, min_obs);
end

function [Z, support_count] = cressman_map(x, y, v, X, Y, radius_r, min_obs)
    Z = NaN(size(X));
    support_count = zeros(size(X));
    good = isfinite(x) & isfinite(y) & isfinite(v) & hypot(x, y) <= 4;
    x = x(good); y = y(good); v = v(good);
    r2_limit = radius_r ^ 2;
    for ii = 1:numel(X)
        d2 = (x - X(ii)).^2 + (y - Y(ii)).^2;
        inside = d2 < r2_limit;
        support_count(ii) = nnz(inside);
        if support_count(ii) >= min_obs
            weight = (r2_limit - d2(inside)) ./ (r2_limit + d2(inside));
            vals = v(inside);
            ok = isfinite(weight) & weight > 0 & isfinite(vals);
            if any(ok)
                Z(ii) = sum(weight(ok) .* vals(ok)) ./ sum(weight(ok));
            end
        end
    end
    Z(hypot(X, Y) > 4) = NaN;
    support_count(hypot(X, Y) > 4) = 0;
end

function write_outputs(group_dir, matches, grid, polarity, band_label, target_lat, intersect_radius_r, radius_r, min_obs)
    header = {'argo_index','platform','argo_time','argo_lon','argo_lat','parking_depth_m','wpk_m_s', ...
        'eddy_track','eddy_time','eddy_lon','eddy_lat','eddy_radius_m','x_over_R','y_over_R','ring'};
    writecell([header; matches], fullfile(group_dir, 'matched_argo1000m_wpk.csv'));
    G = struct();
    G.metadata = struct('polarity', polarity, 'lat_band', band_label, 'grid_mapping', 'cressman', ...
        'target_lat', target_lat, 'intersect_radius_r', intersect_radius_r, ...
        'cressman_radius_r', radius_r, 'cressman_min_obs', min_obs, ...
        'match_count', size(matches,1), 'valid_grid_cells', sum(isfinite(grid.wpk(:))));
    G.x_over_R = grid.x;
    G.y_over_R = grid.y;
    G.wpk_m_s = grid.wpk;
    G.sample_count = grid.count;
    G.mapped_support = grid.support;
    fid = fopen(fullfile(group_dir, 'wpk_composite_grid.json'), 'w');
    fwrite(fid, jsonencode(G), 'char');
    fclose(fid);
    plot_wpk(fullfile(group_dir, 'wpk_composite.png'), grid, [polarity ' ' band_label]);
end

function plot_wpk(path, grid, title_text)
    fig = figure('Visible','off','Position',[100 100 660 560]);
    data = grid.wpk * 1e6;
    lim = max(abs(data(isfinite(data))));
    if isempty(lim) || ~isfinite(lim) || lim == 0
        lim = 2.5;
    end
    imagesc(grid.x(1,:), grid.y(:,1), data, 'AlphaData', isfinite(data));
    set(gca, 'YDir', 'normal', 'Color', [1 1 1]);
    axis image;
    xlim([-4 4]); ylim([-4 4]);
    clim([-lim lim]);
    colormap(redblue_colormap());
    colorbar;
    hold on;
    th = linspace(0, 2*pi, 240);
    plot(cos(th), sin(th), 'k-', 'LineWidth', 1.2);
    plot(4*cos(th), 4*sin(th), 'k-', 'LineWidth', 1.2);
    plot(0, 0, 'k.', 'MarkerSize', 16);
    title([title_text ' I\_Wpk (10^{-6} m s^{-1})'], 'Interpreter', 'tex');
    xlabel('x/R'); ylabel('y/R');
    exportgraphics(fig, path, 'Resolution', 180);
    close(fig);
end

function label = lat_band_label(lat_min, lat_max)
    label = [lat_token(lat_min) '_' lat_token(lat_max)];
end

function token = lat_token(lat)
    if lat < 0
        hemi = 'S';
    else
        hemi = 'N';
    end
    token = sprintf('%02.0f%s', abs(lat), hemi);
end

function cmap = redblue_colormap()
    n = 256;
    r = [(0:(n/2-1))/(n/2), ones(1,n/2)];
    g = [(0:(n/2-1))/(n/2), (n/2-1:-1:0)/(n/2)];
    b = [ones(1,n/2), (n/2-1:-1:0)/(n/2)];
    cmap = [r(:), g(:), b(:)];
end
"""
    for old, new in replacements.items():
        script = script.replace(old, new)
    return script


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Temporary check: composite historical Argo1000m I_Wpk around META eddies."
    )
    parser.add_argument("--argo-mat", type=Path, default=DEFAULT_ARGO_MAT)
    parser.add_argument("--meta-dir", type=Path, default=DEFAULT_META_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--bbox", type=parse_bbox, default=parse_bbox("0,360,-90,90"))
    parser.add_argument("--target-lat", type=float, default=10.0)
    parser.add_argument("--intersect-radius-r", type=float, default=1.0)
    parser.add_argument("--time-window-days", type=float, default=1.0)
    parser.add_argument("--core-min-m", type=float, default=900.0)
    parser.add_argument("--core-max-m", type=float, default=1100.0)
    parser.add_argument("--grid-n", type=int, default=81)
    parser.add_argument("--cressman-radius-r", type=float, default=0.5)
    parser.add_argument("--cressman-min-obs", type=int, default=3)
    args = parser.parse_args()
    for path in run_matlab(args):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
