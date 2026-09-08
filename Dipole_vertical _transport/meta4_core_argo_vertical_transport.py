from __future__ import annotations

import argparse
import json
import math
import struct
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_ARGO_MAT = Path(r"F:\Argo_data\ArgoData_SA_CT_PT_PDen_sigma.mat")
DEFAULT_HISTORY_ARGO_MAT = Path(r"F:\Argo_data\Argo1000m_UVW_TSDen_199601_202306.mat")
DEFAULT_META_DIR = Path(r"F:\Eddy\Eddy\META4.0_DT_allsat")
DEFAULT_BOA_PDEN_ROOT = Path(r"F:\Argo_data\Self_BOA_Argo_PotentialDensity")
DEFAULT_OUTPUT_ROOT = Path(
    r"E:\DATA\01_Eddy_correspond\01_Vertical_asymmetric\META4_CoreArgo_vertical_transport_crossing_global_60S60N_10deg"
)
DEFAULT_BBOX = "0,360,-60,60"
DEFAULT_CROSSING_LATS = "-60,-50,-40,-30,-20,-10,0,10,20,30,40,50,60"
DEFAULT_LAT_BANDS = (
    "-60:-55,-55:-50,-50:-45,-45:-40,-40:-35,-35:-30,"
    "-30:-25,-25:-20,-20:-15,-15:-10,-10:-5,-5:0,"
    "0:5,5:10,10:15,15:20,20:25,25:30,"
    "30:35,35:40,40:45,45:50,50:55,55:60"
)


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


def parse_crossing_lats(value: str) -> list[float]:
    lats: list[float] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        lat = float(item)
        if lat < -90 or lat > 90:
            raise argparse.ArgumentTypeError("crossing latitudes must be within [-90, 90]")
        lats.append(lat)
    if not lats:
        raise argparse.ArgumentTypeError("--crossing-lats must contain at least one latitude")
    return lats


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
            out.extend(struct.pack("<d", float(value) if value is not None else math.nan))
    return bytes(out)


def write_npz_from_grid_json(json_path: Path, npz_path: Path) -> None:
    grid = json.loads(json_path.read_text(encoding="utf-8"))
    array_keys = [
        "x_over_R",
        "y_over_R",
        "z_rho_m",
        "z_rho_raw_m",
        "z_rho_anom_m",
        "u_argo_m_s",
        "v_argo_m_s",
        "wpk_observed_m_s",
        "sample_count",
        "mapped_support",
        "wpk_mapped_support",
        "term1_m_s",
        "term2_m_s",
        "rebuild_w_m_s",
        "term1_depth_positive_m_s",
        "term2_depth_positive_m_s",
        "rebuild_w_raw_depth_positive_m_s",
        "term1_plus_m_s",
        "term1_minus_m_s",
        "term2_abs_m_s",
        "term2_rel_m_s",
        "rebuild_plus_abs_m_s",
        "rebuild_minus_rel_m_s",
        "sample_term1_m_s",
        "sample_term2_m_s",
        "sample_rebuild_w_m_s",
    ]
    arrays = {key: grid[key] for key in array_keys if key in grid}
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
    crossing_lats = " ".join(f"{lat:.12g}" for lat in args.crossing_lats)
    target_label = latitude_crossing_label(args.target_lat, args.intersect_radius_r)
    argo_mat = matlab_quote(args.argo_mat)
    history_argo_mat = matlab_quote(args.history_argo_mat)
    meta_dir = matlab_quote(args.meta_dir)
    boa_pden_root = matlab_quote(args.boa_pden_root)
    output_root = matlab_quote(args.output_root)
    manifest = matlab_quote(manifest_path)
    max_matches = int(args.max_matches_per_group)
    template = """
argo_mat = '@ARGO_MAT@';
history_argo_mat = '@HISTORY_ARGO_MAT@';
meta_dir = '@META_DIR@';
boa_pden_root = '@BOA_PDEN_ROOT@';
output_root = '@OUTPUT_ROOT@';
bbox = [@BBOX@];
lat_bands = [@LAT_BANDS@];
crossing_lats = [@CROSSING_LATS@];
selection_mode = '@SELECTION_MODE@';
target_lat = @TARGET_LAT@;
intersect_radius_r = @INTERSECT_RADIUS_R@;
target_label = '@TARGET_LABEL@';
time_window_days = @TIME_WINDOW_DAYS@;
grid_n = @GRID_N@;
min_bin_count = @MIN_BIN_COUNT@;
plot_filled_gradient = @PLOT_FILLED_GRADIENT@;
grid_mapping = '@GRID_MAPPING@';
smooth_passes = @SMOOTH_PASSES@;
cressman_radius_r = @CRESSMAN_RADIUS_R@;
cressman_min_obs = @CRESSMAN_MIN_OBS@;
sample_gradient_max_profiles = @SAMPLE_GRADIENT_MAX_PROFILES@;
rho0_mode = '@RHO0_MODE@';
z_mode = '@Z_MODE@';
boa_background_mode = '@BOA_BACKGROUND_MODE@';
velocity_source = '@VELOCITY_SOURCE@';
match_mode = '@MATCH_MODE@';
max_matches_per_group = @MAX_MATCHES@;
core_min_m = @CORE_MIN_M@;
core_max_m = @CORE_MAX_M@;
z_rho_min_m = @Z_RHO_MIN_M@;
z_rho_max_m = @Z_RHO_MAX_M@;
min_drho_dz = @MIN_DRHO_DZ@;
max_rho_bracket_dz_m = @MAX_RHO_BRACKET_DZ_M@;
earth_radius_m = 6371000;
deg_m = earth_radius_m * pi / 180;
density_variable = '@DENSITY_VARIABLE@';

if exist(output_root, 'dir') ~= 7
    mkdir(output_root);
end

method_md = fullfile(output_root, 'METHOD_ASSUMPTIONS_ZH.md');
write_method_doc(method_md, argo_mat, history_argo_mat, meta_dir, boa_pden_root, output_root, bbox, lat_bands, crossing_lats, selection_mode, target_lat, intersect_radius_r, target_label, match_mode, velocity_source, z_mode, boa_background_mode, time_window_days, core_min_m, core_max_m, density_variable, z_rho_min_m, z_rho_max_m, min_drho_dz, max_rho_bracket_dz_m);

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

boa_clim = struct();
if strcmp(z_mode, 'anomaly_boa_climatology')
    fprintf('Loading BOA monthly climatology from %s\\n', boa_pden_root);
    boa_clim = load_boa_monthly_climatology(boa_pden_root);
end

fprintf('Computing Argo parking drift\\n');
argo_u = nan(size(argo_time));
argo_v = nan(size(argo_time));
argo_wpk = nan(size(argo_time));
history_match_mask = false(size(argo_time));
if strcmp(velocity_source, 'argo1000m_match')
    H = load(history_argo_mat, 'I_Time', 'I_Lon', 'I_Lat', 'I_ParkDepth', 'I_PF', 'I_Upk', 'I_Vpk', 'I_Wpk');
    [argo_u, argo_v, argo_wpk, history_match_mask] = match_history_argo1000m( ...
        argo_pf, argo_time, argo_lon, argo_lat, argo_park, bbox, core_min_m, core_max_m, H);
    fprintf('History velocity matched profiles: %d\\n', nnz(history_match_mask));
else
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
end

argo_base_mask = argo_lon >= bbox(1) & argo_lon <= bbox(2) & argo_lat >= bbox(3) & argo_lat <= bbox(4) & ...
    argo_park >= core_min_m & argo_park <= core_max_m & isfinite(argo_u) & isfinite(argo_v);

polarities = {'cyclonic','anticyclonic'};
grid_json_files = {};
summary_rows = {};
summary_header = {'polarity','lat_band','match_count','unique_argo_count','duplicate_match_count','ring_0_1R','ring_1_2R','ring_2_4R','valid_grid_cells','valid_grid_fraction','mean_cx_raw_m_s','mean_u_bg_m_s','cx_rel_m_s','mean_radius_km','history_velocity_match_count','boa_bg_valid_count','boa_bg_valid_fraction','mean_wpk_observed_m_s','corr_rebuild_wpk','corr_sample_rebuild_wpk','q95_abs_rebuild_1e6_m_s','q95_abs_sample_rebuild_1e6_m_s','q95_abs_wpk_1e6_m_s','output_dir'};
if strcmp(selection_mode, 'crossing_lat')
    group_count = numel(crossing_lats);
else
    group_count = size(lat_bands, 1);
end
combined_matches = cell(group_count, 1);
for b = 1:group_count
    combined_matches{b} = cell(0, 33);
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

    for b = 1:group_count
        if strcmp(selection_mode, 'crossing_lat')
            this_target_lat = crossing_lats(b);
            band_label = crossing_label(this_target_lat, intersect_radius_r);
            cross_distance_m = abs(meta_lat - this_target_lat) * deg_m;
            meta_band = find(meta_lon >= bbox(1) & meta_lon <= bbox(2) & isfinite(meta_radius) & meta_radius > 0 & ...
                cross_distance_m <= meta_radius * intersect_radius_r);
            if isempty(meta_band)
                argo_band = [];
            else
                argo_lat_window_m = (intersect_radius_r + 4) * max(meta_radius(meta_band));
                argo_band = find(argo_base_mask & abs(argo_lat - this_target_lat) * deg_m <= argo_lat_window_m);
            end
        else
            lat_min = lat_bands(b,1);
            lat_max = lat_bands(b,2);
            band_label = lat_band_label(lat_min, lat_max);
            argo_band = find(argo_base_mask & argo_lat >= lat_min & argo_lat < lat_max);
            meta_band = find(meta_lon >= bbox(1) & meta_lon <= bbox(2) & meta_lat >= lat_min & meta_lat < lat_max & isfinite(meta_radius) & meta_radius > 0);
        end
        group_dir = fullfile(output_root, polarity, band_label);
        if exist(group_dir, 'dir') ~= 7
            mkdir(group_dir);
        end
        [matches, grid] = build_group(argo_band, meta_band, polarity, band_label, ...
            argo_lon, argo_lat, argo_time, argo_park, argo_pf, argo_u, argo_v, argo_wpk, history_match_mask, rho, depth, ...
            meta_lon, meta_lat, meta_time, meta_track, meta_radius, meta_cx, ...
            time_window_days, grid_n, min_bin_count, plot_filled_gradient, grid_mapping, smooth_passes, cressman_radius_r, cressman_min_obs, sample_gradient_max_profiles, rho0_mode, ...
            z_mode, boa_clim, z_rho_min_m, z_rho_max_m, min_drho_dz, max_rho_bracket_dz_m, match_mode, max_matches_per_group, deg_m);
        write_group_outputs(group_dir, matches, grid, polarity, band_label);
        grid_json_files{end+1} = fullfile(group_dir, 'composite_grid.json'); %#ok<SAGROW>
        summary_rows(end+1,:) = summary_from_matches(matches, grid, polarity, band_label, group_dir); %#ok<SAGROW>
        combined_matches{b} = [combined_matches{b}; matches]; %#ok<SAGROW>
    end
end

combined_rows = write_combined_outputs(output_root, lat_bands, crossing_lats, combined_matches, selection_mode, target_label, intersect_radius_r, match_mode, min_bin_count, plot_filled_gradient, grid_mapping, smooth_passes, cressman_radius_r, cressman_min_obs, sample_gradient_max_profiles, grid_n);
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

function [u_out, v_out, wpk_out, matched] = match_history_argo1000m(argo_pf, argo_time, argo_lon, argo_lat, argo_park, bbox, core_min_m, core_max_m, H)
    u_out = nan(size(argo_time));
    v_out = nan(size(argo_time));
    wpk_out = nan(size(argo_time));
    matched = false(size(argo_time));
    hist_lon = double(H.I_Lon);
    hist_lon(hist_lon < 0) = hist_lon(hist_lon < 0) + 360;
    hist_lat = double(H.I_Lat);
    hist_time = double(H.I_Time);
    hist_park = double(H.I_ParkDepth);
    hist_pf = double(H.I_PF);
    hist_mask = hist_lon >= bbox(1) & hist_lon <= bbox(2) & hist_lat >= bbox(3) & hist_lat <= bbox(4) & ...
        hist_park >= core_min_m & hist_park <= core_max_m & isfinite(H.I_Upk) & isfinite(H.I_Vpk) & isfinite(H.I_Wpk);
    argo_mask = argo_lon >= bbox(1) & argo_lon <= bbox(2) & argo_lat >= bbox(3) & argo_lat <= bbox(4) & ...
        argo_park >= core_min_m & argo_park <= core_max_m;
    hist_idx = find(hist_mask);
    argo_idx = find(argo_mask);
    if isempty(hist_idx) || isempty(argo_idx)
        return
    end
    hist_key = argo_match_key(hist_pf(hist_idx), hist_time(hist_idx), hist_lon(hist_idx), hist_lat(hist_idx));
    argo_key = argo_match_key(argo_pf(argo_idx), argo_time(argo_idx), argo_lon(argo_idx), argo_lat(argo_idx));
    [tf, loc] = ismember(argo_key, hist_key);
    if any(tf)
        good_argo = argo_idx(tf);
        good_hist = hist_idx(loc(tf));
        u_out(good_argo) = double(H.I_Upk(good_hist));
        v_out(good_argo) = double(H.I_Vpk(good_hist));
        wpk_out(good_argo) = double(H.I_Wpk(good_hist));
        matched(good_argo) = true;
    end
end

function key = argo_match_key(pf, time, lon, lat)
    key = string(round(double(pf))) + "_" + string(round(double(time) * 1e6)) + "_" + ...
        string(round(double(lon) * 1e4)) + "_" + string(round(double(lat) * 1e4));
end

function boa = load_boa_monthly_climatology(root_dir)
    files = dir(fullfile(root_dir, 'PDen1000_*.mat'));
    if isempty(files)
        error('No BOA potential-density files found in %s', root_dir);
    end
    sums = cell(12, 1);
    counts = cell(12, 1);
    lon_in = [];
    lat_in = [];
    pres = [];
    for k = 1:numel(files)
        name = files(k).name;
        tok = regexp(name, 'PDen1000_\\d{4}(\\d{2})\\.mat', 'tokens', 'once');
        if isempty(tok)
            continue
        end
        mon = str2double(tok{1});
        if ~isfinite(mon) || mon < 1 || mon > 12
            continue
        end
        S = load(fullfile(files(k).folder, files(k).name), 'Den', 'lon_in', 'lat_in', 'pres');
        if isempty(lon_in)
            lon_in = double(S.lon_in(:));
            lat_in = double(S.lat_in(:));
            pres = double(S.pres(:));
        end
        D = double(S.Den);
        ok = isfinite(D);
        D(~ok) = 0;
        if isempty(sums{mon})
            sums{mon} = D;
            counts{mon} = double(ok);
        else
            sums{mon} = sums{mon} + D;
            counts{mon} = counts{mon} + double(ok);
        end
    end
    den = nan([numel(lon_in), numel(lat_in), numel(pres), 12]);
    month_count = zeros(12, 1);
    for mon = 1:12
        if isempty(sums{mon})
            continue
        end
        C = counts{mon};
        tmp = sums{mon} ./ C;
        tmp(C == 0) = NaN;
        den(:,:,:,mon) = tmp;
        month_count(mon) = max(C(:));
    end
    boa = struct('lon', lon_in, 'lat', lat_in, 'pres', pres, 'den', den, 'month_count', month_count);
end

function profile = boa_density_profile_at(boa, lon, lat, month_id)
    profile = nan(size(boa.pres));
    if isempty(fieldnames(boa)) || ~isfinite(lon) || ~isfinite(lat) || ~isfinite(month_id)
        return
    end
    month_id = max(1, min(12, round(month_id)));
    lon = mod(lon, 360);
    if lon < min(boa.lon)
        lon = lon + 360;
    end
    lon2 = boa.lon;
    den = boa.den(:,:,:,month_id);
    if lon > max(lon2)
        lon2 = [lon2; lon2(1) + 360];
        den = cat(1, den, den(1,:,:));
    end
    if lat < min(boa.lat) || lat > max(boa.lat)
        return
    end
    ix2 = find(lon2 >= lon, 1, 'first');
    iy2 = find(boa.lat >= lat, 1, 'first');
    if isempty(ix2) || isempty(iy2) || ix2 <= 1 || iy2 <= 1
        return
    end
    ix1 = ix2 - 1;
    iy1 = iy2 - 1;
    x1 = lon2(ix1); x2 = lon2(ix2);
    y1 = boa.lat(iy1); y2 = boa.lat(iy2);
    if x2 == x1 || y2 == y1
        return
    end
    wx = (lon - x1) / (x2 - x1);
    wy = (lat - y1) / (y2 - y1);
    p11 = squeeze(den(ix1,iy1,:));
    p21 = squeeze(den(ix2,iy1,:));
    p12 = squeeze(den(ix1,iy2,:));
    p22 = squeeze(den(ix2,iy2,:));
    profile = (1-wx) * (1-wy) * p11 + wx * (1-wy) * p21 + (1-wx) * wy * p12 + wx * wy * p22;
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
    argo_lon, argo_lat, argo_time, argo_park, argo_pf, argo_u, argo_v, argo_wpk, history_match_mask, rho, depth, ...
    meta_lon, meta_lat, meta_time, meta_track, meta_radius, meta_cx, ...
    time_window_days, grid_n, min_bin_count, plot_filled_gradient, grid_mapping, smooth_passes, cressman_radius_r, cressman_min_obs, sample_gradient_max_profiles, rho0_mode, ...
    z_mode, boa_clim, z_rho_min_m, z_rho_max_m, min_drho_dz, max_rho_bracket_dz_m, match_mode, max_matches_per_group, deg_m)

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
        rho0 = interp1(depth, double(rho(ii,:)), argo_park(ii), 'linear', NaN);
        if ~isfinite(rho0)
            continue
        end
        if strcmp(match_mode, 'all')
            use_pos = find(isfinite(r_norm) & r_norm <= 4);
        else
            [best_r, best_pos] = min(r_norm);
            if isfinite(best_r) && best_r <= 4
                use_pos = best_pos;
            else
                use_pos = [];
            end
        end
        for pp = 1:numel(use_pos)
            pos = use_pos(pp);
            jj = candidates(pos);
            best_dx = dx(pos);
            best_dy = dy(pos);
            this_r = r_norm(pos);
            row_count = row_count + 1;
            ring = ring_label(this_r);
            rows(row_count,:) = {polarity, band_label, ii, argo_pf(ii), argo_time(ii), argo_lon(ii), argo_lat(ii), ...
                argo_park(ii), argo_u(ii), argo_v(ii), argo_wpk(ii), rho0, NaN, NaN, NaN, NaN, NaN, NaN, ...
                meta_track(jj), meta_time(jj), meta_lon(jj), meta_lat(jj), meta_radius(jj), ...
                best_dx / meta_radius(jj), best_dy / meta_radius(jj), this_r, ring, meta_cx(jj), history_match_mask(ii), ...
                NaN, NaN, NaN, false}; %#ok<AGROW>
            if max_matches_per_group > 0 && row_count >= max_matches_per_group
                break
            end
        end
        if max_matches_per_group > 0 && row_count >= max_matches_per_group
            break
        end
    end
    rows = apply_rho0_mode(rows, rho, depth, argo_park, rho0_mode, z_mode, boa_clim, z_rho_min_m, z_rho_max_m, min_drho_dz, max_rho_bracket_dz_m);
    matches = rows;
    grid = composite_grid(matches, grid_n, min_bin_count, plot_filled_gradient, grid_mapping, smooth_passes, cressman_radius_r, cressman_min_obs, sample_gradient_max_profiles);
    grid.match_mode = match_mode;
    grid.rho0_mode = rho0_mode;
    grid.z_mode = z_mode;
    grid.z_rho_min_m = z_rho_min_m;
    grid.z_rho_max_m = z_rho_max_m;
    grid.min_drho_dz = min_drho_dz;
    grid.max_rho_bracket_dz_m = max_rho_bracket_dz_m;
end

function rows = apply_rho0_mode(rows, rho, depth, argo_park, rho0_mode, z_mode, boa_clim, z_rho_min_m, z_rho_max_m, min_drho_dz, max_rho_bracket_dz_m)
    if isempty(rows)
        return
    end
    profile_rho0 = cell2mat(rows(:,12));
    if strcmp(rho0_mode, 'band_median')
        target_rho0 = median(profile_rho0, 'omitnan');
        target_rho0 = repmat(target_rho0, size(rows, 1), 1);
    else
        target_rho0 = profile_rho0;
    end
    keep = false(size(rows, 1), 1);
    for rr = 1:size(rows, 1)
        ii = rows{rr,3};
        [z_rho, crossing_count, bracket_dz, local_drho_dz] = isopycnal_depth_qc(depth, double(rho(ii,:)), target_rho0(rr), argo_park(ii));
        if isfinite(z_rho) && z_rho >= z_rho_min_m && z_rho <= z_rho_max_m && ...
                isfinite(bracket_dz) && bracket_dz <= max_rho_bracket_dz_m && ...
                isfinite(local_drho_dz) && abs(local_drho_dz) >= min_drho_dz
            rows{rr,12} = target_rho0(rr);
            rows{rr,13} = z_rho;
            rows{rr,16} = crossing_count;
            rows{rr,17} = bracket_dz;
            rows{rr,18} = local_drho_dz;
            keep(rr) = true;
        end
    end
    rows = rows(keep,:);
    if isempty(rows)
        return
    end
    z_rho = cell2mat(rows(:,13));
    r_norm = cell2mat(rows(:,26));
    if strcmp(z_mode, 'anomaly_boa_climatology')
        z_bg_all = nan(size(z_rho));
        z_anom = nan(size(z_rho));
        keep_boa = false(size(rows, 1), 1);
        for rr = 1:size(rows, 1)
            [~, month_id, ~] = datevec(rows{rr,5});
            boa_profile = boa_density_profile_at(boa_clim, rows{rr,6}, rows{rr,7}, month_id);
            boa_profile = align_density_units(boa_profile, rows{rr,12});
            [z_bg, bg_crossing_count, bg_bracket_dz, bg_drho_dz] = isopycnal_depth_qc(boa_clim.pres, boa_profile, rows{rr,12}, rows{rr,8});
            rows{rr,30} = bg_crossing_count;
            rows{rr,31} = bg_bracket_dz;
            rows{rr,32} = bg_drho_dz;
            bg_ok = isfinite(z_bg) && z_bg >= z_rho_min_m && z_bg <= z_rho_max_m && ...
                isfinite(bg_bracket_dz) && bg_bracket_dz <= max_rho_bracket_dz_m && ...
                isfinite(bg_drho_dz) && abs(bg_drho_dz) >= min_drho_dz;
            rows{rr,33} = bg_ok;
            if bg_ok
                z_bg_all(rr) = z_bg;
                z_anom(rr) = z_rho(rr) - z_bg;
                keep_boa(rr) = true;
            end
        end
        rows = rows(keep_boa,:);
        if isempty(rows)
            return
        end
        z_bg_all = z_bg_all(keep_boa);
        z_anom = z_anom(keep_boa);
    elseif strcmp(z_mode, 'anomaly_farfield') || strcmp(z_mode, 'anomaly_farfield_plane')
        farfield = z_rho(r_norm >= 2 & r_norm <= 4 & isfinite(z_rho));
        if strcmp(z_mode, 'anomaly_farfield_plane')
            x_norm = cell2mat(rows(:,24));
            y_norm = cell2mat(rows(:,25));
            far_ok = r_norm >= 2 & r_norm <= 4 & isfinite(z_rho) & isfinite(x_norm) & isfinite(y_norm);
            if nnz(far_ok) >= 10
                Xfit = [ones(nnz(far_ok), 1), x_norm(far_ok), y_norm(far_ok)];
                coef = Xfit \\ z_rho(far_ok);
                z_bg_all = [ones(size(z_rho)), x_norm(:), y_norm(:)] * coef;
            elseif isempty(farfield)
                z_bg_all = repmat(median(z_rho, 'omitnan'), size(rows, 1), 1);
            else
                z_bg_all = repmat(median(farfield, 'omitnan'), size(rows, 1), 1);
            end
        elseif isempty(farfield)
            z_bg = median(z_rho, 'omitnan');
            z_bg_all = repmat(z_bg, size(rows, 1), 1);
        else
            z_bg = median(farfield, 'omitnan');
            z_bg_all = repmat(z_bg, size(rows, 1), 1);
        end
        z_anom = z_rho - z_bg_all;
    else
        z_bg_all = nan(size(z_rho));
        z_anom = z_rho;
    end
    for rr = 1:size(rows, 1)
        rows{rr,14} = z_bg_all(rr);
        rows{rr,15} = z_anom(rr);
        if size(rows, 2) < 33 || isempty(rows{rr,33})
            rows{rr,33} = strcmp(z_mode, 'anomaly_boa_climatology') == false;
        end
    end
end

function profile = align_density_units(profile, rho0)
    med = median(profile, 'omitnan');
    if isfinite(med) && isfinite(rho0)
        if med > 1000 && rho0 < 100
            profile = profile - 1000;
        elseif med < 100 && rho0 > 1000
            profile = profile + 1000;
        end
    end
end

function dx = local_dx_m(lon_a, lon_b, lat_ref, deg_m)
    dlon = lon_a - lon_b;
    dlon(dlon > 180) = dlon(dlon > 180) - 360;
    dlon(dlon < -180) = dlon(dlon < -180) + 360;
    dx = dlon .* deg_m .* cosd(lat_ref);
end

function [z, crossing_count, bracket_dz, local_drho_dz] = isopycnal_depth_qc(depth, profile, rho0, target_depth)
    z = NaN;
    crossing_count = 0;
    bracket_dz = NaN;
    local_drho_dz = NaN;
    depth = depth(:);
    profile = profile(:);
    good = isfinite(depth) & isfinite(profile);
    depth = depth(good);
    profile = profile(good);
    if numel(depth) < 3 || ~isfinite(rho0) || ~isfinite(target_depth)
        return
    end
    [depth, order] = sort(depth);
    profile = profile(order);
    [depth, ia] = unique(depth, 'stable');
    profile = profile(ia);
    hits = nan(0, 4);
    for k = 1:numel(depth)-1
        r1 = profile(k) - rho0;
        r2 = profile(k+1) - rho0;
        if r1 == 0 || r2 == 0 || r1 * r2 > 0 || profile(k) == profile(k+1)
            continue
        end
        frac = (rho0 - profile(k)) / (profile(k+1) - profile(k));
        z_hit = depth(k) + frac * (depth(k+1) - depth(k));
        dz = abs(depth(k+1) - depth(k));
        drhodz = (profile(k+1) - profile(k)) / (depth(k+1) - depth(k));
        hits(end+1,:) = [z_hit, dz, drhodz, k]; %#ok<AGROW>
    end
    crossing_count = size(hits, 1);
    if ~isempty(hits)
        [~, idx] = min(abs(hits(:,1) - target_depth));
        z = hits(idx,1);
        bracket_dz = hits(idx,2);
        local_drho_dz = hits(idx,3);
    end
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

function grid = composite_grid(matches, grid_n, min_bin_count, plot_filled_gradient, grid_mapping, smooth_passes, cressman_radius_r, cressman_min_obs, sample_gradient_max_profiles)
    x_vec = linspace(-4, 4, grid_n);
    y_vec = linspace(-4, 4, grid_n);
    [X, Y] = meshgrid(x_vec, y_vec);
    nan_grid = nan(size(X));
    count_grid = zeros(size(X));
    grid = struct('x', X, 'y', Y, 'z', nan_grid, 'z_raw', nan_grid, 'z_anom', nan_grid, 'u', nan_grid, 'v', nan_grid, 'wpk', nan_grid, ...
        'mapped_support', count_grid, 'wpk_mapped_support', count_grid, ...
        'count', count_grid, 'term1', nan_grid, 'term2', nan_grid, 'rebuild_w', nan_grid, ...
        'term1_depth_positive', nan_grid, 'term2_depth_positive', nan_grid, 'rebuild_w_raw_depth_positive', nan_grid, ...
        'term1_plus', nan_grid, 'term1_minus', nan_grid, 'term2_abs', nan_grid, 'term2_rel', nan_grid, ...
        'rebuild_plus_abs', nan_grid, 'rebuild_minus_rel', nan_grid, ...
        'sample_term1', nan_grid, 'sample_term2', nan_grid, 'sample_rebuild_w', nan_grid, 'corr_sample_rebuild_wpk', NaN, ...
        'mean_cx_raw', NaN, 'mean_u_bg', NaN, 'cx_rel', NaN, 'mean_radius_m', NaN, ...
        'mean_wpk_observed', NaN, 'corr_rebuild_wpk', NaN, ...
        'grid_mapping', grid_mapping, 'cressman_radius_r', cressman_radius_r, 'cressman_min_obs', cressman_min_obs, ...
        'rho0_mode', '', 'z_mode', '', 'z_rho_min_m', NaN, 'z_rho_max_m', NaN, 'min_drho_dz', NaN, 'max_rho_bracket_dz_m', NaN);
    if isempty(matches)
        return
    end
    x = cell2mat(matches(:,24)); x = x(:);
    y = cell2mat(matches(:,25)); y = y(:);
    z_raw = cell2mat(matches(:,13)); z_raw = z_raw(:);
    z = cell2mat(matches(:,15)); z = z(:);
    u = cell2mat(matches(:,9)); u = u(:);
    v = cell2mat(matches(:,10)); v = v(:);
    wpk = cell2mat(matches(:,11)); wpk = wpk(:);
    cx_raw = cell2mat(matches(:,28)); cx_raw = cx_raw(:);
    radius = cell2mat(matches(:,23)); radius = radius(:);
    sample_term1 = nan(size(z));
    sample_term2 = nan(size(z));
    sample_rebuild_w = nan(size(z));
    grid.mean_cx_raw = mean(cx_raw, 'omitnan');
    grid.mean_u_bg = mean(u, 'omitnan');
    grid.cx_rel = grid.mean_cx_raw - grid.mean_u_bg;
    grid.mean_radius_m = mean(radius, 'omitnan');
    grid.mean_wpk_observed = mean(wpk, 'omitnan');
    edges = linspace(-4, 4, grid_n + 1);
    xb = discretize(x, edges);
    yb = discretize(y, edges);
    n = min([numel(xb), numel(yb), numel(z), numel(z_raw), numel(u), numel(v), numel(wpk)]);
    xb = xb(1:n); yb = yb(1:n); z = z(1:n); z_raw = z_raw(1:n); u = u(1:n); v = v(1:n); wpk = wpk(1:n);
    cx_raw = cx_raw(1:n); radius = radius(1:n);
    [sample_term1, sample_term2, sample_rebuild_w] = sample_gradient_terms(x(1:n), y(1:n), z, u, v, cx_raw, radius, grid.cx_rel, cressman_radius_r, cressman_min_obs, sample_gradient_max_profiles);
    valid = isfinite(xb) & isfinite(yb);
    subs = [yb(valid), xb(valid)];
    grid.count = accumarray(subs, 1, [grid_n grid_n], @sum, 0);
    grid.z = accumarray(subs, z(valid), [grid_n grid_n], @(q) median(q, 'omitnan'), NaN);
    grid.z_raw = accumarray(subs, z_raw(valid), [grid_n grid_n], @(q) median(q, 'omitnan'), NaN);
    grid.z_anom = grid.z;
    grid.u = accumarray(subs, u(valid), [grid_n grid_n], @(q) median(q, 'omitnan'), NaN);
    grid.v = accumarray(subs, v(valid), [grid_n grid_n], @(q) median(q, 'omitnan'), NaN);
    grid.wpk = accumarray(subs, wpk(valid), [grid_n grid_n], @(q) median(q, 'omitnan'), NaN);
    grid.sample_term1 = accumarray(subs, sample_term1(valid), [grid_n grid_n], @(q) median(q, 'omitnan'), NaN);
    grid.sample_term2 = accumarray(subs, sample_term2(valid), [grid_n grid_n], @(q) median(q, 'omitnan'), NaN);
    grid.sample_rebuild_w = accumarray(subs, sample_rebuild_w(valid), [grid_n grid_n], @(q) median(q, 'omitnan'), NaN);
    if strcmp(grid_mapping, 'cressman')
        [mapped, support_all] = cressman_map_multi(x, y, [z, z_raw, u, v, wpk], X, Y, cressman_radius_r, cressman_min_obs);
        grid.z = mapped(:,:,1);
        grid.z_raw = mapped(:,:,2);
        grid.z_anom = grid.z;
        grid.u = mapped(:,:,3);
        grid.v = mapped(:,:,4);
        grid.wpk = mapped(:,:,5);
        grid.mapped_support = support_all;
        grid.wpk_mapped_support = support_all;
    elseif strcmp(grid_mapping, 'scattered')
        grid.z = scattered_map(x, y, z, X, Y);
        grid.z_raw = scattered_map(x, y, z_raw, X, Y);
        grid.z_anom = grid.z;
        grid.u = scattered_map(x, y, u, X, Y);
        grid.v = scattered_map(x, y, v, X, Y);
        grid.wpk = scattered_map(x, y, wpk, X, Y);
        grid.mapped_support = double(isfinite(grid.z) & isfinite(grid.u) & isfinite(grid.v) & hypot(X, Y) <= 4);
        grid.wpk_mapped_support = double(isfinite(grid.wpk) & hypot(X, Y) <= 4);
    else
        grid.mapped_support = grid.count;
        grid.wpk_mapped_support = grid.count;
    end
    if smooth_passes > 0
        support = mapping_support_mask(grid_mapping, grid.mapped_support, grid.count, min_bin_count, cressman_min_obs);
        grid.z = smooth2_supported(grid.z, support, smooth_passes);
        grid.z_raw = smooth2_supported(grid.z_raw, support, smooth_passes);
        grid.z_anom = grid.z;
        grid.u = smooth2_supported(grid.u, support, smooth_passes);
        grid.v = smooth2_supported(grid.v, support, smooth_passes);
        grid.wpk = smooth2_supported(grid.wpk, grid.wpk_mapped_support >= cressman_min_obs, smooth_passes);
    end
    dx_m = mean(diff(x_vec)) * grid.mean_radius_m;
    dy_m = mean(diff(y_vec)) * grid.mean_radius_m;
    if isfinite(dx_m) && dx_m > 0 && isfinite(dy_m) && dy_m > 0
        [dzdx, dzdy] = gradient(fillmissing2(grid.z), dx_m, dy_m);
        support = mapping_support_mask(grid_mapping, grid.mapped_support, grid.count, min_bin_count, cressman_min_obs);
        grid.term1_plus = mask_to_support(grid.cx_rel .* dzdx, support);
        grid.term1_minus = mask_to_support(-grid.cx_rel .* dzdx, support);
        grid.term2_abs = mask_to_support(grid.u .* dzdx + grid.v .* dzdy, support);
        grid.term2_rel = mask_to_support((grid.u - grid.mean_cx_raw) .* dzdx + grid.v .* dzdy, support);
        grid.rebuild_plus_abs = mask_to_support(grid.term1_plus + grid.term2_abs, support);
        grid.rebuild_minus_rel = mask_to_support(grid.term1_minus + grid.term2_rel, support);
        grid.term1_depth_positive = grid.term1_minus;
        grid.term2_depth_positive = grid.term2_rel;
        grid.rebuild_w_raw_depth_positive = grid.rebuild_minus_rel;
        if plot_filled_gradient
            grid.term1 = grid.cx_rel .* dzdx;
        else
            grid.term1 = grid.term1_plus;
        end
        grid.term2 = mask_to_support(-grid.term2_rel, support);
        grid.rebuild_w = mask_to_support(grid.term1 + grid.term2, support);
        grid.corr_rebuild_wpk = spatial_corr(grid.rebuild_w, grid.wpk);
        grid.corr_sample_rebuild_wpk = spatial_corr(grid.sample_rebuild_w, grid.wpk);
    end
end

function [term1, term2, rebuild] = sample_gradient_terms(x, y, z, u, v, cx_raw, radius, cx_rel, radius_r, min_obs, max_profiles)
    term1 = nan(size(z));
    term2 = nan(size(z));
    rebuild = nan(size(z));
    good_all = isfinite(x) & isfinite(y) & isfinite(z) & isfinite(u) & isfinite(v) & isfinite(cx_raw) & isfinite(radius) & radius > 0 & hypot(x, y) <= 4;
    if nnz(good_all) < min_obs || ~isfinite(radius_r) || radius_r <= 0
        return
    end
    eval_idx = find(good_all);
    if isfinite(max_profiles) && max_profiles > 0 && numel(eval_idx) > max_profiles
        eval_idx = eval_idx(unique(round(linspace(1, numel(eval_idx), max_profiles))));
    end
    for kk = 1:numel(eval_idx)
        ii = eval_idx(kk);
        if ~good_all(ii)
            continue
        end
        d2 = (x - x(ii)).^2 + (y - y(ii)).^2;
        inside = good_all & d2 < radius_r ^ 2;
        if nnz(inside) < max(min_obs, 6)
            continue
        end
        Xfit = [ones(nnz(inside), 1), x(inside) - x(ii), y(inside) - y(ii)];
        coef = Xfit \\ z(inside);
        dzdx = coef(2) / radius(ii);
        dzdy = coef(3) / radius(ii);
        term1(ii) = cx_rel * dzdx;
        term2(ii) = -((u(ii) - cx_raw(ii)) * dzdx + v(ii) * dzdy);
        rebuild(ii) = term1(ii) + term2(ii);
    end
end

function c = spatial_corr(A, B)
    a = A(:);
    b = B(:);
    good = isfinite(a) & isfinite(b);
    if nnz(good) < 3
        c = NaN;
        return
    end
    R = corrcoef(a(good), b(good));
    c = R(1,2);
end

function support = mapping_support_mask(grid_mapping, mapped_support, count_grid, min_bin_count, cressman_min_obs)
    if strcmp(grid_mapping, 'cressman')
        support = mapped_support >= cressman_min_obs;
    elseif strcmp(grid_mapping, 'scattered')
        support = mapped_support > 0;
    else
        support = count_grid >= min_bin_count;
    end
end

function [Z, support_count] = cressman_map(x, y, v, X, Y, radius_r, min_obs)
    Z = NaN(size(X));
    support_count = zeros(size(X));
    good = isfinite(x) & isfinite(y) & isfinite(v) & hypot(x, y) <= 4;
    x = x(good); y = y(good); v = v(good);
    if isempty(x) || ~isfinite(radius_r) || radius_r <= 0
        return
    end
    r2_limit = radius_r ^ 2;
    for ii = 1:numel(X)
        d2 = (x - X(ii)).^2 + (y - Y(ii)).^2;
        inside = d2 < r2_limit;
        n_inside = nnz(inside);
        support_count(ii) = n_inside;
        if n_inside >= min_obs
            w = (r2_limit - d2(inside)) ./ (r2_limit + d2(inside));
            ok = isfinite(w) & w > 0;
            if any(ok)
                vals = v(inside);
                vals = vals(ok);
                w = w(ok);
                Z(ii) = sum(w .* vals) ./ sum(w);
            end
        end
    end
    Z(hypot(X, Y) > 4) = NaN;
    support_count(hypot(X, Y) > 4) = 0;
end

function [Z, support_count] = cressman_map_multi(x, y, V, X, Y, radius_r, min_obs)
    Z = NaN([size(X), size(V, 2)]);
    support_count = zeros(size(X));
    good = isfinite(x) & isfinite(y) & all(isfinite(V), 2) & hypot(x, y) <= 4;
    x = x(good);
    y = y(good);
    V = V(good,:);
    if isempty(x) || ~isfinite(radius_r) || radius_r <= 0
        return
    end
    r2_limit = radius_r ^ 2;
    for ii = 1:numel(X)
        d2 = (x - X(ii)).^2 + (y - Y(ii)).^2;
        inside = d2 < r2_limit;
        n_inside = nnz(inside);
        support_count(ii) = n_inside;
        if n_inside >= min_obs
            w = (r2_limit - d2(inside)) ./ (r2_limit + d2(inside));
            ok = isfinite(w) & w > 0;
            if any(ok)
                vals = V(inside,:);
                vals = vals(ok,:);
                w = w(ok);
                for kk = 1:size(V, 2)
                    Z(ii + (kk-1) * numel(X)) = sum(w .* vals(:,kk)) ./ sum(w);
                end
            end
        end
    end
    outside = hypot(X, Y) > 4;
    for kk = 1:size(V, 2)
        tmp = Z(:,:,kk);
        tmp(outside) = NaN;
        Z(:,:,kk) = tmp;
    end
    support_count(outside) = 0;
end

function out = smooth2_supported(A, support, passes)
    out = A;
    for n = 1:passes
        B = out;
        for i = 1:size(out,1)
            for j = 1:size(out,2)
                if support(i,j)
                    i0 = max(1, i-1); i1 = min(size(out,1), i+1);
                    j0 = max(1, j-1); j1 = min(size(out,2), j+1);
                    vals = out(i0:i1, j0:j1);
                    ok = isfinite(vals);
                    if any(ok, 'all')
                        B(i,j) = mean(vals(ok), 'omitnan');
                    end
                end
            end
        end
        out = B;
        out(~support) = NaN;
    end
end

function Z = scattered_map(x, y, v, X, Y)
    Z = NaN(size(X));
    good = isfinite(x) & isfinite(y) & isfinite(v) & hypot(x, y) <= 4;
    if nnz(good) < 3
        return
    end
    try
        F = scatteredInterpolant(x(good), y(good), v(good), 'natural', 'none');
        Z = F(X, Y);
    catch
        F = scatteredInterpolant(x(good), y(good), v(good), 'linear', 'none');
        Z = F(X, Y);
    end
    Z(hypot(X, Y) > 4) = NaN;
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
    grid.match_count = size(matches, 1);
    if isempty(matches)
        grid.unique_argo_count = 0;
        grid.boa_bg_valid_count = 0;
    else
        grid.unique_argo_count = numel(unique(cell2mat(matches(:,3))));
        if size(matches, 2) >= 33
            grid.boa_bg_valid_count = sum(cell2mat(matches(:,33)));
        else
            grid.boa_bg_valid_count = 0;
        end
    end
    grid.duplicate_match_count = grid.match_count - grid.unique_argo_count;
    header = {'polarity','lat_band','argo_index','platform','argo_time','argo_lon','argo_lat','parking_depth_m','u_argo_m_s','v_argo_m_s','wpk_observed_m_s','rho0','z_rho_m','z_rho_bg_m','z_rho_anom_m','rho_crossing_count','rho_bracket_dz_m','local_drho_dz','eddy_track','eddy_time','eddy_lon','eddy_lat','eddy_radius_m','x_over_R','y_over_R','r_over_R','ring','cx_raw_m_s','history_velocity_matched','boa_rho_crossing_count','boa_rho_bracket_dz_m','boa_local_drho_dz','boa_bg_valid'};
    writecell(clean_write_cells([header; matches]), fullfile(group_dir, 'matched_core_argo.csv'));
    write_grid_json(fullfile(group_dir, 'composite_grid.json'), grid, polarity, band_label);
    plot_three_panel(fullfile(group_dir, 'vertical_transport_terms.png'), grid, [polarity ' ' band_label]);
    plot_sensitivity(fullfile(group_dir, 'velocity_sign_sensitivity.png'), grid, [polarity ' ' band_label]);
    plot_wpk_validation(fullfile(group_dir, 'wpk_validation.png'), grid, [polarity ' ' band_label]);
    plot_gradient_order_comparison(fullfile(group_dir, 'gradient_order_comparison.png'), grid, [polarity ' ' band_label]);
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
        unique_argo_count = 0;
        history_velocity_match_count = 0;
        boa_bg_valid_count = 0;
    else
        rings = matches(:,27);
        n = size(matches, 1);
        unique_argo_count = numel(unique(cell2mat(matches(:,3))));
        history_velocity_match_count = sum(cell2mat(matches(:,29)));
        if size(matches, 2) >= 33
            boa_bg_valid_count = sum(cell2mat(matches(:,33)));
        else
            boa_bg_valid_count = 0;
        end
    end
    duplicate_match_count = n - unique_argo_count;
    valid_cells = sum(isfinite(grid.rebuild_w(:)));
    valid_fraction = valid_cells / numel(grid.count);
    rebuild = grid.rebuild_w(:);
    sample_rebuild = grid.sample_rebuild_w(:);
    wpk = grid.wpk(:);
    if any(isfinite(rebuild))
        q95_rebuild = prctile(abs(rebuild(isfinite(rebuild))) * 1e6, 95);
    else
        q95_rebuild = NaN;
    end
    if any(isfinite(sample_rebuild))
        q95_sample_rebuild = prctile(abs(sample_rebuild(isfinite(sample_rebuild))) * 1e6, 95);
    else
        q95_sample_rebuild = NaN;
    end
    if any(isfinite(wpk))
        q95_wpk = prctile(abs(wpk(isfinite(wpk))) * 1e6, 95);
    else
        q95_wpk = NaN;
    end
    row = {polarity, band_label, n, unique_argo_count, duplicate_match_count, sum(strcmp(rings,'0-1R')), sum(strcmp(rings,'1-2R')), sum(strcmp(rings,'2-4R')), ...
        valid_cells, valid_fraction, grid.mean_cx_raw, grid.mean_u_bg, grid.cx_rel, grid.mean_radius_m / 1000, ...
        history_velocity_match_count, boa_bg_valid_count, safe_fraction(boa_bg_valid_count, n), grid.mean_wpk_observed, grid.corr_rebuild_wpk, ...
        grid.corr_sample_rebuild_wpk, q95_rebuild, q95_sample_rebuild, q95_wpk, group_dir};
end

function f = safe_fraction(a, b)
    if b > 0
        f = a / b;
    else
        f = NaN;
    end
end

function combined_rows = write_combined_outputs(output_root, lat_bands, crossing_lats, combined_matches, selection_mode, target_label, intersect_radius_r, match_mode, min_bin_count, plot_filled_gradient, grid_mapping, smooth_passes, cressman_radius_r, cressman_min_obs, sample_gradient_max_profiles, grid_n)
    combined_rows = {};
    for b = 1:numel(combined_matches)
        if strcmp(selection_mode, 'crossing_lat')
            band_label = crossing_label(crossing_lats(b), intersect_radius_r);
        else
            band_label = lat_band_label(lat_bands(b,1), lat_bands(b,2));
        end
        combined_dir = fullfile(output_root, 'combined', band_label);
        if exist(combined_dir, 'dir') ~= 7
            mkdir(combined_dir);
        end
        all_matches = combined_matches{b};
        grid = composite_grid(all_matches, grid_n, min_bin_count, plot_filled_gradient, grid_mapping, smooth_passes, cressman_radius_r, cressman_min_obs, sample_gradient_max_profiles);
        grid.match_mode = match_mode;
        write_group_outputs(combined_dir, all_matches, grid, 'combined', band_label);
        combined_rows(end+1,:) = summary_from_matches(all_matches, grid, 'combined', band_label, combined_dir); %#ok<AGROW>
    end
end

function write_grid_json(path, grid, polarity, band_label)
    G = struct();
    G.metadata = struct('polarity', polarity, 'lat_band', band_label, 'w_positive_direction', 'upward', 'depth_positive_direction', 'downward', ...
        'mean_cx_raw_m_s', grid.mean_cx_raw, ...
        'mean_u_bg_m_s', grid.mean_u_bg, 'cx_rel_m_s', grid.cx_rel, 'mean_radius_m', grid.mean_radius_m, ...
        'mean_wpk_observed_m_s', grid.mean_wpk_observed, 'corr_rebuild_wpk', grid.corr_rebuild_wpk, ...
        'grid_mapping', grid.grid_mapping, 'cressman_radius_r', grid.cressman_radius_r, 'cressman_min_obs', grid.cressman_min_obs, ...
        'match_mode', grid.match_mode, 'rho0_mode', grid.rho0_mode, 'z_mode', grid.z_mode, ...
        'z_rho_min_m', grid.z_rho_min_m, 'z_rho_max_m', grid.z_rho_max_m, 'min_drho_dz', grid.min_drho_dz, ...
        'max_rho_bracket_dz_m', grid.max_rho_bracket_dz_m, ...
        'match_count', grid.match_count, 'unique_argo_count', grid.unique_argo_count, 'duplicate_match_count', grid.duplicate_match_count, ...
        'boa_bg_valid_count', grid.boa_bg_valid_count, ...
        'valid_grid_cells', sum(isfinite(grid.rebuild_w(:))), 'total_grid_cells', numel(grid.count), ...
        'corr_sample_rebuild_wpk', grid.corr_sample_rebuild_wpk);
    G.x_over_R = grid.x;
    G.y_over_R = grid.y;
    G.z_rho_m = grid.z;
    G.z_rho_raw_m = grid.z_raw;
    G.z_rho_anom_m = grid.z_anom;
    G.u_argo_m_s = grid.u;
    G.v_argo_m_s = grid.v;
    G.wpk_observed_m_s = grid.wpk;
    G.sample_count = grid.count;
    G.mapped_support = grid.mapped_support;
    G.wpk_mapped_support = grid.wpk_mapped_support;
    G.term1_m_s = grid.term1;
    G.term2_m_s = grid.term2;
    G.rebuild_w_m_s = grid.rebuild_w;
    G.term1_depth_positive_m_s = grid.term1_depth_positive;
    G.term2_depth_positive_m_s = grid.term2_depth_positive;
    G.rebuild_w_raw_depth_positive_m_s = grid.rebuild_w_raw_depth_positive;
    G.term1_plus_m_s = grid.term1_plus;
    G.term1_minus_m_s = grid.term1_minus;
    G.term2_abs_m_s = grid.term2_abs;
    G.term2_rel_m_s = grid.term2_rel;
    G.rebuild_plus_abs_m_s = grid.rebuild_plus_abs;
    G.rebuild_minus_rel_m_s = grid.rebuild_minus_rel;
    G.sample_term1_m_s = grid.sample_term1;
    G.sample_term2_m_s = grid.sample_term2;
    G.sample_rebuild_w_m_s = grid.sample_rebuild_w;
    fid = fopen(path, 'w');
    fwrite(fid, jsonencode(G), 'char');
    fclose(fid);
end

function label = lat_band_label(lat_min, lat_max)
    label = [lat_token(lat_min) '_' lat_token(lat_max)];
end

function label = crossing_label(target_lat, intersect_radius_r)
    radius_text = regexprep(sprintf('%.6g', intersect_radius_r), '\\.', 'p');
    label = ['cross_' lat_token(target_lat) '_' radius_text 'R'];
end

function token = lat_token(lat)
    if lat < 0
        hemi = 'S';
    else
        hemi = 'N';
    end
    token = sprintf('%02.0f%s', abs(lat), hemi);
end

function plot_three_panel(path, grid, title_prefix)
    fig = figure('Visible','off','Position',[100 100 1500 430]);
    fields = {'term1','term2','rebuild_w'};
    titles = {'+c_x^{rel} dz''_\\rho/dx  (W up+)','-(u_{pk}-c_x, v_{pk}) \\cdot \\nabla z''_\\rho','rebuild W  (up+)'};
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

function plot_wpk_validation(path, grid, title_prefix)
    fig = figure('Visible','off','Position',[100 100 1020 430]);
    fields = {'rebuild_w','wpk'};
    titles = {'rebuild W (up+)','observed I\\_Wpk (up+)'};
    vals = [grid.rebuild_w(:); grid.wpk(:)];
    lim = max(abs(vals(isfinite(vals))));
    if isempty(lim) || ~isfinite(lim) || lim == 0
        lim = 2.5e-6;
    end
    lim = max(lim, 2.5e-6);
    for k = 1:2
        subplot(1,2,k);
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
        title(titles{k}, 'Interpreter', 'tex');
        xlabel('x/R'); ylabel('y/R');
    end
    sgtitle([title_prefix ' validation  (10^{-6} m s^{-1})'], 'Interpreter', 'tex');
    tmp_path = [tempname(fileparts(path)) '.png'];
    try
        exportgraphics(fig, tmp_path, 'Resolution', 180);
        if exist(path, 'file') == 2
            delete(path);
        end
        movefile(tmp_path, path, 'f');
    catch ME
        fallback_path = fullfile(fileparts(path), ['wpk_validation_' datestr(now, 'yyyymmdd_HHMMSS') '.png']);
        if exist(tmp_path, 'file') == 2
            movefile(tmp_path, fallback_path, 'f');
        end
        warning('Could not replace %s: %s. Wrote %s instead.', path, ME.message, fallback_path);
    end
    close(fig);
end

function plot_sensitivity(path, grid, title_prefix)
    fig = figure('Visible','off','Position',[100 100 1180 900]);
    fields = {'rebuild_w','rebuild_w_raw_depth_positive','term1','term2'};
    titles = {'main W upward-positive','raw W depth-positive','term1 W up+','term2 W up+'};
    vals = [];
    for k = 1:numel(fields)
        vals = [vals; grid.(fields{k})(:)]; %#ok<AGROW>
    end
    lim = max(abs(vals(isfinite(vals))));
    if isempty(lim) || ~isfinite(lim) || lim == 0
        lim = 2.5e-6;
    end
    lim = max(lim, 2.5e-6);
    for k = 1:4
        subplot(2,2,k);
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
        title(titles{k}, 'Interpreter', 'tex');
        xlabel('x/R'); ylabel('y/R');
    end
    sgtitle([title_prefix ' velocity/sign sensitivity  (10^{-6} m s^{-1})'], 'Interpreter', 'tex');
    tmp_path = [tempname(fileparts(path)) '.png'];
    try
        exportgraphics(fig, tmp_path, 'Resolution', 180);
        if exist(path, 'file') == 2
            delete(path);
        end
        movefile(tmp_path, path, 'f');
    catch ME
        fallback_path = fullfile(fileparts(path), ['velocity_sign_sensitivity_' datestr(now, 'yyyymmdd_HHMMSS') '.png']);
        if exist(tmp_path, 'file') == 2
            movefile(tmp_path, fallback_path, 'f');
        end
        warning('Could not replace %s: %s. Wrote %s instead.', path, ME.message, fallback_path);
    end
    close(fig);
end

function plot_gradient_order_comparison(path, grid, title_prefix)
    fig = figure('Visible','off','Position',[100 100 1180 430]);
    fields = {'rebuild_w','sample_rebuild_w'};
    titles = {'Cressman z'' then gradient','sample gradient then Cressman W'};
    vals = [grid.rebuild_w(:); grid.sample_rebuild_w(:)];
    lim = max(abs(vals(isfinite(vals))));
    if isempty(lim) || ~isfinite(lim) || lim == 0
        lim = 2.5e-6;
    end
    lim = max(lim, 2.5e-6);
    for k = 1:2
        subplot(1,2,k);
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
        title(titles{k}, 'Interpreter', 'tex');
        xlabel('x/R'); ylabel('y/R');
    end
    sgtitle([title_prefix ' gradient order comparison  (10^{-6} m s^{-1})'], 'Interpreter', 'tex');
    tmp_path = [tempname(fileparts(path)) '.png'];
    try
        exportgraphics(fig, tmp_path, 'Resolution', 180);
        if exist(path, 'file') == 2
            delete(path);
        end
        movefile(tmp_path, path, 'f');
    catch ME
        fallback_path = fullfile(fileparts(path), ['gradient_order_comparison_' datestr(now, 'yyyymmdd_HHMMSS') '.png']);
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

function write_method_doc(path, argo_mat, history_argo_mat, meta_dir, boa_pden_root, output_root, bbox, lat_bands, crossing_lats, selection_mode, target_lat, intersect_radius_r, target_label, match_mode, velocity_source, z_mode, boa_background_mode, time_window_days, core_min_m, core_max_m, density_variable, z_rho_min_m, z_rho_max_m, min_drho_dz, max_rho_bracket_dz_m)
    fid = fopen(path, 'w');
    fprintf(fid, '# META4.0 + Core Argo 垂直速度重建方法与假定\\n\\n');
    fprintf(fid, '- Argo 主数据源：`%s`\\n', argo_mat);
    fprintf(fid, '- 历史 Argo 速度校准源：`%s`\\n', history_argo_mat);
    fprintf(fid, '- META4.0 涡旋源：`%s`\\n', meta_dir);
    fprintf(fid, '- BOA 背景位密源：`%s`\\n', boa_pden_root);
    fprintf(fid, '- 输出根目录：`%s`\\n', output_root);
    fprintf(fid, '- 运行范围：`%.1fE-%.1fE, %.1f-%.1f latitude`\\n', bbox(1), bbox(2), bbox(3), bbox(4));
    if strcmp(selection_mode, 'crossing_lat')
        fprintf(fid, '- 样本选择：默认 crossing。涡旋本体 `%.3gR` 跨过纬线 `%s`，逐纬线输出 `cross_XX_%.3gR` 目录。Argo profile 不再按纬度带预筛，只由 bbox、parking depth、时间窗和 `0-4R` 空间匹配决定。\\n', intersect_radius_r, crossing_list_text(crossing_lats), intersect_radius_r);
    else
        fprintf(fid, '- 纬度带：');
        for i=1:size(lat_bands,1), fprintf(fid, '`%s` ', lat_band_label(lat_bands(i,1), lat_bands(i,2))); end
    end
    fprintf(fid, '\\n- Core Argo 限定：`%.0f-%.0f m` parking depth。\\n', core_min_m, core_max_m);
    fprintf(fid, '- 时间匹配：Argo profile 与 META 轨迹点相差不超过 `%.1f day`。\\n', time_window_days);
    fprintf(fid, '- 匹配模式：`%s`。`nearest` 表示每条 Argo 只归属最近的 `r/R` 涡旋；`all` 表示一条 Argo 可在所有满足时间窗和 `0-4R` 的涡旋坐标系中重复使用。\\n', match_mode);
    fprintf(fid, '- 速度来源：`%s`。`argo1000m_match` 使用历史文件 `I_Upk/I_Vpk/I_Wpk` 与 TEOS profile 的 `I_PF/I_Time/I_Lon/I_Lat` 近似键匹配；`profile_diff` 是旧的相邻 profile 位置差近似。\\n', velocity_source);
    fprintf(fid, '- 空间分区：保存 `0-1R`、`1-2R`、`2-4R`，图像网格为 `x/R, y/R = -4..4`。\\n');
    fprintf(fid, '- 密度变量：`%s`。默认 `rho0` 为每个纬度带/极性在实际 parking depth 处密度的中位数；可用 `--rho0-mode profile` 做旧口径对照。\\n', density_variable);
    fprintf(fid, '- `z_mode`：`%s`。`anomaly_boa_climatology` 使用 BOA 多年同月局地背景密度剖面，`anomaly_farfield_plane` 仅作为内部远场平面对照。\\n', z_mode);
    fprintf(fid, '- BOA 背景模式：`%s`。默认按月份平均 `PDen1000_YYYYMM.mat`，对每个 profile 的 `lon/lat/month` 双线性插值得到背景密度剖面。\\n', boa_background_mode);
    fprintf(fid, '- `z_rho` 反插值：profile 和 BOA 背景均只使用显式 bracket crossing，不再用全剖面 fallback；有效窗口 `%.0f-%.0f m`，`abs(local_drho_dz) >= %.3g`，bracket 厚度 `<= %.0f m`。\\n', z_rho_min_m, z_rho_max_m, min_drho_dz, max_rho_bracket_dz_m);
    fprintf(fid, '- 默认网格化：Cressman objective mapping。权重 `w=(R_c^2-r^2)/(R_c^2+r^2)`，仅使用 `R_c` 内样本；默认 `R_c=0.5R`、每格至少 `3` 个样本。`sample_count` 是原始 bin 覆盖，`mapped_support` 是 Cressman 支撑样本数。\\n');
    fprintf(fid, '- 深度变量约定：`z_rho_m`、`z_rho_bg_m`、`z_rho_anom_m` 均保存为正深度向下，便于海洋剖面阅读。\\n');
    fprintf(fid, '- W 符号约定：`rebuild_w_m_s`、`term1_m_s`、`term2_m_s` 统一为向上为正，与历史 `I_Wpk` 中 `z=-Depth` 后计算 `Dz/Dt` 的口径一致；同时保留 `rebuild_w_raw_depth_positive_m_s` 作为深度向下正公式对照。\\n');
    fprintf(fid, '- `c_x_raw` 来自 META track 相邻点中央差分；`u_bg` 为同 crossing 组、同极性、匹配 Core Argo 的 parking drift 纬向均值；`c_x_rel = mean(c_x_raw) - mean(u_bg)`。主图采用 `term1 = +c_x_rel dz''_rho/dx`，`term2 = -[(u_pk-c_x_raw, v_pk) · grad(z''_rho)]`，`rebuild_W = term1 + term2`。\\n');
    fprintf(fid, '- BOA_Argo 只作为 gridded 密度背景，不直接推导背景速度。\\n\\n');
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
    fprintf(fid, '- mean observed I_Wpk：`%.6g m/s`\\n', grid.mean_wpk_observed);
    fprintf(fid, '- corr(rebuild_W, I_Wpk)：`%.6g`\\n', grid.corr_rebuild_wpk);
    fprintf(fid, '- corr(sample-gradient W, I_Wpk)：`%.6g`\\n', grid.corr_sample_rebuild_wpk);
    fprintf(fid, '- BOA 背景有效样本：`%d / %d`\\n', grid.boa_bg_valid_count, size(matches,1));
    fprintf(fid, '- mean radius：`%.3f km`\\n', grid.mean_radius_m / 1000);
    valid_cells = sum(isfinite(grid.rebuild_w(:)));
    valid_fraction = valid_cells / numel(grid.count);
    fprintf(fid, '- 有样本支撑网格：`%d / %d (%.2f%%)`。\\n', valid_cells, numel(grid.count), valid_fraction * 100);
    fprintf(fid, '- 符号约定：W 向上为正；`z_rho` 和 `z_rho_anom` 为正深度向下。\\n');
    fprintf(fid, '- 输出：`matched_core_argo.csv`、`composite_grid.npz`、`vertical_transport_terms.png`、`wpk_validation.png`、`gradient_order_comparison.png`、`velocity_sign_sensitivity.png`。图像显示为 `10^-6 m/s`，网格文件保存原始 `m/s`，白色为空样本格点。\\n');
    fclose(fid);
end

function write_summary_doc(path, summary_rows, output_root)
    fid = fopen(path, 'w');
    fprintf(fid, '# META4.0 + Core Argo 垂直速度重建运行摘要\\n\\n');
    fprintf(fid, '- 输出根目录：`%s`\\n', output_root);
    fprintf(fid, '- 主图变量：W 向上为正，`term1 = +c_x_rel dz''_rho/dx`，`term2 = -[(u_pk-c_x_raw, v_pk) · grad(z''_rho)]`，`rebuild_W = term1 + term2`。\\n');
    fprintf(fid, '- 深度变量：`z_rho_m`、`z_rho_bg_m`、`z_rho_anom_m` 仍为正深度向下；JSON/NPZ 保存原始 `m/s`；PNG 色标显示为 `10^-6 m/s`；白色为空样本格点。\\n');
    fprintf(fid, '- 若使用 `--max-matches-per-group` 做 smoke run，覆盖率会很低；正式结果应使用默认 `0` 读取全部匹配。\\n\\n');
    fprintf(fid, '| polarity | lat_band | matches | unique Argo | duplicated | 0-1R | 1-2R | 2-4R | valid grid %% | BOA bg %% | c_x_rel m/s | corr W/Wpk | corr sample/Wpk | q95 rebuild | q95 sample | q95 Wpk |\\n');
    fprintf(fid, '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\\n');
    for i=1:size(summary_rows,1)
        fprintf(fid, '| %s | %s | %d | %d | %d | %d | %d | %d | %.2f | %.2f | %.6g | %.3g | %.3g | %.3g | %.3g | %.3g |\\n', summary_rows{i,1}, summary_rows{i,2}, summary_rows{i,3}, summary_rows{i,4}, summary_rows{i,5}, summary_rows{i,6}, summary_rows{i,7}, summary_rows{i,8}, summary_rows{i,10} * 100, summary_rows{i,17} * 100, summary_rows{i,13}, summary_rows{i,19}, summary_rows{i,20}, summary_rows{i,21}, summary_rows{i,22}, summary_rows{i,23});
    end
    fclose(fid);
end

function text = crossing_list_text(crossing_lats)
    parts = cell(1, numel(crossing_lats));
    for i = 1:numel(crossing_lats)
        parts{i} = lat_token(crossing_lats(i));
    end
    text = strjoin(parts, ',');
end
"""
    return (
        template.replace("@ARGO_MAT@", argo_mat)
        .replace("@HISTORY_ARGO_MAT@", history_argo_mat)
        .replace("@META_DIR@", meta_dir)
        .replace("@BOA_PDEN_ROOT@", boa_pden_root)
        .replace("@OUTPUT_ROOT@", output_root)
        .replace("@BBOX@", bbox)
        .replace("@LAT_BANDS@", lat_bands)
        .replace("@CROSSING_LATS@", crossing_lats)
        .replace("@SELECTION_MODE@", str(args.selection_mode).replace("'", "''"))
        .replace("@TARGET_LAT@", f"{float(args.target_lat):.12g}")
        .replace("@INTERSECT_RADIUS_R@", f"{float(args.intersect_radius_r):.12g}")
        .replace("@TARGET_LABEL@", target_label.replace("'", "''"))
        .replace("@TIME_WINDOW_DAYS@", f"{float(args.time_window_days):.12g}")
        .replace("@GRID_N@", str(int(args.grid_n)))
        .replace("@MIN_BIN_COUNT@", str(int(args.min_bin_count)))
        .replace("@PLOT_FILLED_GRADIENT@", "true" if args.plot_filled_gradient else "false")
        .replace("@GRID_MAPPING@", str(args.grid_mapping).replace("'", "''"))
        .replace("@SMOOTH_PASSES@", str(int(args.smooth_passes)))
        .replace("@CRESSMAN_RADIUS_R@", f"{float(args.cressman_radius_r):.12g}")
        .replace("@CRESSMAN_MIN_OBS@", str(int(args.cressman_min_obs)))
        .replace("@SAMPLE_GRADIENT_MAX_PROFILES@", str(int(args.sample_gradient_max_profiles)))
        .replace("@RHO0_MODE@", str(args.rho0_mode).replace("'", "''"))
        .replace("@Z_MODE@", str(args.z_mode).replace("'", "''"))
        .replace("@BOA_BACKGROUND_MODE@", str(args.boa_background_mode).replace("'", "''"))
        .replace("@VELOCITY_SOURCE@", str(args.velocity_source).replace("'", "''"))
        .replace("@MATCH_MODE@", str(args.match_mode).replace("'", "''"))
        .replace("@MAX_MATCHES@", str(max_matches))
        .replace("@CORE_MIN_M@", f"{float(args.core_min_m):.12g}")
        .replace("@CORE_MAX_M@", f"{float(args.core_max_m):.12g}")
        .replace("@Z_RHO_MIN_M@", f"{float(args.z_rho_min_m):.12g}")
        .replace("@Z_RHO_MAX_M@", f"{float(args.z_rho_max_m):.12g}")
        .replace("@MIN_DRHO_DZ@", f"{float(args.min_drho_dz):.12g}")
        .replace("@MAX_RHO_BRACKET_DZ_M@", f"{float(args.max_rho_bracket_dz_m):.12g}")
        .replace("@DENSITY_VARIABLE@", str(args.density_variable).replace("'", "''"))
        .replace("@MANIFEST@", manifest)
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Rebuild vertical transport terms from META4.0 eddies and Core Argo.")
    parser.add_argument("--argo-mat", type=Path, default=DEFAULT_ARGO_MAT)
    parser.add_argument("--history-argo-mat", type=Path, default=DEFAULT_HISTORY_ARGO_MAT)
    parser.add_argument("--meta-dir", type=Path, default=DEFAULT_META_DIR)
    parser.add_argument("--boa-pden-root", type=Path, default=DEFAULT_BOA_PDEN_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--bbox", type=parse_bbox, default=parse_bbox(DEFAULT_BBOX))
    parser.add_argument("--lat-bands", type=parse_lat_bands, default=parse_lat_bands(DEFAULT_LAT_BANDS))
    parser.add_argument(
        "--crossing-lats",
        type=parse_crossing_lats,
        default=None,
        help="Comma-separated crossing latitudes. Defaults to -60,-50,...,60 unless --target-lat is explicitly used as a single-lat shortcut.",
    )
    parser.add_argument(
        "--selection-mode",
        choices=("lat_band", "crossing_lat"),
        default="crossing_lat",
        help="Choose standard latitude-band sampling or eddies whose radius crosses a target latitude.",
    )
    parser.add_argument("--target-lat", type=float, default=20.0)
    parser.add_argument("--intersect-radius-r", type=float, default=1.0)
    parser.add_argument("--time-window-days", type=float, default=1.0)
    parser.add_argument("--core-min-m", type=float, default=900.0)
    parser.add_argument("--core-max-m", type=float, default=1100.0)
    parser.add_argument("--z-rho-min-m", type=float, default=900.0)
    parser.add_argument("--z-rho-max-m", type=float, default=1100.0)
    parser.add_argument("--min-drho-dz", type=float, default=1e-5)
    parser.add_argument("--max-rho-bracket-dz-m", type=float, default=150.0)
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
        "--grid-mapping",
        choices=("cressman", "scattered", "bin"),
        default="cressman",
        help="Map profiles to the composite grid. Cressman is the default objective mapping; scattered is a diagnostic interpolation; bin keeps sampled cells only.",
    )
    parser.add_argument("--smooth-passes", type=int, default=2)
    parser.add_argument(
        "--cressman-radius-r",
        type=float,
        default=0.5,
        help="Cressman influence radius in eddy-radius units for --grid-mapping cressman.",
    )
    parser.add_argument(
        "--cressman-min-obs",
        type=int,
        default=3,
        help="Minimum profiles within the Cressman influence radius required to map a grid point.",
    )
    parser.add_argument(
        "--sample-gradient-max-profiles",
        type=int,
        default=1000,
        help="Deterministic cap for the sample-gradient-then-composite diagnostic. Use 0 for all profiles.",
    )
    parser.add_argument(
        "--rho0-mode",
        choices=("band_median", "profile"),
        default="band_median",
        help="Choose the target isopycnal. band_median uses one shared rho0 per latitude/polarity group; profile keeps the old per-profile parking-density target.",
    )
    parser.add_argument(
        "--z-mode",
        choices=("anomaly_boa_climatology", "anomaly_farfield_plane", "anomaly_farfield", "absolute"),
        default="anomaly_boa_climatology",
        help="Use BOA monthly climatology, far-field plane/scalar isopycnal displacement anomaly, or absolute z_rho.",
    )
    parser.add_argument(
        "--boa-background-mode",
        choices=("monthly_climatology",),
        default="monthly_climatology",
        help="BOA background density mode. monthly_climatology averages PDen1000_YYYYMM files by calendar month.",
    )
    parser.add_argument(
        "--velocity-source",
        choices=("argo1000m_match", "profile_diff"),
        default="argo1000m_match",
        help="Use matched historical Argo1000m I_Upk/I_Vpk/I_Wpk, or the older profile-position-difference velocity proxy.",
    )
    parser.add_argument(
        "--match-mode",
        choices=("nearest", "all"),
        default="nearest",
        help="nearest assigns each Argo profile to the closest normalized eddy; all repeats it for every eddy within 4R and the time window.",
    )
    parser.add_argument(
        "--max-matches-per-group",
        type=int,
        default=0,
        help="Debug/smoke limit. Use 0 for all matched profiles.",
    )
    raw_argv = sys.argv[1:]
    argv: list[str] = []
    i = 0
    while i < len(raw_argv):
        if raw_argv[i] == "--crossing-lats" and i + 1 < len(raw_argv):
            argv.append(f"--crossing-lats={raw_argv[i + 1]}")
            i += 2
        else:
            argv.append(raw_argv[i])
            i += 1
    args = parser.parse_args(argv)
    target_lat_explicit = any(item == "--target-lat" or item.startswith("--target-lat=") for item in raw_argv)
    if args.crossing_lats is None:
        if args.selection_mode == "crossing_lat" and target_lat_explicit:
            args.crossing_lats = [args.target_lat]
        else:
            args.crossing_lats = parse_crossing_lats(DEFAULT_CROSSING_LATS)
    npz_paths = run_matlab_pipeline(args)
    for path in npz_paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
