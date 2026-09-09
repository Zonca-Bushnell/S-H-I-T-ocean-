argo_mat = '@ARGO_MAT@';
history_argo_mat = '@HISTORY_ARGO_MAT@';
meta_dir = '@META_DIR@';
boa_pden_root = '@BOA_PDEN_ROOT@';
cache_root = '@CACHE_ROOT@';
output_root = '@OUTPUT_ROOT@';
bbox = [@BBOX@];
crossing_lats = [@CROSSING_LATS@];
target_lat = @TARGET_LAT@;
intersect_radius_r = @INTERSECT_RADIUS_R@;
target_label = '@TARGET_LABEL@';
time_window_days = @TIME_WINDOW_DAYS@;
grid_n = @GRID_N@;
min_bin_count = @MIN_BIN_COUNT@;
plot_filled_gradient = @PLOT_FILLED_GRADIENT@;
smooth_passes = @SMOOTH_PASSES@;
cressman_radius_r = @CRESSMAN_RADIUS_R@;
cressman_min_obs = @CRESSMAN_MIN_OBS@;
sample_gradient_max_profiles = @SAMPLE_GRADIENT_MAX_PROFILES@;
vertical_mode = '@VERTICAL_MODE@';
fast_sensitivity_2d = @FAST_SENSITIVITY_2D@;
sensitivity_workers = @SENSITIVITY_WORKERS@;
sensitivity_config_names = {@SENSITIVITY_CONFIGS@};
compute_device = '@COMPUTE_DEVICE@';
matlab_profile_enabled = @MATLAB_PROFILE@;
diagnose_reversal_factors = @DIAGNOSE_REVERSAL_FACTORS@;
compare_z_geometry_modes = @COMPARE_Z_GEOMETRY_MODES@;
z_geometry_mode = '@Z_GEOMETRY_MODE@';
write_matched_csv_flag = @WRITE_MATCHED_CSV@;
write_grid_json_flag = @WRITE_GRID_JSON@;
write_grid_nc_flag = @WRITE_GRID_NC@;
write_summary_csv_flag = @WRITE_SUMMARY_CSV@;
depth_levels = [@DEPTH_LEVELS@];
section_axis = '@SECTION_AXIS@';
section_half_width_r = @SECTION_HALF_WIDTH_R@;
boa_background_mode = '@BOA_BACKGROUND_MODE@';
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

matlab_module_root = '@MATLAB_MODULE_ROOT@';
if exist(matlab_module_root, 'dir') == 7
    addpath(genpath(matlab_module_root));
end
global USE_GPU_CRESSMAN;
USE_GPU_CRESSMAN = should_use_gpu(compute_device);
global QC_WORKERS;
QC_WORKERS = sensitivity_workers;
global WRITE_MATCHED_CSV WRITE_GRID_JSON WRITE_GRID_NC WRITE_SUMMARY_CSV;
WRITE_MATCHED_CSV = write_matched_csv_flag;
WRITE_GRID_JSON = write_grid_json_flag;
WRITE_GRID_NC = write_grid_nc_flag;
WRITE_SUMMARY_CSV = write_summary_csv_flag;

if exist(output_root, 'dir') ~= 7
    mkdir(output_root);
end
if exist(cache_root, 'dir') ~= 7
    mkdir(cache_root);
end

if matlab_profile_enabled
    profile clear;
    profile on;
    log_step('MATLAB profiler enabled.');
end

method_md = fullfile(output_root, 'METHOD_ASSUMPTIONS_ZH.md');
write_method_doc(method_md, argo_mat, history_argo_mat, meta_dir, boa_pden_root, output_root, bbox, crossing_lats, target_lat, intersect_radius_r, match_mode, boa_background_mode, time_window_days, core_min_m, core_max_m, density_variable, z_rho_min_m, z_rho_max_m, min_drho_dz, max_rho_bracket_dz_m);

fprintf('Loading Argo vectors from %s\n', argo_mat);
A = load(argo_mat, 'I_Time', 'I_Lon', 'I_Lat', 'I_ParkDepth', 'I_PF', density_variable, 'Depth');
rho = A.(density_variable);
argo_lon = double(A.I_Lon);
argo_lon(argo_lon < 0) = argo_lon(argo_lon < 0) + 360;
argo_lat = double(A.I_Lat);
argo_time = double(A.I_Time);
argo_park = double(A.I_ParkDepth);
argo_pf = double(A.I_PF);
depth = double(A.Depth(:));

fprintf('Loading BOA monthly climatology from %s\n', boa_pden_root);
boa_clim = load_boa_monthly_climatology(boa_pden_root, fullfile(cache_root, 'BOA_PDen1000_monthly_climatology.mat'));

fprintf('Computing Argo parking drift\n');
argo_u = nan(size(argo_time));
argo_v = nan(size(argo_time));
argo_wpk = nan(size(argo_time));
history_match_mask = false(size(argo_time));
H = load(history_argo_mat, 'I_Time', 'I_Lon', 'I_Lat', 'I_ParkDepth', 'I_PF', 'I_Upk', 'I_Vpk', 'I_Wpk');
[argo_u, argo_v, argo_wpk, history_match_mask] = match_history_argo1000m( ...
    argo_pf, argo_time, argo_lon, argo_lat, argo_park, bbox, core_min_m, core_max_m, H);
fprintf('History velocity matched profiles: %d\n', nnz(history_match_mask));

argo_base_mask = argo_lon >= bbox(1) & argo_lon <= bbox(2) & argo_lat >= bbox(3) & argo_lat <= bbox(4) & ...
    argo_park >= core_min_m & argo_park <= core_max_m & isfinite(argo_u) & isfinite(argo_v);

if diagnose_reversal_factors
    grid_files = run_reversal_factor_diagnosis(output_root, meta_dir, bbox, target_lat, intersect_radius_r, ...
        argo_base_mask, argo_lon, argo_lat, argo_time, argo_park, argo_pf, argo_u, argo_v, argo_wpk, history_match_mask, rho, depth, ...
        time_window_days, depth_levels, deg_m, cache_root, boa_clim, max_matches_per_group);
    manifest = struct();
    manifest.grid_files = grid_files;
    manifest.output_root = output_root;
    text = jsonencode(manifest);
    fid = fopen('@MANIFEST@', 'w');
    fwrite(fid, text, 'char');
    fclose(fid);
    return
end

if compare_z_geometry_modes
    grid_files = run_zgeometry_comparison(output_root, meta_dir, bbox, target_lat, intersect_radius_r, ...
        argo_base_mask, argo_lon, argo_lat, argo_time, argo_park, argo_pf, argo_u, argo_v, argo_wpk, history_match_mask, rho, depth, ...
        time_window_days, depth_levels, deg_m, cache_root, boa_clim, max_matches_per_group, z_geometry_mode);
    manifest = struct();
    manifest.grid_files = grid_files;
    manifest.output_root = output_root;
    text = jsonencode(manifest);
    fid = fopen('@MANIFEST@', 'w');
    fwrite(fid, text, 'char');
    fclose(fid);
    return
end

if fast_sensitivity_2d
    grid_files = run_fast_sensitivity_2d(output_root, meta_dir, bbox, crossing_lats, target_lat, intersect_radius_r, ...
        argo_base_mask, argo_lon, argo_lat, argo_time, argo_park, argo_pf, argo_u, argo_v, argo_wpk, history_match_mask, rho, depth, ...
        time_window_days, min_bin_count, plot_filled_gradient, sample_gradient_max_profiles, boa_clim, ...
        z_rho_min_m, z_rho_max_m, min_drho_dz, max_rho_bracket_dz_m, match_mode, max_matches_per_group, deg_m, sensitivity_workers, sensitivity_config_names);
    manifest = struct();
    manifest.grid_files = grid_files;
    manifest.output_root = output_root;
    text = jsonencode(manifest);
    fid = fopen('@MANIFEST@', 'w');
    fwrite(fid, text, 'char');
    fclose(fid);
    return
end

polarities = {'cyclonic','anticyclonic'};
grid_files = {};
summary_rows = {};
if strcmp(vertical_mode, 'isopycnal_depth_stack') || strcmp(vertical_mode, 'thermal_wind_depth_stack')
    summary_header = {'polarity','lat_band','match_count','unique_argo_count','duplicate_match_count','ring_0_1R','ring_1_2R','ring_2_4R','depth_count','valid_voxels','valid_voxel_fraction','mean_valid_cells_per_depth','mean_cx_raw_m_s','mean_u_bg_m_s','cx_rel_m_s','mean_radius_km','history_velocity_match_count','mean_boa_bg_valid_fraction','mean_profile_valid_fraction','q95_abs_w_1e6_m_s','output_dir'};
else
    summary_header = {'polarity','lat_band','match_count','unique_argo_count','duplicate_match_count','ring_0_1R','ring_1_2R','ring_2_4R','valid_grid_cells','valid_grid_fraction','mean_cx_raw_m_s','mean_u_bg_m_s','cx_rel_m_s','mean_radius_km','history_velocity_match_count','boa_bg_valid_count','boa_bg_valid_fraction','mean_wpk_observed_m_s','corr_rebuild_wpk','corr_sample_rebuild_wpk','q95_abs_rebuild_1e6_m_s','q95_abs_sample_rebuild_1e6_m_s','q95_abs_wpk_1e6_m_s','output_dir'};
end
group_count = numel(crossing_lats);
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
    meta_cx = track_cx(meta_lon, meta_lat, meta_time, meta_track, deg_m);

    for b = 1:group_count
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
        group_dir = fullfile(output_root, polarity, band_label);
        if exist(group_dir, 'dir') ~= 7
            mkdir(group_dir);
        end
        if strcmp(vertical_mode, 'isopycnal_depth_stack') || strcmp(vertical_mode, 'thermal_wind_depth_stack')
            group_timer = tic;
            [matches, grid3d] = build_group_3d(argo_band, meta_band, polarity, band_label, ...
                argo_lon, argo_lat, argo_time, argo_park, argo_pf, argo_u, argo_v, argo_wpk, history_match_mask, rho, depth, ...
                meta_lon, meta_lat, meta_time, meta_track, meta_radius, meta_cx, ...
                time_window_days, grid_n, min_bin_count, smooth_passes, cressman_radius_r, cressman_min_obs, ...
                'anomaly_boa_climatology', boa_clim, depth_levels, min_drho_dz, max_rho_bracket_dz_m, match_mode, max_matches_per_group, deg_m, section_axis, section_half_width_r, vertical_mode, cache_root);
            grid_file = write_group_outputs_3d(group_dir, matches, grid3d, polarity, band_label);
            log_step(sprintf('%s %s 3D group finished in %.1f s', polarity, band_label, toc(group_timer)));
            grid_files{end+1} = grid_file; %#ok<SAGROW>
            summary_rows(end+1,:) = summary_from_matches_3d(matches, grid3d, polarity, band_label, group_dir); %#ok<SAGROW>
        else
            [matches, grid] = build_group(argo_band, meta_band, polarity, band_label, ...
                argo_lon, argo_lat, argo_time, argo_park, argo_pf, argo_u, argo_v, argo_wpk, history_match_mask, rho, depth, ...
                meta_lon, meta_lat, meta_time, meta_track, meta_radius, meta_cx, ...
                time_window_days, grid_n, min_bin_count, plot_filled_gradient, smooth_passes, cressman_radius_r, cressman_min_obs, sample_gradient_max_profiles, ...
                boa_clim, z_rho_min_m, z_rho_max_m, min_drho_dz, max_rho_bracket_dz_m, match_mode, max_matches_per_group, deg_m);
            grid_file = write_group_outputs(group_dir, matches, grid, polarity, band_label, true);
            grid_files{end+1} = grid_file; %#ok<SAGROW>
            summary_rows(end+1,:) = summary_from_matches(matches, grid, polarity, band_label, group_dir); %#ok<SAGROW>
        end
    end
end

if strcmp(vertical_mode, 'isopycnal_depth_stack') || strcmp(vertical_mode, 'thermal_wind_depth_stack')
    summary_path = fullfile(output_root, 'SUMMARY_3D_W');
else
    summary_path = fullfile(output_root, 'SUMMARY');
end
write_summary_table_mat([summary_path '.mat'], summary_header, summary_rows);
global WRITE_SUMMARY_CSV;
if WRITE_SUMMARY_CSV
    writecell([summary_header; summary_rows], [summary_path '.csv']);
end
if strcmp(vertical_mode, 'isopycnal_depth_stack') || strcmp(vertical_mode, 'thermal_wind_depth_stack')
    write_summary_doc_3d(fullfile(output_root, 'RUN_SUMMARY_ZH.md'), summary_rows, output_root, depth_levels, section_axis, section_half_width_r);
else
    write_summary_doc(fullfile(output_root, 'RUN_SUMMARY_ZH.md'), summary_rows, output_root);
end

if matlab_profile_enabled
    profile_info = profile('info');
    profile off;
    save(fullfile(output_root, 'matlab_profile_info.mat'), 'profile_info', '-v7.3');
    profsave(profile_info, fullfile(output_root, 'matlab_profile_html'));
    log_step('MATLAB profiler saved.');
end

manifest = struct();
manifest.grid_files = grid_files;
manifest.output_root = output_root;
text = jsonencode(manifest);
fid = fopen('@MANIFEST@', 'w');
fwrite(fid, text, 'char');
fclose(fid);
