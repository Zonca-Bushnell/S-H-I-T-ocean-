argo_mat = '@ARGO_MAT@';
history_argo_mat = '@HISTORY_ARGO_MAT@';
meta_dir = '@META_DIR@';
boa_pden_root = '@BOA_PDEN_ROOT@';
cache_root = '@CACHE_ROOT@';
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
vertical_mode = '@VERTICAL_MODE@';
fast_sensitivity_2d = @FAST_SENSITIVITY_2D@;
sensitivity_workers = @SENSITIVITY_WORKERS@;
sensitivity_config_names = {@SENSITIVITY_CONFIGS@};
compute_device = '@COMPUTE_DEVICE@';
matlab_profile_enabled = @MATLAB_PROFILE@;
diagnose_reversal_factors = @DIAGNOSE_REVERSAL_FACTORS@;
write_matched_csv_flag = @WRITE_MATCHED_CSV@;
write_grid_json_flag = @WRITE_GRID_JSON@;
write_grid_nc_flag = @WRITE_GRID_NC@;
write_summary_csv_flag = @WRITE_SUMMARY_CSV@;
depth_levels = [@DEPTH_LEVELS@];
section_axis = '@SECTION_AXIS@';
section_half_width_r = @SECTION_HALF_WIDTH_R@;
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
write_method_doc(method_md, argo_mat, history_argo_mat, meta_dir, boa_pden_root, output_root, bbox, lat_bands, crossing_lats, selection_mode, target_lat, intersect_radius_r, target_label, match_mode, velocity_source, z_mode, boa_background_mode, time_window_days, core_min_m, core_max_m, density_variable, z_rho_min_m, z_rho_max_m, min_drho_dz, max_rho_bracket_dz_m);

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

boa_clim = struct();
if strcmp(z_mode, 'anomaly_boa_climatology')
    fprintf('Loading BOA monthly climatology from %s\n', boa_pden_root);
    boa_clim = load_boa_monthly_climatology(boa_pden_root, fullfile(cache_root, 'BOA_PDen1000_monthly_climatology.mat'));
end

fprintf('Computing Argo parking drift\n');
argo_u = nan(size(argo_time));
argo_v = nan(size(argo_time));
argo_wpk = nan(size(argo_time));
history_match_mask = false(size(argo_time));
if strcmp(velocity_source, 'argo1000m_match')
    H = load(history_argo_mat, 'I_Time', 'I_Lon', 'I_Lat', 'I_ParkDepth', 'I_PF', 'I_Upk', 'I_Vpk', 'I_Wpk');
    [argo_u, argo_v, argo_wpk, history_match_mask] = match_history_argo1000m( ...
        argo_pf, argo_time, argo_lon, argo_lat, argo_park, bbox, core_min_m, core_max_m, H);
    fprintf('History velocity matched profiles: %d\n', nnz(history_match_mask));
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

if fast_sensitivity_2d
    grid_files = run_fast_sensitivity_2d(output_root, meta_dir, bbox, crossing_lats, target_lat, intersect_radius_r, ...
        argo_base_mask, argo_lon, argo_lat, argo_time, argo_park, argo_pf, argo_u, argo_v, argo_wpk, history_match_mask, rho, depth, ...
        time_window_days, min_bin_count, plot_filled_gradient, grid_mapping, sample_gradient_max_profiles, rho0_mode, z_mode, boa_clim, ...
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
if strcmp(selection_mode, 'crossing_lat')
    group_count = numel(crossing_lats);
else
    group_count = size(lat_bands, 1);
end
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
        if strcmp(vertical_mode, 'isopycnal_depth_stack') || strcmp(vertical_mode, 'thermal_wind_depth_stack')
            group_timer = tic;
            [matches, grid3d] = build_group_3d(argo_band, meta_band, polarity, band_label, ...
                argo_lon, argo_lat, argo_time, argo_park, argo_pf, argo_u, argo_v, argo_wpk, history_match_mask, rho, depth, ...
                meta_lon, meta_lat, meta_time, meta_track, meta_radius, meta_cx, ...
                time_window_days, grid_n, min_bin_count, grid_mapping, smooth_passes, cressman_radius_r, cressman_min_obs, ...
                z_mode, boa_clim, depth_levels, min_drho_dz, max_rho_bracket_dz_m, match_mode, max_matches_per_group, deg_m, section_axis, section_half_width_r, vertical_mode, cache_root);
            grid_file = write_group_outputs_3d(group_dir, matches, grid3d, polarity, band_label);
            log_step(sprintf('%s %s 3D group finished in %.1f s', polarity, band_label, toc(group_timer)));
            grid_files{end+1} = grid_file; %#ok<SAGROW>
            summary_rows(end+1,:) = summary_from_matches_3d(matches, grid3d, polarity, band_label, group_dir); %#ok<SAGROW>
        else
            [matches, grid] = build_group(argo_band, meta_band, polarity, band_label, ...
                argo_lon, argo_lat, argo_time, argo_park, argo_pf, argo_u, argo_v, argo_wpk, history_match_mask, rho, depth, ...
                meta_lon, meta_lat, meta_time, meta_track, meta_radius, meta_cx, ...
                time_window_days, grid_n, min_bin_count, plot_filled_gradient, grid_mapping, smooth_passes, cressman_radius_r, cressman_min_obs, sample_gradient_max_profiles, rho0_mode, ...
                z_mode, boa_clim, z_rho_min_m, z_rho_max_m, min_drho_dz, max_rho_bracket_dz_m, match_mode, max_matches_per_group, deg_m);
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

function use_gpu = should_use_gpu(compute_device)
    use_gpu = false;
    if strcmp(compute_device, 'cpu')
        fprintf('Compute device: CPU requested.\n');
        return
    end
    try
        n_gpu = gpuDeviceCount;
        if n_gpu > 0
            gpuDevice(1);
            use_gpu = true;
            fprintf('Compute device: GPU Cressman enabled on device 1.\n');
        end
    catch ME
        if strcmp(compute_device, 'gpu')
            warning('GPU requested but unavailable; falling back to CPU: %s', ME.message);
        else
            fprintf('Compute device: GPU unavailable, using CPU Cressman.\n');
        end
        use_gpu = false;
    end
end

function grid_files = run_fast_sensitivity_2d(output_root, meta_dir, bbox, crossing_lats, target_lat, intersect_radius_r, ...
    argo_base_mask, argo_lon, argo_lat, argo_time, argo_park, argo_pf, argo_u, argo_v, argo_wpk, history_match_mask, rho, depth, ...
    time_window_days, min_bin_count, plot_filled_gradient, grid_mapping, sample_gradient_max_profiles, rho0_mode, z_mode, boa_clim, ...
    z_rho_min_m, z_rho_max_m, min_drho_dz, max_rho_bracket_dz_m, match_mode, max_matches_per_group, deg_m, sensitivity_workers, sensitivity_config_names)

    if isempty(crossing_lats)
        sensitivity_lat = target_lat;
    else
        sensitivity_lat = crossing_lats(1);
    end
    band_label = crossing_label(sensitivity_lat, intersect_radius_r);
    configs = select_sensitivity_configs(sensitivity_configs(), sensitivity_config_names);
    polarities = {'cyclonic','anticyclonic'};
    summary_rows = {};
    grid_files = {};
    summary_header = {'polarity','config','grid_n','cressman_radius_r','cressman_min_obs','smooth_passes','match_count','unique_argo_count','duplicate_match_count','valid_grid_fraction','mapped_support_median','mapped_support_p10','corr_rebuild_wpk','q95_abs_rebuild_1e6_m_s','q95_abs_wpk_1e6_m_s','roughness_score','dipole_score','output_dir'};
    for p = 1:numel(polarities)
        polarity = polarities{p};
        stage_timer = tic;
        meta_file = find_meta_file(meta_dir, polarity);
        log_step(sprintf('Fast sensitivity loading META %s from %s', polarity, meta_file));
        M = load(meta_file, 'final_lon', 'final_lat', 'final_time', 'final_track', 'final_radius');
        meta_lon = double(M.final_lon);
        meta_lon(meta_lon < 0) = meta_lon(meta_lon < 0) + 360;
        meta_lat = double(M.final_lat);
        meta_time = double(M.final_time);
        meta_track = double(M.final_track);
        meta_radius = double(M.final_radius);
        meta_cx = track_cx(meta_lon, meta_lat, meta_time, meta_track, deg_m);
        cross_distance_m = abs(meta_lat - sensitivity_lat) * deg_m;
        meta_band = find(meta_lon >= bbox(1) & meta_lon <= bbox(2) & isfinite(meta_radius) & meta_radius > 0 & ...
            cross_distance_m <= meta_radius * intersect_radius_r);
        if isempty(meta_band)
            argo_band = [];
        else
            argo_lat_window_m = (intersect_radius_r + 4) * max(meta_radius(meta_band));
            argo_band = find(argo_base_mask & abs(argo_lat - sensitivity_lat) * deg_m <= argo_lat_window_m);
        end
        log_step(sprintf('%s %s candidates: %d Argo profiles, %d META snapshots', polarity, band_label, numel(argo_band), numel(meta_band)));
        match_timer = tic;
        matches = build_group_matches_only(argo_band, meta_band, polarity, band_label, ...
            argo_lon, argo_lat, argo_time, argo_park, argo_pf, argo_u, argo_v, argo_wpk, history_match_mask, rho, depth, ...
            meta_lon, meta_lat, meta_time, meta_track, meta_radius, meta_cx, ...
            time_window_days, rho0_mode, z_mode, boa_clim, z_rho_min_m, z_rho_max_m, min_drho_dz, max_rho_bracket_dz_m, match_mode, max_matches_per_group, deg_m);
        log_step(sprintf('%s %s cache built: %d QC matches, %d unique Argo, %.1f s', polarity, band_label, size(matches, 1), count_unique_argo(matches), toc(match_timer)));
        cache_dir = fullfile(output_root, polarity, band_label, '_cache');
        if exist(cache_dir, 'dir') ~= 7
            mkdir(cache_dir);
        end
        header = {'polarity','lat_band','argo_index','platform','argo_time','argo_lon','argo_lat','parking_depth_m','u_argo_m_s','v_argo_m_s','wpk_observed_m_s','rho0','z_rho_m','z_rho_bg_m','z_rho_anom_m','rho_crossing_count','rho_bracket_dz_m','local_drho_dz','eddy_track','eddy_time','eddy_lon','eddy_lat','eddy_radius_m','x_over_R','y_over_R','r_over_R','ring','cx_raw_m_s','history_velocity_matched','boa_rho_crossing_count','boa_rho_bracket_dz_m','boa_local_drho_dz','boa_bg_valid'};
        write_matched_table_mat(fullfile(cache_dir, 'matched_core_argo_cache.mat'), header, matches);
        global WRITE_MATCHED_CSV;
        if WRITE_MATCHED_CSV
            writecell(clean_write_cells([header; matches]), fullfile(cache_dir, 'matched_core_argo_cache.csv'));
        end
        grids = cell(numel(configs), 1);
        global USE_GPU_CRESSMAN;
        use_parallel = ~USE_GPU_CRESSMAN && maybe_start_parallel_pool(sensitivity_workers);
        map_timer = tic;
        if use_parallel
            parfor c = 1:numel(configs)
                cfg = configs(c);
                grids{c} = composite_grid(matches, cfg.grid_n, min_bin_count, plot_filled_gradient, grid_mapping, cfg.smooth_passes, cfg.cressman_radius_r, cfg.cressman_min_obs, sample_gradient_max_profiles);
            end
        else
            for c = 1:numel(configs)
                cfg = configs(c);
                grids{c} = composite_grid(matches, cfg.grid_n, min_bin_count, plot_filled_gradient, grid_mapping, cfg.smooth_passes, cfg.cressman_radius_r, cfg.cressman_min_obs, sample_gradient_max_profiles);
            end
        end
        log_step(sprintf('%s %s remapped %d sensitivity configs in %.1f s', polarity, band_label, numel(configs), toc(map_timer)));
        for c = 1:numel(configs)
            cfg = configs(c);
            grid = grids{c};
            grid.match_mode = match_mode;
            grid.rho0_mode = rho0_mode;
            grid.z_mode = z_mode;
            grid.z_rho_min_m = z_rho_min_m;
            grid.z_rho_max_m = z_rho_max_m;
            grid.min_drho_dz = min_drho_dz;
            grid.max_rho_bracket_dz_m = max_rho_bracket_dz_m;
            cfg_dir = fullfile(output_root, polarity, band_label, cfg.name);
            if exist(cfg_dir, 'dir') ~= 7
                mkdir(cfg_dir);
            end
            grid_file = write_group_outputs(cfg_dir, matches, grid, polarity, [band_label ' ' cfg.name], false);
            grid_files{end+1} = grid_file; %#ok<AGROW>
            summary_rows(end+1,:) = sensitivity_summary_row(matches, grid, polarity, cfg, cfg_dir); %#ok<AGROW>
        end
        plot_sensitivity_montage(fullfile(output_root, [polarity '_sensitivity_montage.png']), grids, configs, polarity, band_label);
        log_step(sprintf('%s %s fast sensitivity finished in %.1f s', polarity, band_label, toc(stage_timer)));
    end
    write_summary_table_mat(fullfile(output_root, 'SENSITIVITY_SUMMARY.mat'), summary_header, summary_rows);
    global WRITE_SUMMARY_CSV;
    if WRITE_SUMMARY_CSV
        writecell([summary_header; summary_rows], fullfile(output_root, 'SENSITIVITY_SUMMARY.csv'));
    end
    write_best_sensitivity_doc(fullfile(output_root, 'BEST_PARAMETER_RECOMMENDATION_ZH.md'), summary_rows);
end

function grid_files = run_reversal_factor_diagnosis(output_root, meta_dir, bbox, target_lat, intersect_radius_r, ...
    argo_base_mask, argo_lon, argo_lat, argo_time, argo_park, argo_pf, argo_u, argo_v, argo_wpk, history_match_mask, rho, depth, ...
    time_window_days, depth_levels, deg_m, cache_root, boa_clim, max_matches_per_group)

    grid_files = {};
    polarities = {'cyclonic','anticyclonic'};
    band_label = crossing_label(target_lat, intersect_radius_r);
    summary_header = {'polarity','variant','match_count','unique_argo_count','median_corr_w_vs_1000m','deep_reversal_score','first_zero_crossing_depth_m','q95_abs_w_1e6_m_s','output_dir'};
    summary_rows = {};
    grid_n_diag = 61;
    min_bin_count_diag = 1;
    grid_mapping_diag = 'cressman';
    smooth_passes_diag = 4;
    cressman_radius_r_diag = 1.0;
    cressman_min_obs_diag = 8;
    match_mode_diag = 'all';
    vertical_mode_diag = 'thermal_wind_depth_stack';
    section_axis_diag = 'x';
    section_half_width_r_diag = 0.25;
    max_matches_diag = max_matches_per_group;

    for p = 1:numel(polarities)
        polarity = polarities{p};
        meta_file = find_meta_file(meta_dir, polarity);
        log_step(sprintf('Reversal diagnosis loading META %s from %s', polarity, meta_file));
        M = load(meta_file, 'final_lon', 'final_lat', 'final_time', 'final_track', 'final_radius');
        meta_lon = double(M.final_lon);
        meta_lon(meta_lon < 0) = meta_lon(meta_lon < 0) + 360;
        meta_lat = double(M.final_lat);
        meta_time = double(M.final_time);
        meta_track = double(M.final_track);
        meta_radius = double(M.final_radius);
        meta_cx = track_cx(meta_lon, meta_lat, meta_time, meta_track, deg_m);
        cross_distance_m = abs(meta_lat - target_lat) * deg_m;
        meta_band = find(meta_lon >= bbox(1) & meta_lon <= bbox(2) & isfinite(meta_radius) & meta_radius > 0 & ...
            cross_distance_m <= meta_radius * intersect_radius_r);
        if isempty(meta_band)
            argo_band = [];
        else
            argo_lat_window_m = (intersect_radius_r + 4) * max(meta_radius(meta_band));
            argo_band = find(argo_base_mask & abs(argo_lat - target_lat) * deg_m <= argo_lat_window_m);
        end
        log_step(sprintf('%s %s reversal diagnosis candidates: %d Argo profiles, %d META snapshots', polarity, band_label, numel(argo_band), numel(meta_band)));
        [matches, grid3d] = build_group_3d(argo_band, meta_band, polarity, band_label, ...
            argo_lon, argo_lat, argo_time, argo_park, argo_pf, argo_u, argo_v, argo_wpk, history_match_mask, rho, depth, ...
            meta_lon, meta_lat, meta_time, meta_track, meta_radius, meta_cx, ...
            time_window_days, grid_n_diag, min_bin_count_diag, grid_mapping_diag, smooth_passes_diag, cressman_radius_r_diag, cressman_min_obs_diag, ...
            'anomaly_boa_climatology', boa_clim, depth_levels, 1e-5, 150, match_mode_diag, max_matches_diag, deg_m, section_axis_diag, section_half_width_r_diag, vertical_mode_diag, cache_root);
        diag = reversal_factor_terms(grid3d);
        group_dir = fullfile(output_root, polarity, band_label);
        if exist(group_dir, 'dir') ~= 7
            mkdir(group_dir);
        end
        save(fullfile(group_dir, 'reversal_factor_terms.mat'), 'diag', 'grid3d', 'matches', '-v7.3');
        plot_reversal_factor_4panel(fullfile(group_dir, 'reversal_factor_4panel.png'), diag, polarity, band_label);
        write_reversal_factor_doc(fullfile(group_dir, 'REVERSION_FACTOR_DIAGNOSIS_ZH.md'), diag, polarity, band_label);
        grid_files{end+1} = fullfile(group_dir, 'reversal_factor_terms.mat'); %#ok<AGROW>
        match_count = size(matches, 1);
        unique_argo_count = count_unique_argo(matches);
        for vv = 1:numel(diag.variant_names)
            stats = diag.variant_stats(vv);
            summary_rows(end+1,:) = {polarity, diag.variant_names{vv}, match_count, unique_argo_count, ...
                stats.median_corr_w_vs_1000m, stats.deep_reversal_score, stats.first_zero_crossing_depth_m, stats.q95_abs_w_1e6_m_s, group_dir}; %#ok<AGROW>
        end
    end
    write_summary_table_mat(fullfile(output_root, 'REVERSAL_FACTOR_SUMMARY.mat'), summary_header, summary_rows);
    write_reversal_factor_summary_doc(fullfile(output_root, 'REVERSAL_FACTOR_SUMMARY_ZH.md'), summary_rows);
end

function diag = reversal_factor_terms(grid3d)
    rho_abs = grid3d.rho_abs;
    if ~isfield(grid3d, 'rho_abs') || ~any(isfinite(rho_abs(:)))
        rho_abs = grid3d.rho_anom;
    end
    support3 = isfinite(grid3d.z_anom) & isfinite(grid3d.rho_anom);
    radius_m = double(grid3d.mean_radius_m);
    x_vec = grid3d.x(1,:);
    y_vec = grid3d.y(:,1);
    dx_m = median(diff(x_vec), 'omitnan') * radius_m;
    dy_m = median(diff(y_vec), 'omitnan') * radius_m;
    depth_levels = grid3d.depth_levels(:);
    [dzdx_current, dzdy_current] = gradient_stack_depth_positive(grid3d.z_anom, dx_m, dy_m);
    [dzdx_comp_down, dzdy_comp_down] = predecessor_isopycnal_slope(rho_abs, depth_levels, dx_m, dy_m, false);
    [dzdx_comp_up, dzdy_comp_up] = predecessor_isopycnal_slope(rho_abs, depth_levels, dx_m, dy_m, true);
    [u_tw_comp, v_tw_comp] = thermal_wind_velocity_stack(rho_abs, support3, grid3d.u_tw(:,:,nearest_depth_index(depth_levels, 1000)), ...
        grid3d.v_tw(:,:,nearest_depth_index(depth_levels, 1000)), isfinite(grid3d.u_tw(:,:,nearest_depth_index(depth_levels, 1000))) & isfinite(grid3d.v_tw(:,:,nearest_depth_index(depth_levels, 1000))), ...
        depth_levels, dx_m, dy_m, grid3d.thermal_wind_f_s_1);

    variants = struct('name', {}, 'description', {}, 'term1', {}, 'term2', {}, 'w', {}, 'section_w', {}, 'stats', {});
    variants(1) = make_reversal_variant_from_factors('A_current', '当前：BOA z''_rho anomaly 梯度 + rho_anom 热成风 + 当前相对速度 term2', ...
        dzdx_current, dzdy_current, -dzdx_current, -dzdy_current, grid3d.u_tw, grid3d.v_tw, 'current', support3, grid3d);
    variants(2) = make_reversal_variant_from_factors('B_comp_isoslope', '只换前辈式：composite rho 反插得到等密面斜率；速度和 term2 仍用当前口径', ...
        dzdx_comp_down, dzdy_comp_down, dzdx_comp_up, dzdy_comp_up, grid3d.u_tw, grid3d.v_tw, 'current', support3, grid3d);
    variants(3) = make_reversal_variant_from_factors('C_comp_isoslope_comp_tw', '前辈式等密面斜率 + 用 composite density 梯度积分热成风；term2 仍用当前相对速度符号', ...
        dzdx_comp_down, dzdy_comp_down, dzdx_comp_up, dzdy_comp_up, u_tw_comp, v_tw_comp, 'current', support3, grid3d);
    variants(4) = make_reversal_variant_from_factors('D_predecessor_like', '尽量接近前辈：composite rho 等密面斜率 + composite density 热成风 + c0/绝对速度 term2', ...
        dzdx_comp_down, dzdy_comp_down, dzdx_comp_up, dzdy_comp_up, u_tw_comp, v_tw_comp, 'predecessor', support3, grid3d);
    variants(5) = make_reversal_variant_from_factors('E_current_slope_comp_tw', '当前 z''_rho 梯度 + composite density 热成风 + 当前 term2', ...
        dzdx_current, dzdy_current, -dzdx_current, -dzdy_current, u_tw_comp, v_tw_comp, 'current', support3, grid3d);
    variants(6) = make_reversal_variant_from_factors('F_current_slope_pred_term2', '当前 z''_rho 梯度 + 当前热成风 + 前辈式 c0/绝对速度 term2', ...
        dzdx_current, dzdy_current, -dzdx_current, -dzdy_current, grid3d.u_tw, grid3d.v_tw, 'predecessor', support3, grid3d);
    variants(7) = make_reversal_variant_from_factors('G_comp_isoslope_pred_term2', '前辈式等密面斜率 + 当前热成风 + 前辈式 c0/绝对速度 term2', ...
        dzdx_comp_down, dzdy_comp_down, dzdx_comp_up, dzdy_comp_up, grid3d.u_tw, grid3d.v_tw, 'predecessor', support3, grid3d);
    variants(8) = make_reversal_variant_from_factors('H_current_slope_comp_tw_pred_term2', '当前 z''_rho 梯度 + composite density 热成风 + 前辈式 c0/绝对速度 term2', ...
        dzdx_current, dzdy_current, -dzdx_current, -dzdy_current, u_tw_comp, v_tw_comp, 'predecessor', support3, grid3d);

    diag = struct();
    diag.x = grid3d.x;
    diag.y = grid3d.y;
    diag.depth_levels = depth_levels;
    diag.section_axis = grid3d.section_axis;
    diag.section_half_width_r = grid3d.section_half_width_r;
    diag.dzdx_current = dzdx_current;
    diag.dzdy_current = dzdy_current;
    diag.dzdx_comp_density_down = dzdx_comp_down;
    diag.dzdy_comp_density_down = dzdy_comp_down;
    diag.dzdx_comp_density_up = dzdx_comp_up;
    diag.dzdy_comp_density_up = dzdy_comp_up;
    diag.u_tw_current = grid3d.u_tw;
    diag.v_tw_current = grid3d.v_tw;
    diag.u_tw_comp_density = u_tw_comp;
    diag.v_tw_comp_density = v_tw_comp;
    diag.z_rho_anom = grid3d.z_anom;
    diag.rho_abs = rho_abs;
    diag.rho_anom = grid3d.rho_anom;
    diag.variant_names = {variants.name};
    diag.variant_descriptions = {variants.description};
    diag.variants = variants;
    diag.variant_stats = [variants.stats];
end

function variant = make_reversal_variant(name, description, term1, term2, support3, grid3d)
    term1 = mask_stack(term1, support3);
    term2 = mask_stack(term2, support3);
    w = mask_stack(term1 + term2, support3);
    section_w = section_stack(w, grid3d.y(:,1), grid3d.section_half_width_r);
    stats = reversal_section_stats(section_w, grid3d.depth_levels(:));
    stats.q95_abs_w_1e6_m_s = q95_abs(w(:) * 1e6);
    variant = struct('name', name, 'description', description, 'term1', term1, 'term2', term2, 'w', w, 'section_w', section_w, 'stats', stats);
end

function variant = make_reversal_variant_from_factors(name, description, dzdx_down, dzdy_down, dzdx_up, dzdy_up, u_field, v_field, term2_mode, support3, grid3d)
    if strcmp(term2_mode, 'predecessor')
        c0 = abs(grid3d.mean_cx_raw);
        term1 = c0 .* dzdx_up;
        term2 = u_field .* dzdx_up + v_field .* dzdy_up;
    else
        term1 = grid3d.cx_rel .* dzdx_down;
        term2 = -((u_field - grid3d.mean_cx_raw) .* dzdx_down + v_field .* dzdy_down);
    end
    variant = make_reversal_variant(name, description, term1, term2, support3, grid3d);
end

function [dzdx_stack, dzdy_stack] = gradient_stack_depth_positive(z_stack, dx_m, dy_m)
    [ny, nx, nz] = size(z_stack);
    dzdx_stack = nan(ny, nx, nz);
    dzdy_stack = nan(ny, nx, nz);
    for zz = 1:nz
        z_grid = fillmissing2(z_stack(:,:,zz));
        [dzdx, dzdy] = gradient(z_grid, dx_m, dy_m);
        dzdx_stack(:,:,zz) = dzdx;
        dzdy_stack(:,:,zz) = dzdy;
    end
end

function [dzdx_stack, dzdy_stack] = predecessor_isopycnal_slope(rho_stack, depth_levels, dx_m, dy_m, upward_coordinate)
    [ny, nx, nz] = size(rho_stack);
    rho_clean = rho_stack;
    for ii = 1:ny
        for jj = 1:nx
            rho_clean(ii,jj,:) = monotonic_density_profile(squeeze(rho_clean(ii,jj,:)));
        end
    end
    z_axis = depth_levels(:);
    if upward_coordinate
        z_axis = -z_axis;
    end
    dzdx_stack = nan(ny, nx, nz);
    dzdy_stack = nan(ny, nx, nz);
    for ii = 1:ny
        for jj = 1:nx
            center = squeeze(rho_clean(ii,jj,:));
            if nnz(isfinite(center)) < 3
                continue
            end
            if jj == 1
                left = squeeze(rho_clean(ii,jj,:));
                right = squeeze(rho_clean(ii,jj+1,:));
                scale_x = 1 / dx_m;
            elseif jj == nx
                left = squeeze(rho_clean(ii,jj-1,:));
                right = squeeze(rho_clean(ii,jj,:));
                scale_x = 1 / dx_m;
            else
                left = squeeze(rho_clean(ii,jj-1,:));
                right = squeeze(rho_clean(ii,jj+1,:));
                scale_x = 0.5 / dx_m;
            end
            z_left = interp_density_to_depth(left, z_axis, center);
            z_right = interp_density_to_depth(right, z_axis, center);
            dzdx_stack(ii,jj,:) = (z_right - z_left) .* scale_x;
            if ii == 1
                south = squeeze(rho_clean(ii,jj,:));
                north = squeeze(rho_clean(ii+1,jj,:));
                scale_y = 1 / dy_m;
            elseif ii == ny
                south = squeeze(rho_clean(ii-1,jj,:));
                north = squeeze(rho_clean(ii,jj,:));
                scale_y = 1 / dy_m;
            else
                south = squeeze(rho_clean(ii-1,jj,:));
                north = squeeze(rho_clean(ii+1,jj,:));
                scale_y = 0.5 / dy_m;
            end
            z_south = interp_density_to_depth(south, z_axis, center);
            z_north = interp_density_to_depth(north, z_axis, center);
            dzdy_stack(ii,jj,:) = (z_north - z_south) .* scale_y;
        end
    end
end

function profile = monotonic_density_profile(profile)
    profile = profile(:);
    for kk = 2:numel(profile)
        if isfinite(profile(kk-1)) && isfinite(profile(kk)) && profile(kk) <= profile(kk-1)
            profile(kk) = profile(kk-1) + max(abs(profile(kk-1)) * 1e-9, 1e-6);
        end
    end
end

function z = interp_density_to_depth(profile, z_axis, rho_targets)
    z = nan(size(rho_targets));
    good = isfinite(profile) & isfinite(z_axis);
    if nnz(good) < 3
        return
    end
    p = profile(good);
    z_good = z_axis(good);
    [p, ia] = unique(p, 'stable');
    z_good = z_good(ia);
    if numel(p) < 3
        return
    end
    z = interp1(p, z_good, rho_targets, 'linear', NaN);
end

function idx = nearest_depth_index(depth_levels, target_depth)
    [~, idx] = min(abs(depth_levels(:) - target_depth));
end

function A = mask_stack(A, support3)
    A(~support3) = NaN;
end

function section = section_stack(field3d, y_vec, half_width_r)
    if ~isfinite(half_width_r) || half_width_r <= 0
        half_width_r = 0.25;
    end
    y_mask = abs(y_vec) <= half_width_r;
    tmp = field3d(y_mask,:,:);
    section = squeeze(median(tmp, 1, 'omitnan'))';
end

function stats = reversal_section_stats(section_w, depth_levels)
    anchor = nearest_depth_index(depth_levels, 1000);
    base = section_w(anchor,:);
    corr_by_depth = nan(numel(depth_levels), 1);
    for kk = 1:numel(depth_levels)
        a = section_w(kk,:);
        good = isfinite(a) & isfinite(base);
        if nnz(good) >= 5
            C = corrcoef(a(good), base(good));
            corr_by_depth(kk) = C(1,2);
        end
    end
    shallow_mask = depth_levels <= 700;
    deep_mask = depth_levels >= 1200;
    shallow_pattern = median(section_w(shallow_mask,:), 1, 'omitnan');
    deep_pattern = median(section_w(deep_mask,:), 1, 'omitnan');
    good = isfinite(shallow_pattern) & isfinite(deep_pattern);
    if nnz(good) >= 5
        C = corrcoef(shallow_pattern(good), deep_pattern(good));
        deep_reversal_score = -C(1,2);
    else
        deep_reversal_score = NaN;
    end
    profile = median(section_w, 2, 'omitnan');
    first_zero = NaN;
    for kk = 2:numel(profile)
        if isfinite(profile(kk-1)) && isfinite(profile(kk)) && profile(kk-1) * profile(kk) < 0
            first_zero = 0.5 * (depth_levels(kk-1) + depth_levels(kk));
            break
        end
    end
    stats = struct('corr_w_vs_1000m', corr_by_depth, ...
        'median_corr_w_vs_1000m', median(corr_by_depth, 'omitnan'), ...
        'deep_reversal_score', deep_reversal_score, ...
        'first_zero_crossing_depth_m', first_zero, ...
        'q95_abs_w_1e6_m_s', NaN);
end

function q = q95_abs(values)
    values = abs(values(isfinite(values)));
    if isempty(values)
        q = NaN;
    else
        q = quantile(values, 0.95);
    end
end

function plot_reversal_factor_4panel(path, diag, polarity, band_label)
    panel_names = {'A_current','B_comp_isoslope','C_comp_isoslope_comp_tw','D_predecessor_like'};
    idx = zeros(1, numel(panel_names));
    vals = [];
    for i = 1:numel(panel_names)
        idx(i) = find(strcmp(diag.variant_names, panel_names{i}), 1);
        vals = [vals; diag.variants(idx(i)).section_w(:) * 1e6]; %#ok<AGROW>
    end
    lim = q95_abs(vals);
    if ~isfinite(lim) || lim <= 0
        lim = 2.5;
    end
    fig = figure('Visible','off','Color','w','Position',[100 100 1500 1000]);
    tl = tiledlayout(fig, 2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
    x = diag.x(1,:);
    depth = diag.depth_levels(:);
    titles = {'A current', 'B comp-rho isoslope', 'C comp-rho isoslope + TW', 'D predecessor-like'};
    for i = 1:numel(idx)
        ax = nexttile(tl);
        data = diag.variants(idx(i)).section_w * 1e6;
        contourf(ax, x, depth, data, 28, 'LineStyle', 'none');
        set(ax, 'YDir', 'reverse');
        colormap(ax, redblue_colormap());
        clim(ax, [-lim lim]);
        cb = colorbar(ax);
        ylabel(cb, '10^{-6} m s^{-1}');
        xlabel(ax, 'x/R');
        ylabel(ax, 'Depth (m)');
        title(ax, titles{i}, 'Interpreter', 'none');
    end
    sgtitle(tl, [polarity ' ' band_label ' reversal factor W sections'], 'Interpreter', 'none');
    exportgraphics(fig, path, 'Resolution', 180);
    close(fig);
end

function write_reversal_factor_doc(path, diag, polarity, band_label)
    fid = fopen(path, 'w');
    fprintf(fid, '# %s %s 三因素反转诊断\n\n', polarity, band_label);
    fprintf(fid, '本诊断用于判断深层 W 反转是否由 `等密面斜率`、`热成风速度` 或 `term2` 组合方式造成。\n\n');
    fprintf(fid, '| variant | 口径 | median corr W/1000m | deep reversal score | first zero depth m | q95 W |\n');
    fprintf(fid, '|---|---|---:|---:|---:|---:|\n');
    for i = 1:numel(diag.variant_names)
        s = diag.variant_stats(i);
        fprintf(fid, '| %s | %s | %.3g | %.3g | %.3g | %.3g |\n', diag.variant_names{i}, diag.variant_descriptions{i}, ...
            s.median_corr_w_vs_1000m, s.deep_reversal_score, s.first_zero_crossing_depth_m, s.q95_abs_w_1e6_m_s);
    end
    fprintf(fid, '\n判读：如果 B 开始反转，主因是 `composite rho -> isopycnal slope`；如果 C 开始反转，主因偏热成风速度；如果 D 开始反转，主因偏 term2 的符号/速度/斜率组合。\n');
    fclose(fid);
end

function write_reversal_factor_summary_doc(path, summary_rows)
    fid = fopen(path, 'w');
    fprintf(fid, '# 20N 三因素反转诊断汇总\n\n');
    fprintf(fid, '| polarity | variant | matches | unique Argo | median corr W/1000m | deep reversal score | first zero depth m | q95 W | output |\n');
    fprintf(fid, '|---|---|---:|---:|---:|---:|---:|---:|---|\n');
    for i = 1:size(summary_rows,1)
        fprintf(fid, '| %s | %s | %d | %d | %.3g | %.3g | %.3g | %.3g | `%s` |\n', ...
            summary_rows{i,1}, summary_rows{i,2}, summary_rows{i,3}, summary_rows{i,4}, summary_rows{i,5}, summary_rows{i,6}, summary_rows{i,7}, summary_rows{i,8}, summary_rows{i,9});
    end
    fclose(fid);
end

function configs = sensitivity_configs()
    configs = struct('name', {}, 'grid_n', {}, 'cressman_radius_r', {}, 'cressman_min_obs', {}, 'smooth_passes', {});
    configs(1) = struct('name', 'baseline', 'grid_n', 81, 'cressman_radius_r', 0.5, 'cressman_min_obs', 3, 'smooth_passes', 2);
    configs(2) = struct('name', 'recommended', 'grid_n', 61, 'cressman_radius_r', 1.0, 'cressman_min_obs', 8, 'smooth_passes', 4);
    configs(3) = struct('name', 'smoother', 'grid_n', 61, 'cressman_radius_r', 1.25, 'cressman_min_obs', 8, 'smooth_passes', 4);
    configs(4) = struct('name', 'strong_support', 'grid_n', 61, 'cressman_radius_r', 1.0, 'cressman_min_obs', 12, 'smooth_passes', 4);
    configs(5) = struct('name', 'low_res_smooth', 'grid_n', 51, 'cressman_radius_r', 1.25, 'cressman_min_obs', 8, 'smooth_passes', 4);
    configs(6) = struct('name', 'high_smooth', 'grid_n', 61, 'cressman_radius_r', 1.0, 'cressman_min_obs', 8, 'smooth_passes', 6);
end

function selected = select_sensitivity_configs(configs, names)
    if isempty(names)
        selected = configs;
        return
    end
    selected = struct('name', {}, 'grid_n', {}, 'cressman_radius_r', {}, 'cressman_min_obs', {}, 'smooth_passes', {});
    config_names = {configs.name};
    for i = 1:numel(names)
        name = char(names{i});
        idx = find(strcmp(config_names, name), 1);
        if isempty(idx)
            error('Unknown sensitivity config: %s', name);
        end
        selected(end+1) = configs(idx); %#ok<AGROW>
    end
end

function rows = preallocate_match_rows(n_argo)
    n_rows = max(1024, min(max(1, n_argo) * 2, 200000));
    rows = cell(n_rows, 33);
end

function rows = grow_match_rows(rows)
    rows(end + size(rows, 1), 33) = {[]};
end

function n = count_unique_argo(matches)
    if isempty(matches)
        n = 0;
    else
        n = numel(unique(cell2mat(matches(:,3))));
    end
end

function log_step(message)
    fprintf('[%s] %s\n', datestr(now, 'yyyy-mm-dd HH:MM:SS'), message);
end

function values = profile_values_at_depth_by_index(depth, rho, profile_idx, target_depth)
    target_depth = target_depth(:);
    values = nan(numel(target_depth), 1);
    for kk = 1:numel(depth)-1
        z1 = depth(kk);
        z2 = depth(kk+1);
        if z2 == z1
            continue
        end
        if kk == numel(depth)-1
            mask = target_depth >= z1 & target_depth <= z2;
        else
            mask = target_depth >= z1 & target_depth < z2;
        end
        if ~any(mask)
            continue
        end
        w = (target_depth(mask) - z1) ./ (z2 - z1);
        rows = profile_idx(mask);
        v1 = double(rho(rows, kk));
        v2 = double(rho(rows, kk+1));
        values(mask) = v1 .* (1 - w) + v2 .* w;
    end
end

function idx = time_window_indices(sorted_time, t0, window_days)
    if ~isfinite(t0) || isempty(sorted_time)
        idx = [];
        return
    end
    lo = lower_bound(sorted_time, t0 - window_days);
    hi = upper_bound(sorted_time, t0 + window_days) - 1;
    if lo > hi
        idx = [];
    else
        idx = lo:hi;
    end
end

function idx = lower_bound(values, target)
    lo = 1;
    hi = numel(values) + 1;
    while lo < hi
        mid = floor((lo + hi) / 2);
        if values(mid) < target
            lo = mid + 1;
        else
            hi = mid;
        end
    end
    idx = lo;
end

function idx = upper_bound(values, target)
    lo = 1;
    hi = numel(values) + 1;
    while lo < hi
        mid = floor((lo + hi) / 2);
        if values(mid) <= target
            lo = mid + 1;
        else
            hi = mid;
        end
    end
    idx = lo;
end

function ok = maybe_start_parallel_pool(workers)
    ok = false;
    if ~isfinite(workers) || workers <= 1
        return
    end
    try
        if license('test', 'Distrib_Computing_Toolbox')
            pool = gcp('nocreate');
            if isempty(pool)
                parpool('threads', workers);
            end
            ok = true;
        end
    catch ME
        warning('Parallel pool unavailable, using serial sensitivity loop: %s', ME.message);
        ok = false;
    end
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

function boa = load_boa_monthly_climatology(root_dir, cache_path)
    if nargin >= 2 && exist(cache_path, 'file') == 2
        C = load(cache_path, 'boa', 'source_root');
        if isfield(C, 'boa') && isfield(C, 'source_root') && strcmp(C.source_root, root_dir)
            boa = C.boa;
            fprintf('Loaded BOA monthly climatology cache: %s\n', cache_path);
            return
        end
    end
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
        tok = regexp(name, 'PDen1000_\d{4}(\d{2})\.mat', 'tokens', 'once');
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
    if nargin >= 2
        source_root = root_dir; %#ok<NASGU>
        cache_dir = fileparts(cache_path);
        if exist(cache_dir, 'dir') ~= 7
            mkdir(cache_dir);
        end
        save(cache_path, 'boa', 'source_root', '-v7.3');
        fprintf('Saved BOA monthly climatology cache: %s\n', cache_path);
    end
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

function matches = build_group_matches_only(argo_idx, meta_idx, polarity, band_label, ...
    argo_lon, argo_lat, argo_time, argo_park, argo_pf, argo_u, argo_v, argo_wpk, history_match_mask, rho, depth, ...
    meta_lon, meta_lat, meta_time, meta_track, meta_radius, meta_cx, ...
    time_window_days, rho0_mode, z_mode, boa_clim, z_rho_min_m, z_rho_max_m, min_drho_dz, max_rho_bracket_dz_m, match_mode, max_matches_per_group, deg_m)

    if strcmp(match_mode, 'all')
        rows = build_match_rows_time_blocks(argo_idx, meta_idx, polarity, band_label, ...
            argo_lon, argo_lat, argo_time, argo_park, argo_pf, argo_u, argo_v, argo_wpk, history_match_mask, rho, depth, ...
            meta_lon, meta_lat, meta_time, meta_track, meta_radius, meta_cx, ...
            time_window_days, max_matches_per_group, deg_m);
    else
        rows = build_match_rows_nearest(argo_idx, meta_idx, polarity, band_label, ...
            argo_lon, argo_lat, argo_time, argo_park, argo_pf, argo_u, argo_v, argo_wpk, history_match_mask, rho, depth, ...
            meta_lon, meta_lat, meta_time, meta_track, meta_radius, meta_cx, ...
            time_window_days, max_matches_per_group, deg_m);
    end
    matches = apply_rho0_mode(rows, rho, depth, argo_park, rho0_mode, z_mode, boa_clim, z_rho_min_m, z_rho_max_m, min_drho_dz, max_rho_bracket_dz_m);
end

function [matches, grid] = build_group(argo_idx, meta_idx, polarity, band_label, ...
    argo_lon, argo_lat, argo_time, argo_park, argo_pf, argo_u, argo_v, argo_wpk, history_match_mask, rho, depth, ...
    meta_lon, meta_lat, meta_time, meta_track, meta_radius, meta_cx, ...
    time_window_days, grid_n, min_bin_count, plot_filled_gradient, grid_mapping, smooth_passes, cressman_radius_r, cressman_min_obs, sample_gradient_max_profiles, rho0_mode, ...
    z_mode, boa_clim, z_rho_min_m, z_rho_max_m, min_drho_dz, max_rho_bracket_dz_m, match_mode, max_matches_per_group, deg_m)

    if strcmp(match_mode, 'all')
        matches = build_group_matches_only(argo_idx, meta_idx, polarity, band_label, ...
            argo_lon, argo_lat, argo_time, argo_park, argo_pf, argo_u, argo_v, argo_wpk, history_match_mask, rho, depth, ...
            meta_lon, meta_lat, meta_time, meta_track, meta_radius, meta_cx, ...
            time_window_days, rho0_mode, z_mode, boa_clim, z_rho_min_m, z_rho_max_m, min_drho_dz, max_rho_bracket_dz_m, match_mode, max_matches_per_group, deg_m);
        grid = composite_grid(matches, grid_n, min_bin_count, plot_filled_gradient, grid_mapping, smooth_passes, cressman_radius_r, cressman_min_obs, sample_gradient_max_profiles);
        grid.match_mode = match_mode;
        grid.rho0_mode = rho0_mode;
        grid.z_mode = z_mode;
        grid.z_rho_min_m = z_rho_min_m;
        grid.z_rho_max_m = z_rho_max_m;
        grid.min_drho_dz = min_drho_dz;
        grid.max_rho_bracket_dz_m = max_rho_bracket_dz_m;
        return
    end

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

function [matches, grid3d] = build_group_3d(argo_idx, meta_idx, polarity, band_label, ...
    argo_lon, argo_lat, argo_time, argo_park, argo_pf, argo_u, argo_v, argo_wpk, history_match_mask, rho, depth, ...
    meta_lon, meta_lat, meta_time, meta_track, meta_radius, meta_cx, ...
    time_window_days, grid_n, min_bin_count, grid_mapping, smooth_passes, cressman_radius_r, cressman_min_obs, ...
    z_mode, boa_clim, depth_levels, min_drho_dz, max_rho_bracket_dz_m, match_mode, max_matches_per_group, deg_m, section_axis, section_half_width_r, vertical_mode, cache_root)

    match_timer = tic;
    if strcmp(match_mode, 'all')
        rows = build_match_rows_time_blocks(argo_idx, meta_idx, polarity, band_label, ...
            argo_lon, argo_lat, argo_time, argo_park, argo_pf, argo_u, argo_v, argo_wpk, history_match_mask, rho, depth, ...
            meta_lon, meta_lat, meta_time, meta_track, meta_radius, meta_cx, ...
            time_window_days, max_matches_per_group, deg_m);
    else
        rows = build_match_rows_nearest(argo_idx, meta_idx, polarity, band_label, ...
            argo_lon, argo_lat, argo_time, argo_park, argo_pf, argo_u, argo_v, argo_wpk, history_match_mask, rho, depth, ...
            meta_lon, meta_lat, meta_time, meta_track, meta_radius, meta_cx, ...
            time_window_days, max_matches_per_group, deg_m);
    end
    log_step(sprintf('%s %s 3D matching produced %d rows in %.1f s', polarity, band_label, size(rows, 1), toc(match_timer)));
    matches = rows;
    composite_timer = tic;
    grid3d = composite_grid_3d(matches, rho, depth, boa_clim, depth_levels, grid_n, min_bin_count, grid_mapping, smooth_passes, cressman_radius_r, cressman_min_obs, min_drho_dz, max_rho_bracket_dz_m, section_axis, section_half_width_r, vertical_mode, cache_root, polarity, band_label, match_mode);
    log_step(sprintf('%s %s 3D composite finished in %.1f s', polarity, band_label, toc(composite_timer)));
    grid3d.match_mode = match_mode;
    grid3d.z_mode = z_mode;
    grid3d.min_drho_dz = min_drho_dz;
    grid3d.max_rho_bracket_dz_m = max_rho_bracket_dz_m;
end

function rows = build_match_rows_time_blocks(argo_idx, meta_idx, polarity, band_label, ...
    argo_lon, argo_lat, argo_time, argo_park, argo_pf, argo_u, argo_v, argo_wpk, history_match_mask, rho, depth, ...
    meta_lon, meta_lat, meta_time, meta_track, meta_radius, meta_cx, ...
    time_window_days, max_matches_per_group, deg_m)

    rows = preallocate_match_rows(numel(argo_idx));
    row_count = 0;
    if isempty(argo_idx) || isempty(meta_idx)
        rows = rows(1:0,:);
        return
    end

    block_days = 3;
    sub_block_size = 1024;
    [argo_time_sorted, argo_order] = sort(argo_time(argo_idx));
    argo_idx_sorted = argo_idx(argo_order);
    [meta_time_sorted, meta_order] = sort(meta_time(meta_idx));
    meta_idx_sorted = meta_idx(meta_order);
    rho0_sorted = profile_values_at_depth_by_index(depth, rho, argo_idx_sorted, argo_park(argo_idx_sorted));
    t_min = floor(min(argo_time_sorted, [], 'omitnan'));
    t_max = ceil(max(argo_time_sorted, [], 'omitnan'));
    if ~isfinite(t_min) || ~isfinite(t_max)
        rows = rows(1:0,:);
        return
    end

    block_starts = t_min:block_days:t_max;
    for bb = 1:numel(block_starts)
        t0 = block_starts(bb);
        t1 = min(t0 + block_days, t_max + 1);
        a_lo = lower_bound(argo_time_sorted, t0);
        a_hi = lower_bound(argo_time_sorted, t1) - 1;
        m_lo = lower_bound(meta_time_sorted, t0 - time_window_days);
        m_hi = lower_bound(meta_time_sorted, t1 + time_window_days) - 1;
        if a_lo > a_hi || m_lo > m_hi
            continue
        end
        a_block = argo_idx_sorted(a_lo:a_hi);
        rho0_block = rho0_sorted(a_lo:a_hi);
        m_block = meta_idx_sorted(m_lo:m_hi);
        if isempty(a_block) || isempty(m_block)
            continue
        end
        for a0 = 1:sub_block_size:numel(a_block)
            a1 = min(a0 + sub_block_size - 1, numel(a_block));
            a_sub = a_block(a0:a1);
            rho0_sub = rho0_block(a0:a1);

            dt_ok = abs(argo_time(a_sub(:)) - meta_time(m_block(:))') <= time_window_days;
            dlon = argo_lon(a_sub(:)) - meta_lon(m_block(:))';
            dlon(dlon > 180) = dlon(dlon > 180) - 360;
            dlon(dlon < -180) = dlon(dlon < -180) + 360;
            dx = dlon .* deg_m .* cosd(argo_lat(a_sub(:)));
            dy = (argo_lat(a_sub(:)) - meta_lat(m_block(:))') .* deg_m;
            r_norm = hypot(dx, dy) ./ meta_radius(m_block(:))';
            mask = dt_ok & isfinite(r_norm) & r_norm <= 4 & isfinite(rho0_sub);
            if ~any(mask(:))
                continue
            end
            [ai, mi] = find(mask);
            lin = sub2ind(size(mask), ai, mi);
            n_add = numel(ai);
            while row_count + n_add > size(rows, 1)
                rows = grow_match_rows(rows);
            end
            for kk = 1:n_add
                ii = a_sub(ai(kk));
                jj = m_block(mi(kk));
                rr = row_count + kk;
                this_r = r_norm(lin(kk));
                rows(rr,:) = {polarity, band_label, ii, argo_pf(ii), argo_time(ii), argo_lon(ii), argo_lat(ii), ...
                    argo_park(ii), argo_u(ii), argo_v(ii), argo_wpk(ii), rho0_sub(ai(kk)), NaN, NaN, NaN, NaN, NaN, NaN, ...
                    meta_track(jj), meta_time(jj), meta_lon(jj), meta_lat(jj), meta_radius(jj), ...
                    dx(lin(kk)) / meta_radius(jj), dy(lin(kk)) / meta_radius(jj), this_r, ring_label(this_r), meta_cx(jj), history_match_mask(ii), ...
                    NaN, NaN, NaN, false};
            end
            row_count = row_count + n_add;
            if max_matches_per_group > 0 && row_count >= max_matches_per_group
                rows = rows(1:max_matches_per_group,:);
                return
            end
        end
        if max_matches_per_group == 0 && (mod(bb, 500) == 0 || bb == numel(block_starts))
            log_step(sprintf('%s %s match blocks: %d/%d, raw rows so far: %d', polarity, band_label, bb, numel(block_starts), row_count));
        end
    end
    rows = rows(1:row_count,:);
end

function rows = build_match_rows_nearest(argo_idx, meta_idx, polarity, band_label, ...
    argo_lon, argo_lat, argo_time, argo_park, argo_pf, argo_u, argo_v, argo_wpk, history_match_mask, rho, depth, ...
    meta_lon, meta_lat, meta_time, meta_track, meta_radius, meta_cx, ...
    time_window_days, max_matches_per_group, deg_m)

    rows = preallocate_match_rows(numel(argo_idx));
    row_count = 0;
    meta_time_band = meta_time(meta_idx);
    [meta_time_sorted, meta_order] = sort(meta_time_band);
    meta_idx_sorted = meta_idx(meta_order);
    for a = 1:numel(argo_idx)
        ii = argo_idx(a);
        candidate_sorted = time_window_indices(meta_time_sorted, argo_time(ii), time_window_days);
        if isempty(candidate_sorted)
            continue
        end
        candidates = meta_idx_sorted(candidate_sorted);
        dx = local_dx_m(argo_lon(ii), meta_lon(candidates), argo_lat(ii), deg_m);
        dy = (argo_lat(ii) - meta_lat(candidates)) * deg_m;
        r_norm = hypot(dx, dy) ./ meta_radius(candidates);
        rho0 = interp1(depth, double(rho(ii,:)), argo_park(ii), 'linear', NaN);
        if ~isfinite(rho0)
            continue
        end
        [best_r, best_pos] = min(r_norm);
        if ~isfinite(best_r) || best_r > 4
            continue
        end
        jj = candidates(best_pos);
        row_count = row_count + 1;
        if row_count > size(rows, 1)
            rows = grow_match_rows(rows);
        end
        rows(row_count,:) = {polarity, band_label, ii, argo_pf(ii), argo_time(ii), argo_lon(ii), argo_lat(ii), ...
            argo_park(ii), argo_u(ii), argo_v(ii), argo_wpk(ii), rho0, NaN, NaN, NaN, NaN, NaN, NaN, ...
            meta_track(jj), meta_time(jj), meta_lon(jj), meta_lat(jj), meta_radius(jj), ...
            dx(best_pos) / meta_radius(jj), dy(best_pos) / meta_radius(jj), best_r, ring_label(best_r), meta_cx(jj), history_match_mask(ii), ...
            NaN, NaN, NaN, false};
        if max_matches_per_group > 0 && row_count >= max_matches_per_group
            break
        end
    end
    rows = rows(1:row_count,:);
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
    argo_ids = cell2mat(rows(:,3));
    [unique_argo_ids, first_row, row_to_unique] = unique(argo_ids);
    log_step(sprintf('profile rho QC start: %d raw rows, %d unique Argo', size(rows, 1), numel(unique_argo_ids)));
    profile_timer = tic;
    unique_z_rho = nan(numel(unique_argo_ids), 1);
    unique_crossing_count = nan(numel(unique_argo_ids), 1);
    unique_bracket_dz = nan(numel(unique_argo_ids), 1);
    unique_drho_dz = nan(numel(unique_argo_ids), 1);
    unique_ok = false(numel(unique_argo_ids), 1);
    unique_rho0 = nan(numel(unique_argo_ids), 1);
    global QC_WORKERS;
    use_qc_parallel = maybe_start_parallel_pool(QC_WORKERS);
    if use_qc_parallel
        parfor uu = 1:numel(unique_argo_ids)
            rr0 = first_row(uu);
            ii = unique_argo_ids(uu);
            this_rho0 = target_rho0(rr0);
            [z_rho, crossing_count, bracket_dz, local_drho_dz] = isopycnal_depth_qc(depth, double(rho(ii,:)), this_rho0, argo_park(ii));
            unique_rho0(uu) = this_rho0;
            unique_z_rho(uu) = z_rho;
            unique_crossing_count(uu) = crossing_count;
            unique_bracket_dz(uu) = bracket_dz;
            unique_drho_dz(uu) = local_drho_dz;
            unique_ok(uu) = isfinite(z_rho) && z_rho >= z_rho_min_m && z_rho <= z_rho_max_m && ...
                isfinite(bracket_dz) && bracket_dz <= max_rho_bracket_dz_m && ...
                isfinite(local_drho_dz) && abs(local_drho_dz) >= min_drho_dz;
        end
    else
        for uu = 1:numel(unique_argo_ids)
            rr0 = first_row(uu);
            ii = unique_argo_ids(uu);
            unique_rho0(uu) = target_rho0(rr0);
            [z_rho, crossing_count, bracket_dz, local_drho_dz] = isopycnal_depth_qc(depth, double(rho(ii,:)), unique_rho0(uu), argo_park(ii));
            unique_z_rho(uu) = z_rho;
            unique_crossing_count(uu) = crossing_count;
            unique_bracket_dz(uu) = bracket_dz;
            unique_drho_dz(uu) = local_drho_dz;
            unique_ok(uu) = isfinite(z_rho) && z_rho >= z_rho_min_m && z_rho <= z_rho_max_m && ...
                isfinite(bracket_dz) && bracket_dz <= max_rho_bracket_dz_m && ...
                isfinite(local_drho_dz) && abs(local_drho_dz) >= min_drho_dz;
        end
    end
    log_step(sprintf('profile rho QC done: %d unique kept, %.1f s', nnz(unique_ok), toc(profile_timer)));
    for rr = 1:size(rows, 1)
        uu = row_to_unique(rr);
        if unique_ok(uu)
            rows{rr,12} = unique_rho0(uu);
            rows{rr,13} = unique_z_rho(uu);
            rows{rr,16} = unique_crossing_count(uu);
            rows{rr,17} = unique_bracket_dz(uu);
            rows{rr,18} = unique_drho_dz(uu);
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
        argo_ids = cell2mat(rows(:,3));
        [unique_argo_ids, first_row, row_to_unique] = unique(argo_ids);
        unique_z_bg = nan(numel(unique_argo_ids), 1);
        unique_bg_crossing_count = nan(numel(unique_argo_ids), 1);
        unique_bg_bracket_dz = nan(numel(unique_argo_ids), 1);
        unique_bg_drho_dz = nan(numel(unique_argo_ids), 1);
        unique_bg_ok = false(numel(unique_argo_ids), 1);
        log_step(sprintf('BOA rho QC start: %d unique Argo', numel(unique_argo_ids)));
        boa_timer = tic;
        if use_qc_parallel
            parfor uu = 1:numel(unique_argo_ids)
                rr0 = first_row(uu);
                [~, month_id, ~] = datevec(rows{rr0,5});
                boa_profile = boa_density_profile_at(boa_clim, rows{rr0,6}, rows{rr0,7}, month_id);
                boa_profile = align_density_units(boa_profile, rows{rr0,12});
                [z_bg, bg_crossing_count, bg_bracket_dz, bg_drho_dz] = isopycnal_depth_qc(boa_clim.pres, boa_profile, rows{rr0,12}, rows{rr0,8});
                unique_z_bg(uu) = z_bg;
                unique_bg_crossing_count(uu) = bg_crossing_count;
                unique_bg_bracket_dz(uu) = bg_bracket_dz;
                unique_bg_drho_dz(uu) = bg_drho_dz;
                unique_bg_ok(uu) = isfinite(z_bg) && z_bg >= z_rho_min_m && z_bg <= z_rho_max_m && ...
                    isfinite(bg_bracket_dz) && bg_bracket_dz <= max_rho_bracket_dz_m && ...
                    isfinite(bg_drho_dz) && abs(bg_drho_dz) >= min_drho_dz;
            end
        else
            for uu = 1:numel(unique_argo_ids)
                rr0 = first_row(uu);
                [~, month_id, ~] = datevec(rows{rr0,5});
                boa_profile = boa_density_profile_at(boa_clim, rows{rr0,6}, rows{rr0,7}, month_id);
                boa_profile = align_density_units(boa_profile, rows{rr0,12});
                [z_bg, bg_crossing_count, bg_bracket_dz, bg_drho_dz] = isopycnal_depth_qc(boa_clim.pres, boa_profile, rows{rr0,12}, rows{rr0,8});
                unique_z_bg(uu) = z_bg;
                unique_bg_crossing_count(uu) = bg_crossing_count;
                unique_bg_bracket_dz(uu) = bg_bracket_dz;
                unique_bg_drho_dz(uu) = bg_drho_dz;
                unique_bg_ok(uu) = isfinite(z_bg) && z_bg >= z_rho_min_m && z_bg <= z_rho_max_m && ...
                    isfinite(bg_bracket_dz) && bg_bracket_dz <= max_rho_bracket_dz_m && ...
                    isfinite(bg_drho_dz) && abs(bg_drho_dz) >= min_drho_dz;
            end
        end
        log_step(sprintf('BOA rho QC done: %d unique kept, %.1f s', nnz(unique_bg_ok), toc(boa_timer)));
        for rr = 1:size(rows, 1)
            uu = row_to_unique(rr);
            rows{rr,30} = unique_bg_crossing_count(uu);
            rows{rr,31} = unique_bg_bracket_dz(uu);
            rows{rr,32} = unique_bg_drho_dz(uu);
            rows{rr,33} = unique_bg_ok(uu);
            bg_ok = unique_bg_ok(uu);
            if bg_ok
                z_bg_all(rr) = unique_z_bg(uu);
                z_anom(rr) = z_rho(rr) - unique_z_bg(uu);
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
                coef = Xfit \ z_rho(far_ok);
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
        if r1 == 0 && depth(k) > 0
            dz = abs(depth(k+1) - depth(k));
            drhodz = (profile(k+1) - profile(k)) / (depth(k+1) - depth(k));
            hits(end+1,:) = [depth(k), dz, drhodz, k]; %#ok<AGROW>
            continue
        elseif r2 == 0 && depth(k+1) > 0
            dz = abs(depth(k+1) - depth(k));
            drhodz = (profile(k+1) - profile(k)) / (depth(k+1) - depth(k));
            hits(end+1,:) = [depth(k+1), dz, drhodz, k]; %#ok<AGROW>
            continue
        end
        if r1 * r2 > 0 || profile(k) == profile(k+1)
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
        coef = Xfit \ z(inside);
        dzdx = coef(2) / radius(ii);
        dzdy = coef(3) / radius(ii);
        term1(ii) = cx_rel * dzdx;
        term2(ii) = -((u(ii) - cx_raw(ii)) * dzdx + v(ii) * dzdy);
        rebuild(ii) = term1(ii) + term2(ii);
    end
end

function grid3d = composite_grid_3d(matches, rho, depth, boa_clim, depth_levels, grid_n, min_bin_count, grid_mapping, smooth_passes, cressman_radius_r, cressman_min_obs, min_drho_dz, max_rho_bracket_dz_m, section_axis, section_half_width_r, vertical_mode, cache_root, polarity, band_label, match_mode)
    x_vec = linspace(-4, 4, grid_n);
    y_vec = linspace(-4, 4, grid_n);
    [X, Y] = meshgrid(x_vec, y_vec);
    nz = numel(depth_levels);
    nan3 = nan(grid_n, grid_n, nz);
    count3 = zeros(grid_n, grid_n, nz);
    grid3d = struct('x', X, 'y', Y, 'depth_levels', depth_levels(:), 'w', nan3, 'term1', nan3, 'term2', nan3, ...
        'z_anom', nan3, 'rho_anom', nan3, 'rho_abs', nan3, 'u_tw', nan3, 'v_tw', nan3, 'count', count3, 'mapped_support', count3, 'valid_profile_count', zeros(nz,1), ...
        'boa_bg_valid_count', zeros(nz,1), 'mean_cx_raw', NaN, 'mean_u_bg', NaN, 'cx_rel', NaN, ...
        'mean_radius_m', NaN, 'section_axis', section_axis, 'section_half_width_r', section_half_width_r, ...
        'section_coord', [], 'section_w', [], 'match_count', 0, 'unique_argo_count', 0, 'duplicate_match_count', 0, ...
        'match_mode', '', 'z_mode', '', 'vertical_mode', vertical_mode, 'thermal_wind_anchor_depth_m', 1000, ...
        'thermal_wind_f_s_1', NaN, 'min_drho_dz', NaN, 'max_rho_bracket_dz_m', NaN);
    if isempty(matches) || isempty(fieldnames(boa_clim))
        return
    end
    x = cell2mat(matches(:,24)); x = x(:);
    y = cell2mat(matches(:,25)); y = y(:);
    u = cell2mat(matches(:,9)); u = u(:);
    v = cell2mat(matches(:,10)); v = v(:);
    cx_raw = cell2mat(matches(:,28)); cx_raw = cx_raw(:);
    radius = cell2mat(matches(:,23)); radius = radius(:);
    argo_indices = cell2mat(matches(:,3)); argo_indices = argo_indices(:);
    argo_lon_match = cell2mat(matches(:,6)); argo_lon_match = argo_lon_match(:);
    argo_lat_match = cell2mat(matches(:,7)); argo_lat_match = argo_lat_match(:);
    argo_time_match = cell2mat(matches(:,5)); argo_time_match = argo_time_match(:);
    grid3d.mean_cx_raw = mean(cx_raw, 'omitnan');
    grid3d.mean_u_bg = mean(u, 'omitnan');
    grid3d.cx_rel = grid3d.mean_cx_raw - grid3d.mean_u_bg;
    grid3d.mean_radius_m = mean(radius, 'omitnan');
    grid3d.thermal_wind_f_s_1 = 2 * 7.2921159e-5 * sind(mean(argo_lat_match, 'omitnan'));
    edges = linspace(-4, 4, grid_n + 1);
    dx_m = mean(diff(x_vec)) * grid3d.mean_radius_m;
    dy_m = mean(diff(y_vec)) * grid3d.mean_radius_m;
    profile_timer = tic;
    profile_cache = profile_depth_stack_cache(matches, rho, depth, boa_clim, depth_levels, min_drho_dz, max_rho_bracket_dz_m, cache_root, polarity, band_label, match_mode);
    log_step(sprintf('%s %s profile depth cache ready in %.1f s', polarity, band_label, toc(profile_timer)));
    [~, unique_pos] = ismember(argo_indices, profile_cache.argo_indices);
    finite_unique = unique_pos > 0;
    base_timer = tic;
    [base_uv, base_support] = cressman_map_multi(x(finite_unique), y(finite_unique), [u(finite_unique), v(finite_unique)], X, Y, cressman_radius_r, cressman_min_obs);
    log_step(sprintf('%s %s base velocity Cressman ready in %.1f s', polarity, band_label, toc(base_timer)));
    u_base = base_uv(:,:,1);
    v_base = base_uv(:,:,2);
    base_support = base_support >= cressman_min_obs;
    rho_grids = nan3;
    support3 = false(grid_n, grid_n, nz);
    z_samples = nan(numel(x), nz);
    rho_samples = nan(numel(x), nz);
    rho_abs_samples = nan(numel(x), nz);
    z_samples(finite_unique,:) = profile_cache.z_anom(unique_pos(finite_unique), :);
    rho_samples(finite_unique,:) = profile_cache.rho_anom(unique_pos(finite_unique), :);
    if isfield(profile_cache, 'rho_abs')
        rho_abs_samples(finite_unique,:) = profile_cache.rho_abs(unique_pos(finite_unique), :);
    end
    base_good = isfinite(x) & isfinite(y) & isfinite(u) & isfinite(v) & isfinite(cx_raw) & hypot(x, y) <= 4;
    z_samples(~base_good,:) = NaN;
    rho_samples(~base_good,:) = NaN;
    rho_abs_samples(~base_good,:) = NaN;
    valid_pair = isfinite(z_samples) & isfinite(rho_samples);
    z_samples(~valid_pair) = NaN;
    rho_samples(~valid_pair) = NaN;
    rho_abs_samples(~valid_pair) = NaN;
    map_timer = tic;
    if strcmp(grid_mapping, 'cressman')
        [mapped_stack, support_stack] = cressman_map_multi_missing(x, y, [z_samples, rho_samples, rho_abs_samples], X, Y, cressman_radius_r, cressman_min_obs);
        for zz = 1:nz
            grid3d.valid_profile_count(zz) = nnz(profile_cache.profile_valid(:,zz));
            grid3d.boa_bg_valid_count(zz) = nnz(profile_cache.boa_valid(:,zz));
            good = base_good & valid_pair(:,zz);
            xb = discretize(x(good), edges);
            yb = discretize(y(good), edges);
            bin_ok = isfinite(xb) & isfinite(yb);
            if any(bin_ok)
                subs = [yb(bin_ok), xb(bin_ok)];
                grid3d.count(:,:,zz) = accumarray(subs, 1, [grid_n grid_n], @sum, 0);
            end
            z_grid = mapped_stack(:,:,zz);
            rho_grid = mapped_stack(:,:,nz+zz);
            rho_abs_grid = mapped_stack(:,:,2*nz+zz);
            support = support_stack(:,:,zz) >= cressman_min_obs & support_stack(:,:,nz+zz) >= cressman_min_obs & support_stack(:,:,2*nz+zz) >= cressman_min_obs;
            grid3d.mapped_support(:,:,zz) = min(min(support_stack(:,:,zz), support_stack(:,:,nz+zz)), support_stack(:,:,2*nz+zz));
            if smooth_passes > 0
                z_grid = smooth2_supported(z_grid, support, smooth_passes);
                rho_grid = smooth2_supported(rho_grid, support, smooth_passes);
                rho_abs_grid = smooth2_supported(rho_abs_grid, support, smooth_passes);
            end
            grid3d.z_anom(:,:,zz) = mask_to_support(z_grid, support);
            grid3d.rho_anom(:,:,zz) = mask_to_support(rho_grid, support);
            grid3d.rho_abs(:,:,zz) = mask_to_support(rho_abs_grid, support);
            rho_grids(:,:,zz) = rho_grid;
            support3(:,:,zz) = support;
        end
    else
        for zz = 1:nz
            z_anom_sample = z_samples(:,zz);
            rho_anom_sample = rho_samples(:,zz);
            rho_abs_sample = rho_abs_samples(:,zz);
            good = base_good & valid_pair(:,zz);
            grid3d.valid_profile_count(zz) = nnz(profile_cache.profile_valid(:,zz));
            grid3d.boa_bg_valid_count(zz) = nnz(profile_cache.boa_valid(:,zz));
            xb = discretize(x(good), edges);
            yb = discretize(y(good), edges);
            bin_ok = isfinite(xb) & isfinite(yb);
            if any(bin_ok)
                subs = [yb(bin_ok), xb(bin_ok)];
                grid3d.count(:,:,zz) = accumarray(subs, 1, [grid_n grid_n], @sum, 0);
            end
            if strcmp(grid_mapping, 'scattered')
                z_grid = scattered_map(x(good), y(good), z_anom_sample(good), X, Y);
                rho_grid = scattered_map(x(good), y(good), rho_anom_sample(good), X, Y);
                rho_abs_grid = scattered_map(x(good), y(good), rho_abs_sample(good), X, Y);
                support = isfinite(z_grid) & isfinite(rho_grid) & isfinite(rho_abs_grid) & hypot(X, Y) <= 4;
                grid3d.mapped_support(:,:,zz) = double(support);
            else
                z_grid = nan(size(X)); rho_grid = nan(size(X)); rho_abs_grid = nan(size(X));
                if any(bin_ok)
                    subs = [yb(bin_ok), xb(bin_ok)];
                    z_vals = z_anom_sample(good); rho_vals = rho_anom_sample(good); rho_abs_vals = rho_abs_sample(good);
                    z_grid = accumarray(subs, z_vals(bin_ok), [grid_n grid_n], @(q) median(q, 'omitnan'), NaN);
                    rho_grid = accumarray(subs, rho_vals(bin_ok), [grid_n grid_n], @(q) median(q, 'omitnan'), NaN);
                    rho_abs_grid = accumarray(subs, rho_abs_vals(bin_ok), [grid_n grid_n], @(q) median(q, 'omitnan'), NaN);
                end
                support = grid3d.count(:,:,zz) >= min_bin_count;
                grid3d.mapped_support(:,:,zz) = grid3d.count(:,:,zz);
            end
            if smooth_passes > 0
                z_grid = smooth2_supported(z_grid, support, smooth_passes);
                rho_grid = smooth2_supported(rho_grid, support, smooth_passes);
                rho_abs_grid = smooth2_supported(rho_abs_grid, support, smooth_passes);
            end
            grid3d.z_anom(:,:,zz) = mask_to_support(z_grid, support);
            grid3d.rho_anom(:,:,zz) = mask_to_support(rho_grid, support);
            grid3d.rho_abs(:,:,zz) = mask_to_support(rho_abs_grid, support);
            rho_grids(:,:,zz) = rho_grid;
            support3(:,:,zz) = support;
        end
    end
    log_step(sprintf('%s %s depth-stack Cressman mapped %d layers in %.1f s', polarity, band_label, nz, toc(map_timer)));
    if strcmp(vertical_mode, 'thermal_wind_depth_stack')
        tw_timer = tic;
        [grid3d.u_tw, grid3d.v_tw] = thermal_wind_velocity_stack(rho_grids, support3, u_base, v_base, base_support, depth_levels, dx_m, dy_m, grid3d.thermal_wind_f_s_1);
        log_step(sprintf('%s %s thermal-wind velocity stack ready in %.1f s', polarity, band_label, toc(tw_timer)));
    else
        grid3d.u_tw = repmat(u_base, 1, 1, nz);
        grid3d.v_tw = repmat(v_base, 1, 1, nz);
    end
    w_timer = tic;
    for zz = 1:nz
        z_grid = grid3d.z_anom(:,:,zz);
        support = support3(:,:,zz) & isfinite(grid3d.u_tw(:,:,zz)) & isfinite(grid3d.v_tw(:,:,zz));
        if isfinite(dx_m) && dx_m > 0 && isfinite(dy_m) && dy_m > 0
            [dzdx, dzdy] = gradient(fillmissing2(z_grid), dx_m, dy_m);
            term1 = mask_to_support(grid3d.cx_rel .* dzdx, support);
            term2 = mask_to_support(-((grid3d.u_tw(:,:,zz) - grid3d.mean_cx_raw) .* dzdx + grid3d.v_tw(:,:,zz) .* dzdy), support);
            grid3d.term1(:,:,zz) = term1;
            grid3d.term2(:,:,zz) = term2;
            grid3d.w(:,:,zz) = mask_to_support(term1 + term2, support);
        end
    end
    log_step(sprintf('%s %s W terms computed for %d layers in %.1f s', polarity, band_label, nz, toc(w_timer)));
    [grid3d.section_coord, grid3d.section_w] = section_from_grid3d(grid3d, section_axis, section_half_width_r);
end

function profile_cache = profile_depth_stack_cache(matches, rho, depth, boa_clim, depth_levels, min_drho_dz, max_rho_bracket_dz_m, cache_root, polarity, band_label, match_mode)
    argo_indices_all = cell2mat(matches(:,3));
    [argo_indices, first_pos] = unique(argo_indices_all(:), 'stable');
    lon = cell2mat(matches(first_pos,6));
    lat = cell2mat(matches(first_pos,7));
    time = cell2mat(matches(first_pos,5));
    n_unique = numel(argo_indices);
    nz = numel(depth_levels);
    depth_tag = sprintf('%gm_%gm_%03dlev', min(depth_levels), max(depth_levels), nz);
    cache_file = fullfile(cache_root, ['boa_profile_qc_' sanitize_filename(band_label) '_' sanitize_filename(match_mode) '_' depth_tag '.mat']);
    key = matlab.lang.makeValidName([polarity '_' band_label '_' match_mode '_' depth_tag]);
    if exist(cache_file, 'file') == 2
        C = load(cache_file, 'profile_cache_store');
        if isfield(C, 'profile_cache_store') && isfield(C.profile_cache_store, key)
            cached = C.profile_cache_store.(key);
            if isequal(cached.argo_indices(:), argo_indices(:)) && isequal(cached.depth_levels(:), depth_levels(:)) && isfield(cached, 'rho_abs')
                profile_cache = cached;
                log_step(sprintf('3D profile BOA/QC cache hit: %s [%s]', cache_file, key));
                return
            end
        end
    end
    log_step(sprintf('3D profile BOA/QC cache build: %d unique Argo x %d depths', n_unique, nz));
    z_anom = nan(n_unique, nz);
    rho_anom = nan(n_unique, nz);
    rho_abs = nan(n_unique, nz);
    boa_rho = nan(n_unique, nz);
    profile_valid = false(n_unique, nz);
    boa_valid = false(n_unique, nz);
    global QC_WORKERS;
    use_parallel = maybe_start_parallel_pool(QC_WORKERS);
    if use_parallel
        parfor uu = 1:n_unique
            [z_row, rho_row, rho_abs_row, boa_rho_row, pv_row, bv_row] = profile_depth_stack_one(argo_indices(uu), lon(uu), lat(uu), time(uu), rho, depth, boa_clim, depth_levels, min_drho_dz, max_rho_bracket_dz_m);
            z_anom(uu,:) = z_row;
            rho_anom(uu,:) = rho_row;
            rho_abs(uu,:) = rho_abs_row;
            boa_rho(uu,:) = boa_rho_row;
            profile_valid(uu,:) = pv_row;
            boa_valid(uu,:) = bv_row;
        end
    else
        for uu = 1:n_unique
            [z_row, rho_row, rho_abs_row, boa_rho_row, pv_row, bv_row] = profile_depth_stack_one(argo_indices(uu), lon(uu), lat(uu), time(uu), rho, depth, boa_clim, depth_levels, min_drho_dz, max_rho_bracket_dz_m);
            z_anom(uu,:) = z_row;
            rho_anom(uu,:) = rho_row;
            rho_abs(uu,:) = rho_abs_row;
            boa_rho(uu,:) = boa_rho_row;
            profile_valid(uu,:) = pv_row;
            boa_valid(uu,:) = bv_row;
        end
    end
    profile_cache = struct('argo_indices', argo_indices(:), 'depth_levels', depth_levels(:), ...
        'z_anom', z_anom, 'rho_anom', rho_anom, 'rho_abs', rho_abs, 'boa_rho', boa_rho, 'profile_valid', profile_valid, 'boa_valid', boa_valid);
    if exist(cache_file, 'file') == 2
        C = load(cache_file, 'profile_cache_store');
        if isfield(C, 'profile_cache_store')
            profile_cache_store = C.profile_cache_store; %#ok<NASGU>
        else
            profile_cache_store = struct(); %#ok<NASGU>
        end
    else
        profile_cache_store = struct(); %#ok<NASGU>
    end
    profile_cache_store.(key) = profile_cache; %#ok<STRNU>
    save(cache_file, 'profile_cache_store', '-v7.3');
    log_step(sprintf('3D profile BOA/QC cache saved: %s [%s]', cache_file, key));
end

function [z_row, rho_row, rho_abs_row, boa_rho_row, profile_valid, boa_valid] = profile_depth_stack_one(argo_index, lon, lat, time_value, rho, depth, boa_clim, depth_levels, min_drho_dz, max_rho_bracket_dz_m)
    nz = numel(depth_levels);
    z_row = nan(1, nz);
    rho_row = nan(1, nz);
    rho_abs_row = nan(1, nz);
    boa_rho_row = nan(1, nz);
    profile_valid = false(1, nz);
    boa_valid = false(1, nz);
    [~, month_id, ~] = datevec(time_value);
    boa_profile = boa_density_profile_at(boa_clim, lon, lat, month_id);
    target_profile_raw = double(rho(argo_index,:));
    for zz = 1:nz
        z0 = depth_levels(zz);
        rho0 = interp1(boa_clim.pres, boa_profile, z0, 'linear', NaN);
        if ~isfinite(rho0)
            continue
        end
        boa_rho_row(zz) = rho0;
        target_profile = align_density_units(target_profile_raw, rho0);
        boa_profile_aligned = align_density_units(boa_profile, rho0);
        rho_profile_z0 = interp1(depth, target_profile, z0, 'linear', NaN);
        if isfinite(rho_profile_z0)
            rho_abs_row(zz) = rho_profile_z0;
            rho_row(zz) = rho_profile_z0 - rho0;
        end
        [z_rho, ~, bracket_dz, local_drho_dz] = isopycnal_depth_qc(depth, target_profile, rho0, z0);
        [z_bg, ~, bg_bracket_dz, bg_drho_dz] = isopycnal_depth_qc(boa_clim.pres, boa_profile_aligned, rho0, z0);
        profile_ok = isfinite(z_rho) && isfinite(bracket_dz) && bracket_dz <= max_rho_bracket_dz_m && ...
            isfinite(local_drho_dz) && abs(local_drho_dz) >= min_drho_dz;
        bg_ok = isfinite(z_bg) && isfinite(bg_bracket_dz) && bg_bracket_dz <= max_rho_bracket_dz_m && ...
            isfinite(bg_drho_dz) && abs(bg_drho_dz) >= min_drho_dz;
        profile_valid(zz) = profile_ok;
        boa_valid(zz) = bg_ok;
        if profile_ok && bg_ok
            z_row(zz) = z_rho - z_bg;
        end
    end
end

function [u_tw, v_tw] = thermal_wind_velocity_stack(rho_grids, support3, u_base, v_base, base_support, depth_levels, dx_m, dy_m, f)
    [ny, nx, nz] = size(rho_grids);
    u_tw = nan(ny, nx, nz);
    v_tw = nan(ny, nx, nz);
    if ~isfinite(f) || abs(f) < 1e-7 || ~isfinite(dx_m) || dx_m <= 0 || ~isfinite(dy_m) || dy_m <= 0
        return
    end
    g = 9.81;
    rho_ref = 1025;
    du_dD = nan(ny, nx, nz);
    dv_dD = nan(ny, nx, nz);
    for zz = 1:nz
        rho_grid = fillmissing2(rho_grids(:,:,zz));
        [drhodx, drhody] = gradient(rho_grid, dx_m, dy_m);
        du_dD(:,:,zz) = mask_to_support(g ./ (f * rho_ref) .* drhody, support3(:,:,zz));
        dv_dD(:,:,zz) = mask_to_support(-g ./ (f * rho_ref) .* drhodx, support3(:,:,zz));
    end
    [~, anchor] = min(abs(depth_levels(:) - 1000));
    u_tw(:,:,anchor) = mask_to_support(u_base, base_support & support3(:,:,anchor));
    v_tw(:,:,anchor) = mask_to_support(v_base, base_support & support3(:,:,anchor));
    for zz = anchor+1:nz
        dD = depth_levels(zz) - depth_levels(zz-1);
        u_tw(:,:,zz) = u_tw(:,:,zz-1) + 0.5 .* (du_dD(:,:,zz-1) + du_dD(:,:,zz)) .* dD;
        v_tw(:,:,zz) = v_tw(:,:,zz-1) + 0.5 .* (dv_dD(:,:,zz-1) + dv_dD(:,:,zz)) .* dD;
        u_tw(:,:,zz) = mask_to_support(u_tw(:,:,zz), base_support & support3(:,:,zz));
        v_tw(:,:,zz) = mask_to_support(v_tw(:,:,zz), base_support & support3(:,:,zz));
    end
    for zz = anchor-1:-1:1
        dD = depth_levels(zz+1) - depth_levels(zz);
        u_tw(:,:,zz) = u_tw(:,:,zz+1) - 0.5 .* (du_dD(:,:,zz+1) + du_dD(:,:,zz)) .* dD;
        v_tw(:,:,zz) = v_tw(:,:,zz+1) - 0.5 .* (dv_dD(:,:,zz+1) + dv_dD(:,:,zz)) .* dD;
        u_tw(:,:,zz) = mask_to_support(u_tw(:,:,zz), base_support & support3(:,:,zz));
        v_tw(:,:,zz) = mask_to_support(v_tw(:,:,zz), base_support & support3(:,:,zz));
    end
end

function text = sanitize_filename(text)
    text = regexprep(char(text), '[^A-Za-z0-9]+', '_');
    text = regexprep(text, '^_+|_+$', '');
end

function [coord, section_w] = section_from_grid3d(grid3d, section_axis, half_width_r)
    if strcmp(section_axis, 'y')
        mask = abs(grid3d.x(1,:)) <= half_width_r;
        coord = grid3d.y(:,1);
        section_w = squeeze(median(grid3d.w(:,mask,:), 2, 'omitnan'))';
    else
        mask = abs(grid3d.y(:,1)) <= half_width_r;
        coord = grid3d.x(1,:);
        section_w = squeeze(median(grid3d.w(mask,:,:), 1, 'omitnan'))';
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
    global USE_GPU_CRESSMAN;
    if USE_GPU_CRESSMAN
        try
            [Z, support_count] = cressman_map_multi_gpu(x, y, V, X, Y, radius_r, min_obs);
            return
        catch ME
            warning('GPU Cressman failed; falling back to CPU for this map: %s', ME.message);
        end
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

function [Z, support_count] = cressman_map_multi_missing(x, y, V, X, Y, radius_r, min_obs)
    n_var = size(V, 2);
    Z = NaN([size(X), n_var]);
    support_count = zeros([size(X), n_var]);
    good_xy = isfinite(x) & isfinite(y) & any(isfinite(V), 2) & hypot(x, y) <= 4;
    x = x(good_xy);
    y = y(good_xy);
    V = V(good_xy,:);
    if isempty(x) || ~isfinite(radius_r) || radius_r <= 0
        return
    end
    global USE_GPU_CRESSMAN;
    if USE_GPU_CRESSMAN
        try
            [Z, support_count] = cressman_map_multi_missing_gpu(x, y, V, X, Y, radius_r, min_obs);
            return
        catch ME
            warning('GPU missing-aware Cressman failed; falling back to CPU for this map: %s', ME.message);
        end
    end
    r2_limit = radius_r ^ 2;
    finite_v = isfinite(V);
    V0 = V;
    V0(~finite_v) = 0;
    for ii = 1:numel(X)
        d2 = (x - X(ii)).^2 + (y - Y(ii)).^2;
        inside = d2 < r2_limit;
        if ~any(inside)
            continue
        end
        w = (r2_limit - d2(inside)) ./ (r2_limit + d2(inside));
        ok_w = isfinite(w) & w > 0;
        if ~any(ok_w)
            continue
        end
        w = w(ok_w);
        finite_inside = finite_v(inside,:);
        finite_inside = finite_inside(ok_w,:);
        vals = V0(inside,:);
        vals = vals(ok_w,:);
        counts = sum(finite_inside, 1);
        denom = sum(w .* finite_inside, 1);
        mapped = sum(w .* vals, 1) ./ denom;
        ok = counts >= min_obs & denom > 0;
        support_count(ii + (0:n_var-1) * numel(X)) = counts;
        if any(ok)
            Z(ii + find(ok) * numel(X) - numel(X)) = mapped(ok);
        end
    end
    outside = hypot(X, Y) > 4;
    for kk = 1:n_var
        tmp = Z(:,:,kk);
        tmp(outside) = NaN;
        Z(:,:,kk) = tmp;
        tmp_count = support_count(:,:,kk);
        tmp_count(outside) = 0;
        support_count(:,:,kk) = tmp_count;
    end
end

function [Z, support_count] = cressman_map_multi_missing_gpu(x, y, V, X, Y, radius_r, min_obs)
    n_grid = numel(X);
    n_var = size(V, 2);
    Z = NaN([size(X), n_var]);
    support_count = zeros([size(X), n_var]);
    xg = gpuArray(single(x(:)));
    yg = gpuArray(single(y(:)));
    valid_cpu = isfinite(V);
    V0 = single(V);
    V0(~valid_cpu) = 0;
    Vg = gpuArray(V0);
    valid_g = gpuArray(single(valid_cpu));
    Xv = single(X(:)');
    Yv = single(Y(:)');
    r2_limit = single(radius_r ^ 2);
    grid_chunk = 1024;
    var_chunk = 64;
    for start_idx = 1:grid_chunk:n_grid
        stop_idx = min(n_grid, start_idx + grid_chunk - 1);
        cols = start_idx:stop_idx;
        Xg = gpuArray(Xv(cols));
        Yg = gpuArray(Yv(cols));
        d2 = (xg - Xg) .^ 2 + (yg - Yg) .^ 2;
        inside = d2 < r2_limit;
        W = (r2_limit - d2) ./ (r2_limit + d2);
        W(~inside) = 0;
        Wt = W';
        inside_t = single(inside');
        for var0 = 1:var_chunk:n_var
            var1 = min(n_var, var0 + var_chunk - 1);
            vars = var0:var1;
            valid_chunk = valid_g(:,vars);
            counts = inside_t * valid_chunk;
            denom = Wt * valid_chunk;
            numerator = Wt * Vg(:,vars);
            mapped = numerator ./ denom;
            counts_cpu = double(gather(counts));
            denom_cpu = double(gather(denom));
            mapped_cpu = double(gather(mapped));
            ok = counts_cpu >= min_obs & denom_cpu > 0;
            block = nan(numel(cols), numel(vars));
            block(ok) = mapped_cpu(ok);
            linear_idx = cols(:) + (vars - 1) * n_grid;
            Z(linear_idx) = block;
            support_count(linear_idx) = counts_cpu;
        end
    end
    outside = hypot(X, Y) > 4;
    for kk = 1:n_var
        tmp = Z(:,:,kk);
        tmp(outside) = NaN;
        Z(:,:,kk) = tmp;
        tmp_count = support_count(:,:,kk);
        tmp_count(outside) = 0;
        support_count(:,:,kk) = tmp_count;
    end
end

function [Z, support_count] = cressman_map_multi_gpu(x, y, V, X, Y, radius_r, min_obs)
    Z = NaN([size(X), size(V, 2)]);
    support_count = zeros(size(X));
    xg = gpuArray(single(x(:)));
    yg = gpuArray(single(y(:)));
    Vg = gpuArray(single(V));
    Xv = single(X(:)');
    Yv = single(Y(:)');
    r2_limit = single(radius_r ^ 2);
    n_grid = numel(X);
    n_var = size(V, 2);
    chunk = 2048;
    for start_idx = 1:chunk:n_grid
        stop_idx = min(n_grid, start_idx + chunk - 1);
        cols = start_idx:stop_idx;
        Xg = gpuArray(Xv(cols));
        Yg = gpuArray(Yv(cols));
        d2 = (xg - Xg) .^ 2 + (yg - Yg) .^ 2;
        inside = d2 < r2_limit;
        support = sum(inside, 1);
        W = (r2_limit - d2) ./ (r2_limit + d2);
        W(~inside) = 0;
        denom = sum(W, 1);
        support_cpu = gather(support);
        support_count(cols) = double(support_cpu);
        ok_cols = support_cpu >= min_obs & gather(denom) > 0;
        if any(ok_cols)
            for kk = 1:n_var
                val = Vg(:,kk);
                mapped = sum(W .* val, 1) ./ denom;
                mapped_cpu = double(gather(mapped));
                out = nan(1, numel(cols));
                out(ok_cols) = mapped_cpu(ok_cols);
                Z(cols + (kk-1) * n_grid) = out;
            end
        end
    end
    outside = hypot(X, Y) > 4;
    for kk = 1:n_var
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

function grid_file = write_group_outputs(group_dir, matches, grid, polarity, band_label, write_matches)
    if nargin < 6
        write_matches = true;
    end
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
    global WRITE_MATCHED_CSV WRITE_GRID_JSON WRITE_GRID_NC;
    if write_matches
        write_matched_table_mat(fullfile(group_dir, 'matched_core_argo.mat'), header, matches);
        if WRITE_MATCHED_CSV
            writecell(clean_write_cells([header; matches]), fullfile(group_dir, 'matched_core_argo.csv'));
        end
    end
    grid_file = fullfile(group_dir, 'composite_grid.mat');
    write_grid_mat(grid_file, grid, polarity, band_label);
    if WRITE_GRID_NC
        write_grid_nc_file(fullfile(group_dir, 'composite_grid.nc'), grid, polarity, band_label);
    end
    if WRITE_GRID_JSON
        write_grid_json(fullfile(group_dir, 'composite_grid.json'), grid, polarity, band_label);
    end
    plot_three_panel(fullfile(group_dir, 'vertical_transport_terms.png'), grid, [polarity ' ' band_label]);
    plot_sensitivity(fullfile(group_dir, 'velocity_sign_sensitivity.png'), grid, [polarity ' ' band_label]);
    plot_wpk_validation(fullfile(group_dir, 'wpk_validation.png'), grid, [polarity ' ' band_label]);
    plot_gradient_order_comparison(fullfile(group_dir, 'gradient_order_comparison.png'), grid, [polarity ' ' band_label]);
    write_group_doc(fullfile(group_dir, 'METHOD_ASSUMPTIONS_ZH.md'), matches, grid, polarity, band_label);
end

function grid_file = write_group_outputs_3d(group_dir, matches, grid3d, polarity, band_label)
    grid3d.match_count = size(matches, 1);
    if isempty(matches)
        grid3d.unique_argo_count = 0;
    else
        grid3d.unique_argo_count = numel(unique(cell2mat(matches(:,3))));
    end
    grid3d.duplicate_match_count = grid3d.match_count - grid3d.unique_argo_count;
    header = {'polarity','lat_band','argo_index','platform','argo_time','argo_lon','argo_lat','parking_depth_m','u_argo_m_s','v_argo_m_s','wpk_observed_m_s','rho0_parking','z_rho_m','z_rho_bg_m','z_rho_anom_m','rho_crossing_count','rho_bracket_dz_m','local_drho_dz','eddy_track','eddy_time','eddy_lon','eddy_lat','eddy_radius_m','x_over_R','y_over_R','r_over_R','ring','cx_raw_m_s','history_velocity_matched','boa_rho_crossing_count','boa_rho_bracket_dz_m','boa_local_drho_dz','boa_bg_valid'};
    global WRITE_MATCHED_CSV WRITE_GRID_JSON WRITE_GRID_NC;
    io_timer = tic;
    stage_timer = tic;
    write_matched_table_mat(fullfile(group_dir, 'matched_core_argo_3d.mat'), header, matches);
    log_step(sprintf('%s %s wrote matched MAT in %.1f s', polarity, band_label, toc(stage_timer)));
    if WRITE_MATCHED_CSV
        stage_timer = tic;
        writecell(clean_write_cells([header; matches]), fullfile(group_dir, 'matched_core_argo_3d.csv'));
        log_step(sprintf('%s %s wrote matched CSV in %.1f s', polarity, band_label, toc(stage_timer)));
    end
    grid_file = fullfile(group_dir, 'w_3d_grid.mat');
    stage_timer = tic;
    write_grid_3d_mat(grid_file, grid3d, polarity, band_label);
    log_step(sprintf('%s %s wrote 3D grid MAT in %.1f s', polarity, band_label, toc(stage_timer)));
    if WRITE_GRID_NC
        stage_timer = tic;
        write_grid_3d_nc_file(fullfile(group_dir, 'w_3d_grid.nc'), grid3d, polarity, band_label);
        log_step(sprintf('%s %s wrote 3D grid NetCDF in %.1f s', polarity, band_label, toc(stage_timer)));
    end
    if WRITE_GRID_JSON
        stage_timer = tic;
        write_grid_json_3d(fullfile(group_dir, 'w_3d_grid.json'), grid3d, polarity, band_label);
        log_step(sprintf('%s %s wrote 3D grid JSON in %.1f s', polarity, band_label, toc(stage_timer)));
    end
    stage_timer = tic;
    plot_3d_section(fullfile(group_dir, ['w_3d_section_' grid3d.section_axis '.png']), grid3d, [polarity ' ' band_label]);
    log_step(sprintf('%s %s wrote section PNG in %.1f s', polarity, band_label, toc(stage_timer)));
    stage_timer = tic;
    plot_3d_depth_slices(fullfile(group_dir, 'w_3d_depth_slices.png'), grid3d, [polarity ' ' band_label]);
    log_step(sprintf('%s %s wrote depth-slice PNG in %.1f s', polarity, band_label, toc(stage_timer)));
    write_group_doc_3d(fullfile(group_dir, 'METHOD_3D_W_ZH.md'), matches, grid3d, polarity, band_label);
    log_step(sprintf('%s %s total 3D output write time %.1f s', polarity, band_label, toc(io_timer)));
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

function write_matched_table_mat(path, header, matches)
    matched = struct();
    matched.header = header;
    matched.row_count = size(matches, 1);
    matched.note = 'Large matched tables are stored as MAT by default; use --write-matched-csv to opt in to CSV.';
    field_names = matlab.lang.makeValidName(header);
    for c = 1:numel(header)
        if isempty(matches)
            matched.(field_names{c}) = [];
            continue
        end
        col = matches(:, c);
        numeric_col = true;
        for r = 1:numel(col)
            if ~(isnumeric(col{r}) || islogical(col{r})) || ~isscalar(col{r})
                numeric_col = false;
                break
            end
        end
        if numeric_col
            matched.(field_names{c}) = cell2mat(col);
        else
            matched.(field_names{c}) = col;
        end
    end
    save(path, 'matched');
end

function write_summary_table_mat(path, header, rows)
    summary = struct();
    summary.header = header;
    summary.row_count = size(rows, 1);
    field_names = matlab.lang.makeValidName(header);
    for c = 1:numel(header)
        if isempty(rows) || c > size(rows, 2)
            summary.(field_names{c}) = [];
            continue
        end
        col = rows(:, c);
        numeric_col = true;
        for r = 1:numel(col)
            if ~(isnumeric(col{r}) || islogical(col{r})) || ~isscalar(col{r})
                numeric_col = false;
                break
            end
        end
        if numeric_col
            summary.(field_names{c}) = cell2mat(col);
        else
            summary.(field_names{c}) = col;
        end
    end
    save(path, 'summary');
end

function write_grid_mat(path, grid, polarity, band_label)
    metadata = grid_metadata(grid, polarity, band_label);
    save(path, 'grid', 'metadata');
end

function write_grid_3d_mat(path, grid3d, polarity, band_label)
    metadata = grid3d_metadata(grid3d, polarity, band_label);
    save(path, 'grid3d', 'metadata', '-v7.3');
end

function write_grid_nc_file(path, grid, polarity, band_label)
    if exist(path, 'file') == 2
        delete(path);
    end
    [ny, nx] = size(grid.x);
    write_nc_2d(path, 'x_over_R', grid.x);
    write_nc_2d(path, 'y_over_R', grid.y);
    vars = {'z','z_raw','z_anom','u','v','wpk','count','mapped_support','wpk_mapped_support', ...
        'term1','term2','rebuild_w','term1_depth_positive','term2_depth_positive','rebuild_w_raw_depth_positive', ...
        'term1_plus','term1_minus','term2_abs','term2_rel','rebuild_plus_abs','rebuild_minus_rel', ...
        'sample_term1','sample_term2','sample_rebuild_w'};
    names = {'z_rho_m','z_rho_raw_m','z_rho_anom_m','u_argo_m_s','v_argo_m_s','wpk_observed_m_s','sample_count','mapped_support','wpk_mapped_support', ...
        'term1_m_s','term2_m_s','rebuild_w_m_s','term1_depth_positive_m_s','term2_depth_positive_m_s','rebuild_w_raw_depth_positive_m_s', ...
        'term1_plus_m_s','term1_minus_m_s','term2_abs_m_s','term2_rel_m_s','rebuild_plus_abs_m_s','rebuild_minus_rel_m_s', ...
        'sample_term1_m_s','sample_term2_m_s','sample_rebuild_w_m_s'};
    for i = 1:numel(vars)
        if isfield(grid, vars{i})
            write_nc_2d(path, names{i}, grid.(vars{i}));
        end
    end
    ncwriteatt(path, '/', 'polarity', polarity);
    ncwriteatt(path, '/', 'lat_band', band_label);
    ncwriteatt(path, '/', 'w_positive_direction', 'upward');
    ncwriteatt(path, '/', 'depth_positive_direction', 'downward');
    ncwriteatt(path, '/', 'nx', nx);
    ncwriteatt(path, '/', 'ny', ny);
end

function write_nc_2d(path, name, value)
    [ny, nx] = size(value);
    nccreate(path, name, 'Dimensions', {'y', ny, 'x', nx}, 'Datatype', 'double', 'DeflateLevel', 4);
    ncwrite(path, name, double(value));
end

function write_grid_3d_nc_file(path, grid3d, polarity, band_label)
    if exist(path, 'file') == 2
        delete(path);
    end
    [ny, nx, nz] = size(grid3d.w);
    nccreate(path, 'depth_m', 'Dimensions', {'depth', nz}, 'Datatype', 'double', 'DeflateLevel', 4);
    ncwrite(path, 'depth_m', double(grid3d.depth_levels(:)));
    write_nc_2d(path, 'x_over_R', grid3d.x);
    write_nc_2d(path, 'y_over_R', grid3d.y);
    write_nc_3d(path, 'w_3d_m_s', grid3d.w);
    write_nc_3d(path, 'term1_3d_m_s', grid3d.term1);
    write_nc_3d(path, 'term2_3d_m_s', grid3d.term2);
    write_nc_3d(path, 'z_rho_anom_3d_m', grid3d.z_anom);
    write_nc_3d(path, 'rho_anom_3d', grid3d.rho_anom);
    write_nc_3d(path, 'u_thermal_wind_3d_m_s', grid3d.u_tw);
    write_nc_3d(path, 'v_thermal_wind_3d_m_s', grid3d.v_tw);
    write_nc_3d(path, 'sample_count_3d', grid3d.count);
    write_nc_3d(path, 'mapped_support_3d', grid3d.mapped_support);
    ncwriteatt(path, '/', 'polarity', polarity);
    ncwriteatt(path, '/', 'lat_band', band_label);
    ncwriteatt(path, '/', 'w_positive_direction', 'upward');
    ncwriteatt(path, '/', 'depth_positive_direction', 'downward');
end

function write_nc_3d(path, name, value)
    [ny, nx, nz] = size(value);
    nccreate(path, name, 'Dimensions', {'y', ny, 'x', nx, 'depth', nz}, 'Datatype', 'double', 'DeflateLevel', 4);
    ncwrite(path, name, double(value));
end

function metadata = grid_metadata(grid, polarity, band_label)
    metadata = struct('polarity', polarity, 'lat_band', band_label, 'w_positive_direction', 'upward', 'depth_positive_direction', 'downward', ...
        'mean_cx_raw_m_s', grid.mean_cx_raw, 'mean_u_bg_m_s', grid.mean_u_bg, 'cx_rel_m_s', grid.cx_rel, ...
        'mean_radius_m', grid.mean_radius_m, 'mean_wpk_observed_m_s', grid.mean_wpk_observed, ...
        'corr_rebuild_wpk', grid.corr_rebuild_wpk, 'corr_sample_rebuild_wpk', grid.corr_sample_rebuild_wpk, ...
        'grid_mapping', grid.grid_mapping, 'cressman_radius_r', grid.cressman_radius_r, 'cressman_min_obs', grid.cressman_min_obs, ...
        'match_mode', grid.match_mode, 'rho0_mode', grid.rho0_mode, 'z_mode', grid.z_mode, ...
        'z_rho_min_m', grid.z_rho_min_m, 'z_rho_max_m', grid.z_rho_max_m, 'min_drho_dz', grid.min_drho_dz, ...
        'max_rho_bracket_dz_m', grid.max_rho_bracket_dz_m, 'match_count', grid.match_count, ...
        'unique_argo_count', grid.unique_argo_count, 'duplicate_match_count', grid.duplicate_match_count, ...
        'boa_bg_valid_count', grid.boa_bg_valid_count, 'valid_grid_cells', sum(isfinite(grid.rebuild_w(:))), ...
        'total_grid_cells', numel(grid.count));
end

function metadata = grid3d_metadata(grid3d, polarity, band_label)
    metadata = struct('polarity', polarity, 'lat_band', band_label, 'vertical_mode', grid3d.vertical_mode, ...
        'w_positive_direction', 'upward', 'depth_positive_direction', 'downward', 'section_axis', grid3d.section_axis, ...
        'section_half_width_r', grid3d.section_half_width_r, 'mean_cx_raw_m_s', grid3d.mean_cx_raw, ...
        'mean_u_bg_m_s', grid3d.mean_u_bg, 'cx_rel_m_s', grid3d.cx_rel, 'mean_radius_m', grid3d.mean_radius_m, ...
        'match_count', grid3d.match_count, 'unique_argo_count', grid3d.unique_argo_count, ...
        'duplicate_match_count', grid3d.duplicate_match_count, 'match_mode', grid3d.match_mode, ...
        'z_mode', grid3d.z_mode, 'min_drho_dz', grid3d.min_drho_dz, ...
        'max_rho_bracket_dz_m', grid3d.max_rho_bracket_dz_m, ...
        'thermal_wind_anchor_depth_m', grid3d.thermal_wind_anchor_depth_m, 'thermal_wind_f_s_1', grid3d.thermal_wind_f_s_1);
end

function row = summary_from_matches_3d(matches, grid3d, polarity, band_label, group_dir)
    if isempty(matches)
        rings = [0 0 0];
        unique_count = 0;
        history_count = 0;
    else
        r = cell2mat(matches(:,26));
        rings = [sum(r <= 1), sum(r > 1 & r <= 2), sum(r > 2 & r <= 4)];
        unique_count = numel(unique(cell2mat(matches(:,3))));
        history_count = sum(cell2mat(matches(:,29)));
    end
    match_count = size(matches, 1);
    duplicate_count = match_count - unique_count;
    valid_voxels = sum(isfinite(grid3d.w(:)));
    valid_fraction = safe_fraction(valid_voxels, numel(grid3d.w));
    valid_by_depth = squeeze(sum(sum(isfinite(grid3d.w), 1), 2));
    mean_valid_cells = mean(valid_by_depth, 'omitnan');
    boa_fraction = mean(grid3d.boa_bg_valid_count ./ max(match_count, 1), 'omitnan');
    profile_fraction = mean(grid3d.valid_profile_count ./ max(match_count, 1), 'omitnan');
    q95_w = quantile(abs(grid3d.w(isfinite(grid3d.w))) * 1e6, 0.95);
    if isempty(q95_w) || ~isfinite(q95_w)
        q95_w = NaN;
    end
    row = {polarity, band_label, match_count, unique_count, duplicate_count, rings(1), rings(2), rings(3), ...
        numel(grid3d.depth_levels), valid_voxels, valid_fraction, mean_valid_cells, grid3d.mean_cx_raw, grid3d.mean_u_bg, ...
        grid3d.cx_rel, grid3d.mean_radius_m / 1000, history_count, boa_fraction, profile_fraction, q95_w, group_dir};
end

function row = sensitivity_summary_row(matches, grid, polarity, cfg, group_dir)
    n = size(matches, 1);
    if isempty(matches)
        unique_count = 0;
    else
        unique_count = numel(unique(cell2mat(matches(:,3))));
    end
    support = grid.mapped_support(:);
    support = support(isfinite(support) & support > 0);
    if isempty(support)
        support_median = NaN;
        support_p10 = NaN;
    else
        support_median = median(support, 'omitnan');
        support_p10 = quantile(support, 0.10);
    end
    rebuild = grid.rebuild_w(:);
    wpk = grid.wpk(:);
    q95_rebuild = quantile(abs(rebuild(isfinite(rebuild))) * 1e6, 0.95);
    q95_wpk = quantile(abs(wpk(isfinite(wpk))) * 1e6, 0.95);
    if isempty(q95_rebuild), q95_rebuild = NaN; end
    if isempty(q95_wpk), q95_wpk = NaN; end
    rough = roughness_score(grid.rebuild_w);
    dipole = dipole_score(grid.rebuild_w, grid.x, grid.y);
    row = {polarity, cfg.name, cfg.grid_n, cfg.cressman_radius_r, cfg.cressman_min_obs, cfg.smooth_passes, ...
        n, unique_count, n - unique_count, safe_fraction(sum(isfinite(grid.rebuild_w(:))), numel(grid.rebuild_w)), ...
        support_median, support_p10, grid.corr_rebuild_wpk, q95_rebuild, q95_wpk, rough, dipole, group_dir};
end

function score = roughness_score(W)
    Wf = fillmissing2(W);
    L = del2(Wf);
    support = isfinite(W);
    vals = abs(L(support));
    if isempty(vals)
        score = NaN;
    else
        score = median(vals, 'omitnan');
    end
end

function score = dipole_score(W, X, Y)
    mask = hypot(X, Y) <= 2 & isfinite(W);
    ideal = X;
    score = spatial_corr(W(mask), ideal(mask));
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

function write_grid_json_3d(path, grid3d, polarity, band_label)
    G = struct();
    G.metadata = struct('polarity', polarity, 'lat_band', band_label, 'vertical_mode', grid3d.vertical_mode, ...
        'w_positive_direction', 'upward', 'depth_positive_direction', 'downward', 'section_axis', grid3d.section_axis, ...
        'section_half_width_r', grid3d.section_half_width_r, 'mean_cx_raw_m_s', grid3d.mean_cx_raw, ...
        'mean_u_bg_m_s', grid3d.mean_u_bg, 'cx_rel_m_s', grid3d.cx_rel, 'mean_radius_m', grid3d.mean_radius_m, ...
        'match_count', grid3d.match_count, 'unique_argo_count', grid3d.unique_argo_count, ...
        'duplicate_match_count', grid3d.duplicate_match_count, 'match_mode', grid3d.match_mode, ...
        'z_mode', grid3d.z_mode, 'min_drho_dz', grid3d.min_drho_dz, ...
        'max_rho_bracket_dz_m', grid3d.max_rho_bracket_dz_m, ...
        'thermal_wind_anchor_depth_m', grid3d.thermal_wind_anchor_depth_m, 'thermal_wind_f_s_1', grid3d.thermal_wind_f_s_1);
    G.depth_m = grid3d.depth_levels(:)';
    G.x_over_R = grid3d.x;
    G.y_over_R = grid3d.y;
    G.w_3d_m_s = permute(grid3d.w, [3 1 2]);
    G.term1_3d_m_s = permute(grid3d.term1, [3 1 2]);
    G.term2_3d_m_s = permute(grid3d.term2, [3 1 2]);
    G.z_rho_anom_3d_m = permute(grid3d.z_anom, [3 1 2]);
    G.rho_anom_3d = permute(grid3d.rho_anom, [3 1 2]);
    if isfield(grid3d, 'rho_abs')
        G.rho_abs_3d = permute(grid3d.rho_abs, [3 1 2]);
    end
    G.u_thermal_wind_3d_m_s = permute(grid3d.u_tw, [3 1 2]);
    G.v_thermal_wind_3d_m_s = permute(grid3d.v_tw, [3 1 2]);
    G.sample_count_3d = permute(grid3d.count, [3 1 2]);
    G.mapped_support_3d = permute(grid3d.mapped_support, [3 1 2]);
    G.valid_profile_count_by_depth = grid3d.valid_profile_count(:)';
    G.boa_bg_valid_count_by_depth = grid3d.boa_bg_valid_count(:)';
    G.section_axis_coord_over_R = grid3d.section_coord(:)';
    G.section_depth_m = grid3d.depth_levels(:)';
    G.section_w_m_s = grid3d.section_w;
    fid = fopen(path, 'w');
    fwrite(fid, jsonencode(G), 'char');
    fclose(fid);
end

function plot_filled_2d_field(grid, data, lim_micro)
    if any(isfinite(data(:)))
        contourf(grid.x(1,:), grid.y(:,1), data, 24, 'LineStyle', 'none');
    else
        h = imagesc(grid.x(1,:), grid.y(:,1), data);
        set(h, 'AlphaData', isfinite(data));
    end
    set(gca, 'YDir', 'normal');
    set(gca, 'Color', [1 1 1]);
    axis image;
    xlim([-4 4]); ylim([-4 4]);
    clim([-lim_micro lim_micro]);
    colormap(redblue_colormap());
    colorbar;
end

function label = lat_band_label(lat_min, lat_max)
    label = [lat_token(lat_min) '_' lat_token(lat_max)];
end

function label = crossing_label(target_lat, intersect_radius_r)
    radius_text = regexprep(sprintf('%.6g', intersect_radius_r), '\.', 'p');
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
    titles = {'+c_x^{rel} dz''_\rho/dx  (W up+)','-(u_{pk}-c_x, v_{pk}) \cdot \nabla z''_\rho','rebuild W  (up+)'};
    vals = [grid.term1(:); grid.term2(:); grid.rebuild_w(:)];
    lim = max(abs(vals(isfinite(vals))));
    if isempty(lim) || ~isfinite(lim) || lim == 0
        lim = 2.5e-6;
    end
    lim = max(lim, 2.5e-6);
    for k = 1:3
        subplot(1,3,k);
        data = grid.(fields{k}) * 1e6;
        plot_filled_2d_field(grid, data, lim * 1e6);
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
    titles = {'rebuild W (up+)','observed I\_Wpk (up+)'};
    vals = [grid.rebuild_w(:); grid.wpk(:)];
    lim = max(abs(vals(isfinite(vals))));
    if isempty(lim) || ~isfinite(lim) || lim == 0
        lim = 2.5e-6;
    end
    lim = max(lim, 2.5e-6);
    for k = 1:2
        subplot(1,2,k);
        data = grid.(fields{k}) * 1e6;
        plot_filled_2d_field(grid, data, lim * 1e6);
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
        plot_filled_2d_field(grid, data, lim * 1e6);
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
        plot_filled_2d_field(grid, data, lim * 1e6);
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

function plot_3d_section(path, grid3d, title_prefix)
    fig = figure('Visible','off','Position',[100 100 760 760]);
    data = grid3d.section_w * 1e6;
    coord = grid3d.section_coord(:)';
    depth_plot = grid3d.depth_levels(:);
    lim = max(abs(data(isfinite(data))));
    if isempty(lim) || ~isfinite(lim) || lim == 0
        lim = 2.5;
    end
    lim = max(lim, 2.5);
    contourf(coord, depth_plot, data, 24, 'LineStyle', 'none');
    set(gca, 'YDir', 'reverse');
    set(gca, 'Color', [1 1 1]);
    colormap(redblue_colormap());
    clim([-lim lim]);
    colorbar;
    if strcmp(grid3d.section_axis, 'y')
        xlabel('y/R');
    else
        xlabel('x/R');
    end
    ylabel('Depth (m)');
    display_label = strrep(title_prefix, 'cross_', '');
    display_label = strrep(display_label, '_', ' ');
    title({display_label, 'W upward-positive (10^{-6} m s^{-1})'}, 'Interpreter', 'tex');
    tmp_path = [tempname(fileparts(path)) '.png'];
    try
        exportgraphics(fig, tmp_path, 'Resolution', 180);
        if exist(path, 'file') == 2
            delete(path);
        end
        movefile(tmp_path, path, 'f');
    catch ME
        fallback_path = fullfile(fileparts(path), ['w_3d_section_' datestr(now, 'yyyymmdd_HHMMSS') '.png']);
        if exist(tmp_path, 'file') == 2
            movefile(tmp_path, fallback_path, 'f');
        end
        warning('Could not replace %s: %s. Wrote %s instead.', path, ME.message, fallback_path);
    end
    close(fig);
end

function plot_sensitivity_montage(path, grids, configs, polarity, band_label)
    fig = figure('Visible','off','Position',[100 100 1500 940]);
    vals = [];
    for c = 1:numel(grids)
        vals = [vals; grids{c}.rebuild_w(:)]; %#ok<AGROW>
    end
    lim = max(abs(vals(isfinite(vals))));
    if isempty(lim) || ~isfinite(lim) || lim == 0
        lim = 2.5e-6;
    end
    lim = max(lim, 2.5e-6);
    for c = 1:numel(grids)
        subplot(2, 3, c);
        grid = grids{c};
        data = grid.rebuild_w * 1e6;
        plot_filled_2d_field(grid, data, lim * 1e6);
        hold on;
        th = linspace(0, 2*pi, 240);
        plot(cos(th), sin(th), 'k-', 'LineWidth', 1.0);
        plot(4*cos(th), 4*sin(th), 'k-', 'LineWidth', 1.0);
        plot(0, 0, 'k.', 'MarkerSize', 14);
        cfg = configs(c);
        title(sprintf('%s: N=%d Rc=%.2g min=%d sm=%d', cfg.name, cfg.grid_n, cfg.cressman_radius_r, cfg.cressman_min_obs, cfg.smooth_passes), 'Interpreter', 'none');
        xlabel('x/R'); ylabel('y/R');
    end
    sgtitle([polarity ' ' band_label ' rebuild W sensitivity  (10^{-6} m s^{-1})'], 'Interpreter', 'tex');
    tmp_path = [tempname(fileparts(path)) '.png'];
    try
        exportgraphics(fig, tmp_path, 'Resolution', 180);
        if exist(path, 'file') == 2
            delete(path);
        end
        movefile(tmp_path, path, 'f');
    catch ME
        fallback_path = fullfile(fileparts(path), [polarity '_sensitivity_montage_' datestr(now, 'yyyymmdd_HHMMSS') '.png']);
        if exist(tmp_path, 'file') == 2
            movefile(tmp_path, fallback_path, 'f');
        end
        warning('Could not replace %s: %s. Wrote %s instead.', path, ME.message, fallback_path);
    end
    close(fig);
end

function plot_3d_depth_slices(path, grid3d, title_prefix)
    fig = figure('Visible','off','Position',[100 100 1500 660]);
    requested = [100 500 1000 1500 1900];
    idx = zeros(size(requested));
    for i = 1:numel(requested)
        [~, idx(i)] = min(abs(grid3d.depth_levels - requested(i)));
    end
    idx = unique(idx, 'stable');
    vals = grid3d.w(:,:,idx) * 1e6;
    lim = max(abs(vals(isfinite(vals))));
    if isempty(lim) || ~isfinite(lim) || lim == 0
        lim = 2.5;
    end
    lim = max(lim, 2.5);
    for k = 1:numel(idx)
        subplot(2, ceil(numel(idx)/2), k);
        data = grid3d.w(:,:,idx(k)) * 1e6;
        plot_filled_2d_field(grid3d, data, lim);
        hold on;
        th = linspace(0, 2*pi, 240);
        plot(cos(th), sin(th), 'k-', 'LineWidth', 1.0);
        plot(4*cos(th), 4*sin(th), 'k-', 'LineWidth', 1.0);
        plot(0, 0, 'k.', 'MarkerSize', 14);
        title(sprintf('%.0f m', grid3d.depth_levels(idx(k))));
        xlabel('x/R'); ylabel('y/R');
    end
    sgtitle([title_prefix ' W depth slices  (10^{-6} m s^{-1})'], 'Interpreter', 'tex');
    tmp_path = [tempname(fileparts(path)) '.png'];
    try
        exportgraphics(fig, tmp_path, 'Resolution', 180);
        if exist(path, 'file') == 2
            delete(path);
        end
        movefile(tmp_path, path, 'f');
    catch ME
        fallback_path = fullfile(fileparts(path), ['w_3d_depth_slices_' datestr(now, 'yyyymmdd_HHMMSS') '.png']);
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
    fprintf(fid, '# META4.0 + Core Argo 垂直速度重建方法与假定\n\n');
    fprintf(fid, '- Argo 主数据源：`%s`\n', argo_mat);
    fprintf(fid, '- 历史 Argo 速度校准源：`%s`\n', history_argo_mat);
    fprintf(fid, '- META4.0 涡旋源：`%s`\n', meta_dir);
    fprintf(fid, '- BOA 背景位密源：`%s`\n', boa_pden_root);
    fprintf(fid, '- 输出根目录：`%s`\n', output_root);
    fprintf(fid, '- 运行范围：`%.1fE-%.1fE, %.1f-%.1f latitude`\n', bbox(1), bbox(2), bbox(3), bbox(4));
    if strcmp(selection_mode, 'crossing_lat')
        fprintf(fid, '- 样本选择：默认 crossing。涡旋本体 `%.3gR` 跨过纬线 `%s`，逐纬线输出 `cross_XX_%.3gR` 目录。Argo profile 不再按纬度带预筛，只由 bbox、parking depth、时间窗和 `0-4R` 空间匹配决定。\n', intersect_radius_r, crossing_list_text(crossing_lats), intersect_radius_r);
    else
        fprintf(fid, '- 纬度带：');
        for i=1:size(lat_bands,1), fprintf(fid, '`%s` ', lat_band_label(lat_bands(i,1), lat_bands(i,2))); end
    end
    fprintf(fid, '\n- Core Argo 限定：`%.0f-%.0f m` parking depth。\n', core_min_m, core_max_m);
    fprintf(fid, '- 时间匹配：Argo profile 与 META 轨迹点相差不超过 `%.1f day`。\n', time_window_days);
    fprintf(fid, '- 匹配模式：`%s`。`nearest` 表示每条 Argo 只归属最近的 `r/R` 涡旋；`all` 表示一条 Argo 可在所有满足时间窗和 `0-4R` 的涡旋坐标系中重复使用。\n', match_mode);
    fprintf(fid, '- 速度来源：`%s`。`argo1000m_match` 使用历史文件 `I_Upk/I_Vpk/I_Wpk` 与 TEOS profile 的 `I_PF/I_Time/I_Lon/I_Lat` 近似键匹配；`profile_diff` 是旧的相邻 profile 位置差近似。\n', velocity_source);
    fprintf(fid, '- 空间分区：保存 `0-1R`、`1-2R`、`2-4R`，图像网格为 `x/R, y/R = -4..4`。\n');
    fprintf(fid, '- 密度变量：`%s`。默认 `rho0` 为每个纬度带/极性在实际 parking depth 处密度的中位数；可用 `--rho0-mode profile` 做旧口径对照。\n', density_variable);
    fprintf(fid, '- `z_mode`：`%s`。`anomaly_boa_climatology` 使用 BOA 多年同月局地背景密度剖面，`anomaly_farfield_plane` 仅作为内部远场平面对照。\n', z_mode);
    fprintf(fid, '- BOA 背景模式：`%s`。默认按月份平均 `PDen1000_YYYYMM.mat`，对每个 profile 的 `lon/lat/month` 双线性插值得到背景密度剖面。\n', boa_background_mode);
    fprintf(fid, '- `z_rho` 反插值：profile 和 BOA 背景均只使用显式 bracket crossing，不再用全剖面 fallback；有效窗口 `%.0f-%.0f m`，`abs(local_drho_dz) >= %.3g`，bracket 厚度 `<= %.0f m`。\n', z_rho_min_m, z_rho_max_m, min_drho_dz, max_rho_bracket_dz_m);
    fprintf(fid, '- 默认网格化：Cressman objective mapping。权重 `w=(R_c^2-r^2)/(R_c^2+r^2)`，仅使用 `R_c` 内样本；默认 `R_c=0.5R`、每格至少 `3` 个样本。`sample_count` 是原始 bin 覆盖，`mapped_support` 是 Cressman 支撑样本数。\n');
    fprintf(fid, '- 默认输出格式：大匹配表写为 `.mat`；网格写为 `.mat` 和 `.nc`；SUMMARY 写为 `.mat`。CSV 与网格 JSON 默认关闭，可用 `--write-matched-csv`、`--write-summary-csv` 和 `--write-grid-json` 显式打开。\n');
    fprintf(fid, '- 默认绘图：二维 W 图使用 `contourf(..., ''LineStyle'', ''none'')`，只显示填色块，不叠加等值线描边。\n');
    fprintf(fid, '- 深度变量约定：`z_rho_m`、`z_rho_bg_m`、`z_rho_anom_m` 均保存为正深度向下，便于海洋剖面阅读。\n');
    fprintf(fid, '- W 符号约定：`rebuild_w_m_s`、`term1_m_s`、`term2_m_s` 统一为向上为正，与历史 `I_Wpk` 中 `z=-Depth` 后计算 `Dz/Dt` 的口径一致；同时保留 `rebuild_w_raw_depth_positive_m_s` 作为深度向下正公式对照。\n');
    fprintf(fid, '- `c_x_raw` 来自 META track 相邻点中央差分；`u_bg` 为同 crossing 组、同极性、匹配 Core Argo 的 parking drift 纬向均值；`c_x_rel = mean(c_x_raw) - mean(u_bg)`。主图采用 `term1 = +c_x_rel dz''_rho/dx`，`term2 = -[(u_pk-c_x_raw, v_pk) · grad(z''_rho)]`，`rebuild_W = term1 + term2`。\n');
    fprintf(fid, '- BOA_Argo 只作为 gridded 密度背景，不直接推导背景速度。\n\n');
    fprintf(fid, '参考：NOAA AOML Argo overview, NOAA Argo best practices, Lin et al. 2019 Remote Sensing, Zhou et al. 2023 JGR Oceans, JAMSTEC Argo gridded products。\n');
    fclose(fid);
end

function write_group_doc(path, matches, grid, polarity, band_label)
    fid = fopen(path, 'w');
    fprintf(fid, '# %s %s Core Argo 垂直速度重建摘要\n\n', polarity, band_label);
    fprintf(fid, '- 匹配样本数：`%d`\n', size(matches,1));
    fprintf(fid, '- mean c_x_raw：`%.6g m/s`\n', grid.mean_cx_raw);
    fprintf(fid, '- mean u_bg：`%.6g m/s`\n', grid.mean_u_bg);
    fprintf(fid, '- c_x_rel：`%.6g m/s`\n', grid.cx_rel);
    fprintf(fid, '- mean observed I_Wpk：`%.6g m/s`\n', grid.mean_wpk_observed);
    fprintf(fid, '- corr(rebuild_W, I_Wpk)：`%.6g`\n', grid.corr_rebuild_wpk);
    fprintf(fid, '- corr(sample-gradient W, I_Wpk)：`%.6g`\n', grid.corr_sample_rebuild_wpk);
    fprintf(fid, '- BOA 背景有效样本：`%d / %d`\n', grid.boa_bg_valid_count, size(matches,1));
    fprintf(fid, '- mean radius：`%.3f km`\n', grid.mean_radius_m / 1000);
    valid_cells = sum(isfinite(grid.rebuild_w(:)));
    valid_fraction = valid_cells / numel(grid.count);
    fprintf(fid, '- 有样本支撑网格：`%d / %d (%.2f%%)`。\n', valid_cells, numel(grid.count), valid_fraction * 100);
    fprintf(fid, '- 符号约定：W 向上为正；`z_rho` 和 `z_rho_anom` 为正深度向下。\n');
    fprintf(fid, '- 输出：默认 `matched_core_argo.mat`、`composite_grid.mat`、`composite_grid.nc`、`vertical_transport_terms.png`、`wpk_validation.png`、`gradient_order_comparison.png`、`velocity_sign_sensitivity.png`。图像显示为 `10^-6 m/s`，网格文件保存原始 `m/s`，白色为空样本格点。\n');
    fclose(fid);
end

function write_group_doc_3d(path, matches, grid3d, polarity, band_label)
    fid = fopen(path, 'w');
    fprintf(fid, '# %s %s Core Argo 三维 W 重建摘要\n\n', polarity, band_label);
    fprintf(fid, '- 垂向模式：`%s`，逐名义深度层重建 `W(x/R,y/R,z)`。\n', grid3d.vertical_mode);
    fprintf(fid, '- 深度层：');
    for i = 1:numel(grid3d.depth_levels)
        fprintf(fid, '`%.0f m` ', grid3d.depth_levels(i));
    end
    fprintf(fid, '\n- 匹配样本数：`%d`，唯一 Argo profile：`%d`。\n', size(matches,1), grid3d.unique_argo_count);
    fprintf(fid, '- mean c_x_raw：`%.6g m/s`，mean u_bg：`%.6g m/s`，c_x_rel：`%.6g m/s`。\n', grid3d.mean_cx_raw, grid3d.mean_u_bg, grid3d.cx_rel);
    if strcmp(grid3d.vertical_mode, 'thermal_wind_depth_stack')
        fprintf(fid, '- 热成风口径：以 1000 m `I_Upk/I_Vpk` 为锚定速度，使用 `rho_anom` 的水平梯度按正深度向下积分得到 `u_tw(z), v_tw(z)`。\n');
        fprintf(fid, '- 热成风剪切：`du/dD = g/(f*rho_ref) * d rho''/dy`，`dv/dD = -g/(f*rho_ref) * d rho''/dx`，`D` 为正深度向下，`f=%.6g s^-1`。\n', grid3d.thermal_wind_f_s_1);
        fprintf(fid, '- 公式：`term1 = +c_x_rel dz''_rho/dx`，`term2 = -[(u_tw-c_x_raw,v_tw)·grad(z''_rho)]`，`W = term1 + term2`。\n');
    else
        fprintf(fid, '- 公式：`term1 = +c_x_rel dz''_rho/dx`，`term2 = -[(u_pk-c_x_raw,v_pk)·grad(z''_rho)]`，`W = term1 + term2`。\n');
    end
    fprintf(fid, '- W 符号：向上为正；深度和 `z_rho_anom` 按正深度向下保存和标注。\n');
    fprintf(fid, '- 横截面：`%s` 方向，半宽 `%.3gR`，纵坐标显示正深度数值。\n', grid3d.section_axis, grid3d.section_half_width_r);
    fprintf(fid, '- 输出：默认 `matched_core_argo_3d.mat`、`w_3d_grid.mat`、`w_3d_grid.nc`、`w_3d_section_%s.png`、`w_3d_depth_slices.png`；网格内包含 `z_anom/rho_anom/u_tw/v_tw/term1/term2/w`。\n', grid3d.section_axis);
    fclose(fid);
end

function write_summary_doc(path, summary_rows, output_root)
    fid = fopen(path, 'w');
    fprintf(fid, '# META4.0 + Core Argo 垂直速度重建运行摘要\n\n');
    fprintf(fid, '- 输出根目录：`%s`\n', output_root);
    fprintf(fid, '- 主图变量：W 向上为正，`term1 = +c_x_rel dz''_rho/dx`，`term2 = -[(u_pk-c_x_raw, v_pk) · grad(z''_rho)]`，`rebuild_W = term1 + term2`。\n');
    fprintf(fid, '- 深度变量：`z_rho_m`、`z_rho_bg_m`、`z_rho_anom_m` 仍为正深度向下；默认 MAT/NetCDF 保存原始 `m/s`；PNG 色标显示为 `10^-6 m/s`；白色为空样本格点。\n');
    fprintf(fid, '- 若使用 `--max-matches-per-group` 做 smoke run，覆盖率会很低；正式结果应使用默认 `0` 读取全部匹配。\n\n');
    fprintf(fid, '| polarity | lat_band | matches | unique Argo | duplicated | 0-1R | 1-2R | 2-4R | valid grid %% | BOA bg %% | c_x_rel m/s | corr W/Wpk | corr sample/Wpk | q95 rebuild | q95 sample | q95 Wpk |\n');
    fprintf(fid, '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n');
    for i=1:size(summary_rows,1)
        fprintf(fid, '| %s | %s | %d | %d | %d | %d | %d | %d | %.2f | %.2f | %.6g | %.3g | %.3g | %.3g | %.3g | %.3g |\n', summary_rows{i,1}, summary_rows{i,2}, summary_rows{i,3}, summary_rows{i,4}, summary_rows{i,5}, summary_rows{i,6}, summary_rows{i,7}, summary_rows{i,8}, summary_rows{i,10} * 100, summary_rows{i,17} * 100, summary_rows{i,13}, summary_rows{i,19}, summary_rows{i,20}, summary_rows{i,21}, summary_rows{i,22}, summary_rows{i,23});
    end
    fclose(fid);
end

function write_summary_doc_3d(path, summary_rows, output_root, depth_levels, section_axis, section_half_width_r)
    fid = fopen(path, 'w');
    fprintf(fid, '# META4.0 + Core Argo 三维 W 重建运行摘要\n\n');
    fprintf(fid, '- 输出根目录：`%s`\n', output_root);
    fprintf(fid, '- 深度层：');
    for i = 1:numel(depth_levels)
        fprintf(fid, '`%.0f m` ', depth_levels(i));
    end
    fprintf(fid, '\n- 横截面：`%s` 方向，半宽 `%.3gR`。纵坐标为正深度数值，W 向上为正。\n', section_axis, section_half_width_r);
    fprintf(fid, '- 公式：`term1 = +c_x_rel dz''_rho/dx`，`term2 = -[(u_pk-c_x_raw,v_pk)·grad(z''_rho)]`，`W = term1 + term2`。\n\n');
    fprintf(fid, '| polarity | lat_band | matches | unique Argo | depth count | valid voxels | valid voxel %% | mean cells/depth | BOA bg %% | profile valid %% | c_x_rel m/s | q95 W |\n');
    fprintf(fid, '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n');
    for i=1:size(summary_rows,1)
        fprintf(fid, '| %s | %s | %d | %d | %d | %d | %.2f | %.1f | %.2f | %.2f | %.6g | %.3g |\n', ...
            summary_rows{i,1}, summary_rows{i,2}, summary_rows{i,3}, summary_rows{i,4}, summary_rows{i,9}, summary_rows{i,10}, ...
            summary_rows{i,11} * 100, summary_rows{i,12}, summary_rows{i,18} * 100, summary_rows{i,19} * 100, summary_rows{i,15}, summary_rows{i,20});
    end
    fclose(fid);
end

function write_best_sensitivity_doc(path, summary_rows)
    fid = fopen(path, 'w');
    fprintf(fid, '# 2D 20N W 参数敏感度推荐\n\n');
    fprintf(fid, '本轮使用缓存后的 20N crossing 样本表，只重做 Cressman、平滑、梯度和出图，用于快速判断碎片/锯齿来源。\n\n');
    if isempty(summary_rows)
        fprintf(fid, '没有可用结果。\n');
        fclose(fid);
        return
    end
    names = unique(summary_rows(:,2), 'stable');
    best_name = '';
    best_score = Inf;
    for i = 1:numel(names)
        name = names{i};
        rows = summary_rows(strcmp(summary_rows(:,2), name), :);
        rough = cell2mat(rows(:,15));
        corrv = abs(cell2mat(rows(:,12)));
        valid = cell2mat(rows(:,9));
        score = mean(rough, 'omitnan') ./ max(mean(valid, 'omitnan'), eps) - 0.1 * mean(corrv, 'omitnan');
        if isfinite(score) && score < best_score
            best_score = score;
            best_name = name;
        end
    end
    fprintf(fid, '- 自动推荐组合：`%s`。\n', best_name);
    fprintf(fid, '- 推荐依据：优先降低 `roughness_score`，同时保留有效覆盖并避免 `corr(rebuild_W,I_Wpk)` 明显恶化。\n');
    fprintf(fid, '- 最终仍需人工看 `cyclonic_sensitivity_montage.png` 和 `anticyclonic_sensitivity_montage.png`，确认是否保留中心东西偶极。\n\n');
    fprintf(fid, '| polarity | config | valid grid %% | support median | support p10 | corr W/Wpk | q95 W | roughness | dipole score |\n');
    fprintf(fid, '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n');
    for i=1:size(summary_rows,1)
        fprintf(fid, '| %s | %s | %.2f | %.1f | %.1f | %.3g | %.3g | %.3g | %.3g |\n', ...
            summary_rows{i,1}, summary_rows{i,2}, summary_rows{i,9} * 100, summary_rows{i,10}, summary_rows{i,11}, ...
            summary_rows{i,12}, summary_rows{i,13}, summary_rows{i,15}, summary_rows{i,16});
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
