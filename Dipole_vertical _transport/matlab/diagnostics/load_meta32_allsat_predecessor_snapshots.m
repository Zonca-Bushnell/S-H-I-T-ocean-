function meta = load_meta32_allsat_predecessor_snapshots(meta32_allsat_dir, polarity, output_root, deg_m)
    nc_file = find_meta32_allsat_file(meta32_allsat_dir, polarity);
    log_step(sprintf('Loading META3.2 allsat %s from %s', polarity, nc_file));

    lat = double(ncread(nc_file, 'latitude'));
    lon = double(ncread(nc_file, 'longitude'));
    time_raw = double(ncread(nc_file, 'time'));
    radius = double(ncread(nc_file, 'speed_radius'));
    track = double(ncread(nc_file, 'track'));

    lon(lon < 0) = lon(lon < 0) + 360;
    time = time_raw;
    cx = track_cx(lon, lat, time, track, deg_m);

    [track_sorted, sort_idx] = sort(track);
    track_change = [true; diff(track_sorted(:)) ~= 0];
    start_pos = find(track_change);
    end_pos = [start_pos(2:end) - 1; numel(track_sorted)];
    keep = false(size(track));
    for ii = 1:numel(start_pos)
        pos = sort_idx(start_pos(ii):end_pos(ii));
        [~, order] = sort(time(pos));
        pos = pos(order);
        life_len = numel(pos);
        dis = fix(life_len / 7);
        if dis < 1
            continue
        end
        local_index = 1:dis:(dis * 7);
        local_index = local_index(local_index <= life_len);
        if numel(local_index) > 7
            local_index = local_index(1:7);
        end
        keep(pos(local_index)) = true;
    end

    t_vec = datevec(time + datenum('1950-01-01')); %#ok<DATNM>
    year_vec = t_vec(:,1);
    keep = keep & year_vec > 2003 & year_vec < 2021 & isfinite(lat) & isfinite(lon) & isfinite(radius) & radius > 0;

    meta = struct();
    meta.source_file = nc_file;
    meta.polarity = polarity;
    meta.lon = lon(keep);
    meta.lat = lat(keep);
    meta.time = time(keep);
    meta.track = track(keep);
    meta.radius = radius(keep);
    meta.cx = cx(keep);
    meta.keep_count = nnz(keep);
    meta.total_count = numel(track);
    meta.track_count = numel(start_pos);
    meta.selection = 'predecessor seven-snapshot per track, years 2004-2020, META3.2 allsat substitute for missing twosat';

    if exist(output_root, 'dir') ~= 7
        mkdir(output_root);
    end
    out_file = fullfile(output_root, sprintf('META32_allsat_%s_predecessor_seven_snapshots.mat', polarity));
    save(out_file, 'meta', '-v7.3');
    meta.cache_file = out_file;
end

function file = find_meta32_allsat_file(meta32_allsat_dir, polarity)
    if strcmp(polarity, 'cyclonic')
        pattern = '*Cyclonic*long*.nc';
        reject = 'Anticyclonic';
    else
        pattern = '*Anticyclonic*long*.nc';
        reject = '';
    end
    files = dir(fullfile(meta32_allsat_dir, pattern));
    if ~isempty(reject)
        keep = true(size(files));
        for kk = 1:numel(files)
            keep(kk) = isempty(strfind(files(kk).name, reject)); %#ok<STREMP>
        end
        files = files(keep);
    end
    if isempty(files)
        error('No META3.2 allsat NetCDF found for %s in %s', polarity, meta32_allsat_dir);
    end
    [~, idx] = max([files.bytes]);
    file = fullfile(files(idx).folder, files(idx).name);
end
