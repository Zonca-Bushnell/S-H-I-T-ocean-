function origin_precompute_candidates(filterRoot, outDir, startDateText, endDateText, windowCells, maxCandidates, useGpu)
%ORIGIN_PRECOMPUTE_CANDIDATES Precompute Hua surface SSH seeds from filtered CMEMS NetCDF.
%
% This backend is deliberately limited to the matrix-friendly surface seed
% stage. Python still performs the Hua velocity center, circle checks,
% strict-contiguous vertical extension, tracking, catalog, and shape stages.

if nargin < 5 || isempty(windowCells)
    windowCells = 7;
end
if nargin < 6 || isempty(maxCandidates)
    maxCandidates = 80;
end
if nargin < 7 || isempty(useGpu)
    useGpu = false;
end

filterRoot = char(filterRoot);
outDir = char(outDir);
if ~exist(outDir, 'dir')
    mkdir(outDir);
end

startDay = datetime(char(startDateText), 'InputFormat', 'yyyy-MM-dd');
endDay = datetime(char(endDateText), 'InputFormat', 'yyyy-MM-dd');
years = year(startDay):year(endDay);
manifestRows = {};

for yy = years
    ncPath = fullfile(filterRoot, sprintf('global_phy_%d.nc', yy));
    if ~isfile(ncPath)
        error('Missing filtered NetCDF file: %s', ncPath);
    end
    lon = double(ncread(ncPath, 'longitude'));
    lat = double(ncread(ncPath, 'latitude'));
    time = double(ncread(ncPath, 'time'));
    timeUnits = ncreadatt(ncPath, 'time', 'units');
    fileDays = origin_time_to_days(time, timeUnits);
    info = ncinfo(ncPath, 'zos_glor');
    dimNames = string({info.Dimensions.Name});

    for ti = 1:numel(fileDays)
        day = fileDays(ti);
        if day < startDay || day > endDay
            continue;
        end
        outPath = fullfile(outDir, sprintf('candidates_%s.csv', datestr(day, 'yyyymmdd')));
        if isfile(outPath)
            manifestRows(end + 1, :) = {datestr(day, 'yyyy-mm-dd'), outPath, 0, 'existing'}; %#ok<AGROW>
            continue;
        end
        zos = origin_read_zos_day(ncPath, dimNames, ti, numel(time), numel(lat), numel(lon));
        rows = origin_local_extrema(zos, windowCells, maxCandidates, useGpu);
        origin_write_candidates(outPath, rows, lon, lat, day);
        manifestRows(end + 1, :) = {datestr(day, 'yyyy-mm-dd'), outPath, size(rows, 1), 'written'}; %#ok<AGROW>
        fprintf('[matlab-candidates] %s rows=%d\n', datestr(day, 'yyyy-mm-dd'), size(rows, 1));
    end
end

manifestPath = fullfile(outDir, 'candidate_cache_manifest.csv');
fid = fopen(manifestPath, 'w');
fprintf(fid, 'date,path,n_candidates,status\n');
for ii = 1:size(manifestRows, 1)
    fprintf(fid, '%s,%s,%d,%s\n', manifestRows{ii, 1}, manifestRows{ii, 2}, manifestRows{ii, 3}, manifestRows{ii, 4});
end
fclose(fid);
end


function days = origin_time_to_days(time, unitsText)
unitsText = char(unitsText);
tok = regexp(unitsText, 'seconds since (\d{4})-(\d{2})-(\d{2})', 'tokens', 'once');
if isempty(tok)
    error('Unsupported time units: %s', unitsText);
end
base = datetime(str2double(tok{1}), str2double(tok{2}), str2double(tok{3}));
days = dateshift(base + seconds(time), 'start', 'day');
end


function zos = origin_read_zos_day(ncPath, dimNames, timeIndex, nTime, nLat, nLon)
start = ones(1, numel(dimNames));
count = ones(1, numel(dimNames));
for ii = 1:numel(dimNames)
    name = lower(dimNames(ii));
    if name == "time"
        start(ii) = timeIndex;
        count(ii) = 1;
    elseif name == "latitude"
        count(ii) = nLat;
    elseif name == "longitude"
        count(ii) = nLon;
    else
        error('Unexpected zos_glor dimension: %s', dimNames(ii));
    end
end
raw = squeeze(ncread(ncPath, 'zos_glor', start, count));
spaceDims = dimNames(dimNames ~= "time");
if numel(spaceDims) == 2 && lower(spaceDims(1)) == "longitude" && lower(spaceDims(2)) == "latitude"
    zos = double(raw');
elseif numel(spaceDims) == 2 && lower(spaceDims(1)) == "latitude" && lower(spaceDims(2)) == "longitude"
    zos = double(raw);
elseif isequal(size(raw), [nLat, nLon])
    zos = double(raw);
elseif isequal(size(raw), [nLon, nLat])
    zos = double(raw');
else
    error('Cannot orient zos_glor day slice. Got size [%s], expected [%d,%d].', num2str(size(raw)), nLat, nLon);
end
zos(abs(zos) > 1e30) = NaN;
end


function rows = origin_local_extrema(zos, windowCells, maxCandidates, useGpu)
if useGpu && gpuDeviceCount("available") > 0
    z = gpuArray(zos);
else
    z = zos;
end
finite = isfinite(z);
fillMax = z;
fillMax(~finite) = -Inf;
fillMin = z;
fillMin(~finite) = Inf;

radius = floor(double(windowCells) / 2);
maxMask = finite;
minMask = finite;
for di = -radius:radius
    for dj = -radius:radius
        shiftedMax = circshift(fillMax, [dj, di]);
        shiftedMin = circshift(fillMin, [dj, di]);
        maxMask = maxMask & (fillMax >= shiftedMax);
        minMask = minMask & (fillMin <= shiftedMin);
    end
end
if radius > 0
    maxMask(1:radius, :) = false;
    maxMask(end-radius+1:end, :) = false;
    maxMask(:, 1:radius) = false;
    maxMask(:, end-radius+1:end) = false;
    minMask(1:radius, :) = false;
    minMask(end-radius+1:end, :) = false;
    minMask(:, 1:radius) = false;
    minMask(:, end-radius+1:end) = false;
end

maxMask = gather(maxMask);
minMask = gather(minMask);
zosCpu = gather(z);
[maxJ, maxI] = find(maxMask);
[minJ, minI] = find(minMask);
maxVal = zosCpu(sub2ind(size(zosCpu), maxJ, maxI));
minVal = zosCpu(sub2ind(size(zosCpu), minJ, minI));
kind = [repmat({'ssh_max'}, numel(maxVal), 1); repmat({'ssh_min'}, numel(minVal), 1)];
i0 = [maxI; minI] - 1;
j0 = [maxJ; minJ] - 1;
val = [maxVal; minVal];
absVal = abs(val);
[~, order] = sort(absVal, 'descend');
if maxCandidates > 0
    order = order(1:min(maxCandidates, numel(order)));
end
rows = table(kind(order), i0(order), j0(order), val(order), absVal(order), ...
    ones(numel(order), 1), ...
    'VariableNames', {'ssh_extremum_type', 'seed_i', 'seed_j', 'ssh_value_m', 'abs_ssh_value_m', 'component_pixels'});
end


function origin_write_candidates(outPath, rows, lon, lat, day)
if isempty(rows)
    rows = table(strings(0, 1), zeros(0, 1), zeros(0, 1), zeros(0, 1), zeros(0, 1), zeros(0, 1), ...
        'VariableNames', {'ssh_extremum_type', 'seed_i', 'seed_j', 'ssh_value_m', 'abs_ssh_value_m', 'component_pixels'});
end
n = height(rows);
dateCol = repmat(string(datestr(day, 'yyyy-mm-dd')), n, 1);
seedLon = zeros(n, 1);
seedLat = zeros(n, 1);
for ii = 1:n
    seedLon(ii) = lon(rows.seed_i(ii) + 1);
    seedLat(ii) = lat(rows.seed_j(ii) + 1);
end
rows = addvars(rows, dateCol, seedLon, seedLat, 'Before', 1, ...
    'NewVariableNames', {'date', 'seed_lon', 'seed_lat'});
tmp = [outPath '.tmp'];
writetable(rows, tmp, 'FileType', 'text');
movefile(tmp, outPath, 'f');
end
