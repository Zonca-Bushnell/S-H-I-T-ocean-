function rows = hua_day_batch_kernel(uDay, vDay, lon, lat, seedCenters, radii, params, targetDegree, windowRadiusCells, minFiniteFraction, stopAtFirstFailed)
%HUA_DAY_BATCH_KERNEL Day-level vertical Hua continuation.
%
% rows = hua_day_batch_kernel(uDay,vDay,lon,lat,seedCenters,radii,params,...)
%
% uDay/vDay are nz-by-ny-by-nx arrays already staged by Python. seedCenters
% are 1-based [i,j] surface SSH seed locations. The kernel performs, inside
% MATLAB, the same daily vertical continuation that Python used to drive one
% depth layer at a time:
%
%   SSH seed -> integer velocity weak center -> pre-Hua subgrid refinement
%   -> Hua boundary check -> next-layer anchor update
%
% It returns one struct row per attempted candidate-layer. Python adds the
% source seed metadata and writes the usual parquet/csv/json products.

if nargin < 11
    stopAtFirstFailed = true;
end
params = day_default_params(params);
seedCenters = double(seedCenters);
lon = double(lon(:));
lat = double(lat(:));

nStates = size(seedCenters, 1);
nz = size(uDay, 1);
rows = struct([]);
if nStates == 0 || nz == 0
    return;
end

prevCenters = zeros(nStates, 2);
surfaceCenters = zeros(nStates, 2);
surfaceSpeeds = nan(nStates, 1);
surfaceSteps = zeros(nStates, 1);
active = true(nStates, 1);

speed0 = hypot(squeeze(uDay(1, :, :)), squeeze(vDay(1, :, :)));
[ci0, cj0, cs0, st0] = seeded_speed_min_batch(speed0, seedCenters(:, 1), seedCenters(:, 2), params.surface_search_cells);
prevCenters(:, 1) = ci0;
prevCenters(:, 2) = cj0;
surfaceCenters = prevCenters;
surfaceSpeeds = cs0;
surfaceSteps = st0;

for kk = 1:nz
    activeIdx = find(active);
    if isempty(activeIdx)
        break;
    end
    uLayer = squeeze(uDay(kk, :, :));
    vLayer = squeeze(vDay(kk, :, :));
    speed = hypot(uLayer, vLayer);
    if kk == 1
        centerI = surfaceCenters(activeIdx, 1);
        centerJ = surfaceCenters(activeIdx, 2);
        centerSpeed = surfaceSpeeds(activeIdx);
        minSteps = surfaceSteps(activeIdx);
    else
        anchors = prevCenters(activeIdx, :);
        [centerI, centerJ, centerSpeed, minSteps] = seeded_speed_min_batch(speed, anchors(:, 1), anchors(:, 2), params.deep_search_cells);
    end
    gridCenters = [centerI(:), centerJ(:)];
    [uniqueGrid, ~, gridInverse] = unique(gridCenters, 'rows', 'stable');
    refinedUnique = refine_speed_min_batch_kernel(speed, uLayer, vLayer, lon, lat, uniqueGrid, targetDegree, windowRadiusCells, minFiniteFraction, params);
    refinedRows = refinedUnique(gridInverse);
    huaCenters = [[refinedRows.center_i_refined]', [refinedRows.center_j_refined]'];
    [uniqueHua, ~, huaInverse] = unique(huaCenters, 'rows', 'stable');
    if strcmp(string(params.boundary_mode), "velocity_streamline_contour")
        checkUnique = streamline_gpu_batch_kernel(uLayer, vLayer, uniqueHua, radii, params);
    else
        checkUnique = hua_circle_batch_kernel(uLayer, vLayer, uniqueHua, params);
    end
    checks = checkUnique(huaInverse);

    nextActive = active;
    for localIdx = 1:numel(activeIdx)
        stateIdx = activeIdx(localIdx);
        refined = refinedRows(localIdx);
        check = checks(localIdx);
        passed = logical(check.hua_pass);
        if passed
            prevCenters(stateIdx, 1) = min(max(round(refined.center_i_refined), 1), size(speed, 2));
            prevCenters(stateIdx, 2) = min(max(round(refined.center_j_refined), 1), size(speed, 1));
        elseif stopAtFirstFailed
            nextActive(stateIdx) = false;
        end
        row = merge_row( ...
            check, refined, stateIdx, kk, centerI(localIdx), centerJ(localIdx), centerSpeed(localIdx), minSteps(localIdx), ...
            numel(activeIdx), size(uniqueGrid, 1), size(uniqueHua, 1));
        if isempty(rows)
            rows = row;
        else
            rows(end + 1) = row; %#ok<AGROW>
        end
    end
    active = nextActive;
end
end


function params = day_default_params(params)
if ~isfield(params, 'surface_search_cells') || isempty(params.surface_search_cells)
    params.surface_search_cells = 3;
end
if ~isfield(params, 'deep_search_cells') || isempty(params.deep_search_cells)
    params.deep_search_cells = 3;
end
if ~isfield(params, 'boundary_mode') || isempty(params.boundary_mode)
    params.boundary_mode = "circle_strict_original";
end
end


function row = merge_row(check, refined, stateIdx, depthIndex, centerI, centerJ, centerSpeed, minSteps, batchSize, uniqueRefineSize, uniqueHuaSize)
row = check;
row.state_index = stateIdx;
row.depth_index = depthIndex - 1;
row.speed_min_i_grid = centerI - 1;
row.speed_min_j_grid = centerJ - 1;
row.center_lon_grid = NaN;
row.center_lat_grid = NaN;
row.center_speed_grid_ms = centerSpeed;
row.local_min_steps = minSteps;
row.center_i_refined = refined.center_i_refined - 1;
row.center_j_refined = refined.center_j_refined - 1;
row.center_lon_refined = refined.center_lon_refined;
row.center_lat_refined = refined.center_lat_refined;
row.refined_speed_ms = refined.refined_speed_ms;
row.refined_offset_km = refined.refined_offset_km;
row.refined_ok = refined.refined_ok;
row.subgrid_fit_quality = refined.subgrid_fit_quality;
row.hua_center_i = refined.center_i_refined - 1;
row.hua_center_j = refined.center_j_refined - 1;
row.matlab_batch_size = batchSize;
row.matlab_refine_batch_size = batchSize;
row.matlab_unique_refine_batch_size = uniqueRefineSize;
row.matlab_unique_hua_batch_size = uniqueHuaSize;
row.matlab_refine_dedup_count = batchSize - uniqueRefineSize;
row.matlab_hua_dedup_count = batchSize - uniqueHuaSize;
row.matlab_day_batch = true;
end


function [outI, outJ, outSpeed, outSteps] = seeded_speed_min_batch(speed, seedI, seedJ, radiusCells)
n = numel(seedI);
outI = nan(n, 1);
outJ = nan(n, 1);
outSpeed = nan(n, 1);
outSteps = zeros(n, 1);
radius = max(0, round(radiusCells));
[ny, nx] = size(speed);
for ii = 1:n
    si = min(max(round(seedI(ii)), 1), nx);
    sj = min(max(round(seedJ(ii)), 1), ny);
    x0 = max(1, si - radius);
    x1 = min(nx, si + radius);
    y0 = max(1, sj - radius);
    y1 = min(ny, sj + radius);
    window = speed(y0:y1, x0:x1);
    [yy, xx] = ndgrid(y0:y1, x0:x1);
    mask = (xx - si).^2 + (yy - sj).^2 <= radius.^2 & isfinite(window);
    if ~any(mask(:))
        [oi, oj, os, steps] = iterative_speed_min(speed, si, sj);
    else
        [rr, cc] = find(mask);
        order = sortrows([rr, cc], [1, 2]);
        linear = sub2ind(size(window), order(:, 1), order(:, 2));
        values = window(linear);
        [~, pick] = min(values);
        oi = x0 + order(pick, 2) - 1;
        oj = y0 + order(pick, 1) - 1;
        [oi, oj, os, steps] = iterative_speed_min(speed, oi, oj);
    end
    outI(ii) = oi;
    outJ(ii) = oj;
    outSpeed(ii) = os;
    outSteps(ii) = steps;
end
end


function [ii, jj, val, steps] = iterative_speed_min(speed, startI, startJ)
[ny, nx] = size(speed);
ii = min(max(round(startI), 1), nx);
jj = min(max(round(startJ), 1), ny);
lastI = -1;
lastJ = -1;
steps = 0;
while (ii ~= lastI || jj ~= lastJ) && steps < 20
    lastI = ii;
    lastJ = jj;
    x0 = max(1, ii - 2);
    x1 = min(nx, ii + 2);
    y0 = max(1, jj - 2);
    y1 = min(ny, jj + 2);
    window = speed(y0:y1, x0:x1);
    finite = isfinite(window);
    if ~any(finite(:))
        break;
    end
    [rr, cc] = find(finite);
    order = sortrows([rr, cc], [1, 2]);
    linear = sub2ind(size(window), order(:, 1), order(:, 2));
    values = window(linear);
    [~, pick] = min(values);
    ii = x0 + order(pick, 2) - 1;
    jj = y0 + order(pick, 1) - 1;
    steps = steps + 1;
end
val = speed(jj, ii);
if isa(val, 'gpuArray')
    val = gather(val);
end
if ~isfinite(val)
    val = NaN;
end
end
