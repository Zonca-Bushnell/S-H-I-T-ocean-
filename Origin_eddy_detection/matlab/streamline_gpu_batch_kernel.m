function rows = streamline_gpu_batch_kernel(u, v, centers, radii, params)
%STREAMLINE_GPU_BATCH_KERNEL Batch fixed-step closed-streamline Hua checks.
%
% rows = streamline_gpu_batch_kernel(u, v, centers, radii, params)
%
% This is the GPU-oriented companion to velocity_streamline_boundary_kernel.
% It keeps the Python/reference scientific semantics but changes streamline
% tracing to a fixed-step, multi-start batch:
%
%   center x radius x start_angle x direction x max_step
%
% Completed/failed paths are masked and frozen instead of breaking out of the
% step loop. That makes the integration stage suitable for GPU execution while
% preserving the first-closure path length used by the reference kernel.

if nargin < 5
    params = struct();
end
params = streamline_default_params(params);
centers = double(centers);
radii = double(radii(:)');
n = size(centers, 1);
template = streamline_empty_row();
rows = repmat(template, n, 1);

if isfield(params, 'use_gpu') && params.use_gpu
    try
        if isa(u, 'gpuArray')
            uWork = u;
        else
            uWork = gpuArray(double(u));
        end
        if isa(v, 'gpuArray')
            vWork = v;
        else
            vWork = gpuArray(double(v));
        end
    catch
        uWork = double(u);
        vWork = double(v);
        params.use_gpu = false;
    end
else
    uWork = double(u);
    vWork = double(v);
    params.use_gpu = false;
end

bestRows = repmat(template, n, 1);
firstFailRows = repmat(template, n, 1);
hasBest = false(n, 1);
hasFirstFail = false(n, 1);
active = true(n, 1);

for radius = radii
    activeIdx = find(active);
    if isempty(activeIdx)
        break;
    end
    candidates = best_streamlines_for_radius_batch_all(uWork, vWork, centers(activeIdx, :), radius, params);
    for localIdx = 1:numel(activeIdx)
        ii = activeIdx(localIdx);
        candidate = candidates(localIdx);
        if ~candidate.streamline_closed
            row = template;
            row.radius_cells = radius;
            row.dominant_failure = "no_closed_streamline";
            row.dominant_failure_code = 10;
            row.first_hard_failure = "no_closed_streamline";
            row.first_hard_failure_code = 10;
            firstFailRows(ii) = row;
            hasFirstFail(ii) = true;
            active(ii) = false;
            continue;
        end
        row = score_streamline_contour(uWork, vWork, candidate.points, centers(ii, 1), centers(ii, 2), candidate.radius_cells, params);
        if row.circle_passed
            bestRows(ii) = row;
            hasBest(ii) = true;
        else
            firstFailRows(ii) = row;
            hasFirstFail(ii) = true;
            active(ii) = false;
        end
    end
end

for ii = 1:n
    if hasBest(ii)
        row = bestRows(ii);
        row.hua_pass = true;
        row.accepted_radius_cells = row.radius_cells;
    elseif hasFirstFail(ii)
        row = firstFailRows(ii);
        row.hua_pass = false;
        row.accepted_radius_cells = 0;
    else
        row = template;
        row.hua_pass = false;
        row.accepted_radius_cells = 0;
    end
    rows(ii) = row;
end
end


function candidates = best_streamlines_for_radius_batch_all(u, v, centers, radius, params)
template = streamline_empty_row();
nCenters = size(centers, 1);
candidates = repmat(template, nCenters, 1);
for cc = 1:nCenters
    candidates(cc).points = [];
end
if nCenters == 0
    return;
end

nAngles = max(1, round(params.streamline_start_angles));
angles = linspace(0, 2*pi, nAngles + 1);
angles(end) = [];
directions = [1, -1];
[centerGrid, angleGrid, directionGrid] = ndgrid(1:nCenters, angles, directions);
centerVec = centerGrid(:)';
angleVec = angleGrid(:)';
directionVec = directionGrid(:)';
nPaths = numel(centerVec);
maxSteps = round(params.streamline_max_steps);
minPoints = round(params.streamline_min_points);
step = params.streamline_step_cells;
cxHost = centers(centerVec, 1)';
cyHost = centers(centerVec, 2)';

if isa(u, 'gpuArray')
    cx = gpuArray(double(cxHost));
    cy = gpuArray(double(cyHost));
    xNow = gpuArray(double(cxHost + radius * cos(angleVec)));
    yNow = gpuArray(double(cyHost + radius * sin(angleVec)));
    dirNow = gpuArray(double(directionVec));
    pointsX = gpuArray(nan(maxSteps + 1, nPaths));
    pointsY = gpuArray(nan(maxSteps + 1, nPaths));
    closed = gpuArray(false(1, nPaths));
    failed = gpuArray(false(1, nPaths));
    closureAt = gpuArray(zeros(1, nPaths, 'uint16'));
    prevAngle = atan2(yNow - cy, xNow - cx);
    windingNow = gpuArray(zeros(1, nPaths));
else
    cx = double(cxHost);
    cy = double(cyHost);
    xNow = double(cxHost + radius * cos(angleVec));
    yNow = double(cyHost + radius * sin(angleVec));
    dirNow = double(directionVec);
    pointsX = NaN(maxSteps + 1, nPaths);
    pointsY = NaN(maxSteps + 1, nPaths);
    closed = false(1, nPaths);
    failed = false(1, nPaths);
    closureAt = zeros(1, nPaths, 'uint16');
    prevAngle = atan2(yNow - cy, xNow - cx);
    windingNow = zeros(1, nPaths);
end
pointsX(1, :) = xNow;
pointsY(1, :) = yNow;

for kk = 1:maxSteps
    activePath = ~(closed | failed);
    if ~any(gather(activePath))
        break;
    end
    [u1, v1] = bilinear_uv_vectorized(u, v, xNow, yNow);
    s1 = hypot(u1, v1);
    bad1 = activePath & (~isfinite(s1) | s1 <= 0);
    failed = failed | bad1;
    activePath = activePath & ~bad1;

    midX = xNow;
    midY = yNow;
    midX(activePath) = xNow(activePath) + 0.5 * step .* dirNow(activePath) .* u1(activePath) ./ s1(activePath);
    midY(activePath) = yNow(activePath) + 0.5 * step .* dirNow(activePath) .* v1(activePath) ./ s1(activePath);
    [u2, v2] = bilinear_uv_vectorized(u, v, midX, midY);
    s2 = hypot(u2, v2);
    nextX = xNow;
    nextY = yNow;
    nextX(activePath) = xNow(activePath) + step .* dirNow(activePath) .* u1(activePath) ./ s1(activePath);
    nextY(activePath) = yNow(activePath) + step .* dirNow(activePath) .* v1(activePath) ./ s1(activePath);
    useMid = activePath & isfinite(s2) & s2 > 0;
    nextX(useMid) = xNow(useMid) + step .* dirNow(useMid) .* u2(useMid) ./ s2(useMid);
    nextY(useMid) = yNow(useMid) + step .* dirNow(useMid) .* v2(useMid) ./ s2(useMid);
    [ny, nx] = size(u);
    outOfBounds = activePath & (nextX < 1 | nextY < 1 | nextX > nx | nextY > ny);
    failed = failed | outOfBounds;
    nextX(outOfBounds) = xNow(outOfBounds);
    nextY(outOfBounds) = yNow(outOfBounds);
    angleActive = activePath & ~outOfBounds;
    newAngle = atan2(nextY - cy, nextX - cx);
    windingNow(angleActive) = windingNow(angleActive) + angle_diff(newAngle(angleActive), prevAngle(angleActive));
    prevAngle(angleActive) = newAngle(angleActive);
    xNow = nextX;
    yNow = nextY;
    pointsX(kk + 1, :) = xNow;
    pointsY(kk + 1, :) = yNow;

    if kk + 1 >= minPoints
        closure = hypot(pointsX(kk + 1, :) - pointsX(1, :), pointsY(kk + 1, :) - pointsY(1, :));
        newlyClosed = activePath & abs(windingNow) >= 2*pi*params.streamline_min_winding_turns & closure <= params.streamline_closure_tolerance_cells;
        closed = closed | newlyClosed;
        closureAt(newlyClosed) = uint16(kk + 1);
    end
end

pointsXg = gather_if_needed(pointsX);
pointsYg = gather_if_needed(pointsY);
closedg = gather_if_needed(closed);
failedg = gather_if_needed(failed);
closureAtg = gather_if_needed(closureAt);
windingFinalg = gather_if_needed(windingNow);
for cc = 1:nCenters
    pathIdx = find(centerVec == cc & closedg);
    if isempty(pathIdx)
        continue;
    end
    bestScore = Inf;
    bestPath = [];
    bestMeta = [];
    cx0 = centers(cc, 1);
    cy0 = centers(cc, 2);
    for pp = pathIdx
        count = double(closureAtg(pp));
        if count < minPoints
            continue;
        end
        pts = [pointsXg(1:count, pp), pointsYg(1:count, pp)];
        windingTurns = windingFinalg(pp) / (2*pi);
        closureErr = hypot(pts(end, 1) - pts(1, 1), pts(end, 2) - pts(1, 2));
        radial = hypot(pts(:, 1) - cx0, pts(:, 2) - cy0);
        radialCv = std(radial, 'omitnan') / max(mean(radial, 'omitnan'), 1.0e-6);
        score = closureErr + 2 * abs(1 - min(abs(windingTurns), 1.5)) + radialCv;
        if failedg(pp)
            score = score + 5;
        end
        if score < bestScore
            bestScore = score;
            bestPath = pts;
            bestMeta = [abs(windingTurns), closureErr, count];
        end
    end
    if isempty(bestPath)
        continue;
    end
    candidates(cc).streamline_closed = true;
    candidates(cc).radius_cells = radius;
    candidates(cc).accepted_radius_cells = radius;
    candidates(cc).streamline_score = bestScore;
    candidates(cc).streamline_winding_turns = bestMeta(1);
    candidates(cc).streamline_closure_error_cells = bestMeta(2);
    candidates(cc).streamline_points = bestMeta(3);
    candidates(cc).points = bestPath;
end
end


function candidate = best_streamline_for_radius_batch(u, v, cx, cy, radius, params)
candidate = streamline_empty_row();
candidate.points = [];

nAngles = max(1, round(params.streamline_start_angles));
angles = linspace(0, 2*pi, nAngles + 1);
angles(end) = [];
directions = [1, -1];
[angleGrid, directionGrid] = ndgrid(angles, directions);
angleVec = angleGrid(:);
directionVec = directionGrid(:);
nPaths = numel(angleVec);
maxSteps = round(params.streamline_max_steps);
minPoints = round(params.streamline_min_points);
step = params.streamline_step_cells;

if isa(u, 'gpuArray')
    xNow = gpuArray(double((cx + radius * cos(angleVec))'));
    yNow = gpuArray(double((cy + radius * sin(angleVec))'));
    dirNow = gpuArray(double(directionVec'));
    pointsX = gpuArray(nan(maxSteps + 1, nPaths));
    pointsY = gpuArray(nan(maxSteps + 1, nPaths));
    closed = gpuArray(false(1, nPaths));
    failed = gpuArray(false(1, nPaths));
    closureAt = gpuArray(zeros(1, nPaths, 'uint16'));
    prevAngle = atan2(yNow - cy, xNow - cx);
    windingNow = gpuArray(zeros(1, nPaths));
else
    xNow = double(cx + radius * cos(angleVec))';
    yNow = double(cy + radius * sin(angleVec))';
    dirNow = double(directionVec)';
    pointsX = NaN(maxSteps + 1, nPaths);
    pointsY = NaN(maxSteps + 1, nPaths);
    closed = false(1, nPaths);
    failed = false(1, nPaths);
    closureAt = zeros(1, nPaths, 'uint16');
    prevAngle = atan2(yNow - cy, xNow - cx);
    windingNow = zeros(1, nPaths);
end
pointsX(1, :) = xNow;
pointsY(1, :) = yNow;

for kk = 1:maxSteps
    active = ~(closed | failed);
    if ~any(gather(active))
        break;
    end
    [u1, v1] = bilinear_uv_vectorized(u, v, xNow, yNow);
    s1 = hypot(u1, v1);
    bad1 = active & (~isfinite(s1) | s1 <= 0);
    failed = failed | bad1;
    active = active & ~bad1;

    midX = xNow;
    midY = yNow;
    midX(active) = xNow(active) + 0.5 * step .* dirNow(active) .* u1(active) ./ s1(active);
    midY(active) = yNow(active) + 0.5 * step .* dirNow(active) .* v1(active) ./ s1(active);
    [u2, v2] = bilinear_uv_vectorized(u, v, midX, midY);
    s2 = hypot(u2, v2);
    nextX = xNow;
    nextY = yNow;
    nextX(active) = xNow(active) + step .* dirNow(active) .* u1(active) ./ s1(active);
    nextY(active) = yNow(active) + step .* dirNow(active) .* v1(active) ./ s1(active);
    useMid = active & isfinite(s2) & s2 > 0;
    nextX(useMid) = xNow(useMid) + step .* dirNow(useMid) .* u2(useMid) ./ s2(useMid);
    nextY(useMid) = yNow(useMid) + step .* dirNow(useMid) .* v2(useMid) ./ s2(useMid);
    [ny, nx] = size(u);
    outOfBounds = active & (nextX < 1 | nextY < 1 | nextX > nx | nextY > ny);
    failed = failed | outOfBounds;
    nextX(outOfBounds) = xNow(outOfBounds);
    nextY(outOfBounds) = yNow(outOfBounds);
    angleActive = active & ~outOfBounds;
    newAngle = atan2(nextY - cy, nextX - cx);
    windingNow(angleActive) = windingNow(angleActive) + angle_diff(newAngle(angleActive), prevAngle(angleActive));
    prevAngle(angleActive) = newAngle(angleActive);
    xNow = nextX;
    yNow = nextY;
    pointsX(kk + 1, :) = xNow;
    pointsY(kk + 1, :) = yNow;

    if kk + 1 >= minPoints
        sx = pointsX(1, :);
        sy = pointsY(1, :);
        closure = hypot(pointsX(kk + 1, :) - sx, pointsY(kk + 1, :) - sy);
        newlyClosed = active & abs(windingNow) >= 2*pi*params.streamline_min_winding_turns & closure <= params.streamline_closure_tolerance_cells;
        closed = closed | newlyClosed;
        closureAt(newlyClosed) = uint16(kk + 1);
    end
end

pointsXg = gather_if_needed(pointsX);
pointsYg = gather_if_needed(pointsY);
closedg = gather_if_needed(closed);
failedg = gather_if_needed(failed);
closureAtg = gather_if_needed(closureAt);
windingFinalg = gather_if_needed(windingNow);
if ~any(closedg)
    return;
end

bestScore = Inf;
bestPath = [];
bestMeta = [];
for pp = 1:nPaths
    if ~closedg(pp)
        continue;
    end
    count = double(closureAtg(pp));
    if count < minPoints
        continue;
    end
    pts = [pointsXg(1:count, pp), pointsYg(1:count, pp)];
    windingTurns = windingFinalg(pp) / (2*pi);
    closureErr = hypot(pts(end, 1) - pts(1, 1), pts(end, 2) - pts(1, 2));
    radial = hypot(pts(:, 1) - cx, pts(:, 2) - cy);
    radialCv = std(radial, 'omitnan') / max(mean(radial, 'omitnan'), 1.0e-6);
    score = closureErr + 2 * abs(1 - min(abs(windingTurns), 1.5)) + radialCv;
    if failedg(pp)
        score = score + 5;
    end
    if score < bestScore
        bestScore = score;
        bestPath = pts;
        bestMeta = [abs(windingTurns), closureErr, count];
    end
end

if isempty(bestPath)
    return;
end
candidate.streamline_closed = true;
candidate.radius_cells = radius;
candidate.accepted_radius_cells = radius;
candidate.streamline_score = bestScore;
candidate.streamline_winding_turns = bestMeta(1);
candidate.streamline_closure_error_cells = bestMeta(2);
candidate.streamline_points = bestMeta(3);
candidate.points = bestPath;
end


function total = angle_accumulated(x, y, cx, cy)
ang = atan2(y - cy, x - cx);
d = diff(ang, 1, 1);
d = mod(d + pi, 2*pi) - pi;
total = sum(d, 1);
end


function [uu, vv] = bilinear_uv_vectorized(u, v, x, y)
uu = interp2(u, x, y, 'linear', NaN);
vv = interp2(v, x, y, 'linear', NaN);
end


function value = gather_if_needed(value)
if isa(value, 'gpuArray')
    value = gather(value);
end
end


function params = streamline_default_params(params)
defaults = struct( ...
    'streamline_step_cells', 0.5, ...
    'streamline_max_steps', 180, ...
    'streamline_start_angles', 4, ...
    'streamline_closure_tolerance_cells', 1.75, ...
    'streamline_min_winding_turns', 0.75, ...
    'streamline_min_points', 16, ...
    'streamline_direction_exception_fraction', 0.10, ...
    'speed_ratio_max', 3.0, ...
    'angle_jump_max_deg', 150.0, ...
    'tangent_tolerance_deg', 24.0, ...
    'min_tangent_fraction', 0.55, ...
    'symmetry_tolerance_deg', 120.0, ...
    'min_reversal_fraction', 0.55, ...
    'min_finite_fraction', 0.75, ...
    'use_gpu', false);
names = fieldnames(defaults);
for ii = 1:numel(names)
    name = names{ii};
    if ~isfield(params, name) || isempty(params.(name))
        params.(name) = defaults.(name);
    end
end
end


function row = streamline_empty_row()
row = struct( ...
    'circle_passed', false, ...
    'hua_pass', false, ...
    'radius_cells', NaN, ...
    'accepted_radius_cells', 0, ...
    'finite_fraction', 0, ...
    'speed_ratio', NaN, ...
    'max_angle_jump_deg', NaN, ...
    'tangent_fraction', 0, ...
    'tangent_fraction_24deg', 0, ...
    'tangent_fraction_30deg', 0, ...
    'tangent_fraction_36deg', 0, ...
    'tangent_fraction_45deg', 0, ...
    'symmetry_fraction', 0, ...
    'reversal_fraction', 0, ...
    'streamline_closed', false, ...
    'streamline_points', 0, ...
    'streamline_closure_error_cells', NaN, ...
    'streamline_winding_turns', NaN, ...
    'streamline_direction_exception_fraction', NaN, ...
    'streamline_score', Inf, ...
    'circulation_sign', NaN, ...
    'dominant_failure_code', 10, ...
    'dominant_failure', "no_closed_streamline", ...
    'first_hard_failure_code', 10, ...
    'first_hard_failure', "no_closed_streamline", ...
    'hard_failure_order', "no_closed_streamline->finite->velocity_ratio->angle_jump->boundary_monotonic->tangent->opposite_reversal");
end


function row = score_streamline_contour(u, v, points, cx, cy, radius, params)
row = streamline_empty_row();
row.circle_passed = false;
row.radius_cells = radius;
row.accepted_radius_cells = radius;
failureCounts = zeros(1, 11);
row.streamline_closed = true;
row.streamline_points = size(points, 1);
row.streamline_closure_error_cells = hypot(points(end, 1) - points(1, 1), points(end, 2) - points(1, 2));
ang = unwrap(atan2(points(:, 2) - cy, points(:, 1) - cx));
row.streamline_winding_turns = abs((ang(end) - ang(1)) / (2*pi));

[uu, vv] = bilinear_uv_vectorized(u, v, points(:, 1), points(:, 2));
speed = hypot(uu, vv);
finite = isfinite(uu) & isfinite(vv) & speed > 0;
row.finite_fraction = scalarize(mean(finite));
rotationFailed = false;
if row.finite_fraction < params.min_finite_fraction
    rotationFailed = true;
    failureCounts(1) = failureCounts(1) + scalarize(sum(~finite));
end

xSmooth = smooth_wrap(points(:, 1), 1.0);
ySmooth = smooth_wrap(points(:, 2), 1.0);
dxy = centered_gradient([xSmooth, ySmooth]);
tmag = hypot(dxy(:, 1), dxy(:, 2));
tx = dxy(:, 1) ./ max(tmag, eps);
ty = dxy(:, 2) ./ max(tmag, eps);
cosang = (uu .* tx + vv .* ty) ./ max(speed, eps);
cosang = max(-1, min(1, cosang));
aligned24 = acosd(abs(cosang(finite))) <= 24;
aligned30 = acosd(abs(cosang(finite))) <= 30;
aligned36 = acosd(abs(cosang(finite))) <= 36;
aligned45 = acosd(abs(cosang(finite))) <= 45;
row.tangent_fraction_24deg = scalarize(mean(aligned24));
row.tangent_fraction_30deg = scalarize(mean(aligned30));
row.tangent_fraction_36deg = scalarize(mean(aligned36));
row.tangent_fraction_45deg = scalarize(mean(aligned45));
tolName = sprintf('tangent_fraction_%ddeg', round(params.tangent_tolerance_deg));
if isfield(row, tolName)
    row.tangent_fraction = row.(tolName);
else
    row.tangent_fraction = scalarize(mean(acosd(abs(cosang(finite))) <= params.tangent_tolerance_deg));
end
if row.tangent_fraction < params.min_tangent_fraction
    rotationFailed = true;
    failureCounts(8) = failureCounts(8) + max(1, round((params.min_tangent_fraction - row.tangent_fraction) * size(points, 1)));
end

polarDelta = diff(ang);
positiveDiffs = sum(polarDelta > 1.0e-9);
negativeDiffs = sum(polarDelta < -1.0e-9);
directionExceptions = min(positiveDiffs, negativeDiffs);
directionTotal = max(1, positiveDiffs + negativeDiffs);
row.streamline_direction_exception_fraction = scalarize(directionExceptions / directionTotal);
if row.streamline_direction_exception_fraction > params.streamline_direction_exception_fraction
    rotationFailed = true;
    failureCounts(10) = failureCounts(10) + max(1, round((row.streamline_direction_exception_fraction - params.streamline_direction_exception_fraction) * size(points, 1)));
end

velocityAngle = atan2(vv, uu);
nextIdx = [2:size(points, 1), 1];
pairFinite = finite & finite(nextIdx);
failureCounts(1) = failureCounts(1) + scalarize(sum(~pairFinite));
ratio = speed(nextIdx) ./ speed;
ratioSym = max(ratio, 1 ./ max(ratio, eps));
ratioSym(~pairFinite) = 0;
if any(gather_if_needed(pairFinite))
    maxRatio = scalarize(max(ratioSym(pairFinite)));
else
    maxRatio = 0;
end
ratioFail = pairFinite & (ratio > params.speed_ratio_max | ratio < 1 / params.speed_ratio_max);
failureCounts(2) = failureCounts(2) + scalarize(sum(ratioFail));
dtheta = angle_diff(velocityAngle, velocityAngle(nextIdx));
angleDeg = abs(dtheta) * 180 / pi;
angleDeg(~pairFinite) = 0;
if any(gather_if_needed(pairFinite))
    maxAngle = scalarize(max(angleDeg(pairFinite)));
else
    maxAngle = 0;
end
failureCounts(3) = failureCounts(3) + scalarize(sum(pairFinite & angleDeg > params.angle_jump_max_deg));
row.speed_ratio = maxRatio;
row.max_angle_jump_deg = maxAngle;

half = floor(size(points, 1) / 2);
symmetryOk = 0;
symmetryTotal = 0;
reversalOk = 0;
reversalTotal = 0;
if half >= 2
    jj = 1:half;
    mm = mod(jj - 1 + half, size(points, 1)) + 1;
    oppositeFinite = finite(jj) & finite(mm);
    diffValue = abs(angle_diff(velocityAngle(jj), velocityAngle(mm)));
    symmetryTotal = scalarize(sum(oppositeFinite));
    symmetryOk = scalarize(sum(oppositeFinite & abs(diffValue - pi) <= deg2rad(params.symmetry_tolerance_deg)));
    reversalTotal = symmetryTotal;
    reversalOk = scalarize(sum(oppositeFinite & (uu(jj) .* uu(mm) + vv(jj) .* vv(mm) < 0)));
end
if symmetryTotal > 0
    row.symmetry_fraction = symmetryOk / symmetryTotal;
    if symmetryOk < symmetryTotal
        failureCounts(7) = failureCounts(7) + symmetryTotal - symmetryOk;
    end
end
if reversalTotal > 0
    row.reversal_fraction = reversalOk / reversalTotal;
end
if row.reversal_fraction < params.min_reversal_fraction
    failureCounts(9) = failureCounts(9) + max(1, round((params.min_reversal_fraction - row.reversal_fraction) * max(reversalTotal, 1)));
end

if failureCounts(2) > 0 || failureCounts(3) > 0 || failureCounts(9) > 0
    rotationFailed = true;
end
tangential = uu .* tx + vv .* ty;
if any(finite)
    row.circulation_sign = scalarize(sign(median(tangential(finite), 'omitnan')));
else
    row.circulation_sign = NaN;
end
[dominantCount, dominantIdx] = max(failureCounts);
if dominantCount > 0
    row.dominant_failure_code = dominantIdx - 1;
    row.dominant_failure = failure_label(dominantIdx - 1);
else
    row.dominant_failure_code = -1;
    row.dominant_failure = "none";
end
row.circle_passed = ~rotationFailed;
row.hua_pass = row.circle_passed;
[row.first_hard_failure_code, row.first_hard_failure] = first_hard_failure( ...
    row.finite_fraction < params.min_finite_fraction, ...
    failureCounts(2) > 0, ...
    failureCounts(3) > 0, ...
    row.streamline_direction_exception_fraction > params.streamline_direction_exception_fraction, ...
    row.tangent_fraction < params.min_tangent_fraction, ...
    row.reversal_fraction < params.min_reversal_fraction);
end


function value = scalarize(value)
if isa(value, 'gpuArray')
    value = gather(value);
end
value = double(value);
if isempty(value)
    value = NaN;
else
    value = value(1);
end
end


function out = smooth_wrap(values, sigma)
values = values(:);
radius = max(1, ceil(4 * sigma));
x = (-radius:radius)';
kernel = exp(-0.5 * (x / sigma).^2);
kernel = kernel / sum(kernel);
extended = [values(end-radius+1:end); values; values(1:radius)];
smoothed = conv(extended, kernel, 'same');
out = smoothed(radius+1:radius+numel(values));
end


function grad = centered_gradient(points)
grad = zeros(size(points));
n = size(points, 1);
if n >= 3
    grad(2:end-1, :) = (points(3:end, :) - points(1:end-2, :)) / 2;
    grad(1, :) = points(2, :) - points(1, :);
    grad(end, :) = points(end, :) - points(end-1, :);
else
    grad(:) = NaN;
end
end


function [code, label] = first_hard_failure(finiteFail, velocityFail, angleFail, boundaryFail, tangentFail, reversalFail)
if finiteFail
    code = 0; label = "invalid_velocity";
elseif velocityFail
    code = 1; label = "velocity_ratio";
elseif angleFail
    code = 2; label = "angle_jump";
elseif boundaryFail
    code = 9; label = "boundary_monotonic_rotation";
elseif tangentFail
    code = 7; label = "tangent_alignment";
elseif reversalFail
    code = 8; label = "opposite_reversal";
else
    code = -1; label = "none";
end
end


function label = failure_label(code)
labels = ["invalid_velocity","velocity_ratio","angle_jump","rotation_direction","too_many_direction_exceptions","dead_zone","symmetry","tangent_alignment","opposite_reversal","boundary_monotonic_rotation","no_closed_streamline"];
if code >= 0 && code < numel(labels)
    label = labels(code + 1);
else
    label = "none";
end
end


function d = angle_diff(a, b)
d = mod(a - b + pi, 2*pi) - pi;
end


function [uu, vv] = bilinear_uv_many(u, v, x, y)
uu = NaN(size(x));
vv = NaN(size(x));
for ii = 1:numel(x)
    [uu(ii), vv(ii)] = bilinear_uv(u, v, x(ii), y(ii));
end
end


function [uu, vv] = bilinear_uv(u, v, x, y)
[ny, nx] = size(u);
x0 = floor(x);
y0 = floor(y);
if x0 < 1 || y0 < 1 || x0 >= nx || y0 >= ny
    uu = NaN;
    vv = NaN;
    return;
end
wx = x - x0;
wy = y - y0;
u00 = u(y0, x0); u10 = u(y0, x0 + 1); u01 = u(y0 + 1, x0); u11 = u(y0 + 1, x0 + 1);
v00 = v(y0, x0); v10 = v(y0, x0 + 1); v01 = v(y0 + 1, x0); v11 = v(y0 + 1, x0 + 1);
if any(~isfinite([u00, u10, u01, u11, v00, v10, v01, v11]))
    uu = NaN;
    vv = NaN;
    return;
end
w00 = (1 - wx) * (1 - wy);
w10 = wx * (1 - wy);
w01 = (1 - wx) * wy;
w11 = wx * wy;
uu = w00*u00 + w10*u10 + w01*u01 + w11*u11;
vv = w00*v00 + w10*v10 + w01*v01 + w11*v11;
end
