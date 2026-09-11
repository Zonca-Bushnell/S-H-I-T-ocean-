function rows = hua_circle_batch_kernel(u, v, centers, params)
%HUA_CIRCLE_BATCH_KERNEL Vectorized fixed-circle Hua checks for one layer.
%
% rows = hua_circle_batch_kernel(u, v, centers, params)
%
% Inputs
%   u, v    : ny-by-nx velocity arrays.
%   centers : n-by-2 matrix [center_i, center_j] in MATLAB 1-based columns/rows.
%   params  : struct with fields matching the Python DetectionParams names.
%
% This function is an acceleration kernel only. The Python reference path is
% still responsible for NetCDF IO, vertical continuation, parquet/csv/json
% output, and final pass-rate accounting.

if nargin < 4
    params = struct();
end
params = hua_default_params(params);
centers = double(centers);
n = size(centers, 1);
template = hua_empty_row();
rows = repmat(template, n, 1);

if n == 0
    return;
end

for ii = 1:n
    best = [];
    firstFail = [];
    for radius = params.start_radius_cells:params.max_radius_cells
        row = hua_circle_one(u, v, centers(ii, 1), centers(ii, 2), radius, params);
        if row.circle_passed
            best = row;
        else
            firstFail = row;
            break;
        end
    end
    if isempty(best)
        if isempty(firstFail)
            row = template;
        else
            row = firstFail;
        end
        row.hua_pass = false;
        row.accepted_radius_cells = 0;
    else
        row = best;
        row.hua_pass = true;
        row.accepted_radius_cells = row.radius_cells;
    end
    rows(ii) = row;
end
end


function params = hua_default_params(params)
defaults = struct( ...
    'start_radius_cells', 2, ...
    'max_radius_cells', 8, ...
    'speed_ratio_max', 3.0, ...
    'angle_jump_max_deg', 150.0, ...
    'tangent_tolerance_deg', 24.0, ...
    'symmetry_tolerance_deg', 120.0, ...
    'min_tangent_fraction', 0.55, ...
    'min_reversal_fraction', 0.55, ...
    'min_finite_fraction', 0.75, ...
    'direction_exception_extra', 2, ...
    'require_boundary_monotonic_rotation', false, ...
    'boundary_monotonic_exception_limit', 0);
names = fieldnames(defaults);
for ii = 1:numel(names)
    name = names{ii};
    if ~isfield(params, name) || isempty(params.(name))
        params.(name) = defaults.(name);
    end
end
end


function row = hua_empty_row()
row = struct( ...
    'circle_passed', false, ...
    'hua_pass', false, ...
    'radius_cells', NaN, ...
    'accepted_radius_cells', 0, ...
    'finite_fraction', 0, ...
    'speed_ratio', NaN, ...
    'max_angle_jump_deg', NaN, ...
    'tangent_fraction', 0, ...
    'symmetry_fraction', 0, ...
    'reversal_fraction', 0, ...
    'direction_exception_count', NaN, ...
    'circulation_sign', NaN, ...
    'dominant_failure_code', 0, ...
    'dominant_failure', "invalid_velocity", ...
    'first_hard_failure_code', 0, ...
    'first_hard_failure', "invalid_velocity", ...
    'hard_failure_order', "finite->velocity_ratio->angle_jump->boundary_monotonic->tangent->opposite_reversal");
end


function row = hua_circle_one(u, v, cx, cy, radius, params)
row = hua_empty_row();
row.radius_cells = radius;
failureCounts = zeros(1, 11);

offsets = circle_offsets(radius);
x = cx + offsets(:, 1)';
y = cy + offsets(:, 2)';
if max(abs(x - round(x))) < 1e-9 && max(abs(y - round(y))) < 1e-9
    [uu, vv] = sample_grid(u, v, round(x), round(y));
else
    [uu, vv] = hua_bilinear(u, v, x, y);
end
speed = hypot(uu, vv);
finite = isfinite(uu) & isfinite(vv) & speed > 0;
row.finite_fraction = mean(finite);
rotationFailed = false;
if row.finite_fraction < params.min_finite_fraction
    rotationFailed = true;
    failureCounts(1) = failureCounts(1) + sum(~finite);
end

angle = atan2(vv, uu);
maxRatio = 0;
maxAngle = 0;
positiveDiffs = 0;
negativeDiffs = 0;
for jj = 1:numel(x)
    mm = mod(jj, numel(x)) + 1;
    if ~(finite(jj) && finite(mm))
        rotationFailed = true;
        failureCounts(1) = failureCounts(1) + 1;
        continue;
    end
    ratio = speed(mm) / speed(jj);
    maxRatio = max([maxRatio, ratio, 1 / max(ratio, eps)]);
    if ratio > params.speed_ratio_max || ratio < 1 / params.speed_ratio_max
        rotationFailed = true;
        failureCounts(2) = failureCounts(2) + 1;
    end
    dtheta = angle_diff(angle(jj), angle(mm));
    maxAngle = max(maxAngle, abs(dtheta) * 180 / pi);
    if abs(dtheta) * 180 / pi > params.angle_jump_max_deg
        rotationFailed = true;
        failureCounts(3) = failureCounts(3) + 1;
    end
    if dtheta > 0
        positiveDiffs = positiveDiffs + 1;
    elseif dtheta < 0
        negativeDiffs = negativeDiffs + 1;
    end
end
row.speed_ratio = maxRatio;
row.max_angle_jump_deg = maxAngle;

maxExceptions = floor(radius / 5.0) + 1 + params.direction_exception_extra;
directionExceptions = min(positiveDiffs, negativeDiffs);
row.direction_exception_count = directionExceptions;
if params.require_boundary_monotonic_rotation
    monotonicLimit = params.boundary_monotonic_exception_limit;
else
    monotonicLimit = maxExceptions;
end
if directionExceptions > maxExceptions
    rotationFailed = true;
    failureCounts(5) = failureCounts(5) + directionExceptions - maxExceptions;
end
if params.require_boundary_monotonic_rotation && directionExceptions > monotonicLimit
    rotationFailed = true;
    failureCounts(10) = failureCounts(10) + directionExceptions - monotonicLimit;
end

theta = atan2(offsets(:, 2)', offsets(:, 1)');
tx = -sin(theta);
ty = cos(theta);
tangentCos = abs((uu .* tx + vv .* ty) ./ max(speed, eps));
tangentOk = finite & tangentCos >= cosd(params.tangent_tolerance_deg);
row.tangent_fraction = sum(tangentOk) / max(sum(finite), 1);
if row.tangent_fraction < params.min_tangent_fraction
    rotationFailed = true;
    failureCounts(8) = failureCounts(8) + max(1, round((params.min_tangent_fraction - row.tangent_fraction) * numel(x)));
end

half = floor(numel(x) / 2);
symmetryOk = 0;
symmetryTotal = 0;
reversalOk = 0;
reversalTotal = 0;
if half >= 2
    for jj = 1:half
        mm = mod(jj - 1 + half, numel(x)) + 1;
        if ~(finite(jj) && finite(mm))
            continue;
        end
        diffValue = abs(angle_diff(angle(jj), angle(mm)));
        symmetryTotal = symmetryTotal + 1;
        if abs(diffValue - pi) <= deg2rad(params.symmetry_tolerance_deg)
            symmetryOk = symmetryOk + 1;
        end
        reversalTotal = reversalTotal + 1;
        if uu(jj) * uu(mm) + vv(jj) * vv(mm) < 0
            reversalOk = reversalOk + 1;
        end
    end
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
    rotationFailed = true;
    failureCounts(9) = failureCounts(9) + max(1, round((params.min_reversal_fraction - row.reversal_fraction) * max(reversalTotal, 1)));
end

tangential = uu .* tx + vv .* ty;
if any(finite)
    row.circulation_sign = sign(median(tangential(finite), 'omitnan'));
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
[row.first_hard_failure_code, row.first_hard_failure] = first_hard_failure( ...
    row.finite_fraction < params.min_finite_fraction, ...
    failureCounts(2) > 0, ...
    failureCounts(3) > 0, ...
    params.require_boundary_monotonic_rotation && directionExceptions > monotonicLimit, ...
    row.tangent_fraction < params.min_tangent_fraction, ...
    row.reversal_fraction < params.min_reversal_fraction);
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


function offsets = circle_offsets(radius)
n = max(16, round(8 * radius));
theta = linspace(-pi / 2, 3*pi / 2, n + 1);
theta(end) = [];
points = [];
last = [NaN, NaN];
for ii = 1:numel(theta)
    point = [round(radius * cos(theta(ii))), round(radius * sin(theta(ii)))];
    if isempty(points) || any(point ~= last)
        points = [points; point]; %#ok<AGROW>
        last = point;
    end
end
if size(points, 1) > 1 && all(points(1, :) == points(end, :))
    points(end, :) = [];
end
offsets = points;
end


function d = angle_diff(a, b)
d = mod(a - b + pi, 2*pi) - pi;
end


function [uu, vv] = sample_grid(u, v, x, y)
[ny, nx] = size(u);
valid = x >= 1 & y >= 1 & x <= nx & y <= ny;
uu = NaN(size(x));
vv = NaN(size(x));
if ~any(valid)
    return;
end
idx = find(valid);
ind = sub2ind([ny, nx], y(idx), x(idx));
uu(idx) = u(ind);
vv(idx) = v(ind);
end


function [uu, vv] = hua_bilinear(u, v, x, y)
[ny, nx] = size(u);
x0 = floor(x);
y0 = floor(y);
valid = x0 >= 1 & y0 >= 1 & x0 < nx & y0 < ny;
uu = NaN(size(x));
vv = NaN(size(x));
if ~any(valid)
    return;
end
idx = find(valid);
wx = x(idx) - x0(idx);
wy = y(idx) - y0(idx);
i00 = sub2ind([ny, nx], y0(idx), x0(idx));
i10 = sub2ind([ny, nx], y0(idx), x0(idx) + 1);
i01 = sub2ind([ny, nx], y0(idx) + 1, x0(idx));
i11 = sub2ind([ny, nx], y0(idx) + 1, x0(idx) + 1);
valsU = [u(i00); u(i10); u(i01); u(i11)];
valsV = [v(i00); v(i10); v(i01); v(i11)];
finite = all(isfinite(valsU), 1) & all(isfinite(valsV), 1);
w00 = (1 - wx) .* (1 - wy);
w10 = wx .* (1 - wy);
w01 = (1 - wx) .* wy;
w11 = wx .* wy;
uu(idx(finite)) = w00(finite).*u(i00(finite)) + w10(finite).*u(i10(finite)) + w01(finite).*u(i01(finite)) + w11(finite).*u(i11(finite));
vv(idx(finite)) = w00(finite).*v(i00(finite)) + w10(finite).*v(i10(finite)) + w01(finite).*v(i01(finite)) + w11(finite).*v(i11(finite));
end
