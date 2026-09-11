function rows = velocity_streamline_boundary_kernel(u, v, centers, radii, params)
%VELOCITY_STREAMLINE_BOUNDARY_KERNEL Closed velocity-streamline boundary checks.
%
% rows = velocity_streamline_boundary_kernel(u, v, centers, radii, params)
%
% Inputs
%   u, v    : ny-by-nx velocity arrays for one depth layer.
%   centers : n-by-2 matrix [center_i, center_j] in MATLAB 1-based columns/rows.
%   radii   : vector of candidate start radii in grid cells.
%   params  : struct with fields matching the Python streamline parameters.
%
% The kernel evaluates closed local streamlines and returns continuous
% diagnostics used by Python for threshold sweeps. It is designed for
% matrix-oriented MATLAB/GPU optimization while preserving the Python
% reference semantics.

if nargin < 5
    params = struct();
end
params = streamline_default_params(params);
centers = double(centers);
radii = double(radii(:)');
n = size(centers, 1);
template = streamline_empty_row();
rows = repmat(template, n, 1);

for ii = 1:n
    best = [];
    firstFail = [];
    for radius = radii
        candidate = best_streamline_for_radius(u, v, centers(ii, 1), centers(ii, 2), radius, params);
        if ~candidate.streamline_closed
            row = template;
            row.radius_cells = radius;
            row.dominant_failure = "no_closed_streamline";
            row.dominant_failure_code = 10;
            row.first_hard_failure = "no_closed_streamline";
            row.first_hard_failure_code = 10;
            firstFail = row;
            break;
        end
        row = score_streamline_contour(u, v, candidate.points, centers(ii, 1), centers(ii, 2), candidate.radius_cells, params);
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
    'min_finite_fraction', 0.75);
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


function best = best_streamline_for_radius(u, v, cx, cy, radius, params)
best = streamline_empty_row();
best.points = [];
angles = linspace(0, 2*pi, params.streamline_start_angles + 1);
angles(end) = [];
for aa = 1:numel(angles)
    start = [cx + radius*cos(angles(aa)), cy + radius*sin(angles(aa))];
    for direction = [1, -1]
        candidate = trace_streamline(u, v, cx, cy, start, radius, direction, params);
        if candidate.streamline_closed && candidate.streamline_score < best.streamline_score
            best = candidate;
        end
    end
end
end


function row = trace_streamline(u, v, cx, cy, startPoint, radius, direction, params)
row = streamline_empty_row();
row.radius_cells = radius;
points = NaN(params.streamline_max_steps + 1, 2);
points(1, :) = startPoint;
step = params.streamline_step_cells;
failedFinite = false;
prevAngle = atan2(startPoint(2) - cy, startPoint(1) - cx);
winding = 0;
for kk = 1:params.streamline_max_steps
    p = points(kk, :);
    [u1, v1] = bilinear_uv(u, v, p(1), p(2));
    s1 = hypot(u1, v1);
    if ~isfinite(s1) || s1 <= 0
        failedFinite = true;
        break;
    end
    mid = p + 0.5 * step * direction * [u1, v1] / s1;
    [u2, v2] = bilinear_uv(u, v, mid(1), mid(2));
    s2 = hypot(u2, v2);
    if isfinite(s2) && s2 > 0
        stepVelocity = [u2, v2] / s2;
    else
        stepVelocity = [u1, v1] / s1;
    end
    nextPoint = p + step * direction * stepVelocity;
    if nextPoint(1) < 1 || nextPoint(2) < 1 || nextPoint(1) > size(u, 2) || nextPoint(2) > size(u, 1)
        failedFinite = true;
        break;
    end
    newAngle = atan2(nextPoint(2) - cy, nextPoint(1) - cx);
    winding = winding + angle_diff(newAngle, prevAngle);
    prevAngle = newAngle;
    points(kk + 1, :) = nextPoint;
    if kk + 1 >= params.streamline_min_points
        partial = points(1:kk + 1, :);
        partialClosure = hypot(partial(end, 1) - partial(1, 1), partial(end, 2) - partial(1, 2));
        if abs(winding) >= 2*pi*params.streamline_min_winding_turns && partialClosure <= params.streamline_closure_tolerance_cells
            points(kk + 2:end, :) = NaN;
            break;
        end
    end
end
valid = all(isfinite(points), 2);
points = points(valid, :);
row.streamline_points = size(points, 1);
if row.streamline_points < params.streamline_min_points
    return;
end
closure = hypot(points(end, 1) - points(1, 1), points(end, 2) - points(1, 2));
absWinding = abs(winding) / (2*pi);
radial = hypot(points(:, 1) - cx, points(:, 2) - cy);
radialCv = std(radial, 'omitnan') / max(mean(radial, 'omitnan'), 1.0e-6);
row.streamline_winding_turns = absWinding;
row.streamline_closure_error_cells = closure;
row.streamline_closed = absWinding >= params.streamline_min_winding_turns && closure <= params.streamline_closure_tolerance_cells;
row.streamline_score = closure + 2 * abs(1 - min(absWinding, 1.5)) + radialCv;
if failedFinite
    row.streamline_score = row.streamline_score + 5;
end
row.points = points;
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

[uu, vv] = bilinear_uv_many(u, v, points(:, 1), points(:, 2));
speed = hypot(uu, vv);
finite = isfinite(uu) & isfinite(vv) & speed > 0;
row.finite_fraction = mean(finite);
rotationFailed = false;
if row.finite_fraction < params.min_finite_fraction
    rotationFailed = true;
    failureCounts(1) = failureCounts(1) + sum(~finite);
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
row.tangent_fraction_24deg = mean(aligned24);
row.tangent_fraction_30deg = mean(aligned30);
row.tangent_fraction_36deg = mean(aligned36);
row.tangent_fraction_45deg = mean(aligned45);
tolName = sprintf('tangent_fraction_%ddeg', round(params.tangent_tolerance_deg));
if isfield(row, tolName)
    row.tangent_fraction = row.(tolName);
else
    row.tangent_fraction = mean(acosd(abs(cosang(finite))) <= params.tangent_tolerance_deg);
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
row.streamline_direction_exception_fraction = directionExceptions / directionTotal;
if row.streamline_direction_exception_fraction > params.streamline_direction_exception_fraction
    rotationFailed = true;
    failureCounts(10) = failureCounts(10) + max(1, round((row.streamline_direction_exception_fraction - params.streamline_direction_exception_fraction) * size(points, 1)));
end

velocityAngle = atan2(vv, uu);
maxRatio = 0;
maxAngle = 0;
for jj = 1:size(points, 1)
    mm = mod(jj, size(points, 1)) + 1;
    if ~(finite(jj) && finite(mm))
        failureCounts(1) = failureCounts(1) + 1;
        continue;
    end
    ratio = speed(mm) / speed(jj);
    maxRatio = max([maxRatio, ratio, 1 / max(ratio, eps)]);
    if ratio > params.speed_ratio_max || ratio < 1 / params.speed_ratio_max
        failureCounts(2) = failureCounts(2) + 1;
    end
    dtheta = angle_diff(velocityAngle(jj), velocityAngle(mm));
    maxAngle = max(maxAngle, abs(dtheta) * 180 / pi);
    if abs(dtheta) * 180 / pi > params.angle_jump_max_deg
        failureCounts(3) = failureCounts(3) + 1;
    end
end
row.speed_ratio = maxRatio;
row.max_angle_jump_deg = maxAngle;

half = floor(size(points, 1) / 2);
symmetryOk = 0;
symmetryTotal = 0;
reversalOk = 0;
reversalTotal = 0;
if half >= 2
    for jj = 1:half
        mm = mod(jj - 1 + half, size(points, 1)) + 1;
        if ~(finite(jj) && finite(mm))
            continue;
        end
        diffValue = abs(angle_diff(velocityAngle(jj), velocityAngle(mm)));
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
    failureCounts(9) = failureCounts(9) + max(1, round((params.min_reversal_fraction - row.reversal_fraction) * max(reversalTotal, 1)));
end

if failureCounts(2) > 0 || failureCounts(3) > 0 || failureCounts(9) > 0
    rotationFailed = true;
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
row.hua_pass = row.circle_passed;
[row.first_hard_failure_code, row.first_hard_failure] = first_hard_failure( ...
    row.finite_fraction < params.min_finite_fraction, ...
    failureCounts(2) > 0, ...
    failureCounts(3) > 0, ...
    row.streamline_direction_exception_fraction > params.streamline_direction_exception_fraction, ...
    row.tangent_fraction < params.min_tangent_fraction, ...
    row.reversal_fraction < params.min_reversal_fraction);
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
