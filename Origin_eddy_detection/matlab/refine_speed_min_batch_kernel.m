function rows = refine_speed_min_batch_kernel(speed, u, v, lon, lat, centers, target_degree, window_radius_cells, min_finite_fraction, params)
%REFINE_SPEED_MIN_BATCH_KERNEL Batch subgrid speed-minimum refinement.
%
% rows = refine_speed_min_batch_kernel(speed,u,v,lon,lat,centers,...)
%
% centers are 1-based [i,j] grid anchors.  The returned center_i_refined and
% center_j_refined are also 1-based so the Python caller can subtract one and
% keep its native zero-based convention.  The main fit mirrors Python's
% quadratic fit to speed^2 before the Hua boundary checks.

if nargin < 10
    params = struct();
end
if ~isfield(params, 'use_gpu')
    params.use_gpu = false;
end
if params.use_gpu && ~isa(speed, 'gpuArray')
    try
        speed = gpuArray(double(speed));
        u = gpuArray(double(u));
        v = gpuArray(double(v));
    catch
        speed = double(speed);
        u = double(u);
        v = double(v);
        params.use_gpu = false;
    end
end

lon = double(lon(:));
lat = double(lat(:));
centers = double(centers);
n = size(centers, 1);
template = refine_empty_row();
rows = repmat(template, n, 1);
[ny, nx] = size(speed);
dlon = median(abs(diff(lon)), 'omitnan');
dlat = median(abs(diff(lat)), 'omitnan');
midLat = median(lat, 'omitnan');
dxKm = abs(deg2rad(dlon) * 6371000.0 * cosd(midLat) / 1000.0);
dyKm = abs(deg2rad(dlat) * 6371000.0 / 1000.0);

for rr = 1:n
    centerI = round(centers(rr, 1));
    centerJ = round(centers(rr, 2));
    centerI = max(1, min(nx, centerI));
    centerJ = max(1, min(ny, centerJ));
    rows(rr) = fallback_row(speed, lon, lat, centerI, centerJ);

    if target_degree <= 0 || window_radius_cells < 1
        rows(rr).subgrid_fit_quality = "disabled";
        continue;
    end
    x0 = max(1, centerI - round(window_radius_cells));
    x1 = min(nx, centerI + round(window_radius_cells));
    y0 = max(1, centerJ - round(window_radius_cells));
    y1 = min(ny, centerJ + round(window_radius_cells));
    if x1 <= x0 || y1 <= y0
        rows(rr).subgrid_fit_quality = "window_too_small";
        continue;
    end

    speedWindow = double(speed(y0:y1, x0:x1));
    uWindow = double(u(y0:y1, x0:x1));
    vWindow = double(v(y0:y1, x0:x1));
    finite = isfinite(speedWindow) & isfinite(uWindow) & isfinite(vWindow);
    finiteFraction = gather(mean(finite(:)));
    if finiteFraction < min_finite_fraction
        rows(rr).subgrid_fit_quality = "insufficient_finite";
        continue;
    end
    if dlon <= 0 || dlat <= 0 || ~isfinite(dlon) || ~isfinite(dlat)
        rows(rr).subgrid_fit_quality = "invalid_grid_spacing";
        continue;
    end

    stepI = max(target_degree / dlon, 1.0e-3);
    stepJ = max(target_degree / dlat, 1.0e-3);
    xi = 0:stepI:((x1 - x0) + 0.5 * stepI);
    yj = 0:stepJ:((y1 - y0) + 0.5 * stepJ);
    if numel(xi) < 3 || numel(yj) < 3
        rows(rr).subgrid_fit_quality = "refined_grid_too_small";
        continue;
    end
    [quadraticDense, quality] = quadratic_speed_surface(speedWindow, finite, xi, yj);
    if isempty(quadraticDense) || ~any(gather(isfinite(quadraticDense(:))))
        rows(rr).subgrid_fit_quality = quality;
        continue;
    end

    denseHost = gather(quadraticDense);
    [~, pick] = min(denseHost(:), [], 'omitnan');
    [pickJ, pickI] = ind2sub(size(denseHost), pick);
    if pickI == 1 || pickI == size(denseHost, 2) || pickJ == 1 || pickJ == size(denseHost, 1)
        rows(rr).subgrid_fit_quality = "minimum_on_refined_boundary";
        continue;
    end

    refinedI = x0 + xi(pickI);
    refinedJ = y0 + yj(pickJ);
    offsetKm = hypot((refinedI - centerI) * dxKm, (refinedJ - centerJ) * dyKm);
    rows(rr).center_i_refined = refinedI;
    rows(rr).center_j_refined = refinedJ;
    rows(rr).center_lon_refined = interp1(1:numel(lon), lon, refinedI, 'linear', 'extrap');
    rows(rr).center_lat_refined = interp1(1:numel(lat), lat, refinedJ, 'linear', 'extrap');
    rows(rr).refined_speed_ms = denseHost(pickJ, pickI);
    rows(rr).refined_offset_km = offsetKm;
    rows(rr).refined_ok = true;
    rows(rr).subgrid_fit_quality = quality;
end
end


function row = fallback_row(speed, lon, lat, centerI, centerJ)
row = refine_empty_row();
row.center_i_refined = centerI;
row.center_j_refined = centerJ;
row.center_lon_refined = lon(centerI);
row.center_lat_refined = lat(centerJ);
value = speed(centerJ, centerI);
if isa(value, 'gpuArray')
    value = gather(value);
end
row.refined_speed_ms = double(value);
if ~isfinite(row.refined_speed_ms)
    row.refined_speed_ms = NaN;
end
row.refined_offset_km = 0;
row.refined_ok = false;
row.subgrid_fit_quality = "fallback_grid";
end


function [dense, quality] = quadratic_speed_surface(speedWindow, finite, xi, yj)
dense = [];
quality = "quadratic_lstsq_failed";
[yy0, xx0] = ndgrid(0:size(speedWindow, 1)-1, 0:size(speedWindow, 2)-1);
if isa(speedWindow, 'gpuArray')
    xx0 = gpuArray(double(xx0));
    yy0 = gpuArray(double(yy0));
else
    xx0 = double(xx0);
    yy0 = double(yy0);
end
x = xx0(finite);
y = yy0(finite);
z = speedWindow(finite) .^ 2;
if gather(numel(z)) < 9
    quality = "quadratic_insufficient_points";
    return;
end
X = [x.^2, y.^2, x.*y, x, y, ones(size(x), 'like', x)];
normal = X' * X;
rhs = X' * z;
diagMax = max(abs(diag(normal)));
if isa(diagMax, 'gpuArray')
    diagMaxScalar = gather(diagMax);
else
    diagMaxScalar = diagMax;
end
normal = normal + eye(6, 'like', normal) * max(double(diagMaxScalar), 1.0) * 1.0e-10;
try
    coeff = normal \ rhs;
catch
    quality = "quadratic_lstsq_failed";
    return;
end
h00 = 2.0 * coeff(1);
h11 = 2.0 * coeff(2);
detH = h00 * h11 - coeff(3) * coeff(3);
convex = gather(isfinite(h00) & isfinite(h11) & isfinite(detH) & h00 > 0 & detH > 0);
if ~convex
    quality = "quadratic_not_convex";
    return;
end
[yy, xx] = ndgrid(yj, xi);
if isa(speedWindow, 'gpuArray')
    xx = gpuArray(double(xx));
    yy = gpuArray(double(yy));
else
    xx = double(xx);
    yy = double(yy);
end
denseSq = coeff(1).*xx.^2 + coeff(2).*yy.^2 + coeff(3).*xx.*yy + coeff(4).*xx + coeff(5).*yy + coeff(6);
if ~gather(any(isfinite(denseSq(:))))
    quality = "quadratic_no_finite";
    dense = [];
    return;
end
dense = sqrt(max(denseSq, 0));
quality = "quadratic_speed2_1_24deg";
end


function row = refine_empty_row()
row = struct( ...
    'center_i_refined', NaN, ...
    'center_j_refined', NaN, ...
    'center_lon_refined', NaN, ...
    'center_lat_refined', NaN, ...
    'refined_speed_ms', NaN, ...
    'refined_offset_km', NaN, ...
    'refined_ok', false, ...
    'subgrid_fit_quality', "fallback_grid");
end
