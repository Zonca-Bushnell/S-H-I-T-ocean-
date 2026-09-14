function [out, info] = regularize_field_for_plot(field, fill_passes, smooth_passes, varargin)
    if nargin < 2 || ~isfinite(fill_passes)
        fill_passes = 2;
    end
    if nargin < 3 || ~isfinite(smooth_passes)
        smooth_passes = 3;
    end
    opts = struct('sigma', [1.1 1.1], 'min_neighbors', 3);
    if mod(numel(varargin), 2) ~= 0
        error('regularize_field_for_plot: options must be name/value pairs');
    end
    for kk = 1:2:numel(varargin)
        key = lower(string(varargin{kk}));
        switch key
            case "sigma"
                opts.sigma = double(varargin{kk+1});
                if isscalar(opts.sigma)
                    opts.sigma = [opts.sigma opts.sigma];
                end
            case "min_neighbors"
                opts.min_neighbors = double(varargin{kk+1});
            otherwise
                error('regularize_field_for_plot: unknown option %s', key);
        end
    end

    out = double(field);
    original_support = isfinite(out);
    info = struct();
    info.input_valid_count = nnz(original_support);
    info.fill_passes = fill_passes;
    info.smooth_passes = smooth_passes;
    info.sigma_y = opts.sigma(1);
    info.sigma_x = opts.sigma(2);

    if info.input_valid_count == 0
        info.filled_count = 0;
        info.output_valid_count = 0;
        return
    end

    fill_support = original_support;
    fill_kernel = [1 2 1; 2 4 2; 1 2 1];
    fill_kernel = fill_kernel ./ sum(fill_kernel(:));
    for pass = 1:fill_passes
        valid = isfinite(out);
        numerator = conv2(replace_nan(out, 0) .* valid, fill_kernel, 'same');
        denominator = conv2(double(valid), fill_kernel, 'same');
        neighbor_count = conv2(double(valid), ones(3), 'same');
        candidate = out;
        can_fill = ~valid & denominator > 0 & neighbor_count >= opts.min_neighbors;
        candidate(can_fill) = numerator(can_fill) ./ denominator(can_fill);
        fill_support(can_fill) = true;
        if isequaln(candidate, out)
            break
        end
        out = candidate;
    end

    smooth_kernel = gaussian_kernel_2d(opts.sigma(1), opts.sigma(2));
    for pass = 1:smooth_passes
        valid = isfinite(out) & fill_support;
        numerator = conv2(replace_nan(out, 0) .* valid, smooth_kernel, 'same');
        denominator = conv2(double(valid), smooth_kernel, 'same');
        candidate = out;
        smoothable = fill_support & denominator > 0;
        candidate(smoothable) = numerator(smoothable) ./ denominator(smoothable);
        out = candidate;
        out(~fill_support) = NaN;
    end

    info.output_valid_count = nnz(isfinite(out));
    info.filled_count = nnz(isfinite(out) & ~original_support);
end

function kernel = gaussian_kernel_2d(sigma_y, sigma_x)
    sigma_y = max(double(sigma_y), 0.5);
    sigma_x = max(double(sigma_x), 0.5);
    ry = max(1, ceil(3 * sigma_y));
    rx = max(1, ceil(3 * sigma_x));
    yy = (-ry:ry)';
    xx = -rx:rx;
    kernel = exp(-(yy.^2 ./ (2 * sigma_y.^2) + xx.^2 ./ (2 * sigma_x.^2)));
    kernel = kernel ./ sum(kernel(:));
end

function out = replace_nan(in, value)
    out = in;
    out(~isfinite(out)) = value;
end
