function [xq, yq, zq] = upsample_field_for_plot(x, y, z, x_factor, y_factor)
%UPSAMPLE_FIELD_FOR_PLOT Interpolate a display-only field onto a denser grid.
%   This function is used only after PNG regularization. It does not alter
%   saved scientific MAT fields.
    if nargin < 4 || ~isfinite(x_factor) || x_factor < 1
        x_factor = 1;
    end
    if nargin < 5 || ~isfinite(y_factor) || y_factor < 1
        y_factor = x_factor;
    end

    x = double(x(:)');
    y = double(y(:));
    z = double(z);
    xq = linspace(min(x), max(x), max(numel(x), round(numel(x) * x_factor)));
    yq = linspace(min(y), max(y), max(numel(y), round(numel(y) * y_factor)));

    valid = isfinite(z);
    if nnz(valid) < 4
        zq = nan(numel(yq), numel(xq));
        return
    end

    z_filled = z;
    if any(~valid, 'all')
        fill_value = median(z(valid), 'omitnan');
        z_filled(~valid) = fill_value;
    end
    [X, Y] = meshgrid(x, y);
    [Xq, Yq] = meshgrid(xq, yq);
    zq = interp2(X, Y, z_filled, Xq, Yq, 'makima');

    maskq = interp2(X, Y, double(valid), Xq, Yq, 'linear', 0) > 0.25;
    zq(~maskq) = NaN;
end
