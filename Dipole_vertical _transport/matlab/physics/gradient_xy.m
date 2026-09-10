function [dfdx, dfdy] = gradient_xy(F, dx, dy)
%GRADIENT_XY Explicit horizontal gradients for eddy composites.
%   Matrix rows are y/R (south-north) and columns are x/R (west-east).
%   dfdx is therefore the finite difference along columns; dfdy is along
%   rows. This wrapper avoids ambiguity in MATLAB gradient output ordering.
    if nargin < 3
        error('gradient_xy requires F, dx, and dy.');
    end
    if ~isfinite(dx) || dx <= 0 || ~isfinite(dy) || dy <= 0
        dfdx = nan(size(F));
        dfdy = nan(size(F));
        return
    end
    [dfdx, dfdy] = gradient(F, dx, dy);
end
