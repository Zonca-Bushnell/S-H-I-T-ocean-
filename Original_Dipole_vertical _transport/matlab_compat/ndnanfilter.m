function out = ndnanfilter(in, kernel, win)
%NDNANFILTER Minimal compatibility replacement for predecessor scripts.
% Supports the 2-D rectangular-window calls used by the original programs.
if nargin < 3 || isempty(win)
    win = [1 1];
end
if ischar(kernel) || isstring(kernel)
    kernel = ones(max(1, round(win(1))), max(1, round(win(2))));
end
if isvector(win) && numel(win) >= 2
    kernel = ones(max(1, round(win(1))), max(1, round(win(2))));
end
x = double(in);
good = isfinite(x);
x0 = x;
x0(~good) = 0;
den = conv2(double(good), kernel, 'same');
num = conv2(x0, kernel, 'same');
out = num ./ den;
out(den == 0) = NaN;
end
