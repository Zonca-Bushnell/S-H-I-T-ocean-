function out = fillmissing2(A)
    out = A;
    for n = 1:4
        B = out;
        for i = 1:size(out,1)
            for j = 1:size(out,2)
                if ~isfinite(out(i,j))
                    i0 = max(1, i-1); i1 = min(size(out,1), i+1);
                    j0 = max(1, j-1); j1 = min(size(out,2), j+1);
                    vals = out(i0:i1, j0:j1);
                    if any(isfinite(vals), 'all')
                        B(i,j) = mean(vals(isfinite(vals)), 'omitnan');
                    end
                end
            end
        end
        out = B;
    end
end
