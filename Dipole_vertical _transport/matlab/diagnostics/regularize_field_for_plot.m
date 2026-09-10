function [out, info] = regularize_field_for_plot(field, fill_passes, smooth_passes)
    if nargin < 2 || ~isfinite(fill_passes)
        fill_passes = 2;
    end
    if nargin < 3 || ~isfinite(smooth_passes)
        smooth_passes = 3;
    end

    out = double(field);
    original_support = isfinite(out);
    info = struct();
    info.input_valid_count = nnz(original_support);
    info.fill_passes = fill_passes;
    info.smooth_passes = smooth_passes;

    if info.input_valid_count == 0
        info.filled_count = 0;
        info.output_valid_count = 0;
        return
    end

    fill_support = original_support;
    for pass = 1:fill_passes
        candidate = out;
        for ii = 1:size(out, 1)
            i0 = max(1, ii - 1);
            i1 = min(size(out, 1), ii + 1);
            for jj = 1:size(out, 2)
                if isfinite(out(ii, jj))
                    continue
                end
                j0 = max(1, jj - 1);
                j1 = min(size(out, 2), jj + 1);
                vals = out(i0:i1, j0:j1);
                ok = isfinite(vals);
                if nnz(ok) >= 4
                    candidate(ii, jj) = median(vals(ok), 'omitnan');
                    fill_support(ii, jj) = true;
                end
            end
        end
        if isequaln(candidate, out)
            break
        end
        out = candidate;
    end

    for pass = 1:smooth_passes
        candidate = out;
        for ii = 1:size(out, 1)
            i0 = max(1, ii - 1);
            i1 = min(size(out, 1), ii + 1);
            for jj = 1:size(out, 2)
                if ~fill_support(ii, jj)
                    continue
                end
                j0 = max(1, jj - 1);
                j1 = min(size(out, 2), jj + 1);
                vals = out(i0:i1, j0:j1);
                ok = isfinite(vals);
                if any(ok, 'all')
                    candidate(ii, jj) = mean(vals(ok), 'omitnan');
                end
            end
        end
        out = candidate;
        out(~fill_support) = NaN;
    end

    info.output_valid_count = nnz(isfinite(out));
    info.filled_count = nnz(isfinite(out) & ~original_support);
end
