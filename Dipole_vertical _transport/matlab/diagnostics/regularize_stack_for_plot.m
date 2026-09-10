function [out, info] = regularize_stack_for_plot(stack, fill_passes, smooth_passes)
    out = double(stack);
    info = struct('input_valid_count', nnz(isfinite(stack)), 'filled_count', 0, ...
        'output_valid_count', 0, 'fill_passes', fill_passes, 'smooth_passes', smooth_passes);

    for kk = 1:size(stack, 3)
        [out(:,:,kk), one] = regularize_field_for_plot(stack(:,:,kk), fill_passes, smooth_passes);
        info.filled_count = info.filled_count + one.filled_count;
    end
    info.output_valid_count = nnz(isfinite(out));
end
