function benchmark_streamline_batch_kernel(inputMat, outputCsv, repeats)
%BENCHMARK_STREAMLINE_BATCH_KERNEL Compare reference, batch CPU, and batch GPU.
%
% inputMat must contain u, v, centers, radii, params. The output CSV records
% wall time and key parity counts for each backend variant.

if nargin < 3 || isempty(repeats)
    repeats = 3;
end
S = load(inputMat);
u = S.u;
v = S.v;
centers = S.centers;
radii = S.radii;
if isfield(S, 'params')
    params = S.params;
else
    params = struct( ...
        'start_radius_cells', 2, ...
        'max_radius_cells', 8, ...
        'speed_ratio_max', 3, ...
        'angle_jump_max_deg', 150, ...
        'tangent_tolerance_deg', 24, ...
        'symmetry_tolerance_deg', 120, ...
        'min_tangent_fraction', 0.55, ...
        'min_reversal_fraction', 0.55, ...
        'min_finite_fraction', 0.75, ...
        'direction_exception_extra', 2, ...
        'require_boundary_monotonic_rotation', true, ...
        'boundary_monotonic_exception_limit', 0, ...
        'streamline_direction_exception_fraction', 0.10, ...
        'streamline_step_cells', 0.5, ...
        'streamline_max_steps', 180, ...
        'streamline_start_angles', 4, ...
        'streamline_closure_tolerance_cells', 1.75, ...
        'streamline_min_winding_turns', 0.75, ...
        'streamline_min_points', 16);
end
params.use_gpu = false;

rows = {};
ref = [];
variants = ["reference", "batch_cpu", "batch_gpu"];
for vv = 1:numel(variants)
    variant = variants(vv);
    if variant == "batch_gpu"
        try
            gpuDevice;
            params.use_gpu = true;
        catch
            params.use_gpu = false;
            rows(end + 1, :) = {char(variant), NaN, false, "gpu_unavailable", 0, 0, 0, NaN, NaN}; %#ok<AGROW>
            continue;
        end
    else
        params.use_gpu = false;
    end
    elapsed = NaN(repeats, 1);
    out = [];
    for rr = 1:repeats
        tic;
        if variant == "reference"
            out = velocity_streamline_boundary_kernel(u, v, centers, radii, params);
        else
            out = streamline_gpu_batch_kernel(u, v, centers, radii, params);
        end
        elapsed(rr) = toc;
    end
    if variant == "reference"
        ref = out;
        passMismatch = 0;
        radiusMismatch = 0;
        firstFailureMismatch = 0;
        maxClosureDiff = 0;
        maxWindingDiff = 0;
    else
        passMismatch = sum([out.hua_pass] ~= [ref.hua_pass]);
        radiusMismatch = sum([out.accepted_radius_cells] ~= [ref.accepted_radius_cells]);
        firstFailureMismatch = sum(string({out.first_hard_failure}) ~= string({ref.first_hard_failure}));
        maxClosureDiff = max(abs([out.streamline_closure_error_cells] - [ref.streamline_closure_error_cells]), [], 'omitnan');
        maxWindingDiff = max(abs([out.streamline_winding_turns] - [ref.streamline_winding_turns]), [], 'omitnan');
    end
    rows(end + 1, :) = {char(variant), median(elapsed, 'omitnan'), true, "", numel(out), passMismatch, radiusMismatch, firstFailureMismatch, maxClosureDiff, maxWindingDiff}; %#ok<AGROW>
end

T = cell2table(rows, 'VariableNames', {
    'variant', 'median_seconds', 'completed', 'note', 'n_centers', ...
    'pass_mismatch_count', 'radius_mismatch_count', 'first_failure_mismatch_count', ...
    'max_closure_error_abs_diff', 'max_winding_abs_diff'});
writetable(T, outputCsv);
disp(T);
end
