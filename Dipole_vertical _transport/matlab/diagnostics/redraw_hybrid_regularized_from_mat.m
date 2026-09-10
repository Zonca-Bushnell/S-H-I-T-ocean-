function redraw_hybrid_regularized_from_mat(result_root)
    if nargin < 1 || isempty(result_root)
        result_root = 'E:\DATA\01_Eddy_correspond\05_Original_Dipole_vertical _transport\ISAS_term2_crossing_20N_regularized_display';
    end

    polarities = {'cyclonic', 'anticyclonic'};
    for pp = 1:numel(polarities)
        polarity = polarities{pp};
        mat_path = fullfile(result_root, polarity, 'cross_20N_1R', 'argo_absolute_term1_isas_term2_terms.mat');
        if exist(mat_path, 'file') ~= 2
            warning('Missing result MAT: %s', mat_path);
            continue
        end
        S = load(mat_path, 'hybrid');
        out_dir = fileparts(mat_path);
        plot_hybrid_term1_isas_term2_3panel(fullfile(out_dir, 'argo_absolute_term1_isas_term2_section_regularized.png'), S.hybrid, polarity, 'cross_20N_1R');
        plot_hybrid_depth_slices(fullfile(out_dir, 'argo_absolute_term1_isas_term2_depth_slices_regularized.png'), S.hybrid, polarity, 'cross_20N_1R');
    end
end
