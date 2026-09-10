function path_out = resolved_isas_density_path(isas_density_mat, polarity)
    path_out = '';
    if isempty(isas_density_mat)
        return
    end
    candidate = char(isas_density_mat);
    if strcmp(polarity, 'anticyclonic')
        ae_candidate = strrep(candidate, '_ce_', '_ae_');
        ae_candidate = strrep(ae_candidate, '_CE_', '_AE_');
        if exist(ae_candidate, 'file') == 2
            path_out = ae_candidate;
            return
        end
        if exist(candidate, 'file') == 2
            path_out = candidate;
        end
        return
    end
    if exist(candidate, 'file') == 2
        path_out = candidate;
    end
end
