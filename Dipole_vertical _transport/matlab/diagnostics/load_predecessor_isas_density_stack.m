function rho_stack = load_predecessor_isas_density_stack(isas_density_mat, polarity, X, Y, depth_levels)
    rho_stack = nan([size(X), numel(depth_levels)]);
    density_path = resolved_isas_density_path(isas_density_mat, polarity);
    if isempty(density_path)
        return
    end
    try
        S = load(density_path, 'Den_compound_all');
        if ~isfield(S, 'Den_compound_all')
            return
        end
        src = double(S.Den_compound_all);
        if ndims(src) == 4
            src = mean(src, 4, 'omitnan');
        end
        if ndims(src) ~= 3
            return
        end
        src_x = linspace(-4, 4, size(src, 2));
        src_y = linspace(-4, 4, size(src, 1));
        src_depth = predecessor_isas_depth_axis(size(src, 3));
        [Xsrc, Ysrc, Zsrc] = meshgrid(src_x, src_y, src_depth);
        [Xq, Yq, Zq] = meshgrid(X(1,:), Y(:,1), depth_levels(:));
        rho_stack = interp3(Xsrc, Ysrc, Zsrc, src, Xq, Yq, Zq, 'linear', NaN);
    catch ME
        warning('Could not load predecessor ISAS density field %s: %s', density_path, ME.message);
        rho_stack = nan([size(X), numel(depth_levels)]);
    end
end

function depth_axis = predecessor_isas_depth_axis(nz)
    temp_path = 'F:\Argo_data\ISAS_Argo\data\2004\ISAS20_ARGO_20040615_dat_TEMP.nc';
    depth_axis = [];
    if exist(temp_path, 'file') == 2
        try
            info = ncinfo(temp_path);
            var_names = {info.Variables.Name};
            if any(strcmp(var_names, 'DEPH'))
                depth_axis = double(ncread(temp_path, 'DEPH'));
            elseif any(strcmp(var_names, 'depth'))
                depth_axis = double(ncread(temp_path, 'depth'));
            end
        catch
            depth_axis = [];
        end
    end
    depth_axis = depth_axis(:);
    if numel(depth_axis) >= nz
        depth_axis = depth_axis(1:nz);
        depth_axis(1) = 0;
    else
        depth_axis = linspace(0, 2000, nz)';
    end
end
