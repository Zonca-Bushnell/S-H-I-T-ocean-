function selected = select_sensitivity_configs(configs, names)
    if isempty(names)
        selected = configs;
        return
    end
    selected = struct('name', {}, 'grid_n', {}, 'cressman_radius_r', {}, 'cressman_min_obs', {}, 'smooth_passes', {});
    config_names = {configs.name};
    for i = 1:numel(names)
        name = char(names{i});
        idx = find(strcmp(config_names, name), 1);
        if isempty(idx)
            error('Unknown sensitivity config: %s', name);
        end
        selected(end+1) = configs(idx); %#ok<AGROW>
    end
end
