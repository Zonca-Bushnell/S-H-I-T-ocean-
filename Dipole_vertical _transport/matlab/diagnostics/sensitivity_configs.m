function configs = sensitivity_configs()
    configs = struct('name', {}, 'grid_n', {}, 'cressman_radius_r', {}, 'cressman_min_obs', {}, 'smooth_passes', {});
    configs(1) = struct('name', 'baseline', 'grid_n', 81, 'cressman_radius_r', 0.5, 'cressman_min_obs', 3, 'smooth_passes', 2);
    configs(2) = struct('name', 'recommended', 'grid_n', 61, 'cressman_radius_r', 1.0, 'cressman_min_obs', 8, 'smooth_passes', 4);
    configs(3) = struct('name', 'smoother', 'grid_n', 61, 'cressman_radius_r', 1.25, 'cressman_min_obs', 8, 'smooth_passes', 4);
    configs(4) = struct('name', 'strong_support', 'grid_n', 61, 'cressman_radius_r', 1.0, 'cressman_min_obs', 12, 'smooth_passes', 4);
    configs(5) = struct('name', 'low_res_smooth', 'grid_n', 51, 'cressman_radius_r', 1.25, 'cressman_min_obs', 8, 'smooth_passes', 4);
    configs(6) = struct('name', 'high_smooth', 'grid_n', 61, 'cressman_radius_r', 1.0, 'cressman_min_obs', 8, 'smooth_passes', 6);
end
