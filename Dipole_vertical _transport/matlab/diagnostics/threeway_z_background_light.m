function light = threeway_z_background_light(comparison)
    light = struct();
    light.x = comparison.x;
    light.y = comparison.y;
    light.depth_levels = comparison.depth_levels;
    light.section_axis = comparison.section_axis;
    light.section_half_width_r = comparison.section_half_width_r;
    light.isas_info = comparison.isas_info;
    light.panel_names = comparison.panel_names;
    light.panel_stats = comparison.panel_stats;
    panels = struct('name', {}, 'title', {}, 'description', {}, 'section_w', {}, 'section_term2', {}, 'stats', {});
    for ii = 1:numel(comparison.panels)
        panels(ii).name = comparison.panels(ii).name; %#ok<AGROW>
        panels(ii).title = comparison.panels(ii).title;
        panels(ii).description = comparison.panels(ii).description;
        panels(ii).section_w = comparison.panels(ii).section_w;
        panels(ii).section_term2 = comparison.panels(ii).section_term2;
        panels(ii).stats = comparison.panels(ii).stats;
    end
    light.panels = panels;
end
