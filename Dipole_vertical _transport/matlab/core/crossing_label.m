function label = crossing_label(target_lat, intersect_radius_r)
    radius_text = regexprep(sprintf('%.6g', intersect_radius_r), '\.', 'p');
    label = ['cross_' lat_token(target_lat) '_' radius_text 'R'];
end
