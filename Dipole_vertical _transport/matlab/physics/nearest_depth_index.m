function idx = nearest_depth_index(depth_levels, target_depth)
    [~, idx] = min(abs(depth_levels(:) - target_depth));
end
