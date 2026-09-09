function write_grid_3d_mat(path, grid3d, polarity, band_label)
    metadata = grid3d_metadata(grid3d, polarity, band_label);
    save(path, 'grid3d', 'metadata', '-v7.3');
end
