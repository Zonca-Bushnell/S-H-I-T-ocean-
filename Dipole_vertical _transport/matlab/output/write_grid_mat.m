function write_grid_mat(path, grid, polarity, band_label)
    metadata = grid_metadata(grid, polarity, band_label);
    save(path, 'grid', 'metadata');
end
