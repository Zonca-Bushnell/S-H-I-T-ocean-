function write_nc_3d(path, name, value)
    [ny, nx, nz] = size(value);
    nccreate(path, name, 'Dimensions', {'y', ny, 'x', nx, 'depth', nz}, 'Datatype', 'double', 'DeflateLevel', 4);
    ncwrite(path, name, double(value));
end
