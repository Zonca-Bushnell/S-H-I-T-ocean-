function write_nc_2d(path, name, value)
    [ny, nx] = size(value);
    nccreate(path, name, 'Dimensions', {'y', ny, 'x', nx}, 'Datatype', 'double', 'DeflateLevel', 4);
    ncwrite(path, name, double(value));
end
