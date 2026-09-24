function export_filtered_velocity_binary(inputPath, outputDir)
% Stream one filtered NetCDF day to Python-readable little-endian binaries.
if ~isfolder(outputDir)
    mkdir(outputDir);
end
info = ncinfo(inputPath, 'uo_glor');
shape = info.Size;
if numel(shape) ~= 4 || shape(4) ~= 1
    error('Unexpected uo_glor dimensions in %s', inputPath);
end
nx = shape(1);
ny = shape(2);
nz = shape(3);
lon = double(ncread(inputPath, 'longitude'));
lat = double(ncread(inputPath, 'latitude'));
depth = double(ncread(inputPath, 'depth'));

write_vector(fullfile(outputDir, 'longitude.f64'), lon, 'double');
write_vector(fullfile(outputDir, 'latitude.f64'), lat, 'double');
write_vector(fullfile(outputDir, 'depth.f64'), depth, 'double');

uPart = fullfile(outputDir, 'uo_glor.f32.part');
vPart = fullfile(outputDir, 'vo_glor.f32.part');
fidU = fopen(uPart, 'w', 'ieee-le');
fidV = fopen(vPart, 'w', 'ieee-le');
if fidU < 0 || fidV < 0
    error('Cannot create velocity bridge files in %s', outputDir);
end
cleanup = onCleanup(@() close_files(fidU, fidV));
for k = 1:nz
    u = single(ncread(inputPath, 'uo_glor', [1 1 k 1], [nx ny 1 1]));
    v = single(ncread(inputPath, 'vo_glor', [1 1 k 1], [nx ny 1 1]));
    fwrite(fidU, u, 'single');
    fwrite(fidV, v, 'single');
end
clear cleanup;
movefile(uPart, fullfile(outputDir, 'uo_glor.f32'), 'f');
movefile(vPart, fullfile(outputDir, 'vo_glor.f32'), 'f');

metadata = struct();
metadata.source_netcdf = inputPath;
metadata.shape_depth_lat_lon = [nz ny nx];
metadata.dtype = 'float32_le';
metadata.layout = 'C_order_depth_lat_lon';
metadata.exporter = 'MATLAB_ncread_layer_stream';
fid = fopen(fullfile(outputDir, 'metadata.json'), 'w');
if fid < 0
    error('Cannot write metadata in %s', outputDir);
end
fprintf(fid, '%s', jsonencode(metadata, PrettyPrint=true));
fclose(fid);
end


function write_vector(path, values, precision)
fid = fopen(path, 'w', 'ieee-le');
if fid < 0
    error('Cannot write %s', path);
end
cleanup = onCleanup(@() fclose(fid));
fwrite(fid, values, precision);
end


function close_files(fidU, fidV)
if fidU > 0
    fclose(fidU);
end
if fidV > 0
    fclose(fidV);
end
end
