function hua_day_batch_to_mat(outputPath, uDay, vDay, lon, lat, seedCenters, radii, params, targetDegree, windowRadiusCells, minFiniteFraction, stopAtFirstFailed)
%HUA_DAY_BATCH_TO_MAT Run day-level Hua continuation and write column arrays.
%
% This is the production MATLAB-to-Python bridge. It avoids returning large
% struct arrays through MATLAB Engine as JSON strings. MATLAB writes a temporary
% .mat file; Python reads it with scipy.io.loadmat and deletes it by default.

rows = hua_day_batch_kernel(uDay, vDay, lon, lat, seedCenters, radii, params, targetDegree, windowRadiusCells, minFiniteFraction, stopAtFirstFailed);
out = rows_to_bridge(rows);
save(outputPath, '-struct', 'out', '-v7');
end


function out = rows_to_bridge(rows)
out = struct();
n = numel(rows);
out.bridge_n_rows = double(n);
if n == 0
    return;
end
names = fieldnames(rows);
stringKeep = ["subgrid_fit_quality"];
for ii = 1:numel(names)
    name = names{ii};
    first = rows(1).(name);
    if isnumeric(first) || islogical(first)
        values = nan(n, 1);
        for rr = 1:n
            value = rows(rr).(name);
            if isempty(value)
                values(rr) = NaN;
            else
                if isa(value, 'gpuArray')
                    value = gather(value);
                end
                values(rr) = double(value(1));
            end
        end
        out.(name) = values;
    elseif isstring(first) || ischar(first)
        if ~any(strcmp(string(name), stringKeep))
            continue;
        end
        values = cell(n, 1);
        for rr = 1:n
            value = rows(rr).(name);
            if isstring(value)
                values{rr} = char(value);
            elseif ischar(value)
                values{rr} = value;
            else
                values{rr} = char(string(value));
            end
        end
        out.(name) = values;
    end
end
end
