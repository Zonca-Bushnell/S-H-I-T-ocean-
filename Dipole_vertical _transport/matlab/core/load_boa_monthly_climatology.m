function boa = load_boa_monthly_climatology(root_dir, cache_path)
    if nargin >= 2 && exist(cache_path, 'file') == 2
        C = load(cache_path, 'boa', 'source_root');
        if isfield(C, 'boa') && isfield(C, 'source_root') && strcmp(C.source_root, root_dir)
            boa = C.boa;
            fprintf('Loaded BOA monthly climatology cache: %s\n', cache_path);
            return
        end
    end
    files = dir(fullfile(root_dir, 'PDen1000_*.mat'));
    if isempty(files)
        error('No BOA potential-density files found in %s', root_dir);
    end
    sums = cell(12, 1);
    counts = cell(12, 1);
    lon_in = [];
    lat_in = [];
    pres = [];
    for k = 1:numel(files)
        name = files(k).name;
        tok = regexp(name, 'PDen1000_\d{4}(\d{2})\.mat', 'tokens', 'once');
        if isempty(tok)
            continue
        end
        mon = str2double(tok{1});
        if ~isfinite(mon) || mon < 1 || mon > 12
            continue
        end
        S = load(fullfile(files(k).folder, files(k).name), 'Den', 'lon_in', 'lat_in', 'pres');
        if isempty(lon_in)
            lon_in = double(S.lon_in(:));
            lat_in = double(S.lat_in(:));
            pres = double(S.pres(:));
        end
        D = double(S.Den);
        ok = isfinite(D);
        D(~ok) = 0;
        if isempty(sums{mon})
            sums{mon} = D;
            counts{mon} = double(ok);
        else
            sums{mon} = sums{mon} + D;
            counts{mon} = counts{mon} + double(ok);
        end
    end
    den = nan([numel(lon_in), numel(lat_in), numel(pres), 12]);
    month_count = zeros(12, 1);
    for mon = 1:12
        if isempty(sums{mon})
            continue
        end
        C = counts{mon};
        tmp = sums{mon} ./ C;
        tmp(C == 0) = NaN;
        den(:,:,:,mon) = tmp;
        month_count(mon) = max(C(:));
    end
    boa = struct('lon', lon_in, 'lat', lat_in, 'pres', pres, 'den', den, 'month_count', month_count);
    if nargin >= 2
        source_root = root_dir; %#ok<NASGU>
        cache_dir = fileparts(cache_path);
        if exist(cache_dir, 'dir') ~= 7
            mkdir(cache_dir);
        end
        save(cache_path, 'boa', 'source_root', '-v7.3');
        fprintf('Saved BOA monthly climatology cache: %s\n', cache_path);
    end
end
