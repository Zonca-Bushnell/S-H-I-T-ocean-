function key = argo_match_key(pf, time, lon, lat)
    key = string(round(double(pf))) + "_" + string(round(double(time) * 1e6)) + "_" + ...
        string(round(double(lon) * 1e4)) + "_" + string(round(double(lat) * 1e4));
end
