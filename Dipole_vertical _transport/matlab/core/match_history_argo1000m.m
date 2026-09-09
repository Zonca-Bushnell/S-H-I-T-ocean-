function [u_out, v_out, wpk_out, matched] = match_history_argo1000m(argo_pf, argo_time, argo_lon, argo_lat, argo_park, bbox, core_min_m, core_max_m, H)
    u_out = nan(size(argo_time));
    v_out = nan(size(argo_time));
    wpk_out = nan(size(argo_time));
    matched = false(size(argo_time));
    hist_lon = double(H.I_Lon);
    hist_lon(hist_lon < 0) = hist_lon(hist_lon < 0) + 360;
    hist_lat = double(H.I_Lat);
    hist_time = double(H.I_Time);
    hist_park = double(H.I_ParkDepth);
    hist_pf = double(H.I_PF);
    hist_mask = hist_lon >= bbox(1) & hist_lon <= bbox(2) & hist_lat >= bbox(3) & hist_lat <= bbox(4) & ...
        hist_park >= core_min_m & hist_park <= core_max_m & isfinite(H.I_Upk) & isfinite(H.I_Vpk) & isfinite(H.I_Wpk);
    argo_mask = argo_lon >= bbox(1) & argo_lon <= bbox(2) & argo_lat >= bbox(3) & argo_lat <= bbox(4) & ...
        argo_park >= core_min_m & argo_park <= core_max_m;
    hist_idx = find(hist_mask);
    argo_idx = find(argo_mask);
    if isempty(hist_idx) || isempty(argo_idx)
        return
    end
    hist_key = argo_match_key(hist_pf(hist_idx), hist_time(hist_idx), hist_lon(hist_idx), hist_lat(hist_idx));
    argo_key = argo_match_key(argo_pf(argo_idx), argo_time(argo_idx), argo_lon(argo_idx), argo_lat(argo_idx));
    [tf, loc] = ismember(argo_key, hist_key);
    if any(tf)
        good_argo = argo_idx(tf);
        good_hist = hist_idx(loc(tf));
        u_out(good_argo) = double(H.I_Upk(good_hist));
        v_out(good_argo) = double(H.I_Vpk(good_hist));
        wpk_out(good_argo) = double(H.I_Wpk(good_hist));
        matched(good_argo) = true;
    end
end
