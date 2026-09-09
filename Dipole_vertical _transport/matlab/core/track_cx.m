function cx = track_cx(lon, lat, time, track, deg_m)
    cx = nan(size(time));
    for i = 2:numel(time)-1
        if track(i-1) == track(i+1)
            dt = (time(i+1) - time(i-1)) * 86400;
            if isfinite(dt) && dt > 0
                dlon = lon(i+1) - lon(i-1);
                if dlon > 180
                    dlon = dlon - 360;
                elseif dlon < -180
                    dlon = dlon + 360;
                end
                cx(i) = dlon * deg_m * cosd(lat(i)) / dt;
            end
        end
    end
end
