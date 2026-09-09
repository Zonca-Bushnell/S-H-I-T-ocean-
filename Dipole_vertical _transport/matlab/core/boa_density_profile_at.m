function profile = boa_density_profile_at(boa, lon, lat, month_id)
    profile = nan(size(boa.pres));
    if isempty(fieldnames(boa)) || ~isfinite(lon) || ~isfinite(lat) || ~isfinite(month_id)
        return
    end
    month_id = max(1, min(12, round(month_id)));
    lon = mod(lon, 360);
    if lon < min(boa.lon)
        lon = lon + 360;
    end
    lon2 = boa.lon;
    den = boa.den(:,:,:,month_id);
    if lon > max(lon2)
        lon2 = [lon2; lon2(1) + 360];
        den = cat(1, den, den(1,:,:));
    end
    if lat < min(boa.lat) || lat > max(boa.lat)
        return
    end
    ix2 = find(lon2 >= lon, 1, 'first');
    iy2 = find(boa.lat >= lat, 1, 'first');
    if isempty(ix2) || isempty(iy2) || ix2 <= 1 || iy2 <= 1
        return
    end
    ix1 = ix2 - 1;
    iy1 = iy2 - 1;
    x1 = lon2(ix1); x2 = lon2(ix2);
    y1 = boa.lat(iy1); y2 = boa.lat(iy2);
    if x2 == x1 || y2 == y1
        return
    end
    wx = (lon - x1) / (x2 - x1);
    wy = (lat - y1) / (y2 - y1);
    p11 = squeeze(den(ix1,iy1,:));
    p21 = squeeze(den(ix2,iy1,:));
    p12 = squeeze(den(ix1,iy2,:));
    p22 = squeeze(den(ix2,iy2,:));
    profile = (1-wx) * (1-wy) * p11 + wx * (1-wy) * p21 + (1-wx) * wy * p12 + wx * wy * p22;
end
