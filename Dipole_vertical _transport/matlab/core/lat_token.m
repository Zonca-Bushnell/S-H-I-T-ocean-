function token = lat_token(lat)
    if lat < 0
        hemi = 'S';
    else
        hemi = 'N';
    end
    token = sprintf('%02.0f%s', abs(lat), hemi);
end
