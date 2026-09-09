function score = roughness_score(W)
    Wf = fillmissing2(W);
    L = del2(Wf);
    support = isfinite(W);
    vals = abs(L(support));
    if isempty(vals)
        score = NaN;
    else
        score = median(vals, 'omitnan');
    end
end
