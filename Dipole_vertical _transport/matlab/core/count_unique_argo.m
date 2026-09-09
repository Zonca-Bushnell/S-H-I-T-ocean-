function n = count_unique_argo(matches)
    if isempty(matches)
        n = 0;
    else
        n = numel(unique(cell2mat(matches(:,3))));
    end
end
