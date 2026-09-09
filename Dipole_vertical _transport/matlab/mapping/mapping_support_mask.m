function support = mapping_support_mask(~, mapped_support, ~, ~, cressman_min_obs)
    support = mapped_support >= cressman_min_obs;
end
