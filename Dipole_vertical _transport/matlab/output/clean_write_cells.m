function C = clean_write_cells(C)
    for ii = 1:numel(C)
        try
            if ismissing(C{ii})
                C{ii} = '';
            end
        catch
        end
    end
end
