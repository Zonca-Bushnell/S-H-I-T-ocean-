function out = mask_to_support(A, support)
    out = A;
    out(~support) = NaN;
end
