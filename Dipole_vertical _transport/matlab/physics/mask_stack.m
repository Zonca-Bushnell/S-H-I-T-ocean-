function A = mask_stack(A, support3)
    A(~support3) = NaN;
end
