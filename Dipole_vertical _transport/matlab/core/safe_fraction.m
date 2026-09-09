function f = safe_fraction(a, b)
    if b > 0
        f = a / b;
    else
        f = NaN;
    end
end
