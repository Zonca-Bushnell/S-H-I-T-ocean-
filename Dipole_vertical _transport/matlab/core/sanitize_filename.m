function text = sanitize_filename(text)
    text = regexprep(char(text), '[^A-Za-z0-9]+', '_');
    text = regexprep(text, '^_+|_+$', '');
end
