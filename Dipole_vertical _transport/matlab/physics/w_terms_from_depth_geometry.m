function [term1_up, term2_up, w_up, term1_depth, term2_depth, w_depth] = w_terms_from_depth_geometry(dDdx, dDdy, u, v, cx_rel, u_bg, support)
%W_TERMS_FROM_DEPTH_GEOMETRY Compute W from positive-down isopycnal depth.
%   D is positive downward. W_UP is positive upward. The split used by the
%   production pipeline is:
%       term1_up = c_rel * dD/dx
%       term2_up = -[(u-u_bg) * dD/dx + v * dD/dy]
%       W_up     = term1_up + term2_up
%
%   The depth-positive counterparts are exactly -term*_up and are retained
%   only for sign auditing.
    if nargin < 7 || isempty(support)
        support = true(size(dDdx));
    end
    u_rel = u - u_bg;
    term1_up = cx_rel .* dDdx;
    term2_up = -(u_rel .* dDdx + v .* dDdy);
    w_up = term1_up + term2_up;
    term1_depth = -term1_up;
    term2_depth = -term2_up;
    w_depth = -w_up;
    term1_up = mask_to_support(term1_up, support);
    term2_up = mask_to_support(term2_up, support);
    w_up = mask_to_support(w_up, support);
    term1_depth = mask_to_support(term1_depth, support);
    term2_depth = mask_to_support(term2_depth, support);
    w_depth = mask_to_support(w_depth, support);
end
