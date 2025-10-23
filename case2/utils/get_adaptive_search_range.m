function [x_min, x_max] = get_adaptive_search_range(t, alpha)
% GET_ADAPTIVE_SEARCH_RANGE - More precise search range based on theoretical analysis
%
% Inputs:
%   t - time point
%   alpha - fractional derivative order
%
% Outputs:
%   x_min, x_max - search range boundaries

% Calculate analytical solution dominant terms
if t < 1e-3  % Ultra-early, sin(x) dominant
    x_min = 0.55; x_max = 0.65;  % sin(x) peak in [0,1] is around x=π/2≈0.785, but modulated by (2x(1-x))^2
elseif t < 0.1  % Early, t^α starts to influence
    x_min = 0.45; x_max = 0.65;  % Transition region
elseif t < 0.5  % Mid-early, t^α influence increases
    x_min = 0.40; x_max = 0.60;  % Polynomial influence grows
elseif t < 1.0  % Mid, t^α influence further increases
    x_min = 0.35; x_max = 0.55;  % Strong polynomial influence
else  % Late, t^α dominant
    x_min = 0.35; x_max = 0.55;  % Polynomial part (2x(1-x))^2 peak at x=0.5, but consider overall evolution
end

end
