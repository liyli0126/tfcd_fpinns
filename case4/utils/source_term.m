function s = source_term(x, t, alpha, D, v)
% SOURCE_TERM - Compute the source term for time-fractional advection-diffusion equation
%
% The source term is defined as:
% f(x,t) = x(1-x)[-E_α(-t^α) + t^(-α)/Γ(1-α)] + 0.002(1-2x)E_α(-t^α) + (1-2x)E_α(-t^α)
%
% Inputs:
%   x     - spatial coordinates (can be dlarray or numeric)
%   t     - temporal coordinates (can be dlarray or numeric) 
%   alpha - fractional order
%   D     - diffusion coefficient
%   v     - velocity
%
% Outputs:
%   s     - source term values (preserves input type)

% Handle both dlarray and numeric inputs
if isa(x, 'dlarray') || isa(t, 'dlarray')
    % Extract data for computation
    x_val = extractdata(x);
    t_val = extractdata(t);
    
    % Compute source term numerically
    s_val = compute_source_term_numeric(x_val, t_val, alpha, D, v);
    
    % Convert back to dlarray with appropriate format
    if isa(x, 'dlarray')
        s = dlarray(s_val, dims(x));
    else
        s = dlarray(s_val, dims(t));
    end
else
    % Direct numeric computation
    s = compute_source_term_numeric(x, t, alpha, D, v);
end

end

function s = compute_source_term_numeric(x, t, alpha, D, v)
% Numeric computation of source term

% Ensure x and t have compatible dimensions
if length(x) == 1 && length(t) > 1
    x = repmat(x, size(t));
elseif length(t) == 1 && length(x) > 1
    t = repmat(t, size(x));
end

% Initialize output
s = zeros(size(x));

% Compute source term: f(x,t) = x(1-x)[-E_α(-t^α) + t^(-α)/Γ(1-α)] + 0.002(1-2x)E_α(-t^α) + (1-2x)E_α(-t^α)
for i = 1:numel(x)
    if t(i) == 0
        % At t=0, E_α(0) = 1, and t^(-α)/Γ(1-α) approaches infinity
        % For numerical stability, use a small time value
        t_small = 1e-10;
        E_alpha_val = mlf(alpha, 1, -t_small^alpha);
        gamma_term = t_small^(-alpha) / gamma(1-alpha);
    else
        % Compute Mittag-Leffler function E_α(-t^α)
        E_alpha_val = mlf(alpha, 1, -t(i)^alpha);
        gamma_term = t(i)^(-alpha) / gamma(1-alpha);
    end
    
    % Spatial terms
    x_term = x(i) * (1 - x(i));
    diff_term = 1 - 2*x(i);
    
    % Source term: f(x,t) = x(1-x)[-E_α(-t^α) + t^(-α)/Γ(1-α)] + 0.002(1-2x)E_α(-t^α) + (1-2x)E_α(-t^α)
    s(i) = x_term * (-E_alpha_val + gamma_term) + ...
           0.002 * diff_term * E_alpha_val + ...
           diff_term * E_alpha_val;
end

% Handle numerical stability
s(isnan(s) | isinf(s)) = 0;

end
