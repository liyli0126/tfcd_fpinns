function f = source_term_case2(t, x, alpha)
% SOURCE_TERM_CASE2  Computes the source term f(x,t) for case 2
% for the fractional advection-diffusion equation with exact solution
% u(x,t) = (2x(1 - x))^2 (t^alpha + sin(x)).
%
% INPUT:
%   t     - time variable (scalar)
%   x     - spatial variable (vector or scalar)
%   alpha - fractional order (0 < alpha < 1)
%
% OUTPUT:
%   f     - evaluated source term f(x,t)

% Ensure x is a column vector for broadcasting
x = x(:);

% Common terms
ta = t.^alpha;
Gamma_term = gamma(1 + alpha);
sinx = sin(x);
cosx = cos(x);

% Polynomial expressions
phi1 = -8 + 56*x - 72*x.^2 + 16*x.^3;                % Coefficient of (t^α + sinx)
phi2 = -16*x + 52*x.^2 - 40*x.^3 + 4*x.^4;           % Coefficient of cosx
phi3 = 4*x.^2 .* (1 - x).^2;                         % Appears in both Γ-term and sinx-term

% Source term f(x,t) without scaling factor
f = (Gamma_term .* phi3 ...
    + (ta + sinx) .* phi1 ...
    + cosx .* phi2 ...
    + phi3 .* sinx);

end
