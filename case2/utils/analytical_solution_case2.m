function u = analytical_solution_case2(x, t, alpha)
% ANALYTICAL_SOLUTION_CASE2 - Analytical solution for case 2
% u(x,t) = (2x(1 - x))^2 (t^alpha + sin(x))
%
% Inputs:
%   x - spatial variable (vector or scalar)
%   t - time variable (scalar)
%   alpha - fractional order (0 < alpha < 1)
%
% Outputs:
%   u - analytical solution

% Create meshgrid
[T, X] = meshgrid(t, x);

% Compute analytical solution
u = (2*X.*(1-X)).^2 .* (T.^alpha + sin(X));

% Ensure boundary conditions
u(abs(X) < 1e-10) = 0;      % u(0,t) = 0
u(abs(X-1) < 1e-10) = 0;    % u(1,t) = 0

end
