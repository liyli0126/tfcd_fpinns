function u = analytical_solution(x, t, alpha, D, v, N_terms)
% ANALYTICAL_SOLUTION - Compute analytical solution for fractional convection-diffusion equation
%
% Equation: D_t^α u(x,t) - D ∂²u/∂x² + v ∂u/∂x = f(x,t)
% Parameters: D = 0.001, v = 1, T = 1, L = 1
% Initial Condition: u(x,0) = x(1-x)
% Boundary Conditions: u(0,t) = 0, u(1,t) = 0
% Exact Solution: u(x,t) = x(1-x) * E_α(-t^α)
%
% Inputs:
%   x       - spatial coordinates
%   t       - time coordinate
%   alpha   - fractional order
%   D       - diffusion coefficient
%   v       - velocity
%   N_terms - number of terms in series (unused for this simple solution)
%
% Output:
%   u       - analytical solution values

% Handle scalar inputs
if isscalar(x)
    x = x(:);
end
if isscalar(t)
    t = t(:);
end

% Initialize solution matrix
u = zeros(length(x), length(t));

% Compute analytical solution: u(x,t) = x(1-x) * E_α(-t^α)
for j = 1:length(t)
    for i = 1:length(x)
        if t(j) == 0
            % Initial condition: u(x,0) = x(1-x)
            u(i, j) = x(i) * (1 - x(i));
        else
            % Mittag-Leffler function E_α(-t^α)
            E_alpha = mlf(alpha, 1, -t(j)^alpha);
            u(i, j) = x(i) * (1 - x(i)) * E_alpha;
        end
    end
end

end
