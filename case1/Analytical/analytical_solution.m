function u = analytical_solution(x, t, alpha, D, v, N_terms)
% ANALYTICAL_SOLUTION - Analytical solution for time-fractional advection-diffusion
% Optimized for moderate Péclet numbers (v/(2D) ≤ 100)
%
% Inputs:
%   x, t - spatial and temporal coordinates
%   alpha - fractional order
%   D - diffusion coefficient
%   v - velocity
%   N_terms - number of series terms
%
% Outputs:
%   u - analytical solution

% Parameter initialization and validation
if nargin < 1
    x = linspace(0, 1, 100);
end
if nargin < 2
    t = linspace(0, 1, 100);
end
if nargin < 3
    alpha = 0.8;
end
if nargin < 4
    D = 1.296e-2;
end
if nargin < 5
    v = 0.1296;  % convection velocity
end
if nargin < 6
    N_terms = 60; % series terms
end

% Initialize result matrix
u = zeros(length(x), length(t));

% Check numerical regime
v_over_2D = v / (2 * D);

% Precompute coefficients and eigenvalues
[c_n, lambda_n] = precompute_coefficients(N_terms, v, D);

% Check coefficient stability (simple check)
stable_coeffs = isfinite(c_n) & (abs(c_n) > 1e-20);

% Compute solution
u = compute_solution(x, t, alpha, v, D, c_n, lambda_n, N_terms, stable_coeffs);

% Ensure boundary conditions
u(1, :) = 0;      % u(0,t) = 0
u(end, :) = 0;    % u(1,t) = 0

end

function [c_n, lambda_n] = precompute_coefficients(N, v, D)
% Precompute coefficients c_n and eigenvalues lambda_n using correct formulas
% Simplified version without complex smoothing

c_n = zeros(1, N);
lambda_n = zeros(1, N);
a = v/(2*D); % compute parameter a

% Compute all coefficients directly
for n = 1:N
    if n == 1
        % n=1 case
        numerator = 32 * pi^2 * D^3 * (1 - exp(-a));
        denominator = v * (v^2 + 16 * pi^2 * D^2);
        c_n(1) = numerator / denominator;
    else
        % n>=2 case
        term1 = (-a) * (exp(-a) * (-1)^(n+1) - 1);
        numerator = 4 * n * pi^2;
        denominator1 = (a^2 + (n-1)^2 * pi^2);
        denominator2 = (a^2 + (n+1)^2 * pi^2);
        c_n(n) = term1 * numerator / (denominator1 * denominator2);
    end
    
    % Compute eigenvalues
    lambda_n(n) = v^2/(4*D) + D * (n*pi)^2;
    
end

end

function u = compute_solution(x, t, alpha, v, D, c_n, lambda_n, N_terms, stable_coeffs)
% Compute analytical solution

u = zeros(length(x), length(t));
a = v/(2*D);

for j = 1:length(t)
    for i = 1:length(x)
        sum_val = 0;
        
        for n = 1:N_terms
            if stable_coeffs(n)
                % Compute Mittag-Leffler function directly
                E_alpha = mlf(alpha, 1, -lambda_n(n) * t(j)^alpha);
                
                % Standard computation (for moderate Péclet numbers)
                term = c_n(n) * exp(a * x(i)) * sin(n * pi * x(i)) * E_alpha;
                
                % Check for numerical issues
                if isfinite(term)
                    sum_val = sum_val + term;
                end
            end
        end
        
        u(i, j) = sum_val;
    end
end

end


