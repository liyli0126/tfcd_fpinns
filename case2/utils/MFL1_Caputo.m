function D_alpha_U = MFL1_Caputo(u_hist, t_hist, u_curr, t_curr, alpha, T, lambda_soe, theta)
% MFL1_CAPUTO - Modified Fast L1 scheme for Caputo fractional derivative
%
% Inputs:
%   u_hist, t_hist - historical solution values and times
%   u_curr, t_curr - current solution value and time
%   alpha - fractional order
%   T - total time
%   lambda_soe, theta - SOE parameters
%
% Outputs:
%   D_alpha_U - fractional derivative value

% Merge history and current point
t_all = [t_hist, t_curr];
u_all = [u_hist, u_curr];
[t_sorted, idx] = sort(t_all);
u_sorted = u_all(idx);
n = find(t_sorted == t_curr, 1, 'last');  % Index of current point
t_n = t_curr;

% Get numerical constants for stability
constants = numerical_constants();
epsilon = constants.epsilon; % Small value for numerical stability

if n < 3
    if n < 2
        D_alpha_U = 0;
        return;
    end
    dt = diff(t_sorted(1:n));
    dt(dt==0) = epsilon;
    du = diff(u_sorted(1:n));
    w = zeros(1, n-1);
    for k = 1:n-1
        tau_k = max(t_n - t_sorted(k), epsilon);
        tau_k1 = max(t_n - t_sorted(k+1), epsilon);
        w(k) = (tau_k^(1 - alpha) - tau_k1^(1 - alpha)) / gamma(2 - alpha);
    end
    % Correct L1 formula: D_alpha_U = sum(w .* (du ./ dt))
    % du is the function difference, dt is the time step
    % We need to normalize by dt to approximate f'(τ)
    % This ensures mathematical consistency with the continuous definition
    D_alpha_U = sum(w .* (du ./ dt));
    if isnan(D_alpha_U) || isinf(D_alpha_U)
        D_alpha_U = 0;
    end
    return;
end

% Current term
tau_k = max(t_n - t_sorted(n-1), epsilon);
dt_current = t_sorted(n) - t_sorted(n-1);  % Current time step
% Current term needs time step normalization: (du/dt) * integral
d_current = (u_sorted(n) - u_sorted(n-1)) * (tau_k^(1-alpha)) / (gamma(2-alpha) * dt_current);

% Transition term
tau_k0 = max(t_n - t_sorted(n-2), epsilon);
tau_k1 = max(t_n - t_sorted(n-1), epsilon);
d_transition = 0;
for j = 1:length(lambda_soe)
    denom = lambda_soe(j) * tau_k1 / T + epsilon;
    term = theta(j) * (exp(-lambda_soe(j)*tau_k1/T) - exp(-lambda_soe(j)*tau_k0/T)) / denom;
    d_transition = d_transition - term * u_sorted(n-1);
end
d_transition = d_transition * T^(-alpha)/gamma(1-alpha);

% History term
d_history = 0;
for j = 1:length(lambda_soe)
    sum_hist = 0;
    for m = 2:n-1
        delta_u = u_sorted(m) - u_sorted(m-1);
        tau1 = max(t_n - t_sorted(m), epsilon);
        tau2 = max(t_n - t_sorted(m-1), epsilon);
        denom = lambda_soe(j) * (t_sorted(m) - t_sorted(m-1)) / T + epsilon;
        exp1 = exp(-lambda_soe(j)*tau1/T);
        exp2 = exp(-lambda_soe(j)*tau2/T);
        sum_hist = sum_hist + delta_u * (exp1 - exp2) / denom;
    end
    d_history = d_history + theta(j) * sum_hist;
end
d_history = d_history * T^(-alpha)/gamma(1-alpha);

D_alpha_U = d_current + d_transition + d_history;

% Final check for NaN/Inf using tolerance constants
if isnan(D_alpha_U) || abs(D_alpha_U) > constants.tolerance_inf
    D_alpha_U = 0;
end

end
