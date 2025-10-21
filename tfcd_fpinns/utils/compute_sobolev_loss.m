function L_sobolev = compute_sobolev_loss(net, t_r, x_r, t_hist, x_hist, alpha, T, lambda_soe, theta, epoch, params)
% COMPUTE_SOBOLEV_LOSS - Compute improved Sobolev loss
%
% Implements Sobolev regularization for spatial and fractional time derivatives
% Uses MFL1_Caputo for fractional derivative computation
%
% Inputs:
%   net - neural network
%   t_r, x_r - current time-space coordinates
%   t_hist, x_hist - historical time-space coordinates
%   alpha - fractional order
%   T - total time
%   lambda_soe, theta - SOE parameters
%   epoch - current training epoch
%   params - parameter struct
%
% Outputs:
%   L_sobolev - Sobolev loss

% Get numerical constants
constants = numerical_constants();

% Sobolev loss parameters
lambda_s0 = constants.sobolev_lambda_s0;  % Initial weight
beta = constants.sobolev_beta;            % Decay parameter

% Compute network output
u_r = predict(net, [t_r; x_r]);

% Compute spatial derivative u_x
u_x = dlgradient(sum(u_r, 'all'), x_r, 'EnableHigherDerivatives', true);

% Compute fractional time derivative D_α u
% Using MFL1_Caputo method
tx_hist = dlarray([t_hist; x_hist], "CB");
u_hist = predict(net, tx_hist);
u_hist = extractdata(u_hist);  % Extract data for MFL1_Caputo
t_curr = extractdata(t_r);
u_curr = extractdata(u_r);

D_alpha_u = zeros(1, length(t_curr));
for i = 1:length(t_curr)
    D_alpha_u(i) = MFL1_Caputo(u_hist, t_hist, u_curr(i), t_curr(i), alpha, T, lambda_soe, theta);
end
D_alpha_u = dlarray(D_alpha_u, "CB");

% Adaptive weight: λ_s = λ_s0 * exp(-β * epoch)
lambda_s = lambda_s0 * exp(-beta * epoch);

% Sobolev loss components
% L_H1_x: L2 norm of spatial derivative
L_H1_x = mean(u_x.^2, 'all');

% L_Halpha: L2 norm of fractional time derivative
L_Halpha = mean(D_alpha_u.^2, 'all');

% Total Sobolev loss
L_sobolev = lambda_s * (L_H1_x + L_Halpha);

% Numerical stability handling
L_sobolev = stripdims(L_sobolev);
if isnan(L_sobolev) || isinf(L_sobolev)
    L_sobolev = dlarray(0.0);
end

end
