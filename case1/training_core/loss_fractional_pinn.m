function [loss, gradients, L_pde, L_bc, L_ic] = loss_fractional_pinn(net, ...
    t_r, x_r, t_hist, x_hist, t_0, x_0, t_L, x_L, t_ic, x_ic, v, D, alpha, lambda, T, lambda_soe, theta, epoch, L, params)
% LOSS_FRACTIONAL_PINN - Compute loss for time-fractional advection-diffusion PINN
%
% Inputs:
%   net, t_r, x_r, t_hist, x_hist, t_0, x_0, t_L, x_L, t_ic, x_ic - training data
%   v, D, alpha - PDE parameters
%   lambda - loss weights
%   T - total time
%   lambda_soe, theta - SOE parameters
%   epoch - current epoch
%   L - domain length
%   params - parameter struct (for weight functions)
%
% Outputs:
%   loss - total loss
%   gradients - network gradients
%   L_pde, L_bc, L_ic - individual loss components

% All loss calculations must remain in the dlarray domain for autodiff
u_r = predict(net, [t_r; x_r]);
u_x = dlgradient(sum(u_r, 'all'), x_r, 'EnableHigherDerivatives', true);
u_xx = dlgradient(sum(u_r, 'all'), x_r, 'EnableHigherDerivatives', true, 'EnableHigherDerivatives', true);

% Compute fractional time derivative using modified fast L1 scheme
tx_hist = dlarray([t_hist; x_hist], "CB");
u_hist = predict(net, tx_hist);
u_hist = extractdata(u_hist);  % Only for MFL1_Caputo, not for loss
t_curr = extractdata(t_r);
u_curr = extractdata(u_r);
D_alpha_u = zeros(1, length(t_curr));
for i = 1:length(t_curr)
    D_alpha_u(i) = MFL1_Caputo(u_hist, t_hist, u_curr(i), t_curr(i), alpha, T, lambda_soe, theta);
end
D_alpha_u = dlarray(D_alpha_u, "CB");

% PDE residual: fractional time derivative + advection - diffusion
residual = D_alpha_u + v * u_x - D * u_xx;

% Enhanced adaptive local weighting using unified weight functions
local_weight = weight_functions('local_weighting', params, x_r, residual);

% Enhanced time weighting using unified weight functions
t_weight = weight_functions('time_weighting', params, t_r);
L_pde = mean(local_weight .* t_weight .* (residual.^2), 'all');

% Long-term prediction penalty using unified weight functions
% TEMPORARILY DISABLED for debugging - using simple weights
% longterm_weights = weight_functions('longterm_weighting', params, t_r);
% L_longterm = 3.0 * mean(longterm_weights .* (residual.^2), 'all');

% Simple long-term penalty
t_r_val = extractdata(t_r);
longterm_mask = (t_r_val > T*0.7);
if any(longterm_mask)
    L_longterm = 3.0 * mean(residual(longterm_mask).^2, 'all');
else
    L_longterm = dlarray(0.0);
end

% Get numerical constants
constants = numerical_constants();

% Enhanced Sobolev regularization
x_perturbed = x_r + constants.perturbation_factor*L*(rand(size(x_r))-0.5);
u_r_perturbed = predict(net, [t_r; x_perturbed]);
u_x_perturbed = dlgradient(sum(u_r_perturbed, 'all'), x_perturbed);
derivative_diff = abs(u_x - u_x_perturbed);
residual_weight = abs(residual).^2;
residual_weight = residual_weight / mean(residual_weight, 'all'); % normalize
L_sobolev = mean(residual_weight .* derivative_diff.^2, 'all');



% Boundary Condition Loss
u_0 = predict(net, dlarray([t_0; x_0], "CB"));
u_L = predict(net, dlarray([t_L; x_L], "CB"));
L_bc = mean(u_0.^2, 'all') + mean(u_L.^2, 'all');

% Enhanced Initial Condition Loss using unified weight functions
u_ic = predict(net, dlarray([t_ic; x_ic], "CB"));
ic_target = sin(pi * x_ic);

% Dynamic peak region emphasis in initial condition based on convection velocity
x_ic_val = extractdata(x_ic);
t_ic_val = extractdata(t_ic);

% Calculate expected peak position for each initial condition point
ic_weights = ones(size(x_ic_val));
for i = 1:length(x_ic_val)
    % For initial condition, peak starts at 0.5 but will move based on convection
    expected_peak_x = 0.5 + (params.v / params.D) * t_ic_val(i) * 0.05;
    expected_peak_x = max(0.1, min(0.9, expected_peak_x));
    
    % Calculate distance to expected peak position
    distance_to_peak = abs(x_ic_val(i) - expected_peak_x);
    
    % Assign weights based on distance to expected peak
    if distance_to_peak < 0.1 * params.L
        ic_weights(i) = 3.0;  % Higher weight for peak region
    elseif distance_to_peak < 0.2 * params.L
        ic_weights(i) = 2.0;  % Medium weight for near-peak region
    end
end
ic_weights = dlarray(ic_weights, "CB");

L_ic = mean(ic_weights .* (u_ic - ic_target).^2, 'all');

% Adaptive peak region protection loss
L_peak = calculate_adaptive_peak_loss(net, params, T);

% Compose total loss
L_pde = stripdims(L_pde); 
L_bc = stripdims(L_bc); 
L_ic = stripdims(L_ic); 
L_sobolev = stripdims(L_sobolev);
L_peak = stripdims(L_peak);

% Enhanced loss composition with peak protection and convection constraint
if epoch < 200
    loss = 15 * L_ic + 5 * L_peak; % Strong weight on IC and peak loss
else
    loss = compose_losses(L_pde + L_longterm, L_bc, L_ic, lambda, epoch) + ...
           min(1.0, epoch/500) * 1.5 * L_sobolev + ...  % Sobolev regularization
           2.0 * L_peak;  % Peak protection loss
end

gradients = dlgradient(loss, net.Learnables);

end