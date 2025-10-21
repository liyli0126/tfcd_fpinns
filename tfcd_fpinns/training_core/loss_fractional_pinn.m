function [loss, gradients, L_pde, L_bc, L_ic, L_sobolev] = loss_fractional_pinn(net, ...
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

% Compute source term: f(x,t) = x(1-x)[-E_α(-t^α) + t^(-α)/Γ(1-α)] + 0.002(1-2x)E_α(-t^α) + (1-2x)E_α(-t^α)
s_val = source_term(x_r, t_r, alpha, D, v);

% PDE residual with source term: D_t^α u - D ∂²u/∂x² + v ∂u/∂x - f(x,t) = 0
residual = D_alpha_u - D * u_xx + v * u_x - s_val;

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

% Compute Sobolev regularization loss (improved version) - COMMENTED OUT
% Temporarily disabled due to implementation issues causing worse results
constants = numerical_constants();
% if constants.sobolev_enabled && epoch >= constants.sobolev_epoch_start && epoch <= constants.sobolev_epoch_end
%     L_sobolev = compute_sobolev_loss(net, t_r, x_r, t_hist, x_hist, alpha, T, lambda_soe, theta, epoch, params);
% else
    L_sobolev = dlarray(0.0);
% end



% Standard Boundary Condition Loss
u_0 = predict(net, dlarray([t_0; x_0], "CB"));
u_L = predict(net, dlarray([t_L; x_L], "CB"));
L_bc = mean(u_0.^2, 'all') + mean(u_L.^2, 'all');

% Enhanced Initial Condition Loss using unified weight functions
u_ic = predict(net, dlarray([t_ic; x_ic], "CB"));
ic_target = x_ic .* (1 - x_ic);  % u(x,0) = x(1-x)

% Dynamic peak region emphasis in initial condition based on convection velocity
x_ic_val = extractdata(x_ic);
t_ic_val = extractdata(t_ic);

% Reduced peak region weighting to allow PDE-driven time evolution
ic_weights = ones(size(x_ic_val));

% Peak region: x ∈ [0.4, 0.6] - moderate weight for peak learning
peak_mask = (x_ic_val > 0.4) & (x_ic_val < 0.6);
ic_weights(peak_mask) = 3.2;  % Reduced from 5.0 to allow time evolution

% Near-peak regions: x ∈ [0.3, 0.4] and [0.6, 0.7] - reduced weight
near_peak_mask = ((x_ic_val > 0.3) & (x_ic_val <= 0.4)) | ...
                 ((x_ic_val >= 0.6) & (x_ic_val < 0.7));
ic_weights(near_peak_mask) = 2.0;  % Reduced from 2.5

% Very precise peak region: x ∈ [0.45, 0.55] - significantly reduced
precise_peak_mask = (x_ic_val > 0.45) & (x_ic_val < 0.55);
ic_weights(precise_peak_mask) = 4.5;  % Reduced from 8.0 to prevent peak locking

ic_weights = dlarray(ic_weights, "CB");

L_ic = mean(ic_weights .* (u_ic - ic_target).^2, 'all');

% Adaptive peak region protection loss
L_peak = calculate_adaptive_peak_loss(net, params, T);

% Use natural training without forced peak constraints

% Compose total loss
L_pde = stripdims(L_pde); 
L_bc = stripdims(L_bc); 
L_ic = stripdims(L_ic); 
L_peak = stripdims(L_peak);
L_sobolev = stripdims(L_sobolev);

% Simplified balanced training strategy - focus on stable PDE learning
if epoch < 100  % Early IC+BC learning
    loss = lambda.ic * L_ic + lambda.bc * L_bc;
elseif epoch < 300  % Gradual PDE introduction
    pde_intro = (epoch - 100) / 200;
    L_pde_clamped = min(L_pde, 2.0);  % Conservative PDE clamp
    loss = lambda.ic * L_ic + lambda.bc * L_bc + lambda.pde * (0.1 * pde_intro) * L_pde_clamped;
else
    % Full balanced training with enhanced PDE constraint
    L_pde_clamped = min(L_pde, 7.0);  % Increased PDE clamp for stronger time evolution
    loss = lambda.pde * L_pde_clamped + lambda.bc * L_bc + lambda.ic * L_ic;
end

gradients = dlgradient(loss, net.Learnables);

end