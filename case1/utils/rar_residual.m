function [res_val, x_r_out] = rar_residual(net, t_r, x_r, t_hist, x_hist, v, D, alpha, T, lambda_soe, theta)
% RAR_RESIDUAL - Compute residual for Residual Adaptive Refinement (RAR)
%
% Inputs:
%   net, t_r, x_r - network and current points
%   t_hist, x_hist - historical points
%   v, D, alpha - PDE parameters
%   T - total time
%   lambda_soe, theta - SOE parameters
%
% Outputs:
%   res_val - residual values
%   x_r_out - corresponding x coordinates

% All inputs must be dlarray, shape [1,N] and "CB"
input_dl = dlarray([t_r; x_r], "CB");
U_r = predict(net, input_dl);

U_x = dlgradient(sum(U_r, 'all'), x_r, 'EnableHigherDerivatives', true);
U_xx = dlgradient(sum(U_r, 'all'), x_r, 'EnableHigherDerivatives', true, 'EnableHigherDerivatives', true);

% Historical prediction
tx_hist = dlarray([t_hist; x_hist], "CB");
u_hist = predict(net, tx_hist);
u_hist = extractdata(u_hist);

t_curr = extractdata(t_r);
u_curr = extractdata(U_r);

D_alpha_U = zeros(1, length(t_curr));
for i = 1:length(t_curr)
    D_alpha_U(i) = MFL1_Caputo(u_hist, t_hist, u_curr(i), t_curr(i), alpha, T, lambda_soe, theta);
end
D_alpha_U = dlarray(D_alpha_U, "CB");

residual = D_alpha_U + v * U_x - D * U_xx;
res_val = abs(extractdata(residual));
x_r_out = extractdata(x_r);

end

