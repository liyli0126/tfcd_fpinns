function plot_results(net, params, loss_history, adaptive_params)
% PLOT_RESULTS - Main function to plot PINN results and analysis
%
% Inputs:
%   net             - trained neural network
%   params          - parameter struct
%   loss_history    - training loss history
%   adaptive_params - adaptive activation parameters (optional)

% Generate validation points
x_val = linspace(0, params.L, 100);
t_val = linspace(0, params.T, 100);
[T_val, X_val] = meshgrid(t_val, x_val);  

% Display key parameters
disp(['D = ', num2str(params.D), ', v = ', num2str(params.v)]);
disp(['alpha = ', num2str(params.alpha), ', N_terms = ', num2str(params.N_terms)]);

% PINN predictions using dual pathway system
input_dl = dlarray([T_val(:)'; X_val(:)'], "CB");  % PINN input format: [t;x]

% Handle optional adaptive_params
if nargin < 4 || isempty(adaptive_params)
    % Standard forward pass
    u_pinn = predict(net, input_dl);
    fprintf('Using STANDARD activation pathway for visualization\n');
else
    % Dual pathway forward pass
    u_pinn = dual_forward(net, input_dl, adaptive_params);
    if adaptive_params.enabled
        fprintf('Using ADAPTIVE activation pathway for visualization\n');
    else
        fprintf('Using STANDARD activation pathway for visualization\n');
    end
end
u_pinn = reshape(extractdata(u_pinn), size(X_val));
u_pinn = max(u_pinn, 0); % All predictions are non-negative

% Analytical solution
u_analytical = analytical_solution(x_val, t_val, params.alpha, params.D, params.v, params.N_terms);

% Call individual plotting functions
plot_3d_comparison(X_val, T_val, u_analytical, u_pinn);
plot_2d_snapshots(x_val, t_val, u_analytical, u_pinn);
plot_error_analysis(X_val, T_val, u_analytical, u_pinn);
%plot_sampling_points(params);
plot_loss_function(loss_history);

end

