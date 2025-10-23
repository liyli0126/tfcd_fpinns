function plot_results(net, params, loss_history)
% PLOT_RESULTS - Main function to plot PINN results and analysis
%
% Inputs:
%   net          - trained neural network
%   params       - parameter struct
%   loss_history - training loss history

% Generate validation points
x_val = linspace(0, params.L, 100);
t_val = linspace(0, params.T, 100);
[T_val, X_val] = meshgrid(t_val, x_val);  

% Debug output
disp('=== Analytical Solution Parameters ===');
disp(['D = ', num2str(params.D), ', v = ', num2str(params.v)]);


% PINN predictions
input_dl = dlarray([T_val(:)'; X_val(:)'], "CB");  % PINN input format: [t;x]
u_pinn = predict(net, input_dl);
u_pinn = reshape(extractdata(u_pinn), size(X_val));
u_pinn = max(u_pinn, 0); % All predictions are non-negative

% Analytical solution
u_analytical = arrayfun(@(x,t) analytical_solution_case2(x,t,params.alpha), X_val, T_val);

% Call individual plotting functions
plot_3d_comparison(X_val, T_val, u_analytical, u_pinn);
plot_2d_snapshots(x_val, t_val, u_analytical, u_pinn);
plot_error_analysis(X_val, T_val, u_analytical, u_pinn);
plot_sampling_points(params);
plot_loss_function(loss_history);

end

