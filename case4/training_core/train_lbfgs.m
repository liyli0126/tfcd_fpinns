function [net, loss_history] = train_lbfgs(net, params, maxIter)
% TRAIN_LBFGS - L-BFGS optimization phase for PINN training
%
% Inputs:
%   net     - neural network
%   params  - parameter struct
%   maxIter - maximum iterations
%
% Outputs:
%   net          - updated network
%   loss_history - loss history during L-BFGS phase

% Convert network parameters to vector
params_vec = net2vec(net);

% L-BFGS configuration
options = optimoptions('fmincon', ...
    'Algorithm', 'interior-point', ...
    'SpecifyObjectiveGradient', true, ...
    'Display', 'iter-detailed', ...
    'MaxIterations', maxIter, ...
    'HessianApproximation', 'lbfgs', ...
    'OptimalityTolerance', 1e-6);

% Optimization objective function
objfun = @(p) lbfgs_objective(p, net, params);

% Run L-BFGS
[opt_params, fval, ~, output] = fmincon(objfun, params_vec, [], [], [], [], [], [], [], options);

% Convert optimized parameters back to network
net = vec2net(opt_params, net);

% Get loss history
loss_history = fval;

end

