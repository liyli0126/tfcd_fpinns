function [net, loss_history] = train_two_stage_pinn(net, params)
% TRAIN_TWO_STAGE_PINN - Two-stage training for PINN
%
% Inputs:
%   net    - neural network
%   params - parameter struct
%
% Outputs:
%   net          - trained network
%   loss_history - training loss history

% Parameter parsing
maxIterAdam = params.maxIterAdam;   % Adam phase iterations
maxIterLBFGS = params.maxIterLBFGS; % L-BFGS phase iterations

% Generate initial training data
[params, params.t_hist, params.x_hist, params.t_bc, params.x0, params.xL, params.t_ic, params.x_ic] = ...
    generate_training_data(params);

% First phase: Adam optimization
fprintf('=== Phase 1: Adam Optimization (maxIter=%d) ===\n', maxIterAdam);
[net, loss_history_adam] = train_adam(net, params, maxIterAdam);

% Second phase: L-BFGS optimization
fprintf('\n=== Phase 2: L-BFGS Optimization (maxIter=%d) ===\n', maxIterLBFGS);
[net, loss_history_lbfgs] = train_lbfgs(net, params, maxIterLBFGS);

% Merge loss history
loss_history = [loss_history_adam; loss_history_lbfgs];

end

