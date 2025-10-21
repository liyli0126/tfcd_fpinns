function [net, loss_history] = train_two_stage_pinn(net, params)
% TRAIN_TWO_STAGE_PINN - Two-stage PINN training with Adam + L-BFGS
%
% Implements a two-stage training strategy:
% Stage 1: Adam optimizer for global exploration
% Stage 2: L-BFGS for fine-tuning and convergence
%
% Inputs:
%   net    - neural network to train
%   params - training parameters
%
% Outputs:
%   net          - trained network
%   loss_history - combined loss history from both stages

% Extract training parameters
maxIterAdam = params.maxIterAdam;   
maxIterLBFGS = params.maxIterLBFGS; 

% Initialize training data
params = generate_training_data(params);

% Stage 1: Adam optimization (global exploration)
fprintf('=== Stage 1: Adam Optimization (%d iterations) ===\n', maxIterAdam);
[net, loss_adam] = train_adam(net, params, maxIterAdam);

% Stage 2: L-BFGS optimization (fine-tuning)
fprintf('=== Stage 2: L-BFGS Optimization (%d iterations) ===\n', maxIterLBFGS);
[net, loss_lbfgs] = train_lbfgs(net, params, maxIterLBFGS);

% Combine loss histories
loss_history = [loss_adam; loss_lbfgs];

end

