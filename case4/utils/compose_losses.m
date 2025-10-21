function loss = compose_losses(L_pde, L_bc, L_ic, lambda, epoch)
% COMPOSE_LOSSES - Dynamic loss balancing using exponential moving average
%
% Inputs:
%   L_pde, L_bc, L_ic - individual loss components
%   lambda - loss weights
%   epoch - current epoch
%
% Outputs:
%   loss - composed total loss

% Dynamic loss balancing using exponential moving average
persistent loss_means
if isempty(loss_means)
    loss_means = struct('pde',1, 'bc',1, 'ic',1);
end

% Get numerical constants
constants = numerical_constants();

decay = constants.weight_decay;
loss_means.pde = decay*loss_means.pde + (1-decay)*double(L_pde);
loss_means.bc = decay*loss_means.bc + (1-decay)*double(L_bc);
loss_means.ic = decay*loss_means.ic + (1-decay)*double(L_ic);

eps = constants.epsilon_loss;
weight_pde = 1/(loss_means.pde + eps);
weight_bc = 1/(loss_means.bc + eps);
weight_ic = 1/(loss_means.ic + eps);

total_weight = weight_pde + weight_bc + weight_ic;
weight_pde = weight_pde / total_weight;
weight_bc = weight_bc / total_weight;
weight_ic = weight_ic / total_weight;

dynamic_lambda = struct(...
    'pde', lambda.pde * weight_pde, ...
    'bc', lambda.bc * weight_bc, ...
    'ic', lambda.ic * weight_ic);

loss = dynamic_lambda.pde * L_pde + ...
       dynamic_lambda.bc * L_bc + ...
       dynamic_lambda.ic * L_ic;

end

