function [loss, grad] = lbfgs_objective(params_vec, net_template, params)
% LBFGS_OBJECTIVE - Objective function for L-BFGS optimization
%
% Inputs:
%   params_vec   - parameter vector
%   net_template - network template
%   params       - parameter struct
%
% Outputs:
%   loss - objective value
%   grad - gradient vector

net = vec2net(params_vec, net_template);
[t_r, x_r, t_0, x_0, t_L, x_L, t_ic, x_ic] = sample_training_data(params, 1, net);

[loss, grads, ~, ~, ~, ~] = dlfeval(@loss_fractional_pinn, net, ...
    t_r, x_r, params.t_hist, params.x_hist, t_0, x_0, t_L, x_L, t_ic, x_ic, ...
    params.v, params.D, params.alpha, params.lambda, params.T, ...
    params.lambda_soe, params.theta, inf, params.L, params);

% Convert gradients to vector
grad = [];
for i = 1:height(grads)
    value = grads.Value{i};
    if iscell(value)
        value = value{1};
    end
    grad = [grad; extractdata(value(:))];
end
grad = double(grad);
grad = grad(:);

% Key correction
loss = double(extractdata(loss));

% Debug output
assert(isa(loss, 'double') && isscalar(loss), 'loss must be scalar double');
assert(isa(grad, 'double') && isvector(grad), 'grad must be vector double');

end

