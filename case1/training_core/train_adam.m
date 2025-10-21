function [net, loss_history] = train_adam(net, params, maxIter)
% TRAIN_ADAM - Adam optimization phase for PINN training
%
% Inputs:
%   net     - neural network
%   params  - parameter struct
%   maxIter - maximum iterations
%
% Outputs:
%   net          - updated network
%   loss_history - loss history during Adam phase

% Initialize Adam parameters
avgGrad = [];
avgSqGrad = [];
loss_history = zeros(maxIter, 1);

for iter = 1:maxIter
    % Dynamic learning rate decay
    lr = params.lr * exp(-iter/params.lr_decay_steps);
    
    % Data sampling (including RAR logic)
    [t_r, x_r, t_0, x_0, t_L, x_L, t_ic, x_ic] = sample_iteration_data(params, iter, net);
    
    % Compute loss and gradients
    [loss, grads, ~, ~, ~] = dlfeval(@loss_fractional_pinn, net, ...
        t_r, x_r, params.t_hist, params.x_hist, ...
        t_0, x_0, t_L, x_L, t_ic, x_ic, ...
        params.v, params.D, params.alpha, params.lambda, params.T, ...
        params.lambda_soe, params.theta, iter, params.L, params);
    
    % Adam update
    [net, avgGrad, avgSqGrad] = adamupdate(net, grads, avgGrad, avgSqGrad, ...
                                           iter, lr, params.beta1, params.beta2, params.eps);
    
    % Gradient clipping
    grads = dlupdate(@(g) min(max(g, -1e3), 1e3), grads);
    
    % Record loss
    loss_history(iter) = extractdata(loss);
    
    % Progress display
    if mod(iter,100)==0 || iter==1
        fprintf('Iter %4d | Loss = %.3e | lr = %.1e\n', iter, loss_history(iter), lr);
    end
    
    % RAR resampling
    if mod(iter, params.rar_freq) == 0
        [params, params.t_hist, params.x_hist, params.t_bc, params.x0, params.xL, params.t_ic, params.x_ic] = ...
            generate_training_data(params, net, iter);
    end
end

end