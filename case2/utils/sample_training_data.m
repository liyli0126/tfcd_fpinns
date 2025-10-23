function [t_r, x_r, t_0, x_0, t_L, x_L, t_ic, x_ic] = sample_training_data(params, iter, net)
% SAMPLE_TRAINING_DATA - Sample training data for each iteration
%
% Inputs:
%   params - parameter struct
%   iter   - current iteration
%   net    - neural network (for RAR)
%
% Outputs:
%   t_r, x_r - collocation points
%   t_0, x_0 - left boundary points
%   t_L, x_L - right boundary points
%   t_ic, x_ic - initial condition points

% Fixed boundary/initial condition points
t_0 = params.t_bc;
x_0 = params.x0;
t_L = params.t_bc;
x_L = params.xL;
t_ic = params.t_ic;
x_ic = params.x_ic;

% Dynamically generate internal points (including RAR)
if iter == 1 || mod(iter, params.rar_freq) ~= 0
    % Strategic non-random sampling with gentle epoch-aware adjustments
    [t_r, x_r] = strategic_sampling(params, iter);
else
    % RAR resampling with enhanced candidate generation
    [t_r, x_r] = rar_sampling_enhanced(params, net, iter);
end

% Convert to dlarray format
t_r = dlarray(t_r, "CB");
x_r = dlarray(x_r, "CB");
t_0 = dlarray(t_0, "CB");
x_0 = dlarray(x_0, "CB");
t_L = dlarray(t_L, "CB");
x_L = dlarray(x_L, "CB");
t_ic = dlarray(t_ic, "CB");
x_ic = dlarray(x_ic, "CB");

end
