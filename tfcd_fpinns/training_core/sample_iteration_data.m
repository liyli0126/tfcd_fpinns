function [t_r, x_r, t_0, x_0, t_L, x_L, t_ic, x_ic] = sample_iteration_data(params, iter, net)
% SAMPLE_ITERATION_DATA - Sample training data for PINN iteration
%
% Provides training data for each iteration, including:
% - Collocation points (with optional RAR sampling)
% - Boundary condition points
% - Initial condition points
%
% Inputs:
%   params - parameter struct with training data
%   iter   - current iteration number
%   net    - neural network (for RAR sampling)
%
% Outputs:
%   t_r, x_r - collocation points for PDE residual
%   t_0, x_0 - left boundary points (x = 0)
%   t_L, x_L - right boundary points (x = L)
%   t_ic, x_ic - initial condition points (t = 0)

% Fixed boundary and initial condition points
t_0 = params.t_bc;
x_0 = params.x0;
t_L = params.t_bc;
x_L = params.xL;
t_ic = params.t_ic;
x_ic = params.x_ic;

% Generate collocation points
if should_use_rar(iter, params)
    % RAR sampling for better accuracy
    [t_r, x_r] = rar_sampling(params, net);
else
    % Regular random sampling
    t_r = params.T * rand(1, params.miniNr);
    x_r = params.L * rand(1, params.miniNr);
end

% Convert to dlarray format for deep learning
[t_r, x_r, t_0, x_0, t_L, x_L, t_ic, x_ic] = convert_to_dlarray(...
    t_r, x_r, t_0, x_0, t_L, x_L, t_ic, x_ic);

end

function use_rar = should_use_rar(iter, params)
% Check if RAR sampling should be used
use_rar = (iter > 1) && (mod(iter, params.rar_freq) == 0);
end

function [t_r, x_r, t_0, x_0, t_L, x_L, t_ic, x_ic] = convert_to_dlarray(...
    t_r, x_r, t_0, x_0, t_L, x_L, t_ic, x_ic)
% Convert all arrays to dlarray format
t_r = dlarray(t_r, "CB");
x_r = dlarray(x_r, "CB");
t_0 = dlarray(t_0, "CB");
x_0 = dlarray(x_0, "CB");
t_L = dlarray(t_L, "CB");
x_L = dlarray(x_L, "CB");
t_ic = dlarray(t_ic, "CB");
x_ic = dlarray(x_ic, "CB");
end
