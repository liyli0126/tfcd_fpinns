function [params, t_hist, x_hist, t_bc, x0, xL, t_ic, x_ic] = generate_training_data(params, net, iter)
% GENERATE_TRAINING_DATA - Generate and update training data for PINN
%
% This function serves two purposes:
% 1. Initial data generation (when called without net/iter)
% 2. Dynamic data updates with RAR sampling during training
%
% Inputs:
%   params - parameter struct
%   net    - neural network (optional, for RAR sampling)
%   iter   - current iteration (optional, for RAR sampling)
%
% Outputs:
%   params - updated parameter struct with sampling info
%   t_hist, x_hist - historical points (including RAR points)
%   t_bc, x0, xL   - boundary condition points
%   t_ic, x_ic     - initial condition points

% Check if this is initial generation or dynamic update
is_initial_generation = (nargin < 2) || isempty(net);

if is_initial_generation
    % Initial data generation
    [t_hist, x_hist, t_bc, x0, xL, t_ic, x_ic] = generate_initial_data(params);
else
    % Dynamic update with RAR sampling
    [t_hist, x_hist, t_bc, x0, xL, t_ic, x_ic] = update_training_data(params, net, iter);
end

% Store in params struct
params.t_hist = t_hist;
params.x_hist = x_hist;
params.t_bc = t_bc;
params.x0 = x0;
params.xL = xL;
params.t_ic = t_ic;
params.x_ic = x_ic;

end

function [t_hist, x_hist, t_bc, x0, xL, t_ic, x_ic] = generate_initial_data(params)
% Generate initial training data without RAR sampling

% Historical points (power-law time distribution)
N_hist = params.N_hist;
r = params.r;
t_hist = params.T * ((0:N_hist-1)/N_hist).^r;
x_hist = params.L * rand(1, N_hist);

% Sort by time for fractional derivative computation
[t_hist, idx] = sort(t_hist);
x_hist = x_hist(idx);

% Boundary condition points (equispaced in time)
N_bc = params.miniNb;
t_bc = linspace(0, params.T, N_bc);
x0 = zeros(1, N_bc);                % x = 0 boundary
xL = params.L * ones(1, N_bc);      % x = L boundary

% Initial condition points (Chebyshev nodes for better accuracy)
N_ic = 2 * params.miniNb;
x_ic = 0.5 * (1 - cos(pi * (0:N_ic-1) / (N_ic-1))) * params.L;
t_ic = zeros(1, N_ic);

end

function [t_hist, x_hist, t_bc, x0, xL, t_ic, x_ic] = update_training_data(params, net, iter)
% Update training data with RAR sampling

% Get current historical points
t_hist = params.t_hist;
x_hist = params.x_hist;

% RAR sampling in t > 0.05T region (every rar_freq steps)
if iter > 1 && mod(iter, params.rar_freq) == 0
    % Generate RAR candidates
    rar_num = 200; % number of candidate points
    rar_add = 50;  % number of points to add
    
    t_rar_cand = params.T * (0.05 + 0.95*rand(1, rar_num));
    x_rar_cand = params.L * rand(1, rar_num);
    
    % Convert to dlarray for network evaluation
    t_rar_dl = dlarray(t_rar_cand, "CB");
    x_rar_dl = dlarray(x_rar_cand, "CB");
    
    % Compute residuals for RAR selection
    [res_val, ~] = dlfeval(@rar_residual, net, t_rar_dl, x_rar_dl, ...
                           t_hist, x_hist, params.v, params.D, ...
                           params.alpha, params.T, params.lambda_soe, params.theta);
    
    % Handle numerical issues
    res_val(isnan(res_val) | isinf(res_val)) = -Inf;
    
    % Select points with highest residuals
    [~, idx] = maxk(res_val, min(rar_add, numel(t_rar_cand)));
    valid_idx = idx(idx <= numel(t_rar_cand));
    
    % Add RAR points to historical data
    t_hist = [t_hist, t_rar_cand(valid_idx)];
    x_hist = [x_hist, x_rar_cand(valid_idx)];
    
    % Store RAR information
    params.t_hist_rar = t_rar_cand(valid_idx);
    params.x_hist_rar = x_rar_cand(valid_idx);
    
    % Accumulate all RAR points
    if ~isfield(params, 't_hist_rar_all')
        params.t_hist_rar_all = [];
        params.x_hist_rar_all = [];
    end
    params.t_hist_rar_all = [params.t_hist_rar_all, t_rar_cand(valid_idx)];
    params.x_hist_rar_all = [params.x_hist_rar_all, x_rar_cand(valid_idx)];
    
    % Sort updated historical points
    [t_hist, idx] = sort(t_hist);
    x_hist = x_hist(idx);
else
    % No RAR update in this iteration
    params.t_hist_rar = [];
    params.x_hist_rar = [];
end

% Boundary and initial condition points remain the same
t_bc = params.t_bc;
x0 = params.x0;
xL = params.xL;
t_ic = params.t_ic;
x_ic = params.x_ic;

end