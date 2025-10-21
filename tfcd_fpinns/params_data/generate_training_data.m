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

% Use balanced point generation (early time + late downstream emphasis)
[x_hist, t_hist] = generate_balanced_points(params);

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
    % Generate RAR candidates with minimum spacing constraint
    rar_num = 200; % number of candidate points
    rar_add = 50;  % number of points to add
    t_min_spacing = 1e-4;  % Minimum time spacing for numerical stability
    
    % Generate time candidates with minimum spacing constraint
    t_rar_cand = [];
    x_rar_cand = [];
    
    % Generate candidates in batches to ensure minimum spacing
    batch_size = 20;
    num_batches = ceil(rar_num / batch_size);
    
    for batch = 1:num_batches
        batch_candidates = min(batch_size, rar_num - length(t_rar_cand));
        
        % Generate time candidates for this batch
        t_batch = params.T * (0.05 + 0.95*rand(1, batch_candidates));
        x_batch = params.L * rand(1, batch_candidates);
        
        % Filter candidates to ensure minimum spacing from existing points
        valid_mask = true(1, batch_candidates);
        for i = 1:batch_candidates
            % Check minimum spacing from all existing historical points
            if any(abs(t_hist - t_batch(i)) < t_min_spacing)
                valid_mask(i) = false;
            end
            % Check minimum spacing from already selected candidates
            if any(abs(t_rar_cand - t_batch(i)) < t_min_spacing)
                valid_mask(i) = false;
            end
        end
        
        % Add valid candidates
        t_rar_cand = [t_rar_cand, t_batch(valid_mask)];
        x_rar_cand = [x_rar_cand, x_batch(valid_mask)];
        
        % Stop if we have enough candidates
        if length(t_rar_cand) >= rar_num
            break;
        end
    end
    
    % Ensure we have at least some candidates
    if length(t_rar_cand) < 10
        % Fallback: generate candidates with larger minimum spacing
        t_min_spacing_fallback = 5e-4;
        t_rar_cand = params.T * (0.05 + 0.95*rand(1, min(50, rar_num)));
        x_rar_cand = params.L * rand(1, min(50, rar_num));
        
        % Filter with fallback spacing
        valid_mask = true(1, length(t_rar_cand));
        for i = 1:length(t_rar_cand)
            if any(abs(t_hist - t_rar_cand(i)) < t_min_spacing_fallback)
                valid_mask(i) = false;
            end
        end
        t_rar_cand = t_rar_cand(valid_mask);
        x_rar_cand = x_rar_cand(valid_mask);
    end
    
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
    
    % Add RAR points to historical data with enhanced minimum spacing constraint
    % Note: t_min_spacing already defined above
    
    % Filter RAR points to ensure minimum spacing from existing points
    t_rar_filtered = [];
    x_rar_filtered = [];
    
    for i = 1:length(valid_idx)
        t_candidate = t_rar_cand(valid_idx(i));
        x_candidate = x_rar_cand(valid_idx(i));
        
        % Check minimum spacing from all existing historical points
        if all(abs(t_hist - t_candidate) >= t_min_spacing)
            % Also check minimum spacing from already selected RAR points
            if all(abs(t_rar_filtered - t_candidate) >= t_min_spacing)
                t_rar_filtered = [t_rar_filtered, t_candidate];
                x_rar_filtered = [x_rar_filtered, x_candidate];
            end
        end
    end
    
    % Add filtered RAR points to historical data
    t_hist = [t_hist, t_rar_filtered];
    x_hist = [x_hist, x_rar_filtered];
    
    % Debug information
    if length(t_rar_filtered) > 0
        fprintf('RAR sampling: Added %d points (min spacing: %.1e)\n', length(t_rar_filtered), t_min_spacing);
    else
        fprintf('RAR sampling: No points added (all candidates violate min spacing constraint)\n');
    end
    
    % Store RAR information
    params.t_hist_rar = t_rar_filtered;
    params.x_hist_rar = x_rar_filtered;
    
    % Accumulate all RAR points
    if ~isfield(params, 't_hist_rar_all')
        params.t_hist_rar_all = [];
        params.x_hist_rar_all = [];
    end
    params.t_hist_rar_all = [params.t_hist_rar_all, t_rar_filtered];
    params.x_hist_rar_all = [params.x_hist_rar_all, x_rar_filtered];
    
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