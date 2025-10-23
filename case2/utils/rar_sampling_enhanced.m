function [t_r, x_r] = rar_sampling_enhanced(params, net, iter)
% RAR_SAMPLING_ENHANCED - Enhanced RAR sampling with strategic candidate generation
%
% Inputs:
%   params - parameter struct
%   net    - neural network
%   iter   - current iteration number
%
% Outputs:
%   t_r, x_r - selected collocation points

% Parameters for RAR sampling
rar_candidates = 3 * params.miniNr;  % 3x candidates for selection
rar_add = params.miniNr;             % number of points to add

% Strategic RAR candidate generation
t_rar_cand = [];
x_rar_cand = [];

% 1. Time-stratified candidates (better coverage across time domain)
time_segments = 5;
for seg = 1:time_segments
    t_start = (seg - 1) * params.T / time_segments;
    t_end = seg * params.T / time_segments;
    n_cand_seg = round(rar_candidates / time_segments);
    t_seg_cand = t_start + (t_end - t_start) * rand(1, n_cand_seg);
    x_seg_cand = params.L * rand(1, n_cand_seg);
    t_rar_cand = [t_rar_cand, t_seg_cand];
    x_rar_cand = [x_rar_cand, x_seg_cand];
end

% 2. Optimized boundary sampling (reduced for Dirichlet BC u=0)
% Since boundary conditions are u(0,t) = u(L,t) = 0, we need fewer boundary points
n_boundary_cand = round(rar_candidates * 0.03);  % Reduced from 0.1 to 0.03 (3%)
if n_boundary_cand > 0
    t_boundary_cand = params.T * rand(1, n_boundary_cand);
    
    % Smart boundary sampling: focus on regions where boundary effects propagate
    % For advection-diffusion, boundary effects are more important at early times
    % and in regions where the solution has significant gradients
    
    % Early time boundary sampling (t < 0.3T)
    n_early_boundary = round(n_boundary_cand * 0.6);
    t_early_boundary = params.T * (0.05 + 0.25 * rand(1, n_early_boundary));
    
    % Late time boundary sampling (t > 0.7T) - minimal sampling
    n_late_boundary = round(n_boundary_cand * 0.2);
    t_late_boundary = params.T * (0.7 + 0.3 * rand(1, n_late_boundary));
    
    % Middle time boundary sampling
    n_mid_boundary = n_boundary_cand - n_early_boundary - n_late_boundary;
    t_mid_boundary = params.T * (0.3 + 0.4 * rand(1, n_mid_boundary));
    
    % Combine boundary time points
    t_boundary_cand = [t_early_boundary, t_mid_boundary, t_late_boundary];
    
    % Spatial distribution: focus on boundary regions with potential gradients
    % Avoid sampling exactly at x=0 and x=L where u=0
    % FIXED: Ensure integer division for rand function
    n_left_boundary = round(n_boundary_cand / 2);
    n_right_boundary = n_boundary_cand - n_left_boundary;
    
    x_left_boundary = params.L * (0.02 + 0.08 * rand(1, n_left_boundary));
    x_right_boundary = params.L * (0.9 + 0.1 * rand(1, n_right_boundary));
    x_boundary_cand = [x_left_boundary, x_right_boundary];
    
    t_rar_cand = [t_rar_cand, t_boundary_cand];
    x_rar_cand = [x_rar_cand, x_boundary_cand];
end

% 3. Critical time region candidates (focus on middle time where dynamics are complex)
n_critical_cand = round(rar_candidates * 0.12);  % Increased from 0.1 to 0.12
t_critical_cand = params.T * (0.3 + 0.4 * rand(1, n_critical_cand)); % Focus on middle time
x_critical_cand = params.L * rand(1, n_critical_cand);
t_rar_cand = [t_rar_cand, t_critical_cand];
x_rar_cand = [x_rar_cand, x_critical_cand];

% 4. Late time region candidates (important for long-term prediction)
n_late_cand = round(rar_candidates * 0.20);  % Reduced from 0.25 to 0.20
t_late_cand = params.T * (0.6 + 0.4 * rand(1, n_late_cand)); % Focus on late time
x_late_cand = params.L * rand(1, n_late_cand);
t_rar_cand = [t_rar_cand, t_late_cand];
x_rar_cand = [x_rar_cand, x_late_cand];

% 5. Interior region sampling (compensate for reduced boundary sampling)
n_interior_cand = round(rar_candidates * 0.15);  % New: 15% for interior regions
% Focus on interior regions where solution gradients are significant
x_interior_cand = params.L * (0.1 + 0.8 * rand(1, n_interior_cand));
t_interior_cand = params.T * rand(1, n_interior_cand);
t_rar_cand = [t_rar_cand, t_interior_cand];
x_rar_cand = [x_rar_cand, x_interior_cand];

% 6. Peak region sampling (focus on dynamic peak location)
n_peak_cand = round(rar_candidates * 0.10);  % 10% for peak region
if n_peak_cand > 0
    % Dynamic peak location
    if isfield(params, 'peak_region') && isfield(params.peak_region, 'dynamic') && params.peak_region.dynamic
        % Calculate peak location at current time
        t_current = iter / params.epochs * params.T;
        [peak_x, ~] = find_peak_location_singular(t_current, params.alpha, params.peak_detection.n_points);
    else
        % Use default peak location
        peak_x = params.peak_region.center;
        t_current = 0.5;  % Default time point
    end
    
    % Adjusted for dynamic peak location
    [region_width, ~] = get_adaptive_sampling_params(t_current, params);
    
    % Generate candidate points near the peak with enhanced density
    % Main sampling region: peak_x ± region_width
    n_main_peak_cand = round(n_peak_cand * 0.7);  % 70% in main region
    x_main_peak_cand = peak_x + region_width * (rand(1, n_main_peak_cand) - 0.5);
    
    % Dense sampling region: peak_x ± 0.05 (more concentrated sampling)
    n_dense_peak_cand = n_peak_cand - n_main_peak_cand;  % 30% in dense region
    dense_region_width = 0.05;  % ±0.05 dense region
    x_dense_peak_cand = peak_x + dense_region_width * (rand(1, n_dense_peak_cand) - 0.5);
    
    % Combine sampling points
    x_peak_cand = [x_main_peak_cand, x_dense_peak_cand];
    x_peak_cand = max(0, min(1, x_peak_cand));  % Ensure in [0,1] range
% Time distribution: enhanced emphasis on ultra-early and early times for peak region
n_ultra_early_cand = round(n_peak_cand * 0.4);  % 40% ultra-early
n_early_cand = round(n_peak_cand * 0.35);      % 35% early
n_mid_cand = round(n_peak_cand * 0.15);        % 15% mid
n_late_cand = n_peak_cand - n_ultra_early_cand - n_early_cand - n_mid_cand;  % Remaining for late

% Ultra-early sampling using exponential distribution
if n_ultra_early_cand > 0
    r_ultra_cand = rand(1, n_ultra_early_cand);
    t_peak_ultra_early = params.T * 0.001 * (1 - exp(-5 * r_ultra_cand));  % Exponential distribution
else
    t_peak_ultra_early = [];
end

% Early sampling using dense uniform distribution
if n_early_cand > 0
    t_peak_early = params.T * (0.001 + 0.009 * rand(1, n_early_cand));  % Dense uniform in [0.001T, 0.01T]
else
    t_peak_early = [];
end

% Middle and late sampling
if n_mid_cand > 0
    t_peak_mid = params.T * (0.3 + 0.4 * rand(1, n_mid_cand));      % Middle time
else
    t_peak_mid = [];
end

if n_late_cand > 0
    t_peak_late = params.T * (0.7 + 0.3 * rand(1, n_late_cand));     % Late time
else
    t_peak_late = [];
end

t_peak_cand = [t_peak_ultra_early, t_peak_early, t_peak_mid, t_peak_late];

t_rar_cand = [t_rar_cand, t_peak_cand];
x_rar_cand = [x_rar_cand, x_peak_cand];

% Ensure we have the right number of candidates
if length(t_rar_cand) > rar_candidates
    idx_keep = randperm(length(t_rar_cand), rar_candidates);
    t_rar_cand = t_rar_cand(idx_keep);
    x_rar_cand = x_rar_cand(idx_keep);
end

% Compute residuals for RAR selection
t_rar_dl = dlarray(t_rar_cand, "CB");
x_rar_dl = dlarray(x_rar_cand, "CB");

try
    % CORRECTED: Fixed parameter order to match rar_residual function signature
    % rar_residual(net, t_r, x_r, t_hist, x_hist, v, D, alpha, T, lambda_soe, theta, params)
    [res_val, ~] = dlfeval(@rar_residual, net, t_rar_dl, x_rar_dl, ...
                           params.t_hist, params.x_hist, params.v, params.D, ...
                           params.alpha, params.T, params.lambda_soe, params.theta, params);
    
    % Handle numerical issues
    res_val(isnan(res_val) | isinf(res_val)) = -Inf;
    
    % Enhanced RAR selection with peak region emphasis
    selection_weights = ones(size(res_val));
    
    % Identify boundary-near points
    boundary_mask = (x_rar_cand < 0.1 * params.L) | (x_rar_cand > 0.9 * params.L);
    
    % Identify peak region points using dynamic peak location
    if isfield(params, 'peak_region') && isfield(params.peak_region, 'dynamic') && params.peak_region.dynamic
        % Calculate current peak location for RAR selection
        t_current = iter / params.epochs * params.T;
        [peak_x_rar, ~] = find_peak_location_singular(t_current, params.alpha, params.peak_detection.n_points);
        peak_region_width = 0.1;  % Use ±0.05 range for peak region identification
        peak_mask = (x_rar_cand > peak_x_rar - peak_region_width) & (x_rar_cand < peak_x_rar + peak_region_width);
    else
        % Fallback to fixed peak region
        peak_mask = (x_rar_cand > 0.4 * params.L) & (x_rar_cand < 0.6 * params.L);
    end
    
    % Reduce weight for boundary-near points (since we have fewer of them)
    selection_weights(boundary_mask) = 0.8;
    
    % Increase weight for peak region points (since this is where we want high accuracy)
    selection_weights(peak_mask) = 1.3;
    
    % Apply weights to residuals
    weighted_residuals = res_val .* selection_weights;
    
    % Select points with highest weighted residuals
    [~, idx] = maxk(weighted_residuals, min(rar_add, numel(t_rar_cand)));
    valid_idx = idx(idx <= numel(t_rar_cand));
    
    t_r = t_rar_cand(valid_idx);
    x_r = x_rar_cand(valid_idx);
    
catch ME
    warning('RAR sampling failed: %s. Falling back to random sampling.', ME.message);
    % Fallback to random sampling
    t_r = params.T * rand(1, params.miniNr);
    x_r = params.L * rand(1, params.miniNr);
end

end

