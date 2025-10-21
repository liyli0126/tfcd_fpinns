function [t_r, x_r] = rar_sampling(params, net)
% RAR_SAMPLING - Residual Adaptive Refinement sampling with strategic candidate generation
%
% This function implements an enhanced RAR sampling strategy that generates
% collocation points based on residual magnitude and strategic positioning.
% It focuses on regions where the solution has high gradients or where
% boundary effects are significant.
%
% Inputs:
%   params - parameter struct containing domain and sampling parameters
%   net    - trained neural network for residual computation
%
% Outputs:
%   t_r, x_r - selected collocation points for training
%
% Algorithm:
%   1. Time-stratified sampling across the entire time domain
%   2. Strategic boundary sampling with time-dependent emphasis
%   3. Critical time region sampling (middle time dynamics)
%   4. Late time region sampling (long-term prediction)
%   5. Interior region sampling (solution gradients)
%   6. Peak region sampling (maximum solution values)
%
% Reference: Based on adaptive sampling strategies for PINN training

% Get numerical constants for consistency
constants = numerical_constants();

% Initialize RAR sampling parameters
rar_candidates = constants.rar_candidate_multiplier * params.miniNr;  % 3x candidates for selection
rar_add = params.miniNr;                                             % number of points to add

% Initialize candidate arrays
t_rar_cand = [];
x_rar_cand = [];

%% 1. Time-stratified sampling (uniform coverage across time domain)
time_segments = constants.default_time_segments;
for seg = 1:time_segments
    t_start = (seg - 1) * params.T / time_segments;
    t_end = seg * params.T / time_segments;
    n_cand_seg = round(rar_candidates / time_segments);
    
    % Generate candidates within this time segment
    t_seg_cand = t_start + (t_end - t_start) * rand(1, n_cand_seg);
    x_seg_cand = params.L * rand(1, n_cand_seg);
    
    t_rar_cand = [t_rar_cand, t_seg_cand];
    x_rar_cand = [x_rar_cand, x_seg_cand];
end

%% 2. Strategic boundary sampling (reduced for Dirichlet BC u=0)
% Since boundary conditions are u(0,t) = u(L,t) = 0, we need fewer boundary points
% Focus on regions where boundary effects propagate into the domain
n_boundary_cand = round(rar_candidates * 0.03);  % 3% of candidates for boundary regions

if n_boundary_cand > 0
    % Time-stratified boundary sampling based on physical importance
    % Early time boundary effects are more important for advection-diffusion
    
    % Early time boundary sampling (t < 0.3T) - highest weight
    n_early_boundary = round(n_boundary_cand * 0.6);
    t_early_boundary = params.T * (constants.boundary_avoidance + ...
        (constants.early_time_threshold - constants.boundary_avoidance) * rand(1, n_early_boundary));
    
    % Late time boundary sampling (t > 0.7T) - minimal sampling
    n_late_boundary = round(n_boundary_cand * 0.2);
    t_late_boundary = params.T * (constants.late_time_threshold + ...
        (1 - constants.late_time_threshold) * rand(1, n_late_boundary));
    
    % Middle time boundary sampling
    n_mid_boundary = n_boundary_cand - n_early_boundary - n_late_boundary;
    t_mid_boundary = params.T * (constants.mid_time_start + ...
        (constants.mid_time_end - constants.mid_time_start) * rand(1, n_mid_boundary));
    
    % Combine boundary time points
    t_boundary_cand = [t_early_boundary, t_mid_boundary, t_late_boundary];
    
    % Spatial distribution: avoid exact boundaries where u=0
    % Focus on regions with potential gradients
    n_left_boundary = round(n_boundary_cand / 2);
    n_right_boundary = n_boundary_cand - n_left_boundary;
    
    % Left boundary region (x ≈ 0.02L to 0.1L)
    x_left_boundary = params.L * (constants.boundary_avoidance + ...
        constants.boundary_near_threshold * rand(1, n_left_boundary));
    
    % Right boundary region (x ≈ 0.9L to 0.98L)
    x_right_boundary = params.L * ((1 - constants.boundary_near_threshold) + ...
        constants.boundary_avoidance * rand(1, n_right_boundary));
    
    x_boundary_cand = [x_left_boundary, x_right_boundary];
    
    % Add boundary candidates to main arrays
    t_rar_cand = [t_rar_cand, t_boundary_cand];
    x_rar_cand = [x_rar_cand, x_boundary_cand];
end

%% 3. Critical time region sampling (middle time dynamics)
% Focus on time regions where the solution dynamics are most complex
n_critical_cand = round(rar_candidates * 0.12);  % 12% for critical regions
t_critical_cand = params.T * (constants.mid_time_start + ...
    (constants.mid_time_end - constants.mid_time_start) * rand(1, n_critical_cand));
x_critical_cand = params.L * rand(1, n_critical_cand);

t_rar_cand = [t_rar_cand, t_critical_cand];
x_rar_cand = [x_rar_cand, x_critical_cand];

%% 4. Late time region sampling (long-term prediction accuracy)
% Important for capturing long-term behavior and stability
n_late_cand = round(rar_candidates * 0.20);  % 20% for late time regions
t_late_cand = params.T * (0.6 + 0.4 * rand(1, n_late_cand));
x_late_cand = params.L * rand(1, n_late_cand);

t_rar_cand = [t_rar_cand, t_late_cand];
x_rar_cand = [x_rar_cand, x_late_cand];

%% 5. Interior region sampling (solution gradients)
% Compensate for reduced boundary sampling, focus on interior gradients
n_interior_cand = round(rar_candidates * 0.15);  % 15% for interior regions
x_interior_cand = params.L * (0.1 + 0.8 * rand(1, n_interior_cand));
t_interior_cand = params.T * rand(1, n_interior_cand);

t_rar_cand = [t_rar_cand, t_interior_cand];
x_rar_cand = [x_rar_cand, x_interior_cand];

%% 6. Adaptive peak region sampling (maximum solution values)
% Dynamically track peak position based on convection-diffusion evolution
n_peak_cand = round(rar_candidates * 0.15);  % Increased to 15% for peak region

% Get constants for adaptive peak sampling
constants = numerical_constants();

if constants.adaptive_peak_enabled
    % Dynamic peak position detection using analytical solution
    % Time-stratified sampling with adaptive peak tracking
    
    % Early time peak sampling (40% of peak candidates)
    n_peak_early = round(n_peak_cand * 0.4);
    t_peak_early = params.T * (constants.boundary_avoidance + ...
        constants.early_time_threshold * rand(1, n_peak_early));
    
    % Middle time peak sampling (30% of peak candidates)
    n_peak_mid = round(n_peak_cand * 0.3);
    t_peak_mid = params.T * (constants.mid_time_start + ...
        (constants.mid_time_end - constants.mid_time_start) * rand(1, n_peak_mid));
    
    % Late time peak sampling (30% of peak candidates)
    n_peak_late = round(n_peak_cand * 0.3);
    t_peak_late = params.T * (constants.late_time_threshold + ...
        (1 - constants.late_time_threshold) * rand(1, n_peak_late));
    
    t_peak_all = [t_peak_early, t_peak_mid, t_peak_late];
    
    % Calculate expected peak positions for each time point using improved estimation
    peak_positions = zeros(1, length(t_peak_all));
    for i = 1:length(t_peak_all)
        try
            % Try to use detect_peak_position for accurate peak location
            [actual_peak_pos, ~] = detect_peak_position(params, constants, [], t_peak_all(i));
            peak_positions(i) = actual_peak_pos;
        catch
            % Fallback to physically reasonable analytical estimation
            % For fractional advection-diffusion with high Pe number
            fractional_factor = t_peak_all(i)^params.alpha / gamma(params.alpha + 1);
            % Use much smaller movement factor for high Pe number (v/D = 1000)
            movement_factor = min(0.001, params.D / params.v); % Limit extreme Pe effects
            expected_peak_x = 0.5 + params.v * fractional_factor * movement_factor;
            expected_peak_x = max(0.1, min(0.9, expected_peak_x));
            peak_positions(i) = expected_peak_x;
        end
    end
    
    % Adaptive peak width based on convection velocity and time
    adaptive_peak_width = constants.adaptive_peak_width_base + ...
        constants.adaptive_peak_width_factor * params.v / params.D;
    adaptive_peak_width = min(0.3, adaptive_peak_width); % Cap maximum width
    
    % Sample around calculated peak positions
    x_peak_cand = zeros(1, length(t_peak_all));
    for i = 1:length(t_peak_all)
        % Random sampling around calculated peak position
        x_peak_cand(i) = peak_positions(i) + ...
            (rand(1) - 0.5) * adaptive_peak_width * params.L;
        % Ensure within domain bounds
        x_peak_cand(i) = max(0, min(params.L, x_peak_cand(i)));
    end
    
    t_peak_cand = t_peak_all;
    
else
    % Fallback to fixed peak sampling
    % Spatial concentration around fixed peak center
    x_peak_cand = params.L * (constants.peak_center - constants.peak_width/2 + ...
        constants.peak_width * rand(1, n_peak_cand));
    
    % Time distribution: emphasize early and late times for peak region
    % Early time peak sampling (40% of peak candidates)
    n_peak_early = round(n_peak_cand * 0.4);
    t_peak_early = params.T * (constants.boundary_avoidance + ...
        constants.early_time_threshold * rand(1, n_peak_early));
    
    % Middle time peak sampling (30% of peak candidates)
    n_peak_mid = round(n_peak_cand * 0.3);
    t_peak_mid = params.T * (constants.mid_time_start + ...
        (constants.mid_time_end - constants.mid_time_start) * rand(1, n_peak_mid));
    
    % Late time peak sampling (30% of peak candidates)
    n_peak_late = round(n_peak_cand * 0.3);
    t_peak_late = params.T * (constants.late_time_threshold + ...
        (1 - constants.late_time_threshold) * rand(1, n_peak_late));
    
    t_peak_cand = [t_peak_early, t_peak_mid, t_peak_late];
end

t_rar_cand = [t_rar_cand, t_peak_cand];
x_rar_cand = [x_rar_cand, x_peak_cand];

%% Ensure correct number of candidates
if length(t_rar_cand) > rar_candidates
    idx_keep = randperm(length(t_rar_cand), rar_candidates);
    t_rar_cand = t_rar_cand(idx_keep);
    x_rar_cand = x_rar_cand(idx_keep);
end

%% Compute residuals for RAR selection
% Convert to dlarray for network evaluation
t_rar_dl = dlarray(t_rar_cand, "CB");
x_rar_dl = dlarray(x_rar_cand, "CB");

try
    % Compute residuals using the trained network
    % Function signature: rar_residual(net, t_r, x_r, t_hist, x_hist, v, D, alpha, T, lambda_soe, theta)
    [res_val, ~] = dlfeval(@rar_residual, net, t_rar_dl, x_rar_dl, ...
                           params.t_hist, params.x_hist, params.v, params.D, ...
                           params.alpha, params.T, params.lambda_soe, params.theta);
    
    % Handle numerical issues in residuals
    res_val(isnan(res_val) | isinf(res_val)) = -Inf;
    
    % Apply strategic selection weights using unified weight functions
    selection_weights = weight_functions('rar_selection', params, x_rar_cand, t_rar_cand);
    
    % Weight residuals for selection
    weighted_residuals = res_val .* selection_weights;
    
    % Select points with highest weighted residuals
    [~, idx] = maxk(weighted_residuals, min(rar_add, numel(t_rar_cand)));
    valid_idx = idx(idx <= numel(t_rar_cand));
    
    t_r = t_rar_cand(valid_idx);
    x_r = x_rar_cand(valid_idx);
    
catch ME
    warning('RAR sampling failed: %s. Falling back to random sampling.', ME.message);
    % Fallback to random sampling if RAR fails
    t_r = params.T * rand(1, params.miniNr);
    x_r = params.L * rand(1, params.miniNr);
end

end

