function [t_r, x_r] = strategic_sampling(params, iter)
% STRATEGIC_SAMPLING - Strategic non-random sampling based on backup_good_case1
% This function implements the same sampling strategy as backup_good_case1
% to ensure consistent solution morphology
%
% Inputs:
%   params - parameter struct with sampling configuration
%
% Outputs:
%   t_r, x_r - strategically sampled collocation points

% Get total number of points needed
total_points = params.miniNr;

% Progress within Adam phase (0~1). If iter is empty, default to 0.
if nargin < 2 || isempty(iter)
    progress = 0;
else
    progress = min(1, iter / max(1, params.maxIterAdam));
end

% Initialize arrays
t_r = [];
x_r = [];

% Gentle rebalancing factors
crit_frac_base = 0.12;                 % base 12%
crit_frac = crit_frac_base + 0.08*progress;   % up to +8%
peak_frac_base = params.peak_sampling_ratio;  % base (e.g., 10%)
peak_frac = min(0.20, peak_frac_base + 0.05*progress); % cap at 20%
late_frac = 0.20 - 0.05*progress;      % slightly reduce late-time share
late_frac = max(0.12, late_frac);
time_strat_frac = 0.25;                 % keep constant
interior_frac = 0.15;                   % keep constant

% Boundary fraction gently decreases over time (collocation near boundaries)
boundary_frac_base = params.boundary_sampling_ratio;    % e.g., 3%
boundary_frac = max(0.01, boundary_frac_base*(1 - 0.5*progress));

% Normalize to avoid exceeding 100%
other_sum = crit_frac + peak_frac + late_frac + time_strat_frac + interior_frac + boundary_frac;
if other_sum > 0.98
    scale = 0.98 / other_sum;
    crit_frac = crit_frac*scale;
    peak_frac = peak_frac*scale;
    late_frac = late_frac*scale;
    interior_frac = interior_frac*scale;
    time_strat_frac = time_strat_frac*scale;
    boundary_frac = boundary_frac*scale;
end

% 1. Time-stratified sampling (better coverage across time domain)
time_segments = params.time_segments;
n_time_stratified = round(total_points * time_strat_frac);
if n_time_stratified > 0
    for seg = 1:time_segments
        t_start = (seg - 1) * params.T / time_segments;
        t_end = seg * params.T / time_segments;
        n_cand_seg = round(n_time_stratified / time_segments);
        t_seg = t_start + (t_end - t_start) * rand(1, n_cand_seg);
        x_seg = params.L * rand(1, n_cand_seg);
        t_r = [t_r, t_seg];
        x_r = [x_r, x_seg];
    end
end

% 2. Boundary region sampling (reduced for Dirichlet BC u=0)
n_boundary = round(total_points * boundary_frac);
if n_boundary > 0
    % Early time boundary sampling (t < 0.3T)
    n_early_boundary = round(n_boundary * params.boundary_early_weight);
    t_early_boundary = params.T * (0.05 + 0.25 * rand(1, n_early_boundary));
    
    % Late time boundary sampling (t > 0.7T) - minimal sampling
    n_late_boundary = round(n_boundary * params.boundary_late_weight);
    t_late_boundary = params.T * (0.7 + 0.3 * rand(1, n_late_boundary));
    
    % Middle time boundary sampling
    n_mid_boundary = n_boundary - n_early_boundary - n_late_boundary;
    t_mid_boundary = params.T * (0.3 + 0.4 * rand(1, n_mid_boundary));
    
    % Combine boundary time points
    t_boundary = [t_early_boundary, t_mid_boundary, t_late_boundary];
    
    % Spatial distribution: focus on boundary regions with potential gradients
    % Avoid sampling exactly at x=0 and x=L where u=0
    n_left_boundary = round(n_boundary / 2);
    n_right_boundary = n_boundary - n_left_boundary;
    
    x_left_boundary = params.L * (0.02 + 0.08 * rand(1, n_left_boundary));
    x_right_boundary = params.L * (0.9 + 0.1 * rand(1, n_right_boundary));
    x_boundary = [x_left_boundary, x_right_boundary];
    
    t_r = [t_r, t_boundary];
    x_r = [x_r, x_boundary];
end

% 3. Critical time region sampling (focus on early + middle time where dynamics are complex)
n_critical = round(total_points * crit_frac);
if n_critical > 0
    % Enhanced initial layer sampling: 70% ultra-early + early (0~0.01T), 30% middle (0.3~0.7T)
    n_critical_early = round(n_critical * 0.7);  % Increased from 0.8 to 0.7 for better balance
    n_critical_mid = n_critical - n_critical_early;
    
    % Ultra-early and early critical sampling with enhanced density
    if n_critical_early > 0
        % Enhanced early sampling: 70% ultra-early (0~0.001T), 30% early (0.001~0.01T)
        n_ultra_early = round(n_critical_early * 0.7);  % Increased from 0.6 to 0.7
        n_early = n_critical_early - n_ultra_early;
        
        % Ultra-early sampling using exponential distribution for better coverage near t=0
        if n_ultra_early > 0
            % Enhanced exponential distribution: t = 0.001 * (1 - exp(-8*r)) for denser sampling
            r_ultra = rand(1, n_ultra_early);
            t_ultra_early = params.T * 0.001 * (1 - exp(-8 * r_ultra));  % Increased from -5 to -8
            x_ultra_early = params.L * rand(1, n_ultra_early);
            t_r = [t_r, t_ultra_early];
            x_r = [x_r, x_ultra_early];
        end
        
        % Early sampling using dense uniform distribution
        if n_early > 0
            t_early = params.T * (0.001 + 0.009 * rand(1, n_early));  % Dense uniform in [0.001T, 0.01T]
            x_early = params.L * rand(1, n_early);
            t_r = [t_r, t_early];
            x_r = [x_r, x_early];
        end
    end
    
    % Middle critical sampling
    if n_critical_mid > 0
        t_critical_mid = params.T * (0.3 + 0.4 * rand(1, n_critical_mid));
        x_critical_mid = params.L * rand(1, n_critical_mid);
        t_r = [t_r, t_critical_mid];
        x_r = [x_r, x_critical_mid];
    end
end

% 4. Late time region sampling (important for long-term prediction)
n_late = round(total_points * late_frac);
if n_late > 0
    t_late = params.T * (0.6 + 0.4 * rand(1, n_late)); % Focus on late time
    x_late = params.L * rand(1, n_late);
    t_r = [t_r, t_late];
    x_r = [x_r, x_late];
end

% 5. Interior region sampling (compensate for reduced boundary sampling)
n_interior = round(total_points * params.interior_sampling_ratio);
if n_interior > 0
    % Focus on interior regions where solution gradients are significant
    x_interior = params.L * (0.1 + 0.8 * rand(1, n_interior));
    t_interior = params.T * rand(1, n_interior);
    t_r = [t_r, t_interior];
    x_r = [x_r, x_interior];
end

% 6. Peak region sampling (focus on dynamic peak location)
n_peak = round(total_points * peak_frac);
if n_peak > 0
        % Dynamic peak location calculation
    if isfield(params, 'peak_region') && isfield(params.peak_region, 'dynamic') && params.peak_region.dynamic
        % Calculate peak location at current time point
        t_current = iter / params.epochs * params.T;
        [peak_x, ~] = find_peak_location_singular(t_current, params.alpha, params.peak_detection.n_points);
    else
        % Use configured default peak location
        peak_x = params.peak_region.center;
    end
    
    % Adjust sampling region width based on time
    [region_width, n_peak_points] = get_adaptive_sampling_params(t_current, params);
    
    % Generate sampling points near peak with enhanced density
    % Main sampling region: peak_x ± region_width
    n_main_peak = round(n_peak * 0.7);  % 70% in main region
    x_main_peak = peak_x + region_width * (rand(1, n_main_peak) - 0.5);
    
    % Dense sampling region: peak_x ± 0.05 (more dense sampling)
    n_dense_peak = n_peak - n_main_peak;  % 30% in dense region
    dense_region_width = 0.05;  % ±0.05 dense region
    x_dense_peak = peak_x + dense_region_width * (rand(1, n_dense_peak) - 0.5);
    
    % Combine sampling points
    x_peak = [x_main_peak, x_dense_peak];
    x_peak = max(0, min(1, x_peak));  % Ensure within [0,1] range
    
    % Time distribution: enhanced bias toward ultra-early and early time
    % Calculate exact numbers for each time segment
    n_peak_ultra_early = round(n_peak * 0.5);  % Increased from 0.4 to 0.5 for better initial layer capture
    n_peak_early = round(n_peak * 0.3);       % Reduced from 0.35 to 0.3
    n_peak_mid = round(n_peak * 0.1);         % Reduced from 0.15 to 0.1
    n_peak_late = n_peak - n_peak_ultra_early - n_peak_early - n_peak_mid;  % Remaining for late
    
    % Generate time points for each segment with enhanced early sampling
    if n_peak_ultra_early > 0
        % Ultra-early sampling using exponential distribution
        r_ultra_peak = rand(1, n_peak_ultra_early);
        t_peak_ultra_early = params.T * 0.001 * (1 - exp(-8 * r_ultra_peak));  % Increased from -5 to -8
    else
        t_peak_ultra_early = [];
    end
    
    if n_peak_early > 0
        % Early sampling using dense uniform distribution
        t_peak_early = params.T * (0.001 + 0.009 * rand(1, n_peak_early));  % Dense uniform in [0.001T, 0.01T]
    else
        t_peak_early = [];
    end
    
    if n_peak_mid > 0
        t_peak_mid = params.T * (0.3 + 0.4 * rand(1, n_peak_mid));     % Middle time
    else
        t_peak_mid = [];
    end
    
    if n_peak_late > 0
        t_peak_late = params.T * (0.7 + 0.3 * rand(1, n_peak_late));    % Late time
    else
        t_peak_late = [];
    end
    
    % Combine time points
    t_peak = [t_peak_ultra_early, t_peak_early, t_peak_mid, t_peak_late];
    
    % Ensure we have exactly n_peak points
    if length(t_peak) > n_peak
        % If too many, randomly select n_peak
        idx_keep = randperm(length(t_peak), n_peak);
        t_peak = t_peak(idx_keep);
        x_peak = x_peak(idx_keep);
    elseif length(t_peak) < n_peak
        % If too few, add more points (distribute proportionally with enhanced early bias)
        n_add = n_peak - length(t_peak);
        n_add_ultra_early = round(n_add * 0.4);
        n_add_early = round(n_add * 0.35);
        n_add_mid = round(n_add * 0.15);
        n_add_late = n_add - n_add_ultra_early - n_add_early - n_add_mid;
        
        if n_add_ultra_early > 0
            r_add_ultra = rand(1, n_add_ultra_early);
            t_add_ultra_early = params.T * 0.001 * (1 - exp(-5 * r_add_ultra));
            x_add_ultra_early = peak_x + 0.1 * (rand(1, n_add_ultra_early) - 0.5);
            t_peak = [t_peak, t_add_ultra_early];
            x_peak = [x_peak, x_add_ultra_early];
        end
        
        if n_add_early > 0
            t_add_early = params.T * (0.001 + 0.009 * rand(1, n_add_early));
            x_add_early = peak_x + 0.1 * (rand(1, n_add_early) - 0.5);
            t_peak = [t_peak, t_add_early];
            x_peak = [x_peak, x_add_early];
        end
        
        if n_add_mid > 0
            t_add_mid = params.T * (0.3 + 0.4 * rand(1, n_add_mid));
            x_add_mid = peak_x + 0.1 * (rand(1, n_add_mid) - 0.5);
            t_peak = [t_peak, t_add_mid];
            x_peak = [x_peak, x_add_mid];
        end
        
        if n_add_late > 0
            t_add_late = params.T * (0.7 + 0.3 * rand(1, n_add_late));
            x_add_late = peak_x + 0.1 * (rand(1, n_add_late) - 0.5);
            t_peak = [t_peak, t_add_late];
            x_peak = [x_peak, x_add_late];
        end
    end
    
    t_r = [t_r, t_peak];
    x_r = [x_r, x_peak];
end

% Enhanced t=0 moment sampling for initial layer capture
n_t0 = round(total_points * 0.08);  % Increased from 0.05 to 0.08 for better initial layer capture
if n_t0 > 0
    t_t0 = zeros(1, n_t0);  % All points at t=0
    x_t0 = params.L * rand(1, n_t0);  % Random spatial distribution
    
    % Enhanced sampling near peak location
    if isfield(params, 'peak_region') && isfield(params.peak_region, 'dynamic') && params.peak_region.dynamic
        [peak_x_t0, ~] = find_peak_location_singular(0, params.alpha, params.peak_detection.n_points);
    else
        peak_x_t0 = params.peak_region.center;
    end
    
    % Enhanced peak-focused sampling at t=0: 70% near peak, 30% random distribution
    n_peak_t0 = round(n_t0 * 0.7);  % Increased from 0.4 to 0.7 for better peak capture
    n_random_t0 = n_t0 - n_peak_t0;
    
    if n_peak_t0 > 0
        peak_region_width = 0.1;
        x_peak_t0 = peak_x_t0 + peak_region_width * (rand(1, n_peak_t0) - 0.5);
        x_peak_t0 = max(0, min(params.L, x_peak_t0));
    else
        x_peak_t0 = [];
    end
    
    if n_random_t0 > 0
        x_random_t0 = params.L * rand(1, n_random_t0);
    else
        x_random_t0 = [];
    end
    
    x_t0 = [x_peak_t0, x_random_t0];
    
    t_r = [t_r, t_t0];
    x_r = [x_r, x_t0];
end

% 7. Fill remaining points with uniform distribution to reach exact total
current_points = length(t_r);
if current_points < total_points
    n_remaining = total_points - current_points;
    t_remaining = params.T * rand(1, n_remaining);
    x_remaining = params.L * rand(1, n_remaining);
    t_r = [t_r, t_remaining];
    x_r = [x_r, x_remaining];
elseif current_points > total_points
    % If we have too many points, randomly select the required number
    idx_keep = randperm(current_points, total_points);
    t_r = t_r(idx_keep);
    x_r = x_r(idx_keep);
end

% Ensure we have exactly the right number of points
assert(length(t_r) == total_points, 'Sampling point count mismatch');
assert(length(x_r) == total_points, 'Sampling point count mismatch');

% Verify all points are within valid ranges
assert(all(t_r >= 0) && all(t_r <= params.T), 'Time points out of range');
assert(all(x_r >= 0) && all(x_r <= params.L), 'Space points out of range');

end
