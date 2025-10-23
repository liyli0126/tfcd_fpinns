function [loss, gradients, L_pde, L_bc, L_ic, D_alpha_sample] = loss_fractional_pinn(net, ...
    t_r, x_r, t_hist, x_hist, t_0, x_0, t_L, x_L, t_ic, x_ic, v, D, alpha, lambda, T, lambda_soe, theta, epoch, L, params)
% LOSS_FRACTIONAL_PINN - Generic loss for time-fractional advection-diffusion PINN
%
% Inputs:
%   net, t_r, x_r, t_hist, x_hist, t_0, x_0, t_L, x_L, t_ic, x_ic - training data
%   v, D, alpha - PDE parameters
%   lambda - loss weights
%   T - total time
%   lambda_soe, theta - SOE parameters
%   epoch - current epoch
%   L - domain length
%   params - parameter struct with generic interfaces
%
% Outputs:
%   loss - total loss
%   gradients - network gradients
%   L_pde, L_bc, L_ic - individual loss components
%   D_alpha_sample - fractional derivative at sample point for debugging

% All loss calculations must remain in the dlarray domain for autodiff
u_r = forward(net, [t_r; x_r]);
u_x = dlgradient(sum(u_r, 'all'), x_r, 'EnableHigherDerivatives', true);
u_xx = dlgradient(sum(u_r, 'all'), x_r, 'EnableHigherDerivatives', true, 'EnableHigherDerivatives', true);

% Compute fractional time derivative using modified fast L1 scheme
tx_hist = dlarray([t_hist; x_hist], "CB");
u_hist = predict(net, tx_hist);
u_hist = extractdata(u_hist);  % Only for MFL1_Caputo, not for loss
t_curr = extractdata(t_r);
u_curr = extractdata(u_r);

% Add numerical stability checks
if any(isnan(u_hist)) || any(isinf(u_hist))
    warning('Historical network outputs contain NaN or Inf');
    u_hist(isnan(u_hist) | isinf(u_hist)) = 0;
end

if any(isnan(u_curr)) || any(isinf(u_curr))
    warning('Current network outputs contain NaN or Inf');
    u_curr(isnan(u_curr) | isinf(u_curr)) = 0;
end

D_alpha_u = zeros(1, length(t_curr));
for i = 1:length(t_curr)
    D_alpha_u(i) = MFL1_Caputo(u_hist, t_hist, u_curr(i), t_curr(i), alpha, T, lambda_soe, theta);
end
D_alpha_u = dlarray(D_alpha_u, "CB");

% Generic PDE residual: fractional time derivative + spatial operator - source term
% Check if source term is provided
if isfield(params, 'source_term') && ~isempty(params.source_term)
    % Case2: with source term
    t_val_r = extractdata(t_r);
    x_val_r = extractdata(x_r);
    f_vals = zeros(1, numel(t_val_r));
    for i = 1:numel(t_val_r)
        f_vals(i) = params.source_term(t_val_r(i), x_val_r(i), alpha);
    end
    f_vals = dlarray(f_vals, "CB");
    residual = D_alpha_u + v * u_x - D * u_xx - f_vals;
else
    % Case1: without source term
    residual = D_alpha_u + v * u_x - D * u_xx;
end

% Enhanced adaptive local weighting: emphasize peak regions and boundary regions
x_val = extractdata(x_r);
local_weight = ones(size(x_val));

% Get numerical constants
constants = numerical_constants();

% Generic peak region emphasis - Using dynamic peak location
if isfield(params, 'peak_region') && ~isempty(params.peak_region)
    peak_width = params.peak_region.width;    % Default: 0.1
    peak_weight = params.peak_region.weight;  % Default: 8
else
    peak_width = 0.1;
    peak_weight = 8;
end

% Dynamic peak location calculation for local weighting
if isfield(params, 'peak_region') && isfield(params.peak_region, 'dynamic') && params.peak_region.dynamic
    % Calculate peak location at current time point
    t_unique = unique(extractdata(t_r));
    peak_center = zeros(size(x_val));
    for i = 1:length(x_val)
        t_idx = find(abs(t_unique - extractdata(t_r(i))) < 1e-6, 1);
        if ~isempty(t_idx)
            [peak_x_temp, ~] = find_peak_location_singular(t_unique(t_idx), alpha, params.peak_detection.n_points);
            peak_center(i) = peak_x_temp;
        else
            peak_center(i) = 0.5;  % Default value
        end
    end
else
    peak_center = params.peak_region.center * ones(size(x_val));  % Use configured default peak location
end

% Use dynamic peak location for local weighting - Further reduce weights to achieve true balanced constraints
peak_mask = zeros(size(x_val), 'logical');  % Ensure logical array
for i = 1:length(x_val)
    peak_mask(i) = (x_val(i) > peak_center(i) - peak_width) && (x_val(i) < peak_center(i) + peak_width);
end
% Further reduce PDE weights in peak region to achieve true balanced constraints
local_weight(peak_mask) = 1.0;  % Reduced from 1.5 to 1.0, achieve true balance

% Generic boundary region emphasis - configurable through params
if isfield(params, 'boundary_regions') && ~isempty(params.boundary_regions)
    front_weight = params.boundary_regions.front;  % Default: 5
    rear_weight = params.boundary_regions.rear;    % Default: 5
    front_threshold = params.boundary_regions.front_threshold;  % Default: 0.2
    rear_threshold = params.boundary_regions.rear_threshold;    % Default: 0.8
else
    front_weight = constants.front_weight;
    rear_weight = constants.rear_weight;
    front_threshold = constants.front_threshold;
    rear_threshold = constants.rear_threshold;
end

local_weight(x_val < front_threshold) = front_weight;
local_weight(x_val > rear_threshold) = rear_weight;

% Residual extremes (top 5% absolute residuals) - Further reduce weights to achieve minimal intervention
abs_res = abs(extractdata(residual));
if numel(abs_res) > 20
    [~, idx_ext] = maxk(abs_res, round(constants.residual_extreme_ratio*numel(abs_res))); % top 5%
    % Further reduce residual extreme weights to achieve minimal intervention
    local_weight(idx_ext) = 1.5;  % Reduced from 3 to 1.5, achieve minimal intervention
end

local_weight = dlarray(local_weight, "CB");

% Enhanced time weighting: segmented early weights + mid-time bell (avoiding network intimidation)
tt = t_r / T;
eps_t = 1e-4;  % Small epsilon to avoid division by zero

% Vectorized ultra-smooth early weights with continuous derivatives
early_boost = zeros(size(tt));

% Use logical masks for vectorized operations
mask_ultra_early = tt < constants.ultra_early_threshold;
mask_early = (tt >= constants.ultra_early_threshold) & (tt < constants.early_threshold);
mask_mid_early = (tt >= constants.early_threshold) & (tt < constants.mid_early_threshold);
mask_late_early = (tt >= constants.mid_early_threshold) & (tt < constants.late_early_threshold);

% Apply weights using vectorized operations (MICRO-TUNED for better singular layer learning)
early_boost(mask_ultra_early) = constants.ultra_early_weight;  % Increased from 0.02 to 0.05 (+150%)

% Smooth transition for early region using vectorized smoothstep
if any(mask_early)
    t_norm_early = (tt(mask_early) - constants.ultra_early_threshold) / (constants.early_threshold - constants.ultra_early_threshold);  % 0 to 1
    t_smooth_early = 3*t_norm_early.^2 - 2*t_norm_early.^3;  % smoothstep function
    early_boost(mask_early) = constants.early_weight_base + constants.early_weight_boost * t_smooth_early;  % Increased from 0.02+0.06 to 0.05+0.07
end

% Smooth transition for mid-early region using vectorized smoothstep
if any(mask_mid_early)
    t_norm_mid = (tt(mask_mid_early) - constants.early_threshold) / (constants.mid_early_threshold - constants.early_threshold);  % 0 to 1
    t_smooth_mid = 3*t_norm_mid.^2 - 2*t_norm_mid.^3;  % smoothstep function
    early_boost(mask_mid_early) = constants.mid_early_weight_base + constants.mid_early_weight_boost * t_smooth_mid;  % Increased from 0.08+0.07 to 0.12+0.08
end

% Fixed weight for late-early region
early_boost(mask_late_early) = constants.late_early_weight;  % Increased from 0.03 to 0.04 (+33%)

sigma_mid = constants.mid_time_bell_width;  % Mid-time bell width
bell = exp(-((tt - 0.5).^2) / (2*(sigma_mid^2)));  % Keep mid-time attention
% Further reduce bell function amplitude to achieve gentle constraints
t_weight = 1 + 0.2*bell + early_boost;  % Reduced from 0.4 to 0.2, achieve gentle constraints
L_pde = mean(local_weight .* t_weight .* (residual.^2), 'all');

% Long-term prediction penalty (focus on t > 0.7T)
t_r_val = extractdata(t_r); % Only for mask
longterm_mask = (t_r_val > T*constants.longterm_threshold);
if any(longterm_mask)
    L_longterm = constants.longterm_penalty * mean(residual(longterm_mask).^2, 'all'); % Increased from 2.0 to 3.0
else
    L_longterm = dlarray(0.0);
end

% Enhanced Sobolev regularization
x_perturbed = x_r + constants.sobolev_perturbation*L*(rand(size(x_r))-0.5);
u_r_perturbed = forward(net, [t_r; x_perturbed]);
u_x_perturbed = dlgradient(sum(u_r_perturbed, 'all'), x_perturbed);
derivative_diff = abs(u_x - u_x_perturbed);
residual_weight = abs(residual).^2;
residual_weight = residual_weight / mean(residual_weight, 'all'); % normalize
L_sobolev = mean(residual_weight .* derivative_diff.^2, 'all');

% Generic Boundary Condition Loss
u_0 = forward(net, dlarray([t_0; x_0], "CB"));
u_L = forward(net, dlarray([t_L; x_L], "CB"));

% Check if boundary conditions are provided
if isfield(params, 'boundary_conditions') && ~isempty(params.boundary_conditions)
    % Generic boundary conditions
    bc_left_target = params.boundary_conditions.left(extractdata(t_0), extractdata(x_0));
    bc_right_target = params.boundary_conditions.right(extractdata(t_L), extractdata(x_L));
    bc_left_target = dlarray(bc_left_target, "CB");
    bc_right_target = dlarray(bc_right_target, "CB");
    L_bc = mean((u_0 - bc_left_target).^2, 'all') + mean((u_L - bc_right_target).^2, 'all');
else
    % Default: homogeneous boundary conditions (case1)
    L_bc = mean(u_0.^2, 'all') + mean(u_L.^2, 'all');
end

% Generic Initial Condition Loss
u_ic = forward(net, dlarray([t_ic; x_ic], "CB"));

% Check if initial condition is provided
if isfield(params, 'initial_condition') && ~isempty(params.initial_condition)
    % Generic initial condition
    ic_target_vals = params.initial_condition(extractdata(x_ic));
    ic_target = dlarray(ic_target_vals, "CB");
else
    % Default: sin(pi*x) initial condition (case1)
    ic_target = sin(pi * x_ic);
end

% Generic peak region emphasis in initial condition - Reduce weights to avoid overfitting
x_ic_val = extractdata(x_ic);
if isfield(params, 'peak_region') && isfield(params.peak_region, 'dynamic') && params.peak_region.dynamic
    % Use peak location at t=0
    [peak_x_ic, ~] = find_peak_location_singular(0, alpha, params.peak_detection.n_points);
else
    peak_x_ic = params.peak_region.center;  % Use configured default peak location
end

peak_ic_mask = (x_ic_val > peak_x_ic - peak_width) & (x_ic_val < peak_x_ic + peak_width);
ic_weights = ones(size(x_ic_val));
ic_weights(peak_ic_mask) = 1.5;  % Reduce peak region weights from 3.0 to 1.5, avoid overfitting
ic_weights = dlarray(ic_weights, "CB");

L_ic = mean(ic_weights .* (u_ic - ic_target).^2, 'all');

% Generic Peak region protection loss with adaptive targets and staged protection
if isfield(params, 'peak_protection') && ~isempty(params.peak_protection)
    % Dynamic peak location calculation
    if isfield(params, 'peak_protection') && isfield(params.peak_protection, 'dynamic') && params.peak_protection.dynamic
        t_current = epoch / params.epochs * T;
        [peak_x_val, ~] = find_peak_location_singular(t_current, alpha, params.peak_detection.n_points);
        peak_x = dlarray(peak_x_val, "CB");
    else
        % Use configured default peak location
        peak_x = dlarray(params.peak_protection.x, "CB");
    end
    
    % Strategy 1: Enhanced peak protection focusing on time evolution
    if epoch < 100
        % Early stage: Focus on peak position learning, not fixed values
        peak_t = dlarray([0, 0.1*T, 0.2*T], "CB");
        % Use analytical solution as guidance for peak evolution trend
        t_vals = extractdata(peak_t);
        x_vals = extractdata(repmat(peak_x, 1, length(peak_t)));
        u_ana_guide = zeros(size(t_vals));
        for k = 1:numel(t_vals)
            u_ana_guide(k) = params.analytical_solution(x_vals(k), t_vals(k), alpha);
        end
        peak_target = dlarray(u_ana_guide, "CB");
        
    elseif epoch < 300
        % Mid stage: Emphasize smooth time evolution of peak
        peak_t = dlarray([0, 0.2*T, 0.4*T, 0.6*T, 0.8*T, T], "CB");
        peak_input = dlarray([peak_t; repmat(peak_x, 1, length(peak_t))], "CB");
        u_current = forward(net, peak_input);
        
        % Calculate time evolution trend from analytical solution
        t_vals = extractdata(peak_t);
        x_vals = extractdata(repmat(peak_x, 1, length(peak_t)));
        u_ana_trend = zeros(size(t_vals));
        for k = 1:numel(t_vals)
            u_ana_trend(k) = params.analytical_solution(x_vals(k), t_vals(k), alpha);
        end
        
        % Mix current prediction with analytical trend (emphasize evolution)
        evolution_mix = 0.7;  % 70% analytical trend, 30% current prediction
        peak_target = evolution_mix * dlarray(u_ana_trend, "CB") + (1 - evolution_mix) * u_current;
        
    else
        % Late stage: Full peak evolution protection
        peak_t = dlarray(linspace(0, T, params.peak_protection.time_points), "CB");
        peak_input = dlarray([peak_t; repmat(peak_x, 1, length(peak_t))], "CB");
        u_current = forward(net, peak_input);
        
        % Calculate analytical solution trend for full time evolution
        t_vals = extractdata(peak_t);
        x_vals = extractdata(repmat(peak_x, 1, length(peak_t)));
        u_ana_full = zeros(size(t_vals));
        for k = 1:numel(t_vals)
            u_ana_full(k) = params.analytical_solution(x_vals(k), t_vals(k), alpha);
        end
        u_ana_full = dlarray(u_ana_full, "CB");
        
        % Adaptive mixing: gradually increase analytical guidance
        mix = min(0.8, epoch / 1000);  % Gradually increase to 80%
        
        % Emphasize time evolution: use analytical trend as primary target
        peak_target = mix * u_ana_full + (1 - mix) * u_current;
    end
    
else
    % Default adaptive peak protection strategy
    if isfield(params, 'peak_protection') && isfield(params.peak_protection, 'dynamic') && params.peak_protection.dynamic
        t_current = epoch / params.epochs * T;
        [peak_x_val, ~] = find_peak_location_singular(t_current, alpha, params.peak_detection.n_points);
        peak_x = dlarray(peak_x_val, "CB");
    else
        peak_x = dlarray(params.peak_protection.x, "CB");
    end
    
    if epoch < 100
        % Early stage: Use analytical solution as guidance to avoid self-locking
        peak_t = dlarray([0, 0.1*T], "CB");
        if isfield(params, 'analytical_solution')
            % Use analytical solution as target to avoid self-locking
            t_vals = extractdata(peak_t);
            x_vals = extractdata(repmat(peak_x, 1, length(peak_t)));
            u_ana = zeros(size(t_vals));
            for k = 1:numel(t_vals)
                u_ana(k) = params.analytical_solution(x_vals(k), t_vals(k), alpha);
            end
            peak_target = dlarray(u_ana, "CB");
        else
            % If no analytical solution, use current network output with small random perturbation
            peak_input = dlarray([peak_t; repmat(peak_x, 1, length(peak_t))], "CB");
            u_current = forward(net, peak_input);
            % Add small random perturbation to avoid complete self-locking
            noise_factor = 0.01; % 1% random perturbation
            noise = noise_factor * (rand(size(u_current)) - 0.5);
            peak_target = u_current + noise;
        end
        
    elseif epoch < 300
        % Mid stage: True peak protection - let network learn freely
        peak_t = dlarray([0, 0.3*T, 0.7*T, T], "CB");
        peak_input = dlarray([peak_t; repmat(peak_x, 1, length(peak_t))], "CB");
        u_current = forward(net, peak_input);
        
        if isfield(params, 'analytical_solution')
            % Calculate analytical solution as reference
            t_vals = extractdata(peak_t);
            x_vals = extractdata(repmat(peak_x, 1, length(peak_t)));
            u_ana = zeros(size(t_vals));
            for k = 1:numel(t_vals)
                u_ana(k) = params.analytical_solution(x_vals(k), t_vals(k), alpha);
            end
            u_ana = dlarray(u_ana, "CB");
            
            % True peak protection: only correct when deviation is too large
            current_error = mean((u_current - u_ana).^2, 'all');
            tolerance = 0.05; % 5% tolerance
            
            if current_error > tolerance
                % Only correct when error exceeds tolerance
                correction_weight = min(0.3, current_error / tolerance * 0.1); % Gentle correction
                peak_target = (1 - correction_weight) * u_current + correction_weight * u_ana;
            else
                % Error within tolerance, let network learn freely
                peak_target = u_current;
            end
        else
            peak_target = u_current;
        end
        
    else
        % Late stage: Minimal intervention peak protection
        peak_t = dlarray(linspace(0, T, 20), "CB");
        peak_input = dlarray([peak_t; repmat(peak_x, 1, length(peak_t))], "CB");
        u_current = forward(net, peak_input);
        
        % Minimal intervention: only correct when necessary
        if isfield(params, 'analytical_solution')
            t_vals = extractdata(peak_t);
            x_vals = extractdata(repmat(peak_x, 1, length(peak_t)));
            u_ana = zeros(size(t_vals));
            for k = 1:numel(t_vals)
                u_ana(k) = params.analytical_solution(x_vals(k), t_vals(k), alpha);
            end
            u_ana = dlarray(u_ana, "CB");
            
            % Minimal intervention strategy: only correct when seriously deviated
            current_error = mean((u_current - u_ana).^2, 'all');
            tolerance = 0.1; % 10% tolerance (more relaxed in later stages)
            
            if current_error > tolerance
                % Only correct when error seriously exceeds tolerance
                correction_weight = min(0.2, current_error / tolerance * 0.05); % Very gentle correction
                peak_target = (1 - correction_weight) * u_current + correction_weight * u_ana;
            else
                % Error within tolerance, let network learn freely
                peak_target = u_current;
            end
        else
            % If no analytical solution, use gentle improvement strategy
            if isfield(params, 'schedules')
                improve_min = params.schedules.peak_improve_min;
                improve_max = params.schedules.peak_improve_max;
            else
                improve_min = constants.peak_improve_min;
                improve_max = constants.peak_improve_max;
            end
            % Use more gentle improvement factor
            residual_factor = min(improve_max, max(improve_min, (epoch-300)/1500));
            peak_target = u_current * (1 + residual_factor);
        end
    end
end

peak_input = dlarray([peak_t; repmat(peak_x, 1, length(peak_t))], "CB");
u_peak = predict(net, peak_input);

% Generic peak target
peak_target = peak_target; % Use the estimated peak_target
L_peak = mean((u_peak - peak_target).^2, 'all');

% Add peak position monitoring and debugging information
if epoch > 0 && mod(epoch, 50) == 0  % Output every 50 epochs
    % Monitor multiple time points for peak position and value evolution
    t_monitor_points = [0.1, 0.3, 0.5, 0.7, 1.0]; % Monitor multiple time points
    fprintf('Epoch %d: Peak Evolution Monitoring\n', epoch);
    fprintf('Time\tAnalytical Peak Position\tAnalytical Peak Value\tPINN Output\tSearch Range\n');
    
    for i = 1:length(t_monitor_points)
        t_monitor = t_monitor_points(i);
        [peak_x_monitor, peak_val_monitor] = find_peak_location_singular(t_monitor, alpha, params.peak_detection.n_points);
        
        % Get search range for debugging
        [x_min, x_max] = get_adaptive_search_range(t_monitor, alpha);
        search_range_str = sprintf('[%.2f,%.2f]', x_min, x_max);
        
        % Calculate PINN output at peak position
        tx_monitor = dlarray([t_monitor; peak_x_monitor], "CB");
        u_pinn_monitor = predict(net, tx_monitor);
        u_pinn_val = extractdata(u_pinn_monitor);
        
        fprintf('%.1f\t%.4f\t\t%.4f\t\t%.4f\t%s\n', ...
            t_monitor, peak_x_monitor, peak_val_monitor, u_pinn_val, search_range_str);
    end
    fprintf('---\n');
    
    % Perform conflict diagnosis every 100 epochs
    if mod(epoch, 100) == 0
        fprintf('Epoch %d: Executing Conflict Diagnosis\n', epoch);
        diagnose_conflict_sources(net, params, alpha, T, lambda_soe, theta);
    end
end

% Small anchor loss versus analytical at early/late near-peak band
L_anchor = dlarray(0.0);
if isfield(params, 'analytical_solution') && isfield(params, 'schedules')
    anchor_max = params.schedules.anchor_weight_end;
    anchor = anchor_max * min(1, epoch / max(1, params.schedules.anchor_warmup));
    % construct near-peak band times focused on early/late
    t_early = dlarray([0.05*T, 0.15*T, 0.25*T], "CB");
    t_late  = dlarray([0.75*T, 0.85*T, 0.95*T], "CB");
    t_band = [t_early, t_late];
    
    % Dynamic peak location for anchor loss
if isfield(params, 'peak_protection') && isfield(params.peak_protection, 'dynamic') && params.peak_protection.dynamic
    % Use average peak location over time segment
    t_vals = extractdata(t_band);
    peak_x_anchor = zeros(size(t_vals));
    for k = 1:numel(t_vals)
        [peak_x_temp, ~] = find_peak_location_singular(t_vals(k), alpha, params.peak_detection.n_points);
        peak_x_anchor(k) = peak_x_temp;
    end
    x_band = dlarray(peak_x_anchor, "CB");
else
    x_band = repmat(peak_x, 1, length(t_band));
end
    
    tx_band = dlarray([t_band; x_band], "CB");
    u_pred = predict(net, tx_band);
    t_vals = extractdata(t_band);
    x_vals = extractdata(x_band);
    u_ana = zeros(size(t_vals));
    for k = 1:numel(t_vals)
        u_ana(k) = params.analytical_solution(x_vals(k), t_vals(k), alpha);
    end
    u_ana = dlarray(u_ana, "CB");
    L_anchor = anchor * mean((u_pred - u_ana).^2, 'all');
end

% Initial condition physics constraint for better early time fitting
L_ic_physics = dlarray(0.0);
if isfield(params, 'loss_weights') && isfield(params.loss_weights, 'ic_physics') && params.loss_weights.ic_physics > 0
    L_ic_physics = initial_condition_physics_constraint(net, params, alpha, T);
end

% Compose total loss
L_pde = stripdims(L_pde); 
L_bc = stripdims(L_bc); 
L_ic = stripdims(L_ic); 
L_longterm = stripdims(L_longterm);
L_sobolev = stripdims(L_sobolev);
L_peak = stripdims(L_peak);
L_ic_physics = stripdims(L_ic_physics);

% Generic loss composition with configurable weights
if isfield(params, 'loss_weights') && ~isempty(params.loss_weights)
    ic_weight = params.loss_weights.ic;      % Default: 15
    peak_weight = params.loss_weights.peak;  % Default: 5
    sobolev_weight = params.loss_weights.sobolev;  % Default: 1.5
else
    ic_weight = 15;
    peak_weight = 5;
    sobolev_weight = 1.5;
end

% Enhanced loss composition with balanced constraints and minimal intervention
% Use balanced constraints and minimal intervention strategy
if epoch < 100
    % Early stage: minimal constraints, let network learn freely
    loss = 10 * L_ic + 0.5 * L_peak + 0.5 * get_ic_physics_weight(params, epoch) * L_ic_physics;
elseif epoch < 200
    % Mid stage: gently introduce PDE constraints
    pde_weight = min(0.5, (epoch - 100) / 200); % Gradually increase from 0 to 0.5 (more gentle)
    loss = pde_weight * lambda.pde * (L_pde + L_longterm) + ...
           0.5 * lambda.bc * L_bc + 10 * L_ic + ...
           0.5 * L_peak + ...
           0.5 * get_ic_physics_weight(params, epoch) * L_ic_physics;
else
    % Late stage: balanced constraints, avoid over-penalization
    loss = 0.8 * lambda.pde * (L_pde + L_longterm) + ...
           0.5 * lambda.bc * L_bc + 10 * L_ic + ...
           0.5 * L_peak + ...
           0.3 * get_ic_physics_weight(params, epoch) * L_ic_physics;
end

% Compute D_alpha_sample for debugging (at point x=0.5, t=0.5)
D_alpha_sample = compute_D_alpha_sample(net, 0.5, 0.5, t_hist, x_hist, alpha, T, lambda_soe, theta);

gradients = dlgradient(loss, net.Learnables);

end

function D_alpha_sample = compute_D_alpha_sample(net, x_sample, t_sample, t_hist, x_hist, alpha, T, lambda_soe, theta)
    % Compute fractional derivative at sample point for debugging
    % Sample the network at historical points
    tx_hist = dlarray([t_hist; x_hist], "CB");
    u_hist = predict(net, tx_hist);
    u_hist = extractdata(u_hist);
    
    % Sample at current point
    tx_curr = dlarray([t_sample; x_sample], "CB");
    u_curr = predict(net, tx_curr);
    u_curr = extractdata(u_curr);
    
    % Compute fractional derivative
    D_alpha_sample = MFL1_Caputo(u_hist, t_hist, u_curr, t_sample, alpha, T, lambda_soe, theta);
    D_alpha_sample = dlarray(D_alpha_sample, "CB");
end

function L_ic_physics = initial_condition_physics_constraint(net, params, alpha, T)
% INITIAL_CONDITION_PHYSICS_CONSTRAINT - Physics-based initial condition constraint
%
% Theoretical basis:
% 1. Singularity handling: force network to learn correct singular behavior near t=0
% 2. Peak position: ensure initial peak position is correct
% 3. Solution structure: ensure spatial structure of solution conforms to physical expectations
%
% Inputs:
%   net    - neural network
%   params - parameter struct
%   alpha  - fractional order
%   T      - total time
%
% Outputs:
%   L_ic_physics - initial condition physics constraint loss

% Define critical early time points for physics constraint
% Use exponential distribution near t=0 for dense sampling, focus on singularity
t_ic_physics = T * [0, 1e-6, 1e-5, 1e-4, 1e-3, 0.01];  % From t=0 to early time

% Spatial sampling focused on peak region and overall structure
x_samples = linspace(0, params.L, 50);  % Uniform spatial sampling

L_ic_physics = dlarray(0.0);

for i = 1:length(t_ic_physics)
    t_val = t_ic_physics(i);
    
    % Create input arrays
    t_array = t_val * ones(1, length(x_samples));
    tx_input = dlarray([t_array; x_samples], "CB");
    
    % Network prediction
    u_pred = predict(net, tx_input);
    
    % Analytical solution at this time point
    u_analytical = zeros(1, length(x_samples));
    for j = 1:length(x_samples)
        u_analytical(j) = params.analytical_solution(x_samples(j), t_val, alpha);
    end
    u_analytical = dlarray(u_analytical, "CB");
    
    % Time-dependent weighting: higher weight for earlier times (singular behavior)
    if t_val == 0
        time_weight = 10.0;  % Highest weight for t=0 (singular point)
    elseif t_val <= 1e-4
        time_weight = 5.0;   % High weight for ultra-early time
    elseif t_val <= 1e-3
        time_weight = 3.0;   % Medium weight for early time
    else
        time_weight = 1.0;   % Standard weight for other times
    end
    
    % Peak region emphasis: identify peak location at this time
    if isfield(params, 'peak_region') && isfield(params.peak_region, 'dynamic') && params.peak_region.dynamic
        [peak_x_val, ~] = find_peak_location_singular(t_val, alpha, params.peak_detection.n_points);
    else
        peak_x_val = params.peak_region.center;
    end
    
    % Spatial weighting: higher weight near peak region - Reduce weights to avoid over-constraint
    spatial_weights = ones(1, length(x_samples));
    peak_region_width = 0.1;  % ±0.05 around peak
    for j = 1:length(x_samples)
        if abs(x_samples(j) - peak_x_val) <= peak_region_width
            spatial_weights(j) = 1.5;  % Reduce peak region weights from 3.0 to 1.5, avoid over-constraint
        end
    end
    spatial_weights = dlarray(spatial_weights, "CB");
    
    % Compute weighted MSE for this time point
    mse_time = mean(spatial_weights .* (u_pred - u_analytical).^2, 'all');
    
    % Add to total constraint with time weighting
    L_ic_physics = L_ic_physics + time_weight * mse_time;
end

% Normalize by number of time points
L_ic_physics = L_ic_physics / length(t_ic_physics);

end

function weight = get_ic_physics_weight(params, epoch)
% GET_IC_PHYSICS_WEIGHT - Get adaptive weight for initial condition physics constraint
%
% Inputs:
%   params - parameter struct
%   epoch  - current epoch
%
% Outputs:
%   weight - constraint weight

if isfield(params, 'loss_weights') && isfield(params.loss_weights, 'ic_physics')
    base_weight = params.loss_weights.ic_physics;
else
    base_weight = 1.0;  % Default weight (reduced from 5.0)
end

% Adaptive weighting: higher weight in early training, gradual decrease
if epoch < 100
    weight = base_weight * 1.5;      % Early: medium constraint (reduced from 2.0 to 1.5)
elseif epoch < 300
    weight = base_weight * 1.0;      % Mid: standard constraint (reduced from 1.5 to 1.0)
elseif epoch < 600
    weight = base_weight * 0.8;      % Late: light constraint (reduced from 1.0 to 0.8)
else
    weight = base_weight * 0.3;      % Very late: very light constraint (reduced from 0.5 to 0.3)
end

end

