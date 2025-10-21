function weights = weight_functions(type, params, varargin)
% WEIGHT_FUNCTIONS - Unified weight calculation for PINN training
%
% Inputs:
%   type   - weight type: 'peak_region', 'boundary_region', 'time_weighting', 
%            'local_weighting', 'rar_selection', 'ic_weighting'
%   params - parameter struct containing domain and sampling parameters
%   varargin - additional arguments depending on weight type
%
% Outputs:
%   weights - calculated weights (dlarray or numeric array)
%
% Note: This function now supports both dlarray and numeric inputs
%       to reduce the need for extractdata conversions

switch lower(type)
    case 'peak_region'
        % Peak region weighting for x ≈ 0.5 where sin(πx) has maximum
        x_coords = varargin{1};
        % For peak region, we need both x and t coordinates for adaptive weighting
        if length(varargin) >= 2
            t_coords = varargin{2};
        else
            % If no time coordinates provided, create default ones
            t_coords = zeros(size(x_coords));
        end
        weights = calculate_peak_region_weights(x_coords, params.L, t_coords, params);
        
    case 'boundary_region'
        % Boundary region weighting with time-stratified emphasis
        x_coords = varargin{1};
        t_coords = varargin{2};
        weights = calculate_boundary_region_weights(x_coords, t_coords, params);
        
    case 'time_weighting'
        % Time-dependent weighting for enhanced temporal accuracy
        t_coords = varargin{1};
        weights = calculate_time_weighting(t_coords, params.T);
        
    case 'local_weighting'
        % Local weighting based on spatial position and residual magnitude
        x_coords = varargin{1};
        residual_values = varargin{2};
        weights = calculate_local_weighting(x_coords, residual_values, params.L);
        
    case 'rar_selection'
        % RAR selection weighting for strategic point selection
        x_coords = varargin{1};
        t_coords = varargin{2};
        weights = calculate_rar_selection_weights(x_coords, t_coords, params);
        
    case 'ic_weighting'
        % Initial condition weighting with peak region emphasis
        x_coords = varargin{1};
        weights = calculate_ic_weighting(x_coords, params.L);
        
    case 'longterm_weighting'
        % Long-term prediction weighting for t > 0.7T
        t_coords = varargin{1};
        weights = calculate_longterm_weighting(t_coords, params.T);
        
    otherwise
        error('Unknown weight type: %s', type);
end

end

function weights = calculate_peak_region_weights(x_coords, L, t_coords, params)
% Calculate adaptive weights for peak region based on dynamic peak position
% Higher weights for regions where solution has maximum values at each time
% Supports both dlarray and numeric inputs

% Handle dlarray input
if isa(x_coords, 'dlarray')
    x_vals = extractdata(x_coords);
    t_vals = extractdata(t_coords);
    weights = calculate_adaptive_peak_weights_numeric(x_vals, t_vals, L, params);
    weights = dlarray(weights, "CB");
else
    weights = calculate_adaptive_peak_weights_numeric(x_coords, t_coords, L, params);
end

end

function weights = calculate_adaptive_peak_weights_numeric(x_coords, t_coords, L, params)
% Numeric implementation of adaptive peak region weights

weights = ones(size(x_coords));

% Get constants
constants = numerical_constants();

if constants.adaptive_peak_enabled
    % Adaptive peak weighting based on dynamic peak position
    unique_times = unique(t_coords);
    
    for i = 1:length(unique_times)
        t = unique_times(i);
        
        % Calculate expected peak position based on convection velocity
        % For time-fractional advection-diffusion, peak moves approximately as: x_peak(t) = 0.5 + v*t
        expected_peak_x = 0.5 + (params.v / params.D) * t * 0.05; % Scale factor for numerical stability
        expected_peak_x = max(0.1, min(0.9, expected_peak_x)); % Ensure within domain bounds
        
        % Find coordinates at current time
        time_mask = (t_coords == t);
        x_at_time = x_coords(time_mask);
        
        % Calculate distances to expected peak position
        distances = abs(x_at_time - expected_peak_x);
        
        % Assign weights based on distance to expected peak
        peak_region_mask = distances < 0.1 * L;
        near_peak_mask = (distances >= 0.1 * L) & (distances < 0.2 * L);
        
        weights(time_mask & peak_region_mask) = 10.0;      % Peak region
        weights(time_mask & near_peak_mask) = 6.0;         % Near peak region
    end
else
    % Fallback to fixed peak weighting
    weights = calculate_fixed_peak_weights_numeric(x_coords, L);
end

end

function weights = calculate_fixed_peak_weights_numeric(x_coords, L)
% Fixed peak region weights (original implementation)

weights = ones(size(x_coords));

% Peak region: x ∈ [0.4L, 0.6L] - highest weight
peak_mask = (x_coords > 0.4*L) & (x_coords < 0.6*L);
weights(peak_mask) = 8.0;

% Near-peak regions: x ∈ [0.3L, 0.4L] and [0.6L, 0.7L] - medium weight
near_peak_mask = ((x_coords > 0.3*L) & (x_coords <= 0.4*L)) | ...
                 ((x_coords >= 0.6*L) & (x_coords < 0.7*L));
weights(near_peak_mask) = 4.0;

% Front and rear regions: x < 0.2L or x > 0.8L - moderate weight
front_rear_mask = (x_coords < 0.2*L) | (x_coords > 0.8*L);
weights(front_rear_mask) = 5.0;

end

function weights = calculate_boundary_region_weights(x_coords, t_coords, params)
% Calculate boundary region weights with time-stratified emphasis
% Higher weights for early time boundary effects
% Supports both dlarray and numeric inputs

% Handle dlarray inputs
if isa(x_coords, 'dlarray')
    x_vals = extractdata(x_coords);
    t_vals = extractdata(t_coords);
    weights = calculate_boundary_region_weights_numeric(x_vals, t_vals, params);
    weights = dlarray(weights, "CB");
else
    weights = calculate_boundary_region_weights_numeric(x_coords, t_coords, params);
end

end

function weights = calculate_boundary_region_weights_numeric(x_coords, t_coords, params)
% Numeric implementation of boundary region weights

weights = ones(size(x_coords));

% Identify boundary regions (avoid exact boundaries where u=0)
boundary_mask = (x_coords < 0.1*params.L) | (x_coords > 0.9*params.L);

if any(boundary_mask)
    % Time-stratified boundary weighting
    t_vals = t_coords;
    
    % Early time boundary (t < 0.3T) - highest weight
    early_time_mask = boundary_mask & (t_vals < 0.3*params.T);
    weights(early_time_mask) = 6.0;
    
    % Middle time boundary (0.3T ≤ t ≤ 0.7T) - medium weight
    mid_time_mask = boundary_mask & (t_vals >= 0.3*params.T) & (t_vals <= 0.7*params.T);
    weights(mid_time_mask) = 3.0;
    
    % Late time boundary (t > 0.7T) - lower weight
    late_time_mask = boundary_mask & (t_vals > 0.7*params.T);
    weights(late_time_mask) = 2.0;
end

end

function weights = calculate_time_weighting(t_coords, T)
% Calculate time-dependent weights for enhanced temporal accuracy
% Higher weights for early and late times where dynamics are complex
% Supports both dlarray and numeric inputs

% Handle dlarray input
if isa(t_coords, 'dlarray')
    t_vals = extractdata(t_coords);
    weights = calculate_time_weighting_numeric(t_vals, T);
    weights = dlarray(weights, "CB");
else
    weights = calculate_time_weighting_numeric(t_coords, T);
end

end

function weights = calculate_time_weighting_numeric(t_coords, T)
% Numeric implementation of time weighting

t_vals = t_coords;
weights = 1 + 8 * (t_vals / T);  % Linear increase with time

% Additional emphasis on critical time regions
early_mask = t_vals < 0.3*T;
late_mask = t_vals > 0.7*T;

weights(early_mask) = weights(early_mask) * 1.5;  % Early time emphasis
weights(late_mask) = weights(late_mask) * 1.3;    % Late time emphasis

end

function weights = calculate_local_weighting(x_coords, residual_values, L)
% Calculate local weights based on spatial position and residual magnitude
% Combines spatial and residual-based weighting
% Supports both dlarray and numeric inputs

% Handle dlarray inputs
if isa(x_coords, 'dlarray') || isa(residual_values, 'dlarray')
    x_vals = extractdata(x_coords);
    res_vals = extractdata(residual_values);
    weights = calculate_local_weighting_numeric(x_vals, res_vals, L);
    weights = dlarray(weights, "CB");
else
    weights = calculate_local_weighting_numeric(x_coords, residual_values, L);
end

end

function weights = calculate_local_weighting_numeric(x_coords, residual_values, L)
% Numeric implementation of local weighting

weights = ones(size(x_coords));

% Spatial weighting (peak region emphasis)
spatial_weights = calculate_fixed_peak_weights_numeric(x_coords, L);

% Residual-based weighting (emphasize regions with high residuals)
abs_res = abs(residual_values);
if numel(abs_res) > 20
    % Top 5% residual points get extra weight
    [~, idx_ext] = maxk(abs_res, round(0.05*numel(abs_res)));
    residual_weights = ones(size(weights));
    residual_weights(idx_ext) = 10.0;
else
    residual_weights = ones(size(weights));
end

% Combine spatial and residual weights
weights = spatial_weights .* residual_weights;

end

function weights = calculate_rar_selection_weights(x_coords, t_coords, params)
% Calculate RAR selection weights for strategic point selection
% Balances boundary, peak, and interior region sampling
% Supports both dlarray and numeric inputs

% Handle dlarray inputs
if isa(x_coords, 'dlarray') || isa(t_coords, 'dlarray')
    x_vals = extractdata(x_coords);
    t_vals = extractdata(t_coords);
    weights = calculate_rar_selection_weights_numeric(x_vals, t_vals, params);
    weights = dlarray(weights, "CB");
else
    weights = calculate_rar_selection_weights_numeric(x_coords, t_coords, params);
end

end

function weights = calculate_rar_selection_weights_numeric(x_coords, t_coords, params)
% Numeric implementation of RAR selection weights

weights = ones(size(x_coords));

% Identify different regions
boundary_mask = (x_coords < 0.1*params.L) | (x_coords > 0.9*params.L);
peak_mask = (x_coords > 0.4*params.L) & (x_coords < 0.6*params.L);
interior_mask = ~boundary_mask & ~peak_mask;

% Boundary region: reduced weight (since we have fewer boundary points)
weights(boundary_mask) = 0.8;

% Peak region: increased weight (high accuracy needed)
weights(peak_mask) = 1.3;

% Interior region: standard weight
weights(interior_mask) = 1.0;

end

function weights = calculate_ic_weighting(x_coords, L)
% Calculate initial condition weights with peak region emphasis
% Higher weights for regions where initial condition accuracy is critical
% Supports both dlarray and numeric inputs

% Handle dlarray input
if isa(x_coords, 'dlarray')
    x_vals = extractdata(x_coords);
    weights = calculate_ic_weighting_numeric(x_vals, L);
    weights = dlarray(weights, "CB");
else
    weights = calculate_ic_weighting_numeric(x_coords, L);
end

end

function weights = calculate_ic_weighting_numeric(x_coords, L)
% Numeric implementation of initial condition weighting

weights = ones(size(x_coords));

% Peak region emphasis in initial condition
peak_mask = (x_coords > 0.4*L) & (x_coords < 0.6*L);
weights(peak_mask) = 3.0;

% Near-peak regions
near_peak_mask = ((x_coords > 0.3*L) & (x_coords <= 0.4*L)) | ...
                 ((x_coords >= 0.6*L) & (x_coords < 0.7*L));
weights(near_peak_mask) = 2.0;

end

function weights = calculate_longterm_weighting(t_coords, T)
% Calculate long-term prediction weights for t > 0.7T
% Emphasizes accuracy in long-term predictions
% Supports both dlarray and numeric inputs

% Handle dlarray input
if isa(t_coords, 'dlarray')
    t_vals = extractdata(t_coords);
    weights = calculate_longterm_weighting_numeric(t_vals, T);
    weights = dlarray(weights, "CB");
else
    weights = calculate_longterm_weighting_numeric(t_coords, T);
end

end

function weights = calculate_longterm_weighting_numeric(t_coords, T)
% Numeric implementation of long-term weighting

t_vals = t_coords;
weights = ones(size(t_vals));

% Long-term region (t > 0.7T) gets higher weight
longterm_mask = (t_vals > 0.7*T);
weights(longterm_mask) = 3.0;

end