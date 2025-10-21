function L_peak_pos = calculate_peak_position_loss(net, params)
% CALCULATE_PEAK_POSITION_LOSS - Specialized loss to enforce correct peak position
%
% This function computes a loss term that specifically penalizes deviations
% of the predicted peak position from the analytical peak position at x=0.5
%
% Inputs:
%   net    - neural network
%   params - parameters struct
%
% Outputs:
%   L_peak_pos - peak position correction loss

% Get numerical constants
constants = numerical_constants();

% Time points to check peak position using configurable parameters
t_check = constants.peak_position_check_times * params.T;
L_peak_pos = 0;

for i = 1:length(t_check)
    t_curr = t_check(i);
    
    % Create a fine grid around the expected peak position using configurable parameters
    peak_center = constants.peak_center * params.L;
    search_half_width = constants.peak_search_half_width * params.L;
    x_peak_grid = linspace(peak_center - search_half_width, ...
                          peak_center + search_half_width, ...
                          constants.peak_search_resolution);
    t_grid = t_curr * ones(size(x_peak_grid));
    
    % Get network predictions
    input_grid = dlarray([t_grid; x_peak_grid], "CB");
    u_pred = predict(net, input_grid);
    u_pred_vals = extractdata(u_pred);
    
    % Find predicted peak position
    [~, max_idx] = max(u_pred_vals);
    x_pred_peak = x_peak_grid(max_idx);
    
    % Analytical peak position from constants
    x_analytical_peak = constants.peak_center * params.L;
    
    % Peak position error
    peak_position_error = (x_pred_peak - x_analytical_peak)^2;
    
    % Time-dependent weighting using configurable factor
    time_weight = 1.0 + constants.peak_position_time_weight_factor * (t_curr / params.T);
    
    L_peak_pos = L_peak_pos + time_weight * peak_position_error;
end

% Normalize by number of time points
L_peak_pos = L_peak_pos / length(t_check);

% Convert to dlarray
L_peak_pos = dlarray(L_peak_pos);

end
