function [peak_x, peak_value] = detect_peak_position(t, params)
% DETECT_PEAK_POSITION - Dynamically detect the peak position at given time points
%
% Inputs:
%   t      - time points (scalar or vector)
%   params - parameter structure
%
% Outputs:
%   peak_x     - peak position (scalar)
%   peak_value - peak value (scalar)

% Persistent cache for peak detection results
persistent peak_cache cache_times cache_positions cache_values cache_size

% Initialize cache if empty
if isempty(peak_cache)
    peak_cache = containers.Map('KeyType', 'double', 'ValueType', 'any');
    cache_times = [];
    cache_positions = [];
    cache_values = [];
    cache_size = 100; % Default cache size
end

% Input validation
if isempty(t) || isempty(params)
    warning('Invalid inputs to detect_peak_position, using default position x = 0.5');
    peak_x = 0.5 * params.L;
    % Use analytical solution at default position instead of hardcoded value
    try
        peak_value = analytical_solution(peak_x, t, 0.8, 1.296e-2, 0.1296, 35);
    catch
        peak_value = 0.5; % Fallback value
    end
    return;
end

% Ensure t is scalar for peak detection
if ~isscalar(t)
    warning('Time input should be scalar, using first element');
    t = t(1);
end

% Check cache first for performance optimization
constants = numerical_constants();
if constants.peak_detection_cache && isKey(peak_cache, t)
    cached_result = peak_cache(t);
    peak_x = cached_result.position;
    peak_value = cached_result.value;
    
    % Debug output (if enabled)
    if constants.verbose_peak_detection
        fprintf('Cache hit for t = %.4f: position = %.4f, value = %.4f\n', t, peak_x, peak_value);
    end
    return;
end

% Calculate expected peak position based on convection velocity
% For time-fractional advection-diffusion, peak moves approximately as: x_peak(t) = 0.5 + v*t
% Scale factor accounts for the fact that peak doesn't move linearly with time due to diffusion
expected_peak_x = 0.5 + (params.v / params.D) * t * 0.05; % Reduced scale factor for more realistic movement
expected_peak_x = max(0.1, min(0.9, expected_peak_x)); % Ensure within domain bounds

% Use analytical solution to refine the peak position
try
    % Validate params structure
    if ~exist('params', 'var') || isempty(params) || ~isfield(params, 'L')
        warning('Invalid params structure in detect_peak_position, using expected position');
        peak_x = expected_peak_x;
        % Use analytical solution at expected position
        try
            peak_value = analytical_solution(peak_x, t, params.alpha, params.D, params.v, params.N_terms);
        catch
            peak_value = 0.5; % Fallback value
        end
        return;
    end
    
    % Create a fine grid around the expected peak position
    peak_search_width = 0.3; % Wider search width to capture peak movement
    x_start = max(0, expected_peak_x - peak_search_width);
    x_end = min(params.L, expected_peak_x + peak_search_width);
    x_fine = linspace(x_start, x_end, 150); % Increased resolution
    
    u_analytical = analytical_solution(x_fine, t, params.alpha, params.D, params.v, params.N_terms);
    
    % Debug: check analytical solution
    if any(isnan(u_analytical)) || any(isinf(u_analytical))
        warning('Analytical solution contains NaN or Inf values at t = %.4f', t);
    end
    
    % Find peak position and value
    [peak_value, peak_idx] = max(u_analytical);
    peak_x = x_fine(peak_idx);
    
    % Debug: print peak detection results (only when verbose mode is enabled)
    if isfield(params, 'verbose_peak_detection') && params.verbose_peak_detection
        fprintf('Peak detection at t = %.4f: expected = %.4f, actual = %.4f, value = %.4f\n', ...
                t, expected_peak_x, peak_x, peak_value);
    end
    
    % Basic validation
    if isempty(peak_idx) || isnan(peak_x) || isinf(peak_x)
        warning('Peak detection failed, using expected position');
        peak_x = expected_peak_x;
        % Use analytical solution at expected position
        try
            peak_value = analytical_solution(peak_x, t, params.alpha, params.D, params.v, params.N_terms);
        catch
            peak_value = 0.5; % Fallback value
        end
    end
    
catch ME
    warning('Peak detection error: %s, using expected position', ME.message);
    peak_x = expected_peak_x;
    % Use analytical solution at expected position
    try
        peak_value = analytical_solution(peak_x, t, params.alpha, params.D, params.v, params.N_terms);
    catch
        peak_value = 0.5; % Fallback value
    end
end

% Cache the result for future use (if not already cached)
if constants.peak_detection_cache && ~isKey(peak_cache, t)
    % Limit cache size to prevent memory issues
    if length(peak_cache) >= cache_size
        % Remove oldest entry
        oldest_key = cache_times(1);
        remove(peak_cache, oldest_key);
        cache_times = cache_times(2:end);
        cache_positions = cache_positions(2:end);
        cache_values = cache_values(2:end);
    end
    
    % Add new result to cache
    peak_cache(t) = struct('position', peak_x, 'value', peak_value);
    cache_times = [cache_times, t];
    cache_positions = [cache_positions, peak_x];
    cache_values = [cache_values, peak_value];
end

end
