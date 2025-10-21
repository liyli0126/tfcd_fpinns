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
    constants = numerical_constants();
    cache_size = constants.peak_detection_cache_size; % Use configurable cache size
end

% Input validation
if isempty(t) || isempty(params)
    warning('Invalid inputs to detect_peak_position, using default position x = 0.5');
    % Use default values if params is not available
    if isempty(params)
        peak_x = 0.5;  % Default domain length = 1
        peak_value = 0.5; % Simple fallback
    else
        peak_x = 0.5 * params.L;
        % Use params values instead of hardcoded values
        try
            peak_value = analytical_solution(peak_x, t, params.alpha, params.D, params.v, params.N_terms);
        catch
            peak_value = 0.5; % Fallback value
        end
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

% Dynamic peak detection based on analytical solution
% Instead of using fixed movement formula, sample the analytical solution over the domain
% to find the actual peak position at time t

% Create a coarse grid first for initial peak estimation
x_coarse = linspace(0, params.L, 50);
try
    u_coarse = analytical_solution(x_coarse, t, params.alpha, params.D, params.v, params.N_terms);
    [~, coarse_peak_idx] = max(u_coarse);
    coarse_peak_x = x_coarse(coarse_peak_idx);
    
    % Use analytical solution-based peak as the expected position
    expected_peak_x = coarse_peak_x;
catch
    % Fallback: peak is fixed at x=0.5 for solution u(x,t) = x(1-x)*E_α(-t^α)
    expected_peak_x = 0.5 * params.L;
end

% Ensure expected position is within reasonable bounds
expected_peak_x = max(constants.peak_search_boundary_min * params.L, min(constants.peak_search_boundary_max * params.L, expected_peak_x));

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
    peak_search_width = constants.peak_search_width; % Use configurable search width
    x_start = max(0, expected_peak_x - peak_search_width);
    x_end = min(params.L, expected_peak_x + peak_search_width);
    x_fine = linspace(x_start, x_end, constants.peak_search_resolution); % Use configurable resolution
    
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
