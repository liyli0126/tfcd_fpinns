function [peak_x, peak_value] = find_peak_location_singular(t, alpha, n_points)
% FIND_PEAK_LOCATION_SINGULAR - Peak location detection considering time singularity
%
% Inputs:
%   t - time point
%   alpha - fractional derivative order
%   n_points - number of sampling points
%
% Outputs:
%   peak_x - peak location
%   peak_value - peak value

% Get numerical constants
constants = numerical_constants();

% Get adaptive search range based on time
[x_min, x_max] = get_adaptive_search_range(t, alpha);

% Dense sampling in limited range
x_samples = linspace(x_min, x_max, n_points);

% Calculate analytical solution
u_ana = zeros(size(x_samples));
for i = 1:length(x_samples)
    u_ana(i) = analytical_solution_case2(x_samples(i), t, alpha);
end

% Find peak location and value
[peak_value, idx] = max(u_ana);
peak_x = x_samples(idx);

end
