function [region_width, n_peak_points] = get_adaptive_sampling_params(t, params)
% GET_ADAPTIVE_SAMPLING_PARAMS - Get adaptive sampling parameters based on time
%
% Inputs:
%   t - time point
%   params - parameter structure
%
% Outputs:
%   region_width - sampling region width
%   n_peak_points - number of peak region sampling points

% Get numerical constants
constants = numerical_constants();

% Adjust sampling parameters based on time
if t < constants.singularity_early_threshold
    % Ultra-early: narrow region, high precision
    region_width = constants.region_width_ultra_early;
    n_peak_points = round(params.miniNr * constants.peak_frac_ultra_early);
elseif t < constants.singularity_mid_threshold
    % Early: medium region
    region_width = constants.region_width_early;
    n_peak_points = round(params.miniNr * constants.peak_frac_early);
elseif t < constants.singularity_late_threshold
    % Mid: medium region
    region_width = constants.region_width_mid;
    n_peak_points = round(params.miniNr * constants.peak_frac_mid);
else
    % Late: narrow region
    region_width = constants.region_width_late;
    n_peak_points = round(params.miniNr * constants.peak_frac_late);
end

end
