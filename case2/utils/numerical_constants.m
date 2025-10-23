function constants = numerical_constants()
% NUMERICAL_CONSTANTS - Return numerical constants for numerical stability
%
% Outputs:
%   constants - struct containing numerical constants

constants = struct();

% Basic numerical stability constants
constants.epsilon = 1e-8;        % Small value for numerical stability (increased from 1e-12 to 1e-8)
constants.tolerance_inf = 1e6;    % Infinity tolerance
constants.tolerance_nan = 1e-12;  % NaN tolerance

% Time weighting constants
constants.ultra_early_threshold = 0.0001;  % Ultra-early time threshold
constants.early_threshold = 0.001;          % Early time threshold
constants.mid_early_threshold = 0.01;      % Mid-early time threshold
constants.late_early_threshold = 0.1;      % Late-early time threshold
constants.longterm_threshold = 0.7;        % Long-term prediction threshold

% Time weighting values
constants.ultra_early_weight = 0.05;       % Ultra-early weight
constants.early_weight_base = 0.05;        % Early base weight
constants.early_weight_boost = 0.07;       % Early boost weight
constants.mid_early_weight_base = 0.12;    % Mid-early base weight
constants.mid_early_weight_boost = 0.08;   % Mid-early boost weight
constants.late_early_weight = 0.04;        % Late-early weight

% Time weighting smoothing parameters
constants.mid_time_bell_width = 0.18;      % Mid-time bell width
constants.mid_time_bell_amplitude = 0.8;   % Mid-time bell amplitude
constants.early_boost_amplitude = 1.0;     % Early boost amplitude

% Loss weighting constants
constants.longterm_penalty = 3.0;           % Long-term prediction penalty factor
constants.residual_extreme_ratio = 0.05;   % Residual extreme ratio (5%)
constants.residual_extreme_weight = 10;    % Residual extreme weight

% Sobolev regularization constants
constants.sobolev_perturbation = 0.01;     % Sobolev perturbation amplitude

% Peak detection constants
constants.peak_search_points = 1000;        % Peak search points
constants.peak_region_ratio = 0.2;         % Peak region ratio

% Sampling parameter constants
constants.region_width_ultra_early = 0.1;  % Ultra early region width (±0.05)
constants.region_width_early = 0.1;        % Early region width (±0.05)
constants.region_width_mid = 0.1;          % Mid region width (±0.05)
constants.region_width_late = 0.1;         % Late region width (±0.05)

% Peak sampling ratios
constants.peak_frac_ultra_early = 0.6;     % Ultra early peak sampling ratio (increased from 0.4)
constants.peak_frac_early = 0.4;          % Early peak sampling ratio (increased from 0.3)
constants.peak_frac_mid = 0.15;           % Mid peak sampling ratio (decreased from 0.2)
constants.peak_frac_late = 0.15;         % Late peak sampling ratio (increased from 0.1)

% Boundary threshold constants
constants.front_threshold = 0.2;          % Front boundary threshold
constants.rear_threshold = 0.8;           % Rear boundary threshold
constants.front_weight = 5;               % Front boundary weight
constants.rear_weight = 5;                % Rear boundary weight      

% Peak protection constants
constants.peak_improve_min = 0.02;        % Peak improvement minimum
constants.peak_improve_max = 0.10;        % Peak improvement maximum
constants.analytical_mix = 0.25;          % Analytical solution mix ratio

% Time singularity constants
constants.singularity_early_threshold = 1e-3;  % Singularity early threshold
constants.singularity_mid_threshold = 0.1;     % Singularity mid threshold
constants.singularity_late_threshold = 1.0;    % Singularity late threshold

end
