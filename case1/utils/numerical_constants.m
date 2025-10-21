function constants = numerical_constants()
% NUMERICAL_CONSTANTS - Centralized numerical stability constants
%
% Returns:
%   constants - struct containing all numerical stability parameters
%
% This function centralizes all numerical constants used throughout
% the PINN solver to ensure consistency and easy maintenance.

constants = struct();

% Numerical stability parameters
constants.epsilon = 1e-12;        % Small value for numerical stability
constants.epsilon_loss = 1e-8;    % Small value for loss function stability
constants.epsilon_grad = 1e-10;   % Small value for gradient stability

% Perturbation parameters for numerical differentiation
constants.perturbation_factor = 0.01;  % Factor for spatial perturbation in Sobolev regularization

% Tolerance parameters
constants.tolerance_nan = 1e-15;   % Tolerance for NaN detection
constants.tolerance_inf = 1e+15;   % Tolerance for Inf detection

% Weight decay parameters
constants.weight_decay = 0.95;     % Exponential decay for loss balancing

% RAR sampling parameters
constants.rar_candidate_multiplier = 3;  % Multiply mini-batch size for RAR candidates
constants.rar_selection_ratio = 0.05;   % Top 5% residual points for extreme weighting

% Time segmentation parameters
constants.default_time_segments = 5;     % Default number of time segments for stratified sampling

% Boundary sampling parameters
constants.boundary_avoidance = 0.02;     % Avoid sampling exactly at boundaries (2% of domain)
constants.boundary_near_threshold = 0.1; % Threshold for "near boundary" classification

% Peak region parameters
constants.peak_center = 0.5;             % Center of peak region (x = 0.5)
constants.peak_width = 0.2;             % Width of peak region (±0.1 around center)
constants.near_peak_width = 0.4;        % Width of near-peak region (±0.2 around center)

% Adaptive peak sampling parameters
constants.adaptive_peak_enabled = true;    % Enable adaptive peak sampling
constants.peak_detection_resolution = 200; % Spatial resolution for peak detection
constants.peak_tracking_threshold = 0.1;   % Threshold for peak tracking
constants.peak_smoothing_factor = 0.8;     % Smoothing factor for peak position
constants.adaptive_peak_width_base = 0.1;  % Base width for adaptive peak sampling
constants.adaptive_peak_width_factor = 0.05; % Factor for velocity-dependent width adjustment

% Performance optimization flags
constants.verbose_peak_detection = false;     % Enable/disable peak detection output
constants.peak_detection_cache = true;        % Enable caching for peak detection
constants.peak_detection_cache_size = 100;    % Cache size for peak detection results

% Path guidance parameters
constants.path_guidance_enabled = true;           % Enable path guidance constraint
constants.path_guidance_threshold = 0.1;         % Minimum expected peak value
constants.path_guidance_weight = 0.3;            % Weight for path guidance in combined loss
constants.peak_protection_weight = 0.7;          % Weight for peak protection in combined loss

% Peak position parameters
constants.initial_peak_position = 0.5;            % Initial peak position for sin(πx)
constants.peak_path_tolerance = 0.05;             % Tolerance for peak path deviation

% Fourier feature mapping parameters
constants.fourier_features_enabled = false;    % Disable Fourier feature mapping (temporary)
constants.fourier_mapping_size = 256;        % Dimension of Fourier feature mapping
constants.fourier_sigma = 15.0;              % Scaling factor for frequency distribution
constants.fourier_include_original = true;   % Include original coordinates in mapping

% Adaptive activation function parameters
constants.adaptive_activation_enabled = true; % Enable adaptive activation functions
constants.activation_n_min = 0.1;            % Minimum scaling factor for activation
constants.activation_n_max = 10.0;           % Maximum scaling factor for activation
constants.activation_n_init = 1.0;           % Initial scaling factor for activation
constants.activation_learning_rate = 0.0001; % Learning rate for activation parameters



% Critical time thresholds
constants.early_time_threshold = 0.3;    % Early time threshold (t < 0.3T)
constants.late_time_threshold = 0.7;    % Late time threshold (t > 0.7T)
constants.mid_time_start = 0.3;         % Middle time start
constants.mid_time_end = 0.7;           % Middle time end

end
