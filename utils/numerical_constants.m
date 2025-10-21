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

% Peak region parameters - enhanced for precise peak position learning
constants.peak_center = 0.5;             % Center of peak region (x = 0.5)
constants.peak_width = 0.15;            % Narrowed peak region (±0.075 around center)
constants.near_peak_width = 0.25;       % Reduced near-peak region (±0.125 around center)
constants.peak_precision_region = 0.05; % Very precise region around x=0.5 (±0.025)

% Peak position loss parameters - enhanced for final time accuracy
constants.peak_position_check_times = [0.1, 0.3, 0.5, 0.7, 0.85, 0.95]; % More emphasis on late times
constants.peak_search_half_width = 0.05;   % Half-width for peak search around center (±0.05)
constants.peak_search_resolution = 21;     % Number of points in peak search grid
constants.peak_position_time_weight_factor = 2.0; % Factor for time-dependent weighting

% Initial condition weighting parameters
constants.ic_peak_precise_threshold = 0.025;  % Very precise peak region threshold
constants.ic_peak_region_threshold = 0.075;   % Peak region threshold
constants.ic_near_peak_threshold = 0.15;      % Near-peak region threshold
constants.ic_weight_precise_peak = 2.5;       % Weight for very precise peak region
constants.ic_weight_peak_region = 2.0;        % Weight for peak region
constants.ic_weight_near_peak = 1.5;          % Weight for near-peak region
constants.ic_weight_base = 1.0;               % Base weight for other regions
constants.ic_sin_enhancement_factor = 0.3;    % Enhancement factor for sin-based weighting
constants.ic_symmetry_enhancement = 0.1;      % Enhancement for symmetry around peak
constants.ic_symmetry_sharpness = 20;         % Sharpness of Gaussian symmetry weighting

% Peak position loss weighting schedule - enhanced for final accuracy
constants.peak_pos_weight_early = 0.5;        % Early stage weight (epochs < 200)
constants.peak_pos_weight_intro_start = 0.5;  % Introduction stage start weight (epochs 200-400)
constants.peak_pos_weight_intro_end = 1.2;    % Increased introduction stage end weight
constants.peak_pos_weight_transition = 1.2;   % Increased transition stage weight (epochs 400-800)
constants.peak_pos_weight_final = 1.0;        % Increased final stage weight (epochs > 800)

% Additional parameters for final time enhancement
constants.final_time_emphasis_threshold = 0.8;   % Threshold for final time emphasis (t > 0.8T)
constants.final_time_weight_factor = 1.5;        % Additional weight factor for final time points

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

% Peak detection parameters
constants.peak_movement_scale_factor = 0.05;      % Scale factor for peak movement calculation
constants.peak_search_boundary_min = 0.1;         % Minimum boundary for peak search (relative to L)
constants.peak_search_boundary_max = 0.9;         % Maximum boundary for peak search (relative to L)
constants.peak_search_width = 0.3;                % Width of peak search region
constants.peak_search_resolution = 150;           % Resolution for peak detection grid

% Boundary detection parameters
constants.boundary_detection_threshold = 0.1;     % Threshold for boundary region detection (relative to L)
constants.boundary_adaptive_enabled = true;       % Enable adaptive boundary detection based on analytical solution

% Weight coefficients for different regions and times
constants.weights = struct();
constants.weights.boundary_early = 6.0;           % Early time boundary weight
constants.weights.boundary_mid = 3.0;             % Middle time boundary weight  
constants.weights.boundary_late = 2.0;            % Late time boundary weight
constants.weights.peak_region_high = 3.0;         % High weight for peak region
constants.weights.peak_region_medium = 2.0;       % Medium weight for near-peak region
constants.weights.interior_base = 1.0;            % Base weight for interior region

% Network architecture parameters
constants.network = struct();
constants.network.input_features = 2;             % Input features [t, x]
constants.network.hidden_layers = [128, 128, 64, 32]; % Reduced network size for better training
constants.network.output_features = 1;            % Output features
constants.network.activation = 'tanh';            % Standard activation: 'tanh', 'relu', 'sigmoid'
constants.network.use_adaptive_activation = false; % Enable adaptive activation (experimental)

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

% Sobolev regularization parameters - COMMENTED OUT but preserved
% Temporarily disabled due to implementation issues causing worse results
% To re-enable: set sobolev_enabled = true and uncomment Sobolev terms in loss function
constants.sobolev_enabled = false;       % Disable Sobolev regularization temporarily
constants.sobolev_lambda_s0 = 0.1;       % Initial Sobolev weight
constants.sobolev_beta = 0.001;          % Exponential decay parameter for Sobolev weight
constants.sobolev_epoch_start = 200;     % Epoch to start Sobolev regularization
constants.sobolev_epoch_end = 1000;      % Epoch to end Sobolev regularization

end
