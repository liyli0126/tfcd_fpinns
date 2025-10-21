function params = initialize_parameters()
% INITIALIZE_PARAMETERS - Initialize all problem and training parameters
% Returns a struct containing all parameters for the time-fractional 
% advection-diffusion PINN solver

params = struct();

% Problem parameters
params.alpha = 0.8;                    % Fractional order
params.D = 1.296e-2;                   % Diffusion coefficient (m^2/h) - FIXED
params.v = 0.1296;                     % Velocity (m/h)
params.L = 1;                          % Domain length (m)
params.T = 1;                          % Simulation time (h)

% Enhanced loss weights for peak accuracy
params.lambda = struct('pde', 2.5, 'bc', 8.0, 'ic', 8.0); % Increased IC weight

% Generate SOE approximation parameters
[params.lambda_soe, params.theta] = generate_SOE(1 - params.alpha, 1e-8, 1e-6);

% Training parameters
params.numCollocation = 600;           % Number of collocation points for PINN training
params.N_hist = 1000;                  % Number of historical training points
params.epochs = 1500;                   % Total training epochs
params.r = 2.0;                        % Power law exponent for time sampling
params.miniNr = 500;                   % Number of residual points per mini-batch
params.miniNb = 120;                   % Number of boundary points per mini-batch
params.N_terms = 35;                   % Number of terms in analytical solution series

% Two-stage training parameters
params.maxIterAdam = 1000;             % Maximum iterations for Adam optimizer
params.maxIterLBFGS = 500;             % Maximum iterations for L-BFGS optimizer
params.lr = 0.0008;                    % Learning rate for Adam optimizer
params.lr_decay_steps = 1000;          % Learning rate decay steps
params.beta1 = 0.9;                    % Adam beta1 parameter
params.beta2 = 0.999;                  % Adam beta2 parameter
params.eps = 1e-8;                     % Adam epsilon parameter
params.rar_freq = 150;                 % RAR sampling frequency (every N epochs)

% RAR sampling parameters
params.rar_candidates = 1500;          % Number of candidate points for RAR selection
params.rar_add_points = 60;            % Number of points to add in each RAR iteration
params.time_segments = 6;              % Number of time segments for stratified sampling

% Boundary sampling optimization parameters
params.boundary_sampling_ratio = 0.03; % Reduced boundary sampling ratio (3%)
params.boundary_early_weight = 0.6;    % Weight for early time boundary sampling
params.boundary_late_weight = 0.2;     % Weight for late time boundary sampling
params.boundary_mid_weight = 0.2;      % Weight for middle time boundary sampling
params.interior_sampling_ratio = 0.15; % Interior region sampling ratio (15%)
params.boundary_weight_reduction = 0.8; % Weight reduction for boundary-near points in RAR selection

% Peak region optimization parameters
params.peak_sampling_ratio = 0.10;     % Peak region sampling ratio (10%)
params.peak_early_weight = 0.4;        % Early time weight for peak region
params.peak_mid_weight = 0.3;          % Middle time weight for peak region
params.peak_late_weight = 0.3;         % Late time weight for peak region

end
