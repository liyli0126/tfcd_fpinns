function params = initialize_parameters()
% INITIALIZE_PARAMETERS - Initialize all problem and training parameters
% Returns a struct containing all parameters for the time-fractional 
% advection-diffusion PINN solver

params = struct();

% Problem parameters
params.alpha = 0.8;                    % Fractional order
params.D = 0.001;                      % Diffusion coefficient 
params.v = 1;                          % Velocity 
params.L = 1;                          % Domain length 
params.T = 1;                          % Simulation time 

% Optimized loss weights - enhanced PDE learning for time evolution
params.lambda = struct('pde', 0.9, 'bc', 1.9, 'ic', 1.9); % Increased PDE weight for stronger time evolution

% Generate SOE approximation parameters
[params.lambda_soe, params.theta] = generate_SOE(1 - params.alpha, 1e-8, 1e-6);

% Training parameters - optimized for peak learning
params.numCollocation = 800;           % Increased collocation points for better peak resolution
params.N_hist = 2500;                  % Increased historical points for better temporal dynamics
params.epochs = 1800;                  % Increased epochs for better convergence
params.r = 1.8;                        % Reduced power law for more uniform time sampling
params.miniNr = 600;                   % Increased residual points per mini-batch
params.miniNb = 300;                   % Increased boundary points per mini-batch for better BC enforcement
params.N_terms = 35;                   % Number of terms in analytical solution series

% Two-stage training parameters - optimized for peak learning
params.maxIterAdam = 1000;             % Increased Adam iterations for better peak learning
params.maxIterLBFGS = 200;             % Increased L-BFGS iterations for fine-tuning
params.lr = 0.002;                    % Increased learning rate for better convergence
params.lr_decay_steps = 1200;          % Learning rate decay steps
params.beta1 = 0.9;                    % Adam beta1 parameter
params.beta2 = 0.999;                  % Adam beta2 parameter
params.eps = 1e-8;                     % Adam epsilon parameter
params.rar_freq = 80;                  % More frequent RAR sampling for better peak accuracy

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
