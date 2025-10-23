function params = initialize_parameters()
% INITIALIZE_PARAMETERS - Initialize all problem and training parameters
% Returns a struct containing all parameters for the time-fractional 
% advection-diffusion PINN solver with generic interfaces


params = struct();

% Set problem type (single supported case)
params.problem_type = 'case2';

% Case2 parameters (only supported case)
params.alpha = 0.99;  
params.D = 1;
params.v = 1;  % Constant convection coefficient
params.variable_convection = false;

% ===== GENERIC INTERFACES =====
% These interfaces make the PINN solver generic and configurable

% 1. Source term interface (optional - for case2)
params.source_term = @(t,x,alpha) source_term_case2(t,x,alpha);

% 2. Initial condition interface (optional - for case2)
params.initial_condition = @(x) 4*x.^2.*(1-x).^2;  % u(x,0) = 4x^2(1-x)^2

% 3. Boundary conditions interface (optional - for case2)
params.boundary_conditions = struct();
params.boundary_conditions.left = @(t,x) zeros(size(t));   % u(0,t) = 0
params.boundary_conditions.right = @(t,x) zeros(size(t));  % u(1,t) = 0

% 4. Peak region configuration (configurable) - Modified for dynamic peak detection
params.peak_region = struct();
params.peak_region.center = 0.5;    % Peak center (default: x = 0.5) - Will be replaced by dynamic calculation
params.peak_region.width = 0.1;     % Peak width (default: ±0.1)
params.peak_region.weight = 4;      % Peak region weight in local weighting (reduced from 8 to 4)
params.peak_region.dynamic = true;  % Enable dynamic peak detection

% 5. Boundary regions configuration (configurable)
params.boundary_regions = struct();
params.boundary_regions.front = 5;              % Front region weight (x < threshold)
params.boundary_regions.rear = 5;               % Rear region weight (x > threshold)
params.boundary_regions.front_threshold = 0.2;  % Front boundary threshold
params.boundary_regions.rear_threshold = 0.8;   % Rear boundary threshold

% 6. Peak protection configuration (configurable) - Modified for dynamic peak protection
params.peak_protection = struct();
params.peak_protection.x = 0.5;           % Peak location for protection - Will be replaced by dynamic calculation
params.peak_protection.time_points = 20;  % Number of time points to sample
params.peak_protection.staged = true;     % Enable staged peak protection
params.peak_protection.strategy = 'gentle';  % Gentle strategy identifier
params.peak_protection.max_improvement = 0.20;  % Maximum improvement factor 20%
params.peak_protection.early_intervention = 0.05;  % Early intervention factor 5%
params.peak_protection.dynamic = true;    % Enable dynamic peak protection

% 7. Loss weights configuration (configurable) 
params.loss_weights = struct();
params.loss_weights.ic = 15;        % Initial condition weight
params.loss_weights.peak = 1.5;       % Peak protection weight (increased from 1 to 1.5, balanced learning)
params.loss_weights.sobolev = 1.5;  % Sobolev regularization weight
params.loss_weights.ic_physics = 1.0;  % Initial condition physics constraint weight (reduced from 5.0 to 1.0)

% 8. Analytical solution interface (for validation)
params.analytical_solution = @(x,t,alpha) analytical_solution_case2(x,t,alpha);
params.description = 'New case with source term: u(x,t) = (2x(1-x))^2 (t^alpha + sin(x))';

% ===== New: Peak detection and time singularity parameters =====
% Peak detection parameters - Only keep n_points parameter
params.peak_detection = struct();
params.peak_detection.n_points = 1000;  % Number of points for peak detection

% Time singularity parameters
params.singularity = struct();
params.singularity.early_threshold = 1e-3;
params.singularity.mid_threshold = 0.1;
params.singularity.late_threshold = 1.0;

% ===== COMMON PARAMETERS =====
% These parameters are common for all cases

% Domain parameters
params.L = 1;                          % Domain length (m)
params.T = 1;                          % Simulation time (h)

% Enhanced loss weights for peak accuracy
params.lambda = struct('pde', 2.0, 'bc', 1.0, 'ic', 15.0);

% Generate SOE approximation parameters with enhanced stability for α close to 1
[params.lambda_soe, params.theta] = generate_SOE(1 - params.alpha, 1e-10, 1e-8);

% Training parameters
params.numCollocation = 600;           % Increased from 500 to 600
params.N_hist = 500;                  % Reduced from 1200 to 500 for numerical stability
params.epochs = 1000;                  % Increased from 2000 to 2500
params.r = 1.2;                        % Reduced from 1.5 to 1.2 for more uniform early-time distribution
params.miniNr = 1000;                  % Increase PDE collocation points
params.miniNb = 80;                    % Reduce BC points (IC will be 2x)

% Two-stage training parameters
params.maxIterAdam = 800;
params.maxIterLBFGS =200;
params.lr = 0.001;                     % Slightly higher, slower decay
params.lr_decay_steps = 1000;          % Slower LR decay
params.beta1 = 0.9;                    % Adam beta1 parameter
params.beta2 = 0.999;                  % Adam beta2 parameter
params.eps = 1e-8;                     % Adam epsilon parameter
params.rar_freq = 150;                 % Reduced from 200 to 150 for more frequent RAR

% RAR sampling parameters
params.rar_candidates = 1500;          % Increased from 1200 to 1500 (3x miniNr)
params.rar_add_points = 60;            % Increased from 50 to 60
params.time_segments = 6;              % Increased from 5 to 6 for finer time stratification

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
