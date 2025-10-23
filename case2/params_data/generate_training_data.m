function [params, t_hist, x_hist, t_bc, x0, xL, t_ic, x_ic] = generate_training_data(params, net, iter)
% GENERATE_TRAINING_DATA - Generate training data points for PINN
%
% Inputs:
%   params - parameter struct
%   net    - neural network (optional, for RAR sampling)
%   iter   - current iteration (optional, for RAR sampling)
%
% Outputs:
%   params - updated parameter struct with sampling info
%   t_hist, x_hist - historical points
%   t_bc, x0, xL   - boundary condition points
%   t_ic, x_ic     - initial condition points

% Historical points with adaptive distribution for initial layer
% Use different r values for different time regions
N_hist = min(params.N_hist, 500);  % Limit number of historical points
r = params.r;

% Define time regions
t_initial = 0.1;  % Initial layer region
t_mid = 0.5;     % Mid-time region

% Generate points for different regions
N_initial = round(N_hist * 0.3);  % 30% points for initial layer
N_mid = round(N_hist * 0.4);     % 40% points for mid-time
N_late = N_hist - N_initial - N_mid;  % Remaining for late time

% Initial layer: dense sampling with r=1.2 (less aggressive)
t_hist1 = t_initial * ((0:N_initial-1)/N_initial).^1.2;

% Mid-time: moderate sampling with r=1.5
t_hist2 = t_initial + (t_mid - t_initial) * ((0:N_mid-1)/N_mid).^1.5;

% Late time: sparse sampling with r=1.0 (uniform)
t_hist3 = t_mid + (params.T - t_mid) * ((0:N_late-1)/N_late).^1.0;

% Combine all regions
t_hist1 = [t_hist1, t_hist2, t_hist3];

% Ensure minimum time step to avoid numerical issues
min_dt = 1e-6;
for i = 2:length(t_hist1)
    if t_hist1(i) - t_hist1(i-1) < min_dt
        t_hist1(i) = t_hist1(i-1) + min_dt;
    end
end

t_hist = t_hist1;
x_hist = params.L * rand(1, N_hist);
params.t_hist_powerlaw = t_hist1;
params.x_hist_powerlaw = x_hist;

% RAR sampling in t > 0.05T region (after initial epoch, every rar_freq steps)
if nargin > 1 && ~isempty(net) && nargin > 2 && iter > 1 && mod(iter, params.rar_freq) == 0
    rar_num = 200; % number of candidate points
    rar_add = 50;  % number of points to add
    t_rar_cand = params.T * (0.05 + 0.95*rand(1, rar_num));
    x_rar_cand = params.L * rand(1, rar_num);
    t_rar_dl = dlarray(t_rar_cand, "CB");
    x_rar_dl = dlarray(x_rar_cand, "CB");
    
    [res_val, ~] = dlfeval(@rar_residual, net, t_rar_dl, x_rar_dl, t_hist, x_hist, ...
                           params.v, params.D, params.alpha, params.T, params.lambda_soe, params.theta, params);
    res_val(isnan(res_val) | isinf(res_val)) = -Inf;
    [~, idx] = maxk(res_val, min(rar_add, numel(t_rar_cand)));
    valid_idx = idx(idx <= numel(t_rar_cand));
    t_hist = [t_hist, t_rar_cand(valid_idx)];
    x_hist = [x_hist, x_rar_cand(valid_idx)];
    params.t_hist_rar = t_rar_cand(valid_idx);
    params.x_hist_rar = x_rar_cand(valid_idx);
    
    % Accumulate all RAR points
    if ~isfield(params, 't_hist_rar_all')
        params.t_hist_rar_all = [];
        params.x_hist_rar_all = [];
    end
    params.t_hist_rar_all = [params.t_hist_rar_all, t_rar_cand(valid_idx)];
    params.x_hist_rar_all = [params.x_hist_rar_all, x_rar_cand(valid_idx)];
else
    params.t_hist_rar = [];
    params.x_hist_rar = [];
end

% Sort historical points by time
[t_hist, idx] = sort(t_hist);
x_hist = x_hist(idx);

% Save historical points to params for later access
params.t_hist = t_hist;
params.x_hist = x_hist;

% Boundary condition points (equispaced in time)
N_bc = params.miniNb;
t_bc = linspace(0, params.T, N_bc);
x0 = zeros(1, N_bc);                % x=0 boundary
xL = params.L * ones(1, N_bc);      % x=L boundary

% Initial condition points with enhanced peak region sampling
N_ic = 2 * params.miniNb;

% Calculate peak location at t=0 for enhanced initial condition sampling
if isfield(params, 'peak_region') && isfield(params.peak_region, 'dynamic') && params.peak_region.dynamic
    [peak_x_ic, ~] = find_peak_location_singular(0, params.alpha, params.peak_detection.n_points);
else
    peak_x_ic = params.peak_region.center;
end

% Enhanced initial condition sampling: 70% Chebyshev + 30% peak region
n_chebyshev = round(N_ic * 0.7);
n_peak_ic = N_ic - n_chebyshev;

% Chebyshev nodes for general coverage
x_ic_chebyshev = 0.5 * (1 - cos(pi * (0:n_chebyshev-1) / (n_chebyshev-1))) * params.L;

% Peak region sampling for enhanced accuracy
if n_peak_ic > 0
    peak_region_width = 0.1;  % ±0.05 around peak
    x_ic_peak = peak_x_ic + peak_region_width * (rand(1, n_peak_ic) - 0.5);
    x_ic_peak = max(0, min(params.L, x_ic_peak));  % Ensure in [0, L] range
else
    x_ic_peak = [];
end

% Combine and sort initial condition points
x_ic = [x_ic_chebyshev, x_ic_peak];
x_ic = sort(x_ic);  % Sort for better numerical stability
t_ic = zeros(1, N_ic);

end

