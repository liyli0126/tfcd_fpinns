function [x_hist, t_hist] = generate_balanced_points(params)
% GENERATE_BALANCED_POINTS - Generate spatially and temporally balanced sampling points
%
% This function implements a sampling strategy that:
% - Emphasizes early time points (for fractional derivative memory effects)
% - Emphasizes downstream spatial regions in late time (for peak migration)
%
% Inputs:
%   params - parameter struct with N_hist, T, L, r, etc.
%
% Outputs:
%   x_hist, t_hist - balanced historical points

    N_hist = params.N_hist;
    T = params.T;
    L = params.L;
    r = params.r;
    t_min_spacing = 1e-4;  % Minimum time spacing for numerical stability
    
    % Generate time distribution (early emphasis)
    t_hist = generate_early_emphasized_time(params);
    
    % Generate spatial distribution (late-time downstream emphasis)
    x_hist = generate_late_downstream_space(params, t_hist);
    
    % Ensure minimum time spacing
    t_hist = enforce_minimum_spacing(t_hist, t_min_spacing);
    
    % Sort by time for fractional derivative computation
    [t_hist, idx] = sort(t_hist);
    x_hist = x_hist(idx);
    
    % Ensure we don't exceed T
    t_hist = min(t_hist, T);
end

function t_hist = generate_early_emphasized_time(params)
% Generate time points with early time emphasis
    N_hist = params.N_hist;
    T = params.T;
    r = params.r;
    
    % Power-law distribution with early emphasis
    t_hist = zeros(1, N_hist);
    t_hist(1) = 0;  % Always start at t=0
    
    for i = 2:N_hist
        % Power-law distribution: t_i = T * ((i-1)/N_hist)^r
        % Smaller r gives more emphasis to early times
        t_power = T * ((i-1)/N_hist).^r;
        t_hist(i) = t_power;
    end
    
    % Add some early time clustering for better memory effect capture
    early_ratio = 0.3;  % 30% of points in early time
    n_early = round(N_hist * early_ratio);
    
    % Redistribute early points with higher density
    for i = 2:n_early
        t_hist(i) = T * 0.1 * ((i-1)/n_early).^0.5;  % Denser early sampling
    end
end

function x_hist = generate_late_downstream_space(params, t_hist)
% Generate spatial points with moderate late-time downstream emphasis
% Peak at x=0.5 is fixed, so we need balanced coverage around it
    N_hist = params.N_hist;
    L = params.L;
    T = params.T;
    
    x_hist = zeros(1, N_hist);
    
    for i = 1:N_hist
        t_frac = t_hist(i) / T;
        
        if t_frac < 0.3
            % Early time: uniform spatial distribution with peak emphasis
            x_hist(i) = L * rand()^0.9;
        elseif t_frac < 0.7
            % Mid time: balanced with slight peak region emphasis
            x_hist(i) = L * (0.1 + 0.8 * rand()^0.85);
        else
            % Late time: moderate downstream emphasis but keep peak coverage
            % Ensure sufficient sampling around x=0.5 peak region
            x_hist(i) = L * (0.2 + 0.7 * rand()^0.75);
        end
    end
end

function t_hist = enforce_minimum_spacing(t_hist, t_min_spacing)
% Ensure minimum spacing between consecutive time points
    for i = 2:length(t_hist)
        if t_hist(i) - t_hist(i-1) < t_min_spacing
            t_hist(i) = t_hist(i-1) + t_min_spacing;
        end
    end
end
