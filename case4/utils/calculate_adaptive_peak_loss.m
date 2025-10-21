function L_peak = calculate_adaptive_peak_loss(net, params, T)
% CALCULATE_ADAPTIVE_PEAK_LOSS - Calculate adaptive peak protection loss with path guidance
%
% Inputs:
%   net    - trained neural network
%   params - parameter struct
%   T      - total simulation time
%
% Outputs:
%   L_peak - peak protection loss with path guidance

% Get constants
constants = numerical_constants();

if constants.adaptive_peak_enabled
    % Combined constraint: peak protection + path guidance
    % Sample multiple time points for peak tracking
    t_peak_samples = linspace(0, T, 20);
    peak_losses = zeros(size(t_peak_samples));
    path_guidance_losses = zeros(size(t_peak_samples));
    
    % Initial peak position from constants
    x_peak_initial = constants.initial_peak_position;
    
    for i = 1:length(t_peak_samples)
        t = t_peak_samples(i);
        
        % 1. Peak protection: detect current peak and protect it
        [peak_x, peak_analytical] = detect_peak_position(t, params);
        peak_input = dlarray([t; peak_x], "CB");
        u_peak = predict(net, peak_input);
        
        % Peak should be close to analytical solution value
        L_peak_single = (u_peak - peak_analytical)^2;
        peak_losses(i) = extractdata(L_peak_single);
        
        % 2. Path guidance: peak position should remain at x=0.5
        % Analytical solution u(x,t) = x(1-x)*E_α(-t^α) has fixed peak at x=0.5
        x_expected = 0.5 * params.L;
        
        % Calculate network prediction at expected peak position
        expected_input = dlarray([t; x_expected], "CB");
        u_expected = predict(net, expected_input);
        
        % Path guidance constraint: peak should be near expected position
        % Use threshold-based constraint: u should be above threshold at expected position
        threshold = constants.path_guidance_threshold;  % Minimum expected peak value
        path_guidance_losses(i) = extractdata(relu(threshold - u_expected));
    end
    
    % Combine losses: peak protection + path guidance
    L_peak_protection = mean(peak_losses);
    L_path_guidance = mean(path_guidance_losses);
    
    % Weighted combination
    L_peak = constants.peak_protection_weight * L_peak_protection + ...
             constants.path_guidance_weight * L_path_guidance;
    L_peak = dlarray(L_peak, "CB");
    
else
    % Fallback to fixed peak loss
    L_peak = calculate_fixed_peak_loss(net, params, T);
end

end

function L_peak = calculate_fixed_peak_loss(net, params, T)
% Fixed peak loss (original implementation)

% Fixed peak position from constants
peak_x = dlarray(constants.initial_peak_position, "CB");
peak_t = dlarray(linspace(0, T, 20), "CB");
peak_input = dlarray([peak_t; repmat(peak_x, 1, 20)], "CB");
u_peak = predict(net, peak_input);

% Peak should be close to analytical solution value at x = 0.5
try
    peak_target = analytical_solution(0.5, extractdata(peak_t), params.alpha, params.D, params.v, params.N_terms);
    peak_target = dlarray(peak_target, "CB");
catch
    % Fallback to theoretical maximum of sin(πx) = 1.0
    peak_target = ones(size(u_peak));
end
L_peak = mean((u_peak - peak_target).^2, 'all');

end
