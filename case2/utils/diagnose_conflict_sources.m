function diagnose_conflict_sources(net, params, alpha, T, lambda_soe, theta)
% DIAGNOSE_CONFLICT_SOURCES - Diagnose conflict sources between PDE and peak constraints
%
% Inputs:
%   net - neural network
%   params - parameter structure
%   alpha - fractional derivative order
%   T - total time
%   lambda_soe, theta - SOE parameters

if nargin < 6
    error('Need 6 input parameters: net, params, alpha, T, lambda_soe, theta');
end

fprintf('=== Conflict Source Diagnosis ===\n');

try
    % 1. Check PDE residual at peak region
    t_test = 0.5;
    [peak_x, ~] = find_peak_location_singular(t_test, alpha, params.peak_detection.n_points);
    
    fprintf('\n1. Peak Region PDE Residual Analysis:\n');
    residual_peak = compute_pde_residual_at_point(net, t_test, peak_x, params, alpha, T, lambda_soe, theta);
    fprintf('   Peak position (t=%.1f, x=%.4f) PDE residual: %.6e\n', t_test, peak_x, residual_peak);
    
    % 2. Check PDE residual at non-peak region
    non_peak_x = 0.2;  % Non-peak position
    fprintf('\n2. Non-Peak Region PDE Residual Analysis:\n');
    residual_non_peak = compute_pde_residual_at_point(net, t_test, non_peak_x, params, alpha, T, lambda_soe, theta);
    fprintf('   Non-peak position (t=%.1f, x=%.4f) PDE residual: %.6e\n', t_test, non_peak_x, residual_non_peak);
    
    % 3. Compare residual magnitudes
    fprintf('\n3. Residual Comparison:\n');
    if abs(residual_peak) > abs(residual_non_peak)
        fprintf('   Peak region PDE residual is larger, potential peak constraint conflict\n');
        fprintf('   Residual ratio: %.2f\n', abs(residual_peak) / abs(residual_non_peak));
    else
        fprintf('   Non-peak region PDE residual is larger, peak constraint may be effective\n');
        fprintf('   Residual ratio: %.2f\n', abs(residual_non_peak) / abs(residual_peak));
    end
    
    % 4. Check network outputs
    fprintf('\n4. Network Output Check:\n');
    tx_peak = dlarray([t_test; peak_x], "CB");
    u_peak = predict(net, tx_peak);
    u_peak_val = extractdata(u_peak);
    
    tx_non_peak = dlarray([t_test; non_peak_x], "CB");
    u_non_peak = predict(net, tx_non_peak);
    u_non_peak_val = extractdata(u_non_peak);
    
    fprintf('   Peak position output: %.6e\n', u_peak_val);
    fprintf('   Non-peak position output: %.6e\n', u_non_peak_val);
    fprintf('   Output ratio: %.2f\n', u_peak_val / u_non_peak_val);
    
    % 5. Numerical stability check
    fprintf('\n5. Numerical Stability Check:\n');
    if isnan(residual_peak) || isinf(residual_peak)
        fprintf('   Warning: Peak region PDE residual contains NaN or Inf\n');
    else
        fprintf('   Peak region PDE residual is numerically stable\n');
    end
    
    if isnan(residual_non_peak) || isinf(residual_non_peak)
        fprintf('   Warning: Non-peak region PDE residual contains NaN or Inf\n');
    else
        fprintf('   Non-peak region PDE residual is numerically stable\n');
    end
    
    % 6. Conflict assessment
    fprintf('\n6. Conflict Assessment:\n');
    if abs(residual_peak) > 1e3 * abs(residual_non_peak)
        fprintf('   Severe conflict: Peak region PDE residual much larger than non-peak region\n');
    elseif abs(residual_peak) > 10 * abs(residual_non_peak)
        fprintf('   Moderate conflict: Peak region PDE residual significantly larger than non-peak region\n');
    elseif abs(residual_peak) > abs(residual_non_peak)
        fprintf('   Minor conflict: Peak region PDE residual slightly larger than non-peak region\n');
    else
        fprintf('   No conflict: Peak region PDE residual less than or equal to non-peak region\n');
    end
    
catch ME
    fprintf('Error occurred during diagnosis: %s\n', ME.message);
    fprintf('Skip diagnosis and continue training\n');
end

fprintf('=== Diagnosis Complete ===\n\n');

end

function residual = compute_pde_residual_at_point(net, t_val, x_val, params, alpha, T, lambda_soe, theta)
% Compute PDE residual at specified point (simplified version, avoid gradient computation)

% Create historical points for fractional derivative calculation
t_hist = linspace(0, t_val, 100);
x_hist = x_val * ones(size(t_hist));
tx_hist = dlarray([t_hist; x_hist], "CB");
u_hist = predict(net, tx_hist);
u_hist = extractdata(u_hist);

% Current point
tx_curr = dlarray([t_val; x_val], "CB");
u_curr = predict(net, tx_curr);
u_curr_val = extractdata(u_curr);

% Calculate fractional derivative
D_alpha_u = MFL1_Caputo(u_hist, t_hist, u_curr_val, t_val, alpha, T, lambda_soe, theta);

% Simplified: use finite difference approximation for spatial derivatives to avoid dlgradient issues
dx = 1e-6;
x_plus = x_val + dx;
x_minus = x_val - dx;

tx_plus = dlarray([t_val; x_plus], "CB");
tx_minus = dlarray([t_val; x_minus], "CB");

u_plus = predict(net, tx_plus);
u_minus = predict(net, tx_minus);

u_plus_val = extractdata(u_plus);
u_minus_val = extractdata(u_minus);

% Finite difference calculation of first and second derivatives
u_x = (u_plus_val - u_minus_val) / (2 * dx);
u_xx = (u_plus_val - 2 * u_curr_val + u_minus_val) / (dx^2);

% Calculate source term
if isfield(params, 'source_term') && ~isempty(params.source_term)
    f_val = params.source_term(t_val, x_val, alpha);
else
    f_val = 0;
end

% PDE residual
residual = D_alpha_u + 0.1 * u_x - 0.01 * u_xx - f_val;

end
