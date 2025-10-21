function plot_error_analysis(X_val, T_val, u_analytical, u_pinn)
% PLOT_ERROR_ANALYSIS - Plot error analysis and statistics
%
% Inputs:
%   X_val, T_val - spatial and temporal grids
%   u_analytical - analytical solution
%   u_pinn       - PINN prediction

% Error Analysis
error = abs(u_analytical - u_pinn);

disp('=== Error Analysis ===');
disp(['Number of points with error > 0.1: ', num2str(sum(error(:) > 0.1))]);

[max_err, max_idx] = max(error(:));
[max_i, max_j] = ind2sub(size(error), max_idx);
disp(['Maximum error location: t = ', num2str(T_val(max_i,max_j)), ...
    ', x = ', num2str(X_val(max_i,max_j))]);
disp(['Error value at this point: ', num2str(max_err)]);
disp(['Analytical solution at max error: ', num2str(u_analytical(max_i,max_j))]);
disp(['PINN prediction at max error: ', num2str(u_pinn(max_i,max_j))]);

% Error visualization
figure;
surf(X_val, T_val, error, 'EdgeColor', 'none');
title('Absolute Error');
ylabel('x (m)'); xlabel('t (h)'); zlabel('Error');
view(35,30); colormap jet; colorbar;

fprintf('Maximum Absolute Error: %.4e\n', max(error(:)));
fprintf('Mean Absolute Error: %.4e\n', mean(error(:)));

end

