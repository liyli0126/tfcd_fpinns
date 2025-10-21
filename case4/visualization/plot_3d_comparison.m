function plot_3d_comparison(X_val, T_val, u_analytical, u_pinn)
% PLOT_3D_COMPARISON - Plot 3D comparison between analytical and PINN solutions
%
% Inputs:
%   X_val, T_val - spatial and temporal grids
%   u_analytical - analytical solution
%   u_pinn       - PINN prediction

figure;
subplot(1,2,1);
surf(X_val, T_val, u_analytical, 'EdgeColor', 'none');
title('Analytical Solution');
xlabel('x (m)'); ylabel('t (h)'); zlabel('u(x,t)');
view(30,35); colormap jet; colorbar;
zlim([0 0.3]);

subplot(1,2,2);
surf(X_val, T_val, u_pinn, 'EdgeColor', 'none');
title('PINN Prediction');
xlabel('x (m)'); ylabel('t (h)'); zlabel('u(x,t)');
view(30,35); colormap jet; colorbar;
zlim([0 0.3]);

end

