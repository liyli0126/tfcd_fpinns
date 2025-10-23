function plot_2d_snapshots(x_val, t_val, u_analytical, u_pinn)
% PLOT_2D_SNAPSHOTS - Plot 2D snapshots at specific times
%
% Inputs:
%   x_val, t_val - spatial and temporal grids
%   u_analytical - analytical solution
%   u_pinn       - PINN prediction

figure;
t_plot = [0.1, 0.5, 0.9];

for i = 1:length(t_plot)
    t_idx = find(abs(t_val - t_plot(i)) == min(abs(t_val - t_plot(i))), 1);
    subplot(1,3,i);
    plot(x_val, u_analytical(:,t_idx), 'b-', 'LineWidth', 2, 'DisplayName', 'Analytical');
    hold on;
    plot(x_val, u_pinn(:,t_idx), 'r--', 'LineWidth', 2, 'DisplayName', 'PINN');
    title(['t = ', num2str(t_plot(i))]);
    xlabel('x'); ylabel('u(x,t)');
    legend('Location', 'best');
    grid on;
    ylim([0 1.2]);
end

end

