function plot_sampling_points(params)
% PLOT_SAMPLING_POINTS - Plot sampling points by type
%
% Inputs:
%   params - parameter struct containing sampling information

figure;
hold on;
legend_entries = {};

if isfield(params, 't_hist_powerlaw') && ~isempty(params.t_hist_powerlaw)
    scatter(params.x_hist_powerlaw, params.t_hist_powerlaw, 10, 'b', 'filled');
    legend_entries{end+1} = 'Power-law';
end

if isfield(params, 't_hist_rar_all') && ~isempty(params.t_hist_rar_all)
    scatter(params.x_hist_rar_all, params.t_hist_rar_all, 20, 'r', 'filled');
    legend_entries{end+1} = 'RAR';
end

if ~isempty(legend_entries)
    legend(legend_entries);
end

title('Sampling Points by Type');
xlabel('x');
ylabel('t');
grid on;
hold off;

end

