function plot_sampling_points(params)
% PLOT_SAMPLING_POINTS - Visualize sampling points before and after RAR
%
% This function produces two separate figures:
%   1) Initial sampling design (without any RAR points)
%   2) Initial sampling overlaid with all accumulated RAR points
%
% Inputs:
%   params - parameter struct that contains sampling arrays:
%            t_bc, x0, xL, t_ic, x_ic, t_hist, x_hist
%            optionally t_hist_rar_all, x_hist_rar_all

    % Safety checks for required fields
    requiredFields = {"t_bc","x0","xL","t_ic","x_ic","t_hist","x_hist","L","T"};
    for k = 1:numel(requiredFields)
        if ~isfield(params, requiredFields{k})
            error("plot_sampling_points:MissingField", "params.%s is required.", requiredFields{k});
        end
    end

    % Extract base data
    t_bc = params.t_bc;   % 1xN_bc time samples on boundary lines
    x0 = params.x0;       % 1xN_bc (zeros) for left boundary x=0
    xL = params.xL;       % 1xN_bc (L) for right boundary x=L
    t_ic = params.t_ic;   % 1xN_ic (zeros) for initial line t=0
    x_ic = params.x_ic;   % 1xN_ic positions along x for IC
    t_hist_all = params.t_hist; % historical/internal points (may include RAR already)
    x_hist_all = params.x_hist;
    L = params.L;
    T = params.T;

    % RAR accumulated points (may be absent if never used)
    if isfield(params, 't_hist_rar_all') && isfield(params, 'x_hist_rar_all')
        t_rar_all = params.t_hist_rar_all;
        x_rar_all = params.x_hist_rar_all;
    else
        t_rar_all = [];
        x_rar_all = [];
    end

    % Separate initial-only internal points by excluding RAR-accumulated points
    % Use tolerance for floating point pairing
    [t_hist_initial, x_hist_initial] = exclude_pairs_with_tolerance( ...
        t_hist_all, x_hist_all, t_rar_all, x_rar_all, 1e-10);

    % Figure 1: initial sampling only (no RAR)
    figure;
    hold on; grid on;
    % Boundary points: left (x=0) and right (x=L)
    scatter(x0, t_bc, 36, 'b', 's', 'filled', 'DisplayName', 'Boundary x=0');
    scatter(xL, t_bc, 36, 'b', 'd', 'filled', 'DisplayName', 'Boundary x=L');
    % Initial condition points (t=0)
    scatter(x_ic, t_ic, 36, 'g', '^', 'filled', 'DisplayName', 'Initial t=0');
    % Internal historical points (initial only)
    scatter(x_hist_initial, t_hist_initial, 10, [0.5 0.5 0.5], 'o', 'filled', 'DisplayName', 'Internal (initial)');
    xlabel('x'); ylabel('t');
    title('Initial Sampling (No RAR)');
    axis([0 L 0 T]);
    legend('Location','best');
    hold off;

    % Save figure 1
    try
        saveas(gcf, fullfile(fileparts(fileparts(mfilename('fullpath'))), 'main', 'Figure', 'Sampling_initial.png'));
    catch
        % Silent if save path not available
    end

    % Figure 2: overlay RAR points (if any)
    figure;
    hold on; grid on;
    scatter(x0, t_bc, 36, 'b', 's', 'filled', 'DisplayName', 'Boundary x=0');
    scatter(xL, t_bc, 36, 'b', 'd', 'filled', 'DisplayName', 'Boundary x=L');
    scatter(x_ic, t_ic, 36, 'g', '^', 'filled', 'DisplayName', 'Initial t=0');
    scatter(x_hist_initial, t_hist_initial, 10, [0.5 0.5 0.5], 'o', 'filled', 'DisplayName', 'Internal (initial)');
    if ~isempty(t_rar_all)
        scatter(x_rar_all, t_rar_all, 28, 'r', 'x', 'LineWidth', 1.25, 'DisplayName', 'RAR (accumulated)');
    end
    xlabel('x'); ylabel('t');
    title('Sampling with RAR (Overlay)');
    axis([0 L 0 T]);
    legend('Location','best');
    hold off;

    % Save figure 2
    try
        saveas(gcf, fullfile(fileparts(fileparts(mfilename('fullpath'))), 'main', 'Figure', 'Sampling_with_RAR.png'));
    catch
        % Silent if save path not available
    end
end

function [t_keep, x_keep] = exclude_pairs_with_tolerance(t_all, x_all, t_ex, x_ex, tol)
% Exclude pairs (t_all(i), x_all(i)) that match any (t_ex(j), x_ex(j)) within tolerance
    if isempty(t_ex)
        t_keep = t_all;
        x_keep = x_all;
        return;
    end
    mask_keep = true(size(t_all));
    for i = 1:numel(t_all)
        ti = t_all(i);
        xi = x_all(i);
        % Find any exclusion candidate close in both t and x
        is_close_t = abs(t_ex - ti) <= tol;
        if any(is_close_t)
            idx = find(is_close_t);
            if any(abs(x_ex(idx) - xi) <= tol)
                mask_keep(i) = false;
            end
        end
    end
    t_keep = t_all(mask_keep);
    x_keep = x_all(mask_keep);
end


