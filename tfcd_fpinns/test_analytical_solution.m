%% Test Analytical Solution
% Test the analytical solution for the fractional convection-diffusion equation

clear; clc; close all;

fprintf('=== Analytical Solution Test ===\n');

%% Add paths
addpath('Analytical');
addpath('utils');

%% Test parameters
alpha = 0.5;  % Fractional order
D = 0.001;     % Diffusion coefficient
v = 1;         % Velocity
T = 1;         % Total time
L = 1;         % Domain length
N_terms = 35;  % Number of terms in series

fprintf('Test parameters:\n');
fprintf('  α = %.1f\n', alpha);
fprintf('  D = %.3f\n', D);
fprintf('  v = %.1f\n', v);
fprintf('  T = %.1f\n', T);
fprintf('  L = %.1f\n', L);

%% Test 1: Initial condition check
fprintf('\n=== Test 1: Initial Condition Check ===\n');

x_fine = linspace(0, 1, 100);
t_initial = 0;

u_initial = analytical_solution(x_fine, t_initial, alpha, D, v, N_terms);
ic_expected = x_fine .* (1 - x_fine);

% Check initial condition
ic_error = max(abs(u_initial - ic_expected));
fprintf('Initial condition max error: %.2e\n', ic_error);

if ic_error < 1e-10
    fprintf('✓ Initial condition correct\n');
else
    fprintf('✗ Initial condition incorrect\n');
end

% Visualize initial condition
figure('Position', [100, 100, 800, 600]);

subplot(2, 2, 1);
plot(x_fine, u_initial, 'b-', 'LineWidth', 2, 'DisplayName', 'Analytical');
hold on;
plot(x_fine, ic_expected, 'r--', 'LineWidth', 2, 'DisplayName', 'Expected');
xlabel('Position x');
ylabel('u(x,0)');
title('Initial Condition Check');
legend;
grid on;

%% Test 2: Boundary condition check
fprintf('\n=== Test 2: Boundary Condition Check ===\n');

t_fine = linspace(0, T, 50);

% Left boundary x = 0
u_left = analytical_solution(0, t_fine, alpha, D, v, N_terms);
bc_left_error = max(abs(u_left));

% Right boundary x = 1
u_right = analytical_solution(1, t_fine, alpha, D, v, N_terms);
bc_right_error = max(abs(u_right));

fprintf('Left boundary max error: %.2e\n', bc_left_error);
fprintf('Right boundary max error: %.2e\n', bc_right_error);

if bc_left_error < 1e-10 && bc_right_error < 1e-10
    fprintf('✓ Boundary conditions correct\n');
else
    fprintf('✗ Boundary conditions incorrect\n');
end

% Visualize boundary conditions
subplot(2, 2, 2);
plot(t_fine, u_left, 'b-', 'LineWidth', 2, 'DisplayName', 'Left BC u(0,t)');
hold on;
plot(t_fine, u_right, 'r-', 'LineWidth', 2, 'DisplayName', 'Right BC u(1,t)');
xlabel('Time t');
ylabel('Boundary value');
title('Boundary Condition Check');
legend;
grid on;

%% Test 3: Peak position check
fprintf('\n=== Test 3: Peak Position Check ===\n');

t_test = [0, 0.2, 0.4, 0.6, 0.8, 1.0];
x_fine = linspace(0, 1, 1000);

for i = 1:length(t_test)
    t_val = t_test(i);
    u_test = analytical_solution(x_fine, t_val, alpha, D, v, N_terms);
    
    % Find peak position
    [~, peak_idx] = max(u_test);
    peak_pos = x_fine(peak_idx);
    peak_amp = u_test(peak_idx);
    
    fprintf('t = %.1f: Peak position = %.3f, Peak amplitude = %.6f\n', t_val, peak_pos, peak_amp);
end

%% Test 4: Time evolution check
fprintf('\n=== Test 4: Time Evolution Check ===\n');

% Check time evolution of solution
x_center = 0.5;  % Center point
t_fine = linspace(0, T, 100);
u_center = analytical_solution(x_center, t_fine, alpha, D, v, N_terms);

% Check if solution is monotonically decreasing (due to Mittag-Leffler decay)
is_decreasing = all(diff(u_center) <= 0);
fprintf('Is center point solution monotonically decreasing: %s\n', is_decreasing ? 'Yes' : 'No');

% Visualize time evolution
subplot(2, 2, 3);
plot(t_fine, u_center, 'b-', 'LineWidth', 2);
xlabel('Time t');
ylabel('u(0.5,t)');
title('Center Point Time Evolution');
grid on;

%% Test 5: Spatial distribution check
fprintf('\n=== Test 5: Spatial Distribution Check ===\n');

% Check spatial distribution at different times
t_test = [0, 0.25, 0.5, 0.75, 1.0];
x_fine = linspace(0, 1, 100);

subplot(2, 2, 4);
colors = {'b', 'r', 'g', 'm', 'k'};
for i = 1:length(t_test)
    t_val = t_test(i);
    u_test = analytical_solution(x_fine, t_val, alpha, D, v, N_terms);
    
    plot(x_fine, u_test, colors{i}, 'LineWidth', 2, ...
         'DisplayName', sprintf('t = %.2f', t_val));
    hold on;
end
xlabel('Position x');
ylabel('u(x,t)');
title('Spatial Distribution at Different Times');
legend;
grid on;

sgtitle('Analytical Solution Test Results', 'FontSize', 16, 'FontWeight', 'bold');

%% Test 6: Numerical stability check
fprintf('\n=== Test 6: Numerical Stability Check ===\n');

% Check Mittag-Leffler function computation
t_test = [0.001, 0.01, 0.1, 0.5, 1.0];
for i = 1:length(t_test)
    t_val = t_test(i);
    E_alpha = mlf(alpha, 1, -t_val^alpha);
    fprintf('t = %.3f: E_α(-t^α) = %.6f\n', t_val, E_alpha);
end

%% Summary
fprintf('\n=== Test Summary ===\n');
fprintf('1. Initial condition error: %.2e\n', ic_error);
fprintf('2. Left boundary error: %.2e\n', bc_left_error);
fprintf('3. Right boundary error: %.2e\n', bc_right_error);
fprintf('4. Time evolution monotonicity: %s\n', is_decreasing ? 'Correct' : 'Incorrect');

if ic_error < 1e-10 && bc_left_error < 1e-10 && bc_right_error < 1e-10
    fprintf('\n✓ Analytical solution test passed!\n');
else
    fprintf('\n✗ Analytical solution test failed!\n');
end

fprintf('\n=== Test Complete ===\n');
