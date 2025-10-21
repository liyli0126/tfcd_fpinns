%% Time-Fractional Advection-Diffusion PINN Solver
% Main execution script for modularized tfCD PINN solver
% Based on Pariyar et al. (2025) air pollution model
% with Caputo fractional derivative (α=0.8)

% Clear workspace
%clear; clc; close all;

% Set random seed for reproducible results
set_random_seed(26);

%% Add paths to all modules (relative to main directory)
addpath('../params_data');
addpath('../training_core');
addpath('../utils');
addpath('../visualization');
addpath('../Analytical');

%% Initialize parameters
fprintf('=== Initializing Parameters ===\n');
params = initialize_parameters();

%% Create neural network
fprintf('=== Creating Neural Network ===\n');
[net, adaptive_params] = create_network();

% Quick activation mode switching:
% Use switch_activation_mode('adaptive') or switch_activation_mode('standard')
% Or check status with switch_activation_mode('status')

%% Train PINN
fprintf('=== Starting PINN Training ===\n');
[net, loss_history] = train_two_stage_pinn(net, params);

%% Plot results
fprintf('=== Generating Results and Plots ===\n');
plot_results(net, params, loss_history, adaptive_params);

fprintf('=== PINN Training and Analysis Complete ===\n');
