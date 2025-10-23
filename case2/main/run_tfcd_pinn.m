%% Time-Fractional Convection-Diffusion PINN Solver
% Main execution script for modularized tfCD PINN solver
% Based on Mu. et al. (2025) singularity
% with Caputo fractional derivative (α=0.8)

% Clear workspace
clear; clc; close all;

%% Add paths to all modules (relative to main directory)
addpath('../params_data');
addpath('../training_core');
addpath('../utils');
addpath('../visualization');

%% Initialize parameters
fprintf('=== Initializing Parameters ===\n');
params = initialize_parameters();

%% Create neural network
fprintf('=== Creating Neural Network ===\n');
net = create_network();

%% Train PINN
fprintf('=== Starting PINN Training ===\n');
[net, loss_history] = train_two_stage_pinn(net, params);

%% Plot results
fprintf('=== Generating Results and Plots ===\n');
plot_results(net, params, loss_history);

fprintf('=== PINN Training and Analysis Complete ===\n');