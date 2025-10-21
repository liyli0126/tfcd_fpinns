function [net, adaptive_params] = create_network()
% CREATE_NETWORK - Create PINN neural network architecture with dual activation pathways
%
% This function creates a network that supports both:
% 1. Standard activation functions (tanh, relu, sigmoid)
% 2. Adaptive activation functions (experimental)
%
% Returns:
%   net             - dlnetwork object for the PINN
%   adaptive_params - struct containing adaptive activation parameters



% Get constants
constants = numerical_constants();

% Create standard network using configurable architecture
layers = [featureInputLayer(constants.network.input_features)]; % Input layer

% Add hidden layers dynamically based on configuration
for i = 1:length(constants.network.hidden_layers)
    layers = [layers, fullyConnectedLayer(constants.network.hidden_layers(i))];
    
    % Add activation layer based on configuration
    switch constants.network.activation
        case 'tanh'
            layers = [layers, tanhLayer];
        case 'relu'
            layers = [layers, reluLayer];
        case 'sigmoid'
            layers = [layers, sigmoidLayer];
        otherwise
            layers = [layers, tanhLayer]; % Default to tanh
    end
end

% Add output layer
layers = [layers, fullyConnectedLayer(constants.network.output_features)];

net = dlnetwork(layers);

% Initialize adaptive activation parameters (always available)
adaptive_params = struct();
adaptive_params.enabled = constants.adaptive_activation_enabled && constants.network.use_adaptive_activation;

% Initialize adaptive parameters for each hidden layer (regardless of enabled status)
for i = 1:length(constants.network.hidden_layers)
    param_name = ['layer_' num2str(i)];
    adaptive_params.(param_name) = dlarray(constants.activation_n_init, "CB");
end

% Display activation pathway information
if adaptive_params.enabled
    fprintf('Network created with DUAL activation pathways:\n');
    fprintf('  ✓ Standard activation: %s\n', constants.network.activation);
    fprintf('  ✓ Adaptive activation: ENABLED (%d adaptive layers)\n', length(constants.network.hidden_layers));
    fprintf('  → Current mode: ADAPTIVE\n');
else
    fprintf('Network created with DUAL activation pathways:\n');
    fprintf('  ✓ Standard activation: %s\n', constants.network.activation);
    fprintf('  ✓ Adaptive activation: AVAILABLE (but disabled)\n');
    fprintf('  → Current mode: STANDARD\n');
end
fprintf('  Note: Switch modes by changing constants.network.use_adaptive_activation\n');

end

