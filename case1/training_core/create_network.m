function net = create_network()
% CREATE_NETWORK - Create PINN neural network architecture with Fourier features
%
% Returns:
%   net - dlnetwork object for the PINN



% Get constants
constants = numerical_constants();

if constants.fourier_features_enabled
    % Create Fourier feature mapping matrix B
    input_dims = 2;  % [t, x]
    mapping_size = constants.fourier_mapping_size;
    sigma = constants.fourier_sigma;
    
    % Generate Gaussian matrix B for frequency mapping
    B = sigma * randn(mapping_size, input_dims);
    
    % Calculate output dimension after Fourier mapping
    if constants.fourier_include_original
        fourier_output_dim = 2 * mapping_size + input_dims;
    else
        fourier_output_dim = 2 * mapping_size;
    end
    
    % Create network with Fourier features
    layers = [
        featureInputLayer(fourier_output_dim)            % Input: [fourier_features; batch_size]
        fullyConnectedLayer(256)                         % First hidden layer
        tanhLayer
        fullyConnectedLayer(256)                         % Second hidden layer
        tanhLayer
        fullyConnectedLayer(128)                         % Third hidden layer
        tanhLayer
        fullyConnectedLayer(64)                          % Fourth hidden layer
        tanhLayer
        fullyConnectedLayer(1)                           % Output: u(x,t)
    ];
    
    net = dlnetwork(layers);
    
    % Store Fourier matrix B in global variable for later use
    global fourier_matrix_B;
    fourier_matrix_B = B;
    
else
    % Standard network without Fourier features
    layers = [
        featureInputLayer(2)      % Input: [t; x]
        fullyConnectedLayer(256)  % First hidden layer with 256 neurons
        tanhLayer
        fullyConnectedLayer(256)  % Second hidden layer with 256 neurons
        tanhLayer
        fullyConnectedLayer(128)  % Third hidden layer with 128 neurons
        tanhLayer
        fullyConnectedLayer(64)   % New layer for better expressiveness
        tanhLayer
        fullyConnectedLayer(1)    % Output: u(x,t)
    ];
    
    net = dlnetwork(layers);
end

end

