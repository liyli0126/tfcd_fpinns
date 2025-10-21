function output = adaptive_forward(net, input_data, adaptive_params)
% ADAPTIVE_FORWARD - Forward pass with adaptive activation functions
%
% This function provides an alternative forward pass that applies adaptive
% activation functions to selected layers of the network.
%
% Inputs:
%   net            - standard dlnetwork
%   input_data     - input data as dlarray
%   adaptive_params - struct containing adaptive parameters for each layer
%
% Outputs:
%   output         - network output with adaptive activations applied

% Get constants
constants = numerical_constants();

if ~constants.adaptive_activation_enabled || ~constants.network.use_adaptive_activation
    % Use standard forward pass
    output = predict(net, input_data);
    return;
end

% Manual forward pass with adaptive activations
current_input = input_data;
layer_count = 0;

% Get network layers
layers = net.Layers;

for i = 1:numel(layers)
    layer = layers(i);
    
    if isa(layer, 'nnet.cnn.layer.FullyConnectedLayer')
        % Apply fully connected transformation
        weights = net.Learnables.Value{find(strcmp(net.Learnables.Layer, layer.Name) & strcmp(net.Learnables.Parameter, 'Weights'))};
        bias = net.Learnables.Value{find(strcmp(net.Learnables.Layer, layer.Name) & strcmp(net.Learnables.Parameter, 'Bias'))};
        
        % Linear transformation: W*x + b
        current_input = weights * current_input + bias;
        layer_count = layer_count + 1;
        
    elseif isa(layer, 'nnet.cnn.layer.TanhLayer') || isa(layer, 'nnet.cnn.layer.ReLULayer') || isa(layer, 'nnet.cnn.layer.SigmoidLayer')
        % Apply adaptive activation instead of standard activation
        if isfield(adaptive_params, ['layer_' num2str(layer_count)]) && constants.network.use_adaptive_activation
            % Get adaptive parameter for this layer
            n_param = adaptive_params.(['layer_' num2str(layer_count)]);
            
            % Apply adaptive activation
            current_input = adaptive_activation(current_input, n_param, 'adaptive_tanh');
        else
            % Fall back to standard activation
            if isa(layer, 'nnet.cnn.layer.TanhLayer')
                current_input = tanh(current_input);
            elseif isa(layer, 'nnet.cnn.layer.ReLULayer')
                current_input = max(0, current_input);
            elseif isa(layer, 'nnet.cnn.layer.SigmoidLayer')
                current_input = sigmoid(current_input);
            end
        end
        
    elseif isa(layer, 'nnet.cnn.layer.FeatureInputLayer')
        % Input layer - no operation needed
        continue;
    end
end

output = current_input;

end
