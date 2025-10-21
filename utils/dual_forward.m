function output = dual_forward(net, input_data, adaptive_params)
% DUAL_FORWARD - Unified forward pass supporting both standard and adaptive activations
%
% This function automatically chooses between standard and adaptive activation
% pathways based on the adaptive_params.enabled flag, providing seamless
% switching between the two approaches.
%
% Inputs:
%   net            - dlnetwork object
%   input_data     - input data as dlarray
%   adaptive_params - struct containing adaptive parameters and enabled flag
%
% Outputs:
%   output         - network output

% Get constants
constants = numerical_constants();

% Pathway 1: Adaptive Activation (if enabled)
if adaptive_params.enabled && constants.adaptive_activation_enabled
    output = adaptive_forward_pathway(net, input_data, adaptive_params);
    return;
end

% Pathway 2: Standard Activation (default)
output = predict(net, input_data);

end

function output = adaptive_forward_pathway(net, input_data, adaptive_params)
% ADAPTIVE_FORWARD_PATHWAY - Forward pass with adaptive activation functions

% Get constants
constants = numerical_constants();

% Manual forward pass with adaptive activations
current_input = input_data;
layer_count = 0;

% Get network layers
layers = net.Layers;

for i = 1:numel(layers)
    layer = layers(i);
    
    if isa(layer, 'nnet.cnn.layer.FullyConnectedLayer')
        % Apply fully connected transformation
        weights_idx = find(strcmp(net.Learnables.Layer, layer.Name) & strcmp(net.Learnables.Parameter, 'Weights'));
        bias_idx = find(strcmp(net.Learnables.Layer, layer.Name) & strcmp(net.Learnables.Parameter, 'Bias'));
        
        if ~isempty(weights_idx) && ~isempty(bias_idx)
            weights = net.Learnables.Value{weights_idx};
            bias = net.Learnables.Value{bias_idx};
            
            % Linear transformation: W*x + b
            current_input = weights * current_input + bias;
            layer_count = layer_count + 1;
        end
        
    elseif isa(layer, 'nnet.cnn.layer.TanhLayer') || isa(layer, 'nnet.cnn.layer.ReLULayer') || isa(layer, 'nnet.cnn.layer.SigmoidLayer')
        % Apply adaptive activation for hidden layers
        param_name = ['layer_' num2str(layer_count)];
        
        if isfield(adaptive_params, param_name) && layer_count <= length(constants.network.hidden_layers)
            % Get adaptive parameter for this layer
            n_param = adaptive_params.(param_name);
            
            % Apply adaptive activation based on original layer type
            if isa(layer, 'nnet.cnn.layer.TanhLayer')
                current_input = adaptive_activation(current_input, n_param, 'adaptive_tanh');
            elseif isa(layer, 'nnet.cnn.layer.ReLULayer')
                current_input = adaptive_activation(current_input, n_param, 'swish'); % More sophisticated than ReLU
            elseif isa(layer, 'nnet.cnn.layer.SigmoidLayer')
                current_input = adaptive_activation(current_input, n_param, 'adaptive_tanh');
            end
        else
            % Output layer or layers without adaptive params: use standard activation
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
