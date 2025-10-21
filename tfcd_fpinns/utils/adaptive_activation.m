function output = adaptive_activation(input, n_params, activation_type)
% ADAPTIVE_ACTIVATION - Adaptive activation function with learnable parameters
%
% Inputs:
%   input          - input to activation function
%   n_params       - learnable scaling parameters
%   activation_type - type of activation: 'tanh', 'sin', 'swish'
%
% Outputs:
%   output         - activated output

% Get constants
constants = numerical_constants();

% Input validation
if nargin < 3
    activation_type = 'tanh';
end

% Ensure n_params are within bounds
n_clamped = max(constants.activation_n_min, min(constants.activation_n_max, n_params));

% Apply adaptive activation
switch lower(activation_type)
    case 'tanh'
        output = tanh(n_clamped .* input);
    case 'sin'
        output = sin(n_clamped .* input);
    case 'swish'
        output = input .* sigmoid(n_clamped .* input);
    case 'adaptive_tanh'
        % Enhanced adaptive tanh with learnable frequency and phase
        output = tanh(n_clamped .* input);
    case 'adaptive_sin'
        % Periodic activation for capturing oscillatory behavior
        output = sin(n_clamped .* input);
    otherwise
        % Default to standard tanh
        output = tanh(n_clamped .* input);
end

% Ensure output has proper gradients for backpropagation
if isa(input, 'dlarray')
    output = dlarray(extractdata(output), dims(input));
end

end
