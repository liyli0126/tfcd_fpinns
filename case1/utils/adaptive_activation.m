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
n_params = max(constants.activation_n_min, min(constants.activation_n_max, n_params));

% Apply adaptive activation based on type
switch lower(activation_type)
    case 'tanh'
        % Adaptive tanh: tanh(n * x)
        output = tanh(n_params .* input);
        
    case 'sin'
        % Adaptive sin: sin(n * x)
        output = sin(n_params .* input);
        
    case 'swish'
        % Adaptive swish: x * sigmoid(n * x)
        output = input .* sigmoid(n_params .* input);
        
    case 'gelu'
        % Adaptive GELU: x * Φ(n * x) where Φ is normal CDF
        output = input .* 0.5 .* (1 + erf((n_params .* input) / sqrt(2)));
        
    otherwise
        % Default to adaptive tanh
        output = tanh(n_params .* input);
end

end

function y = sigmoid(x)
% Sigmoid function
y = 1 ./ (1 + exp(-x));
end
