function output = forward_with_fourier(net, input_coords)
% FORWARD_WITH_FOURIER - Forward pass with Fourier feature mapping
%
% Inputs:
%   net          - neural network with Fourier features
%   input_coords - input coordinates [t; x]
%
% Outputs:
%   output       - network output

% Get constants
constants = numerical_constants();

% Initialize output variable
output = [];

if constants.fourier_features_enabled
    % Get Fourier matrix B from global variable
    global fourier_matrix_B;
    
    % Check if Fourier matrix B exists and is valid
    if ~isempty(fourier_matrix_B) && all(size(fourier_matrix_B) == [constants.fourier_mapping_size, 2])
        % Apply Fourier feature mapping while preserving dlarray format
        % input_coords is [2, N] format, B is [mapping_size, 2]
        % We need: [2, N]' * [mapping_size, 2]' = [N, 2] * [2, mapping_size] = [N, mapping_size]
        
        % Extract data for matrix operations while preserving gradient information
        coords_data = extractdata(input_coords);  % [2, N]
        coords_t = coords_data';                  % [N, 2]
        
        % Matrix multiplication: [N, 2] * [2, mapping_size] = [N, mapping_size]
        proj = 2 * pi * coords_t * fourier_matrix_B';  % [N, mapping_size]
        
        % Apply sine and cosine transformations
        sin_features = sin(proj);  % [N, mapping_size]
        cos_features = cos(proj);  % [N, mapping_size]
        
        % Combine features
        if constants.fourier_include_original
            % Include original coordinates: [sin_features, cos_features, coords_t]
            mapped_coords = [sin_features, cos_features, coords_t];  % [N, 2*mapping_size+2]
        else
            % Only Fourier features: [sin_features, cos_features]
            mapped_coords = [sin_features, cos_features];  % [N, 2*mapping_size]
        end
        
        % Convert back to dlarray for network processing
        mapped_input = dlarray(mapped_coords', "CB");  % [features, N]
        
        % Forward pass through the network
        output = predict(net, mapped_input);
    else
        % Fourier matrix B is missing or invalid, fallback to standard forward pass
        warning('Fourier matrix B is missing or invalid, using standard forward pass');
        output = predict(net, input_coords);
    end
else
    % Standard forward pass without Fourier features
    output = predict(net, input_coords);
end

% Final safety check - ensure output is assigned
if isempty(output)
    error('Output not assigned in forward_with_fourier function');
end
