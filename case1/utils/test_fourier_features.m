function test_fourier_features()
% TEST_FOURIER_FEATURES - Test Fourier feature mapping functionality
%
% This function tests the Fourier feature mapping implementation

fprintf('=== Testing Fourier Feature Mapping ===\n');

% Get constants
constants = numerical_constants();

% Test parameters
input_dims = 2;
mapping_size = constants.fourier_mapping_size;
sigma = constants.fourier_sigma;

fprintf('Input dimensions: %d\n', input_dims);
fprintf('Mapping size: %d\n', mapping_size);
fprintf('Sigma: %.2f\n', sigma);

% Generate test coordinates
t_test = [0.1, 0.5, 0.9];
x_test = [0.2, 0.5, 0.8];
coords = [t_test; x_test]';

fprintf('\nTest coordinates:\n');
fprintf('t: [%.1f, %.1f, %.1f]\n', t_test);
fprintf('x: [%.1f, %.1f, %.1f]\n', x_test);

% Generate Gaussian matrix B
B = sigma * randn(mapping_size, input_dims);

fprintf('\nGaussian matrix B:\n');
fprintf('Size: %d x %d\n', size(B));
fprintf('Mean: %.4f\n', mean(B(:)));
fprintf('Std: %.4f\n', std(B(:)));

% Test Fourier feature mapping
fprintf('\nTesting Fourier feature mapping...\n');

% Without original coordinates
mapped_coords_no_orig = fourier_feature_mapping(coords, B, false);
fprintf('Without original coordinates:\n');
fprintf('  Input size: %s\n', mat2str(size(coords)));
fprintf('  Output size: %s\n', mat2str(size(mapped_coords_no_orig)));
fprintf('  Expected size: [%d, %d]\n', size(coords, 1), 2*mapping_size);

% With original coordinates
mapped_coords_with_orig = fourier_feature_mapping(coords, B, true);
fprintf('\nWith original coordinates:\n');
fprintf('  Output size: %s\n', mat2str(size(mapped_coords_with_orig)));
fprintf('  Expected size: [%d, %d]\n', size(coords, 1), 2*mapping_size + input_dims);

% Test network creation
fprintf('\nTesting network creation...\n');
try
    net = create_network();
    fprintf('✓ Network created successfully\n');
    fprintf('  Network size: %s\n', mat2str(size(net.Layers)));
    
    if isfield(net, 'B')
        fprintf('✓ Fourier matrix B stored in network\n');
        fprintf('  B size: %s\n', mat2str(size(net.B)));
    else
        fprintf('✗ Fourier matrix B not found in network\n');
    end
    
catch ME
    fprintf('✗ Network creation failed: %s\n', ME.message);
end

% Test forward pass
fprintf('\nTesting forward pass...\n');
try
    test_input = dlarray([0.5; 0.5], "CB");
    output = forward_with_fourier(net, test_input);
    fprintf('✓ Forward pass successful\n');
    fprintf('  Output size: %s\n', mat2str(size(output)));
    fprintf('  Output value: %.6f\n', extractdata(output));
    
catch ME
    fprintf('✗ Forward pass failed: %s\n', ME.message);
end

fprintf('\n=== Fourier Feature Testing Complete ===\n');

end
