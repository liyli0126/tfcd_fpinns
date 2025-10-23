function net = create_network()
% CREATE_NETWORK - Create PINN neural network architecture
%
% Returns:
%   net - dlnetwork object for the PINN

layers = [
    featureInputLayer(2)      % Input: [t; x]
    fullyConnectedLayer(256)  % Increased from 128 to 256
    tanhLayer
    fullyConnectedLayer(256)  % Increased from 128 to 256
    tanhLayer
    fullyConnectedLayer(128)  % Increased from 64 to 128
    tanhLayer
    fullyConnectedLayer(64)   % New layer for better expressiveness
    tanhLayer
    fullyConnectedLayer(1)    % Output: u(x,t)
];

net = dlnetwork(layers);

end

