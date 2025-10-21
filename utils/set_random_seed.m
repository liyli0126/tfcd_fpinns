function set_random_seed(seed)
% SET_RANDOM_SEED - Set random seed for reproducible results
%
% Inputs:
%   seed - random seed value (default: 2024)

if nargin < 1
    seed = 2024;
end

% Set MATLAB random seed
rng(seed, 'twister');

% Set global random stream for consistent results
global RandomStream;
RandomStream = RandStream('mt19937ar', 'Seed', seed);
RandStream.setGlobalStream(RandomStream);

fprintf('Random seed set to: %d\n', seed);

end
