function [lambda_soe, theta] = generate_SOE(beta, epsilon, delta)
% GENERATE_SOE: Generate SOE approximation parameters for t^(-beta)
% 
% Inputs:
%   beta    - fractional exponent (0 < beta < 1)
%   epsilon - relative error tolerance (e.g., 1e-8)
%   delta   - lower bound of time interval (e.g., 1e-6)
%
% Outputs:
%   lambda_soe - vector of SOE exponents
%   theta      - vector of SOE weights

% Validate input
if beta <= 0 || beta >= 1
    error('beta must be in (0,1)');
end

if nargin < 2
    epsilon = 1e-10;
end
if nargin < 3
    delta = 1e-6;
end

% Parameters for exponential sum
M = ceil(2*log(1/epsilon));  % Number of exponentials
s = linspace(log(delta), 0, M);  % Log-spaced nodes

lambda_soe = zeros(1, M);
theta = zeros(1, M);

for j = 1:M
    lambda_soe(j) = exp(-s(j));
    theta(j) = lambda_soe(j)^(1 - beta) * (log(1 + 1/lambda_soe(j))) / gamma(1 - beta);
end

% Normalize weights
theta = theta / sum(theta);

end

