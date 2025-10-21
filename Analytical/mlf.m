function E = mlf(alpha, beta, z)
% MLF - Mittag-Leffler function E_α,β(z)
%
% E_α,β(z) = Σ(k=0 to ∞) z^k / Γ(αk + β)
%
% Inputs:
%   alpha - fractional order (0 < alpha ≤ 1)
%   beta  - parameter (usually 1)
%   z     - complex argument
%
% Output:
%   E     - Mittag-Leffler function value

% Handle scalar inputs
if isscalar(z)
    z = z(:);
end

% Initialize output
E = zeros(size(z));

% Maximum number of terms for series expansion
max_terms = 100;

% Compute Mittag-Leffler function using series expansion
for i = 1:length(z)
    z_val = z(i);
    
    if abs(z_val) < 1e-10
        % For z ≈ 0, E_α,β(0) = 1/Γ(β)
        E(i) = 1 / gamma(beta);
    else
        % Series expansion: E_α,β(z) = Σ(k=0 to ∞) z^k / Γ(αk + β)
        sum_val = 0;
        term = 1;
        
        for k = 0:max_terms
            if k == 0
                term = 1 / gamma(beta);
            else
                term = term * z_val / gamma(alpha * k + beta);
            end
            
            sum_val = sum_val + term;
            
            % Check convergence
            if abs(term) < 1e-15
                break;
            end
        end
        
        E(i) = sum_val;
    end
end

% Handle special cases
if isscalar(z)
    E = E(1);
end

end
