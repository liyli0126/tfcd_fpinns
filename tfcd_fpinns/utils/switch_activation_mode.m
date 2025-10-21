function switch_activation_mode(mode)
% SWITCH_ACTIVATION_MODE - Quick utility to switch between activation pathways
%
% This function provides an easy way to switch between standard and adaptive
% activation functions by modifying the numerical constants.
%
% Usage:
%   switch_activation_mode('standard')  - Use standard activations
%   switch_activation_mode('adaptive')  - Use adaptive activations
%   switch_activation_mode('status')    - Show current status
%
% Input:
%   mode - 'standard', 'adaptive', or 'status'

if nargin < 1
    mode = 'status';
end

% Get current constants
constants = numerical_constants();

switch lower(mode)
    case 'standard'
        fprintf('🔄 Switching to STANDARD activation mode...\n');
        fprintf('   Modify numerical_constants.m:\n');
        fprintf('   constants.network.use_adaptive_activation = false;\n');
        fprintf('   Then restart MATLAB or clear functions.\n');
        
    case 'adaptive'
        fprintf('🔄 Switching to ADAPTIVE activation mode...\n');
        fprintf('   Modify numerical_constants.m:\n');
        fprintf('   constants.adaptive_activation_enabled = true;\n');
        fprintf('   constants.network.use_adaptive_activation = true;\n');
        fprintf('   Then restart MATLAB or clear functions.\n');
        
    case 'status'
        fprintf('📊 Current Activation Pathway Status:\n');
        fprintf('   Adaptive activation enabled: %s\n', logical2str(constants.adaptive_activation_enabled));
        fprintf('   Network uses adaptive activation: %s\n', logical2str(constants.network.use_adaptive_activation));
        
        if constants.adaptive_activation_enabled && constants.network.use_adaptive_activation
            fprintf('   🟢 Current mode: ADAPTIVE\n');
        else
            fprintf('   🔵 Current mode: STANDARD\n');
        end
        
        fprintf('\n   Available activation types:\n');
        fprintf('   📋 Standard: %s\n', constants.network.activation);
        fprintf('   📋 Adaptive: adaptive_tanh, adaptive_sin, swish\n');
        
    otherwise
        fprintf('❌ Unknown mode: %s\n', mode);
        fprintf('   Available modes: ''standard'', ''adaptive'', ''status''\n');
end

end

function str = logical2str(val)
% Convert logical to string
if val
    str = 'true';
else
    str = 'false';
end
end
