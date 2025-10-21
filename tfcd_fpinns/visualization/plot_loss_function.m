function plot_loss_function(loss_history)
% PLOT_LOSS_FUNCTION - Plot training loss history
%
% Inputs:
%   loss_history - training loss history

figure;
semilogy(loss_history, 'LineWidth', 1.5);
title('Loss Function vs. Iteration');
xlabel('Iteration');
ylabel('Loss (log scale)');

end

