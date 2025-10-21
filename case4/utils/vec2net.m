function net = vec2net(vec, net_template)
% VEC2NET - Convert parameter vector back to network for L-BFGS optimization
%
% Inputs:
%   vec          - parameter vector
%   net_template - network template
%
% Outputs:
%   net - updated dlnetwork object

lgraph = layerGraph(net_template.Layers);
start_idx = 1;

for i = 1:numel(lgraph.Layers)
    layer = lgraph.Layers(i);
    
    % Only process layers with Weights/Bias
    if isprop(layer, 'Weights')
        sz = size(layer.Weights);
        numW = prod(sz);
        layer.Weights = reshape(vec(start_idx:start_idx+numW-1), sz);
        start_idx = start_idx + numW;
    end
    
    if isprop(layer, 'Bias')
        sz = size(layer.Bias);
        numB = prod(sz);
        layer.Bias = reshape(vec(start_idx:start_idx+numB-1), sz);
        start_idx = start_idx + numB;
    end
    
    lgraph = replaceLayer(lgraph, layer.Name, layer);
end

net = dlnetwork(lgraph); % Automatic initialization

end

