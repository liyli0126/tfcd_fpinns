function vec = net2vec(net)
% NET2VEC - Convert network parameters to vector for L-BFGS optimization
%
% Inputs:
%   net - dlnetwork object
%
% Outputs:
%   vec - parameter vector

params = net.Learnables;
vec = [];

for i = 1:height(params)
    value = params.Value{i};
    if iscell(value)
        value = value{1};
    end
    if isa(value, 'dlarray')
        value = extractdata(value);
    end
    vec = [vec; double(value(:))];
end

end

