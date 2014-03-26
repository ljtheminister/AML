function [errors, gradient] = computeCost(layers, labels, cost)

output = layers(end).output;
[nSamples nLabels] = size(output);
nLayers = length(layers);

% Compute the cost due to weight decays.
if isfield(layers(1), 'wdCost')
	wdCost = sum([layers.wdCost]);
else
	wdCost = 0;
end

switch cost
case 'mse'
	diff = output - labels;
	errors = .5*sum(diff.^2,2) + wdCost;
	gradient = diff/nSamples;
case 'ce'
	errors = -sum(output.*labels - log(1 + exp(output)), 2) + wdCost;

	% Avoid overflows.
	errors(output > 20) = -sum(output(output > 20).*labels(output > 20) - output(output > 20), 2) + wdCost;
	gradient = -(labels - sigm(output))/nSamples;

case 'nll'
	softmaxOutput = softmax(output, 2);
	errors = sum(-log(softmaxOutput).*labels,2) + wdCost;
	gradient = (- labels + softmaxOutput)/nSamples;
case 'class'
	if size(output, 2) == 1
		errors = ( sign(output) ~= (2*labels-1) );
	else
		[~, valueOutput] = max(output, [], 2);
		errors = ( valueOutput ~= labels*(1:nLabels)' );
		if nargout > 1, error('This cost is not designed for training'); end
    end

case 'huber mse'
    diff = output - labels;
    delta = .25;
    errors = sum(arrayfun(@Huber, diff));
    gradient = delta*sign(diff)/nSamples;
    
case 'student mse'
    v = 1; %degrees of freedom: n-1
    diff = output - labels;
    errors = .5*sum(log(v+diff.^2),2) + wdCost;
    gradient = 2*(diff./(v+diff.^2))/nSamples;
end
end

function [a] = Huber(a)
    delta = .25;
    if abs(a) <= delta
        a = .5*a^2;
    else
        a = delta*(abs(a) - delta/2);
    end
end
