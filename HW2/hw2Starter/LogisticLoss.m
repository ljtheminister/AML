function [nll,g] = LogisticLoss(w,X,y)
% w(feature,1)
% X(instance,feature)
% y(instance,1)

[n,p] = size(X);

Xw = X*w;
yXw = y.*Xw;

nll = sum(mylogsumexp([zeros(n,1) -yXw]));

if nargout > 1
        g = -(X.'*(y./(1+exp(yXw))));
end