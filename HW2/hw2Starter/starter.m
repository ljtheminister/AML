clear all; close all; clc

% set up data
nInst = 100;   % number of measurements
nVars = 100;   % number of variables
X = randn(nInst,nVars);  % Gaussian (feature) matrix
w = randn(nVars,1);      % 'true' values of weights. 
y = sign(X*w + randn(nInst,1)); % 'true' simulated observations (two classes)

%

w_init = zeros(nVars,1);      % initial value (all zeros)
[f g] = LogisticLoss(w_init,X,y);  




%% here's how you can optimize unconstrained objective using minFunc
%objFun = @(w)LogisticLoss(w, X, y); 
%options.Method = 'lbfgs';
%wMF = minFunc(objFun, w_init, options);





 


