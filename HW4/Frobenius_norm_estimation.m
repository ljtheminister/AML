%% set seed for random number generation
s = RandStream('mt19937ar','Seed',1);
RandStream.setGlobalStream(s);
%rng default

%% generate random matrix A
n = 500;
p = 1000;

A = rand(n, p);
%A = randi5, n, p);
%A = randn(n, p);

norm_Frobenius = norm(A, 'fro')^2



