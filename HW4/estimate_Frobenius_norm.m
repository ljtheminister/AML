function [norm_F] = estimate_Frobenius_norm(A, N)
% estimate Frobenius norm of matrix A by hitting A with random vectors that
% are uniformly distributed on n-sphere (spherical uniform distribution)

%A := matrix for which Frobenius norm is estimated
%N := number of iterations aka 

n = size(A,1);
p = size(A,2);

norm_sum = 0;

for i = 1:N
   w = randn(1,p)';
   w = w/norm(w);
   %w = w./sqrt(w*w')
   %w = bsxfun(@rdivide,w,sqrt(sum(v.^2,2)));
   X = A*w;
   norm2 = norm(X,2)^2
   norm_sum = norm_sum + norm2;
end

norm_F = norm_sum/N


end
