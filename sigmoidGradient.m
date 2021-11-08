function g = sigmoidGradient(z)
%   g = SIGMOIDGRADIENT(z) computes the gradient of the sigmoid function
%   evaluated at z. This should work regardless if z is a matrix or a
%   vector. In particular, if z is a vector or matrix, it returns
%   the gradient for each element.

g = zeros(size(z));
g = (1./(1+exp(-z))).*(1-(1./(1+exp(-z))));

end
