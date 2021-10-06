function [theta, J_history] = gradientDescentMultiReg(X, y, theta, alpha, num_iters,lambda)
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

%   Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
J = 0;
grad = zeros(size(theta));

for iter = 1:num_iters
  
    % Compute cost without Regularization
    J = (1/(2*m))*(X*theta-y)'*(X*theta-y);

    % Regularization sumation term of cost function
    R = 0;
    for i = 2:size(theta,1)
        R = R +(lambda/(2*m))*theta(i)^2;
    end
    
    % Save the cost J with Regularization in every iteration
    
    J_history(iter) = J + R;
    
    % Compute gradient J without Regularization
    grad = (1/m)*(X'*((X*theta)-y));

    % Substitute bias term on theta by 0 to avoid regularize Theta_0
    
    temp = theta;
    temp(1) = 0;
    
    % gradient J with Regularization
    grad = grad + (lambda/m)*temp;
    
    % Update theta vector
    theta = theta - (alpha/m)*grad;
end
end
