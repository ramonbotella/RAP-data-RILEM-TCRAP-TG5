function [J grad] = nnCostFunctionContOut(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   hidden_layer_size2,...
                                   num_labels, ...
                                   X, y, lambda)
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad is an "unrolled" vector of the
%   partial derivatives of the neural network.
%
%   Variables with nume of elements of each Theta
T1el = hidden_layer_size*(input_layer_size+1);
T2el = hidden_layer_size2*(hidden_layer_size+1);
T3el = num_labels*(hidden_layer_size2+1);

%   Reshape nn_params back into the parameters Theta1, Theta2 and Theta3
Theta1 = reshape(nn_params(1:T1el), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params(1+T1el:T1el+T2el), ...
                 hidden_layer_size2, (hidden_layer_size + 1));

Theta3 = reshape(nn_params(T1el+T2el+1:end), ...
                 num_labels, (hidden_layer_size2 + 1));             
%   Auxiliary parameter
m = size(X, 1);

%   Initialize cost and theta matrixes
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));
Theta3_grad = zeros(size(Theta3));

%   Add bias term to features
X = [ones(m,1) X];

% Part 1: Feedforward the neural network and return the cost in the
%         variable J.

a2 = sigmoid(X*Theta1');
a2 = [ones(size(a2,1),1) a2];
a3 = sigmoid(a2*Theta2');
a3 = [ones(size(a3,1),1) a3];
a4 = sigmoid(a3*Theta3');

for i =1:m
   J = J +(1/m)*(-y(i,:)*log(a4(i,:)')-(1-y(i,:))*log(1-(a4(i,:)')));
end

% Regularization term

Theta1_R = Theta1(:,2:end);
Theta2_R = Theta2(:,2:end);
Theta3_R = Theta3(:,2:end);

R  = (lambda/(2*m))*(sum(Theta1_R.^2,'all') + sum(Theta2_R.^2,'all')...
    + sum(Theta3_R.^2,'all'));
    

J = J + R;

% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad, Theta2_grad and Theta3_grad. 
%   Initialize Delta matrixes
Delta_1 = zeros(size(Theta1));
Delta_2 = zeros(size(Theta2));
Delta_3 = zeros(size(Theta3));

for t = 1:m
   a_1 = X(t,:);
   z_2 = a_1*Theta1';
   a_2 = sigmoid(z_2);
   a_2 = [1 a_2];
   z_3 = a_2*Theta2';
   a_3 = sigmoid(z_3);
   a_3 = [1 a_3];
   z_4 = a_3*Theta3';
   a_4 = sigmoid(z_4);
   d_4 = a_4-y(t,:);
% Take out the Bias terms of Theta2 and Theta3   
   d_3 = (Theta3(:,2:end)'*d_4').*sigmoidGradient(z_3)';
   d_2 = (Theta2(:,2:end)'*d_3).*sigmoidGradient(z_2)';
   Delta_1 = Delta_1 + d_2*a_1;
   Delta_2 = Delta_2 + d_3*a_2;
   Delta_3 = Delta_3 + d_4'*a_3;
end
Theta1_grad = (1/m)*Delta_1;
Theta2_grad = (1/m)*Delta_2;
Theta3_grad = (1/m)*Delta_3;

% Part 3: Implement regularization with the cost function and gradients.
%
R_1 = [zeros(size(Theta1,1),1) (lambda/m)*Theta1(:,2:end)];
R_2 = [zeros(size(Theta2,1),1) (lambda/m)*Theta2(:,2:end)];
R_3 = [zeros(size(Theta3,1),1) (lambda/m)*Theta3(:,2:end)];

Theta1_grad = Theta1_grad + R_1;
Theta2_grad = Theta2_grad + R_2;
Theta3_grad = Theta3_grad + R_3;

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:) ; Theta3_grad(:)];

end
