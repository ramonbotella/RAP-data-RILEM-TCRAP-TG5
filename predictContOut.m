function p = predictContOut(Theta1, Theta2, Theta3, X)
%   p = PREDICT(Theta1, Theta2, Theta3, X) outputs the predicted label of X 
%   given the trained weights of a neural network (Theta1, Theta2, Theta3)

% Auxiliary variables
m = size(X, 1);
num_labels = size(Theta3, 1);

% Initialize probability vector
p = zeros(size(X, 1), 1);

% Apply the neural network to the features
h1 = sigmoid([ones(m, 1) X] * Theta1');
h2 = sigmoid([ones(m, 1) h1] * Theta2');
h3 = sigmoid([ones(m, 1) h2] * Theta3');

% Probability 0<=p<=100 output
p = h3;

end
