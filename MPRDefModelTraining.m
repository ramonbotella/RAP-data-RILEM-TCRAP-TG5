%% 
% Loading data (no heading, file with i columns and m rows, features stored 
% in column 1 to i-1 and labes are stored in column i)

%  The first two columns contains the X values and the third column
%  contains the label (y).
data = load('rapdataMatlabMPR.csv');
%   Shuffleling the rows of data
%   Create a vector of m components with index randomized with the
%   following comand
random_state = randperm(size(data, 1));
%   In order to reproduce the same result the random state vector should be
%   saved for future compilations of the model
%   Uncomment next line if a random state has been previously saved
%random_state = load('random_state_vector.csv');
dataR = data(random_state, :);

%  Separate Train/Test data
m = size(dataR,1);
X = dataR(:, 1:(size(data,2)-1)); 
y = dataR(:,size(data,2));
%% 
% Map features to 6th degree polinomial

% Add Polynomial Features
% Note that mapFeature also adds a column of ones for us, so the intercept term is handled
%X = mapFeature(X(:,1),X(:,2),X(:,3));
%X_test = mapFeature(X_test(:,1),X_test(:,2),X_test(:,3));
X = mapFeature(X(:,1),X(:,2),X(:,3));
%% 
% Normalization of features

% Scale features and set them to zero mean
[X, mu, sigma] = featureNormalize(X);   
%% 
% Running gradient descend with the optimun lambda that minimizes cost on test 
% function and prevents overfiting on the train set 

% Run gradient descent
alpha = 1;
num_iters = 500000;
lambda = 0.08;

% Init Theta and Run Gradient Descent 
theta = zeros(size(X,2), 1);
[theta,Cost] = gradientDescentMultiReg(X, y, theta, alpha, num_iters, lambda);
%% 
% Goodness of fit on Train set

% Accuracy on train set predictions
y_pred = X*theta;
error_train = sum(abs(y-y_pred))/size(y,1);
%% 
% Cost on the train set set

J_train = (1/(2*(size(X,1))))*(X*theta-y)'*(X*theta-y);
%% 
% Printing values

fprintf('**************************************************')
fprintf('**************************************************')
% Printing lambda value applied
fprintf('Regularization parameter: %f',  lambda)
% Printing Mean Absolute Error on Train set
fprintf('Main Absolute Error on Train set: %f',  error_train)
fprintf('**************************************************')
fprintf('**************************************************')

%% 
%
