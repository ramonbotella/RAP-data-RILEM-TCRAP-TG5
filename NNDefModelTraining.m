%% 
% Load RAP Data

% First columns of data file are features and last column contains labels
% Data without heading
data = load('rapdataMatlabANN.csv');
%   Shuffleling the rows of data
%   Create a vector of m components with index randomized with the
%   following comand
%%%%%%%%%%%%%%%%%%%%%%%%%%
random_state = randperm(size(data, 1));
%%%%%%%%%%%%%%%%%%%%%%%%%%
%   In order to reproduce the same result the random state vector should be
%   saved for future compilations of the model
%   Comment line 8 and uncomment line 13 if you want to reproduce a
%   previous randomization
%   random_state = load('random_state_vector.csv');
dataR = data(random_state, :);
%  Separate Train/Test data
m = size(dataR,1);
X = dataR(:, 1:(size(data,2)-1)); 
y = (dataR(:,size(data,2)))/100;
%% 
% Initialization of main parameters

input_layer_size  = 3;   % Temperature, Air voids and ITS
hidden_layer_size = 5;   % 5 hidden units
hidden_layer_size2 = 5;  % 2 hidden layers
num_labels = 1;          % 1 labes 0-1
%% 
% Initialization of Theta1, Theta2 and Theta3 matrixes

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, hidden_layer_size2);
initial_Theta3 = randInitializeWeights(hidden_layer_size2, num_labels);

% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:); initial_Theta3(:)];

% Auxiliary variables for later reshaping Thetas
T1el = hidden_layer_size*(input_layer_size+1);
T2el = hidden_layer_size2*(hidden_layer_size+1);
T3el = num_labels*(hidden_layer_size2+1);
%% 
% Get the best Theta1, Theta2 and Theta3 coefficients using fmincg optimization 
% algorithm, i.e., training the neural network

options = optimset('MaxIter', 10000);
lambda = 0.08;

% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunctionContOut(p, input_layer_size, hidden_layer_size, hidden_layer_size2, num_labels, X, y, lambda);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params,progres,t] = fmincg(costFunction, initial_nn_params, options);
[Cost_train, ~] = nnCostFunctionContOut(nn_params, input_layer_size, hidden_layer_size, hidden_layer_size2, num_labels, X, y, lambda);

%% 
% Neural network has been trained. Now we measure its accuracy in predicting 
% the training set labels

% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:T1el), hidden_layer_size, (input_layer_size + 1));
Theta2 = reshape(nn_params(1 + T1el:T1el+T2el), hidden_layer_size2, (hidden_layer_size + 1));
Theta3 = reshape(nn_params(T1el+T2el+1:end), num_labels, (hidden_layer_size2 + 1));

y_pred = predictContOut(Theta1, Theta2, Theta3, X);
Error_train = sum(abs(y-y_pred))/(size(y,1));


fprintf('\nRegularization parameter: %f\n', lambda);
fprintf('\nMean Error on Train Set: %f\n', 100*Error_train);
