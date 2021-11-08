function [X_norm, mu, sigma] = featureNormalize(X)
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. 

X_norm = X;
mu = zeros(1, size(X, 2));
sigma = zeros(1, size(X, 2));

%     For each feature dimension, the mean
%     of the feature is subtracted from the dataset,
%     storing the mean value in mu. Next,the 
%     standard deviation of each feature is computed and 
%     each feature is divided by it's standard deviation, storing
%     the standard deviation in sigma. 
%     Note that X should be a matrix where each column is a 
%     feature and each row is an example.
%     Loop starts in 2 to avoid normalizing the bias term      

for i = 2:size(X,2)
    mu(1,i) = mean(X(:,i));
    sigma(1,i) = std(X(:,i));
    X_norm(:,i) = (X(:,i)- mu(1,i))/sigma(1,i);
end


end
