function out = mapFeature(X1, X2, X3)
%   MAPFEATURE(X1, X2) maps the two input features
%   to quadratic features used in the regularization exercise.
%
%   Returns a new feature array with more features, comprising of 
%   X1, X2, X3, X1.^2, X2.^2, X3^2, X1*X2*X3, X1*X2.^2*X3, etc..
%
%   Inputs X1, X2, X3 must be the same size
%

X = [X1 X2 X3];

degree = 6;
out = ones(size(X1(:,1)));
for t = 1:2
    for i = 1:degree
        for j = 0:i
            out(:, end+1) = (X(:,t).^(i-j)).*(X(:,t+1).^j);
        end
    end
end
for i = 1:degree
    for j = 0:i
        out(:, end+1) = (X(:,1).^(i-j)).*(X(:,3).^j);
    end
end
end