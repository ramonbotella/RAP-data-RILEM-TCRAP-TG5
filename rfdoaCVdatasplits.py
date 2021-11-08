# -*- coding: utf-8 -*-
"""
Modified on Fri Jul 17 16:36:08 2021

@author: ramonbotella, modified by kbernatowicz
"""

import pandas as pd                                      # Pandas is used for data manipulation
import numpy as np                                       # Numpy used for math 
import joblib                                            # Joblib to save final RF model                          
from sklearn.ensemble import RandomForestRegressor       # Sklearn to perform RF regression
from sklearn.model_selection import RandomizedSearchCV   # Module to crossvalitade model
from sklearn.model_selection import train_test_split     # Package to split data into train and test sets
from sklearn.model_selection import GridSearchCV         # Packege to grid search best hyperparameters combination
train_size = 0.8                                         # Determine train&test as 80% & 20% of the data
# Defining function to evaluate models
#####################################################
def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    mape = 100 * np.mean(errors / test_labels)
    accuracy = 100 - mape
    mae = np.mean(errors)
    rsquare = model.score(test_features,test_labels)
    print('Model Performance')
    print('Mean Absolute Error: {:0.2f}%max.ITS'.format(np.mean(errors)))
    print('R2: {:0.2f}%max.ITS'.format(rsquare))
    print('Accuracy = {:0.2f}%'.format(accuracy))
    
    return accuracy, mae, rsquare
#######################################################
file_path = r'rapdataPyRF.csv'                           # Read in data
features = pd.read_csv(file_path,encoding='utf-8')       # Features are the input variables
labels = np.array(features['DoA (% max.ITS)'])           # Labels are the values we want to predict
features= features.drop('DoA (% max.ITS)', axis = 1)     # Remove the labels from the features
                                                         # axis 1 refers to the columns
feature_list = list(features.columns)                    # Saving feature names for later use
features = np.array(features)                            # Convert to numpy array

# Renamed to generalize code applications
x = features
y = labels
# Number of iterations
n_iters = 20
# Initalize arrays to store model precision scores
accuracies = np.zeros((n_iters,2))
errors = np.zeros ((n_iters,2))
rsquares = np.zeros((n_iters,2))

# Starts loop to train n_iters models with different
# data splits with the optimum hyperparameters
for i in range(0,n_iters-1):
    # Split data randomly into train and test sets
    x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8)

    # First create the base model to tune
    rf = RandomForestRegressor(n_estimators = 1000,
                                 max_depth= 40, 
                                 min_samples_split= 7,
                                 min_samples_leaf = 2,
                                 bootstrap = True,
                                 random_state = 2,)
    # Fit the model to the train data
    rf.fit(x_train, y_train)
    # Compute precision scores on test data
    accuracies [i,1] = evaluate(rf, x_test, y_test)[0]
    errors [i,1] = evaluate(rf, x_test, y_test)[1]
    rsquares[i,1] = evaluate(rf, x_test, y_test)[2]

# Store resutls
np.savetxt('accuraciesall.dat',accuracies, delimiter=',')
np.savetxt('errorsall.dat',errors, delimiter=',')
np.savetxt('rsquaredall.dat',rsquares, delimiter=',')
