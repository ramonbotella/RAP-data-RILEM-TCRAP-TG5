# -*- coding: utf-8 -*-
"""
@author: ramonbotella, modified by kbernatowicz
"""

import pandas as pd                                      # Pandas is used for data manipulation
import numpy as np                                       # Numpy used for math 
import joblib                                            # Joblib to save final RF model                          
from sklearn.ensemble import RandomForestRegressor       # Sklearn to perform RF regression

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

#%%Loading data 

file_path = r'rapdataPyRF.csv'                           # Read in data
features = pd.read_csv(file_path,encoding='utf-8')       # Features are the input variables
labels = np.array(features['DoA (% max.ITS)'])           # Labels are the values we want to predict
features= features.drop('DoA (% max.ITS)', axis = 1)     # Remove the labels from the features
                                                         # axis 1 refers to the columns
feature_list = list(features.columns)                    # Saving feature names for later use
features = np.array(features)                            # Convert to numpy array

#Renamed to generalize code applications
x = features
y = labels
rndState = 2
   
#%% Create model with optimum hyperparameters
rf = RandomForestRegressor(n_estimators = 1000,
                                 max_depth= 40, 
                                 min_samples_split= 7,
                                 min_samples_leaf = 2,
                                 bootstrap = True,
                                 random_state = 2,)
# Fit the model to the whole data set
rf.fit(x, y)
# Store the definitive model in a file
joblib.dump(rf,"./random_forest.joblib")

# Evaluate de model
scores = evaluate(rf,x,y)

# Get numerical feature importances
importances = list(rf.feature_importances_)

# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]

# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)

# Print out the feature and importances

print('*******************************')
[print('Feature: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];

