import numpy as np
import matplotlib.pyplot as plt
############################################################################
#   by Ramon Botella, 11/22/2021
#
#   This script trains a Support Vector Regression to
#   to predict Degree of binder Activation (DoA) in % max. ITS
#   of a Reclaimed Asphalt Pavement sample from the compaction
#   temperature in ºC, the air voids in % and the Indirect
#   Tensile Strength at 25ºC of Marshall compacted specimens.
#
#   To data end the data is splitt in train and test sets (80/20)
#   the hyperparameters C and epsilon are tunned using grid_search
#   and the final model precision is measured. This process is repeated
#   with 10 different random data splits and the average MAE of all models
#   is reported.
############################################################################

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV         

# Reading data
dataset = pd.read_csv('rapdataALLMarshall.csv')
x = dataset.drop('DoA (% max.ITS)',axis = 1)
x = np.array(x)
y = np.array(dataset['DoA (% max.ITS)'])

mae = np.zeros((10,1))
mae_std = np.zeros((10,1))
for i in range(0,9):
    # Splitting data into randomized train and test sets
    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2,
                                                        random_state = i+5)
                                                    
    # Reshape y for StandardScaler
    y_train = y_train.reshape(-1,1)
    y_test = y_test.reshape(-1,1)
    ##
    ##### Feature scaling
    sc_x = StandardScaler()
    sc_y = StandardScaler()
    x_train = sc_x.fit_transform(x_train)
    y_train = sc_y.fit_transform(y_train)
    x_test = sc_x.transform(x_test)

    ### Reshape back y for SVR
    y_train = y_train.reshape(-1,)
    y_test = y_test.reshape(-1,)

    # Call SVR function
    regressor = SVR()
    ### Tunning hyperparameters
    Cfine = [1,3,5,6,7,9,10,11,13,15,20]
    epsilon_fine = [0.1,0.2,0.3,0.4,0.5,0.6,0.8,1,1.3]
    param_grid = {'C':Cfine,
              'epsilon':epsilon_fine}
    grid_search = GridSearchCV(estimator = regressor,
                           param_grid = param_grid,
                           cv = 5,
                           n_jobs = -1,
                           verbose = 2)
    # Fitting models with fine tuned parameters
    grid_search.fit(x_train, y_train)
    best_hyperparams = grid_search.best_params_

    ##### Fitting SVR to dataset
    final_model = SVR(kernel = 'rbf',
                      C = best_hyperparams['C'],
                      epsilon = best_hyperparams['epsilon'])
    final_model.fit(x_train,y_train)
    

    ##### Measuring precision
    y_pred_test = final_model.predict(x_test)
    y_pred_test =sc_y.inverse_transform(y_pred_test)
    mae[i] = np.mean(abs(y_test-y_pred_test))
    mae_std[i] = np.std(abs(y_test-y_pred_test))
    print('*******************')
    print('Model evaluation')
    print('Iteration:  %.0f  ' % (i+1))
    print('Random state: %.0f ' % (i+5))
    print('Model R2:  %.2f  ' % final_model.score(x_train,y_train))
    print('Average MAE on test set:  %.2f  ' % mae[i])
    print('Std. Dev. on MAE: %.2f ' % mae_std[i])
    print('Best Hyperparameters')
    print(best_hyperparams)
    print('*******************')

print('*******************')
print('*******************')
print('Average MAE all models: %.2f ' % np.mean(mae))
print('Average Std. Dev. on MAE all models: %.2f ' % np.mean(mae_std))


