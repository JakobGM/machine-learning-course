#!/usr/bin/env python
# coding: utf-8

# # Santander Customer Transaction Prediction

# In[49]:

import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import mean_squared_error, roc_auc_score, roc_curve, confusion_matrix


train_df = pd.read_csv("project/data/train_small.csv", header=None)
test_df = pd.read_csv("project/data/test_small.csv")

train_df.columns = list([test_df.columns[0],"target",*test_df.columns[1:]])

response = np.array(train_df['target'])
features = np.array(train_df.iloc[:,2:])
features_list = list(train_df.iloc[:,2:].columns)

from sklearn.model_selection import train_test_split
train_features, test_features, train_response, test_response = train_test_split(features, response, test_size = 0.25, random_state = 42)


# In[72]:

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
# Create the parameter grid based on the results of random search 
param_grid = {
    'bootstrap': [False],
    'max_depth': [20, 30, 40],
    'max_features': [2, 3],
    'min_samples_leaf': [4, 5, 6],
    'min_samples_split': [2, 3, 4],
    'n_estimators': [1800, 2000, 2200]
}

# Create a base model
rf = RandomForestRegressor(random_state = 42)

# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                          cv = 3, n_jobs = -1, verbose = 2, return_train_score=True)
grid_search.fit(train_features, train_response)
pickle.dump(grid_search.best_params_, open('best_params_g1.pkl',  'wb'))
pickle.dump(grid_search.best_estimator_, open('best_estimator_g1.pkl',  'wb'))