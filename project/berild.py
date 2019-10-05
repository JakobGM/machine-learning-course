#!/usr/bin/env python
# coding: utf-8

# # Santander Customer Transaction Prediction

# In[49]:


import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, roc_auc_score, roc_curve


train_df = pd.read_csv("data/train_small.csv", header=None)
test_df = pd.read_csv("data/test_small.csv")

train_df.columns = list([test_df.columns[0],"target",*test_df.columns[1:]])


# ## Visualization

# In[52]:


train_df.head()


# In[53]:


test_df.head()


# In[62]:


train_df.iloc[:,2:].describe()


# In[61]:


test_df.describe()


# ## Data Preprocessing

# In[69]:


response = np.array(train_df['target'])
features = np.array(train_df.iloc[:,2:])
features_list = list(train_df.iloc[:,2:].columns)

from sklearn.model_selection import train_test_split
train_features, test_features, train_response, test_response = train_test_split(features, response, test_size = 0.25, random_state = 42)


# ## Feature Selection 

# In[70]:


from sklearn.ensemble import RandomForestRegressor



# ## Hyperparameter Tuning

# In[71]:


def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    mape = 100 * np.mean(errors / test_labels)
    accuracy = 100 - mape
    print('Model Performance')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))
    
    return accuracy


# ## Random search

# In[72]:


from sklearn.model_selection import RandomizedSearchCV
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]

# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

# First create the base model to tune
rf = RandomForestRegressor(random_state = 42)
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid,
                              n_iter = 100, scoring='neg_mean_absolute_error', 
                              cv = 3, verbose=2, random_state=42, n_jobs=-1,
                              return_train_score=True)

# Fit the random search model
rf_random.fit(train_features, train_response);


# In[ ]:


rf_random.best_params_


# In[ ]:


base_model = RandomForestRegressor(n_estimators = 10, random_state = 42)
base_model.fit(train_features, train_response)
base_accuracy = evaluate(base_model, test_features, test_response)


# In[ ]:


best_random = rf_random.best_estimator_
random_accuracy = evaluate(best_random, test_features, test_response)
