# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 15:14:50 2019

@author: Médéric Carriat
"""

## Importing the libraries
import numpy as np
from sklearn.metrics import confusion_matrix
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV, cross_val_score, RandomizedSearchCV

import feature_extractors

### To test different combinations of hyper-parameters on our best classifiers

test_size = 0.20
dataset = pd.read_csv('dataset/groups.csv')
X_raw = np.load("dataset/X_train_kaggle.npy")
y = dataset.iloc[:, -1].values
groups = dataset.iloc[:, 1].values

le = LabelEncoder()
y = le.fit_transform(y)

X = feature_extractors.deviationer_plus(X_raw, 1703)

# Parameters grids
rf_param_grid = {'n_estimators': [10, 30, 100, 300, 1000]}

et_param_grid = {'n_estimators': [10, 30, 100, 300, 1000]}

boost_param_grid = {'n_estimators': [10, 30, 100, 300, 1000],
                    'max_depth': [2, 3, 4, 5],
                    'min_samples_leaf': [1, 2, 3]}


## Fitting and printing out best score and params
print("Fitting models")
rf_est = RandomForestClassifier()
rf_gs_cv = GridSearchCV(rf_est, rf_param_grid, cv=3).fit(X, y)
print("RF grid search", rf_gs_cv.best_score_, rf_gs_cv.best_params_)
rf_rs_cv = RandomizedSearchCV(rf_est, param_distributions=rf_param_grid, cv=4, verbose=1).fit(X, y, groups=groups)
print("RF randomized search", rf_rs_cv.best_score_, rf_rs_cv.best_params_)

boost_est = GradientBoostingClassifier()
boost_gs_cv = GridSearchCV(boost_est, boost_param_grid, cv=3).fit(X, y)
print("Boost grid search", boost_gs_cv.best_score_, boost_gs_cv.best_params_)
boost_rs_cv = RandomizedSearchCV(boost_est, param_distributions=boost_param_grid, cv=4, verbose=1).fit(X, y, groups=groups)
print("Boost randomized search", boost_rs_cv.best_score_, boost_rs_cv.best_params_)

et = ExtraTreesClassifier()
et_gs_cv = GridSearchCV(estimator=et, param_grid=et_param_grid, cv=3).fit(X, y, groups=groups)
print("ET grid search", et_gs_cv.best_score_, et_gs_cv.best_params_)
et_rs_cv = RandomizedSearchCV(et, param_distributions=et_param_grid, cv=4, verbose=1).fit(X, y, groups=groups)
print("ET randomized search", et_rs_cv.best_score_, et_rs_cv.best_params_)
print('\n')


## Check score with optimised params
print("Score test with otpimised params")
clf_rf = RandomForestClassifier(n_estimators=rf_gs_cv.best_params_['n_estimators'])
scores = cross_val_score(clf_rf, X, y, cv=3)
print("RF ", scores.mean())

clf_boost = GradientBoostingClassifier(n_estimators=boost_gs_cv.best_params_['n_estimators'],
                                 max_depth=boost_gs_cv.best_params_['max_depth'],
                                 min_samples_leaf=boost_gs_cv.best_params_['min_samples_leaf'])
scores = cross_val_score(clf_boost, X, y, cv=3)
print("Boost ", scores.mean())

clf_et = ExtraTreesClassifier(n_estimators=et_rs_cv.best_params_['n_estimators'])
scores = cross_val_score(clf_et, X, y, cv=3, groups=groups)
print("ET ", scores.mean())

    