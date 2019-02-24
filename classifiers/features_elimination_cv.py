import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression 
from sklearn.svm import LinearSVC, SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from utilities import features_extractors2
from utilities import feature_selectors
from utilities import tools


## Load data
X_raw = np.load("./dataset/X_train_kaggle.npy")
X_kaggle_raw = np.load("./dataset/X_test_kaggle.npy")
dataset = pd.read_csv("./dataset/groups.csv")
y = dataset.iloc[:, -1].values
groups = dataset.iloc[:, 1].values


## Encode y
le = LabelEncoder()
y = le.fit_transform(y)


## Extract features

# Without orientation 
#X = features_extractors2.features_extraction_no_ori(X_raw)
#X_kaggle = features_extractors2.features_extraction_no_ori(X_kaggle_raw)

# With orientation
X = features_extractors2.features_extraction(X_raw)
X_kaggle = features_extractors2.features_extraction(X_kaggle_raw)


## Feature selection
X, X_kaggle = feature_selectors.rfe(X, y, X_kaggle)
#X = feature_selectors.pca(X)
#X, X_kaggle = feature_selectors.boruta(X, y, X_kaggle)


## Grid search

# Declare classifiers and there params
etc = ExtraTreesClassifier()
param_etc = {'max_features' : ['sqrt', 'log2'],
             'n_estimators' : [200, 300, 500, 700, 1000, 1500, 2000]}

lr = LogisticRegression()
param_lr = {'penalty' : ['l1', 'l2'],
            'C' : np.logspace(0, 4, 10)}

mlda = LinearDiscriminantAnalysis()
param_mlda = {}

dt = DecisionTreeClassifier()
param_dt = {}

rf = RandomForestClassifier()
param_rf = {
    'bootstrap': [True],
    'max_depth': [80, 90, 100, 110],
    'max_features': [2, 3],
    'min_samples_leaf': [3, 4, 5],
    'min_samples_split': [8, 10, 12],
    'n_estimators': [100, 200, 300, 1000]
}

gb = GradientBoostingClassifier()
param_gb = {'n_estimators': [10, 30, 100, 300, 1000],
                    'max_depth': [2, 3, 4, 5],
                    'min_samples_leaf': [1, 2, 3]}

# Chose classifier and its params
clf = etc
params = param_etc

# Start grid search and print results
grid = GridSearchCV(clf, params, scoring='accuracy', n_jobs=4, verbose=1)
grid.fit(X, y, groups)
print('Best score and parameter combination = ')
print(grid.best_score_)    
print(grid.best_params_)    

# Set best params for classifier
clf.set_params(**grid.best_params_)


## Print results
tools.accuracy_average2(clf, X, y)

## Save results
clf.fit(X, y)
y_kaggle = clf.predict(X_kaggle)
tools.CSVOutput(y_kaggle, le)




