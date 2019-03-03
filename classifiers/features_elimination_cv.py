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
from sklearn.model_selection import StratifiedKFold
from utilities import feature_extractors, features_extractors2
from utilities import feature_selectors
from utilities import tools

#try :
#    import xgboost as xgb
#    from xgboost import XGBClassifier
#except :
#    XGB_installed = 0

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
#X = feature_extractors.euler_angles(X_raw, len(X_raw))
#X_kaggle = feature_extractors.euler_angles(X_kaggle_raw, len(X_kaggle_raw))

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
             'n_estimators' : [200, 300, 500, 700, 1000, 1500, 2000, 10000]}

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
param_gb = {'n_estimators': [10, 30, 100, 300, 1000, 10000],
                    'max_depth': [2, 3, 4, 5],
                    'min_samples_leaf': [1, 2, 3]}

#if(XGB_installed) :
#        print("Training XGB")
#        xgb = XGBClassifier(learning_rate=0.1, n_estimators=140, max_depth=5,
#                            min_child_weight=3, gamma=0.2, subsample=0.6, colsample_bytree=1.0,
#                            objective='binary:logistic', nthread=4, scale_pos_weight=1, seed=27)
#
#        #print(" Start Feeding Data")
#        #cv_folds = 5
#        #early_stopping_rounds = 50
#        #xgb_param = xgb_alg.get_xgb_params()
#        #xgtrain = xgb.DMatrix(X_train, label=y_train)
#        ## xgtest = xgb.DMatrix(X_test.values, label=y_test.values)
#        #cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=xgb_alg.get_params()['n_estimators'], nfold=cv_folds,
#        #                  early_stopping_rounds=early_stopping_rounds)
#        #xgb_alg.set_params(n_estimators=cvresult.shape[0])
#        
#        #print(" Start Training")
#        #xgb_alg.fit(X_train, y_train, eval_metric='auc')
#        param_xgb = {
#        'min_child_weight': [1, 5, 10],
#        'gamma': [0.5, 1, 1.5, 2, 5],
#        'subsample': [0.6, 0.8, 1.0],
#        'colsample_bytree': [0.6, 0.8, 1.0],
#        'max_depth': [3, 4, 5]
#        }

# Chose classifier and its params
clf = etc
params = param_etc

# Start grid search and print results
grid = GridSearchCV(clf, params, scoring='accuracy', n_jobs=4, verbose=1, cv=5)

#folds = 3
#param_comb = 5
#skf = StratifiedKFold(n_splits=folds, shuffle = True, random_state = 1001)
#grid = RandomizedSearchCV(clf, param_distributions=params, n_iter=param_comb, n_jobs=4, cv=skf.split(X,y), verbose=3, random_state=1001 )

grid.fit(X, y)
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




